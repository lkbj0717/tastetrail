"""Live, swappable v13 inference. Importing this module performs no API calls.

Set TASTETRAIL_V13_SERVICE_CONFIG to a model JSON with an optional ``service``
block: judge_backend=none|cards2|hierarchical, decision_backend=fusion|judge,
version, frozen. The default is the current development agent_config.json.
Each comparison shares the initial GLM result and, when used, one planner.
No evaluation cases, answer keys, or reported accuracies are loaded here.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
import unicodedata
import uuid
from pathlib import Path

import numpy as np

from common import ROOT, PRIVATE, recovered_output, token_count, parse_ranking, usage_complete, usage_upper_bound
from evidence_cards import EvidenceCardBuilder
from inference_v13 import predict, qloo_review_evidence_available


_SERVICE = None
_SERVICE_LOCK = threading.Lock()
_QLOO_NUMERIC = {"pooled_log_ratio", "signed_history_log_ratio", "best_like_log_ratio",
                 "signed_agreement", "mf_weighted_qloo", "cohort_weighted_qloo",
                 "balanced_interest_qloo", "target_interest_qloo"}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _normal_title(title):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", title)).strip().casefold()


def _save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_suffix(f".{threading.get_ident()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.chmod(0o600)
    temporary.replace(path)


def _cached(path, fingerprint, calculate):
    if path.exists():
        record = json.loads(path.read_text())
        if record.get("request_fingerprint") != fingerprint:
            raise ValueError("The cached request does not match the visible input.")
        return record, True
    record = calculate()
    record["request_fingerprint"] = fingerprint
    _save(path, record)
    return record, False


def _not_called_record(reason=None):
    """Only deliberate skips may have a known zero model cost."""
    return {"error": reason, "metrics": {}, "calls": [], "physical_requests": 0,
            "reported_tokens": 0, "accounted_tokens": 0}


def _sum_known(values):
    values = list(values)
    return None if any(v is None for v in values) else sum(values)


def _record_cost(record):
    complete = usage_complete(record)
    observed = token_count(record)
    upper = usage_upper_bound(record)
    physical = record.get("physical_requests")
    if physical is None and record.get("calls"):
        physical = len(record["calls"])
    return {"reported_tokens": observed if complete else None,
            "observed_reported_tokens": observed,
            "reported_tokens_complete": complete,
            "accounted_tokens": upper,
            "model_requests": physical,
            "usage_is_upper_bound": not complete,
            "usage_unknown": not complete and upper is None}


class RecommendationService:
    """Dependencies may be injected for tests; production tools load lazily.

    Hooks: baseline(history,candidates,case_id), cards_review(...,mode,initial),
    planner(history,candidates,case_id,baseline_tokens),
    hierarchy_review(...,mode,plan,initial), affinity(history,candidates,case_id).
    They return the same private record dictionaries as the research runners.
    ``extra_feature_tools`` maps feature prefixes to objects with evaluate(h,c).
    """
    def __init__(self, *, private_path=None, config=None, metadata=None,
                 community=None, feature_builder=None, feature_tools=None, culture_builder=None,
                 hierarchy_culture_builder=None,
                 baseline=None, cards_review=None, planner=None,
                 hierarchy_review=None, affinity=None, grouped=None,
                 extra_feature_tools=None):
        self.private = Path(private_path or os.environ.get("V13_PRIVATE", str(PRIVATE)))
        self.old = self.private.parent / "culture_v11"
        config_path = Path(os.environ.get("TASTETRAIL_V13_SERVICE_CONFIG", str(ROOT / "agent_config.json")))
        self.config = json.loads(json.dumps(config if config is not None else json.loads(config_path.read_text())))
        options = self.config.get("service", {})
        judge = options.get("judge_backend", self.config.get("judge_version", "none")) or "none"
        self.judge_backend = "hierarchical" if judge.startswith("hierarchical") else ("cards2" if judge.startswith("cards2") else judge)
        self.decision_backend = options.get("decision_backend", "fusion")
        if self.judge_backend not in {"none", "cards2", "hierarchical"}:
            raise ValueError("Unsupported judge backend in the service configuration.")
        if self.decision_backend not in {"fusion", "judge"}:
            raise ValueError("Unsupported decision backend in the service configuration.")
        if self.decision_backend == "judge" and self.judge_backend == "none":
            raise ValueError("Direct judge decisions require a judge backend.")
        self.frozen = options.get("frozen", False) is True
        self.version = str(options.get("version", f"v13-{self.judge_backend}-{self.decision_backend}-development"))
        if self.decision_backend == "fusion":
            for name in ("selected_agent", "selected_qloo"):
                if self.config.get(name) not in self.config.get("models", {}):
                    raise ValueError("The selected fusion model is missing.")
                names = self.config["models"][self.config[name]]["features"]
                if any(n.startswith("judge_") for n in names) and any(n.startswith("hier_") for n in names):
                    raise ValueError("A production branch cannot combine short and hierarchical reviews within the bounded budget.")
            selected = set().union(*(set(self.config["models"][self.config[n]]["features"]) for n in ("selected_agent", "selected_qloo")))
            self.short_modes = {m for m in ("agent", "qloo") if f"judge_{m}" in selected}
            self.hier_modes = {m for m in ("agent", "qloo") if f"hier_{m}" in selected}
        else:
            self.short_modes = {"agent", "qloo"} if self.judge_backend == "cards2" else set()
            self.hier_modes = {"agent", "qloo"} if self.judge_backend == "hierarchical" else set()
        if self.hier_modes and not options.get("version"):
            self.version = f"v13-{self.config.get('hierarchical_version', 'hierarchical')}-{self.decision_backend}-development"
        if metadata is None:
            path = self.private / "metadata.json"
            metadata = json.loads((path if path.exists() else self.old / "metadata.json").read_text())
        self.metadata = {str(mid): {k: row.get(k) for k in ("title", "year", "genres")}
                         for mid, row in metadata.items()}
        self.lookup = {}
        for mid, row in self.metadata.items():
            key = (_normal_title(row["title"]), row["year"])
            if key in self.lookup:
                raise ValueError("The research catalog contains an ambiguous title/year pair.")
            self.lookup[key] = int(mid)
        self.builder = EvidenceCardBuilder(self.private, movie_metadata=self.metadata)
        self.community = community
        self.feature_builder = feature_builder
        self.feature_tools = feature_tools
        self.culture_builder = culture_builder
        self.hierarchy_culture_builder = hierarchy_culture_builder
        self.hooks = {"baseline": baseline, "cards_review": cards_review, "planner": planner,
                      "hierarchy_review": hierarchy_review, "affinity": affinity, "grouped": grouped}
        self.extra_feature_tools = extra_feature_tools or {}
        self.config_digest = _digest(self.config)
        self.lock = threading.Lock()

    def resolve(self, body):
        # The same HTTP schema is reused for direct Python calls.
        from app_v13 import Request
        request = Request.model_validate(body)
        def movie(m):
            key = (_normal_title(m.title), m.year)
            if key not in self.lookup:
                raise ValueError(f"Unsupported research-catalog film: {m.title} ({m.year}). Choose from the catalog.")
            return self.lookup[key]
        history = [(movie(m), m.rating) for m in request.likes + request.dislikes]
        candidates = [movie(m) for m in request.candidates]
        self.builder._validate(history, candidates)
        return history, candidates

    def _community(self):
        if self.community is None:
            from tastetrail.cluster_tools import TasteTools
            self.community = TasteTools(dict(np.load(self.old / "tools.npz")))
        return self.community

    def _culture(self):
        if self.culture_builder is None:
            from compact_cards import CompactEvidenceCardBuilder
            self.culture_builder = CompactEvidenceCardBuilder(self.private, movie_metadata=self.metadata)
        return self.culture_builder

    def _hierarchy_culture(self):
        if self.hierarchy_culture_builder is None:
            # v2's reviewer uses bounded theme/style/tone descriptions. Its
            # no-Qloo builder remains a separate, never-culture-loaded object.
            self.hierarchy_culture_builder = EvidenceCardBuilder(self.private, movie_metadata=self.metadata)
        return self.hierarchy_culture_builder

    def _features(self):
        if self.feature_tools is None:
            from feature_tools_v13 import FeatureTools
            self.feature_tools = FeatureTools(community=self._community(), tools_path=self.old / "tools.npz",
                                               private_dir=self.private)
        return self.feature_tools

    def local_values(self, history, candidates, initial, requested=()):
        """Independent rating features; this path never opens Qloo metadata."""
        from feature_tools_v13 import COMMUNITY, INTERACTION_CF, MIXTURE
        independent = COMMUNITY | INTERACTION_CF | {n for n in MIXTURE if not n.endswith("_qloo")}
        built = self._features().build(history, candidates, initial,
                                       required_features=set(requested) & independent)
        return built["values"], built["trace"]

    def _call_model(self, prompt, payload, *, max_tokens, reasoning_effort=None):
        # Use the same global semaphore and paced transport as the hierarchy.
        from hierarchical_agent import SingleAttemptPacedTransport, _MODEL_SLOTS
        from tastetrail.model_client import ModelClient
        user = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        bound = len((prompt + user).encode()) + 256
        out = {"error": None, "request_data": payload, "prompt_sha": _digest(prompt),
               "max_completion_tokens": max_tokens, "reasoning_effort_requested": reasoning_effort,
               "prompt_utf8_upper_bound": bound}
        transport = SingleAttemptPacedTransport()
        client = ModelClient(provider="glm", max_tokens=max_tokens, transport=transport,
                             reasoning_effort=reasoning_effort)
        started = time.monotonic()
        try:
            with _MODEL_SLOTS:
                message = client.complete([{"role": "system", "content": prompt},
                                           {"role": "user", "content": user}], json_output=True)
            out["output"] = parse_ranking(message.get("content", ""))
        except Exception as exc:
            out["error"] = type(exc).__name__ + ": " + str(exc)[:180]
        finally:
            out.update(calls=client.calls, metrics=client.metrics(), seconds=time.monotonic() - started,
                       physical_requests=transport.physical_requests, pacing_events=transport.events)
            known = any(isinstance(c.get("usage", {}).get("prompt_tokens"), (int, float)) and
                        isinstance(c.get("usage", {}).get("completion_tokens"), (int, float)) for c in client.calls)
            out["reported_tokens"] = token_count(out)
            out["usage_is_upper_bound"] = bool(transport.physical_requests and not known)
            out["accounted_tokens"] = max(out["reported_tokens"], bound + max_tokens if out["usage_is_upper_bound"] else 0)
            transport.close()
        return out

    def _baseline(self, history, candidates, case_id):
        if self.hooks["baseline"]:
            return self.hooks["baseline"](history, candidates, case_id), False
        from baseline_snapshot import PROMPT
        public = self.builder.build(history, candidates, include_qloo=False)
        payload = {"liked_movies": [{k: v for k, v in h.items() if k != "history_id"} for h in public["history"][:5]],
                   "disliked_movies": [{k: v for k, v in h.items() if k != "history_id"} for h in public["history"][5:]],
                   "candidates": [{"id": c["candidate_id"], **{k: v for k, v in c.items() if k != "candidate_id"}} for c in public["candidates"]]}
        fingerprint = _digest([payload, PROMPT, 16384])
        # Baseline cache is shared across service versions, independently of planner/fusion selection.
        path = self.private / "live/baseline" / f"{fingerprint}.json"
        return _cached(path, fingerprint, lambda: self._call_model(PROMPT, payload, max_tokens=16384))

    def _cards_review(self, history, candidates, case_id, mode, initial):
        if self.hooks["cards_review"]:
            return self.hooks["cards_review"](history, candidates, case_id, mode, initial), False
        from compact_cards import JUDGE_SYSTEM_PROMPT
        builder = self._culture() if mode == "qloo" else self.builder
        payload = builder.build(history, candidates, include_qloo=mode == "qloo")
        bound = len((JUDGE_SYSTEM_PROMPT + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))).encode()) + 256
        budget = 9 * token_count(initial)
        limit = min(4096, budget - bound)
        fingerprint = _digest([payload, JUDGE_SYSTEM_PROMPT, limit, "low"])
        path = self.private / "live/cards2" / mode / f"{case_id}.json"
        def calculate():
            if limit < 768:
                return {"error": "budget_preflight_skip", "calls": [], "metrics": {},
                        "reported_tokens": 0, "accounted_tokens": 0, "request_data": payload}
            return self._call_model(JUDGE_SYSTEM_PROMPT, payload, max_tokens=limit, reasoning_effort="low")
        return _cached(path, fingerprint, calculate)

    def _planner(self, history, candidates, case_id, initial):
        if self.hooks["planner"]:
            return self.hooks["planner"](history, candidates, case_id, token_count(initial)), False
        from hierarchy_backend import get_backend
        backend=get_backend(self.config.get("hierarchical_version"))
        cached = (self.private / backend.VERSION / "plans" / f"{case_id}.json").exists()
        return backend.run_plan(history, candidates, case_id, token_count(initial), private_path=self.private,
                        builder=self.builder), cached

    def _hierarchy_review(self, history, candidates, case_id, mode, plan, initial):
        if self.hooks["hierarchy_review"]:
            return self.hooks["hierarchy_review"](history, candidates, case_id, mode, plan, initial), False
        from hierarchy_backend import get_backend
        backend=get_backend(self.config.get("hierarchical_version"))
        cached = (self.private / backend.VERSION / "reviews" / mode / f"{case_id}.json").exists()
        return backend.run_review(history, candidates, case_id, mode, plan, initial, private_path=self.private,
                          builder=self.builder, community_tools=self._community(),
                          culture_builder=self._hierarchy_culture() if mode == "qloo" else None), cached

    def _affinity(self, history, candidates, case_id):
        if self.hooks["affinity"]:
            return self.hooks["affinity"](history, candidates, case_id)
        from tastetrail.qloo_http import QlooClient
        catalog = json.loads((self.old / "catalog.json").read_text())
        wanted = [catalog[str(mid)] for mid in candidates]
        expected = [str(uuid.UUID(qid)) for qid in wanted]
        signals = [[catalog[str(mid)] for mid, _ in history[:5]]] + [[catalog[str(mid)]] for mid, _ in history]
        requests = [{"filter.type": "urn:entity:movie", "signal.interests.entities": ",".join(seed),
                     "filter.results.entities": ",".join(wanted), "take": 2} for seed in signals]
        fingerprint = _digest(requests)
        def calculate():
            client = QlooClient(ttl=30 * 86400)
            out = {"error": None, "rows": [], "requests": requests}
            try:
                for params in requests:
                    entities = client.get("/v2/insights", params)["results"]["entities"]
                    values = {str(uuid.UUID(e["entity_id"])): e.get("query", {}).get("affinity") for e in entities}
                    if len(entities) != 2 or set(values) != set(expected):
                        raise ValueError("Exact candidate pair is unavailable.")
                    row = [values[qid] for qid in expected]
                    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0 for x in row):
                        raise ValueError("Missing or invalid cultural affinity.")
                    out["rows"].append(row)
            except Exception as exc:
                out["error"] = type(exc).__name__ + ": " + str(exc)[:180]
            out["calls"] = client.calls
            return out
        out, cached = _cached(self.private / "live/affinity" / f"{case_id}.json", fingerprint, calculate)
        return {**out, "record_cache_hit": cached}

    def _qloo_features(self, history, candidates, case_id, initial, requested, values):
        """Only called for the Qloo condition; never mutates common values."""
        out = {"values": dict(values), "available": True, "groups_available": True,
               "calls": [], "evidence": {}, "trace": []}
        needs_affinity = bool(requested & _QLOO_NUMERIC or any(n.startswith("qgroup_") for n in requested))
        evidence = self._affinity(history, candidates, case_id) if needs_affinity else None
        if evidence:
            out["calls"].extend(evidence.get("calls", []))
            out["evidence"]["pair_affinities"] = {"source": "Qloo exact candidate queries", "rows": evidence.get("rows", []),
                                                   "row_order": ["pooled liked films"] + [f"H{i:02d}" for i in range(1, 9)],
                                                   "column_order": ["C01", "C02"],
                                                   "available": not bool(evidence.get("error"))}
            out["trace"].append({"name": "Retrieve exact-pair Qloo associations",
                                 "status": "fallback" if evidence.get("error") else "ok",
                                 "detail": "One pooled-like query and eight single-history queries; requests contain public movie UUIDs only."})
            if evidence.get("error"):
                out["available"] = False
                return out
        group = None
        if any(n.startswith("qgroup_") for n in requested):
            from grouped_qloo import query_history
            group = (self.hooks["grouped"] or query_history)(history, candidates, case_id=case_id)
            out["calls"].extend(group.get("calls", []))
            if group.get("error"):
                out["groups_available"] = False
        from feature_tools_v13 import SUPPORTED
        needed = requested & SUPPORTED
        if needed - set(values):
            tool = self.feature_builder or self._features()
            built = tool.build(history, candidates, initial, evidence,
                               required_features=needed, group_record=group)
            # Keep control values immutable even if another tool returns them.
            out["values"].update({k: v for k, v in built["values"].items() if k in requested and k not in values})
            source_trace = built.get("trace", {})
            if source_trace.get("groups", {}).get("valid") is False:
                out["groups_available"] = False
            if source_trace.get("source_availability", {}).get("required_tools_valid") is False:
                out["available"] = False
            out["trace"].append({"name": "Qloo cultural numerical tools", "status": "ok" if out["available"] else "fallback",
                                 "detail": "Shared offline/live feature implementation; only selected cultural features enter the final score.",
                                 "sources": source_trace.get("feature_sources", {})})
        if any(n.startswith("learned_") for n in requested) and "learned_" not in self.extra_feature_tools:
            from learned_culture import LearnedCultureTools
            self.extra_feature_tools["learned_"] = LearnedCultureTools()
        if any(n.startswith("qaff_") for n in requested) and "qaff_" not in self.extra_feature_tools:
            from affinity_graph import SparseAffinityTools
            self.extra_feature_tools["qaff_"] = SparseAffinityTools()
        for prefix, tool in self.extra_feature_tools.items():
            if any(name.startswith(prefix) for name in requested):
                extra = tool.evaluate(history, candidates)
                if isinstance(extra, tuple):
                    extra = extra[0]
                if extra is None:
                    out["available"] = False
                else:
                    out["values"].update({k: v for k, v in extra.items() if k in requested and k not in values})
        missing = requested - set(out["values"]) - {"judge_qloo", "judge_agent", "hier_qloo", "hier_agent"}
        if missing and out["groups_available"]:
            raise ValueError("The selected model requires a feature backend that is not configured: " + ", ".join(sorted(missing)))
        cultural_values = {k: out["values"][k] for k in requested if k in out["values"] and k not in values}
        if cultural_values:
            out["evidence"]["numerical_cultural_tools"] = {
                "source": "Qloo cultural metadata or affinities, combined with independent rating tools",
                "selected_features": cultural_values,
                "scores_are_personal_probabilities": False}
        return out

    def recommend(self, body, compare=True):
        with self.lock:
            return self._recommend(body, compare)

    def _recommend(self, body, compare):
        started = time.monotonic()
        history, candidates = self.resolve(body)
        case_id = "LIVE_" + _digest([history, candidates, self.config_digest])[:24]
        trace = [{"name": "Validate the eight ratings and two candidates", "status": "ok",
                  "detail": f"All ten films resolved to the {len(self.metadata)}-film research catalog."}]
        warnings = [] if self.frozen else ["This is a development configuration; the final version and confirmation results are not yet frozen."]
        initial, initial_cached = self._baseline(history, candidates, case_id)
        first = recovered_output(initial)
        baseline_tokens = token_count(initial)
        if first is None:
            warnings.append("The initial GLM answer is invalid. It remains unavailable in the comparison; Agent fusion can still use independent local preference tools.")
        if baseline_tokens <= 0:
            warnings.append("Initial model usage is unavailable or zero. No planner or reviewer is called because a relative token budget cannot be established.")
        trace.append({"name": "Direct GLM preference comparison", "status": "ok" if first else "failed", "cache_hit": initial_cached,
                      "detail": "One initial call, shared by both Agent conditions." if first else "No valid ranking returned; use zero initial margin in local Agent fusion, matching confirmation inference."})
        selected_names = set().union(*(set(self.config["models"][self.config[n]]["features"]) for n in ("selected_agent", "selected_qloo"))) if self.decision_backend == "fusion" else set()
        values, community_trace = self.local_values(history, candidates, initial, selected_names)
        trace.append({"name": "Reconstruct preference from independent rating data", "status": "ok",
                      "detail": "The same collaborative and matrix-factorization evidence is available to both Agent conditions."})
        agent_names = set(self.config["models"][self.config["selected_agent"]]["features"]) if self.decision_backend == "fusion" else set()
        if agent_names - set(values) - {"judge_agent", "hier_agent"}:
            raise ValueError("The no-Qloo model contains features outside the independent rating tools.")
        qloo_names = set(self.config["models"][self.config["selected_qloo"]]["features"]) if self.decision_backend == "fusion" else set()
        qdata = {"values": dict(values), "available": True, "groups_available": True, "calls": [], "evidence": {}, "trace": []}
        if qloo_names - set(values) - {"judge_qloo", "hier_qloo"}:
            qdata = self._qloo_features(history, candidates, case_id, initial, qloo_names, values)
            trace.extend(qdata["trace"])
        plan = None
        plan_cached = False
        modes = ["agent", "qloo"] if compare else ["qloo"]
        active_hier_modes = {m for m in modes if m in self.hier_modes and (m != "qloo" or qdata["available"])}
        if active_hier_modes and baseline_tokens > 0:
            plan, plan_cached = self._planner(history, candidates, case_id, initial)
            trace.append({"name": "Shared preference and query planner", "status": "fallback" if plan.get("fallback") else "ok",
                          "cache_hit": plan_cached, "detail": "Up to three hypotheses, counterevidence, and two liked plus one disliked seed group; no candidate ranking.",
                          "preference_plan": plan.get("output")})
        judges, hierarchical, judge_cached, hierarchy_cached = {}, {}, {}, {}
        used_plan_modes = set()
        for mode in modes:
            if mode == "qloo" and not qdata["available"]:
                continue
            for kind, wanted, records, cached_records in (("short", self.short_modes, judges, judge_cached),
                                                          ("hierarchical", self.hier_modes, hierarchical, hierarchy_cached)):
                if mode not in wanted:
                    continue
                try:
                    if baseline_tokens <= 0:
                        row, cached = _not_called_record("initial_usage_unavailable_no_extra_model_calls"), False
                    elif kind == "hierarchical":
                        used_plan_modes.add(mode)
                        row, cached = self._hierarchy_review(history, candidates, case_id, mode, plan, initial)
                    else:
                        row, cached = self._cards_review(history, candidates, case_id, mode, initial)
                    records[mode], cached_records[mode] = row, cached
                except (OSError, ValueError, KeyError):
                    records[mode] = {"error": "The selected evidence backend is unavailable.", "metrics": {}, "calls": []}
                    cached_records[mode] = False
                row = records[mode]
                trace.append({"name": ("Agent + Qloo" if mode == "qloo" else "Agent") + f" {kind} evidence comparison",
                              "status": "fallback" if row.get("error") else "ok", "cache_hit": cached_records[mode],
                              "detail": "Reviews support and counterevidence without seeing the initial GLM choice."})
                if row.get("repair_attempted"):
                    trace.append({"name": ("Agent + Qloo" if mode == "qloo" else "Agent") + " bounded response-format recovery",
                                  "status": "ok" if row.get("repair_succeeded") else "fallback",
                                  "detail": "One low-effort retry for an invalid or truncated response; both attempts are charged within the same 18× branch budget."})
                if mode == "qloo":
                    qdata["calls"].extend(row.get("qloo_calls", []))
                    public_evidence = row.get("request_data", {})
                    for key in ("qloo_cultural_evidence", "qloo_group_evidence"):
                        if key in public_evidence:
                            qdata["evidence"][key] = public_evidence[key]
                    # A failed model/JSON review does not erase already retrieved
                    # tool evidence. The shared inference layer then substitutes
                    # the initial margin for that review without another call.
                    has_culture = qloo_review_evidence_available(row,hierarchical=kind == "hierarchical")
                    if not has_culture:
                        qdata["available"] = False
        if self.decision_backend == "fusion":
            result = predict(qdata["values"], initial, self.config, judges, hierarchical=hierarchical,
                             qloo_available=qdata["available"], groups_available=qdata["groups_available"])
        else:
            direct_records = hierarchical if self.judge_backend == "hierarchical" else judges
            outputs = {"glm": first,
                       "agent": recovered_output(direct_records.get("agent", {})) or first,
                       "agent_qloo": recovered_output(direct_records.get("qloo", {})) or first}
            if not qdata["available"]:
                outputs["agent_qloo"] = first
            result = {"choices": {k: (0 if v["ranking"][0] == "C01" else 1) if v else None for k, v in outputs.items()},
                      "probabilities": {k: v["probability_c01"] / 100 if v else None for k, v in outputs.items()},
                      "qloo_fallback": not qdata["available"]}
        if result["qloo_fallback"]:
            warnings.append("Qloo evidence or its review was unavailable. The final choice uses available no-Qloo evidence; no Qloo improvement is claimed.")
        if result["qloo_fallback"]:
            warnings.append("The fallback uses the configured Agent fusion with review margins replaced by the initial GLM margin; no extra review is run or borrowed from the comparison." if self.decision_backend == "fusion" else "The direct-judge prototype falls back to the initial GLM answer without an additional model call.")
        if plan and plan.get("fallback"):
            warnings.append("The preference planner did not complete reliably; candidate-independent fixed query groups were used.")
        for mode, row in list(judges.items()) + list(hierarchical.items()):
            if row.get("error"):
                warnings.append(f"The {mode} review did not complete reliably. Its fallback is recorded in the cost and trace details.")
        trace.append({"name": "Verify candidates and apply the configured decision", "status": "ok",
                      "detail": "Numerical fusion selected on development data." if self.decision_backend == "fusion" else "The bounded counterevidence judge supplies the final ordering."})
        recommendations, explanations = {}, {}
        for key in ("glm", "agent", "agent_qloo"):
            if key == "agent" and not compare:
                continue
            choice = result["choices"][key]
            if choice is None:
                recommendations[key] = None
                explanations[key] = "No valid model ranking was returned. This method has no recommendation; its failure is retained in the comparison."
                continue
            movie = self.metadata[str(candidates[choice])]
            evidence_mode = "agent" if key == "agent" or result["qloo_fallback"] else "qloo"
            record = initial if key == "glm" else hierarchical.get(evidence_mode, judges.get(evidence_mode, {}))
            explanation_output = recovered_output(record)
            consistent = explanation_output and explanation_output["ranking"][0] == f"C{choice + 1:02d}"
            explanation = explanation_output["explanation"] if consistent else "The configured preference model combines the available evidence to select this film."
            if key == "agent_qloo" and result["qloo_fallback"]:
                explanation = "Qloo evidence was unavailable. Independent rating tools and the initial GLM judgment supply a degraded Agent fusion; no extra review was called." if self.decision_backend == "fusion" else "Qloo evidence was unavailable; this direct-judge prototype returned the initial GLM answer without another model call."
            recommendations[key] = {"title": movie["title"], "year": movie["year"], "candidate_id": f"C{choice + 1:02d}", "explanation": explanation}
            explanations[key] = explanation
        base_cost = _record_cost(initial)
        plan_cost = _record_cost(plan) if plan is not None else _record_cost(_not_called_record())
        baseline_complete = base_cost["reported_tokens_complete"] and baseline_tokens > 0
        def branch_cost(mode):
            short = _record_cost(judges[mode]) if mode in judges else _record_cost(_not_called_record())
            hier = _record_cost(hierarchical[mode]) if mode in hierarchical else _record_cost(_not_called_record())
            own = {k: _sum_known([short[k], hier[k]]) for k in ("reported_tokens", "accounted_tokens", "model_requests")}
            used_plan = plan_cost if mode in used_plan_modes else _record_cost(_not_called_record())
            pieces = [base_cost, used_plan, short, hier]
            total = _sum_known(p["reported_tokens"] for p in pieces)
            charged = _sum_known(p["accounted_tokens"] for p in pieces)
            observed = sum(p["observed_reported_tokens"] for p in pieces)
            return {"reported_tokens": total, "observed_reported_tokens": observed,
                    "reported_tokens_complete": all(p["reported_tokens_complete"] for p in pieces),
                    "accounted_tokens_upper_bound": charged,
                    "token_ratio": total / baseline_tokens if baseline_complete and total is not None else None,
                    "token_ratio_upper_bound": charged / baseline_tokens if baseline_complete and charged is not None else None,
                    "shared_initial_tokens": base_cost["reported_tokens"], "shared_planner_tokens": used_plan["reported_tokens"],
                    "review_tokens": own["reported_tokens"],
                    "model_requests": _sum_known(p["model_requests"] for p in pieces),
                    "short_review_cache_hit": judge_cached.get(mode, False),
                    "hierarchical_review_cache_hit": hierarchy_cached.get(mode, False),
                    "usage_is_upper_bound": any(p["usage_is_upper_bound"] for p in pieces),
                    "usage_unknown": charged is None}
        costs = {"glm": {**base_cost, "token_ratio": 1 if baseline_complete else None, "cache_hit": initial_cached},
                 "agent_qloo": branch_cost("qloo")}
        if compare:
            costs["agent"] = branch_cost("agent")
        all_rows = [initial] + ([plan] if plan is not None else []) + list(judges.values()) + list(hierarchical.values())
        all_costs = [_record_cost(r) for r in all_rows]
        costs["comparison"] = {"reported_tokens": _sum_known(r["reported_tokens"] for r in all_costs),
                               "observed_reported_tokens": sum(r["observed_reported_tokens"] for r in all_costs),
                               "reported_tokens_complete": all(r["reported_tokens_complete"] for r in all_costs),
                               "accounted_tokens_upper_bound": _sum_known(r["accounted_tokens"] for r in all_costs),
                               "shared_planner_counted_once": True,
                               "logical_qloo_queries": len(qdata["calls"]),
                               "qloo_cache_hits": sum(bool(c.get("cache_hit")) for c in qdata["calls"]),
                               "note": "Counts describe logical production requests, including original usage when a response cache is reused; they are not new billing on every page load. Unknown usage remains null; observed partial totals are separate."}
        ratio_upper = costs["agent_qloo"]["token_ratio_upper_bound"]
        costs["agent_qloo"]["within_20x_budget"] = ratio_upper < 20 if ratio_upper is not None else None
        if ratio_upper is None:
            warnings.append("Complete token usage or its conservative bound is unavailable; compliance with the 20× token budget cannot be established.")
        elif ratio_upper >= 20:
            warnings.append("This configuration exceeded the 20× conservative token budget and should not be presented as the final efficient version.")
        return {"recommendations": recommendations, "explanations": explanations,
                "probabilities": result["probabilities"], "selected_version": self.version,
                "configuration_frozen": self.frozen, "comparison_requested": bool(compare),
                "trace": trace, "qloo_evidence": qdata["evidence"], "costs": costs,
                "seconds": time.monotonic() - started, "warnings": warnings,
                "qloo_fallback": result["qloo_fallback"],
                "fallbacks": {"policy": result.get("qloo_fallback_policy"),
                              "short_reviews": result.get("judge_fallbacks", {}),
                              "hierarchical_reviews": result.get("hierarchical_fallbacks", {})},
                "community_evidence": {"source": "Independent background ratings", "trace": community_trace,
                                       "margins": {k: values[k] for k in ("community_margin", "mf_32_10", "neighbors_100") if k in values}},
                "scope": {"catalog_size": len(self.metadata), "task": "Two-candidate movie preference comparison",
                          "city_itinerary_validated": False},
                "scores_are_personal_probabilities": False}


def recommend(body, compare=True):
    global _SERVICE
    if _SERVICE is None:
        with _SERVICE_LOCK:
            if _SERVICE is None:
                _SERVICE = RecommendationService()
    return _SERVICE.recommend(body, compare=compare)
