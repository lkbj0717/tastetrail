"""Shared local feature construction for batch evaluation and the live service.

Every numerical formula delegates to its original research tool. Features are
loaded on demand: the no-Qloo path reads only the background tool file and, if
requested, plain/CF mixture weights. Public Qloo metadata is never loaded just
because a FeatureTools instance was constructed. This module makes no API calls
and accepts no case IDs, hidden ratings, or outcome labels.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from common import OLD, PRIVATE, recovered_output
from tastetrail.cluster_tools import TasteTools
from tastetrail.fusion import qloo_features


QBASE = ("pooled_log_ratio", "signed_history_log_ratio", "best_like_log_ratio", "signed_agreement")
AFFINITY = set(QBASE) | {"balanced_interest_qloo", "target_interest_qloo",
                        "cohort_weighted_qloo", "mf_weighted_qloo"}
COMMUNITY = {
    "community_margin", "genre_margin", "quality_margin", "neighbors_30", "neighbors_100",
    "latent_margin", "cohort_margin", "cohort_entropy", "balanced_interest_cf", "target_interest_cf",
    "mf_16_10", "mf_32_10", "mf_32_30", "ease_10", "ease_100", "ease_1000",
    "glm_log_odds", "cohort_gated_glm",
}
METADATA = {f"qmeta_{kind}_{suffix}" for kind in ("tags", "curated", "text", "people")
            for suffix in ("contrast", "ridge_0.1", "ridge_1", "ridge_10")}
GRAPH = {"qgraph_loo", "qgraph_asymmetric", "qgraph_background", "qgraph_background_loo"}
INTERACTION_CF = {"interaction_cf_ridge_1", "interaction_cf_ridge_10"}
INTERACTION_MIXED = {"interaction_mixed_ridge_1", "interaction_mixed_ridge_10"}
MIXTURE = {f"bernoulli_k{k}_{kind}" for k in (12, 24) for kind in ("plain", "cf", "qloo")}
GROUPS = {"qgroup_weighted_contrast", "qgroup_pooled_contrast"}
SUPPORTED = COMMUNITY | AFFINITY | METADATA | GRAPH | INTERACTION_CF | INTERACTION_MIXED | MIXTURE | GROUPS


def _required(names):
    # GLM is always parsed here. Judges/hierarchical outputs are merged by the
    # common inference layer, not fabricated or recalculated by local tools.
    if names is None:
        return set(COMMUNITY)
    if isinstance(names, str):
        raise ValueError("required_features must be a collection of feature names")
    result = set(names)
    if any(not isinstance(n, str) for n in result):
        raise ValueError("Feature names must be strings")
    result = {n for n in result if not n.startswith(("judge_", "hier_"))}
    unknown = result - SUPPORTED
    if unknown:
        raise ValueError("Unsupported local features: " + ", ".join(sorted(unknown)))
    return result


def requires_qloo(required_features):
    """Whether the requested numerical inputs use Qloo evidence/derived data."""
    names = _required(required_features)
    return bool(names & (AFFINITY | METADATA | GRAPH | INTERACTION_MIXED | GROUPS)
                or any(n.endswith("_qloo") for n in names & MIXTURE))


class FeatureTools:
    def __init__(self, community=None, *, tools_path=None, private_dir=None):
        """Optionally share an existing TasteTools instance with the service.

        Construction does not read files. Default paths point to the already
        fitted background model and private cached numerical weights.
        """
        if community is not None and (not hasattr(community, "evaluate") or not hasattr(community, "a")):
            raise ValueError("community must be an existing TasteTools instance")
        self._community = community
        self.tools_path = Path(tools_path) if tools_path is not None else OLD / "tools.npz"
        self.private_dir = Path(private_dir) if private_dir is not None else PRIVATE
        self._culture = None
        self._graph = None
        self._gram = None
        self._interactions = {}
        self._mixtures = {}

    @property
    def community(self):
        if self._community is None:
            with np.load(self.tools_path) as stored:
                self._community = TasteTools({k: stored[k] for k in stored.files})
        return self._community

    def _ids(self):
        return list(map(int, self.community.a["ids"]))

    def _validate(self, history, candidates):
        if isinstance(history, dict) or len(history) != 8 or len(candidates) != 2:
            raise ValueError("Exactly eight visible history rows and two candidates required")
        if any(not isinstance(row, (list, tuple)) or len(row) != 2 for row in history):
            raise ValueError("History contains only (movie ID, visible rating) pairs")
        ids = [int(row[0]) for row in history]
        candidates = list(map(int, candidates))
        ratings = np.asarray([row[1] for row in history], float)
        if (len(set(ids + candidates)) != 10 or not np.isin(ratings, [1, 2, 4, 5]).all()
                or np.sum(ratings >= 4) != 5 or np.sum(ratings <= 2) != 3):
            raise ValueError("Five likes, three dislikes, and two distinct unseen candidates required")
        known = set(self._ids())
        if not set(ids + candidates) <= known:
            raise ValueError("Unknown movie ID for fitted feature catalog")
        return [(i, float(r)) for i, r in zip(ids, ratings)], candidates

    def _cultural(self):
        if self._culture is None:
            from audit_features import CulturalMetadataTools
            tool = CulturalMetadataTools(self.private_dir / "qloo_metadata.json")
            if tool.ids != self._ids():
                raise ValueError("Qloo metadata and background catalog order differ")
            self._culture = tool
        return self._culture

    def _cultural_graph(self):
        if self._graph is None:
            from graph_models import GraphTools
            tool = GraphTools(self.private_dir / "qloo_graph_tools.npz")
            if tool.ids != self._ids():
                raise ValueError("Qloo graph and background catalog order differ")
            self._graph = tool
        return self._graph

    def _cf_gram(self):
        if self._gram is None:
            from interaction_tools import normalized_gram
            self._gram = normalized_gram(self.community.a["mf_32_10"])
        return self._gram

    def _interaction(self, mixed):
        key = "mixed" if mixed else "cf"
        if key not in self._interactions:
            from interaction_tools import InteractionTools
            cf = self._cf_gram()
            # The original implementation computes both CF and mixed features.
            # A zero dummy cultural Gram lets its exact CF formula be reused
            # without reading any Qloo artifact. Mixed outputs from this dummy
            # object are never exported. The true mixed route loads real Qloo.
            culture = self._cultural().kernels["curated"] if mixed else np.zeros_like(cf)
            self._interactions[key] = InteractionTools(self._ids(), cf, culture, self.community.a["item_bias"])
        return self._interactions[key]

    def _mixture(self, name):
        if name not in MIXTURE:
            raise ValueError("Unknown mixture configuration")
        if name not in self._mixtures:
            from mixture_tools import MixtureTools
            path = self.private_dir / "mixture_tools" / (name + ".npz")
            with np.load(path) as stored:
                ids = list(map(int, stored["ids"]))
                if ids != self._ids():
                    raise ValueError("Mixture and background catalog order differ")
                self._mixtures[name] = MixtureTools(ids, stored["probability"], stored["prior"])
        return self._mixtures[name]

    def build(self, history, candidates, initial, evidence=None, required_features=None, group_record=None):
        """Build only explicitly requested extra feature families.

        Always returns community + normalized initial-GLM values. Other values
        are added only when requested. Missing raw affinity/group evidence is
        represented by a neutral margin with an explicit invalid flag; metadata
        and fitted graph weights are never silently replaced if their files fail.
        """
        required = _required(required_features)
        history, candidates = self._validate(history, candidates)
        output = recovered_output(initial or {})
        rows, qbase, affinity_error = None, None, None
        if evidence is not None:
            if not isinstance(evidence, dict):
                raise ValueError("evidence must be a cached evidence record")
            if evidence.get("error"):
                affinity_error = str(evidence["error"])
            else:
                try:
                    candidate_rows = evidence.get("rows")
                    # When the record carries explicit order, align both columns
                    # before passing rows to original formulas. Legacy records
                    # without order are already in the supplied candidate order.
                    if evidence.get("candidates") is not None:
                        source_order = list(map(int, evidence["candidates"]))
                        if len(source_order) != 2 or set(source_order) != set(candidates):
                            raise ValueError("Affinity evidence belongs to another candidate pair")
                        take = [source_order.index(i) for i in candidates]
                        candidate_rows = [[row[j] for j in take] for row in candidate_rows]
                    qbase = qloo_features(candidate_rows, [r for i, r in history])
                    rows = candidate_rows
                except (ValueError, TypeError, IndexError, KeyError) as error:
                    affinity_error = str(error)
        use_affinity = bool(required & AFFINITY)
        values, extra, community_trace = self.community.evaluate(history, candidates, rows if use_affinity else None)
        values["glm_log_odds"] = output["log_odds"] if output else 0.
        values["cohort_gated_glm"] = values["glm_log_odds"] * (1 - values["cohort_entropy"])
        trace = {"community": community_trace,
                 "feature_sources": {"requested": sorted(required), "uses_qloo": requires_qloo(required),
                                     "culture_metadata_loaded": False, "culture_graph_loaded": False,
                                     "mixture_weights": []},
                 "source_availability": {"community": True, "initial_model": output is not None,
                                         "affinity": rows is not None, "groups": None,
                                         "metadata": None, "graph": None, "mixtures": {},
                                         "required_tools_valid": not use_affinity or rows is not None}}
        if use_affinity:
            available = dict(zip(QBASE, qbase)) if rows is not None else {}
            available.update(extra)
            values.update({name: float(available.get(name, 0.)) for name in required & AFFINITY})
        if affinity_error:
            trace["affinity_error"] = affinity_error
        if required & METADATA:
            available = self._cultural().evaluate(history, candidates)
            values.update({name: available[name] for name in required & METADATA})
            trace["feature_sources"]["culture_metadata_loaded"] = True
            trace["source_availability"]["metadata"] = True
        if required & GRAPH:
            available, graph_trace = self._cultural_graph().evaluate(history, candidates)
            values.update({name: available[name] for name in required & GRAPH})
            trace["culture_graph"] = {name: graph_trace[name] for name in required & GRAPH}
            trace["feature_sources"]["culture_graph_loaded"] = True
            trace["source_availability"]["graph"] = True
        interactions = required & (INTERACTION_CF | INTERACTION_MIXED)
        if interactions:
            mixed = bool(required & INTERACTION_MIXED)
            available = self._interaction(mixed).evaluate(history, candidates)
            values.update({name: available[name] for name in interactions})
            trace["feature_sources"]["interaction"] = "CF + Qloo curated" if mixed else "CF only"
            trace["feature_sources"]["culture_metadata_loaded"] |= mixed
            if mixed:
                trace["source_availability"]["metadata"] = True
        for name in sorted(required & MIXTURE):
            values[name], mixture_trace = self._mixture(name).evaluate(history, candidates)
            trace.setdefault("mixture", {})[name] = mixture_trace
            trace["feature_sources"]["mixture_weights"].append(name)
            trace["source_availability"]["mixtures"][name] = True
        if required & GROUPS:
            from grouped_qloo import features_from_rows
            group_error = None
            group_values = {}
            try:
                if group_record is None:
                    raise ValueError("Grouped Qloo evidence was not supplied")
                if evidence is None or not evidence.get("rows"):
                    raise ValueError("Pooled-likes Qloo row was not supplied")
                # Group features only require pooled row zero, and can remain
                # usable when a later singleton row makes full affinity invalid.
                group_values = features_from_rows(history, candidates, group_record,
                                                  {"rows": evidence["rows"],
                                                   "candidates": evidence.get("candidates", candidates)})
            except (ValueError, TypeError, IndexError, KeyError) as error:
                group_error = str(error)
            values.update({name: float(group_values.get(name, 0.)) for name in required & GROUPS})
            trace["groups"] = {"valid": group_error is None, "error": group_error}
            trace["source_availability"]["groups"] = group_error is None
            trace["source_availability"]["required_tools_valid"] &= group_error is None
        absent = required - values.keys()
        if absent:
            raise ValueError("Requested feature unavailable: " + ", ".join(sorted(absent)))
        if any(not np.isfinite(float(values[name])) for name in required):
            raise ValueError("Requested feature is non-finite")
        return {"values": values, "trace": trace, "initial_valid": output is not None,
                "affinity_valid": rows is not None}
