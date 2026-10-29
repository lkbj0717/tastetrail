"""Explicit-feedback Bernoulli taste mixtures, with matched graph priors.

Six prespecified configurations: K in {12,24} x {plain,CF,Qloo-curated}.
Only the 3640 independent background users train mixture sufficient statistics.
Three-star and missing entries are unobserved, never negative feedback.
No API calls, provider credentials, or confirmation-set loading occur here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
PRIVATE = Path.home() / ".local/share/tastetrail/culture_v13"
OLD = PRIVATE.parent / "culture_v11"
SEED = 20261030
CONFIGS = [(k, kind) for k in (12, 24) for kind in ("plain", "cf", "qloo")]
SETTINGS = {"seed": SEED, "K": [12, 24], "iterations_max": 30,
            "beta_strength": 20., "user_effective_feedback_cap": 40.,
            "graph_neighbors": 24, "graph_prior_residual_strength": .5,
            "fusion_ridge": 10., "graph_diagonal": "excluded"}


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


def logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p) - np.log1p(-p)


def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    p = np.exp(x)
    return p / p.sum(axis=-1, keepdims=True)


def weighted_feedback(raw, cap=40.):
    """Balance each user's two explicit feedback classes; no pseudo-negatives.

    A user without either sign cannot distinguish rating leniency from taste,
    so contributes no class-balanced mixture evidence. Such users are counted
    explicitly in provenance; no new negative labels are invented for them.
    """
    r = np.asarray(raw, float)
    if r.ndim != 2 or not np.isin(r, [0, 1, 2, 3, 4, 5]).all():
        raise ValueError("Expected a user x item matrix of ratings 1..5 or missing 0")
    y = r >= 4
    n = (r > 0) & (r <= 2)
    py, pn = y.sum(axis=1), n.sum(axis=1)
    informative = (py > 0) & (pn > 0)
    strength = np.minimum(py + pn, cap) * informative
    positives = y * (strength / (2 * np.maximum(py, 1)))[:, None]
    negatives = n * (strength / (2 * np.maximum(pn, 1)))[:, None]
    return positives, negatives, informative


def centered_probabilities(p, base, weights):
    """Remove one general likes-everything intercept from every group."""
    delta = logit(p) - logit(base)[None, :]
    mean = (delta @ weights) / max(float(weights.sum()), 1e-12)
    return np.clip(sigmoid(logit(base)[None, :] + delta - mean[:, None]), 1e-5, 1 - 1e-5)


def fit_mixture(raw, k, *, seed=SEED, iterations=30, beta=20., cap=40.):
    """Regularized generalized EM; fixed balance/intercept constraints are imposed.

    The E-step is a class-balanced Bernoulli likelihood. Beta posterior means
    provide the M-step, followed by a group-intercept constraint. Because of
    that constraint this is not claimed to maximize an unconstrained ordinary
    Bernoulli likelihood monotonically. All iteration diagnostics are retained.
    """
    positive, negative, informative = weighted_feedback(raw, cap)
    nusers, nitems = positive.shape
    if informative.sum() < k or k < 2 or iterations < 1:
        raise ValueError("Not enough users with both explicit feedback signs")
    total = positive + negative
    item_count = total.sum(axis=0)
    base = np.clip((positive.sum(axis=0) + 1) / (item_count + 2), .01, .99)
    rng = np.random.default_rng(seed)
    initial = rng.choice(np.flatnonzero(informative), k, replace=False)
    direction = (np.asarray(raw)[initial] >= 4).astype(float)
    direction -= ((np.asarray(raw)[initial] > 0) & (np.asarray(raw)[initial] <= 2))
    p = centered_probabilities(sigmoid(logit(base)[None, :] + 1.5 * direction), base, item_count)
    prior = np.ones(k) / k
    trace = []
    for iteration in range(iterations):
        score = positive @ np.log(p).T + negative @ np.log1p(-p).T + np.log(prior)[None, :]
        posterior = softmax(score)
        success = posterior.T @ positive
        count = posterior.T @ total
        prior = (posterior[informative].sum(axis=0) + 1) / (informative.sum() + k)
        updated = centered_probabilities((success + beta * base[None, :]) / (count + beta),
                                         base, item_count)
        change = float(np.max(np.abs(updated - p)))
        maximum = score.max(axis=1)
        likelihood = maximum + np.log(np.exp(score - maximum[:, None]).sum(axis=1))
        trace.append({"iteration": iteration + 1, "mean_weighted_log_likelihood": float(likelihood[informative].mean()),
                      "parameter_change_max": change,
                      "effective_group_min": float(posterior[informative].sum(axis=0).min()),
                      "effective_group_max": float(posterior[informative].sum(axis=0).max())})
        p = updated
        if change < 1e-5:
            break
    # Recompute the sufficient statistics from the final background posterior.
    score = positive @ np.log(p).T + negative @ np.log1p(-p).T + np.log(prior)[None, :]
    posterior = softmax(score)
    success = posterior.T @ positive
    count = posterior.T @ total
    prior = (posterior[informative].sum(axis=0) + 1) / (informative.sum() + k)
    p = centered_probabilities((success + beta * base[None, :]) / (count + beta), base, item_count)
    arrays = {"probability": p, "prior": prior, "base": base, "success": success,
              "count": count, "item_count": item_count}
    metrics = {"k": k, "background_users_loaded": nusers, "background_users_informative": int(informative.sum()),
               "excluded_no_positive_or_no_negative": int((~informative).sum()),
               "rated3_cells_ignored": int((np.asarray(raw) == 3).sum()),
               "missing_cells_ignored": int((np.asarray(raw) == 0).sum()),
               "explicit_positive_cells": int((np.asarray(raw) >= 4).sum()),
               "explicit_negative_cells": int(((np.asarray(raw) > 0) & (np.asarray(raw) <= 2)).sum()),
               "prior_group_entropy": float(-(prior * np.log(prior)).sum() / np.log(k)),
               "iterations": trace, "beta_strength": beta,
               "background_candidate_ratings": "Only disjoint background observations; target user's candidate ratings never enter."}
    return arrays, metrics


def neighbor_kernel(gram, neighbors=24):
    """Matched nonnegative cosine-squared neighbors, excluding item self-loops."""
    gram = np.asarray(gram, float)
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1] or not np.isfinite(gram).all():
        raise ValueError("Expected a finite square item kernel")
    similarities = np.maximum(gram, 0) ** 2
    np.fill_diagonal(similarities, 0.)
    weights = np.zeros_like(similarities)
    for i, row in enumerate(similarities):
        take = np.argsort(-row, kind="stable")[:neighbors]
        weights[i, take] = row[take]
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    return weights


def with_graph_prior(arrays, weights=None, *, beta=20., strength=.5):
    """Same group counts, same Beta mass; only the prior center differs.

    Qloo supplies public item relationships. It is never treated as a user's
    positive/negative feedback. A missing neighbor contributes no association.
    """
    base = arrays["base"]
    if weights is None:
        prior_probability = np.broadcast_to(base, arrays["probability"].shape)
    else:
        weights = np.asarray(weights, float)
        if weights.shape != (len(base), len(base)) or not np.allclose(np.diag(weights), 0):
            raise ValueError("Aligned graph weights must exclude self-loops")
        residual = logit(arrays["probability"]) - logit(base)[None, :]
        prior_probability = sigmoid(logit(base)[None, :] + strength * (residual @ weights.T))
    p = (arrays["success"] + beta * prior_probability) / (arrays["count"] + beta)
    return centered_probabilities(p, base, arrays["item_count"])


class MixtureTools:
    def __init__(self, ids, probability, prior):
        self.ids = list(map(int, ids))
        self.index = {movie: i for i, movie in enumerate(self.ids)}
        self.p = np.asarray(probability, float)
        self.prior = np.asarray(prior, float)
        if (len(self.index) != len(self.ids) or self.p.shape != (len(self.prior), len(self.ids))
                or not np.isfinite(self.p).all() or not ((self.p > 0) & (self.p < 1)).all()
                or not np.isfinite(self.prior).all() or not (self.prior > 0).all()
                or not np.isclose(self.prior.sum(), 1)):
            raise ValueError("Invalid mixture shape/probabilities")
        self.population = self.prior @ self.p

    def evaluate(self, history, candidates):
        """Eight visible ratings and two distinct item IDs only; no target label."""
        if isinstance(history, dict) or len(history) != 8 or len(candidates) != 2:
            raise ValueError("Eight visible history pairs and two candidates required")
        movies = [int(i) for i, r in history]
        candidate_ids = list(map(int, candidates))
        ratings = np.array([r for i, r in history], float)
        if (len(set(movies + candidate_ids)) != 10 or not np.isin(ratings, [1, 2, 4, 5]).all()
                or np.sum(ratings >= 4) != 5 or np.sum(ratings <= 2) != 3):
            raise ValueError("Five likes, three dislikes, and distinct unseen candidates required")
        h = [self.index[i] for i in movies]
        c = [self.index[i] for i in candidate_ids]
        positive, negative = ratings >= 4, ratings <= 2
        score = np.log(self.prior)
        score = score + 4 * np.log(self.p[:, np.asarray(h)[positive]]).mean(axis=1)
        score = score + 4 * np.log1p(-self.p[:, np.asarray(h)[negative]]).mean(axis=1)
        posterior = softmax(score)
        preference = posterior @ self.p[:, c]
        residual = logit(preference) - logit(self.population[c])
        return float(residual[0] - residual[1]), {
            "posterior": posterior.tolist(),
            "posterior_entropy": float(-(posterior * np.log(np.maximum(posterior, 1e-12))).sum() / np.log(len(posterior))),
            "candidate_like_probabilities": preference.tolist(),
            "population_like_probabilities": self.population[c].tolist(),
            "candidate_personalized_log_odds": residual.tolist(),
        }


def _save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    if PRIVATE in path.parents:
        path.chmod(0o600)


def _model_name(k, kind):
    return f"bernoulli_k{k}_{kind}"


def _load_development():
    """Decode only D/E objects from the shared pretty-printed case manifest.

    The storage file also contains historical formal cases. Those JSON objects
    are discarded as text before decoding, so no C/F/R preference fields are
    inspected or used by this experiment. N is stored elsewhere and not read.
    """
    import re
    text = (OLD / "cases.json").read_text()
    chunks = re.findall(r"(?ms)^  \{\n.*?^  \}(?:,)?$", text)
    cases = []
    for chunk in chunks:
        if not re.search(r'"id": "[DE]\d{3}"', chunk):
            continue
        row = json.loads(chunk.rstrip(","))
        if row["split"] != "dev" or row["id"][0] not in {"D", "E"}:
            raise ValueError("Development case filter failed")
        cases.append(row)
    if len(cases) != 219:
        raise ValueError("Expected fixed 219-case D/E development set")
    return cases


def train_and_evaluate():
    """Six fixed mixtures; nested folds choose K within each graph condition."""
    from audit_features import CulturalMetadataTools
    from common import recovered_output
    from calibrate import folds_for, loss
    from tastetrail.fusion import fit, probability

    start = time.monotonic()
    protocol = {**SETTINGS, "configurations": [{"k": k, "prior": kind} for k, kind in CONFIGS],
                "selection": "Outer fixed 5 folds; inner remaining 4 folds select K=12/24 within each prior type; ridge fixed10.",
                "no_new_model_calls": True, "no_new_qloo_calls": True,
                "target_input_fields": ["eight visible (movie ID,rating) pairs", "two candidate movie IDs"],
                "labels_used": "D/E development only; background users independently disjoint",
                "token_cost": 0, "excluded_splits": ["C", "F", "R", "N"],
                "warning": "All six results retained; this experiment does not promote a formal winner."}
    _save(ROOT / "research/mixture_protocol.json", protocol)
    original = dict(np.load(OLD / "tools.npz"))
    assert original["raw"].shape == (3640, 473)
    cultural = CulturalMetadataTools()
    ids = list(map(int, original["ids"]))
    assert ids == cultural.ids
    cf = original["mf_32_10"].astype(float)
    cf /= np.maximum(np.linalg.norm(cf, axis=1, keepdims=True), 1e-12)
    weights = {"plain": None, "cf": neighbor_kernel(cf @ cf.T),
               "qloo": neighbor_kernel(cultural.kernels["curated"])}
    tools, training = {}, []
    weights_folder = PRIVATE / "mixture_tools"
    weights_folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    for k in SETTINGS["K"]:
        t = time.monotonic()
        arrays, metrics = fit_mixture(original["raw"], k)
        metrics["seconds"] = time.monotonic() - t
        training.append(metrics)
        for kind in ["plain", "cf", "qloo"]:
            p = with_graph_prior(arrays, weights[kind])
            name = _model_name(k, kind)
            tools[name] = MixtureTools(ids, p, arrays["prior"])
            path = weights_folder / (name + ".npz")
            np.savez_compressed(path, ids=np.array(ids), probability=p, prior=arrays["prior"])
            path.chmod(0o600)
        print(json.dumps({"trained_k": k, "seconds": metrics["seconds"],
                          "iterations": len(metrics["iterations"]),
                          "informative_background": metrics["background_users_informative"]}), flush=True)
    cs = _load_development()
    base_values = json.loads((PRIVATE / "graph_features_dev.json").read_text())
    BASE = ["glm_log_odds", "mf_32_10", "community_margin"]
    all_values, traces = {}, {}
    swap_error = 0.
    for c in cs:
        ident = c["id"]
        raw = json.loads((OLD / "predictions" / (ident + ".json")).read_text())
        model = recovered_output(raw)
        if model is None:
            raise ValueError("Missing development baseline ranking")
        all_values[ident] = {n: base_values[ident][n] for n in BASE}
        all_values[ident]["glm_log_odds"] = model["log_odds"]
        traces[ident] = {}
        for name, tool in tools.items():
            margin, detail = tool.evaluate(c["likes"] + c["dislikes"], c["candidates"])
            reverse, _ = tool.evaluate(c["likes"] + c["dislikes"], c["candidates"][::-1])
            swap_error = max(swap_error, abs(margin + reverse))
            all_values[ident][name] = margin
            traces[ident][name] = detail
    _save(PRIVATE / "mixture_features_dev.json", all_values)
    _save(PRIVATE / "mixture_traces_dev.json", traces)
    y = np.array([c["candidates"][0] == c["preferred"] for c in cs], float)
    fold = folds_for(cs)
    matrices = {"base": np.array([[all_values[c["id"]][n] for n in BASE] for c in cs])}
    for name in tools:
        matrices[name] = np.array([[all_values[c["id"]][n] for n in BASE + [name]] for c in cs])
    oof, rows, models = {}, [], {}
    ridge = SETTINGS["fusion_ridge"]

    def summarize(name, p):
        return {"name": name, "total": len(y), "correct": int(((p >= .5) == y).sum()),
                "accuracy": float(((p >= .5) == y).mean()), "log_loss": loss(y, p)}

    for name, X in matrices.items():
        p = np.zeros(len(y))
        for f in range(5):
            tr, te = fold != f, fold == f
            p[te] = probability(fit(X[tr], y[tr], ridge), X[te])
        oof[name] = {c["id"]: float(v) for c, v in zip(cs, p)}
        rows.append(summarize(name, p))
        models[name] = {"features": BASE if name == "base" else BASE + [name],
                        "model": fit(X, y, ridge)}
    nested = []
    for kind in ["plain", "cf", "qloo"]:
        outer_p, selections = np.zeros(len(y)), []
        for outer in range(5):
            train, test = fold != outer, fold == outer
            trials = []
            for k in SETTINGS["K"]:
                X = matrices[_model_name(k, kind)]
                p = np.zeros(len(y))
                for inner in sorted(set(fold[train])):
                    a, b = train & (fold != inner), train & (fold == inner)
                    p[b] = probability(fit(X[a], y[a], ridge), X[b])
                trials.append((loss(y[train], p[train]), k))
            k = min(trials)[1]
            X = matrices[_model_name(k, kind)]
            outer_p[test] = probability(fit(X[train], y[train], ridge), X[test])
            selections.append({"outer_fold": outer, "selected_k": k,
                               "inner_trials": [{"k": q, "log_loss": v} for v, q in trials]})
        name = "nested_" + kind
        oof[name] = {c["id"]: float(v) for c, v in zip(cs, outer_p)}
        nested.append({**summarize(name, outer_p), "selections": selections})
    report = {"protocol": protocol, "training": training, "development": {"D": 120, "E": 99},
              "fixed_config_oof": rows, "nested_k_selection": nested,
              "candidate_swap_error_max": swap_error,
              "total_seconds": time.monotonic() - start,
              "extra_api_calls": 0, "extra_tokens": 0,
              "source_hashes": {"implementation": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                "background_tools": hashlib.sha256((OLD / "tools.npz").read_bytes()).hexdigest(),
                                "public_qloo_metadata": hashlib.sha256((PRIVATE / "qloo_metadata.json").read_bytes()).hexdigest()},
              "feature_file": "private culture_v13/mixture_features_dev.json",
              "weights_directory": "private culture_v13/mixture_tools"}
    _save(ROOT / "research/mixture_development.json", report)
    _save(ROOT / "mixture_candidate_config.json", {"models": models, "protocol": protocol})
    _save(PRIVATE / "mixture_dev_oof.json", oof)
    print(json.dumps({"fixed": rows, "nested": [{k: v for k, v in r.items() if k != "selections"} for r in nested],
                      "seconds": report["total_seconds"], "swap_error": swap_error}, indent=2), flush=True)


if __name__ == "__main__":
    train_and_evaluate()
