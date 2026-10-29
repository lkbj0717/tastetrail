"""Development-only Schur-product cultural interaction experiment; no APIs.

The two factors are explicit Gram matrices, not raw Qloo affinity values.
K_CF = normalized MF32 item embeddings @ their transpose.
K_Q = the existing curated Qloo-tag TF-IDF Gram matrix.
K_mix = .5 K_CF + .5 (K_CF * K_Q), where * is elementwise multiplication.

Four fixed tool features: CF and interaction, each with fold-in ridge 1 or 10.
Four fusion families x three calibration ridges = twelve configurations. All
selection uses D/E development only, with nested calibration ridge selection.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np


def normalized_gram(embeddings):
    z = np.asarray(embeddings, dtype=float)
    if z.ndim != 2 or not np.all(np.isfinite(z)):
        raise ValueError("Expected a finite item embedding matrix")
    z = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
    return z @ z.T


def interaction_kernel(cf_gram, cultural_gram):
    """The Schur product theorem applies only after validating both Grams."""
    c, q = np.asarray(cf_gram, float), np.asarray(cultural_gram, float)
    if c.ndim != 2 or c.shape[0] != c.shape[1] or c.shape != q.shape:
        raise ValueError("Aligned square Gram matrices are required")
    for name, k in [("CF", c), ("Qloo metadata", q)]:
        if not np.all(np.isfinite(k)) or not np.allclose(k, k.T, atol=1e-10):
            raise ValueError(f"{name} must be finite and symmetric")
        if float(np.linalg.eigvalsh(k).min()) < -1e-8:
            raise ValueError(f"{name} is not a positive semidefinite Gram")
    return .5 * c + .5 * (c * q)


class InteractionTools:
    def __init__(self, ids, cf_gram, cultural_gram, item_bias):
        self.ids = list(map(int, ids))
        if len(self.ids) != len(set(self.ids)):
            raise ValueError("Duplicate item IDs")
        self.index = {mid: i for i, mid in enumerate(self.ids)}
        self.cf = np.asarray(cf_gram, float)
        self.culture = np.asarray(cultural_gram, float)
        self.mixed = interaction_kernel(self.cf, self.culture)
        self.bias = np.asarray(item_bias, float)
        if self.cf.shape != (len(ids), len(ids)) or self.bias.shape != (len(ids),):
            raise ValueError("Item vectors, kernels, and biases must align")
        if not np.all(np.isfinite(self.bias)):
            raise ValueError("Bias must be finite")

    @classmethod
    def from_existing(cls):
        from audit_features import CulturalMetadataTools, OLD
        cultural = CulturalMetadataTools()
        arrays = np.load(OLD / "tools.npz")
        ids = list(map(int, arrays["ids"]))
        if ids != cultural.ids:
            raise ValueError("MF and Qloo catalog ordering must match exactly")
        if len(arrays["raw"]) != 3640:
            raise ValueError("Only the current disjoint background is allowed")
        return cls(ids, normalized_gram(arrays["mf_32_10"]),
                   cultural.kernels["curated"], arrays["item_bias"])

    def evaluate(self, history, candidates):
        """Accept only eight visible pairs and two item IDs, never a case dict."""
        if isinstance(history, dict) or len(history) != 8 or len(candidates) != 2:
            raise ValueError("Exactly eight history ratings and two candidates required")
        if any(len(row) != 2 for row in history):
            raise ValueError("History rows must contain item and visible rating only")
        movie_ids = [int(row[0]) for row in history]
        candidate_ids = list(map(int, candidates))
        if len(set(movie_ids + candidate_ids)) != 10:
            raise ValueError("History and candidate items must be distinct")
        r = np.asarray([row[1] for row in history], float)
        if (not np.all(np.isin(r, [1., 2., 4., 5.])) or
                np.sum(r >= 4) != 5 or np.sum(r <= 2) != 3):
            raise ValueError("Expected five likes and three dislikes")
        h = [self.index[i] for i in movie_ids]
        c = [self.index[i] for i in candidate_ids]
        y = r - r.mean() - self.bias[h]
        out = {}
        for kind, kernel in [("cf", self.cf), ("mixed", self.mixed)]:
            hh = kernel[np.ix_(h, h)]
            hc = kernel[np.ix_(h, c)]
            for ridge in [1., 10.]:
                alpha = np.linalg.solve(hh + ridge * np.eye(8), y)
                score = self.bias[c] + hc.T @ alpha
                out[f"interaction_{kind}_ridge_{ridge:g}"] = float(score[0] - score[1])
        return out


def run_development():
    # These imports perform no I/O until explicitly called below.
    from common import ROOT, PRIVATE, OLD, read, save, sha, dev_cases, recovered_output
    from calibrate import nested_family, folds_for, loss

    report_path = ROOT / "research/interaction_development.json"
    feature_path = PRIVATE / "interaction_features_dev.json"
    oof_path = PRIVATE / "interaction_oof_dev.json"
    tool_path = PRIVATE / "interaction_tools.npz"
    for p in [report_path, feature_path, oof_path, tool_path]:
        if p.exists():
            raise RuntimeError(f"Refuse overwrite of an existing experiment: {p.name}")
    start = time.monotonic()
    tool = InteractionTools.from_existing()
    cs = dev_cases()  # roles filtered immediately; C/F/R/N labels never inspected
    if len(cs) != 219 or any(c["split"] != "dev" or c["id"][0] not in "DE" for c in cs):
        raise ValueError("Expected precisely 219 D/E development cases")
    old = read(OLD / "features_dev.json")
    features, model_failures, repairs = {}, [], []
    max_swap_error, max_history_permutation_error = 0., 0.
    for case in cs:
        ident = case["id"]
        initial = read(OLD / "predictions" / f"{ident}.json")
        output = recovered_output(initial)
        if output is None:
            model_failures.append(ident)
        elif output["parser_flags"]:
            repairs.append(ident)
        h = case["likes"] + case["dislikes"]
        c = case["candidates"]
        values = tool.evaluate(h, c)
        rev = tool.evaluate(h, c[::-1])
        perm = tool.evaluate(h[::-1], c)
        max_swap_error = max(max_swap_error, max(abs(values[n] + rev[n]) for n in values))
        max_history_permutation_error = max(max_history_permutation_error,
                                           max(abs(values[n] - perm[n]) for n in values))
        features[ident] = {**old[ident]["values"], **values,
                           "glm_log_odds": output["log_odds"] if output else 0.}
        # Same unavailable-affinity convention as existing development controls.
        for name in ["signed_history_log_ratio", "mf_weighted_qloo"]:
            features[ident].setdefault(name, 0.)
    assert max_swap_error < 1e-10 and max_history_permutation_error < 1e-10

    base = ["glm_log_odds", "mf_32_10", "community_margin"]
    qloo = ["signed_history_log_ratio", "mf_weighted_qloo"]
    cf = ["interaction_cf_ridge_1", "interaction_cf_ridge_10"]
    mixed = ["interaction_mixed_ridge_1", "interaction_mixed_ridge_10"]
    families = {"base3": base, "old_qloo5": base + qloo,
                "cf_control5": base + cf, "hadamard_culture5": base + mixed}
    y = np.array([c["candidates"][0] == c["preferred"] for c in cs], float)
    fold = folds_for(cs)
    trials, models, oofs = [], {}, {}
    for name, names in families.items():
        X = np.array([[features[c["id"]][n] for n in names] for c in cs])
        pred, model, inner_ridges = nested_family(X, y, fold)
        decisions = pred >= .5
        row = {"family": name, "features": names,
               "correct": int(np.sum(decisions == y)), "total": len(cs),
               "accuracy": float(np.mean(decisions == y)), "log_loss": loss(y, pred),
               "nested_selected_ridges": inner_ridges, "final_ridge": model["ridge"]}
        trials.append(row)
        models[name] = {"features": names, "model": model}
        oofs[name] = {c["id"]: float(v) for c, v in zip(cs, pred)}
    rows = {r["family"]: r for r in trials}
    a = np.array([oofs["cf_control5"][c["id"]] >= .5 for c in cs])
    q = np.array([oofs["hadamard_culture5"][c["id"]] >= .5 for c in cs])
    paired = {"fixed": int(np.sum((q == y) & (a != y))),
              "harmed": int(np.sum((q != y) & (a == y))),
              "unchanged": int(np.sum(q == a)),
              "accuracy_difference_pp": 100 * (rows["hadamard_culture5"]["accuracy"] - rows["cf_control5"]["accuracy"]),
              "log_loss_difference": rows["hadamard_culture5"]["log_loss"] - rows["cf_control5"]["log_loss"]}

    save(feature_path, features)
    save(oof_path, oofs)
    np.savez_compressed(tool_path, ids=np.array(tool.ids), cf=tool.cf,
                        culture=tool.culture, mixed=tool.mixed, item_bias=tool.bias)
    tool_path.chmod(0o600)
    report = {
        "development_only": True, "task": "Hadamard cultural interaction vs identical CF-only fold-in",
        "data_roles": "D/E219 only; current3640 background item embeddings",
        "calls_to_apis": 0, "additional_llm_tokens": 0,
        "tool_feature_count": 4, "fusion_family_count": 4,
        "calibration_ridges": [1., 10., 100.], "full_configuration_count": 12,
        "selection": "Within each fixed family: five outer folds; ridge selected on remaining inner folds by log loss. Family comparison remains exploratory development.",
        "fold_seed": 20261025, "foldin_ridges": [1., 10.],
        "kernel_formula": "Kmix=.5*K_CF+.5*(K_CF elementwise K_Q); K_Q is curated-tag TF-IDF Gram, not raw affinity",
        "kernel_diagnostics": {"cf_min_eigenvalue": float(np.linalg.eigvalsh(tool.cf).min()),
                               "culture_min_eigenvalue": float(np.linalg.eigvalsh(tool.culture).min()),
                               "mixed_min_eigenvalue": float(np.linalg.eigvalsh(tool.mixed).min()),
                               "zero_cultural_diagonal_items": int(np.sum(np.diag(tool.culture) < 1e-12))},
        "candidate_swap_max_error": max_swap_error,
        "history_permutation_max_error": max_history_permutation_error,
        "baseline_unparseable_count": len(model_failures), "confidence_normalization_count": len(repairs),
        "trials": trials, "matched_hadamard_vs_cf": paired,
        "development_best_by_log_loss": min(trials, key=lambda r: (r["log_loss"], -r["correct"], r["family"]))["family"],
        "models": models,
        "source_hashes": {"current_background_tools": sha(OLD / "tools.npz"),
                          "qloo_metadata": sha(PRIVATE / "qloo_metadata.json"),
                          "source_code": sha(Path(__file__)),
                          "shared_nested_calibration": sha(ROOT / "calibrate.py"),
                          "development_features": sha(OLD / "features_dev.json")},
        "private_feature_sha256": sha(feature_path), "private_tools_sha256": sha(tool_path),
        "seconds": time.monotonic() - start,
        "limitations": ["No confirmation or historical formal outcomes read or scored.",
                        "Finite-sample development selection is not proof of generalization or70%.",
                        "Schur PSD guarantee applies to the two constructed Gram matrices only.",
                        "No LOO or routing variants were added to this small experiment.",
                        "All219 cases remain in the denominator; failure handling matches shared development controls."]}
    save(report_path, report)
    print(json.dumps({"trials": trials, "matched_hadamard_vs_cf": paired,
                      "api_calls": 0, "seconds": report["seconds"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run_development()
