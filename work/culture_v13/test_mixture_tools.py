import numpy as np
import pytest

from mixture_tools import MixtureTools, fit_mixture, neighbor_kernel, weighted_feedback, with_graph_prior


def synthetic_ratings():
    rng = np.random.default_rng(41)
    r = rng.choice([0, 1, 2, 3, 4, 5], size=(80, 12), p=[.2, .1, .1, .2, .2, .2])
    r[:, 0], r[:, 1] = 4, 2
    return r


def test_neutral_missing_not_negative_and_leniency_balanced():
    r = synthetic_ratings()
    p, n, valid = weighted_feedback(r)
    assert np.all(n[(r == 0) | (r == 3)] == 0)
    assert np.all(p[(r == 0) | (r == 3)] == 0)
    assert np.allclose(p.sum(axis=1)[valid], n.sum(axis=1)[valid])
    replaced = r.copy()
    replaced[replaced == 3] = 0
    a, _ = fit_mixture(r, 3, iterations=3)
    b, _ = fit_mixture(replaced, 3, iterations=3)
    assert np.array_equal(a["probability"], b["probability"])


def test_no_graph_self_loop_and_zero_graph_matches_plain():
    rng = np.random.default_rng(9)
    z = rng.normal(size=(12, 4))
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    gram = z @ z.T
    a = neighbor_kernel(gram, neighbors=4)
    np.fill_diagonal(gram, 999)
    assert np.array_equal(a, neighbor_kernel(gram, neighbors=4))
    assert np.all(np.diag(a) == 0)
    arrays, _ = fit_mixture(synthetic_ratings(), 3, iterations=3)
    assert np.allclose(with_graph_prior(arrays), with_graph_prior(arrays, np.zeros((12, 12))))


def test_candidate_swap_history_order_and_no_hidden_label_argument():
    arrays, _ = fit_mixture(synthetic_ratings(), 3, iterations=3)
    tool = MixtureTools(range(12), arrays["probability"], arrays["prior"])
    history = [(0, 5), (1, 4), (2, 4), (3, 5), (4, 4), (5, 2), (6, 1), (7, 2)]
    forward, trace = tool.evaluate(history, [8, 9])
    reverse, _ = tool.evaluate(history[::-1], [9, 8])
    assert forward == pytest.approx(-reverse, abs=1e-12)
    assert sum(trace["posterior"]) == pytest.approx(1.)
    with pytest.raises(TypeError):
        tool.evaluate(history, [8, 9], preferred=8)
    with pytest.raises(ValueError):
        tool.evaluate(history, [7, 8])


def test_no_personal_association_means_no_popularity_only_margin():
    # Every cluster has the same probabilities: popular vs unpopular alone
    # must produce zero personalized preference, despite an absolute gap.
    p = np.array([[.7, .6, .8, .4, .3, .8, .3, .4, .95, .05]] * 3)
    tool = MixtureTools(range(10), p, np.array([.2, .3, .5]))
    history = [(0, 5), (1, 4), (2, 4), (3, 5), (4, 4), (5, 2), (6, 1), (7, 2)]
    value, _ = tool.evaluate(history, [8, 9])
    assert value == pytest.approx(0., abs=1e-12)
