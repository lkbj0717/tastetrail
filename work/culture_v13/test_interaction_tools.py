"""Kernel validity and input-boundary tests on synthetic public-size arrays."""
import numpy as np
import pytest

from interaction_tools import InteractionTools, interaction_kernel, normalized_gram


def fixture():
    rng = np.random.default_rng(104)
    z, q = rng.normal(size=(12, 4)), rng.normal(size=(12, 5))
    cf, culture = normalized_gram(z), normalized_gram(q)
    tool = InteractionTools(list(range(1, 13)), cf, culture, rng.normal(0, .2, 12))
    history = [(1, 5), (2, 4), (3, 5), (4, 4), (5, 5), (6, 1), (7, 2), (8, 1)]
    return tool, history, [9, 10], z, q


def test_schur_product_matches_tensor_feature_gram_and_is_psd():
    tool, _, _, z, q = fixture()
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    tensor = np.einsum('ij,ik->ijk', z, q).reshape(12, -1)
    assert np.allclose(tool.cf * tool.culture, tensor @ tensor.T, atol=1e-12)
    assert np.linalg.eigvalsh(tool.mixed).min() > -1e-10
    # Matrix multiplication is not the intended interaction.
    assert not np.allclose(tool.cf * tool.culture, tool.cf @ tool.culture)


def test_candidate_reversal_and_history_permutation_are_invariant():
    tool, h, c, _, _ = fixture()
    first, reverse, permuted = tool.evaluate(h, c), tool.evaluate(h, c[::-1]), tool.evaluate(h[::-1], c)
    assert len(first) == 4
    for key, value in first.items():
        assert np.isfinite(value)
        assert abs(value + reverse[key]) < 1e-11
        assert abs(value - permuted[key]) < 1e-11


def test_rejects_non_psd_raw_affinity_and_misalignment():
    with pytest.raises(ValueError, match='positive semidefinite'):
        interaction_kernel(np.eye(2), np.array([[1., 2.], [2., 1.]]))
    with pytest.raises(ValueError, match='Aligned'):
        interaction_kernel(np.eye(2), np.eye(3))
    with pytest.raises(ValueError, match='symmetric'):
        interaction_kernel(np.eye(2), np.array([[1., .2], [.3, 1.]]))


def test_visible_input_boundary_and_unknown_ids():
    tool, h, c, _, _ = fixture()
    with pytest.raises(ValueError, match='distinct'):
        tool.evaluate(h, [1, 9])
    with pytest.raises(ValueError, match='Exactly'):
        tool.evaluate({'likes': h, 'hidden_ratings': {9: 5}}, c)
    with pytest.raises(ValueError, match='visible rating only'):
        tool.evaluate([(*h[0], 'hidden-field')] + h[1:], c)
    with pytest.raises(KeyError):
        tool.evaluate(h, [9, 999])


def test_all_ones_cultural_gram_reduces_to_matched_cf_control():
    tool, h, c, _, _ = fixture()
    same = InteractionTools(tool.ids, tool.cf, np.ones_like(tool.cf), tool.bias)
    out = same.evaluate(h, c)
    for ridge in [1, 10]:
        assert abs(out[f'interaction_cf_ridge_{ridge}'] - out[f'interaction_mixed_ridge_{ridge}']) < 1e-12
