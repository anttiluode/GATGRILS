import numpy as np

from gatgrils.grammar import (
    align_labels,
    bag_coordinate_nll,
    fit_bag_coordinate_model,
    fit_response_coordinates,
    fit_transition_model,
    transition_nll,
)


def test_response_clustering_is_deterministic_and_selects_from_candidates():
    rng = np.random.default_rng(4)
    a = rng.normal([-2.0, 0.0], 0.15, size=(80, 2))
    b = rng.normal([2.0, 0.0], 0.15, size=(80, 2))
    features = np.vstack([a[:60], b[:60]])
    validation = np.vstack([a[60:], b[60:]])
    m1 = fit_response_coordinates(features, validation, (2, 3, 4), seed=9)
    m2 = fit_response_coordinates(features, validation, (2, 3, 4), seed=9)
    assert m1.k in {2, 3, 4}
    assert m1.k == m2.k
    assert np.array_equal(m1.predict(features), m2.predict(features))
    assert np.allclose(m1.centers, m2.centers)


def test_align_labels_recovers_permuted_coordinate_sequence():
    reference = np.array([0, 1, 2, 0, 2, 1, 1, 0, 2], dtype=np.int64)
    permutation = np.array([2, 0, 1], dtype=np.int64)
    candidate = permutation[reference]
    aligned, mapping = align_labels(reference, candidate)
    assert np.array_equal(aligned, reference)
    assert set(mapping) == {0, 1, 2}


def test_ordered_transition_model_beats_bag_on_order_sensitive_fixture():
    # Same coordinate counts, highly structured order. An event-conditioned
    # model sees the alternating grammar; a bag model only sees marginals.
    coords = np.tile(np.array([0, 1, 0, 2], dtype=np.int64), 80)
    symbols = np.tile(np.array([0, 0, 1, 1], dtype=np.int64), 80)
    train_n = 220
    model = fit_transition_model(coords[:train_n], symbols[:train_n], smoothing=0.2)
    ordered = transition_nll(model, coords[train_n - 1 :], symbols[train_n - 1 :])
    bag = fit_bag_coordinate_model(coords[:train_n], coordinate_count=3, smoothing=0.2)
    bag_score = bag_coordinate_nll(bag, coords[train_n:])
    assert np.isfinite(ordered)
    assert np.isfinite(bag_score)
    assert ordered < bag_score


def test_bag_model_discards_order_but_preserves_counts():
    a = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2], dtype=np.int64)
    b = np.array([2, 1, 0, 2, 1, 0, 2, 1, 0], dtype=np.int64)
    assert np.array_equal(np.bincount(a, minlength=3), np.bincount(b, minlength=3))
    bag_a = fit_bag_coordinate_model(a, 3, smoothing=0.5)
    bag_b = fit_bag_coordinate_model(b, 3, smoothing=0.5)
    assert np.allclose(bag_a, bag_b)
