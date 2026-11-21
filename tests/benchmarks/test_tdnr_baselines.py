"""TDNR-01 Stage 2A: causal baseline contracts."""

from __future__ import annotations

import numpy as np
import pytest

from adaptive_reservoir.benchmarks.tdnr_baselines import (
    LagBufferRidgeBaseline,
    PersistenceBaseline,
    ZeroBaseline,
)
from adaptive_reservoir.benchmarks.temporal_drift_noise import (
    TemporalDriftNoiseConfig,
    generate_temporal_drift_noise_stream,
)

BASELINES = (ZeroBaseline, PersistenceBaseline, LagBufferRidgeBaseline)


def _stream():
    return generate_temporal_drift_noise_stream(
        TemporalDriftNoiseConfig(
            seed=17,
            n_samples=96,
            drift_at=48,
            delay_before=3,
            delay_after=12,
            input_noise=0.2,
            target_noise=0.1,
        )
    )


def _predictions(model, rows) -> np.ndarray:
    return np.asarray(
        [model.step(sample, target) for sample, target in rows],
        dtype=np.float64,
    )


def test_zero_always_predicts_zero() -> None:
    model = ZeroBaseline()
    assert model.name == "zero"
    assert model.step((1.0, 0.0), 4.0) == 0.0
    assert model.step((-3.0, 1.0), -8.0) == 0.0
    assert model.step((2.0, -1.0), 99.0) == 0.0


def test_persistence_uses_previous_training_target_only() -> None:
    model = PersistenceBaseline()
    assert model.name == "persistence"
    assert model.step((1.0, 0.0), 4.0) == 0.0
    assert model.step((3.0, 2.0), -2.5) == 4.0
    assert model.step((5.0, 2.0), 9.0) == -2.5
    assert model.step((0.0, -5.0), 0.0) == 9.0


def test_ridge_defaults_and_cold_start() -> None:
    model = LagBufferRidgeBaseline()
    assert model.name == "lag_buffer_ridge"
    assert model.max_lag == 16
    assert model.window_size == 128
    assert model.alpha == 0.001
    assert model.samples_seen == 0
    assert model.window_count == 0

    # The current target cannot influence the first prediction.
    assert model.step((1.0, 0.0), 3.0) == 0.0
    assert model.samples_seen == 1
    assert model.window_count == 1

    # With one supervised sample, the unpenalized intercept is its target.
    assert model.step((1.0, 0.0), -2.0) == pytest.approx(3.0, abs=1e-10)
    assert model.samples_seen == 2


def test_ridge_matches_independent_batch_normal_equations() -> None:
    """Check lag order, zero padding, intercept, refits, and eviction."""
    observations = [0.2, -0.4, 0.7, 0.1, -0.9, 0.6, 0.3, -0.2]
    targets = [1.0, 2.0, -1.0, 0.5, 3.0, -0.5, 1.5, 0.25]
    model = LagBufferRidgeBaseline(max_lag=2, window_size=3, alpha=0.1)
    feature_history: list[np.ndarray] = []
    target_history: list[float] = []

    for index, (observation, target) in enumerate(
        zip(observations, targets, strict=True)
    ):
        features = np.array(
            [observations[index - lag] if index >= lag else 0.0 for lag in range(3)],
            dtype=np.float64,
        )

        if not feature_history:
            expected = 0.0
        else:
            recent_features = np.asarray(feature_history[-3:], dtype=np.float64)
            recent_targets = np.asarray(target_history[-3:], dtype=np.float64)
            design = np.column_stack(
                (recent_features, np.ones(len(recent_features), dtype=np.float64))
            )
            penalty = np.diag([0.1, 0.1, 0.1, 0.0])
            coefficients = np.linalg.solve(
                design.T @ design + penalty,
                design.T @ recent_targets,
            )
            expected = float(np.dot(features, coefficients[:3]) + coefficients[3])

        actual = model.step((observation, 999.0), target)
        assert actual == pytest.approx(expected, rel=1e-9, abs=1e-9)
        feature_history.append(features)
        target_history.append(target)
        assert model.samples_seen == index + 1
        assert model.window_count == min(index + 1, 3)


def test_ridge_supports_current_observation_only() -> None:
    model = LagBufferRidgeBaseline(max_lag=0, window_size=2, alpha=0.01)
    assert model.step((2.0, 0.0), 4.0) == 0.0
    assert model.step((3.0, 1.0), 5.0) == pytest.approx(4.0, abs=1e-10)
    assert model.samples_seen == 2
    assert model.window_count == 2


@pytest.mark.parametrize("factory", BASELINES)
def test_replay_is_deterministic(factory) -> None:
    rows = list(_stream().iter_training())
    first = _predictions(factory(), rows)
    second = _predictions(factory(), rows)

    np.testing.assert_array_equal(first, second)
    assert first.shape == (len(rows),)
    assert np.all(np.isfinite(first))


@pytest.mark.parametrize("factory", BASELINES)
def test_future_changes_cannot_affect_past_predictions(factory) -> None:
    rows = list(_stream().iter_training())
    altered = list(rows)
    boundary = 24
    for index in range(boundary, len(altered)):
        altered[index] = ((1000.0 + index, -500.0), -100.0 - index)

    original = _predictions(factory(), rows)
    changed = _predictions(factory(), altered)
    np.testing.assert_array_equal(original[:boundary], changed[:boundary])


@pytest.mark.parametrize("factory", BASELINES)
def test_current_target_does_not_affect_its_own_prediction(factory) -> None:
    rows = list(_stream().iter_training())
    prefix = rows[:12]
    current_sample, current_target = rows[12]
    first = factory()
    second = factory()
    _predictions(first, prefix)
    _predictions(second, prefix)

    prediction_a = first.step(current_sample, current_target)
    prediction_b = second.step(current_sample, current_target + 50.0)
    assert prediction_a == prediction_b

    # Persistence and ridge must admit that difference on a later step.
    next_sample, next_target = rows[13]
    later_a = first.step(next_sample, next_target)
    later_b = second.step(next_sample, next_target)
    if factory is not ZeroBaseline:
        assert later_a != pytest.approx(later_b, abs=1e-8)


def test_ridge_ignores_redundant_difference_component() -> None:
    observations = [0.1, 0.4, -0.2, 0.8, -0.7, 0.3]
    targets = [1.0, -1.0, 0.5, 2.0, -0.5, 0.0]
    first = LagBufferRidgeBaseline(max_lag=3, window_size=4)
    second = LagBufferRidgeBaseline(max_lag=3, window_size=4)

    for observation, target in zip(observations, targets, strict=True):
        prediction_a = first.step((observation, 0.0), target)
        prediction_b = second.step((observation, 1000.0), target)
        assert prediction_a == prediction_b


def test_ridge_does_not_alias_mutable_input_arrays() -> None:
    first = LagBufferRidgeBaseline(max_lag=2, window_size=3)
    second = LagBufferRidgeBaseline(max_lag=2, window_size=3)
    sample = np.array([0.5, 0.0], dtype=np.float64)

    first.step(sample, 2.0)
    second.step((0.5, 0.0), 2.0)
    sample[:] = 999.0

    assert first.step((0.2, -0.3), 1.0) == second.step((0.2, -0.3), 1.0)


@pytest.mark.parametrize("factory", BASELINES)
def test_rejected_step_does_not_change_state(factory) -> None:
    model = factory()
    reference = factory()
    assert model.step((0.2, 0.0), 1.0) == reference.step((0.2, 0.0), 1.0)

    with pytest.raises(ValueError):
        model.step((float("nan"), 0.0), 2.0)
    with pytest.raises(ValueError):
        model.step((0.3, 0.0), float("inf"))

    assert model.step((0.4, 0.2), -1.0) == reference.step((0.4, 0.2), -1.0)


@pytest.mark.parametrize("factory", BASELINES)
@pytest.mark.parametrize(
    "sample",
    [
        (),
        (1.0,),
        (1.0, 2.0, 3.0),
        ((1.0, 2.0),),
        (float("nan"), 0.0),
        (1.0, float("inf")),
        (float("-inf"), 0.0),
        ("invalid", 0.0),
        None,
    ],
)
def test_invalid_sample_is_rejected(factory, sample) -> None:
    with pytest.raises(ValueError):
        factory().step(sample, 1.0)


@pytest.mark.parametrize("factory", BASELINES)
@pytest.mark.parametrize(
    "target",
    [float("nan"), float("inf"), float("-inf"), "invalid", None, True],
)
def test_invalid_target_is_rejected(factory, target) -> None:
    with pytest.raises(ValueError):
        factory().step((1.0, 0.0), target)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_lag": -1}, "max_lag"),
        ({"max_lag": True}, "max_lag"),
        ({"max_lag": 1.5}, "max_lag"),
        ({"window_size": 0}, "window_size"),
        ({"window_size": -2}, "window_size"),
        ({"window_size": True}, "window_size"),
        ({"window_size": 2.5}, "window_size"),
        ({"alpha": 0.0}, "alpha"),
        ({"alpha": -0.1}, "alpha"),
        ({"alpha": float("nan")}, "alpha"),
        ({"alpha": float("inf")}, "alpha"),
        ({"alpha": True}, "alpha"),
        ({"alpha": "0.001"}, "alpha"),
    ],
)
def test_invalid_ridge_configuration_is_rejected(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        LagBufferRidgeBaseline(**kwargs)


def test_baselines_do_not_change_global_numpy_rng_state() -> None:
    rows = list(_stream().iter_training())
    before = np.random.get_state()
    for factory in BASELINES:
        _predictions(factory(), rows)
    after = np.random.get_state()

    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
