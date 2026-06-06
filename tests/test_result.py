import math

import pytest

from adaptive_reservoir import AdaptiveChannels, AdaptiveStepResult, StepMetrics, TraceNorms


def test_default_channels_are_normalized() -> None:
    channels = AdaptiveChannels()

    assert channels.novelty == 0.0
    assert channels.stability == 1.0
    assert channels.drift_pressure == 0.0
    assert channels.confidence == 0.0
    assert channels.saturation == 0.0


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("novelty", -0.1),
        ("novelty", 1.1),
        ("stability", math.nan),
        ("drift_pressure", math.inf),
        ("confidence", -math.inf),
        ("saturation", 1.1),
    ],
)
def test_channel_values_must_be_normalized(field_name: str, value: float) -> None:
    with pytest.raises(ValueError, match=f"{field_name} must be in the range"):
        AdaptiveChannels(**{field_name: value})


def test_step_metrics_accepts_zero_samples_seen() -> None:
    metrics = StepMetrics(samples_seen=0)

    assert metrics.samples_seen == 0
    assert metrics.prediction_available is False
    assert metrics.target_available is False
    assert metrics.readout_updated is False


def test_step_metrics_accepts_diagnostic_values() -> None:
    metrics = StepMetrics(
        samples_seen=1,
        state_norm=1.0,
        state_delta=0.5,
        feature_norm=2.0,
        saturation_rate=0.25,
        trace_norms=TraceNorms(fast=0.1, mid=0.2, slow=0.3),
    )

    assert metrics.state_norm == 1.0
    assert metrics.state_delta == 0.5
    assert metrics.feature_norm == 2.0
    assert metrics.saturation_rate == 0.25
    assert metrics.trace_norms == TraceNorms(fast=0.1, mid=0.2, slow=0.3)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"samples_seen": -1}, "samples_seen must be non-negative"),
        ({"samples_seen": 0, "state_norm": -0.1}, "state_norm must be finite"),
        ({"samples_seen": 0, "state_delta": -0.1}, "state_delta must be finite"),
        ({"samples_seen": 0, "feature_norm": math.nan}, "feature_norm must be finite"),
        ({"samples_seen": 0, "saturation_rate": 1.1}, "saturation_rate must be in"),
        (
            {"samples_seen": 0, "trace_norms": TraceNorms(fast=-0.1, mid=0.0, slow=0.0)},
            "trace_norms.fast must be finite",
        ),
        (
            {"samples_seen": 0, "trace_norms": TraceNorms(fast=0.0, mid=math.nan, slow=0.0)},
            "trace_norms.mid must be finite",
        ),
        (
            {"samples_seen": 0, "trace_norms": TraceNorms(fast=0.0, mid=0.0, slow=math.inf)},
            "trace_norms.slow must be finite",
        ),
        ({"samples_seen": 0, "prediction_error": -0.1}, "prediction_error must be finite"),
        ({"samples_seen": 0, "us_per_sample": math.inf}, "us_per_sample must be finite"),
    ],
)
def test_invalid_step_metrics_raise_clear_errors(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        StepMetrics(**kwargs)  # type: ignore[arg-type]


def test_adaptive_step_result_accepts_valid_values() -> None:
    result = AdaptiveStepResult(
        prediction=None,
        features=(0.1, -0.2),
        channels=AdaptiveChannels(),
        metrics=StepMetrics(samples_seen=1),
    )

    assert result.features == (0.1, -0.2)
    assert result.state is None


@pytest.mark.parametrize(
    ("prediction", "features", "message"),
    [
        (math.nan, (0.1,), "prediction must be finite"),
        (None, (math.inf,), "features must contain only finite values"),
    ],
)
def test_invalid_step_result_values_raise_clear_errors(
    prediction: float | None,
    features: tuple[float, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        AdaptiveStepResult(
            prediction=prediction,
            features=features,
            channels=AdaptiveChannels(),
            metrics=StepMetrics(samples_seen=1),
        )
