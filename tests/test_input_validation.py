from __future__ import annotations

import numpy as np
import pytest

from adaptive_reservoir import AdaptiveReservoir, ReservoirConfig


@pytest.mark.parametrize(
    "x",
    [
        [1.0, -1.0],
        (1.0, -1.0),
        np.array([1.0, -1.0], dtype=np.float32),
        np.array([1.0, -1.0], dtype=np.float64),
        [1, -1],
    ],
)
def test_step_accepts_1d_numeric_inputs(x: object) -> None:
    model = AdaptiveReservoir(_config())

    result = model.step(x)  # type: ignore[arg-type]

    assert result.metrics.samples_seen == 1


@pytest.mark.parametrize(
    "x",
    [
        [1.0, -1.0],
        (1.0, -1.0),
        np.array([1.0, -1.0], dtype=np.float32),
        np.array([1.0, -1.0], dtype=np.float64),
        [1, -1],
    ],
)
def test_predict_accepts_same_1d_numeric_inputs_as_step(x: object) -> None:
    model = AdaptiveReservoir(_config())

    prediction = model.predict(x)  # type: ignore[arg-type]

    assert isinstance(prediction, float)
    assert model.samples_seen == 0


@pytest.mark.parametrize(
    "x",
    [
        [1.0],
        [1.0, 2.0, 3.0],
    ],
)
def test_step_rejects_wrong_input_dim(x: object) -> None:
    model = AdaptiveReservoir(_config())

    with pytest.raises(ValueError, match="expected input_dim=2"):
        model.step(x)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "x",
    [
        [1.0],
        [1.0, 2.0, 3.0],
    ],
)
def test_predict_rejects_wrong_input_dim(x: object) -> None:
    model = AdaptiveReservoir(_config())

    with pytest.raises(ValueError, match="expected input_dim=2"):
        model.predict(x)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "x",
    [
        [[1.0, 2.0]],
        np.array([[1.0, 2.0]]),
        1.0,
        "1,2",
        b"1,2",
    ],
)
def test_step_rejects_non_1d_input(x: object) -> None:
    model = AdaptiveReservoir(_config())

    with pytest.raises(ValueError, match="x must be a 1D numeric vector"):
        model.step(x)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "x",
    [
        [[1.0, 2.0]],
        np.array([[1.0, 2.0]]),
        1.0,
        "1,2",
        b"1,2",
    ],
)
def test_predict_rejects_non_1d_input(x: object) -> None:
    model = AdaptiveReservoir(_config())

    with pytest.raises(ValueError, match="x must be a 1D numeric vector"):
        model.predict(x)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "x",
    [
        [1.0, float("nan")],
        [1.0, float("inf")],
        [1.0, -float("inf")],
    ],
)
def test_step_rejects_non_finite_input_values(x: object) -> None:
    model = AdaptiveReservoir(_config())

    with pytest.raises(ValueError, match="x must contain only finite values"):
        model.step(x)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "x",
    [
        [1.0, float("nan")],
        [1.0, float("inf")],
        [1.0, -float("inf")],
    ],
)
def test_predict_rejects_non_finite_input_values(x: object) -> None:
    model = AdaptiveReservoir(_config())

    with pytest.raises(ValueError, match="x must contain only finite values"):
        model.predict(x)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "x",
    [
        [1.0, "bad"],
        [object(), 1.0],
        None,
    ],
)
def test_step_rejects_non_numeric_input_values(x: object) -> None:
    model = AdaptiveReservoir(_config())

    with pytest.raises(ValueError, match="x must contain only numeric values"):
        model.step(x)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "x",
    [
        [1.0, "bad"],
        [object(), 1.0],
        None,
    ],
)
def test_predict_rejects_non_numeric_input_values(x: object) -> None:
    model = AdaptiveReservoir(_config())

    with pytest.raises(ValueError, match="x must contain only numeric values"):
        model.predict(x)  # type: ignore[arg-type]


@pytest.mark.parametrize("target", [0, 1.0, np.float32(0.5)])
def test_step_accepts_numeric_finite_target(target: object) -> None:
    model = AdaptiveReservoir(_config())

    result = model.step([1.0, -1.0], target=target)  # type: ignore[arg-type]

    assert result.metrics.target_available is True
    assert result.metrics.readout_updated is True


@pytest.mark.parametrize(
    "target",
    [
        float("nan"),
        float("inf"),
        -float("inf"),
    ],
)
def test_step_rejects_non_finite_target(target: object) -> None:
    model = AdaptiveReservoir(_config())

    with pytest.raises(ValueError, match="target must be finite"):
        model.step([1.0, -1.0], target=target)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "target",
    [
        "1.0",
        b"1.0",
        True,
        object(),
    ],
)
def test_step_rejects_non_numeric_target(target: object) -> None:
    model = AdaptiveReservoir(_config())

    with pytest.raises(ValueError, match="target must be numeric"):
        model.step([1.0, -1.0], target=target)  # type: ignore[arg-type]


def _config() -> ReservoirConfig:
    return ReservoirConfig(
        input_dim=2,
        n_cells=4,
        topology="ring_shortcuts",
        seed=42,
    )
