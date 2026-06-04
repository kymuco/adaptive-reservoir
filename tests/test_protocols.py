from collections.abc import Mapping

import numpy as np

from adaptive_reservoir import (
    AdaptiveChannels,
    ChannelCalculatorProtocol,
    FeatureExtractorProtocol,
    ReadoutProtocol,
    ReservoirConfig,
    ReservoirState,
    TopologyBuilderProtocol,
)
from adaptive_reservoir.core.protocols import FloatArray


class DummyReadout:
    def predict(self, features: FloatArray) -> float:
        return float(features.sum())

    def update(self, features: FloatArray, target: float) -> None:
        return None

    def snapshot(self) -> Mapping[str, object]:
        return {"kind": "dummy"}

    def restore(self, snapshot: Mapping[str, object]) -> None:
        return None


class DummyTopologyBuilder:
    def build(self, config: ReservoirConfig) -> FloatArray:
        return np.zeros((config.n_cells, config.n_cells), dtype=config.dtype)


class DummyFeatureExtractor:
    def extract(self, state: ReservoirState, x: FloatArray) -> FloatArray:
        return np.concatenate((state.activations, x))


class DummyChannelCalculator:
    def update(
        self,
        x: FloatArray,
        state: ReservoirState,
        features: FloatArray,
        prediction: float | None,
        target: float | None = None,
    ) -> AdaptiveChannels:
        return AdaptiveChannels(confidence=1.0 if prediction is not None else 0.0)


def test_dummy_readout_satisfies_protocol() -> None:
    assert isinstance(DummyReadout(), ReadoutProtocol)


def test_dummy_topology_builder_satisfies_protocol() -> None:
    assert isinstance(DummyTopologyBuilder(), TopologyBuilderProtocol)


def test_dummy_feature_extractor_satisfies_protocol() -> None:
    assert isinstance(DummyFeatureExtractor(), FeatureExtractorProtocol)


def test_dummy_channel_calculator_satisfies_protocol() -> None:
    assert isinstance(DummyChannelCalculator(), ChannelCalculatorProtocol)


def test_dummy_protocols_can_be_used_together() -> None:
    config = ReservoirConfig(input_dim=2, n_cells=4)
    state = ReservoirState.zeros(n_cells=config.n_cells, dtype=config.dtype)
    x = np.array([0.1, -0.2], dtype=config.dtype)

    topology = DummyTopologyBuilder().build(config)
    features = DummyFeatureExtractor().extract(state, x)
    prediction = DummyReadout().predict(features)
    channels = DummyChannelCalculator().update(x, state, features, prediction)

    assert topology.shape == (4, 4)
    assert features.shape == (6,)
    assert isinstance(prediction, float)
    assert channels.confidence == 1.0
