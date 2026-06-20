from __future__ import annotations

from collections.abc import Mapping

import numpy as np

import adaptive_reservoir
from adaptive_reservoir.adapters import (
    __all__ as adapters_exports,
    EventVectorizer,
    FloatArray,
)


class DictEventVectorizer:
    def transform(self, event: object) -> FloatArray:
        if not isinstance(event, Mapping):
            msg = "event must be a mapping"
            raise TypeError(msg)
        return np.asarray(
            (
                float(event["signal"]),
                float(event.get("pressure", 0.0)),
            ),
            dtype=np.float64,
        )


class MissingTransform:
    pass


def test_event_vectorizer_accepts_structural_implementation() -> None:
    vectorizer = DictEventVectorizer()

    assert isinstance(vectorizer, EventVectorizer)

    vector = vectorizer.transform({"signal": 0.75, "pressure": -0.25})

    assert isinstance(vector, np.ndarray)
    assert vector.shape == (2,)
    assert vector.dtype == np.float64
    assert np.all(np.isfinite(vector))
    assert np.allclose(vector, np.asarray((0.75, -0.25), dtype=np.float64))


def test_event_vectorizer_rejects_objects_without_transform() -> None:
    assert not isinstance(MissingTransform(), EventVectorizer)


def test_adapters_package_exports_protocol_contract() -> None:
    assert set(adapters_exports) == {"EventVectorizer", "FloatArray"}


def test_event_vectorizer_is_not_root_exported_before_stable_api_review() -> None:
    assert not hasattr(adaptive_reservoir, "EventVectorizer")
