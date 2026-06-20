from __future__ import annotations

from pathlib import Path

_REPORT_PATH = Path("docs/reports/experimental_readouts.md")


def _report_text() -> str:
    return _REPORT_PATH.read_text(encoding="utf-8")


def test_experimental_readouts_report_exists() -> None:
    assert _REPORT_PATH.is_file()


def test_experimental_readouts_report_names_m10_components() -> None:
    text = _report_text()

    assert "SparseOnlineReadout" in text
    assert "RLSReadout" in text
    assert "OjaCompressor" in text


def test_experimental_readouts_report_separates_oja_from_readouts() -> None:
    text = _report_text()

    assert "`OjaCompressor` is not a readout." in text
    assert "compressor/projection layer, not supervised prediction logic" in text


def test_experimental_readouts_report_preserves_stable_api_boundary() -> None:
    text = _report_text()

    required_phrases = (
        "not part of the stable adaptive-reservoir public API",
        "no new default readout",
        "no stable readout factory registration",
        "no root package export",
        "not registered in the stable readout factory",
        "The stable adaptive-reservoir core should remain unchanged.",
    )
    for phrase in required_phrases:
        assert phrase in text


def test_experimental_readouts_report_states_explicit_non_claims() -> None:
    text = _report_text()

    required_non_claims = (
        "any experimental component is better than the stable readouts",
        "RLS is production-ready",
        "sparse readout is stable enough for default use",
        "Oja compression improves downstream prediction",
        "benchmark results generalize to real user data",
        "experimental components should be exposed through the stable API",
    )
    for claim in required_non_claims:
        assert claim in text


def test_experimental_readouts_report_includes_promotion_criteria() -> None:
    text = _report_text()

    criteria = (
        "deterministic benchmark evidence across multiple seeds",
        "no regression of CPU-friendly behavior",
        "clear snapshot/restore compatibility",
        "well-understood failure modes",
        "no heavy dependency requirement",
        "no semantic, policy, identity, or user-data coupling",
    )
    for criterion in criteria:
        assert criterion in text
