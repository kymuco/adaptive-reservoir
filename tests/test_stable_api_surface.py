EXPECTED_ROOT_EXPORTS = (
    "AdaptiveChannels",
    "AdaptiveReservoir",
    "AdaptiveReservoirMetricsSnapshot",
    "AdaptiveStepResult",
    "ChannelCalculatorProtocol",
    "ChannelCalculatorSnapshot",
    "ChannelConfig",
    "FeatureExtractorProtocol",
    "ReadoutConfig",
    "ReadoutProtocol",
    "ReadoutSnapshot",
    "ReservoirConfig",
    "ReservoirCore",
    "ReservoirSnapshot",
    "ReservoirState",
    "StateDiagnostics",
    "StepMetrics",
    "TopologyBuilderProtocol",
    "TraceConfig",
    "TraceNorms",
    "__version__",
    "calculate_state_diagnostics",
    "extract_features",
    "rms_norm",
)


def test_root_export_surface_is_explicitly_reviewed() -> None:
    adaptive_reservoir = __import__("adaptive_reservoir")

    assert tuple(adaptive_reservoir.__all__) == EXPECTED_ROOT_EXPORTS


def test_adapter_protocol_is_package_exported_but_not_root_exported() -> None:
    adaptive_reservoir = __import__("adaptive_reservoir")
    adapters = __import__("adaptive_reservoir.adapters", fromlist=["__all__"])

    assert set(adapters.__all__) == {"EventVectorizer", "FloatArray"}
    assert not hasattr(adaptive_reservoir, "EventVectorizer")
    assert not hasattr(adaptive_reservoir, "FloatArray")


def test_stable_api_surface_doc_mentions_all_root_exports() -> None:
    pathlib = __import__("pathlib", fromlist=["Path"])
    text = pathlib.Path("docs/stable_api_surface.md").read_text(encoding="utf-8")

    for exported_name in EXPECTED_ROOT_EXPORTS:
        assert exported_name in text


def test_stable_api_surface_doc_records_compatibility_boundaries() -> None:
    pathlib = __import__("pathlib", fromlist=["Path"])
    text = pathlib.Path("docs/stable_api_surface.md").read_text(encoding="utf-8")

    required_phrases = (
        "Stable user-facing API",
        "Stable adapter boundary",
        "Extension-facing API",
        "Advanced diagnostics API",
        "Advanced implementation API",
        "Internal / non-stable areas",
        "What can change without a breaking change",
        "intentionally not root-exported in M12",
        "M12 keeps the root API unchanged",
    )
    for phrase in required_phrases:
        assert phrase in text
