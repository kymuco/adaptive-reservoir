from __future__ import annotations

import json

import pytest

from adaptive_reservoir.benchmarks.runner import main


def test_runner_csv_format_outputs_report_columns(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main([
        "delayed-xor",
        "--seed",
        "3",
        "--samples",
        "240",
        "--delay-a",
        "2",
        "--delay-b",
        "5",
        "--score-window",
        "40",
        "--cells",
        "8",
        "--format",
        "csv",
    ])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert output.startswith("benchmark,model,topology,feature_mode,readout,seed")
    assert "delayed-xor" in output
    assert "final_score" in output
    assert "pre_score" not in output


def test_runner_markdown_format_outputs_summary_table(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main([
        "concept-drift",
        "--seed",
        "3",
        "--samples",
        "220",
        "--drift-at",
        "110",
        "--score-window",
        "24",
        "--cells",
        "8",
        "--format",
        "markdown",
    ])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert output.startswith("| benchmark | model | topology |")
    assert "| concept-drift | adaptive_reservoir |" in output
    assert "| --- | --- | --- |" in output


def test_runner_json_format_outputs_report_array(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main([
        "temporal-drift",
        "--seed",
        "3",
        "--samples",
        "260",
        "--drift-at",
        "130",
        "--delay-before",
        "2",
        "--delay-after",
        "8",
        "--score-window",
        "32",
        "--cells",
        "8",
        "--format",
        "json",
    ])

    assert exit_code == 0
    rows = json.loads(capsys.readouterr().out)
    assert len(rows) == 1
    assert rows[0]["benchmark"] == "temporal-drift"
    assert rows[0]["model"] == "adaptive_reservoir"
    assert isinstance(rows[0]["final_score"], float)
    assert "pre_score" not in rows[0]


def test_runner_rejects_unknown_output_format(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["delayed-xor", "--format", "xml"])

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "invalid choice" in captured.err
    assert "xml" in captured.err


def test_runner_output_writes_report_file_without_stdout(
    capsys: pytest.CaptureFixture[str],
    tmp_path,
) -> None:
    output_path = tmp_path / "benchmark.csv"

    exit_code = main([
        "delayed-xor",
        "--seed",
        "3",
        "--samples",
        "240",
        "--delay-a",
        "2",
        "--delay-b",
        "5",
        "--score-window",
        "40",
        "--cells",
        "8",
        "--format",
        "csv",
        "--output",
        str(output_path),
    ])

    assert exit_code == 0
    assert capsys.readouterr().out == ""
    assert output_path.read_text(encoding="utf-8").startswith(
        "benchmark,model,topology,feature_mode,readout,seed"
    )
