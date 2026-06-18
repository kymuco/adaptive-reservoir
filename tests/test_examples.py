from __future__ import annotations

from examples import temporal_drift_demo


def test_temporal_drift_demo_main_outputs_expected_sections(capsys) -> None:
    exit_code = temporal_drift_demo.main()

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Temporal drift demo" in output
    assert "deterministic synthetic temporal drift stream" in output
    assert "stream -> prediction -> metrics" in output
    assert "benchmark: temporal-drift" in output
    assert "samples_seen: 360" in output
    assert "pre_score:" in output
    assert "post_score:" in output
    assert "final_score:" in output
    assert "adapt_steps:" in output
    assert "us_per_sample:" in output
    assert "Markdown report:" in output
    assert "| benchmark | model | topology |" in output
    assert "| temporal-drift | adaptive_reservoir |" in output
