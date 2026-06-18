from __future__ import annotations

from examples import behavior_bias_demo, presence_state_demo, temporal_drift_demo


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


def test_behavior_bias_demo_main_outputs_expected_sections(capsys) -> None:
    exit_code = behavior_bias_demo.main()

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Behavior bias demo" in output
    assert "synthetic numeric interaction events only" in output
    assert "does not infer real user state" in output
    assert "does not integrate with Character_OS" in output
    assert "events -> adaptive channels -> host decision hints" in output
    assert "message_length" in output
    assert "pause_seconds" in output
    assert "interruptions" in output
    assert "error_count" in output
    assert "time_pressure" in output
    assert "initiative_bias" in output
    assert "interrupt_risk" in output
    assert "confidence" in output
    assert "drift_pressure" in output
    assert "060   calm" in output
    assert "120   pressure" in output
    assert "180   recovery" in output
    assert "240   recovery" in output


def test_behavior_bias_demo_events_are_bounded() -> None:
    for step in range(1, behavior_bias_demo.N_STEPS + 1):
        phase, event = behavior_bias_demo.synthetic_behavior_event(step)
        assert phase in {"calm", "pressure", "recovery"}
        assert len(event) == len(behavior_bias_demo.EVENT_NAMES)
        assert all(0.0 <= value <= 1.0 for value in event)


def test_behavior_bias_demo_targets_are_bounded() -> None:
    event = (0.5, 0.5, 0.2, 0.1, 0.4)

    for phase in ("calm", "pressure", "recovery"):
        targets = behavior_bias_demo.behavior_targets(event, phase=phase)
        assert tuple(targets) == behavior_bias_demo.CHANNELS
        assert all(0.0 <= value <= 1.0 for value in targets.values())


def test_behavior_bias_demo_returns_checkpoint_snapshots() -> None:
    snapshots = behavior_bias_demo.run_behavior_bias_demo()

    assert tuple(snapshot.step for snapshot in snapshots) == behavior_bias_demo.CHECKPOINTS
    assert {snapshot.phase for snapshot in snapshots} == {"calm", "pressure", "recovery"}
    for snapshot in snapshots:
        assert 0.0 <= snapshot.initiative_bias <= 1.0
        assert 0.0 <= snapshot.interrupt_risk <= 1.0
        assert 0.0 <= snapshot.confidence <= 1.0
        assert 0.0 <= snapshot.drift_pressure <= 1.0


def test_presence_state_demo_main_outputs_expected_sections(capsys) -> None:
    exit_code = presence_state_demo.main()

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Presence state demo" in output
    assert "synthetic numeric desktop/system events only" in output
    assert "does not read real desktop activity" in output
    assert "does not infer real user state" in output
    assert "events -> presence channels -> host decision hints" in output
    assert "idle_time" in output
    assert "window_switch_rate" in output
    assert "typing_burst" in output
    assert "failed_action_count" in output
    assert "notification_density" in output
    assert "should_wait" in output
    assert "should_notify" in output
    assert "attention_state" in output
    assert "060   focused" in output
    assert "120   overloaded" in output
    assert "180   available" in output
    assert "240   available" in output


def test_presence_state_demo_events_are_bounded() -> None:
    for step in range(1, presence_state_demo.N_STEPS + 1):
        phase, event = presence_state_demo.synthetic_presence_event(step)
        assert phase in {"focused", "overloaded", "available"}
        assert len(event) == len(presence_state_demo.EVENT_NAMES)
        assert all(0.0 <= value <= 1.0 for value in event)


def test_presence_state_demo_targets_are_bounded() -> None:
    event = (0.4, 0.2, 0.5, 0.1, 0.3)

    for phase in ("focused", "overloaded", "available"):
        targets = presence_state_demo.presence_targets(event, phase=phase)
        assert tuple(targets) == presence_state_demo.CHANNELS
        assert all(0.0 <= value <= 1.0 for value in targets.values())


def test_presence_state_demo_returns_checkpoint_snapshots() -> None:
    snapshots = presence_state_demo.run_presence_state_demo()

    assert tuple(snapshot.step for snapshot in snapshots) == presence_state_demo.CHECKPOINTS
    assert {snapshot.phase for snapshot in snapshots} == {"focused", "overloaded", "available"}
    for snapshot in snapshots:
        assert 0.0 <= snapshot.should_wait <= 1.0
        assert 0.0 <= snapshot.should_notify <= 1.0
        assert 0.0 <= snapshot.attention_state <= 1.0
