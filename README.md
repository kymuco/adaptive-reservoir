# adaptive-reservoir

A small online temporal adaptation layer for systems that need numeric reflexes,
not another LLM, agent framework, or memory database.

`adaptive-reservoir` turns numeric event streams into predictions, metrics, and
adaptive state channels such as drift pressure, confidence, interruption risk,
or wait/notify hints.

It is designed to sit below a host system:

```text
numeric feature stream
        |
        v
adaptive-reservoir
        |
        v
predictions / metrics / adaptive channels
        |
        v
host interpretation + policy + action layer
```

## Why this exists

Many adaptive systems receive changing streams: interaction rhythm, system
presence, practice-session signals, sensor-like events, and other numeric
features.

Static rules and one-shot predictions are often too brittle for those streams.
The useful question is not only "what is happening now?" but also:

```text
is the stream changing?
is the system stable?
should the host wait, adapt, or ask for more context?
```

`adaptive-reservoir` provides a lightweight online substrate for those questions.
It helps a host system build a small numeric reflex layer while the host still
owns meaning, memory, consent, policy, and actions.

## What it is

- A lightweight Python library for online temporal adaptation.
- A CPU-friendly reservoir-style state engine.
- A reusable substrate for software agents, presence engines, practice systems,
  and streaming applications.
- A foundation for adaptive state channels such as drift pressure, confidence,
  saturation, stability, interruption risk, and wait/notify hints.
- A deterministic benchmark and example suite for checking adaptation behavior.

## What it is not

- Not an LLM.
- Not an agent framework.
- Not a memory database.
- Not HDE Core.
- Not a consent, policy, identity, or audit layer.
- Not a semantic interpreter.
- Not a replacement for host-side privacy and safety checks.

`adaptive-reservoir` processes numeric feature streams only. Host systems own raw
data access, feature extraction, interpretation, memory, consent, policy,
actions, and audit.

## When to use it

Use `adaptive-reservoir` when you have a numeric stream and want online adaptive
signals that a host system can interpret.

Good fits include:

- concept-drift and temporal-drift experiments
- adaptive state channels for agent-like systems
- synthetic presence or interaction streams
- practice-session dynamics
- low-level numeric adaptation layers in larger systems
- HDE-like hosts that need adaptive numeric channels but own policy elsewhere

Poor fits include:

- natural-language reasoning by itself
- storing user memories
- making policy decisions
- enforcing consent
- directly reading desktop, email, chat, or private documents
- executing user-facing actions without a host decision layer

## Quick start

Install locally for development:

```bash
python -m pip install -e .
python -m pip install pytest ruff
```

Run a benchmark:

```bash
adaptive-reservoir-bench temporal-drift --format markdown
```

Run examples:

```bash
python examples/temporal_drift_demo.py
python examples/behavior_bias_demo.py
python examples/presence_state_demo.py
```

Run checks:

```bash
python -m pytest
python -m ruff check .
```

## Examples

### Temporal drift

Use the benchmark runner when you want a compact, reproducible check that the
model can adapt when a temporal stream changes:

```bash
adaptive-reservoir-bench temporal-drift --format markdown
```

Or run the executable demo:

```bash
python examples/temporal_drift_demo.py
```

This demonstrates:

```text
stream -> prediction -> metrics
```

### Adaptive channels

`adaptive-reservoir` can also be used as a low-level numeric layer for adaptive
state channels. The host application decides how to interpret the channels and
what to do with them.

```bash
python examples/behavior_bias_demo.py
python examples/presence_state_demo.py
```

These demos use deterministic synthetic numeric streams only:

```text
events -> adaptive channels -> host decision hints
```

They do not read real desktop activity, process message content, infer real user
state, or integrate with Character_OS/HDE.

## Benchmarks

The benchmark suite is intentionally deterministic and synthetic. It is evidence
for online temporal adaptation behavior, not a production-readiness claim.

Current benchmarks:

| Benchmark | What it stresses |
|---|---|
| `concept-drift` | abrupt input-to-target mapping change |
| `temporal-drift` | changed temporal dependency / delay |
| `delayed-xor` | delayed memory plus nonlinear binary logic |

Benchmark reports include metrics such as `final_score`, `adapt_steps`,
`us_per_sample`, `saturation_rate`, and `readout_sparsity`.

See [Benchmark Methodology](docs/benchmarks.md) for how to read the results and
what they do not prove.

## Documentation map

- [Adaptive State Channels](docs/adaptive_state_channels.md) explains channel
  meaning, computation, ranges, interpretation, and limitations.
- [Benchmark Methodology](docs/benchmarks.md) explains why the benchmark suite
  uses concept drift, temporal drift, and delayed XOR.
- [HDE Integration Note](docs/integration/hde.md) defines the boundary between
  `adaptive-reservoir` and HDE-like host systems.
- [Character OS Design Note](docs/integration/character_os.md) sketches a
  companion-facing adapter boundary.
- [Machine Presence Design Note](docs/integration/machine_presence.md) sketches a
  system-presence adapter boundary.
- [PracticeLens Design Note](docs/integration/practicelens.md) sketches a
  practice-session adapter boundary.

## Development

Run the local checks:

```bash
python -m pip install -e .
python -m pip install pytest ruff
python -m pytest
python -m ruff check .
```

## License

Apache-2.0.
