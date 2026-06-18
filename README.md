# adaptive-reservoir

CPU-friendly temporal adaptation layer for software agents and streaming systems.

`adaptive-reservoir` implements an Adaptive Temporal Substrate: a small reservoir-style state engine designed to turn event streams into predictions and adaptive state channels.

## What it is

- A lightweight Python library for online temporal adaptation.
- A reusable substrate for software agents, presence engines, and streaming systems.
- A foundation for adaptive state channels such as novelty, stability, drift pressure, confidence, and saturation.

## What it is not

- Not an LLM.
- Not an agent framework.
- Not a memory database.
- Not an HDE core component.
- Not a replacement for semantic or episodic memory.

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
