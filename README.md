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

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check .
