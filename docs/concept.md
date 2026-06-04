# Project Concept

`adaptive-reservoir` is a CPU-friendly temporal adaptation layer, not an agent framework, not a memory system, and not an LLM replacement.

It implements an **Adaptive Temporal Substrate**: a small reservoir-style state engine that turns numeric event streams into predictions and adaptive state channels.

The project is designed to be embedded into other systems that need lightweight online temporal adaptation, such as software agents, presence engines, streaming behavior controllers, practice systems, robotics prototypes, and HDE-like environments.

## What adaptive-reservoir is

`adaptive-reservoir` is a reusable Python library for tracking short-term temporal dynamics in a stream of numeric events.

At a high level, it follows this shape:

```text
event stream -> temporal state -> adaptive channels -> prediction / behavior signal
```

The library is intended to provide:

- a deterministic reservoir-style state engine;
- sparse, CPU-friendly recurrent dynamics;
- short-term and multi-timescale traces;
- simple online readouts;
- adaptive state channels such as novelty, stability, drift pressure, confidence, and saturation;
- metrics and benchmarks for temporal adaptation.

The goal is not to make every host system intelligent by itself. The goal is to give host systems a small, cheap temporal layer that can notice change, instability, confidence shifts, and recurring patterns without turning every micro-decision into an LLM call.

## What adaptive-reservoir is not

`adaptive-reservoir` is intentionally narrow.

It is not:

- an LLM;
- an agent framework;
- a chat system;
- a prompt builder;
- a memory database;
- a vector database;
- an HDE core component;
- a consent or policy engine;
- a replacement for semantic or episodic memory;
- a claim about AGI, consciousness, or biological realism.

The library consumes numeric vectors and produces numeric predictions, state diagnostics, and adaptive channels. It does not know what a user, companion, project, memory, conversation, or robot is.

Those meanings belong to the host system and its adapters.

## Adaptive Temporal Substrate

The broader concept behind the library is an **Adaptive Temporal Substrate**.

An Adaptive Temporal Substrate is a small runtime layer that tracks how a stream changes over time. It is useful when a system needs to answer questions like:

- Is the current situation familiar or novel?
- Is the recent state stable or unstable?
- Is prediction error increasing?
- Can the current prediction be trusted?
- Is the internal state becoming saturated or overloaded?
- Should the host system be more cautious, more responsive, or less intrusive?

In this project, the substrate is implemented with a fixed or slowly changing reservoir state, trace features, and lightweight online readouts.

The substrate is not the whole application. It is a component that host systems can use to build adaptive behavior.

## Why not another ESN wrapper

Reservoir computing and Echo State Networks are established ideas. This project does not exist to repackage them as a generic forecasting toolkit.

The difference is the target interface and use case.

A classic reservoir tool often focuses on:

```text
time series -> prediction
```

`adaptive-reservoir` focuses on:

```text
event stream -> temporal state -> adaptive state channels -> behavior/control signal
```

The main differentiator is not just the reservoir. The main differentiator is the combination of:

- a small CPU-friendly state engine;
- online readout updates;
- explicit adaptive state channels;
- runtime metrics;
- clean embedding into agent, presence, and behavior systems.

This means the library should remain technically boring internally, while being specific and useful in how it is applied.

## Main use cases

### Software agents

A software agent can use adaptive state channels to adjust behavior without making every small decision with an LLM.

Example outputs:

- initiative bias;
- interruption risk;
- confidence pressure;
- drift pressure;
- stability score.

### Presence engines

A presence engine can use the substrate to track whether a user or environment appears stable, busy, interrupted, idle, or changing rapidly.

Example outputs:

- should wait;
- should notify;
- attention state;
- readiness score.

### Companion systems

A companion runtime can use the substrate as a behavior-level adaptation layer.

The substrate should not store relationship memory or private facts. Instead, it can track temporal rhythm and short-term behavioral dynamics after a trusted host adapter converts events into numeric features.

Example outputs:

- reply length bias;
- warmth bias;
- initiative bias;
- silence/wait pressure;
- memory recall pressure.

### Practice and skill systems

A practice system can use the substrate to track temporal changes in a user's performance.

Example outputs:

- timing instability;
- recurring error pressure;
- fatigue-like state;
- improvement trend.

### Robotics and interactive systems

A robot or interactive device can use the substrate for lightweight reactive state tracking.

Example outputs:

- speak now vs wait;
- repeat shorter;
- call for help;
- confidence to act;
- unusual interaction pressure.

## Boundary with HDE

`adaptive-reservoir` is not HDE Core and must not depend on HDE repositories.

HDE-like systems may use this library as an optional computational substrate, but the boundary is strict:

```text
HDE / host system -> policy and consent filtering -> numeric adapter -> adaptive-reservoir -> adaptive signals -> host system policy/action layer
```

The host system is responsible for:

- consent;
- access control;
- audit logging;
- identity;
- memory scopes;
- semantic and episodic memory;
- deciding which events can be converted into numeric vectors;
- deciding which downstream systems may consume adaptive signals.

`adaptive-reservoir` is responsible only for:

- accepting numeric vectors;
- updating temporal state;
- updating readouts when targets are provided;
- computing adaptive state channels;
- exposing metrics and snapshots of its mathematical state.

It must not store private user facts, conversations, memories, or policy decisions.

## Design principles

1. **Independent core**  
   The library must not depend on HDE, Character_OS, PracticeLens, Runplane, Machine Presence, or any specific host system.

2. **Numeric boundary**  
   Inputs are numeric vectors. Domain events are converted by external adapters.

3. **Boring implementation, special purpose**  
   The core should be deterministic, testable, and simple. The distinctive value comes from adaptive state channels and embedding use cases.

4. **Readout-first adaptation**  
   Fast learning pressure should live primarily in the readout. Recurrent plasticity is not part of the MVP.

5. **Metrics over vibes**  
   Claims must be backed by benchmarks and runtime metrics such as adaptation lag, drift pressure, saturation, and microseconds per sample.

6. **Embeddable by default**  
   The library should be small enough to run inside larger applications without becoming their architecture.

## MVP scope

The MVP should include:

- deterministic package structure;
- reservoir config and state;
- random sparse, ring shortcuts, and modular small-world topologies;
- state, state+slow, and multi-trace feature modes;
- NLMS, replay ridge, and sliding-window ridge readouts;
- adaptive channels for novelty, stability, drift pressure, confidence, and saturation;
- import tests and smoke benchmarks.

The MVP should exclude:

- recurrent plasticity;
- STDP;
- LLM integration;
- HDE integration code;
- project-specific adapters;
- GUI;
- claims about consciousness or biological realism.
