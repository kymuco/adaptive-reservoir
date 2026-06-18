# PracticeLens Adapter Design Note

This note sketches how a PracticeLens-like host could use `adaptive-reservoir`
as a numeric temporal adaptation layer for practice-session dynamics.

It is a design note only. It does not define an adapter API and does not
implement PracticeLens integration.

## Purpose

A PracticeLens-like host may use `adaptive-reservoir` to model numeric practice
signals over time and produce adaptive hints for host-owned practice logic.

The intended pattern is:

```text
PracticeLens host -> numeric practice features -> adaptive-reservoir -> numeric practice channels -> host practice/policy layer
```

## Boundary summary

| Concern | Owned by adaptive-reservoir? | Owner |
|---|---:|---|
| Numeric temporal adaptation | Yes | adaptive-reservoir |
| Numeric predictions/channels | Yes | adaptive-reservoir |
| Practice records | No | PracticeLens host |
| Skill taxonomy | No | PracticeLens host |
| Lesson plan | No | PracticeLens host |
| Feedback text | No | PracticeLens host |
| Grading | No | PracticeLens host |
| Coaching policy | No | PracticeLens host |
| Consent, privacy, and audit | No | PracticeLens host |

## What the host could send

The host must send numeric feature streams only. Raw student/user content,
practice history, feedback text, and private records must stay outside
`adaptive-reservoir`.

Example numeric features:

- `error_rate_window`
- `retry_count_normalized`
- `response_time_normalized`
- `stability_score`
- `difficulty_level_normalized`
- `fatigue_pressure_hint`
- `improvement_delta`

These features must be computed by the host from permitted practice data. The
host owns feature definitions, privacy filtering, retention, and policy.

## What adaptive-reservoir could return

Example numeric channels:

- `practice_stability`
- `difficulty_pressure`
- `retry_pressure`
- `confidence`
- `drift_pressure`
- `session_adaptation_hint`

These channels are hints for a host-owned practice system. They are not grades,
recommendations, diagnoses, or lesson-plan decisions.

## What the host owns

The PracticeLens host owns:

- practice records
- skill taxonomy
- lesson plans
- feedback text
- recommendation policy
- grading policy, if any
- teacher/student privacy rules
- consent checks
- memory and history
- audit records

A high `difficulty_pressure`-like channel does not mean the curriculum should be
changed automatically. The host must interpret numeric hints with practice policy,
user settings, and context.

## Example safe flow

1. The host checks whether practice data can be used for feature computation.
2. The host converts permitted practice data into numeric features.
3. `adaptive-reservoir` receives only numeric vectors.
4. `adaptive-reservoir` returns numeric predictions or practice channels.
5. The host interprets channels with skill taxonomy and practice policy.
6. The host decides whether feedback, review, or plan changes are allowed.
7. The host records audit information when required.

## Unsafe patterns

Do not:

- use `adaptive-reservoir` as a grading system
- store student/user profiles in reservoir state
- pass raw student text, audio, or private records to `adaptive-reservoir`
- treat channels as final diagnosis
- auto-change curriculum only because a channel is high
- use channels to bypass teacher/student policy
- treat model state as long-term practice memory
- use adaptive channels as standalone evaluation decisions

## Non-goals

This document does not define:

- PracticeLens runtime behavior
- adapter classes
- schemas
- grading logic
- coaching logic
- lesson-plan generation
- feedback text generation
- practice memory storage
- consent implementation
- policy enforcement

Those concerns belong to the PracticeLens host or future explicit integration
documents.
