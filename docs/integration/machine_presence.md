# Machine Presence Adapter Design Note

This note sketches how a Machine Presence-like host could use
`adaptive-reservoir` as a numeric temporal adaptation layer for system-presence
signals.

It is a design note only. It does not define an adapter API and does not
implement desktop, device, or operating-system monitoring.

## Purpose

A Machine Presence-like host may use `adaptive-reservoir` to turn permitted
system-style numeric features into adaptive presence channels.

The intended pattern is:

```text
Machine Presence host -> numeric system features -> adaptive-reservoir -> numeric presence channels -> host policy/action layer
```

## Boundary summary

| Concern | Owned by adaptive-reservoir? | Owner |
|---|---:|---|
| Numeric temporal adaptation | Yes | adaptive-reservoir |
| Numeric predictions/channels | Yes | adaptive-reservoir |
| OS event collection | No | Machine Presence host |
| Keyboard/mouse hooks | No | Machine Presence host |
| Window/app inspection | No | Machine Presence host |
| Notification access | No | Machine Presence host |
| Screen/file access | No | Machine Presence host |
| Consent and policy | No | Machine Presence host |
| Actions and audit | No | Machine Presence host |

## What the host could send

The host must send numeric feature streams only. `adaptive-reservoir` must not
read operating-system events directly.

Example numeric features:

- `idle_time_normalized`
- `window_switch_rate`
- `typing_burst_score`
- `failed_action_count_window`
- `notification_density`
- `focus_session_age`
- `device_load_hint`

These features must be computed by the host after consent, policy, and privacy
filters have already allowed them.

## What adaptive-reservoir could return

Example numeric channels:

- `should_wait`
- `should_notify`
- `attention_state`
- `context_switch_pressure`
- `presence_stability`
- `drift_pressure`

These channels are host-facing hints. They are not permissions, OS actions, or
mental-state classifications.

`attention_state` should be interpreted only as a synthetic operational attention
channel defined by the host's numeric features. It must not be treated as mental
state inference, emotion detection, or surveillance output.

## What the host owns

The Machine Presence host owns:

- permission to observe system events
- OS integration
- privacy filtering
- feature retention
- notification policy
- action decisions
- user settings
- consent checks
- audit records

A high `should_notify`-like channel is not permission to notify. The host must
still apply policy, context, and user settings before any user-facing behavior.

## Example safe flow

1. The host checks whether system observation is allowed.
2. The host reads permitted events through host-owned integrations.
3. The host converts permitted events into numeric features.
4. `adaptive-reservoir` receives only numeric vectors.
5. `adaptive-reservoir` returns numeric presence channels.
6. The host interprets channels with policy and user settings.
7. The host decides whether any notification or action is allowed.
8. The host records audit information when required.

## Unsafe patterns

Do not:

- read OS activity inside `adaptive-reservoir`
- pass screen contents to `adaptive-reservoir`
- pass app names, window titles, or notification text as raw content
- pass keyboard or mouse event logs as raw content
- treat `attention_state` as mental-state detection
- notify the user only because a channel is high
- use channels to bypass consent or quiet-hours policy
- treat reservoir state as a system activity log

## Non-goals

This document does not define:

- desktop monitoring
- OS event readers
- keyboard or mouse hooks
- window/app inspection
- notification access
- screen capture
- action execution
- consent implementation
- policy enforcement

Those concerns belong to the Machine Presence host or future explicit integration
documents.
