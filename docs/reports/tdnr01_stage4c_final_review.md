# TDNR-01 — Final Evidence Review

**Review status:** Independent analysis of recorded evidence and static source inspection completed; formal scientific sign-off pending.

**Scientific conclusion:** Adaptive Reservoir demonstrates strong absolute prediction accuracy in several fixed synthetic conditions, but does not establish universal superiority or universal noise robustness. The experiment contains material negative results, particularly under target noise and during early adaptation.

This document reviews the frozen TDNR-01 experiment without changing its protocol, selecting a new configuration, excluding unfavorable results, or claiming an independent model execution.

## 1. Execution and provenance

| Item | Recorded value |
| --- | --- |
| Repository | `kymuco/adaptive-reservoir` |
| Frozen scientific implementation | PR #65, `fc4823b96e3c8c065952485a5765a01a67ce3001` |
| Execution orchestrator | PR #67, `caeef46e8fb62b9544eee2bf121f5e80ce2e3da9` |
| Preflight run used for seed-0 comparison | `45986280290` |
| Full study run | `46048648400` |
| Full run attempt | `1` |
| Workflow deployment commit | `525f5c3baff0fc97224855ee1509567d84573a28` |
| Complete evidence artifact ID | `15225991694` |
| Artifact name | `tdnr01-complete-46048648400-attempt-1` |
| Report SHA-256 | `c9baed32b97c17fe313636d94cb4474bb9eb6fc2f589b185135dc74c68239f29` |
| ZIP SHA-256 reported by GitHub | `e5760cf7176400f76b69b32fe51d5333f967b3add34d04606c8b6d5d17ecaa45` |
| Original artifact expiration | 2026-12-27, approximately 19:03 UTC |

The full workflow's authorization, preflight, seed jobs and consolidation completed successfully. The recorded evidence inspection found that the report SHA-256 matched the expected digest and that partition artifact hashes and record counts were internally consistent.

The ZIP digest and report digest identify different objects. They must not be interchanged. Artifact hashes establish byte integrity, not independent authentication of the scientific source or experimental environment.

The orchestrator checks exact Git HEAD values and requires clean, separate checkouts. It also checks that experimental modules resolve from the frozen PR #65 checkout rather than the PR #67 orchestration checkout. The generated report nevertheless retains `source_commit_independently_verified: false`. No independently signed source-provenance attestation or attributable reviewer identity was supplied for this document.

**Provenance correction:** The seed-0 numerical comparison used outputs from preflight run `45986280290` and full run `46048648400`. Both executions used the same frozen scientific implementation. Neither execution was an independently implemented or independently conducted model experiment. Their comparison is a cross-run consistency check, not independent experimental replication.

## 2. Frozen scientific protocol

The fixed benchmark uses 4,000 samples, an abrupt drift at sample 2,000, a sinusoidal signal, delay changing from 3 to 12, and a 128-sample scoring window. Development uses seeds 0–9; holdout uses seeds 100–129. The partitions are disjoint.

The experiment evaluates two noise axes (`input`, `target`) at levels `0`, `0.05`, `0.1`, `0.2`, `0.4` and `0.8`. Each seed has 12 indexed axis-level cells, including two representations of the shared zero-noise reference, but only 11 distinct noise conditions. Four models are evaluated: `adaptive_reservoir`, `zero`, `persistence` and `lag_buffer_ridge`.

The scoring windows are `pre` [1872, 2000), `early_post` [2000, 2128) and `late_post` [3872, 4000). Models learn from training targets, while clean targets are reserved for offline MAE/RMSE evaluation. No parameter selection is performed using holdout results.

| Partition | Seeds | Scores | Aggregates | Contrasts |
| --- | ---: | ---: | ---: | ---: |
| Development | 10 | 1,440 | 144 | 108 |
| Holdout | 30 | 4,320 | 144 | 108 |
| **Total** | **40** | **5,760** | **288** | **216** |

All 40 seed receipts are represented in the full-run execution evidence. Negative results and intervals containing zero are retained.

## 3. Independent analysis of saved numerical evidence

The supplied independent analysis reports agreement for **4,896 aggregate and contrast numerical fields**:

- 288 noise aggregates × 8 mean/interval fields = 2,304 comparisons.
- 216 baseline contrasts × 12 mean/interval fields = 2,592 comparisons.
- Total = 4,896 comparisons.

These comparisons cover MAE, RMSE, noise-induced changes and their descriptive bootstrap confidence intervals across development and holdout. The analysis checks derived numerical evidence against saved experimental records. It is independent analysis of existing results, **not independent model execution**.

### 3.1 Seed-0 cross-run consistency

The seed-0 comparison uses two previously executed GitHub Actions runs:

| Property | Preflight | Full study |
| --- | --- | --- |
| Run ID | `45986280290` | `46048648400` |
| Scientific implementation | Frozen PR #65 | Same frozen PR #65 |
| Seed | 0 | 0 |
| Purpose | Preflight validation | Official full study |

The comparison covers **144 corresponding score records and 864 numerical values**.

- Maximum observed absolute difference: **3.801153836135995 × 10⁻¹²**.
- Prespecified comparison tolerance: **1 × 10⁻¹⁰**.
- Result: all 864 compared values satisfy the stated absolute tolerance.
- Bitwise identity: **not established; the values are not bitwise identical**.

This result supports numerical consistency between two executions of the **same frozen implementation**. It does not demonstrate independent algorithmic reproduction, independent implementation validation, or independently executed scientific replication.

The preflight and full study are separate workflow runs, but their separation does not make either one an independent scientific replication. Both share the same scientific source and may share implementation assumptions or defects.

No independent model execution was performed as part of this review. No independently implemented model was run, and no complete independent 40-seed rerun was performed.

### 3.2 Scope of verification

The evidence review consists of artifact-integrity checks, independent analysis of saved scores and derived statistics, comparison of seed-0 outputs from the two existing runs, and static inspection of the frozen source.

The 4,896-field analysis and seed-0 comparison are distinct checks. Neither should be described as independent regeneration of all model predictions.

The recorded seed-0 comparison tolerance is an absolute numerical acceptance threshold, not a proof of bitwise determinism or equivalence across arbitrary hardware and numerical libraries.

## 4. Static causality and leakage review

Inspection of the frozen `AdaptiveReservoir.step()` implementation confirms that the readout prediction is computed before the current target is admitted to the supervised readout update. The reservoir state may incorporate the current observation, which is available at prediction time; the current training target does not enter the readout before that prediction.

`PersistenceBaseline.step()` predicts the previous observed training target, or zero initially, then stores the current training target for the next step. Thus persistence uses **past labels**, not future labels or the current label. This is causal under the benchmark's information policy.

`LagBufferRidgeBaseline.step()` constructs features from current and past observations, predicts with its existing fitted readout, and then updates using the current training target. The zero baseline predicts zero without learning.

The Stage 2B comparison generates chronological predictions from training inputs and targets and subsequently computes scoring-window errors against clean targets. Static inspection therefore supports predict-before-update and offline clean-target scoring. It does not substitute for a complete dynamic information-flow proof covering every dependency.

An important comparability limitation remains: persistence consumes the previous supervised label directly, while the other models use different feature representations and learning mechanisms. Consequently, robustness comparisons measure these specific causal systems under their specified information access; they are not a controlled isolation of reservoir architecture alone.

## 5. Absolute prediction accuracy

Positive baseline advantage means the baseline has greater error than Adaptive Reservoir. Negative advantage favors the baseline. Absolute prediction accuracy must be interpreted separately from degradation caused by noise.

For holdout MAE, the representative mean errors are:

| Condition | Window | Adaptive | Zero | Persistence | Lag-buffer ridge |
| --- | --- | ---: | ---: | ---: | ---: |
| No noise | pre | 0.00463 | 0.35406 | 0.02293 | 0.00678 |
| No noise | early_post | 0.03242 | 0.35540 | 0.02379 | 0.10668 |
| No noise | late_post | 0.00716 | 0.35539 | 0.02285 | 0.00738 |
| Input noise 0.8 | pre | 0.03038 | 0.35406 | 0.02293 | 0.08511 |
| Input noise 0.8 | early_post | 0.03859 | 0.35540 | 0.02379 | 0.13769 |
| Input noise 0.8 | late_post | 0.02302 | 0.35539 | 0.02285 | 0.08545 |
| Target noise 0.8 | pre | 0.15799 | 0.35406 | 0.27490 | 0.06000 |
| Target noise 0.8 | early_post | 0.16737 | 0.35540 | 0.27441 | 0.12631 |
| Target noise 0.8 | late_post | 0.15397 | 0.35539 | 0.27650 | 0.06109 |

The adaptive model is substantially better than the constant-zero predictor in absolute error throughout these examples. It also performs particularly well against lag-buffer ridge under input noise.

However, persistence is better in the early-post-drift window under input noise, and lag-buffer ridge is markedly better under strong target noise. These failures are central scientific findings, not outliers to remove.

## 6. Descriptive confidence-interval directions

The following counts summarize the recorded 95% percentile-bootstrap intervals. Entries are ordered as **adaptive favored / baseline favored / interval includes zero**. The intervals are descriptive and unadjusted for multiple comparisons.

| Metric | Development | Holdout |
| --- | --- | --- |
| Absolute MAE, 108 contrasts | 83 / 19 / 6 | 84 / 20 / 4 |
| Absolute RMSE, 108 contrasts | 83 / 20 / 5 | 83 / 20 / 5 |
| Noise-delta MAE, 90 nonzero-noise contrasts | 25 / 54 / 11 | 27 / 56 / 7 |
| Noise-delta RMSE, 90 nonzero-noise contrasts | 27 / 54 / 9 | 28 / 58 / 4 |

The 18 zero-noise contrasts per partition are excluded from the noise-delta direction counts because their noise-induced differences are zero by construction.

Development and holdout show a similar qualitative pattern: the adaptive model is often favored in absolute error, but baselines are more frequently favored on the noise-degradation estimand. These are counts across correlated experimental comparisons, not independent trials or probabilities of superiority.

## 7. Noise robustness is not absolute accuracy

For each model, noise degradation is the error under a specified nonzero noise level minus its paired zero-noise error. A model can have a low absolute error while experiencing a larger increase from its own baseline. Conversely, a consistently inaccurate model can have almost no noise degradation.

The constant-zero baseline illustrates this distinction: its predictions do not change with input or target noise, so its noise-induced degradation is zero even though its absolute error is relatively large.

Persistence is unaffected by input noise in this benchmark because it predicts from the preceding training target rather than the current input. This is causal behavior, not evidence of an implementation leak. It does, however, limit any architectural interpretation of input-noise robustness comparisons.

In holdout, Adaptive Reservoir beats lag-buffer ridge on absolute MAE in 17 of 18 input-noise axis/window combinations, and its noise degradation is favored in all 15 nonzero input-noise combinations. Under target noise, lag-buffer ridge is favored on absolute MAE in 11 of 18 combinations and on noise degradation in all 15 nonzero combinations.

Against persistence, Adaptive Reservoir is favored on absolute MAE in 10 of 18 holdout input-noise combinations, persistence is favored in 7, and one interval includes zero. In the `early_post` window, persistence is favored at every input-noise level. Under target noise, Adaptive Reservoir is favored in 15 of 18 absolute-MAE combinations.

The evidence therefore supports **conditional** robustness claims, not a general statement that the adaptive architecture is more noise-resistant than all baselines.

## 8. Negative and inconclusive findings

A representative negative holdout result occurs with target noise 0.8 in the `pre` window:

- Adaptive Reservoir mean MAE: approximately 0.15799.
- Lag-buffer ridge mean MAE: approximately 0.06000.
- Ridge-minus-adaptive MAE advantage: approximately −0.0980.
- Descriptive 95% bootstrap interval: approximately [−0.1149, −0.0806].

This condition favors lag-buffer ridge by a meaningful absolute error margin. Other baseline-favored results occur in the early-post-drift window, demonstrating that adaptation is not uniformly fast or accurate.

Intervals containing zero must be reported as directionally unresolved under the chosen descriptive interval criterion. They must not be converted into claims of equivalence, absence of an effect, or statistical significance.

The adaptive model's strongest supported findings are its low clean-signal error, recovery in the late-post window, and relative performance against lag-buffer ridge under input noise. Its principal weaknesses are early-post adaptation against persistence and sensitivity to target noise relative to ridge.

## 9. Scope and limitations

1. The scientific evidence is synthetic. It covers a fixed abrupt-drift sinusoidal generator, not real-world streams, recurring drift, gradual drift, AR(1), or unknown deployment distributions.
2. Only the frozen model configurations, noise levels, scoring windows and seed partitions were evaluated. Results do not establish general superiority over other architectures or tuned baselines.
3. Bootstrap intervals are descriptive, unadjusted and based on paired synthetic seeds. No familywise-error or false-discovery-rate correction was applied.
4. The 108 contrasts per partition and four related metrics are correlated. Direction counts are descriptive summaries, not independent significance tests.
5. Development and holdout results were preserved separately. The holdout findings must not be used to retroactively tune this frozen experiment.
6. Static causality review supports predict-before-update, but is not a formal proof of the entire execution stack.
7. Independent numerical analysis covers 4,896 reported aggregate/contrast fields. This is independent analysis of saved evidence, not independent model execution.
8. Seed-0 numerical consistency was assessed between preflight run `45986280290` and full run `46048648400`, both using the same frozen implementation. The comparison covered 144 records and 864 numerical values, with maximum absolute difference `3.801153836135995e-12`, within absolute tolerance `1e-10`.
9. The seed-0 comparison does not constitute independent replication. It does not establish bitwise identity, independently implemented model correctness, or general cross-environment reproducibility.
10. No independently conducted model execution or complete independent 40-seed rerun was performed during this review. Shared implementation defects could affect both existing runs.
11. Artifact integrity and declared Git SHAs are not a cryptographic attestation of execution provenance. The report's source-verification flag remains false.
12. Reviewer identity, attributable independent analysis record, execution-environment details and formal sign-off have not been supplied in the present evidence record.
13. No computational cost, latency, memory-use or deployment-efficiency superiority was established by this accuracy-focused benchmark.
14. These results do not establish statistically significant, universally robust or real-world superior behavior.

## 10. Review determination

**Evidence integrity:** Accepted within the recorded artifact, digest and schema checks.

**Derived statistics:** Accepted within the reported 4,896-field independent analysis of saved numerical evidence.

**Causal implementation:** Static predict-before-update ordering confirmed for the adaptive model and inspected baselines. Persistence's use of previous training labels is explicit and causal.

**Cross-run numerical consistency:** Demonstrated for seed 0 between the preflight and full study runs using the same frozen implementation. All 864 compared numerical values fall within absolute tolerance `1e-10`; the maximum absolute difference is `3.801153836135995e-12`. Bitwise identity is not established.

**Independent experimental replication:** Not performed. The cross-run comparison must not be presented as an independent model execution, independent implementation or independent scientific rerun.

**Scientific conclusion:** The frozen experiment supports conditional predictive advantages and reveals material limitations. Universal superiority and universal noise robustness are not supported.

**Formal independent scientific sign-off:** Pending attributable reviewer approval, documented independent analysis methods and environment, and source-provenance attestation. This report does not by itself authorize a `program_complete` declaration.

## 11. Required follow-up

First, preserve the original ZIP, all 40 seed artifacts, workflow logs, manifests, hashes and seed-0 comparison records from **both** preflight run `45986280290` and full run `46048648400` before the relevant GitHub Actions retention deadlines. The recorded complete full-run artifact expires on 27 December 2026 unless independently archived.

Second, attach the independent analyst's identity or attributable review record, analysis methods, the absolute comparison tolerance `1e-10`, numerical comparison details, environment versions and independent verification of the frozen source checkout. Record clearly that the seed-0 comparison involved two executions of the same implementation.

Third, obtain an explicit scientific sign-off or a documented finding that additional verification is required. An independently authorized, independently conducted model execution or separately implemented validation could strengthen reproducibility evidence. Such work must not silently replace or modify the original frozen results.

Finally, if further research is approved, preregister a new protocol focusing on target-noise robustness, early-post adaptation, information-equivalent baselines and additional drift scenarios. Preserve TDNR-01 as a historical fixed experiment; do not tune it using the existing holdout.

**Decision boundary:** This document is a final review draft of the available TDNR-01 evidence. It neither modifies the scientific protocol nor merges PRs, executes experiments, publishes results or declares program completion. Any formal completion decision requires separate authorization.
