# TDNR-01 Stage 4C — Frozen Execution and Independent Review

Status: PROPOSED — NOT EXECUTED.

## Objective

Execute the already implemented fixed TDNR-01 protocol, preserve complete development and holdout evidence, independently audit the results, and report negative and inconclusive outcomes without filtering. Stage 4C is not complete until actual evidence and an independent review exist.

## Frozen source

Reviewed implementation commit: `fc4823b96e3c8c065952485a5765a01a67ce3001` (PR #65, CI PASS). The PR was still open and draft when this runbook was prepared. Do not substitute a different commit without a separate review. The manifest records a declared commit; it does not independently authenticate the checkout.

## Approval boundary

The following PowerShell commands are PROPOSALS ONLY. They must undergo separate Controller interpretation and human approval before execution. Run them from the repository root in a working Python environment with the project installed. Do not execute from an unverified or dirty checkout. Do not change model parameters, seeds, noise levels, scoring windows, or bootstrap settings after freezing the protocol.

## Proposed PowerShell execution

```powershell
$expected = 'fc4823b96e3c8c065952485a5765a01a67ce3001'
$sha = (git rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $sha -ne $expected) { throw 'Frozen source commit mismatch' }
$dirty = @(git status --porcelain)
if ($LASTEXITCODE -ne 0 -or $dirty.Count -ne 0) { throw 'Working tree must be clean' }
python --version
if ($LASTEXITCODE -ne 0) { throw 'Python unavailable' }
python -c 'import numpy, adaptive_reservoir; print("NumPy", numpy.__version__)'
if ($LASTEXITCODE -ne 0) { throw 'Project environment unavailable' }
$root = Join-Path $HOME ('tdnr01-evidence-' + $sha.Substring(0, 12))
if (Test-Path -LiteralPath $root) { throw 'Evidence destination already exists' }
New-Item -ItemType Directory -Path $root -ErrorAction Stop | Out-Null
python -m adaptive_reservoir.benchmarks.tdnr_experiment --partition development --output-dir (Join-Path $root 'development') --source-commit $sha
if ($LASTEXITCODE -ne 0) { throw 'Development experiment failed; preserve partial evidence' }
python -m adaptive_reservoir.benchmarks.tdnr_experiment --partition holdout --output-dir (Join-Path $root 'holdout') --source-commit $sha
if ($LASTEXITCODE -ne 0) { throw 'Holdout experiment failed; preserve partial evidence' }
python -m adaptive_reservoir.benchmarks.tdnr_report --development-dir (Join-Path $root 'development') --holdout-dir (Join-Path $root 'holdout') --output (Join-Path $root 'report.json')
if ($LASTEXITCODE -ne 0) { throw 'Evidence audit or report generation failed' }
Get-FileHash -Algorithm SHA256 (Join-Path $root 'development/manifest.json')
Get-FileHash -Algorithm SHA256 (Join-Path $root 'holdout/manifest.json')
Get-FileHash -Algorithm SHA256 (Join-Path $root 'report.json')
```

These commands create evidence outside the repository. They do not install dependencies, modify tracked source, commit, merge, or publish. The experiment and reporting commands do write new evidence files and therefore require explicit authorization. Each partition uses a write-once destination. An interrupted execution may leave incomplete evidence; preserve it for diagnosis rather than treating it as a successful run.

## Frozen protocol expectations

- Development: seeds 0–9; holdout: seeds 100–129. They must remain disjoint.
- One zero-noise reference and five nonzero levels on each of two noise axes, producing 11 unique conditions per seed.
- Four models and three scoring windows per condition.
- Expected development records: 1,440 scores, 144 aggregates, 108 contrasts.
- Expected holdout records: 4,320 scores, 144 aggregates, 108 contrasts.
- Combined evidence: 5,760 scores, 288 aggregates, 216 contrasts.
- Full runs are computationally expensive; no shortened substitute qualifies as the official experiment.

## Independent review requirements

1. Independently establish that the executed Git commit matches the approved frozen source and that the working tree was clean before execution. Record the verification method and reviewer identity outside the generated manifest.
2. Preserve original manifest and artifact bytes. Verify SHA-256, record counts, partition identities, protocol equality, and independent regeneration of aggregates and contrasts from saved scores.
3. Confirm that no holdout results were used to tune parameters, select configurations, change hypotheses, or exclude unfavorable outcomes.
4. Examine all models, noise levels, windows, MAE/RMSE values, and paired contrasts. Explicitly identify cases favoring baselines, cases favoring the adaptive reservoir, and intervals including zero.
5. Report absolute effect sizes and descriptive confidence intervals. Do not equate unadjusted bootstrap intervals with formal significance or infer real-world superiority from synthetic results.
6. If independently rerunning experiments, record hardware, Python, NumPy, BLAS and package versions. Compare numerical results with documented tolerances; byte-identical output across different environments is not guaranteed.
7. Record unresolved anomalies, incomplete runs, numerical discrepancies, and any inability to independently reproduce results. A failed or inconclusive study is a valid outcome.

## Completion gate

Stage 4C may be declared complete only after actual development and holdout runs, successful evidence verification, a complete unfiltered report, preserved hashes, and a documented independent review. TDNR-01 program completion requires a separate scientific assessment and explicit program-level decision. CI PASS alone is insufficient.
