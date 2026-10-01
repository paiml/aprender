---
spike: 027
idea: llm-decision-classifier
name: laya-calibration-slice-and-tcap
type: standard
validates: "Given TweetEval stance, when Laya en-root is fully fine-tuned on s16 and s64 (12- vs 48-row calibration slices) with the fixed 12-epoch and the early-stopping recipes over seeds 13/17/23, then post-calibration ECE at T<=5 and T<=10, fitted-T vs eval-optimal T, and macro-F1 margin show whether any recipe clears ECE<=0.10 with margin>=0.05 within Laya's cap — or how much cap is needed"
verdict: PARTIAL
related: [024, 025, 026]
tags: [laya, calibration, temperature-scaling, ece, gate, few-shot, distribution-shift, mps-noise]
---

# Spike 027: Laya calibration slice × temperature cap

> **These are MEASUREMENTS, not gate runs.** No run here is a declared `laya-finetune-gate-v1` run, and none
> may be read as one. Every run dir carries `NOT-A-GATE-RUN.txt`. Whatever this spike recommends must be
> **declared in `contracts/laya-finetune-gate-v1.yaml` before the next gate run is read (D-07)**. `gate_max_ece`
> was not touched and is not proposed to move.

## What This Validates
Plan 08-08's D-19 stance demo failed the pre-declared gate twice, both times on the ECE clause
(fixed 12 epochs: ECE 0.377 at a clamped T of 5.0; early stopping: ECE 0.222 at T 3.14). The brief
(`.planning/todos/pending/spike-laya-calibration-slice-and-temperature-cap.md`) names two suspects: the
12-row calibration slice, and Laya's served temperature cap `TEMP_MAX = 5.0`. This spike puts
numbers on both over 3 seeds, and adds 2 recipe columns and 4 replicate runs.

## Research
- **Where 5.0 comes from.** Laya @ `4066d5d` `laya/common.py:376-377` sets `TEMP_MIN = 0.5` and `TEMP_MAX = 5.0`.
  The comment above them justifies only the **minimum**: the shipped `choice:11+` T of 0.10 sharpens a 0.24
  probability into 0.99. **Upstream gives no rationale for 5.0.** `Agent`'s loader clamps every stored
  temperature into the range and warns (`agent.py:425`), so a T above 5 written into `rl_agent_config.json` is
  served as 5. Raising the cap therefore means changing `laya-parity-v1`, the Rust clamp
  (`calibration_temp_max`), and diverging from upstream's loader.
- **The splits.** The TweetEval stance_abortion `train`/`validation` rows come from the SemEval-2016 training
  data, and `test` from its test set. Label priors are similar: test is 67.5 % against, and the in-distribution
  held-out set below is 63 % against. Difficulty is not: every model here is 0.1 to 0.2 accuracy points worse on
  test than on any train-distribution set.
- **Calibration split rule** (`data.calibration_split`): per class, `max(2, ceil(0.25 * n_class))` rows, seeded
  by the declared seed 13, grouped by text. That gives 4 per class = 12 rows at s16 and 16 per class = 48 rows at
  s64. Fit rows are 36 and 144.

## How to Run
```bash
uv run --frozen --project scripts/laya_train python .planning/spikes/027-laya-calibration-slice-and-tcap/prepare_data.py
.planning/spikes/027-laya-calibration-slice-and-tcap/run_grid.sh            # 24 runs, ~48 min on M4 Pro MPS
SPIKE027_KEEP=9 SPIKE027_REP=-rep uv run --frozen --project scripts/laya_train python \
  .planning/spikes/027-laya-calibration-slice-and-tcap/run_cell.py --size s64 --recipe es12 --seeds 13,17,23   # replicates
python3 .planning/spikes/027-laya-calibration-slice-and-tcap/aggregate.py   # RESULTS.md, results/summary.json, report.html
uv run --frozen --project scripts/laya_train python .planning/spikes/027-laya-calibration-slice-and-tcap/diagnose_indist.py
# also: demo_oracle.py (read-only on the D-19 run dirs), diagnose_prior.py, diagnose_valsplit.py
```
`scripts/laya_train` is imported and **not modified**. `run_cell.py` drives `train.SeedRun.train_and_score`,
the exact path `train.main` runs for every seed: MPS training, then the complete F16 checkpoint, then an
fp32 CPU reload, then the bounded T fit on the slice, then a second reload, then eval probabilities. It uses
`data.calibration_split` with the declared seed 13, and `gate.evaluate_gate` / `metrics.py`.

The spike-only hooks:
1. Epochs are passed straight through, so `r1` (4 epochs) runs at s16 too, although the contract fixes 12 there.
2. `es12m10` widens only the early-stopping monitor's T bound to 10.
3. Calibration-slice and eval logits are kept, so T can be refitted offline unclamped, at ≤5 and at ≤10.
4. The kept checkpoint is the run's own seed, and the split stays seed 13.

Cross-check: the wrapper's `s16/fixed12/seed13` and `s16/es12/seed13` recipe bytes hash to the demo's
`d0f4e40d…` and `3d4b91da…` (asserted in code), and the rebuilt s16 data is byte-identical to
`data/decide/tweet-stance-16`.

## What to Expect
- `RESULTS.md`: the per-cell, per-run and replicate tables. `results/runs/*.json`: every number per run.
  `results/{indist,valsplit,prior-shift,demo-oracle}.json`: the diagnostics.
- `report.html`: mean eval ECE against applied T per cell. The cap lines at 5 and 10 and the 0.10 bar show where
  each recipe's curve dips under the bar. The curve reads eval labels, so it is not a way to choose T.
- Run dirs (gitignored) are under `models/decide/spike-027/<size>-<recipe>-seed<s>[-rep]/`, in `train.py`'s
  `run_dir_layout` plus `spike-metrics.json`. Only 4 checkpoints are kept, because the disk had about 12 GB free.

## Investigation Trail
1. **Reproduce the demo first.** A read-only look at the three existing D-19 run dirs (`demo_oracle.py`) shows
   the brief's "ECE 0.1126 @ T≤5, 0.0389 @ T=10" belongs to the **early-stopping** checkpoint. The
   **fixed-epochs** checkpoint needs **T ≈ 29** (its best ECE at T≤10 is 0.263).
2. **Main grid** (s16/s64 × fixed12/es12 × seeds 13/17/23, 12 runs, about 29 min). **0/12 pass at T≤5, and
   0/12 at T≤10.** The two recipes fail differently:
   - **fixed 12 epochs:** the slice already asks for T of 12 to 26, so the cap binds. Even T≤10 leaves ECE at
     0.24 (s16 and s64). Eval would want T of 18 to 34.
   - **early stopping (the declared rule):** the slice's fitted T is **below 5 in 8 of 9 runs, replicates
     included**, so the cap does not bind. It fails because eval wants a higher T
     (eval-oracle 4 to 23).
3. **A bigger slice moves T in the right direction and reduces its spread, but does not close the gap.** Under
   fixed12, s64 fits T 13.7 ± 1.2 against s16's 18.7 ± 7.1. The eval-oracle T is still 24 ± 6. The slice
   accuracy is above eval accuracy in 26 of 28 runs: slice 0.70 on average at s64, against 0.55 on eval.
4. **Is the slice-vs-eval gap noise or shift?** Three diagnostics, all pointing the same way:
   - **Prior reweighting** (`diagnose_prior.py`): reweighting eval to a balanced prior lowers the eval-optimal
     T only partly. The gap is not label prior.
   - **TweetEval validation, 66 rows** (`diagnose_valsplit.py`): this is train-distribution data that no run
     saw. It fits T **at or below** the slice's T (for example s64-r1-seed13: slice 8.76, validation 4.89,
     eval-oracle 11.96), and its accuracy is 0.70 to 0.80 against eval's 0.58.
   - **An in-distribution held-out set of 459 rows** (`diagnose_indist.py`): validation plus every train-pool row
     in no shot, disjoint from the shots by text. On it, the **same checkpoints at their own slice-fitted T** are
     calibrated: **ECE 0.051 to 0.069 for all 3 s64 early-stopping replicates, with margin 0.23 to 0.31**. The
     in-distribution oracle T is 1.3 to 7.3.
   - **Conclusion:** the demo fails its ECE clause because of the **SemEval train→test shift**. The model is
     markedly less accurate on test than on anything drawn like its shots, so no train-side slice of any size
     can see the extra softening test needs. The 12-row slice's noise is real (the brief's SE estimate stands),
     but it is second-order.
5. **Cheap columns.**
   - **`r1` (4 epochs):** under-training keeps logits soft. At s16 the ECE before calibration falls to
     0.09 to 0.28, but the 12-row slice then fits T of 0.88 to 1.84. On seed 23 calibration made ECE *worse*
     (0.092 at Laya's shipped 1.76, 0.160 after the fit). s16-r1-seed13 misses by **1e-4** (ECE 0.1001).
     At s64, r1 has the best margin (0.221 ± 0.012) and wants T of about 9: **1/3 pass at T≤10, 0/3 at T≤5.**
   - **`es12m10`:** a monitor that may pick T up to 10 does not help. On s16 seed 23 it let over-fit epochs look
     good (best epoch 9 instead of 4), which drove the model back into the memorised regime (fitted T 14, ECE
     at T≤10 0.23). The monitor bound must stay at the served cap, whatever that cap is.
6. **MPS noise decides outcomes near the bar.** I re-ran s64 es12 on seeds 13/17/23 and s64 r1 on seed 23:
   - **The seed-17 replicate PASSES at T≤5**: ECE 0.0959, margin 0.206, T 1.51, best epoch 1. The original run
     of the same seed failed with ECE 0.144 at best epoch 2.
   - Replicate deltas: margin up to **0.074**, ECE at T≤5 up to **0.048**, the best epoch flips 2→1.
   - Other pairs:
     - s16/fixed12/seed13 against the demo `d0f4e40d`: margin 0.101 vs 0.130, ECE 0.405 vs 0.377.
     - s16/es12/seed13 against the demo `3d4b91da`: 0.106/0.221 vs 0.106/0.222 (a close match).
     - s64 es12m10 seed 13 stopped at the same epoch as es12 seed 13, yet its margin is 0.216 vs 0.152.
   - A single-seed gate run on MPS is a draw from a distribution whose spread straddles 0.10.
7. **Cut / not run.** Nothing from the requested grid was cut: 24 grid runs plus 4 replicates, about 57 min of
   MPS in total. The following were not measured:
   - x86_64, and the Rust / float64 re-score tails (these are spike 028's job);
   - more than 3 seeds per cell;
   - s32;
   - early-stopping at s64 with 4 epochs as the maximum;
   - any recipe fitting T on an in-distribution held-out set bigger than the slice (the validation-T result in
     step 4 predicts it would not help on test).

## Results

**Per cell, on the gate's eval set (280 TweetEval stance test rows): mean ± sd over seeds 13/17/23.**
ECE is the house top-label ECE with 15 bins. Zero-shot macro-F1 is 0.340 (the same base and rows as the demo).

| size / slice | recipe | margin | ECE pre | T fit (slice, unclamped) | ECE @T≤5 | ECE @T≤10 | EVAL-ORACLE T† (its ECE) | pass @5 | pass @10 |
|---|---|---|---|---|---|---|---|---|---|
| s16 / 12 | fixed 12 ep | 0.122 ± 0.031 | 0.408 | 18.7 ± 7.1 | 0.362 ± 0.073 | 0.239 ± 0.050 | 28.8 ± 7.1 (0.016) | 0/3 | 0/3 |
| s16 / 12 | early stop (declared) | 0.122 ± 0.037 | 0.335 | 3.5 ± 0.6 | 0.229 ± 0.113 | 0.229 ± 0.113 | 12.3 ± 9.5 (0.040) | 0/3 | 0/3 |
| s16 / 12 | fixed 4 ep (r1) ‡ | 0.112 ± 0.042 | 0.165 | 1.4 ± 0.5 | 0.194 ± 0.115 | 0.194 ± 0.115 | 4.3 ± 2.5 (0.049) | 0/3 | 0/3 |
| s16 / 12 | early stop, monitor T≤10 ‡ | 0.126 ± 0.018 | 0.382 | 6.4 ± 6.7 | 0.319 ± 0.081 | 0.275 ± 0.082 | 14.6 ± 7.0 (0.025) | 0/3 | 0/3 |
| s64 / 48 | fixed 12 ep | 0.190 ± 0.037 | 0.417 | 13.7 ± 1.2 | 0.366 ± 0.036 | 0.243 ± 0.047 | 24.2 ± 5.8 (0.029) | 0/3 | 0/3 |
| s64 / 48 | early stop (declared) | 0.188 ± 0.032 | 0.307 | 3.7 ± 1.3 | 0.161 ± 0.018 | 0.161 ± 0.018 | 7.8 ± 4.6 (0.043) | 0/3 | 0/3 |
| s64 / 48 | early stop, + 3 replicates (n=6) | 0.200 ± 0.025 | – | 3.6 ± 1.5 | 0.143 ± 0.028 | 0.143 ± 0.028 | – | **1/6** | 1/6 |
| s64 / 48 | fixed 4 ep (r1, epoch rule allows it) | **0.221 ± 0.012** | 0.375 | 9.3 ± 0.8 | 0.261 ± 0.033 | **0.125 ± 0.029** | 14.6 ± 2.8 (0.040) | 0/3 | **1/3** |
| s64 / 48 | early stop, monitor T≤10 ‡ | 0.206 ± 0.019 | 0.373 | 6.1 ± 2.3 | 0.198 ± 0.118 | 0.157 ± 0.048 | 11.9 ± 7.4 (0.072) | 0/3 | 0/3 |

† **Reads eval labels.** This is a diagnostic of how far away the needed T is, never a selection rule.
‡ Outside the declared recipe space: 4 epochs at ≤16 shots/class, and a monitor bound other than the served cap.

- **Slice vs eval accuracy:** the slice reads 0.58 to 0.71 and eval 0.48 to 0.59, cell by cell.
- **Epochs used:**
  - early stopping at s16: best epoch 4, stopped after 7;
  - early stopping at s64: best epoch 2, stopped after 5 (one replicate: best 1, stopped after 4);
  - es12m10: best epoch 2 to 9.

**The same checkpoints on 459 in-distribution held-out rows** (`results/indist.json`; zero-shot macro-F1 0.409;
each run's own slice-fitted T, so no label of this set chooses anything):

| run | in-dist margin | in-dist ECE @T≤5 | in-dist oracle T | vs gate eval: margin / ECE @T≤5 |
|---|---|---|---|---|
| s64 early-stop seed 13 (rep) | 0.313 | 0.069 | 4.0 | 0.226 / 0.139 |
| s64 early-stop seed 17 (rep) | 0.229 | 0.052 | 1.3 | 0.206 / 0.096 |
| s64 early-stop seed 23 (rep) | 0.263 | 0.051 | 3.0 | 0.203 / 0.139 |
| s64 r1 seed 13 | 0.323 | 0.092 (0.080 @T≤10) | 7.2 | 0.235 / 0.230 |
| s64 r1 seed 17 | 0.281 | 0.100 (0.064 @T≤10) | 7.3 | 0.216 / 0.259 |
| s16 early-stop seed 23 | 0.091 | 0.055 | 3.4 | 0.095 / 0.119 |
| s16 r1 seed 13 | 0.161 | 0.114 | 1.8 | 0.140 / 0.100 |

**Verdict: PARTIAL ⚠.**
- On the gate's eval set, no cell passes on all 3 seeds, at T≤5 or at T≤10.
- A pass exists but is not reproducible:
  - s64 + the contract's own early-stopping rule passes at T≤5 on **1 of 6** runs (a replicate of seed 17);
  - s64 + 4 epochs passes at T≤10 on **1 of 3** runs.

The central findings:
- **The cap is not the main lever.** Raising the cap to 10 rescues only fixed-epoch recipes, only at s64, and
  only 1/3 of the time. The declared early-stopping rule almost never reaches the cap: its slices fit T of 1.5 to 5.3,
  below 5 in 8 of 9 runs.
- **A 48-row slice is not a sufficient lever either.** It halves the fitted T's spread and lifts the margin
  (0.12 → 0.19 to 0.22), but the slice measures train-distribution confidence.
- **The binding constraint is the SemEval train→test shift.** On in-distribution held-out rows, the same
  s64 early-stopping checkpoints are calibrated, 3/3 replicates at ECE 0.05 to 0.07 within the unchanged cap.

**Candidate checkpoints for spike 028** (run dirs in `train.py`'s layout under `models/decide/spike-027/`):
1. **`s64-es12-seed17-rep/`**
   - Recipe: s64-seed13 data, `early_stopping` (the contract default; max 12 epochs; best epoch 1); seed 17;
     recipe_id `ccc5c233…`.
   - The only run that clears the gate at T≤5: margin 0.206, ECE 0.0959, T 1.51, inside Laya's cap.
   - Its `gate-report.json` says `pass: true`. It is **not** a gate run (`NOT-A-GATE-RUN.txt`), and ECE 0.0959
     leaves 0.004 of headroom, which is exactly where the 1e-5 re-score bar matters.
2. **`s64-r1-seed13/`**
   - Recipe: s64, fixed 4 epochs (legal under the epoch rule above 16 shots); seed 13; recipe_id `97e1db30…`.
   - Passes only at T≤10: T 8.76, ECE 0.0981, margin 0.235. The checkpoint's config holds T = 5.0, because
     Laya clamps anything higher.
   - 028 should re-score it at T = 8.76 to probe a cap-10 world.
3. **`s16-r1-seed13/`**
   - Recipe: s16, fixed 4 epochs (outside the ≤16 epoch rule); seed 13; recipe_id `dc7926e6…`.
   - A boundary vector: ECE 0.1001 at T 1.84, failing by 1e-4. A re-score drift of about 1e-5 could flip it.
4. Spare: `s64-es12-seed23-rep/`, the declared recipe, a typical fail (ECE 0.139).

**Recommendation for the one declared amendment.**
- **Do not raise the cap, and do not declare an s64-only amendment against the unchanged test eval.**
  - s64 + the early-stopping rule (with `--epochs 12` declared) has an expected pass rate of about
    **1/6 ≈ 17 %** for a seed-13 run (95 % CI roughly 0.4 % to 64 %). ECE at T≤5 sits at 0.143 ± 0.028.
  - Adding cap 10 plus 4 epochs gives about **1/3** (CI roughly 1 % to 91 %). It changes `laya-parity-v1` and the
    Rust clamp, and diverges from upstream, which has no stated reason for 5.0 but whose loader enforces it.
  - A third declared attempt of either kind most likely fails. D-07 would record it honestly, but it would teach
    nothing this spike has not already measured.
- **The one amendment I would bring to the user is a D-19 eval-set decision, taken together with s64:**
  - Keep `gate_max_ece` 0.10, the T bounds [0.5, 5] and the early-stopping rule exactly as declared.
  - Move D-19 to `s64-seed13` shots and declare `--epochs 12` (the maximum).
  - Declare the gate's eval set as an **in-distribution held-out set, defined by rule**: TweetEval stance
    validation plus every train-pool row in no s64-seed13 shot and not in the manifest exclusions, deduplicated
    by `nfc-trim-ws-v1`. That is 459 rows here.
  - Keep the SemEval test split as a **reported shift probe, not a gate clause**.
- **Expected outcome:** margin about 0.27 ± 0.04 (bar 0.05) and ECE about 0.057 ± 0.010 (bar 0.10).
  - Seed 13 passes with high but not certain probability: 3/3 unselected replicates passed, and the rule of three
    gives a lower 95 % bound of about 37 %.
  - MPS noise (ECE ± 0.02 to 0.05 between replicates) is small against this 0.04 headroom.
- **The cost is a claims change.** The gate would then certify calibration on data drawn like the tenant's shots.
  It would not certify robustness to a shifted input population, which is precisely what the test split exposed.
  That is a product decision for the user, not a tuning choice.
- If the user wants the shifted-population claim kept, no measured recipe delivers it. The honest record is
  "Laya few-shot FT is over-confident under split shift at this data size", with the D-18 deploy staying deferred.

**Surprises**
- The cap is rarely the problem for the declared rule: the slice fits T below 5 in 8 of 9 early-stopping runs,
  replicates included.
- Calibration made ECE worse (s16-r1-seed23: 0.092 → 0.160). A 12-row slice can pull T below 1.
- A monitor allowed a wider T bound picks over-fit epochs (es12m10 s16 seed 23: best epoch 9 instead of 4).
- One MPS replicate turned a gate FAIL into a PASS on the declared recipe, so single-seed gate runs on MPS are
  near-coin-flips at this margin.
