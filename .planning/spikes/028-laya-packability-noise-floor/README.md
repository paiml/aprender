---
spike: 028
idea: llm-decision-classifier
name: laya-packability-noise-floor
type: standard
validates: "Given spike 027's candidate checkpoints, when each is re-scored over the full eval set in Rust fp32, torch fp32 and float64, then we know whether pack's 1e-5 probability bar and the ladder's 1e-4 final-block bar leave room for a gate-passing model, and if not, what noise-based bar the measurements support"
verdict: INVALIDATED
related: [025, 027]
tags: [laya, parity, pack, rescore, fp32-noise-floor, float64-reference, tolerance, contract-amendment]
---

# Spike 028: Laya packability vs the fp32 noise floor

> **These are MEASUREMENTS, not gate runs and not deploys.** Every `just laya-pack` below ran against a spike-027
> run dir carrying `NOT-A-GATE-RUN.txt`. No pack accepted, so no `.apr` was written (`models/decide/spike-028/` was
> created and removed empty). No contract, crate source or `scripts/laya_train` file was changed. The throwaway
> Rust example lived in `crates/aprender-decide/examples/` only while measuring; its copy is `examples/zz_triad_dump.rs`.

## What This Validates
Given 027's four candidate checkpoints, re-score every eval row three ways (Rust fp32 from the Rust builder's own
ids, which is what pack does; torch fp32 through `train.py`'s own `Scorer`, which is how the stored probability
files were written; and the same model in float64), then answer:
- Does `pack_rescore_probs_abs` = 1e-5 (laya-parity-v1) accept a model that passes the gate?
- Does the ladder's `final_norm_abs` = 1e-4 leave any headroom on real rows?
- If not, what bar do the measurements support?

## Research
- **Prior art:** the resolved debug session `.planning/debug/resolved/laya-rescore-drift.md`. It found that the
  1e-5 / 1e-4 bars come from spike 025's worst of **14 base-model rows** (2.6× / 3.8× headroom). On the old
  fixed-epochs checkpoint, torch's own fp32 answer is 3.7e-5 from float64. After the RoPE `inv_freq` fix
  (8e55e0bed), Rust's per-sub-op error equals torch's. Its throwaway tools (`torch_f64.py`, `triad.py`,
  `zz_rescore_dump.rs`) are the basis of the ones here.
- **What pack compares** (`crates/aprender-decide/src/verify.rs::rescore`): max over rows and components of
  |Rust p − stored torch p|, with argmax exact. The fine-tuned re-score runs first, and a `RescoreDrift` there
  returns before the zero-shot re-score. The gate is then recomputed from the **stored torch** probabilities.
- **Approach:** the one available. A float64 forward is the only reference that tells "Rust is wrong" apart from
  "torch is noisy". A second fp32 implementation cannot do that.

## How to Run
```bash
S=.planning/spikes/028-laya-packability-noise-floor
$S/run_packs.sh                        # real `just laya-pack` x4 on the 280-row test eval -> results/pack/
uv run --frozen --project scripts/laya_train python $S/build_heldout.py   # 027's 459-row in-dist set -> data/indist/
cp $S/examples/zz_triad_dump.rs crates/aprender-decide/examples/   # TEMPORARY
cargo build --release -p aprender-decide --example zz_triad_dump
$S/run_triads.sh test && $S/run_triads.sh indist   # Rust dump -> torch fp32 + float64 -> results/triad/*.json
rm crates/aprender-decide/examples/zz_triad_dump.rs            # before any commit
uv run --frozen --project scripts/laya_train python $S/analyze.py         # RESULTS.md, results/summary.json
uv run --frozen --project scripts/laya_train python $S/final_norm_locate.py base \
  .planning/spikes/027-laya-calibration-slice-and-tcap/data/s64 $S/results/final-norm-locate-base.json
```
Wall time on the M4 Pro (14 cores): about 3 min per pack, about 1 min for each Rust dump, and 20 to 40 s of torch
fp32 plus 40 to 70 s of float64 per 280 rows.

## What to Expect
- `results/pack/*.log`: one `PACKED`/`REFUSED` line per candidate, with the exit code and wall time.
- `results/triad/<set>-<run>.json`: summaries, per-row maxima, and **the full probabilities and logits of all
  three sides, including float64**. An x86_64 host can therefore be compared against the same f64 reference
  without re-running torch.
- `RESULTS.md`: every table, generated.

## Investigation Trail
1. **Real pack first.** `run_packs.sh` ran on the 280-row test eval, with each run's own 027 data dir
   (`data/s64` or `data/s16`; the hashes match each `gate-report.json`, or pack would have refused earlier).
   - **The only gate-passing checkpoint is refused:** s64-es12-seed17-rep gives `REFUSED RescoreDrift
     which=fine_tuned row=189 max_abs=1.445e-5`, exit 2.
   - s64-r1-seed13 is refused too (1.255e-5, row 95).
   - The two gate failers reach the gate and fail it on `ece_post` (exit 3), with 4.1e-6 and 3.7e-6.
2. **Triad on the same rows.** For every checkpoint the torch fp32 rerun is **bitwise equal to the stored file on
   280/280 rows**, and the manual fp32 forward used for f64 reproduces the Scorer's logits exactly
   (control 0.0). The Rust builder's ids equal torch's on 280/280 and 459/459. Rust−torch maxima reproduce pack's
   numbers exactly (1.445e-5 and 1.255e-5). The noise decomposition:
   - **seed17-rep row 189:** torch fp32 is **6.27e-5 from float64**, Rust is 4.83e-5 from float64, and they are
     1.45e-5 apart. **Rust is closer to the exact answer than the file it is judged against.** The model is refused
     for torch's rounding, not for a Rust defect.
   - **r1-seed13 row 95:** torch is 1.73e-5 from exact (2 rows over 1e-5); Rust's worst is 8.4e-6 (0 rows).
3. **Is drift gate-relevant?** Recomputing ECE and macro-F1 from each side's probabilities: |ΔECE| is at most
   1.5e-7, macro-F1 is identical, and argmax agrees 280/280 (459/459) for every pair. At this scale the drift is
   **four to five orders of magnitude below anything the gate can see**. What the bar actually decides is whether
   two fp32 rounding walks happened to land within 1e-5 of each other.
4. **Cap-10 probe.** r1-seed13 at T = 8.76 (softmax of each side's logits; the checkpoint config holds 5.0, so this
   was not a real pack): Rust−torch 1.01e-5 on test (1 row) and 2.94e-5 on held-out. A higher T does not rescue it.
5. **The held-out 459-row in-distribution set** (027's recommended gate eval set, rebuilt by the same rule:
   459 rows, classes [111, 291, 57], matching 027). Pack's check on it would be **worse**:
   - seed17-rep 2.98e-5 (2 rows, 3× over the bar);
   - r1-seed13 1.42e-5 (4 rows);
   - base (the zero-shot re-score pack also runs) 7.35e-6, while torch itself is 1.50e-5 from exact there.
   - More rows means a larger maximum from a heavy-tailed distribution. Moving the gate to 459 rows makes today's
     bar strictly harder to meet.
6. **Final-norm rung.** On real eval rows, the absolute 1e-4 bar is exceeded by **torch's own fp32 error** on 26 to
   90 % of rows (max 1.4e-2 to 4.6e-2). `final_norm_locate.py` on the base found:
   - the worst element sits on outlier channels 195 and 963 (values 4 to 12, token rms about 1), at non-marker tokens;
   - per-row relative rms error is at most 1.8e-4, well inside `per_layer_rel_rms` 1e-3;
   - marker tokens alone still exceed 1e-4 on 20/280 rows.
   The rung is only asserted on spike 025's 158-token fixture row (base, 3.05e-5 there), so it does not block pack,
   but it has **no headroom as a real-row bar** (1e-4 ÷ max = 0.00 to 0.04×). `logits_abs` 1e-4 likewise breaks on
   real rows: torch vs f64 reaches 7.9e-4 on test and 2.8e-3 on held-out for r1-seed13 (|z| up to 20).
7. **Surprise, correlated error.** On s64-es12-seed23-rep test row 97, Rust and torch agree to 3.7e-6 while *both*
   sit 3.7e-5 from float64. The two fp32 walks are not always independent draws. They share some rounding: the
   F16-widened weights, and (a caveat carried from the debug session) the f32 RoPE `inv_freq` buffer the f64 model
   inherits. So |Rust − torch| can understate both errors. The proposal below bounds it from the other side.
8. **Disk.** One Rust dump died with `StorageFull`: the volume transiently hit 0 bytes free from outside this spike
   (it read 5.3 GB free at start and 57 GB later). It was re-run cleanly. This spike never wrote more than about
   200 MB at a time.

## Results

### Per checkpoint, gate test eval (280 rows), aarch64 (Apple M4 Pro)

| checkpoint | gate (027) | `just laya-pack` | Rust−torch32 max (rows>1e-5) | Rust−f64 max (rows) | torch32−f64 max (rows) | final-norm Rust−torch max (1e-4 headroom) |
|---|---|---|---|---|---|---|
| **s64-es12-seed17-rep** | **PASS** (ECE 0.0959) | **exit 2 REFUSED RescoreDrift** ft row 189, 1.445e-5; zs not reached | 1.45e-5 (1) | 4.83e-5 (1) | **6.27e-5 (1)** | 1.07e-2 (0.01×) |
| s64-r1-seed13 (T=5) | fail @5, pass @10 | **exit 2 REFUSED RescoreDrift** ft row 95, 1.255e-5; zs not reached | 1.25e-5 (1) | 8.40e-6 (0) | 1.73e-5 (2) | 4.09e-2 (0.00×) |
| ↳ same, T = 8.76 (probe) | | not packable as-is (config holds 5.0) | 1.01e-5 (1) | 8.15e-6 (0) | 1.83e-5 (1) | – |
| s16-r1-seed13 | fail by 1e-4 | exit 3 GateFailed[ece_post] 0.10015; rescore 4.11e-6, zs 6.82e-6 | 4.11e-6 (0) | 4.36e-6 (0) | 3.48e-6 (0) | 2.51e-3 (0.04×) |
| s64-es12-seed23-rep | fail | exit 3 GateFailed[ece_post] 0.13907; rescore 3.67e-6, zs 6.82e-6 | 3.67e-6 (0) | 3.67e-5 (1) | 3.80e-5 (1) | 2.25e-2 (0.00×) |
| base (zero-shot) | – | (zs re-score inside the two exit-3 packs: 6.82e-6) | 6.82e-6 (0) | 6.71e-6 (0) | 7.92e-6 (0) | 1.53e-2 (0.01×) |

The p99 is 3 to 5e-6 and the median 3e-7 to 8e-7 everywhere (see `RESULTS.md`). **The tail is one or two rows
per checkpoint.** Argmax agrees 280/280 on every pair.

### Held-out in-distribution set (459 rows): what pack would see if the gate moved there (027's recommendation)

| checkpoint | Rust−torch32 max (rows>1e-5) | Rust−f64 | torch32−f64 | would pass 1e-5? |
|---|---|---|---|---|
| s64-es12-seed17-rep | **2.98e-5 (2)** | 3.48e-5 (4) | 4.66e-5 (2) | **no** |
| s64-es12-seed23-rep | 6.68e-6 (0) | 1.80e-5 (1) | 1.13e-5 (1) | yes (1.5×) |
| s64-r1-seed13 (T=5 / T=8.76) | 1.42e-5 (4) / 2.94e-5 (3) | 9.5e-6 / 7.1e-6 | 1.35e-5 / 2.99e-5 | no / no |
| base (zero-shot) | 7.35e-6 (0) | 1.56e-5 (2) | 1.50e-5 (1) | yes (1.36×) |

Gate metrics recomputed from Rust, torch or f64 probabilities differ by at most 1.5e-7 in ECE, with macro-F1 and
argmax identical.

### Verdict: INVALIDATED ✗
- **Under today's bars, a gate-passing model cannot be packed.** Seed17-rep, the only one of 28 spike-027 runs that
  passes at T ≤ 5, is refused with `RescoreDrift` (exit 2) on the gate's own eval set.
  - On 027's recommended 459-row set it would be refused by 3×.
  - The cap-10 candidate is refused at T = 5 and would be at T = 8.76.
- In every refusal, torch's own fp32 error against float64 is **larger** than the Rust−torch gap. That is 1.7 to
  6.3× the bar, and Rust is the more accurate side on the refusing row.
- Nothing about Rust is wrong: ids are exact, argmax is exact, and ECE and F1 are unchanged to 1e-7.
  **The 1e-5 bar is below the fp32 noise floor of real Laya checkpoints.**
- The ladder's 1e-4 final-norm bar has no headroom at all on real rows (torch vs f64 up to 4.6e-2). It survives only
  because it is asserted on a single base fixture row. It does not affect pack.

### Proposed bar (a contract amendment for the user; NOT made here)
Ground it in the triangle inequality, |Rust − torch| ≤ |Rust − f64| + |torch − f64|, and in what the measurements
show: Rust's own error is at most **1.6×** torch's (rust−f64 ÷ torch−f64 ranges from 0.24 to 1.59 across 11
checkpoint×set×T cells).

**Candidate A (recommended): noise-referenced.**
- Bar: `pack_rescore_probs_abs_c = max(1e-5, 4 × max_r |p_torch32(r) − p_f64(r)|)`, per checkpoint.
- The torch-vs-f64 term is measured by `train.py` at scoring time over the same rows (+40 to 70 s per 280 rows),
  written to the run dir and bound into the gate report's hashes.
- k = 4 means "Rust may be up to 3× as far from exact as torch is". k_needed peaks at **1.18** measured (1.26 for
  the old fixed-epochs checkpoint).
- Margin (bar ÷ observed Rust−torch) on every measured checkpoint:

  | | test 280 | held-out 459 |
  |---|---|---|
  | seed17-rep | 17.4× | 6.3× |
  | r1-seed13 T5 / T8.76 | 5.5× / 7.2× | 3.8× / 4.1× |
  | s16-r1-seed13 | **3.4×** (floor-bound: 4×3.5e-6 = 1.4e-5) | – |
  | seed23-rep | 41× | 6.8× |
  | base | 4.6× | 8.1× |
  | debug-session fixed-epochs (post-fix, test) | 3.2× | – |

  The minimum is **3.2×**. With k = 3 the minimum is 2.4×.
- Cost: a float64 forward in the training harness, and one more hash-bound file. The bar then moves with the
  checkpoint's conditioning (logits up to 20, T up to 8.8) instead of being fixed from 14 base rows.

**Candidate B: fixed absolute 1e-4.**
- Simpler, and needs no f64 at train time. Margins: 3.4× (seed17 held-out), 3.4× (r1 T8.76 held-out), 6.9× (seed17
  test); **2.1×** on the old fixed-epochs checkpoint.
- It still bounds the served-vs-gated ECE difference far below anything that matters (measured |ΔECE| ≤ 1.5e-7).
- It does not scale with conditioning: a more over-confident checkpoint than any measured here could exceed it for
  the same reason 1e-5 fails now.

Under either candidate:
- Keep **argmax exact**: 0 flips across 3 pairs × 3,236 row scorings.
- Keep the **1e-5 literal for the tiny fixtures and the spike-025 fixture rows**, where it has measured headroom.
- If `final_norm_abs` / `logits_abs` are ever extended to real rows, they must become relative (like
  `per_layer_rel_rms`) or noise-referenced too.
- **Declare the chosen bar in laya-parity-v1 before the next gate run is read (D-07).**

### x86_64: unmeasured (risk)
- No x86 host was available. Every number above is aarch64: Rust on the NEON 8×6 BLIS kernel with Apple libm
  `sinf`/`cosf`, and torch 2.14 CPU on the same box, which is where the stored probability files were written.
- If pack or `laya-verify` ever runs on x86_64 (a CI runner, or a Lambda-side check), Rust takes a different rounding
  walk: AVX2 or scalar GEMM, a different accumulation order, glibc libm. It is compared against an **aarch64 torch
  file**. Under 1e-5 that is a coin flip on every checkpoint above. The debug session showed a 1-ULP `sincos` codegen
  change is enough to flip a row.
- **The one measurement that settles it:** on an x86_64 host, build `examples/zz_triad_dump.rs` (temporarily, as
  here) and dump seed17-rep and the base on both row sets. Then compare Rust-x86 logits against the **float64 logits
  already committed** in `results/triad/*.json`, so no torch is needed on the host.
- If rust_x86−f64 stays within 3× torch32−f64 (the k = 4 assumption), candidate A carries over unchanged. If it does
  not, the bar needs a per-architecture term. Record the value in laya-parity-v1, as its `qa_gate` already demands
  ("first x86_64 value recorded here").

### Surprises
- **The refused model's Rust answer is closer to the truth than the torch file that refuses it** (row 189: Rust
  4.8e-5 off exact, torch 6.3e-5).
- Moving the gate to 027's larger in-distribution set makes the parity bar *harder* (the max over 459 rows is bigger),
  so 027's recommendation and today's 1e-5 bar are incompatible.
- Correlated fp32 error: Rust and torch can agree to 3.7e-6 while both are 3.7e-5 from f64. A Rust-vs-torch bar can
  therefore under-report as well as over-report.
- The final-norm error lives in two outlier channels (195 and 963) of non-marker tokens: absolute errors of 1e-2 at
  relative rms 1e-4.
