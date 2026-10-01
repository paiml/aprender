# Phase 5 Plan 01 — Production Calibration Measurements

**Purpose:** the measured evidence input to the D-04 human checkpoint (plan 05-03's deliberate
three-place contract edit). This file MEASURES. It edits no contract, no threshold table and no
gate.

**Host:** local dev box (Darwin 25.6.0, arm64), CPU only — not lambda-vector.
**Branch:** `gsd/phase-2-contract-gate` (02-01 policy; no PR).
**Harness:** `production_calibration_matrix` in
`crates/aprender-train/src/train/setfit/evidence.rs`, `#[ignore]`d and env-gated on
`APRENDER_MINILM_DIR`.

**Gate-independence (why measuring is legal today):** the `UncalibratedRegime` refusal lives in
`validate_evidence` (`tune.rs`, judgement time). `run_tuning` and
`UpdateEvidence::from_tune_output` carry no regime gate, so the production envelope can be
measured on today's code with zero relaxation. Nothing in this plan widens a gate.

---

## Frozen production hyperparameters

Read from the **pinned, hash-locked `uv` environment**, not from documentation, memory, or the
web (`scripts/setfit_fixtures/`, `setfit==1.1.3`).

### The literal command

```bash
cd scripts/setfit_fixtures && uv run python -c \
  "from setfit import TrainingArguments; a = TrainingArguments(); \
   print(a.num_epochs, a.batch_size, a.body_learning_rate)"
```

### Its stdout

```text
(1, 16) (16, 2) (2e-05, 1e-05)
```

### The environment those numbers came from

```text
setfit 1.1.3
transformers 4.57.6
sentence_transformers 5.7.0
torch 2.13.0
num_epochs (1, 16)
batch_size (16, 2)
body_learning_rate (2e-05, 1e-05)
max_length None
warmup_proportion 0.1
```

### Reading the tuples

Each of the three is a `(body, head)` pair. `SetFitTrainConfig` configures the **contrastive
body** stage — the only stage the evidence table measures — so the **body** member is the one
that maps onto its knobs. The head members (`16` epochs, batch `2`, lr `1e-05`) belong to the
logistic head, which this gate does not measure.

### The frozen values

```text
epochs = 1
batch  = 16
```

(reference encoder lr = `2e-05`, the harness's REAL condition.)

**Cross-check against the in-repo recipe.** `crates/aprender-train/src/train/setfit/config.rs`
already carries `REFERENCE_EPOCHS = 1`, `REFERENCE_BATCH_SIZE = 16`,
`REFERENCE_ENCODER_LR = 2e-5`. The harness **asserts** this agreement at construction
(`production_config`), so a future divergence between the pinned Python env and the Rust
reference recipe turns the harness red instead of silently measuring cells that production runs
never enter.

**Consequence (F-R3):** every cell label this milestone can enumerate is
`s{shots}e1b16`, and therefore the D-02 contract entry's `cells=` component is
`s8e1b16,s16e1b16,s32e1b16,s64e1b16`. These two values were frozen **before** any calibration
pass, which is the ordering the plan's must-have truth requires — they are baked into every cell
label and cannot be chosen after seeing a result.

---

## Probe timing and projection

The probe ran **before** the boundary matrix, and the matrix projection below is derived from it.
That ordering is the point (CLAUDE.md: check in BEFORE >1 hr compute on non-lambda-vector hosts).

### The literal command

```bash
CARGO_INCREMENTAL=0 APRENDER_CALIBRATION_PROBE=1 \
  cargo test -p aprender-train --lib --features setfit production_calibration \
  -- --ignored --nocapture > /tmp/probe.log 2>&1
rc=$?
```

Status is read on the next line, never through a pipe (CLAUDE.md verification rule 1). The
timing wrapper was a throwaway script outside the tracked tree; it is exactly:

```bash
START=$(date +%s); START_ISO=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
CARGO_INCREMENTAL=0 APRENDER_CALIBRATION_PROBE=1 \
  cargo test -p aprender-train --lib --features setfit production_calibration \
  -- --ignored --nocapture >/tmp/probe.log 2>&1
rc=$?
END=$(date +%s)
echo "probe_wall_clock_secs=$((END - START)) rc=${rc}"
```

### Measured (debug profile)

```text
probe_start_iso=2026-08-17T04:11:31Z
probe_end_iso=2026-08-17T04:41:42Z
probe_wall_clock_secs=1811
rc=0
```

Breakdown, from the harness's own printed report and the libtest summary:

| Quantity | Value |
|---|---|
| whole `cargo test` invocation | 1811 s (30.2 min) |
| — of which incremental recompile + binary start | ~46 s |
| libtest `finished in` | 1764.62 s |
| `run_tuning` alone (harness `wall_clock`) | 1758.6 s |
| fixed per-pass overhead (corpus + checkout load + prepare + evidence) | ~6.0 s |
| optimizer steps in the pass | 24 |
| **cost per optimizer step** | **73.3 s/step** |

### The regime id the run rendered

Quoted verbatim from `/tmp/probe.log`:

```text
CALIBRATION REGIME: minilm-slice-h384-l6-a12-i1536-v30522@1110a243
```

```text
minilm-slice-h384-l6-a12-i1536-v30522@1110a243|seeds=13|cells=s8e1b16
```

It begins with the required `minilm-slice-h384-l6-a12-i1536-v30522@1110a243` — the
`minilm-slice-` prefix is hardcoded in `BertSentenceEncoder::architecture_fingerprint` and
renders for the full model too (F-R2), and the `h384-l6-a12-i1536-v30522` dimensions prove the
**production** encoder was loaded, not the 97-token slice. This string was **rendered by the
production code path** (`calibration_regime_id`), never composed by the harness (T-05-01-01).

### Projection arithmetic for the 18-pass boundary matrix

Step count per cell is set by the **contracted default pair budget**, not by the shot count
directly. For a uniform selection of `n` rows across `K = 3` classes
(`aprender-contrastive-data/src/pairs.rs`):

```text
positive_capacity = K · C(n,2)      = 3 · n(n−1)/2
negative_capacity = C(K,2) · n²     = 3 · n²
budget            = 2 · max(pos, neg) = 6n²      (negatives dominate for all n ≥ 1)
steps             = ceil(budget / batch) = ceil(6n² / 16)
```

| cell | pos_cap | neg_cap | budget | steps @ b16, e1 | steps relative to s8 |
|---|---|---|---|---|---|
| `s8e1b16`  | 84   | 192   | 384    | **24**   | 1× |
| `s16e1b16` | 360  | 768   | 1536   | 96       | 4× |
| `s32e1b16` | 1488 | 3072  | 6144   | 384      | 16× |
| `s64e1b16` | 6048 | 12288 | 24576  | **1536** | **64×** |

The measured `steps=24` for the probe confirms the closed form against the running code.

> **The plan's `~8×` weighting is REFUTED by this measurement.** 05-01 projected "s64 cells
> weighted ~8x s8 (rows scale)". Rows do scale 8×, but the *budget* is **quadratic** in shots
> because `negative_capacity = 3n²` dominates, so an s64 pass is **64×** an s8 pass, not 8×.
> Correcting this is the difference between a ~6-hour projection and a ~12-day one, so it is
> recorded here rather than silently applied.

**Debug-profile projection**, at 73.3 s/step + 6.0 s fixed per pass:

| group | passes | steps/pass | s/pass | subtotal |
|---|---|---|---|---|
| `s8e1b16` × {13,31,53} × {real, ctrl, near-null} | 9 | 24 | 1 765 s | 15 881 s (4.4 h) |
| `s64e1b16` × {13,31,53} × {real, ctrl, near-null} | 9 | 1 536 | 112 556 s | 1 013 004 s (281.4 h) |
| **total** | **18** | — | — | **1 028 885 s ≈ 285.8 h ≈ 11.9 days** |

### Release-profile cross-check (and a result that matters for 05-03)

285 hours is far enough over the compute gate that "a release build would fix it" had to be
**measured**, not assumed (CLAUDE.md verification rule 2). It is also the more faithful profile:
a user's `apr setfit train` is a release binary (`cargo install aprender`).

```bash
CARGO_INCREMENTAL=0 APRENDER_CALIBRATION_PROBE=1 \
  cargo test --release -p aprender-train --lib --features setfit production_calibration \
  -- --ignored --nocapture > /tmp/probe_release.log 2>&1
```

```text
release_probe_start_iso=2026-08-17T04:42:24Z
release_probe_end_iso=2026-08-17T04:46:04Z
release_probe_wall_clock_secs=220
rc=0
```

| Quantity | debug | release | ratio |
|---|---|---|---|
| `run_tuning` wall clock (s8, 24 steps) | 1758.6 s | **51.0 s** | 34.5× |
| per optimizer step | 73.3 s | **2.125 s** | 34.5× |
| fixed per-pass overhead | ~6.0 s | ~0.2 s | — |

**The two profiles produced BIT-IDENTICAL measurements.** Every printed relative delta, every
binding parameter, every `delta_norm` / `init_norm` / `grad_norm_max` / noise floor agrees to
the last printed digit across the two runs — e.g. `embedding real_min 1.813e-3`,
`projection_weight real_min 1.231e-3`, `attention_key_bias real_min 1.714e-7`, and the binding
row `embeddings.word_embeddings.weight … delta_norm=1.159e-2 init_norm=1.904e2`.

This is a load-bearing result for 05-03, not a performance footnote: **the profile is not a
degree of freedom in the calibration.** Whatever ε is frozen from a release-profile matrix is
the same ε a debug-profile matrix would have produced, so choosing release for the compute
budget does not change what is being measured.

**Release-profile projection**, at 2.125 s/step + 0.2 s fixed per pass:

| group | passes | steps/pass | s/pass | subtotal |
|---|---|---|---|---|
| `s8e1b16` × {13,31,53} × {real, ctrl, near-null} | 9 | 24 | 51.2 s | 461 s (7.7 min) |
| `s64e1b16` × {13,31,53} × {real, ctrl, near-null} | 9 | 1 536 | 3 264 s | 29 378 s (8.16 h) |
| **total** | **18** | — | — | **29 839 s ≈ 8.29 h** |

### Preliminary observation carried into Task 2 (not a conclusion)

`attention_key_bias` — the class the fixture leaves **ungated** on the gradient-free argument
(`dL/db_k = 0` by softmax shift-invariance) — is not bit-frozen at production scale:
`grad_norm_max = 8.084e-10`, `real_min relative_delta = 1.714e-7`, about 150× its own
`rounding_noise_floor` of `1.133e-9`. Whether that is f32 reduction residue or a real gradient
cannot be decided without the 1e-30 and 1e-8 controls, which is precisely what the boundary
matrix runs. Recorded here so the Task 2 re-derivation is answering a question that was already
open, rather than one invented after seeing its own result.

---

## COMPUTE GATE — checkpoint reached, boundary matrix NOT run

CLAUDE.md, "Check in BEFORE acting": *compute spend > 1 hr on non-lambda-vector hosts*
(lambda-vector is pre-authorized). Plan 05-01 Task 1 step (4) restates it as a hard 60-minute
projection check before the matrix.

**Projection: 8.29 h (release) / 285.8 h (debug). Both exceed 60 minutes by more than 8×.**
Execution stopped here. The boundary matrix has NOT been run, Tasks 2 and 3 are not started, and
no ε has been derived.

### Why this is larger than the plan expected

Two independent factors, both measured above:

1. **The pair budget is quadratic in shots, not linear.** The plan projected s64 at ~8× s8
   ("rows scale"). The contracted default budget is `2·max(pos_cap, neg_cap)` and
   `neg_cap = 3n²` dominates, so s64 is **64×** s8. This alone is an 8× projection error.
2. Nothing about the per-step cost was surprising — 2.1 s/step in release for a 22M-parameter
   encoder over 32 texts is ordinary. The matrix is expensive because it is 18 passes of which
   nine are 1 536-step passes.

### Options for the human (go / no-go)

| # | Option | Measured/derived cost | What it costs in evidence |
|---|---|---|---|
| A | Run the full matrix locally, **release** profile | **8.29 h** wall clock on this dev box | Nothing — this is the plan as written, at the measured price. Release is bit-identical to debug (proven above). |
| B | Move the matrix to **lambda-vector** (pre-authorized) | 8.29 h scaled by that host's CPU; not measured here | Nothing, plus it removes the check-in requirement entirely. Needs the 86.7 MB checkout materialized there. |
| C1 | Trim the shot boundary to **{s8, s16}** | 9×51.2 + 9×204.2 = **2 299 s ≈ 38 min** — under the gate | Weakest. s32/s64 would be covered by extrapolating a 16×/64× budget factor the matrix never observed. |
| C2 | Trim the shot boundary to **{s8, s32}** | 9×51.2 + 9×816.2 = **7 807 s ≈ 2.17 h** | Moderate: s64 covered by a single 4× budget extrapolation. Still over the gate. |
| C3 | Keep {s8, s64}, trim to **one seed (31, the median)** | 3×51.2 + 3×3 264 = **9 947 s ≈ 2.76 h** | Loses the cross-seed spread, which is what makes the window a window rather than one run's number. Still over the gate. |
| D | Keep all cells but **pin an explicit small pair budget** | ~38 min | **Not recommended, and flagged rather than offered as equal.** The cell label `s{shots}e{E}b{B}` does not record the budget, so an entry claiming `s64e1b16` coverage measured at a trimmed budget would label cells that trained 64× less than the runs the entry licenses. That is a fidelity break, not a trim. |

**Recommendation (mine, non-binding): B, else A.** The measurement is the keystone of the whole
milestone (D-01), the ε it produces gates every Phase 5 benchmark cell, and options C1–C3 all
buy time by making the margin argument weaker in exactly the dimension the cross-AI review
already flagged as the plan's soft spot (six measured cells as an engineering margin for the
other 34). Paying 8.3 h once is cheaper than defending a thinner envelope at the D-04 checkpoint.

### What is already in hand regardless of the decision

- The harness (`production_calibration_matrix`) exists, is `#[ignore]`d, env-gated, and green.
- E/B are frozen from the pinned env with the command and stdout recorded.
- The production regime id is rendered and quoted verbatim.
- Debug/release measurement equivalence is proven.
- The plan's 8× projection error is corrected to 64×.

Resuming after a go/no-go needs only Task 2 onward; nothing above is re-run.

---

## COMPUTE GATE RESOLVED — Option B chosen; dispatch BLOCKED

**Human decision (recorded 2026-08-17): Option B — run the full contracted boundary matrix on
the lambda-vector host.** Full matrix as specified: `{s8, s64}` × seeds `{13, 31, 53}` ×
`{real, control, near-null}` = 18 passes. Trims C1/C2/C3 rejected; option D rejected. The
rationale on record: lambda-vector is pre-authorized for compute per CLAUDE.md, so the >1 hr
check-in requirement is *removed rather than waived*, and it costs zero evidence.

**The dispatch could not be performed: `lambda-vector` is not reachable from this host.**

### The reachability evidence (mechanism, not intent — CLAUDE.md verification rule 2)

Executing host, read from the machine rather than assumed:

```text
$ hostname
MacBook-Pro-7.local
$ uname -a
Darwin MacBook-Pro-7.local 25.6.0 ... RELEASE_ARM64_T6041 arm64
```

Every resolution and connection path was tried, and each failed:

| Probe | Command | Result |
|---|---|---|
| DNS / mDNS | `ping -c 1 lambda-vector` | `ping: cannot resolve lambda-vector: Unknown host` |
| Directory service | `dscacheutil -q host -a name lambda-vector` | no records returned |
| SSH (non-interactive) | `ssh -o BatchMode=yes -o ConnectTimeout=5 lambda-vector hostname` | **rc=255** — `ssh: Could not resolve hostname lambda-vector: nodename nor servname provided, or not known` |
| SSH client config | `grep -i '^Host ' ~/.ssh/config` | one entry only: `rvsc  ec2-54-72-80-139.eu-west-1.compute.amazonaws.com` — an AWS EC2 box, not lambda-vector |
| Known hosts | `grep -ci lambda ~/.ssh/known_hosts` | `0` |
| Overlay network | `command -v tailscale` | not installed |
| In-repo dispatch path | `grep -rIn ssh scripts/` | no script dispatches to lambda-vector; the three `scripts/` hits naming it are prose comments about its disk layout and GPU, not a connection |
| Named pre-auth doc | `~/.claude/projects/…/memory/feedback_compute_pre_authorized.md` | not present (the memory dir holds six unrelated files) |

Every status above was read directly (`cmd > log 2>&1; rc=$?`), never through a pipe
(CLAUDE.md verification rule 1).

**The matrix was NOT run locally.** Falling back to the local box would have spent the 8.29 h the
human's decision explicitly redirected, and would have converted a pre-authorized spend into an
unauthorized one. Execution halted instead.

### A caveat the next dispatcher needs

lambda-vector is a **GPU** host, but this measurement is **CPU-bound by construction**:
`production_config` sets `device: "cpu"`, and the SetFit contrastive trainer has no GPU path.
So option B's benefit is *authorization*, not speed — exactly as the human's rationale stated.
The 8.29 h figure is this box's Apple-silicon CPU; **lambda-vector's wall-clock is unknown and
could be higher or lower**, and must be re-projected there with the same one-cell probe before
the full matrix is launched. Do not carry 8.29 h over as if it were a measurement of that host —
that is precisely the "label a run by intent" error.

### Exact commands to run once a dispatch path exists

```bash
# 0. prerequisite: materialize the 86.7 MB pinned checkout on the target host
cd scripts/setfit_fixtures && uv run python fetch_full_weights.py

# 1. re-project on THAT host first (one cell, REAL only) — same gate discipline
CARGO_INCREMENTAL=0 APRENDER_CALIBRATION_PROBE=1 \
  cargo test --release -p aprender-train --lib --features setfit production_calibration \
  -- --ignored --nocapture > /tmp/probe.log 2>&1
rc=$?

# 2. the full 18-pass boundary matrix (probe env var unset)
CARGO_INCREMENTAL=0 \
  cargo test --release -p aprender-train --lib --features setfit production_calibration \
  -- --ignored --nocapture > /tmp/matrix.log 2>&1
rc=$?
```

`--release` is correct and is **not** a weakening of the calibration: the debug/release
bit-identity proven above means the profile is a price, not a degree of freedom. A reader who
does not know that will assume a release-mode calibration is the weaker one, so it is stated
here explicitly as well as in the cross-check section.

The harness writes its report to `$SETFIT_PRODUCTION_CALIBRATION_REPORT` (default
`$TMPDIR/setfit-production-calibration.txt`), which is the artifact to transport back — a
summarizing wrapper around `cargo test` drops `--nocapture` output entirely.

### What Task 2 still owes when it runs

- 6 per-cell tables (`s8`/`s64` × seeds 13/31/53) over every `ParameterClass`, with the columns
  the contract's DERIVATION invariant names.
- `ctrl_max < real_min` recorded for every gated class in every cell (the harness asserts it).
- At least one full regime id quoted verbatim, architecture component
  `minilm-slice-h384-l6-a12-i1536-v30522@1110a243`.
- The **`attention_key_bias` verdict**, which is the open question from the probe and is
  load-bearing for 05-03: at production scale that class shows
  `grad_norm_max = 8.084e-10` and `real_min relative_delta = 1.714e-7`, about 150× its own
  `rounding_noise_floor` of `1.133e-9`. The fixture leaves the class **ungated** on the
  gradient-free argument (`dL/db_k = 0` by softmax shift-invariance). The 1e-30 and 1e-8
  controls decide whether that argument survives production scale. If it does **not**, that
  changes 05-03's proposed regime entry and must be surfaced loudly, not absorbed.

---

## DISPATCH RESOLVED — Option A, run locally, explicitly authorized

**Human decision (recorded 2026-08-17, superseding the Option B attempt): Option A — run the
FULL contracted boundary matrix on this host, in release. ~8.29 h of local compute explicitly
authorized.** The CLAUDE.md ">1 hr on non-lambda-vector hosts" check-in was satisfied by that
authorization, not bypassed. No trim (not C1/C2/C3) and no pinned pair budget (not D).

### Why running locally is the right engineering choice here, not a convenience

**A future reader must not conclude the matrix was run locally because lambda-vector was
awkward to reach.** It was not a compromise; on the evidence it is plausibly the *fastest*
available option. The reason is that this measurement is **CPU-bound by construction, on every
host**:

1. `production_config` (in `evidence.rs`) hardcodes `device: "cpu"` — verified by reading.
2. Phase 3 **refuses** any non-CPU resolved device outright. `tune_rejects_a_non_cpu_resolved_device`
   in `tune.rs` asserts that `preflight(&mut encoder, Device::Cuda { index: 0 }, …)` returns
   `SetFitTrainError::UnsupportedDeviceForPhase3 { resolved: "cuda:0" }`, and carries a CPU
   control (`preflight(…, Device::Cpu).is_ok()`) so the refusal is not vacuous. A
   CUDA-resolved device **fails closed by design**.

So lambda-vector — a GPU host — would have run this on *its CPU* too. Option B's benefit was
never speed; it was that lambda-vector is pre-authorized for compute. With Option A the human
supplied that authorization directly, and the work runs on an Apple-silicon CPU that is
plausibly faster than the remote host's.

This is recorded rather than assumed: the harness now reads the **probed** device off
`ResolvedSetFitConfig` after `prepare()` and asserts it is CPU, then prints it per pass
(`device=Cpu` in the wall-clock table). A report that said "cpu" because the request string said
"cpu" would prove nothing about what executed — CLAUDE.md verification rule 2.

### Release profile is a price, not a degree of freedom

Stated again here because it is the single most misreadable choice in this document: the matrix
runs under `--release`. **This is not a weaker calibration.** The debug/release cross-check above
showed the two profiles agree to the last printed digit on every relative delta, every binding
parameter, and every `delta_norm` / `init_norm` / `grad_norm_max` / noise floor. Release changes
what the run *costs* (34.5× faster), never what it *measures*. Release is also the profile a
user's `apr setfit train` actually executes, since that ships as a `cargo install` binary.

### Sizing uses the measured closed form, never the plan's 8× assumption

All projections in this document use `budget = 2·max(pos_cap, neg_cap)` with
`neg_cap = 3n²` — quadratic in shots — confirmed against the harness's own measured `steps=24`
at s8. s64 is 1536 steps, **64×** s8. The plan's "~8×" is stale and is not used anywhere.

### The command that was run

```bash
CARGO_INCREMENTAL=0 \
  cargo test --release -p aprender-train --lib --features setfit production_calibration \
  -- --ignored --nocapture > /tmp/matrix.log 2>&1
rc=$?
```

Status read directly on the next line, never through a pipe (verification rule 1). Host identity
captured from the machine (`hostname`, `uname -a`) rather than asserted.

---

## Boundary matrix — PARTIAL (s8 half complete, s64 half not run)

**Status: 9 of 18 passes measured.** The `s8e1b16` half is complete across all three seeds with
the full three-condition treatment. The `s64e1b16` half has **not** been run. No ε is frozen in
this document; that is Task 3, and it needs the s64 cells.

### What happened to the first full-matrix attempt

The unchunked 18-pass run was launched (`matrix_start_iso=2026-08-17T05:12:29Z`, host
`MacBook-Pro-7.local`, `Darwin … RELEASE_ARM64_T6041 arm64`) and was **killed externally at
pass 9 of 18**, about 40 minutes in.

This was **not** a defect and not a measurement failure:

- no panic, no test-assertion failure, no compile error — `grep -in "panic\|assertion\|FAILED\|error\["`
  over `/tmp/matrix.log` matched only two build-script lines about YAML contract assertion counts;
- the last log line is an ordinary progress line;
- the wrapper never reached its epilogue, so `/tmp/matrix_timing.txt` has no `rc=` line — the
  signature of a terminated process rather than an exited one;
- all nine completed passes were `s8`, all `device=Cpu`, all `steps=24`, all ~48.7 s.

It cost the per-class tables for those nine passes, because the harness only wrote its report
after the final pass. **That defect is now fixed** (see below), and the s8 half was re-measured.

### The fix, and why it is not a scope change

Two changes to the harness, verified by running them rather than by inspection:

1. **Incremental persistence.** The report is rewritten after every cell with a
   `STATUS: PARTIAL` banner instructing the reader to treat any absent cell as *unmeasured*,
   never as passing; the final write carries `STATUS: COMPLETE`. A kill now costs at most the
   cell in flight.
2. **`APRENDER_CALIBRATION_CELLS="s8:13,s8:31,s8:53"`** — a named cell subset run with the
   **full three conditions**, so the 18 passes can execute as resumable chunks. This
   **partitions** the work; it does not shrink it. Each chunk runs exactly the passes,
   conditions and separation assertions the unchunked matrix would have run for those cells, so
   `{s8,s64} × {13,31,53}` split across invocations is the *same measurement* as one
   invocation. It is deliberately distinct from `APRENDER_CALIBRATION_PROSPECTIVE`, which stays
   REAL-only because it validates rather than derives. The cross-cell ε basis is unaffected —
   the window rule is defined over "all measured cells", so it composes from the union of the
   per-cell tables.

### The s8 half, measured

`APRENDER_CALIBRATION_CELLS="s8:13,s8:31,s8:53"`, release profile, `rc=0`, 542 s wall clock,
439.4 s in `run_tuning` over 9 passes, `STATUS: COMPLETE`. Every pass `device=Cpu`,
`steps=24` — the probed device read off `ResolvedSetFitConfig`, not echoed from the request.

| cell | class | real_min | real_median | real_max | ctrl_max | nnull_max | nnull_moved | noise_floor | support_frac | all_moved |
|---|---|---|---|---|---|---|---|---|---|---|
| s13 | embedding | 1.813e-3 | 2.611e-3 | 3.068e-3 | 0.000e0 | 2.761e-6 | true | 1.775e-6 | 0.0195 | true |
| s13 | layer_norm_weight | 1.891e-4 | 2.177e-4 | 2.605e-4 | 0.000e0 | 6.635e-9 | false | 5.960e-8 | 1.0000 | true |
| s13 | layer_norm_bias | 7.281e-4 | 2.315e-3 | 2.989e-3 | 0.000e0 | 1.678e-6 | true | 5.960e-8 | 1.0000 | true |
| s13 | projection_weight | 1.231e-3 | 2.085e-3 | 3.154e-3 | 0.000e0 | 1.804e-6 | true | 5.960e-8 | 1.0000 | true |
| s13 | projection_bias | 3.447e-4 | 1.784e-3 | 3.182e-3 | 0.000e0 | 1.814e-6 | true | 5.960e-8 | 1.0000 | true |
| s13 | attention_key_bias | 1.714e-7 | 2.033e-7 | 3.110e-7 | 0.000e0 | 3.105e-11 | true | 1.133e-9 | 0.9987 | true |
| s31 | embedding | 1.823e-3 | 2.642e-3 | 3.043e-3 | 0.000e0 | 2.846e-6 | true | 1.741e-6 | 0.0195 | true |
| s31 | layer_norm_weight | 1.914e-4 | 2.141e-4 | 2.617e-4 | 0.000e0 | 5.529e-9 | false | 5.960e-8 | 1.0000 | true |
| s31 | layer_norm_bias | 7.112e-4 | 2.208e-3 | 2.924e-3 | 0.000e0 | 1.650e-6 | true | 5.960e-8 | 1.0000 | true |
| s31 | projection_weight | 1.232e-3 | 2.054e-3 | 3.095e-3 | 0.000e0 | 1.788e-6 | true | 5.960e-8 | 1.0000 | true |
| s31 | projection_bias | 3.730e-4 | 1.633e-3 | 3.140e-3 | 0.000e0 | 1.785e-6 | true | 5.960e-8 | 1.0000 | true |
| s31 | attention_key_bias | 1.721e-7 | 2.419e-7 | 2.727e-7 | 0.000e0 | 5.162e-11 | true | 1.133e-9 | 0.9961 | true |
| s53 | embedding | 1.859e-3 | 2.631e-3 | 3.073e-3 | 0.000e0 | 2.808e-6 | true | 1.779e-6 | 0.0195 | true |
| s53 | layer_norm_weight | 1.895e-4 | 2.202e-4 | 2.578e-4 | 0.000e0 | 6.635e-9 | false | 5.960e-8 | 1.0000 | true |
| s53 | layer_norm_bias | 7.148e-4 | 2.241e-3 | 2.900e-3 | 0.000e0 | 1.592e-6 | true | 5.960e-8 | 1.0000 | true |
| s53 | projection_weight | 1.261e-3 | 2.063e-3 | 3.109e-3 | 0.000e0 | 1.745e-6 | true | 5.960e-8 | 1.0000 | true |
| s53 | projection_bias | 3.734e-4 | 1.838e-3 | 3.157e-3 | 0.000e0 | 1.738e-6 | true | 5.960e-8 | 1.0000 | true |
| s53 | attention_key_bias | 1.801e-7 | 2.028e-7 | 3.364e-7 | 0.000e0 | 4.430e-11 | true | 1.133e-9 | 0.9974 | true |

(cell label is `s{seed}` above; every row is cell `s8e1b16`.)

**Separation holds for every gated class in every s8 cell.** `ctrl_max` is `0.000e0`
throughout — the 1e-30 control underflows every parameter's ULP and writes back bit-identical
weights, exactly as the fixture matrix found — so `ctrl_max < real_min` is satisfied with the
widest possible margin. The harness asserts this per class per cell and the run exited `rc=0`,
so the assertion is load-bearing rather than decorative.

### s8 cross-cell basis (NOT the frozen ε — s64 is still missing)

| class | worst_ctrl | worst_nnull | best_real | 10× lower | 10× upper | noise_floor | eps/noise | nnull_moved | window exists |
|---|---|---|---|---|---|---|---|---|---|
| embedding | 0.000e0 | 2.846e-6 | 1.813e-3 | 2.846e-5 | 1.813e-4 | 1.779e-6 | 1.02e2 | true | yes |
| layer_norm_weight | 0.000e0 | 6.635e-9 | 1.891e-4 | 6.635e-8 | 1.891e-5 | 5.960e-8 | 3.17e2 | **false** | yes |
| layer_norm_bias | 0.000e0 | 1.678e-6 | 7.112e-4 | 1.678e-5 | 7.112e-5 | 5.960e-8 | 1.19e3 | true | yes |
| projection_weight | 0.000e0 | 1.804e-6 | 1.231e-3 | 1.804e-5 | 1.231e-4 | 5.960e-8 | 2.07e3 | true | yes |
| projection_bias | 0.000e0 | 1.814e-6 | 3.447e-4 | 1.814e-5 | 3.447e-5 | 5.960e-8 | 5.78e2 | true | yes |
| attention_key_bias | 0.000e0 | 5.162e-11 | 1.714e-7 | 5.162e-10 | 1.714e-8 | 1.133e-9 | **1.51e1** | true | yes |

A window exists for every class on the s8 half. **These numbers are not the frozen ε**: the
window rule takes the worst control and the best real across *all* measured cells, and the s64
cells — 64× the optimizer steps — are precisely the ones most likely to move `best_real`.

---

## gradient_free re-derivation — PRELIMINARY (s8 only)

This is the question 05-03's contract edit depends on, so it is stated carefully and its
current evidentiary status is stated with it.

**The claim under test.** The fixture leaves `attention_key_bias` **ungated** on a physics
argument: `dL/db_k = 0` because softmax is invariant to a constant shift of its logits, so the
attention key bias cannot receive gradient. The contract's own
"RE-DERIVATION, NOT RELAXATION" clause requires re-checking that at production scale.

**What the s8 half measured** (3 seeds, full 3 conditions):

| quantity | seed 13 | seed 31 | seed 53 |
|---|---|---|---|
| real min relative delta | 1.714e-7 | 1.721e-7 | 1.801e-7 |
| control (1e-30) max | 0.000e0 | 0.000e0 | 0.000e0 |
| near-null (1e-8) max | 3.105e-11 | 5.162e-11 | 4.430e-11 |
| `moved` for every member | true | true | true |
| class rounding-noise floor | 1.133e-9 | 1.133e-9 | 1.133e-9 |

Binding row: `encoder.layer.4.attention.self.key.bias`, `grad_norm_max = 8.084e-10`,
`grad_norm_mean = 2.252e-10`, `init_norm = 1.519e-2`, own noise floor `9.051e-10`.

**Preliminary verdict, consistent across all three seeds:**

1. **The strict bit-exactness reading of the gradient-free argument does NOT survive production
   scale.** The key bias moves. Its gradient is not zero — `grad_norm_max ≈ 8.1e-10` — and its
   movement *scales with the learning rate* (1e-8 → ~4e-11; 2e-5 → ~1.8e-7, a ~3600× ratio for
   a 2000× learning-rate ratio). A parameter whose delta tracks the learning rate is being
   trained, not held fixed. `dL/db_k = 0` is exact in real arithmetic; in `f32` the softmax
   shift-invariance is only approximate, so a residual gradient at the 1e-10 level is the
   expected numerical consequence, not a bug.
2. **But the class is not vacuous either.** It separates real training from the near-null
   control by ~3300× (1.714e-7 vs 5.162e-11), and its real deltas sit ~150× above its own
   rounding-noise floor. A window `[5.162e-10, 1.714e-8]` exists.
3. **Its margin is the narrowest of the six classes by an order of magnitude** — `eps/noise`
   15.1, against 102 for the next-narrowest (embedding) and up to 2070 for
   `projection_weight`. So if any class is going to fail to support a frozen ε, this is the one.

**Why this is labelled PRELIMINARY and not a verdict.** Three seeds at one shot count is three
samples of one cell. The s64 cells run 64× the optimizer steps, and accumulated residual is
exactly the quantity that could move `real_min` — in either direction. CLAUDE.md rule 6: one
failing input is an anecdote; vary it before naming a cause. The verdict is not final until the
s64 half is measured.

**Consequence if it holds.** This does not automatically force the class to be *gated* — a
gate needs a threshold that separates training from not-training, and the measurements show one
exists. What it does mean is that the *justification* recorded in the contract cannot remain
"this parameter receives no gradient", because at production scale it demonstrably does. That
is a phase-level finding for 05-03 and is flagged as such rather than absorbed.

### What 05-03's replacement justification must be grounded in

So that 05-03 inherits a decision it can act on rather than a number it has to re-interpret:

**The sentence that must go.** Any contract or code comment asserting that
`attention_key_bias` is excluded from gating *because it receives no gradient* — the
`dL/db_k = 0` softmax shift-invariance argument as a statement about the executing code. That
sentence is refuted by this plan's own measurement: `grad_norm_max = 8.084e-10`, and the delta
scales with the learning rate. Keeping it would be recording an argument in place of a
measurement, which is exactly what the contract's own clause N-02 ("an argument is not a
measurement") forbids.

**What is still true and may be kept, restated precisely.** `dL/db_k = 0` holds in *exact*
arithmetic. What the encoder executes is `f32`, in which softmax shift-invariance is only
approximate, so the key bias receives a residual gradient at the ~1e-10 level. The physics
argument survives as an explanation of *why the gradient is tiny*; it does not survive as a
claim that the gradient is *zero*.

**What the replacement justification must be grounded in — the measured window and its margin,
not the physics.** Whichever disposition 05-03 chooses, it must cite measured quantities:

- if the class stays **ungated**, the justification is no longer "it cannot move" but a measured
  statement that its movement is not a usable discriminator at the margin available — and that
  claim must carry the `eps/noise` margin (**15.1** on the s8 half, the narrowest of the six
  classes by an order of magnitude) and the observation that it *does* separate real from
  near-null by ~3300×, so the reader can see what is being given up;
- if the class becomes **gated**, the frozen ε comes from the same window rule as every other
  class — `[10 × worst control, best real / 10]`, upper edge rounded DOWN to two significant
  figures — and must be reported with its noise-floor clearance and binding parameter, exactly
  like the other five.

**The number that decides between them is the s64 margin**, because s64 runs 64× the optimizer
steps and accumulated `f32` residual is precisely what would erode a 15.1× clearance. That
figure is reported per seed in the s64 section, and until all three s64 seeds are in, the
verdict above stays PRELIMINARY.

---

## HALTED — s64 half awaiting a decision

The standing instruction for a partway failure: *do NOT silently restart the whole matrix or
quietly reduce scope to fit; report what completed, what failed, and the measured cost, and halt
for a decision.* That is this section.

### Ledger

| | passes | measured cost | status |
|---|---|---|---|
| `s8e1b16` × {13,31,53} × 3 conditions | 9 | 439.4 s tuning / 542 s wall | **COMPLETE**, `rc=0`, tables above |
| `s64e1b16` × {13,31,53} × 3 conditions | 9 | not run | **NOT RUN** |
| first unchunked attempt | 9 of 18 | ~40 min, tables lost | killed externally; no defect |

### Re-projection from the measured s8 cost

Using the measured 48.8 s/pass at 24 steps → **2.033 s/step**, and the closed-form step count
(`budget = 2·max(pos_cap, neg_cap)`, `neg_cap = 3n²`, so s64 = 1536 steps = 64× s8):

- one s64 pass ≈ 1536 × 2.033 ≈ **3 123 s ≈ 52 min**
- nine s64 passes ≈ **28 105 s ≈ 7.81 h**

That is the whole remaining cost; the s8 half is banked.

### Why it cannot simply be relaunched as-is

A single unattended invocation of ~7.8 h exceeds the background-task lifetime available in this
session — that is what killed the first attempt, and a straight relaunch would die at roughly
the same point. The harness is now chunk-capable and crash-persistent precisely so this is
survivable.

### Options

| # | Option | Shape | Cost | Evidence cost |
|---|---|---|---|---|
| **A′** | Per-seed chunks: `APRENDER_CALIBRATION_CELLS="s64:13"`, then `s64:31`, then `s64:53` | 3 invocations of ~52 min, 3 conditions each | 7.81 h | **None.** Identical passes, conditions and assertions; ε composes from the union of per-cell tables |
| B′ | One unattended relaunch of the whole s64 half | 1 invocation, 7.81 h | 7.81 h | None if it survives; now bounded to the in-flight cell if killed |
| C′ | Human runs the s64 half outside this session and returns the reports | — | 7.81 h, none of it in-session | None |

**Recommendation (non-binding): A′.** It is the same 18-pass matrix already authorized, at the
same cost, partitioned so no invocation must survive longer than ~52 minutes. It trims no cells,
does not touch the pair budget, and does not weaken the D-02 margin argument. Each chunk lands
its own `STATUS: COMPLETE` report, so progress is monotonic and a failure costs at most one seed.

Commands for A′:

```bash
for SEED in 13 31 53; do
  CARGO_INCREMENTAL=0 \
    APRENDER_CALIBRATION_CELLS="s64:${SEED}" \
    SETFIT_PRODUCTION_CALIBRATION_REPORT="/tmp/s64_seed${SEED}_report.txt" \
    cargo test --release -p aprender-train --lib --features setfit production_calibration \
    -- --ignored --nocapture > "/tmp/s64_seed${SEED}.log" 2>&1
  rc=$?
  echo "seed=${SEED} rc=${rc}"
done
```

### CORRECTION to the A′ chunk arithmetic — a chunk is ~2.6 h, not ~52 min

The A′ dispatch was issued on the understanding that one chunk is a ~52-minute unit that "fits
comfortably inside a background task lifetime". **That conflates a pass with a cell, and the
distinction is load-bearing.**

- **52 min is one PASS** (1536 steps × 2.033 s/step).
- **A chunk is one CELL**, and a cell is **three passes** — real (2e-5), control (1e-30) and
  near-null (1e-8). So `APRENDER_CALIBRATION_CELLS="s64:13"` is **3 × 52 min ≈ 2.6 h**.

Measured confirmation rather than arithmetic alone: the `s64:13` chunk launched at
`2026-08-17T06:39:08Z` had **not** completed its first pass at 31 minutes elapsed, which is
consistent with ~52 min/pass and rules out the 52-min-per-chunk reading.

**The cell cannot be subdivided further without losing the assertion that makes the matrix
mean anything.** The separation check `ctrl_max < real_min` compares the real condition against
the control *within the same cell*, so real and control must exist in the same process. Running
conditions as separate invocations would require persisting per-parameter raw data between runs
and re-deriving the comparison outside the harness — a much larger change that moves the
assertion out of the measurement. A cell is therefore genuinely atomic at ~2.6 h.

**Consequence for persistence.** The per-cell flush fires only after all three conditions
complete, so a kill *mid-cell* still loses that cell entirely. The crash-persistence added
earlier bounds the loss to one cell — which at s64 is 2.6 h, not the ~2.5 min it was at s8.

**So each of the three A′ chunks carries the same ~2.6 h exposure** the chunking was meant to
avoid; A′ reduces the exposure from 7.81 h to 2.6 h, not to 52 min. That is still a 3× reduction
and still the best in-session option, but the premise should be corrected rather than inherited.

### Disk exhaustion during this chunk (recorded because it nearly cost the run)

Partway through the `s64:13` chunk the volume filled completely: every shell invocation failed
with `ENOSPC`, including `true`, because the harness must create a task output file before
running anything. Freeing stale task outputs restored 4.6 GiB, and removing the **regenerable**
`target/debug` tree (4.7 GB; all remaining work is release-profile) restored the volume to
41 GiB free. The running job survived — it had not yet needed to write.

This is the recurring ENOSPC that `CLAUDE.md` already warns about on this host. It matters for
the remaining chunks because the harness's FINAL report write panics on failure: an ENOSPC at
the end of a 2.6 h cell would destroy the cell's measurements at the last step. **Check free
space before dispatching each remaining chunk.**

### A′ chunk 1 (`s64:13`) — ATTEMPTED, killed at 55.8 min; 1 of 3 passes done, no tables

The correction above predicted this failure; the run then produced the measurement that proves
it. Recorded in full because it settles the execution-shape question with numbers.

**What completed.** One pass — the REAL condition — quoted verbatim from `/tmp/s64_seed13.log`:

```text
[progress] pass done: seed=13 cell=s64e1b16 condition=real device=Cpu steps=1536 \
  wall_clock=3246.2s regime=minilm-slice-h384-l6-a12-i1536-v30522@1110a243|seeds=13|cells=s64e1b16
```

Three things this establishes:

1. **The s64 step count is exactly the closed form.** `steps=1536`, as
   `budget = 2·max(pos_cap, neg_cap) = 6n² = 24576` over batch 16 predicts. The quadratic
   budget is now confirmed by measurement at *both* ends of the envelope.
2. **The s64 cell label renders as predicted**, and the architecture component is byte-identical
   to every s8 id: `minilm-slice-h384-l6-a12-i1536-v30522@1110a243`. This is the first verbatim
   `s64e1b16` id, and it is what plan 05-03's contract entry copies.
3. **Measured s64 pass cost: 3 246.2 s (54.1 min)**, i.e. 2.113 s/step — 4 % above the
   2.033 s/step measured at s8, consistent with a larger selection touching more embedding rows.

**What failed.** The process was terminated externally during pass 2. Same signature as before
and again not a defect: no panic, no assertion failure, and `/tmp/s64_seed13_timing.txt` has no
`rc=` line. Timeline from the machine — started `06:39:08Z`, last log write `07:34:54Z`, so it
**survived 55.8 minutes** and died roughly 100 seconds after pass 1 completed.

**What was lost.** Everything except that one progress line. No report file was written, because
the per-cell flush fires only after all three conditions complete and the cell is atomic for the
separation assertion. **Zero s64 per-class tables. ~55.8 minutes of compute spent for one timing
number and one regime id.**

### The measured constraint that decides the execution shape

| quantity | measured |
|---|---|
| survivable unattended window | **~55.8 min** |
| one s64 pass | **54.1 min** |
| one s64 cell (3 conditions, atomic for the assertion) | **2.71 h** |
| the s64 half (9 passes) | **8.12 h** |
| the full 18-pass matrix (s8 banked + s64) | **8.24 h** |

**How to project a pass, and the overhead term that must be in it.** Every wall-clock projection
in this file is `steps × per-step cost`, and every one of them was too low, because a pass costs
`cargo test` startup on top of tuning:

| quantity | measured |
|---|---|
| per-step cost, s8 | 2.033 s |
| per-step cost, s64 | 2.113 s (4 % higher — a larger selection touches more embedding rows) |
| per-invocation `cargo test` overhead | **~2 min** (no-op rebuild check, link, test-harness start) |

That overhead is a rounding error on a 54-minute s64 pass and **roughly triples a sub-minute s8
pass** — which is exactly how the s8 re-bank estimate came out at 8 min against 28.7 min actual.
**Any future per-pass projection must add the ~2 min invocation term, not just tuning time.** The
error is invisible when pass ≫ overhead and dominant when it is not, which is why it survived
until a batch of nine short passes exposed it.

**A cell does not fit in the window, and cannot be made to.** One *pass* fits with under two
minutes of headroom — far too tight to rely on. So **no in-session chunking at cell granularity
can ever complete an s64 cell**: A′ is infeasible as specified, and re-dispatching `s64:31` or
`s64:53` unchanged would burn ~56 min each and produce nothing but another timing line. That is
why this halts here instead of continuing to the next seed.

### Options, revised against the measurement

| # | Option | Viability | Evidence cost |
|---|---|---|---|
| **C′** | A human runs the s64 half outside this session (no task-lifetime ceiling) and returns the three `STATUS: COMPLETE` reports | **Works today**, no code change | **None** |
| **D′** | Add per-CONDITION persistence: each pass writes its own per-parameter table to disk; a final cheap invocation loads the three and performs the separation assertion over the persisted numbers | Makes a chunk one pass (54.1 min) and, crucially, **retryable** — a kill costs one pass, not a cell | **None.** The assertion is over recorded numbers either way; performing it across persisted tables is exactly as sound as in-process, and is the same mechanism any off-host run would need to transport results back |
| A′ as dispatched | Per-seed cell chunks | **Refuted by measurement** — 2.71 h against a ~55.8 min window | — |
| B′ | One 8.12 h relaunch | Refuted a fortiori | — |

**Recommendation: D′ then C′-if-preferred.** D′ is a contained harness change with no effect on
what is measured, it makes progress monotonic and retryable instead of all-or-nothing, and it
also produces exactly the artifact an out-of-session or off-host run would need. It is not a
scope reduction: the same 18 passes, the same three conditions, the same window rule, the same
assertion.

### D′ implemented, and its precondition PROVEN before any s64 pass ran

D′ was approved on condition that cross-process determinism be proven first, at s8, before
spending anything at s64 — because D′ moves `ctrl_max < real_min` from comparing values computed
in ONE process to comparing values persisted from THREE, and the in-process proof does not
transfer to the new scope (CLAUDE.md verification rule 4).

**The mechanism.** Three new modes on the existing harness, no change to what is measured:

| env var | effect |
|---|---|
| `APRENDER_CALIBRATION_PASS="s64:13:real"` | run ONE cell under ONE condition, persist it, stop |
| `APRENDER_CALIBRATION_COMBINE="s64:13,…"` | no training: load each cell's three conditions and run the analysis + separation assertion |
| `APRENDER_CALIBRATION_STORE=<dir>` | where passes are persisted |

**The assertion is not re-implemented for the combine path — it is the same lines.** The only
thing the new mode changes is where the three `ProductionPass` values come from:

```rust
let obtain = |cell, condition| if load_from_store { load_pass(…) } else { production_pass(…) };
```

Everything downstream — the regime-label assertion, the per-class rows, `ctrl_max < real_min`,
the epsilon accumulators — executes unchanged and cannot tell which door the values came
through. That is what makes "as sound" structural rather than asserted.

**Persistence is split by determinism, deliberately.** `<stem>.evidence.json` is exactly
`to_canonical_bytes()` — only what the assertion consumes. `<stem>.meta.json` holds
`elapsed_secs` and the probed device — only what the report prints. Had timing been folded into
the evidence file, two identical runs would serialize differently and the bit-identity check
below would have been unsatisfiable, leaving nothing to check. Loading also asserts the bytes
still hash to their recorded digest AND that the parsed table re-serializes to those same bytes,
so a field silently dropped by deserialization cannot sail through.

**THE PROOF — bit-identical across two fresh processes.** Confirmed twice, once by the in-tree
test `cross_process_determinism_of_persisted_evidence` (re-executes the test binary via
`current_exe`, one child per run, separate stores) and once independently at the shell level so
the test could not self-certify:

```text
5838b3d290e77751db25799ac72efcb9bdebc9d619cde2ff1b0981a91f8357dc  /tmp/dprove-a/s8e1b16-seed13-real.evidence.json
5838b3d290e77751db25799ac72efcb9bdebc9d619cde2ff1b0981a91f8357dc  /tmp/dprove-b/s8e1b16-seed13-real.evidence.json
BIT-IDENTICAL: yes        48555 bytes each        steps=24        device=Cpu
elapsed_secs: 64.53 vs 52.18   ← differs by design; timing is metadata, never evidence
```

The two processes disagree by 12.35 s of wall clock and by **zero bytes** of measurement. That
is the whole claim D′ rests on, and it is now measured rather than argued. **D′ is sound on this
host**, so the separation assertion over persisted tables is the assertion the in-process path
makes.

**A false-green caught while proving it, worth recording.** The first attempt passed
`--exact production_calibration_matrix` to the child. `--exact` matches the FULL test path, so
the filter matched nothing, the child ran **zero tests — and exited 0**. Only the subsequent
missing-file panic exposed it. A proof harness whose child can silently run nothing is worse
than no proof, so the child's own count is now asserted (`stdout` must contain `1 passed`) and
the path is derived from `module_path!()` rather than written out, so a module rename cannot
quietly reintroduce the mismatch.

**Disk preflight.** `MIN_FREE_GIB = 10`, checked before every pass, failing closed below the
floor and open (with a warning) if free space cannot be measured — an unparseable `df` is a
reason to warn, not to block an authorized run. This exists because the final report write
panics on failure: exhausting the disk at the end of a ~54 min pass destroys it at its last
step.

### D′ pass 1 of 9 BANKED — `s64:13:real`

The first s64 pass to survive and be persisted. It is committed as a reviewable artifact rather
than only summarised in prose:

```text
.planning/phases/05-benchmark-and-claims-gate/calibration-store/
  s64e1b16-seed13-real.evidence.json   67 697 bytes
  s64e1b16-seed13-real.meta.json
```

```text
[progress] pass done: seed=13 cell=s64e1b16 condition=real device=Cpu steps=1536 \
  wall_clock=3327.1s regime=minilm-slice-h384-l6-a12-i1536-v30522@1110a243|seeds=13|cells=s64e1b16
[persist] s64e1b16-seed13-real evidence_sha256=23e8b60da8773216d36f15b6280c0858ffc6faf06583f2500cc5ef7122700665
STATUS: PARTIAL — one persisted pass … No separation assertion has run
```

`shasum -a 256` on the committed file reproduces `23e8b60d…0665` independently of the harness
that wrote it. `steps=1536` again, and the regime id is **byte-identical** to the one the lost
chunk rendered — the same pass, reproduced, and this time kept.

**The banner is `STATUS: PARTIAL`, correctly.** One condition cannot support
`ctrl_max < real_min`; nothing is claimed here beyond one persisted table.

**A finding that matters for the remaining eight passes: this pass took 3 327.1 s = 55.45 min,
against a measured survivable window of ~55.8 min.** It fit with about **20 seconds of
headroom** — and it ran 2.5 % slower than the lost chunk's 3 246.2 s for exactly the same 1 536
steps, so the variance is real and the margin is inside it. D′ is therefore the right shape for
reasons beyond convenience: at this margin some passes will lose the race, and the only thing
that makes that acceptable is that a lost pass now costs **one pass, retryable**, instead of a
2.71 h cell. Expect retries and treat them as normal operation, not as failures.

### Recommendation: re-bank the nine s8 passes into the store (~8 min)

The s8 half was measured before D′ existed, so its numbers live only as text in this file while
the store holds s64 tables. That leaves the cross-cell epsilon basis — worst control and best
real *across all measured cells* — to be assembled by hand from two different media.

Re-running the nine s8 passes under `APRENDER_CALIBRATION_PASS` costs about **8 minutes total**
(s8 measured 49–65 s per pass), after which `APRENDER_CALIBRATION_COMBINE="s8:13,s8:31,s8:53,
s64:13,s64:31,s64:53"` derives the whole basis mechanically from the persisted tables, with the
separation assertion running over all six cells in one place. That removes a transcription step
from the number 05-03 freezes ε from.

Not done here, because the standing instruction is one pass per dispatch and this is nine — but
the instruction's purpose (a 54-min pass against a ~55.8-min window) does not bind at s8, where
a pass is under a minute. Flagged for the coordinator to dispatch rather than assumed.

### D′ END-TO-END VALIDATED at s8 — `bank → combine → assert` reproduces the in-process numbers EXACTLY

Bit-identity of a persisted pass was proven earlier, and one s64 pass was banked, but the full
D′ pipeline had never been run: nothing had yet gone `bank → combine → separation assertion`
over a complete cell. That was validated at s8, where it is cheap, rather than discovered to be
broken after eight hours of s64 passes.

**Nine s8 passes re-banked**, each its own process, all `rc=0` and all persisting exactly one
pass (`s8:{13,31,53}:{real,control,near-null}`). **Correction to my own estimate:** I projected
~8 minutes; it took **28.7 minutes** (09:11:26Z → 09:40:10Z). The per-pass tuning time was right
(49.5–64.8 s) but I ignored per-invocation `cargo test` overhead, which roughly triples a
sub-minute pass. Cheap either way, and the estimate is corrected rather than quietly forgotten.

**`APRENDER_CALIBRATION_COMBINE="s8:13,s8:31,s8:53"` then ran with no training and `rc=0`**, so
`ctrl_max < real_min` was asserted for all six classes in all three cells over the persisted
tables.

**The comparison that licenses the remaining eight s64 passes.** Combined-from-store output
versus the in-process numbers recorded earlier in this file:

| surface | result |
|---|---|
| per-cell table, 18 rows × 9 columns | **identical in every cell** |
| cross-cell ε basis, 6 classes × 9 columns | **identical in every cell** |
| `attention_key_bias` `eps/noise` | **1.51e1 both** — the 15.1 margin, unchanged |
| every `ctrl_max` | `0.000e0` both |
| `EMBEDDING DELTA MIN` | `1.813e-3` both |
| binding rows (all six classes) | **identical**, same parameter names and same figures |
| `attention_key_bias` binding row | `encoder.layer.4.attention.self.key.bias`, `delta_norm=1.714e-7`, `init_norm=1.519e-2`, `grad_norm_max=8.084e-10`, `grad_norm_mean=2.252e-10`, own noise floor `9.051e-10` — **all identical** |
| wall clock | 501.4 s vs 439.4 s — **differs, correctly**; timing is metadata, never evidence |

Not one measured digit moved. The only thing that differs is the one quantity that *must*
differ, and it differs because it is excluded from the evidence file by design. **The combine
path is the equivalence it was argued to be — now measured, not asserted.** The remaining eight
s64 passes are licensed.

**A dishonest header caught and fixed while validating.** The report printed
`CROSS-CELL EPSILON BASIS (plan 05-03 freezes from these)` regardless of how many cells were
measured — so a three-cell s8-only run produced a table inviting 05-03 to freeze ε from half the
matrix. Since the window rule takes the worst control and best real across *all* measured cells,
an unmeasured cell can still move `best_real` and shrink every window. The header now states
coverage and refuses the invitation:

```text
CROSS-CELL EPSILON BASIS — PROVISIONAL, NOT THE FROZEN EPSILON.
Derived from 3 of 6 boundary cells. MISSING: s64:13, s64:31, s64:53.
… Plan 05-03 must NOT freeze epsilon from this table.
```

It prints the unqualified "freezes from these" heading only when all six cells are present. The
ε values were byte-identical before and after this change, confirming the fix touches only the
banner.

**The store now holds 10 of 18 passes** — the complete s8 half plus `s64:13:real` — all committed
as reviewable artifacts, so the ε basis lives in ONE medium and 05-03 derives it mechanically
rather than transcribing from prose.

### D′ pass 2 of 9 BANKED — `s64:13:control`

```text
[progress] pass done: seed=13 cell=s64e1b16 condition=control device=Cpu steps=1536 \
  wall_clock=3348.9s regime=minilm-slice-h384-l6-a12-i1536-v30522@1110a243|seeds=13|cells=s64e1b16
[persist] s64e1b16-seed13-control evidence_sha256=9c969aa9f10be6579f607e72744883bb4fa1e7b7de2e59703fdd644a4e59c704
STATUS: PARTIAL
```

`shasum -a 256` on the committed file reproduces `9c969aa9…c704` independently. `steps=1536`,
`device=Cpu`, regime byte-identical to the real pass of the same cell. **11 of 18 passes banked.**

**The survivable window is looser than measured, and the earlier figure should be read as a floor
rather than a limit.** This pass ran 3 348.9 s = **55.82 min** of tuning inside a task that lived
09:49:10Z → 10:46:40Z = **57.5 min**, and completed. The earlier 55.8 min figure was a *death*
observation, not a ceiling — the process that died did so ~100 s into its second pass, so all it
established was that the window is *at least* that long. It is at least 57.5 min. Still an order
of magnitude short of a 2.71 h cell, so nothing about D′ changes; but a single s64 pass has more
headroom than the ~20 s I reported last time, and I should not have implied a hard boundary from
one death.

**An early finding worth having before the near-null pass: the control leg does NOT erode at
s64.** Read directly off the persisted table:

| quantity | s64 control, 1536 steps |
|---|---|
| rows | 101 |
| max `relative_delta` | **0.0** |
| rows with `moved = true` | **0** |
| max `delta_support_count` | **0** |
| max `delta_norm` | **0.0** |

At 64× the optimizer steps of the s8 half, the 1e-30 control **still writes back bit-identical
weights** — every parameter, no exceptions. The worry that 1 536 steps of accumulation might let
the control drift and narrow the separation window is **refuted for the control leg**: 1e-30
underflows every parameter's ULP however many times it is applied, so `ctrl_max` will be
`0.000e0` at s64 exactly as at s8, and `ctrl_max < real_min` will hold with the widest possible
margin.

**This sharpens where the remaining risk to `attention_key_bias`'s 15.1 margin actually lies.**
The window is `[10 × max(worst_ctrl, worst_nnull), best_real / 10]`. With `worst_ctrl` pinned at
zero, the lower bound is set entirely by the **near-null (1e-8)** leg and the upper bound by
`best_real` — so the two passes that can still move that margin are `s64:13:near-null` and the
real passes of seeds 31 and 53. The control passes, while still required for the assertion, can
no longer shrink anything.

### METHOD NOTE — a number derived from a failure is not a measurement of capacity

This went wrong twice in this plan, in the same shape, and the second time it propagated into the
coordinator's own reasoning before it was caught. Recorded because the fix is a rule, not
vigilance.

**The two instances.** Both concerned "how long can an unattended task live here":

1. Attempt one died and I reported "~40 min of background task life", then used that figure to
   reason about what would fit in a chunk.
2. The `s64:13` chunk died at 55.8 min and I reported a "~55.8 min survivable window", then a
   "~20 s headroom" margin against it — which the coordinator carried forward.

**The logical error, stated precisely.** A task that DIED at time `T` proves the window is **at
least `T`** — it survived that long. It says **nothing about the upper bound** unless the death
is known to have been *caused* by the window. Both times I inverted this and read `T` as a
ceiling. The 57.5 min completion then contradicted the "55.8 min limit" immediately, which is the
tell: a ceiling that a later ordinary run walks straight through was never a ceiling.

**The rule.** Capacity claims come from **completions**, never from deaths:

- a completed task of duration `T` proves capacity `≥ T` — usable as a floor;
- a death at `T` also proves only `≥ T`, and is evidence about the *cause*, not the *capacity*;
- an upper bound requires either a diagnosed kill mechanism or repeated deaths clustered at a
  duration that completions never exceed.

This is the same family as CLAUDE.md verification rule 6 ("one failing input is an anecdote") and
rule 2 ("never label a run by intent — prove the mechanism engaged"): in both cases the missing
step is asking what the observation actually licenses. **Every remaining timing claim in this
plan is stated as a floor from a completion.**

### PHASE-LEVEL FINDING — the 10× window rule COLLAPSES at s64 for five of six classes

`s64:13:near-null` is banked (`evidence_sha256=78d45f99…7e16`, `steps=1536`, `wall_clock=3317.3s`,
digest reproduced independently by `shasum`). It was the most consequential remaining pass because
`worst_ctrl` is pinned at exactly `0.0`, so the near-null leg alone sets the lower bound of every
window. **It moved that bound far more than the s8 half predicted, and the ε window no longer
exists for five of the six classes.**

**`attention_key_bias` first, as the narrowest class.**

| quantity | s8 (24 steps) | s64 (1536 steps) | factor |
|---|---|---|---|
| near-null `max relative_delta` | 3.105e-11 (s13), 5.162e-11 (s31), 4.430e-11 (s53) | **9.278e-9** | **299× vs s13; 180× vs the worst s8 seed** |
| `best_real` | 1.714e-7 | 3.683e-7 | 2.1× |
| `real / near-null` ratio | **5 520** | **39.7** | the window rule needs **> 100** |

**The margin does not narrow — it ceases to be defined.** The window is
`[10 × max(worst_ctrl, worst_nnull), best_real / 10]`, so with `worst_nnull = 9.278e-9`:

```text
lower = 9.278e-8      upper = 1.714e-8      lower > upper by 5.4x   →  EMPTY
```

There is no interval for ε to sit in. **The previously reported `eps/noise = 15.1` must not be
carried forward as a margin**: the report still prints `1.51e1`, but that column is
`upper / noise_floor` and is computed whether or not `lower < upper`. When `supports_margin` is
`false` the "ε" it divides by the noise floor is not a legal ε, so the number is meaningless
rather than merely smaller. That is a reporting defect worth fixing before 05-03 reads this table.

**It is not confined to `attention_key_bias`, and not an artifact of cross-cell mixing.** From
the shipped `COMBINE` path over four cells (`supports_margin` column):

| class | worst_nnull (s64) | best_real | 10× lower | upper | window |
|---|---|---|---|---|---|
| embedding | 1.553e-4 | 1.813e-3 | 1.553e-3 | 1.813e-4 | **no** |
| layer_norm_weight | 4.802e-7 | 1.891e-4 | 4.802e-6 | 1.891e-5 | yes |
| layer_norm_bias | 9.844e-5 | 7.112e-4 | 9.844e-4 | 7.112e-5 | **no** |
| projection_weight | 1.051e-4 | 1.231e-3 | 1.051e-3 | 1.231e-4 | **no** |
| projection_bias | 1.101e-4 | 3.447e-4 | 1.101e-3 | 3.447e-5 | **no** |
| attention_key_bias | 9.278e-9 | 1.714e-7 | 9.278e-8 | 1.714e-8 | **no** |

Cross-cell mixing makes it worse (the smallest `best_real` comes from s8's 24 steps while the
largest `worst_nnull` comes from s64's 1536), but **the collapse is real within s64 alone**:
`3.683e-7 / 9.278e-9 = 39.7 < 100`. Only `layer_norm_weight` survives, and it is also the one
class whose near-null does not move every row (`nnull_moved = false`). Near-null growth by class
is 56–72× everywhere except `attention_key_bias` at **299×** — it degraded ~4–5× faster than the
rest, which is consistent with it having been the class closest to the noise floor to begin with.

**Likely mechanism, stated as a hypothesis and NOT as a measured result.** Growth of 299× across a
64× increase in steps is superlinear, so this is not a simple longer walk. The plausible reading is
that at 24 steps most 1e-8 updates fall below the parameters' ULP and vanish, so the s8 near-null
figure reflects an *underflow-dominated* regime; at 1536 steps, with warmup complete and Adam
normalising each step to roughly the learning rate, far more updates survive rounding. Emerging
from an underflow-dominated regime produces exactly this superlinear jump. **This is untested.**
It would be tested by measuring near-null at an intermediate step count and checking whether the
delta grows linearly once out of the underflow regime — not by assuming it.

**What this does and does not mean.**

- It does **not** invalidate the separation assertion. `ctrl_max = 0.000e0 < real_min` holds for
  all six classes in the s64 cell; `COMBINE="s64:13"` exited `rc=0`. Real training is still
  cleanly distinguishable from no training.
- It does mean **ε cannot be frozen by the current 10×/10× window rule**, because for five of six
  classes there is no window at the measured boundary. Either the rule's safety factors have to be
  revisited on the evidence, or the near-null condition's role in setting the lower bound does —
  and that is a decision for 05-03 with this table in hand, not something to be resolved by
  picking a number here.
- **`rc=0` on a run whose ε basis is empty for five classes is the clearest example yet of why an
  exit code cannot stand in for a result.** The separation assertion is the only thing asserted;
  `supports_margin` is *reported*, not enforced. Nothing in the harness fails when the window
  vanishes.

Seeds 31 and 53 remain. Their **real** passes can raise `best_real` and their **near-null** passes
can raise `worst_nnull` further; on the s8 evidence the seeds agreed closely, so the direction is
unlikely to reverse, but that is an expectation and not a measurement.

### PRE-REGISTERED CHECK — the architecture component across cell labels: **PASSES**

Answered by a genuinely cross-half combine — `COMBINE="s8:13,s8:31,s8:53,s64:13"`, twelve passes
across **both** cell labels in one invocation, so the harness's assertion actually evaluated s64
ids against an architecture component taken from an s8 id.

```text
minilm-slice-h384-l6-a12-i1536-v30522@1110a243|seeds=13|cells=s8e1b16     (x3)
minilm-slice-h384-l6-a12-i1536-v30522@1110a243|seeds=31|cells=s8e1b16     (x3)
minilm-slice-h384-l6-a12-i1536-v30522@1110a243|seeds=53|cells=s8e1b16     (x3)
minilm-slice-h384-l6-a12-i1536-v30522@1110a243|seeds=13|cells=s64e1b16    (x3)
```

The architecture component `minilm-slice-h384-l6-a12-i1536-v30522@1110a243` is **byte-identical
across `cells=s8e1b16` and `cells=s64e1b16`**, and this is stated as a result: `architecture` was
taken from the first id (an s8 pass) and `id.starts_with(&architecture)` was then evaluated against
all twelve, including every s64 id. `PASSES RUN: 12`, `rc=0`.

**Both halves measured the same production encoder** — the full 22M-parameter MiniLM
(`h384-l6-a12-i1536-v30522`), not the 97-token fixture slice. The s8 and s64 relative deltas are
therefore comparable, which is what makes the finding above a real result about step count rather
than an artifact of two different models. Still pending: the same check with `s64:31` and `s64:53`
included, which will follow automatically as those cells land.

### PRE-REGISTERED CHECK — the architecture component across cell labels (superseded, see above)

Recorded before the evidence exists, so it is reported as a result rather than assumed to have
passed because a run exited 0.

**What is proven.** Within the s8 half, nine passes rendered nine regime ids and the harness
asserted every one starts with the same architecture component
`minilm-slice-h384-l6-a12-i1536-v30522@1110a243`. That is proof **within** one cell label.

**What is NOT proven.** The assertion has never run across `cells=s8e1b16` and `cells=s64e1b16`
together. Every combine so far named cells from a single half, so the cross-label comparison has
simply never been evaluated.

**Why it is the backbone rather than a detail.** That component is what proves the **production
encoder** was loaded in every cell — `h384-l6-a12-i1536-v30522` is the full 22M-parameter
MiniLM, not the 97-token fixture slice the earlier phases used. If s8 and s64 cells disagreed on
it, the two halves would have measured *different models*, and combining their relative deltas
into one ε window would be meaningless no matter how cleanly the separation assertion passed
within each half.

**The check.** On the first combine spanning both halves, report explicitly whether
`minilm-slice-h384-l6-a12-i1536-v30522@1110a243` is byte-identical across both cell labels — as a
stated result, quoting the ids. **If it does not hold, that is a phase-level finding and it stops
05-03.** Deferring to the run's exit status is not acceptable here: the assertion is
`id.starts_with(&architecture)` where `architecture` is taken from the FIRST regime seen, so a
green run proves agreement only among the ids that run actually loaded.

## PLAN CLOSEOUT — what 05-03 inherits

Measurement stopped by human decision at 12 of 18 passes, **not** by failure. The remaining six
passes cannot change the verdict: `lower = 10 × worst_nnull` can only **rise** as more near-null
passes land, and `upper = best_real / 10` can only **fall** as more real passes land, so every
window can only get narrower. Spending a further ~5.5 h to confirm a determined outcome was
rejected. All 12 passes are committed as reviewable artifacts, so `s64:31` and `s64:53` can be
measured later without redoing anything.

### FINDING 1 — the 10×/10× window rule does not survive production step counts

**Five of six classes have no legal ε** at the measured boundary. From the shipped `COMBINE` path:

| class | lower `10 × worst_nnull` | upper `best_real / 10` | window | lower exceeds upper by |
|---|---|---|---|---|
| `embedding` | 1.553e-3 | 1.813e-4 | **EMPTY** | 8.56× |
| `layer_norm_weight` | 4.802e-6 | 1.891e-5 | EXISTS | — |
| `layer_norm_bias` | 9.844e-4 | 7.112e-5 | **EMPTY** | 13.84× |
| `projection_weight` | 1.051e-3 | 1.231e-4 | **EMPTY** | 8.54× |
| `projection_bias` | 1.101e-3 | 3.447e-5 | **EMPTY** | 31.94× |
| `attention_key_bias` | 9.278e-8 | 1.714e-8 | **EMPTY** | 5.41× |

Only `layer_norm_weight` retains a window — also the one class whose near-null does not move every
row. ε cannot be frozen for the other five by this rule. **05-03 must either revisit the rule's
safety factors against this evidence or reconsider the near-null condition's role in setting the
lower bound.** It must not pick a number to make the table close.

### FINDING 2 — what did NOT fail: separation is intact

**`ctrl_max = 0.000e0 < real_min` for all six classes, in every measured cell.** The 1e-30 control
writes back bit-identical weights even at 1536 steps (101 rows, zero moved, zero support). Real
training remains cleanly distinguishable from no training, with the widest possible margin.

**05-03 must not over-read Finding 1 as "calibration is impossible."** What failed is *freezing an
ε by the 10×/10× rule*. What works is the underlying discrimination the gate depends on.

### FINDING 3 — `COMBINE` exits `rc=0` with an empty ε basis (fail-closed hole)

A real defect in the surface F-10 exists to protect, recorded here rather than fixed because making
it fail-closed changes gate semantics and belongs to 05-03's replan.

- The separation predicate is **asserted** (`assert!(control_max < real_min)`).
- `supports_margin` / `window` is only **reported**. Nothing fails when a window vanishes.
- Consequently **`rc=0` is currently reachable with five of six ε windows empty** — verified: the
  four-cell combine exited `rc=0` while printing `EMPTY` for five classes.

A green run is therefore not evidence that an ε basis exists. **05-03 must make an empty window a
hard failure** in whatever surface freezes ε, or the gate can be satisfied by a run that proves
nothing about ε.

### FINDING 4 — `attention_key_bias`'s original justification is REFUTED

The contract's stated reason for leaving this class ungated — *it receives no gradient, since
`dL/db_k = 0` by softmax shift-invariance* — is refuted by measurement:

- `grad_norm_max = 8.084e-10` (s8), `7.298e-10` (s64) — non-zero, on the binding row
  `encoder.layer.4.attention.self.key.bias` / `encoder.layer.1...` respectively;
- the delta **scales with the learning rate** (real 2e-5 → 1.714e-7; near-null 1e-8 → 3.105e-11),
  which a parameter receiving no gradient cannot do.

`dL/db_k = 0` holds in **exact** arithmetic; the encoder executes `f32`, where shift-invariance is
only approximate. The physics argument survives as an explanation of why the gradient is *tiny*, not
as a claim that it is *zero*. **This sentence cannot be carried into 05-03 as written.** Note the
class now also has **no window**, so "ungated because it cannot move" and "gated at a frozen ε" are
*both* unavailable on this evidence — which is precisely the decision 05-03 has to make explicitly.

### FINDING 5 — superlinear near-null growth (HYPOTHESIS, not tested)

Near-null `max relative_delta` grew **299×** for `attention_key_bias` across a **64×** increase in
steps (3.105e-11 → 9.278e-9); other classes grew 56–72×. Superlinear growth is not a longer walk.

**Hypothesis:** at 24 steps most 1e-8 updates fall below the parameters' ULP and vanish, so the s8
figure is *underflow-dominated*; at 1536 steps, warmup complete and Adam normalising each step to
roughly the learning rate, far more updates survive rounding. Emerging from an underflow-dominated
regime produces exactly this jump.

**This was NOT tested.** It would be tested by measuring near-null at an intermediate step count and
checking whether growth becomes linear once out of the underflow regime. Recorded as a hypothesis so
05-03 does not inherit it as a mechanism.

### FINDING 6 — the cross-label check PASSED

`COMBINE="s8:13,s8:31,s8:53,s64:13"` — 12 passes across **both** cell labels in one invocation, so
`architecture` was taken from an s8 id and `starts_with` evaluated against every s64 id.
`minilm-slice-h384-l6-a12-i1536-v30522@1110a243` is **byte-identical across `cells=s8e1b16` and
`cells=s64e1b16`**.

Both halves measured the same 22M-parameter production encoder, not the 97-token fixture slice.
**This is what licenses Finding 1 as a result about step count rather than an artifact of two
different models.**

### Reporting defects fixed in this plan

1. **`eps/noise` is no longer printed for a class with an empty window** — it reads `n/a`. That
   column is `10x_upper / noise_floor`, and `10x_upper` is only a legal ε while `lower < upper`. The
   old behaviour produced a plausible-looking margin for a class with no ε at all; this plan tracked
   `1.51e1` for `attention_key_bias` across several reports before noticing it was meaningless.
   Suppressed rather than adjusted, because there is no correct value to print.
2. **Per-class window status is now explicit** (`EXISTS` / `EMPTY`) plus a block naming every empty
   window, the factor by which `lower` exceeds `upper`, and the instruction not to read a suppressed
   margin as small rather than absent.
3. The **PROVISIONAL** ε-basis banner (coverage + named missing cells) is now permanently accurate
   for this plan rather than temporarily so.

Both the production and fixture ε tables were fixed, so the trap does not survive in a sibling
report.

### Task status, honestly

| task | status | why |
|---|---|---|
| **Task 1** — frozen hyperparameters, rendered regime id, compute projection | **COMPLETE** | recorded above, verbatim |
| **Task 2** — measure the 18-pass boundary matrix | **PARTIAL, 12 of 18** | complete s8 half (9) + complete `s64:13` cell (3); stopped by human decision, verdict already determined |
| **Task 3** — derive per-class ε windows and the proposed regime entry string | **NOT DONE, and not completable as specified** | the windows do not exist for 5 of 6 classes, so there is no ε to freeze. Emitting a regime entry would mean publishing a number nobody should use |

**Task 3's one deliverable that IS met:** the verbatim regime ids for both cell labels, proven
byte-identical in their architecture component (Finding 6). The frozen ε is not delivered, and
should not be.

### What Task 3 still owes once the s64 half lands

Frozen ε per class (window upper edge rounded DOWN to two significant figures) with noise-floor
clearance and binding parameter; the production `embedding_delta_floor`; the proposed regime
entry with its architecture component byte-copied from a measured id; the MEASURED-vs-COVERED
table over all 40 cells; the two-cell prospective validation (`s16` seed 41, `s32` seed 29); and
the **final** `attention_key_bias` verdict.

