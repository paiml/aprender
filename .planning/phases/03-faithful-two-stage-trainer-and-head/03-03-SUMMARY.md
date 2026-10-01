---
phase: 03-faithful-two-stage-trainer-and-head
plan: 03
subsystem: training
tags: [setfit, typestate, philox, feature-matrix, serde, lr-scheduler, determinism, cargo-features]

requires:
  - phase: 01-differentiable-minilm-conformance
    provides: "SetFitMiniLm behind the `setfit` feature, FreezeGroup + apply_freeze, MAX_SEQUENCE_LENGTH = 256"
  - phase: 02-deterministic-pair-and-data-protocol
    provides: "PreparedDataset<Canonical>, Split<Train>, Selection, PairConfig/resolve_budget, the Philox construction shape and its frozen constants"
provides:
  - "The `setfit` feature in aprender-train, dependency-closed and composing with every CPU-buildable feature"
  - "Sealed LifecycleState trait + four state markers; SetFitRun<Prepared> with private fields and prepare() as the only door"
  - "SetFitTrainConfig (requested/serializable) and ResolvedSetFitConfig (Serialize-only) — 12 fail-closed knobs behind one validating constructor serde cannot bypass"
  - "WarmupLinearDecayLR + warmup_steps_from_ratio (ceil, pinned to HF) — the reference schedule the repo did not have"
  - "reduce.rs: fixed-order f64 reductions, the trainer's only reduction door"
  - "epoch.rs: epoch_pair_order under the trainer's own frozen tag apr-setfit-train-v1"
  - "make setfit-feature-matrix extended with three aprender-train legs plus a two-sided dependency-closure negative"
affects: [03-05, 03-06, 03-07, 03-08, 03-09, 03-10]

tech-stack:
  added:
    - "aprender-contrastive-data (optional, under `setfit`) — the single new dependency edge D-05 introduces"
    - "aprender-rand / trueno_rand (optional, under `setfit`) — Philox for the epoch shuffle"
  patterns:
    - "Sealed typestate with an Evidence ASSOCIATED TYPE, so a state with no evidence has no evidence field"
    - "Requested-vs-resolved config split; the resolved form is Serialize-only"
    - "Private wire struct + #[serde(try_from)] so deserialization routes through the one validating constructor"
    - "Two-sided feature-matrix guards: an absence check paired with a presence check so the absence cannot pass vacuously"
    - "Golden derivation whose independent reference first reproduces a PRIOR phase's frozen constants as a control"

key-files:
  created:
    - crates/aprender-train/src/train/setfit/mod.rs
    - crates/aprender-train/src/train/setfit/config.rs
    - crates/aprender-train/src/train/setfit/reduce.rs
    - crates/aprender-train/src/train/setfit/epoch.rs
    - crates/aprender-train/src/optim/scheduler/warmup_linear_decay.rs
    - .planning/phases/03-faithful-two-stage-trainer-and-head/deferred-items.md
  modified:
    - crates/aprender-train/Cargo.toml
    - crates/aprender-train/src/train/mod.rs
    - crates/aprender-train/src/optim/mod.rs
    - crates/aprender-train/src/optim/scheduler/mod.rs
    - Makefile

key-decisions:
  - "03-03: the plan's headline leg `cargo check -p aprender-train --no-default-features --features setfit` is UNSATISFIABLE and the cause is pre-existing — `src/monitor/mod.rs:45` declares `pub mod tui;` unconditionally while `presentar-terminal` is gated behind the `tui` feature. Control-measured: the same command WITHOUT setfit produces a BYTE-IDENTICAL 8-error stream, so setfit contributes zero. Wired as a two-sided DIFF leg instead of a plain green check, so Phase 3 neither inherits the red nor hides it (D-ITEM-05)"
  - "03-03: the D-05 closure claim is discharged by a TWO-SIDED cargo tree check — a DEFAULT aprender-train build must contain no aprender-contrastive-data / aprender-rand / tokenizers node, AND a --features setfit build must contain all three. The presence half is what stops the absence half passing vacuously if cargo tree ever stopped resolving those packages"
  - "03-03: SetFitRun holds the PreparedDataset as well as the Selection, because SelectedExample carries no text; without it 03-05 and 03-07 have no string to encode"
  - "03-03: prepare() compares each selected row's EXACT HASH against the dataset's, not merely its id — an id that resolves to a row whose bytes changed is a different example wearing the same name, and it would silently invalidate every provenance claim the run goes on to make (SelectionRowContentMismatch, a Rule-2 addition)"
  - "03-03: pair-budget validation is DEFERRED to prepare() in full rather than partially duplicated at construction. The three request rungs (zero cap, zero budget, over-cap) are knowable without class sizes, but re-checking them here would be a second implementation of Phase 2's ladder that could drift; one call to resolve_budget stays the only answer"
  - "03-03: the device knob's grammar is validated at construction by CALLING resolve_device and accepting both Ok and CudaNotAvailable — grammar is host-independent, availability is a prepare()-time fact. This satisfies `contains resolve_device, contains no independent device parsing` without splitting device.rs, which is outside this plan's file set"
  - "03-03: a resolved non-CPU device is rejected at prepare(), including `auto` on a CUDA host. A phase that only supports CPU should make the operator say so rather than quietly disagree with the machine"
  - "03-03: reference_defaults uses SklearnEquivalentC { c: 1.0 } rather than Lambda(0.0). SetFit's reference head is sklearn LogisticRegression at its default C; Lambda(0.0) would mean NO regularization, which is not the reference"
  - "03-03: PairStrategy and SingletonPolicy are #[non_exhaustive], so they are COMPARED against the v1 values rather than matched with a wildcard. A wildcard arm would accept a future variant the wire form cannot represent and silently serialize it as the default"
  - "03-03: warmup_steps_from_ratio CLAMPS to total_steps, which discharges the cross-field `warmup_steps > total_steps` rule structurally. A runtime comparison in prepare() would have needed the pair budget, epochs and batch size — none of which the config knows — and could be forgotten by a caller"
  - "03-03: the epoch goldens were derived by an independent Python Philox that FIRST reproduced all eight of Phase 2's frozen constants under the old tag. A reference implementation that reproduces eight independently frozen values is not guessing at the ninth"

patterns-established:
  - "Vacuity guards on comparison gates: assert rc and diagnostics AGREE before comparing two streams, so a guard cannot pass because both sides were empty"
  - "String-surgery test helpers that assert the substitution applied — a `.replace()` that matched nothing turns a rejection test into a test of the valid payload"
  - "Mutation-before-trust: every new Makefile guard had its failure mode induced, observed and reverted before it was wired"

requirements-completed: [TRN-01, TRN-02]

duration: ~4h15m
completed: 2026-08-09
---

# Phase 3 Plan 03: Trainer Home, Lifecycle and Determinism Primitives Summary

**The `setfit` feature opened in aprender-train — dependency-closed across a 37-module crate — carrying a sealed four-state typestate whose only door is `prepare()`, a 12-knob fail-closed configuration that serde cannot route around, and the three determinism primitives (HF-pinned warmup-linear-decay schedule, fixed-order f64 reductions, Philox epoch shuffle under the trainer's own tag).**

## Performance

- **Duration:** ~4h15m (roughly 45 min of it lost to two host ENOSPC stalls)
- **Tasks:** 3 of 3
- **Files created:** 6 · **Files modified:** 5

## Accomplishments

- `setfit = ["aprender/setfit", "dep:aprender-contrastive-data", "dep:aprender-rand"]` composes with every CPU-buildable feature of a crate that also carries GPU/LoRA/distill/server, and its closure is now gated two-sidedly.
- The lifecycle is four TYPES, not a runtime field: `SetFitRun<Prepared>` has six private fields, `LifecycleState` is sealed, and `Evidence` is an associated type so a state without evidence has no evidence field to `expect` on.
- 35 FALSIFY rows covering all twelve knobs plus the deserialization path, all reaching one validating constructor.
- `WarmupLinearDecayLR` — the schedule the repo did not have — with its rounding rule pinned to HF's `math.ceil` rather than the plan's guessed `round`.
- Three deferred/pre-existing defects measured with controls rather than assumed.

## Task Commits

1. **Task 1: setfit feature wiring + sealed typestate skeleton + feature-matrix legs** — `f9e76856c` (feat)
2. **Task 2: SetFitTrainConfig — 12 knobs, fail-closed, validated deserialization** — `47a431ac7` (feat)
3. **Task 3: WarmupLinearDecayLR + fixed-order reductions + epoch-shuffle domain** — `e0116dc7c` (feat)

## Measured Verification

| Command | rc | Note |
|---|---|---|
| `cargo check -p aprender-train --features setfit` | **0** | leg (b) |
| `cargo check -p aprender-train --features setfit,cpu-fallback,gguf,monitor,tui,citl,server,tracing,ruchy-sessions,parquet,hub,viz` | **0** | leg (c), maximal CPU-safe |
| …the same list WITHOUT `setfit` (leg (c) CONTROL) | **0** | control green, so leg (c) wired as-is |
| `cargo check -p aprender-train --no-default-features --features setfit` | **101** | leg (a) — see "Known reds" |
| …WITHOUT `setfit` (leg (a) CONTROL) | **101** | byte-identical diagnostics |
| `make setfit-feature-matrix` (standalone, rc captured directly) | **0** | |
| `cargo test -p aprender-train --lib --features setfit config_` | **0** | 409 passed, 0 failed |
| `cargo test … setfit::config::tests::` (scoped) | **0** | **35** tests (criterion: ≥ 22) |
| `cargo test -p aprender-train --lib --features setfit reduce_` | **0** | 13 passed |
| `cargo test -p aprender-train --lib --features setfit epoch_` | **0** | 42 passed |
| `cargo test -p aprender-train --lib warmup_linear` | **0** | 13 passed |
| `cargo clippy -p aprender-train --lib --features setfit -- -D warnings` | **101** | 0 diagnostics in any file this plan owns — see "Known reds" |

### Source assertions

| Assertion | Result |
|---|---|
| `setfit = [` lists all three entries | yes |
| four markers + sealed `LifecycleState` with `type Evidence` | yes |
| `SetFitRun` has a `dataset: PreparedDataset<Canonical>` field | yes |
| `SetFitRun` FIELD lines containing `pub ` | **0** (see deviation 4) |
| `PhantomData<S>` in the `SetFitRun` range | yes |
| `-p aprender-train` occurrences in the `setfit-feature-matrix` block | **7** (criterion: ≥ 3) |
| `--all-features` in a RECIPE line of that block | **0** (see deviation 5) |
| `SetFitTrainConfig` carries `#[serde(try_from = "SetFitTrainConfigWire")]` | yes |
| `ResolvedSetFitConfig` derives `Serialize`, not `Deserialize` | yes |
| `config.rs` contains `resolve_device` / independent device parsing | 7 / **0** |
| `warmup_steps_from_ratio` body contains `ceil` / `round` | 1 / **0** |
| `apr-setfit-train-v1` in `epoch.rs` | yes (tag is `b"apr-setfit-train-v1\0"`, 20 bytes) |
| `par_iter|rayon` in NON-COMMENT lines of `reduce.rs` | **0** (see deviation 6) |
| `WarmupLinearDecayLR` exported from `scheduler/mod.rs` | yes |

## Known reds — measured, expected, NOT regressions

### 1. `cargo package -p aprender-train --no-verify` (Pitfall 9, as predicted)

```
error: failed to prepare local package for uploading
Caused by:
  no matching package named `aprender-contrastive-data` found
  location searched: crates.io index
  required by package `aprender-train v0.63.0`
```

rc=101. This confirms Phase 2's control-verified finding for a second crate: `--no-verify`
skips the packaged-crate BUILD, not the MANIFEST RESOLUTION that rewrites the path dep into
a registry dep, and resolution is where it breaks. **The publish cascade is now
`aprender-contrastive-data` → `{apr-cli, aprender-train}`.** `/gsd:verify-work` must read a
red `pre-release` Gate 5 as this expected state.

(Note for whoever re-measures: without `--allow-dirty` the command fails EARLIER, on
uncommitted files, with a completely different message. That is not the known-red.)

### 2. The minimal build (D-ITEM-05, new)

`cargo check -p aprender-train --no-default-features` is red at HEAD with 8 errors, all
`presentar_terminal` unlinked under `src/monitor/tui/`, because `src/monitor/mod.rs:45`
declares `pub mod tui;` unconditionally while its dependency is gated behind the `tui`
feature. **Control:** the identical command with `--features setfit` produces a
byte-identical 26-line diagnostic stream — setfit contributes zero. Full write-up and fix
direction in `deferred-items.md`.

### 3. Clippy (D-ITEM-02, Phase 2, re-measured)

`cargo clippy -p aprender-train --lib --features setfit -- -D warnings` exits 101 with
`aprender-compute` (19 errors) and `aprender-present-terminal` (1). Diagnostics citing
`crates/aprender-train/`: **0**. Diagnostics citing `train/setfit/` or
`warmup_linear_decay.rs`: **0**. The stream is identical with and without `--features
setfit`. `aprender-compute` belongs to plan 03-02 in this same wave.

## Guards falsified before being trusted

No new gate was wired on the strength of it being green. Each had its failure mode induced,
observed, and reverted:

| Guard | Induced mutation | Observed |
|---|---|---|
| leg (a) minimal-build diff | a module under `cfg(not(feature = "setfit"))` importing `aprender_contrastive_data` | FAIL, printing the exact extra error as a diff; legs (b)/(c) stayed green, so the leg was reached on its own merit |
| tree closure negative | `aprender-contrastive-data` made non-optional **and** removed from the feature list (the manifest-valid form — the naive mutation is rejected by cargo before the check runs) | `FAIL: a setfit-only dependency leaked into the DEFAULT aprender-train build` |
| the 12-knob table | all validators disabled | 15 of 35 rows RED — exactly the value-rejection rows; the serde shape-rejection rows correctly stayed green |

The leg-(a) guard **also caught two defects in itself** on first run, which is the reason its
vacuity checks exist: it read `$?` after `if ! cargo check`, which is the status of the
NEGATION and therefore always 0 (CLAUDE.md rule 1, in a form the rule's own examples do not
list); and its location pattern `^ *--> ` matched warning locations from unrelated crates.
Both fixed; the corrected leg reports the true `control rc=101, setfit rc=101`.

## Epoch golden derivation

Constants were produced by an independent Python implementation of Philox 4x32-10 written
from Salmon et al. (2011) and the byte-encoding table, **not** by running the Rust and
blessing its output. PROJECT.md permits Python solely as a numerical reference during
verification; nothing Python entered the repository.

**The derivation carries a control.** Before deriving anything under the new tag, that
Python reproduced every constant Phase 2 froze under the OLD tag — three key-lane pairs, the
block at ordinal 7, the 64-bit assembly, and all three bounded draws including the
high-ordinal / non-zero-stream case. All eight matched.

```
tag    = b"apr-setfit-train-v1\0"                 (19 ASCII bytes + NUL = 20)
key    = trunc64_le(SHA256(tag ‖ seed.to_le_bytes() ‖ "epoch-shuffle"))
block  = Philox4x32-10(key, [ordinal_lo, ordinal_hi, epoch, 0])
x      = (block[1] << 32) | block[0]
draw   = (x * n) >> 64                            (multiply-shift, never modulo)
order  = Fisher-Yates DESCENDING over 0..n, one draw per swap, ordinal from 0
```

| Input | Golden |
|---|---|
| `key(42, "epoch-shuffle")` | `[0x01270e6e, 0x0c648cc6]` |
| `epoch_pair_order(42, 0, 8)` | `[2, 3, 0, 7, 1, 5, 6, 4]` |
| `epoch_pair_order(42, 1, 8)` | `[7, 2, 5, 1, 3, 4, 0, 6]` |
| `epoch_pair_order(43, 0, 8)` | `[7, 2, 1, 4, 3, 0, 6, 5]` |

The Rust implementation reproduced all four on its first run.

## The warmup ceil-vs-round divergence

`warmup_steps_from_ratio(11, 0.1)` = **2** under `ceil`, **1** under `round`
(11 × 0.1 = 1.1000000000000001). A second, differently-shaped case is pinned so the first is
not an artifact: `warmup_steps_from_ratio(101, 0.001)` = **1** under `ceil`, **0** under
`round`. And an agreeing case (`100, 0.1` → 10) is pinned too, so the test is not merely
detecting "always ceil". HF's `TrainingArguments.get_warmup_steps` uses `math.ceil`; the
plan's earlier `round` was a guess.

`get_lr()` at step 0 with `warmup_steps > 0` is asserted with `assert_eq!(…, 0.0)` — an exact
equality, not a tolerance — because HF's `LambdaLR` lambda(0) is 0 and the first optimizer
step genuinely runs at lr 0. Documented on the type so the 03-05 loop ordering is not later
"fixed" by a reader who thinks it is an off-by-one.

## Cross-field rules and where they ended up

- **`warmup_steps > total_steps`** — discharged STRUCTURALLY, not by a runtime comparison.
  `warmup_steps_from_ratio` clamps to `total_steps`, and a test sweeps 8 totals × 8 ratios
  (including 1.5 and infinity) asserting `steps <= total`. A `prepare()`-time check would
  have needed the pair budget, epochs and batch size, none of which the config knows, and
  would have been vacuous anyway: with `ratio ∈ [0,1]`, `ceil(ratio · total) ≤ total` always.
- **Pair budget vs selection capacity** — deferred to `prepare()` in full, via one call to
  Phase 2's `resolve_budget`. See deviation 3.
- **Freeze-policy zero-match** — stays with `apply_freeze`, which is the only place that
  knows the encoder's parameter names. The construction-time half is the wire enum's shape
  (a group missing its `layer`, or an unknown group, is a typed error) plus canonicalization
  (sort + dedup, matching `apply_freeze`) so equivalent policies hash identically for 03-08.
  Documented on the constructor that **03-05 must call `apply_freeze` BEFORE the initial
  parameter snapshot and BEFORE the pre-tuning baseline encode**.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] `config.rs` created in Task 1, not Task 2**
- **Found during:** Task 1
- **Issue:** Task 1's `SetFitRun` holds `config: ResolvedSetFitConfig` and its error enum wraps `SetFitConfigError`, but `config.rs` is listed only in Task 2's files. Task 1 could not compile.
- **Fix:** Task 1 created `config.rs` with the type skeleton; Task 2 replaced it with the validated form (its diff is +1065/−84).
- **Committed in:** `f9e76856c`, then `47a431ac7`

**2. [Rule 3 - Blocking] `Cargo.lock` committed with Task 1**
- **Issue:** Adding two optional deps rewrites the lockfile; leaving it uncommitted leaves the tree dirty and the lock inconsistent with the manifest.
- **Fix:** Staged with Task 1 by explicit pathspec. Not in `files_modified`, but a direct consequence of a change that is.

**3. [Rule 1 - Correctness] Pair-budget validation NOT duplicated at construction**
- **Found during:** Task 2
- **Issue:** The knob table marks `pair_config` "delegated" but also expects a rejected form at construction. Three of `resolve_budget`'s rungs (zero cap, zero budget, over-cap) are knowable without class sizes — re-checking them in `config.rs` would have been a second implementation of Phase 2's ladder, which the same plan forbids ("never reimplement capacity logic").
- **Fix:** Deferred wholly to `prepare()`. The rejected forms are proven by two tests that build a config through the validating constructor and confirm `ZeroBudget` and `BudgetExceedsHardCap` arrive from Phase 2 verbatim — delegation demonstrated rather than copied.

**4. [Rule 1 - Plan defect] The private-field acceptance criterion is unsatisfiable as written**
- **Issue:** `awk '/pub struct SetFitRun/,/^}/' … | grep -c 'pub '` returns **1** for ANY `pub struct`, because the awk range's first line is the declaration and `pub struct` contains the substring `pub `. No conforming code can return 0.
- **Fix:** Ran the literal form (returns 1, from the declaration line only) and the field-scoped form `… | tail -n +2 | grep -c 'pub '`, which returns **0**. All six fields are private; `PhantomData<S>` present. The criterion's substance holds; its wording needs the `tail -n +2`.

**5. [Rule 1 - Plan defect] `--all-features` must appear, as a comment**
- **Issue:** The criterion says the literal string must not appear in the Makefile block, but the same task's action text requires explaining why the leg is absent — which requires naming it. The pre-existing aprender-core block above does exactly this (line 337).
- **Fix:** Zero occurrences in any RECIPE line (measured with a tab-anchored pattern); the one occurrence is the comment explaining why the leg cannot exist on a CPU profile.

**6. [Rule 1 - Plan defect] `par_iter|rayon` and `apr-contrastive-v1` counts**
- **Issue:** Same shape. `grep -c 'par_iter|rayon' reduce.rs` returns 3 and `grep -c apr-contrastive-v1 epoch.rs` returns 3 — all in doc comments explaining why those things are absent, plus one assertion MESSAGE in the test that proves `epoch.rs` does NOT derive under Phase 2's tag.
- **Fix:** Measured on NON-COMMENT lines: `par_iter|rayon` = **0**; `apr-contrastive-v1` = **1**, and that one is the negative test's failure message.

**7. [Rule 2 - Missing critical] `SelectionRowContentMismatch` added to `prepare()`**
- **Issue:** The plan asks `prepare()` to assert every selected id resolves to a dataset row. Membership alone does not catch a row whose BYTES changed since selection — the same name, different example — which would silently invalidate every provenance claim the run makes.
- **Fix:** `prepare()` compares the selected row's `exact_hash` against `Split<Train>::exact_hash_of`, with its own typed variant.

**8. [Rule 1 - Bug] Two defects in the leg-(a) Makefile guard, caught by its own vacuity check**
- **Issue:** (a) `if ! cargo check …; then rc=$?; fi` reads the status of the negation, always 0 — so the guard reported `control rc=0` for a build that exited 101. (b) The pattern `^ *--> ` matched warning locations from unrelated crates, so the compared stream was not the error stream.
- **Fix:** `cmd > log 2>&1; rc=$?` on the following line (CLAUDE.md rule 1's prescribed form) and `grep -A1 -E '^error'`. Both recorded in the Makefile comment so the next reader sees the trap.

**9. [Rule 1 - Bug] Two test defects, each caught by an assertion placed for that purpose**
- **Issue:** (a) `falsify_config_deserialize_rejects_negative_encoder_lr` keyed a `.replace()` on the literal `2e-5`, but `serde_json` renders it `0.00002` — the substitution matched nothing and the test was asserting that an unmodified VALID payload failed to deserialize. (b) `reduce_is_a_pure_function_of_the_slice_in_order` asserted `[1.0, 1e30, -1e30]` sums to 1.0; in f64 the ULP of 1e30 is ~1.5e14, so the 1.0 is discarded and the sum is 0.0.
- **Fix:** (a) A `substitute()` helper that asserts the replacement changed the string, applied to all eight deserialization substitutions. (b) The assertion corrected to the measured values (0.0 vs 1.0 for the two orders) and turned into the module's statement of *why* index order must be fixed rather than hoped for.

---

**Total deviations:** 9 auto-fixed (3 blocking, 4 plan-defect measurement corrections, 1 missing-critical, 4 bugs — two entries cover two defects each)
**Impact on plan:** No scope creep. Three deviations are corrections to acceptance criteria that could not be satisfied as literally written; the rest are correctness fixes inside this plan's own files.

## Issues Encountered

**Host ENOSPC, twice — the recurring blocker, and it is worse under per-executor worktrees.**
The volume reached 117 MiB and then 0 bytes free, at which point even a zero-byte tool-output
file could not be created and no command would run. `CARGO_INCREMENTAL=0` was set on every
build as STATE.md recommends and the incremental cache stayed small; the pressure came from
three concurrent worktrees each holding a full `target/`. Recovered by deleting **this
worktree's own** `target/` (regenerable, gitignored, owned solely by this executor — no
shared state, no git operation, no other agent's files). Task 1 was committed immediately on
recovery so the work could not be lost to a worktree teardown. Roughly 45 minutes lost.

**A `git checkout --` aimed at the wrong file.** While reverting a mutation probe, `git
checkout -- crates/aprender-train/src/train/mod.rs` reverted Task 1's module registration
rather than the probe (which lived in an untracked file and was therefore unreachable by
`git checkout`). Caught immediately by reading the file back, and re-applied. Recorded
because it is a live argument for the standing rule against blanket working-tree resets:
targeted or not, `checkout --` on a file with real uncommitted work is a one-keystroke loss.

**RTK hook interference with measurement.** `grep`, `diff`, `wc` and `cargo` are rewritten by
the rtk hook, whose summarized output silently broke several redirections (a `diff` reported
"[ok] Files are identical" for files it had not compared as I expected; `grep -c` returned a
count in a different format). Every load-bearing measurement in this summary was therefore
taken through `rtk proxy`, extending Phase 2's `git status --porcelain` finding to the whole
measurement toolchain. Also re-confirmed: BSD `grep` has no `-P` (D-ITEM-01's root cause) —
it exits 2, and a `|| true` would have made the guard vacuous.

## Known Stubs

None. Every type this plan ships is fully implemented for its declared scope. The three
lifecycle markers `EncoderTuned`, `HeadFitted` and `ArtifactReloadedAndVerified` are
DECLARED without `LifecycleState` impls, which is the plan's interface-first intent, not a
stub: an impl would need an `Evidence` type that does not exist until the transition that
produces it lands (03-05, 03-07, 03-08). A marker with a placeholder evidence type would be
a lie that compiles; a marker without an impl is a contract.

## Threat Flags

None. Every file this plan touches is inside the plan's declared `<threat_model>` surface —
no new network endpoint, no filesystem access (`train/setfit/` opens no file), no schema at a
trust boundary beyond the config wire form, which T-3-36 already covers. Mitigations applied:
T-3-08 (per-knob fail-closed table), T-3-09 (sealed trait + private fields + PhantomData),
T-3-10 (own frozen tag, independent golden), T-3-11 (both scheduler zero-division branches
tested), T-3-36 (`#[serde(try_from)]` + six invalid-payload tests), T-3-50
(`ResolvedSetFitConfig` Serialize-only), T-3-51 (single `warmup_steps_from_ratio`, ceil,
divergence-tested).

## Next Phase Readiness

Wave 2 can program against all of it:

- **03-05** consumes `SetFitRun<Prepared>`, `warmup_steps_from_ratio`, `WarmupLinearDecayLR`, `epoch_pair_order`, `reduce::*`, and the ADAMW_* constants. It must implement `LifecycleState for EncoderTuned` with its evidence type, and **must call `apply_freeze` before the initial parameter snapshot and before the pre-tuning baseline encode**.
- **03-06** authors `setfit-train-lifecycle-v1`; every error Display in this plan already cites that contract id and TRN-02, so the bindings have call sites waiting.
- **03-08** embeds both config forms; `ResolvedSetFitConfig` serializes `requested` and `resolved_device` side by side, and the freeze policy is canonicalized so equivalent policies hash identically.
- **03-09**'s trybuild non-constructibility proof has what it needs: sealed trait, private supertrait module, six private fields, `PhantomData<S>`.

**Concerns for the orchestrator:**
1. **Disk.** Three concurrent worktrees exhausted the volume twice. Wave 2 should either serialize or provision headroom; this executor deleted only its own `target/`.
2. **Publish cascade widened.** It is now `aprender-contrastive-data` → `{apr-cli, aprender-train}`. Measured, not inferred.
3. **D-ITEM-05** is new and belongs in STATE.md's blockers alongside D-ITEM-02.
4. This plan did **not** touch STATE.md or ROADMAP.md, per the worktree protocol.

## Self-Check: PASSED

Files verified present, commits verified in `git log`:

- `crates/aprender-train/src/train/setfit/mod.rs` — FOUND
- `crates/aprender-train/src/train/setfit/config.rs` — FOUND
- `crates/aprender-train/src/train/setfit/reduce.rs` — FOUND
- `crates/aprender-train/src/train/setfit/epoch.rs` — FOUND
- `crates/aprender-train/src/optim/scheduler/warmup_linear_decay.rs` — FOUND
- `.planning/phases/03-faithful-two-stage-trainer-and-head/deferred-items.md` — FOUND
- `f9e76856c` — FOUND
- `47a431ac7` — FOUND
- `e0116dc7c` — FOUND

---
*Phase: 03-faithful-two-stage-trainer-and-head*
*Plan: 03*
*Completed: 2026-08-09*
