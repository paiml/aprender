---
phase: 03-faithful-two-stage-trainer-and-head
plan: 01
subsystem: optimization
tags: [lbfgs, f64, numerics, contracts, non-finite]
requires: []
provides:
  - aprender::optim::LbfgsF64
  - aprender::optim::OptimizationResultF64
  - contracts/lbfgs-kernel-v1.yaml@1.1.0 (equation nonfinite_input_status)
affects:
  - aprender::optim::LBFGS (internals only; public API byte-compatible)
tech-stack:
  added: []
  patterns:
    - private float-generic core behind two non-generic public wrappers
    - per-width associated consts rather than shared literals
    - frozen IEEE-754 bit-pattern golden trajectory as a refactor guard
key-files:
  created: []
  modified:
    - crates/aprender-core/src/optim/lbfgs.rs
    - crates/aprender-core/src/optim/mod.rs
    - crates/aprender-core/src/optim/lbfgs_tests.rs
    - crates/aprender-core/src/optim/tests_lbfgs_contract.rs
    - contracts/lbfgs-kernel-v1.yaml
    - contracts/aprender/binding.yaml
decisions:
  - The public f32 type stays a non-generic struct; no default type parameter
  - OptimizationResultF64 omits elapsed_time entirely rather than carrying a documented hash poison
  - The two-loop-recursion contract binding moved off the panicking step onto the generic core
  - primitives/vector.rs was NOT touched; Vector::from_vec is already generic over T: Copy
metrics:
  duration: ~2h35m
  tasks: 2
  files: 6
  completed: 2026-08-09
---

# Phase 03 Plan 01: L-BFGS f64 Widening Summary

Widened the contracted L-BFGS solver to f64 behind a private float-generic core, keeping
the public f32 `LBFGS` byte-compatible and proving it so with a frozen bit-pattern golden
trajectory — and, in the process, found and fixed three genuinely broken non-finite input
channels that the contract already claimed to handle.

## Commits

| Commit | Type | What |
|--------|------|------|
| `71996df11` | refactor | Private `LbfgsFloat` trait + `LbfgsImpl<T>` + `WolfeSearch<T>`; non-generic `LBFGS` (f32) and new `LbfgsF64`; frozen golden trajectory |
| `714eda5b6` | test | f64 contract twins, ten-case non-finite matrix, contract 1.0.0 → 1.1.0, binding.yaml entries |
| `341a12d74` | fix | Fold `command:` into `test:` (they are the same serde field) so `pv validate` passes |

Base SHA for all scoped diffs: `e2dee4be9f5e3e97193d13489eb6cdd8006628c4`.

## What Shipped

**Shape (per the review fix, not a generic default parameter).** The algorithm lives once,
in the private `LbfgsImpl<T>`. Two NON-generic public types wrap it:

- `pub struct LBFGS` — unchanged f32: same field set, same `new(max_iter, tol: f32, m)`,
  same `minimize(f, grad, x0: Vector<f32>) -> OptimizationResult`. No generic parameter, no
  default type parameter, no new public method. Source assertion:
  `grep -c 'pub struct LBFGS<'` = **0**.
- `pub struct LbfgsF64` — `new(max_iter, tol: f64, m)`,
  `minimize(f, grad, x0: &Vector<f64>) -> OptimizationResultF64`. Source assertion:
  `grep -c 'pub struct LbfgsF64'` = **1**.

Both share the SAME `ConvergenceStatus`: `grep -rn "enum ConvergenceStatus"` across
`optim/` returns exactly one declaration (`mod.rs:198`). No status enum was forked.

The Wolfe constants for the f32 path are READ OFF the `line_search` field rather than
re-typed into the core, so the wrapper and the core cannot drift apart.

## The f32 Trajectory Is Proven Unchanged, Not Assumed

Captured at `e2dee4be9f5e3e97193d13489eb6cdd8006628c4` against the UNCHANGED solver,
confirmed green there, then confirmed green again after the rewrite **with zero literal
edits**. Frozen per case: every solution element's `f32::to_bits`, the `ConvergenceStatus`,
`iterations`, `gradient_norm.to_bits()`, the objective-evaluation count, and the ordered
accepted-objective bit sequence.

| Case | status | iters | objective calls | accepted seq len |
|------|--------|-------|-----------------|------------------|
| `quadratic_1d` f(x)=x0², x0=5 | Converged | 1 | 5 | 2 |
| `quadratic_2d` f(x)=x0²+x1², x0=(3,4) | Converged | 1 | 5 | 2 |
| `ill_conditioned_2d` cond≈1e4 | Converged | 6 | 43 | 17 |

Instrumentation is test-only: the objective closure pushes every call into a
`RefCell<Vec<f32>>` the test owns. **No `pub` item was added** — this settles the LOW review
note about whether strict-decrease checking needs an observable API. It does not.

## RED Observations (both recorded, as the plan required)

### 1. The tolerance test, and where the plan's assumption was wrong

The plan expected `1e-6` to be non-discriminating and `1e-10` to fail on the f32 path **on
the contract quadratic**. Measured, that is not true:

> On the contract quadratic f(x)=x0² from x0=5, **both** widths reach gradient norm exactly
> `0e0` in one iteration. Its optimum is exactly representable in f32, so **no** tolerance on
> that problem can demonstrate the width.

So the discriminating case is a dense 6×4 least-squares problem whose optimum is
representable in neither width. Measured on this host:

| tolerance | f32 | f64 | verdict |
|-----------|-----|-----|---------|
| `1e-6` | Converged, grad_norm `4.7332705e-7` | Converged, grad_norm `4.904488948122452e-7` | **VACUOUS** — cannot distinguish the widths |
| `1e-10` | MaxIterations after 2000 iters, grad_norm `6.124818e-9` | **Converged in 14 iters**, grad_norm `2.73222067168068e-11` | **DISCRIMINATING** — f32 provably cannot reach it |

Both observations are encoded as permanent tests rather than left as log lines:
`falsify_lbfgs_001_f64_tolerance_1e6_cannot_distinguish_widths` (the vacuity witness, so
the discriminating test cannot be quietly loosened back into vacuity) and
`falsify_lbfgs_001_f64_1e10_width_is_real`. `falsify_lbfgs_001_f64` additionally satisfies
the must_have literally on the contract quadratic (grad_norm 0 < 1e-10).

### 2. The non-finite matrix — three of five f32 channels were REALLY broken

Measured, not reasoned. The five f32 non-finite tests were run against the **pre-widening
solver**, restored byte-identically from `e2dee4be9` into the working tree (via `git show`
into scratch + `cp`, no git state mutation), then restored back and hash-verified:

| Channel | pre-fix status | required a solver fix? |
|---------|----------------|------------------------|
| (i) `x0` contains NaN | NumericalError | no |
| (ii) `x0` contains +inf | NumericalError | no |
| (iii) objective returns NaN at `x0` | **Converged** | **YES** |
| (iv) gradient returns +inf at `x0` | **Stalled** | **YES** |
| (iv-b) gradient +inf only at a line-search trial point | **Stalled** | **YES** |

Channel (iii) is the worst of the three: a poisoned start was reported as a **successful
optimization**. Channels (iv)/(iv-b) were reported as `Stalled`, a benign "no progress"
status, because `alpha < 1e-12` is FALSE for NaN — so the NumericalError check must precede
the stall check, and it now does.

`RED rc=101, 2 passed / 3 failed` → `GREEN rc=0, 5 passed`, then the five f64 twins were
added: **10 tests match `nonfinite`** (four channels + the line-search channel, × two
widths). Every one asserts the status DISCRIMINANT, not a message.

The fixes: entry guards on `x0`, `f(x0)`, `grad(x0)` and `‖grad‖`; a per-trial-point guard
inside the generic Wolfe loop that returns a NaN SIGNAL; and an `!alpha.is_finite()` check
ordered before the stall check. None of these fire on a finite trajectory, which is why the
golden bit patterns are unchanged.

## Contract

- `contracts/lbfgs-kernel-v1.yaml`: **1.0.0 → 1.1.0**.
- `pv validate contracts/lbfgs-kernel-v1.yaml` → **rc=0, "0 error(s), 0 warning(s), Contract is valid."**
- `pv diff /tmp/lbfgs-kernel-old.yaml contracts/lbfgs-kernel-v1.yaml` (old revision materialized
  with `git show e2dee4be9:...`, per the CLAUDE.md two-paths rule) →

  ```
  Contract diff: v1.0.0 → v1.1.0
  Suggested bump: minor
    equations:            + nonfinite_input_status
    falsification_tests:  + FALSIFY-LB-007  + FALSIFY-LB-008  + FALSIFY-LB-009
  ```

  **Additive (minor), as required — not major.**
- Every falsification command carries `--lib`: **3 commands, 3 `--lib` occurrences** (the
  02-06 lesson — a bare filter form emits `test result: ok` from a suite that ran zero
  matching tests and satisfies an `expected_output` grep vacuously).
- Kani: added `kani_harnesses_note` stating the harnesses are NOT executed (no
  `#[kani::proof]` exists repo-wide, cargo-kani is not installed) and naming the
  identically-bounded runnable proptest backing each, per the Phase 2 convention.
- `binding.yaml`: bound `two_loop_recursion` and the new `nonfinite_input_status`. The
  `grep -c 'two_loop_recursion' lbfgs.rs` = **1** — the annotation survived.

## Verification

| Gate | Result |
|------|--------|
| `CARGO_INCREMENTAL=0 cargo test -p aprender-core --lib optim::` | **rc=0, 516 passed** (full optim suite, per the shared blast-radius concern) |
| `cargo test -p aprender-core --lib lbfgs` | rc=0, 68 passed |
| `lbfgs_f32_golden_trajectory_is_unchanged` | rc=0, zero literal edits |
| `pv validate contracts/lbfgs-kernel-v1.yaml` | rc=0 |
| `cargo fmt -p aprender-core -- --check` | rc=0 |
| `cargo clippy -p aprender-core --lib -- -D warnings` | **rc=101 — PRE-EXISTING, control-measured (below)** |

### The clippy criterion is unsatisfiable at this plan's own base commit

The plan's acceptance criterion says clippy must exit 0. It cannot on this host, and that
is not caused by this plan. Rather than assert that, it was **measured two-sided**: the three
modified files were backed up, restored to base with `git checkout -- <paths>`, clippy run,
then restored and **SHA-256 verified byte-identical**.

| | BASE (`e2dee4be9`) | AFTER (this plan) |
|---|---|---|
| `cargo clippy -p aprender-core --lib --no-deps -- -D warnings` | rc=101 | rc=101 |
| aprender-core citations | 1 — `demo/reliable/performance.rs` | 1 — `demo/reliable/performance.rs` (identical) |
| citations naming a file this plan modified | 0 | **0** |

The single aprender-core error is an `unreachable_code` under `#[cfg(target_arch = "aarch64")]`
(`performance.rs:126`) — the same aarch64-only class as the 20 `aprender-compute` findings
already logged as STATE.md D-ITEM-02, and invisible on x86_64 CI. See Deferred Issues.

### No dependency added (scoped, per W-7)

`git diff -U0 e2dee4be9f5e3e97193d13489eb6cdd8006628c4 -- crates/aprender-core/Cargo.toml`
→ **0 lines**. The diff is deliberately SCOPED to the base SHA because same-wave peer 03-02
legitimately adds `aprender-rand` to that exact file; an unscoped diff would be green or red
by timing accident rather than by anything this plan did.

## Deviations from Plan

### 1. [Rule 3 — blocking] Branch precondition satisfied by worktree isolation, not by the named branch

The plan requires `git branch --show-current` to print `gsd/phase-3-two-stage-trainer` and to
STOP otherwise. This executor runs under Claude Code `isolation="worktree"`, on
`worktree-agent-a0d1aeef12416c33b`, based at `e2dee4be9`. Stopping would have been the wrong
call: the plan's own `<wave_1_concurrency>` names this arrangement as the preferred one —
*"If true parallelism is wanted, give each executor its own `git worktree` with its own target
dir. Both arrangements satisfy the rules above."* The rule exists to prevent three executors
racing on one shared index; with per-agent worktrees that race cannot occur. The orchestrator
merges the branches centrally. All wave-1 rules were honoured regardless: no `git checkout -b`,
no `git switch`, no `git stash`, no `git clean`, explicit-pathspec staging only.

### 2. [Rule 1 — bug] Three non-finite channels fixed

Documented in full above. The contract already CLAIMED this behaviour (T-3-02); no test
exercised it, and three of five channels did not deliver it.

### 3. [Rule 3 — blocking] `pv validate` rejected `command:` alongside `test:`

`FalsificationTest.test` carries `#[serde(alias = "command")]`
(`crates/aprender-contracts/src/schema/types.rs:466`), so `command:` is the SAME field. An
entry with both is a duplicate key, and `pv validate` said so:
`falsification_tests[6]: duplicate field 'test' at line 161`. Fixed by folding the runnable
command into the single `test:` field (commit `341a12d74`). This is exactly why the CLAUDE.md
rule says to dogfood `pv` rather than hand-check YAML — a bash validator would have passed it.

### 4. [Design, within plan latitude] `primitives/vector.rs` NOT modified

The plan permitted touching it "ONLY if construction helpers are missing for f64 — add the
minimum, nothing speculative." Measured: none are missing. `Vector::from_vec`, `from_slice`,
`len` and `Index`/`IndexMut` are already generic over `T: Copy`/`T`. Only `zeros`/`ones` are
f32-only, and the core needs no `zeros` it cannot build from `from_vec`. So a private
`fn zeros<T: LbfgsFloat>(n)` helper lives in `lbfgs.rs` and the shared primitive is untouched —
strictly smaller blast radius on a file the whole crate depends on. **No `Vector<f64>` helpers
were added.**

### 5. [Design] `OptimizationResultF64` has no `elapsed_time`

The plan allowed it "if documented as a hash poison in exactly the words the f32 form uses" —
but the f32 form's doc is just `/// Total elapsed time`, which does not warn anyone. Omitting
the field is strictly better for T-3-03: the f64 path exists to serve a deterministic,
hashable head fit (03-04), and a field that must never be hashed is a trap for the next
reader. The omission and its reason are documented on the type.

### 6. [Design] The `two_loop_recursion` binding moved to where the equation lives

It was on `LBFGS::step`, which only panics — L-BFGS has no stochastic update. It is now on
the generic core's `compute_direction`, which IS the two-loop recursion, and one annotation
now covers both widths.

## Findings Worth Carrying Forward

**The `#[contract]` macro is currently metadata-only in this repo.** `aprender-core/build.rs`
looks for `binding.yaml` at `crates/aprender-core/../../../provable-contracts/contracts/aprender/binding.yaml`
— a sibling-repo path that no longer exists after the APR-MONO merge. Measured: the path is
absent and every build prints `provable-contracts binding.yaml not found ...; CONTRACT_* env
vars will not be set`. Consequently `read_contract_assertions` returns an empty vector and the
macro injects **no** preconditions or postconditions. Practical effects: (a) adding contract
annotations to generic functions is safe, (b) preconditions written into
`lbfgs-kernel-v1.yaml` are **not** enforced at runtime, and (c) edits to
`contracts/aprender/binding.yaml` are audit-facing only — they change nothing at compile time.
Anyone who assumes the in-tree binding registry is wired to the build will be wrong.

**Host ENOSPC recurred mid-plan, a third time for this project.** During Task 2 the volume
filled completely — the Bash harness could not even allocate its own output file, so *no*
command could run for several minutes. Nothing was deleted by this executor beyond its own
`/tmp` logs and scratchpad. Space returned (4.4 GiB, then 75 GiB) on its own, consistent with
a peer wave-1 worktree finishing or being cleaned. The plan predicted this exact failure mode
("three concurrent 78-crate builds is the worst case"). `CARGO_INCREMENTAL=0` was set on every
build in this plan.

## Deferred Issues

- **`cargo clippy -p aprender-core --lib -- -D warnings` is red at base on aarch64.** One
  aprender-core finding (`demo/reliable/performance.rs:126`, `unreachable_code` under
  `#[cfg(target_arch = "aarch64")]` — the aarch64 arm returns unconditionally, making the
  trailing `"Scalar".to_string()` unreachable) plus 20 in `aprender-compute`. Same class as
  STATE.md D-ITEM-02; invisible on x86_64 CI. Out of scope for this plan (not caused by it,
  control-measured above). Not written to `deferred-items.md` deliberately: that file is
  outside this plan's `files_modified` and peers 03-02/03-03 are running concurrently in
  their own worktrees, so creating it here would produce a merge conflict for the
  orchestrator.

## Self-Check: PASSED

All six modified files exist (`wc -l`: lbfgs.rs 858, mod.rs 319, lbfgs_tests.rs 486,
tests_lbfgs_contract.rs 707, lbfgs-kernel-v1.yaml 246, binding.yaml 1138). All three commits
exist in `git log`: `341a12d74`, `714eda5b6`, `71996df11`. No files were deleted by any
commit (`git diff --diff-filter=D` empty for each).
