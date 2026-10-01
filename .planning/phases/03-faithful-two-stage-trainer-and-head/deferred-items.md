# Phase 3 — Deferred Items

Pre-existing defects discovered during Phase 3 execution that are OUT OF SCOPE for the plan
that found them. Each is measured, not assumed. Phase 3 does not inherit these as failures
and does not hide them either.

Numbering continues from Phase 2's `deferred-items.md` (D-ITEM-01..04), so a reader who
follows a reference from either phase lands on the right entry.

---

## D-ITEM-05 — `cargo check -p aprender-train --no-default-features` cannot compile

**Found by:** plan 03-03, Task 1 (2026-08-09)
**Owner crate:** `aprender-train` (`src/monitor/`) — NOT a Phase 3 module
**Severity:** blocks one acceptance criterion of plan 03-03; blocks nothing else

### Measurement

```
cargo check -p aprender-train --no-default-features                     -> rc=101, 8 errors
cargo check -p aprender-train --no-default-features --features setfit   -> rc=101, 8 errors
```

The two diagnostic streams (`grep -A1 '^error'`, 26 lines each) are **byte-identical**.
Every one of the 8 errors is in `src/monitor/tui/{app,dashboard}.rs`:

- 3 x `E0433: failed to resolve: use of unresolved module or unlinked crate presentar_terminal`
- 5 x `E0282: type annotations needed` (the shadow of the first three)

### Cause

`crates/aprender-train/src/monitor/mod.rs:45` declares `pub mod tui;` **unconditionally**,
while its only dependency is gated: `tui = ["dep:presentar-terminal"]`, and `tui` is in
`default`. So the module is compiled in a build that does not link what it uses. Nothing to
do with Phase 3 — the `setfit` feature contributes exactly zero errors, which is what the
byte-identical streams prove.

### Why plan 03-03 did not fix it

`src/monitor/` is outside the plan's declared `files_modified`, and wave-1 plans are
explicitly instructed to stay inside their own file sets. It is also not a one-line fix to
verify: gating the module means auditing every `pub use tui::{...}` re-export in
`train/mod.rs` and `monitor/mod.rs` for the same treatment.

### What 03-03 did instead

`make setfit-feature-matrix` leg (a) is wired as a **two-sided diff** rather than a plain
green check: it runs the minimal build with and without `setfit` and fails if the diagnostic
streams differ. It therefore asserts the property Phase 3 owns ("setfit does not leak into
the minimal build") without importing a red Phase 3 did not cause. The leg was falsified
before being trusted — a module compiled only under `cfg(not(feature = "setfit"))` and
importing `aprender_contrastive_data` made it fire, and was reverted.

### Fix direction

Gate `monitor::tui` on `feature = "tui"` together with its re-exports, then re-run the leg;
if the streams are then both empty and both rc 0, replace the diff leg with the plain
`cargo check --no-default-features --features setfit` the plan originally asked for. Its own
PMAT ticket.

---

## D-ITEM-02 (Phase 2) — re-measured in Phase 3

`cargo clippy -p aprender-train --lib --features setfit -- -D warnings` exits **101** at
Phase 3 wave 1, and every diagnostic is pre-existing:

```
error: could not compile `aprender-compute` (lib) due to 19 previous errors
error: could not compile `aprender-present-terminal` (lib) due to 1 previous error
```

Diagnostics citing `crates/aprender-train/` : **0**. Diagnostics citing `train/setfit/` or
`scheduler/warmup_linear_decay.rs` : **0**. The error stream is identical with and without
`--features setfit`. `aprender-compute` is owned by plan 03-02 in this wave, so 03-03 did
not touch it. Unchanged from Phase 2's entry apart from the crate counts.

---

## D-ITEM-06 (Phase 3, plan 03-05) — the MiniLM slice loader leaves 24 operations on the tape

`SetFitMiniLm::from_slice_fixture` (and, by construction, the shared
`BertSentenceEncoder::from_import` path behind it) appends **24 entries** to the thread-local
autograd tape while merely LOADING a model. Measured with the accessor 03-05 added:

```
clear_graph(); slice_encoder(SEED); graph_tape_len()  ->  24
```

The count is independent of any subsequent encode, and `no_grad` itself is honoured — the
same probe reports `graph_tape_len() == 0` after `no_grad(|| encode_texts(n))` for n = 1, 2
and 4, against a control of `98` for the same encode outside `no_grad`. So this is a LOADER
property, not a `no_grad` defect.

### Why it is not a correctness bug today

The trainer's step (b) clears the tape before every forward, so those 24 entries are gone by
the time the first backward runs. They are, however:

* a small permanent allocation for any consumer that loads an encoder and never trains
  (inference callers included), and
* a live trap for any future code that assumes a freshly loaded model implies an empty tape.
  Plan 03-05 wrote exactly that assumption first, and its test went red on it.

### Scope

Out of scope for 03-05: the recording happens in `crates/aprender-core/src/setfit/encoder.rs`
/ `import.rs`, which 03-05 does not own, and the fix is a Phase 1 change. 03-05's
`tune_baseline_encode_records_no_operations` asserts the **delta** across the baseline encode
rather than an absolute zero, and says in its doc comment why.

### Fix direction

Find the operation(s) `from_import` performs on tensors that already require grad — most
likely a reshape/transpose during projection installation — and either perform them on
detached data or wrap the constructor body in `autograd::no_grad`. Then tighten
`tune_baseline_encode_records_no_operations` to the absolute `== 0` form, whose current
non-vacuity assertion (`baseline_encode_tape.0 > 0`) will turn red and point here. Its own
PMAT ticket.

---

## D-ITEM-07 — `aprender-core` has ONE pre-existing arm64 clippy error, in a file Phase 3 does not own

Surfaced by plan 03-08, whose acceptance criteria run
`cargo clippy -p aprender-core --lib --features setfit -- -D warnings`.

**Measured**, scoped with `--no-deps` so the finding is attributable:

```
crates/aprender-core/src/demo/reliable/performance.rs:126:5
error: unreachable expression   (`-D unreachable-code` implied by `-D warnings`)
```

`cpu_backend_name()` returns unconditionally inside `#[cfg(target_arch = "aarch64")]`, so
the trailing `"Scalar".to_string()` is unreachable on this host. CI runs X64-Linux only,
where the arm arm is not compiled, so the line is live there and the error does not appear.

### Control

`git diff f56c34481 HEAD -- crates/aprender-core/src/demo/` is EMPTY: the file is
byte-identical to the wave-5 base commit, and no plan-03-08 change is in it. This matches
STATE.md's re-measurement at 02-08, which counted the arm64 clippy baseline as
"aprender-compute 38, zram-core 3, present-terminal 1, **core 1**, serve 1" — this is that
one.

### Scope

Out of scope for 03-08 under the executor's scope boundary: a pre-existing finding in an
unrelated file. Recorded here so that a reader of 03-08's clippy leg does not mistake it for
something the plan introduced, and so the honest scoped form is on record:

| Command | Result |
|---------|--------|
| `cargo clippy -p aprender-train --lib --features setfit --no-deps -- -D warnings` | rc=0 |
| `cargo clippy -p aprender-train --lib --features setfit --no-deps --tests -- -D warnings` | rc=0 |
| `cargo clippy -p aprender-core --lib --features setfit --no-deps -- -D warnings` | rc=101, ONE finding, the one above |
| either command WITHOUT `--no-deps` | rc=101, plus 10 `aprender-compute` arm64 findings |

### Fix direction

Add the `#[cfg(not(target_arch = "aarch64"))]` guard the x86 branch already has, or restructure
`cpu_backend_name` so every branch is an expression rather than an early return. One line, but
it belongs to whoever owns `demo/reliable/`, and it wants its own two-sided measurement on both
architectures rather than a blind edit from an arm64 host.
