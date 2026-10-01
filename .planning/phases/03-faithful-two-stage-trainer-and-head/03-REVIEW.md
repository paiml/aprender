---
phase: 03-faithful-two-stage-trainer-and-head
reviewed: 2026-08-12T01:05:12Z
depth: standard
files_reviewed: 68
files_reviewed_list:
  - contracts/aprender/binding.yaml
  - contracts/lbfgs-kernel-v1.yaml
  - contracts/multinomial-head-v1.yaml
  - contracts/setfit-train-lifecycle-v1.yaml
  - crates/aprender-compute/src/blis/parallel.rs
  - crates/aprender-core/Cargo.toml
  - crates/aprender-core/src/autograd/mod.rs
  - crates/aprender-core/src/classification/mod.rs
  - crates/aprender-core/src/classification/multinomial.rs
  - crates/aprender-core/src/classification/tests_multinomial_contract.rs
  - crates/aprender-core/src/nn/transformer/mod.rs
  - crates/aprender-core/src/nn/transformer/tests_seeded_attention_dropout.rs
  - crates/aprender-core/src/optim/lbfgs_tests.rs
  - crates/aprender-core/src/optim/lbfgs.rs
  - crates/aprender-core/src/optim/mod.rs
  - crates/aprender-core/src/optim/tests_lbfgs_contract.rs
  - crates/aprender-core/src/setfit/dropout_rng.rs
  - crates/aprender-core/src/setfit/encoder_tests.rs
  - crates/aprender-core/src/setfit/encoder.rs
  - crates/aprender-core/src/setfit/error.rs
  - crates/aprender-core/src/setfit/import.rs
  - crates/aprender-core/src/setfit/mod.rs
  - crates/aprender-core/src/setfit/model_tests.rs
  - crates/aprender-core/src/setfit/tokenizer_tests.rs
  - crates/aprender-core/src/setfit/tokenizer.rs
  - crates/aprender-core/tests/gemm_thread_determinism.rs
  - crates/aprender-train/Cargo.toml
  - crates/aprender-train/src/optim/mod.rs
  - crates/aprender-train/src/optim/scheduler/mod.rs
  - crates/aprender-train/src/optim/scheduler/warmup_linear_decay.rs
  - crates/aprender-train/src/train/mod.rs
  - crates/aprender-train/src/train/setfit/baseline.rs
  - crates/aprender-train/src/train/setfit/bundle_tests.rs
  - crates/aprender-train/src/train/setfit/bundle.rs
  - crates/aprender-train/src/train/setfit/config.rs
  - crates/aprender-train/src/train/setfit/epoch.rs
  - crates/aprender-train/src/train/setfit/evaluate_tests.rs
  - crates/aprender-train/src/train/setfit/evaluate.rs
  - crates/aprender-train/src/train/setfit/evidence.rs
  - crates/aprender-train/src/train/setfit/head_input.rs
  - crates/aprender-train/src/train/setfit/lock_tests.rs
  - crates/aprender-train/src/train/setfit/lock.rs
  - crates/aprender-train/src/train/setfit/mod.rs
  - crates/aprender-train/src/train/setfit/negative.rs
  - crates/aprender-train/src/train/setfit/reduce.rs
  - crates/aprender-train/src/train/setfit/test_fixtures.rs
  - crates/aprender-train/src/train/setfit/thresholds.rs
  - crates/aprender-train/src/train/setfit/tune.rs
  - crates/aprender-train/src/train/setfit/verify_tests.rs
  - crates/aprender-train/src/train/setfit/verify.rs
  - crates/aprender-train/tests/setfit_repro.rs
  - crates/aprender-train/tests/ui.rs
  - crates/aprender-train/tests/ui/setfit_direct_state_construction.rs
  - crates/aprender-train/tests/ui/setfit_direct_state_construction.stderr
  - crates/aprender-train/tests/ui/setfit_external_codec_impl.rs
  - crates/aprender-train/tests/ui/setfit_external_codec_impl.stderr
  - crates/aprender-train/tests/ui/setfit_fit_head_before_tune.rs
  - crates/aprender-train/tests/ui/setfit_fit_head_before_tune.stderr
  - crates/aprender-train/tests/ui/setfit_metric_value_asserted.rs
  - crates/aprender-train/tests/ui/setfit_metric_value_asserted.stderr
  - crates/aprender-train/tests/ui/setfit_pairs_into_fit_head.rs
  - crates/aprender-train/tests/ui/setfit_pairs_into_fit_head.stderr
  - crates/aprender-train/tests/ui/setfit_probe_claims_setfit.rs
  - crates/aprender-train/tests/ui/setfit_probe_claims_setfit.stderr
  - crates/aprender-train/tests/ui/setfit_token_without_lock.rs
  - crates/aprender-train/tests/ui/setfit_token_without_lock.stderr
  - Makefile
  - scripts/gen_multinomial_sklearn_fixture.py
findings:
  critical: 4
  warning: 11
  info: 4
  total: 19
status: issues_found
---

# Phase 3: Code Review Report

**Reviewed:** 2026-08-12T01:05:12Z
**Depth:** standard
**Files Reviewed:** 68
**Status:** issues_found

## Summary

Reviewed the Phase 3 diff (`git diff e2dee4be9..HEAD`, ~26,900 inserted lines) at standard
depth: the multinomial head and its L-BFGS widening in `aprender-core`, the native SetFit
two-stage trainer in `aprender-train`, the counter-based dropout RNG, the three new Makefile
reproducibility gates, and the three contract YAMLs plus the binding file.

The *implementation* arithmetic holds up under tracing. The sklearn parity fixture is
independently derived and its factor-2 control genuinely bites; `WarmupLinearDecayLR` matches
HuggingFace's `get_linear_schedule_with_warmup` exactly at both endpoints; `reduce.rs` is a
correct fixed-order f64 reduction; `epoch.rs`'s Fisher-Yates and `dropout_rng.rs`'s
multiply-shift keep rule are both frozen against externally-derived goldens; the L-BFGS f32
golden trajectory pins real bit patterns rather than re-blessed output; `blis/parallel.rs`'s
extraction is behaviour-preserving and the D-13 gate is genuinely mechanism-connected
(`Matrix::matmul` does route to `gemm_blis_parallel`, and `gemm_m_partitions` really does move
1 -> 2 bands between pool sizes 1 and 2).

The defects are almost all in the *verification* layer, which is the layer this phase spends
most of its prose claiming to have hardened. Four are blockers:

1. The phase's entire `aprender-train` test surface — every `#[cfg(test)]` module under
   `src/train/setfit/` and all seven trybuild compile-fail cases — is executed by **no tier and
   no CI job**, because `setfit` is not a default feature and nothing in the workspace enables
   it. The file that most loudly warns "a target outside the tiers is a target that stops being
   run" left ~6,000 lines of its own tests exactly there.
2. `setfit_repro_recorded_matches_expected_replay` — the one test that distinguishes
   "reproducible" from "correct" — is selected by neither Makefile filter, and libtest exits 0
   on a zero-match filter, so both repro gates are one rename away from silent vacuity.
3. The `loss_trace_hash` contract precondition ("a NaN or infinite step is a typed failure
   before hashing") has **no implementation at all**, and the consequence is worse than a
   missing check: `serde_json` renders every non-finite f64 as `null`, so the evidence table's
   "canonical bytes" hash cannot distinguish `+inf` from `-inf` from `NaN`, and the artifact
   the phase just wrote fails its own reload.
4. `thresholds_match_the_contract` cannot detect a Rust-side widening of the calibrated regime
   set — the exact T-3-21 loosening it exists to close.

## Critical Issues

### CR-01: Phase 3's `aprender-train` test surface is run by no tier and no CI job

**File:** `crates/aprender-train/src/train/mod.rs:48-50`, `crates/aprender-train/Cargo.toml:54`, `Makefile:287`, `crates/aprender-train/tests/ui.rs:55`

**Issue:** `train::setfit` is feature-gated:

```rust
// crates/aprender-train/src/train/mod.rs
#[cfg(feature = "setfit")]
pub mod setfit;
```

and `setfit` is **not** a default feature (`crates/aprender-train/Cargo.toml:54`,
`default = ["tui"]`). No workspace member declares `aprender-train` with
`features = ["setfit"]` (verified by grep across every `crates/*/Cargo.toml` and the root
manifest), and the workspace uses `resolver = "2"`, so feature unification never turns it on
either.

Consequently:

| Runner | Command | Does it compile `train::setfit`? |
|---|---|---|
| `make tier2` | `cargo test --lib` (root facade only) + `-p aprender-core --lib … setfit::` | no |
| `make tier3` | `cargo test --all` (default features) | **no** |
| CI `workspace-test` | `cargo nextest run --profile ci --workspace --lib` (`.github/workflows/ci.yml:272`) | **no** |
| `make setfit-feature-matrix` | `cargo check -p aprender-train --features setfit …` | compiles, **runs no tests** |
| `make setfit-repro-inproc/-crossproc` | `cargo test -p aprender-train --test setfit_repro --features setfit <filter>` | only that one integration target |

So none of the following ever executes in any gate: `bundle_tests.rs` (1045 lines),
`lock_tests.rs` (753), `verify_tests.rs` (694), `evaluate_tests.rs` (378), `config.rs`'s test
module (~400), `evidence.rs`'s test module (~1000, including the calibration matrix),
`tune.rs`'s test module (~600, including `tune_step_order_is_pinned` and
`tune_digest_is_recorded_not_recomputed`), `head_input.rs`'s tests, `thresholds.rs`'s tests
(including CR-04's `thresholds_match_the_contract`), `negative.rs` (the TRN-05 adversary), and
`tests/ui.rs`'s seven trybuild cases.

`tests/ui.rs:47-53` explicitly reasons about the feature gate ("Without the gate the harness
would compile in a default build … and report SEVEN PASSING compile-fail tests") and then does
not wire the gated form anywhere.

**Fix:** wire the two missing invocations beside the existing repro targets, and assert a test
count so a filter cannot go vacuous:

```make
setfit-train-unit: ## Phase 3: the setfit trainer's own unit + trybuild suites
	@mkdir -p target
	@set +e; \
	CARGO_INCREMENTAL=0 cargo test -p aprender-train --lib --features setfit setfit:: \
		> target/setfit-train-lib.log 2>&1; rc=$$?; \
	set -e; \
	tail -3 target/setfit-train-lib.log; \
	ran=$$(grep -oE '^test result: ok\. [0-9]+ passed' target/setfit-train-lib.log \
	       | grep -oE '[0-9]+' | head -1); \
	if [ "$$rc" -ne 0 ]; then echo "FAIL: setfit trainer unit suite is red (rc=$$rc)"; exit $$rc; fi; \
	if [ -z "$$ran" ] || [ "$$ran" -lt 100 ]; then \
	  echo "FAIL: the filter selected $$ran tests; this gate would pass vacuously"; exit 1; \
	fi
	@set +e; \
	CARGO_INCREMENTAL=0 cargo test -p aprender-train --test ui --features setfit \
		> target/setfit-train-ui.log 2>&1; rc=$$?; \
	set -e; \
	tail -3 target/setfit-train-ui.log; \
	if [ "$$rc" -ne 0 ]; then echo "FAIL: the seven trybuild cases are red (rc=$$rc)"; exit $$rc; fi
```

and add `@$(MAKE) setfit-train-unit` to `tier3` next to `setfit-repro-crossproc`
(`Makefile:336`), plus the target name to `.PHONY`.

---

### CR-02: The only test that separates "reproducible" from "correct" is never executed, and both repro gates pass vacuously on a rename

**File:** `crates/aprender-train/tests/setfit_repro.rs:512-513`, `Makefile:1424-1425`, `Makefile:1439-1441`

**Issue:** `setfit_repro.rs` ships three tests and documents (lines 3-24) that the third is the
one that can tell "the pipeline is a faithful function of the wrong thing" from "correct".
Neither Makefile target selects it:

```make
# Makefile:1424 — filter `in_process` matches only setfit_repro_in_process_two_runs_agree
cargo test -p aprender-train --test setfit_repro --features setfit in_process
# Makefile:1439 — filter `setfit_repro_cross_process` matches only that one test
cargo test -p aprender-train --test setfit_repro --features setfit setfit_repro_cross_process
```

`setfit_repro_recorded_matches_expected_replay` matches neither substring, and `cargo test
--all` compiles the file out (`#![cfg(feature = "setfit")]`, line 50). It therefore never runs
in any tier.

Worse, both gates are *structurally* vacuous: libtest exits **0** when a name filter selects
zero tests. Renaming `setfit_repro_in_process_two_runs_agree` (or moving it) leaves
`make tier2` printing `in-process: every composite component agreed` while having run nothing.
The Makefile comment at line 268 records "1 test" as a measured fact but nothing asserts it.

**Fix:** drop the filters and run the whole target once, then assert the executed count:

```make
setfit-repro: ## TRN-06/D-16: all three reproducibility claims, count-asserted
	@mkdir -p target
	@set +e; \
	CARGO_INCREMENTAL=0 cargo test -p aprender-train --test setfit_repro --features setfit \
		> target/setfit-repro.log 2>&1; rc=$$?; \
	set -e; \
	tail -5 target/setfit-repro.log; \
	if [ "$$rc" -ne 0 ]; then echo "FAIL: TRN-06 reproducibility is red (rc=$$rc)"; exit $$rc; fi; \
	passed=$$(grep -oE '^test result: ok\. [0-9]+ passed' target/setfit-repro.log \
	          | grep -oE '[0-9]+' | head -1); \
	if [ "$$passed" != "4" ]; then \
	  echo "FAIL: expected 4 tests (child + in-process + cross-process + replay), ran $$passed;"; \
	  echo "      a filter or a rename has made this gate vacuous."; exit 1; \
	fi
```

(If the in-process/cross-process split across tiers must be kept, keep the two filtered
targets but add the same `passed` assertion to each, and add the replay test to tier3.)

---

### CR-03: The `loss_trace_hash` precondition is unimplemented, and non-finite metrics collapse the evidence hash to `null`

**File:** `crates/aprender-train/src/train/setfit/tune.rs:641-646`, `crates/aprender-train/src/train/setfit/tune.rs:995-1001`, `crates/aprender-train/src/train/setfit/evidence.rs:409-412`, `contracts/setfit-train-lifecycle-v1.yaml:491-492`

**Issue:** The contract states, as a precondition of `loss_trace_hash`:

```yaml
    preconditions:
      - 'every loss value is finite; a NaN or infinite step is a typed failure before hashing'
```

No such check exists. `run_batch` pushes the value unconditionally:

```rust
let loss_value = loss.data()[0];
loss.backward();
// …
ctx.loss_trace.push(loss_value);        // tune.rs — no finiteness check
```

and `loss_trace_hash_of` hashes the bit pattern of whatever arrived
(`tune.rs:995-1001`). `grep -n "is_finite" tune.rs` finds finiteness checks only in
`validate_evidence`/`gated_row_failure` (lines 1146, 1228-1232) and in two test bodies —
`tune_loss_trace_is_finite` (line 1487) *observes* the fixture's trace, it does not enforce
anything about an arbitrary run.

The downstream consequence is a real data-integrity defect, not merely a missing guard:

* `UpdateEvidence::to_canonical_bytes` is `serde_json::to_vec(self)`
  (`evidence.rs:409-412`), and `serde_json` serializes every non-finite `f64` as `null`.
  `first_k_mean`, `last_k_mean`, `pre_clip_norm_max` and any ungated row's
  `relative_delta`/`grad_norm_*` therefore all render identically. `table_hash` — the value
  `evidence_table_hash()` publishes as the binding between summary and table — is **not
  injective** over non-finite measurements: `+inf`, `-inf` and `NaN` produce the same digest.
  `min_of`/`max_of` (`evidence.rs:596-611`) compound this, because `f64::min`/`f64::max`
  silently return the other operand when one side is `NaN`.
* The same struct travels into the bundle (`bundle.rs`, "the bound evidence summary"), and
  `serde_json` **cannot deserialize `null` into `f64`** — so a run with one non-finite
  aggregate produces an artifact that fails its own `verify_artifact` round trip with a
  `Codec`/`Bundle` parse error rather than with the honest "your loss diverged".

**Fix:** enforce the precondition where the value is produced, and keep the typed channel:

```rust
// crates/aprender-train/src/train/setfit/mod.rs — new variant
    /// A recorded per-step measurement was not finite.
    NonFiniteStepMeasurement {
        /// The global optimizer step.
        global_step: u64,
        /// Which measurement: `"loss"` or `"pre_clip_norm"`.
        what: &'static str,
        /// The offending value's bit pattern (a rendered NaN is not diagnostic).
        value_bits: u32,
    },

// crates/aprender-train/src/train/setfit/tune.rs — in run_batch, before (m)
    for (what, value) in [("loss", loss_value), ("pre_clip_norm", pre_clip)] {
        if !value.is_finite() {
            return Err(SetFitTrainError::NonFiniteStepMeasurement {
                global_step: ctx.global_step,
                what,
                value_bits: value.to_bits(),
            });
        }
    }
```

Then extend `gated_row_failure`'s finiteness conjunction to the *ungated* rows as well (they
are recorded in the hashed table even though they carry no verdict), and assert
`first_k_mean`/`last_k_mean`/`pre_clip_norm_max` finite in `UpdateEvidence::from_tune_output`
before `to_canonical_bytes` can be called.

---

### CR-04: `thresholds_match_the_contract` cannot detect a Rust-side widening of the calibrated regime set

**File:** `crates/aprender-train/src/train/setfit/thresholds.rs:451-468`, `crates/aprender-train/src/train/setfit/thresholds.rs:151-158`

**Issue:** The module header states the test's purpose: "Editing either side alone turns that
test red, which is what makes loosening an epsilon after a failing comparison require a
contract edit `pv diff` flags (T-3-21)". For the *epsilons* that is true — they are compared
field by field. For the **calibrated regime set** it is not:

```rust
let contracted_regimes = &parsed.equations.calibration_regime.calibrated_regimes;
assert_eq!(contracted_regimes.len(), 1, …);
assert_eq!(frozen.calibrated_regimes().len(), contracted_regimes.len(), …);
for regime in contracted_regimes {
    assert!(frozen.is_calibrated(regime), …);
}
```

`is_calibrated` resolves to `parse_calibrated(entry).covers(&observed)`, and `covers` is a
**subset** test (`thresholds.rs:155-157`):

```rust
self.architecture == run.architecture
    && run.seeds.is_subset(&self.seeds)
    && run.cells.is_subset(&self.cells)
```

So editing the Rust constant to

```rust
pub(crate) const CALIBRATED_REGIMES: &[&str] =
    &["minilm-slice-h64-l2-a2-i256-v97@1110a243|seeds=1,42,7,99|cells=s16e2b8,s8e1b4,s32e4b16"];
```

leaves the entry *count* at 1 and leaves the contract's seed/cell sets a subset of the widened
Rust sets — the assertion passes, and the gate now admits runs at seed 99 and cell `s32e4b16`
that were never calibrated. That is exactly the one-sided loosening the test's own docstring
claims to make impossible, and no other test closes it (`regime_every_calibrated_entry_parses`
only checks grammar; `calibrated_architecture()` only reads the architecture component).

**Fix:** assert set equality in both directions rather than coverage:

```rust
        let contracted: Vec<RegimeCoordinates> = contracted_regimes
            .iter()
            .map(|r| RegimeCoordinates::parse(r).expect("a contracted regime must parse"))
            .collect();
        let rust: Vec<RegimeCoordinates> = frozen
            .calibrated_regimes()
            .iter()
            .map(|r| RegimeCoordinates::parse(r).expect("a frozen regime must parse"))
            .collect();
        assert_eq!(
            rust, contracted,
            "the Rust calibrated set and the contract's must be EQUAL, not merely compatible. \
             `covers` is a SUBSET test, so a Rust-side widening (an extra seed or cell) passes \
             a coverage assertion while admitting runs nobody calibrated — the T-3-21 \
             loosening this test exists to block.",
        );
```

(`RegimeCoordinates` already derives `PartialEq, Eq`, so this compiles as written.)

---

## Warnings

### WR-01: The three new Makefile gates use the `rc=$$?` shape without `set +e`, so their entire failure-reporting block is unreachable

**File:** `Makefile:1424-1433`, `Makefile:1439-1449`, `Makefile:1455-1463`

**Issue:** The Makefile sets `.SHELLFLAGS := -e -c` with `.ONESHELL:` (lines 39-40). This
phase *documents* the resulting hazard twice — once for `setfit-feature-matrix` leg (a)
(lines 419-429: "a FAILING `cargo check` aborts the whole recipe before `ctl_rc=$$?` on the
same line can run … Reproduced directly: `bash -e -c 'false > /tmp/x 2>&1; rc=$$?; …'` prints
NOTHING") and once for `contract-audit-phase3` (lines 1305-1319) — and fixes both with
`set +e`. The three new reproducibility targets did not get the fix:

```make
	@CARGO_INCREMENTAL=0 cargo test -p aprender-train --test setfit_repro \
		--features setfit in_process > target/setfit-repro-inproc.log 2>&1; rc=$$?; \
	tail -3 target/setfit-repro-inproc.log; \
	if [ $$rc -ne 0 ]; then \
		echo "FAIL: the in-process two-run comparison is red (rc=$$rc)"; \
		…
		exit $$rc; \
	fi
```

Under `-e` the shell exits at the `cargo test` line, so `tail -3`, every `FAIL:` line, the
"tier2 as a WHOLE is red on arm64…" note and `exit $$rc` are dead code. The gate still fails
closed (make sees a non-zero shell), but the diagnostics the comment block promises never
print — the same "reported one contract where three were asked for" defect recorded at
line 1315. The comment at line 1401 asserting "EVERY RECIPE BELOW READS `$$?` ON THE LINE
AFTER THE REDIRECT" is therefore true textually and false operationally.

**Fix:** bracket the cargo invocation exactly as the two fixed targets do:

```make
	@set +e; \
	CARGO_INCREMENTAL=0 cargo test -p aprender-train --test setfit_repro \
		--features setfit in_process > target/setfit-repro-inproc.log 2>&1; rc=$$?; \
	set -e; \
	tail -3 target/setfit-repro-inproc.log; \
	if [ $$rc -ne 0 ]; then …; exit $$rc; fi
```

Apply to `setfit-repro-inproc`, `setfit-repro-crossproc` and `gemm-thread-determinism`, then
re-induce a failure in each and confirm the `FAIL:` text actually appears (the old proof does
not transfer — CLAUDE.md discipline 4).

---

### WR-02: A `pub` method was deleted from the published `aprender` crate with no CHANGELOG or semver note

**File:** `crates/aprender-core/src/nn/transformer/mod.rs:256-283`

**Issue:** `MultiHeadAttention::with_attention_dropout_seed(self, seed: u64) -> Self` and
`attention_dropout_seed()` were `pub` on a `pub struct` of a crates.io-published crate
(`[lib] name = "aprender"`). They are removed and replaced by
`pub(crate) with_attention_dropout_masks(…) -> Self`, whose parameter type
(`Arc<dyn AttentionDropoutMasks>`) is itself `pub(crate)`. `mix_call_seed` (`pub(crate)`) is
also gone. No in-tree caller exists (verified by grep), but any downstream `use` site breaks,
and the diff contains no CHANGELOG entry.

**Fix:** add a `### Breaking` entry to `CHANGELOG.md` naming both removed items and the
replacement, and confirm the version bump is a minor bump (pre-1.0 semantics) rather than a
patch. If a public extension point is wanted, re-export `AttentionDropoutMasks` publicly
instead of removing the door silently.

---

### WR-03: Two new contract bindings name a function that does not implement the equation

**File:** `contracts/aprender/binding.yaml` (new entries for `loss_trace_hash` and `pair_loss_endpoint`)

**Issue:** Both bind to `entrenar::train::setfit::evidence::from_tune_output`:

```yaml
- contract: setfit-train-lifecycle-v1.yaml
  equation: loss_trace_hash
  module_path: entrenar::train::setfit::evidence
  function: from_tune_output
  signature: 'fn from_tune_output(out: &TuneOutput, calibration_regime_id: &str) -> …'
  status: implemented
```

`from_tune_output` only hex-encodes a digest it receives (`evidence.rs:384`,
`loss_trace_hash: hex::encode(out.loss_trace_hash)`). The equation's formula — SHA-256 over
the f32 little-endian bit patterns in step order — is implemented by
`tune::loss_trace_hash_of` (`tune.rs:995-1001`). The same applies to `pair_loss_endpoint`,
whose statistic is computed by `tune::endpoint_means` (`tune.rs:1004-1012`).

`pv audit`'s BIND-001 checks that an *entry exists* for each equation; it does not resolve
`module_path::function` against the source. So `contract-audit-phase3` reports "every equation
is bound" while two equations point at a function that copies a value rather than computing
it — the mis-binding is invisible to the tier3 gate by construction.

**Fix:** re-point both entries at the functions that own the arithmetic:

```yaml
  equation: loss_trace_hash
  module_path: entrenar::train::setfit::tune
  function: loss_trace_hash_of
  signature: 'fn loss_trace_hash_of(trace: &[f32]) -> [u8; 32]'
```

and likewise `pair_loss_endpoint` -> `tune::endpoint_means`. Consider a follow-up ticket for a
`pv audit` rule that resolves `function` against the crate's symbols, since presence-only
auditing cannot see this class.

---

### WR-04: The reproducibility-accessor exhaustiveness count is a substring scan that misses `pub const fn`

**File:** `crates/aprender-train/src/train/setfit/verify_tests.rs:572-587`

**Issue:** The guard that replaced the old (admittedly blind) source scan is itself a source
scan:

```rust
let declared = block[..end].matches("\n    pub fn ").count();
assert_eq!(declared, ACCESSORS_CALLED_ABOVE, …);
```

Two gaps:

* `"\n    pub fn "` does not match `pub const fn`, `pub async fn`, or an accessor at any other
  indentation. `pub const fn` is the prevailing style in this very phase
  (`evaluate.rs:118 pub const fn metric_kind`, `evaluate.rs:124 pub const fn value`,
  `lock.rs:157 pub const fn rule`, `lock.rs:293 pub const fn chosen_index`), so a
  thirteenth accessor written in that style would be invisible to the count — exactly the
  drift the previous form was replaced for missing.
* `end = block.find("\n}\n")` truncates at the first column-0 `}` in the remainder of the
  file. Today the impl block closes there; any accessor body containing a raw string or macro
  with a column-0 `}` silently shortens the scanned region.

**Fix:** count both spellings and pin the pattern with a case table:

```rust
    let declared = block[..end]
        .lines()
        .filter(|l| {
            let t = l.trim_start();
            t.starts_with("pub fn ") || t.starts_with("pub const fn ") || t.starts_with("pub async fn ")
        })
        .count();
```

and add a unit test over a literal fixture block that must-match `pub const fn foo` and
must-not-match `pub(crate) fn foo` / `fn foo`, per CLAUDE.md discipline 7.

---

### WR-05: A trybuild snapshot pins an incidental rustc note about an unrelated third-party crate

**File:** `crates/aprender-train/tests/ui/setfit_external_codec_impl.stderr:11`

**Issue:** The blessed snapshot contains:

```
   = note: `MyCodec` implements similarly named trait `unicode_width::private::Sealed`, but not `verify::sealed::Sealed`
```

This line is a function of (a) `unicode_width` being in the dev-dependency closure and (b)
rustc's "similarly named trait" heuristic. Neither is related to the claim under test (that
`SetFitCodec` is sealed). Dropping or adding a transitive dependency that also has a `Sealed`
trait turns this case red for a reason `ui.rs:34-45`'s re-baseline instructions cannot
distinguish from a real regression — and the instructions tell the reviewer to check that the
*names* are still present, which they would be.

**Fix:** either add `trybuild`'s note-trimming (`t.compile_fail` with a `.stderr` reduced to
the `error[E0277]` block and the `required by a bound in` note), or narrow the case so rustc
has no similarly-named candidate to suggest. At minimum, add a comment in the `.rs` case
naming this line as dependency-derived so a re-baseline diff on it is not read as a claim
change.

---

### WR-06: The SAFE-03 baseline is a second encode path with a hard-coded window, so it is not bitwise comparable to the SetFit path

**File:** `crates/aprender-train/src/train/setfit/baseline.rs:36-47`, `crates/aprender-train/src/train/setfit/baseline.rs:161-176`

**Issue:** `head_input.rs:314-324` and `verify.rs` both record that a second encode path "would
measure a model the trainer never ran", and `head_input::encode_eval_rows` exists precisely so
out-of-`head_dataset` callers reach the one path. `FrozenProbeRun::fit` does not use it — it
re-implements the windowing loop with its own constant:

```rust
const PROBE_ENCODE_WINDOW: usize = 8;
…
for window in texts.chunks(PROBE_ENCODE_WINDOW) {
    let embedded = self.encoder.encode_texts(window)…;
    super::head_input::push_rows(&mut features, &embedded, window.len())?;
}
```

Consequences: (a) the probe pads to different per-window sequence lengths than the SetFit path
(reference `batch_size` is 16), which changes the GEMM shapes and therefore the low bits of the
embeddings; (b) the probe never checks `EncodeWitness::require_isolated()`, so a future encoder
change that starts recording under `no_grad` is red on the SetFit path and green here. The
constant's own doc comment records (a) as a KNOWN LIMIT, but the `fit` docstring two screens
below still asserts unqualified that "a probe and a SetFit run encode identical strings in
identical order — which is what makes the two comparable at all", and this is the control the
phase's headline claim is measured against.

**Fix:** thread the window size in and route through the shared door:

```rust
pub fn new(
    encoder: SetFitMiniLm,
    dataset: PreparedDataset<Canonical>,
    selection: Selection,
    regularization: Regularization,
    encode_batch_size: u32,      // from the SetFit run's ResolvedSetFitConfig
) -> Self { … }

// in fit():
let rows: Vec<(&str, &str)> =
    self.selection.ordered_ids().into_iter().zip(texts.iter().copied()).collect();
let features = super::head_input::encode_eval_rows(&self.encoder, &rows, self.encode_batch_size)?;
```

That deletes `PROBE_ENCODE_WINDOW`, picks up the isolation witness for free, and makes the
comparability claim true rather than aspirational.

---

### WR-07: The GEMM determinism gate's `cfg` names the wrong crate's `parallel` feature, and its skip message misattributes the cause

**File:** `crates/aprender-core/tests/gemm_thread_determinism.rs:96-106`, `crates/aprender-core/tests/gemm_thread_determinism.rs:264-272`

**Issue:** The test gates its pool-size reader on `#[cfg(feature = "parallel")]`, which in a
`tests/` target refers to **`aprender-core`'s** feature (`default = ["parallel"]`,
`parallel = ["rayon"]`). The partitioning under test is controlled by **`trueno`'s**
independent `parallel` feature, which `aprender-core` does *not* enable — the workspace dep is
`trueno = { path = …, version = "0.63.0", package = "aprender-compute" }` with no `features`,
and `aprender-compute` has `default = []`. `trueno/parallel` is on for this test only
incidentally, via the dev-dependency edge `crates/aprender-train/Cargo.toml:86`
(`trueno = { workspace = true, features = ["parallel"] }`) unified into the build graph.

The fallback's justification is therefore false as written:

```rust
/// Without `parallel` there is no pool; the GEMM is serial and the gate will
/// correctly report that the partitioning never moved.
#[cfg(not(feature = "parallel"))]
fn pool_threads() -> usize { 1 }
```

If `trueno/parallel` were ever off (e.g. that dev-dep is dropped, or the crate is tested
standalone), `gemm_partition_count_for` returns the constant `1` for every pool size, the
`partitionings.len() < 2` branch fires, and the test **passes** while printing:

```
SKIPPED-WITH-EVIDENCE: … (Likely cause: the FLOP ladder capped max_threads at 1.)
```

which names the wrong cause — and `make gemm-thread-determinism` then prints
`GEMM: identical hashes at pool sizes 1, 2 and 3` as though the claim held. The message also
interpolates `partitionings[0]` twice, the second occurrence labelled as `max_threads`.

**Fix:** make the dependency explicit and the skip cause honest:

```toml
# crates/aprender-core/Cargo.toml
[dev-dependencies]
trueno = { workspace = true, features = ["parallel"] }   # D-13 gate needs the M-partitioner
```

and in the skip branch report the mechanism rather than guessing:

```rust
        println!(
            "SKIPPED-WITH-EVIDENCE: every pool size (1, 2, 3) partitioned {m}x{k}x{n} into \
             {bands} band(s). Either trueno was built without `parallel` (in which case the \
             GEMM is serial and this gate is inapplicable) or the FLOP ladder capped \
             max_threads on this host. A3 remains an assumption.",
            bands = partitionings[0],
        );
```

---

### WR-08: `encode_ledger_hash`'s doc describes the NUL-terminated scheme that was deliberately replaced

**File:** `crates/aprender-train/src/train/setfit/mod.rs:928-931` (accessor doc) vs `crates/aprender-train/src/train/setfit/mod.rs:806-820` (`digest_of_ordered`)

**Issue:** `digest_of_ordered` length-prefixes each entry, and its own doc explains why the NUL
terminator was removed ("a Rust `String` may contain a NUL byte, so `["a\0b"]` and
`["a", "b"]` both hashed the byte stream `a 00 b 00`"). The accessor 120 lines below still
says:

```rust
    /// SHA-256 over the RECORDED encode ledger, entries NUL-terminated in order.
```

A stale statement about a wire format is exactly the sentence a future reader reimplements
from — and this digest is one of the ten components of the cross-process composite hash.

**Fix:** `/// SHA-256 over the RECORDED encode ledger, each entry LENGTH-PREFIXED with its
u64 little-endian byte length, in order.`

---

### WR-09: A Python script was committed into a tree whose own docs prohibit Python and assert none entered the repository

**File:** `scripts/gen_multinomial_sklearn_fixture.py`, `crates/aprender-train/src/train/setfit/epoch.rs:161-163`, `crates/aprender-train/CLAUDE.md`

**Issue:** `crates/aprender-train/CLAUDE.md` opens its toolchain section with "**CRITICAL:
Python is PROHIBITED.** This project uses only pure Rust tools from the Sovereign AI Stack",
and `epoch.rs:161-163` states "(PROJECT.md permits Python solely as a numerical reference
during verification; nothing Python entered the repository.)". This phase commits a 252-line
Python generator. Additionally, `make lint-scripts` / `bashrs` covers `scripts/*.sh` only, so
nothing lints or gates this file, and there is no test that the committed Rust constants in
`tests_multinomial_contract.rs:347-446` still match what the script produces.

**Fix:** pick one and make it consistent. Either (a) amend the prohibition to carve out
`scripts/reference/` verification-only generators and correct `epoch.rs`'s parenthetical, or
(b) move the generator out of the tree and keep only its output plus the recorded provenance
header. Whichever is chosen, drop the "nothing Python entered the repository" claim, since it
is now false as written.

---

### WR-10: Non-finite measurements are checked only for gated rows, and `min_of`/`max_of` swallow NaN

**File:** `crates/aprender-train/src/train/setfit/tune.rs:1224-1232`, `crates/aprender-train/src/train/setfit/evidence.rs:596-611`, `crates/aprender-train/src/train/setfit/evidence.rs:379-397`

**Issue:** `gated_row_failure` requires finiteness of five fields — but only for rows whose
class is `gated`. `attention_key_bias` rows (`gated: false`) are excluded from the conjunction
by construction (`gated` is filtered before the loop, `tune.rs:1124-1125`), yet their
`relative_delta` is recorded in the hashed table. The run-level aggregates
(`pre_clip_norm_max`, `first_k_mean`, `last_k_mean`) are never checked at all.

`min_of` / `max_of` then hide the residue:

```rust
values.iter().copied().fold(f64::INFINITY, f64::min)
```

`f64::min` returns the non-NaN operand, so one `NaN` relative delta leaves the per-class
`best`/`worst` looking clean, while `median_of`'s `partial_cmp(…).unwrap_or(Equal)` sort places
it at an arbitrary position. The doc says "every value here is finite by construction"; nothing
enforces that for the classes and aggregates above.

**Fix:** extend the finiteness conjunction to every row (gated or not) before the `gated`
filter, and validate the three run-level aggregates in `UpdateEvidence::from_tune_output`.
Pairs with CR-03's fix.

---

### WR-11: `ui.rs` globs its compile-fail cases and asserts nothing about how many were found

**File:** `crates/aprender-train/tests/ui.rs:57-61`

**Issue:**

```rust
let t = trybuild::TestCases::new();
t.compile_fail("tests/ui/*.rs");
```

The module doc enumerates "seven cases" in a table and reasons explicitly about the
"SEVEN PASSING compile-fail tests" vacuity mode, then leaves the case set to a glob. Deleting,
renaming to a non-`.rs` extension, or moving a case reduces the suite silently — the harness
reports success over whatever it found.

**Fix:** assert the count before running:

```rust
#[test]
fn ui() {
    let cases: Vec<_> = std::fs::read_dir("tests/ui")
        .expect("the ui case directory must exist")
        .filter_map(Result::ok)
        .filter(|e| e.path().extension().is_some_and(|x| x == "rs"))
        .collect();
    assert_eq!(
        cases.len(), 7,
        "the seven cases in this file's table are the claim; found {}: {:?}",
        cases.len(),
        cases.iter().map(std::fs::DirEntry::file_name).collect::<Vec<_>>(),
    );
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/*.rs");
}
```

---

## Info

### IN-01: A trybuild case's "Expected diagnostic" comment quotes a message the blessed snapshot does not contain

**File:** `crates/aprender-train/tests/ui/setfit_token_without_lock.rs:13-14`

**Issue:** The comment says the expected diagnostic is
``cannot construct 'CanonicalTestToken' with struct literal syntax due to private fields``,
while the committed `.stderr` says
``fields `lock_hash`, `artifact_hash`, `dataset_fingerprint` and `validation_split_fingerprint` of struct `CanonicalTestToken` are private``.
A reviewer following `ui.rs`'s re-baseline instructions compares the snapshot against these
comments; a comment that never matched cannot serve that purpose.

**Fix:** quote the actual first line of the snapshot.

---

### IN-02: `argmax_lowest_index(&[])` returns `0`, an out-of-range index

**File:** `crates/aprender-core/src/classification/multinomial.rs:509-522`

**Issue:** `best` is initialised to `0` and the loop never runs for an empty slice, so the
function returns an index no caller may use. Every current caller comes through
`predict_proba`, where `K >= 2` is validated, but the function is the bound implementation of
the contract's `label_order_semantics` equation and `label_order_semantics`'s precondition is
`indices.len() > 0` — which nothing in the function enforces.

**Fix:** return `Option<usize>` and let `predict_indices` map the `None` arm to
`HeadFitError::NotFitted`, or add a `debug_assert!(!values.is_empty(), …)` and state the
precondition in the docstring.

---

### IN-03: `fit` can store `f32` coefficients that its own reload door refuses

**File:** `crates/aprender-core/src/classification/multinomial.rs:1138-1140`, `crates/aprender-core/src/classification/multinomial.rs:951-963`

**Issue:** `fit` narrows with a plain cast:

```rust
self.weights = solution[..off].iter().map(|&w| w as f32).collect();
self.intercepts = self.intercepts_f64.iter().map(|&b| b as f32).collect();
```

Rust's `f64 -> f32` `as` cast saturates to `±inf` above `f32::MAX`, and
`from_stored_coefficients` rejects any non-finite stored coefficient with
`NonFiniteCoefficient`. A fit whose f64 solution exceeded f32 range would therefore return
`Ok(HeadFitReport)` and produce an artifact that cannot be reloaded. Unreachable for any
plausible penalised logistic fit, but the two doors disagree about the same invariant.

**Fix:** check finiteness after the narrowing and return `HeadFitError::NumericalError { iterations }`
(or a new `CoefficientOverflow` variant) so both doors enforce the same predicate.

---

### IN-04: `LbfgsOutcome` still calls `Instant::now()` on the "deterministic" f64 path

**File:** `crates/aprender-core/src/optim/lbfgs.rs:243`, `crates/aprender-core/src/optim/lbfgs.rs:826-841`

**Issue:** `LbfgsImpl::minimize` unconditionally records `start_time.elapsed()` into
`LbfgsOutcome::elapsed_time`, which `LbfgsF64::minimize` then deliberately drops because
"wall-clock time is a semantic-hash poison". The value never escapes, so this is not a
correctness defect — but the f64 path pays two clock reads per fit for a field it is
documented as refusing to carry, and the field's presence in the shared core is the kind of
thing a later change would plumb through to `OptimizationResultF64` "for symmetry".

**Fix:** make `elapsed_time` an `Option<Duration>` set only by the f32 wrapper's path, or gate
the timing behind a const generic / separate outcome type, so the f64 result type cannot grow
the field by accident.

---

## Notes on what was checked and found sound

Recorded so a later reviewer does not re-derive it:

- `WarmupLinearDecayLR::get_lr` matches HuggingFace's `get_linear_schedule_with_warmup`
  exactly at `current_step == warmup_steps` (peak), at `total_steps` (floor), and at
  `total_steps - 1` (`1/decay_steps`). `warmup_steps_from_ratio` uses `ceil` per the reference
  and clamps to `total_steps`.
- `validate_label_set`'s `HashMap` duplicate scan is order-independent: it keeps the minimum
  `first` index, which reproduces the pairwise scan's answer for every input.
- `worst_failing_gated_parameter`, `EvidenceSummary::of`'s `worst_param_name`, and
  `median_of` are all deterministic (`BTreeMap` iteration, name tie-break, total sort).
- `SelectionRule::apply` uses `f64::total_cmp` with a strictly-greater test and a value
  carried across iterations — deterministic, lowest-index tie-break, and `from_candidates`
  rejects non-finite values before the rule can rank a NaN above `+inf`.
- `CanonicalTestAccess::grant`'s dataset-fingerprint check is sufficient to cover the
  validation-split identity, since the dataset fingerprint digests the whole corpus.
- The D-13 GEMM gate is mechanism-connected: `Tensor::matmul` -> `trueno::Matrix::matmul`
  (`crates/aprender-compute/src/matrix/ops/arithmetic.rs:112-120`) -> `gemm_blis_parallel`, the
  `(m, n, k)` argument order agrees at both call sites, and `gemm_m_partitions(80, 384, 384)`
  yields 1 band at pool size 1 and 2 bands at pool sizes 2-3 (`max_threads = 2.min(phys_cores)`
  under the `flops < 64M` arm), so the "FALSIFIED" branch is the one taken on any host with
  two or more physical cores.
- The sklearn parity fixture is not a tautology: the constants were produced by an independent
  generator, the factor-2 control is asserted to diverge by >1e-3 (measured 2.89e-2), and
  `falsify_multinomial_001_sklearn_optimum_is_stationary_only_at_the_halved_lambda` checks
  stationarity without running an optimizer at all.
- No `unwrap()` in non-test code across the phase's new files; no `unsafe`; no SATD markers;
  no `matmul_*_colmajor` import; no inference/serving/KV-cache code in the training crates.

---

_Reviewed: 2026-08-12T01:05:12Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_

---

## Resolution — applied by the orchestrator 2026-08-12

The reviewer's findings above are preserved verbatim. This section records what happened
to them; it does not edit the findings.

| Finding | Status | Commit | Evidence |
|---------|--------|--------|----------|
| CR-01 | **OPEN — escalated to the human** | — | Fix requires editing `.github/workflows/*.yml`, which project CLAUDE.md reserves for explicit human approval. Item 4 of `03-HUMAN-UAT.md`. Confirmed independently: `setfit` declared at `aprender-train/Cargo.toml:79`, `default = ["tui"]` (:54), module gated at `train/mod.rs:51`, no workspace member enables it. tier3 only `cargo check`s the feature (`setfit-feature-matrix`, Makefile:338). |
| CR-02 | **FIXED** | `1038f6414` | Both halves. (a) `setfit-repro-replay` target added and wired into tier3. (b) `assert_tests_ran` refuses a gate that ran nothing — proven two-sided: real filter → rc=0/1 test; filter + `XX` → libtest prints `ok. 0 passed; 4 filtered out` and WOULD have exited 0, gate now exits 2. |
| CR-03 | **FIXED** | `0158a758d` | Two enforcement points. `SetFitTrainError::NonFiniteLoss` in `run_batch` before `backward()`; `EvidenceError::NonFiniteMeasurement` from a null-scan in `to_canonical_bytes`. Premise measured: `+inf`/`-inf`/`NaN` all render `null`, and `from_str::<f64>("null")` errors. Both RED-proven (check defeated → rc=101, returned bytes contain `"relative_delta":null`), then reverted. |
| CR-04 | **FIXED** | `51db85ace` | Regime ids compared for EQUALITY, not `is_calibrated`'s subset. RED-proven: widening the Rust set to `seeds=1,42,7,99\|cells=...,s32e4b16` left the OLD test at rc=0; the new assertion is rc=101 on the same mutation. Mutation reverted. |
| WR-01 | **FIXED** | `1038f6414` | `set +e` added to all four recipes under `.SHELLFLAGS := -e -c`. Proven reachable: pointing the GEMM recipe at a non-existent test target now prints its `FAIL:` diagnostic (rc=2) where it previously printed nothing. Gates already failed CLOSED, so this was lost diagnostics, not a false green. |
| WR-02..WR-11, INFO-01..04 | **OPEN — not triaged** | — | Not addressed in this pass. The scope decision covered the blockers plus WR-01 only. |

### Verification of the fixes

- `cargo test -p aprender-train --features setfit --lib setfit::` → **234 passed, 0 failed, 1 ignored**
- All four Phase 3 gates green with their real, asserted test counts: `setfit-repro-inproc` 1, `setfit-repro-crossproc` 1, `setfit-repro-replay` 1, `gemm-thread-determinism` 2
- `cargo clippy -p aprender-train --features setfit --lib --no-deps -- -D warnings` → **rc=0, 0 errors**
- `cargo fmt --check -p aprender-train` → **rc=0**

### Two caveats on the verification itself

1. **`cargo clippy` without `--no-deps` cannot reach this crate.** It fails first in
   `aprender-compute` with 19 lint errors under `-D warnings` — D-ITEM-02, which the
   Makefile already documents as the pre-existing arm64 tier2 red. So the `--no-deps`
   result above is the honest scope of what was linted: my changes, not the workspace.
2. **`bashrs` is not installed on this host**, so the project's mandated Makefile lint
   (`bashrs make lint Makefile`) did NOT run against the CR-02/WR-01 edits. `make -n`
   parses clean and no recipe pipes into `tee`, but that is not the same check.

### A defect found while reconciling the mutation inventory

`cargo mutants --list -f <form>` **exits 0 with empty output** when the form matches
nothing, rather than erroring. `-f 'src/train/setfit/**'` and `-f '*/setfit/*'` both
report zero mutants silently; the working forms are `-f '**/setfit/**'` and the full
repo-relative `-f 'crates/aprender-train/src/train/setfit/*.rs'`. This is CR-02's
zero-match vacuity in a second tool: a scoped mutation run could report a perfect score
having tested nothing. Any future run must assert a non-zero inventory first. Recorded in
`03-HUMAN-UAT.md` item 1.
