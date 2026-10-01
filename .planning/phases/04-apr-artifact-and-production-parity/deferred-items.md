# Phase 4 — deferred items

Out-of-scope discoveries logged during execution. Not fixed; recorded so they are
not rediscovered as if new.

---

## D-04-04-A — `cargo clippy -p aprender-core ... -D warnings` cannot pass, even with `--no-deps`

Discovered by plan 04-04, wave 4.

Orchestrator note F-03 established that `--no-deps` rescues the clippy gate for
`aprender-train` by excluding `aprender-compute`'s pre-existing debt. **That does not
extend to `aprender-core`**, which carries one of its own:

```
error: unreachable expression
   --> crates/aprender-core/src/demo/reliable/performance.rs:126:5
124 |         return "NEON".to_string();
126 |     "Scalar".to_string()
```

Measured, not assumed: `rtk proxy cargo clippy -p aprender-core --features setfit --lib
--no-deps -- -D warnings` exits **101** with exactly **one** error, the one above. The
20 `aprender-compute` entries in the same output are `warning`, not `error` — `--no-deps`
is working; the crate under test simply has debt of its own.

Consequence for any plan that states `cargo clippy -p aprender-core ... --no-deps --
-D warnings` as a gate: it fails today and would fail identically on an empty diff, so it
cannot distinguish "my code is clean" from "never linted" — the exact F-03 defect, one
crate over.

**What 04-04 did instead** (and what a Make target should do): run the same command and
assert **zero diagnostics whose path is under the plan's own files**. Verified at 04-04's
final state: zero lines matching `setfit/` in the clippy output.

**Action for 04-10:** either fix `demo/reliable/performance.rs:126` (a `cfg`-shaped
`return` followed by a fallback — the same pattern `aprender-compute` has four of), or
scope the core clippy leg by path. Do not "fix" it by dropping `-D warnings`.

---

## D-04-04-B — 24 `aprender-train --lib` tests fail in this worktree, in modules that link nothing this phase touches

Discovered by plan 04-04, wave 4. **Cause not diagnosed. Not claimed pre-existing** —
that was not measured.

`cargo test -p aprender-train --features setfit --lib` → `7865 passed; 24 failed`. The
failures are entirely:

- `gpu::guard::tests::*` (8)
- `gpu::ledger::tests::*` (12)
- `gpu::wait::tests::test_timeout_when_full` (1)
- `prune::snapshot_tests::tests::*` (3)

Sample: `gpu::ledger::tests::test_reserve_and_release` asserts `total_reserved() == 8000`
immediately after a successful `try_reserve(8000, …)` and observes `0`. Re-running with
`--test-threads=1` still fails 21 of them, so intra-binary parallelism is not the cause.
The ledger path is per-process (`temp_dir()/entrenar-ledger-test/test-ledger-{n}-{pid}.json`),
so cross-process contention with the parallel wave-4 agent is not the cause either.

**Proven NOT caused by this plan**, by import graph rather than by argument:

- `grep -rn "predict_proba\|predict_logits\|MultinomialLogisticRegression\|softmax"
  crates/aprender-train/src/gpu/ crates/aprender-train/src/prune/` → **zero matches**.
- `gpu/ledger.rs` imports `std`, `chrono`, `fs4`, `serde`, `super::{error, profiler}` and
  `crate::trace` — **nothing from `aprender-core`**.
- `prune/snapshot_tests.rs` imports only `crate::prune::…`.

The one control that would have been decisive — reverting `classification/multinomial.rs`
to the base blob and re-running — does not compile, because 04-04's `classify` calls the
`predict_logits` that refactor introduces. Reverting the whole task to run it was judged a
worse trade than recording the import-level proof.

Positive evidence that the refactor is behaviour-preserving where it matters:
`cargo test -p aprender-train --features setfit --lib setfit::` → **256 passed**, and
`cargo test -p aprender-core --features setfit --lib` → **14419 passed**. Those suites
include the probe-replay comparisons that check head logits and probabilities against
recorded artifact values.

**Action:** someone should run these four modules on a clean checkout of `main` to
establish whether they are pre-existing or environmental to this machine.

---

## D-04-08-A — 51 `aprender-serve --lib` tests fail on ONE arithmetic overflow in `contract_gate.rs:428`

Discovered by plan 04-08, wave 7. **Not fixed** — `contract_gate.rs` is outside this
plan's files and the fix needs a decision about the intended width.

`cargo test -p aprender-serve --features setfit --lib` → `15389 passed; 51 failed`. Every
one of the 51 panics at the SAME line, and the message is the same:

```
thread '...' panicked at crates/aprender-serve/src/contract_gate.rs:428:21:
attempt to multiply with overflow
```

The failing tests are `apr_transformer::tests::{q4k_bytes_q6k, tests_08, tests_10}::*` (49),
`contract_gate::tests::test_small_model_passes_resource_check` (1) and
`convert::convert_tests_q4k_converter::test_q4k_convert_roundtrip_loadable` (1). They are
one defect with 51 witnesses, not 51 defects.

**Not caused by this plan.** This plan's diff touches `api/` (the `AppState` slot, the
route, `HealthResponse`), `Cargo.toml` and test files. It does not touch `contract_gate.rs`,
`apr_transformer.rs` or `convert/`, and none of those modules names `AppState`,
`HealthResponse` or anything under `setfit`. The panic is an integer multiply in a
resource-estimation path reached from model construction — a code path with no edge to the
HTTP surface.

**Also pre-existing and unrelated:** `cargo check -p aprender-serve --tests` is red at the
base commit for two integration targets that have drifted from their types —
`tests/ffn_coverage.rs` (5 × missing `OwnedQuantizedLayer::{post_attn_norm_weight,
post_ffw_norm_weight}`) and `tests/gguf_extended_coverage.rs` (14 × missing
`GGUFConfig::query_pre_attn_scalar`). 04-08 fixed only the `HealthResponse` initializers its
own change broke (8 sites, 3 files) and left these 19 alone.

**Action for 04-10 (gates):** an `aprender-serve` test leg must either fix
`contract_gate.rs:428` first or be scoped by filter, because a whole-crate `cargo test
-p aprender-serve --lib` cannot go green today and so cannot distinguish a regression from
the standing red. 04-08's own leg is scoped: `--lib setfit` (see its SUMMARY).

---

## D-04-09-A — `cargo check -p apr-cli --no-default-features` does not compile (pre-existing)

**Found by:** plan 04-09, while verifying the Cargo.toml note plan 04-09 Task 1 step 1
requires, which names that command as the leg SAFE-02's gating evidence is read from.

**Measured twice** — on 04-09's tree, and again with the BASE manifest restored
(`git checkout f2824611a -- crates/apr-cli/Cargo.toml`), which is a true base measurement for
a lib-only check because 04-09's diff touches no `src/` file at all. Identical result both
times, same four errors:

```
$ cargo check -p apr-cli --no-default-features > log 2>&1; echo "rc=$?"
rc=101   error: could not compile `apr-cli` (lib) due to 4 previous errors
```

The four are `inference`-gated code that is not `cfg`-gated:

| file | error |
| ---- | ----- |
| `src/commands/explain.rs:231` | `realizar::safetensors::find_sibling_file` — unlinked crate |
| `src/commands/explain.rs:344` | same |
| `src/commands/diff_05_aprt_stage.rs:100` | `realizar::inference_trace::save_tensor::read_tensor_file` |
| `src/lib.rs:63` | re-exports `commands::serve::auth::apply`, which is `#[cfg(feature = "inference")]` |

**Not fixed here.** All four are in files 04-09 does not own, the fix is a `cfg` decision on
the `explain`/`diff` command surface, and 04-15 is editing `crates/apr-cli/` in the same wave.

**Action for 04-10 (gates) and 04-11 (audit): do NOT wire `--no-default-features` as a leg.**
It cannot distinguish a regression from this standing red. The leg that is green, and the one
04-06 and 04-07 actually used, is `cargo check -p apr-cli --all-targets` with DEFAULT features
(`setfit` OFF, `inference` ON) — the ungated build is what proves the `setfit` gating.
04-09's Cargo.toml carries the same note beside the dev-dependency block.

---

## D-04-10-A — `aprender-serve`'s minimal TEST build is red; the LIBRARY check is green

**Found by:** plan 04-10, while measuring the SAFE-02 run legs for the feature matrix. The
CHECK cells for `aprender-serve` at profiles (a) and (b) are both rc=0, so the discrepancy is
specific to the test target and would not have surfaced from a build-only matrix.

**Measured** (status captured directly off cargo, never through a pipe):

```
$ cargo check -p aprender-serve --no-default-features                        rc=0
$ cargo check -p aprender-serve --no-default-features --features setfit      rc=0
$ cargo test  -p aprender-serve --no-default-features --lib setfit           rc=101
$ cargo test  -p aprender-serve --no-default-features --features setfit --lib setfit  rc=101
```

**Cause:** `#[cfg(test)]` code imports feature-gated items unconditionally —
`crate::gguf::OwnedQuantizedModelCached`, `crate::gguf::OwnedQuantizedModelCachedSync`,
`crate::gguf::DequantizedFFNWeights`, `crate::gguf::DequantizedWeightCache`, `crate::gpu`, and
the `crate::api` GPU request/response types (`GpuBatchRequest`, `GpuStatusResponse`, ...). The
library compiles without `server`/`gpu`; only the test target needs them.

**Pre-existing and unrelated to setfit.** The two runs above produce the SAME first errors with
the feature OFF and ON, and the first four errors are `gguf`/`gpu` imports that setfit does not
touch. This is the same class as D-04-09-A one crate over: gated code that is not `cfg`-gated,
here inside `#[cfg(test)]` rather than in `src/`.

**Action taken by 04-10:** `setfit-feature-matrix` runs `aprender-serve` at profile (c) only,
and says so in the recipe. The (a)/(b) CHECK cells are still wired — they are green and they
are what the SAFE-02 build claim needs. `setfit-serve-tests` is likewise scoped to
`--features setfit --lib setfit` (10 passed / 0 failed).

**Action for whoever fixes it:** the fix is a `cfg` decision on `aprender-serve`'s test module
imports, not on setfit. Once green, add the two missing RUN cells to `setfit-feature-matrix`
beside the profile-(c) leg. Do NOT wire the whole-crate `-p aprender-serve --lib` suite in the
process — that is a separate standing red (D-04-08-A, 51 failures from one overflow at
`contract_gate.rs:428`).

---

## D-04-11-A — two PRODUCTION mutation survivors in `api/setfit_handlers.rs`, one of them surprising

**Found by:** plan 04-11's mutation gate. `crates/aprender-serve/src/api/setfit_handlers.rs` is
04-08's file, not 04-11's, so neither survivor was fixed here.

Invocation (note the two corrections — see 04-11-SUMMARY deviations 4 and 5):

```
cargo mutants --no-times --timeout 180 --package aprender-serve \
  --features setfit --cargo-arg=--lib \
  -f crates/aprender-serve/src/api/setfit_handlers.rs -- setfit
```

Baseline `ok`. The run was INTERRUPTED at ~68 min before finishing all 101 mutants, so there is
**no score** — but the survivors it did report are real and are recorded here rather than lost.

**Nine survivors reported; SEVEN are proven-equivalent by construction** — they mutate code inside
the file's own `#[cfg(test)]` module (five `tests::<fn> -> ()` and two in `fixture::Filler::next`).
A test suite cannot detect the deletion or alteration of one of its own tests, so these are not
coverage gaps. This is the F-05 self-scan class at the mutation tier: a `-f <file>` glob mutates
the file's tests along with its production code, and those survivors must be triaged out rather
than counted against the score.

**The two that matter:**

| # | Mutation | Source | Assessment |
|---|----------|--------|------------|
| A | `replace AppState::has_setfit_model -> bool with false` | `:70-72`, body is `self.setfit_model.is_some()` | **Real gap.** Under the `setfit`-filtered suite — the same filter `make setfit-serve-tests` uses — nothing pins this returning `true`. A readiness path that always reported "no classifier resident" would pass. |
| B | `replace > with == in setfit_classify_handler` | `:158`, `if request.texts.len() > MAX_BATCH_TEXTS` | **Real, and it should not have survived.** A co-located test is *named* `setfit_classify_refuses_a_batch_one_over_the_contract_bound`, i.e. `len == MAX+1` — the exact input that distinguishes `>` from `==`. Either that test does not exercise the branch its name claims, or it is not reached under this filter. **Not diagnosed here; one failing input is an anecdote (CLAUDE.md rule 6).** |

B is the more valuable finding and is precisely what mutation testing is for: a test whose NAME
asserts boundary coverage, beside a boundary mutant that lives. That is "labelling by intent"
(CLAUDE.md rule 2) one level down — at the test name rather than at the run.

**Action for the owner of `api/setfit_handlers.rs` (04-08's surface, or Phase 5):**
1. Re-run B's mutant alone and read why the named test does not kill it.
2. Add an assertion pinning `has_setfit_model() == true` on a state that has a resident model.
3. When re-running, exclude `#[cfg(test)]` functions from the mutation set so the score is not
   diluted by seven survivors that are equivalent by construction.

---

## D-04-11-B — the full mutation gate does not fit a single session: MEASURED projection

**Found by:** plan 04-11, which was asked to run a scoped mutation gate over all four crates.

Denominators, enumerated with `cargo mutants --list` (not estimated):

| Crate | Files | Mutants |
| ----- | ----- | ------- |
| aprender-core | `setfit/artifact.rs` (311), `setfit/classify.rs` (69) | 380 |
| aprender-train | `bundle.rs` (188), `lock.rs` (81), `config.rs` (63), `apr_reload.rs` (14), `apr_codec.rs` (12) | 358 |
| aprender-serve | `api/setfit_handlers.rs` | 101 |
| apr-cli | `setfit_train.rs` (25), `predict.rs` (20), `setfit_io.rs` (6) | 51 |
| **total** | | **890** |

Two timing measurements on the SMALLEST crate:

- `--shard 1/25` (4 mutants + baseline): **453 s**, 4/4 caught.
- full 101 mutants: **interrupted at 4,094 s (68 min) without finishing**.

From the second: the average cost per mutant, baseline included, is **> 40.5 s** — a lower bound,
since the run had not completed. `aprender-core` and `aprender-train` carry far heavier builds and
test suites than `aprender-serve`, so their per-mutant cost is strictly worse.

**Projection: ≥ 10 hours of wall clock for all 890 mutants**, ≥ 8.9 h for the 789 not yet
attempted, plus four baselines. That is beyond a single execution session, and the run that was
attempted was killed rather than completing.

**Reported and stopped rather than shrinking scope silently**, per the plan's own instruction and
the Phase 3 compute-budget precedent. **Recommendations for whoever runs it:**
1. Run it as a scheduled/nightly job per crate, not inside a plan execution.
2. Pass `--features` / `--cargo-arg`, never `-- --features` (deviation 4) — otherwise the whole
   run happens with `setfit` compiled out.
3. Pass `--cargo-arg=--lib` (deviation 5) or the baseline cannot build on `aprender-serve`.
4. Exclude `#[cfg(test)]` functions, or ~7 of every 9 survivors will be equivalent by construction.

## D-04-11-A — `setfit-api-boundary` excluded from CI (user ruling, 2026-08-15)

**Status:** deferred by explicit user decision at the 04-11 checkpoint. Excluded on QUOTING
grounds, not value grounds.

`setfit-api-boundary` is a `cargo tree` dependency-direction gate implemented as a Make `for` loop
with `$$`-escaped variables and single-quoted patterns. It cannot be embedded in the CI step's
single-quoted `bash -c '...'` without rewriting the quoting — and that rewrite is precisely the
drift the gate exists to detect. It continues to run locally via `make setfit-api-boundary`
(orchestrator-measured rc=0).

**Risk explicitly accepted by the user:** a Linux-only dependency-closure regression — one
introduced via `cfg(target_os)` so that the Linux closure differs from macOS — would NOT be caught,
because the gate now only ever runs on developer machines. This is the one class of regression a
local-only run cannot cover, and it is the counter-argument the executor recorded in the patch
beside the exclusion.

**If revisited:** the clean fix is to extract the loop into `scripts/setfit_api_boundary.sh` and
have both the Make target and a CI step invoke that script, so no quoting rewrite is needed. Note
`bashrs` (which CLAUDE.md mandates over shellcheck) is NOT installed on this host, so any new
script would need linting elsewhere.

## D-04-12-A — 04-REVIEW.md's non-Critical findings are open and now TRACKED (W-03)

The phase-04 code review returned **2 Critical, 6 Warning, 5 Info**. Both Criticals were fixed and
committed at `0fb47958f`:
- **CR-02 (security)** `POST /v1/classify` was mounted WITHOUT the AuthGate, so with `APR_API_KEY`
  set the classifier server was unauthenticated behind `CorsLayer::permissive()` — and since
  `AuthGate::from_env()` was never called, its "routes are unauthenticated" warning never printed
  either. Fixed with the same `auth::layer` the APR path already used.
- **CR-01 (correctness)** `apr eval --split test` scored artifact-head indices against dataset
  indices with no label-map gate, so a mismatched corpus produced a confidently wrong accuracy
  rather than an error — the exact failure the validation evaluator refuses by name
  (`apr_evaluate.rs`, `LabelMapMismatch`). Gate added in `run_test`, with a source guard asserting
  ORDER (not mere presence) plus non-vacuity on both sides, shown red by deleting the gate.

**The remaining 6 Warning + 5 Info are NOT fixed.** They were previously untracked, which is how
review findings quietly die. The two worth doing first, both write-path defects:

- **WR-02** (verifier re-confirmed present at `eval/setfit.rs:620-636`): `write_lock`'s temp file is
  predictable, symlink-following, and opened with `.create(true)` rather than `create_new` — while
  this repo's own precedent (`setfit_train::temp_path` + `create_new`) does the opposite.
- **WR-01**: `atomic_write` / `write_lock` both clobber through a check -> `fs::rename` window that
  their own doc comments claim to have closed.

Full findings with file:line and mechanism are in
`.planning/phases/04-apr-artifact-and-production-parity/04-REVIEW.md`.

## D-04-12-B — 04-11's mutation gate never produced a score (verifier gap, not F-10)

The only phase-04 gap NOT caused by F-10. One crate attempted, interrupted at 68 minutes against a
measured >=10 h projection for the lightest crate alone; cargo-mutants prints its summary only at
the end, so **no per-crate or aggregate score exists**. Two real production survivors in
`setfit_handlers.rs` remain undiagnosed — the notable one is `>` -> `==` at
`texts.len() > MAX_BATCH_TEXTS` surviving beside a test named
`..._refuses_a_batch_one_over_the_contract_bound`, i.e. the exact distinguishing input.

Two recipe defects were found and fixed en route, and both invalidate any earlier confidence at
this tier: `-- --features setfit` never reached cargo (so setfit was compiled OUT — F-04 vacuity),
and cargo-mutants had no baseline at all for `aprender-serve` (fixed with `--cargo-arg=--lib`).

## D-04-13-A — cleanup-pass findings deliberately NOT applied (2026-08-16)

A four-angle cleanup review (reuse / simplification / efficiency / altitude) over the phase-4
diff applied ten fixes and deferred the rest. The deferred items are real, were each named by at
least one angle, and are listed here rather than dropped. They were skipped because each is a
cross-crate refactor or a multi-site mechanical change beyond a cleanup pass, not because they
were judged wrong.

**Highest value first:**

1. **Three functions now exceed the ENFORCED `max_complexity = 10`** (`.pmat-gates.toml`),
   measured with `pmat analyze complexity`. This is a live gate violation, not a preference:
   - `commands/setfit_train.rs::run` — cyclomatic **15**, cognitive 17, nesting 5
   - `setfit_tag.rs::read_setfit_tag` — cyclomatic **12**, cognitive **27** (threshold 15)
   - `train/setfit/apr_evaluate.rs::evaluate_validation_from_artifact` — cyclomatic **12**
   Item 2 below fixes the third for free.

2. **The classify-and-score half of `apr_evaluate.rs` is re-implemented in the CLI.**
   `eval/setfit.rs`'s label-map comparison and its chunk-by-`MAX_BATCH_TEXTS` / `position()` /
   arity-check loop duplicate `apr_evaluate.rs` almost line for line — and the CLI copy silently
   scores an unrecognised predicted label as WRONG where the library refuses with a typed error.
   Fix: extract `check_label_map(model, dataset)` and `classify_rows_to_indices(model, rows,
   labels)` as `pub` in `apr_evaluate.rs`; both callers use them.

3. **`eval/setfit.rs` computes test accuracy outside the ONE reduction door.** It tallies in a
   `usize` and divides, while the validation side averages an indicator vector through
   `reduce::mean_in_index_order` — "the trainer's ONLY reduction door (D-13)", whose purpose
   (TRN-06) is bitwise-stable index-order f64 accumulation. Both numbers land in the same
   `EvalRow` schema carrying `value_bits`, which exists precisely so a determinism assertion can
   compare bit patterns — the comparison this divergence makes meaningless ACROSS SPLITS.

4. **~100 lines of atomic-writer stack duplicated VERBATIM** (including doc-comment prose)
   between `commands/setfit_train.rs:110-207` and `commands/data_contrastive.rs:83-192`. This
   pass removed the third, degraded copy (`write_lock`) by delegating to `atomic_write`; the
   remaining two should collapse into one `crate::atomic_io` module. Net ~-140 lines.

5. **A bounded small-file reader written twice**: `predict.rs::read_request_document` and
   `eval/setfit.rs::read_lock` differ only in the cap constant and the noun. The subtle
   `take(CAP + 1)` re-check now exists in two places. Fix: `read_bounded(path, cap, what)` in
   `setfit_io.rs`, whose module header already declares it the crate's bounded-read door.

6. **`dispatch_analysis.rs::dispatch_classify_eval` takes 13 positional params** behind
   `#[allow(clippy::too_many_arguments, clippy::fn_params_excessive_bools)]` — the lints fired
   and were silenced rather than heeded — plus a hand-maintained flag-refusal table. `--split`'s
   "was it supplied?" is reconstructed by comparing against clap's own `default_value` instead of
   being typed as `Option<String>`. Fix: a `#[derive(Args)] struct` with `#[command(flatten)]`.

7. **`AppState` gained the identical `setfit_model: None` line at 17 struct-literal sites**
   because none of the 16 other constructors delegates. Now that `Default` exists, rewrite them as
   `Self { .., ..Self::default() }`. Entirely compiler-checked; highest noise-removed-per-risk.

8. **`inspect.rs`: the SetFit section travels as a parallel positional argument** to
   `output_json`/`output_json_with_quality` even though `MetadataInfo` already carries
   `setfit_doc`. A second artifact family adds another positional param to both functions and
   every call site. Fix: build it from `metadata.setfit_doc` where `InspectResult` is assembled.

9. **`predict.rs`: `--text`/`--input` exclusion is expressed at two altitudes** — clap's
   `conflicts_with` AND `RequestSource::choose`'s arm. clap wins at parse time, so the arm is
   unreachable through the CLI and its better-worded message never ships; the unit test calls
   `choose` directly, so the TESTED message is not the SHIPPED message. CLAUDE.md verification
   item 5 in miniature. Fix: drop `conflicts_with`, keep the good message.

10. **`eval/setfit.rs::run_test` re-derives the lock path with a weaker, unreachable refusal**
    (`check_split_flags` already guaranteed it), and the lock file is read only AFTER the corpus
    ingest and the full artifact verification — contradicting the file's own comment that a run
    "has told the operator something it knew before it started". Fix: read and validate the lock
    immediately after `check_split_flags`, pass the `SelectionLock` into `run_test`.

**Confirmed clean by the review, recorded so it is not re-litigated:** the retained ~1.8 MB
artifact buffer is never cloned or copied in non-test source (moved end to end); `drop(reloaded)`
keeps its measured position; `POST /v1/classify` has no hoistable per-request work and installs as
a fn pointer, not a capturing closure; the sealed `SetFitCredential` and the consuming
`into_artifact_bytes` are at the right altitude — and the consuming signature means the
read-then-take ordering is enforced by BORROWCK, not by call-site discipline.

## D-04-14-A — round-2 review: WR-07 fixed, three warnings open (2026-08-16)

Round 2 returned **0 Critical**. Both round-1 Criticals verified genuinely closed (not cosmetically),
WR-02 closed by the `atomic_write` delegation, IN-03 closed. All six cleanup-pass code paths were
verified correct — the reviewer recovered the deleted tag detector from `a523db6b3` to compare
semantics rather than taking the claim on trust.

**WR-07 — FIXED, and it was a regression introduced BY the CR-02 auth fix.** Gating the shared
router put auth OUTSIDE `CorsLayer::permissive()`, so a browser's `OPTIONS /v1/classify` preflight
got 401 with no `Access-Control-Allow-Origin` before CORS saw it, and every `/health/ready` probe
401'd — the endpoint 04-08 added specifically so an orchestrator could admit a classifier. The
security fix would have made the server unreachable from a browser and unschedulable under k8s.
Closed by `auth::layer_public_ops`, which exempts only CORS preflight (defined to travel without
credentials, so gating it is unsatisfiable by a conforming client) and `/health*` (no model data;
both were fully public before the classify surface was gated at all). `POST /v1/classify` stays
gated. The APR path is untouched — a new function rather than a change to shared `apply`. Three
tests drive the REAL middleware through a real router, and a mutation removing the exemptions fails
exactly the two operability tests while leaving the security test green.

**STILL OPEN — three warnings, highest value first:**

**WR-09 (do this first).** `commands/inspect.rs:548-566` bounds `metadata_size` — an
attacker-controlled `u32` with no checksum branch — only by file length. `truncate -s 4297M` costs
nothing on a sparse filesystem, so `apr inspect` allocates ~4 GiB and then SUCCEEDS. The 16 MiB
`MAX_TAG_METADATA_BYTES` added to `setfit_tag.rs` in the cleanup pass exists for exactly this class
and is two files away. Fix: apply the same cap.

**WR-08.** The cap's `Ok(None)` fail-open is justified in its own comment by "the caller falls
through to the pre-existing APR path" — true for `apr serve` ONLY. `apr predict` has no APR path
and emits "not a SetFit classifier … run apr inspect"; `apr inspect` has no cap and renders the
full APR-05 section. So the three tools now disagree about one file, and the message routes the
operator INTO the contradiction. Fix alongside WR-09.

**WR-10.** `--lock-out`'s no-clobber check runs AFTER the full multi-candidate sweep, inverting the
ordering discipline `setfit_train.rs:12-20` states and that `refuse_existing_output` was made
standalone to support.

## D-04-14-B — IN-06: the unwrap() ban is NOT machine-enforced in apr-cli

`crates/apr-cli/src/lib.rs:9-16` carries crate-wide
`#![allow(clippy::all, clippy::pedantic, clippy::disallowed_methods, unused_imports, dead_code, …)]`.

`clippy::disallowed_methods` is the mechanism `.clippy.toml` uses to ban `unwrap()`, and CLAUDE.md
states "0 unwrap()" as a project threshold — so the project's headline quality gate is silently
inert across the entire CLI crate. Verified empirically by the reviewer, not inferred: a fresh
`cargo check -p apr-cli --lib --features setfit` recompiles and emits ZERO diagnostics for
`apr-cli/src`, including for a genuinely dead `Write as _` import at `eval/setfit.rs:40`.

Phase 4's own code is clean — 0 `unwrap()` across all 13 new modules, counted — but by discipline
rather than by gate, which is exactly the distinction CLAUDE.md's verification discipline warns
about. This is a PROJECT-level finding, far beyond phase 4: removing the blanket allow will surface
a backlog across the whole crate and needs its own ticket and its own pass.

---

## D-04-21-A — the `setfit_handlers.rs` survivor set RE-MEASURED at HEAD; D-04-11-A corrected in three ways (2026-08-16)

**Supersedes the measurement in `## D-04-11-A` (the mutation-survivor one at line 199), which is
left byte-unchanged above.** It was true when written. Three commits landed after it, one of them
deleted the function its Survivor A mutates, and its own run was INTERRUPTED at 68 min before
reaching the whole set — so it is incomplete rather than wrong.

Measured by plan 04-21 at HEAD `91eba6f1c`, `cargo-mutants 25.3.1`.

### The enumeration (`--list`, not estimated)

```
cargo mutants --list --package aprender-serve --features setfit --cargo-arg=--lib \
  -f crates/aprender-serve/src/api/setfit_handlers.rs
```

**99 mutants total; 78 are `tests::` / `fixture::`; 21 are PRODUCTION.** The 78 stay triaged out on
D-04-11-A's own argument — a suite cannot detect the deletion or alteration of one of its own tests
— and the two counts are recorded so the exclusion is auditable rather than a hand-wave. D-04-11-B
recorded 101 for this file; the drop to 99 is the `b47acc4fe` cleanup.

### The run

```
cargo mutants --no-times --timeout 180 --package aprender-serve \
  --features setfit --cargo-arg=--lib \
  -f crates/aprender-serve/src/api/setfit_handlers.rs \
  -E 'tests::' -E 'fixture::' -- setfit
```

Baseline `ok`. **COMPLETED in 593 s** (D-04-11-A's was interrupted at 4,094 s, which is why it never
produced a score). Verbatim summary line:

```
21 mutants tested: 3 missed, 5 caught, 13 unviable
```

Viable production mutants = 8. `-- setfit` is load-bearing, not decoration: the unfiltered
`-p aprender-serve --lib` suite is 51-red (D-04-08-A), so a baseline reported `ok` is itself the
proof the filter engaged. The `setfit` feature was genuinely ON — the whole file is behind
`#[cfg(feature = "setfit")]` and produced 21 mutants, which a build with it compiled out cannot do
(the F-04 vacuity check D-04-11-B's deviation 4 exists for).

### Correction 1 — Survivor A is MOOT BY DELETION, not fixed

`AppState::has_setfit_model` was added at `82416fcc1` and **deleted at `b47acc4fe`** ("Deleted
`has_setfit_model` (zero callers)"). At HEAD `grep -rn has_setfit_model crates/` returns nothing
(rc=1) and the `--list` log contains **zero** occurrences. The mutant D-04-11-A named cannot be
re-run because the code is gone. **No test was written for it** — writing one would have been
writing a test for deleted code.

Its successor surface is `AppState::setfit_model()` (`setfit_handlers.rs:65`), whose ONLY production
caller is the readiness assembly at `router.rs:249`
(`state.setfit_model().map_or((None, None), …)`); the `model_loaded` half is separate and reads the
FIELD directly (`mod_app_state_gpu.rs:420`). The successor mutant
`replace AppState::setfit_model -> Option<&Arc<VerifiedSetFitModel>> with None` was **MEASURED
CAUGHT** — `setfit_readiness_is_200_and_reports_the_exact_artifact_hash` (`:811`) asserts the exact
hash VALUE, and under `-> None` the key is absent. So the readiness gap D-04-11-A's action item 2
asked to close **does not exist at HEAD**, and no test was added for it either. Recorded as a
measurement rather than closed with a test that would kill nothing.

### Correction 2 — Survivor B is TWO mutants, not one, and moved line

D-04-11-A records `:158`; at HEAD the expression is `setfit_handlers.rs:152`. Three comparison
mutants live at `:152:28`, and they do NOT behave alike:

| Mutant | Verdict at HEAD (before this plan) |
|--------|------------------------------------|
| `replace > with ==` | **MISSED** |
| `replace > with >=` | **MISSED** |
| `replace > with <`  | CAUGHT |

### Correction 3 — a THIRD survivor D-04-11-A never reported

`:108:9: delete match arm ClassifyError::EmptyInput | BatchTooLarge{..} | UnsupportedSchemaVersion{..}
in classify_error_response` — MISSED. D-04-11-A did not report it because its run was interrupted
before reaching it, not because it was absent. See "Still open" below: it is **equivalent by
construction**, with evidence.

### The varied-input diagnosis for Survivor B (CLAUDE.md rule 6)

D-04-11-A explicitly declined to diagnose from one input. **Three inputs were EXECUTED** — through
the real router via `oneshot`, under the correct code and under each hand-applied mutant — not
reasoned about:

| batch size | correct `>` | mutant `==` | mutant `>=` |
|-----------|-------------|-------------|-------------|
| `MAX_BATCH_TEXTS - 1` = 255 | **200** | 200 | 200 |
| `MAX_BATCH_TEXTS` = 256     | **200** | **400** `the request carried 256 texts; the bound is 256` | **400** (identical) |
| `MAX_BATCH_TEXTS + 1` = 257 | **400** `the request carried 257 texts; the bound is 256` | **400** (byte-identical) | **400** (byte-identical) |

**The cause, named from the measurements.** At 257 all three implementations answer `400` with the
same body. Under the mutants the transport check falls through, core re-checks the same bound at
`classify.rs:624`, returns `BatchTooLarge { max: 256, got: 257 }`, and `classify_error_response`
maps it to `BAD_REQUEST` — the same status and the same `Display` string the transport `refuse`
produces. So `setfit_classify_refuses_a_batch_one_over_the_contract_bound` (`:725`), which asserts
400 plus the substrings `257` and `256`, **provably cannot distinguish the implementations**.

The diagnosis is therefore **neither of the two D-04-11-A offered**. The test is not mis-scoped and
it is not unreached — it reaches the branch and asserts truthfully. A redundant INNER check makes
the outer branch unobservable through the response *for that input*. This is the cost of defense in
depth (T-04-23), and it is worth paying; it just means the boundary needs a different witness.

**The distinguishing input is exactly `MAX_BATCH_TEXTS`**, where `>` admits and both `==` and `>=`
refuse. `aprender-core` already had that acceptance test (`classify.rs:1729`); `aprender-serve` had
none, which is precisely why two mutants lived there.

### What was done

`setfit_classify_admits_a_batch_at_exactly_the_contract_bound` added to `setfit_handlers.rs`,
reading the bound from the exported constant and asserting the body sits far under
`classify_body_limit_bytes()` so it cannot silently become a body-limit test.

Proven RED/GREEN asymmetric by hand-applied probe — the finding itself, so recorded verbatim:

| hand-applied probe at `:152` | `…admits_a_batch_at_exactly_the_contract_bound` | `…refuses_a_batch_one_over_the_contract_bound` |
|------------------------------|--------------------------------------------------|--------------------------------------------------|
| `> → ==` | **FAILED** | ok |
| `> → >=` | **FAILED** | ok |

Then confirmed by cargo-mutants itself, `-F 'replace > with .* in setfit_classify_handler|delete match arm'`,
baseline `ok`, verbatim: **`4 mutants tested: 1 missed, 3 caught`** — all three `:152` mutants
CAUGHT, zero missed among the mutants this plan claims to have killed.

### Still open

**`:108` delete-match-arm — MISSED, and triaged EQUIVALENT BY CONSTRUCTION.** No test was added,
because none can exist at this tier. `classify_error_response` has exactly ONE caller
(`setfit_handlers.rs:166`, the `map_err` on `model.classify`), so the arm is reachable only if
`classify()` can return one of its three variants after the transport pre-checks pass:

* `EmptyInput` — pre-empted by `:146`, unreachable;
* `BatchTooLarge` — pre-empted by `:152`, unreachable;
* `UnsupportedSchemaVersion` — **not producible on the request path at all.**
  `ClassifyRequestDocument` is `{ texts, include_logits }` with `#[serde(deny_unknown_fields)]` and
  carries no `schema_version`; that variant belongs to `ClassifyResponseWire`'s `TryFrom`
  (`classify.rs:559`), and `classify()` builds its result through `ClassifyResponse::new`
  (`classify.rs:722`), never that `TryFrom`. Measured, not inferred: posting
  `{"schema_version":2,"texts":["ok"]}` answers **422** from axum's extractor
  (`Invalid request body. Check that the JSON structure matches…`), so the variant never reaches
  the mapper.

Every error `classify()` CAN return under those pre-conditions (`EncodeFailed`, `HeadFailed`,
`LabelCountMismatch`, `NonFiniteResponse`, `NegativeLatency`, `ProbabilityMassOutOfRange`) already
lands on the `_ => INTERNAL_SERVER_ERROR` wildcard, so deleting the arm changes no HTTP response.
Note this is the SAME structural cause as Survivor B — the transport pre-checks shadow core's
equivalents — but here it makes the mutant genuinely equivalent rather than merely hard to observe.
The arm is still correct to keep: it is the fail-closed mapping if a future edit ever removes a
pre-check.

**Adjusted production score after this plan: 7 caught / 7 non-equivalent viable = 100%**
(21 mutants − 13 unviable = 8 viable, less the 1 equivalent).

**NOT closed by this plan, and not claimed:** 04-11's must-have 4 — the four-crate per-crate
mutation gate with four baselines and an explicitly computed aggregate adjusted score. D-04-11-B
measured ≥ 10 h wall clock for its 890 mutants; this plan ran 25 mutants over 865 s against ONE file
in ONE crate. That gate remains a `human_verification` item, exactly as D-04-11-B left it.

## D-04-22-A — the bashrs inventory: measured exit-code semantics, two proven TOOL false positives, and the triaged repo-wide backlog

Measured by plan 04-22, wave 12, on `gsd/phase-2-contract-gate` at base `9ae41fcaa` (2026-08-16).
Every status below was captured with `cmd > log 2>&1; rc=$?` — never through a pipe, never with
`tee`. CLAUDE.md Verification Discipline rule 1 records two shipped defects of exactly that shape
(#2336, #2360), and this entry is the inventory for a finding of the same class, so committing it
here would have been self-refuting.

### 0. The tool, proven before any number it printed was trusted

CLAUDE.md rule 8 records two cases where a shadowed artifact made edits look effective while
changing nothing. So the binary was proven, not assumed:

| probe | result |
|-------|--------|
| `whence -a bashrs` (zsh) | **exactly one path**: `/Users/guy/.cargo/bin/bashrs` |
| `command -v bashrs` | rc=0, same single path |
| `bashrs --version` | rc=0, **`bashrs 6.66.3`** |

`type -aP bashrs` — the form CLAUDE.md rule 8 suggests — is **not available here**: the login shell
is zsh, whose `type` has no `-P` (`(eval):type:1: bad option: -P`, rc=1). `whence -a` is zsh's
equivalent and was used instead. Recorded rather than silently substituted.

**This contradicts F-07's premise and D-04-11-A's closing note, both of which say bashrs is absent.**
The version matches what the planner measured, so the planner's counts were re-derived rather than
inherited — and they reproduced (see §3).

### 1. Exit-code semantics, established by CONTROL — and the help text is misleading

`bashrs --help` documents a global `--strict` as *"Enable strict mode (fail on warnings)"*, which
implies the default does NOT fail on warnings. **That implication is false.** Throwaway inputs were
constructed under the scratch directory, one per severity state, for BOTH subcommands. They are
different code paths and were measured separately rather than assumed to share semantics.

**`bashrs lint` (shell scripts):**

| # | input | findings | default rc | `--strict` rc |
|---|-------|----------|-----------|--------------|
| 1 | `probe_a.sh` — info-only | 0E 0W 3I | **0** | **0** |
| 2 | `probe_w.sh` — warning-present | 0E 1W 2I | **1** | **1** |
| 3 | `probe_b.sh` — error-present (a REAL unterminated quote) | 1E 0W 0I | **2** | **2** |

**`bashrs make lint` (Makefiles):**

| # | input | findings | default rc | `--strict` rc |
|---|-------|----------|-----------|--------------|
| 4 | `Mk_min.mk` — no findings at all | 0E 0W 0I | **0** | **0** |
| 5 | `Mk_warn.mk` — warning-only | 0E 6W 0I | **1** | **1** |
| 6 | `Mk_err.mk` — error-present (a REAL `local` in a recipe body) | 1E 0W 0I | **2** | **2** |
| 7 | `Makefile` (the real one, 2432 lines) | 1E 33W 0I | **2** | **2** |

Twelve measured cells over the two subcommands. Three conclusions, all load-bearing for the gate
built in Task 2:

1. **The mapping is severity-tiered, identically for both subcommands:** `0` = nothing above info,
   `1` = warnings present, `2` = at least one error. It is NOT pass/fail.
2. **`--strict` is a no-op in all seven states measured.** Warnings already fail without it (rc=1),
   and it does not escalate info to failure. The gate does not use it, and the reason is recorded
   here so a future editor does not add it expecting an effect.
3. **The info-only cell is not constructible for `make lint`.** No info-severity Makefile rule was
   observed to fire — zero infos on the real 2432-line Makefile and on every probe. The row is
   therefore replaced by the strictly-more-informative *no-findings* and *warning-only* rows rather
   than fabricated. Stated explicitly so the missing cell is not read as an omission.

**The finding that decided Task 2's design — rc=2 is AMBIGUOUS:**

| control | rc |
|---------|----|
| `bashrs make lint Makefile` (1 error-severity finding) | **2** |
| `bashrs make lint /nonexistent/Makefile` (*"The specified file was not found"*) | **2** |
| `bashrs make lint Makefile` with bashrs absent from `PATH` | **127** |

**A gate keyed on the exit code alone cannot distinguish "this file has an error" from "the gate was
pointed at nothing".** That is the vacuous-pass failure mode this plan exists to remove, arriving
through the back door. Every gate written in Task 2 therefore parses the report and FAILS
explicitly when no parseable report was produced, rather than trusting `rc`.

A second parsing hazard, measured: a **fully clean** file prints `✓ No issues found in <f>` and
**no `Summary:` line at all** (`scripts/verify-parity.sh`, rc=0). A parser that greps only for
`Summary:` reads nothing there and, if it defaults to zero, cannot tell clean from unparsed. Both
shapes are handled.

### 2. The two MEASURED FALSE POSITIVES — bashrs defects, not code defects

Of the four files Task 2 gates, exactly two carry an error-severity finding, and **both are wrong**.
Each was checked against an independent shell-semantic control rather than accepting the linter's
severity verdict as ground truth. This is the most reusable output of the plan: in both cases,
"fixing" what bashrs flagged would have caused a real regression.

**FP-1 — `scripts/check_apr_bin_pinned.sh:72`, SC1078**

| field | value |
|-------|-------|
| rule | `SC1078: Did you forget to close this double-quoted string?` (error), cols **83-84** |
| flagged text | `ABS_APR='(^\|[[:space:]]"'"'"'=(])(/\|~/\|\$HOME/)[A-Za-z0-9_.$/-]*/apr([[:space:]]"'"'"']\|$)'` — the standard `'"'"'` single-quote-escaping idiom inside the regex |
| refuting control | **`bash -n scripts/check_apr_bin_pinned.sh` → rc=`0`** (empty output). bash's own parser accepts the file. |
| control is non-vacuous | The same control on `probe_b.sh`, which has a genuine unterminated quote, → rc=`2` (`unexpected EOF while looking for matching '"'`). So `bash -n` does fire when a quote really is unterminated; it declines to fire here. |
| avoided consequence | Editing a regex whose own lines 70-71 read *"This regex class has now been gotten wrong four times in this repo; if you change it, re-run the table rather than reading it."* The `apr`-invocation patterns were wrong five times (CLAUDE.md rule 7) and every one was caught by the 12-case must-match/must-not-match table, none by review. Rewriting it to appease a parser bug is the single most dangerous edit available in this plan. |

**FP-2 — `Makefile:2327`, SC2168**

| field | value |
|-------|-------|
| rule | `SC2168: 'local' is only valid in functions` (error), cols **22-28** |
| flagged text | cols 22-28 of `dev-setup: ## Set up local dev environment with sibling repo overrides` are exactly **`local d`** — English prose inside a `##` help comment |
| refuting control | **`make -n dev-setup` → rc=`0`**, and the fully expanded recipe it prints contains **zero** shell `local` (`grep -cE '(^\|[^A-Za-z_])local[[:space:]]' → 0`, grep rc=1). There is no `local` anywhere in the target's actual shell. |
| avoided consequence | A blocking tier3 gate that trips on any future help string containing "local", "declare" or "typeset". A gate satisfied by deleting a word from prose is a gate that will be disabled, and it takes the real gates beside it down with it. |

**Neither is suppressed.** Both are carried as *justified baseline entries* in Task 2's gate, where
each entry must name its refuting control or the gate rejects it. If a later bashrs release fixes
either rule, the entry is **removed from the baseline**, not left as permanent slack — the gate's
own recipe says so.

The other two gated files carry **no** error-severity finding at all (`apr_bin.sh` 0E,
`check_sourced_libs_option_neutral.sh` 0E), so no control was required for them.

### 3. The per-leg inventory — every leg's status, read directly off the command

**Leg A — `bashrs make lint Makefile`: rc=`2`, `1 error(s), 33 warning(s), 0 info(s)`.**

| code | severity | count |
|------|----------|-------|
| MAKE012 (recursive make invocation) | warning | 18 |
| MAKE010 (missing error handling) | warning | 11 |
| MAKE003 (unquoted variable in command) | warning | 2 |
| MAKE018 | warning | 1 |
| MAKE001 (non-deterministic `$(wildcard)`) | warning | 1 |
| **SC2168** | **error** | **1** (FP-2 above) |

Identical to the planner's pre-04-21 tally. **04-21's two-line Makefile edit produced a delta of
zero** — recorded because the plan flagged a small delta as expected, and "expected but absent" is
itself a measurement.

**Leg B — `bashrs lint` over each of `scripts/*.sh`, ONE FILE AT A TIME.** The corpus form
(`bashrs lint scripts/*.sh`) hides which file failed, so 59 separate invocations were made.
**59 scripts, 51 non-zero.** With §1's semantics the "51 red" headline decomposes into something
more useful:

| rc | meaning | files |
|----|---------|-------|
| 0 | nothing above info | **8** |
| 1 | warnings, **no errors** | **29** |
| 2 | **at least one error-severity finding** | **22** |

So the actionable error-severity surface is **22 files**, not 51.

Corpus-wide severity tally, ANSI stripped before counting (the output is coloured and a naive grep
over the raw bytes miscounts): **106 error**, **754 warning**, **1412 info**.

- error: SC1078 (34), SC1100 (30), SC2296 (11), SC1007 (11), SC1035 (7), SC2122 (5), SC2188 (4),
  SC1028 (2), SC2105 (1), SC2104 (1)
- warning: SC2086 (180), PERF002 (133), SC2047 (65), SC2154 (51), SC2164 (30), SC2161 (29),
  REL005 (26), **SEC014 (24)**, **SEC013 (24)**, SC2198 (21), SC2036 (20), SC2297 (15), SC2155 (15),
  SC2031 (12), **SEC020 (4)**, **SEC006 (2)**
- info: SC1012 (216), SC2081 (133), SC2227 (129), PERF003 (104), SC2016 (94), SC2233 (83)

Per-file exit codes, all 59 (`E`/`W`/`I` = error/warning/info counts):

| script | rc | findings | | script | rc | findings |
|--------|----|----------|-|--------|----|----------|
| `apr_bin.sh` | 1 | 0E 14W 33I | | `dispatch-distill-phase-3-gx10.sh` | 2 | 12E 78W 38I |
| `bench.sh` | 0 | 0E 0W 7I | | `dispatch-distill-phase-5-humaneval.sh` | 1 | 0E 14W 37I |
| `benchmark-2x-ollama.sh` | 2 | 10E 33W 152I | | `dispatch-distill-stage-d.sh` | 1 | 0E 35W 55I |
| `benchmark-matrix.sh` | 2 | 2E 52W 141I | | `dispatch-phase5-humaneval-gx10.sh` | 2 | 10E 23W 38I |
| `book-ci-local.sh` | 2 | 4E 14W 22I | | `dispatch-phase6-publish.sh` | 2 | 5E 9W 26I |
| `book-gate.sh` | 2 | 1E 5W 38I | | `dogfood-book.sh` | 1 | 0E 38W 21I |
| `build-wasm-noise.sh` | 1 | 0E 8W 7I | | `dogfood-use.sh` | 2 | 3E 22W 33I |
| `capture_golden_traces.sh` | 1 | 0E 9W 19I | | `extract-book-examples.sh` | 1 | 0E 2W 1I |
| `cascade-drain.sh` | 1 | 0E 10W 10I | | `gen-cli-chapter-stubs.sh` | 2 | 6E 24W 12I |
| `cascade-publish.sh` | 2 | 7E 22W 46I | | `gen-lib-chapter-stubs.sh` | 2 | 3E 13W 5I |
| `check_apr_bin_pinned.sh` | **2** | 1E 17W 35I | | `gen-summary.sh` | 2 | 1E 0W 6I |
| `check_beat_baseline_env.sh` | 2 | 1E 28W 27I | | `gpu_2x_benchmark.sh` | 1 | 0E 22W 45I |
| `check_beat_measurements.sh` | 1 | 0E 11W 11I | | `lint-self-referential-default.sh` | 2 | 12E 0W 2I |
| `check_beats_gated.sh` | 1 | 0E 17W 13I | | `pcu-batch.sh` | 2 | 10E 18W 19I |
| `check_book_cli_parity.sh` | 1 | 0E 5W 8I | | `prepare-release.sh` | 0 | 0E 0W 11I |
| `check_book_example_block.sh` | 1 | 0E 5W 3I | | `qualify-matrix.sh` | 2 | 5E 11W 91I |
| `check_book_examples_compile.sh` | 1 | 0E 2W 3I | | `qwen-story.sh` | 2 | 2E 23W 96I |
| `check_book_examples_executable.sh` | 2 | 3E 49W 44I | | `release.sh` | 0 | 0E 0W 4I |
| `check_book_lib_example_block.sh` | 1 | 0E 5W 3I | | `repro_qwen.sh` | 0 | 0E 0W 3I |
| `check_book_lib_parity.sh` | 1 | 0E 6W 2I | | `stack_release.sh` | 2 | 2E 5W 12I |
| `check_book_linkcheck.sh` | 1 | 0E 2W 2I | | `verify_pmat_116.sh` | 1 | 0E 12W 15I |
| `check_build_rs_paths.sh` | 2 | 2E 9W 4I | | `verify-chat-models.sh` | 1 | 0E 7W 32I |
| `check_format_sovereignty.sh` | 1 | 0E 1W 10I | | `verify-parity.sh` | 0 | 0E 0W 0I † |
| `check_include_files.sh` | 1 | 0E 5W 5I | | `watch-distill-phase-3-gx10.sh` | 0 | 0E 0W 3I |
| `check_msrv.sh` | 1 | 0E 4W 4I | | `cleanup-gx10-runs.sh` | 1 | 0E 8W 11I |
| `check_package_includes.sh` | 1 | 0E 5W 4I | | `crux_bulk_pmat_work.sh` | 2 | 4E 5W 17I |
| `check_pass_grep_anchored.sh` | 1 | 0E 16W 30I | | `cublas_fp8_per_layer_diff.sh` | 1 | 0E 8W 25I |
| `check_publish_safety.sh` | 1 | 0E 7W 28I | | `ci.sh` | 0 | 0E 0W 6I |
| `check_readme_claims.sh` | 0 | 0E 0W 5I | | `check_runner_labels.sh` | 1 | 0E 2W 1I |
| `check_sourced_libs_option_neutral.sh` | 1 | 0E 14W 31I | | | | |

† `verify-parity.sh` is genuinely clean: bashrs prints `✓ No issues found` and emits **no `Summary:`
line**. Recorded because the first pass through the corpus logged it as `NO-SUMMARY`, and a gate
that greps only for `Summary:` would read nothing there. Investigated rather than tidied away.

**Leg C — `bashrs score` on the three guard scripts** (advisory, RECORDED, not a gate):

| script | rc | grade | score |
|--------|----|-------|-------|
| `scripts/apr_bin.sh` | 0 | B- | 7.1 / 10.0 |
| `scripts/check_apr_bin_pinned.sh` | 0 | C- | 5.6 / 10.0 |
| `scripts/check_sourced_libs_option_neutral.sh` | 0 | B | 7.9 / 10.0 |

**Leg D — `bashrs gate`: RUN, and it is VACUOUS. It is NOT recorded as a pass.**

The invocation CLAUDE.md documents, `bashrs gate --strict .`, **does not exist in 6.66.3**:

```
$ bashrs gate --strict .            → rc=2
error: unexpected argument '--strict' found
Usage: bashrs gate [OPTIONS]
```

`gate` takes neither a path nor `--strict` in this version — only `--tier` and `--report`. The valid
form was then run at every tier, and every tier reports success **having enabled nothing**:

```
$ bashrs gate --tier 1              → rc=0
Executing Tier 1 Quality Gates...
Gates enabled:
----------------------------------------
----------------------------------------
✅ Tier 1 Gates Passed!
```

`--tier 2` and `--tier 3` are byte-identical in shape: rc=0, `Gates enabled:` **empty**,
`✅ Tier N Gates Passed!`. **This is a check that reports passing having checked nothing — the exact
defect class F-07 is about, inside the tool this plan was sent to adopt.** It is recorded in the
NOT-RUN list below as `NOT RUN (vacuous)`, never as a green leg, and nothing in this phase is
gated on it. CLAUDE.md's documented `bashrs gate --strict .` invocation is stale for 6.66.3 and
should be corrected when someone next edits that section.

### 3b. The control that reframes the whole backlog: NO script fails `bash -n`

Run across the corpus after the two per-file false positives were found, because if bash's own
parser refuted two of them it was worth asking how far that generalises:

| control | result |
|---------|--------|
| `bash -n` over all **59** scripts | **0 failures.** Every script parses. |
| files carrying at least one SC1078 | **8**, and **all 8** have `bash -n` rc=0 |

**84 of the 106 error-severity findings are parse-class rules** — SC1078 (34), SC1100 (30),
SC1007 (11), SC1035 (7), SC1028 (2) — **and bash accepts every file they appear in.** For a bash
script, bash is the authority on whether it parses. That does not make all 84 provably wrong (a
quote can parse and still nest differently than the author intended, which is why the ABS_APR regex
is protected by its own 12-case table rather than by `bash -n` alone), but it does mean the
"106 error-severity findings" headline is not 106 broken scripts, and a fixer who starts there will
mostly be editing correct code.

The remaining **22 are semantic, not parse-class** — SC2296 (11), SC2122 (5), SC2188 (4),
SC2104 (1), SC2105 (1) — and `bash -n` says nothing about them either way. **They are where a fixer
should start.** Measured precedent for why the distinction matters: SC2168 is also semantic, `bash -n`
does NOT catch it, and the induced probe `@local probe=1` in a Makefile recipe produced a genuine
runtime failure (`/bin/bash: line 0: local: can only be used in a function`). Semantic findings can
be real even when the file parses.

### 4. The triage, grouped by what a fixer would actually DO

Grouped by fix-shape rather than by lint code, because the code is not the unit of work.

**Group A — SECURITY (50 findings). Highest value, and the same defect class this phase closed in
Rust as WR-02.**

| code | n | meaning | representative files |
|------|---|---------|----------------------|
| SEC013 | **24** | hardcoded `/tmp` path, vulnerable to symlink attack | `benchmark-2x-ollama.sh`, `benchmark-matrix.sh`, `dogfood-book.sh` |
| SEC014 | **24** | related insecure-temp handling | `book-gate.sh`, `capture_golden_traces.sh`, `cascade-drain.sh` |
| SEC020 | **4** | (see per-file logs) | `check_book_examples_executable.sh` |
| SEC006 | **2** | unsafe temp file — use `mktemp` for random names | `benchmark-2x-ollama.sh`, `benchmark-matrix.sh` |

**WR-02 connection, stated because it is the argument for doing this group first:** WR-02 was the
Rust-side instance of exactly this — a `.{name}.tmp` file opened with `create(true)`, which follows
a symlink. Phase 4 closed it in Rust and left the identical exposure standing in 59 shell scripts. A
predictable path in a world-writable directory is a symlink-attack surface whatever language opens
it. **Mechanical fix is plausible** for SEC006/SEC013 (`/tmp/fixed-name` → `$(mktemp)`), but each
site needs a read: several of these scripts pass the path between steps or expect a stable name
across runs, and a blind rewrite breaks those. Budget per-site judgement, not sed.

**Group B — SEMANTIC CORRECTNESS (22 findings). Small, and the only error-severity group `bash -n`
does not contradict.**

| code | n | representative files |
|------|---|----------------------|
| SC2296 | 11 | `benchmark-2x-ollama.sh`, `benchmark-matrix.sh`, `check_beat_baseline_env.sh` |
| SC2122 | 5 | (per-file logs) |
| SC2188 | 4 | (per-file logs) |
| SC2104 | 1 | (per-file logs) |
| SC2105 | 1 | (per-file logs) |

**Per-site judgement required, no mechanical fix.** 22 sites is a single focused session.

**Group C — PARSE-CLASS (84 findings). Do NOT start here.**

SC1078 (34), SC1100 (30), SC1007 (11), SC1035 (7), SC1028 (2), across files including
`cascade-publish.sh`, `check_apr_bin_pinned.sh`, `dispatch-distill-phase-3-gx10.sh`,
`book-ci-local.sh`, `check_book_examples_executable.sh`, `lint-self-referential-default.sh`.
**Every file involved passes `bash -n`** (section 3b). At least one — `check_apr_bin_pinned.sh:72` —
is a PROVEN false positive whose "fix" would have been a real regression. Triage each against
`bash -n` and the file's own intent BEFORE editing anything. Expect a high false-positive rate;
report the pattern upstream rather than rewriting the repo around it.

**Group D — QUOTING / WORD-SPLITTING (180 findings, SC2086).** `benchmark-2x-ollama.sh`,
`benchmark-matrix.sh`, `book-ci-local.sh` are the heaviest. **A mechanical fix is exactly what must
NOT happen here.** Adding quotes changes word-splitting behaviour and can silently alter what a
script does — a script relying on splitting an argument list will break silently, and the failure
surfaces in CI, not at edit time. Whoever takes this MUST ship a must-match / must-not-match case
table per CLAUDE.md rule 7. That rule exists because the `apr`-invocation patterns in this repo were
wrong five times and every one was caught by such a table, none by review.

**Group E — PERFORMANCE / STYLE (PERF002 133, PERF003 104, SC1012 216, SC2081 133, SC2227 129,
SC2016 94, SC2233 83, and others).** Advisory. Lowest priority; do not let it dilute Groups A and B.

### 5. Owner and shape

**Owner: a standalone repo-wide shell hygiene ticket, filed against the tooling queue before Phase 5
planning starts, and carried by whoever owns `scripts/` maintenance — NOT a Phase 4 plan and NOT a
Phase 5 plan.** Phase 5 is production encoder calibration; absorbing 59 scripts into it would repeat
the mistake this entry exists to prevent.

Required shape, so it is actionable rather than a wish:

1. **Order: Group A (security) → Group B (semantic) → Group D (quoting, only with a case table) →
   Group C (parse-class, triage-only) → Group E (optional).**
2. **Every group's changes ship a must-match / must-not-match case table (CLAUDE.md rule 7), RUN and
   recorded, not written.** Group D cannot be merged without one.
3. **Every claimed false positive names its independent control and that control's exit code as a
   number.** A false positive asserted without its control is the same defect one level up.
4. **Do not wire `lint-scripts` into a tier until the corpus is green.** 51 of 59 are non-zero today;
   a permanently red blocking gate gets disabled, and the real gates beside it go with it (Ph1 D-26).
   The scoped `bashrs-scoped-lint` added by this plan is the interim: it covers the four files this
   phase's own verification rests on, and it is green.
5. **Adjacent finding, cross-referenced so both are found together:** `D-ITEM-01` in `STATE.md`
   (Phase 2) records vacuous macOS `grep -oP` guards in this same file family. Same directory, same
   class of portability defect, same owner.

### 6. The two bashrs FALSE POSITIVES, as TOOL defects — and the upstream decision

The full five-field records are in section 2 above (file:line, rule id, flagged text, refuting
control with its numeric exit code, avoided consequence). Restated here as the reusable output:

| # | site | rule | control | verdict |
|---|------|------|---------|---------|
| FP-1 | `scripts/check_apr_bin_pinned.sh:72` | SC1078 | `bash -n` rc=**0** (and rc=**2** on a real unterminated quote) | tool defect |
| FP-2 | `Makefile:2327` at base `9ae41fcaa` | SC2168 | `make -n dev-setup` rc=**0**, zero shell `local` in the expanded recipe | tool defect |

**FP-2's line number moved** to 2686 once this plan inserted its gate block above it. Recorded
because the finding is unchanged and only its coordinates moved — which is exactly why nothing in
the gate is keyed on a line number.

**Neither is suppressed and neither is baselined-and-forgotten.** Each is DISCRIMINATED by a control
that is re-run on every gate invocation: SC2168 is discounted only when the flagged columns fall
after a `##` on a non-recipe line, and SC1078 only for a file whose own `bash -n` exits 0. Both
discriminators were proven to still catch the real thing — an induced `@local probe=1` recipe and an
induced unterminated quote both turn the gate RED. If a later bashrs release fixes either rule,
DELETE the discriminator rather than leaving it as permanent slack.

**Upstream reporting: RECOMMENDED, and explicitly NOT DONE by this plan.** Both are cleanly
reproducible from single lines and would be good bug reports (SC2168 parsing a `##` help comment as
shell; SC1078 mis-parsing the `'"'"'` escape idiom). Filing them requires opening issues against a
third-party repository, which is outside this plan's autonomous scope and was not done. Recorded as
a decision rather than left ambiguous, so nobody assumes it was handled.

### 7. The stale premise in D-04-11-A, corrected ADDITIVELY

**2026-08-16.** `## D-04-11-A` (the `setfit-api-boundary` CI-exclusion entry) closes with: *"Note
`bashrs` (which CLAUDE.md mandates over shellcheck) is NOT installed on this host, so any new script
would need linting elsewhere."* **That premise is now FALSE.** bashrs **6.66.3 is installed at
exactly one path**, `/Users/guy/.cargo/bin/bashrs` (section 0). That section is left byte-unchanged;
this is the correction.

Consequence: the clean fix D-04-11-A proposed — extracting the `cargo tree` loop into
`scripts/setfit_api_boundary.sh` so that one script serves both the Make target and a CI step, with
no quoting rewrite — **is now UNBLOCKED on the tooling side.** A new script can be linted, and this
plan's `bashrs-scoped-lint` is the pattern for adding it to a scoped gate.

**It is NOT unblocked on the process side, and this changes nothing about that.** The extraction only
pays off if a CI step invokes the script, which needs a `.github/workflows/ci.yml` edit. CLAUDE.md
places workflow edits outside autonomous scope, so it stays a human decision. **The user's
risk-acceptance recorded in D-04-11-A still stands unaltered:** a Linux-only dependency-closure
regression introduced via `cfg(target_os)` would still not be caught, because the gate still only
runs on developer machines.

### 8. The NOT-RUN set — nothing here may be read as closed

Stated as its own list so a verifier reading only this section cannot mistake any item for done.

1. **NOT RUN — the 51-of-59 script backlog is MEASURED and TRIAGED, but NOT FIXED.** 106
   error-severity, 754 warning, 1412 info findings remain in the tree. Only 4 files are gated
   (section 4, group ordering). No script was modified by this plan: `git diff -- scripts/` is empty.
2. **NOT RUN (vacuous) — `bashrs gate` at every tier.** rc=0 with `Gates enabled:` EMPTY at tiers 1,
   2 and 3 (section 3, Leg D). It reports success having checked nothing. It is NOT a passing leg and
   nothing is gated on it. `bashrs gate --strict .`, the form CLAUDE.md documents, does not exist in
   6.66.3 at all (rc=2, `unexpected argument '--strict'`).
3. **NOT RUN — the info-only cell of the `bashrs make lint` control table.** No info-severity Makefile
   rule was observed to fire, on the real 2432-line Makefile or on any probe, so the cell is not
   constructible. Substituted with no-findings and warning-only rows and recorded as a substitution,
   not silently omitted.
4. **NOT RUN — the per-crate `cargo-mutants` gate (04-11 must-have 4).** D-04-11-B measured **≥ 10 h**
   wall clock for its 890 mutants. Untouched here; still a `human_verification` item.
5. **NOT RUN — SAFE-02's "in CI" clause.** The 16 legs applied at `.github/workflows/ci.yml:378-397`
   have never been executed. Nothing was pushed, no PR was opened, and no file under `.github/` was
   modified by this plan.
6. **NOT RUN — the full `make tier3`.** The two new targets were verified standalone (rc=0 each) and
   their presence in the tier3 recipe confirmed by `make -n tier3`. A full tier3 was not run; it
   carries known standing reds unrelated to this plan (D-04-04-B's 24 `aprender-train` names,
   D-04-08-A's 51 `aprender-serve` failures).
7. **NOT RUN — upstream bug reports for FP-1 and FP-2.** Recommended in section 6, not filed.

**F-10, OPS-01, OPS-02, SC1, SC2, SC3 and the "over a produced artifact" halves of SC4/SC5 remain
OPEN. SAFE-01 and SAFE-02 remain UNCHECKED.** This plan improves the honesty of the gate surface
those requirements are measured on; it does not deliver any of them.
