---
phase: 04-apr-artifact-and-production-parity
plan: 05
subsystem: aprender-train/setfit
tags: [codec, setfit-apr-v1, bijection, closure, typed-errors, apr-03, tdd]
requires:
  - phase: 04-01
    provides: "contracts/setfit-apr-v1.yaml — `doc_bundle_bijection`, both directions, and the EXACT-tolerance closure equation this plan implements"
  - phase: 04-02
    provides: "write_setfit_apr + SetFitArtifactView (20 fields, 1:1 with SetFitBundle) + SetFitArtifactError deriving Debug/Clone/PartialEq, which is what lets CodecError carry it typed"
  - phase: 04-03
    provides: "read_setfit_apr_parts (rungs 1-5, the parse-only door), load_setfit_apr (rungs 1-7), artifact_sha256_hex, SetFitAprParts"
  - phase: 04-13
    provides: "ProvenanceRecord as SetFitBundle field 20 — without it the bijection is not total and closure is unachievable (review B3)"
  - phase: 03-faithful-two-stage-trainer-and-head
    provides: "the sealed SetFitCodec seam, CodecError, run_verify_policy's byte-closure check, Tolerance::EXACT, and the fixture run builders"
provides:
  - "AprCodec — the sealed setfit-apr-v1 codec behind Phase 3's seam; APR_FORMAT_ID"
  - "CodecError::Artifact { format_id, source: SetFitArtifactError } — a core failure reaching the caller TYPED"
  - "pub(crate) mod sealed in verify.rs — the one sanctioned visibility edit"
  - "bundle::f32_to_hex / bundle::hex_to_f32 widened to pub(super), so there is still exactly ONE hex codec"
  - "an APR-capable tiny fixture on the aprender-train side (256 position rows, 48-token tokenizer, vocab_remap: None)"
  - "the measured finding that the phase-3 slice fixture CANNOT carry a setfit-apr-v1 artifact, kept executable"
affects:
  - "04-06 / 04-07 / 04-08 (every adapter that wants train-time bytes now has a codec that produces production-loadable ones)"
  - "04-12 (OPS-01's lifecycle proof: `AprCodec` is the codec, and `round_trip::apr_capable_run`'s shape is the recipe for a run that can be closed to APR)"
  - "04-10 (a gate over aprender-train clippy must carry --no-deps; the counted filters here are setfit::apr_codec:: and ::round_trip)"
  - "04-09 (parity: the artifact hash a run records is core's artifact_sha256_hex over the same bytes — asserted, not assumed)"
tech-stack:
  added: []
  patterns:
    - "the codec is an ADAPTER: every format decision stays in aprender-core, the bundle<->view mapping is all this file owns"
    - "a bijection proven field by field, each assertion labelled `field N: name`, so a defaulted field is NAMED by the failure"
    - "two typed error channels that do not collapse into one: BundleError for the bundle layer, SetFitArtifactError for the core layer"
    - "the in-band negative is a codec that recovers 19 of 20 fields — the defect the review said was asserted and shown nowhere"
    - "test modules as DIRECT children of the module under test, so a counted filter cannot match zero (F-04)"
key-files:
  created:
    - "crates/aprender-train/src/train/setfit/apr_codec.rs (1362 lines)"
  modified:
    - "crates/aprender-train/src/train/setfit/verify.rs (+40 -2: the seal's visibility, the Artifact variant, its Display arm)"
    - "crates/aprender-train/src/train/setfit/bundle.rs (+13 -2: two fn visibilities, doc-only otherwise)"
    - "crates/aprender-train/src/train/setfit/mod.rs (+2: pub mod apr_codec)"
decisions:
  - "CodecError gains an `Artifact` variant rather than stringifying SetFitArtifactError into BundleError::Serialization — the review's HIGH finding, and mutation M4 shows the stringification turns four typed-error tests red"
  - "serialize reads the bundle through its PUBLIC accessors (tokenizer_bytes / named_tensors / head_parts), so the hex limits and diagnostics stay the bundle layer's"
  - "deserialize calls read_setfit_apr_parts, never load_setfit_apr: the rebuild and probe-replay rungs belong to the production loader and the trusted policy, and a codec that replayed probes would be a second verification policy"
  - "the format-id self-check is kept on deserialize even though `decode` also makes it — the two cover different surfaces, exactly as SerdeJsonCodec's comment says; M3 shows removing it turns precisely one test red"
  - "the phase-3 slice fixture cannot carry an APR artifact, so the round-trip run substitutes the ENCODER and HEAD and keeps the dataset, selection, config, evidence and the entire trusted policy real"
metrics:
  duration_seconds: 9000
  tasks_completed: 2
  files_changed: 4
  completed: 2026-08-15
---

# Phase 4 Plan 05: The `setfit-apr-v1` Codec Summary

`AprCodec` is a thin adapter whose twenty-field mapping onto `write_setfit_apr` /
`read_setfit_apr_parts` is proven — not asserted — field by field, closes
byte-canonically at `Tolerance::EXACT` through the unchanged phase-3 policy, produces
bytes the production loader accepts including probe replay, and reports a core failure
as a typed `SetFitArtifactError` rather than a sentence.

## Task Commits

| Task | Deliverable | Commit |
| ---- | ----------- | ------ |
| 1 | `AprCodec`, both bijection directions, `CodecError::Artifact`, the seal's visibility, 10 tests | `ef118629c` |
| 2 | APR-03 through the real `verify_artifact`, 7 tests | `06b48cf32` |

Both commits compile and pass their gates **in isolation** — `ef118629c` was checked
with `cargo fmt --check` (rc=0), scoped clippy (rc=0) and its counted filter (10
passed) on exactly the tree that was committed, before the round-trip module was
added. That is deliberate: 04-13 recorded (F-08) an intermediate commit that did not
build alone, and this plan's task split was arranged so the same thing does not happen
here.

## Test Counts (the plan's verify commands, verbatim, status captured directly)

Every status is `cmd > log 2>&1; rc=$?` — never read through a pipe (CLAUDE.md
verification rule 1). Every command carries `--features setfit` and every count is a
stated non-zero number compared against a stated minimum (F-04).

```
$ CARGO_INCREMENTAL=0 cargo test -p aprender-train --features setfit --lib setfit::apr_codec::
rc=0    17 passed                       (Task 1 criterion: >= 8)

$ CARGO_INCREMENTAL=0 cargo test -p aprender-train --features setfit --lib setfit::apr_codec::round_trip
rc=0     7 passed                       (Task 2 criterion: >= 5)

$ CARGO_INCREMENTAL=0 cargo test -p aprender-train --features setfit --lib setfit::apr_codec::bijection
rc=0    10 passed

$ CARGO_INCREMENTAL=0 cargo test -p aprender-train --features setfit --lib setfit::
rc=0   273 passed, 1 ignored            (baseline before this plan: 256 — exactly +17)

$ CARGO_INCREMENTAL=0 cargo test -p aprender-train --features setfit --lib
rc=101 7882 passed, 24 failed, 15 ignored

$ CARGO_INCREMENTAL=0 cargo clippy -p aprender-train --features setfit --lib --all-targets --no-deps -- -D warnings
rc=0

$ cargo fmt -p aprender-train -- --check
rc=0
```

**The 24 whole-suite failures are the phase-3 known-red baseline, diffed rather than
asserted:**

```
$ diff /tmp/failures-baseline.txt /tmp/failures-now.txt ; DIFF_RC=$?
DIFF_RC=0
```

21 `gpu::` + 3 `prune::snapshot_tests`, the same names as
`.planning/phases/03-faithful-two-stage-trainer-and-head/known-red-baseline.md`. Zero
new failures. Not fixed — out of scope, as the plan directs.

**The `setfit::` baseline was measured on the merge base before any edit** (256 passed,
1 ignored, rc=0), so the +17 is a delta between two measurements rather than a
subtraction from a quoted figure.

## The Bijection, Field By Field

`apr_every_one_of_the_twenty_bundle_fields_is_recovered_by_name` carries twenty
assertions in wire order, each labelled so the failure says WHICH field moved. Measured
on the committed tree:

```
$ grep -oE '"field [0-9]+: [a-z_.]+' crates/aprender-train/src/train/setfit/apr_codec.rs
field 1: schema_version          field 11: root_seed
field 2: format_id               field 12: tensors
field 3: architecture            field 13: head_weights_hex
field 4: tokenizer_bytes_hex     field 14: head_intercepts_hex
field 5: pooling                 field 15: head_n_features
field 6: normalization           field 16: ordered_labels
field 7: l2_epsilon (bit)        field 17: requested_config
field 8: truncation_max_...      field 18: resolved_config
field 9: padding_mode            field 19: evidence
field 10: max_length             field 20: provenance
                                 field 20: provenance.dataset_fingerprint
```

Twenty distinct field numbers, plus one sub-assertion on field 20. Three of them are
worth a sentence each:

- **field 7** is compared by `to_bits()`, not by `==`. It is an epsilon, and `==` on
  floats is exactly the comparison that would not notice a NaN written into the field —
  the same reasoning `check_policy_matches_this_build` already applies to it.
- **fields 8 and 10** are both sequence-length `u32`s and are the pair a mapping is
  most likely to swap. The fixture gives them **different** values (256 and 96) and the
  test asserts `assert_ne!` between the two recovered values, so "they both
  round-tripped" cannot stand in for "neither was substituted for the other". This was
  not theoretical — see mutation M2, where the run-derived bundle (whose two values are
  equal) noticed nothing and only the bijection fixture went red.
- **field 20** is read back with `serde_json::from_value(doc.provenance)` and is never
  recomputed, because it is not a function of the other nineteen. It is the field 04-13
  added, and its recovery is what makes the bijection total.

The whole-struct `assert_eq!(reloaded, original)` lives beside the per-field test on
purpose: the struct comparison is the one a forgotten assertion cannot fool, and the
per-field one is the one that says which field moved.

## Closure and EXACT Tolerance

`serialize(deserialize(bytes)) == bytes` holds, byte for byte, and was run **twice** —
once proves the identity, twice proves it is stable against a writer whose output
depends on how many times it has run. Separately,
`apr_two_serializations_of_one_bundle_are_byte_identical` asserts determinism on its
own, so a failure says which of the two properties broke.

`Tolerance::EXACT` **holds. A2 was not falsified.** Through the real
`verify_artifact`:

| quantity | observed |
| -------- | -------- |
| `tolerance_embedding_abs` | 0.0 |
| `tolerance_probability_abs` | 0.0 |
| `max_embedding_abs_diff` | 0.0 |
| `max_probability_abs_diff` | 0.0 |
| `round_trip_closed` | true |

Both bounds AND both observed maxima are zero, which is what distinguishes "matched
exactly" from "matched inside a tolerance somebody widened".

## The Typed Error Mapping (the review's HIGH finding, closed)

`CodecError` gained a third variant rather than a conversion:

```rust
Artifact { format_id: String, source: aprender::setfit::SetFitArtifactError }
```

`CodecError::Bundle` carries a `BundleError`, and the two enumerate different worlds — a
contracted allocation limit, a container CRC failure, a non-finite payload, an
incomplete tensor set and a probe-replay divergence have no `BundleError` counterparts.
The seam's own rule is that the inner error is preserved rather than rendered to a
string; honouring it with one variant leaves only stringification (which erases the
distinction) or inventing a `BundleError` that never happened (worse — the caller then
matches successfully on a diagnosis nothing produced).

It is constructible because `SetFitArtifactError` derives `Debug + Clone + PartialEq`,
which 04-02 took as an acceptance criterion for exactly this reason. **Verified on the
merged tree before relying on it:** `#[derive(Debug, Clone, PartialEq, Eq)]` +
`#[non_exhaustive]` at `artifact.rs:437-439`.

Both channels are tested and both stay distinct: `apr_a_malformed_tensor_hex_payload_is_a_typed_bundle_error`
asserts a `BundleError::MalformedHexPayload` naming `head_weights`, while three tests
assert `SetFitArtifactError` variants. Every negative assertion is `matches!` on the
VARIANT and its discriminating field. There is no `to_string()` in any of them.

## APR-03, End To End

`round_trip_verify_artifact_mints_the_verified_state_and_records_the_artifact_hash`
drives the shipped `SetFitRun::<HeadFitted>::verify_artifact` — not a reimplementation —
which delegates to `verify::run_verify_policy` at `Tolerance::EXACT`. Nothing in the
policy changed for phase 4; `verify.rs`'s diff is 40 lines across three regions, none
of them a policy step, a tolerance or an ordering.

| APR-03 clause | how it is executed |
| ------------- | ------------------ |
| "training closes the in-memory model" | `close` consumes the encoder and head by value; the caller has no binding afterwards |
| "reloads the written APR" | `decode` -> `AprCodec::deserialize` -> `read_setfit_apr_parts` |
| "through the production core loader" | `round_trip_the_same_bytes_load_through_the_production_core_loader` runs `load_setfit_apr` on the SAME bytes — rungs 1-7, probe replay included — then embeds through the verified model |
| "verifies exact tokenizer/configuration/tensor state" | the twenty-field bijection + the byte-closure check |
| "plus tolerance-bounded outputs" | `compare_probes` at `Tolerance::EXACT`, maxima asserted zero |

**One hash, two witnesses.** The state's recorded `artifact_hash()` (from the trusted
free function `verify::artifact_hash`, which a codec can never reach) is asserted equal
to core's `artifact_sha256_hex` over the same bytes. Two modules, one digest, and a
consumer holding only the file can recompute it.

**Provenance, end to end.** `round_trip_the_recovered_provenance_equals_the_runs_selection_fingerprints`
reads `doc_view().provenance` off the model the PRODUCTION loader returned and compares
all six values against the run's own `Selection` accessors — four 64-character digests
plus the root seed and shots-per-class. That is the witness that 04-13's field 20 is
real rather than merely compiled.

## The Bijection Shown Able To Fail

Four mutations of the production code, each reverted with
`cp /tmp/apr_codec.rs.pristine <file>` immediately afterwards and the green baseline
re-measured (17 passed, rc=0) after every revert.

| # | Mutation | `setfit::apr_codec::` | What went red |
| - | -------- | --------------------- | ------------- |
| — | baseline | 17 passed, rc=0 | — |
| M1 | `bundle_of` DEFAULTS `provenance` instead of recovering it | **11 passed, 6 FAILED, rc=101** | the whole-struct comparison, the per-field test **naming `field 20: provenance`**, byte closure, and three round-trip tests |
| M2 | `view_of` swaps fields 8 and 10 | **14 passed, 3 FAILED, rc=101** | the per-field test **naming `field 8: truncation_max_sequence_length`**, plus the struct comparison and closure |
| M3 | the redundant format-id self-check disabled (`if false && ...`) | **16 passed, 1 FAILED, rc=101** | exactly `apr_a_bundle_declaring_a_foreign_format_id_is_refused_by_the_codecs_own_check` |
| M4 | `artifact_error` stringifies into `CodecError::Bundle` | **13 passed, 4 FAILED, rc=101** | all four typed-error tests |
| — | all reverted | 17 passed, rc=0 | — |

M1's failure message, quoted verbatim, is the property T-04-45 asks for — the defaulted
field is NAMED, not inferred from a struct dump:

```
assertion `left == right` failed: field 20: provenance
  left:  ProvenanceRecord { dataset_fingerprint: "", ... selection_root_seed: 0, shots_per_class: 0 }
  right: ProvenanceRecord { dataset_fingerprint: "dddd...", ... selection_root_seed: 289643248593403923, shots_per_class: 11 }
```

**M2 is the one that earned its keep.** The three tests it turned red are all in
`bijection`; **the round-trip tests stayed green**, because the run-derived bundle's
`max_length` and `truncation_max_sequence_length` are both 256 and the swap is
therefore invisible there. A suite built only on run-derived bundles would have passed
this mutation. That is precisely why the bijection fixture assigns the two fields
different values and asserts `assert_ne!` between them.

**M3 is the redundancy argument, measured.** Removing the codec's own check leaves the
trusted `decode`'s check in place — and exactly one test notices, the one that reaches
`AprCodec::deserialize` directly without going through the policy. Neither check
subsumes the other, which is what `SerdeJsonCodec`'s comment claims and what this now
demonstrates for a second implementor.

**M4 is the review's HIGH finding made concrete.** The stringification the review
warned about compiles, produces a plausible message, and turns four typed-error tests
red.

In addition, the in-band negative is a **production-shaped** version of M1:
`round_trip_a_codec_that_defaults_one_bundle_field_cannot_close` ships a codec whose
`serialize` is honest and whose `deserialize` recovers nineteen fields and invents the
twentieth. The trusted policy refuses it with `ReloadNotFromBytes`, and the SAME run
under the honest codec is asserted to close first, so the refusal is attributable to
the defaulted field and to nothing else.

### The clippy leg was proven to reach the file

Per F-03 the scoped command carries `--no-deps`; without it the run exits 101 on
`aprender-compute`'s pre-existing debt and "no findings in my crate" is
indistinguishable from "my crate was never linted". So the scoped run was falsified
before being believed — a deliberate `let _clippy_probe = format!("{}", "reached");`
was inserted into `artifact_error`:

```
error: useless use of `format!`
   --> crates/aprender-train/src/train/setfit/apr_codec.rs:137:25
   = note: `-D clippy::useless-format` implied by `-D warnings`
rc=101
```

Probe reverted; rc back to 0. The 28 remaining warnings are all in
`crates/aprender-compute/` — `grep -c` for any of this plan's four files in the clippy
log returns **0**. Stated as a scoped-clean result, not a whole-command pass.

## THE FINDING: the phase-3 slice fixture cannot carry a `setfit-apr-v1` artifact

The plan's Task 2 says "a completed tiny-fixture run -> `verify_artifact(AprCodec)`
succeeds". **It does not, and this was measured rather than predicted.** The first
attempt returned:

```
Codec(Artifact { format_id: "setfit-apr-v1", source: ProbeComputation {
    probe: "probe_unicode",
    reason: "SetFitError::VocabOutOfSlice(canonical id 5915 is outside the slice closure)" } })
```

`setfit-apr-v1` requires all six contract-resident probes to be computable from the
model, and the phase-3 slice fixture cannot compute two of them:

| probe | what the slice lacks | source |
| ----- | -------------------- | ------ |
| `probe_unicode` | its `vocab_remap` is a **97-row closure** built for the synthetic fixture corpus; the probe tokenizes to canonical ids outside it | `slice_config.json`: `"vocab": 97`; `vocab_remap.json` |
| `probe_truncation_boundary` | it declares **64 position rows**, and the probe is 64 repeats of a 7-word unit truncated at `MAX_SEQUENCE_LENGTH` = 256, so `max_seq() = min(256, 64) = 64` and the encode is refused with `OversizeInput` | `slice_config.json`: `"positions": 64`; `encoder.rs:914-919` |

**Neither is a production defect.** The pinned MiniLM-L6-v2 carries the full 30522-entry
vocabulary with no remap and 512 position rows, so both probes are computable there.
They are FIXTURE capability gaps — and they are exactly why 04-02 declined to reuse the
slice fixture for the writer and built a self-contained tiny model with
`positions = MAX_SEQUENCE_LENGTH` and a 48-word tokenizer whose every id is in range
(`artifact.rs:2890-2899` states the position reason in as many words).

### Why the plan's assumption could not be repaired inside the typestate

`tune_encoder` gates on `calibration_regime_id(&encoder, &selection, &config)`, whose
first component is `encoder.architecture_fingerprint()`, judged against
`Thresholds::frozen()`. So **no synthetic encoder can reach `SetFitRun<HeadFitted>`
through the shipped transitions**, and the only encoder that can is the one that cannot
carry an artifact. This is a genuine circularity in the fixture estate, not something a
different call order would have avoided.

### What was done instead

`round_trip::apr_capable_run()` takes a genuine calibrated run built through the shipped
doors and substitutes **the encoder and the head, and nothing else**. The dataset, the
selection, the resolved configuration, the stage-one evidence summary, the entire
trusted policy and `Tolerance::EXACT` are all real. The substituted encoder is itself
built through the SHIPPED reload door `SetFitMiniLm::from_bundle_parts` — the same door
the artifact's own rung-6 rebuild uses. The struct literal that assembles the run is
in-crate; the lifecycle seal is against OUT-OF-CRATE minting, and phase 3's own test
modules already assemble bundles directly through `from_run_parts`.

### The finding is EXECUTABLE, not a paragraph

`round_trip_the_phase_three_slice_fixture_cannot_carry_an_apr_artifact` asserts **both**
structural gaps by name — `vocab_remap.is_some()` and `positions < MAX_SEQUENCE_LENGTH` —
and then asserts the typed refusal. Asserting both matters: only one of them fires
first, and a test that recorded only that one would silently narrow the finding to
whichever probe the encoder happens to reach soonest. If the slice fixture ever gains
vocabulary coverage, this test turns red and points at `fixture`'s module docs.

**For 04-12 (OPS-01).** The public-API lifecycle proof needs a run it can close to APR.
`round_trip::apr_capable_run()` is the recipe, and the gap above is the reason a plain
`fx::head_fitted_run(..)` will not do. If OPS-01 wants a lifecycle over the *slice*
fixture specifically, the phase needs a fixture-estate change (a wider remap closure and
256 position rows in `tests/fixtures/setfit/`) that no plan currently owns.

## Deviations from Plan

### Auto-fixed

**1. [Rule 3 — blocking] `bundle.rs` was edited: two `fn` visibilities widened to `pub(super)`**

- **Found during:** Task 1, writing the reverse mapping.
- **Plan text:** "hex fields decoded via bundle.rs's existing helpers ... reuse them, do
  not re-implement hex". `bundle.rs` is **not** in the plan's `files_modified`.
- **Issue:** `f32_to_hex` and `hex_to_f32` are module-private in `bundle.rs`, and
  `apr_codec` is a SIBLING module, not a descendant — so the plan's own instruction is
  not satisfiable without a visibility change. The forward direction needs no helper
  (`tokenizer_bytes()`, `named_tensors()` and `head_parts()` decode through the bundle's
  public accessors), but the REVERSE direction must hex-ENCODE, and there is no public
  door for that.
- **Alternatives rejected:** re-implementing the encoder in `apr_codec.rs` would be two
  copies of the exact fact the closure equation is about — a copy differing in case,
  byte order or subnormal handling would break closure with nothing turning red — and
  the obvious re-implementation (`extend_from_slice` into a scratch `Vec<u8>`, then
  `hex::encode`) reintroduces the `4N`-byte allocation `f32_to_hex`'s own doc comment
  exists to explain away (~90 MB on a full pin).
- **Fix:** `pub(super)` — the tightest visibility that works, narrower than `pub(crate)`
  — with the reason recorded on both functions. **No behaviour changed**: the diff is
  two `fn` keywords and doc comments (`+13 -2`).
- **Commit:** `ef118629c`

**2. [Rule 3 — blocking] the Task 2 fixture: the slice fixture cannot carry an artifact**

Recorded in full under "THE FINDING" above. Summarised as a deviation because it changed
what Task 2 could execute: the plan says "reuse the fixture-run builders, do not build a
run by hand", and what shipped reuses them for everything except the encoder and head.

- **Commit:** `06b48cf32`

**3. [Rule 1 — fixture bug] `layer_norm_eps` had to be `f64::from(1e-12_f32)`, not `1e-12`**

- **Found during:** Task 1, by `apr_the_fixture_encoder_reports_the_fixture_architecture`
  failing.
- **Issue:** `EncoderArchitecture::layer_norm_eps` is an `f64` "widened from the `f32`
  the encoder holds". `f32 -> f64 -> f32` is lossless, but `f64 -> f32 -> f64` is not: a
  declared `1e-12` comes back off the built encoder as `9.999999960041972e-13`. The
  fixture's architecture record therefore described an encoder `fixture::encoder()` does
  not build.
- **Why it mattered even though closure held:** the doc carries the declared `f64`
  verbatim, so the bijection round-tripped either way. The defect was in the FIXTURE's
  self-consistency, and a fixture whose declared architecture and rebuilt architecture
  disagree is one whose later failures point at the wrong place.
- **Fix:** `f64::from(1e-12_f32)`, with the measurement in a comment beside it.
  The coherence test is what caught it, which is why it was written before the
  bijection tests rather than after.
- **Commit:** `ef118629c`

**4. [Rule 1 — bug] `.err().expect(..)` is a clippy error; `let ... else` instead**

- **Found during:** Task 2 verification.
- **Issue:** `clippy::err_expect` fired twice. `expect_err` — its suggestion — requires
  the `Ok` type to be `Debug`, and `SetFitRun` deliberately is not.
- **Fix:** `let Err(err) = ... else { panic!(..) }`, which needs no `Debug` bound. The
  reason is in a comment so the next author does not "fix" it back.
- **Commit:** `06b48cf32`

### Process deviation

**5. TDD RED was captured as targeted falsification, not as a pre-implementation red run**

Both tasks are marked `tdd="true"`. In Rust, writing these tests first against a type
that does not exist yields a COMPILE error, which carries no information about whether
any individual assertion can fail and masks every other test in the module. So the RED
evidence here is the four-mutation transcript above — each turning a *specific named
assertion* red, each reverted, with the green baseline re-measured after every revert —
plus the in-band `ProvenanceDefaultingCodec` negative, which is a permanent RED-capable
test rather than a transient one. Recorded as a deviation rather than reported as a TDD
cycle that was not run in that shape (the same call 04-13 made, for the same reason).

**Honesty note on the first green.** The bijection suite passed 16 of 17 on its first
run after implementation, which CLAUDE.md says to distrust rather than celebrate. The
one failure was deviation 3 above — a real fixture defect the coherence test found — and
the four mutations are what turn "it passed" into "it can fail, and here is which test
notices".

## A Tooling Hazard, Caught In Passing

`head -1021 file > file2` produced a **682-line** file on this host: `head` output is
filtered by the RTK CLI proxy, so the redirect captured the FILTERED render rather than
the file's bytes. This was written into `apr_codec.rs` and reverted within one command;
no committed state was affected. `awk 'NR<=N'` was verified to produce the correct line
count before being used.

**Consequence for later plans in this phase:** do not use `head`/`tail` to produce file
CONTENT here — they are display commands under the proxy, not byte-exact filters. This
is CLAUDE.md rule 8's shadowed-artifact class (a name that does not do what the name
does) applied to a coreutil, and it is exactly the sort of thing that produces a
"mysteriously truncated file" with no error and rc=0.

## Source Assertions (all measured on the committed tree)

| Assertion | Criterion | Observed |
| --------- | --------- | -------- |
| `grep -c "impl SetFitCodec for AprCodec"` | >= 1 | **1** (line 88) |
| `grep -c "setfit-apr-v1"` in apr_codec.rs | present | **18** |
| `grep -c "read_setfit_apr_parts"` in apr_codec.rs | >= 1 | **2** (the import and the call) |
| distinct `"field N: ..."` assertion labels | 20 | **20**, in wire order, + 1 sub-assertion on field 20 |
| `to_string()` inside any typed-error assertion | 0 | **0** — every negative uses `matches!` on the variant |
| `grep -c "serde(skip_serializing_if"` in the setfit dir | 0 | **0** (F-05: the ATTRIBUTE form, not the bare token) |
| `git diff --name-only 302d461ab..HEAD` under `crates/aprender-core/` | 0 | **0** — wave-4 ownership held |
| `git diff --name-only 302d461ab..HEAD` | the 4 declared files | exactly those 4 |
| `git diff --diff-filter=D --name-only 302d461ab..HEAD` | empty | **empty** — no deletions in either commit |
| `verify.rs` diff | seal visibility + variant + Display arm only | **3 hunks, +40 -2**; no policy step, tolerance or ordering touched |
| `git status --short --untracked-files=all` | empty | **empty** |
| `.pv/` side effect | none | none — `pv` was never run (F-02) |

## Authentication Gates

None. Nothing in this plan touches the network, the filesystem or any credential.

## Threat Flags

No NEW surface outside the plan's threat register. The register's assignments, as
shipped:

| Threat ID | Mitigation as shipped |
| --------- | --------------------- |
| T-04-14 | BOTH checks kept: the trusted `decode`'s and the codec's own. M3 shows exactly one test notices the second one's removal, so the redundancy is load-bearing rather than decorative |
| T-04-15 | the codec computes no digest. Measured precisely rather than loosely: `Sha256` appears at three lines in `apr_codec.rs` and **all three are inside `#[cfg(test)] mod fixture`** (the import at :408, `sha256_hex` at :549, and its one caller at :576, which computes the fixture tokenizer's own digest). The production half (lines 1-355) mentions `sha256` twice — once in a doc comment and once as `artifact_sha256: _`, where `SetFitAprParts`'s digest is explicitly DROPPED on the way into the bundle rather than carried. The hash a run records stays `verify::artifact_hash`, a trusted free function the trait cannot reach |
| T-04-16 | 04-02's cross-process determinism, plus this plan's double-closure test AND a separate two-writes-are-identical test so the two properties fail separately |
| T-04-45 | the 20-field per-field bijection test; M1 shows a defaulted field is NAMED, and `ProvenanceDefaultingCodec` keeps a permanent RED-capable version of the same defect |
| T-04-SC | zero new packages |

## Known Stubs

None. Every declared function is implemented and exercised. Nothing is hardcoded,
defaulted or placeheld on the production path — the one place a default appears is
inside `ProvenanceDefaultingCodec`, a `#[cfg(test)]` negative whose entire purpose is to
be refused.

## Notes for Later Plans

- **04-12 (OPS-01).** Use `AprCodec::new()` as the codec and
  `round_trip::apr_capable_run()`'s recipe for the run. A plain
  `fx::head_fitted_run(..)` cannot be closed to APR — see THE FINDING.
- **04-06 / 04-07 / 04-08.** `AprCodec::serialize` produces bytes `load_setfit_apr`
  accepts, asserted. Adapters should still read through
  `read_setfit_apr_bytes_bounded` (04-03's rule); the codec takes a `&[u8]` and has no
  opinion about where it came from.
- **04-10 (gates).** The counted filters are `setfit::apr_codec::` (17) and
  `setfit::apr_codec::round_trip` (7), both requiring `--features setfit`. The clippy
  leg needs `--no-deps` (F-03). A `skip_serializing_if` gate must match the ATTRIBUTE
  form (F-05); the bare token count in `apr_codec.rs` is 0 either way.
- **04-11 (requirements audit).** APR-03 is executable and green. The
  `doc_bundle_bijection` equation now binds to real code in `apr_codec.rs`
  (`view_of` / `bundle_of`); `contracts/aprender/binding.yaml` is not in this plan's
  `files_modified` and is shared across concurrent agents, so no status was flipped —
  flagged explicitly so its absence reads as a decision rather than an omission (F-01).
- **The fixture estate.** `crates/aprender-train/src/train/setfit/apr_codec.rs`'s
  `mod fixture` is the aprender-train twin of `artifact.rs`'s `mod fixture`. They are
  deliberately the same recipe at the same dimensions; if one moves, the other should.
  Neither is derived from the other, because core cannot name `aprender-train`'s types
  and `aprender-train` cannot reach core's `#[cfg(test)]` module.

## Self-Check: PASSED

Files claimed, checked on disk:

```
FOUND: crates/aprender-train/src/train/setfit/apr_codec.rs   (1362 lines)
FOUND: crates/aprender-train/src/train/setfit/verify.rs      (+40 -2)
FOUND: crates/aprender-train/src/train/setfit/bundle.rs      (+13 -2)
FOUND: crates/aprender-train/src/train/setfit/mod.rs         (+2)
```

Commits claimed, checked in the log:

```
FOUND: 06b48cf32 test(04-05): prove APR-03 end to end through the trusted verify policy
FOUND: ef118629c feat(04-05): AprCodec — the setfit-apr-v1 adapter with a proven bijection (APR-03)
```

`git diff --diff-filter=D --name-only 302d461ab..HEAD` is **empty** — no deletions.
`git status --short --untracked-files=all` is **empty** — nothing generated and left
untracked. No file outside `crates/aprender-train/src/train/setfit/` was touched;
`STATE.md` and `ROADMAP.md` were not modified (the orchestrator owns them), and nothing
under `crates/aprender-core/` was touched (the wave-4 parallelism contract with 04-04).
