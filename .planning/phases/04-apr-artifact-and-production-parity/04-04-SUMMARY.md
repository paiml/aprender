---
phase: 04-apr-artifact-and-production-parity
plan: 04
subsystem: setfit-classification
tags: [OPS-04, OPS-06, D-08, D-12, review-B6, review-M1, review-M2]
requires:
  - "04-03: VerifiedSetFitModel typestate + fail-closed load ladder"
  - "04-02: write_setfit_apr + the artifact fixture view"
provides:
  - "ClassifyResponse / ClassifyResult / ClassifyRequestDocument — the ONE versioned envelope core owns"
  - "VerifiedSetFitModel::classify — the single classification path"
  - "ExecutionBackend + SetFitMiniLm::encode_texts_traced — execution-derived backend identity"
  - "MultinomialLogisticRegression::predict_logits — the single logit implementation"
affects:
  - "04-07 (CLI): serializes ClassifyResponse, parses ClassifyRequestDocument, zero surface-local types"
  - "04-08 (HTTP): same two types over the wire"
  - "04-09 (parity harness): builds its skewed in-band negative through the public validating constructors"
tech-stack:
  added: []
  patterns:
    - "serde(into/try_from) over a PRIVATE wire struct — the SetFitTrainConfig precedent, so Deserialize cannot bypass validation"
    - "identity returned BY the operation, never read from capability detection"
    - "source assertions scan a production_source() slice, never their own needles"
key-files:
  created:
    - crates/aprender-core/src/setfit/classify.rs
  modified:
    - crates/aprender-core/src/setfit/encoder.rs
    - crates/aprender-core/src/setfit/mod.rs
    - crates/aprender-core/src/setfit/artifact.rs
    - crates/aprender-core/src/classification/multinomial.rs
decisions:
  - "\"logits absent\" is an explicit JSON null, not a missing key — skip_serializing_if is proscribed"
  - "the golden fixture uses a FIXTURE backend value so the kernel literal lives only in encoder.rs"
  - "predict_proba refactored onto a new predict_logits rather than adding a third copy of the logit loop"
metrics:
  duration: ~2h15m
  completed: 2026-08-15
---

# Phase 4 Plan 04: The Classification Envelope and Execution-Derived Backend Identity — Summary

`VerifiedSetFitModel::classify` plus the one core-owned `ClassifyResponse`/`ClassifyRequestDocument`
pair, with the backend identity produced by the encode invocation that ran rather than by a
CPU-feature probe — the review-B6 disagreement resolved in favour of Codex's reading, enforced by a
grep gate at zero and by a mutation that turned it red.

## What shipped

| Task | Commit | What |
|------|--------|------|
| 1 | `b1d1055fd` | `classify.rs`: the D-08 request/response pair, private fields, validating constructors, both serde directions routed through private wire structs, committed golden |
| 2 | `261b6d6fa` | `ExecutionBackend` in `encoder.rs`, `encode_with_backend`, `encode_texts_traced` in `mod.rs`, the B6 gates |
| 3 | `62bca9760` | `VerifiedSetFitModel::classify`, `predict_logits`, the `classify_path` suite |

## Test counts per filter (all with `--features setfit`, all non-zero)

| Filter | Passed | Required |
|--------|--------|----------|
| `setfit::classify::envelope` | **26** | >= 6 |
| `setfit::classify::backend` | **7** | >= 3 |
| `setfit::classify::classify_path` | **16** | — |
| `setfit::classify::` | **49** | >= 15 |
| `setfit::` | **236** (was 187 at base) | green |
| `-p aprender-core --features setfit --lib` | **14419** passed, 2 ignored | green |
| `--features conformance-fixtures --test setfit_conformance` | **27** passed, 1 ignored | unchanged |
| `-p aprender-train --features setfit --lib setfit::` | **256** passed | green |

**F-04 reproduced live, and it matters here:** without the feature flag,
`cargo test -p aprender-core --lib setfit::classify::` returns **`0 passed, 14185 filtered out`,
rc=0**. Every gate that names this module must carry `--features setfit` *and* assert a non-zero
minimum, or it is a vacuous green.

## The B6 evidence (backend identity)

**Observed identity on this host:** `cpu:setfit-core:autograd-trueno-matmul`.

It is not asserted against a literal copied into the test. The test calls
`encode_texts_traced`, takes the returned identity, and asserts
`contracts/setfit-apr-v1.yaml` contains `v1 = "<that identity>"` — so the value is pinned to the
contract itself, and a copy cannot drift from it silently.

**Gate 1 — no capability-detection symbol anywhere in the setfit surface:**

```
$ grep -rn "select_backend\|Backend::AVX\|detect_x86_backend\|detect_arm_backend" \
      crates/aprender-core/src/setfit/
$ echo $?
1        # zero matches
```

**Gate 2 — the kernel literal lives only in `encoder.rs`:**

```
$ grep -rc "autograd-trueno-matmul" crates/aprender-core/src/setfit/
crates/aprender-core/src/setfit/encoder.rs:3      <-- the constant + 2 doc references
crates/aprender-core/src/setfit/classify.rs:0     <-- the identity arrives as a VALUE
crates/aprender-core/src/setfit/model_tests.rs:0
crates/aprender-core/src/setfit/artifact.rs:0
crates/aprender-core/src/setfit/error.rs:0
crates/aprender-core/src/setfit/import.rs:0
crates/aprender-core/src/setfit/encoder_tests.rs:0
crates/aprender-core/src/setfit/tokenizer_tests.rs:0
crates/aprender-core/src/setfit/mod.rs:0
crates/aprender-core/src/setfit/dropout_rng.rs:0
crates/aprender-core/src/setfit/loss_tests.rs:0
crates/aprender-core/src/setfit/loss.rs:0
crates/aprender-core/src/setfit/tokenizer.rs:0
crates/aprender-core/src/setfit/import_tests.rs:0
```

Both gates also run as tests (`the_setfit_surface_names_no_capability_detection_symbol`,
`the_kernel_literal_lives_only_in_encoder_rs`), over a `read_dir` of the directory rather than a
hardcoded file list — a hardcoded list goes stale the moment a file is added, and a file added later
is exactly where such a symbol would arrive unnoticed. The scan asserts it saw >= 10 files, so a
path that resolved to nothing is loud.

**`encode` is untouched.** `git diff` on `encoder.rs`: **110 insertions, 0 deletions**, and
`git diff ... | grep -E "^[-+].*fn encode"` produces nothing — `encode_with_backend` was added
beside `encode`, not carved out of it. The behavioural half is asserted too:
`the_traced_and_untraced_encode_paths_agree_elementwise` compares `encode_texts` against
`encode_texts_traced` bit for bit.

## The v1 limitation, recorded rather than overstated

`autograd-trueno-matmul` names the kernel ENTRY POINT the encoder invoked. It is a true statement
about what the encoder called and an **incomplete** one about what silicon executed: trueno exposes
no per-dispatch execution report, and `trueno::Matrix::matmul` selects among `matmul_naive`,
`gemm_blis_parallel` and a GPU path **by SIZE**. So an AVX2 detection result is fully consistent
with a scalar execution of a small batch, which is why detection may never appear in this field.
Upgrading to per-dispatch reporting needs a trueno API that does not exist; it is a deferred item,
not a Phase 4 deliverable. This is documented in full at `ExecutionBackend`'s definition, citing
CLAUDE.md Verification Discipline rule 2.

## The committed serialization golden

```json
{"schema_version":1,"artifact_sha256":"9f2c7a1d4e8b60315a7c9e0d2f4b6813a5c7e9f1b3d50729468a0c2e4f6a8b1d","backend":"cpu:setfit-core:fixture-kernel","latency_ms":0.0,"results":[{"label":"positive","probabilities":[0.25,0.75],"logits":null,"margin":0.5,"token_count":7,"truncated":false},{"label":"negative","probabilities":[0.875,0.125],"logits":null,"margin":0.75,"token_count":5,"truncated":false}]}
```

Compared byte-for-byte against a committed literal, not against a re-serialization of the same
struct (which would prove only that serde is deterministic).

The `backend` value here is a deliberate **fixture** value, not the real v1 identity. This golden
pins the SCHEMA — field names, declaration order, exact bytes. Writing the real identity here would
have put the kernel literal in `classify.rs` and broken Gate 2 above; the identity must arrive as a
value the encode call returns, never as a string this module knows how to spell.

`"logits":null` is deliberate: **"absent" is an explicit null, not a missing key.** A missing key is
invisible to a reviewer's diff and to a null-walking guard; an explicit null is loud to both, and
`skip_serializing_if` is already proscribed on the adjacent artifact sub-documents.

## Deviations from Plan

### 1. [Rule 3 — Blocking] `artifact.rs` gained `pub(crate)` accessors (undeclared file)

- **Found during:** Task 3
- **Issue:** `VerifiedSetFitModel`'s fields are private to the `artifact` module. `classify` lives in
  the sibling `classify` module, so `impl VerifiedSetFitModel { pub fn classify }` **could not be
  written at all** — Rust privacy is module-based, and the public accessors 04-03 shipped
  (`artifact_sha256`, `ordered_labels`, `doc_view`, `embed`) expose neither the encoder nor the head.
  `embed` additionally routes through the UNTRACED `encode_texts`, so it could not have supplied the
  D-12 identity even if it had returned the right shape.
- **Fix:** added `pub(crate) fn model()` and `pub(crate) fn head()` — READ borrows, no construction,
  no mutation. The D-08 seal and the typestate's private constructor are untouched.
- **Also:** `mod fixture` and `fixture_view_full_pin_shape` widened from `pub(super)` to
  `pub(crate)` (both `#[cfg(test)]`), so the classify suites run against the SAME fixture the writer
  and loader suites use. A second fixture next door would have been a second definition of "the
  artifact shape under test", free to drift.
- **Commit:** `261b6d6fa` (fixture visibility), `62bca9760` (accessors)

### 2. [Rule 2 — Missing critical functionality] `multinomial.rs` gained `predict_logits` (undeclared file)

- **Found during:** Task 3
- **Issue:** the response must carry logits *and* probabilities. `predict_proba` returns only
  probabilities and does not expose the logits it computes internally. The logit accumulation loop
  already exists in two places (`predict_proba`, and `artifact.rs::replay_logits` for probe replay);
  writing a third copy inside `classify` would violate OPS-03 and would let the two reported vectors
  drift out of agreement.
- **Fix:** extracted `predict_logits`, and `predict_proba` is now defined as exactly that plus its
  existing softmax. **Accumulation order is unchanged** (intercept first, then `j` ascending) — that
  order is what the artifact writer recorded its probe logits in, so it is load-bearing.
  `classify` calls `predict_logits` once and reuses the head's own `softmax_into`, so the reported
  logits and probabilities cannot disagree. `probabilities_match_the_heads_predict_proba_exactly`
  asserts the equivalence rather than arguing for it.
- **Evidence it is behaviour-preserving:** `aprender-core --lib` 14419 passed;
  `aprender-train --lib setfit::` 256 passed, including the probe-replay comparisons that check head
  logits and probabilities against recorded artifact values.
- **Commit:** `62bca9760`

### 3. [Rule 2] Three error variants beyond the plan's list

`UnsupportedSchemaVersion`, `EncodeFailed`, `HeadFailed` were added to `ClassifyError`. The first is
required for enforceable deserialization (a v2 payload silently parsing as v1 is how a renamed field
becomes a missing field nobody notices); the latter two are required because `classify` calls two
fallible compute steps that the plan's five variants cannot describe. The enum is `#[non_exhaustive]`
and every variant is small — no boxed payload needed, so 04-03's `result_large_err` hazard does not
recur.

`ProbabilityMassOutOfRange` carries `{ mass }` and not `{ row, mass }`: a single `ClassifyResult`
has no row index, and reporting `row: 0` from a value that might be row 5 would have been a lie.

### 4. Test-module layout (plan-mandated, worth flagging)

The plan fixes `mod envelope` / `mod backend` / `mod classify_path` **inside `classify.rs`**. That
had to be honoured literally: a `#[path = "classify_tests.rs"] mod classify_tests;` wrapper — the
idiom every other file in this directory uses — inserts a path segment, and all three filters would
have selected **zero** tests while exiting 0. That is precisely the CR-02 vacuity that bit 04-13's
`bundle_nullable`. `classify.rs` is therefore 1921 lines with its tests inline.

## Findings for the orchestrator notes

### F-05 generalizes, and it bit twice in this plan — once from the plan's own instructions

Note F-05 records that a `skip_serializing_if` gate must match `serde(skip_serializing_if`, not the
bare token, because the bare token appears in the documentation explaining the prohibition. **The
general rule is: a source assertion must not scan its own needle.** Both instances here were
observed, not anticipated:

1. `logits_key_is_present_and_null_when_not_requested` asserted
   `!source.contains("skip_serializing_if")` and turned RED on its own doc comment. Fixed by
   matching the attribute form.
2. **The plan instructed** the `ExecutionBackend` doc comment to say *"reporting
   `trueno::Backend::AVX2`-style detection here is FORBIDDEN"*, **and** carried a grep gate
   requiring zero matches for `Backend::AVX` across the whole directory. The two instructions
   contradict each other: writing the doc as specified turned the plan's own gate red. Resolved by
   naming those symbols descriptively in prose ("trueno's `Backend` enum, its AVX2 / NEON variants")
   and recording *in the doc comment itself* why they are never spelled literally.

The structural defences now in place, worth copying:
- `production_source()` returns this file's source cut at the test-module banner, so no source
  assertion can scan its own text.
- Needles that a scan must not match in itself are assembled with `concat!("select_", "backend")` —
  compile-time concatenation, so the comparison is against the whole symbol while the file contains
  only fragments.

### The `--no-deps` clippy rescue does NOT extend to `aprender-core`

See `deferred-items.md` D-04-04-A. `cargo clippy -p aprender-core --features setfit --lib --no-deps
-- -D warnings` exits **101** on one pre-existing error of its own
(`demo/reliable/performance.rs:126`, unreachable expression), with the 20 `aprender-compute` entries
correctly demoted to warnings. F-03's fix works; the crate simply has debt `--no-deps` cannot hide.

## Verification: guards proven able to fail, not assumed

Every load-bearing gate in this plan was mutated and observed RED, then reverted.

| Mutation | Gate | Result |
|----------|------|--------|
| RED phase: envelope types with no validation | `setfit::classify::envelope` | **13 failed / 8 passed** |
| RED phase: identity reported as `cpu:setfit-core:avx2` | `the_identity_carries_no_simd_capability_token` | **RED** — the review-B6 failure mode itself |
| RED phase: `pub fn new` on `ExecutionBackend` | `execution_backend_has_no_public_constructor_or_setter` | **RED** |
| attention-mask count `+ 1` | `a_truncated_text_reports_..._pinned_token_count` | **RED** on `token_count` |
| `truncated` forced `false` | same test | **RED** on `truncated` |
| `backend.identity()` replaced by a literal | `the_response_backend_equals_the_identity_encode_texts_traced_returns` | **RED** |
| `format!("clippy probe")` planted in `classify.rs` | scoped clippy filter | **0 -> 1** finding under `setfit/` |

The scoped clippy result at final state: **zero** diagnostics matching `setfit/` or
`classification/multinomial` (`rc=101` comes solely from the pre-existing `demo/` error above).

Boundary checks rather than one-sided ones: 257 texts are refused **and** 256 accepted, so the batch
bound is shown off-by-none; mass `1.0` and mass `1.0 ± 5e-7` are accepted while mass `0.5` is
refused; a zero latency is asserted **legal** so no future gate is tempted to require `> 0`.

## File ownership (wave-4 contract with 04-05)

```
$ git diff --name-only 302d461ab..HEAD
.planning/phases/04-apr-artifact-and-production-parity/deferred-items.md
crates/aprender-core/src/classification/multinomial.rs
crates/aprender-core/src/setfit/artifact.rs
crates/aprender-core/src/setfit/classify.rs
crates/aprender-core/src/setfit/encoder.rs
crates/aprender-core/src/setfit/mod.rs
```

Zero `crates/aprender-train/` paths. 04-05 owns only `aprender-train`, so the overlap is empty,
including for the two undeclared files above.

## Known Stubs

None. Every field `classify` reports is computed from the model and the batch: `token_count` from
the encoded batch's attention mask, `truncated` from its truncation facts, `backend` from the encode
invocation, `artifact_sha256` from the verified model, `latency_ms` from a real `Instant`.

## Threat Flags

None. No new network endpoint, auth path, file-access pattern or schema at a trust boundary beyond
those the plan's `<threat_model>` already registers (T-04-11 batch bound, T-04-12 backend
misreporting, T-04-13 NaN-through-JSON — all mitigated as specified).

## Notes for 04-07 / 04-08 / 04-09

- Serialize `ClassifyResponse` and parse `ClassifyRequestDocument` directly. Define no surface-local
  DTO: D-08 field parity is by construction only if there is exactly one type.
- `ClassifyRequestDocument` is `deny_unknown_fields` with `include_logits` defaulting to `false`, so
  `{"texts":["…"]}` is a complete, legal body.
- Core enforces `EmptyInput` and `BatchTooLarge` **before** tokenizing. 04-08 still owes the
  `max_request_body_bytes = 1048576` bound at the transport layer — core cannot see the body size.
- 04-09's skewed in-band negative: `ClassifyResponse::new` and `ClassifyResult::new` are public and
  accept any FINITE value, so perturbing one probability by 10x the tolerance is a legitimate
  construction. Note that `ClassifyResult::new` enforces mass ≈ 1, so a skew must be compensated
  within the row (or applied to `margin`/`logits`, which carry no mass constraint).
- Exclude `latency_ms` from every cross-surface comparison, and never assert it is `> 0`.

## Self-Check: PASSED

- `crates/aprender-core/src/setfit/classify.rs` — FOUND (1921 lines, min_lines 200)
- contains `pub struct ClassifyResponse` — FOUND
- contains `ExecutionBackend` link (`crate::setfit::encoder::ExecutionBackend::identity`) — FOUND
- contains `impl crate::setfit::artifact::VerifiedSetFitModel` — FOUND
- commit `b1d1055fd` — FOUND
- commit `261b6d6fa` — FOUND
- commit `62bca9760` — FOUND
