---
phase: 04-apr-artifact-and-production-parity
plan: 02
subsystem: aprender-core/setfit
tags: [writer, apr-v2, setfit, determinism, null-guard, probes, tdd]
requires:
  - "contracts/setfit-apr-v1.yaml (04-01) — the storage map, doc field list, allowlist, probe strings"
provides:
  - "aprender::setfit::artifact::write_setfit_apr — the pure deterministic view -> APR v2 bytes function"
  - "SetFitArtifactView — 20 fields, 1:1 with SetFitBundle, so 04-05's codec can fill it without inventing anything"
  - "SetFitArtifactError — Debug + Clone + PartialEq + non_exhaustive, embeddable in 04-05's CodecError as a typed source"
  - "artifact_sha256_hex — the ONE hashing path for artifact identity"
  - "NULLABLE_PATH_ALLOWLIST (4) + WALKED_SUBDOCUMENTS (5) + first_unallowed_null_path"
  - "build_hf_name_map / canonical_name_for_hf / expected_tensor_names — the architecture-derived name table"
  - "SETFIT_ARTIFACT_DOC_FIELDS — the contract's 16-field list as a checkable constant"
  - "probe_inputs() + PROBE_IDS — the six contract-resident synthetic probes"
  - "fixture_view_full_pin_shape() — the DEFAULT test fixture, in the production nullability shape"
  - "a committed golden artifact hash for the full-pin fixture"
affects:
  - "04-03 (loader reuses expected_tensor_names, the error enum, the doc field list)"
  - "04-05 (codec populates SetFitArtifactView and maps SetFitArtifactError as a typed source)"
  - "04-13 (the allowlist completeness gate asserts against NULLABLE_PATH_ALLOWLIST)"
  - "04-09 / 04-12 / 04-15 (reuse fixture_view_full_pin_shape's recipe)"
tech-stack:
  added: []
  patterns:
    - "contract-resident constants read from one place, never restated"
    - "allowlisted null scan over opaque sub-documents (evidence.rs CR-03 precedent, widened by exception set)"
    - "bit-pattern hex for every stored float, byte order identical to bundle.rs f32_to_hex"
    - "cross-process determinism proof via current_exe + Output.status (never a pipe)"
    - "test modules as DIRECT children of the module under test, so counted filters actually match"
key-files:
  created:
    - "crates/aprender-core/src/setfit/artifact.rs (2457 lines)"
  modified:
    - "crates/aprender-core/src/setfit/mod.rs (module registration + re-exports)"
decisions:
  - "The view carries the TYPED EncoderArchitecture only; the architecture sub-document is derived with to_value inside the writer. One fact, one copy."
  - "The three test modules are DIRECT children of `artifact`, not nested under `tests` — a nested layout makes both counted filters match zero tests, which exits 0 (CR-02)."
  - "The fixture ships a self-contained 48-entry WordPiece tokenizer rather than reading tests/fixtures/setfit/tokenizer.json, because that path honours APRENDER_SETFIT_FIXTURES and would make the golden hash environment-dependent."
  - "NonFiniteValue.path names the FIRST offender (contract postcondition); the walk still collects all of them and exposes them to callers and tests."
  - "Public API named after contracts/aprender/binding.yaml's declared function names, now, before 04-03/04-05 depend on the symbols."
metrics:
  duration_seconds: 3400
  tasks_completed: 2
  files_changed: 2
  completed: 2026-08-15
---

# Phase 4 Plan 02: The setfit-apr-v1 Writer Summary

`write_setfit_apr` is a pure function of one view to APR v2 bytes — architecture-derived canonical
tensor names, the classifier head as two named F32 tensors, the raw tokenizer as a U8 blob, one
custom metadata key holding the contract's 16-field document, and six probes recomputed from the
view's own parts — proven byte-identical across two processes and pinned to a committed golden hash,
with the null guard shown to accept the production shape and to refuse a null in all five walked
sub-documents.

## What Shipped

| Task | Deliverable | Commit |
| ---- | ----------- | ------ |
| RED (1+2) | 39 failing tests, the view/error types, the constants, the name table, the walk | `9e986c034` |
| 1 | `write_setfit_apr` + the seven-rung order + `build_artifact_doc` + `write_container` | `22e6a2cd2` |
| 2 | the golden hash, measured and blessed | `28b488382` |
| 1+2 | binding-registry name alignment + `first_unallowed_null_path` / `canonical_name_for_hf` | `5f982ed6e` |

## Test Counts (the plan's verify commands, verbatim, status captured directly)

```
$ CARGO_INCREMENTAL=0 cargo test -p aprender-core --features setfit --lib setfit::artifact::
rc=0    41 passed, 0 failed        (criterion: >= 15)

$ CARGO_INCREMENTAL=0 cargo test -p aprender-core --features setfit --lib setfit::artifact::nullable
rc=0    12 passed, 0 failed        (criterion: >= 6)

$ CARGO_INCREMENTAL=0 cargo test -p aprender-core --features setfit --lib setfit::artifact::determinism
rc=0     7 passed, 0 failed        (criterion: >= 5)

$ CARGO_INCREMENTAL=0 cargo test -p aprender-core --features setfit --lib
rc=0   14328 passed, 2 ignored     (whole-crate regression check)
```

**These counts are non-vacuous, and that was measured rather than asserted** — see the falsification
transcript below. A filter matching zero tests exits 0; both counted filters were shown to match a
non-zero set AND to go red under a mutation of the thing they claim to guard.

### The `setfit::artifact::nullable` breakdown (12)

| Kind | Count | Tests |
| ---- | ----- | ----- |
| ACCEPT | 4 | `accept_the_production_full_pin_shape_whose_vocab_remap_is_none`, `accept_a_view_whose_pair_config_budget_and_hard_cap_are_both_none`, `accept_a_view_whose_evidence_epsilon_used_is_none`, `the_full_pin_fixture_emits_exactly_the_four_allowlisted_nulls_and_no_others` |
| REJECT | 4 | `reject_a_null_inside_provenance_and_name_the_path`, `reject_a_null_at_resolved_config_resolved_device_and_name_the_path`, `reject_a_non_finite_f64_smuggled_into_evidence_as_a_null`, `reject_a_null_at_architecture_hidden_act_and_name_the_path` |
| Structural | 4 | `the_allowlist_constant_is_exactly_the_contracts_four_paths`, `the_walk_covers_all_five_sub_documents`, `the_walk_collects_every_offender_and_the_refusal_names_the_first`, `the_walk_does_not_cover_the_containers_typed_metadata_nulls` |

Criterion asked for >= 3 accept and >= 3 reject, one of which injects inside `provenance`. Shipped
4 and 4, with the provenance reject going end-to-end through `write_setfit_apr`.

## The Falsification Transcript (required: the guard must be shown able to fail)

Two mutations, each reverted with `git checkout -- <one named file>` immediately afterwards.

| Mutation | `setfit::artifact::nullable` | `setfit::artifact::` | What went red |
| -------- | ---------------------------- | -------------------- | ------------- |
| baseline | 12 passed, rc=0 | 41 passed, rc=0 | — |
| `WALKED_SUBDOCUMENTS`: `"provenance"` -> a duplicate `"evidence"` | **9 passed, 2 FAILED, rc=101** | — | `reject_a_null_inside_provenance_and_name_the_path`, `the_walk_covers_all_five_sub_documents` |
| `NULLABLE_PATH_ALLOWLIST`: `"architecture.vocab_remap"` -> a bogus path | **1 passed, 10 FAILED, rc=101** | **11 passed, 28 FAILED, rc=101** | everything that writes the production shape, all with `NonFiniteValue { path: "architecture.vocab_remap" }` |
| both reverted | 12 passed, rc=0 | 41 passed, rc=0 | — |

The second row is the checker blocker made concrete. With `architecture.vocab_remap` out of the
allowlist the writer **refuses 28 of 39 tests**, every one of them a full-pin artifact — which is
exactly what would have shipped had the fixture used the slice shape, because the slice shape sets
`vocab_remap: Some(..)` and emits no null at that path at all. The ACCEPT tests are what convert that
from a silent production failure into a loud test failure.

The first row is checker warning W-A made concrete: with `provenance` dropped from the walk, a null
smuggled into `provenance.dataset_fingerprint` is accepted **silently** and only the provenance
reject test notices.

## What The Artifact Contains, As Shipped

### Tensor set (fixture: 2 layers; the same rule gives 104 for the pinned 6-layer model)

`|expected| = 5 global + 16 per layer + 3 schema-owned`. For the fixture that is `5 + 32 + 3 = 40`
index entries, asserted by `writer_writes_exactly_the_architecture_derived_tensor_set` against a set
built by the same architecture-derived function the writer uses, plus an independent arithmetic
check so the two cannot agree by construction alone.

### Head tensor shapes observed

| Tensor | dtype | fixture shape | pinned shape |
| ------ | ----- | ------------- | ------------ |
| `setfit.head.weight` | F32 | `[3, 8]` | `[num_labels, 384]` |
| `setfit.head.bias` | F32 | `[3]` | `[num_labels]` |
| `tokenizer.blob` | U8 | `[tokenizer_bytes.len()]` — asserted equal to the input byte count, not a quoted figure | `[466247]` |

Both head tensors round-trip **bit-exactly** through `AprV2Reader::get_f32_tensor` (asserted with
`assert_eq!` on the `Vec<f32>`, not a tolerance). Review B1 is closed at the code layer: the head is
in the tensor index, so the tensor-set rule can refuse an artifact missing either half.

### The metadata field list, exactly as shipped

Typed container keys set by this writer: **`model_type = "setfit"` only.** `created_at` is written
explicitly as `None`. `license` / `data_source` / `data_license` remain the container's deterministic
explicit nulls and are deliberately outside the walk.

Custom keys: **exactly one**, `"setfit"`, holding a `serde_json::Map` with these 16 fields and no
others (`SETFIT_ARTIFACT_DOC_FIELDS`, asserted as a set equality so an added or renamed field is a
loud failure):

```
schema, schema_version, bundle_schema_version, format_id, architecture, tokenizer_sha256,
preprocessing, root_seed, head, ordered_labels, requested_config, resolved_config,
evidence, provenance, hf_name_map, probes
```

`preprocessing = {pooling, normalization, l2_epsilon_hex, truncation_max_sequence_length,
padding_mode, max_length}`; `head = {n_features, num_labels}`; each probe record is exactly
`{input, embedding_hex, logits_hex, probabilities_hex, label}`.

### The allowlist constant, as shipped

```rust
pub const NULLABLE_PATH_ALLOWLIST: [&str; 4] = [
    "architecture.vocab_remap",                 // EncoderArchitecture,  Option<Vec<u32>>
    "requested_config.pair_config.budget",      // PairConfigWire,       Option<u64>
    "requested_config.pair_config.hard_cap",    // PairConfigWire,       Option<u64>
    "evidence.epsilon_used",                    // EvidenceSummary,      Option<f64>
];
```

### The five walked sub-document prefixes, as shipped

```rust
pub const WALKED_SUBDOCUMENTS: [&str; 5] = [
    "architecture", "requested_config", "resolved_config", "evidence", "provenance",
];
```

Five walked, four allowlisted. `resolved_config` and `provenance` contribute zero and are walked
anyway; `the_walk_covers_all_five_sub_documents` asserts the exact literal list, and the falsification
above shows the fifth entry is load-bearing rather than decorative.

### The golden hash

```
GOLDEN_SHA256_FIXTURE_VIEW_FULL_PIN_SHAPE
  = 13e5c2965e95fc970c19a93f298a33b123f5c524a03c1e33e4a0e36967000bf4
```

It pins `write_setfit_apr(fixture_view_full_pin_shape())` — the DEFAULT builder, whose
`architecture.vocab_remap` is `None`. The constant's name states the shape so a later reader does not
have to open the builder to find out which fixture it pins. It was **measured** (placeholder ->
observed value -> blessed in its own commit `28b488382`), never chosen.

The same hex string is what the cross-process child printed and what the parent computed
independently, so the golden, the same-process double write and the two-process write all agree.

## Determinism, Proven Where It Can Actually Break

| Property | Test |
| -------- | ---- |
| same-process double write is byte-identical, same sha256 | `two_writes_of_one_view_are_byte_identical_in_one_process` |
| metadata write -> parse -> write is the identity; exactly one custom key | `metadata_survives_a_parse_and_re_serialize_byte_identically` |
| all FIVE sub-documents survive Value -> bytes -> Value -> bytes, including the allowlisted nulls and a fractional f64 | `all_five_sub_documents_round_trip_byte_identically_including_the_allowlisted_nulls` |
| **cross-process**: a re-invoked child writes the same view and reports the same sha256 | `cross_process_writes_produce_the_same_artifact_sha256` |
| a pinned golden hash for the full-pin fixture | `the_fixture_artifact_hash_matches_the_committed_golden` |
| in-band negative: two custom keys are unstable; the public path can produce only one | `the_public_writer_path_can_never_produce_two_custom_keys` |

The cross-process test reads the child's status **directly off `Output.status`** (CLAUDE.md
verification rule 1) and refuses to pass if the marker line is absent — a missing marker is an
`expect`, not an `unwrap_or_default`. It also asserts the recovered value is 64 characters, so a
truncated or empty capture cannot masquerade as agreement.

The sub-document round-trip test asserts a fractional `f64` is actually present in the serialized
evidence (`0.3333333333333333`) rather than assuming the number-formatting path was exercised.

## Deviations from Plan

### Auto-fixed / design changes

**1. [Rule 2 — correctness] The view carries the TYPED `EncoderArchitecture` only; the
sub-document is derived**

- **Found during:** Task 1 design
- **Plan text:** "`architecture` (as a `serde_json::Value` sub-document plus the typed
  `EncoderArchitecture` needed to rebuild)".
- **Issue:** carrying both is transporting one fact twice, and two copies of one fact are two values
  that can disagree. The contract's forward bijection row is `architecture = to_value(bundle.architecture)`
  — a FUNCTION of the typed record — so a codec that supplied a divergent `Value` would break the
  bijection with nothing turning red.
- **Fix:** the view carries the typed record; `build_artifact_doc` computes `serde_json::to_value`
  itself. A pleasant consequence, recorded in the code: the ONLY `null` the `architecture` subtree can
  emit is the allowlisted `vocab_remap`, because every other field is non-`Option`. The subtree is
  still walked, so a future `Option` there is caught.
- **Consequence for the reject test:** no view can smuggle a non-allowlisted null into
  `architecture`, so `reject_a_null_at_architecture_hidden_act_and_name_the_path` exercises
  `guard_subdocument_nulls` — the exact function `write_setfit_apr` calls — with the null injected
  into the built document. The other three reject cases go end-to-end through `write_setfit_apr`.
- **Files:** `crates/aprender-core/src/setfit/artifact.rs`
- **Commit:** `9e986c034`, `22e6a2cd2`

**2. [Rule 3 — blocking] `mod nullable` and `mod determinism` are DIRECT children of `artifact`,
not nested inside a `tests` module**

- **Found during:** Task 1, before writing any test
- **Plan text:** "Put the allowlist tests in their own `mod nullable` inside the test module".
- **Issue:** `cargo test` filters on the full test path. Nested, the paths would be
  `setfit::artifact::tests::nullable::…`, and the acceptance criteria's filters
  `setfit::artifact::nullable` and `setfit::artifact::determinism` would match **zero** tests. **A
  filter matching zero tests exits 0** — both counted gates would have reported a green pass over an
  empty set, which is precisely the CR-02 vacuity class this phase exists to prevent.
- **Fix:** three sibling modules (`tests`, `nullable`, `determinism`) plus a shared `fixture` module,
  all direct children of `artifact`. The reason is recorded in a comment at the module boundary.
- **Commit:** `9e986c034`

**3. [Rule 1 — bug] The cross-process harness's marker parse could never find the marker**

- **Found during:** Task 2 (the test failed, which is how it was found)
- **Issue:** `strip_prefix(CHILD_MARKER)` requires the marker at the START of the line. libtest with
  `--nocapture` prints the child's stdout on the SAME line as its `test <name> ... ` prefix, so the
  marker is mid-line and the search found nothing.
- **Fix:** search for the marker anywhere in the line (`line.find(..)`), plus a length assertion on
  the recovered value. The `expect` on a missing marker is deliberate: a missing marker must FAIL,
  never silently pass.
- **Commit:** `22e6a2cd2`

**4. [Rule 2 — future-proofing] Public API named after the binding registry's declarations**

- **Found during:** post-GREEN review of `contracts/aprender/binding.yaml`
- **Issue:** the registry already declares the function each equation binds to. Three of this plan's
  symbols did not match. Renaming after 04-03 and 04-05 depend on them would be a breaking change.
- **Fix:** `expected_container_tensor_names` -> `expected_tensor_names`; added
  `first_unallowed_null_path` (the decision `guard_subdocument_nulls` makes) and
  `canonical_name_for_hf` (implemented BY `build_hf_name_map`, so there is no second copy of the
  mapping). Two tests added for the new surface.
- **Commit:** `5f982ed6e`

### Fixture decision worth recording

The tiny fixture ships a **self-contained 48-entry WordPiece tokenizer** built in-code, in the pinned
file's exact structural shape (BertNormalizer + BertPreTokenizer + TemplateProcessing + WordPiece),
rather than reading the committed `tests/fixtures/setfit/tokenizer.json`. Two reasons:

1. `fixtures_dir()` honours the `APRENDER_SETFIT_FIXTURES` environment override
   (`tokenizer_tests.rs:41-49`). A golden hash whose expected value depends on an environment
   variable is not a gate.
2. Every id the tiny tokenizer can emit is `< 48`, which is what lets the fixture carry
   `vocab_remap: None` — the PRODUCTION shape — with a 48-row embedding table instead of the pin's
   30522. Using the real tokenizer would have forced a ~1 MB embedding table into every test.

`positions` is deliberately **not** reduced (it is `MAX_SEQUENCE_LENGTH = 256`): the tokenizer
truncates at 256, so `probe_truncation_boundary` produces a 256-position row and an encoder with
fewer position rows would refuse it with `OversizeInput` before a probe could be recorded. The
fixture keeps the exact per-layer NAME topology and the exact NULLABILITY topology; only the
dimensions and the layer count are reduced.

## Criterion Defect Found (reported rather than papered over)

Acceptance criterion: *"`grep -c "skip_serializing_if" crates/aprender-core/src/setfit/artifact.rs`
== 0"*.

**Measured: 3, not 0.** All three are prose in doc comments — lines 46, 141 and 175 — explaining that
`skip_serializing_if` is FORBIDDEN on the five sub-document types and why (it would change
`SetFitBundle::to_canonical_bytes` and break Phase 3's committed closure tests, while silently
emptying the allowlist with no test turning red). The contract's `nullable_path_allowlist` invariant
requires that prohibition be recorded, so satisfying the criterion literally would mean deleting the
explanation the contract mandates.

The assertion that actually tests the intent is the attribute form, and it scores zero:

```
$ grep -c 'skip_serializing_if *=' crates/aprender-core/src/setfit/artifact.rs
0   (exit 1 — no matches)
```

Reporting this as a plain pass would have been the "check how it was measured" failure.

## Other Source Assertions (all measured)

| Assertion | Criterion | Observed |
| --------- | --------- | -------- |
| `pub fn write_setfit_apr(` present | yes | yes |
| `pub fn artifact_sha256_hex(` present | yes | yes |
| `grep -c "provenance"` | >= 3 | **16** |
| `grep -c "NULLABLE_PATH_ALLOWLIST"` | >= 2 | **9** |
| `grep -c "fixture_view_full_pin_shape"` | >= 2 | **37**, and it is the builder every non-slice test calls |
| `grep -c "setfit.head.weight"` / `"setfit.head.bias"` | >= 1 each | present as named constants + tests |
| `SetFitArtifactError` derives | Debug, Clone, PartialEq | `#[derive(Debug, Clone, PartialEq, Eq)]` + `#[non_exhaustive]` |
| `created_at` | never `Some(...)` | 3 occurrences: a doc heading, `created_at: None`, and a test asserting `None` |
| `current_exe` | present | 1 |
| golden hex literal near a `GOLDEN` name | present | `GOLDEN_SHA256_FIXTURE_VIEW_FULL_PIN_SHAPE` |
| `artifact.rs` line count | >= 280 | **2457** |
| `git diff --name-only` | only `crates/aprender-core/` | `setfit/artifact.rs`, `setfit/mod.rs` — nothing else |
| every typed error asserted with `matches!` | yes | every refusal test uses `matches!`, none matches message text |

## Clippy

```
$ CARGO_INCREMENTAL=0 cargo clippy -p aprender-core --features setfit --lib --tests
rc=0
$ grep -c "artifact.rs" <clippy log>
0   (no findings in the new module)
```

`cargo clippy -p aprender-core --features setfit -- -D warnings` returns rc=101, and **every finding
is in `crates/aprender-compute/`** — a dependency this plan does not touch (`git diff --name-only`
above). That is the KNOWN-RED arm64 baseline the phase records as D-ITEM-02, not a regression from
this plan. Stated as a scoped-clean result rather than a whole-command pass.

`rustfmt --edition 2021 --check crates/aprender-core/src/setfit/artifact.rs` → rc=0.

## TDD Gate Compliance

| Gate | Commit | Evidence |
| ---- | ------ | -------- |
| RED | `9e986c034` `test(04-02): …` | measured **4 passed, 35 failed** of 39 under `setfit::artifact::`. The 4 that passed are the ones that touch no writer: the two constant assertions, the hex-precedent unit test, and the guarded cross-process child (a no-op without its env var). |
| GREEN | `22e6a2cd2` `feat(04-02): …` | 38 passed, 1 failed — only the golden placeholder. |
| GREEN (Task 2) | `28b488382` `test(04-02): …` | 39 passed, 0 failed. |
| REFACTOR | `5f982ed6e` `refactor(04-02): …` | 41 passed, 0 failed. |

**Honesty note on Task 2's RED.** At the RED commit the determinism suite was genuinely red along
with everything else (the writer was stubbed). After the writer landed, five of the seven determinism
tests passed immediately, because they characterise properties of the writer built in Task 1 rather
than new behaviour. The two that were genuinely red at that point — the golden-hash placeholder and
the broken marker parse — are the two that produced real work. Reporting this rather than claiming a
seven-test RED I did not observe.

## Notes for Later Plans

- **04-03 (loader).** `SetFitArtifactError` is `#[non_exhaustive]` precisely so the loader's rungs
  join this enum rather than starting a second one. `expected_tensor_names(num_layers)` is already the
  architecture-derived rule the loader's rung 5 needs, and `SETFIT_ARTIFACT_DOC_FIELDS` is the field
  list its `deny_unknown_fields` struct must match. **`SetFitArtifactDoc` — the parse struct — is
  yours**: this plan builds the document as a `serde_json::Map` (as the plan directs) and owns only
  the normative field list; `binding.yaml` currently notes `artifact_doc_schema` as landing with
  04-02, which is not what shipped.
- **04-05 (codec).** `SetFitArtifactView` has exactly 20 public fields, one per `SetFitBundle` field,
  so population is mechanical. `SetFitArtifactError` derives `Debug + Clone + PartialEq` so it embeds
  in `CodecError` as a typed source with no stringification. Note the cost of probe recomputation:
  `compute_probes` **clones the tensor map** because `from_bundle_parts` drains what it is given, so
  peak memory on a full pin is roughly double the encoder payload (~180 MB). That is deliberate —
  computing probes from anything but the view's own tensors would record expectations for a model the
  artifact does not contain — but it is worth knowing before the codec is called in a memory-tight
  context.
- **04-13 (completeness gate).** Assert against `aprender::setfit::NULLABLE_PATH_ALLOWLIST` (it is
  re-exported from `setfit/mod.rs`) rather than a local literal, so the two halves of the guarantee
  cannot disagree about the allowlist's contents. `WALKED_SUBDOCUMENTS` is re-exported too.
- **04-09 / 04-12 / 04-15.** "Build the fixture artifact the same way" means
  `fixture_view_full_pin_shape()` — the DEFAULT builder, `vocab_remap: None`. The slice builder exists
  only for the tests that need the slice topology.
- **The binding flip is DEFERRED, deliberately.** `contracts/aprender/binding.yaml` is not in this
  plan's `files_modified` and is a shared file that the two concurrent wave-2 agents (04-13, 04-14)
  could also touch. Flipping it here risked a merge conflict across the wave for no test benefit.
  Three equations are now implementable-and-implemented and should be flipped from `pending` to
  `implemented` by the next plan that owns that file: `artifact_storage_map` -> `write_setfit_apr`,
  `nullable_path_allowlist` -> `first_unallowed_null_path`, `canonical_tensor_names` ->
  `canonical_name_for_hf`. All three now exist under those exact names (that is what commit
  `5f982ed6e` was for). `architecture_derived_tensor_set` -> `expected_tensor_names` also exists,
  though its registry note assigns it to 04-03.

## Platform Assumption, Stated Rather Than Assumed

The golden hash covers bytes that include the six probe expectations, which are the output of a real
encoder forward pass. The fixture's own float data is exactly representable (`k / 65536 - 0.5`), so it
cannot itself drift; the probe floats are the product of scalar `f32`/`f64` IEEE arithmetic in a fixed
source order (the fixture's hidden width of 8 is below trueno's `SIMD_THRESHOLD` of 64, so the matmul
path is `matmul_naive`). The test comment says plainly that a golden divergence on another platform is
a **real finding, not a flake**: the artifact hash IS the identity every Phase 4 response carries, so
two platforms producing different bytes for one view is exactly the parity defect SAFE-01 exists to
catch. It must not be "fixed" by relaxing the test.

## Threat Flags

None. This plan introduces no network endpoint, no auth path, no file access and no schema change at
a trust boundary. `write_setfit_apr` reads nothing from disk and consults no environment. The
mitigations the plan's threat register assigns to these files are all present and counted:

| Threat ID | Mitigation as shipped |
| --------- | --------------------- |
| T-04-03 | one custom key + sorted `serde_json::Map` + `created_at: None` + a cross-process byte-equality test |
| T-04-04 | the six probe inputs are the contract's synthetic strings only, asserted verbatim; no dataset text |
| T-04-05 | typed `NonFiniteValue` before serialization; the walk covers all five sub-documents, collects all paths, rejects anything outside the four-entry allowlist; bit-pattern hex for every stored float |
| T-04-58 | the allowlist matches the contract's four derived paths; the DEFAULT fixture carries the production full-pin shape; four counted ACCEPT tests, falsified |
| T-04-64 | `provenance` and `resolved_config` walked with zero allowlisted paths; a counted REJECT test injects into `provenance`, and dropping it from the walk was shown to turn that test red |
| T-04-38 | head written as two named F32 tensors with declared shapes; an incomplete tensor set is a typed error |
| T-04-SC | zero new packages |

## Known Stubs

None. Every function this plan declares is implemented and exercised. `SetFitArtifactError` is
`#[non_exhaustive]` with a documented seam for 04-03's loader variants — a declared extension point,
not a stub, and no code path returns a placeholder.

## Self-Check: PASSED

Files claimed, checked on disk:

```
FOUND: crates/aprender-core/src/setfit/artifact.rs   (2457 lines)
FOUND: crates/aprender-core/src/setfit/mod.rs        (modified: pub mod artifact + re-exports)
```

Commits claimed, checked in the log:

```
FOUND: 5f982ed6e refactor(04-02): name the writer's API after the binding registry's declarations
FOUND: 28b488382 test(04-02): bless the golden sha256 of the full-pin fixture artifact
FOUND: 22e6a2cd2 feat(04-02): implement write_setfit_apr — the setfit-apr-v1 writer (GREEN)
FOUND: 9e986c034 test(04-02): failing writer suite for setfit-apr-v1 (RED)
```

`git diff --diff-filter=D --name-only b30bff96a HEAD` is **empty** — no file deletions in any commit.
`git status --short --untracked-files=all` is **empty** — nothing generated and left untracked.
`git diff --name-only b30bff96a HEAD` lists only the two `crates/aprender-core/` paths, so the wave-2
ownership contract with 04-13 and 04-14 held.
