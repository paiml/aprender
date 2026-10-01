---
phase: 03-faithful-two-stage-trainer-and-head
plan: 08
subsystem: training
tags: [setfit, persistence, codec, serde, sha256, typestate, reproducibility, contracts]

requires:
  - phase: 03-07
    provides: SetFitRun<HeadFitted> with HeadFittedEvidence — the fitted head, the ordered labels, the encode ledger and the passed stage-one chain
  - phase: 03-05
    provides: the in-band execution digests (consumed_pair_digest, batch_boundary_digest, batch_boundary_list, parameter_registry_hash) recorded at consumption
  - phase: 03-06
    provides: contracts/setfit-train-lifecycle-v1.yaml and the frozen thresholds the evidence gate reads
  - phase: 01
    provides: SetFitMiniLm, BertSentenceEncoder, MiniLmTokenizer and the frozen MiniLM slice fixture
provides:
  - SetFitBundle — the complete deterministic state of a finished run, with a canonical wire form and four bounds enforced before the allocations they bound
  - A sealed SetFitCodec seam (three methods, no power to hash, compare, set a tolerance or mint a state) plus SerdeJsonCodec, the phase-3 implementation
  - SetFitRun<HeadFitted>::verify_artifact — the trusted close/reload/rebuild/re-predict/compare policy, and the only door to ArtifactReloadedAndVerified
  - SetFitMiniLm::from_bundle_parts / tokenizer_bytes / architecture / named_parameters / root_seed — the aprender-core bytes-reconstruction path
  - MultinomialLogisticRegression::from_stored_coefficients — the head's reload door
  - Ten public read-only reproducibility accessors on the final state
  - Three contract equations (reload_verify_roundtrip, bundle_completeness, bundle_limits), six obligations, six falsification tests, three binding entries
affects: [03-09, 03-10, phase-04-apr-artifact, phase-05-benchmark]

tech-stack:
  added:
    - "serde_json feature `float_roundtrip` in aprender-core AND aprender-train (no new package; `float_roundtrip = []`)"
  patterns:
    - "Codec/policy split: an implementable seam that owns the FORMAT and cannot reach the VERDICT"
    - "Round-trip closure check: re-serialize what a codec returned and require byte equality with what was hashed, so the reloaded value is provably a function of the bytes"
    - "Structural close: the policy takes the live model BY VALUE, so the borrow checker enforces what a drop() call would only document"
    - "Bounds as a parameter with a `#[cfg(test)]` shrinking constructor, so the MECHANISM is falsified on a real payload while the VALUES are pinned against the contract"
    - "Hex-of-bit-pattern payloads, which make f32 round-tripping exact by construction and element counts knowable from a string length"

key-files:
  created:
    - crates/aprender-train/src/train/setfit/bundle.rs
    - crates/aprender-train/src/train/setfit/bundle_tests.rs
    - crates/aprender-train/src/train/setfit/verify.rs
    - crates/aprender-train/src/train/setfit/verify_tests.rs
  modified:
    - crates/aprender-core/src/setfit/tokenizer.rs
    - crates/aprender-core/src/setfit/encoder.rs
    - crates/aprender-core/src/setfit/import.rs
    - crates/aprender-core/src/setfit/mod.rs
    - crates/aprender-core/src/setfit/model_tests.rs
    - crates/aprender-core/src/setfit/tokenizer_tests.rs
    - crates/aprender-core/src/classification/multinomial.rs
    - crates/aprender-core/Cargo.toml
    - crates/aprender-train/Cargo.toml
    - crates/aprender-train/src/train/setfit/mod.rs
    - crates/aprender-train/src/train/setfit/head_input.rs
    - crates/aprender-train/src/train/setfit/tune.rs
    - contracts/setfit-train-lifecycle-v1.yaml
    - contracts/aprender/binding.yaml

key-decisions:
  - "serde_json's `float_roundtrip` feature is REQUIRED, not an optimization: without it the fixture bundle differed from its own re-serialization at byte offset 936,551, so every honest codec would have failed the round-trip closure check"
  - "Tensor data and tokenizer bytes travel as lowercase hex of little-endian bit patterns, which makes f32 exactness structural and makes element counts knowable BEFORE any Vec<f32> is allocated — the plan's bounding requirement is unsatisfiable with decimal JSON numbers"
  - "The architecture record carries the vocabulary remap; SliceConfig's field set alone rebuilds a slice that gathers the wrong embedding row for every token and still looks valid"
  - "There is no From<&SliceConfig> for the architecture record: SliceConfig is conformance-gated, absent on the pin path, and carries no remap, so such a door would mint incomplete records"
  - "EncoderArchitecture lives in setfit/mod.rs, not import.rs, because import.rs's guard forbids deny_unknown_fields there and this record has the opposite obligation"
  - "The EchoCodec cheat perturbs source_revision only — pure provenance with no second catcher — because an extra label is caught by the head's arity check and a perturbed tensor by the probe comparison, and either would have made the negative green for the wrong reason"
  - "The close is structural: run_verify_policy takes the encoder and head BY VALUE into its close step and never returns them"
  - "TRN-01 and TRN-06 are left UNCHECKED; 03-10's out-of-crate cross-process gate is the plan that closes them at the 'a user can' tier"

patterns-established:
  - "Prove the mechanism, not the flag: `bundle_serde_json_parses_floats_round_trip_exactly` and `json_roundtrip_of_an_f64_is_bit_exact` assert the BEHAVIOUR a Cargo feature buys, because a feature can be declared and inert"
  - "A guard whose enumeration must grow is widened with the argument written down beside it, plus a new test for the property the enumeration stands for"
  - "An in-band negative is MEASURED red before the mechanism that reddens it is added, and the measurement goes in the commit message"

requirements-completed: []

duration: ~2h35m
completed: 2026-08-10
---

# Phase 3 Plan 08: Verify by Reload — the Codec Seam and the Complete Bundle

**The persistence half of the lifecycle: a sealed three-method codec that owns the format and
cannot reach the verdict, a bundle complete enough that a model rebuilt from its bytes alone
embeds bit-identically, and a trusted policy that drops the live model, reloads, re-serializes
what it got back, and refuses anything whose serialization is not the artifact it hashed.**

## Performance

- **Duration:** ~2h35m (first commit 22:43:45Z, last code commit 23:21:32Z; work began ~21:05Z)
- **Started:** 2026-08-10T21:05:00Z (approximate — the base-commit reset)
- **Completed:** 2026-08-10T23:45:00Z
- **Tasks:** 3 of 3 (Task 2 is TDD: RED then GREEN)
- **Files modified:** 18 (4 created, 14 modified)

## Accomplishments

- `SetFitRun<ArtifactReloadedAndVerified>` exists and is reachable ONLY through a real
  close → reload → rebuild → re-encode → re-predict → compare round trip. The full lifecycle
  `Prepared -> EncoderTuned -> HeadFitted -> ArtifactReloadedAndVerified` runs end to end.
- A bundle that genuinely reloads: `bundle_rebuilds_a_bit_identical_encoder_from_bytes_alone`
  rebuilds through the wire form and compares raw `f32` bit patterns, and
  `bundle_rebuilds_a_head_that_predicts_identically` does the same for the head's probabilities.
- Two real defects found, both by taking byte-canonicity seriously (see Deviations).
- The reproducibility surface: ten public read-only accessors, with the perturbed-consumption
  companion proving `pair_order_digest()` reports what the loop CONSUMED rather than what it
  was configured to consume.

## Task Commits

1. **Task 1: Complete bundle + the aprender-core bytes-reconstruction path** — `6599fbc4b` (feat)
2. **Task 2 RED: behaviour tests + the seam, without the closure check** — `ead8cf252` (test)
3. **Task 2 GREEN: the round-trip closure check** — `2ef3a39b6` (feat)
4. **Task 3: contract growth and binding entries** — `1a223ab9f` (docs)

## The bundle schema

`schema_version = 1`, `format_id = "setfit-serde-json-v1"`. Nineteen fields in fixed wire
(declaration) order — the normative completeness list, restated in the `bundle_completeness`
equation and asserted field by field by
`bundle_declares_every_field_of_the_normative_completeness_list`:

| # | Field | Note |
|---|-------|------|
| 1 | `schema_version: u32` | a future version is refused, never partially interpreted |
| 2 | `format_id: String` | the codec's own; never the shipped container's name |
| 3 | `architecture: EncoderArchitecture` | 14 fields — `SliceConfig`'s set PLUS `vocab_remap` |
| 4 | `tokenizer_bytes_hex: String` | the exact `tokenizer.json` bytes |
| 5 | `pooling: String` | from `aprender::setfit::POOLING_POLICY` |
| 6 | `normalization: String` | from `NORMALIZATION_POLICY` |
| 7 | `l2_epsilon: f32` | from `L2_EPS`, the constant the encoder clamped with |
| 8 | `truncation_max_sequence_length: u32` | from `MAX_SEQUENCE_LENGTH` |
| 9 | `padding_mode: String` | from `PADDING_MODE` |
| 10 | `max_length: u32` | the run's requested value |
| 11 | `root_seed: u64` | read off the ENCODER, not the configuration |
| 12 | `tensors: BTreeMap<String, BundleTensor>` | every named parameter, `(shape, hex data)` |
| 13-15 | `head_weights_hex`, `head_intercepts_hex`, `head_n_features` | |
| 16 | `ordered_labels: Vec<String>` | the head's weight-row index |
| 17 | `requested_config: SetFitTrainConfig` | what was ASKED FOR |
| 18 | `resolved_config: ResolvedConfigRecord` | what was RESOLVED, as provenance only |
| 19 | `evidence: EvidenceSummary` | including its binding `table_hash` |

`#[serde(deny_unknown_fields)]` on the bundle, the tensor and the two records.

## The codec trait phase 4 must implement, and the sealing consequence

```rust
pub trait SetFitCodec: sealed::Sealed {
    fn format_id(&self) -> &'static str;
    fn serialize(&self, bundle: &SetFitBundle) -> Result<Vec<u8>, CodecError>;
    fn deserialize(&self, bytes: &[u8]) -> Result<SetFitBundle, CodecError>;
}
```

Three methods. No hashing (`artifact_hash` is a free function), no tolerance, no comparison,
no way to construct a lifecycle state. `verify_codec_trait_is_a_pure_sealed_codec` asserts the
method count AND scans the trait body for `hash`, `tolerance`, `compare`, `verify` and
`SetFitRun`.

**The sealing consequence, for the user to confirm:** `sealed::Sealed` lives in a private
module, so phase 4's APR codec lands as a thin **adapter module inside `aprender-train`** —
`impl Sealed for AprCodec` plus the codec impl, calling `aprender-core`'s APR format code. The
APR **format** stays in `aprender-core`; only the adapter moves. Sealing later would be
breaking and un-sealing later is not, which is why phase 3 seals; deleting the supertrait later
is a non-breaking change if an openly implementable codec is preferred.

**The canonical-serialization obligation is on every implementor**, stated in the module doc
and as an explicit clause of `reload_verify_roundtrip`: `serialize(deserialize(b)) == b` byte
for byte. A phase-4 writer with padding, unordered metadata or a trailing checksum is a
contract-visible incompatibility to amend deliberately — not a mysterious `ReloadNotFromBytes`
to debug.

## Measured sizes

| Quantity | Value |
|----------|-------|
| Retained tokenizer bytes (pinned `tokenizer.json`) | **466,247 B** |
| Fixture bundle: tensor elements | 110,528 `f32` |
| Fixture bundle: raw payload (tensors + tokenizer) | 908,359 B |
| **Fixture bundle: serialized** | **1,823,868 B** |
| Fixture amplification | **2.01x** |
| Full MiniLM-L6-v2: tensors | 101 |
| Full MiniLM-L6-v2: largest tensor | 11,720,448 elements (30522x384) |
| Full MiniLM-L6-v2: total elements | 22,565,376 |
| **Full MiniLM-L6-v2: projected serialized** | **182,455,502 B (~182 MB)** |

The amplification is why phase 4 replaces the FORMAT behind the same codec trait. The full-pin
figures are COMPUTED from the pinned architecture by
`bundle_limits_clear_the_full_minilm_figures`, not quoted, so a change to the pin moves the
comparison with it.

## The four bounds, with headroom

| Limit | Value | Full-pin figure it clears | Headroom |
|-------|-------|---------------------------|----------|
| `max_bundle_bytes` | 536,870,912 (512 MiB) | 182,455,502 | **2.94x** |
| `max_tensor_count` | 4,096 | 101 | **40.6x** |
| `max_elements_per_tensor` | 134,217,728 (2^27) | 11,720,448 | **11.45x** |
| `max_total_elements` | 268,435,456 (2^28) | 22,565,376 | **11.90x** |

All four are enforced BEFORE the allocation each bounds. The input length is checked on the raw
slice before serde is handed anything; tensor data travels as hex, so an element count is
`len(hex)/8` and is known from a string length rather than from a decoded vector.

The **values** are parsed out of the contract by `bundle_limits_match_the_contract` (serde_yaml,
field by field, following `OBLIG-STL-THRESHOLDS-PARSED`). The **mechanism** is falsified
separately: four tests set deliberately tiny bounds on a real fixture bundle, each with a
CONTROL at exactly the bound that must be accepted — so none of the four is a refusal of
everything. Proving the 512 MiB bound by materializing half a gigabyte would have made the suite
pay 512 MB to learn that a comparison compares.

## What the EchoCodec negative proves, and what it does not

`EchoCodec.deserialize` ignores its input bytes entirely and returns a bundle cached at
construction; `serialize` is honest and never refreshes that cache.

**It proves** that the trusted policy forces the reloaded value to be a FUNCTION OF THE HASHED
BYTES. The round-trip closure check re-serializes what the codec returned and requires byte
equality with what was hashed, so a codec cannot substitute an arbitrary object for the
artifact's contents.

**It does not prove** durability. An in-process codec that round-trips faithfully through a
buffer it never writes anywhere satisfies every check here and is indistinguishable from one
that wrote a file, because nothing in this crate observes the filesystem. Durability is a claim
about I/O; phase 4's format is where it acquires one.

**It was MEASURED red, not assumed red.** Commit `ead8cf252` shipped the policy without the
closure check and recorded the observation: `verify_artifact(&EchoCodec)` returned `Ok` and
minted `ArtifactReloadedAndVerified`. Choosing the cheat took two attempts, and the failures are
the finding:

| Perturbation | Caught by | Verdict |
|--------------|-----------|---------|
| One extra label | the head's coefficient-arity check | green for the wrong reason |
| A perturbed tensor | the probe comparison | green for the wrong reason |
| `source_revision` | **nothing else** — pure provenance, read by no rebuild step and touched by no comparison | attributable to the closure check alone |

The closure check runs immediately after `deserialize` and BEFORE the rebuild, and
`verify_policy_closes_the_live_model_before_it_reloads` asserts that ordering in source. Placing
it after the comparison would not work: a codec whose cache described a behaviourally identical
model passes the comparison, so the substitution would never be seen.

## Decisions Made

- **Hex-of-bit-pattern payloads instead of JSON decimal numbers.** The plan discussed ryu
  exactness, but the plan's *normative* requirement is that the four bounds be enforced BEFORE
  the allocation. With decimal numbers an element count is unknowable until the `Vec<f32>` is
  built, which makes "bounded before allocation" unsatisfiable without a custom counting
  `Deserialize`. Hex makes the count a string length, makes `f32` round-tripping exact for
  subnormals, `-0.0` and NaN by construction rather than by trusting a float formatter, and
  costs 8 characters per `f32`. Base64 would be denser and was deliberately not introduced: the
  threat model dispositions package installs as `accept / none this plan`, and taking a
  dependency to shrink a format scheduled for replacement is the wrong trade.
- **No `From<&SliceConfig>` for the architecture record.** `SliceConfig` is parsed only on the
  `conformance-fixtures` path, does not exist on the full-pin path, and carries no vocabulary
  remap. Such a conversion would be a door that mints incomplete records for exactly the models
  that need the missing field. `SetFitMiniLm::architecture()` is the single producer and reads
  every field off the encoder that would be rebuilt.
- **`EncoderArchitecture` lives in `setfit/mod.rs`.** Placing it in `import.rs` made
  `import_pin_constructors_are_sealed` fire, and that guard is right: no wire form on the
  pin-import path may deny unknown fields, because the real `config.json` carries metadata this
  crate does not model. A closed artifact schema has the opposite obligation. Moving the type
  keeps the guard exactly as strong instead of weakening it to accommodate a type it was never
  about.
- **The close is structural.** `run_verify_policy` hands the live encoder and head BY VALUE into
  its `close` step, which never returns them. After that call there is no binding to the
  pre-close model, so the comparison cannot be built from it even by accident — the borrow
  checker enforces what a `drop(...)` call would only document.
- **`HeadFittedEvidence::into_parts` splits recorded facts from the live head.** The head is
  MODEL STATE, whose replacement by a reloaded one is the whole claim; the digests, the ledger,
  the report and the lambda are MEASUREMENTS of a run that already happened, and a measurement
  is not made truer or falser by being carried forward.
- **TRN-01 and TRN-06 are left unchecked.** The MECHANISM of TRN-01 is complete — all four
  lifecycle stages exist and the chain runs end to end — but its "a developer can" tier is
  demonstrated here only from inside the crate. 03-10's out-of-crate cross-process gate is the
  plan that closes both, and this phase's policy (inherited from 02-09) is to mark each
  requirement at the plan that actually closes it rather than at the plan that makes it
  possible.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] serde_json's default float parser is not correctly rounded, and it broke the round-trip closure check for every honest codec**

- **Found during:** Task 1 (`bundle_f32_round_trip_is_exact_for_adversarial_values`)
- **Issue:** `serde_json` without the `float_roundtrip` feature uses a fast float parser that
  can land one ULP from the value ryu wrote. Measured on the fixture's own evidence summary:
  `0.00009142675446597625` parsed back and re-serialized as `...624`, so re-serializing a parsed
  bundle differed from its input at **byte offset 936,551**. Since the verify policy hashes a
  bundle's bytes and then compares them against the re-serialization of the reloaded bundle, this
  would have failed `ReloadNotFromBytes` on every real bundle and every faithful codec — and the
  failure would have read as "the codec is cheating" rather than "the float parser is
  approximate".
- **Fix:** Declared `serde_json = { features = ["float_roundtrip"] }` in **both**
  `aprender-train` and `aprender-core`. No new package (`float_roundtrip = []` in serde_json's
  manifest); it only selects the correctly-rounding parse path. Declared in `aprender-core` too
  rather than inherited by feature unification, because that crate's tests assert the guarantee
  and a crate that asserts a guarantee has to be the crate that requests it.
- **Files modified:** `crates/aprender-train/Cargo.toml`, `crates/aprender-core/Cargo.toml`
- **Verification:** `bundle_serde_json_parses_floats_round_trip_exactly` asserts the BEHAVIOUR
  on six literals (a feature can be declared and inert); `bundle_round_trip_is_byte_stable_and_closed` is green.
- **Committed in:** `6599fbc4b`

**2. [Rule 1 - Bug] aprender-core carried a test that ASSERTED the imprecision, with instructions to delete it — now discharged**

- **Found during:** Task 1 regression run
- **Issue:** `json_roundtrip_of_an_f64_is_not_bit_exact` (in `multinomial.rs`) documented the
  one-ULP drift and said in its own failure message: *"If it is now exact, the dependency was
  fixed: delete this test and tighten the report round-trip assertion back to a full equality."*
  Enabling `float_roundtrip` turned it red. Somebody had hit this exact defect before and worked
  around it by documenting the imprecision.
- **Fix:** Followed the instruction. The test is now `json_roundtrip_of_an_f64_is_bit_exact`,
  keeping the measured history in its doc comment and asserting the behaviour rather than the
  flag. `head_fit_report_serializes_stably_across_two_runs` had its one-ULP tolerance replaced
  with bitwise equality on `objective` and `final_grad_norm`, plus a re-serialization check — a
  tolerance kept after its cause is gone is a hole that admits a real drift later.
- **Files modified:** `crates/aprender-core/src/classification/multinomial.rs`
- **Verification:** `cargo test -p aprender-core --lib classification` — 245 passed.
- **Committed in:** `6599fbc4b`

**3. [Rule 2 - Missing critical functionality] The bundle could not rebuild a working model without the vocabulary remap, and the plan's field list did not include it**

- **Found during:** Task 1
- **Issue:** The plan's normative completeness list is `SliceConfig`'s field set. A slice encoder
  gathers embeddings through `slice_to_orig`; a record without it rebuilds an encoder that reads
  the wrong embedding row for **every token** and still looks structurally valid. The plan's own
  acceptance criterion (bit-identical embeddings) is unsatisfiable without it.
- **Fix:** `EncoderArchitecture` carries `vocab_remap: Option<Vec<u32>>` (the `slice_to_orig`
  table). Only that direction travels; the reverse is DERIVED at rebuild by
  `VocabRemap::from_slice_to_orig`, so the two cannot be transported inconsistently. Contracted
  in `bundle_completeness`.
- **Files modified:** `crates/aprender-core/src/setfit/mod.rs`, `import.rs`, `encoder.rs`
- **Verification:** `bundle_rebuilds_a_bit_identical_encoder_from_bytes_alone`;
  `bundle_records_the_architecture_and_policy_the_encoder_actually_uses` asserts the remap is
  present and has one row per embedding row.
- **Committed in:** `6599fbc4b`

**4. [Rule 3 - Blocking] There was no way to rebuild a fitted head, so "re-predict from the rebuilt model" was unwritable**

- **Found during:** Task 1
- **Issue:** `MultinomialLogisticRegression` had `new()` and `fit()` and no constructor from
  stored coefficients. `predict_proba` needs `n_features`, `weights`, `intercepts` and `labels`,
  all private. The alternative — re-implementing softmax in `aprender-train` — would have made
  the comparison compare against something production never runs.
- **Fix:** `from_stored_coefficients`, sharing the fit path's label-set rule via an extracted
  `validate_label_set` (a second copy is how a reloaded head starts accepting a label map the fit
  would have refused). Two `HeadInputError` variants added for the checks a fit cannot get wrong.
  `report()` stays `None`: no optimizer ran in this process and a synthesized report would assert
  a convergence status nobody observed.
- **Files modified:** `crates/aprender-core/src/classification/multinomial.rs`
- **Verification:** `bundle_rebuilds_a_head_that_predicts_identically`.
- **Committed in:** `6599fbc4b`

**5. [Rule 3 - Blocking] Two pre-existing guards fired; both were right, and both were widened deliberately rather than weakened**

- **Found during:** Task 1 regression run
- **Issue:** (a) `import_pin_constructors_are_sealed` forbids `deny_unknown_fields` anywhere in
  `import.rs`, and the new architecture record needs it. (b)
  `setfit_model_exposes_exactly_two_public_constructors` enumerates `SetFitMiniLm`'s public
  constructors, and `from_bundle_parts` is a third.
- **Fix:** (a) The type moved to `setfit/mod.rs`. The guard's invariant is about the pin-import
  path, where denying unknown fields would reject the pinned `config.json` itself; a closed
  artifact schema has the opposite obligation and does not belong in that file. The guard is now
  exactly as strong as before. (b) The enumeration became `PUBLIC_CONSTRUCTORS` with the argument
  written beside it — every entry builds the tokenizer and encoder from ONE source, and
  `from_bundle_parts` qualifies because the architecture record carries the tokenizer's sha256
  and the reload path checks it before a tensor is installed. A new test,
  `setfit_model_bundle_constructor_checks_tokenizer_identity_before_building`, asserts that
  ordering in source, so the list now stands for a property rather than only for a count.
- **Files modified:** `crates/aprender-core/src/setfit/model_tests.rs`, `mod.rs`, `import.rs`
- **Verification:** `cargo test -p aprender-core --lib --features conformance-fixtures setfit` — 183 passed.
- **Committed in:** `6599fbc4b`

**6. [Rule 3 - Blocking] The verify probe read a `#[cfg(test)]` accessor**

- **Found during:** Task 2 (clippy)
- **Issue:** `probe_model` called `HeadDataset::encode_ledger()`, which is `#[cfg(test)]` because
  the shipped path MOVES the ledger out rather than borrowing it. The test build compiled; the
  library build did not.
- **Fix:** `HeadDataset::into_probe_parts`, which moves the embeddings and the recorded ledger
  out **together** from the one object the encode filled. Deriving the ids from
  `selection.ordered_ids()` instead was rejected for the reason `head_input.rs` already records:
  a list rebuilt from the selection agrees with the selection by construction and cannot see a
  windowing defect.
- **Files modified:** `crates/aprender-train/src/train/setfit/head_input.rs`, `verify.rs`
- **Verification:** `cargo clippy -p aprender-train --lib --features setfit --no-deps -- -D warnings` rc=0.
- **Committed in:** `2ef3a39b6`

**7. [Rule 3 - Blocking] Two source-assertion tests matched their own forbidden tokens**

- **Found during:** Tasks 1 and 2
- **Issue:** (a) `bundle_resolved_config_is_provenance_with_no_reconstruction_path` scanned for
  `TryFrom<ResolvedConfigRecord>` — which the doc comment explaining its absence contained, and
  which the test's own literal would have added to the directory the plan's `grep` scans. (b) The
  same test's `Deserialize` scan matched the doc comment reading *"NOT `Deserialize`,
  deliberately"*.
- **Fix:** (a) The doc comment no longer names the forbidden conversion (and says why), and the
  test assembles the token at runtime. `grep -rc 'TryFrom<ResolvedConfigRecord>'` over the
  directory now returns nothing. (b) The scan reads the ATTRIBUTE lines immediately above the
  declaration, not the surrounding text. This is the same class of defect `import_tests.rs`
  already guards against with its comment-line filter.
- **Files modified:** `crates/aprender-train/src/train/setfit/bundle.rs`, `bundle_tests.rs`
- **Verification:** the test is green and the directory grep is clean.
- **Committed in:** `6599fbc4b`

**8. [Rule 2 - Missing critical functionality] Test-only reachability for the perturbed-consumption companion**

- **Found during:** Task 2
- **Issue:** The plan requires a run whose consumption order was perturbed to report a DIFFERENT
  digest **through the accessor**. `TuningProbes::REVERSE_INTRA_BATCH_PULL` existed but could not
  reach `SetFitRun`.
- **Fix:** `tune::run_tuning_with_probes` (`#[cfg(test)]`) and
  `SetFitRun::<Prepared>::tune_encoder_with_probes` (module-private). A shipped build can name
  only `TuningProbes::NONE`, so the door is exactly `tune_encoder`. This follows `tune.rs`'s own
  precedent: a production knob whose only use is to make a run lie about itself would be worse
  than the bug it tests for.
- **Files modified:** `crates/aprender-train/src/train/setfit/tune.rs`, `mod.rs`
- **Verification:** `verify_pair_order_digest_changes_when_consumption_order_changes` — the two
  runs agree on batch boundaries, step count and selection hash and DISAGREE on the pair digest.
- **Committed in:** `ead8cf252`

**9. [Rule 2 - Missing critical functionality] `pv` requires falsification_tests >= proof_obligations**

- **Found during:** Task 3
- **Issue:** Six new obligations against five new falsification tests made 13 vs 12, and
  `pv validate` returned `PROVABILITY-001`.
- **Fix:** Added FALSIFY-STL-013 for `OBLIG-STL-RESOLVED-CONFIG-ONE-WAY`, which had a real test
  behind it already. Merging obligations to reach parity was rejected — the obligation count is
  the honest one.
- **Files modified:** `contracts/setfit-train-lifecycle-v1.yaml`
- **Verification:** `pv validate` rc=0; `make contract-audit-phase3` rc=0.
- **Committed in:** `1a223ab9f`

---

**Total deviations:** 9 auto-fixed (2 x Rule 1, 3 x Rule 2, 4 x Rule 3). No Rule 4 escalations.
**Impact on plan:** Every one was necessary for correctness or for the plan's own acceptance
criteria to be satisfiable. Two (the float parser, the head reload door) were prerequisites
without which the plan's headline claims could not have been made at all. No scope creep: the
aprender-core additions are the minimum the reload surface needs, and each is documented as
such at its declaration.

## Issues Encountered

- **The full-pipeline test suite is fast; the compile is not.** A `cargo test ... bundle_` run is
  ~6 s of tests behind ~40 s of compilation. An early run that appeared to hang for ten minutes
  was compilation plus a multi-megabyte `assert_eq!` on two `Vec<u8>`; `assert_bytes_eq` now
  reports the first differing offset and a 120-character window instead.
- **`cargo check -p aprender-train --no-default-features --features setfit` is RED**, with 8
  `presentar_terminal` errors under `src/monitor/tui/`. This is pre-existing and documented as
  D-ITEM-05: `src/monitor/mod.rs:45` declares `pub mod tui;` unconditionally while its dependency
  is `tui`-gated. The honest gate is `make setfit-feature-matrix`'s leg (a), a two-sided DIFF
  that asserts the diagnostics are byte-identical with and without `setfit`. **Measured: rc=0,
  "identical with and without setfit (control rc=101, setfit rc=101)".**
- **`cargo clippy -p aprender-core --lib --features setfit -- -D warnings` is RED** for
  pre-existing arm64 reasons. Scoped with `--no-deps`, `aprender-core` has exactly ONE finding —
  `demo/reliable/performance.rs:126`, unreachable expression on the aarch64 branch — in a file
  this plan does not touch. **Control: `git diff <base> HEAD -- crates/aprender-core/src/demo/`
  is EMPTY.** This matches STATE.md's re-measurement at 02-08 ("core 1"). Recorded as D-ITEM-07
  in the phase's `deferred-items.md` with the full scoped table. `aprender-train --no-deps` is
  rc=0 with and without `--tests`.

## Verification

| Command | Result |
|---------|--------|
| `cargo test -p aprender-core --lib --features setfit tokenizer_bytes` | rc=0, **4 passed** |
| `cargo test -p aprender-train --lib --features setfit bundle_` | rc=0, **29 passed** |
| `cargo test -p aprender-train --lib --features setfit verify_` | rc=0, **42 passed** |
| `cargo test -p aprender-core --lib --features conformance-fixtures setfit` | rc=0, **183 passed** |
| `cargo test -p aprender-train --lib --features setfit setfit` | rc=0, **184 passed**, 1 ignored |
| `cargo test -p aprender-core --lib --features conformance-fixtures` (whole crate) | rc=0, **14,366 passed** |
| `cargo test -p aprender-train --lib --features setfit -- --test-threads=1` (whole crate) | 7,793 passed, **24 failed** — exactly the known-red baseline, name for name |
| `pv validate contracts/setfit-train-lifecycle-v1.yaml` | rc=0, 0 errors, 0 warnings |
| `make contract-audit-phase3` | rc=0 — 13 equations, 13 bound, 13 obligations, 13 falsification tests |
| `make setfit-feature-matrix` | rc=0 |
| `cargo clippy -p aprender-train --lib --features setfit --no-deps -- -D warnings` | rc=0 |
| `cargo clippy -p aprender-train --lib --features setfit --no-deps --tests -- -D warnings` | rc=0 |
| `cargo clippy -p aprender-core --lib --features setfit --no-deps -- -D warnings` | rc=101, one pre-existing finding (D-ITEM-07) |
| `grep -icE '\bapr\b' .../bundle.rs` | **0** |
| `grep -rc 'TryFrom<ResolvedConfigRecord>' .../setfit/` | nothing (clean) |
| `grep -c -- '--lib'` in the contract (via `rtk proxy`) | **26** = 13 tests x 2 command lines |

The known-red 24 are the phase's documented baseline (`known-red-baseline.md`): 21 `gpu::` and 3
`prune::snapshot_tests`. Zero failures under `setfit` or `classification`.

## TDD Gate Compliance

Task 2 carries `tdd="true"` and both gates are in the log:

- **RED** — `ead8cf252` `test(03-08): ...`. 40 passed, 2 failed. The failures are the EchoCodec
  negative (which returned `Ok` and minted the final state) and its faithful-codec control.
- **GREEN** — `2ef3a39b6` `feat(03-08): ...`. 42 passed.
- **REFACTOR** — none; the clippy-driven changes in the GREEN commit were behaviour-preserving
  and are described there.

Task 1 and Task 3 are not TDD tasks, and their tests were written alongside their
implementations because the interfaces did not exist to compile a test against — the
produced-before-consumed constraint 02-09 also recorded. Their gates are the falsification tests
in the contract plus the induced-failure measurements documented above.

## Known Stubs

None. Every field of the bundle is populated from the encoder, head, configuration or evidence
that produced it, and the bit-identical rebuild test is what makes that checkable rather than
claimed.

## Threat Flags

None. The plan's `<threat_model>` assigns `mitigate` to T-3-26, T-3-35, T-3-38, T-3-41 and
T-3-59, and each is discharged above (typed rejections and the tokenizer-hash check; the pure
sealed codec plus the closure check; the recorded accessors plus the perturbed-consumption
companion; the four bounds; the completeness list plus the bit-identical rebuild). No new
network endpoint, auth path, file access or trust-boundary schema was introduced — this plan
adds no I/O at all, which is exactly why the durability limitation is documented rather than
claimed.

## Next Steps

- **03-09** takes the SelectionLock / CanonicalTestToken work that was originally this plan's
  Task 3, plus the trybuild non-constructibility proof.
- **03-10** consumes the ten accessors from OUTSIDE the crate and is the plan that closes TRN-01
  and TRN-06 at the "a user can" tier. It constructs `SerdeJsonCodec` directly, which is why the
  type is `pub` with a `pub const fn new()`.
- **Phase 4** implements `SetFitCodec` for the APR format. Two things to read first: the
  canonical-serialization clause of `reload_verify_roundtrip` (a non-canonical writer fails
  `ReloadNotFromBytes` and must amend the contract deliberately), and the sealing consequence
  above (the codec lands as an adapter module inside `aprender-train`; the format stays in
  `aprender-core`).

## Self-Check: PASSED

All five created files exist on disk, and all five commits are in `git log --all`:

| Artifact | Result |
|----------|--------|
| `crates/aprender-train/src/train/setfit/bundle.rs` | FOUND |
| `crates/aprender-train/src/train/setfit/bundle_tests.rs` | FOUND |
| `crates/aprender-train/src/train/setfit/verify.rs` | FOUND |
| `crates/aprender-train/src/train/setfit/verify_tests.rs` | FOUND |
| `.planning/phases/03-faithful-two-stage-trainer-and-head/03-08-SUMMARY.md` | FOUND |
| `6599fbc4b` Task 1 | FOUND |
| `ead8cf252` Task 2 RED | FOUND |
| `2ef3a39b6` Task 2 GREEN | FOUND |
| `1a223ab9f` Task 3 | FOUND |
| `28a8881e6` SUMMARY | FOUND |

`git diff --diff-filter=D` across all five commits is empty: no tracked file was deleted.
`git status --short` is clean; no generated artifact is left untracked (the `.snap.new` files
the known-red `prune::snapshot_tests` produce are gitignored).

STATE.md and ROADMAP.md are deliberately untouched — this plan ran as a parallel worktree
executor and the orchestrator owns those writes.
