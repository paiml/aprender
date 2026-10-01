//! `SetFitBundle` tests (plan 03-08, TRN-01).
//!
//! Every test name starts `bundle_`, which is the filter plan 03-08's Task 1
//! verification runs. The prefix returned zero pre-existing matches in `crates/`
//! before it was chosen, so the filter selects exactly this file.
//!
//! # What the headline test is
//!
//! `bundle_rebuilds_a_bit_identical_encoder_from_bytes_alone` is the one that
//! makes "reload" mean something. Everything else here — byte stability, the four
//! limits, the typed rejections — bounds and protects a claim that test is the
//! only one actually making.

use std::collections::BTreeMap;

use aprender::setfit::SetFitMiniLm;
use aprender_contrastive_data::ledger::AccessLedger;
use aprender_contrastive_data::select::{FewShotSelector, Selection, SelectionConfig};

use super::super::test_fixtures as fx;
use super::*;
use crate::train::setfit::{HeadFitted, SetFitRun};

/// The format identifier the serde codec owns (plan 03-08 task 2 re-states it).
const FIXTURE_FORMAT_ID: &str = "setfit-serde-json-v1";

/// A complete calibrated pipeline: prepare -> tune_encoder -> fit_head.
fn head_fitted_run() -> SetFitRun<HeadFitted> {
    fx::head_fitted_run(fx::calibrated_variant())
}

/// The bundle a finished fixture run produces, through the shipped assembly path.
fn fixture_bundle(run: &SetFitRun<HeadFitted>) -> SetFitBundle {
    bundle_of(run, run.selection())
}

/// The same assembly, at a caller-chosen selection.
///
/// Exists so the provenance tests can vary the ONE input provenance is read off
/// without paying for a second `tune_encoder` + `fit_head` pipeline. Every other
/// argument is the run's own, so a difference in the record is attributable to the
/// selection alone.
fn bundle_of(run: &SetFitRun<HeadFitted>, selection: &Selection) -> SetFitBundle {
    SetFitBundle::from_run_parts(
        FIXTURE_FORMAT_ID,
        run.encoder(),
        run.evidence().head(),
        run.evidence().ordered_labels(),
        selection,
        run.config(),
        run.evidence().passed().summary(),
    )
    .expect("a finished run must assemble into a bundle")
}

/// A selection over the fixture corpus, drawn independently of any run.
///
/// Built through the shipped door (`FewShotSelector::select` over
/// `fx::synthetic_dataset`), so a selection produced here is the same KIND of
/// object a run carries — the point of the determinism test below is that the same
/// arguments reproduce the same hashes, which is only meaningful if this path is
/// the production one.
fn selection_at(root_seed: u64, shots_per_class: u32) -> Selection {
    let mut ledger = AccessLedger::new();
    let dataset = fx::synthetic_dataset(&mut ledger);
    FewShotSelector::select(&dataset, &SelectionConfig { root_seed, shots_per_class }, &mut ledger)
        .expect("the synthetic corpus must support this selection")
}

/// The probe texts: the fixture's test split, which the run never trained on.
fn probe_texts(run: &SetFitRun<HeadFitted>) -> Vec<String> {
    run.dataset().test().rows().iter().map(|r| r.input.clone()).collect()
}

/// `[B, H]` embeddings as raw bit patterns, so the comparison admits no epsilon.
fn embedding_bits(model: &SetFitMiniLm, texts: &[&str]) -> Vec<u32> {
    model
        .encode_texts(texts)
        .expect("probe texts must encode")
        .data()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

/// Byte equality that reports the FIRST difference and a window around it.
///
/// `assert_eq!` on two multi-megabyte `Vec<u8>` renders both in full, which is
/// several MB of scrollback in place of the one offset that matters — and the
/// rendering is a decimal byte list, so the JSON is unreadable even after paging
/// through it.
fn assert_bytes_eq(left: &[u8], right: &[u8], what: &str) {
    if left == right {
        return;
    }
    let at = left
        .iter()
        .zip(right.iter())
        .position(|(a, b)| a != b)
        .unwrap_or_else(|| left.len().min(right.len()));
    let from = at.saturating_sub(60);
    let window =
        |bytes: &[u8]| String::from_utf8_lossy(&bytes[from..bytes.len().min(at + 60)]).into_owned();
    panic!(
        "{what}: byte streams differ at offset {at} (lengths {} vs {})\n  left:  ...{}...\n  \
         right: ...{}...",
        left.len(),
        right.len(),
        window(left),
        window(right),
    );
}

/// Rebuild a model from a bundle, through the public reload door only.
fn rebuild(bundle: &SetFitBundle) -> SetFitMiniLm {
    SetFitMiniLm::from_bundle_parts(
        &bundle.tokenizer_bytes().expect("tokenizer bytes decode"),
        bundle.architecture(),
        bundle.named_tensors().expect("tensors decode"),
        bundle.root_seed(),
    )
    .expect("a complete bundle must rebuild a model")
}

// ===========================================================================================
// The headline: a bundle is a reload, not a description
// ===========================================================================================

/// A model rebuilt from the bundle's bytes alone embeds BIT-IDENTICALLY.
///
/// This is the criterion that separates a persistence artifact from a manifest. A
/// bundle missing the tokenizer bytes cannot tokenize at all; one missing the
/// vocabulary remap gathers the wrong embedding rows for every token and still
/// produces plausible-looking floats; one missing the LayerNorm epsilon or the
/// pooling policy produces embeddings that are close and not equal. Bit equality
/// on the raw `f32` patterns is the only comparison that fails on all three.
#[test]
fn bundle_rebuilds_a_bit_identical_encoder_from_bytes_alone() {
    let run = head_fitted_run();
    let bundle = fixture_bundle(&run);

    // Through the WIRE, not through the in-memory value: a rebuild from the live
    // struct would prove nothing about the bytes.
    let bytes = bundle.to_canonical_bytes().expect("canonical bytes");
    let reloaded = SetFitBundle::from_canonical_bytes(&bytes).expect("canonical bytes parse");
    let rebuilt = rebuild(&reloaded);

    let owned = probe_texts(&run);
    let texts: Vec<&str> = owned.iter().map(String::as_str).collect();
    assert!(!texts.is_empty(), "the fixture must have probe rows");

    let original = embedding_bits(run.encoder(), &texts);
    let after = embedding_bits(&rebuilt, &texts);
    assert!(!original.is_empty(), "a probe must produce embeddings");
    assert_eq!(
        original, after,
        "a model rebuilt from the bundle's bytes must embed BIT-IDENTICALLY; a difference \
         here means the bundle is not complete enough for `reload` to mean anything",
    );

    // And the rebuilt half agrees on identity as well as on arithmetic.
    assert_eq!(rebuilt.tokenizer_sha256(), run.encoder().tokenizer_sha256());
    assert_eq!(rebuilt.architecture_fingerprint(), run.encoder().architecture_fingerprint(),);
    assert_eq!(rebuilt.root_seed(), run.encoder().root_seed());
}

/// The head rebuilt from stored coefficients predicts identically.
///
/// The encoder half above and this half together are what "the artifact is the
/// model" means: same embeddings AND same probabilities from the same bytes.
#[test]
fn bundle_rebuilds_a_head_that_predicts_identically() {
    let run = head_fitted_run();
    let bundle = fixture_bundle(&run);
    let bytes = bundle.to_canonical_bytes().expect("canonical bytes");
    let reloaded = SetFitBundle::from_canonical_bytes(&bytes).expect("canonical bytes parse");

    let (labels, n_features, weights, intercepts) =
        reloaded.head_parts().expect("head parts decode");
    let rebuilt_head = MultinomialLogisticRegression::from_stored_coefficients(
        labels, n_features, weights, intercepts,
    )
    .expect("stored coefficients must rebuild a head");

    let owned = probe_texts(&run);
    let texts: Vec<&str> = owned.iter().map(String::as_str).collect();
    let embedded = run.encoder().encode_texts(&texts).expect("probe encodes");
    let hidden = embedded.shape()[1];
    let rows: Vec<Vec<f32>> = embedded.data().chunks(hidden).map(<[f32]>::to_vec).collect();

    let before = run.evidence().head().predict_proba(&rows).expect("the live head predicts");
    let after = rebuilt_head.predict_proba(&rows).expect("the rebuilt head predicts");
    assert!(!before.is_empty(), "a probe must produce predictions");
    assert_eq!(
        before, after,
        "the head rebuilt from the bundle's stored coefficients must produce the same \
         probabilities as the head that was serialized",
    );
    assert_eq!(
        run.evidence().head().labels(),
        rebuilt_head.labels(),
        "the label map indexes the weight rows; a reordering would silently relabel every \
         prediction",
    );
    assert!(
        rebuilt_head.report().is_none(),
        "a reloaded head must not claim a convergence status no optimizer produced here",
    );
}

// ===========================================================================================
// Canonicity
// ===========================================================================================

/// Serializing twice produces identical bytes, and the round trip is closed.
///
/// `serialize(deserialize(b)) == b` is not a nicety: plan 03-08's verify policy
/// re-serializes what a codec hands back and compares it against the bytes it
/// hashed, so a bundle whose wire form is not canonical would make every honest
/// codec look like a cheating one.
#[test]
fn bundle_round_trip_is_byte_stable_and_closed() {
    let run = head_fitted_run();
    let bundle = fixture_bundle(&run);

    let first = bundle.to_canonical_bytes().expect("first serialize");
    let second = bundle.to_canonical_bytes().expect("second serialize");
    assert_bytes_eq(
        &first,
        &second,
        "two serializations of one bundle must be byte-identical (a map with unspecified \
         iteration order or a formatted float would break this)",
    );

    let parsed = SetFitBundle::from_canonical_bytes(&first).expect("parse");
    let reserialized = parsed.to_canonical_bytes().expect("re-serialize");
    assert_bytes_eq(
        &first,
        &reserialized,
        "re-serializing a parsed bundle must reproduce the input bytes exactly",
    );
    assert!(parsed == bundle, "the parsed value must equal the original");
}

/// Two independently built pipelines produce byte-identical bundles.
///
/// Byte stability of ONE value only says serialization is a function. This says
/// the whole path — pair stream, tuning, encode-once, L-BFGS, assembly — is one
/// too, which is what a reproducibility artifact has to be.
#[test]
fn bundle_two_identical_pipelines_agree_byte_for_byte() {
    let first = fixture_bundle(&head_fitted_run()).to_canonical_bytes().expect("first bundle");
    let second = fixture_bundle(&head_fitted_run()).to_canonical_bytes().expect("second bundle");
    assert_bytes_eq(&first, &second, "two identical runs must produce byte-identical bundles");
}

// ===========================================================================================
// Completeness — the normative field list, asserted field by field
// ===========================================================================================

/// Every field of the normative completeness list is declared, by name.
///
/// A source assertion rather than a round-trip assertion on purpose: a round trip
/// preserves whatever is there, so it is green for a bundle that never carried
/// the tokenizer bytes at all. This is the check that notices a field being
/// REMOVED, which is the only way completeness regresses.
#[test]
fn bundle_declares_every_field_of_the_normative_completeness_list() {
    let src = include_str!("bundle.rs");
    for field in [
        "pub(crate) schema_version: u32,",
        "pub(crate) format_id: String,",
        "pub(crate) architecture: EncoderArchitecture,",
        "pub(crate) tokenizer_bytes_hex: String,",
        "pub(crate) pooling: String,",
        "pub(crate) normalization: String,",
        "pub(crate) l2_epsilon: f32,",
        "pub(crate) truncation_max_sequence_length: u32,",
        "pub(crate) padding_mode: String,",
        "pub(crate) max_length: u32,",
        "pub(crate) root_seed: u64,",
        "pub(crate) tensors: BTreeMap<String, BundleTensor>,",
        "pub(crate) head_weights_hex: String,",
        "pub(crate) head_intercepts_hex: String,",
        "pub(crate) head_n_features: usize,",
        "pub(crate) ordered_labels: Vec<String>,",
        "pub(crate) requested_config: SetFitTrainConfig,",
        "pub(crate) resolved_config: ResolvedConfigRecord,",
        "pub(crate) evidence: EvidenceSummary,",
        // Field 20 (plan 04-13). The phase-3 contract's `bundle_completeness`
        // enumerates the nineteen above as a normative MINIMUM ("a field MISSING
        // is a finding"); `setfit-apr-v1`'s bijection table enumerates all twenty
        // and recovers every one of them from artifact bytes.
        "pub(crate) provenance: ProvenanceRecord,",
    ] {
        assert!(
            src.contains(field),
            "the completeness list requires `{field}`; a bundle missing it cannot rebuild \
             the model it claims to describe",
        );
    }
    assert!(
        src.contains("#[serde(deny_unknown_fields)]"),
        "an unknown field must be a rejection, not a silently ignored payload",
    );
}

/// The architecture record and the policy fields carry the values the model has.
///
/// The source assertion above proves the fields EXIST; this proves they are
/// populated from the encoder rather than from a default or a literal.
#[test]
fn bundle_records_the_architecture_and_policy_the_encoder_actually_uses() {
    let run = head_fitted_run();
    let bundle = fixture_bundle(&run);
    let arch = bundle.architecture();

    assert_eq!(arch.hidden, 64, "the fixture slice is 64-wide");
    assert_eq!(arch.num_layers, 2);
    assert_eq!(arch.heads, 2);
    assert_eq!(arch.head_dim, arch.hidden / arch.heads);
    assert_eq!(arch.hidden_act, aprender::setfit::PINNED_ACTIVATION);
    assert_eq!(arch.tokenizer_sha256, run.encoder().tokenizer_sha256());
    assert!(
        arch.vocab_remap.is_some(),
        "the fixture is a SLICE; without its remap a rebuild gathers the wrong rows",
    );
    assert_eq!(
        arch.vocab_remap.as_ref().map(Vec::len),
        Some(arch.vocab),
        "the remap must have one row per embedding row",
    );

    assert_eq!(bundle.pooling, aprender::setfit::POOLING_POLICY);
    assert_eq!(bundle.normalization, aprender::setfit::NORMALIZATION_POLICY);
    assert_eq!(bundle.l2_epsilon, aprender::setfit::L2_EPS);
    assert_eq!(bundle.padding_mode, aprender::setfit::PADDING_MODE);
    assert_eq!(
        bundle.truncation_max_sequence_length as usize,
        aprender::setfit::MAX_SEQUENCE_LENGTH,
    );
    assert_eq!(bundle.max_length, run.config().requested().max_length());
    assert_eq!(bundle.root_seed, run.encoder().root_seed());

    // Every named parameter of the live encoder appears, with its shape.
    let live: BTreeMap<String, Vec<usize>> = run
        .encoder()
        .named_parameters()
        .into_iter()
        .map(|(name, t)| (name, t.shape().to_vec()))
        .collect();
    let stored = bundle.named_tensors().expect("tensors decode");
    assert!(!live.is_empty(), "the encoder must have parameters");
    assert_eq!(live.len(), stored.len(), "every named encoder parameter must be in the bundle",);
    for (name, shape) in &live {
        let (stored_shape, data) =
            stored.get(name).unwrap_or_else(|| panic!("bundle is missing tensor `{name}`"));
        assert_eq!(stored_shape, shape, "{name}: shape");
        assert_eq!(data.len(), shape.iter().product::<usize>(), "{name}: element count",);
    }

    // Both configuration forms, and the evidence with its binding hash.
    assert_eq!(&bundle.requested_config, run.config().requested());
    assert_eq!(bundle.resolved_config.resolved_device, "cpu");
    assert!(
        !bundle.evidence().table_hash.is_empty(),
        "the summary must carry the hash that binds it to its table",
    );
    assert_eq!(bundle.ordered_labels(), run.evidence().ordered_labels());
    assert_eq!(bundle.format_id(), FIXTURE_FORMAT_ID);
    assert_eq!(bundle.schema_version(), BUNDLE_SCHEMA_VERSION);
}

/// The resolved configuration travels as a RECORD and never as a capability.
///
/// T-3-50 (plan 03-03) is that a probed device must not arrive from a file. This
/// is the wave where the pressure to break it appears — a bundle that embedded
/// [`ResolvedSetFitConfig`] would not derive `Deserialize`, and the cheapest fix
/// for that compile error is the one that retires the criterion.
#[test]
fn bundle_resolved_config_is_provenance_with_no_reconstruction_path() {
    let bundle_src = include_str!("bundle.rs");
    // Assembled rather than written out: the same scan is run as a `grep` over
    // this whole directory, and a test that spells the forbidden token is a file
    // in that directory containing the forbidden token.
    let forbidden = format!("TryFrom<{}>", "ResolvedConfigRecord");
    assert!(
        !bundle_src.contains(&forbidden),
        "the record is provenance; a conversion back to a runtime resolved config would \
         reconstruct a probed device from bytes",
    );
    assert!(
        !bundle_src.contains("resolved_config: ResolvedSetFitConfig"),
        "the bundle must carry the RECORD, not the runtime type",
    );

    // The ATTRIBUTE lines immediately above the declaration, doc comments
    // excluded. The doc comment there says the words "NOT `Deserialize`,
    // deliberately", so a naive substring scan over the preamble matches the
    // sentence that documents the invariant and reports it as a violation.
    let config_src = include_str!("config.rs");
    let attributes: Vec<&str> = config_src
        .lines()
        .take_while(|line| !line.contains("pub struct ResolvedSetFitConfig"))
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .take_while(|line| line.trim_start().starts_with("#["))
        .collect();
    assert!(
        !attributes.is_empty(),
        "ResolvedSetFitConfig must carry derive attributes for this scan to mean anything",
    );
    assert!(
        attributes.iter().any(|line| line.contains("Serialize")),
        "the scan must be looking at the right declaration: {attributes:?}",
    );
    assert!(
        !attributes.iter().any(|line| line.contains("Deserialize")),
        "T-3-50: ResolvedSetFitConfig must remain Serialize-but-not-Deserialize, so a probed \
         device cannot arrive from a file; its attributes are {attributes:?}",
    );
}

// ===========================================================================================
// Typed rejections
// ===========================================================================================

/// A truncated payload is a typed parse error naming the failure.
#[test]
fn bundle_truncated_payload_is_a_typed_parse_error() {
    let bundle = fixture_bundle(&head_fitted_run());
    let bytes = bundle.to_canonical_bytes().expect("canonical bytes");
    for cut in [1, bytes.len() / 3, bytes.len() / 2, bytes.len() - 1] {
        match SetFitBundle::from_canonical_bytes(&bytes[..cut]) {
            Err(BundleError::Serialization { context, detail }) => {
                assert_eq!(context, "parse");
                assert!(!detail.is_empty(), "a parse failure must name what went wrong",);
            }
            other => panic!("truncation at {cut} must be a typed parse error, got {other:?}"),
        }
    }
}

/// A single corrupted STRUCTURAL byte is a typed parse error.
///
/// Split from the hex-payload case below deliberately. The two corruptions surface
/// at different places — one at parse, one at decode — and a test that accepted
/// "some error happened" for both would not notice if one of them stopped being
/// detected at all.
#[test]
fn bundle_structural_byte_corruption_is_a_typed_parse_error() {
    let bundle = fixture_bundle(&head_fitted_run());
    let bytes = bundle.to_canonical_bytes().expect("canonical bytes");

    // Byte 0 is `{`. Replacing it cannot leave a parseable document.
    let mut corrupted = bytes.clone();
    corrupted[0] = b'~';
    match SetFitBundle::from_canonical_bytes(&corrupted) {
        Err(BundleError::Serialization { context, detail }) => {
            assert_eq!(context, "parse");
            assert!(
                detail.contains("line") || detail.contains("column"),
                "the parse failure should name its position, got `{detail}`",
            );
        }
        other => panic!("a corrupted structural byte must be a typed parse error, got {other:?}"),
    }
}

/// A corrupted byte INSIDE a hex payload is a typed decode error naming the field.
#[test]
fn bundle_hex_payload_corruption_is_typed_and_names_the_field() {
    let bundle = fixture_bundle(&head_fitted_run());
    let text = String::from_utf8(bundle.to_canonical_bytes().expect("canonical bytes"))
        .expect("canonical bytes are UTF-8 JSON");

    // Land inside the tokenizer's hex payload and put a non-hex character there.
    let marker = "\"tokenizer_bytes_hex\":\"";
    let at = text.find(marker).expect("the field must exist") + marker.len();
    let mut corrupted = text.into_bytes();
    corrupted[at + 4] = b'z';

    let parsed = SetFitBundle::from_canonical_bytes(&corrupted)
        .expect("a hex payload with a bad character is still valid JSON");
    match parsed.tokenizer_bytes() {
        Err(BundleError::MalformedHexPayload { field, reason }) => {
            assert_eq!(field, "tokenizer_bytes");
            assert!(!reason.is_empty());
        }
        other => panic!("a corrupted hex payload must be typed, got {other:?}"),
    }
}

/// A version-bumped payload is refused before any field is trusted.
#[test]
fn bundle_version_bump_is_a_typed_version_error() {
    let bundle = fixture_bundle(&head_fitted_run());
    let text = String::from_utf8(bundle.to_canonical_bytes().expect("canonical bytes"))
        .expect("canonical bytes are UTF-8 JSON");
    // Both sides read the CONSTANT. A literal `"schema_version":1` on the left
    // silently stopped matching the moment 04-13 bumped the constant to 2, and a
    // `replacen` that matches nothing leaves the payload valid — the test would
    // then have asserted that a CURRENT-version bundle is refused, which is the
    // opposite of what it claims, and it would have failed for the right-looking
    // reason. The `assert_ne!` below is what catches that class.
    let bumped = text.replacen(
        &format!("\"schema_version\":{BUNDLE_SCHEMA_VERSION}"),
        &format!("\"schema_version\":{}", BUNDLE_SCHEMA_VERSION + 1),
        1,
    );
    assert_ne!(bumped, text, "the replacement must have applied");

    match SetFitBundle::from_canonical_bytes(bumped.as_bytes()) {
        Err(BundleError::UnsupportedSchemaVersion { got, supported }) => {
            assert_eq!(got, BUNDLE_SCHEMA_VERSION + 1);
            assert_eq!(supported, BUNDLE_SCHEMA_VERSION);
        }
        other => panic!("a future schema must be refused, got {other:?}"),
    }
}

/// A tokenizer-hash mismatch is refused BEFORE a single tensor is installed.
#[test]
fn bundle_tokenizer_hash_mismatch_is_a_typed_rejection() {
    let run = head_fitted_run();
    let mut bundle = fixture_bundle(&run);
    let genuine = bundle.architecture.tokenizer_sha256.clone();
    bundle.architecture.tokenizer_sha256 = "0".repeat(64);

    let err = SetFitMiniLm::from_bundle_parts(
        &bundle.tokenizer_bytes().expect("tokenizer bytes decode"),
        bundle.architecture(),
        bundle.named_tensors().expect("tensors decode"),
        bundle.root_seed(),
    )
    .expect_err("a tokenizer whose bytes do not match the record must be refused");
    match err {
        aprender::setfit::SetFitError::TokenizerHashMismatch { expected, got } => {
            assert_eq!(expected, "0".repeat(64));
            assert_eq!(got, genuine, "the error must name the digest of the BYTES supplied");
        }
        other => panic!("expected a tokenizer-hash mismatch, got {other:?}"),
    }
}

// ===========================================================================================
// The reload-door guards, each shown BITING
//
// These three refusals were added by a review pass that shipped no tests with them. Each one
// is a NEW way for a legitimate bundle to be rejected as well as a new way to catch a bad
// one, so a guard that had only ever been observed on the happy path -- where by
// construction it does nothing -- was evidence of neither direction. Every test below pairs
// the refusal with the control that the untampered fixture still passes.
// ===========================================================================================

/// An extra tensor the architecture never names is refused, naming it.
///
/// Nothing else catches this: the shape and element checks only fire on names that ARE read,
/// `deny_unknown_fields` rejects unknown struct KEYS and says nothing about entries of the
/// tensor MAP, and the verify policy's round-trip closure check re-serializes the residue
/// happily because it is genuinely part of the hashed bytes.
#[test]
fn bundle_an_unreferenced_tensor_is_refused_naming_it() {
    let run = head_fitted_run();
    let bundle = fixture_bundle(&run);

    // Control: untouched, the same bundle rebuilds.
    let _ = rebuild(&bundle);

    let mut tensors = bundle.named_tensors().expect("tensors decode");
    assert!(
        tensors.insert("attacker.payload".to_string(), (vec![2], vec![1.0, 2.0])).is_none(),
        "the injected name must not already exist",
    );

    let err = SetFitMiniLm::from_bundle_parts(
        &bundle.tokenizer_bytes().expect("tokenizer bytes decode"),
        bundle.architecture(),
        tensors,
        bundle.root_seed(),
    )
    .expect_err("a bundle carrying a tensor the architecture does not name must be refused");
    let rendered = err.to_string();
    assert!(
        rendered.contains("attacker.payload"),
        "the refusal must NAME the unreferenced tensor, got `{rendered}`",
    );
}

/// An architecture declaring more layers than the supplied tensors could satisfy is refused
/// TYPED, rather than aborting the process on the allocation.
///
/// `assemble` builds a `Vec` of capacity `num_layers` before it reads a single tensor, and
/// the bundle's four contracted bounds cover input length, tensor count and element counts —
/// none of them covers this struct. So the record is an allocation request arriving ahead of
/// every validation the reader has.
#[test]
fn bundle_an_absurd_layer_count_is_refused_before_it_allocates() {
    let run = head_fitted_run();
    let bundle = fixture_bundle(&run);
    let tensors = bundle.named_tensors().expect("tensors decode");

    let mut arch = bundle.architecture().clone();
    arch.num_layers = 1 << 60;

    let err = SetFitMiniLm::from_bundle_parts(
        &bundle.tokenizer_bytes().expect("tokenizer bytes decode"),
        &arch,
        tensors,
        bundle.root_seed(),
    )
    .expect_err("a layer count no supplied tensor set could satisfy must be refused");
    match err {
        aprender::setfit::SetFitError::ImportConfigMismatch { field, .. } => {
            assert_eq!(field, "num_layers", "the refusal must blame the dimension that is absurd");
        }
        other => panic!("expected a typed ImportConfigMismatch, got {other:?}"),
    }
}

/// A bundle recording a pooling/normalization policy this build does not implement is refused.
///
/// The rebuild applies whatever policy is COMPILED IN, so without this a payload written
/// under `pooling = "cls"` would be rebuilt under ours and produce different embeddings from
/// the same weights — and `run_verify_policy` could not notice, because both sides of its
/// comparison are the same build.
#[test]
fn bundle_a_foreign_pooling_policy_is_refused_naming_the_field() {
    let run = head_fitted_run();
    let mut bundle = fixture_bundle(&run);

    // Control first: as written, the fixture agrees with this build.
    bundle
        .check_policy_matches_this_build()
        .expect("the fixture must match the build that wrote it");

    let genuine = bundle.pooling.clone();
    assert_ne!(genuine, "cls", "the perturbation must actually change the policy");
    bundle.pooling = "cls".to_string();

    match bundle.check_policy_matches_this_build() {
        Err(BundleError::PolicyMismatch { field, expected, got }) => {
            assert_eq!(field, "pooling");
            assert_eq!(expected, genuine, "the error must name what this build implements");
            assert_eq!(got, "cls", "and what the payload recorded");
        }
        other => panic!("expected a typed PolicyMismatch, got {other:?}"),
    }
}

/// A missing tensor is a typed error naming the tensor.
#[test]
fn bundle_missing_tensor_is_a_typed_rejection_naming_it() {
    let run = head_fitted_run();
    let bundle = fixture_bundle(&run);
    let mut tensors = bundle.named_tensors().expect("tensors decode");
    let dropped = "embeddings.word_embeddings.weight".to_string();
    assert!(tensors.remove(&dropped).is_some(), "the tensor to drop must have been present",);

    let err = SetFitMiniLm::from_bundle_parts(
        &bundle.tokenizer_bytes().expect("tokenizer bytes decode"),
        bundle.architecture(),
        tensors,
        bundle.root_seed(),
    )
    .expect_err("a bundle missing a tensor must be refused");
    let rendered = err.to_string();
    assert!(
        rendered.contains(&dropped),
        "the refusal must name the missing tensor, got `{rendered}`",
    );
}

// ===========================================================================================
// The four bounds
// ===========================================================================================

/// Bounds tight enough that a fixture bundle trips exactly one of them.
fn only_input_bytes_bound(limit: u64) -> BundleLimits {
    BundleLimits::tiny(limit, u64::MAX, u64::MAX, u64::MAX)
}

/// The input-length bound bites, on the raw slice, before serde is handed anything.
#[test]
fn bundle_limit_input_bytes_is_enforced_before_the_parse() {
    let bundle = fixture_bundle(&head_fitted_run());
    let bytes = bundle.to_canonical_bytes().expect("canonical bytes");
    let limit = (bytes.len() - 1) as u64;

    match SetFitBundle::from_canonical_bytes_within(&bytes, &only_input_bytes_bound(limit)) {
        Err(BundleError::BundleLimitExceeded { what, limit: reported, observed }) => {
            assert_eq!(what, "input_bytes");
            assert_eq!(reported, limit);
            assert_eq!(observed, bytes.len() as u64);
        }
        other => panic!("an oversized payload must be refused, got {other:?}"),
    }

    // CONTROL: one more byte of headroom and the same payload is accepted, so the
    // bound is a bound rather than a refusal of everything.
    SetFitBundle::from_canonical_bytes_within(&bytes, &only_input_bytes_bound(bytes.len() as u64))
        .expect("a payload exactly at the bound must be accepted");
}

/// The tensor-count bound bites.
#[test]
fn bundle_limit_tensor_count_is_enforced() {
    let bundle = fixture_bundle(&head_fitted_run());
    let bytes = bundle.to_canonical_bytes().expect("canonical bytes");
    let count = bundle.tensors.len() as u64;
    assert!(count > 1, "the fixture must carry several tensors");

    let limits = BundleLimits::tiny(u64::MAX, count - 1, u64::MAX, u64::MAX);
    match SetFitBundle::from_canonical_bytes_within(&bytes, &limits) {
        Err(BundleError::BundleLimitExceeded { what, limit, observed }) => {
            assert_eq!(what, "tensor_count");
            assert_eq!(limit, count - 1);
            assert_eq!(observed, count);
        }
        other => panic!("too many tensors must be refused, got {other:?}"),
    }
    SetFitBundle::from_canonical_bytes_within(
        &bytes,
        &BundleLimits::tiny(u64::MAX, count, u64::MAX, u64::MAX),
    )
    .expect("a payload exactly at the bound must be accepted");
}

/// The per-tensor element bound bites, computed from the hex LENGTH.
#[test]
fn bundle_limit_tensor_elements_is_enforced_before_allocation() {
    let bundle = fixture_bundle(&head_fitted_run());
    let bytes = bundle.to_canonical_bytes().expect("canonical bytes");
    let largest = bundle
        .tensors
        .values()
        .map(|t| (t.data_hex.len() / 8) as u64)
        .max()
        .expect("the fixture has tensors");
    assert!(largest > 1, "the largest tensor must have elements to bound");

    let limits = BundleLimits::tiny(u64::MAX, u64::MAX, largest - 1, u64::MAX);
    match SetFitBundle::from_canonical_bytes_within(&bytes, &limits) {
        Err(BundleError::BundleLimitExceeded { what, limit, observed }) => {
            assert_eq!(what, "tensor_elements");
            assert_eq!(limit, largest - 1);
            assert!(observed > limit, "the reported observation must be the offending count",);
        }
        other => panic!("an oversized tensor must be refused, got {other:?}"),
    }
    SetFitBundle::from_canonical_bytes_within(
        &bytes,
        &BundleLimits::tiny(u64::MAX, u64::MAX, largest, u64::MAX),
    )
    .expect("a payload exactly at the bound must be accepted");
}

/// The cumulative element bound bites, and is not implied by the per-tensor one.
#[test]
fn bundle_limit_total_elements_is_enforced() {
    let bundle = fixture_bundle(&head_fitted_run());
    let bytes = bundle.to_canonical_bytes().expect("canonical bytes");
    let total: u64 = bundle.tensors.values().map(|t| (t.data_hex.len() / 8) as u64).sum();
    let largest = bundle
        .tensors
        .values()
        .map(|t| (t.data_hex.len() / 8) as u64)
        .max()
        .expect("the fixture has tensors");
    assert!(
        total > largest,
        "the total must exceed the largest single tensor, or this bound would be the \
         per-tensor bound wearing another name",
    );

    // Per-tensor bound generous, total bound tight: only the cumulative check can fire.
    let limits = BundleLimits::tiny(u64::MAX, u64::MAX, u64::MAX, total - 1);
    match SetFitBundle::from_canonical_bytes_within(&bytes, &limits) {
        Err(BundleError::BundleLimitExceeded { what, limit, observed }) => {
            assert_eq!(what, "total_elements");
            assert_eq!(limit, total - 1);
            assert!(observed > limit);
        }
        other => panic!("an oversized total must be refused, got {other:?}"),
    }
    SetFitBundle::from_canonical_bytes_within(
        &bytes,
        &BundleLimits::tiny(u64::MAX, u64::MAX, u64::MAX, total),
    )
    .expect("a payload exactly at the bound must be accepted");
}

/// The contracted bounds clear the full pinned MiniLM by a stated factor.
///
/// The full-pin figures are COMPUTED from the pinned architecture rather than
/// quoted, so a change to the pin moves this test with it instead of leaving a
/// stale number in a comment.
#[test]
fn bundle_limits_clear_the_full_minilm_figures() {
    // sentence-transformers/all-MiniLM-L6-v2.
    let (hidden, layers, intermediate, vocab, positions, type_vocab) =
        (384_u64, 6_u64, 1536_u64, 30522_u64, 512_u64, 2_u64);

    let embeddings = vocab * hidden + positions * hidden + type_vocab * hidden + 2 * hidden;
    let per_layer = 3 * (hidden * hidden + hidden)   // q, k, v
        + (hidden * hidden + hidden)                 // attention output dense
        + 2 * hidden                                 // attention output LayerNorm
        + (intermediate * hidden + intermediate)     // FFN intermediate
        + (hidden * intermediate + hidden)           // FFN output dense
        + 2 * hidden; // FFN LayerNorm
    let total_elements = embeddings + layers * per_layer;
    let largest_tensor = vocab * hidden;
    let tensor_count = 5 + layers * 16;
    // Eight hex characters per f32, plus JSON structure and the hex tokenizer.
    let projected_bytes = total_elements * 8 + 2 * 466_247 + 1_000_000;

    assert_eq!(total_elements, 22_565_376, "full-pin element total");
    assert_eq!(largest_tensor, 11_720_448, "full-pin largest tensor");
    assert_eq!(tensor_count, 101, "full-pin tensor count");

    assert!(
        MAX_TOTAL_ELEMENTS >= total_elements * 10,
        "MAX_TOTAL_ELEMENTS {MAX_TOTAL_ELEMENTS} must clear the full pin's {total_elements} \
         by at least 10x",
    );
    assert!(
        MAX_ELEMENTS_PER_TENSOR >= largest_tensor * 10,
        "MAX_ELEMENTS_PER_TENSOR {MAX_ELEMENTS_PER_TENSOR} must clear {largest_tensor} by at \
         least 10x",
    );
    assert!(
        MAX_TENSOR_COUNT >= tensor_count * 10,
        "MAX_TENSOR_COUNT {MAX_TENSOR_COUNT} must clear {tensor_count} by at least 10x",
    );
    assert!(
        MAX_BUNDLE_BYTES >= projected_bytes * 2,
        "MAX_BUNDLE_BYTES {MAX_BUNDLE_BYTES} must clear the projected {projected_bytes} by at \
         least 2x",
    );

    // And the contract states the same figures, so the headroom claim it makes is
    // about the numbers this computation produced rather than about numbers a prose
    // paragraph remembers.
    let contracted = contract_bundle_limits();
    assert_eq!(contracted.full_minilm_figures.total_elements, total_elements);
    assert_eq!(contracted.full_minilm_figures.largest_tensor_elements, largest_tensor);
    assert_eq!(contracted.full_minilm_figures.tensor_count, tensor_count);
    assert_eq!(contracted.full_minilm_figures.serialized_bytes, projected_bytes);
}

// ---------------------------------------------------------------------------
// The contracted numbers, PARSED rather than restated
// ---------------------------------------------------------------------------

/// The committed contract, embedded rather than read at runtime.
///
/// `include_str!` for the reason `thresholds.rs` uses it: a test that silently
/// skips when a file is missing is a test that proves nothing on the machine where
/// the file went missing.
const CONTRACT_YAML: &str = include_str!("../../../../../contracts/setfit-train-lifecycle-v1.yaml");

#[derive(Debug, serde::Deserialize)]
struct ContractFile {
    equations: ContractEquations,
}

#[derive(Debug, serde::Deserialize)]
struct ContractEquations {
    bundle_limits: ContractBundleLimits,
}

#[derive(Debug, serde::Deserialize)]
struct ContractBundleLimits {
    limits: ContractLimitValues,
    full_minilm_figures: ContractFullPinFigures,
}

#[derive(Debug, serde::Deserialize)]
struct ContractLimitValues {
    max_bundle_bytes: u64,
    max_tensor_count: u64,
    max_elements_per_tensor: u64,
    max_total_elements: u64,
}

#[derive(Debug, serde::Deserialize)]
struct ContractFullPinFigures {
    serialized_bytes: u64,
    tensor_count: u64,
    largest_tensor_elements: u64,
    total_elements: u64,
}

fn contract_bundle_limits() -> ContractBundleLimits {
    let parsed: ContractFile =
        serde_yaml::from_str(CONTRACT_YAML).expect("the committed contract must deserialize");
    parsed.equations.bundle_limits
}

/// Every limit constant is PARSED from the contract and compared.
///
/// Not a substring search. The contract is deserialized into typed structs and
/// compared field by field, so a number that is right but in the wrong slot is red,
/// and so is a number that appears only in a prose paragraph. Editing either side
/// alone turns this red, which is what makes loosening a bound require a contract
/// edit that `pv diff` flags.
#[test]
fn bundle_limits_match_the_contract() {
    let contracted = contract_bundle_limits().limits;
    assert_eq!(
        contracted.max_bundle_bytes, MAX_BUNDLE_BYTES,
        "max_bundle_bytes disagrees with the contract",
    );
    assert_eq!(
        contracted.max_tensor_count, MAX_TENSOR_COUNT,
        "max_tensor_count disagrees with the contract",
    );
    assert_eq!(
        contracted.max_elements_per_tensor, MAX_ELEMENTS_PER_TENSOR,
        "max_elements_per_tensor disagrees with the contract",
    );
    assert_eq!(
        contracted.max_total_elements, MAX_TOTAL_ELEMENTS,
        "max_total_elements disagrees with the contract",
    );

    // And the shipped bounds object is exactly those four values, so a limit could
    // not be contracted at one number and enforced at another.
    assert_eq!(
        BundleLimits::CONTRACTED,
        BundleLimits::tiny(
            contracted.max_bundle_bytes,
            contracted.max_tensor_count,
            contracted.max_elements_per_tensor,
            contracted.max_total_elements,
        ),
    );
}

/// Record the measured sizes, so the amplification is visible rather than asserted.
///
/// Run with `--nocapture` to read the numbers; the assertions keep it honest when
/// nobody is looking.
#[test]
fn bundle_size_is_recorded_for_the_fixture_and_projected_for_the_pin() {
    let run = head_fitted_run();
    let bundle = fixture_bundle(&run);
    let bytes = bundle.to_canonical_bytes().expect("canonical bytes");

    let tokenizer_len = run.encoder().tokenizer_bytes().len();
    let elements: usize = bundle.tensors.values().map(|t| t.data_hex.len() / 8).sum();
    let raw = elements * 4 + tokenizer_len;

    eprintln!(
        "bundle sizes: tokenizer {tokenizer_len} B | tensors {elements} f32 | raw {raw} B | \
         serialized {} B | amplification {:.2}x",
        bytes.len(),
        bytes.len() as f64 / raw as f64,
    );

    assert_eq!(tokenizer_len, 466_247, "the pinned tokenizer.json is 466,247 bytes",);
    assert_eq!(elements, 110_528, "the fixture slice's f32 element total");
    assert!(
        bytes.len() > raw,
        "the serialized form is necessarily larger than the raw bytes it carries",
    );
}

// ===========================================================================================
// f32 exactness
// ===========================================================================================

/// Adversarial `f32` values survive the wire exactly, bit for bit.
///
/// Bit comparison, not value comparison: `NaN != NaN`, and a codec that turned a
/// quiet NaN into a different quiet NaN — or a subnormal into zero — would pass a
/// value comparison on every other member of the list.
#[test]
fn bundle_f32_round_trip_is_exact_for_adversarial_values() {
    let adversarial: Vec<f32> = vec![
        0.0,
        -0.0,
        f32::MIN_POSITIVE,
        f32::from_bits(1), // smallest subnormal
        -f32::from_bits(1),
        f32::MAX,
        f32::MIN,
        f32::from_bits(0x7F7F_FFFF), // just below MAX
        f32::EPSILON,
        1.0 / 3.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::from_bits(0x7FC0_0001), // a quiet NaN with a payload
    ];

    let run = head_fitted_run();
    let mut bundle = fixture_bundle(&run);
    let name = "adversarial".to_string();
    bundle.tensors.clear();
    bundle.tensors.insert(
        name.clone(),
        BundleTensor { shape: vec![adversarial.len()], data_hex: f32_to_hex(&adversarial) },
    );

    let bytes = bundle.to_canonical_bytes().expect("canonical bytes");
    let parsed = SetFitBundle::from_canonical_bytes(&bytes).expect("parse");
    let decoded = parsed.named_tensors().expect("decode");
    let (shape, values) = decoded.get(&name).expect("the tensor survives");

    assert_eq!(shape, &vec![adversarial.len()]);
    assert_eq!(values.len(), adversarial.len());
    for (index, (before, after)) in adversarial.iter().zip(values.iter()).enumerate() {
        assert_eq!(
            before.to_bits(),
            after.to_bits(),
            "value {index} ({before:?}) changed bit pattern across the wire: {:#010x} -> {:#010x}",
            before.to_bits(),
            after.to_bits(),
        );
    }

    // And the round trip is still closed with those values in it.
    assert_bytes_eq(
        &bytes,
        &parsed.to_canonical_bytes().expect("re-serialize"),
        "non-finite values must not break byte canonicity",
    );
}

/// serde_json parses `f64` back to the value ryu wrote — the `float_roundtrip` proof.
///
/// # Why this is a test and not a comment on a Cargo feature
///
/// `serde_json`'s default float parser is fast and may land one ULP from the value
/// the serializer wrote. It is not a hypothetical: this exact literal is from the
/// fixture's evidence summary, and it round-tripped to `...624` before the
/// `float_roundtrip` feature was enabled, which made the bundle's re-serialization
/// differ from its own input at offset 936,551.
///
/// That matters far beyond this crate's tidiness. Plan 03-08's verify policy hashes
/// a bundle's bytes and then asserts that re-serializing what the codec handed back
/// reproduces them exactly. A one-ULP parse would fail that check for every honest
/// codec, on every real bundle, and the failure would read as "the codec is
/// cheating" rather than "the float parser is approximate".
///
/// The test asserts the BEHAVIOUR rather than the feature flag, because a flag can
/// be present and inert — feature unification, a vendored copy, a future default
/// change — and what the policy depends on is the behaviour.
#[test]
fn bundle_serde_json_parses_floats_round_trip_exactly() {
    for literal in [
        "0.00009142675446597625",
        "0.00020186550022704763",
        "0.0002177860093569460",
        "9.999999960041972e-13",
        "0.1",
        "1e-300",
    ] {
        let parsed: f64 = serde_json::from_str(literal).expect("a JSON number parses");
        let reserialized = serde_json::to_string(&parsed).expect("an f64 serializes");
        let reparsed: f64 = serde_json::from_str(&reserialized).expect("the round trip parses");
        assert_eq!(
            parsed.to_bits(),
            reparsed.to_bits(),
            "`{literal}` -> {parsed:?} -> `{reserialized}` -> {reparsed:?} changed value; \
             serde_json's `float_roundtrip` feature is not engaged, and every bundle's \
             round-trip closure check will fail as a result",
        );
    }
}

/// A hex payload that is not a whole number of `f32`s is refused, naming the field.
#[test]
fn bundle_ragged_hex_payload_is_refused() {
    match hex_to_f32("head_weights", "abcdef") {
        Err(BundleError::MalformedHexPayload { field, reason }) => {
            assert_eq!(field, "head_weights");
            assert!(reason.contains('6'), "the reason should state the length");
        }
        other => panic!("a partial f32 must be refused, got {other:?}"),
    }
}

// ===========================================================================================
// Field 20: provenance (plan 04-13, APR-01 + APR-05)
// ===========================================================================================

/// Every provenance value is READ OFF the selection, never restated from config.
///
/// The distinction is the whole point of the field. `selection_root_seed` and
/// `shots_per_class` both also exist in the run's CONFIGURATION, so a record built
/// from `config` would look identical on the fixture and would be a description of
/// what was ASKED FOR rather than of what was DRAWN. The two hashes and the two
/// fingerprints have no configuration counterpart at all: they exist only on the
/// selection object, which is why they are the ones that make this a measurement.
#[test]
fn bundle_provenance_is_read_off_the_runs_own_selection() {
    let run = head_fitted_run();
    let bundle = fixture_bundle(&run);
    let selection = run.selection();
    let provenance = bundle.provenance();

    assert_eq!(provenance.dataset_fingerprint(), selection.dataset_fingerprint_hex());
    assert_eq!(provenance.validation_split_fingerprint(), selection.validation_fingerprint_hex(),);
    assert_eq!(provenance.selection_semantic_hash(), hex::encode(selection.semantic_hash()));
    assert_eq!(provenance.selection_ledger_hash(), hex::encode(selection.ledger_hash()));
    assert_eq!(provenance.selection_root_seed(), selection.root_seed());
    assert_eq!(provenance.shots_per_class(), selection.shots_per_class());

    // Not a vacuous comparison of two empty strings: a 32-byte hash is 64 hex
    // characters, and every character is lowercase hex.
    for (what, hash) in [
        ("selection_semantic_hash", provenance.selection_semantic_hash()),
        ("selection_ledger_hash", provenance.selection_ledger_hash()),
    ] {
        assert_eq!(hash.len(), 64, "{what} must be 32 bytes of lowercase hex");
        assert!(
            hash.chars().all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c)),
            "{what} must be LOWERCASE hex, got `{hash}`",
        );
    }
    assert!(
        !provenance.dataset_fingerprint().is_empty(),
        "APR-05 is the obligation that inspection recovers a data fingerprint; an empty one \
         recovers nothing",
    );
    assert_ne!(
        provenance.dataset_fingerprint(),
        provenance.validation_split_fingerprint(),
        "the whole dataset and the validation split alone must not fingerprint identically, or \
         one of the two is being read off the wrong thing",
    );
}

/// The same selection reproduces the record; a different one moves the hash.
///
/// Two assertions that fail for opposite reasons, deliberately paired. A record
/// hardcoded to constants passes the first and fails the second; a record built
/// from something nondeterministic (a timestamp, an address, an iteration order)
/// fails the first. Only a deterministic function OF THE SELECTION passes both.
#[test]
fn bundle_provenance_is_deterministic_and_moves_with_the_selection() {
    let run = head_fitted_run();
    let mine = fixture_bundle(&run);

    // Same arguments, drawn again through the shipped selector: identical record.
    let redrawn = selection_at(run.selection().root_seed(), run.selection().shots_per_class());
    let same = bundle_of(&run, &redrawn);
    assert_eq!(
        same.provenance(),
        mine.provenance(),
        "the same dataset and the same selection arguments must reproduce the record exactly",
    );

    // A DIFFERENT selection over the same corpus: the semantic hash must move.
    let other_seed = run.selection().root_seed().wrapping_add(1);
    let other = bundle_of(&run, &selection_at(other_seed, run.selection().shots_per_class()));
    assert_ne!(
        other.provenance().selection_semantic_hash(),
        mine.provenance().selection_semantic_hash(),
        "a different draw selected different rows; a semantic hash that did not move is not \
         identifying the selection",
    );
    assert_eq!(
        other.provenance().dataset_fingerprint(),
        mine.provenance().dataset_fingerprint(),
        "the CORPUS did not change, so its fingerprint must not; only the draw did",
    );
    assert_eq!(other.provenance().selection_root_seed(), other_seed);
}

/// The round trip stays byte-closed with field 20 present.
///
/// `bundle_round_trip_is_byte_stable_and_closed` above already asserts closure for
/// the bundle as a whole; this asserts it for the NEW field specifically, and
/// checks the field actually survives the wire rather than being reconstructed by
/// a default. Closure is the property `setfit-apr-v1`'s codec equation rests on
/// (`serialize(deserialize(bytes)) == bytes`), and adding a field is exactly the
/// change that breaks it.
#[test]
fn bundle_provenance_survives_the_round_trip_byte_closed() {
    let run = head_fitted_run();
    let bundle = fixture_bundle(&run);

    let first = bundle.to_canonical_bytes().expect("first serialize");
    let parsed = SetFitBundle::from_canonical_bytes(&first).expect("parse");
    let reserialized = parsed.to_canonical_bytes().expect("re-serialize");
    assert_bytes_eq(&first, &reserialized, "closure must hold with field 20 present");
    assert_eq!(parsed.provenance(), bundle.provenance());

    // The field is IN the bytes, not merely in the parsed value.
    let text = String::from_utf8(first).expect("canonical bytes are UTF-8 JSON");
    assert!(text.contains("\"provenance\":"), "provenance must be on the wire");
    assert!(
        text.contains(bundle.provenance().selection_semantic_hash()),
        "the selection hash must appear verbatim in the payload",
    );
}

/// A payload declaring the PREVIOUS schema version is refused, naming both.
///
/// The companion to `bundle_version_bump_is_a_typed_version_error`, which covers
/// the forward direction. This is the backward one: v1 is a real version that real
/// bytes on a real disk declare, and it must be a typed refusal rather than a
/// partial interpretation of a nineteen-field payload as a twenty-field one.
#[test]
fn bundle_the_previous_schema_version_is_a_typed_version_error() {
    assert!(
        BUNDLE_SCHEMA_VERSION > 1,
        "this test is about a version that PRECEDES the current one; at version 1 there is none \
         and the test would be vacuous",
    );
    let previous = BUNDLE_SCHEMA_VERSION - 1;

    let bundle = fixture_bundle(&head_fitted_run());
    let text = String::from_utf8(bundle.to_canonical_bytes().expect("canonical bytes"))
        .expect("canonical bytes are UTF-8 JSON");
    let downgraded = text.replacen(
        &format!("\"schema_version\":{BUNDLE_SCHEMA_VERSION}"),
        &format!("\"schema_version\":{previous}"),
        1,
    );
    assert_ne!(downgraded, text, "the replacement must have applied");

    match SetFitBundle::from_canonical_bytes(downgraded.as_bytes()) {
        Err(BundleError::UnsupportedSchemaVersion { got, supported }) => {
            assert_eq!(got, previous, "the refusal must name the version the payload declared");
            assert_eq!(supported, BUNDLE_SCHEMA_VERSION, "and the one this build supports");
        }
        other => panic!("a superseded schema must be refused, got {other:?}"),
    }
}

/// A payload with NO provenance key at all is refused at the parse.
///
/// This is what a genuine v1 artifact looks like, and it is a different failure
/// from the version check above: `provenance` is not an `Option`, so serde never
/// reaches the version comparison. Asserted because "refused" and "refused with
/// the error I expected" are different claims, and because a future
/// `#[serde(default)]` on the field would turn this into a SILENT acceptance that
/// fabricates an empty fingerprint — the precise failure APR-05 exists to prevent.
#[test]
fn bundle_a_payload_without_provenance_is_refused_naming_the_field() {
    let bundle = fixture_bundle(&head_fitted_run());
    let mut value: serde_json::Value =
        serde_json::from_slice(&bundle.to_canonical_bytes().expect("canonical bytes"))
            .expect("canonical bytes parse as JSON");
    let removed =
        value.as_object_mut().expect("the bundle serializes as a JSON object").remove("provenance");
    assert!(removed.is_some(), "the key must have been there to remove");

    let without = serde_json::to_vec(&value).expect("the trimmed document re-serializes");
    match SetFitBundle::from_canonical_bytes(&without) {
        Err(BundleError::Serialization { context, detail }) => {
            assert_eq!(context, "parse");
            assert!(
                detail.contains("provenance"),
                "the parse failure must name the missing field, got `{detail}`",
            );
        }
        other => panic!("a payload with no provenance must be refused, got {other:?}"),
    }
}

/// `ProvenanceRecord` declares no `Option` field, asserted against the SOURCE.
///
/// A behavioural test cannot see this: an `Option` field left `None` on the
/// fixture would serialize as a `null` the completeness gate below catches, but an
/// `Option` field that happens to be `Some` on every fixture would pass every
/// runtime assertion here and still put an un-allowlisted nullable path into
/// production. The declaration is the thing under test, so the declaration is what
/// is read.
#[test]
fn bundle_provenance_record_declares_no_option_field() {
    let src = include_str!("bundle.rs");
    let marker = "pub struct ProvenanceRecord {";
    let start = src.find(marker).expect("the record must be declared") + marker.len();
    let len = src[start..].find("\n}").expect("the declaration must close");
    let block = &src[start..start + len];

    assert!(
        block.contains("dataset_fingerprint: String,"),
        "the scan must be looking at the right block, got:\n{block}",
    );
    assert_eq!(
        block.matches("Option<").count(),
        0,
        "ProvenanceRecord must have NO `Option` field: it contributes zero paths to \
         setfit-apr-v1's nullable-path allowlist while still being walked by the writer's null \
         scan, so an Option here makes every honest artifact a NonFiniteValue refusal. The \
         block was:\n{block}",
    );
    assert!(
        !block.contains("skip_serializing_if"),
        "skip_serializing_if is forbidden on all five sub-document types: it changes \
         to_canonical_bytes output, breaks phase 3's closure tests, and silently empties the \
         allowlist with nothing turning red",
    );
}

// ===========================================================================================
// The nullable-path allowlist completeness gate (04-01 item 3, obligation (b))
// ===========================================================================================

/// EVERY dotted path at which the tree holds a `Value::Null`, in document order.
///
/// The shape is `evidence.rs`'s `first_null_path` (lines 199-224) widened from
/// first-only to all. First-only is the right primitive for a REFUSAL — it reports
/// the offending path and stops. It is the wrong one for a completeness gate,
/// which has to compare a SET against the allowlist: a first-only walk would
/// report `architecture.vocab_remap` and never mention the new field added beside
/// it.
fn null_paths(value: &serde_json::Value) -> Vec<String> {
    match value {
        serde_json::Value::Null => vec![String::new()],
        serde_json::Value::Object(map) => map
            .iter()
            .flat_map(|(key, child)| {
                null_paths(child).into_iter().map(move |rest| join_null_path(key, &rest))
            })
            .collect(),
        serde_json::Value::Array(items) => items
            .iter()
            .enumerate()
            .flat_map(|(index, child)| {
                null_paths(child)
                    .into_iter()
                    .map(move |rest| join_null_path(&index.to_string(), &rest))
            })
            .collect(),
        _ => Vec::new(),
    }
}

/// Join one path segment onto a (possibly empty) remainder.
fn join_null_path(head: &str, rest: &str) -> String {
    if rest.is_empty() {
        head.to_string()
    } else {
        format!("{head}.{rest}")
    }
}

/// The null paths one sub-document contributes, prefixed by its doc field name.
///
/// Asserts the value is a JSON OBJECT before walking it. Without that, a
/// sub-document that serialized to `null` outright would contribute the single
/// path `<name>` and a sub-document that serialized to a scalar would contribute
/// NOTHING — and "contributed nothing" is exactly what this gate reads as "has no
/// nullable fields". A gate whose pass condition is reachable by not looking is
/// the CR-02 vacuity class.
fn subdocument_null_paths<T: serde::Serialize>(name: &str, value: &T) -> Vec<String> {
    let json = serde_json::to_value(value).expect("a sub-document must serialize to a Value");
    let object = json
        .as_object()
        .unwrap_or_else(|| panic!("`{name}` must serialize to a JSON object, got {json}"));
    assert!(!object.is_empty(), "`{name}` serialized to an EMPTY object; nothing was walked");
    null_paths(&json).into_iter().map(|path| join_null_path(name, &path)).collect()
}

/// The allowlist in `setfit-apr-v1` is COMPLETE against the five shipped types.
///
/// # What this gate is for
///
/// 04-02's writer refuses an artifact carrying a `null` at any path outside a
/// four-entry allowlist. That guard is an ALLOWLISTED null scan rather than a
/// blanket one because a blanket scan is exact only over types with no `Option`
/// field — the precondition `evidence.rs:455-478` records for the shipped
/// `first_null_path` guard, and which THREE of these five types do not meet.
///
/// An allowlist that drifts behind the types refuses the first PRODUCTION
/// artifact, and no fixture-shaped test catches that: the slice fixture sets
/// `vocab_remap: Some(..)` (`import.rs:620`) while the full pin sets `None`
/// (`import.rs:501`), so the fixture is the one input shape at which the guard
/// cannot fire. This gate makes the allowlist mechanically derived from the types
/// instead of hand-listed beside them.
///
/// # It lives here because it can only live here
///
/// `aprender-train` is the only crate that can name all five types. `aprender-core`
/// owns the writer and the doc, and train depends on core rather than the reverse,
/// so four of the five arrive there already reduced to an opaque
/// `serde_json::Value`.
///
/// # If this fails
///
/// The fix is a new entry in `contracts/setfit-apr-v1.yaml`'s allowlist AND in
/// `NULLABLE_PATH_ALLOWLIST` (`crates/aprender-core/src/setfit/artifact.rs`),
/// together, with the written analysis the contract requires. It is NEVER
/// `skip_serializing_if` — that would change `to_canonical_bytes`'s output and
/// break the closure tests in this very file, while silently emptying the
/// allowlist with nothing turning red.
#[test]
fn bundle_nullable_path_allowlist_is_complete_over_the_five_subdocuments() {
    /// The allowlist, verbatim from `contracts/setfit-apr-v1.yaml`, sorted.
    const ALLOWLIST: [&str; 4] = [
        "architecture.vocab_remap",
        "evidence.epsilon_used",
        "requested_config.pair_config.budget",
        "requested_config.pair_config.hard_cap",
    ];

    // THE SHIPPED CONSTANT, NOT ONLY THE CONTRACT'S TEXT. The gate's whole claim is
    // that the contract, the writer's guard and the types stay in step; comparing the
    // types against a hand-copied literal proved two of those three and left
    // `aprender-core`'s `NULLABLE_PATH_ALLOWLIST` — the list the writer actually
    // enforces — compared against nothing. A drift there would refuse every
    // production artifact with this gate still green.
    let mut shipped: Vec<&str> = aprender::setfit::NULLABLE_PATH_ALLOWLIST.to_vec();
    shipped.sort_unstable();
    assert_eq!(
        shipped,
        ALLOWLIST.to_vec(),
        "aprender-core's NULLABLE_PATH_ALLOWLIST — the list the writer's null walk enforces — \
         has drifted from contracts/setfit-apr-v1.yaml's allowlist. The two must be edited \
         together, with the written analysis the contract requires.",
    );

    let run = head_fitted_run();
    let bundle = fixture_bundle(&run);

    // (1) architecture — the fixture is a SLICE, so `vocab_remap` is `Some`. The
    //     production shape is the full pin's `None`, which is the one this gate
    //     must see. Overriding it is not weakening the fixture; it is aiming the
    //     gate at the shape the fixture structurally cannot produce.
    let mut architecture = bundle.architecture().clone();
    assert!(
        architecture.vocab_remap.is_some(),
        "the slice fixture is expected to carry a remap; if it does not, this override is no \
         longer doing the work its comment claims",
    );
    architecture.vocab_remap = None;

    // (2) requested_config — through the PUBLIC constructor path, never the
    //     private wire type. `reference_defaults` builds `PairConfig::new`, whose
    //     `budget` and `hard_cap` are both `None`. The fixture run's own config
    //     sets `budget: Some(..)` (test_fixtures.rs:293), so reusing it here would
    //     hide one of the two paths and the gate would pass while under-counting.
    let requested_config = SetFitTrainConfig::reference_defaults(fx::FIXTURE_SEED);
    assert!(
        requested_config.pair_config().budget.is_none()
            && requested_config.pair_config().hard_cap.is_none(),
        "this instance must be the all-`None` one, or the two pair_config paths never appear",
    );

    // (3) resolved_config — one `String`; nothing to set to `None`.
    let resolved_config = bundle.resolved_config.clone();

    // (4) evidence — `epsilon_used` is `None` while unjudged, which is the shipped
    //     state (evidence.rs:655). Set explicitly so the gate does not depend on
    //     that remaining true by accident.
    let mut evidence = bundle.evidence().clone();
    evidence.epsilon_used = None;

    // (5) provenance — no `Option` field to set. That is the claim under test.
    let provenance = bundle.provenance().clone();

    let architecture_paths = subdocument_null_paths("architecture", &architecture);
    let requested_paths = subdocument_null_paths("requested_config", &requested_config);
    let resolved_paths = subdocument_null_paths("resolved_config", &resolved_config);
    let evidence_paths = subdocument_null_paths("evidence", &evidence);
    let provenance_paths = subdocument_null_paths("provenance", &provenance);

    // The two zero-contribution subtrees, asserted BY NAME. Folding them into the
    // set comparison below would report a new `Option` on either as an opaque set
    // mismatch; named, it reports which type grew one.
    assert_eq!(
        resolved_paths,
        Vec::<String>::new(),
        "`resolved_config` (ResolvedConfigRecord) must contribute NO nullable path. It is walked \
         by the writer precisely because it contributes none today — an unwalked subtree cannot \
         reject anything, so a new `Option` here would fail on the first production artifact \
         instead of here.",
    );
    assert_eq!(
        provenance_paths,
        Vec::<String>::new(),
        "`provenance` (ProvenanceRecord) must contribute NO nullable path. Field 20 is `four \
         String + u64 + u32` by construction and the type doc forbids an `Option`; if this fires, \
         one was added and the contract's allowlist is now incomplete.",
    );

    let mut observed: Vec<String> = architecture_paths
        .into_iter()
        .chain(requested_paths)
        .chain(resolved_paths)
        .chain(evidence_paths)
        .chain(provenance_paths)
        .collect();
    observed.sort();

    let expected: Vec<String> = ALLOWLIST.iter().map(|p| (*p).to_string()).collect();
    let unexpected: Vec<&String> = observed.iter().filter(|p| !expected.contains(p)).collect();
    let missing: Vec<&String> = expected.iter().filter(|p| !observed.contains(p)).collect();
    assert!(
        unexpected.is_empty() && missing.is_empty(),
        "the nullable-path allowlist in contracts/setfit-apr-v1.yaml is no longer complete \
         against the shipped types.\n  NOT ALLOWLISTED (a new `Option` field): {unexpected:?}\n  \
         ALLOWLISTED BUT NOT OBSERVED (a removed field, or a fixture that no longer sets it to \
         `None`): {missing:?}\n  observed: {observed:?}\n  allowlist: {expected:?}",
    );
    assert_eq!(observed, expected, "the sets agree but the walk must also be exhaustive");
    assert_eq!(observed.len(), 4, "the contract commits to FOUR paths over FIVE walked types");
}
