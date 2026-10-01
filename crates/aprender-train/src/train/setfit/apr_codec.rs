//! The `setfit-apr-v1` adapter behind the sealed codec seam (plan 04-05, APR-03).
//!
//! # A codec, and deliberately nothing more
//!
//! `verify.rs`'s module docs prescribe this file's shape in advance: "phase 4's
//! APR codec lands as a thin ADAPTER module inside this crate — an
//! `impl Sealed for AprCodec` plus a `SetFitCodec` impl that calls `aprender-core`'s
//! APR format code. The APR FORMAT itself stays in `aprender-core`; only the
//! adapter moves." That is exactly what is below. Every format decision —
//! the container, the canonical tensor names, the document schema, the null walk,
//! the probes, the validation ladder — lives in `aprender::setfit::artifact`. This
//! file owns ONE thing: the `SetFitBundle` <-> `SetFitArtifactView`/`SetFitAprParts`
//! mapping, and it owns it in both directions.
//!
//! # THE BIJECTION IS THE POINT (review B3)
//!
//! `contracts/setfit-apr-v1.yaml`'s `doc_bundle_bijection` equation is what this
//! module implements, and its closure obligation is enforced by the trusted policy
//! on every verified run: `serialize(deserialize(bytes)) == bytes`, byte for byte,
//! at `Tolerance::EXACT`. That holds only if EVERY doc field is either a bundle
//! field or a deterministic function of bundle fields, and every one of the
//! bundle's twenty fields is recovered by `deserialize`. Both directions are
//! written out below, field by field, and `bijection::apr_every_bundle_field_*`
//! asserts all twenty INDIVIDUALLY BY NAME — so a mapping that silently defaults a
//! field names the field in its failure instead of producing an opaque struct
//! inequality.
//!
//! Two consequences of that equation are worth stating where the mapping is, not
//! only in the contract:
//!
//! * **Probes are RECOMPUTED inside `write_setfit_apr` on every serialize**, from
//!   the view's own tensors, through a deterministic CPU encode of six fixed
//!   contract-resident inputs. So a re-serialization of a deserialized artifact
//!   reproduces byte-identical probe records, which is precisely why probes are not
//!   a bundle field and never become one. A probe set CARRIED rather than
//!   recomputed would be a 21st bundle field with no source, and closure would fail
//!   the moment it was written.
//! * **`provenance` is NOT derivable from anything else**, which is why plan 04-13
//!   made it bundle field 20. Dataset fingerprint, validation-split fingerprint,
//!   selection semantic hash, selection ledger hash, selection root seed and
//!   shots-per-class are facts about the run's inputs; no function of the other
//!   nineteen fields produces them. Before field 20 existed, this codec could not
//!   have closed at all.
//!
//! # The hex helpers are BORROWED, never re-implemented
//!
//! `bundle::f32_to_hex` and `bundle::hex_to_f32` are `pub(super)` for this module.
//! A second encoder here that agreed with them today would be two copies of one
//! fact, and the fact in question is the one the closure equation is about.

use std::collections::BTreeMap;

use aprender::setfit::{
    read_setfit_apr_parts, write_setfit_apr, SetFitAprParts, SetFitArtifactError,
    SetFitArtifactView,
};

use super::bundle::{f32_to_hex, hex_to_f32, BundleError, BundleTensor, SetFitBundle};
use super::verify::{sealed, CodecError, SetFitCodec};

/// The identifier of the shipped `setfit-apr-v1` codec.
///
/// It is the SAME string the artifact document's `schema` field carries, and that
/// is not a coincidence: for this codec the container schema and the codec's own
/// identity are one thing. `SERDE_JSON_FORMAT_ID` deliberately names a different
/// format, because phase 3's interim debug encoding is not this one.
pub const APR_FORMAT_ID: &str = "setfit-apr-v1";

/// The `setfit-apr-v1` codec: bytes <-> [`SetFitBundle`], and nothing else.
///
/// It cannot hash, cannot compare, cannot set a tolerance and cannot construct a
/// lifecycle state — the seam removed all four, and swapping the FORMAT behind it
/// does not give any of them back (Ph3 D-07 as amended). The check that decides
/// whether a run is verified is byte-for-byte the one phase 3 committed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AprCodec;

impl AprCodec {
    /// The codec. It has no state and no configuration.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl sealed::Sealed for AprCodec {}

impl SetFitCodec for AprCodec {
    fn format_id(&self) -> &'static str {
        APR_FORMAT_ID
    }

    fn serialize(&self, bundle: &SetFitBundle) -> Result<Vec<u8>, CodecError> {
        let view = view_of(bundle)?;
        write_setfit_apr(&view).map_err(artifact_error)
    }

    fn deserialize(&self, bytes: &[u8]) -> Result<SetFitBundle, CodecError> {
        // Rungs 1-5 of the ONE ladder, through core's parse-only door. The
        // rebuild and the probe-replay rung stay in the production loader's and
        // the trusted policy's domain: a codec's job is bytes <-> bundle, and a
        // codec that replayed probes would be a second verification policy with
        // its own tolerances (Pitfall 9, and `load_setfit_apr` CALLS this same
        // function so there is exactly one ladder).
        let parts = read_setfit_apr_parts(bytes).map_err(artifact_error)?;
        let bundle = bundle_of(parts)?;
        // This check is ALSO made by the trusted `decode`, and the redundancy is
        // deliberate: the two cover different surfaces. `decode` covers the policy
        // path for every implementor including ones that forget to check. This one
        // covers THIS codec's own `pub` surface, which a caller can reach without
        // going through the policy — dropping it here would mean
        // `AprCodec.deserialize(foreign_bytes)` returns a foreign bundle happily.
        // Neither subsumes the other. (The same comment, and the same reason, as
        // `SerdeJsonCodec::deserialize`.)
        if bundle.format_id() != APR_FORMAT_ID {
            return Err(CodecError::ForeignFormat {
                expected: APR_FORMAT_ID.to_string(),
                got: bundle.format_id().to_string(),
            });
        }
        Ok(bundle)
    }
}

// ===========================================================================================
// Error mapping — TYPED in both directions, never stringified
// ===========================================================================================

/// A core artifact failure, carried whole.
///
/// The seam's rule is that the inner error is preserved rather than rendered to a
/// string, so a caller can tell a contracted limit from a parse failure from a
/// probe-replay divergence without matching on message text. `CodecError::Bundle`
/// carries a `BundleError` and cannot carry this type, which is why
/// `CodecError::Artifact` exists.
///
/// `pub(super)` for `apr_reload`: the fresh-process door maps the SAME core failure
/// for the same format when it runs the production loader (04-16), and a second
/// spelling there would be a second place for the format id to drift from this one.
pub(super) fn artifact_error(source: SetFitArtifactError) -> CodecError {
    CodecError::Artifact { format_id: APR_FORMAT_ID.to_string(), source }
}

/// A bundle-layer failure, carried whole.
fn bundle_error(source: BundleError) -> CodecError {
    CodecError::Bundle { format_id: APR_FORMAT_ID.to_string(), source }
}

/// A sub-document that would not convert to or from `serde_json::Value`.
///
/// This IS a bundle-layer serialization failure — the value being converted is a
/// bundle field — so it is reported as one rather than invented as an artifact
/// error the core layer never produced.
fn subdocument_error(field: &str, error: &serde_json::Error) -> CodecError {
    bundle_error(BundleError::Serialization {
        context: format!("apr sub-document `{field}`"),
        detail: error.to_string(),
    })
}

// ===========================================================================================
// FORWARD: SetFitBundle -> SetFitArtifactView (the contract's `forward_doc_to_bundle` rows)
// ===========================================================================================

/// Map all twenty bundle fields onto the writer's complete input.
///
/// The view is `aprender-core`'s type and its field list mirrors `SetFitBundle`'s
/// twenty 1:1 (04-02), so this function invents nothing: every assignment below is
/// a row of `contracts/setfit-apr-v1.yaml`'s bijection table. The doc fields the
/// table derives rather than carries — `tokenizer_sha256`, `head.num_labels`,
/// `hf_name_map`, `probes`, `schema`, `schema_version` — are computed by the writer
/// from these same values and are therefore functions of the bundle, not extra
/// inputs this codec has to supply.
fn view_of(bundle: &SetFitBundle) -> Result<SetFitArtifactView, CodecError> {
    // The three hex-carried payloads are decoded through the BUNDLE's own
    // accessors, which is what makes their limits and their typed diagnostics the
    // bundle layer's rather than a second set living here.
    let tokenizer_bytes = bundle.tokenizer_bytes().map_err(bundle_error)?;
    let tensors = bundle.named_tensors().map_err(bundle_error)?;
    let (ordered_labels, head_n_features, head_weights, head_intercepts) =
        bundle.head_parts().map_err(bundle_error)?;

    // All FIVE embedded sub-documents, because 04-02's null walk covers all five
    // (04-01 item 3) — including `resolved_config` and `provenance`, which
    // contribute zero nullable paths today. Walking a zero-contribution subtree is
    // the point, not redundancy: it is what catches the `Option` added tomorrow.
    let requested_config = serde_json::to_value(&bundle.requested_config)
        .map_err(|e| subdocument_error("requested_config", &e))?;
    let resolved_config = serde_json::to_value(&bundle.resolved_config)
        .map_err(|e| subdocument_error("resolved_config", &e))?;
    let evidence =
        serde_json::to_value(&bundle.evidence).map_err(|e| subdocument_error("evidence", &e))?;
    let provenance = serde_json::to_value(&bundle.provenance)
        .map_err(|e| subdocument_error("provenance", &e))?;

    Ok(SetFitArtifactView {
        // 1
        bundle_schema_version: bundle.schema_version,
        // 2
        format_id: bundle.format_id.clone(),
        // 3 — the TYPED record; the `architecture` sub-document is `to_value` of it,
        //     computed inside the writer so one fact has one copy.
        architecture: bundle.architecture.clone(),
        // 4
        tokenizer_bytes,
        // 5
        pooling: bundle.pooling.clone(),
        // 6
        normalization: bundle.normalization.clone(),
        // 7
        l2_epsilon: bundle.l2_epsilon,
        // 8
        truncation_max_sequence_length: bundle.truncation_max_sequence_length,
        // 9
        padding_mode: bundle.padding_mode.clone(),
        // 10
        max_length: bundle.max_length,
        // 11
        root_seed: bundle.root_seed,
        // 12 — HF-keyed on both sides; the canonical names are the writer's business.
        tensors,
        // 13
        head_weights,
        // 14
        head_intercepts,
        // 15
        head_n_features,
        // 16
        ordered_labels,
        // 17
        requested_config,
        // 18
        resolved_config,
        // 19
        evidence,
        // 20
        provenance,
    })
}

// ===========================================================================================
// REVERSE: SetFitAprParts -> SetFitBundle (the contract's `reverse_bundle_from_doc` rows)
// ===========================================================================================

/// Rebuild all twenty bundle fields from the artifact alone.
///
/// Every field below has a NAMED source in the recovered parts. A field that could
/// not be recovered would be a finding to surface, not a default to invent: a
/// defaulted field breaks `serialize(deserialize(bytes)) == bytes` on the next
/// serialize, which is the closure the trusted policy checks on every run.
fn bundle_of(parts: SetFitAprParts) -> Result<SetFitBundle, CodecError> {
    let SetFitAprParts {
        doc,
        tensors,
        head_weights,
        head_intercepts,
        tokenizer_bytes,
        // The artifact's identity is recomputed by the TRUSTED policy from the same
        // bytes (`verify::artifact_hash`), so it is deliberately not carried into
        // the bundle: two copies of one digest are two values that can disagree, and
        // a codec that supplied its own would be hashing its own output.
        artifact_sha256: _,
    } = parts;

    // The one doc-owned float, recovered from its BIT PATTERN. `l2_epsilon` is an
    // epsilon: decimal text is not the identity on f32, and the bundle compares it
    // with `to_bits()` in `check_policy_matches_this_build`.
    let epsilon =
        hex_to_f32("l2_epsilon_hex", &doc.preprocessing.l2_epsilon_hex).map_err(bundle_error)?;
    let l2_epsilon = match epsilon.as_slice() {
        [only] => *only,
        other => {
            return Err(bundle_error(BundleError::MalformedHexPayload {
                field: "l2_epsilon_hex".to_string(),
                reason: format!(
                    "the preprocessing group must carry exactly one f32, got {}",
                    other.len()
                ),
            }))
        }
    };

    let mut bundle_tensors = BTreeMap::new();
    for (hf_name, (shape, data)) in tensors {
        bundle_tensors.insert(hf_name, BundleTensor { shape, data_hex: f32_to_hex(&data) });
    }

    let requested_config = serde_json::from_value(doc.requested_config)
        .map_err(|e| subdocument_error("requested_config", &e))?;
    let resolved_config = serde_json::from_value(doc.resolved_config)
        .map_err(|e| subdocument_error("resolved_config", &e))?;
    let evidence =
        serde_json::from_value(doc.evidence).map_err(|e| subdocument_error("evidence", &e))?;
    // Field 20 is READ BACK DIRECTLY, never recomputed: it is not a function of the
    // other nineteen (plan 04-13). This assignment is the whole reason the bundle
    // grew a twentieth field.
    let provenance =
        serde_json::from_value(doc.provenance).map_err(|e| subdocument_error("provenance", &e))?;

    Ok(SetFitBundle {
        // 1
        schema_version: doc.bundle_schema_version,
        // 2
        format_id: doc.format_id,
        // 3 — the doc carries the TYPED record, so there is no re-parse here.
        architecture: doc.architecture,
        // 4
        tokenizer_bytes_hex: hex::encode(tokenizer_bytes),
        // 5
        pooling: doc.preprocessing.pooling,
        // 6
        normalization: doc.preprocessing.normalization,
        // 7
        l2_epsilon,
        // 8
        truncation_max_sequence_length: doc.preprocessing.truncation_max_sequence_length,
        // 9
        padding_mode: doc.preprocessing.padding_mode,
        // 10
        max_length: doc.preprocessing.max_length,
        // 11
        root_seed: doc.root_seed,
        // 12 — already HF-keyed by core, which inverted the artifact's OWN
        //      `hf_name_map` rather than re-deriving names from a table that may
        //      have moved since the write.
        tensors: bundle_tensors,
        // 13
        head_weights_hex: f32_to_hex(&head_weights),
        // 14
        head_intercepts_hex: f32_to_hex(&head_intercepts),
        // 15
        head_n_features: doc.head.n_features,
        // 16
        ordered_labels: doc.ordered_labels,
        // 17
        requested_config,
        // 18
        resolved_config,
        // 19
        evidence,
        // 20
        provenance,
    })
}

// ===========================================================================================
// Tests
//
// The three test modules are DIRECT children of `apr_codec`, not nested inside a
// `tests` module and not pulled in through a `#[path]` include. `cargo test`
// filters on the FULL test path, so a nested layout would make this plan's
// acceptance filter `setfit::apr_codec::round_trip` match ZERO tests — and a
// filter matching zero tests EXITS 0 (F-04, hit three times in this phase
// already). The counted gate would then have reported a green pass over an empty
// set. Same reason, same shape as `artifact.rs`'s `mod nullable` / `mod
// determinism`.
// ===========================================================================================

#[cfg(test)]
pub(in crate::train::setfit) mod fixture {
    //! An APR-CAPABLE tiny model, and why the phase-3 slice fixture is not one.
    //!
    //! # The measurement that forced this module to exist
    //!
    //! Plan 04-05 assumed `fx::head_fitted_run(..)` — the phase-3 calibrated
    //! fixture run — could be pushed straight through `verify_artifact(&AprCodec)`.
    //! It cannot, and the refusal was MEASURED, not predicted:
    //!
    //! ```text
    //! Codec(Artifact { format_id: "setfit-apr-v1", source: ProbeComputation {
    //!     probe: "probe_unicode",
    //!     reason: "SetFitError::VocabOutOfSlice(canonical id 5915 is outside the slice closure)" } })
    //! ```
    //!
    //! `setfit-apr-v1` requires SIX contract-resident probes to be computable from
    //! the model, and the phase-3 slice fixture cannot compute two of them:
    //!
    //! | probe | what the slice lacks |
    //! | ----- | -------------------- |
    //! | `probe_unicode` | its `vocab_remap` is a **97-row closure** built for the synthetic fixture corpus; the probe's text tokenizes to canonical ids outside it (`VocabOutOfSlice`) |
    //! | `probe_truncation_boundary` | it declares **64 position rows** (`slice_config.json`), and the probe is 64 repeats of a 7-word unit — truncated at `MAX_SEQUENCE_LENGTH` = 256, so the encode is a 256-position row (`OversizeInput`) |
    //!
    //! Neither is a production defect: the pinned MiniLM-L6-v2 carries the full
    //! 30522-entry vocabulary with **no** remap and 512 position rows, so both
    //! probes are computable there. They are FIXTURE capability gaps — and they are
    //! exactly why plan 04-02 declined to reuse the slice fixture for the writer and
    //! built a self-contained tiny model with `positions = MAX_SEQUENCE_LENGTH` and a
    //! 48-word tokenizer whose every id is in range instead. This module is the
    //! aprender-train side of that same recipe, kept deliberately identical in shape.
    //!
    //! `round_trip::round_trip_the_phase_three_slice_fixture_cannot_carry_an_apr_artifact`
    //! keeps the finding EXECUTABLE rather than only written down: if the slice ever
    //! gains vocabulary coverage, that test turns red and points here.
    //!
    //! # What is real and what is reduced
    //!
    //! Reduced: the ENCODER (8-wide, 2 layers, 48-token vocabulary) and the HEAD.
    //! Real, and taken from a genuine calibrated run through the shipped doors:
    //! the dataset, the selection, the resolved configuration and the stage-one
    //! evidence summary — and, in `round_trip`, the entire trusted verify policy
    //! including `Tolerance::EXACT`. The encoder itself is built through the SHIPPED
    //! reload door `SetFitMiniLm::from_bundle_parts`, which is the same door the
    //! artifact's own rung-7 rebuild uses.

    use std::collections::BTreeMap;

    use aprender::classification::MultinomialLogisticRegression;
    use aprender::setfit::{
        EncoderArchitecture, SetFitMiniLm, L2_EPS, MAX_SEQUENCE_LENGTH, NORMALIZATION_POLICY,
        PADDING_MODE, PINNED_ACTIVATION, PINNED_REVISION, POOLING_POLICY,
    };
    use sha2::{Digest, Sha256};

    use crate::train::setfit::bundle::{
        f32_to_hex, BundleTensor, ProvenanceRecord, ResolvedConfigRecord, SetFitBundle,
        BUNDLE_SCHEMA_VERSION,
    };
    use crate::train::setfit::config::SetFitTrainConfig;
    use crate::train::setfit::evidence::{ClassStats, EvidenceSummary, Verdict};
    use crate::train::setfit::test_fixtures as fx;
    use crate::train::setfit::verify::SetFitCodec;
    use crate::train::setfit::{HeadFitted, HeadFittedEvidence, SetFitRun};

    /// Reduced dimensions. `positions` is NOT reduced, for the reason in the module
    /// docs: `probe_truncation_boundary` produces a `MAX_SEQUENCE_LENGTH`-position row.
    pub(super) const HIDDEN: usize = 8;
    const HEADS: usize = 2;
    const INTERMEDIATE: usize = 16;
    const LAYERS: usize = 2;
    const TYPE_VOCAB: usize = 2;

    /// The fixture's own root seed, distinct from every phase-3 fixture seed so a
    /// recovered `root_seed` cannot match by coincidence.
    pub(super) const ROOT_SEED: u64 = 0x0405_0500_0000_0007;

    /// The fixture's `max_length`, deliberately NOT `MAX_SEQUENCE_LENGTH`.
    ///
    /// Bundle fields 8 and 10 are both sequence-length `u32`s, and they are the
    /// pair a mapping is most likely to swap. Giving them different values is what
    /// turns "they both round-tripped" into "neither was substituted for the other".
    pub(super) const MAX_LENGTH: u32 = 96;

    /// A tiny WordPiece vocabulary. Everything outside it becomes `[UNK]`, so every
    /// id the tokenizer can emit is `< TINY_VOCAB.len()` — which is what lets the
    /// fixture carry `vocab_remap: None` (the PRODUCTION shape) and is precisely
    /// what the slice fixture's 97-row closure cannot promise.
    const TINY_VOCAB: [&str; 48] = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "the",
        "quick",
        "brown",
        "fox",
        "jumps",
        "over",
        "lazy",
        "dog",
        "ok",
        "few",
        "shot",
        "classification",
        "with",
        "contrastive",
        "pairs",
        "line",
        "one",
        "two",
        "tabbed",
        "spaced",
        "stance",
        "detection",
        "i",
        "firmly",
        "support",
        "this",
        "position",
        "el",
        "zorro",
        "cafe",
        "naive",
        "pi",
        "##s",
        "##ed",
        ".",
        ",",
        "!",
        "#",
        "@",
        ":",
        "/",
        "-",
        "=",
    ];

    /// A valid, self-contained `tokenizer.json` in the pinned file's exact shape
    /// (BertNormalizer + BertPreTokenizer + TemplateProcessing + WordPiece).
    ///
    /// Self-contained ON PURPOSE, for 04-02's reason: `fx::fixtures_dir()` honours
    /// the `APRENDER_SETFIT_FIXTURES` override, so reading the committed tokenizer
    /// would make the artifact's bytes — and therefore every closure assertion
    /// below — depend on an environment variable.
    pub(super) fn tiny_tokenizer_json() -> Vec<u8> {
        let mut s = String::new();
        s.push_str(r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":["#);
        for (id, content) in ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"].iter().enumerate() {
            if id > 0 {
                s.push(',');
            }
            s.push_str(&format!(
                r#"{{"id":{id},"special":true,"content":"{content}","single_word":false,"lstrip":false,"rstrip":false,"normalized":false}}"#
            ));
        }
        s.push_str(
            r###"],"normalizer":{"type":"BertNormalizer","clean_text":true,"handle_chinese_chars":true,"strip_accents":null,"lowercase":true},"pre_tokenizer":{"type":"BertPreTokenizer"},"post_processor":{"type":"TemplateProcessing","single":[{"SpecialToken":{"id":"[CLS]","type_id":0}},{"Sequence":{"id":"A","type_id":0}},{"SpecialToken":{"id":"[SEP]","type_id":0}}],"pair":[{"SpecialToken":{"id":"[CLS]","type_id":0}},{"Sequence":{"id":"A","type_id":0}},{"SpecialToken":{"id":"[SEP]","type_id":0}},{"Sequence":{"id":"B","type_id":1}},{"SpecialToken":{"id":"[SEP]","type_id":1}}],"special_tokens":{"[CLS]":{"id":"[CLS]","ids":[2],"tokens":["[CLS]"]},"[SEP]":{"id":"[SEP]","ids":[3],"tokens":["[SEP]"]}}},"decoder":{"type":"WordPiece","prefix":"##","cleanup":true},"model":{"type":"WordPiece","unk_token":"[UNK]","continuing_subword_prefix":"##","max_input_chars_per_word":100,"vocab":{"###,
        );
        for (id, token) in TINY_VOCAB.iter().enumerate() {
            if id > 0 {
                s.push(',');
            }
            s.push_str(&format!(r#""{token}":{id}"#));
        }
        s.push_str("}}}");
        s.into_bytes()
    }

    /// A deterministic, platform-independent filler.
    ///
    /// Every produced value is `k / 65536 - 0.5` for an integer `k`, so it is
    /// EXACTLY representable in `f32` on every target: the fixture's own bytes
    /// cannot be a source of cross-platform drift in a closure comparison.
    struct Filler(u64);

    impl Filler {
        fn new(seed: u64) -> Self {
            Self(seed | 1)
        }

        fn next(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let quantum = ((self.0 >> 40) & 0xFFFF) as f32 / 65536.0;
            quantum - 0.5
        }

        fn vec(&mut self, n: usize) -> Vec<f32> {
            (0..n).map(|_| self.next()).collect()
        }
    }

    /// Lowercase-hex SHA-256, the one digest the tokenizer pairing is checked with.
    fn sha256_hex(bytes: &[u8]) -> String {
        hex::encode(Sha256::digest(bytes))
    }

    /// The fixture's architecture record, in the PRODUCTION nullability shape.
    pub(super) fn architecture() -> EncoderArchitecture {
        EncoderArchitecture {
            hidden: HIDDEN,
            heads: HEADS,
            head_dim: HIDDEN / HEADS,
            num_layers: LAYERS,
            intermediate: INTERMEDIATE,
            vocab: TINY_VOCAB.len(),
            positions: MAX_SEQUENCE_LENGTH,
            type_vocab_size: TYPE_VOCAB,
            // `f64::from(1e-12_f32)` and NOT the `1e-12` literal, measured rather than
            // assumed. `EncoderArchitecture::layer_norm_eps` is an `f64` "widened from
            // the `f32` the encoder holds", and while `f32 -> f64 -> f32` is lossless,
            // `f64 -> f32 -> f64` is not: a declared `1e-12` comes back off the built
            // encoder as `9.999999960041972e-13`. Writing the literal would leave
            // `fixture::bundle()`'s architecture record describing an encoder that
            // `fixture::encoder()` does not build —
            // `apr_the_fixture_encoder_reports_the_fixture_architecture` is what caught it.
            layer_norm_eps: f64::from(1e-12_f32),
            pad_token_id: 0,
            hidden_act: PINNED_ACTIVATION.to_string(),
            source_revision: PINNED_REVISION.to_string(),
            tokenizer_sha256: sha256_hex(&tiny_tokenizer_json()),
            // `None` is the PIN's shape (import.rs:501); the slice sets `Some`.
            vocab_remap: None,
        }
    }

    /// The complete architecture-derived tensor set, at the fixture's dimensions.
    pub(super) fn tensors(arch: &EncoderArchitecture) -> BTreeMap<String, (Vec<usize>, Vec<f32>)> {
        let h = arch.hidden;
        let im = arch.intermediate;
        let mut f = Filler::new(0x0405_0001);
        let mut t: BTreeMap<String, (Vec<usize>, Vec<f32>)> = BTreeMap::new();
        let put = |t: &mut BTreeMap<String, (Vec<usize>, Vec<f32>)>,
                   f: &mut Filler,
                   name: String,
                   shape: Vec<usize>| {
            let n = shape.iter().product();
            t.insert(name, (shape, f.vec(n)));
        };

        put(&mut t, &mut f, "embeddings.word_embeddings.weight".to_string(), vec![arch.vocab, h]);
        put(
            &mut t,
            &mut f,
            "embeddings.position_embeddings.weight".to_string(),
            vec![arch.positions, h],
        );
        put(
            &mut t,
            &mut f,
            "embeddings.token_type_embeddings.weight".to_string(),
            vec![arch.type_vocab_size, h],
        );
        put(&mut t, &mut f, "embeddings.LayerNorm.weight".to_string(), vec![h]);
        put(&mut t, &mut f, "embeddings.LayerNorm.bias".to_string(), vec![h]);

        for n in 0..arch.num_layers {
            let p = format!("encoder.layer.{n}");
            for leaf in ["query", "key", "value"] {
                put(&mut t, &mut f, format!("{p}.attention.self.{leaf}.weight"), vec![h, h]);
                put(&mut t, &mut f, format!("{p}.attention.self.{leaf}.bias"), vec![h]);
            }
            put(&mut t, &mut f, format!("{p}.attention.output.dense.weight"), vec![h, h]);
            put(&mut t, &mut f, format!("{p}.attention.output.dense.bias"), vec![h]);
            put(&mut t, &mut f, format!("{p}.attention.output.LayerNorm.weight"), vec![h]);
            put(&mut t, &mut f, format!("{p}.attention.output.LayerNorm.bias"), vec![h]);
            put(&mut t, &mut f, format!("{p}.intermediate.dense.weight"), vec![im, h]);
            put(&mut t, &mut f, format!("{p}.intermediate.dense.bias"), vec![im]);
            put(&mut t, &mut f, format!("{p}.output.dense.weight"), vec![h, im]);
            put(&mut t, &mut f, format!("{p}.output.dense.bias"), vec![h]);
            put(&mut t, &mut f, format!("{p}.output.LayerNorm.weight"), vec![h]);
            put(&mut t, &mut f, format!("{p}.output.LayerNorm.bias"), vec![h]);
        }
        t
    }

    /// The fixture encoder, built through the SHIPPED reload door.
    pub(super) fn encoder() -> SetFitMiniLm {
        let arch = architecture();
        SetFitMiniLm::from_bundle_parts(&tiny_tokenizer_json(), &arch, tensors(&arch), ROOT_SEED)
            .expect("the fixture architecture, tokenizer and tensor set must rebuild")
    }

    /// The head's coefficients: `K * d` weights and `K` intercepts, exactly representable.
    pub(super) fn head_coefficients(num_labels: usize) -> (Vec<f32>, Vec<f32>) {
        let mut f = Filler::new(0x0405_0002);
        (f.vec(num_labels * HIDDEN), f.vec(num_labels))
    }

    /// A fitted head over `labels`, at the fixture encoder's width.
    pub(super) fn head(labels: &[String]) -> MultinomialLogisticRegression {
        let (weights, intercepts) = head_coefficients(labels.len());
        MultinomialLogisticRegression::from_stored_coefficients(
            labels.to_vec(),
            HIDDEN,
            weights,
            intercepts,
        )
        .expect("the fixture head's arity matches its label set")
    }

    /// The three labels every fixture in this module discriminates.
    pub(super) fn labels() -> Vec<String> {
        vec!["against".to_string(), "favor".to_string(), "neutral".to_string()]
    }

    /// A DISTINCT `EvidenceSummary` — every field a value nothing else here carries.
    pub(super) fn evidence() -> EvidenceSummary {
        let mut per_class = BTreeMap::new();
        for (name, min, median, worst, count) in [
            ("against", 0.125_f64, 0.25_f64, 0.5_f64, 8_usize),
            ("favor", 0.062_5, 0.125, 0.25, 9),
            ("neutral", 0.031_25, 0.062_5, 0.125, 10),
        ] {
            per_class.insert(
                name.to_string(),
                ClassStats { min, median, worst, count, all_moved: true },
            );
        }
        EvidenceSummary {
            schema_version: 1,
            verdict: Verdict::Unjudged,
            trainable_count: 4_242,
            frozen_count: 17,
            per_class,
            worst_param_name: "attention_key_bias".to_string(),
            // The one allowlisted `null` this sub-document can emit, in its
            // PRODUCTION shape — see `NULLABLE_PATH_ALLOWLIST`'s `evidence.epsilon_used`.
            epsilon_used: None,
            calibration_regime_id: "apr-05-bijection-regime".to_string(),
            contract_version: "setfit-apr-v1@04-05".to_string(),
            table_hash: "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
                .to_string(),
        }
    }

    /// A DISTINCT `ProvenanceRecord` — bundle field 20, six values nothing derives.
    pub(super) fn provenance() -> ProvenanceRecord {
        ProvenanceRecord {
            dataset_fingerprint: "d".repeat(64),
            validation_split_fingerprint: "e".repeat(64),
            selection_semantic_hash: "5".repeat(64),
            selection_ledger_hash: "1".repeat(64),
            selection_root_seed: 0x0405_0500_0000_0013,
            shots_per_class: 11,
        }
    }

    /// The seed the fixture's `requested_config` is built at.
    pub(super) const REQUESTED_CONFIG_SEED: u64 = 0x0405_0500_0000_0017;

    /// The device tag the fixture's `resolved_config` records.
    pub(super) const RESOLVED_DEVICE: &str = "cpu";

    /// A bundle whose TWENTY fields all carry distinct, non-default values.
    ///
    /// Built by struct literal rather than through `from_run_parts` on purpose: the
    /// assembly path forces five policy fields to this build's constants, and the
    /// point of this fixture is that every field carries a value a mis-mapping could
    /// be caught by. It is still a bundle `write_setfit_apr` accepts in full — the
    /// tensor set is architecture-complete, the tokenizer hashes to the record's
    /// digest, and all six probes are computable.
    pub(super) fn bundle() -> SetFitBundle {
        let arch = architecture();
        let ordered_labels = labels();
        let (head_weights, head_intercepts) = head_coefficients(ordered_labels.len());
        let mut bundle_tensors = BTreeMap::new();
        for (name, (shape, data)) in tensors(&arch) {
            bundle_tensors.insert(name, BundleTensor { shape, data_hex: f32_to_hex(&data) });
        }
        SetFitBundle {
            schema_version: BUNDLE_SCHEMA_VERSION,
            format_id: super::APR_FORMAT_ID.to_string(),
            architecture: arch,
            tokenizer_bytes_hex: hex::encode(tiny_tokenizer_json()),
            pooling: POOLING_POLICY.to_string(),
            normalization: NORMALIZATION_POLICY.to_string(),
            l2_epsilon: L2_EPS,
            truncation_max_sequence_length: MAX_SEQUENCE_LENGTH as u32,
            padding_mode: PADDING_MODE.to_string(),
            max_length: MAX_LENGTH,
            root_seed: ROOT_SEED,
            tensors: bundle_tensors,
            head_weights_hex: f32_to_hex(&head_weights),
            head_intercepts_hex: f32_to_hex(&head_intercepts),
            head_n_features: HIDDEN,
            ordered_labels,
            requested_config: SetFitTrainConfig::reference_defaults(REQUESTED_CONFIG_SEED),
            resolved_config: ResolvedConfigRecord { resolved_device: RESOLVED_DEVICE.to_string() },
            evidence: evidence(),
            provenance: provenance(),
        }
    }

    /// A calibrated run whose encoder and head CAN carry a `setfit-apr-v1` artifact.
    ///
    /// The struct literal is deliberate and it is in-crate: the lifecycle seal is
    /// against OUT-OF-CRATE minting (`mod.rs`'s `sealed::Sealed`), and phase 3's own
    /// test modules already assemble bundles directly through `from_run_parts`. What
    /// is substituted is named here so no reader has to infer it: the encoder, the
    /// head, and nothing else.
    ///
    /// # Why this lives in `fixture` and is visible to the whole `setfit` module
    ///
    /// It was `round_trip`'s private helper until plan 04-16 needed the same run:
    /// `apr_reload`'s door takes real `.apr` bytes, and this is the ONLY model in
    /// this crate that can produce them — the phase-3 slice fixture provably cannot
    /// compute two of the contract's six probes (see this module's header for the
    /// measurement). A second copy there would have been a second APR-capable
    /// fixture, free to drift from this one in exactly the dimensions the probes
    /// are sensitive to, so the definition moved here instead of being duplicated.
    pub(in crate::train::setfit) fn apr_capable_run() -> SetFitRun<HeadFitted> {
        let real = fx::head_fitted_run(fx::calibrated_variant());
        let SetFitRun { encoder: _slice_encoder, dataset, selection, config, evidence, _state } =
            real;
        let labels = evidence.ordered_labels().to_vec();
        assert_eq!(
            labels.len(),
            3,
            "the fixture corpus declares three classes; the substituted head must match",
        );
        let evidence = HeadFittedEvidence { head: head(&labels), ..evidence };
        SetFitRun { encoder: encoder(), dataset, selection, config, evidence, _state }
    }

    /// The exact bytes `close` will produce for this run, computed without consuming it.
    ///
    /// Assembled through the SAME `from_run_parts` the trusted `close` calls, with the
    /// same seven arguments in the same order, so this is the artifact under test and
    /// not a look-alike.
    pub(in crate::train::setfit) fn artifact_bytes_of(run: &SetFitRun<HeadFitted>) -> Vec<u8> {
        let bundle = SetFitBundle::from_run_parts(
            super::APR_FORMAT_ID,
            run.encoder(),
            run.evidence().head(),
            run.evidence().ordered_labels(),
            run.selection(),
            run.config(),
            run.evidence().passed().summary(),
        )
        .expect("the run's parts must assemble into a bundle");
        super::AprCodec::new().serialize(&bundle).expect("the bundle must write as setfit-apr-v1")
    }
}

#[cfg(test)]
mod bijection {
    //! The doc <-> bundle bijection, proven field by field (review B3).
    //!
    //! `contracts/setfit-apr-v1.yaml`'s `doc_bundle_bijection` invariant says it in
    //! one line: "THE BIJECTION IS THE POINT, NOT THE TABLE. Review B3 found that
    //! byte-canonical closure was asserted by 04-05 and shown nowhere." So the
    //! twenty-field test below asserts every field INDIVIDUALLY BY NAME rather than
    //! resting on the whole-struct `assert_eq!` beside it. Both exist on purpose:
    //! the struct comparison is the one that cannot be fooled by a forgotten
    //! assertion, and the per-field one is the one that says WHICH field moved.

    use aprender::setfit::SetFitArtifactError;

    use super::*;
    use crate::train::setfit::bundle::BundleError;
    use crate::train::setfit::verify::{SerdeJsonCodec, SERDE_JSON_FORMAT_ID};

    /// The codec under test, with no state and no configuration.
    fn codec() -> AprCodec {
        AprCodec::new()
    }

    /// The fixture bundle's real `setfit-apr-v1` bytes.
    fn artifact_bytes() -> Vec<u8> {
        codec()
            .serialize(&fixture::bundle())
            .expect("the fixture bundle must serialize to a setfit-apr-v1 artifact")
    }

    /// The codec owns the contract's schema identifier, and never phase 3's.
    #[test]
    fn apr_the_codec_reports_the_contracts_schema_identifier() {
        assert_eq!(codec().format_id(), "setfit-apr-v1");
        assert_eq!(APR_FORMAT_ID, "setfit-apr-v1");
        assert_ne!(
            codec().format_id(),
            SERDE_JSON_FORMAT_ID,
            "the shipped container must never wear the interim debug format's name",
        );
    }

    /// The fixture is COHERENT: the encoder it builds reports the record it was built from.
    ///
    /// Without this, a divergence between the declared architecture and the rebuilt
    /// one would surface as an unrelated probe failure three rungs later.
    #[test]
    fn apr_the_fixture_encoder_reports_the_fixture_architecture() {
        assert_eq!(fixture::encoder().architecture(), fixture::architecture());
        // And the head must be able to CONSUME that encoder's embedding — the pairing
        // `write_setfit_apr` refuses as `head_n_features does not match the encoder's
        // hidden width`. Asserted here so a fixture-side mismatch is a fixture failure
        // rather than a confusing writer refusal in every other test.
        let head = fixture::head(&fixture::labels());
        assert_eq!(head.n_features(), Some(fixture::HIDDEN));
        assert_eq!(head.labels(), fixture::labels().as_slice());
        assert_eq!(fixture::bundle().head_n_features, fixture::HIDDEN);
    }

    /// serialize -> deserialize returns an EQUAL bundle: all twenty fields, one comparison.
    #[test]
    fn apr_serialize_then_deserialize_returns_an_equal_bundle() {
        let original = fixture::bundle();
        let bytes = codec().serialize(&original).expect("serialize");
        let reloaded = codec().deserialize(&bytes).expect("deserialize");
        assert_eq!(
            reloaded, original,
            "a dropped or defaulted bundle field fails here; the per-field test below names it",
        );
    }

    /// THE BIJECTION PROOF: every one of the twenty fields, asserted individually by name.
    ///
    /// A mapping that silently defaults a field is NAMED by this test's failure
    /// instead of producing an opaque struct inequality. The numbering is the
    /// bundle's wire order, which is also the contract's `reverse_bundle_from_doc`
    /// order, so a row here can be read against a row there.
    #[test]
    fn apr_every_one_of_the_twenty_bundle_fields_is_recovered_by_name() {
        let original = fixture::bundle();
        let bytes = codec().serialize(&original).expect("serialize");
        let r = codec().deserialize(&bytes).expect("deserialize");

        // 1
        assert_eq!(r.schema_version, original.schema_version, "field 1: schema_version");
        // 2
        assert_eq!(r.format_id, original.format_id, "field 2: format_id");
        // 3
        assert_eq!(r.architecture, original.architecture, "field 3: architecture");
        // 4
        assert_eq!(
            r.tokenizer_bytes_hex, original.tokenizer_bytes_hex,
            "field 4: tokenizer_bytes_hex",
        );
        // 5
        assert_eq!(r.pooling, original.pooling, "field 5: pooling");
        // 6
        assert_eq!(r.normalization, original.normalization, "field 6: normalization");
        // 7 — compared by BIT PATTERN: it is an epsilon, and `==` on floats is the
        //     comparison that would not notice a NaN written into the field.
        assert_eq!(
            r.l2_epsilon.to_bits(),
            original.l2_epsilon.to_bits(),
            "field 7: l2_epsilon (bit pattern)",
        );
        // 8
        assert_eq!(
            r.truncation_max_sequence_length, original.truncation_max_sequence_length,
            "field 8: truncation_max_sequence_length",
        );
        // 9
        assert_eq!(r.padding_mode, original.padding_mode, "field 9: padding_mode");
        // 10 — and it is NOT field 8's value, so the two cannot have been swapped.
        assert_eq!(r.max_length, original.max_length, "field 10: max_length");
        assert_ne!(
            r.max_length, r.truncation_max_sequence_length,
            "fields 8 and 10 must stay distinguishable, or a swap between them is invisible",
        );
        // 11
        assert_eq!(r.root_seed, original.root_seed, "field 11: root_seed");
        // 12 — the WHOLE map: names, shapes and hex payloads.
        assert_eq!(r.tensors, original.tensors, "field 12: tensors");
        assert!(!r.tensors.is_empty(), "field 12 must not be recovered as an empty map");
        // 13
        assert_eq!(r.head_weights_hex, original.head_weights_hex, "field 13: head_weights_hex");
        // 14
        assert_eq!(
            r.head_intercepts_hex, original.head_intercepts_hex,
            "field 14: head_intercepts_hex",
        );
        // 15
        assert_eq!(r.head_n_features, original.head_n_features, "field 15: head_n_features");
        // 16
        assert_eq!(r.ordered_labels, original.ordered_labels, "field 16: ordered_labels");
        // 17
        assert_eq!(r.requested_config, original.requested_config, "field 17: requested_config");
        // 18
        assert_eq!(r.resolved_config, original.resolved_config, "field 18: resolved_config");
        // 19
        assert_eq!(r.evidence, original.evidence, "field 19: evidence");
        // 20 — the field plan 04-13 added, and the one no function of the other
        //      nineteen could reproduce. Its recovery is what closes review B3.
        assert_eq!(r.provenance, original.provenance, "field 20: provenance");
        assert_eq!(
            r.provenance.dataset_fingerprint(),
            fixture::provenance().dataset_fingerprint(),
            "field 20: provenance.dataset_fingerprint",
        );
    }

    /// The closure equation itself: `serialize(deserialize(bytes)) == bytes`, TWICE.
    ///
    /// Twice because once proves the identity and twice proves it is STABLE — the
    /// second pass would catch a serialize whose output depended on how many times
    /// it had run, which is the shape a cached or accumulating writer takes.
    #[test]
    fn apr_byte_closure_holds_and_is_stable_across_two_serializations() {
        let bytes = artifact_bytes();
        let first = codec()
            .serialize(&codec().deserialize(&bytes).expect("deserialize 1"))
            .expect("re-serialize 1");
        assert_eq!(first, bytes, "serialize(deserialize(bytes)) must reproduce the input bytes");

        let second = codec()
            .serialize(&codec().deserialize(&first).expect("deserialize 2"))
            .expect("re-serialize 2");
        assert_eq!(second, bytes, "the closure must be stable, not true once");
    }

    /// Two serializations of ONE bundle are byte-identical.
    ///
    /// The determinism the closure rests on, asserted separately from it: a writer
    /// that was nondeterministic would fail closure too, and this is what says which
    /// of the two properties broke.
    #[test]
    fn apr_two_serializations_of_one_bundle_are_byte_identical() {
        let bundle = fixture::bundle();
        assert_eq!(
            codec().serialize(&bundle).expect("write 1"),
            codec().serialize(&bundle).expect("write 2"),
        );
    }

    /// Phase 3's own bytes are refused at the CONTAINER, typed.
    ///
    /// The two formats share a bundle and nothing else; a codec that accepted the
    /// other's payload would make the format identifier decoration.
    #[test]
    fn apr_serde_json_codec_bytes_are_refused_as_a_typed_artifact_error() {
        let json_bytes = SerdeJsonCodec::new()
            .serialize(&fixture::bundle())
            .expect("the phase-3 codec serializes any bundle");
        let err = codec()
            .deserialize(&json_bytes)
            .expect_err("a JSON payload is not a setfit-apr-v1 container");
        assert!(
            matches!(
                err,
                CodecError::Artifact { source: SetFitArtifactError::ContainerIntegrity { .. }, .. }
            ),
            "expected a typed container refusal, got {err:?}",
        );
    }

    /// A bundle declaring a FOREIGN format id is refused by the codec's OWN check.
    ///
    /// This is the check `decode` also makes. The redundancy is deliberate and the
    /// two cover different surfaces — see `deserialize`'s comment. This test reaches
    /// the codec's public surface directly, which is the surface `decode` does not
    /// cover.
    #[test]
    fn apr_a_bundle_declaring_a_foreign_format_id_is_refused_by_the_codecs_own_check() {
        let mut bundle = fixture::bundle();
        bundle.format_id = "setfit-some-other-format-v9".to_string();
        let bytes = codec()
            .serialize(&bundle)
            .expect("the writer stamps the schema, so a foreign format id still writes");
        let err = codec().deserialize(&bytes).expect_err("a foreign format id must be refused");
        assert!(
            matches!(
                &err,
                CodecError::ForeignFormat { expected, got }
                    if expected == APR_FORMAT_ID && got == "setfit-some-other-format-v9"
            ),
            "expected ForeignFormat naming both identifiers, got {err:?}",
        );
    }

    /// A truncated artifact arrives as a TYPED artifact error, matched on the variant.
    ///
    /// The assertion is `matches!` on the inner `SetFitArtifactError` variant and its
    /// discriminating field — never on `to_string()`. A caller must be able to tell a
    /// container failure from a contracted limit without parsing English.
    #[test]
    fn apr_a_truncated_artifact_is_a_typed_artifact_error_not_a_message() {
        let bytes = artifact_bytes();
        let truncated = &bytes[..bytes.len() / 2];
        let err = codec().deserialize(truncated).expect_err("half an artifact is not an artifact");
        let CodecError::Artifact { format_id, source } = &err else {
            panic!("expected CodecError::Artifact carrying the core error typed, got {err:?}");
        };
        assert_eq!(format_id, APR_FORMAT_ID);
        assert!(
            matches!(source, SetFitArtifactError::ContainerIntegrity { .. }),
            "expected a typed container refusal, got {source:?}",
        );
        // The magic survives a truncation of the tail, so a refusal blaming it would
        // mean the ladder read something other than what it was handed.
        let SetFitArtifactError::ContainerIntegrity { what, .. } = source else {
            unreachable!("guarded by the assertion above");
        };
        assert_ne!(*what, "magic", "a tail truncation leaves the magic intact");
    }

    /// A malformed hex payload arrives as a TYPED **bundle** error.
    ///
    /// The other typed channel, and the reason `CodecError::Artifact` is an ADDITION
    /// rather than a replacement: a bundle-layer failure must still report a
    /// `BundleError`, and a core-layer failure must still report a
    /// `SetFitArtifactError`. Collapsing the two would erase exactly the distinction
    /// the seam's doc says a caller must be able to make.
    #[test]
    fn apr_a_malformed_tensor_hex_payload_is_a_typed_bundle_error() {
        let mut bundle = fixture::bundle();
        bundle.head_weights_hex.push('f'); // an odd number of hex characters
        let err = codec().serialize(&bundle).expect_err("a partial f32 is not encodable");
        assert!(
            matches!(
                &err,
                CodecError::Bundle {
                    format_id,
                    source: BundleError::MalformedHexPayload { field, .. },
                } if format_id == APR_FORMAT_ID && field == "head_weights"
            ),
            "expected a typed bundle refusal naming the field, got {err:?}",
        );
    }
}

#[cfg(test)]
mod round_trip {
    //! APR-03, end to end through the REAL trusted policy.
    //!
    //! > "training closes the in-memory model, reloads the written APR through the
    //! > production core loader, and verifies exact tokenizer/configuration/tensor
    //! > state plus tolerance-bounded outputs."
    //!
    //! Every test below drives `SetFitRun::<HeadFitted>::verify_artifact` — the
    //! shipped public door — which delegates to `verify::run_verify_policy` at
    //! `Tolerance::EXACT`. Nothing in the policy changed for phase 4: the format
    //! swapped behind the seam and the check did not (Ph3 D-07 as amended).
    //!
    //! # What the run is, exactly
    //!
    //! The dataset, the selection, the resolved configuration and the stage-one
    //! evidence come from a genuine calibrated run built through the shipped doors
    //! (`fx::head_fitted_run`). The ENCODER and HEAD are substituted for an
    //! APR-capable pair, because the phase-3 slice fixture provably cannot compute
    //! the contract's six probes — see `fixture`'s module docs for the measurement,
    //! and `round_trip_the_phase_three_slice_fixture_cannot_carry_an_apr_artifact`
    //! below for the executable record of it.

    use aprender::setfit::{artifact_sha256_hex, load_setfit_apr, SetFitArtifactError};

    use super::fixture::{apr_capable_run, artifact_bytes_of};
    use super::*;
    use crate::train::setfit::bundle::ProvenanceRecord;
    use crate::train::setfit::test_fixtures as fx;
    use crate::train::setfit::SetFitTrainError;

    /// The final state is MINTED, and the hash it records is the artifact's own.
    ///
    /// One hash, two witnesses: the policy's `verify::artifact_hash` (a trusted free
    /// function, never the codec's) and core's `artifact_sha256_hex` over the same
    /// bytes. They come from two modules and must agree, which is what makes the
    /// recorded identity checkable by a consumer that only has the file.
    #[test]
    fn round_trip_verify_artifact_mints_the_verified_state_and_records_the_artifact_hash() {
        let run = apr_capable_run();
        let bytes = artifact_bytes_of(&run);

        let verified = run
            .verify_artifact(&AprCodec::new())
            .expect("the shipped APR codec must complete the trusted round trip");

        assert_eq!(verified.state_name(), "artifact_reloaded_and_verified");
        assert_eq!(verified.artifact_format_id(), APR_FORMAT_ID);
        assert_eq!(
            verified.artifact_hash(),
            artifact_sha256_hex(&bytes),
            "the state's recorded hash must be the sha256 of the artifact it was minted from",
        );

        let report = verified.evidence().verify_report();
        assert!(report.round_trip_closed(), "the byte-canonical closure check must have passed");
        assert_eq!(report.artifact_bytes(), bytes.len());
        assert!(report.probe_rows() > 0, "the verification must have compared rows");
        assert_eq!(report.embedding_dim(), fixture::HIDDEN);
        assert_eq!(report.class_count(), 3);
    }

    /// The SAME bytes load through the PRODUCTION core loader — probe replay included.
    ///
    /// This is the half of APR-03 the codec alone cannot claim: `deserialize` runs
    /// rungs 2-6, and `load_setfit_apr` runs 2-8. A train-time artifact that parsed
    /// but could not be rebuilt or could not reproduce its own probe expectations
    /// would satisfy the codec and fail in production.
    #[test]
    fn round_trip_the_same_bytes_load_through_the_production_core_loader() {
        let run = apr_capable_run();
        let bytes = artifact_bytes_of(&run);

        let model = load_setfit_apr(&bytes)
            .expect("a train-time setfit-apr-v1 artifact IS a production artifact");
        assert_eq!(model.artifact_sha256(), artifact_sha256_hex(&bytes));
        assert_eq!(model.ordered_labels(), run.evidence().ordered_labels());
        assert_eq!(model.doc_view().head.n_features, fixture::HIDDEN);
        assert_eq!(model.doc_view().schema, APR_FORMAT_ID);

        // The verification the codec's own door does NOT perform, performed.
        let embedded = model
            .embed(&["the quick brown fox".to_string()])
            .expect("the verified model embeds through the same path rung 8 replayed");
        assert_eq!(embedded.len(), 1);
        assert_eq!(embedded[0].len(), fixture::HIDDEN);
    }

    /// The recovered provenance equals the RUN'S OWN selection fingerprints.
    ///
    /// The end-to-end witness that plan 04-13's bundle field 20 is real and not
    /// merely compiled: the values are read off the `Selection` at assembly, written
    /// into the artifact, and recovered from the artifact alone by the production
    /// loader — six facts no function of the other nineteen bundle fields produces.
    #[test]
    fn round_trip_the_recovered_provenance_equals_the_runs_selection_fingerprints() {
        let run = apr_capable_run();
        let bytes = artifact_bytes_of(&run);
        let selection = run.selection();

        let model = load_setfit_apr(&bytes).expect("load");
        let provenance = &model.doc_view().provenance;

        for (field, expected) in [
            ("dataset_fingerprint", selection.dataset_fingerprint_hex().to_string()),
            ("validation_split_fingerprint", selection.validation_fingerprint_hex().to_string()),
            ("selection_semantic_hash", hex::encode(selection.semantic_hash())),
            ("selection_ledger_hash", hex::encode(selection.ledger_hash())),
        ] {
            let observed = provenance
                .get(field)
                .and_then(serde_json::Value::as_str)
                .unwrap_or_else(|| panic!("the artifact must carry provenance.{field}"));
            assert_eq!(observed, expected, "provenance.{field}");
            assert_eq!(observed.len(), 64, "provenance.{field} must be a 64-character digest");
        }
        assert_eq!(
            provenance.get("selection_root_seed").and_then(serde_json::Value::as_u64),
            Some(selection.root_seed()),
        );
        assert_eq!(
            provenance.get("shots_per_class").and_then(serde_json::Value::as_u64),
            Some(u64::from(selection.shots_per_class())),
        );
    }

    /// `Tolerance::EXACT` HOLDS: both bounds and both observed maxima are zero.
    ///
    /// A2's falsification point. If this ever fails, the fix is a contracted
    /// tolerance in TRUSTED code keyed on the codec — never a parameter on the
    /// trait, which is the power the codec/policy split removed.
    #[test]
    fn round_trip_tolerance_exact_holds_with_zero_recorded_maxima() {
        let verified =
            apr_capable_run().verify_artifact(&AprCodec::new()).expect("the round trip closes");
        let report = verified.evidence().verify_report();

        assert_eq!(report.tolerance_embedding_abs(), 0.0, "the APR codec verifies at EXACT");
        assert_eq!(report.tolerance_probability_abs(), 0.0);
        assert_eq!(
            report.max_embedding_abs_diff(),
            0.0,
            "an embedding element differed across the APR persistence boundary",
        );
        assert_eq!(
            report.max_probability_abs_diff(),
            0.0,
            "a class probability differed across the APR persistence boundary",
        );
    }

    /// A codec that SILENTLY DEFAULTS one bundle field cannot close. The bijection,
    /// shown able to fail.
    ///
    /// This is the in-band negative for THIS plan's claim. `AprCodec` recovers all
    /// twenty fields; the wrapper below recovers nineteen and invents the twentieth,
    /// which is precisely the defect review B3 said was asserted and shown nowhere.
    /// The trusted policy's round-trip closure check catches it, because
    /// re-serializing a bundle with a different `provenance` cannot reproduce the
    /// bytes that were hashed.
    ///
    /// It is `provenance` and not another field on purpose: field 20 is the one that
    /// is NOT derivable from the artifact's other contents, so a codec could not
    /// recover it by accident.
    #[test]
    fn round_trip_a_codec_that_defaults_one_bundle_field_cannot_close() {
        /// `AprCodec`, minus field 20.
        struct ProvenanceDefaultingCodec(AprCodec);

        impl sealed::Sealed for ProvenanceDefaultingCodec {}

        impl SetFitCodec for ProvenanceDefaultingCodec {
            fn format_id(&self) -> &'static str {
                APR_FORMAT_ID
            }

            fn serialize(&self, bundle: &SetFitBundle) -> Result<Vec<u8>, CodecError> {
                // HONEST, exactly like `EchoCodec`'s: the defect is entirely in the
                // reverse direction, so the closure check has a fair chance to pass.
                self.0.serialize(bundle)
            }

            fn deserialize(&self, bytes: &[u8]) -> Result<SetFitBundle, CodecError> {
                let mut bundle = self.0.deserialize(bytes)?;
                bundle.provenance = ProvenanceRecord {
                    dataset_fingerprint: "0".repeat(64),
                    validation_split_fingerprint: "0".repeat(64),
                    selection_semantic_hash: "0".repeat(64),
                    selection_ledger_hash: "0".repeat(64),
                    selection_root_seed: 0,
                    shots_per_class: 0,
                };
                Ok(bundle)
            }
        }

        // The control: the SAME run closes under the honest codec, so the refusal
        // below is attributable to the defaulted field and to nothing else.
        apr_capable_run()
            .verify_artifact(&AprCodec::new())
            .expect("control: the honest codec must close");

        // `let ... else` and not `expect_err`: the Ok arm is a `SetFitRun`, which is
        // deliberately not `Debug`, and `.err().expect(..)` is a clippy error here.
        let Err(err) =
            apr_capable_run().verify_artifact(&ProvenanceDefaultingCodec(AprCodec::new()))
        else {
            panic!("a codec that defaults a bundle field must not mint the verified state");
        };
        assert!(
            matches!(err, SetFitTrainError::ReloadNotFromBytes { .. }),
            "expected the closure check to refuse, got {err:?}",
        );
    }

    /// One flipped artifact byte is refused, typed, by BOTH doors.
    ///
    /// The container's trailing CRC covers the whole content and is checked before
    /// anything in the file is interpreted, so a flip anywhere in the payload is a
    /// `footer_checksum` refusal rather than a mis-decoded tensor. The DEEPER
    /// corruption class — a flip that is re-signed, so the CRCs agree and only the
    /// probe replay notices — is 04-03's tamper-harness territory and is already
    /// covered there; it needs a re-emitting writer that lives in core's test module.
    #[test]
    fn round_trip_a_flipped_artifact_byte_is_refused_by_both_doors() {
        let bytes = artifact_bytes_of(&apr_capable_run());
        let mut flipped = bytes.clone();
        let middle = flipped.len() / 2;
        flipped[middle] ^= 0x01;
        assert_ne!(flipped, bytes, "the flip must actually change a byte");

        let codec_err = AprCodec::new()
            .deserialize(&flipped)
            .expect_err("the codec must refuse a corrupted artifact");
        assert!(
            matches!(
                &codec_err,
                CodecError::Artifact {
                    source: SetFitArtifactError::ContainerIntegrity { what, .. },
                    ..
                } if *what == "footer_checksum"
            ),
            "expected a typed footer-checksum refusal, got {codec_err:?}",
        );

        assert!(
            load_setfit_apr(&flipped).is_err(),
            "the production loader must refuse the same bytes",
        );
    }

    /// THE FINDING, kept executable: the phase-3 slice fixture cannot carry an artifact.
    ///
    /// Plan 04-05 was written assuming `fx::head_fitted_run(..)` could be pushed
    /// straight through `verify_artifact(&AprCodec)`. It cannot, and this test is the
    /// measurement rather than a paragraph — if the slice fixture ever gains
    /// vocabulary coverage, this turns red and points at `fixture`'s module docs.
    ///
    /// Both structural gaps are asserted, not only the one that happens to fire
    /// first, so the record does not silently narrow to whichever probe the encoder
    /// reaches soonest.
    #[test]
    fn round_trip_the_phase_three_slice_fixture_cannot_carry_an_apr_artifact() {
        let slice = fx::slice_encoder(fx::FIXTURE_SEED);
        let arch = slice.architecture();
        assert!(
            arch.vocab_remap.is_some(),
            "gap 1: the slice is a VOCABULARY CLOSURE, so a probe outside it has no id",
        );
        assert!(
            arch.positions < aprender::setfit::MAX_SEQUENCE_LENGTH,
            "gap 2: the slice declares {} position rows, below the {}-token truncation \
             boundary probe_truncation_boundary produces",
            arch.positions,
            aprender::setfit::MAX_SEQUENCE_LENGTH,
        );

        let Err(err) =
            fx::head_fitted_run(fx::calibrated_variant()).verify_artifact(&AprCodec::new())
        else {
            panic!("the slice fixture cannot compute the contract's six probes");
        };
        assert!(
            matches!(
                &err,
                SetFitTrainError::Codec(CodecError::Artifact {
                    source: SetFitArtifactError::ProbeComputation { probe, .. },
                    ..
                }) if probe == "probe_unicode"
            ),
            "expected a typed probe-computation refusal naming probe_unicode, got {err:?}",
        );
    }
}
