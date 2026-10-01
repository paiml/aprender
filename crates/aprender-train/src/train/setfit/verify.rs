//! The codec seam and the trusted verify-by-reload policy (plan 03-08, D-07, TRN-01).
//!
//! # The seam is implementable and powerless
//!
//! [`SetFitCodec`] is a PURE CODEC: a format identifier and a bytes <-> bundle
//! pair, and nothing else. It has no hashing method, no tolerance parameter, no
//! comparison method and no way to construct a lifecycle state. Everything that
//! DECIDES anything — the artifact hash, the close/reload/re-encode/re-predict
//! sequence, the comparison, and the minting of
//! [`ArtifactReloadedAndVerified`](super::ArtifactReloadedAndVerified) — lives in
//! trusted crate-internal code below and cannot be overridden.
//!
//! The split exists because the earlier shape did not have it. A single
//! `ReloadVerify` trait owning `reload()` lets an implementor return the
//! pre-close bundle unchanged: the comparison then passes and no persistence
//! boundary ever existed. Reducing the trait to a codec removes the comparison
//! and the minting from an implementor's reach, which is the substantive half of
//! that fix; the seal below is the conservative other half.
//!
//! # The trait is SEALED, and what that costs phase 4
//!
//! `sealed::Sealed` is a private-module supertrait, so no out-of-crate type can
//! implement [`SetFitCodec`]. Sealing later would be a breaking change and
//! un-sealing later is not, so phase 3 seals. The consequence is concrete: phase
//! 4's APR codec lands as a thin ADAPTER module inside this crate — an
//! `impl Sealed for AprCodec` plus a `SetFitCodec` impl that calls
//! `aprender-core`'s APR format code. The APR FORMAT itself stays in
//! `aprender-core`; only the adapter moves. If an openly implementable codec is
//! wanted later, deleting the supertrait is a non-breaking change.
//!
//! # CANONICAL-SERIALIZATION OBLIGATION
//!
//! Implementors MUST be byte-canonical: re-serializing a bundle obtained from
//! [`SetFitCodec::deserialize`] must reproduce the input bytes exactly. This is a
//! real constraint, not a free property. It holds for [`SerdeJsonCodec`] because
//! the bundle uses `BTreeMap` ordering, declaration-order fields and bit-pattern
//! floats — and it took a Cargo feature to make even that true (`float_roundtrip`;
//! without it serde_json's float parser lands one ULP away and the check fails on
//! every honest bundle). A phase-4 writer with padding, unordered metadata or a
//! checksum placed after the payload would fail [`SetFitTrainError::ReloadNotFromBytes`]
//! and be blocked until someone edited the contract.
//!
//! It is stated twice on purpose — here and as an explicit clause of the
//! `reload_verify_roundtrip` equation — so that a phase-4 author MEETS it or
//! amends it deliberately, rather than discovering it as a mysterious
//! verification failure.

use std::collections::BTreeMap;

use aprender::classification::MultinomialLogisticRegression;
use aprender::setfit::SetFitMiniLm;
use aprender_contrastive_data::prepared::{Canonical, PreparedDataset};
use aprender_contrastive_data::select::Selection;
use sha2::{Digest, Sha256};

use super::bundle::{BundleError, SetFitBundle};
use super::config::ResolvedSetFitConfig;
use super::evidence::EvidenceSummary;
use super::{
    head_input, ArtifactVerifiedEvidence, RetainedArtifactBytes, SetFitTrainError, VerifyProbe,
    VerifyReport,
};

/// The seal. Crate-visible module, public-in-private trait — the standard Rust idiom.
///
/// `pub(crate)` and not `pub`: the module docs above promise that no OUT-OF-CRATE
/// type can implement [`SetFitCodec`], and `pub(crate)` keeps that promise exactly
/// while letting phase 4's adapter — `super::apr_codec::AprCodec`, a sibling module
/// in this crate — write the `impl Sealed` the docs say it will. A module-private
/// seal would have forced the adapter to live inside this file, which is the one
/// thing the "thin ADAPTER module inside this crate" sentence rules out.
pub(crate) mod sealed {
    /// Implemented only by codecs declared inside this crate.
    pub trait Sealed {}
}

/// The identifier of the phase-3 serde codec.
///
/// It never claims to be the project's shipped model container: an interim debug
/// format wearing that name would make a phase-4 reader believe it had found the
/// real thing.
pub const SERDE_JSON_FORMAT_ID: &str = "setfit-serde-json-v1";

// ===========================================================================================
// The codec seam
// ===========================================================================================

/// A bytes <-> [`SetFitBundle`] codec, and nothing else.
///
/// Three methods. None hashes, none compares, none carries a tolerance, and none
/// can construct a lifecycle state. See the module docs for the sealing decision
/// and the canonical-serialization obligation every implementor must satisfy.
pub trait SetFitCodec: sealed::Sealed {
    /// The identifier this codec stamps into, and expects to find in, a payload.
    fn format_id(&self) -> &'static str;

    /// Encode a bundle.
    ///
    /// # Errors
    ///
    /// [`CodecError`] naming this codec's format and the underlying failure.
    fn serialize(&self, bundle: &SetFitBundle) -> Result<Vec<u8>, CodecError>;

    /// Decode a bundle.
    ///
    /// # Errors
    ///
    /// [`CodecError`] naming this codec's format and the underlying failure.
    fn deserialize(&self, bytes: &[u8]) -> Result<SetFitBundle, CodecError>;
}

/// A codec-level failure, carrying the typed bundle error underneath it.
///
/// The inner [`BundleError`] is preserved rather than rendered to a string: a
/// caller must be able to tell a contracted limit from a parse failure from a
/// schema-version refusal without matching on message text.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum CodecError {
    /// The bundle layer refused the payload.
    Bundle {
        /// The codec that was reading or writing.
        format_id: String,
        /// What the bundle layer said.
        source: BundleError,
    },
    /// The payload was written by a different codec.
    ForeignFormat {
        /// The identifier this codec owns.
        expected: String,
        /// The identifier the payload declares.
        got: String,
    },
    /// The `aprender-core` artifact layer refused the payload, typed.
    ///
    /// # Why this is a SECOND variant and not a conversion into [`Self::Bundle`]
    ///
    /// [`Self::Bundle`] carries a [`BundleError`]; the core artifact layer reports
    /// a [`aprender::setfit::SetFitArtifactError`], and the two enumerate different
    /// worlds — a contracted allocation limit, a container CRC failure, a
    /// non-finite payload, an incomplete tensor set and a probe-replay divergence
    /// have no `BundleError` counterparts at all. This module's rule is that the
    /// inner error is PRESERVED rather than rendered to a string, "so a caller can
    /// tell a contracted limit from a parse failure without matching on message
    /// text" (see [`Self::Bundle`]'s own doc). Honouring that rule with only one
    /// variant leaves two options and both are defects: stringify the core error
    /// into `BundleError::Serialization` — which erases exactly the distinction the
    /// rule exists to keep — or map it onto some `BundleError` variant that never
    /// happened, which is worse, because the caller then matches successfully on a
    /// diagnosis nothing produced.
    ///
    /// Adding a variant is cheap here precisely because `CodecError` is
    /// `#[non_exhaustive]`, and it is constructible because
    /// `SetFitArtifactError` derives `Debug + Clone + PartialEq` — the three this
    /// enum derives (a 04-02 acceptance criterion, taken for this reason).
    Artifact {
        /// The codec that was reading or writing.
        format_id: String,
        /// What the core artifact layer said, typed and whole.
        source: aprender::setfit::SetFitArtifactError,
    },
}

impl core::fmt::Display for CodecError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Bundle { format_id, source } => {
                write!(f, "codec `{format_id}`: {source}")
            }
            Self::ForeignFormat { expected, got } => write!(
                f,
                "codec `{expected}` was handed a payload written by `{got}`; a codec reads \
                 only its own format",
            ),
            Self::Artifact { format_id, source } => {
                write!(f, "codec `{format_id}`: {source}")
            }
        }
    }
}

impl std::error::Error for CodecError {}

/// The phase-3 codec: the bundle's own canonical JSON.
///
/// `pub`, and reachable from outside the crate on purpose — plan 03-10's
/// out-of-crate integration test constructs one directly, and a `pub(crate)` type
/// would make that test unwritable without a backdoor. It is still SEALED against
/// out-of-crate *implementations* of the trait, which is the property that matters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SerdeJsonCodec;

impl SerdeJsonCodec {
    /// The codec. It has no state and no configuration.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl sealed::Sealed for SerdeJsonCodec {}

impl SetFitCodec for SerdeJsonCodec {
    fn format_id(&self) -> &'static str {
        SERDE_JSON_FORMAT_ID
    }

    fn serialize(&self, bundle: &SetFitBundle) -> Result<Vec<u8>, CodecError> {
        bundle.to_canonical_bytes().map_err(|source| CodecError::Bundle {
            format_id: SERDE_JSON_FORMAT_ID.to_string(),
            source,
        })
    }

    fn deserialize(&self, bytes: &[u8]) -> Result<SetFitBundle, CodecError> {
        let bundle = SetFitBundle::from_canonical_bytes(bytes).map_err(|source| {
            CodecError::Bundle { format_id: SERDE_JSON_FORMAT_ID.to_string(), source }
        })?;
        // This check is ALSO made by the trusted `decode`, and the redundancy is deliberate:
        // the two cover different surfaces. `decode` covers the policy path for every
        // implementor including ones that forget to check. This one covers THIS codec's own
        // `pub` surface, which plan 03-10 calls directly without going through the policy —
        // dropping it here would mean `SerdeJsonCodec.deserialize(foreign_bytes)` returns a
        // foreign bundle happily. Neither subsumes the other.
        if bundle.format_id() != SERDE_JSON_FORMAT_ID {
            return Err(CodecError::ForeignFormat {
                expected: SERDE_JSON_FORMAT_ID.to_string(),
                got: bundle.format_id().to_string(),
            });
        }
        Ok(bundle)
    }
}

// ===========================================================================================
// Trusted policy — crate-internal, not overridable
// ===========================================================================================

/// SHA-256 over an artifact's canonical bytes.
///
/// A FREE FUNCTION and deliberately not a trait method: a codec that hashed its
/// own output could report any digest it liked for any bytes, and the digest is
/// the only thing tying a verified run to a specific artifact.
pub(crate) fn artifact_hash(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

/// Decode through `codec`, then enforce that the payload is the codec's OWN format.
///
/// # Why the check lives here and not in the implementor
///
/// `close` stamps the format id via [`SetFitBundle::from_run_parts`]`(codec.format_id(), ..)`
/// — trusted, and unforgeable by an implementor. The matching check on the way back in used
/// to be hand-written inside `SerdeJsonCodec::deserialize`, which made the *encode* half of
/// the invariant a property of the module and the *decode* half a guard each implementor had
/// to remember. That is the wrong altitude for a module whose entire thesis is that an
/// implementor is powerless: `SetFitBundle::from_canonical_bytes` does not check the id
/// either, so a phase-4 `AprCodec` that simply forgot would have a public `deserialize` that
/// silently accepts a foreign payload, with nothing in-crate noticing.
///
/// Inside [`run_verify_policy`] the round-trip closure check happens to mask this; on the
/// codec's own public surface — which plan 03-10 deliberately exposes — nothing does. All
/// three inputs to the existing [`CodecError::ForeignFormat`] are reachable from the trait,
/// so one function covers every present and future implementor. This is the same lift the
/// phase already made for `head_input::push_rows`: one guard, reached from every caller.
fn decode<C: SetFitCodec>(codec: &C, bytes: &[u8]) -> Result<SetFitBundle, CodecError> {
    let bundle = codec.deserialize(bytes)?;
    if bundle.format_id() != codec.format_id() {
        return Err(CodecError::ForeignFormat {
            expected: codec.format_id().to_string(),
            got: bundle.format_id().to_string(),
        });
    }
    Ok(bundle)
}

/// The comparison tolerance. Crate-internal; no codec can set one.
///
/// [`SerdeJsonCodec`]'s verification runs at [`Self::EXACT`]. The type exists
/// because phase 4's format may need a contracted non-zero tolerance, and a
/// tolerance that arrives at that point with nowhere to live tends to arrive as a
/// parameter on the trait — which is exactly the power this split removed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Tolerance {
    /// Maximum accepted absolute difference between two embedding elements.
    pub(crate) embedding_abs: f32,
    /// Maximum accepted absolute difference between two class probabilities.
    pub(crate) probability_abs: f64,
}

impl Tolerance {
    /// No difference at all is accepted.
    pub(crate) const EXACT: Self = Self { embedding_abs: 0.0, probability_abs: 0.0 };
}

/// What the pre-close model answered, recorded before it was dropped.
///
/// Built by [`close`], which CONSUMES the live model, so this is the only thing
/// that survives it.
pub(crate) struct ClosedArtifact {
    /// The serialized artifact.
    pub(crate) bytes: Vec<u8>,
    /// SHA-256 of exactly those bytes.
    pub(crate) hash: [u8; 32],
    /// The pre-close probe.
    pub(crate) probe: VerifyProbe,
}

/// Serialize the run's state, hash it, and DROP the live model.
///
/// # The close is structural, not a discipline
///
/// `encoder` and `head` are taken BY VALUE and never returned. They go out of
/// scope when this function does, so the caller has no binding to the live model
/// afterwards and cannot accidentally compare against it — the borrow checker
/// enforces what a `drop(...)` call would only document.
fn close<C: SetFitCodec>(
    codec: &C,
    mut encoder: SetFitMiniLm,
    head: MultinomialLogisticRegression,
    ordered_labels: &[String],
    dataset: &PreparedDataset<Canonical>,
    selection: &Selection,
    config: &ResolvedSetFitConfig,
    summary: &EvidenceSummary,
) -> Result<ClosedArtifact, SetFitTrainError> {
    // The probe is taken FIRST, while the live model still exists: it is the
    // baseline the reloaded model is judged against.
    let probe = probe_model(&mut encoder, &head, dataset, selection, config)?;

    let bundle = SetFitBundle::from_run_parts(
        codec.format_id(),
        &encoder,
        &head,
        ordered_labels,
        selection,
        config,
        summary,
    )
    .map_err(SetFitTrainError::Bundle)?;
    let bytes = codec.serialize(&bundle).map_err(SetFitTrainError::Codec)?;
    let hash = artifact_hash(&bytes);

    Ok(ClosedArtifact { bytes, hash, probe })
}

/// Encode the selected rows through THIS model and predict with THIS head.
///
/// The encode goes through `head_input::head_dataset`, which is the encode-once
/// path stage two itself uses — eval mode, inside `no_grad`, every result
/// detached. Using a second encode path here would compare the reloaded model
/// against something the trainer never ran.
fn probe_model(
    encoder: &mut SetFitMiniLm,
    head: &MultinomialLogisticRegression,
    dataset: &PreparedDataset<Canonical>,
    selection: &Selection,
    config: &ResolvedSetFitConfig,
) -> Result<VerifyProbe, SetFitTrainError> {
    let input =
        head_input::head_dataset(encoder, dataset, selection, config.requested().batch_size())?;
    // Both halves MOVED out together, from the one object the encode filled. Copying
    // the embeddings and re-deriving the ids from the selection would give two lists
    // that agree by construction and could not disagree with each other.
    let (embeddings, ids) = input.into_probe_parts();
    let probabilities = head.predict_proba(&embeddings).map_err(SetFitTrainError::HeadFit)?;
    let labels = head.predict(&embeddings).map_err(SetFitTrainError::HeadFit)?;
    Ok(VerifyProbe { ids, embeddings, probabilities, labels })
}

/// Whether `delta` is inside `bound`, with an INCOMPARABLE delta counting as outside.
///
/// Written through `partial_cmp` rather than as `delta <= bound` so the NaN case is
/// visible rather than implied. A NaN difference means at least one of the two
/// values was non-finite, which is a divergence in every sense that matters; a
/// bare `delta > bound` test would silently accept it, because every comparison
/// with NaN is false.
fn within(delta: f64, bound: f64) -> bool {
    matches!(
        delta.partial_cmp(&bound),
        Some(core::cmp::Ordering::Less | core::cmp::Ordering::Equal)
    )
}

/// The one shape a probe divergence takes, so the five rungs below cannot disagree on it.
fn diverged(
    field: &'static str,
    row: usize,
    index: usize,
    expected: String,
    observed: String,
    bound: f64,
) -> SetFitTrainError {
    SetFitTrainError::ReloadDiverged { field, row, index, expected, observed, tolerance: bound }
}

/// Rung 1: EVERY one of the four lists on both sides has the same length.
///
/// Not just the ids. The rungs below walk their pairs with `zip`, which STOPS at the shorter
/// side — so a probe whose embeddings, probabilities or labels ever desynchronised from its
/// ids would have the surplus rows silently skipped and the verification would pass on a
/// PARTIAL comparison. They agree by construction today (`into_probe_parts` moves both halves
/// out of one object, and the head answers once per row); this is what keeps "today" from
/// being the whole argument.
fn compare_probe_row_counts(
    before: &VerifyProbe,
    after: &VerifyProbe,
) -> Result<(), SetFitTrainError> {
    for (what, expected, observed) in [
        ("probe_row_count", before.ids.len(), after.ids.len()),
        ("probe_row_count", before.ids.len(), before.embeddings.len()),
        ("probe_row_count", before.ids.len(), after.embeddings.len()),
        ("probe_row_count", before.ids.len(), before.probabilities.len()),
        ("probe_row_count", before.ids.len(), after.probabilities.len()),
        ("probe_row_count", before.ids.len(), before.labels.len()),
        ("probe_row_count", before.ids.len(), after.labels.len()),
    ] {
        if expected != observed {
            return Err(diverged(what, 0, 0, expected.to_string(), observed.to_string(), 0.0));
        }
    }
    Ok(())
}

/// Rung 2 and rung 5: an exact string comparison over one list, named by `field`.
fn compare_probe_strings(
    field: &'static str,
    before: &[String],
    after: &[String],
) -> Result<(), SetFitTrainError> {
    for (row, (a, b)) in before.iter().zip(after.iter()).enumerate() {
        if a != b {
            return Err(diverged(field, row, 0, a.clone(), b.clone(), 0.0));
        }
    }
    Ok(())
}

/// Rung 3: the embeddings, at `tolerance.embedding_abs`, returning the largest delta seen.
fn compare_probe_embeddings(
    before: &VerifyProbe,
    after: &VerifyProbe,
    tolerance: Tolerance,
) -> Result<f32, SetFitTrainError> {
    let bound = f64::from(tolerance.embedding_abs);
    let mut max_embedding: f32 = 0.0;
    for (row, (a, b)) in before.embeddings.iter().zip(after.embeddings.iter()).enumerate() {
        if a.len() != b.len() {
            return Err(diverged(
                "embedding_width",
                row,
                0,
                a.len().to_string(),
                b.len().to_string(),
                0.0,
            ));
        }
        for (index, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            let delta = (x - y).abs();
            if delta > max_embedding {
                max_embedding = delta;
            }
            if !within(f64::from(delta), bound) {
                return Err(diverged(
                    "embedding",
                    row,
                    index,
                    format!("{x:e}"),
                    format!("{y:e}"),
                    bound,
                ));
            }
        }
    }
    Ok(max_embedding)
}

/// Rung 4: the class probabilities, at `tolerance.probability_abs`.
fn compare_probe_probabilities(
    before: &VerifyProbe,
    after: &VerifyProbe,
    tolerance: Tolerance,
) -> Result<f64, SetFitTrainError> {
    let bound = tolerance.probability_abs;
    let mut max_probability: f64 = 0.0;
    for (row, (a, b)) in before.probabilities.iter().zip(after.probabilities.iter()).enumerate() {
        if a.len() != b.len() {
            return Err(diverged(
                "class_count",
                row,
                0,
                a.len().to_string(),
                b.len().to_string(),
                0.0,
            ));
        }
        for (index, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            let delta = (x - y).abs();
            if delta > max_probability {
                max_probability = delta;
            }
            if !within(delta, bound) {
                return Err(diverged(
                    "probability",
                    row,
                    index,
                    format!("{x:e}"),
                    format!("{y:e}"),
                    bound,
                ));
            }
        }
    }
    Ok(max_probability)
}

/// Compare two probes at `tolerance`, naming the first divergence.
///
/// # Five rungs, in this order, and the order is observable
///
/// Split into per-rung functions in plan 03-10 T3 to clear the project's cyclomatic ceiling of
/// 10 (measured 17 before). The row-count rung must stay FIRST — every rung below it walks
/// with `zip` and would silently skip surplus rows — and the field names in the errors are
/// what `verify_tests` asserts on, so the sequence is behaviour, not layout.
fn compare_probes(
    before: &VerifyProbe,
    after: &VerifyProbe,
    tolerance: Tolerance,
) -> Result<(f32, f64), SetFitTrainError> {
    compare_probe_row_counts(before, after)?;
    compare_probe_strings("probe_id", &before.ids, &after.ids)?;
    let max_embedding = compare_probe_embeddings(before, after, tolerance)?;
    let max_probability = compare_probe_probabilities(before, after, tolerance)?;
    compare_probe_strings("label", &before.labels, &after.labels)?;
    Ok((max_embedding, max_probability))
}

/// THE round-trip closure check: `serialize(deserialize(b)) == b`, byte for byte.
///
/// # What it establishes, and what it does not
///
/// It establishes that the value the codec returned is one whose serialization IS
/// the hashed artifact. A codec cannot therefore substitute an arbitrary object
/// for the artifact's contents — the substitute would have to re-serialize to the
/// same bytes, which is the definition of being the same bundle.
///
/// It does NOT establish durability. An in-process codec that round-trips
/// faithfully through a buffer it never writes anywhere satisfies this and is
/// indistinguishable from one that wrote a file, because nothing here observes the
/// filesystem. Durability is a claim about I/O and it is out of this seam's scope.
///
/// # Errors
///
/// [`SetFitTrainError::ReloadNotFromBytes`], carrying both lengths and the offset
/// of the first difference; or [`SetFitTrainError::Codec`] if the re-serialization
/// itself fails.
fn close_round_trip<C: SetFitCodec>(
    codec: &C,
    reloaded: &SetFitBundle,
    hashed: &[u8],
) -> Result<(), SetFitTrainError> {
    let reserialized = codec.serialize(reloaded).map_err(SetFitTrainError::Codec)?;
    if reserialized == hashed {
        return Ok(());
    }
    Err(SetFitTrainError::ReloadNotFromBytes {
        hashed_len: hashed.len(),
        reserialized_len: reserialized.len(),
        first_diff_offset: hashed.iter().zip(reserialized.iter()).position(|(a, b)| a != b),
    })
}

/// Rebuild a model and head from a reloaded bundle, and from nothing else.
fn rebuild_from(
    bundle: &SetFitBundle,
) -> Result<(SetFitMiniLm, MultinomialLogisticRegression), SetFitTrainError> {
    // FIRST, and before a byte of tensor data is decoded: the rebuild applies the policy
    // this build compiles in, so a payload recording a different one would be rebuilt
    // under ours without a word. Reading the recorded fields here is what keeps them
    // normative rather than decorative.
    bundle.check_policy_matches_this_build().map_err(SetFitTrainError::Bundle)?;
    let tokenizer_bytes = bundle.tokenizer_bytes().map_err(SetFitTrainError::Bundle)?;
    let tensors: BTreeMap<String, (Vec<usize>, Vec<f32>)> =
        bundle.named_tensors().map_err(SetFitTrainError::Bundle)?;
    let encoder = SetFitMiniLm::from_bundle_parts(
        &tokenizer_bytes,
        bundle.architecture(),
        tensors,
        bundle.root_seed(),
    )
    .map_err(|e| SetFitTrainError::Encoder { reason: e.to_string() })?;
    let (labels, n_features, weights, intercepts) =
        bundle.head_parts().map_err(SetFitTrainError::Bundle)?;
    let head = MultinomialLogisticRegression::from_stored_coefficients(
        labels, n_features, weights, intercepts,
    )
    .map_err(SetFitTrainError::HeadFit)?;
    Ok((encoder, head))
}

/// Everything the trusted policy produced. Every field traces to the reloaded bytes.
pub(crate) struct VerifiedOutcome {
    /// The encoder REBUILT from the artifact.
    pub(crate) encoder: SetFitMiniLm,
    /// The head REBUILT from the artifact.
    pub(crate) head: MultinomialLogisticRegression,
    /// The measured outcome.
    pub(crate) report: VerifyReport,
    /// What the REBUILT model answered.
    pub(crate) probe: VerifyProbe,
    /// SHA-256 of the artifact's bytes.
    pub(crate) artifact_hash: [u8; 32],
    /// THE bytes — the ones [`Self::artifact_hash`] was taken over.
    ///
    /// Not a re-serialization and not a copy taken somewhere else: this is the same
    /// `Vec` [`close`] hashed and [`close_round_trip`] closed, moved rather than
    /// rebuilt, so no route exists by which the value handed out could differ from the
    /// value that was verified.
    pub(crate) bytes: Vec<u8>,
}

/// The whole trusted sequence, driven from [`super::SetFitRun::verify_artifact`].
///
/// The caller has no binding to the pre-close model at any point after [`close`]
/// returns, because [`close`] consumed it.
pub(crate) fn run_verify_policy<C: SetFitCodec>(
    codec: &C,
    encoder: SetFitMiniLm,
    head: MultinomialLogisticRegression,
    ordered_labels: &[String],
    dataset: &PreparedDataset<Canonical>,
    selection: &Selection,
    config: &ResolvedSetFitConfig,
    summary: &EvidenceSummary,
    tolerance: Tolerance,
) -> Result<VerifiedOutcome, SetFitTrainError> {
    // (1) Probe, serialize, hash, and DROP the live model. `close` consumes it.
    let ClosedArtifact { bytes, hash, probe } =
        close(codec, encoder, head, ordered_labels, dataset, selection, config, summary)?;

    // (2) Reload FROM BYTES, through the trusted `decode` so the format-id check is the
    // module's and not the implementor's.
    let reloaded = decode(codec, &bytes).map_err(SetFitTrainError::Codec)?;

    // (3) ROUND-TRIP CLOSURE CHECK, before anything downstream trusts the value.
    //
    // Re-serialize what the codec handed back and require the result to equal the
    // bytes that were hashed. A faithful codec satisfies this by construction --
    // it is the serialize/deserialize identity the bundle's own tests prove -- and
    // it is what makes the reloaded value provably a FUNCTION OF THE HASHED BYTES
    // rather than of anything the codec had lying around.
    //
    // It runs HERE, not after the comparison, because the comparison judges the
    // rebuilt model against the pre-close one: a codec whose cache described a
    // behaviourally identical model would pass that and still not be returning
    // the artifact. Measured before this check existed: a codec ignoring its input
    // entirely minted the final state.
    close_round_trip(codec, &reloaded, &bytes)?;

    // THE ARTIFACT IS NOW RETAINED, AND THIS COMMENT USED TO SAY THE OPPOSITE.
    //
    // It read: "only its LENGTH is still wanted ... on a full pin it is ~180 MB that
    // would otherwise stay live". The first half was measured false by plan 04-17 —
    // wanting only the length is exactly what left `apr setfit train` able to report an
    // artifact's SHA-256 and unable to write the file that digest is of, because
    // `SetFitBundle::from_run_parts` is `pub(crate)` and there is no other public route
    // to the bytes. Re-serializing them out-of-crate would produce a SECOND
    // implementation of the twenty-field mapping whose output is not what was verified,
    // which is the one claim this whole phase exists to make true.
    //
    // So `bytes` now moves into `VerifiedOutcome` and reaches the caller through the
    // single consuming door `SetFitRun::<ArtifactReloadedAndVerified>::into_artifact_bytes`.
    // The length is still computed HERE, before the move, so `VerifyReport` is unchanged.
    //
    // What that costs, MEASURED and not estimated. `/usr/bin/time -l` around one full
    // verify of the calibrated fixture (`verify_full_pipeline_reaches_the_final_state`,
    // aarch64 Darwin, five runs on each side), maximum resident set size in bytes:
    //
    //           min          median       max
    //   before  59,146,240   59,473,920   65,748,992
    //   after   60,882,944   61,292,544   67,518,464
    //   delta   +1,736,704   +1,818,624   +1,769,472
    //
    // The retained buffer on that fixture is 1,824,298 bytes, so the median delta
    // accounts for it to within 0.3 percent: the cost is the artifact and nothing else.
    // (Re-measure the buffer with `cargo test -p aprender-train --lib --features setfit
    // verify_into_artifact_bytes -- --nocapture`.)
    //
    // The ~180 MB the old text cited was a FULL PIN, and it stays an estimate HERE: no test
    // in THIS crate's default suite produces one, because the production checkout is an
    // 86.7 MB offline prerequisite rather than a committed fixture. (Before Phase 5's 05-03
    // calibration edit, commit a63bb130b, no user-reachable path produced one at all —
    // orchestrator note F-10. That is closed: 05-07's spawned production ladder writes a real
    // artifact, measured at 90,777,156 bytes for the s8 cell.) The scaling is nevertheless the
    // plain one, because the retained buffer IS the artifact:
    // whatever a pinned artifact weighs, this holds exactly that and not a multiple of it.
    //
    // `drop(reloaded)` below is untouched and keeps its position. It is the LARGER
    // buffer — the bundle carries its tensors as hex, so roughly twice the artifact —
    // and it has no caller, so nothing was gained by holding it.
    let artifact_bytes = bytes.len();

    // (4) Rebuild from the reloaded bundle and from nothing else.
    let (mut rebuilt_encoder, rebuilt_head) = rebuild_from(&reloaded)?;
    // Same reasoning, and a larger figure: the bundle carries its tensors as hex, so it
    // is about twice the artifact's size and every tensor it holds has already been
    // decoded into the rebuilt encoder.
    drop(reloaded);

    // (5) Re-encode and re-predict FROM THE REBUILT MODEL.
    let after = probe_model(&mut rebuilt_encoder, &rebuilt_head, dataset, selection, config)?;

    // (6) Compare at the trusted tolerance.
    let (max_embedding_abs_diff, max_probability_abs_diff) =
        compare_probes(&probe, &after, tolerance)?;

    let report = VerifyReport {
        artifact_bytes,
        probe_rows: probe.ids.len(),
        embedding_dim: probe.embeddings.first().map_or(0, Vec::len),
        class_count: probe.probabilities.first().map_or(0, Vec::len),
        max_embedding_abs_diff,
        max_probability_abs_diff,
        tolerance_embedding_abs: tolerance.embedding_abs,
        tolerance_probability_abs: tolerance.probability_abs,
        // Reached only past `close_round_trip`, which returns `Err` otherwise.
        round_trip_closed: true,
    };
    Ok(VerifiedOutcome {
        encoder: rebuilt_encoder,
        head: rebuilt_head,
        report,
        probe: after,
        artifact_hash: hash,
        bytes,
    })
}

/// Assemble the final state's evidence.
///
/// `artifact_bytes` is the buffer [`run_verify_policy`] hashed and closed, moved through
/// unchanged. Nothing here re-serializes, re-hashes or re-derives it: the only thing that
/// happens to it is the newtype wrapper that keeps `Debug` printing a length.
pub(crate) fn verified_evidence(
    parts: super::HeadFittedParts,
    artifact_hash: [u8; 32],
    format_id: &str,
    report: VerifyReport,
    probe: VerifyProbe,
    head: MultinomialLogisticRegression,
    artifact_bytes: Vec<u8>,
) -> ArtifactVerifiedEvidence {
    ArtifactVerifiedEvidence {
        passed: parts.passed,
        head,
        report: parts.report,
        effective_lambda: parts.effective_lambda,
        ordered_labels: parts.ordered_labels,
        encode_ledger: parts.encode_ledger,
        encode_call_count: parts.encode_call_count,
        artifact_hash,
        format_id: format_id.to_string(),
        verify: report,
        probe,
        artifact_bytes: RetainedArtifactBytes(artifact_bytes),
    }
}

#[cfg(test)]
#[path = "verify_tests.rs"]
mod verify_tests;
