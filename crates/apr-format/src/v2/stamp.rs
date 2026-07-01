//! APR v2 provenance stamping — SHIP-009 full-discharge enabler.
//!
//! Read an APR v2 file, patch its provenance metadata (`license`,
//! `data_source`, `data_license`), re-serialize. Tensor bytes are copied
//! verbatim; header flags (`QUANTIZED`, `HAS_VOCAB`, …) are preserved so
//! round-tripping a quantized model does not silently drop the flag that
//! downstream consumers branch on.
//!
//! Motivates: the 7B Q4_K teacher shipped at commit `06a3eae38` predates
//! `GATE-APR-PROV-001/002/003` (commit `8f0607d42`), so its `.apr` has
//! `license: None / data_source: None / data_license: None`. `apr inspect`
//! renders those as `(missing)`, and `GATE-APR-PROV-004` — the algorithm
//! gate for `AC-SHIP1-009` — rejects the `(None, None, None)` triple at
//! full-discharge time. This helper closes the tooling gap; the release
//! cycle (re-stamp → re-upload → manifest-sha256 refresh) is a follow-up.
//!
//! Contract reference: `contracts/apr-provenance-v1.yaml` (v1.1.0,
//! GATE-APR-PROV-001..004). Spec reference:
//! `docs/specifications/aprender-train/ship-two-models-spec.md` §4.2
//! AC-SHIP1-009 + v2.52.0 amendment (teacher provenance gap).

use super::{AprV2Reader, AprV2Writer, V2FormatError};

/// In-place field patches. `None` means "leave unchanged"; `Some("")` is a
/// legitimate explicit clear (not currently contract-approved but kept
/// distinct from `None` so callers can express intent).
///
/// PMAT-690 P0-K extension (2026-05-17): `hf_architecture` and
/// `hf_model_type` were added so pre-P0-K APRs can be patched in place
/// without re-import. The §86 SPEC amendment surfaced this: P2-E's
/// epoch-49 checkpoint (val_loss=4.62, the best MODEL-2 result on
/// record) has architecture="LlamaForCausalLM" (the P0-H fallback) and
/// hf_architecture=null because its init APR pre-dates P0-K. Without
/// in-place stamping, the 50 P2-E checkpoints (~125 GB) are unusable
/// as `--init` for resume training because apr pretrain reads the
/// (wrong) architecture stamp and rejects the load. Stamping the
/// correct hf_architecture + a corrected `architecture` family slug
/// salvages the entire run without a 53-min retrain.
#[derive(Debug, Clone, Default)]
pub struct ProvenancePatch {
    /// SPDX license identifier to stamp into the metadata.
    pub license: Option<String>,
    /// Training-data source (dataset identifier or "teacher-only").
    pub data_source: Option<String>,
    /// SPDX license for `data_source`.
    pub data_license: Option<String>,
    /// HuggingFace class name from `config.json::architectures[0]`
    /// (e.g., "Qwen2ForCausalLM"). PMAT-690 P0-K extension.
    pub hf_architecture: Option<String>,
    /// HuggingFace `config.json::model_type` (e.g., "qwen2").
    /// PMAT-690 P0-K extension.
    pub hf_model_type: Option<String>,
    /// Lowercase architecture family slug (e.g., "qwen2", "llama").
    /// PMAT-690 P0-K extension. Distinct from `hf_architecture` (which
    /// is the HF class name like "Qwen2ForCausalLM"). This is the
    /// field that `apr pretrain --init` reads for arch dispatch, so
    /// patching this is what makes a pre-P0-K checkpoint resumable.
    pub architecture: Option<String>,
    /// Tokenizer vocabulary (token strings indexed by token-id). When
    /// `Some`, the stamp embeds these strings into
    /// `metadata.custom["tokenizer.vocabulary"]` (as a JSON array)
    /// AND sets the HAS_VOCAB header flag — making the resulting APR
    /// self-contained for `apr run` inference (which rejects APRs
    /// without an embedded tokenizer per PMAT-172).
    ///
    /// PMAT-690 P3-C-prep follow-up (2026-05-17, defect 1 from
    /// publish-readiness preflight on P2-E ep49): pre-P0-K APRs lack
    /// embedded tokenizers because the training init didn't have
    /// one. Without this stamp extension, the §86 salvage produces a
    /// 6.0 GB HF-publish-ready directory that fails the headline
    /// `apr run` smoke test.
    pub tokenizer_vocab: Option<Vec<String>>,
    /// BPE merge rules (e.g., `["Ä t", "i n", ...]`). When
    /// `Some`, embedded into `metadata.custom["tokenizer.merges"]`.
    pub tokenizer_merges: Option<Vec<String>>,
    /// Tokenizer model type (e.g., "BPE", "Unigram"). Optional metadata
    /// for `apr inspect` to surface.
    pub tokenizer_model_type: Option<String>,
}

impl ProvenancePatch {
    /// `true` iff at least one field would change. Guards against a
    /// no-op rewrite producing a pointless new file.
    #[must_use]
    pub fn has_any(&self) -> bool {
        self.license.is_some()
            || self.data_source.is_some()
            || self.data_license.is_some()
            || self.hf_architecture.is_some()
            || self.hf_model_type.is_some()
            || self.architecture.is_some()
            || self.tokenizer_vocab.is_some()
            || self.tokenizer_merges.is_some()
            || self.tokenizer_model_type.is_some()
    }
}

/// Patch provenance metadata on an existing APR v2 buffer and return the
/// re-serialized bytes.
///
/// # Errors
/// Returns `V2FormatError::InvalidHeader` if:
///   - `input` is not a valid APR v2 buffer (propagated from
///     `AprV2Reader::from_bytes`)
///   - `patch.has_any()` is `false` — a no-op stamp is rejected
///     up-front so callers cannot accidentally rewrite without
///     changing the artifact
///
/// # Guarantees
///   - Header flags from `input` are preserved in the output (LAYOUT_ROW_MAJOR
///     is always added regardless of input, per LAYOUT-002 jidoka)
///   - Tensor bytes are copied verbatim — no quantize/dequantize round-trip
///   - Sort-by-name ordering matches `AprV2Writer::write()` (tensor index
///     is sorted, so the re-serialized index is canonical)
///
/// # Non-guarantees
///   - Footer checksum WILL change (metadata bytes moved)
///   - sha256 of the output file WILL differ from the input (by design —
///     that is the whole point of a stamp operation)
pub fn stamp_provenance_bytes(
    input: &[u8],
    patch: &ProvenancePatch,
) -> Result<Vec<u8>, V2FormatError> {
    if !patch.has_any() {
        return Err(V2FormatError::InvalidHeader(
            "stamp_provenance_bytes: patch has no fields set — \
             refusing to rewrite without changes"
                .to_string(),
        ));
    }

    let reader = AprV2Reader::from_bytes(input)?;

    let original_flags = reader.header().flags;
    let mut new_metadata = reader.metadata().clone();

    if let Some(ref lic) = patch.license {
        new_metadata.license = Some(lic.clone());
    }
    if let Some(ref ds) = patch.data_source {
        new_metadata.data_source = Some(ds.clone());
    }
    if let Some(ref dl) = patch.data_license {
        new_metadata.data_license = Some(dl.clone());
    }
    // PMAT-690 P0-K extension: HF identity + architecture family
    if let Some(ref ha) = patch.hf_architecture {
        new_metadata.hf_architecture = Some(ha.clone());
    }
    if let Some(ref hmt) = patch.hf_model_type {
        new_metadata.hf_model_type = Some(hmt.clone());
    }
    if let Some(ref arch) = patch.architecture {
        new_metadata.architecture = Some(arch.clone());
    }
    // PMAT-690 P3-C-prep follow-up (defect 1): embed tokenizer into the
    // custom JSON metadata. Mirrors the apr-import path's behaviour
    // (`insert_f32_tokenizer_metadata` in converter::write); we duplicate
    // the key names here so a stamped APR has the same shape as a
    // freshly-imported one.
    let mut set_has_vocab = false;
    if let Some(ref vocab) = patch.tokenizer_vocab {
        if !vocab.is_empty() {
            let vocab_array: Vec<serde_json::Value> = vocab
                .iter()
                .map(|s| serde_json::Value::String(s.clone()))
                .collect();
            new_metadata.custom.insert(
                "tokenizer.vocabulary".to_string(),
                serde_json::Value::Array(vocab_array),
            );
            new_metadata.custom.insert(
                "tokenizer.vocab_size".to_string(),
                serde_json::Value::Number(serde_json::Number::from(vocab.len())),
            );
            set_has_vocab = true;
        }
    }
    if let Some(ref merges) = patch.tokenizer_merges {
        if !merges.is_empty() {
            let merges_array: Vec<serde_json::Value> = merges
                .iter()
                .map(|s| serde_json::Value::String(s.clone()))
                .collect();
            new_metadata.custom.insert(
                "tokenizer.merges".to_string(),
                serde_json::Value::Array(merges_array),
            );
        }
    }
    if let Some(ref mt) = patch.tokenizer_model_type {
        new_metadata.custom.insert(
            "tokenizer.model_type".to_string(),
            serde_json::Value::String(mt.clone()),
        );
    }

    // PMAT-172 (defect 1 root cause): `apr run` checks the HAS_VOCAB
    // flag before allowing inference. Setting tokenizer_vocab without
    // setting the flag would still surface the "missing embedded tokenizer"
    // error.
    let effective_flags = if set_has_vocab {
        original_flags.with(super::AprV2Flags::HAS_VOCAB)
    } else {
        original_flags
    };

    let mut writer = AprV2Writer::new(new_metadata);
    writer.set_header_flags(effective_flags);

    // Copy every tensor by name; AprV2Writer sorts by name internally on
    // write(), so input ordering is irrelevant here.
    for name in reader.tensor_names() {
        let entry = reader
            .get_tensor(name)
            .ok_or_else(|| V2FormatError::InvalidHeader(format!("tensor {name} vanished")))?;
        let data = reader
            .get_tensor_data(name)
            .ok_or_else(|| V2FormatError::InvalidHeader(format!("tensor {name} has no data")))?;
        writer.add_tensor(
            name.to_string(),
            entry.dtype,
            entry.shape.clone(),
            data.to_vec(),
        );
    }

    writer.write()
}

#[cfg(test)]
mod tests {
    use super::super::{AprV2Flags, AprV2Metadata, TensorDType};
    use super::*;

    /// Build a minimal valid APR v2 buffer for round-trip tests.
    fn minimal_apr_with_flags(flags: u16) -> Vec<u8> {
        let metadata = AprV2Metadata::new("stamp-test");
        let mut writer = AprV2Writer::new(metadata);
        writer.set_header_flags(AprV2Flags::from_bits(flags));
        writer.add_tensor(
            "weight",
            TensorDType::F32,
            vec![2, 3],
            vec![0u8; 24], // 2 * 3 * 4 bytes
        );
        writer.write().expect("write test apr")
    }

    #[test]
    fn stamp_populates_all_three_fields_when_source_is_unpopulated() {
        let input = minimal_apr_with_flags(0);
        let patch = ProvenancePatch {
            license: Some("Apache-2.0".into()),
            data_source: Some("huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct".into()),
            data_license: Some("Qwen-License-Agreement-v1".into()),
            hf_architecture: None,
            hf_model_type: None,
            architecture: None,
            tokenizer_vocab: None,
            tokenizer_merges: None,
            tokenizer_model_type: None,
        };

        let output = stamp_provenance_bytes(&input, &patch).expect("stamp must succeed");

        let reader = AprV2Reader::from_bytes(&output).expect("stamped buffer must parse");
        let md = reader.metadata();
        assert_eq!(md.license.as_deref(), Some("Apache-2.0"));
        assert_eq!(
            md.data_source.as_deref(),
            Some("huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct")
        );
        assert_eq!(
            md.data_license.as_deref(),
            Some("Qwen-License-Agreement-v1")
        );
    }

    #[test]
    fn stamp_preserves_tensor_data_byte_for_byte() {
        let input = minimal_apr_with_flags(0);
        let input_reader = AprV2Reader::from_bytes(&input).unwrap();
        let original_bytes: Vec<u8> = input_reader
            .get_tensor_data("weight")
            .expect("input has weight")
            .to_vec();

        let patch = ProvenancePatch {
            license: Some("MIT".into()),
            ..Default::default()
        };
        let output = stamp_provenance_bytes(&input, &patch).unwrap();

        let out_reader = AprV2Reader::from_bytes(&output).unwrap();
        let round_tripped = out_reader
            .get_tensor_data("weight")
            .expect("output has weight");

        assert_eq!(
            original_bytes.as_slice(),
            round_tripped,
            "tensor bytes must survive stamp verbatim"
        );
    }

    #[test]
    fn stamp_preserves_header_flags() {
        // Simulate a quantized source: set QUANTIZED | HAS_VOCAB on input.
        let flags = AprV2Flags::QUANTIZED | AprV2Flags::HAS_VOCAB;
        let input = minimal_apr_with_flags(flags);

        let in_reader = AprV2Reader::from_bytes(&input).unwrap();
        assert!(in_reader.header().flags.contains(AprV2Flags::QUANTIZED));
        assert!(in_reader.header().flags.contains(AprV2Flags::HAS_VOCAB));

        let patch = ProvenancePatch {
            license: Some("Apache-2.0".into()),
            ..Default::default()
        };
        let output = stamp_provenance_bytes(&input, &patch).unwrap();

        let out_reader = AprV2Reader::from_bytes(&output).unwrap();
        // Input flags preserved:
        assert!(
            out_reader.header().flags.contains(AprV2Flags::QUANTIZED),
            "QUANTIZED flag must survive stamp"
        );
        assert!(
            out_reader.header().flags.contains(AprV2Flags::HAS_VOCAB),
            "HAS_VOCAB flag must survive stamp"
        );
        // LAYOUT-002 jidoka still engaged:
        assert!(
            out_reader
                .header()
                .flags
                .contains(AprV2Flags::LAYOUT_ROW_MAJOR),
            "LAYOUT_ROW_MAJOR must always be set"
        );
    }

    #[test]
    fn stamp_rejects_empty_patch() {
        let input = minimal_apr_with_flags(0);
        let empty = ProvenancePatch::default();
        let err = stamp_provenance_bytes(&input, &empty).unwrap_err();
        let msg = format!("{err:?}");
        assert!(
            msg.contains("patch has no fields"),
            "empty-patch error must be explicit: {msg}"
        );
    }

    #[test]
    fn stamp_allows_partial_patch_leaving_other_fields_unchanged() {
        // Input already has a license (pretend) but no data_* fields.
        let mut md = AprV2Metadata::new("partial-test");
        md.license = Some("Apache-2.0".into());
        let mut writer = AprV2Writer::new(md);
        writer.add_tensor("w", TensorDType::F32, vec![4], vec![0u8; 16]);
        let input = writer.write().unwrap();

        // Only patch data_source.
        let patch = ProvenancePatch {
            data_source: Some("teacher-only".into()),
            ..Default::default()
        };
        let output = stamp_provenance_bytes(&input, &patch).unwrap();

        let out_reader = AprV2Reader::from_bytes(&output).unwrap();
        assert_eq!(
            out_reader.metadata().license.as_deref(),
            Some("Apache-2.0"),
            "unchanged license must survive"
        );
        assert_eq!(
            out_reader.metadata().data_source.as_deref(),
            Some("teacher-only"),
            "patched data_source must land"
        );
        assert!(
            out_reader.metadata().data_license.is_none(),
            "untouched data_license must remain None"
        );
    }

    #[test]
    fn stamp_is_idempotent_under_identical_patch() {
        let input = minimal_apr_with_flags(0);
        let patch = ProvenancePatch {
            license: Some("Apache-2.0".into()),
            data_source: Some("teacher-only".into()),
            data_license: Some("Apache-2.0".into()),
            hf_architecture: None,
            hf_model_type: None,
            architecture: None,
            tokenizer_vocab: None,
            tokenizer_merges: None,
            tokenizer_model_type: None,
        };

        let first = stamp_provenance_bytes(&input, &patch).unwrap();
        let second = stamp_provenance_bytes(&first, &patch).unwrap();
        assert_eq!(
            first, second,
            "applying the same patch twice must be byte-identical (idempotent)"
        );
    }
}
