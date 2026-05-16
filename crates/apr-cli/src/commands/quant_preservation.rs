//! CRUX-B-19 — dequant→requant metadata preservation classifier.
//!
//! Validates the round-trip property that `apr dequant` + `apr quantize`
//! preserves all `general.*` GGUF metadata (except the two fields that
//! MUST change to reflect the new qtype: `general.quantization_version`
//! and `general.file_type`) and preserves `tokenizer.*` byte-for-byte.
//!
//! Surface: `apr quant-preservation-lint --reference REF.gguf --requant REQ.gguf [--json]`
//!
//! The classifier is a pure-function over two parsed metadata maps; it
//! gates any future implementation of the dequant→requant pipeline.
//! The full pipeline (the `apr dequant` CLI command) is left to a
//! separate ticket; this PR captures the contract value and the gate.
//!
//! See `contracts/crux-B-19-v1.yaml`.

use std::collections::BTreeMap;
use std::path::Path;

use aprender::format::gguf::{GgufReader, GgufValue};
use serde::Serialize;

use crate::error::{CliError, Result};

/// Fields that MUST change across a quant-format round-trip (and so are
/// excluded from the equality check). Defined by the GGUF spec — the
/// quantization_version is per-qtype, and `general.file_type` encodes
/// the qtype enum on disk.
const QUANT_VOLATILE_KEYS: &[&str] = &[
    "general.quantization_version",
    "general.file_type",
];

#[derive(Debug, Clone, Serialize)]
pub struct DiffEntry {
    pub key: String,
    pub reference: String,
    pub requant: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct PreservationReport {
    pub reference: String,
    pub requant: String,
    pub general_keys_checked: usize,
    pub general_keys_diverged: Vec<DiffEntry>,
    pub general_keys_missing_in_requant: Vec<String>,
    pub general_keys_added_in_requant: Vec<String>,
    pub tokenizer_keys_checked: usize,
    pub tokenizer_keys_diverged: Vec<DiffEntry>,
    pub tokenizer_keys_missing_in_requant: Vec<String>,
    pub tokenizer_keys_added_in_requant: Vec<String>,
    pub passed: bool,
}

/// Read GGUF metadata from disk.
pub fn read_gguf_metadata(path: &Path) -> Result<BTreeMap<String, GgufValue>> {
    let reader = GgufReader::from_file(path).map_err(|e| {
        CliError::ValidationFailed(format!(
            "GGUF parse {}: {e}",
            path.display()
        ))
    })?;
    Ok(reader.metadata)
}

/// Pure-function classifier: compare two metadata maps and return a report.
///
/// The contract property is:
/// - `general.*` keys (excluding `quantization_version` + `file_type`) must
///   appear in both with equal values.
/// - `tokenizer.*` keys must appear in both with equal values.
///
/// Volatile quantization fields are excluded from the equality check.
pub fn classify_preservation(
    reference: &BTreeMap<String, GgufValue>,
    requant: &BTreeMap<String, GgufValue>,
    ref_path: String,
    req_path: String,
) -> PreservationReport {
    let mut general_diverged = Vec::new();
    let mut general_missing = Vec::new();
    let mut general_added = Vec::new();
    let mut tokenizer_diverged = Vec::new();
    let mut tokenizer_missing = Vec::new();
    let mut tokenizer_added = Vec::new();
    let mut general_checked = 0usize;
    let mut tokenizer_checked = 0usize;

    for (key, ref_val) in reference {
        let prefix_general = key.starts_with("general.");
        let prefix_tokenizer = key.starts_with("tokenizer.");
        if !prefix_general && !prefix_tokenizer {
            continue;
        }
        if prefix_general && QUANT_VOLATILE_KEYS.contains(&key.as_str()) {
            continue;
        }
        match requant.get(key) {
            Some(req_val) => {
                if prefix_general {
                    general_checked += 1;
                } else {
                    tokenizer_checked += 1;
                }
                if !values_equal(ref_val, req_val) {
                    let entry = DiffEntry {
                        key: key.clone(),
                        reference: format!("{ref_val:?}"),
                        requant: format!("{req_val:?}"),
                    };
                    if prefix_general {
                        general_diverged.push(entry);
                    } else {
                        tokenizer_diverged.push(entry);
                    }
                }
            }
            None => {
                if prefix_general {
                    general_missing.push(key.clone());
                } else {
                    tokenizer_missing.push(key.clone());
                }
            }
        }
    }

    // Keys that appear in requant but not reference (excluding volatile).
    for key in requant.keys() {
        let prefix_general = key.starts_with("general.");
        let prefix_tokenizer = key.starts_with("tokenizer.");
        if !prefix_general && !prefix_tokenizer {
            continue;
        }
        if prefix_general && QUANT_VOLATILE_KEYS.contains(&key.as_str()) {
            continue;
        }
        if !reference.contains_key(key) {
            if prefix_general {
                general_added.push(key.clone());
            } else {
                tokenizer_added.push(key.clone());
            }
        }
    }

    let passed = general_diverged.is_empty()
        && general_missing.is_empty()
        && general_added.is_empty()
        && tokenizer_diverged.is_empty()
        && tokenizer_missing.is_empty()
        && tokenizer_added.is_empty();

    PreservationReport {
        reference: ref_path,
        requant: req_path,
        general_keys_checked: general_checked,
        general_keys_diverged: general_diverged,
        general_keys_missing_in_requant: general_missing,
        general_keys_added_in_requant: general_added,
        tokenizer_keys_checked: tokenizer_checked,
        tokenizer_keys_diverged: tokenizer_diverged,
        tokenizer_keys_missing_in_requant: tokenizer_missing,
        tokenizer_keys_added_in_requant: tokenizer_added,
        passed,
    }
}

/// Structural equality on `GgufValue` — we compare via Debug formatting to
/// avoid imposing a PartialEq derivation upstream. For all current
/// `GgufValue` variants the Debug output is canonical (no platform-dependent
/// formatting), so this is a stable comparison.
fn values_equal(a: &GgufValue, b: &GgufValue) -> bool {
    format!("{a:?}") == format!("{b:?}")
}

/// Render the report as a human-readable summary.
pub fn render_text(report: &PreservationReport) -> String {
    let mut out = String::new();
    out.push_str(&format!("APR Quant Preservation Lint (CRUX-B-19)\n"));
    out.push_str(&format!("  reference:  {}\n", report.reference));
    out.push_str(&format!("  requant:    {}\n", report.requant));
    out.push_str(&format!(
        "  general.*  : {} checked / {} diverged / {} missing / {} added\n",
        report.general_keys_checked,
        report.general_keys_diverged.len(),
        report.general_keys_missing_in_requant.len(),
        report.general_keys_added_in_requant.len(),
    ));
    out.push_str(&format!(
        "  tokenizer.*: {} checked / {} diverged / {} missing / {} added\n",
        report.tokenizer_keys_checked,
        report.tokenizer_keys_diverged.len(),
        report.tokenizer_keys_missing_in_requant.len(),
        report.tokenizer_keys_added_in_requant.len(),
    ));
    if !report.general_keys_diverged.is_empty() {
        out.push_str("  diverged general.*:\n");
        for e in &report.general_keys_diverged {
            out.push_str(&format!("    {} :: {} → {}\n", e.key, e.reference, e.requant));
        }
    }
    if !report.tokenizer_keys_diverged.is_empty() {
        out.push_str("  diverged tokenizer.*:\n");
        for e in &report.tokenizer_keys_diverged {
            out.push_str(&format!("    {} :: {} → {}\n", e.key, e.reference, e.requant));
        }
    }
    out.push_str(&format!(
        "  verdict:    {}\n",
        if report.passed { "PRESERVED" } else { "VIOLATED" }
    ));
    out
}

/// Entry point.
pub fn run(reference: &Path, requant: &Path, json: bool) -> Result<()> {
    let ref_meta = read_gguf_metadata(reference)?;
    let req_meta = read_gguf_metadata(requant)?;

    let report = classify_preservation(
        &ref_meta,
        &req_meta,
        reference.display().to_string(),
        requant.display().to_string(),
    );

    if json {
        let serialized = serde_json::to_string_pretty(&report).map_err(|e| {
            CliError::ValidationFailed(format!("serialize report: {e}"))
        })?;
        println!("{serialized}");
    } else {
        print!("{}", render_text(&report));
    }

    if !report.passed {
        return Err(CliError::ValidationFailed(
            "quant-preservation: metadata invariant violated (see report)".to_string(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta_kv(pairs: &[(&str, GgufValue)]) -> BTreeMap<String, GgufValue> {
        pairs.iter().map(|(k, v)| (k.to_string(), v.clone())).collect()
    }

    /// FALSIFY-CRUX-B-19-001 — `general.*` keys (except quantization_version +
    /// file_type) must be byte-equal across the round-trip.
    #[test]
    fn falsify_crux_b_19_001_general_preserved_modulo_volatile() {
        let reference = meta_kv(&[
            ("general.architecture", GgufValue::String("llama".to_string())),
            ("general.name", GgufValue::String("test-model".to_string())),
            ("general.quantization_version", GgufValue::Uint32(2)),
            ("general.file_type", GgufValue::Uint32(10)), // Q4_K
            ("tokenizer.ggml.bos_token_id", GgufValue::Uint32(1)),
        ]);
        // requant: same architecture+name; volatile fields differ (Q6_K).
        let requant = meta_kv(&[
            ("general.architecture", GgufValue::String("llama".to_string())),
            ("general.name", GgufValue::String("test-model".to_string())),
            ("general.quantization_version", GgufValue::Uint32(3)),
            ("general.file_type", GgufValue::Uint32(14)), // Q6_K
            ("tokenizer.ggml.bos_token_id", GgufValue::Uint32(1)),
        ]);
        let report = classify_preservation(
            &reference,
            &requant,
            "ref.gguf".into(),
            "req.gguf".into(),
        );
        assert!(report.passed, "expected PRESERVED, got {report:#?}");
        assert_eq!(report.general_keys_checked, 2, "expected 2 non-volatile general keys");
        assert!(report.general_keys_diverged.is_empty());
        assert!(report.tokenizer_keys_diverged.is_empty());
    }

    /// FALSIFY-CRUX-B-19-001 fail case — a non-volatile general.* key changes
    /// → classifier MUST flag VIOLATED.
    #[test]
    fn classifier_flags_general_name_change() {
        let reference = meta_kv(&[
            ("general.name", GgufValue::String("alpha".to_string())),
        ]);
        let requant = meta_kv(&[
            ("general.name", GgufValue::String("beta".to_string())),
        ]);
        let report = classify_preservation(
            &reference,
            &requant,
            "r.gguf".into(),
            "q.gguf".into(),
        );
        assert!(!report.passed);
        assert_eq!(report.general_keys_diverged.len(), 1);
        assert_eq!(report.general_keys_diverged[0].key, "general.name");
    }

    /// FALSIFY-CRUX-B-19-002 — tokenizer.* keys must be byte-identical.
    #[test]
    fn falsify_crux_b_19_002_tokenizer_byte_identical() {
        let vocab_a = GgufValue::ArrayString(vec![
            "<bos>".to_string(),
            "<eos>".to_string(),
            "hello".to_string(),
        ]);
        let merges = GgufValue::ArrayString(vec![
            "h e".to_string(),
            "he ll".to_string(),
        ]);
        let reference = meta_kv(&[
            ("tokenizer.ggml.tokens", vocab_a.clone()),
            ("tokenizer.ggml.merges", merges.clone()),
        ]);
        let requant = meta_kv(&[
            ("tokenizer.ggml.tokens", vocab_a),
            ("tokenizer.ggml.merges", merges),
        ]);
        let report = classify_preservation(
            &reference,
            &requant,
            "r.gguf".into(),
            "q.gguf".into(),
        );
        assert!(report.passed);
        assert_eq!(report.tokenizer_keys_checked, 2);
    }

    /// FALSIFY-CRUX-B-19-002 fail case — vocab order changes → VIOLATED.
    #[test]
    fn classifier_flags_vocab_reorder() {
        let reference = meta_kv(&[(
            "tokenizer.ggml.tokens",
            GgufValue::ArrayString(vec!["a".to_string(), "b".to_string()]),
        )]);
        let requant = meta_kv(&[(
            "tokenizer.ggml.tokens",
            GgufValue::ArrayString(vec!["b".to_string(), "a".to_string()]),
        )]);
        let report = classify_preservation(
            &reference,
            &requant,
            "r.gguf".into(),
            "q.gguf".into(),
        );
        assert!(!report.passed);
        assert_eq!(report.tokenizer_keys_diverged.len(), 1);
    }

    /// Missing-in-requant case — every required general.* key must be present.
    #[test]
    fn classifier_flags_missing_general_key() {
        let reference = meta_kv(&[
            ("general.architecture", GgufValue::String("llama".to_string())),
            ("general.name", GgufValue::String("m".to_string())),
        ]);
        let requant = meta_kv(&[
            ("general.architecture", GgufValue::String("llama".to_string())),
            // general.name missing
        ]);
        let report = classify_preservation(
            &reference,
            &requant,
            "r.gguf".into(),
            "q.gguf".into(),
        );
        assert!(!report.passed);
        assert_eq!(report.general_keys_missing_in_requant, vec!["general.name"]);
    }

    /// Volatile fields are NOT compared even when they differ.
    #[test]
    fn volatile_fields_ignored() {
        for k in QUANT_VOLATILE_KEYS {
            let reference = meta_kv(&[
                (k, GgufValue::Uint32(2)),
                ("general.architecture", GgufValue::String("llama".to_string())),
            ]);
            let requant = meta_kv(&[
                (k, GgufValue::Uint32(99)),
                ("general.architecture", GgufValue::String("llama".to_string())),
            ]);
            let report = classify_preservation(
                &reference,
                &requant,
                "r.gguf".into(),
                "q.gguf".into(),
            );
            assert!(report.passed, "volatile key {k} should be ignored");
        }
    }

    /// Non-general / non-tokenizer keys are ignored.
    #[test]
    fn non_general_non_tokenizer_keys_ignored() {
        let reference = meta_kv(&[
            ("llama.attention.head_count", GgufValue::Uint32(32)),
            ("general.name", GgufValue::String("m".to_string())),
        ]);
        let requant = meta_kv(&[
            // llama.* differs but it's neither general.* nor tokenizer.*
            ("llama.attention.head_count", GgufValue::Uint32(99)),
            ("general.name", GgufValue::String("m".to_string())),
        ]);
        let report = classify_preservation(
            &reference,
            &requant,
            "r.gguf".into(),
            "q.gguf".into(),
        );
        assert!(report.passed, "non-general/non-tokenizer must be ignored");
    }

    /// Added keys in requant (that weren't in reference) — flagged.
    #[test]
    fn classifier_flags_added_tokenizer_key() {
        let reference = meta_kv(&[
            ("tokenizer.ggml.tokens", GgufValue::ArrayString(vec![])),
        ]);
        let requant = meta_kv(&[
            ("tokenizer.ggml.tokens", GgufValue::ArrayString(vec![])),
            ("tokenizer.ggml.merges", GgufValue::ArrayString(vec![])),
        ]);
        let report = classify_preservation(
            &reference,
            &requant,
            "r.gguf".into(),
            "q.gguf".into(),
        );
        assert!(!report.passed);
        assert_eq!(
            report.tokenizer_keys_added_in_requant,
            vec!["tokenizer.ggml.merges"]
        );
    }
}
