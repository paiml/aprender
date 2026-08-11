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
const QUANT_VOLATILE_KEYS: &[&str] = &["general.quantization_version", "general.file_type"];

/// Longest verbatim rendering of a single metadata value kept in a divergence
/// entry. Beyond this the value is summarised.
///
/// GGUF tokenizer metadata is not small: `tokenizer.ggml.merges` on a Qwen
/// vocab renders to 4.8 MB and `tokenizer.ggml.tokens` to 4.4 MB via `Debug`.
/// Emitting those verbatim produced a 13 MB, 19-line report (14.9 MB under
/// `--json`) that wedged terminals and blew CI log budgets, burying the
/// actionable one-line scalar divergences on either side of it.
pub const DIFF_VALUE_MAX_CHARS: usize = 200;

/// Characters of the verbatim rendering retained as a head sample when a
/// value is summarised.
const DIFF_VALUE_HEAD_CHARS: usize = 96;

#[derive(Debug, Clone, Serialize)]
pub struct DiffEntry {
    pub key: String,
    pub reference: String,
    pub requant: String,
    /// Where the two values first differ, when that is computable (same array
    /// variant on both sides). `None` for scalars and cross-type divergences,
    /// where the rendered values already show the difference.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_difference: Option<String>,
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
    let reader = GgufReader::from_file(path)
        .map_err(|e| CliError::ValidationFailed(format!("GGUF parse {}: {e}", path.display())))?;
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
                        reference: render_value_bounded(ref_val),
                        requant: render_value_bounded(req_val),
                        first_difference: describe_first_difference(ref_val, req_val),
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

/// Variant name of a `GgufValue`, used as the label of a summarised value.
fn value_kind(v: &GgufValue) -> &'static str {
    match v {
        GgufValue::Uint8(_) => "Uint8",
        GgufValue::Int8(_) => "Int8",
        GgufValue::Uint16(_) => "Uint16",
        GgufValue::Int16(_) => "Int16",
        GgufValue::Uint32(_) => "Uint32",
        GgufValue::Int32(_) => "Int32",
        GgufValue::Float32(_) => "Float32",
        GgufValue::Bool(_) => "Bool",
        GgufValue::String(_) => "String",
        GgufValue::Uint64(_) => "Uint64",
        GgufValue::Int64(_) => "Int64",
        GgufValue::Float64(_) => "Float64",
        GgufValue::ArrayUint32(_) => "ArrayUint32",
        GgufValue::ArrayInt32(_) => "ArrayInt32",
        GgufValue::ArrayFloat32(_) => "ArrayFloat32",
        GgufValue::ArrayString(_) => "ArrayString",
    }
}

/// Element count for array-valued metadata; `None` for scalars.
fn array_len(v: &GgufValue) -> Option<usize> {
    match v {
        GgufValue::ArrayUint32(a) => Some(a.len()),
        GgufValue::ArrayInt32(a) => Some(a.len()),
        GgufValue::ArrayFloat32(a) => Some(a.len()),
        GgufValue::ArrayString(a) => Some(a.len()),
        _ => None,
    }
}

/// Hex sha256 of the canonical rendering, truncated to 16 chars — enough to
/// tell two vocabularies apart in a log line without reproducing either.
fn digest16(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let full = Sha256::digest(bytes);
    full.iter().take(8).map(|b| format!("{b:02x}")).collect()
}

/// Render a metadata value for a divergence report, bounded in size.
///
/// Short values (every scalar, and small arrays) are rendered verbatim — the
/// `general.architecture :: String("qwen35") → String("qwen2")` form is
/// exactly what a reader needs. Anything longer than
/// [`DIFF_VALUE_MAX_CHARS`] is replaced by a summary carrying the element
/// count (or character count), a content digest, and a head sample.
pub fn render_value_bounded(v: &GgufValue) -> String {
    let full = format!("{v:?}");
    if full.chars().count() <= DIFF_VALUE_MAX_CHARS {
        return full;
    }
    let digest = digest16(full.as_bytes());
    let head: String = full.chars().take(DIFF_VALUE_HEAD_CHARS).collect();
    match array_len(v) {
        Some(n) => format!("{}(len={n}, sha256={digest}, head={head}…)", value_kind(v)),
        None => format!(
            "{}(chars={}, sha256={digest}, head={head}…)",
            value_kind(v),
            full.chars().count()
        ),
    }
}

/// For two arrays of the same variant, the index of the first element that
/// differs (or the point at which one ran out). `None` when the values are
/// not a same-variant array pair.
pub fn describe_first_difference(a: &GgufValue, b: &GgufValue) -> Option<String> {
    fn locate<T: PartialEq>(x: &[T], y: &[T]) -> String {
        match x.iter().zip(y.iter()).position(|(p, q)| p != q) {
            Some(i) => format!(
                "first differing element: index {i} (len {} vs {})",
                x.len(),
                y.len()
            ),
            None => format!(
                "common prefix identical; lengths differ: {} vs {}",
                x.len(),
                y.len()
            ),
        }
    }
    match (a, b) {
        (GgufValue::ArrayUint32(x), GgufValue::ArrayUint32(y)) => Some(locate(x, y)),
        (GgufValue::ArrayInt32(x), GgufValue::ArrayInt32(y)) => Some(locate(x, y)),
        (GgufValue::ArrayString(x), GgufValue::ArrayString(y)) => Some(locate(x, y)),
        (GgufValue::ArrayFloat32(x), GgufValue::ArrayFloat32(y)) => {
            let pos = x
                .iter()
                .zip(y.iter())
                .position(|(p, q)| p.to_bits() != q.to_bits());
            Some(match pos {
                Some(i) => format!(
                    "first differing element: index {i} (len {} vs {})",
                    x.len(),
                    y.len()
                ),
                None => format!(
                    "common prefix identical; lengths differ: {} vs {}",
                    x.len(),
                    y.len()
                ),
            })
        }
        _ => None,
    }
}

/// One divergence line, plus an indented locator line when we could compute
/// where two arrays first differ.
fn push_diff_entry(out: &mut String, e: &DiffEntry) {
    out.push_str(&format!(
        "    {} :: {} → {}\n",
        e.key, e.reference, e.requant
    ));
    if let Some(loc) = &e.first_difference {
        out.push_str(&format!("      {loc}\n"));
    }
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
            push_diff_entry(&mut out, e);
        }
    }
    if !report.tokenizer_keys_diverged.is_empty() {
        out.push_str("  diverged tokenizer.*:\n");
        for e in &report.tokenizer_keys_diverged {
            push_diff_entry(&mut out, e);
        }
    }
    out.push_str(&format!(
        "  verdict:    {}\n",
        if report.passed {
            "PRESERVED"
        } else {
            "VIOLATED"
        }
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
        let serialized = serde_json::to_string_pretty(&report)
            .map_err(|e| CliError::ValidationFailed(format!("serialize report: {e}")))?;
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
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect()
    }

    /// FALSIFY-CRUX-B-19-001 — `general.*` keys (except quantization_version +
    /// file_type) must be byte-equal across the round-trip.
    #[test]
    fn falsify_crux_b_19_001_general_preserved_modulo_volatile() {
        let reference = meta_kv(&[
            (
                "general.architecture",
                GgufValue::String("llama".to_string()),
            ),
            ("general.name", GgufValue::String("test-model".to_string())),
            ("general.quantization_version", GgufValue::Uint32(2)),
            ("general.file_type", GgufValue::Uint32(10)), // Q4_K
            ("tokenizer.ggml.bos_token_id", GgufValue::Uint32(1)),
        ]);
        // requant: same architecture+name; volatile fields differ (Q6_K).
        let requant = meta_kv(&[
            (
                "general.architecture",
                GgufValue::String("llama".to_string()),
            ),
            ("general.name", GgufValue::String("test-model".to_string())),
            ("general.quantization_version", GgufValue::Uint32(3)),
            ("general.file_type", GgufValue::Uint32(14)), // Q6_K
            ("tokenizer.ggml.bos_token_id", GgufValue::Uint32(1)),
        ]);
        let report =
            classify_preservation(&reference, &requant, "ref.gguf".into(), "req.gguf".into());
        assert!(report.passed, "expected PRESERVED, got {report:#?}");
        assert_eq!(
            report.general_keys_checked, 2,
            "expected 2 non-volatile general keys"
        );
        assert!(report.general_keys_diverged.is_empty());
        assert!(report.tokenizer_keys_diverged.is_empty());
    }

    /// FALSIFY-CRUX-B-19-001 fail case — a non-volatile general.* key changes
    /// → classifier MUST flag VIOLATED.
    #[test]
    fn classifier_flags_general_name_change() {
        let reference = meta_kv(&[("general.name", GgufValue::String("alpha".to_string()))]);
        let requant = meta_kv(&[("general.name", GgufValue::String("beta".to_string()))]);
        let report = classify_preservation(&reference, &requant, "r.gguf".into(), "q.gguf".into());
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
        let merges = GgufValue::ArrayString(vec!["h e".to_string(), "he ll".to_string()]);
        let reference = meta_kv(&[
            ("tokenizer.ggml.tokens", vocab_a.clone()),
            ("tokenizer.ggml.merges", merges.clone()),
        ]);
        let requant = meta_kv(&[
            ("tokenizer.ggml.tokens", vocab_a),
            ("tokenizer.ggml.merges", merges),
        ]);
        let report = classify_preservation(&reference, &requant, "r.gguf".into(), "q.gguf".into());
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
        let report = classify_preservation(&reference, &requant, "r.gguf".into(), "q.gguf".into());
        assert!(!report.passed);
        assert_eq!(report.tokenizer_keys_diverged.len(), 1);
    }

    /// Missing-in-requant case — every required general.* key must be present.
    #[test]
    fn classifier_flags_missing_general_key() {
        let reference = meta_kv(&[
            (
                "general.architecture",
                GgufValue::String("llama".to_string()),
            ),
            ("general.name", GgufValue::String("m".to_string())),
        ]);
        let requant = meta_kv(&[
            (
                "general.architecture",
                GgufValue::String("llama".to_string()),
            ),
            // general.name missing
        ]);
        let report = classify_preservation(&reference, &requant, "r.gguf".into(), "q.gguf".into());
        assert!(!report.passed);
        assert_eq!(report.general_keys_missing_in_requant, vec!["general.name"]);
    }

    /// Volatile fields are NOT compared even when they differ.
    #[test]
    fn volatile_fields_ignored() {
        for k in QUANT_VOLATILE_KEYS {
            let reference = meta_kv(&[
                (k, GgufValue::Uint32(2)),
                (
                    "general.architecture",
                    GgufValue::String("llama".to_string()),
                ),
            ]);
            let requant = meta_kv(&[
                (k, GgufValue::Uint32(99)),
                (
                    "general.architecture",
                    GgufValue::String("llama".to_string()),
                ),
            ]);
            let report =
                classify_preservation(&reference, &requant, "r.gguf".into(), "q.gguf".into());
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
        let report = classify_preservation(&reference, &requant, "r.gguf".into(), "q.gguf".into());
        assert!(report.passed, "non-general/non-tokenizer must be ignored");
    }

    /// Added keys in requant (that weren't in reference) — flagged.
    #[test]
    fn classifier_flags_added_tokenizer_key() {
        let reference = meta_kv(&[("tokenizer.ggml.tokens", GgufValue::ArrayString(vec![]))]);
        let requant = meta_kv(&[
            ("tokenizer.ggml.tokens", GgufValue::ArrayString(vec![])),
            ("tokenizer.ggml.merges", GgufValue::ArrayString(vec![])),
        ]);
        let report = classify_preservation(&reference, &requant, "r.gguf".into(), "q.gguf".into());
        assert!(!report.passed);
        assert_eq!(
            report.tokenizer_keys_added_in_requant,
            vec!["tokenizer.ggml.merges"]
        );
    }

    // ── bounded divergence reporting (dogfood 0.63.0 #2377 findings 1+7) ──
    //
    // Two diverging tokenizer vocabularies produced a 13 MB, 19-line text
    // report (14.9 MB under --json) with a single 4.8 MB line, because every
    // diverged value was stored as the untruncated `Debug` of the whole GGUF
    // array. These assert the SIZE of the emitted report, not its shape.

    /// A vocab-sized array is the realistic worst case: 151936 tokens.
    fn big_vocab(seed: &str, n: usize) -> GgufValue {
        GgufValue::ArrayString((0..n).map(|i| format!("{seed}-token-{i}")).collect())
    }

    #[test]
    fn diverging_vocabularies_do_not_produce_a_multi_megabyte_report() {
        let reference = meta_kv(&[("tokenizer.ggml.tokens", big_vocab("ref", 151_936))]);
        let requant = meta_kv(&[("tokenizer.ggml.tokens", big_vocab("req", 151_936))]);
        let report = classify_preservation(&reference, &requant, "r.gguf".into(), "q.gguf".into());
        assert!(!report.passed, "divergent vocabularies must be VIOLATED");

        let text = render_text(&report);
        assert!(
            text.len() < 4096,
            "divergence report must stay bounded; got {} bytes:\n{}",
            text.len(),
            &text[..text.len().min(400)]
        );
        // Every line must be readable in a terminal and a CI log.
        let longest = text.lines().map(str::len).max().unwrap_or(0);
        assert!(longest < 512, "longest line is {longest} chars");

        let json = serde_json::to_string(&report).expect("serialize");
        assert!(json.len() < 4096, "--json payload is {} bytes", json.len());
    }

    #[test]
    fn summarised_value_names_the_length_and_a_digest_not_the_contents() {
        let big = big_vocab("ref", 151_936);
        let rendered = render_value_bounded(&big);
        assert!(rendered.contains("len=151936"), "got: {rendered}");
        assert!(rendered.contains("sha256="), "got: {rendered}");
        assert!(
            rendered.chars().count() < 400,
            "summary is {} chars",
            rendered.chars().count()
        );
    }

    #[test]
    fn two_different_vocabularies_summarise_to_different_digests() {
        // A digest that collided would make the summary useless.
        let a = render_value_bounded(&big_vocab("ref", 8_000));
        let b = render_value_bounded(&big_vocab("req", 8_000));
        assert_ne!(a, b, "distinct vocabularies must not summarise identically");
    }

    #[test]
    fn short_scalar_divergences_are_still_rendered_verbatim() {
        // The actionable one-line signals must not be summarised away.
        let reference = meta_kv(&[
            (
                "general.architecture",
                GgufValue::String("qwen35".to_string()),
            ),
            ("tokenizer.ggml.eos_token_id", GgufValue::Uint32(248_046)),
        ]);
        let requant = meta_kv(&[
            (
                "general.architecture",
                GgufValue::String("qwen2".to_string()),
            ),
            ("tokenizer.ggml.eos_token_id", GgufValue::Uint32(151_645)),
        ]);
        let text = render_text(&classify_preservation(
            &reference,
            &requant,
            "r.gguf".into(),
            "q.gguf".into(),
        ));
        assert!(
            text.contains(r#"general.architecture :: String("qwen35") → String("qwen2")"#),
            "got:\n{text}"
        );
        assert!(
            text.contains("tokenizer.ggml.eos_token_id :: Uint32(248046) → Uint32(151645)"),
            "got:\n{text}"
        );
    }

    #[test]
    fn array_divergence_reports_the_first_differing_index() {
        let reference = meta_kv(&[(
            "tokenizer.ggml.token_type",
            GgufValue::ArrayInt32(vec![1; 4096]),
        )]);
        let mut changed = vec![1i32; 4096];
        changed[2047] = 4;
        let requant = meta_kv(&[("tokenizer.ggml.token_type", GgufValue::ArrayInt32(changed))]);
        let report = classify_preservation(&reference, &requant, "r.gguf".into(), "q.gguf".into());
        let entry = &report.tokenizer_keys_diverged[0];
        assert_eq!(
            entry.first_difference.as_deref(),
            Some("first differing element: index 2047 (len 4096 vs 4096)")
        );
        assert!(render_text(&report).contains("index 2047"));
    }
}
