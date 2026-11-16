//! Single source of the `qwen4_exp` refusal (paiml/aprender#7169).
//!
//! Shared by the inference converter (`aprender-serve`), the CUDA loader and
//! `apr import` (`aprender-core`), so every entry point refuses with the same text.

/// HF `model_type` of the composite checkpoint.
pub const MODEL_TYPE: &str = "qwen4_exp";
/// HF `model_type` of its text sub-config.
pub const TEXT_MODEL_TYPE: &str = "qwen4_exp_text";
/// HF `architectures[0]` of the checkpoint.
pub const ARCHITECTURE: &str = "Qwen4ExpForConditionalGeneration";

/// Names of the ops this build has no forward for, in refusal order.
pub const UNSUPPORTED_OP_NAMES: [&str; 6] = [
    "sparse-attention indexer",
    "ngram embedding",
    "ple",
    "hc residual",
    "sigmoid output gate",
    "mtp",
];

/// Is this `model_type` / `architectures` pair `qwen4_exp`?
#[must_use]
pub fn is_qwen4_exp(model_type: Option<&str>, architectures: &[String]) -> bool {
    matches!(model_type, Some(MODEL_TYPE | TEXT_MODEL_TYPE))
        || architectures.iter().any(|a| a == ARCHITECTURE)
}

/// The named refusal text for the given op names.
#[must_use]
pub fn refusal_reason(op_names: &[&str]) -> String {
    format!(
        "architecture '{MODEL_TYPE}' is recognised but this build has no forward for its new \
         ops: [{}]. This is a refusal, not a fallback: no tensor data was read (headers may \
         have been parsed) and nothing was generated (paiml/aprender#7169 S2/S3).",
        op_names.join(", ")
    )
}

/// Refusal for a `qwen4_exp` config, `None` for any other arch.
#[must_use]
pub fn refusal_for(model_type: Option<&str>, architectures: &[String]) -> Option<String> {
    is_qwen4_exp(model_type, architectures).then(|| refusal_reason(&UNSUPPORTED_OP_NAMES))
}

/// Refusal for a raw `config.json` value (top level or `text_config`), `None` otherwise.
#[must_use]
pub fn refusal_for_json(raw: &serde_json::Value) -> Option<String> {
    let mt = |v: &serde_json::Value| {
        v.get("model_type")
            .and_then(|m| m.as_str())
            .map(String::from)
    };
    let archs: Vec<String> = raw
        .get("architectures")
        .and_then(|a| a.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|s| s.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();
    refusal_for(mt(raw).as_deref(), &archs).or_else(|| {
        raw.get("text_config")
            .and_then(|t| refusal_for(mt(t).as_deref(), &[]))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_refusal_matches_by_type_text_config_and_arch() {
        for j in [
            r#"{"model_type":"qwen4_exp"}"#,
            r#"{"model_type":"qwen4_exp_text"}"#,
            r#"{"architectures":["Qwen4ExpForConditionalGeneration"]}"#,
            r#"{"model_type":"x","text_config":{"model_type":"qwen4_exp_text"}}"#,
        ] {
            let v: serde_json::Value = serde_json::from_str(j).expect("json");
            let r = refusal_for_json(&v).unwrap_or_else(|| panic!("no refusal: {j}"));
            assert!(r.contains("refusal, not a fallback") && r.contains("mtp"));
        }
        let v: serde_json::Value = serde_json::from_str(r#"{"model_type":"qwen3"}"#).expect("json");
        assert!(refusal_for_json(&v).is_none());
    }
}
