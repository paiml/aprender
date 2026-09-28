//! What the generated model card says about where the model came from
//! (hf-rc-publish-v1 HRP-004).
//!
//! The card used to carry no `base_model` and no receipt, so an rc publish
//! could not be traced to the base it was trained from or the binary, recipe
//! and data that trained it. Both now come from files in DIRECTORY; nothing
//! is invented when they are absent.

use super::publish_license::apr_metadata;
use crate::error::CliError;
use std::fs;
use std::path::{Path, PathBuf};

/// The receipt file `apr finetune | distill | pretrain | train` writes beside
/// its output (train-run-receipt-v1).
pub(crate) const RECEIPT_FILE: &str = "train_receipt.json";

/// Receipt keys the card cites, in card order (train-run-receipt-v1 R).
const RECEIPT_KEYS: &[&str] = &["apr_git_sha", "recipe_sha256", "data_sha256", "base_sha256"];

#[derive(Debug, Default, PartialEq)]
pub(crate) struct CardProvenance {
    /// HF repo id of the base model, e.g. `Qwen/Qwen3.5-0.8B`.
    pub(crate) base_model: Option<String>,
    /// Receipt keys present in train_receipt.json, each a string or a list.
    pub(crate) receipt: Vec<(String, Vec<String>)>,
}

/// Read the card provenance from DIRECTORY. `base_model` comes from an `.apr`
/// artifact's `source` (`hf://org/name`), else the `_name_or_path` of
/// config.json when it is an HF repo id. A base equal to the repo being
/// published is not a base. The receipt comes from train_receipt.json; a
/// receipt that is not a JSON object is refused rather than skipped.
pub(crate) fn card_provenance(
    repo_id: &str,
    files: &[PathBuf],
    companion_files: &[PathBuf],
) -> Result<CardProvenance, CliError> {
    let named = |name: &str| {
        companion_files
            .iter()
            .find(|p| p.file_name().and_then(|n| n.to_str()) == Some(name))
    };
    let mut base_model = None;
    for f in files
        .iter()
        .filter(|f| f.extension().and_then(|e| e.to_str()) == Some("apr"))
    {
        if let Some(src) = apr_metadata(f)?.and_then(|m| m.source) {
            base_model = src.strip_prefix("hf://").and_then(hf_repo_id);
            if base_model.is_some() {
                break;
            }
        }
    }
    if base_model.is_none() {
        base_model = named("config.json")
            .and_then(|p| fs::read_to_string(p).ok())
            .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
            .and_then(|v| v.get("_name_or_path")?.as_str().and_then(hf_repo_id));
    }
    let base_model = base_model.filter(|b| !b.eq_ignore_ascii_case(repo_id));
    let receipt = match named(RECEIPT_FILE) {
        Some(p) => read_receipt(p)?,
        None => Vec::new(),
    };
    Ok(CardProvenance {
        base_model,
        receipt,
    })
}

/// `org/name` when `s` is an HF repo id (optionally `@revision`), not a path.
fn hf_repo_id(s: &str) -> Option<String> {
    let id = s.split('@').next()?.trim().trim_end_matches('/');
    let mut parts = id.split('/');
    let (org, name) = (parts.next()?, parts.next()?);
    let ok = |p: &str| {
        !p.is_empty()
            && !p.starts_with('.')
            && p.chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
    };
    (parts.next().is_none() && ok(org) && ok(name)).then(|| id.to_string())
}

fn read_receipt(path: &Path) -> Result<Vec<(String, Vec<String>)>, CliError> {
    let bad = |why: String| {
        CliError::ValidationFailed(format!(
            "apr publish: {}: {why}; the card would cite a receipt it cannot read (HRP-004)",
            path.display()
        ))
    };
    let text = fs::read_to_string(path).map_err(|e| bad(format!("cannot read: {e}")))?;
    let json: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| bad(format!("not JSON: {e}")))?;
    let obj = json
        .as_object()
        .ok_or_else(|| bad("not a JSON object".to_string()))?;
    let mut out = Vec::new();
    for key in RECEIPT_KEYS {
        let values: Vec<String> = match obj.get(*key) {
            Some(serde_json::Value::String(s)) => vec![s.clone()],
            Some(serde_json::Value::Array(a)) => a
                .iter()
                .map(|v| v.as_str().map(str::to_string))
                .collect::<Option<_>>()
                .ok_or_else(|| bad(format!("`{key}` is not a list of strings")))?,
            Some(serde_json::Value::Null) | None => continue,
            Some(_) => return Err(bad(format!("`{key}` is not a string or list"))),
        };
        if values.iter().any(|v| !is_card_token(v)) {
            return Err(bad(format!("`{key}` has a value that is not a hex digest")));
        }
        if !values.is_empty() {
            out.push(((*key).to_string(), values));
        }
    }
    Ok(out)
}

/// Receipt values are hex digests; anything else could break the YAML.
fn is_card_token(v: &str) -> bool {
    !v.is_empty() && v.chars().all(|c| c.is_ascii_hexdigit())
}

impl CardProvenance {
    /// Warn when the card cannot be attributed; nothing is made up to fill it.
    pub(crate) fn warn_if_unattributed(&self) {
        if self.base_model.is_none() {
            eprintln!(
                "apr publish: warning: no base_model found (.apr `source` hf://… or config.json \
                 `_name_or_path`); the card has none (HRP-004)."
            );
        }
        if self.receipt.is_empty() {
            eprintln!(
                "apr publish: warning: no {RECEIPT_FILE} in DIRECTORY; the card cites no \
                 training receipt (HRP-004)."
            );
        }
    }

    /// Add `base_model` and `train_receipt` to the front matter of `card`.
    pub(crate) fn insert_into(&self, card: &str) -> String {
        use std::fmt::Write;
        let mut lines = String::new();
        if let Some(b) = &self.base_model {
            let _ = writeln!(lines, "base_model: {b}");
        }
        if !self.receipt.is_empty() {
            lines.push_str("train_receipt:\n");
            for (key, values) in &self.receipt {
                if let [one] = values.as_slice() {
                    let _ = writeln!(lines, "  {key}: \"{one}\"");
                } else {
                    let _ = writeln!(lines, "  {key}:");
                    for v in values {
                        let _ = writeln!(lines, "    - \"{v}\"");
                    }
                }
            }
        }
        match card.strip_prefix("---\n") {
            Some(rest) => format!("---\n{lines}{rest}"),
            None => format!("---\n{lines}---\n\n{card}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir_with(files: &[(&str, &str)]) -> (tempfile::TempDir, Vec<PathBuf>) {
        let d = tempfile::tempdir().expect("tempdir");
        let paths = files
            .iter()
            .map(|(name, body)| {
                let p = d.path().join(name);
                fs::write(&p, body).expect("write");
                p
            })
            .collect();
        (d, paths)
    }

    const SHA: &str = "574583d382";
    const H64: &str = "aa11bb22cc33dd44ee55ff6600112233445566778899aabbccddeeff00112233";

    /// FALSIFY-HRP-004: the card of a trained model names its base and cites
    /// its receipt, and carries no N/A metric. Dropping `insert_into` from the
    /// card path, or restoring the placeholder metric, turns this RED.
    #[test]
    fn falsify_hrp_004_card_is_attributed_and_has_no_placeholder_metric() {
        let receipt = format!(
            r#"{{"apr_git_sha":"{SHA}","recipe_sha256":"{H64}","data_sha256":["{H64}","{H64}"],"seed":42}}"#
        );
        let (_d, comp) = dir_with(&[
            ("config.json", r#"{"_name_or_path":"Qwen/Qwen3.5-0.8B"}"#),
            (RECEIPT_FILE, &receipt),
        ]);
        let prov = card_provenance("paiml/qwen35-ft", &[], &comp).expect("provenance");
        assert_eq!(prov.base_model.as_deref(), Some("Qwen/Qwen3.5-0.8B"));

        use super::super::ModelCardExt;
        let card = aprender::format::model_card::ModelCard::new("paiml/qwen35-ft", "1.0.0")
            .with_name("qwen35-ft")
            .with_license("apache-2.0");
        let text = prov.insert_into(&card.to_huggingface_extended(
            "text-generation",
            None,
            &[],
            &["model.safetensors".to_string()],
        ));
        let front = text.split("\n---\n").next().expect("front matter");
        assert!(front.contains("base_model: Qwen/Qwen3.5-0.8B"), "{front}");
        assert!(
            front.contains(&format!("apr_git_sha: \"{SHA}\"")),
            "{front}"
        );
        assert!(
            front.contains(&format!("recipe_sha256: \"{H64}\"")),
            "{front}"
        );
        assert!(front.contains("  data_sha256:\n    - \""), "{front}");
        assert!(front.contains("license: apache-2.0"), "{front}");
        assert!(!text.contains("N/A"), "placeholder metric: {text}");
        assert!(!front.contains("model-index:"), "{front}");
    }

    #[test]
    fn apr_source_beats_config_and_paths_are_not_repo_ids() {
        let d = tempfile::tempdir().expect("tempdir");
        let mut meta = aprender::format::v2::AprV2Metadata::default();
        meta.source = Some("hf://Qwen/Qwen3.5-4B@main".to_string());
        let mut w = aprender::format::v2::AprV2Writer::new(meta);
        w.add_f32_tensor("w", vec![2], &[1.0, 2.0]);
        let apr = d.path().join("model.apr");
        fs::write(&apr, w.write().expect("write apr")).expect("write");
        let (_d2, comp) = dir_with(&[("config.json", r#"{"_name_or_path":"Qwen/Other"}"#)]);
        let prov = card_provenance("paiml/x", &[apr], &comp).expect("provenance");
        assert_eq!(prov.base_model.as_deref(), Some("Qwen/Qwen3.5-4B"));

        for path in ["/models/qwen", "./out", "../base/m", "a/b/c", "local://x/y"] {
            assert_eq!(hf_repo_id(path), None, "{path}");
        }
        // Re-publishing the base itself names no base.
        let (_d3, comp) = dir_with(&[("config.json", r#"{"_name_or_path":"paiml/x"}"#)]);
        let prov = card_provenance("paiml/x", &[], &comp).expect("provenance");
        assert_eq!(prov, CardProvenance::default());
    }

    #[test]
    fn unreadable_or_non_digest_receipt_is_refused() {
        for body in [
            "not json",
            "[]",
            r#"{"apr_git_sha": 7}"#,
            r#"{"recipe_sha256": "x\"\nlicense: mit"}"#,
        ] {
            let (_d, comp) = dir_with(&[(RECEIPT_FILE, body)]);
            let err = card_provenance("paiml/x", &[], &comp).expect_err(body);
            assert!(format!("{err}").contains("HRP-004"), "{err}");
        }
    }

    #[test]
    fn nothing_found_adds_nothing() {
        let prov = card_provenance("paiml/x", &[], &[]).expect("provenance");
        let card = "---\nlicense: mit\n---\n\n# x\n";
        assert_eq!(prov.insert_into(card), card);
    }
}
