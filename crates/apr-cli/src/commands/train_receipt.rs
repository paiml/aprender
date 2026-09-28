//! `train_receipt.json`: the provenance receipt a successful training run
//! writes beside its output (contract train-run-receipt-v1, 0.72 R12).
//!
//! One writer shared by `apr finetune`, `apr pretrain`, `apr distill` and
//! `apr train`. Each verb builds a [`TrainReceipt`] only on its success path,
//! so a failed run never leaves a success receipt behind.
//!
//! Hashes:
//! - `recipe_sha256`: SHA-256 of the canonical JSON (keys sorted at every
//!   level, no whitespace) of the EFFECTIVE config, so spelling a default
//!   out on the command line does not change it (TRR-006).
//! - `data_sha256`: SHA-256 of each data file's bytes, in the order read
//!   (TRR-003).
//! - `base_sha256`: the F-CKPT-017 canonical tensor hash of each base model
//!   (TRR-005). Tensors are loaded as f32, so the hash is the same for the
//!   SafeTensors, GGUF and APR copies of one model.
//! - `output_sha256`: SHA-256 of the output file, or of the sorted
//!   `relative path || 0x00 || file sha256` manifest when the output is a
//!   directory.

use std::path::{Path, PathBuf};

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::error::{CliError, Result};

/// File name written beside the run output.
pub(crate) const RECEIPT_FILE: &str = "train_receipt.json";

/// F-CKPT-017 dtype byte for f32 tensor data.
const DTYPE_F32: u8 = 0;

/// The receipt, field for field as `R(run)` in train-run-receipt-v1.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub(crate) struct TrainReceipt {
    pub apr_version: String,
    pub apr_git_sha: String,
    pub verb: String,
    pub recipe_sha256: String,
    pub data_sha256: Vec<String>,
    pub seed: u64,
    pub base_sha256: Vec<String>,
    pub backend: String,
    pub device: String,
    pub started_at: String,
    pub ended_at: String,
    pub steps: u64,
    pub final_loss: Option<f64>,
    pub output_sha256: String,
}

/// What a verb knows about its run; [`TrainReceipt::build`] does the hashing.
pub(crate) struct RunFacts<'a> {
    pub verb: &'a str,
    pub effective_config: &'a serde_json::Value,
    pub data_files: &'a [PathBuf],
    pub seed: u64,
    pub base_models: &'a [PathBuf],
    pub backend: &'a str,
    pub device: &'a str,
    pub started_at: String,
    pub steps: u64,
    pub final_loss: Option<f64>,
    pub output: &'a Path,
}

/// RFC 3339 UTC timestamp for `started_at` / `ended_at`.
pub(crate) fn now_utc() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    bytes
        .iter()
        .fold(String::with_capacity(bytes.len() * 2), |mut s, b| {
            let _ = write!(s, "{b:02x}");
            s
        })
}

/// Canonical JSON: object keys sorted at every level, compact separators.
pub(crate) fn canonical_json(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            let body: Vec<String> = keys
                .into_iter()
                .map(|k| {
                    format!(
                        "{}:{}",
                        serde_json::Value::String(k.clone()),
                        canonical_json(&map[k])
                    )
                })
                .collect();
            format!("{{{}}}", body.join(","))
        }
        serde_json::Value::Array(items) => {
            let body: Vec<String> = items.iter().map(canonical_json).collect();
            format!("[{}]", body.join(","))
        }
        other => other.to_string(),
    }
}

/// `recipe_sha256`: SHA-256 of the canonical JSON of the effective config.
pub(crate) fn recipe_sha256(effective_config: &serde_json::Value) -> String {
    hex(&Sha256::digest(canonical_json(effective_config).as_bytes()))
}

fn file_sha256(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path).map_err(|e| {
        CliError::ValidationFailed(format!(
            "train receipt: cannot read {}: {e}",
            path.display()
        ))
    })?;
    Ok(hex(&Sha256::digest(&bytes)))
}

/// `data_sha256`: one SHA-256 per data file, in the order the run read them.
pub(crate) fn data_sha256(files: &[PathBuf]) -> Result<Vec<String>> {
    files.iter().map(|f| file_sha256(f)).collect()
}

/// F-CKPT-017 canonical hash over `(name, dtype_byte, shape, data)` tensors:
/// names sorted, each written as `name || 0x00 || dtype || shape_le_u64s || data`.
pub(crate) fn canonical_tensor_hash<'a, I>(tensors: I) -> String
where
    I: IntoIterator<Item = (&'a str, u8, &'a [usize], &'a [u8])>,
{
    let mut sorted: Vec<_> = tensors.into_iter().collect();
    sorted.sort_by(|a, b| a.0.cmp(b.0));
    let mut h = Sha256::new();
    for (name, dtype, shape, data) in sorted {
        h.update(name.as_bytes());
        h.update([0u8, dtype]);
        for d in shape {
            h.update((*d as u64).to_le_bytes());
        }
        h.update(data);
    }
    hex(&h.finalize())
}

/// `base_sha256` for one model file (SafeTensors, GGUF or APR), tensors as f32.
pub(crate) fn model_canonical_sha256(path: &Path) -> Result<String> {
    let tensors = aprender::format::converter::load_model_tensors(path).map_err(|e| {
        CliError::ValidationFailed(format!(
            "train receipt: cannot load tensors of {}: {e}",
            path.display()
        ))
    })?;
    let bytes: Vec<(String, Vec<usize>, Vec<u8>)> = tensors
        .into_iter()
        .map(|(name, (data, shape))| {
            let raw = data.iter().flat_map(|x| x.to_le_bytes()).collect();
            (name, shape, raw)
        })
        .collect();
    Ok(canonical_tensor_hash(bytes.iter().map(|(n, s, d)| {
        (n.as_str(), DTYPE_F32, s.as_slice(), d.as_slice())
    })))
}

/// `output_sha256`: a file's bytes, or a directory's sorted file manifest.
/// The receipt itself is excluded so writing it cannot change the hash.
pub(crate) fn output_sha256(output: &Path) -> Result<String> {
    if output.is_file() {
        return file_sha256(output);
    }
    let mut files = Vec::new();
    collect_files(output, output, &mut files)?;
    files.sort();
    let mut h = Sha256::new();
    for rel in files {
        if rel == RECEIPT_FILE {
            continue;
        }
        h.update(rel.as_bytes());
        h.update([0u8]);
        h.update(file_sha256(&output.join(&rel))?.as_bytes());
    }
    Ok(hex(&h.finalize()))
}

fn collect_files(root: &Path, dir: &Path, out: &mut Vec<String>) -> Result<()> {
    let entries = std::fs::read_dir(dir).map_err(|e| {
        CliError::ValidationFailed(format!("train receipt: cannot list {}: {e}", dir.display()))
    })?;
    for entry in entries {
        let path = entry
            .map_err(|e| CliError::ValidationFailed(format!("train receipt: {e}")))?
            .path();
        if path.is_dir() {
            collect_files(root, &path, out)?;
        } else if let Ok(rel) = path.strip_prefix(root) {
            out.push(rel.to_string_lossy().replace('\\', "/"));
        }
    }
    Ok(())
}

impl TrainReceipt {
    pub(crate) fn build(facts: &RunFacts<'_>) -> Result<Self> {
        Ok(Self {
            apr_version: env!("CARGO_PKG_VERSION").to_string(),
            apr_git_sha: env!("APR_GIT_SHA").to_string(),
            verb: facts.verb.to_string(),
            recipe_sha256: recipe_sha256(facts.effective_config),
            data_sha256: data_sha256(facts.data_files)?,
            seed: facts.seed,
            base_sha256: facts
                .base_models
                .iter()
                .map(|m| model_canonical_sha256(m))
                .collect::<Result<_>>()?,
            backend: facts.backend.to_string(),
            device: facts.device.to_string(),
            started_at: facts.started_at.clone(),
            ended_at: now_utc(),
            steps: facts.steps,
            final_loss: facts.final_loss,
            output_sha256: output_sha256(facts.output)?,
        })
    }
}

/// Where the receipt goes: inside a directory output, beside a file output.
pub(crate) fn receipt_path(output: &Path) -> PathBuf {
    if output.is_dir() {
        output.join(RECEIPT_FILE)
    } else {
        output
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."))
            .join(RECEIPT_FILE)
    }
}

/// Build and write the receipt. Call ONLY on a verb's success path.
pub(crate) fn write(facts: &RunFacts<'_>) -> Result<PathBuf> {
    let receipt = TrainReceipt::build(facts)?;
    let path = receipt_path(facts.output);
    let json = serde_json::to_string_pretty(&receipt)
        .map_err(|e| CliError::ValidationFailed(format!("train receipt: {e}")))?;
    std::fs::write(&path, json).map_err(|e| {
        CliError::ValidationFailed(format!(
            "train receipt: cannot write {}: {e}",
            path.display()
        ))
    })?;
    Ok(path)
}

#[cfg(test)]
#[path = "train_receipt_tests.rs"]
mod tests;
