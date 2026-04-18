//! `apr validate-manifest` — verify publish manifests against
//! `contracts/publish-manifest-v1.yaml`.
//!
//! Discharges AC-EX-004 of SPEC-SHIP-TWO-001 §12.3 — closes the "tool gap"
//! noted in `contracts/publish-manifests/paiml-qwen2.5-coder-7b-apache-q4k-v1.yaml`
//! where FALSIFY-PM-001..006 were previously validated via an ad-hoc pyyaml
//! helper.
//!
//! Gates (schema per `contracts/publish-manifest-v1.yaml` §schema):
//!   FALSIFY-PM-001       schema — 12 top + 7 provenance required fields present & non-null
//!   FALSIFY-PM-002       sha256 — declared matches computed over `--artifact` (if provided)
//!   FALSIFY-PM-003       URL liveness — HTTP HEAD + content-length (only with `--live`)
//!   FALSIFY-PM-002-live  streaming remote sha256 (only with `--live`)
//!   FALSIFY-PM-004       SPDX — license / provenance.parent_license / data_license valid
//!   FALSIFY-PM-005       recipe — provenance.recipe_sha256 matches sha256 of `recipe` file
//!   FALSIFY-PM-006       parent-chain — provenance.parent terminates at an HF model id
//!   FALSIFY-PM-007       safetensors header dtype Poka-Yoke (only for format=safetensors
//!                        + `--artifact`) — SHIP-TWO-001 §12.7.2 ship-blocker guard

use crate::error::CliError;
use colored::Colorize;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

/// SPDX identifiers accepted without question. Not exhaustive; extend as
/// new licenses appear in provenance chains.
const SPDX_ALLOWLIST: &[&str] = &[
    "Apache-2.0",
    "MIT",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "MPL-2.0",
    "LGPL-2.1",
    "LGPL-2.1-only",
    "LGPL-3.0",
    "LGPL-3.0-only",
    "GPL-2.0",
    "GPL-2.0-only",
    "GPL-3.0",
    "GPL-3.0-only",
    "CC-BY-4.0",
    "CC-BY-SA-4.0",
    "CC-BY-NC-4.0",
    "CC0-1.0",
    "Unlicense",
    "ISC",
    "Apache-2.0 WITH LLVM-exception",
    "llama2",
    "llama3",
    "llama3.1",
    "gemma",
    "custom",
];

// Field names authoritative per contracts/publish-manifest-v1.yaml §schema.
const REQUIRED_TOP: &[&str] = &[
    "model_id",
    "version",
    "architecture",
    "format",
    "quantization",
    "artifact_url",
    "sha256",
    "size_bytes",
    "license",
    "provenance",
    "published_at",
    "published_by",
];

const REQUIRED_PROVENANCE: &[&str] = &[
    "pipeline",
    "parent",
    "parent_license",
    "data_source",
    "data_license",
    "recipe",
    "recipe_sha256",
];

#[derive(Serialize)]
struct FalsifyResult {
    id: &'static str,
    verdict: &'static str,
    detail: String,
}

#[derive(Serialize)]
struct ManifestReport {
    manifest_path: String,
    artifact_path: Option<String>,
    falsification_results: Vec<FalsifyResult>,
    overall: &'static str,
}

pub(crate) fn run(
    manifest_path: &Path,
    artifact: Option<&Path>,
    json: bool,
    live_check: bool,
) -> Result<(), CliError> {
    let contents = fs::read_to_string(manifest_path)
        .map_err(|e| CliError::ValidationFailed(format!("read manifest: {e}")))?;
    let yaml: serde_yaml::Value = serde_yaml::from_str(&contents)
        .map_err(|e| CliError::ValidationFailed(format!("parse yaml: {e}")))?;

    let top = yaml
        .as_mapping()
        .ok_or_else(|| CliError::ValidationFailed("manifest is not a YAML mapping".into()))?;

    let mut results: Vec<FalsifyResult> = Vec::new();

    let prov_value = get_str_key(top, "provenance");
    let prov = prov_value.and_then(serde_yaml::Value::as_mapping);

    results.push(check_schema(top, prov));
    results.push(check_sha256(top, artifact));
    // FALSIFY-PM-003 (HEAD) + FALSIFY-PM-002-live (streaming sha256) discharged
    // via network when --live is passed, otherwise PM-003 DEFERRED.
    // F-PUBLISH-EXTRA-001::dogfood_ex05 — replaces `uv run python3` streaming
    // check in ex-05-verify-manifest.sh with an apr-native implementation.
    if live_check {
        results.push(check_url_head_live(top));
        results.push(check_sha256_live(top));
    } else {
        results.push(defer_url_liveness());
    }
    results.push(check_spdx(top, prov));
    results.push(check_recipe(prov, manifest_path));
    results.push(check_parent_chain(prov));
    // FALSIFY-PM-007 — safetensors header dtype Poka-Yoke (SHIP-TWO-001 §12.7.2).
    // Applies only when format == "safetensors" AND --artifact is provided.
    results.push(check_safetensors_header_dtype(top, artifact));

    let any_fail = results.iter().any(|r| r.verdict == "FAIL");
    let overall = if any_fail { "FAIL" } else { "PASS" };

    let report = ManifestReport {
        manifest_path: manifest_path.display().to_string(),
        artifact_path: artifact.map(|p| p.display().to_string()),
        falsification_results: results,
        overall,
    };

    if json {
        let s = serde_json::to_string_pretty(&report)
            .map_err(|e| CliError::ValidationFailed(format!("json: {e}")))?;
        println!("{s}");
    } else {
        println!("apr validate-manifest {}", manifest_path.display());
        for r in &report.falsification_results {
            let badge = match r.verdict {
                "PASS" => "PASS".green(),
                "FAIL" => "FAIL".red(),
                _ => "DEFERRED".yellow(),
            };
            println!("  [{}] {}: {}", badge, r.id, r.detail);
        }
        let overall_colored = if overall == "PASS" {
            "PASS".green()
        } else {
            "FAIL".red()
        };
        println!("  overall: {overall_colored}");
    }

    if any_fail {
        return Err(CliError::ValidationFailed(
            "manifest validation FAILED".into(),
        ));
    }
    Ok(())
}

fn check_schema(top: &serde_yaml::Mapping, prov: Option<&serde_yaml::Mapping>) -> FalsifyResult {
    let mut missing_top: Vec<&str> = Vec::new();
    for k in REQUIRED_TOP {
        match get_str_key(top, k) {
            None | Some(serde_yaml::Value::Null) => missing_top.push(k),
            _ => {}
        }
    }
    let missing_prov: Vec<&str> = match prov {
        Some(pm) => REQUIRED_PROVENANCE
            .iter()
            .copied()
            .filter(|k| match get_str_key(pm, k) {
                None | Some(serde_yaml::Value::Null) => true,
                _ => false,
            })
            .collect(),
        None => REQUIRED_PROVENANCE.to_vec(),
    };
    if missing_top.is_empty() && missing_prov.is_empty() {
        FalsifyResult {
            id: "FALSIFY-PM-001",
            verdict: "PASS",
            detail: format!(
                "all {} top + {} provenance required fields present",
                REQUIRED_TOP.len(),
                REQUIRED_PROVENANCE.len()
            ),
        }
    } else {
        FalsifyResult {
            id: "FALSIFY-PM-001",
            verdict: "FAIL",
            detail: format!("missing top={missing_top:?} provenance={missing_prov:?}"),
        }
    }
}

fn check_sha256(top: &serde_yaml::Mapping, artifact: Option<&Path>) -> FalsifyResult {
    let declared = get_str(top, "sha256").unwrap_or_default();
    match artifact {
        None => FalsifyResult {
            id: "FALSIFY-PM-002",
            verdict: "DEFERRED",
            detail: "no --artifact provided for local sha256 check".into(),
        },
        Some(p) => match compute_sha256(p) {
            Ok(sha) if sha == declared => FalsifyResult {
                id: "FALSIFY-PM-002",
                verdict: "PASS",
                detail: format!("sha256 match: {sha}"),
            },
            Ok(sha) => FalsifyResult {
                id: "FALSIFY-PM-002",
                verdict: "FAIL",
                detail: format!("declared={declared} computed={sha}"),
            },
            Err(e) => FalsifyResult {
                id: "FALSIFY-PM-002",
                verdict: "FAIL",
                detail: format!("read artifact {}: {e}", p.display()),
            },
        },
    }
}

fn defer_url_liveness() -> FalsifyResult {
    FalsifyResult {
        id: "FALSIFY-PM-003",
        verdict: "DEFERRED",
        detail: "URL HEAD check requires network; re-run with --live".into(),
    }
}

/// FALSIFY-PM-003 via HEAD. Requires HTTP 200 and content-length == declared
/// `size_bytes`. Implements F-PUBLISH-EXTRA-001::dogfood_ex05 — no Python.
fn check_url_head_live(top: &serde_yaml::Mapping) -> FalsifyResult {
    let Some(url) = get_str(top, "artifact_url") else {
        return FalsifyResult {
            id: "FALSIFY-PM-003",
            verdict: "FAIL",
            detail: "artifact_url missing".into(),
        };
    };
    let declared_size: Option<u64> =
        get_str_key(top, "size_bytes").and_then(serde_yaml::Value::as_u64);

    let resp = match ureq::head(&url).call() {
        Ok(r) => r,
        Err(ureq::Error::Status(code, _)) => {
            return FalsifyResult {
                id: "FALSIFY-PM-003",
                verdict: "FAIL",
                detail: format!("HEAD {url} → HTTP {code}"),
            };
        }
        Err(e) => {
            return FalsifyResult {
                id: "FALSIFY-PM-003",
                verdict: "FAIL",
                detail: format!("HEAD {url}: {e}"),
            };
        }
    };
    let status = resp.status();
    if status != 200 {
        return FalsifyResult {
            id: "FALSIFY-PM-003",
            verdict: "FAIL",
            detail: format!("HEAD {url} returned status {status}"),
        };
    }
    let got_cl = resp
        .header("content-length")
        .and_then(|s| s.parse::<u64>().ok());
    match (declared_size, got_cl) {
        (Some(exp), Some(got)) if got == exp => FalsifyResult {
            id: "FALSIFY-PM-003",
            verdict: "PASS",
            detail: format!("HEAD 200, content-length {got} == declared {exp}"),
        },
        (Some(exp), Some(got)) => FalsifyResult {
            id: "FALSIFY-PM-003",
            verdict: "FAIL",
            detail: format!("content-length {got} != declared {exp}"),
        },
        (Some(_), None) => FalsifyResult {
            id: "FALSIFY-PM-003",
            verdict: "FAIL",
            detail: "content-length header missing from HEAD response".into(),
        },
        (None, _) => FalsifyResult {
            id: "FALSIFY-PM-003",
            verdict: "FAIL",
            detail: "manifest missing size_bytes; cannot verify content-length".into(),
        },
    }
}

/// FALSIFY-PM-002-live: streaming GET and sha256 verification against the
/// declared `sha256`. Equivalent to the former `uv run python3` block.
/// Performs a full download — can be expensive for large artifacts.
fn check_sha256_live(top: &serde_yaml::Mapping) -> FalsifyResult {
    let Some(url) = get_str(top, "artifact_url") else {
        return FalsifyResult {
            id: "FALSIFY-PM-002-live",
            verdict: "FAIL",
            detail: "artifact_url missing".into(),
        };
    };
    let declared_sha = get_str(top, "sha256").unwrap_or_default();
    if declared_sha.is_empty() {
        return FalsifyResult {
            id: "FALSIFY-PM-002-live",
            verdict: "FAIL",
            detail: "manifest missing sha256".into(),
        };
    }

    let resp = match ureq::get(&url).call() {
        Ok(r) => r,
        Err(ureq::Error::Status(code, _)) => {
            return FalsifyResult {
                id: "FALSIFY-PM-002-live",
                verdict: "FAIL",
                detail: format!("GET {url} → HTTP {code}"),
            };
        }
        Err(e) => {
            return FalsifyResult {
                id: "FALSIFY-PM-002-live",
                verdict: "FAIL",
                detail: format!("GET {url}: {e}"),
            };
        }
    };
    let mut reader = resp.into_reader();
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 1 << 20];
    let mut total: u64 = 0;
    loop {
        match reader.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => {
                hasher.update(&buf[..n]);
                total += n as u64;
            }
            Err(e) => {
                return FalsifyResult {
                    id: "FALSIFY-PM-002-live",
                    verdict: "FAIL",
                    detail: format!("stream GET {url}: {e}"),
                };
            }
        }
    }
    let computed = format!("{:x}", hasher.finalize());
    if computed == declared_sha {
        FalsifyResult {
            id: "FALSIFY-PM-002-live",
            verdict: "PASS",
            detail: format!("sha256 {computed} over {total} bytes"),
        }
    } else {
        FalsifyResult {
            id: "FALSIFY-PM-002-live",
            verdict: "FAIL",
            detail: format!("declared={declared_sha} computed={computed} bytes_read={total}"),
        }
    }
}

fn check_spdx(top: &serde_yaml::Mapping, prov: Option<&serde_yaml::Mapping>) -> FalsifyResult {
    let license = get_str(top, "license").unwrap_or_default();
    let parent_license = prov
        .and_then(|pm| get_str(pm, "parent_license"))
        .unwrap_or_default();
    let data_license = prov
        .and_then(|pm| get_str(pm, "data_license"))
        .unwrap_or_default();

    let checks: [(&str, &str); 3] = [
        ("license", &license),
        ("provenance.parent_license", &parent_license),
        ("provenance.data_license", &data_license),
    ];
    let mut invalid: Vec<String> = Vec::new();
    let mut valid = 0usize;
    for (field, val) in checks {
        if val.is_empty() {
            continue;
        }
        if SPDX_ALLOWLIST.iter().any(|a| a.eq_ignore_ascii_case(val)) {
            valid += 1;
        } else {
            invalid.push(format!("{field}={val}"));
        }
    }
    if invalid.is_empty() {
        FalsifyResult {
            id: "FALSIFY-PM-004",
            verdict: "PASS",
            detail: format!("{valid} SPDX identifier(s) valid"),
        }
    } else {
        FalsifyResult {
            id: "FALSIFY-PM-004",
            verdict: "FAIL",
            detail: format!("invalid SPDX: {invalid:?}"),
        }
    }
}

fn check_recipe(prov: Option<&serde_yaml::Mapping>, manifest_path: &Path) -> FalsifyResult {
    let Some(pm) = prov else {
        return FalsifyResult {
            id: "FALSIFY-PM-005",
            verdict: "FAIL",
            detail: "provenance block missing".into(),
        };
    };
    let recipe_path_str = get_str(pm, "recipe").unwrap_or_default();
    let declared = get_str(pm, "recipe_sha256").unwrap_or_default();
    if recipe_path_str.is_empty() || declared.is_empty() {
        return FalsifyResult {
            id: "FALSIFY-PM-005",
            verdict: "FAIL",
            detail: "provenance.recipe or provenance.recipe_sha256 missing".into(),
        };
    }
    let rp = resolve_recipe(&recipe_path_str, manifest_path);
    match compute_sha256(&rp) {
        Ok(computed) if computed == declared => FalsifyResult {
            id: "FALSIFY-PM-005",
            verdict: "PASS",
            detail: format!("recipe_sha256 match ({}): {computed}", rp.display()),
        },
        Ok(computed) => FalsifyResult {
            id: "FALSIFY-PM-005",
            verdict: "FAIL",
            detail: format!("{} declared={declared} computed={computed}", rp.display()),
        },
        Err(e) => FalsifyResult {
            id: "FALSIFY-PM-005",
            verdict: "FAIL",
            detail: format!("read recipe {}: {e}", rp.display()),
        },
    }
}

fn check_parent_chain(prov: Option<&serde_yaml::Mapping>) -> FalsifyResult {
    let Some(pm) = prov else {
        return FalsifyResult {
            id: "FALSIFY-PM-006",
            verdict: "FAIL",
            detail: "provenance block missing".into(),
        };
    };
    let parent = get_str(pm, "parent").unwrap_or_default();
    if parent.is_empty() {
        return FalsifyResult {
            id: "FALSIFY-PM-006",
            verdict: "FAIL",
            detail: "provenance.parent missing".into(),
        };
    }
    // Accept either an HF-style "org/name" id or the literal "base" to indicate
    // a foundation model trained from scratch.
    if parent == "base" || parent.contains('/') {
        FalsifyResult {
            id: "FALSIFY-PM-006",
            verdict: "PASS",
            detail: format!("parent chain terminates at {parent}"),
        }
    } else {
        FalsifyResult {
            id: "FALSIFY-PM-006",
            verdict: "FAIL",
            detail: format!(
                "provenance.parent={parent} — expected HF id 'org/name' or literal 'base'"
            ),
        }
    }
}

/// FALSIFY-PM-007 — safetensors header dtype Poka-Yoke.
///
/// When the manifest declares `format: safetensors` and `--artifact` is
/// provided, parse the safetensors header (first 8 bytes LE u64 = header
/// length, then UTF-8 JSON) and verify per-tensor dtype matches the dtype
/// implied by `manifest.quantization`:
///   fp16 → F16, bf16 → BF16, fp32 → F32
///
/// Norm/bias tensors (name contains "norm" or ends with ".bias") may remain
/// F32 — that's the canonical fp16 export shape. Weight tensors must match.
///
/// Would have caught the 30.46 GiB F32 fp16-manifest bug at publish time.
fn check_safetensors_header_dtype(
    top: &serde_yaml::Mapping,
    artifact: Option<&Path>,
) -> FalsifyResult {
    let format = get_str(top, "format").unwrap_or_default();
    if format != "safetensors" {
        return FalsifyResult {
            id: "FALSIFY-PM-007",
            verdict: "DEFERRED",
            detail: format!("format={format} — not safetensors; skip dtype gate"),
        };
    }
    let Some(path) = artifact else {
        return FalsifyResult {
            id: "FALSIFY-PM-007",
            verdict: "DEFERRED",
            detail: "no --artifact provided for safetensors header check".into(),
        };
    };
    let quant = get_str(top, "quantization").unwrap_or_default();
    let expected = match expected_safetensors_dtype(&quant) {
        Some(s) => s,
        None => {
            return FalsifyResult {
                id: "FALSIFY-PM-007",
                verdict: "DEFERRED",
                detail: format!("unknown quantization '{quant}' — cannot check dtype"),
            };
        }
    };
    match read_safetensors_header_dtypes(path) {
        Err(e) => FalsifyResult {
            id: "FALSIFY-PM-007",
            verdict: "FAIL",
            detail: format!("read header {}: {e}", path.display()),
        },
        Ok(entries) => {
            let mut mismatches: Vec<String> = Vec::new();
            let mut weight_count = 0usize;
            let mut exempt_count = 0usize;
            for (name, dtype) in &entries {
                if is_norm_or_bias(name) {
                    exempt_count += 1;
                    continue;
                }
                weight_count += 1;
                if dtype != expected {
                    mismatches.push(format!("{name}={dtype}"));
                }
            }
            if mismatches.is_empty() {
                FalsifyResult {
                    id: "FALSIFY-PM-007",
                    verdict: "PASS",
                    detail: format!(
                        "{weight_count} weight tensor(s) == {expected}; {exempt_count} norm/bias exempt"
                    ),
                }
            } else {
                let preview: Vec<_> = mismatches.iter().take(5).cloned().collect();
                FalsifyResult {
                    id: "FALSIFY-PM-007",
                    verdict: "FAIL",
                    detail: format!(
                        "{} weight tensor(s) declared {expected} but header has mismatches; first: {preview:?}",
                        mismatches.len()
                    ),
                }
            }
        }
    }
}

fn expected_safetensors_dtype(quant: &str) -> Option<&'static str> {
    match quant.to_ascii_lowercase().as_str() {
        "fp16" | "f16" | "float16" | "half" => Some("F16"),
        "bf16" | "bfloat16" => Some("BF16"),
        "fp32" | "f32" | "float32" | "float" => Some("F32"),
        _ => None,
    }
}

fn is_norm_or_bias(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.contains("norm") || lower.ends_with(".bias")
}

/// Reads the safetensors header and returns a Vec<(tensor_name, dtype)>.
/// The `__metadata__` key (if present) is filtered out.
fn read_safetensors_header_dtypes(path: &Path) -> Result<Vec<(String, String)>, String> {
    let mut f = fs::File::open(path).map_err(|e| e.to_string())?;
    let mut len_bytes = [0u8; 8];
    f.read_exact(&mut len_bytes)
        .map_err(|e| format!("read header length: {e}"))?;
    let header_len = u64::from_le_bytes(len_bytes);
    // Guard against absurd header sizes.
    const MAX_HEADER: u64 = 256 * 1024 * 1024;
    if header_len == 0 || header_len > MAX_HEADER {
        return Err(format!(
            "header_len {header_len} outside sane range [1, {MAX_HEADER}]"
        ));
    }
    let mut buf = vec![0u8; header_len as usize];
    f.read_exact(&mut buf)
        .map_err(|e| format!("read header body ({header_len} bytes): {e}"))?;
    let header: serde_json::Value =
        serde_json::from_slice(&buf).map_err(|e| format!("parse header json: {e}"))?;
    let obj = header
        .as_object()
        .ok_or_else(|| "header is not a JSON object".to_string())?;
    let mut out: Vec<(String, String)> = Vec::with_capacity(obj.len());
    for (name, val) in obj {
        if name == "__metadata__" {
            continue;
        }
        let dtype = val
            .get("dtype")
            .and_then(|v| v.as_str())
            .ok_or_else(|| format!("tensor '{name}' missing dtype"))?;
        out.push((name.clone(), dtype.to_string()));
    }
    Ok(out)
}

// ─────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────

fn get_str_key<'a>(map: &'a serde_yaml::Mapping, key: &str) -> Option<&'a serde_yaml::Value> {
    map.get(serde_yaml::Value::String(key.to_string()))
}

fn get_str(map: &serde_yaml::Mapping, key: &str) -> Option<String> {
    get_str_key(map, key)
        .and_then(serde_yaml::Value::as_str)
        .map(str::to_string)
}

fn compute_sha256(path: &Path) -> std::io::Result<String> {
    let mut f = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 65536];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn resolve_recipe(recipe_path: &str, manifest_path: &Path) -> PathBuf {
    let parent = manifest_path.parent().unwrap_or_else(|| Path::new("."));
    let candidate = parent.join(recipe_path);
    if candidate.exists() {
        return candidate;
    }
    PathBuf::from(recipe_path)
}

// ─────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;

    fn write(dir: &Path, name: &str, contents: &str) -> PathBuf {
        let p = dir.join(name);
        let mut f = fs::File::create(&p).unwrap();
        f.write_all(contents.as_bytes()).unwrap();
        p
    }

    #[test]
    fn compute_sha256_empty_file() {
        let dir = tempdir().unwrap();
        let p = write(dir.path(), "empty", "");
        let sha = compute_sha256(&p).unwrap();
        // sha256("") constant
        assert_eq!(
            sha,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn compute_sha256_known_value() {
        let dir = tempdir().unwrap();
        let p = write(dir.path(), "hello", "hello\n");
        let sha = compute_sha256(&p).unwrap();
        assert_eq!(
            sha,
            "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03"
        );
    }

    #[test]
    fn parent_chain_hf_id_passes() {
        let mut pm = serde_yaml::Mapping::new();
        pm.insert(
            serde_yaml::Value::String("parent".into()),
            serde_yaml::Value::String("Qwen/Qwen2.5-Coder-7B-Instruct".into()),
        );
        let r = check_parent_chain(Some(&pm));
        assert_eq!(r.verdict, "PASS", "{}", r.detail);
    }

    #[test]
    fn parent_chain_base_passes() {
        let mut pm = serde_yaml::Mapping::new();
        pm.insert(
            serde_yaml::Value::String("parent".into()),
            serde_yaml::Value::String("base".into()),
        );
        assert_eq!(check_parent_chain(Some(&pm)).verdict, "PASS");
    }

    #[test]
    fn parent_chain_bareword_fails() {
        let mut pm = serde_yaml::Mapping::new();
        pm.insert(
            serde_yaml::Value::String("parent".into()),
            serde_yaml::Value::String("qwen".into()),
        );
        assert_eq!(check_parent_chain(Some(&pm)).verdict, "FAIL");
    }

    #[test]
    fn spdx_apache_mit_pass() {
        let mut top = serde_yaml::Mapping::new();
        top.insert(
            serde_yaml::Value::String("license".into()),
            serde_yaml::Value::String("Apache-2.0".into()),
        );
        let mut prov = serde_yaml::Mapping::new();
        prov.insert(
            serde_yaml::Value::String("parent_license".into()),
            serde_yaml::Value::String("MIT".into()),
        );
        let r = check_spdx(&top, Some(&prov));
        assert_eq!(r.verdict, "PASS");
    }

    #[test]
    fn spdx_invalid_fails() {
        let mut top = serde_yaml::Mapping::new();
        top.insert(
            serde_yaml::Value::String("license".into()),
            serde_yaml::Value::String("WTFPL-99".into()),
        );
        let r = check_spdx(&top, None);
        assert_eq!(r.verdict, "FAIL");
    }

    #[test]
    fn head_live_missing_url_fails() {
        // F-PUBLISH-EXTRA-001::dogfood_ex05 — precondition: artifact_url is required.
        let top = serde_yaml::Mapping::new();
        let r = check_url_head_live(&top);
        assert_eq!(r.verdict, "FAIL");
        assert!(
            r.detail.contains("artifact_url"),
            "expected artifact_url mention, got: {}",
            r.detail
        );
        assert_eq!(r.id, "FALSIFY-PM-003");
    }

    #[test]
    fn sha256_live_missing_url_fails() {
        let top = serde_yaml::Mapping::new();
        let r = check_sha256_live(&top);
        assert_eq!(r.verdict, "FAIL");
        assert_eq!(r.id, "FALSIFY-PM-002-live");
        assert!(r.detail.contains("artifact_url"), "{}", r.detail);
    }

    #[test]
    fn sha256_live_missing_sha256_fails() {
        // artifact_url present but no declared sha256 → cannot verify.
        let mut top = serde_yaml::Mapping::new();
        top.insert(
            serde_yaml::Value::String("artifact_url".into()),
            serde_yaml::Value::String("https://example.test/file.bin".into()),
        );
        let r = check_sha256_live(&top);
        assert_eq!(r.verdict, "FAIL");
        assert_eq!(r.id, "FALSIFY-PM-002-live");
        assert!(r.detail.contains("sha256"), "{}", r.detail);
    }

    #[test]
    fn defer_url_liveness_id_and_verdict() {
        let r = defer_url_liveness();
        assert_eq!(r.verdict, "DEFERRED");
        assert_eq!(r.id, "FALSIFY-PM-003");
        assert!(
            r.detail.contains("--live"),
            "hint must point user at --live flag: {}",
            r.detail
        );
    }

    #[test]
    fn schema_missing_top_fails() {
        let top = serde_yaml::Mapping::new();
        let r = check_schema(&top, None);
        assert_eq!(r.verdict, "FAIL");
        assert!(r.detail.contains("model_id"));
    }

    // ─────────────────────────────────────────────────────────────
    // FALSIFY-PM-007 (safetensors header dtype Poka-Yoke)
    // ─────────────────────────────────────────────────────────────

    fn write_safetensors(dir: &Path, name: &str, header_json: &str) -> PathBuf {
        let p = dir.join(name);
        let mut f = fs::File::create(&p).unwrap();
        let body = header_json.as_bytes();
        f.write_all(&(body.len() as u64).to_le_bytes()).unwrap();
        f.write_all(body).unwrap();
        // A few bytes of fake tensor body so data_offsets are plausible.
        f.write_all(&[0u8; 16]).unwrap();
        p
    }

    fn top_with(format: &str, quant: &str) -> serde_yaml::Mapping {
        let mut top = serde_yaml::Mapping::new();
        top.insert(
            serde_yaml::Value::String("format".into()),
            serde_yaml::Value::String(format.into()),
        );
        top.insert(
            serde_yaml::Value::String("quantization".into()),
            serde_yaml::Value::String(quant.into()),
        );
        top
    }

    #[test]
    fn pm007_non_safetensors_deferred() {
        let top = top_with("apr", "q4_k");
        let r = check_safetensors_header_dtype(&top, None);
        assert_eq!(r.id, "FALSIFY-PM-007");
        assert_eq!(r.verdict, "DEFERRED");
        assert!(r.detail.contains("not safetensors"), "{}", r.detail);
    }

    #[test]
    fn pm007_missing_artifact_deferred() {
        let top = top_with("safetensors", "fp16");
        let r = check_safetensors_header_dtype(&top, None);
        assert_eq!(r.verdict, "DEFERRED");
        assert!(r.detail.contains("--artifact"), "{}", r.detail);
    }

    #[test]
    fn pm007_unknown_quant_deferred() {
        let dir = tempdir().unwrap();
        let path = write_safetensors(
            dir.path(),
            "m.safetensors",
            r#"{"x.weight":{"dtype":"F16","shape":[1],"data_offsets":[0,2]}}"#,
        );
        let top = top_with("safetensors", "q8_0");
        let r = check_safetensors_header_dtype(&top, Some(&path));
        assert_eq!(r.verdict, "DEFERRED");
        assert!(r.detail.contains("q8_0"), "{}", r.detail);
    }

    #[test]
    fn pm007_all_f16_weights_pass() {
        let dir = tempdir().unwrap();
        let path = write_safetensors(
            dir.path(),
            "m.safetensors",
            r#"{
                "blk.0.attn_q.weight":{"dtype":"F16","shape":[4,4],"data_offsets":[0,32]},
                "blk.0.attn_norm.weight":{"dtype":"F32","shape":[4],"data_offsets":[32,48]},
                "blk.0.attn_q.bias":{"dtype":"F32","shape":[4],"data_offsets":[48,64]},
                "__metadata__":{"format":"pt"}
            }"#,
        );
        let top = top_with("safetensors", "fp16");
        let r = check_safetensors_header_dtype(&top, Some(&path));
        assert_eq!(r.verdict, "PASS", "{}", r.detail);
        assert!(r.detail.contains("F16"), "{}", r.detail);
        // 1 weight, 2 exempt (norm + bias)
        assert!(r.detail.contains("1 weight"), "{}", r.detail);
        assert!(r.detail.contains("2 norm/bias"), "{}", r.detail);
    }

    #[test]
    fn pm007_f32_weight_when_fp16_declared_fails() {
        // This is the exact bug SHIP-TWO-001 §12.7.2 lists as a ship-blocker.
        let dir = tempdir().unwrap();
        let path = write_safetensors(
            dir.path(),
            "m.safetensors",
            r#"{
                "blk.0.attn_q.weight":{"dtype":"F32","shape":[4,4],"data_offsets":[0,64]},
                "blk.0.ffn_up.weight":{"dtype":"F32","shape":[4,4],"data_offsets":[64,128]},
                "blk.0.attn_norm.weight":{"dtype":"F32","shape":[4],"data_offsets":[128,144]}
            }"#,
        );
        let top = top_with("safetensors", "fp16");
        let r = check_safetensors_header_dtype(&top, Some(&path));
        assert_eq!(r.verdict, "FAIL", "{}", r.detail);
        assert!(
            r.detail.contains("attn_q") || r.detail.contains("ffn_up"),
            "mismatch list should name offending tensors: {}",
            r.detail
        );
    }

    #[test]
    fn pm007_bf16_declaration_requires_bf16_weights() {
        let dir = tempdir().unwrap();
        let path = write_safetensors(
            dir.path(),
            "m.safetensors",
            r#"{
                "blk.0.w":{"dtype":"F16","shape":[1],"data_offsets":[0,2]}
            }"#,
        );
        let top = top_with("safetensors", "bf16");
        let r = check_safetensors_header_dtype(&top, Some(&path));
        assert_eq!(r.verdict, "FAIL", "{}", r.detail);
    }

    #[test]
    fn pm007_corrupt_header_fails() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("corrupt.safetensors");
        // Oversize header length guard: 2^62 is well beyond MAX_HEADER.
        let mut f = fs::File::create(&p).unwrap();
        f.write_all(&(u64::MAX).to_le_bytes()).unwrap();
        let top = top_with("safetensors", "fp16");
        let r = check_safetensors_header_dtype(&top, Some(&p));
        assert_eq!(r.verdict, "FAIL");
        assert!(r.detail.contains("header"), "{}", r.detail);
    }

    #[test]
    fn pm007_is_norm_or_bias_heuristic() {
        assert!(is_norm_or_bias("attn_norm.weight"));
        assert!(is_norm_or_bias("blk.0.attn_output_norm.weight"));
        assert!(is_norm_or_bias("model.layers.0.mlp.down_proj.bias"));
        assert!(!is_norm_or_bias("blk.0.attn_q.weight"));
        assert!(!is_norm_or_bias("lm_head.weight"));
        assert!(!is_norm_or_bias("token_embd.weight"));
    }

    #[test]
    fn pm007_expected_dtype_mapping() {
        assert_eq!(expected_safetensors_dtype("fp16"), Some("F16"));
        assert_eq!(expected_safetensors_dtype("FP16"), Some("F16"));
        assert_eq!(expected_safetensors_dtype("bf16"), Some("BF16"));
        assert_eq!(expected_safetensors_dtype("fp32"), Some("F32"));
        assert_eq!(expected_safetensors_dtype("q4_k"), None);
    }
}
