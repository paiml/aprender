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
//!   FALSIFY-PM-008       GGUF tensor-type Poka-Yoke (only for format=gguf
//!                        + `--artifact`) — same class of defense for the third
//!                        shipped format. Predominant tensor type is
//!                        authoritative; `general.file_type` is advisory (real
//!                        llama.cpp quantize output has shipped with stale
//!                        ftype=0 despite fully Q4_K tensors).
//!   FALSIFY-PM-009       APR magic-bytes Poka-Yoke (only for format=apr
//!                        + `--artifact`) — closes the three-format ship
//!                        symmetry (PM-007 safetensors + PM-008 gguf + PM-009
//!                        apr). Catches "wrong file staged" (e.g. manifest
//!                        declares format=apr but .gguf is attached) BEFORE
//!                        network upload. v1.0 scope is magic-only; tensor
//!                        index quant validation deferred to v1.1.

use crate::error::CliError;
use colored::Colorize;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

// The SPDX table lives in `commands::spdx` so the commands that WRITE a
// license (`apr stamp`, `apr publish`) validate against exactly what this
// gate enforces — they used to accept anything (issue #2391).
use crate::commands::spdx::is_accepted as spdx_accepted;

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
    // FALSIFY-PM-008 — GGUF `general.file_type` Poka-Yoke.
    // Applies only when format == "gguf" AND --artifact is provided.
    results.push(check_gguf_file_type(top, artifact));
    // FALSIFY-PM-009 — APR magic-bytes Poka-Yoke.
    // Applies only when format == "apr" AND --artifact is provided.
    results.push(check_apr_magic(top, artifact));

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
        if spdx_accepted(val) {
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
// FALSIFY-PM-008 — GGUF `general.file_type` Poka-Yoke
// ─────────────────────────────────────────────────────────────
// Mirrors PM-007 for the third shipped format. GGUF stores
// `general.file_type` as a u32 whose value matches llama.cpp's
// LLAMA_FTYPE_MOSTLY_* enum. A manifest declaring quantization=q4_k_m
// against a file whose file_type != 15 is lying about the artifact —
// catch it before upload.

fn check_gguf_file_type(top: &serde_yaml::Mapping, artifact: Option<&Path>) -> FalsifyResult {
    let format = get_str(top, "format").unwrap_or_default();
    if format != "gguf" {
        return FalsifyResult {
            id: "FALSIFY-PM-008",
            verdict: "DEFERRED",
            detail: format!("format={format} — not gguf; skip file_type gate"),
        };
    }
    let Some(path) = artifact else {
        return FalsifyResult {
            id: "FALSIFY-PM-008",
            verdict: "DEFERRED",
            detail: "no --artifact provided for gguf file_type check".into(),
        };
    };
    let quant = get_str(top, "quantization").unwrap_or_default();
    let expected_tensor = expected_ggml_tensor_type(&quant);
    let expected_ftype_val = expected_gguf_ftype(&quant);
    if expected_tensor.is_none() && expected_ftype_val.is_none() {
        return FalsifyResult {
            id: "FALSIFY-PM-008",
            verdict: "DEFERRED",
            detail: format!("unknown quantization '{quant}' — cannot check file_type"),
        };
    }

    let sig = match read_gguf_signature(path) {
        Ok(s) => s,
        Err(e) => {
            return FalsifyResult {
                id: "FALSIFY-PM-008",
                verdict: "FAIL",
                detail: format!("read gguf {}: {e}", path.display()),
            };
        }
    };

    // Authoritative signal: predominant tensor type (what the file actually contains).
    // `general.file_type` is advisory and frequently stale in real-world GGUFs —
    // llama.cpp's own quantize tool has shipped artifacts with ftype=0 despite
    // quantized tensors, so tensor types are the source of truth.
    if let Some(observed) = predominant_quant_type(&sig.tensor_types) {
        if let Some(exp) = expected_tensor {
            if observed == exp {
                let ftype_note = match sig.ftype {
                    Some(ft) if Some(ft) == expected_ftype_val => String::new(),
                    Some(ft) => {
                        format!(
                            " (note: general.file_type={ft}={} is stale)",
                            gguf_ftype_name(ft)
                        )
                    }
                    None => String::new(),
                };
                return FalsifyResult {
                    id: "FALSIFY-PM-008",
                    verdict: "PASS",
                    detail: format!(
                        "predominant tensor type = {observed} ({}) matches quantization '{quant}'{ftype_note}",
                        ggml_type_name(observed)
                    ),
                };
            }
            return FalsifyResult {
                id: "FALSIFY-PM-008",
                verdict: "FAIL",
                detail: format!(
                    "manifest declares '{quant}' (ggml type {exp} = {}) but predominant tensor type is {observed} = {}",
                    ggml_type_name(exp),
                    ggml_type_name(observed)
                ),
            };
        }
    }

    // Fallback: header-level general.file_type (used when the file has no tensor
    // metadata, e.g. synthetic fixtures in unit tests).
    let Some(expected_ftype) = expected_ftype_val else {
        return FalsifyResult {
            id: "FALSIFY-PM-008",
            verdict: "FAIL",
            detail: format!(
                "no tensor types in {} and no ftype mapping for quant '{quant}'",
                path.display()
            ),
        };
    };
    match sig.ftype {
        None => FalsifyResult {
            id: "FALSIFY-PM-008",
            verdict: "FAIL",
            detail: "general.file_type not found in GGUF metadata".into(),
        },
        Some(observed) if observed == expected_ftype => FalsifyResult {
            id: "FALSIFY-PM-008",
            verdict: "PASS",
            detail: format!(
                "general.file_type = {observed} ({}) matches quantization '{quant}'",
                gguf_ftype_name(observed)
            ),
        },
        Some(observed) => FalsifyResult {
            id: "FALSIFY-PM-008",
            verdict: "FAIL",
            detail: format!(
                "manifest declares '{quant}' (ftype {expected_ftype} = {}) but header has ftype {observed} = {}",
                gguf_ftype_name(expected_ftype),
                gguf_ftype_name(observed)
            ),
        },
    }
}

/// Map manifest `quantization` strings to GGML tensor type codes
/// (source: ggml-common.h `enum ggml_type`). Coarser than `expected_gguf_ftype`:
/// Q4_K_M and Q4_K_S both materialize as `GGML_TYPE_Q4_K = 12` at the tensor
/// level — the "M"/"S" distinction is about WHICH tensors get Q6_K/Q5_K/Q4_K
/// mixed in, not the per-tensor code.
fn expected_ggml_tensor_type(quant: &str) -> Option<u32> {
    match quant.to_ascii_lowercase().as_str() {
        "fp32" | "f32" => Some(0),
        "fp16" | "f16" => Some(1),
        "q4_0" => Some(2),
        "q4_1" => Some(3),
        "q5_0" => Some(6),
        "q5_1" => Some(7),
        "q8_0" => Some(8),
        "q8_1" => Some(9),
        "q2_k" | "q2_k_s" => Some(10),
        "q3_k" | "q3_k_s" | "q3_k_m" | "q3_k_l" => Some(11),
        "q4_k" | "q4_k_s" | "q4_k_m" => Some(12),
        "q5_k" | "q5_k_s" | "q5_k_m" => Some(13),
        "q6_k" => Some(14),
        "q8_k" => Some(15),
        "iq2_xxs" => Some(16),
        "iq2_xs" => Some(17),
        "iq3_xxs" | "iq3_xs" => Some(18),
        "iq1_s" => Some(19),
        "iq4_nl" => Some(20),
        "iq3_s" | "iq3_m" => Some(21),
        "iq2_s" | "iq2_m" => Some(22),
        "iq4_xs" => Some(23),
        "iq1_m" => Some(29),
        "bf16" | "bfloat16" => Some(30),
        _ => None,
    }
}

fn ggml_type_name(t: u32) -> &'static str {
    match t {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        6 => "Q5_0",
        7 => "Q5_1",
        8 => "Q8_0",
        9 => "Q8_1",
        10 => "Q2_K",
        11 => "Q3_K",
        12 => "Q4_K",
        13 => "Q5_K",
        14 => "Q6_K",
        15 => "Q8_K",
        16 => "IQ2_XXS",
        17 => "IQ2_XS",
        18 => "IQ3_XXS",
        19 => "IQ1_S",
        20 => "IQ4_NL",
        21 => "IQ3_S",
        22 => "IQ2_S",
        23 => "IQ4_XS",
        24 => "I8",
        25 => "I16",
        26 => "I32",
        27 => "I64",
        28 => "F64",
        29 => "IQ1_M",
        30 => "BF16",
        _ => "UNKNOWN",
    }
}

/// Pick the "predominant" quantization type from a tensor-type histogram.
/// Returns the most common non-float type if any weights are quantized; else the
/// most common float type. Returns None only when the map is empty.
///
/// Rationale: bias/norm tensors are ALWAYS kept in F32 even in Q4_K_M models,
/// so raw mode would always pick F32 and give the wrong answer.
fn predominant_quant_type(counts: &HashMap<u32, usize>) -> Option<u32> {
    const FLOAT_TYPES: &[u32] = &[0, 1, 28, 30]; // F32, F16, F64, BF16
    let max_non_float = counts
        .iter()
        .filter(|(t, _)| !FLOAT_TYPES.contains(t))
        .max_by_key(|(_, n)| *n)
        .map(|(t, _)| *t);
    if max_non_float.is_some() {
        return max_non_float;
    }
    counts.iter().max_by_key(|(_, n)| *n).map(|(t, _)| *t)
}

struct GgufSignature {
    ftype: Option<u32>,
    tensor_types: HashMap<u32, usize>,
}

/// Map manifest `quantization` strings to llama.cpp `LLAMA_FTYPE_MOSTLY_*` u32s.
/// Source of truth: llama.cpp `llama.h` enum `llama_ftype`.
fn expected_gguf_ftype(quant: &str) -> Option<u32> {
    match quant.to_ascii_lowercase().as_str() {
        "fp32" | "f32" | "all_f32" => Some(0),
        "fp16" | "f16" | "mostly_f16" => Some(1),
        "q4_0" => Some(2),
        "q4_1" => Some(3),
        "q8_0" => Some(7),
        "q5_0" => Some(8),
        "q5_1" => Some(9),
        "q2_k" => Some(10),
        "q3_k_s" => Some(11),
        "q3_k_m" => Some(12),
        "q3_k_l" => Some(13),
        "q4_k_s" => Some(14),
        "q4_k_m" | "q4_k" => Some(15),
        "q5_k_s" => Some(16),
        "q5_k_m" | "q5_k" => Some(17),
        "q6_k" => Some(18),
        "iq2_xxs" => Some(19),
        "iq2_xs" => Some(20),
        "q2_k_s" => Some(21),
        "iq3_xs" | "iq3_xxs" => Some(23),
        "iq1_s" => Some(24),
        "iq4_nl" => Some(25),
        "iq3_s" => Some(26),
        "iq3_m" => Some(27),
        "iq2_s" => Some(28),
        "iq2_m" => Some(29),
        "iq4_xs" => Some(30),
        "iq1_m" => Some(31),
        "bf16" | "bfloat16" => Some(32),
        _ => None,
    }
}

fn gguf_ftype_name(ftype: u32) -> &'static str {
    match ftype {
        0 => "ALL_F32",
        1 => "MOSTLY_F16",
        2 => "MOSTLY_Q4_0",
        3 => "MOSTLY_Q4_1",
        7 => "MOSTLY_Q8_0",
        8 => "MOSTLY_Q5_0",
        9 => "MOSTLY_Q5_1",
        10 => "MOSTLY_Q2_K",
        11 => "MOSTLY_Q3_K_S",
        12 => "MOSTLY_Q3_K_M",
        13 => "MOSTLY_Q3_K_L",
        14 => "MOSTLY_Q4_K_S",
        15 => "MOSTLY_Q4_K_M",
        16 => "MOSTLY_Q5_K_S",
        17 => "MOSTLY_Q5_K_M",
        18 => "MOSTLY_Q6_K",
        19 => "MOSTLY_IQ2_XXS",
        20 => "MOSTLY_IQ2_XS",
        21 => "MOSTLY_Q2_K_S",
        23 => "MOSTLY_IQ3_XXS",
        24 => "MOSTLY_IQ1_S",
        25 => "MOSTLY_IQ4_NL",
        26 => "MOSTLY_IQ3_S",
        27 => "MOSTLY_IQ3_M",
        28 => "MOSTLY_IQ2_S",
        29 => "MOSTLY_IQ2_M",
        30 => "MOSTLY_IQ4_XS",
        31 => "MOSTLY_IQ1_M",
        32 => "MOSTLY_BF16",
        _ => "UNKNOWN",
    }
}

/// Read the GGUF header + tensor-metadata section, returning:
///   - optional `general.file_type` (advisory; frequently stale in real files)
///   - histogram of GGML tensor-type codes → counts (authoritative)
///
/// Format: magic(4) + version(u32) + tensor_count(u64) + kv_count(u64) +
/// kv_entries + tensor_metadata. Each kv: key_len(u64) + key(utf-8) +
/// value_type(u32) + value. Each tensor_metadata: name_len(u64) + name(utf-8)
/// + n_dims(u32) + dims(n_dims*u64) + type(u32) + offset(u64).
fn read_gguf_signature(path: &Path) -> Result<GgufSignature, String> {
    let mut f = fs::File::open(path).map_err(|e| e.to_string())?;
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)
        .map_err(|e| format!("read magic: {e}"))?;
    if &magic != b"GGUF" {
        return Err(format!("bad magic: {magic:?} (expected GGUF)"));
    }
    let mut u32buf = [0u8; 4];
    let mut u64buf = [0u8; 8];
    f.read_exact(&mut u32buf)
        .map_err(|e| format!("read version: {e}"))?;
    let version = u32::from_le_bytes(u32buf);
    if !(1..=3).contains(&version) {
        return Err(format!("unsupported GGUF version {version}"));
    }
    f.read_exact(&mut u64buf)
        .map_err(|e| format!("read tensor_count: {e}"))?;
    let tensor_count = u64::from_le_bytes(u64buf);
    const MAX_TENSOR_COUNT: u64 = 100_000;
    if tensor_count > MAX_TENSOR_COUNT {
        return Err(format!(
            "tensor_count {tensor_count} exceeds sane limit {MAX_TENSOR_COUNT}"
        ));
    }
    f.read_exact(&mut u64buf)
        .map_err(|e| format!("read kv_count: {e}"))?;
    let kv_count = u64::from_le_bytes(u64buf);
    const MAX_KV_COUNT: u64 = 10_000;
    if kv_count > MAX_KV_COUNT {
        return Err(format!(
            "kv_count {kv_count} exceeds sane limit {MAX_KV_COUNT}"
        ));
    }

    let mut ftype: Option<u32> = None;
    for _ in 0..kv_count {
        f.read_exact(&mut u64buf)
            .map_err(|e| format!("read key_len: {e}"))?;
        let key_len = u64::from_le_bytes(u64buf);
        if key_len > 1024 {
            return Err(format!("key_len {key_len} > 1024"));
        }
        let mut key_buf = vec![0u8; key_len as usize];
        f.read_exact(&mut key_buf)
            .map_err(|e| format!("read key: {e}"))?;
        let key = String::from_utf8(key_buf).map_err(|e| format!("key utf8: {e}"))?;
        f.read_exact(&mut u32buf)
            .map_err(|e| format!("read value_type for {key}: {e}"))?;
        let value_type = u32::from_le_bytes(u32buf);
        if key == "general.file_type" && value_type == 4 {
            f.read_exact(&mut u32buf)
                .map_err(|e| format!("read file_type u32: {e}"))?;
            ftype = Some(u32::from_le_bytes(u32buf));
        } else {
            skip_gguf_value(&mut f, value_type)
                .map_err(|e| format!("skip value for key {key}: {e}"))?;
        }
    }

    let mut tensor_types: HashMap<u32, usize> = HashMap::new();
    for _ in 0..tensor_count {
        f.read_exact(&mut u64buf)
            .map_err(|e| format!("read tensor name_len: {e}"))?;
        let name_len = u64::from_le_bytes(u64buf);
        if name_len > 1024 {
            return Err(format!("tensor name_len {name_len} > 1024"));
        }
        let mut name_buf = vec![0u8; name_len as usize];
        f.read_exact(&mut name_buf)
            .map_err(|e| format!("read tensor name: {e}"))?;
        f.read_exact(&mut u32buf)
            .map_err(|e| format!("read tensor n_dims: {e}"))?;
        let n_dims = u32::from_le_bytes(u32buf);
        if n_dims > 4 {
            return Err(format!("tensor n_dims {n_dims} > 4"));
        }
        for _ in 0..n_dims {
            f.read_exact(&mut u64buf)
                .map_err(|e| format!("read tensor dim: {e}"))?;
        }
        f.read_exact(&mut u32buf)
            .map_err(|e| format!("read tensor type: {e}"))?;
        let ttype = u32::from_le_bytes(u32buf);
        f.read_exact(&mut u64buf)
            .map_err(|e| format!("read tensor offset: {e}"))?;
        *tensor_types.entry(ttype).or_insert(0) += 1;
    }

    Ok(GgufSignature {
        ftype,
        tensor_types,
    })
}

/// Advance `f` past one GGUF value of the given type without materializing it.
/// Required so we can skip over metadata_kv entries we don't care about.
fn skip_gguf_value<R: std::io::Read + std::io::Seek>(
    f: &mut R,
    value_type: u32,
) -> Result<(), String> {
    use std::io::SeekFrom;
    match value_type {
        // uint8, int8, bool
        0 | 1 | 7 => {
            f.seek(SeekFrom::Current(1)).map_err(|e| e.to_string())?;
        }
        // uint16, int16
        2 | 3 => {
            f.seek(SeekFrom::Current(2)).map_err(|e| e.to_string())?;
        }
        // uint32, int32, float32
        4 | 5 | 6 => {
            f.seek(SeekFrom::Current(4)).map_err(|e| e.to_string())?;
        }
        // uint64, int64, float64
        10 | 11 | 12 => {
            f.seek(SeekFrom::Current(8)).map_err(|e| e.to_string())?;
        }
        // string: u64 length + bytes
        8 => {
            let mut u64buf = [0u8; 8];
            f.read_exact(&mut u64buf)
                .map_err(|e| format!("read string len: {e}"))?;
            let len = u64::from_le_bytes(u64buf);
            if len > 10_000_000 {
                return Err(format!("string len {len} absurdly large"));
            }
            f.seek(SeekFrom::Current(len as i64))
                .map_err(|e| e.to_string())?;
        }
        // array: u32 element_type + u64 count + elements
        9 => {
            let mut u32buf = [0u8; 4];
            let mut u64buf = [0u8; 8];
            f.read_exact(&mut u32buf)
                .map_err(|e| format!("read array elem_type: {e}"))?;
            let elem_type = u32::from_le_bytes(u32buf);
            f.read_exact(&mut u64buf)
                .map_err(|e| format!("read array count: {e}"))?;
            let count = u64::from_le_bytes(u64buf);
            if count > 10_000_000 {
                return Err(format!("array count {count} absurdly large"));
            }
            for _ in 0..count {
                skip_gguf_value(f, elem_type).map_err(|e| format!("skip array elem: {e}"))?;
            }
        }
        other => return Err(format!("unknown value_type {other}")),
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────
// FALSIFY-PM-009 — APR magic-bytes Poka-Yoke
// ─────────────────────────────────────────────────────────────
// Closes the three-format ship symmetry (PM-007 safetensors + PM-008 gguf +
// PM-009 apr). v1.0 scope: verify the first 4 bytes match one of the APR
// magic variants (`APR\0`, `APRN`, `APR1`, `APR2`). Catches the "wrong file
// staged under manifest format=apr" ship-blocker class without the complexity
// of parsing the APR v2 tensor index.
//
// Expansion path (deferred to v1.1): use `aprender::format::v2::AprV2Reader`
// to parse the tensor index and compute predominant per-tensor dtype, then
// compare to manifest `quantization` (symmetric to PM-008's tensor-authority
// path). v1.0 unblocks the same ship-blocker class at the magic layer.

const APR_MAGICS: &[&[u8; 4]] = &[b"APR\0", b"APRN", b"APR1", b"APR2"];

fn check_apr_magic(top: &serde_yaml::Mapping, artifact: Option<&Path>) -> FalsifyResult {
    let format = get_str(top, "format").unwrap_or_default();
    if format != "apr" {
        return FalsifyResult {
            id: "FALSIFY-PM-009",
            verdict: "DEFERRED",
            detail: format!("format={format} — not apr; skip magic gate"),
        };
    }
    let Some(path) = artifact else {
        return FalsifyResult {
            id: "FALSIFY-PM-009",
            verdict: "DEFERRED",
            detail: "no --artifact provided for apr magic check".into(),
        };
    };
    match read_apr_magic(path) {
        Err(e) => FalsifyResult {
            id: "FALSIFY-PM-009",
            verdict: "FAIL",
            detail: format!("read apr magic {}: {e}", path.display()),
        },
        Ok(magic) => {
            if APR_MAGICS.iter().any(|m| *m == &magic) {
                FalsifyResult {
                    id: "FALSIFY-PM-009",
                    verdict: "PASS",
                    detail: format!("apr magic = {} (valid)", apr_magic_name(&magic)),
                }
            } else {
                FalsifyResult {
                    id: "FALSIFY-PM-009",
                    verdict: "FAIL",
                    detail: format!(
                        "manifest declares format=apr but file magic is {} {magic:?} (expected one of APR\\0, APRN, APR1, APR2)",
                        ascii_or_hex(&magic),
                    ),
                }
            }
        }
    }
}

fn read_apr_magic(path: &Path) -> Result<[u8; 4], String> {
    let mut f = fs::File::open(path).map_err(|e| e.to_string())?;
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)
        .map_err(|e| format!("read magic: {e}"))?;
    Ok(magic)
}

fn apr_magic_name(magic: &[u8; 4]) -> &'static str {
    match magic {
        b"APR\0" => "APR\\0 (v2)",
        b"APRN" => "APRN (v1)",
        b"APR1" => "APR1",
        b"APR2" => "APR2",
        _ => "UNKNOWN",
    }
}

/// Render magic bytes as a readable string when all bytes are printable ASCII,
/// otherwise fall back to the decimal debug form. Used in FAIL details so
/// `"GGUF"` shows up instead of `[71, 71, 85, 70]`.
fn ascii_or_hex(bytes: &[u8; 4]) -> String {
    if bytes.iter().all(|b| (0x20..0x7f).contains(b)) {
        format!("\"{}\"", String::from_utf8_lossy(bytes))
    } else {
        String::new()
    }
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

    // ─────────────────────────────────────────────────────────────
    // FALSIFY-PM-008 (GGUF general.file_type Poka-Yoke)
    // ─────────────────────────────────────────────────────────────

    /// Write a minimal synthetic GGUF v3 header with the given kv pairs.
    /// Each kv is (key, value_type u32, raw_value_bytes).
    fn write_gguf(dir: &Path, name: &str, kvs: &[(&str, u32, Vec<u8>)]) -> PathBuf {
        let p = dir.join(name);
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes()); // version
        buf.extend_from_slice(&0u64.to_le_bytes()); // tensor_count = 0
        buf.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
        for (key, vtype, val) in kvs {
            let kb = key.as_bytes();
            buf.extend_from_slice(&(kb.len() as u64).to_le_bytes());
            buf.extend_from_slice(kb);
            buf.extend_from_slice(&vtype.to_le_bytes());
            buf.extend_from_slice(val);
        }
        fs::write(&p, &buf).unwrap();
        p
    }

    fn kv_u32(v: u32) -> Vec<u8> {
        v.to_le_bytes().to_vec()
    }

    fn kv_string(s: &str) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&(s.len() as u64).to_le_bytes());
        out.extend_from_slice(s.as_bytes());
        out
    }

    #[test]
    fn pm008_non_gguf_deferred() {
        let top = top_with("safetensors", "fp16");
        let r = check_gguf_file_type(&top, None);
        assert_eq!(r.id, "FALSIFY-PM-008");
        assert_eq!(r.verdict, "DEFERRED");
        assert!(r.detail.contains("not gguf"), "{}", r.detail);
    }

    #[test]
    fn pm008_missing_artifact_deferred() {
        let top = top_with("gguf", "q4_k_m");
        let r = check_gguf_file_type(&top, None);
        assert_eq!(r.verdict, "DEFERRED");
        assert!(r.detail.contains("--artifact"), "{}", r.detail);
    }

    #[test]
    fn pm008_unknown_quant_deferred() {
        let dir = tempdir().unwrap();
        let path = write_gguf(
            dir.path(),
            "m.gguf",
            &[("general.file_type", 4, kv_u32(15))],
        );
        let top = top_with("gguf", "some_future_quant");
        let r = check_gguf_file_type(&top, Some(&path));
        assert_eq!(r.verdict, "DEFERRED");
    }

    #[test]
    fn pm008_q4_k_m_pass() {
        let dir = tempdir().unwrap();
        let path = write_gguf(
            dir.path(),
            "m.gguf",
            &[
                ("general.architecture", 8, kv_string("qwen2")),
                ("general.file_type", 4, kv_u32(15)), // MOSTLY_Q4_K_M
            ],
        );
        let top = top_with("gguf", "q4_k_m");
        let r = check_gguf_file_type(&top, Some(&path));
        assert_eq!(r.verdict, "PASS", "{}", r.detail);
        assert!(r.detail.contains("MOSTLY_Q4_K_M"), "{}", r.detail);
    }

    #[test]
    fn pm008_ftype_mismatch_fails() {
        // Exact ship-blocker: manifest lies about quantization.
        let dir = tempdir().unwrap();
        let path = write_gguf(
            dir.path(),
            "m.gguf",
            &[("general.file_type", 4, kv_u32(1))], // MOSTLY_F16, not q4_k_m
        );
        let top = top_with("gguf", "q4_k_m");
        let r = check_gguf_file_type(&top, Some(&path));
        assert_eq!(r.verdict, "FAIL", "{}", r.detail);
        assert!(r.detail.contains("ftype 1"), "{}", r.detail);
        assert!(r.detail.contains("ftype 15"), "{}", r.detail);
    }

    #[test]
    fn pm008_skips_unrelated_kvs_then_finds_file_type() {
        // Verify the skip path for strings/arrays works — file_type must be found
        // even when it's not the first entry.
        let dir = tempdir().unwrap();
        let mut arr_body: Vec<u8> = Vec::new();
        arr_body.extend_from_slice(&4u32.to_le_bytes()); // elem type: u32
        arr_body.extend_from_slice(&3u64.to_le_bytes()); // count: 3
        for v in [1u32, 2, 3] {
            arr_body.extend_from_slice(&v.to_le_bytes());
        }
        let path = write_gguf(
            dir.path(),
            "m.gguf",
            &[
                ("general.name", 8, kv_string("qwen2.5-coder-7b")),
                ("some.array", 9, arr_body),
                ("general.file_type", 4, kv_u32(18)), // MOSTLY_Q6_K
            ],
        );
        let top = top_with("gguf", "q6_k");
        let r = check_gguf_file_type(&top, Some(&path));
        assert_eq!(r.verdict, "PASS", "{}", r.detail);
    }

    #[test]
    fn pm008_missing_file_type_kv_fails() {
        let dir = tempdir().unwrap();
        let path = write_gguf(
            dir.path(),
            "m.gguf",
            &[("general.architecture", 8, kv_string("qwen2"))],
        );
        let top = top_with("gguf", "q4_k_m");
        let r = check_gguf_file_type(&top, Some(&path));
        assert_eq!(r.verdict, "FAIL", "{}", r.detail);
        assert!(
            r.detail.contains("general.file_type not found"),
            "{}",
            r.detail
        );
    }

    #[test]
    fn pm008_bad_magic_fails() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("bad.gguf");
        fs::write(&p, b"NOPENOPE").unwrap();
        let top = top_with("gguf", "q4_k_m");
        let r = check_gguf_file_type(&top, Some(&p));
        assert_eq!(r.verdict, "FAIL");
        assert!(r.detail.contains("magic"), "{}", r.detail);
    }

    #[test]
    fn pm008_expected_ftype_mapping() {
        assert_eq!(expected_gguf_ftype("q4_k_m"), Some(15));
        assert_eq!(expected_gguf_ftype("Q4_K_M"), Some(15));
        assert_eq!(expected_gguf_ftype("q4_k"), Some(15));
        assert_eq!(expected_gguf_ftype("q5_k_m"), Some(17));
        assert_eq!(expected_gguf_ftype("q6_k"), Some(18));
        assert_eq!(expected_gguf_ftype("q8_0"), Some(7));
        assert_eq!(expected_gguf_ftype("fp16"), Some(1));
        assert_eq!(expected_gguf_ftype("bf16"), Some(32));
        assert_eq!(expected_gguf_ftype("nonsense"), None);
    }

    #[test]
    fn pm008_ftype_name_mapping() {
        assert_eq!(gguf_ftype_name(15), "MOSTLY_Q4_K_M");
        assert_eq!(gguf_ftype_name(18), "MOSTLY_Q6_K");
        assert_eq!(gguf_ftype_name(1), "MOSTLY_F16");
        assert_eq!(gguf_ftype_name(32), "MOSTLY_BF16");
        assert_eq!(gguf_ftype_name(999), "UNKNOWN");
    }

    #[test]
    fn pm008_ggml_type_mapping() {
        assert_eq!(expected_ggml_tensor_type("q4_k"), Some(12));
        assert_eq!(expected_ggml_tensor_type("q4_k_m"), Some(12));
        assert_eq!(expected_ggml_tensor_type("q4_k_s"), Some(12));
        assert_eq!(expected_ggml_tensor_type("q6_k"), Some(14));
        assert_eq!(expected_ggml_tensor_type("fp16"), Some(1));
        assert_eq!(expected_ggml_tensor_type("bf16"), Some(30));
        assert_eq!(expected_ggml_tensor_type("unknown"), None);
        assert_eq!(ggml_type_name(12), "Q4_K");
        assert_eq!(ggml_type_name(14), "Q6_K");
    }

    #[test]
    fn pm008_predominant_prefers_non_float() {
        let mut h = HashMap::new();
        h.insert(0u32, 60); // F32 (bias/norm)
        h.insert(12u32, 280); // Q4_K (weights)
        assert_eq!(predominant_quant_type(&h), Some(12));
    }

    #[test]
    fn pm008_predominant_all_float_falls_back() {
        let mut h = HashMap::new();
        h.insert(0u32, 10);
        h.insert(1u32, 3);
        assert_eq!(predominant_quant_type(&h), Some(0));
    }

    /// Write a synthetic GGUF v3 header with non-zero tensor metadata. Tensors
    /// carry no actual data — only the metadata block matters for PM-008.
    /// Each tensor: (name, dims, ggml_type).
    fn write_gguf_with_tensors(
        dir: &Path,
        name: &str,
        kvs: &[(&str, u32, Vec<u8>)],
        tensors: &[(&str, &[u64], u32)],
    ) -> PathBuf {
        let p = dir.join(name);
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes()); // version
        buf.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        buf.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
        for (key, vtype, val) in kvs {
            let kb = key.as_bytes();
            buf.extend_from_slice(&(kb.len() as u64).to_le_bytes());
            buf.extend_from_slice(kb);
            buf.extend_from_slice(&vtype.to_le_bytes());
            buf.extend_from_slice(val);
        }
        let mut offset: u64 = 0;
        for (tname, dims, ttype) in tensors {
            let nb = tname.as_bytes();
            buf.extend_from_slice(&(nb.len() as u64).to_le_bytes());
            buf.extend_from_slice(nb);
            buf.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for d in *dims {
                buf.extend_from_slice(&d.to_le_bytes());
            }
            buf.extend_from_slice(&ttype.to_le_bytes());
            buf.extend_from_slice(&offset.to_le_bytes());
            offset += 64; // dummy
        }
        fs::write(&p, &buf).unwrap();
        p
    }

    /// Real-world teacher-GGUF scenario: Q4_K tensors but stale
    /// general.file_type = 0 (ALL_F32). Tensor-type evidence must win.
    #[test]
    fn pm008_q4_k_tensors_override_stale_ftype_zero() {
        let dir = tempdir().unwrap();
        let path = write_gguf_with_tensors(
            dir.path(),
            "teacher.gguf",
            &[
                ("general.architecture", 8, kv_string("qwen2")),
                ("general.file_type", 4, kv_u32(0)), // stale: claims ALL_F32
            ],
            &[
                ("blk.0.attn_q.weight", &[3584, 3584], 12),  // Q4_K
                ("blk.0.attn_k.weight", &[3584, 512], 12),   // Q4_K
                ("blk.0.ffn_up.weight", &[3584, 18944], 12), // Q4_K
                ("blk.0.attn_norm.weight", &[3584], 0),      // F32 bias/norm
                ("blk.0.attn_q.bias", &[3584], 0),           // F32 bias
            ],
        );
        let top = top_with("gguf", "q4_k");
        let r = check_gguf_file_type(&top, Some(&path));
        assert_eq!(r.verdict, "PASS", "{}", r.detail);
        assert!(r.detail.contains("predominant tensor type"), "{}", r.detail);
        assert!(r.detail.contains("Q4_K"), "{}", r.detail);
        assert!(
            r.detail.contains("stale"),
            "expected stale-ftype note: {}",
            r.detail
        );
    }

    /// Tensor types are Q6_K but manifest says q4_k → FAIL. Catches the
    /// "wrong file pointed at by manifest" ship-blocker class.
    #[test]
    fn pm008_tensor_type_mismatch_fails() {
        let dir = tempdir().unwrap();
        let path = write_gguf_with_tensors(
            dir.path(),
            "wrong.gguf",
            &[("general.file_type", 4, kv_u32(18))], // MOSTLY_Q6_K (also wrong)
            &[
                ("blk.0.attn_q.weight", &[3584, 3584], 14), // Q6_K
                ("blk.0.attn_k.weight", &[3584, 512], 14),  // Q6_K
                ("blk.0.attn_norm.weight", &[3584], 0),
            ],
        );
        let top = top_with("gguf", "q4_k");
        let r = check_gguf_file_type(&top, Some(&path));
        assert_eq!(r.verdict, "FAIL", "{}", r.detail);
        assert!(r.detail.contains("Q4_K"), "{}", r.detail);
        assert!(r.detail.contains("Q6_K"), "{}", r.detail);
    }

    // ─────────────────────────────────────────────────────────────
    // FALSIFY-PM-009 — APR magic-bytes Poka-Yoke
    // ─────────────────────────────────────────────────────────────

    fn write_apr_magic(dir: &Path, name: &str, magic: &[u8]) -> PathBuf {
        let p = dir.join(name);
        let mut f = fs::File::create(&p).unwrap();
        f.write_all(magic).unwrap();
        // Pad with zero bytes so the file has a plausible header size.
        f.write_all(&[0u8; 32]).unwrap();
        p
    }

    #[test]
    fn pm009_non_apr_deferred() {
        let top = top_with("safetensors", "fp16");
        let r = check_apr_magic(&top, None);
        assert_eq!(r.verdict, "DEFERRED", "{}", r.detail);
        assert!(r.detail.contains("not apr"), "{}", r.detail);
    }

    #[test]
    fn pm009_missing_artifact_deferred() {
        let top = top_with("apr", "q4_k");
        let r = check_apr_magic(&top, None);
        assert_eq!(r.verdict, "DEFERRED", "{}", r.detail);
        assert!(r.detail.contains("no --artifact"), "{}", r.detail);
    }

    #[test]
    fn pm009_apr_null_magic_passes() {
        let dir = tempdir().unwrap();
        let p = write_apr_magic(dir.path(), "m.apr", b"APR\0");
        let top = top_with("apr", "q4_k");
        let r = check_apr_magic(&top, Some(&p));
        assert_eq!(r.verdict, "PASS", "{}", r.detail);
        assert!(r.detail.contains("APR"), "{}", r.detail);
    }

    #[test]
    fn pm009_aprn_magic_passes() {
        let dir = tempdir().unwrap();
        let p = write_apr_magic(dir.path(), "m.apr", b"APRN");
        let top = top_with("apr", "q4_k");
        let r = check_apr_magic(&top, Some(&p));
        assert_eq!(r.verdict, "PASS", "{}", r.detail);
        assert!(r.detail.contains("APRN"), "{}", r.detail);
    }

    #[test]
    fn pm009_apr2_magic_passes() {
        let dir = tempdir().unwrap();
        let p = write_apr_magic(dir.path(), "m.apr", b"APR2");
        let top = top_with("apr", "q4_k");
        let r = check_apr_magic(&top, Some(&p));
        assert_eq!(r.verdict, "PASS", "{}", r.detail);
        assert!(r.detail.contains("APR2"), "{}", r.detail);
    }

    #[test]
    fn pm009_gguf_magic_staged_as_apr_fails() {
        // The exact ship-blocker this gate exists to catch: a .gguf file
        // renamed to .apr or staged under a format=apr manifest.
        let dir = tempdir().unwrap();
        let p = write_apr_magic(dir.path(), "wrong.apr", b"GGUF");
        let top = top_with("apr", "q4_k");
        let r = check_apr_magic(&top, Some(&p));
        assert_eq!(r.verdict, "FAIL", "{}", r.detail);
        assert!(r.detail.contains("GGUF"), "{}", r.detail);
    }

    #[test]
    fn pm009_safetensors_magic_staged_as_apr_fails() {
        // safetensors has no fixed 4-byte magic (starts with u64 header-length
        // LE), but 8 random-ish bytes will almost never match an APR magic.
        let dir = tempdir().unwrap();
        let p = write_apr_magic(dir.path(), "wrong.apr", b"\x80\x00\x00\x00");
        let top = top_with("apr", "q4_k");
        let r = check_apr_magic(&top, Some(&p));
        assert_eq!(r.verdict, "FAIL", "{}", r.detail);
    }

    #[test]
    fn pm009_empty_file_fails() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("empty.apr");
        fs::File::create(&p).unwrap(); // 0 bytes
        let top = top_with("apr", "q4_k");
        let r = check_apr_magic(&top, Some(&p));
        assert_eq!(r.verdict, "FAIL", "{}", r.detail);
        assert!(r.detail.contains("read"), "{}", r.detail);
    }

    #[test]
    fn pm009_magic_name_table() {
        assert_eq!(apr_magic_name(b"APR\0"), "APR\\0 (v2)");
        assert_eq!(apr_magic_name(b"APRN"), "APRN (v1)");
        assert_eq!(apr_magic_name(b"APR1"), "APR1");
        assert_eq!(apr_magic_name(b"APR2"), "APR2");
        assert_eq!(apr_magic_name(b"XXXX"), "UNKNOWN");
    }
}
