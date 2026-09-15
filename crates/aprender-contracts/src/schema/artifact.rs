//! What a `.yaml` file under `contracts/` actually IS, and how to validate it.
//!
//! # Why this exists
//!
//! `pv validate <file>` had exactly one answer to that question: every file is
//! a [`Contract`], so parse it as one. Three of the artifacts the repo keeps
//! under `contracts/` are not contracts, and all five of those files failed
//! with the same message — ``Failed to parse YAML: missing field `metadata` ``:
//!
//! * `contracts/binding.yaml` and `contracts/aprender/binding.yaml` — pv's OWN
//!   binding registries, the equation → implementing-function maps that `pv
//!   audit --binding` and `pv probar --binding` read. pv could not validate its
//!   own artifact.
//! * `contracts/publish-manifests/*.yaml` — three model publish manifests,
//!   whose gate is `apr validate-manifest` (FALSIFY-PM-001..009).
//!
//! Both were already KNOWN not to be contracts: `is_contract_yaml` excludes
//! `binding.yaml` by name and `lint::gates::collect_yaml_files` skips the
//! `publish-manifests/` directory outright, with a comment saying why. The
//! knowledge just never reached `pv validate`, so the single-file surface and
//! the directory surface disagreed — the same two-walkers-disagree defect the
//! `is_contract_yaml` doc comment describes.
//!
//! # The discriminator
//!
//! `metadata:` is a REQUIRED field of every contract. A YAML mapping under
//! `contracts/` that has no `metadata:` is therefore not a contract, whatever
//! else it is, and this module reads its shape to say what it is instead. A
//! file that has no `metadata:` and matches no known artifact shape still
//! fails, with the contract parse error it has today — recognising an artifact
//! is never a way to pass without being checked.

use std::path::{Path, PathBuf};

use serde_yaml::{Mapping, Value};

use super::parser::parse_contract_str;
use super::validator::validate_contract;
use crate::binding::validate_binding_registry;
use crate::error::{ContractError, Severity, Violation};

/// The kind of artifact a `.yaml` file under `contracts/` is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArtifactKind {
    /// A [`crate::schema::Contract`] — has a `metadata:` block.
    Contract,
    /// A pv binding registry (`crate::binding::BindingRegistry`).
    Binding,
    /// A model publish manifest, conforming to
    /// `contracts/publish-manifest-v1.yaml` §schema.
    PublishManifest,
}

impl std::fmt::Display for ArtifactKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Contract => "contract",
            Self::Binding => "binding",
            Self::PublishManifest => "publish-manifest",
        };
        write!(f, "{s}")
    }
}

/// Which artifact `yaml` is, from its shape alone.
///
/// Shape, not filename: `contracts/aprender/binding.yaml` and
/// `contracts/binding.yaml` are both registries and `is_contract_yaml`
/// recognises them by name, but a registry saved under any other name is still
/// a registry and must not be told it is missing a `metadata:` block. A
/// document that is not a mapping, or that has a `metadata:` key, is a
/// contract as far as this function is concerned — the contract parser then
/// gets the final word.
#[must_use]
pub fn classify_artifact(yaml: &str) -> ArtifactKind {
    let Ok(Value::Mapping(map)) = serde_yaml::from_str::<Value>(yaml) else {
        return ArtifactKind::Contract;
    };
    if map.contains_key("metadata") {
        return ArtifactKind::Contract;
    }
    if map.contains_key("bindings") && map.contains_key("target_crate") {
        return ArtifactKind::Binding;
    }
    if map.contains_key("model_id") && map.contains_key("provenance") {
        return ArtifactKind::PublishManifest;
    }
    ArtifactKind::Contract
}

/// Validate whatever kind of artifact lives at `path`.
///
/// Returns the kind that was recognised together with its violations. An
/// `Err` means the file could not be read or parsed at all.
///
/// # Errors
///
/// [`ContractError::Io`] if the file cannot be read, [`ContractError::Yaml`]
/// if it does not parse as its recognised kind.
pub fn validate_artifact(path: &Path) -> Result<(ArtifactKind, Vec<Violation>), ContractError> {
    let content = std::fs::read_to_string(path)?;
    match classify_artifact(&content) {
        ArtifactKind::Contract => {
            let contract = parse_contract_str(&content)?;
            Ok((ArtifactKind::Contract, validate_contract(&contract)))
        }
        ArtifactKind::Binding => {
            let registry = crate::binding::parse_binding_str(&content)?;
            Ok((ArtifactKind::Binding, validate_binding_registry(&registry)))
        }
        ArtifactKind::PublishManifest => {
            let manifest: Value = serde_yaml::from_str(&content)?;
            Ok((
                ArtifactKind::PublishManifest,
                validate_publish_manifest(&manifest, path),
            ))
        }
    }
}

fn violation(rule: &str, message: String, location: &str) -> Violation {
    Violation {
        severity: Severity::Error,
        rule: rule.to_string(),
        message,
        location: Some(location.to_string()),
    }
}

/// The contract that OWNS the publish-manifest schema.
///
/// Located rather than hard-coded: the required-field lists live in
/// `contracts/publish-manifest-v1.yaml` §schema and are already copied once,
/// into `apr validate-manifest`'s `REQUIRED_TOP`/`REQUIRED_PROVENANCE`. A
/// third copy here would be a third thing to drift. Reading the contract means
/// the field list pv checks IS the field list the contract declares.
const PUBLISH_MANIFEST_SCHEMA: &str = "publish-manifest-v1.yaml";

/// Search `path`'s directory and its two ancestors for the schema contract.
fn find_publish_manifest_schema(path: &Path) -> Option<PathBuf> {
    let mut dir = path.parent()?;
    for _ in 0..3 {
        let candidate = dir.join(PUBLISH_MANIFEST_SCHEMA);
        if candidate.is_file() {
            return Some(candidate);
        }
        dir = dir.parent()?;
    }
    None
}

/// The `required_fields` / `provenance_required_fields` lists the schema
/// contract declares.
fn schema_required_fields(schema_path: &Path) -> Option<(Vec<String>, Vec<String>)> {
    let text = std::fs::read_to_string(schema_path).ok()?;
    let doc: Value = serde_yaml::from_str(&text).ok()?;
    let schema = doc.get("schema")?;
    let read = |key: &str| -> Vec<String> {
        schema
            .get(key)
            .and_then(Value::as_sequence)
            .map(|items| {
                items
                    .iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default()
    };
    Some((read("required_fields"), read("provenance_required_fields")))
}

/// Validate a publish manifest against the schema its own contract declares
/// (rules PM-SHAPE-001..005).
///
/// This is the OFFLINE half of `apr validate-manifest`. The gates that need
/// bytes or a network — sha256 over the artifact (FALSIFY-PM-002), URL
/// liveness (PM-003), recipe hashing (PM-005), the per-format Poka-Yokes
/// (PM-007/008/009) — stay there, and pv does not pretend to have run them.
/// What pv can answer without either is whether the document is shaped like a
/// manifest at all, which is FALSIFY-PM-001 and rejection criterion RJ-PM-002.
#[must_use]
pub fn validate_publish_manifest(manifest: &Value, path: &Path) -> Vec<Violation> {
    let mut violations = Vec::new();
    let Some(top) = manifest.as_mapping() else {
        violations.push(violation(
            "PM-SHAPE-001",
            "publish manifest is not a YAML mapping".to_string(),
            "",
        ));
        return violations;
    };

    let Some(schema_path) = find_publish_manifest_schema(path) else {
        violations.push(violation(
            "PM-SHAPE-000",
            format!(
                "cannot locate {PUBLISH_MANIFEST_SCHEMA} near {} — the manifest's \
                 required-field list is declared there, so without it nothing about \
                 this manifest can be checked",
                path.display()
            ),
            "",
        ));
        return violations;
    };
    let Some((required_top, required_provenance)) = schema_required_fields(&schema_path) else {
        violations.push(violation(
            "PM-SHAPE-000",
            format!(
                "{} declares no `schema.required_fields` — the publish-manifest schema \
                 contract has lost the list this gate reads",
                schema_path.display()
            ),
            "",
        ));
        return violations;
    };

    check_required(top, &required_top, "PM-SHAPE-001", "", &mut violations);
    let provenance = top.get("provenance").and_then(Value::as_mapping);
    if let Some(provenance) = provenance {
        check_required(
            provenance,
            &required_provenance,
            "PM-SHAPE-002",
            "provenance.",
            &mut violations,
        );
    }
    check_sha256(top, &mut violations);
    check_size_bytes(top, &mut violations);
    check_artifact_url(top, &mut violations);
    violations
}

/// RJ-PM-002: "Any required field missing OR null OR empty-string" is a hard
/// fail. All three cases, not just the first — a `license:` present but empty
/// reduces provenance to uselessness exactly as a missing one does.
fn check_required(
    map: &Mapping,
    required: &[String],
    rule: &str,
    prefix: &str,
    violations: &mut Vec<Violation>,
) {
    for field in required {
        let why = match map.get(field.as_str()) {
            None => "missing",
            Some(Value::Null) => "null",
            Some(Value::String(s)) if s.trim().is_empty() => "an empty string",
            Some(_) => continue,
        };
        violations.push(violation(
            rule,
            format!(
                "required manifest field `{prefix}{field}` is {why} — \
                 publish-manifest-v1 RJ-PM-002: a manifest missing a required field \
                 cannot ship"
            ),
            &format!("{prefix}{field}"),
        ));
    }
}

/// PM-SHAPE-003: `sha256` is documented as a "64-char hex string, lowercase".
/// Case and length are load-bearing: the comparison downstream is a byte-exact
/// string equality, so an uppercase digest never matches the artifact it names.
fn check_sha256(top: &Mapping, violations: &mut Vec<Violation>) {
    let Some(Value::String(sha)) = top.get("sha256") else {
        return;
    };
    let ok = sha.len() == 64
        && sha
            .chars()
            .all(|c| c.is_ascii_digit() || matches!(c, 'a'..='f'));
    if !ok {
        violations.push(violation(
            "PM-SHAPE-003",
            format!(
                "sha256 {sha:?} is not 64 lowercase hex characters — the published-artifact \
                 check compares this string byte-for-byte, so any other spelling can never \
                 match the artifact it names"
            ),
            "sha256",
        ));
    }
}

/// PM-SHAPE-004: `size_bytes` is documented as a positive integer, and is the
/// value the content-length check compares against.
fn check_size_bytes(top: &Mapping, violations: &mut Vec<Violation>) {
    let Some(value) = top.get("size_bytes") else {
        return;
    };
    if value.as_u64().is_some_and(|n| n > 0) {
        return;
    }
    violations.push(violation(
        "PM-SHAPE-004",
        format!(
            "size_bytes must be a positive integer, got {value:?} — it is what the \
             content-length check compares against, and a zero or non-integer makes \
             a truncated upload undetectable"
        ),
        "size_bytes",
    ));
}

/// PM-SHAPE-005: `artifact_url` must be an HTTPS URL — the schema's own
/// annotation is "HTTPS URL resolving to the binary", and a plaintext or
/// scheme-less URL cannot carry an integrity guarantee.
fn check_artifact_url(top: &Mapping, violations: &mut Vec<Violation>) {
    let Some(Value::String(url)) = top.get("artifact_url") else {
        return;
    };
    if url.starts_with("https://") && url.len() > "https://".len() {
        return;
    }
    violations.push(violation(
        "PM-SHAPE-005",
        format!(
            "artifact_url {url:?} is not an https:// URL — publish-manifest-v1 §schema \
             requires an HTTPS URL resolving to the binary"
        ),
        "artifact_url",
    ));
}

#[cfg(test)]
mod tests {
    include!("artifact_tests.rs");
}
