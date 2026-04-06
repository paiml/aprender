//! Playbook YAML parsing and structural validation (PB-001)
//!
//! Parses YAML into typed `Playbook` and validates structural constraints:
//! - version must be "1.0"
//! - stages must have non-empty cmd
//! - after references must point to valid stage names
//! - param template references must resolve

use super::types::*;
use anyhow::{bail, Context, Result};
use std::path::Path;

/// Parse a playbook from a YAML file path
pub fn parse_playbook_file(path: &Path) -> Result<Playbook> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    parse_playbook(&content).with_context(|| format!("failed to parse {}", path.display()))
}

/// Parse a playbook from a YAML string
pub fn parse_playbook(yaml: &str) -> Result<Playbook> {
    let pb: Playbook = serde_yaml_ng::from_str(yaml).context("invalid playbook YAML")?;
    Ok(pb)
}

/// Validate a parsed playbook, returning warnings for non-fatal issues
pub fn validate_playbook(pb: &Playbook) -> Result<Vec<ValidationWarning>> {
    if pb.version != "1.0" {
        bail!("unsupported playbook version '{}', expected '1.0'", pb.version);
    }
    if pb.name.is_empty() {
        bail!("playbook name must not be empty");
    }
    if pb.stages.is_empty() {
        bail!("playbook must have at least one stage");
    }

    let mut warnings = Vec::new();
    for (name, stage) in &pb.stages {
        validate_stage(name, stage, pb, &mut warnings)?;
    }
    Ok(warnings)
}

/// Validate a single stage within a playbook.
fn validate_stage(
    name: &str,
    stage: &Stage,
    pb: &Playbook,
    warnings: &mut Vec<ValidationWarning>,
) -> Result<()> {
    if stage.cmd.trim().is_empty() {
        bail!("stage '{}' has empty cmd", name);
    }

    for after_ref in &stage.after {
        if !pb.stages.contains_key(after_ref) {
            bail!("stage '{}' references unknown stage '{}' in after", name, after_ref);
        }
        if after_ref == name {
            bail!("stage '{}' references itself in after", name);
        }
    }

    if let Some(target_ref) = &stage.target {
        if !pb.targets.contains_key(target_ref) && !target_ref.is_empty() {
            warnings.push(ValidationWarning {
                message: format!(
                    "stage '{}' references target '{}' which is not defined in targets",
                    name, target_ref
                ),
            });
        }
    }

    validate_template_refs(&stage.cmd, &pb.params, &stage.deps, &stage.outs)
        .with_context(|| format!("stage '{}' cmd template error", name))?;

    if stage.outs.is_empty() {
        warnings.push(ValidationWarning {
            message: format!("stage '{}' has no outputs — will always re-run (no cache key)", name),
        });
    }

    Ok(())
}

/// Validate a single template reference (e.g. "params.model", "deps[0].path")
fn validate_single_ref(
    ref_str: &str,
    global_params: &std::collections::HashMap<String, serde_yaml_ng::Value>,
    deps: &[Dependency],
    outs: &[Output],
) -> Result<()> {
    if let Some(key) = ref_str.strip_prefix("params.") {
        if !global_params.contains_key(key) {
            bail!("template references undefined param '{}'", key);
        }
    } else if let Some(idx_str) =
        ref_str.strip_prefix("deps[").and_then(|s| s.strip_suffix("].path"))
    {
        let idx: usize =
            idx_str.parse().with_context(|| format!("invalid deps index '{}'", idx_str))?;
        if idx >= deps.len() {
            bail!("template references deps[{}] but only {} deps defined", idx, deps.len());
        }
    } else if let Some(idx_str) =
        ref_str.strip_prefix("outs[").and_then(|s| s.strip_suffix("].path"))
    {
        let idx: usize =
            idx_str.parse().with_context(|| format!("invalid outs index '{}'", idx_str))?;
        if idx >= outs.len() {
            bail!("template references outs[{}] but only {} outs defined", idx, outs.len());
        }
    }
    Ok(())
}

/// Validate template references without resolving them (UTF-8 safe)
fn validate_template_refs(
    cmd: &str,
    global_params: &std::collections::HashMap<String, serde_yaml_ng::Value>,
    deps: &[Dependency],
    outs: &[Output],
) -> Result<()> {
    let mut pos = 0;

    while pos < cmd.len() {
        if cmd[pos..].starts_with("{{") {
            let start = pos + 2;
            if let Some(end_offset) = cmd[start..].find("}}") {
                let ref_str = cmd[start..start + end_offset].trim();
                validate_single_ref(ref_str, global_params, deps, outs)?;
                pos = start + end_offset + 2;
            } else {
                pos += 2;
            }
        } else {
            let ch = cmd[pos..].chars().next().expect("iterator empty");
            pos += ch.len_utf8();
        }
    }
    Ok(())
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    fn minimal_yaml() -> String {
        r#"
version: "1.0"
name: test
params: {}
targets: {}
stages:
  hello:
    cmd: "echo hello"
    deps: []
    outs:
      - path: /tmp/out.txt
policy:
  failure: stop_on_first
  validation: checksum
  lock_file: true
"#
        .to_string()
    }

    #[test]
    fn test_PB001_parse_valid_playbook() {
        let pb = parse_playbook(&minimal_yaml()).expect("unexpected failure");
        assert_eq!(pb.version, "1.0");
        assert_eq!(pb.name, "test");
        assert_eq!(pb.stages.len(), 1);
    }

    #[test]
    fn test_PB001_validate_valid_playbook() {
        let pb = parse_playbook(&minimal_yaml()).expect("unexpected failure");
        let warnings = validate_playbook(&pb).expect("unexpected failure");
        assert!(warnings.is_empty());
    }

    #[test]
    fn test_PB001_reject_bad_version() {
        let yaml = minimal_yaml().replace("\"1.0\"", "\"2.0\"");
        let pb = parse_playbook(&yaml).expect("unexpected failure");
        let err = validate_playbook(&pb).unwrap_err();
        assert!(err.to_string().contains("unsupported playbook version"));
    }

    #[test]
    fn test_PB001_reject_empty_name() {
        let yaml = minimal_yaml().replace("name: test", "name: \"\"");
        let pb = parse_playbook(&yaml).expect("unexpected failure");
        let err = validate_playbook(&pb).unwrap_err();
        assert!(err.to_string().contains("name must not be empty"));
    }

    #[test]
    fn test_PB001_reject_empty_cmd() {
        let yaml = minimal_yaml().replace("echo hello", "  ");
        let pb = parse_playbook(&yaml).expect("unexpected failure");
        let err = validate_playbook(&pb).unwrap_err();
        assert!(err.to_string().contains("empty cmd"));
    }

    #[test]
    fn test_PB001_reject_invalid_after_ref() {
        let yaml = r#"
version: "1.0"
name: test
params: {}
targets: {}
stages:
  hello:
    cmd: "echo hello"
    deps: []
    outs:
      - path: /tmp/out.txt
    after:
      - nonexistent
policy:
  failure: stop_on_first
  validation: checksum
  lock_file: true
"#;
        let pb = parse_playbook(yaml).expect("unexpected failure");
        let err = validate_playbook(&pb).unwrap_err();
        assert!(err.to_string().contains("unknown stage 'nonexistent'"));
    }

    #[test]
    fn test_PB001_reject_self_reference() {
        let yaml = r#"
version: "1.0"
name: test
params: {}
targets: {}
stages:
  hello:
    cmd: "echo hello"
    deps: []
    outs:
      - path: /tmp/out.txt
    after:
      - hello
policy:
  failure: stop_on_first
  validation: checksum
  lock_file: true
"#;
        let pb = parse_playbook(yaml).expect("unexpected failure");
        let err = validate_playbook(&pb).unwrap_err();
        assert!(err.to_string().contains("references itself"));
    }

    #[test]
    fn test_PB001_warn_missing_outs() {
        let yaml = r#"
version: "1.0"
name: test
params: {}
targets: {}
stages:
  hello:
    cmd: "echo hello"
    deps: []
    outs: []
policy:
  failure: stop_on_first
  validation: checksum
  lock_file: true
"#;
        let pb = parse_playbook(yaml).expect("unexpected failure");
        let warnings = validate_playbook(&pb).expect("unexpected failure");
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].message.contains("no outputs"));
    }

    #[test]
    fn test_PB001_reject_undefined_param_ref() {
        let yaml = r#"
version: "1.0"
name: test
params: {}
targets: {}
stages:
  hello:
    cmd: "echo {{params.missing_key}}"
    deps: []
    outs:
      - path: /tmp/out.txt
policy:
  failure: stop_on_first
  validation: checksum
  lock_file: true
"#;
        let pb = parse_playbook(yaml).expect("unexpected failure");
        let err = validate_playbook(&pb).unwrap_err();
        let msg = format!("{:#}", err);
        assert!(msg.contains("undefined param"), "error was: {}", msg);
    }

    #[test]
    fn test_PB001_accept_valid_param_ref() {
        let yaml = r#"
version: "1.0"
name: test
params:
  model: "base"
targets: {}
stages:
  hello:
    cmd: "echo {{params.model}}"
    deps: []
    outs:
      - path: /tmp/out.txt
policy:
  failure: stop_on_first
  validation: checksum
  lock_file: true
"#;
        let pb = parse_playbook(yaml).expect("unexpected failure");
        let warnings = validate_playbook(&pb).expect("unexpected failure");
        assert!(warnings.is_empty());
    }

    #[test]
    fn test_PB001_reject_out_of_range_deps_ref() {
        let yaml = r#"
version: "1.0"
name: test
params: {}
targets: {}
stages:
  hello:
    cmd: "cat {{deps[5].path}}"
    deps: []
    outs:
      - path: /tmp/out.txt
policy:
  failure: stop_on_first
  validation: checksum
  lock_file: true
"#;
        let pb = parse_playbook(yaml).expect("unexpected failure");
        let err = validate_playbook(&pb).unwrap_err();
        let msg = format!("{:#}", err);
        assert!(msg.contains("deps[5]"), "error was: {}", msg);
    }

    #[test]
    fn test_PB001_parse_invalid_yaml() {
        let err = parse_playbook("not: valid: yaml: [[[").unwrap_err();
        assert!(err.to_string().contains("invalid playbook YAML"));
    }

    #[test]
    fn test_PB001_multistage_playbook() {
        let yaml = r#"
version: "1.0"
name: multi
params:
  model: base
targets: {}
stages:
  extract:
    cmd: "extract --model {{params.model}}"
    deps:
      - path: /data/input.wav
    outs:
      - path: /data/audio.wav
  transcribe:
    cmd: "transcribe {{deps[0].path}}"
    deps:
      - path: /data/audio.wav
    outs:
      - path: /data/text.txt
    after:
      - extract
policy:
  failure: stop_on_first
  validation: checksum
  lock_file: true
"#;
        let pb = parse_playbook(yaml).expect("unexpected failure");
        let warnings = validate_playbook(&pb).expect("unexpected failure");
        assert!(warnings.is_empty());
        assert_eq!(pb.stages.len(), 2);
    }
}
