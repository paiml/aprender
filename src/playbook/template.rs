//! Template resolution for playbook commands (PB-001)
//!
//! Handles `{{params.key}}`, `{{deps[N].path}}`, `{{outs[N].path}}` substitution.

use super::types::{yaml_value_to_string, Dependency, Output};
use anyhow::{bail, Result};
use std::collections::HashMap;

/// Resolve all template variables in a command string
///
/// Uses UTF-8-safe string scanning (no byte-level char casting).
pub fn resolve_template(
    cmd: &str,
    global_params: &HashMap<String, serde_yaml_ng::Value>,
    _stage_param_keys: &Option<Vec<String>>,
    deps: &[Dependency],
    outs: &[Output],
) -> Result<String> {
    let mut result = String::with_capacity(cmd.len());
    let mut pos = 0;

    while pos < cmd.len() {
        if cmd[pos..].starts_with("{{") {
            let start = pos + 2;
            if let Some(end_offset) = cmd[start..].find("}}") {
                let ref_str = cmd[start..start + end_offset].trim();
                let replacement = resolve_ref(ref_str, global_params, deps, outs)?;
                result.push_str(&replacement);
                pos = start + end_offset + 2;
            } else {
                bail!("unclosed template expression at position {}", pos);
            }
        } else {
            // UTF-8-safe: advance by one character
            let ch = cmd[pos..].chars().next().expect("iterator empty");
            result.push(ch);
            pos += ch.len_utf8();
        }
    }

    Ok(result)
}

fn resolve_ref(
    ref_str: &str,
    global_params: &HashMap<String, serde_yaml_ng::Value>,
    deps: &[Dependency],
    outs: &[Output],
) -> Result<String> {
    // {{params.key}} — resolved from global params
    if let Some(key) = ref_str.strip_prefix("params.") {
        if let Some(val) = global_params.get(key) {
            return Ok(yaml_value_to_string(val));
        }
        bail!("undefined param '{}'", key);
    }

    // {{deps[N].path}}
    if let Some(idx_str) = ref_str.strip_prefix("deps[").and_then(|s| s.strip_suffix("].path")) {
        let idx: usize =
            idx_str.parse().map_err(|_| anyhow::anyhow!("invalid deps index '{}'", idx_str))?;
        if idx >= deps.len() {
            bail!("deps[{}] out of range (only {} deps)", idx, deps.len());
        }
        return Ok(deps[idx].path.clone());
    }

    // {{outs[N].path}}
    if let Some(idx_str) = ref_str.strip_prefix("outs[").and_then(|s| s.strip_suffix("].path")) {
        let idx: usize =
            idx_str.parse().map_err(|_| anyhow::anyhow!("invalid outs index '{}'", idx_str))?;
        if idx >= outs.len() {
            bail!("outs[{}] out of range (only {} outs)", idx, outs.len());
        }
        return Ok(outs[idx].path.clone());
    }

    bail!("unknown template reference '{}'", ref_str);
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    fn make_params(pairs: &[(&str, &str)]) -> HashMap<String, serde_yaml_ng::Value> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), serde_yaml_ng::Value::String(v.to_string())))
            .collect()
    }

    fn make_deps(paths: &[&str]) -> Vec<Dependency> {
        paths.iter().map(|p| Dependency { path: p.to_string(), dep_type: None }).collect()
    }

    fn make_outs(paths: &[&str]) -> Vec<Output> {
        paths.iter().map(|p| Output { path: p.to_string(), out_type: None, remote: None }).collect()
    }

    #[test]
    fn test_PB001_param_substitution() {
        let global = make_params(&[("model", "whisper-base")]);
        let result = resolve_template("run --model {{params.model}}", &global, &None, &[], &[])
            .expect("unexpected failure");
        assert_eq!(result, "run --model whisper-base");
    }

    #[test]
    fn test_PB001_numeric_param_substitution() {
        let mut global = HashMap::new();
        global.insert(
            "chunk_size".to_string(),
            serde_yaml_ng::Value::Number(serde_yaml_ng::Number::from(512)),
        );
        let result =
            resolve_template("split --size {{params.chunk_size}}", &global, &None, &[], &[])
                .expect("unexpected failure");
        assert_eq!(result, "split --size 512");
    }

    #[test]
    fn test_PB001_deps_path_ref() {
        let deps = make_deps(&["/data/input.wav", "/data/config.json"]);
        let result = resolve_template(
            "cat {{deps[0].path}} {{deps[1].path}}",
            &HashMap::new(),
            &None,
            &deps,
            &[],
        )
        .expect("unexpected failure");
        assert_eq!(result, "cat /data/input.wav /data/config.json");
    }

    #[test]
    fn test_PB001_outs_path_ref() {
        let outs = make_outs(&["/tmp/output.txt"]);
        let result =
            resolve_template("echo hello > {{outs[0].path}}", &HashMap::new(), &None, &[], &outs)
                .expect("unexpected failure");
        assert_eq!(result, "echo hello > /tmp/output.txt");
    }

    #[test]
    fn test_PB001_multiple_substitutions() {
        let global = make_params(&[("model", "base"), ("lang", "en")]);
        let deps = make_deps(&["/input.wav"]);
        let outs = make_outs(&["/output.txt"]);
        let result = resolve_template(
            "transcribe --model {{params.model}} --lang {{params.lang}} {{deps[0].path}} > {{outs[0].path}}",
            &global, &None, &deps, &outs,
        ).expect("unexpected failure");
        assert_eq!(result, "transcribe --model base --lang en /input.wav > /output.txt");
    }

    #[test]
    fn test_PB001_no_templates() {
        let result = resolve_template("echo hello world", &HashMap::new(), &None, &[], &[])
            .expect("unexpected failure");
        assert_eq!(result, "echo hello world");
    }

    #[test]
    fn test_PB001_missing_param_error() {
        let err = resolve_template("echo {{params.missing}}", &HashMap::new(), &None, &[], &[])
            .unwrap_err();
        assert!(err.to_string().contains("undefined param"));
    }

    #[test]
    fn test_PB001_deps_out_of_range() {
        let err =
            resolve_template("cat {{deps[5].path}}", &HashMap::new(), &None, &[], &[]).unwrap_err();
        assert!(err.to_string().contains("out of range"));
    }

    #[test]
    fn test_PB001_outs_out_of_range() {
        let err =
            resolve_template("cat {{outs[0].path}}", &HashMap::new(), &None, &[], &[]).unwrap_err();
        assert!(err.to_string().contains("out of range"));
    }

    #[test]
    fn test_PB001_unclosed_template() {
        let err =
            resolve_template("echo {{params.model", &HashMap::new(), &None, &[], &[]).unwrap_err();
        assert!(err.to_string().contains("unclosed"));
    }

    #[test]
    fn test_PB001_whitespace_in_template() {
        let global = make_params(&[("name", "world")]);
        let result = resolve_template("echo {{ params.name }}", &global, &None, &[], &[])
            .expect("unexpected failure");
        assert_eq!(result, "echo world");
    }

    #[test]
    fn test_PB001_unicode_safe() {
        let global = make_params(&[("name", "héllo")]);
        let result = resolve_template("echo {{params.name}} — résumé", &global, &None, &[], &[])
            .expect("unexpected failure");
        assert_eq!(result, "echo héllo — résumé");
    }
}
