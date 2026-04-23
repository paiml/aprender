//! `apr-audit-ship-two` — structural enforcement of SHIP-TWO-001 spec claims.
//!
//! v1 scope (per `contracts/apr-audit-ship-two-v1.yaml` v1.0.0):
//! parse the first ```yaml fenced block that declares a top-level `status:`
//! key in `docs/specifications/aprender-train/ship-two-models-spec.md`,
//! then structurally verify three invariants:
//!
//!   (1) every `{ count, of, ids }` triple has `count == len(ids)`
//!   (2) every `{ count, of, ids }` triple has `count <= of`
//!   (3) every `count:` value is a non-negative integer
//!
//! Exit codes (also encoded in the contract):
//!   0 — all claims consistent
//!   1 — at least one drift finding reported on stderr
//!   2 — parse error (spec YAML block missing / malformed)
//!
//! Future increments (separate PRs, separate contract bumps):
//!   v1.1 — git merge-base --is-ancestor checks on named evidence commits
//!   v2.0 — cross-repo parity via `--include-albor`
//!
//! See:
//!   - contracts/apr-audit-ship-two-v1.yaml
//!   - docs/specifications/aprender-train/ship-two-models-spec.md §3 row #9
//!   - docs/specifications/aprender-train/ship-two-models-spec-audit.md §3.3

use anyhow::{bail, Context, Result};
use clap::Parser;
use serde_yaml::Value;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

/// Default spec path relative to the workspace root.
const DEFAULT_SPEC_PATH: &str = "docs/specifications/aprender-train/ship-two-models-spec.md";

#[derive(Parser, Debug)]
#[command(
    name = "apr-audit-ship-two",
    about = "Structural enforcement of SHIP-TWO-001 spec YAML status block",
    version
)]
struct Args {
    /// Path to the SHIP-TWO-001 spec (markdown file containing the YAML status block).
    #[arg(long, default_value = DEFAULT_SPEC_PATH)]
    spec: PathBuf,

    /// Exit non-zero on any drift (default). Pass --no-fail to only print.
    #[arg(long)]
    no_fail: bool,
}

fn main() -> ExitCode {
    let args = Args::parse();

    match audit(&args.spec) {
        Ok(findings) if findings.is_empty() => {
            println!("apr-audit-ship-two: all count/of/ids claims consistent ✓");
            ExitCode::SUCCESS
        }
        Ok(findings) => {
            eprintln!("apr-audit-ship-two: {} drift finding(s):", findings.len());
            for f in &findings {
                eprintln!("  {}", f);
            }
            if args.no_fail {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(1)
            }
        }
        Err(e) => {
            eprintln!("apr-audit-ship-two: parse error: {e:#}");
            ExitCode::from(2)
        }
    }
}

/// Audit the spec file. Returns a list of human-readable drift findings; empty
/// list means all claims are internally consistent.
fn audit(spec_path: &Path) -> Result<Vec<String>> {
    let markdown = std::fs::read_to_string(spec_path)
        .with_context(|| format!("read spec {}", spec_path.display()))?;
    let yaml_block = extract_first_status_yaml_block(&markdown)
        .with_context(|| "no ```yaml block with top-level `status:` key found in spec")?;
    let root: Value = serde_yaml::from_str(&yaml_block).context("parse status YAML block")?;
    let status = root
        .get("status")
        .context("parsed YAML has no `status:` top-level key")?;
    let mut findings = Vec::new();
    check_count_of_ids(status, "status", &mut findings);
    Ok(findings)
}

/// Find the first ```yaml fenced block whose body contains a top-level `status:`
/// key. Markdown is scanned line by line; only the first match is returned.
fn extract_first_status_yaml_block(markdown: &str) -> Result<String> {
    let mut in_yaml = false;
    let mut buf = String::new();
    for line in markdown.lines() {
        let trimmed = line.trim_start();
        if !in_yaml {
            if trimmed == "```yaml" {
                in_yaml = true;
                buf.clear();
            }
            continue;
        }
        if trimmed == "```" {
            if buf.lines().any(|l| l.starts_with("status:")) {
                return Ok(buf);
            }
            in_yaml = false;
            buf.clear();
            continue;
        }
        buf.push_str(line);
        buf.push('\n');
    }
    bail!("no ```yaml block containing `status:` found")
}

/// Recursively walk a YAML value; whenever a map has all three of `count`,
/// `of`, `ids`, verify the structural invariants (equations 1, 2, 3 in the
/// contract).
fn check_count_of_ids(value: &Value, path: &str, findings: &mut Vec<String>) {
    match value {
        Value::Mapping(map) => {
            let has_count = map.contains_key(Value::String("count".into()));
            let has_of = map.contains_key(Value::String("of".into()));
            let has_ids = map.contains_key(Value::String("ids".into()));
            if has_count && has_of && has_ids {
                check_triple(map, path, findings);
            }
            for (k, v) in map {
                let key_str = match k {
                    Value::String(s) => s.clone(),
                    other => format!("{:?}", other),
                };
                check_count_of_ids(v, &format!("{path}.{key_str}"), findings);
            }
        }
        Value::Sequence(seq) => {
            for (i, item) in seq.iter().enumerate() {
                check_count_of_ids(item, &format!("{path}[{i}]"), findings);
            }
        }
        _ => {}
    }
}

/// Check the three equations against one `{count, of, ids}` triple.
fn check_triple(map: &serde_yaml::Mapping, path: &str, findings: &mut Vec<String>) {
    let count = map.get(Value::String("count".into())).and_then(Value::as_u64);
    let of = map.get(Value::String("of".into())).and_then(Value::as_u64);
    let ids = map
        .get(Value::String("ids".into()))
        .and_then(Value::as_sequence);

    let (count, of, ids) = match (count, of, ids) {
        (Some(c), Some(o), Some(i)) => (c, o, i),
        _ => {
            findings.push(format!(
                "{path}: count/of/ids present but not parseable as (u64, u64, Sequence)"
            ));
            return;
        }
    };

    // Equation 1: count == len(ids)
    if count as usize != ids.len() {
        findings.push(format!(
            "{path}: count={} but ids.len()={} (equations.count_equals_ids_length violated)",
            count,
            ids.len()
        ));
    }
    // Equation 2: count <= of
    if count > of {
        findings.push(format!(
            "{path}: count={count} exceeds of={of} (equations.count_under_of violated)"
        ));
    }
    // Equation 3: count is a non-negative integer — guaranteed by u64 parse above.
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_tmp(content: &str) -> tempfile::NamedTempFile {
        use std::io::Write;
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f
    }

    #[test]
    fn extracts_yaml_block_with_status_key() {
        let md = "# header\n\npre-text\n\n```yaml\nstatus:\n  x: 1\n```\n\npost\n";
        let body = extract_first_status_yaml_block(md).unwrap();
        assert!(body.contains("status:"));
        assert!(body.contains("x: 1"));
    }

    #[test]
    fn skips_yaml_block_without_status_key() {
        let md = "```yaml\nfoo: bar\n```\n\n```yaml\nstatus:\n  x: 1\n```\n";
        let body = extract_first_status_yaml_block(md).unwrap();
        assert!(body.contains("status:"));
        assert!(!body.contains("foo: bar"));
    }

    #[test]
    fn errors_when_no_status_block() {
        let md = "```yaml\nfoo: bar\n```\n";
        assert!(extract_first_status_yaml_block(md).is_err());
    }

    #[test]
    fn clean_status_block_returns_no_findings() {
        let clean = "status:\n  group:\n    count: 3\n    of: 10\n    ids: [A, B, C]\n";
        let spec = format!("```yaml\n{clean}```\n");
        let tmp = write_tmp(&spec);
        let findings = audit(tmp.path()).unwrap();
        assert!(findings.is_empty(), "unexpected: {:?}", findings);
    }

    // FALSIFY-AUDIT-SHIP-TWO-001 — count mismatch detected
    #[test]
    fn falsify_001_detects_count_vs_ids_mismatch() {
        let bad = "status:\n  group:\n    count: 6\n    of: 10\n    ids: [A, B, C, D, E, F, G]\n";
        let spec = format!("```yaml\n{bad}```\n");
        let tmp = write_tmp(&spec);
        let findings = audit(tmp.path()).unwrap();
        assert_eq!(findings.len(), 1);
        assert!(findings[0].contains("count=6"));
        assert!(findings[0].contains("ids.len()=7"));
    }

    // FALSIFY-AUDIT-SHIP-TWO-002 — count > of detected
    #[test]
    fn falsify_002_detects_count_exceeds_of() {
        let bad = "status:\n  group:\n    count: 11\n    of: 10\n    ids: [A, B, C, D, E, F, G, H, I, J, K]\n";
        let spec = format!("```yaml\n{bad}```\n");
        let tmp = write_tmp(&spec);
        let findings = audit(tmp.path()).unwrap();
        assert_eq!(findings.len(), 1);
        assert!(findings[0].contains("count=11 exceeds of=10"));
    }

    // FALSIFY-AUDIT-SHIP-TWO-003 — current on-branch spec passes
    #[test]
    fn falsify_003_current_spec_is_clean() {
        // Walk up from CARGO_MANIFEST_DIR to find workspace root, then check
        // the real spec. This asserts the on-branch spec is internally
        // consistent — fails loud if someone edits the status: block into
        // drift without running the gate.
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let workspace_root = std::path::Path::new(manifest_dir)
            .parent()
            .expect("apr-cli parent")
            .parent()
            .expect("workspace root")
            .to_path_buf();
        let spec = workspace_root.join(DEFAULT_SPEC_PATH);
        if !spec.exists() {
            eprintln!("skipping falsify_003: spec not found at {}", spec.display());
            return;
        }
        let findings = audit(&spec).expect("spec parses");
        assert!(
            findings.is_empty(),
            "SHIP-TWO-001 spec has structural drift: {:?}",
            findings
        );
    }

    #[test]
    fn both_count_and_of_violations_reported() {
        // count=5 vs ids.len()=3 AND count=5 > of=2
        let bad = "status:\n  group:\n    count: 5\n    of: 2\n    ids: [A, B, C]\n";
        let spec = format!("```yaml\n{bad}```\n");
        let tmp = write_tmp(&spec);
        let findings = audit(tmp.path()).unwrap();
        assert_eq!(findings.len(), 2);
    }
}
