//! Port of `scripts/coverage_report_scope.py` (#4023): print the explicit `-p <crate>`
//! list that scopes a `cargo llvm-cov report`, derived from `cargo metadata --no-deps`.
//!
//! Every workspace member is listed, sorted by name, minus each `--exclude`. An
//! `--exclude` that names no member, or an empty result, is refused.

use crate::pystr::py_list_repr;
use serde_json::Value;
use std::collections::BTreeSet;

/// The `-p` line (with its `\n`), or the refusal message.
///
/// # Errors
/// Metadata that is not JSON or has a package without a string `name`; an `--exclude`
/// naming no member; nothing left to report.
pub fn scope(metadata: &str, exclude: &[String]) -> Result<String, String> {
    let meta: Value =
        serde_json::from_str(metadata).map_err(|e| format!("metadata is not JSON: {e}"))?;
    let names = meta
        .get("packages")
        .and_then(Value::as_array)
        .ok_or("metadata has no `packages` list")?
        .iter()
        .map(|p| {
            p.get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| "a package has no string `name`".to_owned())
        })
        .collect::<Result<Vec<&str>, String>>()?;
    let exclude: BTreeSet<&str> = exclude.iter().map(String::as_str).collect();
    let members: BTreeSet<&str> = names.iter().copied().collect();
    let unknown: Vec<&str> = exclude.difference(&members).copied().collect();
    if !unknown.is_empty() {
        return Err(format!(
            "coverage_report_scope: --exclude names no workspace member: {}",
            py_list_repr(&unknown)
        ));
    }
    let mut kept: Vec<&str> = names.into_iter().filter(|n| !exclude.contains(n)).collect();
    kept.sort_unstable();
    if kept.is_empty() {
        return Err("coverage_report_scope: no workspace members left to report".to_owned());
    }
    let parts: Vec<String> = kept.iter().map(|n| format!("-p {n}")).collect();
    Ok(format!("{}\n", parts.join(" ")))
}

#[cfg(test)]
mod tests {
    use super::scope;

    const META: &str =
        r#"{"packages":[{"name":"b"},{"name":"aprender-gpu"},{"name":"a-z"},{"name":"A"}]}"#;

    fn ex(names: &[&str]) -> Vec<String> {
        names.iter().map(|s| (*s).to_owned()).collect()
    }

    /// The case table of handoff PY-INV port 3.
    #[test]
    fn case_table() {
        let ok = [
            (ex(&[]), "-p A -p a-z -p aprender-gpu -p b\n"),
            (ex(&["aprender-gpu"]), "-p A -p a-z -p b\n"),
            (ex(&["aprender-gpu", "b"]), "-p A -p a-z\n"),
            (ex(&["b", "b"]), "-p A -p a-z -p aprender-gpu\n"),
        ];
        for (exclude, want) in ok {
            assert_eq!(scope(META, &exclude).as_deref(), Ok(want), "{exclude:?}");
        }
        assert_eq!(
            scope(META, &ex(&["nope", ""])),
            Err("coverage_report_scope: --exclude names no workspace member: ['', 'nope']".into())
        );
        assert_eq!(
            scope(META, &ex(&["A", "a-z", "aprender-gpu", "b"])),
            Err("coverage_report_scope: no workspace members left to report".into())
        );
        assert!(scope("{", &[]).is_err());
        assert!(scope(r#"{"packages":[{}]}"#, &[]).is_err());
    }
}
