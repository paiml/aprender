//! Port of `scripts/lib/publishable_crates.py`: read `cargo metadata --no-deps
//! --format-version 1` and print `<package name>\t<crate dir>` for every package whose
//! `publish` is not `[]`. Only `[]` is skipped: an absent or `null` `publish`, or any
//! registry list (even one without crates-io), is printed, exactly as the original did.

use crate::pystr::py_dirname;
use serde_json::Value;

/// The output lines, each with its `\n`, or the reason the input is refused.
///
/// # Errors
/// Input that is not JSON, has no `packages` list, or has a package without a string
/// `name` or `manifest_path` (the original raised `KeyError`/`TypeError` on each).
pub fn run(metadata: &str) -> Result<String, String> {
    let meta: Value =
        serde_json::from_str(metadata).map_err(|e| format!("metadata is not JSON: {e}"))?;
    let packages = meta
        .get("packages")
        .and_then(Value::as_array)
        .ok_or("metadata has no `packages` list")?;
    let mut out = String::new();
    for pkg in packages {
        if pkg
            .get("publish")
            .and_then(Value::as_array)
            .is_some_and(Vec::is_empty)
        {
            continue;
        }
        let field = |k: &str| {
            pkg.get(k)
                .and_then(Value::as_str)
                .ok_or_else(|| format!("a package has no string `{k}`"))
        };
        out.push_str(field("name")?);
        out.push('\t');
        out.push_str(py_dirname(field("manifest_path")?));
        out.push('\n');
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::run;

    fn pkg(name: &str, publish: &str) -> String {
        let p = if publish.is_empty() {
            String::new()
        } else {
            format!(r#","publish":{publish}"#)
        };
        format!(r#"{{"name":"{name}","manifest_path":"/w/crates/{name}/Cargo.toml"{p}}}"#)
    }

    fn meta(pkgs: &[String]) -> String {
        format!(r#"{{"packages":[{}],"version":1}}"#, pkgs.join(","))
    }

    /// The case table of `docs`/handoff PY-INV port 1: every `publish` shape.
    #[test]
    fn publish_shapes() {
        let cases = [
            ("", true),
            ("null", true),
            ("[]", false),
            (r#"["crates-io"]"#, true),
            (r#"["other-registry"]"#, true),
            ("false", true),
        ];
        for (publish, printed) in cases {
            let got = run(&meta(&[pkg("a-b", publish)])).expect("valid metadata");
            let want = if printed { "a-b\t/w/crates/a-b\n" } else { "" };
            assert_eq!(got, want, "publish = {publish:?}");
        }
    }

    #[test]
    fn root_facade_and_order() {
        let root = r#"{"name":"aprender","manifest_path":"/w/Cargo.toml"}"#.to_owned();
        let got = run(&meta(&[pkg("z", ""), root, pkg("a", "[]")])).expect("valid");
        assert_eq!(
            got, "z\t/w/crates/z\naprender\t/w\n",
            "input order, root dir is /w"
        );
    }

    #[test]
    fn zero_packages_is_empty_and_ok() {
        assert_eq!(run(r#"{"packages":[]}"#), Ok(String::new()));
    }

    #[test]
    fn refusals() {
        for bad in [
            "",
            "{",
            "[]",
            r#"{"packages":{}}"#,
            r#"{"packages":[{"name":"x"}]}"#,
            r#"{"packages":[{"manifest_path":"/x/Cargo.toml"}]}"#,
        ] {
            assert!(run(bad).is_err(), "must refuse {bad:?}");
        }
    }
}
