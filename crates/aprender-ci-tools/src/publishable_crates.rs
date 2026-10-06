//! Port of `scripts/lib/publishable_crates.py`: read `cargo metadata --no-deps
//! --format-version 1` and print `<package name>\t<crate dir>` for every package whose
//! `publish` is not `[]`. Only `[]` is skipped: an absent or `null` `publish`, or any
//! registry list (even one without crates-io), is printed, exactly as the original did.

use crate::pystr::py_dirname;
use serde_json::Value;

/// The output lines, each with its `\n`, or `(printed, reason)` when the input is refused.
/// `printed` holds the lines of the packages before the bad one: the original printed
/// each line as it went, so it had already written them when it raised, and the caller
/// must write them too.
///
/// # Errors
/// Input that is not JSON, has no `packages` the original could iterate (a list, or an
/// empty dict or string, which yield nothing), or has a package without a string
/// `name` or `manifest_path` (the original raised `KeyError`/`TypeError` on each).
pub fn run(metadata: &str) -> Result<String, (String, String)> {
    let refuse = |reason: String| (String::new(), reason);
    let meta: Value =
        serde_json::from_str(metadata).map_err(|e| refuse(format!("metadata is not JSON: {e}")))?;
    // `for pkg in meta["packages"]` iterates whatever is there: an empty dict or string
    // runs zero times and succeeds; a non-empty one yields `str` items, which have no
    // `.get`, so it raises before printing anything — as does a missing key or a scalar.
    let packages = match meta.get("packages") {
        Some(Value::Array(packages)) => packages,
        Some(Value::Object(m)) if m.is_empty() => return Ok(String::new()),
        Some(Value::String(s)) if s.is_empty() => return Ok(String::new()),
        _ => {
            return Err(refuse(
                "metadata has no iterable `packages` of packages".to_string(),
            ))
        }
    };
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
        // Both fields before any byte of the line: the original built the whole f-string
        // before printing it, so a bad package prints nothing of its own line.
        let line = field("name")
            .and_then(|name| Ok(format!("{name}\t{}\n", py_dirname(field("manifest_path")?))));
        match line {
            Ok(line) => out.push_str(&line),
            Err(reason) => return Err((out, reason)),
        }
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
        // `{}` and `""` are as empty to the original's `for` loop as `[]`.
        for empty in [
            r#"{"packages":[]}"#,
            r#"{"packages":{}}"#,
            r#"{"packages":""}"#,
        ] {
            assert_eq!(run(empty), Ok(String::new()), "{empty}");
        }
    }

    /// A bad package after good ones: the good lines were already printed (the original
    /// printed as it went), and nothing of the bad package's own line is.
    #[test]
    fn refusal_keeps_the_lines_printed_before_it() {
        let bad_path = r#"{"name":"x"}"#.to_owned();
        let bad_name = r#"{"manifest_path":"/w/y/Cargo.toml"}"#.to_owned();
        for bad in [bad_path, bad_name] {
            let got = run(&meta(&[pkg("a", ""), pkg("b", "[]"), bad, pkg("c", "")]));
            let (printed, _) = got.expect_err("a bad package is refused");
            assert_eq!(printed, "a\t/w/crates/a\n");
        }
    }

    #[test]
    fn refusals() {
        for bad in [
            "",
            "{",
            "[]",
            r#"{"packages":{"a":1}}"#,
            r#"{"packages":"ab"}"#,
            r#"{"packages":null}"#,
            r#"{"packages":5}"#,
            r#"{"packages":[5]}"#,
            r#"{"packages":[{"name":"x"}]}"#,
            r#"{"packages":[{"manifest_path":"/x/Cargo.toml"}]}"#,
        ] {
            assert!(run(bad).is_err(), "must refuse {bad:?}");
        }
    }
}
