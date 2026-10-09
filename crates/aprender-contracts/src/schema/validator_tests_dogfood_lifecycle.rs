//! EXT-01 (aprender#4383): `dogfood-model-lifecycle-v1` binds every falsifier
//! to an obligation that exists.
//!
//! `pv validate` does not resolve `falsification_tests[].obligation`: a
//! citation of `EXT-INV-099` passed validate and lint clean (measured
//! 2026-09-25). These tests carry the check the tool does not make, for the
//! one contract whose row claims it.
//!
//! EXT-23 (aprender#4401): `dogfood_gate_paths`, the set I-12 recuses the apr
//! lane from, names only live paths and includes the contract itself.

use crate::schema::parse_contract_str;
use serde_yaml::Value;
use std::collections::BTreeSet;
use std::path::Path;

const DOGFOOD: &str = include_str!("../../../../contracts/dogfood-model-lifecycle-v1.yaml");

/// The spec's §4 invariants (I-1..I-14), all named, none anonymous.
#[test]
fn ext_obligations_are_i1_through_i14_all_named() {
    let c = parse_contract_str(DOGFOOD).expect("dogfood contract parses");
    let ids: Vec<&str> = c
        .proof_obligations
        .iter()
        .map(|o| o.id.as_deref().expect("no anonymous obligation"))
        .collect();
    let want: Vec<String> = (1..=14).map(|n| format!("EXT-INV-{n:03}")).collect();
    assert_eq!(ids, want);
}

/// The spec's §9 falsifiers (FALSIFY-EXT-001..024), each citing only
/// obligations that exist, together citing every obligation, each bound to a
/// test or owed by a row with its planned test.
#[test]
fn ext_falsifiers_cite_existing_obligations_and_cover_all() {
    let c = parse_contract_str(DOGFOOD).expect("dogfood contract parses");
    let ids: BTreeSet<&str> = c
        .proof_obligations
        .iter()
        .filter_map(|o| o.id.as_deref())
        .collect();
    let want: Vec<String> = (1..=24).map(|n| format!("FALSIFY-EXT-{n:03}")).collect();
    let got: Vec<&str> = c
        .falsification_tests
        .iter()
        .map(|f| f.id.as_str())
        .collect();
    assert_eq!(got, want);

    let mut cited = BTreeSet::new();
    for f in &c.falsification_tests {
        let citation = f
            .obligation
            .as_ref()
            .unwrap_or_else(|| panic!("{} cites no obligation", f.id));
        for t in citation.targets() {
            assert!(ids.contains(t), "{} cites unknown obligation {t}", f.id);
            cited.insert(t);
        }
        // Bound (`test:` names a runnable cargo test) XOR owed (the owing row
        // and the planned invocation are named, and nothing is cited yet).
        let owed = f.if_fails.contains("(owed EXT-");
        match f.test.as_deref() {
            Some(test) => {
                assert!(!owed, "{} is bound but still marked owed", f.id);
                assert!(test.starts_with("cargo test -p "), "{}: {test:?}", f.id);
            }
            None => assert!(
                // `ends_with(')')`: an unquoted ` #4384` is a YAML comment and
                // silently truncated every marker once (2026-09-25).
                owed && f.if_fails.contains("; planned: cargo test -p ")
                    && f.if_fails.ends_with(')'),
                "{} is neither bound nor owed with a planned test",
                f.id
            ),
        }
    }
    assert_eq!(cited, ids, "an obligation no falsifier discharges");
}

const SELF_PATH: &str = "contracts/dogfood-model-lifecycle-v1.yaml";

/// Every reason `dogfood_gate_paths` fails I-12. `matches(prefix)` says
/// whether any repo file starts with the prefix. The review lanes' recusal
/// filters read this list, so an entry that matches nothing recuses nothing.
fn gate_path_violations(dogfood: &str, matches: &dyn Fn(&str) -> bool) -> Vec<String> {
    let doc: Value = match serde_yaml::from_str(dogfood) {
        Ok(d) => d,
        Err(e) => return vec![format!("dogfood contract does not parse: {e}")],
    };
    let Some(entries) = doc.get("dogfood_gate_paths").and_then(Value::as_sequence) else {
        return vec!["no dogfood_gate_paths list".to_string()];
    };
    let mut bad = Vec::new();
    let mut names_self = false;
    for e in entries {
        match e.as_str() {
            Some(p) if !p.is_empty() && !p.starts_with('/') && !p.contains("..") => {
                names_self |= p == SELF_PATH;
                if !matches(p) {
                    bad.push(format!("{p}: matches no file"));
                }
            }
            _ => bad.push(format!("{e:?}: not a repo-relative path prefix")),
        }
    }
    if !names_self {
        bad.push(format!("{SELF_PATH} is not in its own recusal set"));
    }
    bad
}

/// A prefix matches when it names a file, a non-empty directory (trailing
/// `/`), or the start of a file name in its parent directory.
fn live_prefix_matches(prefix: &str) -> bool {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let path = root.join(prefix);
    if let Some(dir) = prefix.strip_suffix('/') {
        return std::fs::read_dir(root.join(dir)).is_ok_and(|mut d| d.next().is_some());
    }
    if path.is_file() {
        return true;
    }
    let (Some(parent), Some(stem)) = (path.parent(), path.file_name().and_then(|n| n.to_str()))
    else {
        return false;
    };
    std::fs::read_dir(parent).is_ok_and(|d| {
        d.filter_map(Result::ok)
            .any(|f| f.file_name().to_str().is_some_and(|n| n.starts_with(stem)))
    })
}

#[test]
fn ext_inv_012_gate_paths_are_live() {
    assert_eq!(
        gate_path_violations(DOGFOOD, &live_prefix_matches),
        Vec::<String>::new()
    );
}

/// Each plant must fail exactly once: a stranded entry (what a rename of the
/// gate code leaves behind), an absolute path, and the contract dropping
/// itself from the list.
#[test]
fn ext_inv_012_planted_gate_paths_red() {
    let head = "dogfood_gate_paths:\n";
    assert!(DOGFOOD.contains(head), "the list moved; re-plant");
    let stranded = DOGFOOD.replace(
        head,
        &format!("{head}- crates/apr-cli/src/commands/no_such_gate\n"),
    );
    let absolute = DOGFOOD.replace(head, &format!("{head}- /etc/passwd\n"));
    let self_line = format!("\n- {SELF_PATH}");
    assert!(
        DOGFOOD.contains(&self_line),
        "the self entry moved; re-plant"
    );
    let no_self = DOGFOOD.replacen(&self_line, "", 1);
    for (name, yaml, want) in [
        ("stranded", stranded, "no_such_gate: matches no file"),
        ("absolute", absolute, "not a repo-relative path prefix"),
        ("no_self", no_self, "is not in its own recusal set"),
    ] {
        let got = gate_path_violations(&yaml, &live_prefix_matches);
        assert_eq!(got.len(), 1, "{name}: {got:?}");
        assert!(got[0].contains(want), "{name}: {got:?}");
    }
}
