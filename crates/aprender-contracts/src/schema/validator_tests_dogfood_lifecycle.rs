//! EXT-01 (aprender#4383): `dogfood-model-lifecycle-v1` binds every falsifier
//! to an obligation that exists.
//!
//! `pv validate` does not resolve `falsification_tests[].obligation`: a
//! citation of `EXT-INV-099` passed validate and lint clean (measured
//! 2026-09-25). These tests carry the check the tool does not make, for the
//! one contract whose row claims it.

use crate::schema::parse_contract_str;
use std::collections::BTreeSet;

const DOGFOOD: &str =
    include_str!("../../../../contracts/dogfood-model-lifecycle-v1.yaml");

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
    let got: Vec<&str> = c.falsification_tests.iter().map(|f| f.id.as_str()).collect();
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
