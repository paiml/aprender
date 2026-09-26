//! ONT-001 §5 ONT-4c (v4.14) — the MEASURED claim sets and their two ratchets.
//!
//! `lint-baseline.json` carries `readme.verified_commands[]`, `readme.withdrawn[]`, `claude_md.verified_commands[]`
//! and `claude_md.withdrawn[]`. They are measured, not declared: `pv lint` extracts the sets live and
//!
//! - **F-33** (freshness): the committed `verified_commands[]` must EQUAL the live extraction — a hand edit that
//!   disagrees, or a claim added or dropped without re-running `make ont-ratchet`, is RED naming the difference;
//! - **F-34** (withdrawal): `withdrawn(HEAD) \ withdrawn(comparand) == set(comparand) \ set(current)` against the
//!   merge-base baseline — a verified claim dropped with no new `withdrawn[]` entry is RED naming it, and so is a
//!   new entry for a claim that was not dropped. The ratchet is a SET: a duplicate of a dropped command elsewhere
//!   in the document keeps it in `set(current)`, so it cannot mask a drop and is not one.
//!
//! A key absent from a baseline reads as ∅ — which is what makes the first PR that adds the keys (bootstrap)
//! checkable rather than exempt: at its comparand every set is ∅, so its own `withdrawn[]` must be ∅ too.

use std::collections::BTreeSet;

/// The baseline keys, in report order: `readme` (`extract:readme`) and `claude_md` (`extract:llm-context`).
pub const KEYS: [&str; 2] = ["readme", "claude_md"];

/// The RED message's remedy, verbatim from the row.
pub const REMEDY: &str =
    "withdrawn[] disagrees with set(merge-base) \\ set(current): run make ont-ratchet (after a rebase, re-run it)";

/// One key's committed sets.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Entry {
    pub verified_commands: BTreeSet<String>,
    pub withdrawn: BTreeSet<String>,
}

/// `key`'s sets in a baseline document. No document, or no key, is ∅; a key that is there and is not
/// `{verified_commands: [string], withdrawn: [string]}` is an error (a committed value that cannot be read must
/// not read as empty — that would disarm the check for exactly that commit).
pub fn read(text: Option<&str>, key: &str) -> Result<Entry, String> {
    let Some(text) = text else {
        return Ok(Entry::default());
    };
    let doc: serde_json::Value =
        serde_json::from_str(text).map_err(|e| format!("lint-baseline.json: {e}"))?;
    let Some(v) = doc.get(key) else {
        return Ok(Entry::default());
    };
    let set = |field: &str| -> Result<BTreeSet<String>, String> {
        match v.get(field) {
            None => Ok(BTreeSet::new()),
            Some(serde_json::Value::Array(a)) => a
                .iter()
                .map(|x| {
                    x.as_str().map(String::from).ok_or_else(|| {
                        format!("lint-baseline.json {key}.{field}[] holds a non-string")
                    })
                })
                .collect(),
            Some(_) => Err(format!("lint-baseline.json {key}.{field} is not an array")),
        }
    };
    Ok(Entry {
        verified_commands: set("verified_commands")?,
        withdrawn: set("withdrawn")?,
    })
}

/// F-33: the committed set against the live one. One message per difference, naming the command.
#[must_use]
pub fn freshness(key: &str, committed: &Entry, live: &BTreeSet<String>) -> Vec<String> {
    let mut out = Vec::new();
    for c in live.difference(&committed.verified_commands) {
        out.push(format!(
            "F-33: {key}.verified_commands[] in lint-baseline.json lacks the live claim `{c}` — run make ont-ratchet"
        ));
    }
    for c in committed.verified_commands.difference(live) {
        out.push(format!(
            "F-33: {key}.verified_commands[] in lint-baseline.json carries `{c}`, which the live extraction does not — run make ont-ratchet"
        ));
    }
    out
}

/// F-34: `withdrawn(head) \ withdrawn(comparand)` against `set(comparand) \ set(live)`. One message per
/// difference, naming the command.
#[must_use]
pub fn withdrawal(
    key: &str,
    head: &Entry,
    comparand: &Entry,
    live: &BTreeSet<String>,
) -> Vec<String> {
    let new_withdrawn: BTreeSet<&String> =
        head.withdrawn.difference(&comparand.withdrawn).collect();
    let dropped: BTreeSet<&String> = comparand.verified_commands.difference(live).collect();
    let mut out = Vec::new();
    for c in dropped.difference(&new_withdrawn) {
        out.push(format!(
            "F-34: {key}: verified claim `{c}` was dropped with no new withdrawn[] entry — {REMEDY}"
        ));
    }
    for c in new_withdrawn.difference(&dropped) {
        out.push(format!(
            "F-34: {key}: withdrawn[] names `{c}`, which was not dropped — {REMEDY}"
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set(xs: &[&str]) -> BTreeSet<String> {
        xs.iter().map(|s| (*s).to_string()).collect()
    }

    fn entry(v: &[&str], w: &[&str]) -> Entry {
        Entry {
            verified_commands: set(v),
            withdrawn: set(w),
        }
    }

    #[test]
    fn an_absent_document_or_key_reads_as_empty_and_a_malformed_key_is_an_error() {
        assert_eq!(read(None, "readme").unwrap(), Entry::default());
        assert_eq!(read(Some("{}"), "readme").unwrap(), Entry::default());
        let e = read(
            Some(r#"{"readme":{"verified_commands":["a"],"withdrawn":["b"]}}"#),
            "readme",
        )
        .unwrap();
        assert_eq!(e, entry(&["a"], &["b"]));
        assert!(read(Some(r#"{"readme":{"verified_commands":"a"}}"#), "readme").is_err());
        assert!(read(Some(r#"{"readme":{"withdrawn":[1]}}"#), "readme").is_err());
    }

    #[test]
    fn f33_names_every_difference_in_both_directions() {
        let m = freshness("readme", &entry(&["a", "b"], &[]), &set(&["b", "c"]));
        assert_eq!(m.len(), 2);
        assert!(m[0].contains("lacks the live claim `c`"), "{m:?}");
        assert!(m[1].contains("carries `a`"), "{m:?}");
        assert!(freshness("readme", &entry(&["a"], &[]), &set(&["a"])).is_empty());
    }

    #[test]
    fn f34_a_drop_needs_a_new_withdrawal_and_a_withdrawal_needs_a_drop() {
        let cmp = entry(&["a", "b"], &["old"]);
        // dropped `b`, recorded
        assert!(withdrawal("readme", &entry(&["a"], &["old", "b"]), &cmp, &set(&["a"])).is_empty());
        // dropped `b`, not recorded
        let m = withdrawal("readme", &entry(&["a"], &["old"]), &cmp, &set(&["a"]));
        assert_eq!(m.len(), 1);
        assert!(
            m[0].contains("`b` was dropped") && m[0].contains(REMEDY),
            "{m:?}"
        );
        // withdrew `a`, which is still live
        let m = withdrawal(
            "readme",
            &entry(&["a", "b"], &["old", "a"]),
            &cmp,
            &set(&["a", "b"]),
        );
        assert!(m[0].contains("names `a`, which was not dropped"), "{m:?}");
    }

    #[test]
    fn f34_bootstrap_with_the_key_absent_at_the_comparand_is_checked_against_the_empty_set() {
        let cmp = read(Some("{}"), "readme").unwrap();
        assert!(withdrawal("readme", &entry(&["a"], &[]), &cmp, &set(&["a"])).is_empty());
        assert_eq!(
            withdrawal("readme", &entry(&["a"], &["x"]), &cmp, &set(&["a"])).len(),
            1
        );
    }
}
