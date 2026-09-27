//! The packaged mirror of the thinking-budget table is byte-identical to its source (#3907).
//!
//! Same two-copy arrangement, and the same reason, as `capability_mirror.rs`:
//! `include_str!` cannot escape a crate directory at package time, so a published `apr`
//! can only embed a file inside `apr-cli`; but `pv lint contracts/` reads the workspace
//! root only. One copy is linted, the other is packaged, and this test is what makes the
//! pair safe:
//!
//!   SOURCE  contracts/thinking-budgets-v1.yaml                   linted by `pv lint`
//!   MIRROR  crates/apr-cli/contracts/thinking-budgets-v1.yaml    embedded in the binary
//!
//! Without it the two are a SHADOW — CLAUDE.md Verification Discipline #8, "a shadowed
//! artifact is worse than a missing one". A measured budget added to the linted source
//! would look landed and change nothing the gate reads, which for a FAIL-CLOSED table is
//! the worst direction: the model stays refused and the edit looks done.

//! #4129: the SOURCE lives outside this crate, so it is read at RUN time (an `include_str!` of it
//! cannot compile from the published apr-cli tarball). The MIRROR is inside the crate and ships.
//! In tree a missing source FAILS; out of tree the comparison SKIPs by name. The anti-vacuity
//! test below reads the MIRROR, which the comparison proves identical in tree, so it still runs
//! from the tarball.

const MIRROR: &str = include_str!("../contracts/thinking-budgets-v1.yaml");

/// The linted source, or `None` (after naming the skip) when out of tree.
fn source_or_skip(test: &str) -> Option<String> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    if !root.join("contracts").is_dir() {
        eprintln!(
            "SKIP {test}: out of tree (no {} beside this crate) - the linted source lives in the \
             workspace, which a published crate does not carry (#4129)",
            root.join("contracts").display()
        );
        return None;
    }
    let path = root.join("contracts/thinking-budgets-v1.yaml");
    Some(
        std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("in tree, {} must be readable: {e}", path.display())),
    )
}

#[test]
fn the_packaged_mirror_is_byte_identical_to_the_linted_source() {
    let Some(source) = source_or_skip("the_packaged_mirror_is_byte_identical_to_the_linted_source")
    else {
        return;
    };
    assert_eq!(
        source, MIRROR,
        "contracts/thinking-budgets-v1.yaml and crates/apr-cli/contracts/thinking-budgets-v1.yaml \
         have diverged. The linted copy is the source; copy it over the mirror:\n  \
         cp contracts/thinking-budgets-v1.yaml crates/apr-cli/contracts/thinking-budgets-v1.yaml"
    );
}

/// The mirror test above is satisfied by two EMPTY files. This is the anti-vacuity half:
/// the table must actually carry the things the gate refuses without.
#[test]
fn the_table_declares_a_default_with_a_basis_and_at_least_one_model() {
    let doc: serde_yaml::Value = serde_yaml::from_str(MIRROR).expect("mirror parses");
    let default = doc.get("default").expect("a `default` section");
    assert!(
        default
            .get("budget")
            .and_then(serde_yaml::Value::as_u64)
            .is_some(),
        "`default` must declare a budget"
    );
    let basis = default.get("basis").and_then(|b| b.as_str()).unwrap_or("");
    assert!(
        basis.trim().len() > 40,
        "`default.basis` must state the measurement behind the number, not a word: {basis:?}"
    );
    let models = doc
        .get("models")
        .and_then(serde_yaml::Value::as_mapping)
        .expect("a `models` mapping");
    assert!(
        !models.is_empty(),
        "an empty `models` map means every model silently takes the default — the defect \
         #3907 exists to end"
    );
    // Every listed model either has a budget WITH a basis, or states why it is unmeasured.
    // A bare entry with neither is a silent refusal nobody can act on.
    for (k, v) in models {
        let name = k.as_str().unwrap_or("<non-string key>");
        let has_budget = v
            .get("budget")
            .and_then(serde_yaml::Value::as_u64)
            .is_some();
        let has_basis = v
            .get("basis")
            .and_then(|b| b.as_str())
            .is_some_and(|b| b.trim().len() > 20);
        let has_why = v
            .get("why_unmeasured")
            .and_then(|b| b.as_str())
            .is_some_and(|b| b.trim().len() > 20);
        assert!(
            (has_budget && has_basis) || (!has_budget && has_why),
            "`{name}`: a listed model needs either a budget WITH a basis, or no budget and a \
             `why_unmeasured` saying what is outstanding"
        );
    }
}
