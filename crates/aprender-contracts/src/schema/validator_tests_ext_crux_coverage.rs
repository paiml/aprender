//! EXT-25 (aprender#4407): every EXT-001 surface has a CRUX contract.
//!
//! `dogfood-model-lifecycle-v1.yaml` §`crux_surfaces` lists the surfaces the
//! spec creates or depends on (C1..C7) and the contracts that cover them. This
//! is FALSIFY-EXT-019: a surface with no contract, a listed contract that is
//! missing, one that declares `crux_coverage: none`, or a verb absent from the
//! CLI registry is a violation, and the live tree must have none.
//!
//! `Contract` is not `deny_unknown_fields`, so `crux_surfaces` and the crux
//! `metadata.surface`/`crux_coverage`/`gap_effect` keys are dropped by serde.
//! They are read here from the raw YAML, and the contracts are read at run
//! time so that a deleted file names its surface instead of breaking the build.

use crate::schema::validator::Severity;
use crate::schema::{parse_contract_str, validate_contract};
use serde_yaml::Value;
use std::collections::BTreeSet;
use std::path::Path;

const DOGFOOD: &str = include_str!("../../../../contracts/dogfood-model-lifecycle-v1.yaml");
const CLI_REGISTRY: &str = include_str!("../../../../contracts/apr-cli-commands-v1.yaml");

/// EXT-001 §10, rows C1..C7.
const EXT_SURFACES: [&str; 7] = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"];

/// The review-receipt vocabulary (`scripts/check_pr_review_receipt.sh`).
const GAP_EFFECTS: [&str; 3] = ["closes", "widens", "none"];

fn str_list(v: &Value, key: &str) -> Vec<String> {
    v.get(key)
        .and_then(Value::as_sequence)
        .map(|s| s.iter().filter_map(Value::as_str).map(str::to_string).collect())
        .unwrap_or_default()
}

fn meta_str<'a>(doc: &'a Value, key: &str) -> Option<&'a str> {
    doc.get("metadata")?.get(key)?.as_str()
}

/// Every reason the surface registry fails R-14. `read(name)` returns the
/// text of `contracts/<name>`, or `None` if there is no such file.
fn coverage_violations(
    dogfood: &str,
    read: &dyn Fn(&str) -> Option<String>,
    cli_verbs: &BTreeSet<String>,
    crux_p_files: &BTreeSet<String>,
) -> Vec<String> {
    let mut bad = Vec::new();
    let doc: Value = match serde_yaml::from_str(dogfood) {
        Ok(d) => d,
        Err(e) => return vec![format!("dogfood contract does not parse: {e}")],
    };
    let surfaces = doc.get("crux_surfaces").and_then(Value::as_sequence).cloned().unwrap_or_default();
    let ids: Vec<&str> = surfaces.iter().filter_map(|s| s.get("id")?.as_str()).collect();
    for want in EXT_SURFACES {
        if !ids.contains(&want) {
            bad.push(format!("{want}: EXT surface has no crux_surfaces entry"));
        }
    }

    let mut listed = BTreeSet::new();
    for s in &surfaces {
        let id = s.get("id").and_then(Value::as_str).unwrap_or("?");
        let surface = s.get("surface").and_then(Value::as_str).unwrap_or("");
        for verb in str_list(s, "verbs") {
            if !cli_verbs.contains(&verb) {
                bad.push(format!("{id}: verb `{verb}` is not in apr-cli-commands-v1"));
            }
        }
        let owned = str_list(s, "contracts");
        let extended = str_list(s, "extends");
        if owned.is_empty() && extended.is_empty() {
            bad.push(format!("{id} `{surface}`: no CRUX contract"));
        }
        for name in &extended {
            match read(name).map(|t| parse_contract_str(&t)) {
                None => bad.push(format!("{id}: extended contract {name} is missing")),
                Some(Err(e)) => bad.push(format!("{id}: {name} does not parse: {e}")),
                Some(Ok(_)) => {}
            }
        }
        for name in &owned {
            listed.insert(name.clone());
            let Some(text) = read(name) else {
                bad.push(format!("{id} `{surface}`: contract {name} is missing"));
                continue;
            };
            bad.extend(owned_contract_violations(id, surface, name, &text));
        }
    }
    for orphan in crux_p_files.difference(&listed) {
        bad.push(format!("{orphan}: category-P contract covers no listed EXT surface"));
    }
    bad
}

fn owned_contract_violations(id: &str, surface: &str, name: &str, text: &str) -> Vec<String> {
    let mut bad = Vec::new();
    let raw: Value = match serde_yaml::from_str(text) {
        Ok(v) => v,
        Err(e) => return vec![format!("{id}: {name} is not YAML: {e}")],
    };
    let checks = [
        ("ext_surface", Some(id)),
        ("surface", Some(surface)),
        ("crux_coverage", Some("covered")),
    ];
    for (key, want) in checks {
        let got = meta_str(&raw, key);
        if got != want {
            bad.push(format!("{id}: {name} metadata.{key} is {got:?}, want {want:?}"));
        }
    }
    match meta_str(&raw, "gap_effect") {
        Some(g) if GAP_EFFECTS.contains(&g) => {}
        other => bad.push(format!("{id}: {name} gap_effect {other:?} not in {GAP_EFFECTS:?}")),
    }
    if meta_str(&raw, "gap_note").is_none_or(str::is_empty) {
        bad.push(format!("{id}: {name} gap_effect carries no gap_note"));
    }
    match parse_contract_str(text) {
        Err(e) => bad.push(format!("{id}: {name} does not parse: {e}")),
        Ok(c) => {
            if c.proof_obligations.is_empty() {
                bad.push(format!("{id}: {name} has no obligations"));
            }
            if c.proof_obligations.iter().any(|o| o.id.is_none()) {
                bad.push(format!("{id}: {name} has an anonymous obligation"));
            }
            for v in validate_contract(&c) {
                if v.severity == Severity::Error {
                    bad.push(format!("{id}: {name} {}: {}", v.rule, v.message));
                }
            }
        }
    }
    bad
}

fn contracts_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../contracts")
}

fn read_live(name: &str) -> Option<String> {
    std::fs::read_to_string(contracts_dir().join(name)).ok()
}

fn live_crux_p_files() -> BTreeSet<String> {
    std::fs::read_dir(contracts_dir())
        .expect("contracts/ is readable")
        .filter_map(std::result::Result::ok)
        .filter_map(|e| e.file_name().into_string().ok())
        .filter(|n| n.starts_with("crux-P-") && n.ends_with(".yaml"))
        .collect()
}

fn cli_verbs() -> BTreeSet<String> {
    let doc: Value = serde_yaml::from_str(CLI_REGISTRY).expect("CLI registry parses");
    doc.get("commands")
        .and_then(Value::as_sequence)
        .expect("CLI registry has commands")
        .iter()
        .filter_map(|c| c.get("name")?.as_str().map(str::to_string))
        .collect()
}

/// FALSIFY-EXT-019: the live tree is covered, and each planted gap is RED.
#[test]
fn falsify_ext_019_surface_without_contract_red() {
    let verbs = cli_verbs();
    let files = live_crux_p_files();
    let live = coverage_violations(DOGFOOD, &read_live, &verbs, &files);
    assert!(live.is_empty(), "EXT surfaces not covered:\n{}", live.join("\n"));

    // Deleting one contract: RED, naming the surface.
    let deleted = |n: &str| (n != "crux-P-04-v1.yaml").then(|| read_live(n)).flatten();
    let v = coverage_violations(DOGFOOD, &deleted, &verbs, &files);
    assert!(v.iter().any(|m| m.starts_with("C5 `tracking`: contract crux-P-04")), "{v:?}");

    // `crux_coverage: none` on an EXT surface: RED.
    let uncovered = |n: &str| {
        read_live(n).map(|t| {
            if n == "crux-P-01-v1.yaml" {
                t.replace("crux_coverage: covered", "crux_coverage: none")
            } else {
                t
            }
        })
    };
    let v = coverage_violations(DOGFOOD, &uncovered, &verbs, &files);
    assert!(v.iter().any(|m| m.contains("metadata.crux_coverage is Some(\"none\")")), "{v:?}");

    // A surface listed with no contract, and one dropped from the list: RED.
    let no_contract = DOGFOOD.replace("contracts: [crux-P-06-v1.yaml]", "contracts: []");
    let v = coverage_violations(&no_contract, &read_live, &verbs, &files);
    assert!(v.iter().any(|m| m == "C7 `decision`: no CRUX contract"), "{v:?}");
    assert!(v.iter().any(|m| m.starts_with("crux-P-06-v1.yaml: category-P")), "{v:?}");

    let dropped = DOGFOOD.replace("- {id: C3,", "- {id: C3-retired,");
    let v = coverage_violations(&dropped, &read_live, &verbs, &files);
    assert!(v.iter().any(|m| m == "C3: EXT surface has no crux_surfaces entry"), "{v:?}");
}

/// A verb the CLI does not have, and an anonymous obligation, are RED.
#[test]
fn ext_crux_coverage_rejects_unknown_verbs_and_anonymous_obligations() {
    let verbs = cli_verbs();
    let files = live_crux_p_files();
    let ghost = DOGFOOD.replace("verbs: [eval]", "verbs: [eval, promote-magic]");
    let v = coverage_violations(&ghost, &read_live, &verbs, &files);
    assert!(v.iter().any(|m| m.contains("`promote-magic` is not in")), "{v:?}");

    let anon = |n: &str| {
        read_live(n).map(|t| {
            if n == "crux-P-02-v1.yaml" {
                t.replace("- id: CRUX-P-02-OB-01\n  type:", "- type:")
            } else {
                t
            }
        })
    };
    let v = coverage_violations(DOGFOOD, &anon, &verbs, &files);
    assert!(v.iter().any(|m| m.contains("anonymous obligation")), "{v:?}");
}
