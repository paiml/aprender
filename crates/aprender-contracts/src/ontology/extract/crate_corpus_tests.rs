//! #4538: Σ walks `crates/*/contracts`. Top level is authoritative; a crate copy that is byte-identical to another
//! copy is skipped; a crate stem with one distinct content is admitted; a crate copy with different bytes is refused
//! by name (PV-DUP-001) and never unioned — and a binding that names a refused copy is refused by name, never
//! quietly resolved to the top-level node with different content (infra-83's caveat).

use std::path::Path;

use super::*;
use crate::ontology::extract::code;
use crate::ontology::rdf::iri;

fn write(root: &Path, rel: &str, body: &str) {
    let p = root.join(rel);
    std::fs::create_dir_all(p.parent().expect("parent")).expect("mkdir");
    std::fs::write(p, body).expect("write");
}

fn contract(name: &str) -> String {
    format!("metadata:\n  version: 1.0.0\n  description: {name}\nname: {name}\n")
}

/// A repo with one real crate (`k`) and one manifest-less directory (`staging`).
fn fixture() -> tempfile::TempDir {
    let t = tempfile::tempdir().expect("tempdir");
    let r = t.path();
    write(r, "contracts/top-v1.yaml", &contract("top"));
    write(r, "contracts/same-v1.yaml", &contract("same"));
    write(r, "contracts/diff-v1.yaml", &contract("diff-top"));
    write(r, "crates/k/Cargo.toml", "[package]\nname = \"k\"\n");
    write(r, "crates/k/contracts/uniq-v1.yaml", &contract("uniq"));
    write(r, "crates/k/contracts/same-v1.yaml", &contract("same"));
    write(
        r,
        "crates/k/contracts/diff-v1.yaml",
        &contract("diff-crate"),
    );
    write(
        r,
        "crates/k/contracts/binding.yaml",
        "version: \"1.0.0\"\ntarget_crate: k\nbindings:\n\
         - contract: diff-v1.yaml\n  equation: e\n  module_path: k::m\n  function: f\n  status: implemented\n\
         - contract: uniq-v1.yaml\n  equation: e\n  module_path: k::m\n  function: g\n  status: implemented\n\
         - contract: ../../../contracts/top-v1.yaml\n  equation: e\n  module_path: k::m\n  function: h\n  status: implemented\n",
    );
    write(
        r,
        "crates/staging/contracts/ghost-v1.yaml",
        &contract("ghost"),
    );
    t
}

fn stems(dir: &Path) -> Vec<String> {
    documents(dir).into_iter().map(|(s, _, _)| s).collect()
}

#[test]
fn a_unique_crate_contract_is_admitted_with_its_crate_path() {
    let t = fixture();
    let docs = documents(&t.path().join("contracts"));
    let uniq = docs
        .iter()
        .find(|(s, _, _)| s == "uniq-v1")
        .expect("uniq admitted");
    assert_eq!(uniq.1, "crates/k/contracts/uniq-v1.yaml");
}

#[test]
fn an_identical_crate_copy_is_skipped_and_the_top_level_file_kept() {
    let t = fixture();
    let docs = documents(&t.path().join("contracts"));
    let same: Vec<_> = docs.iter().filter(|(s, _, _)| s == "same-v1").collect();
    assert_eq!(same.len(), 1, "{same:?}");
    assert_eq!(same[0].1, "contracts/same-v1.yaml");
    let refused = corpus(&t.path().join("contracts")).refused;
    assert!(!refused.iter().any(|r| r.stem == "same-v1"), "{refused:?}");
    // Contrast (#4538 re-quorum): the crate copy was walked and compared by bytes, not
    // missed — one changed byte in the same file and the same stem is refused.
    write(
        t.path(),
        "crates/k/contracts/same-v1.yaml",
        &contract("same2"),
    );
    let refused = corpus(&t.path().join("contracts")).refused;
    assert!(refused.iter().any(|r| r.stem == "same-v1"), "{refused:?}");
}

#[test]
fn a_differing_crate_copy_is_refused_by_name_and_not_unioned() {
    let t = fixture();
    let dir = t.path().join("contracts");
    let c = corpus(&dir);
    assert_eq!(
        c.refused,
        vec![RefusedStem {
            stem: "diff-v1".into(),
            paths: vec![
                "contracts/diff-v1.yaml".into(),
                "crates/k/contracts/diff-v1.yaml".into()
            ],
        }]
    );
    // The top level keeps its own copy; the crate copy contributes no triple.
    let nt = extract(&dir).to_ntriples();
    assert!(nt.contains("contracts/diff-v1.yaml"), "{nt}");
    assert!(!nt.contains("crates/k/contracts/diff-v1.yaml"), "{nt}");
    assert!(!nt.contains("diff-crate"), "{nt}");
    // And the lint gate names it against its own (empty) baseline.
    let found = crate::lint::duplicate_stems::scan_crate_refusals(&dir);
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].stem, "diff-v1");
    assert_eq!(found[0].variants, 2);
}

#[test]
fn a_directory_without_a_manifest_is_not_a_crate() {
    let t = fixture();
    assert!(!stems(&t.path().join("contracts")).contains(&"ghost-v1".to_string()));
}

#[test]
fn a_crate_only_stem_with_two_contents_is_refused_whole() {
    let t = fixture();
    let r = t.path();
    write(r, "crates/j/Cargo.toml", "[package]\nname = \"j\"\n");
    write(
        r,
        "crates/j/contracts/uniq-v1.yaml",
        &contract("uniq-other"),
    );
    let dir = r.join("contracts");
    assert!(!stems(&dir).contains(&"uniq-v1".to_string()));
    assert!(corpus(&dir).refused.iter().any(|x| x.stem == "uniq-v1"
        && x.paths
            == vec![
                "crates/j/contracts/uniq-v1.yaml".to_string(),
                "crates/k/contracts/uniq-v1.yaml".to_string()
            ]));
}

#[test]
fn a_binding_that_names_a_refused_crate_copy_is_refused_by_name_not_resolved_to_the_top_level_node()
{
    let t = fixture();
    let dir = t.path().join("contracts");
    let mut g = crate::ontology::rdf::Graph::new();
    let stats = code::extract(&dir, &mut g);
    assert_eq!(
        stats.refused_bindings,
        vec![
            "crates/k/contracts/binding.yaml: crates/k/contracts/diff-v1.yaml -> k::m::f"
                .to_string()
        ]
    );
    let nt = g.to_ntriples();
    // No symbol edge points at the top-level `diff-v1` node on the refused binding's behalf …
    assert!(
        !nt.contains(&format!("<{}>", iri("contract", "diff-v1"))),
        "{nt}"
    );
    // … while the binding to the admitted crate contract is kept.
    assert!(
        nt.contains(&format!("<{}>", iri("contract", "uniq-v1"))),
        "{nt}"
    );
    // A contract named by relative path binds its stem's node, never `contract/../…`.
    assert!(
        nt.contains(&format!("<{}>", iri("contract", "top-v1"))),
        "{nt}"
    );
    assert_eq!(stats.symbols, 2);
}

#[test]
fn a_crate_yaml_that_is_not_a_typed_contract_is_skipped_by_name_not_admitted() {
    let t = fixture();
    write(
        t.path(),
        "crates/k/contracts/cgp-v1.yaml",
        "name: cgp\nkernel: x\n",
    );
    let dir = t.path().join("contracts");
    let c = corpus(&dir);
    assert!(!stems(&dir).contains(&"cgp-v1".to_string()));
    assert!(c
        .files
        .iter()
        .all(|f| crate::schema::parse_contract(f).is_ok() || f.starts_with(&dir)));
    assert!(c
        .unparsed
        .contains(&"crates/k/contracts/cgp-v1.yaml".to_string()));
}

#[test]
fn a_binding_that_reaches_a_refused_copy_through_dot_dot_is_still_refused() {
    let t = fixture();
    write(t.path(), "crates/j/Cargo.toml", "[package]\nname = \"j\"\n");
    write(
        t.path(),
        "crates/j/contracts/binding.yaml",
        "version: \"1.0.0\"\ntarget_crate: j\nbindings:\n\
         - contract: ../../k/contracts/./diff-v1.yaml\n  equation: e\n  module_path: j::m\n  function: f\n  status: implemented\n",
    );
    let mut g = crate::ontology::rdf::Graph::new();
    let stats = code::extract(&t.path().join("contracts"), &mut g);
    assert!(
        stats.refused_bindings.contains(
            &"crates/j/contracts/binding.yaml: crates/k/contracts/diff-v1.yaml -> j::m::f"
                .to_string()
        ),
        "{:?}",
        stats.refused_bindings
    );
}

#[test]
fn a_top_level_binding_to_a_refused_stems_kept_top_level_file_is_not_refused() {
    let t = fixture();
    write(
        t.path(),
        "contracts/binding.yaml",
        "version: \"1.0.0\"\ntarget_crate: top\nbindings:\n\
         - contract: diff-v1.yaml\n  equation: e\n  module_path: top::m\n  function: f\n  status: implemented\n",
    );
    let mut g = crate::ontology::rdf::Graph::new();
    let stats = code::extract(&t.path().join("contracts"), &mut g);
    assert!(
        stats
            .refused_bindings
            .iter()
            .all(|r| !r.starts_with("contracts/binding.yaml")),
        "{:?}",
        stats.refused_bindings
    );
    assert!(
        g.to_ntriples()
            .contains(&format!("<{}>", iri("contract", "diff-v1"))),
        "the kept top-level diff-v1 is bound"
    );
}
