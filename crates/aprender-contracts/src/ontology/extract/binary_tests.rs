//! ONT-4g: the binary extractor over the `tests/fixtures/ont4g/` snapshot (two `apr` targets and `pv`), the four
//! mutations `contracts/binary-surface-v1.yaml` names (FALSIFY-BIN-001..004), and the repo's own census.

use super::*;
use std::fs;

fn fixture(name: &str) -> String {
    let p = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/ont4g")
        .join(name);
    fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

fn write(root: &Path, rel: &str, text: &str) {
    let p = root.join(rel);
    fs::create_dir_all(p.parent().unwrap()).unwrap();
    fs::write(p, text).unwrap();
}

/// A workspace whose manifests declare exactly the fixture's three bin targets: the root facade's `[[bin]] apr`,
/// `apr-cli`'s discovered `src/main.rs` renamed by `[[bin]]`, and `pv`'s.
fn planted(snapshot: &str) -> tempfile::TempDir {
    let d = tempfile::tempdir().unwrap();
    let r = d.path();
    write(
        r,
        "Cargo.toml",
        "[package]\nname = \"aprender\"\n\n[[bin]]\nname = \"apr\"\npath = \"src/bin/apr.rs\"\n\n\
         [workspace]\nmembers = [\"crates/*\"]\n",
    );
    write(r, "src/lib.rs", "");
    write(r, "src/bin/apr.rs", "fn main() {}\n");
    write(
        r,
        "crates/apr-cli/Cargo.toml",
        "[package]\nname = \"apr-cli\"\n\n[[bin]]\nname = \"apr\"\npath = \"src/main.rs\"\n",
    );
    write(r, "crates/apr-cli/src/main.rs", "fn main() {}\n");
    write(
        r,
        "crates/aprender-contracts-cli/Cargo.toml",
        "[package]\nname = \"aprender-contracts-cli\"\n\n[[bin]]\nname = \"pv\"\npath = \"src/main.rs\"\n",
    );
    write(
        r,
        "crates/aprender-contracts-cli/src/main.rs",
        "fn main() {}\n",
    );
    write(r, SNAPSHOT, snapshot);
    write(r, LEDGER, &fixture("surface_audit.csv"));
    d
}

fn run(snapshot: &str) -> (Graph, BinaryStats) {
    let d = planted(snapshot);
    let mut g = Graph::default();
    let stats = extract_at(d.path(), &mut g);
    (g, stats)
}

fn has(g: &Graph, n: &str, pred: &str) -> bool {
    !g.objects(n, &bin(pred)).is_empty()
}

/// Rewrite the line whose `(package, target)` matches with `f`.
fn mutate(package: &str, f: impl Fn(&mut serde_json::Value)) -> String {
    fixture("snapshot.jsonl")
        .lines()
        .map(|l| {
            let mut v: serde_json::Value = serde_json::from_str(l).unwrap();
            if v["package"] == package {
                f(&mut v);
            }
            v.to_string()
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn the_fixture_is_green_three_nodes_two_names() {
    let (g, s) = run(&fixture("snapshot.jsonl"));
    assert!(s.errors.is_empty(), "{:?}", s.errors);
    assert_eq!((s.targets, s.names, s.census), (3, 2, Some(3)));
    assert_eq!(g.instances_of(&ont("Binary")).len(), 3);
    assert_eq!(
        (s.version_lacks_sha, s.name_pair_mismatch, s.help_failed),
        (0, 0, 0)
    );
    assert_eq!((s.unledgered, s.orphans, s.no_audit_row), (0, 0, 0));
    let pv = node("aprender-contracts-cli", "pv");
    assert_eq!(
        g.instances_of(&bin("target/aprender-contracts-cli/pv")),
        [pv.as_str()]
    );
    assert_eq!(g.objects(&pv, &bin("auditRow")).len(), 2);
    // the bare `apr --version` row and the flag after `apr serve` join as the binary and `serve`, not orphans
    assert!(!has(&g, &node("apr-cli", "apr"), "ledgerOrphan"));
}

#[test]
fn two_extractions_are_byte_identical() {
    let a = run(&fixture("snapshot.jsonl")).0.to_ntriples();
    let b = run(&fixture("snapshot.jsonl")).0.to_ntriples();
    assert_eq!(a, b);
    assert!(!a.contains("_:"), "no blank nodes (R-15)");
}

/// FALSIFY-BIN-001: drop a line — the census names the target with no snapshot line.
#[test]
fn falsify_bin_001_a_dropped_line_is_red_naming_the_target() {
    let snap: String = fixture("snapshot.jsonl")
        .lines()
        .filter(|l| !l.contains("\"package\":\"aprender-contracts-cli\""))
        .collect::<Vec<_>>()
        .join("\n");
    let (_, s) = run(&snap);
    assert_eq!((s.targets, s.census), (2, Some(3)));
    assert_eq!(s.errors.len(), 1, "{:?}", s.errors);
    assert!(s.errors[0].what.contains("aprender-contracts-cli/pv"));
}

/// FALSIFY-BIN-002: keyed by name, the apr pair would collapse — keyed by package and target it does not.
#[test]
fn falsify_bin_002_the_apr_pair_is_two_nodes() {
    let (g, s) = run(&fixture("snapshot.jsonl"));
    let a = node("apr-cli", "apr");
    let b = node("aprender", "apr");
    assert_ne!(a, b);
    assert!(g.instances_of(&ont("Binary")).contains(&a.as_str()));
    assert!(g.instances_of(&ont("Binary")).contains(&b.as_str()));
    // the mutation: an IRI from bin:name alone yields one node per name, one short of the census
    assert_eq!(s.names + 1, s.census.unwrap());
}

/// FALSIFY-BIN-003: strip the sha from one version — `bin:versionLacksSha` on that node only.
#[test]
fn falsify_bin_003_a_version_without_the_commit_fires() {
    let snap = mutate("aprender-contracts-cli", |v| {
        v["version"] = "pv 0.70.0".into();
    });
    let (g, s) = run(&snap);
    assert_eq!(s.version_lacks_sha, 1);
    assert!(has(
        &g,
        &node("aprender-contracts-cli", "pv"),
        "versionLacksSha"
    ));
    assert!(!has(&g, &node("apr-cli", "apr"), "versionLacksSha"));
}

/// FALSIFY-BIN-004: change one apr helpSha256 — `bin:namePairMismatch` on BOTH apr nodes, each naming the other.
#[test]
fn falsify_bin_004_the_apr_pair_diverging_fires_on_both() {
    let snap = mutate("apr-cli", |v| {
        v["help_sha256"] = "f".repeat(64).into();
    });
    let (g, s) = run(&snap);
    assert_eq!(s.name_pair_mismatch, 2);
    let a = node("apr-cli", "apr");
    let b = node("aprender", "apr");
    assert_eq!(
        g.objects(&a, &bin("namePairMismatch")),
        [&Term::iri(b.clone())]
    );
    assert_eq!(g.objects(&b, &bin("namePairMismatch")), [&Term::iri(a)]);
    assert!(!has(
        &g,
        &node("aprender-contracts-cli", "pv"),
        "namePairMismatch"
    ));
}

#[test]
fn a_foreign_schema_or_an_omitted_array_is_refused_by_line() {
    let (_, s) = run(&mutate("aprender", |v| {
        v["schema"] = "binary-snapshot/v0".into()
    }));
    assert!(s
        .errors
        .iter()
        .any(|e| e.what.contains("binary-snapshot/v0")));
    let (_, s) = run(&mutate("aprender", |v| {
        v.as_object_mut().unwrap().remove("tools");
    }));
    assert!(s.errors.iter().any(|e| e.what.contains("`tools`")));
    // the refused line is also a census target with no node
    assert!(s.errors.iter().any(|e| e.what.contains("aprender/apr")));
}

#[test]
fn a_workspace_root_with_no_snapshot_is_red_not_vacuous() {
    let d = planted("");
    fs::remove_file(d.path().join(SNAPSHOT)).unwrap();
    let s = extract_at(d.path(), &mut Graph::default());
    assert_eq!(s.targets, 0);
    assert!(
        s.errors.iter().any(|e| e.what.contains("absent")),
        "{:?}",
        s.errors
    );
    let s = extract_at(Path::new("/nonexistent-ont4g"), &mut Graph::default());
    assert!(s.errors.is_empty() && !s.at_workspace_root);
}

/// A workspace that declares no bin target (the ont fixtures' `[workspace]` roots) has nothing to snapshot: zero
/// census, zero lines, no refusal. The census still being read is what keeps the three-target case above red.
#[test]
fn a_workspace_root_with_no_bin_target_needs_no_snapshot() {
    let d = tempfile::tempdir().unwrap();
    write(
        d.path(),
        "Cargo.toml",
        "[workspace]\nmembers = [\"crates/kern\"]\n",
    );
    write(
        d.path(),
        "crates/kern/Cargo.toml",
        "[package]\nname = \"kern\"\n",
    );
    write(d.path(), "crates/kern/src/lib.rs", "");
    let s = extract_at(d.path(), &mut Graph::default());
    assert_eq!((s.targets, s.census), (0, Some(0)));
    assert!(s.errors.is_empty(), "{:?}", s.errors);
}

#[test]
fn the_ledger_join_finds_unledgered_and_orphans_both_ways() {
    let snap = mutate("apr-cli", |v| {
        v["commands"] = serde_json::json!(["run", "serve", "serve plan", "tune"]);
        v["routes"] = serde_json::json!([]);
    });
    let (g, _) = run(&snap);
    let a = node("apr-cli", "apr");
    assert_eq!(
        g.objects(&a, &bin("unledgeredCommand")),
        [&Term::string("tune")]
    );
    assert_eq!(
        g.objects(&a, &bin("ledgerOrphanRoute")),
        [&Term::string("GET /api/tags")]
    );
    assert!(!has(&g, &node("aprender", "apr"), "unledgeredCommand"));
}

#[test]
fn feature_forms_case_table() {
    for (name, feature, want) in [
        ("apr", "apr run", Some("run")),
        ("apr", "apr serve plan <MODEL>", Some("serve plan")),
        ("apr", "apr eval --task humaneval", Some("eval")),
        ("apr", "apr --version", Some("")),
        ("apr", "apr", Some("")),
        ("apr", "aprender-orchestrate run", None),
        ("pv", "apr run", None),
    ] {
        assert_eq!(command_path(name, feature).as_deref(), want, "{feature}");
    }
    assert_eq!(
        route_key("POST /api/chat (stream:true, x)").as_deref(),
        Some("POST /api/chat")
    );
    assert_eq!(route_key("HTTP 404 fallback route index"), None);
    assert_eq!(
        tool_key("mcp:calculator (banco POST /api/v1/mcp)").as_deref(),
        Some("calculator")
    );
    assert_eq!(tool_key("mcp: nothing"), None);
}

#[test]
fn a_quoted_ledger_field_keeps_its_comma() {
    let l = parse_ledger(&fixture("surface_audit.csv")).unwrap();
    assert!(l["apr"].rows.contains("GET /api/tags (list, local)"));
    assert!(l["apr"].routes.contains("GET /api/tags"));
    assert!(parse_ledger("binary,feature\napr,apr run\n").is_err());
}

#[test]
fn the_positive_control_fires() {
    assert!(positive_control());
}

/// The repo's own census reads the bin targets `cargo metadata --no-deps` lists (29 on 2026-09-26, 28 names):
/// both `apr` declarations and `pv` are in it.
#[test]
fn the_repo_census_has_both_aprs_and_pv() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let c = census(&root).expect("the repo root is a workspace");
    for key in [
        ("apr-cli", "apr"),
        ("aprender", "apr"),
        ("aprender-contracts-cli", "pv"),
    ] {
        assert!(
            c.contains(&(key.0.to_string(), key.1.to_string())),
            "{key:?} in {c:?}"
        );
    }
}
