use super::*;

fn m(pkg: &str, lib: Option<&str>) -> (String, Option<String>) {
    (pkg.to_string(), lib.map(str::to_string))
}

/// A tree like this one: renamed packages keeping their old lib names, plus the provable-contracts facades.
fn retired() -> BTreeSet<String> {
    retired_crates(&[
        m("aprender", None),
        m("aprender-compute", Some("trueno")),
        m("aprender-gpu", Some("trueno_gpu")),
        m("aprender-serve", Some("realizar")),
        m("aprender-contracts", Some("provable_contracts")),
        m("provable-contracts", None),
        m("provable-contracts-macros", None),
    ])
}

fn rules(text: &str) -> Vec<&'static str> {
    scan_text(text, &retired())
        .into_iter()
        .map(|v| v.rule)
        .collect()
}

/// FALSIFY-README-ONT-001: the must-flag table. Every row is a pointer the 2026-10-15 tree carried.
#[test]
fn falsify_readme_ont_001_must_flag() {
    let cases: &[(&str, &str)] = &[
        ("[![CI](https://github.com/paiml/trueno/actions/workflows/ci.yml/badge.svg)]", RULE_ARCHIVED_REPO),
        ("see https://github.com/paiml/realizar.", RULE_ARCHIVED_REPO),
        ("git clone https://github.com/paiml/entrenar.git", RULE_ARCHIVED_REPO),
        ("[Batuta](https://github.com/paiml/Batuta)", RULE_ARCHIVED_REPO),
        ("https://github.com/paiml/trueno-viz/issues", RULE_ARCHIVED_REPO),
        ("[book](https://paiml.github.io/trueno/)", RULE_ARCHIVED_REPO),
        ("https://github.com/paiml/provable-contracts", RULE_ARCHIVED_REPO),
        ("<img src=\"https://raw.githubusercontent.com/paiml/probar/main/docs/assets/probar-hero.svg\"/>", RULE_ARCHIVED_REPO),
        ("[![Crates.io](https://img.shields.io/crates/v/trueno.svg)](https://crates.io/crates/trueno)", RULE_RETIRED_CRATE),
        ("[docs](https://docs.rs/realizar)", RULE_RETIRED_CRATE),
        ("https://docs.rs/realizar/latest/realizar/", RULE_RETIRED_CRATE),
        ("https://docs.rs/crate/pacha", RULE_RETIRED_CRATE),
        ("![docs](https://img.shields.io/docsrs/renacer)", RULE_RETIRED_CRATE),
        ("https://lib.rs/crates/simular", RULE_RETIRED_CRATE),
        ("https://docs.rs/trueno_gpu/latest/trueno_gpu/", RULE_RETIRED_CRATE),
        ("[![c](https://img.shields.io/crates/v/trueno-gpu.svg)](https://crates.io/crates/trueno-gpu)", RULE_RETIRED_CRATE),
        ("trueno-gpu = \"0.4\"", RULE_RETIRED_INSTALL),
        ("cargo install batuta", RULE_RETIRED_INSTALL),
        ("cargo add trueno-db --features gpu", RULE_RETIRED_INSTALL),
        ("cargo install --locked renacer", RULE_RETIRED_INSTALL),
        ("trueno = \"0.18\"", RULE_RETIRED_INSTALL),
        ("alimentar = { version = \"0.2\", features = [\"hf\"] }", RULE_RETIRED_INSTALL),
    ];
    for (text, rule) in cases {
        let got = rules(text);
        assert!(got.contains(rule), "must flag {rule}: {text:?} -> {got:?}");
    }
}

/// FALSIFY-README-ONT-002: the must-pass table. Live names, prose, and the facades' own pages.
#[test]
fn falsify_readme_ont_002_must_pass() {
    let cases: &[&str] = &[
        "[![CI](https://github.com/paiml/aprender/actions/workflows/ci.yml/badge.svg)]",
        "https://crates.io/crates/aprender-compute",
        "https://docs.rs/aprender-serve",
        "https://paiml.github.io/aprender/",
        "trueno is the library name of aprender-compute.",
        "https://github.com/paiml/whisper.apr",
        "https://github.com/paiml/truenox",
        "https://raw.githubusercontent.com/paiml/aprender/main/crates/aprender-test/docs/assets/probar-hero.svg",
        "https://github.com/paiml/entrenar-cookbook",
        "https://crates.io/crates/provable-contracts",
        "cargo install aprender",
        "cargo add aprender-compute",
        "trueno = { package = \"aprender-compute\", version = \"0.71\" }",
        "trueno = { path = \"../aprender-compute\" }",
        "aprender-compute = \"0.71\"",
        "use trueno_gpu::ptx::PtxModule;",
        "https://crates.io/crates/aprender-gpu",
    ];
    for text in cases {
        let got = rules(text);
        assert!(got.is_empty(), "must pass: {text:?} -> {got:?}");
    }
}

/// FALSIFY-README-ONT-003: a facade name is live only while a package carries it.
#[test]
fn falsify_readme_ont_003_facade_is_live_only_while_packaged() {
    let text = "https://crates.io/crates/provable-contracts";
    assert!(scan_text(text, &retired()).is_empty());
    let no_facade = retired_crates(&[m("aprender-contracts", Some("provable_contracts"))]);
    assert_eq!(scan_text(text, &no_facade)[0].rule, RULE_RETIRED_CRATE);
}

/// FALSIFY-README-ONT-004: manifests are scanned on the crates.io-rendered keys only.
#[test]
fn falsify_readme_ont_004_manifest_keys() {
    let toml = "[package]\nname = \"aprender-verify\"\n# was https://github.com/paiml/certeza\n\
                repository = \"https://github.com/paiml/certeza\"\ndocumentation = \"https://docs.rs/certeza\"\n\
                homepage = \"https://github.com/paiml/aprender\"\n";
    let got = scan_manifest(toml, &retired());
    let lines: Vec<usize> = got.iter().map(|v| v.line).collect();
    assert_eq!(lines, vec![4, 5], "{got:?}");
    assert_eq!(
        package_and_lib(toml),
        Some(("aprender-verify".to_string(), None))
    );
    assert_eq!(package_and_lib("[workspace]\nmembers = []\n"), None);
    assert_eq!(
        package_and_lib("[package]\nnamespace-ish = 1\nname = \"aprender-x\"\n"),
        Some(("aprender-x".to_string(), None))
    );
}

/// FALSIFY-README-ONT-005: line numbers are 1-based and every hit on a line is reported.
#[test]
fn falsify_readme_ont_005_counts_every_hit() {
    let text = "# t\n[a](https://github.com/paiml/trueno) [b](https://github.com/paiml/realizar)\n";
    let got = scan_text(text, &retired());
    assert_eq!(got.len(), 2);
    assert!(got.iter().all(|v| v.line == 2));
}

/// FALSIFY-README-ONT-006: audit_tree reads the root README, nested crate READMEs and manifests.
#[test]
fn falsify_readme_ont_006_audit_tree() {
    let dir = std::env::temp_dir().join(format!("pv-readme-ont-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let nested = dir.join("crates/aprender-x/examples");
    std::fs::create_dir_all(&nested).expect("mkdir");
    std::fs::create_dir_all(dir.join("crates/aprender-x/target")).expect("mkdir");
    std::fs::write(dir.join("README.md"), "https://github.com/paiml/aprender\n").expect("w");
    std::fs::write(dir.join("Cargo.toml"), "[workspace]\n").expect("w");
    std::fs::write(
        dir.join("crates/aprender-x/Cargo.toml"),
        "[package]\nname = \"aprender-x\"\nrepository = \"https://github.com/paiml/pacha\"\n",
    )
    .expect("w");
    std::fs::write(nested.join("README.md"), "cargo install pacha\n").expect("w");
    std::fs::write(
        dir.join("crates/aprender-x/target/README.md"),
        "https://github.com/paiml/trueno\n",
    )
    .expect("w");
    let audit = audit_tree(&dir);
    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(
        audit.readmes.len(),
        2,
        "target/ is skipped: {:?}",
        audit.readmes
    );
    assert!(audit.packages.contains("aprender-x"));
    let hits: Vec<(String, &str)> = audit
        .findings
        .iter()
        .map(|(p, v)| (p.display().to_string(), v.rule))
        .collect();
    assert_eq!(
        hits,
        vec![
            (
                "crates/aprender-x/examples/README.md".to_string(),
                RULE_RETIRED_INSTALL
            ),
            (
                "crates/aprender-x/Cargo.toml".to_string(),
                RULE_ARCHIVED_REPO
            ),
        ]
    );
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// FALSIFY-README-ONT-007 (the gate): the repo's own READMEs and manifests name no retired repo or crate.
#[test]
fn falsify_readme_ont_007_repo_readmes_name_no_retired_repo() {
    let audit = audit_tree(&repo_root());
    assert!(
        audit.readmes.len() > 100,
        "only {} READMEs read",
        audit.readmes.len()
    );
    assert!(
        audit.packages.len() > 50,
        "only {} packages read",
        audit.packages.len()
    );
    let lines: Vec<String> = audit
        .findings
        .iter()
        .map(|(p, v)| format!("{}:{}: {}: {}", p.display(), v.line, v.rule, v.message))
        .collect();
    assert!(
        lines.is_empty(),
        "{} finding(s) — run `pv ontology readme`:\n{}",
        lines.len(),
        lines.join("\n")
    );
}

/// FALSIFY-README-ONT-008: ARCHIVED_REPOS is the consolidation spec's archive list, no more and no less.
#[test]
fn falsify_readme_ont_008_archived_list_matches_the_spec() {
    let spec = std::fs::read_to_string(
        repo_root().join("docs/specifications/aprender-monorepo-consolidation.md"),
    )
    .expect("consolidation spec");
    let re =
        Regex::new(r"gh api -X PATCH repos/paiml/([A-Za-z0-9_.-]+) -f archived=true").expect("re");
    let from_spec: BTreeSet<String> = re
        .captures_iter(&spec)
        .map(|c| c[1].to_ascii_lowercase())
        .collect();
    let ours: BTreeSet<String> = ARCHIVED_REPOS.iter().map(|s| (*s).to_string()).collect();
    assert_eq!(
        from_spec.len(),
        20,
        "spec lists {} archived repos: {from_spec:?}",
        from_spec.len()
    );
    assert_eq!(ours, from_spec);
}
