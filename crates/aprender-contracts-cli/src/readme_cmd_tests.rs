//! Fixture-only falsifiers for `pv-readme` (#6394): tempdirs, no network, no live repo state.

use super::*;

const SHAPE: &str = "name: readme-root\nentity: {type: readme, ref: README.md}\nshape:\n  targetClass: readme:Readme\n  closed: true\n  properties:\n    - {path: readme:schemaVersion, minCount: 1, maxCount: 1}\n    - {path: readme:kind, in: [library, cli, service, spec]}\n    - {path: readme:entrypoint, minCount: 1, resolves: path}\n    - {path: readme:verifiedCommand, minCount: 1, resolves: ci-step}\n    - {path: readme:section, minCount: 3}\n    - {path: readme:contractCount, maxCount: 1, resolves: census}\n";

const WORKFLOW: &str =
    "on: [pull_request]\njobs:\n  j:\n    runs-on: x\n    steps:\n      - run: make lint\n";

/// walked = 2 (readme-root.yaml, a.yaml); legacy/, dotfiles and ontology.yaml do not count.
fn fixture(shape: &str) -> tempfile::TempDir {
    let d = tempfile::tempdir().expect("tempdir");
    let c = d.path().join("contracts");
    std::fs::create_dir_all(c.join("legacy")).expect("mkdir");
    std::fs::create_dir_all(d.path().join(".github/workflows")).expect("mkdir");
    std::fs::create_dir_all(d.path().join("crates/x")).expect("mkdir");
    std::fs::write(c.join("readme-root.yaml"), shape).expect("w");
    std::fs::write(c.join("a.yaml"), "name: a\n").expect("w");
    std::fs::write(c.join("legacy/old.yaml"), "name: old\n").expect("w");
    std::fs::write(c.join(".hidden.yaml"), "name: h\n").expect("w");
    std::fs::write(c.join("ontology.yaml"), "name: sigma\n").expect("w");
    std::fs::write(c.join("census.json"), "{\"n_files\": 2}").expect("w");
    std::fs::write(d.path().join(".github/workflows/ci.yml"), WORKFLOW).expect("w");
    d
}

fn readme(count: u64, fm: u64, extra: &str, entry: &str) -> String {
    format!(
        "---\nschema_version: \"1.0\"\nkind: library\nentrypoints: [{entry}]\ncontract_count: {fm}\n---\n# T\n## A\n## B\n\nWe carry {START}{count}{END} contracts.\n\n```bash\nmake lint\n```\n{extra}"
    )
}

fn put(d: &tempfile::TempDir, text: &str) -> PathBuf {
    let p = d.path().join("README.md");
    std::fs::write(&p, text).expect("w");
    p
}

fn check(d: &tempfile::TempDir, p: &Path) -> Result<String, Refusal> {
    run(p, &d.path().join("contracts"), false)
}

#[test]
fn walked_count_applies_the_readme_sync_exclusions() {
    let d = fixture(SHAPE);
    assert_eq!(walked_count(&d.path().join("contracts")), Ok(2));
}

#[test]
fn render_touches_only_the_blocks_and_the_frontmatter_line() {
    let t = readme(
        9,
        9,
        "contract_count: 5 in prose is untouched\n",
        "crates/x",
    );
    let r = render(&t, 2, 2);
    assert!(r.contains(&format!("{START}2{END}")));
    assert!(r.contains("\ncontract_count: 2\n"));
    assert!(r.contains("contract_count: 5 in prose is untouched"));
    assert_eq!(render(&r, 2, 2), r, "idempotent");
}

#[test]
fn a_readme_matching_the_graph_passes() {
    let d = fixture(SHAPE);
    let p = put(&d, &readme(2, 2, "", "crates/x"));
    let ok = check(&d, &p).expect("conforms");
    assert!(ok.contains("2 contracts"), "{ok}");
}

#[test]
fn a_wrong_count_fails_check_exits_1_and_writes_nothing() {
    let d = fixture(SHAPE);
    let text = readme(7, 2, "", "crates/x");
    let p = put(&d, &text);
    let e = check(&d, &p).expect_err("stale");
    assert_eq!(e.code, 1);
    assert_eq!(
        std::fs::read_to_string(&p).expect("r"),
        text,
        "--check writes nothing"
    );
}

#[test]
fn write_renders_then_the_check_passes() {
    let d = fixture(SHAPE);
    let p = put(&d, &readme(7, 40, "", "crates/x"));
    let ok = run(&p, &d.path().join("contracts"), true).expect("rendered");
    assert!(ok.ends_with("rewritten"), "{ok}");
    assert!(check(&d, &p).is_ok());
}

#[test]
fn a_missing_required_section_fails_by_shape() {
    let d = fixture(SHAPE);
    let text = readme(2, 2, "", "crates/x").replace("## A\n## B\n", "");
    let p = put(&d, &text);
    let e = check(&d, &p).expect_err("two sections < 3");
    assert_eq!(e.code, 1);
    assert!(e.what.contains("section"), "{}", e.what);
}

#[test]
fn a_missing_entrypoint_and_an_unrun_claim_fail_by_name() {
    let d = fixture(SHAPE);
    let p = put(
        &d,
        &readme(2, 2, "```bash\nmake nope\n```\n", "crates/gone"),
    );
    let e = check(&d, &p).expect_err("refused");
    assert_eq!(e.code, 1);
    assert!(
        e.what.contains("entrypoint unresolved: crates/gone"),
        "{}",
        e.what
    );
    assert!(
        e.what.contains("verifiedCommand unresolved: make nope"),
        "{}",
        e.what
    );
}

#[test]
fn a_readme_with_no_block_or_no_shape_is_exit_3_never_a_pass() {
    let d = fixture(SHAPE);
    let p = put(&d, "---\nschema_version: \"1.0\"\n---\n# T\n");
    assert_eq!(check(&d, &p).expect_err("no block").code, 3);
    let n = fixture("name: other\n");
    let p = put(&n, &readme(2, 2, "", "crates/x"));
    assert_eq!(check(&n, &p).expect_err("no shape").code, 3);
}

#[test]
fn mutated_checker_a_shape_that_cannot_fail_is_refused_by_the_plant() {
    // the shape with every constraint removed accepts anything, including the bare plant
    let toothless = "name: readme-root\nentity: {type: readme, ref: README.md}\nshape:\n  targetClass: readme:Readme\n  properties:\n    - {path: readme:kind}\n";
    let d = fixture(toothless);
    let p = put(
        &d,
        &readme(2, 2, "", "crates/x").replace("## A\n## B\n", ""),
    );
    let e = check(&d, &p).expect_err("plant must refuse the toothless shape");
    assert_eq!(e.code, 3);
    assert!(e.what.contains("plant"), "{}", e.what);
}

#[test]
fn main_with_maps_usage_and_failure_to_nonzero() {
    assert_eq!(main_with(&[]), ExitCode::from(2));
    assert_eq!(main_with(&["--bogus".into()]), ExitCode::from(2));
    let d = fixture(SHAPE);
    let p = put(&d, &readme(7, 2, "", "crates/x"));
    let args = |m: &str| -> Vec<String> {
        vec![
            m.into(),
            "--readme".into(),
            p.display().to_string(),
            "--contracts".into(),
            d.path().join("contracts").display().to_string(),
        ]
    };
    assert_eq!(main_with(&args("--check")), ExitCode::from(1));
    assert_eq!(main_with(&args("--write")), ExitCode::SUCCESS);
    assert_eq!(main_with(&args("--check")), ExitCode::SUCCESS);
}

#[test]
fn write_validates_before_it_replaces_the_readme() {
    let d = fixture(SHAPE);
    // stale count AND an entrypoint that does not exist: the render would fix the first, the shape fails the second
    let text = readme(7, 2, "", "crates/nope");
    let p = put(&d, &text);
    let e = run(&p, &d.path().join("contracts"), true).expect_err("violates the shape");
    assert_eq!(e.code, 1);
    assert_eq!(
        std::fs::read_to_string(&p).expect("r"),
        text,
        "--write must not rewrite a README that fails validation"
    );
}
