use super::*;

#[test]
fn default_is_the_aprender_rule() {
    let c = Config::default();
    assert_eq!((c.max_lines, c.max_entries), (200, 3));
    assert_eq!(c.label, "rollup-ok");
    assert!(c.is_forbidden(".github/workflows/ci.yml"));
    assert!(c.is_forbidden("contracts/a.yaml"));
    assert!(c.is_forbidden("Cargo.lock"));
    assert!(!c.is_forbidden("crates/x/Cargo.lock.rs"));
    assert!(!c.is_forbidden("docs/contracts/x.md"));
}

#[test]
fn a_repo_table_overrides_only_what_it_names() {
    let t = "[repo.infra]\nrepo = \"paiml/infra\"\nmax_entries = 7\nforbidden = [\"fleet/gates/\", \"flake.lock\"]\n";
    let c = Config::from_toml(t, "infra").expect("parses");
    assert_eq!(c.repo, "paiml/infra");
    assert_eq!(
        (c.max_entries, c.max_lines, c.base.as_str()),
        (7, 200, "main")
    );
    assert!(c.is_forbidden("fleet/gates/a.yml"));
    assert!(c.is_forbidden("flake.lock"));
    assert!(
        !c.is_forbidden("Cargo.lock"),
        "forbidden list replaces the default"
    );
}

#[test]
fn repo_args_pin_the_repo_only_when_set() {
    assert!(Config::default().repo_args().is_empty());
    let c = Config {
        repo: "paiml/infra".into(),
        ..Config::default()
    };
    assert_eq!(c.repo_args(), ["--repo", "paiml/infra"]);
}

#[test]
fn bad_config_is_an_error_not_a_default() {
    let one = "[repo.a]\nmax_entires = 3\n";
    assert!(Config::from_toml(one, "a")
        .unwrap_err()
        .contains("max_entires"));
    assert!(Config::from_toml(one, "b")
        .unwrap_err()
        .contains("no [repo.b]"));
    assert!(Config::from_toml("not toml [", "a").is_err());
    assert!(Config::from_toml("[repo.a]\nmax_lines = \"x\"\n", "a").is_err());
}
