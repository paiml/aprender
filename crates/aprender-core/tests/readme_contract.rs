//! README.md Contract Enforcement
//!
//! Enforces: contracts/apr-docs-v1.yaml
//! FALSIFY-README-001 through FALSIFY-README-005 + FALSIFY-SVG-002/003
//!
//! These tests prevent README drift from the actual workspace state.

use std::path::Path;
use std::process::Command;

fn workspace_root() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}

fn read_readme() -> String {
    std::fs::read_to_string(workspace_root().join("README.md"))
        .expect("README.md must exist at workspace root")
}

/// FALSIFY-README-001: README contains `cargo install aprender`
#[test]
fn test_readme_has_install_command() {
    let readme = read_readme();
    assert!(
        readme.contains("cargo install aprender"),
        "FALSIFY-README-001: README.md must contain 'cargo install aprender'"
    );
}

/// FALSIFY-README-002: README contains `apr run` example
#[test]
fn test_readme_has_apr_run_example() {
    let readme = read_readme();
    assert!(
        readme.contains("apr run"),
        "FALSIFY-README-002: README.md must contain 'apr run' usage example"
    );
}

/// FALSIFY-README-003: README does NOT reference `cargo install apr-cli`
#[test]
fn test_readme_no_apr_cli_install() {
    let readme = read_readme();
    assert!(
        !readme.contains("cargo install apr-cli"),
        "FALSIFY-README-003: README.md must not contain 'cargo install apr-cli'"
    );
}

/// FALSIFY-README-004: README does NOT reference old repos as installable
#[test]
fn test_readme_no_stale_install_refs() {
    let readme = read_readme();
    for old in &[
        "cargo install trueno",
        "cargo install realizar",
        "cargo install entrenar",
        "cargo install batuta",
    ] {
        assert!(
            !readme.contains(old),
            "FALSIFY-README-004: README.md must not contain '{old}'"
        );
    }
}

/// FALSIFY-README-005: the README's workspace-crate count matches CARGO.
///
/// PREVIOUSLY THIS GATE PROVED THE WRONG PROPOSITION. It counted directory
/// entries under `crates/` (82) and asserted the README contained `**82**`.
/// The README duly said "82 workspace crates" and the gate went green — while
/// `cargo metadata --no-deps` reports **78** packages. Four `crates/` entries
/// are `exclude`d in the root Cargo.toml (old workspace-root shells
/// aprender-viz-ttop / aprender-present / aprender-test, plus the dev-only
/// aprender-train-canary), and a fifth (aprender-contracts-staging) has no
/// Cargo.toml at all. A directory is not a workspace crate.
///
/// The old comment justified the choice as "not `cargo metadata`, which returns
/// MORE packages than directories" — the opposite of the truth.
///
/// Hand-rolled counting cannot be made to agree with cargo. Four methods, four
/// answers, measured 2026-07-27:
///     ls crates/                    -> 82   (directories, incl. a non-crate)
///     members + root                -> 84
///     members - exclude + root      -> 81
///     cargo metadata --no-deps      -> 78   <- the only authoritative one
/// So this asks cargo, via the $CARGO the harness already sets for us.
///
/// It also now asserts the SPECIFIC claims-table row rather than a bare
/// `**78**` occurring anywhere in the file: the old form would have been
/// satisfied by any unrelated bold number.
#[test]
fn test_readme_crate_count_matches_workspace() {
    let readme = read_readme();

    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let out = std::process::Command::new(cargo)
        .args(["metadata", "--no-deps", "--format-version", "1"])
        .current_dir(workspace_root())
        .output()
        .expect("run cargo metadata");
    assert!(
        out.status.success(),
        "FALSIFY-README-005: `cargo metadata` failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let json = String::from_utf8_lossy(&out.stdout);

    // Count `"name":` keys inside the top-level "packages" array. --no-deps
    // means packages == workspace members exactly.
    let packages_start = json.find("\"packages\":").expect("metadata has packages");
    let crate_count = json[packages_start..].matches("\"manifest_path\":").count();
    assert!(
        crate_count > 1,
        "FALSIFY-README-005: parsed {crate_count} packages from cargo metadata — parser is wrong"
    );

    let expected_row = format!("| Workspace crates | **{crate_count}** workspace crates |");
    assert!(
        readme.contains(&expected_row),
        "FALSIFY-README-005: README claims-table row does not match cargo.\n\
         expected: {expected_row}\n\
         `cargo metadata --no-deps` reports {crate_count} workspace packages. Note that\n\
         `ls crates/` is NOT the same number — some crates/ entries are `exclude`d in the\n\
         root Cargo.toml and one has no Cargo.toml at all. A directory is not a crate."
    );
}

/// FALSIFY-README-007: Contract count in README matches `find contracts/ -name '*.yaml'`.
///
/// The "**M** provable contracts" claim was previously checked ONLY by
/// `scripts/check_readme_claims.sh`, which is executable but wired into NO
/// workflow (`grep -rn check_readme_claims .github/workflows` = 0 hits). So the
/// count drifted freely: README said **1331** while the tree held **1766**
/// (Fable rank-7, PMAT-DRIFT-GATES-001). This test rides the already-wired
/// `cargo test` job, so the claim can no longer drift without failing a PR.
/// Counts `*.yaml` recursively to match the canonical script method.
#[test]
fn test_readme_contract_count_matches_workspace() {
    let readme = read_readme();
    let contracts_dir = workspace_root().join("contracts");

    fn count_yaml(dir: &Path) -> usize {
        let mut n = 0;
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    n += count_yaml(&path);
                } else if path.extension().is_some_and(|ext| ext == "yaml") {
                    n += 1;
                }
            }
        }
        n
    }

    let contract_count = count_yaml(&contracts_dir);
    let count_str = format!("**{contract_count}** provable contracts");
    assert!(
        readme.contains(&count_str),
        "FALSIFY-README-007: README lacks `**{contract_count}** provable contracts` \
         matching `find contracts/ -name '*.yaml'` — update the README claims table row"
    );
}

/// FALSIFY-SVG-002: Hero SVG is accessible
#[test]
fn test_hero_svg_accessible() {
    let svg = std::fs::read_to_string(workspace_root().join("docs/hero.svg"))
        .expect("docs/hero.svg must exist");
    assert!(
        svg.contains(r#"role="img""#),
        "FALSIFY-SVG-002: hero.svg missing role=img"
    );
    assert!(
        svg.contains("aria-label"),
        "FALSIFY-SVG-002: hero.svg missing aria-label"
    );
    assert!(
        svg.contains("<title>"),
        "FALSIFY-SVG-002: hero.svg missing <title>"
    );
}

/// FALSIFY-README-006: README cites upstream POC benchmark repos.
///
/// The Performance section must cite `candle-vs-apr` and
/// `ground-truth-apr-ludwig` so readers can reproduce the performance
/// claims. Hard-coded token/s numbers are NOT asserted here — those
/// drift with tuning runs and should be re-derived from the POC repos
/// at review time, not frozen into a regression test (which is how the
/// pre-rewrite version got stuck on 369.9/3,220 long after those numbers
/// stopped reflecting the best-known configuration).
#[test]
fn test_readme_has_framework_comparison() {
    let readme = read_readme();
    assert!(
        readme.contains("candle-vs-apr"),
        "README must cite paiml/candle-vs-apr for reproducible inference benchmarks"
    );
    assert!(
        readme.contains("ground-truth-apr-ludwig"),
        "README must cite paiml/ground-truth-apr-ludwig for reproducible training benchmarks"
    );
}

/// FALSIFY-SVG-003: Hero SVG is valid XML
#[test]
fn test_hero_svg_valid() {
    let svg_path = workspace_root().join("docs/hero.svg");
    let content = std::fs::read_to_string(&svg_path).expect("read hero.svg");
    // Basic XML validation — starts with <svg, ends with </svg>
    assert!(
        content.trim().starts_with("<svg"),
        "FALSIFY-SVG-003: not valid SVG"
    );
    assert!(
        content.trim().ends_with("</svg>"),
        "FALSIFY-SVG-003: not valid SVG"
    );
}

/// FALSIFY-README-CRATE-001: Every crate has README.md
#[test]
fn test_every_crate_has_readme() {
    let ws_root = workspace_root();
    let crates_dir = ws_root.join("crates");
    let mut missing = Vec::new();

    if let Ok(entries) = std::fs::read_dir(&crates_dir) {
        for entry in entries.flatten() {
            if !entry.path().join("Cargo.toml").exists() {
                continue; // Not a crate
            }
            if !entry.path().join("README.md").exists() {
                missing.push(entry.file_name().to_string_lossy().to_string());
            }
        }
    }

    assert!(
        missing.is_empty(),
        "FALSIFY-README-CRATE-001: Crates missing README.md: {:?}",
        missing
    );
}

/// FALSIFY-README-CRATE-002: Every README links to monorepo
#[test]
fn test_every_readme_links_monorepo() {
    let ws_root = workspace_root();
    let crates_dir = ws_root.join("crates");
    let mut no_link = Vec::new();

    if let Ok(entries) = std::fs::read_dir(&crates_dir) {
        for entry in entries.flatten() {
            let readme = entry.path().join("README.md");
            if !readme.exists() {
                continue;
            }
            let content = std::fs::read_to_string(&readme).unwrap_or_default();
            if !content.contains("paiml/aprender") {
                no_link.push(entry.file_name().to_string_lossy().to_string());
            }
        }
    }

    assert!(
        no_link.is_empty(),
        "FALSIFY-README-CRATE-002: READMEs without monorepo link: {:?}",
        no_link
    );
}
