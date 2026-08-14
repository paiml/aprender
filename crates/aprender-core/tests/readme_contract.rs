//! Published-docs Contract Enforcement — README.md, docs/BEATS.md, CLAUDE.md
//!
//! Enforces: contracts/apr-docs-v1.yaml
//! FALSIFY-README-001..007 + FALSIFY-SVG-002/003
//! FALSIFY-DOCS-BEATS-001..003, FALSIFY-DOCS-CLAUDE-001
//!
//! These tests prevent our *published* claims from drifting away from the tree
//! and from `contracts/`. A doc claim nobody re-measures is a claim nobody can
//! trust — see the withdrawn 1.371x Ollama beat in docs/BEATS.md.
//!
//! WHY ALL THREE DOCS SHARE ONE TEST TARGET. `.github/workflows/ci.yml` runs
//! `--lib` across the workspace, and gates integration targets by *explicit name*
//! on one physical line. A brand-new `tests/*.rs` file would therefore never run
//! (547 of 573 integration targets in this repo are dark for exactly this
//! reason), and only one PR at a time may edit that line without a merge-queue
//! conflict. `readme_contract` is already on it, so the doc gates live here.

use std::path::Path;

fn workspace_root() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("workspace root (crates/aprender-core/../..) must resolve")
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
/// `cargo metadata --no-deps` reports **78** packages. Three `crates/` entries
/// are not workspace members (old workspace-root shells aprender-viz-ttop /
/// aprender-present, which are `exclude`d, plus the dev-only
/// aprender-train-canary, which is simply unlisted), and a fourth
/// (aprender-contracts-staging) has no Cargo.toml at all. A directory is not a
/// workspace crate.
///
/// The old comment justified the choice as "not `cargo metadata`, which returns
/// MORE packages than directories" — the opposite of the truth.
///
/// Hand-rolled counting cannot be made to agree with cargo. Four methods, four
/// answers, re-measured 2026-08-14 after #2470 deleted a fourth shell
/// (crates/aprender-test, the vendored upstream probar workspace root — it had
/// no [package] at all, so it was never one of the 78):
///     ls crates/                    -> 81   (directories, incl. a non-crate)
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

// ---------------------------------------------------------------------------
// docs/BEATS.md — the public scoreboard must match contracts/
// ---------------------------------------------------------------------------
//
// #2349 carve-out. docs/BEATS.md published "apr 1.371x faster than Ollama" as a
// WON headline beat with "gate >=1.10x" long after the claim was withdrawn. The
// contract had already been corrected to `beat_threshold: 0.9000` (a NO-COLLAPSE
// FLOOR) and the harness to `ENFORCED_THRESHOLD: f64 = 0.90`; only the
// scoreboard still said "win". Nothing failed, because nothing compared the two.

const OLLAMA_BEAT_CONTRACT: &str = "contracts/beat-ollama-decode-throughput-speed-v1.yaml";

fn read_doc(rel: &str) -> String {
    let path = workspace_root().join(rel);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{rel} must be readable: {e}"))
}

/// Parse `beat_threshold:` out of a beat contract.
///
/// Deliberately reads the CONTRACT rather than a constant duplicated here — a
/// gate that hardcodes the number it is checking proves only that someone typed
/// the same digits twice.
fn contract_beat_threshold(yaml: &str) -> f64 {
    for line in yaml.lines() {
        if let Some(rest) = line.trim_start().strip_prefix("beat_threshold:") {
            let value = rest.split('#').next().unwrap_or("").trim();
            return value
                .parse::<f64>()
                .unwrap_or_else(|e| panic!("beat_threshold {value:?} is not a number: {e}"));
        }
    }
    panic!("{OLLAMA_BEAT_CONTRACT} has no `beat_threshold:` key");
}

/// Every `<date> apr <tps> ollama <tps> <ratio>x` row recorded in the contract
/// description, as the literal strings a reader would look for.
fn contract_measurement_figures(yaml: &str) -> Vec<String> {
    let mut figures = Vec::new();
    for line in yaml.lines() {
        let t = line.trim();
        if !t.starts_with("20") || !t.contains(" apr ") || !t.contains(" ollama ") {
            continue;
        }
        // 2026-06-15  apr 412.3  ollama 300.7  1.371x  <note...>
        let fields: Vec<&str> = t.split_whitespace().collect();
        if fields.len() < 6 {
            continue;
        }
        figures.push(fields[2].to_string()); // apr tok/s
        figures.push(fields[4].to_string()); // ollama tok/s
        figures.push(fields[5].trim_end_matches('x').to_string()); // ratio
    }
    figures
}

/// FALSIFY-DOCS-BEATS-001: docs/BEATS.md states the Ollama gate exactly as the
/// contract carries it, and never restates the withdrawn 1.10x gate as live.
#[test]
fn test_beats_md_ollama_gate_matches_contract() {
    let beats = read_doc("docs/BEATS.md");
    let threshold = contract_beat_threshold(&read_doc(OLLAMA_BEAT_CONTRACT));

    let rendered = format!("beat_threshold: {threshold:.4}");
    assert!(
        beats.contains(&rendered),
        "FALSIFY-DOCS-BEATS-001: docs/BEATS.md must publish the Ollama gate as the \
         contract carries it (`{rendered}`, from {OLLAMA_BEAT_CONTRACT}). The scoreboard \
         is public; a gate figure it invents is a false claim about what CI enforces."
    );

    // The withdrawn gate, in the exact forms BEATS.md used to publish it.
    const WITHDRAWN_GATE_PHRASES: [&str; 3] = [
        "gate \u{2265}1.10\u{d7}",     // "gate ≥1.10×"
        "\u{2265} ollama \u{d7} 1.10", // "≥ ollama × 1.10"
        "ollama median x 1.10",
    ];
    let restated: Vec<&str> = WITHDRAWN_GATE_PHRASES
        .into_iter()
        .filter(|p| beats.contains(p))
        .collect();
    assert!(
        restated.is_empty(),
        "FALSIFY-DOCS-BEATS-001: docs/BEATS.md restates the WITHDRAWN 1.10x Ollama gate as \
         live: {restated:?}. The contract enforces {threshold:.4}."
    );
}

/// FALSIFY-DOCS-BEATS-002: while the contract's threshold is a floor
/// (`beat_threshold < 1.0`), no Ollama-decode row in docs/BEATS.md may be marked
/// WON. A floor proves apr did not collapse; it cannot prove a win.
#[test]
fn test_beats_md_ollama_decode_is_not_marked_won_under_a_floor() {
    let beats = read_doc("docs/BEATS.md");
    let threshold = contract_beat_threshold(&read_doc(OLLAMA_BEAT_CONTRACT));

    assert!(
        threshold < 1.0,
        "FALSIFY-DOCS-BEATS-002: {OLLAMA_BEAT_CONTRACT} now carries \
         beat_threshold={threshold:.4} >= 1.0, i.e. it asserts a WIN again. That is a \
         scoreboard promotion, not a doc edit: re-measure, then update docs/BEATS.md and \
         this test together."
    );

    let mut won_rows: Vec<String> = Vec::new();
    for line in beats.lines() {
        let lower = line.to_lowercase();
        if lower.contains("ollama") && lower.contains("decode") && line.contains("**WON**") {
            won_rows.push(line.trim().chars().take(120).collect());
        }
    }
    assert!(
        won_rows.is_empty(),
        "FALSIFY-DOCS-BEATS-002: docs/BEATS.md marks an Ollama-decode row **WON** while the \
         contract gate is a no-collapse floor (beat_threshold={threshold:.4}). \
         Offending line(s): {won_rows:?}"
    );
}

/// FALSIFY-DOCS-BEATS-003: docs/BEATS.md publishes EVERY measurement the contract
/// records — including the ones that killed the claim.
///
/// This is the under-claiming guard. Deleting the retracted numbers would let the
/// scoreboard pass FALSIFY-DOCS-BEATS-001/002 while telling the reader less than
/// we know. A withdrawal has to cite what replaced the claim.
#[test]
fn test_beats_md_publishes_every_contract_measurement() {
    let beats = read_doc("docs/BEATS.md");
    let figures = contract_measurement_figures(&read_doc(OLLAMA_BEAT_CONTRACT));

    assert!(
        figures.len() >= 12,
        "FALSIFY-DOCS-BEATS-003: parsed only {} figures from {OLLAMA_BEAT_CONTRACT} — the \
         parser is wrong, or the contract's measurement table changed shape. Fix the parser \
         before trusting this gate.",
        figures.len()
    );

    let missing: Vec<&String> = figures.iter().filter(|f| !beats.contains(*f)).collect();
    assert!(
        missing.is_empty(),
        "FALSIFY-DOCS-BEATS-003: docs/BEATS.md omits measurement(s) the contract records: \
         {missing:?}. Under-claiming is a reporting failure too — publish the numbers that \
         replaced the withdrawn claim, do not just delete the claim."
    );
}

// ---------------------------------------------------------------------------
// CLAUDE.md / docs/BEATS.md — every path they cite must exist
// ---------------------------------------------------------------------------

/// FALSIFY-DOCS-CLAUDE-001: every repo-relative source path cited in CLAUDE.md
/// (and docs/BEATS.md) resolves on disk.
///
/// CLAUDE.md advertised six pre-monorepo paths for months —
/// `realizar/src/inference_trace.rs`, `realizar/src/quantize/fused_gate_up.rs`,
/// `quantize/fused_gate_up.rs`, `src/format/layout_contract.rs`,
/// `src/format/converter/write.rs`, `src/format/converter/mod.rs` — none of which
/// have existed since APR-MONO moved everything under `crates/`. Nothing checked
/// them, so nothing complained.
///
/// Scope is deliberately narrow so the gate cannot cry wolf: a token counts only
/// if it sits inside backticks, contains `/`, carries no shell/glob
/// metacharacters, and ends in a source extension. A path that legitimately is
/// not in the tree (e.g. the gitignored `.cargo/config.toml`) is marked
/// `[gitignored]` on the same line — information the reader wants anyway.
const DOCS_WITH_PATHS: [&str; 2] = ["CLAUDE.md", "docs/BEATS.md"];

/// Does this backticked token look like a repo-relative source path we can check?
fn looks_like_repo_path(token: &str) -> bool {
    const EXTENSIONS: [&str; 6] = [".rs", ".yaml", ".yml", ".toml", ".sh", ".md"];
    const SKIP_PREFIXES: [&str; 5] = ["http", "hf://", "~", "/", "target/"];
    const METACHARACTERS: [char; 12] = [' ', '*', '<', '>', '|', '(', ')', '[', ']', '{', '}', '?'];

    token.contains('/')
        && !token.contains("..")
        && !token.contains(METACHARACTERS)
        && !SKIP_PREFIXES.iter().any(|p| token.starts_with(p))
        && EXTENSIONS.iter().any(|e| token.ends_with(e))
}

/// Every `(1-based line, path)` a document cites, skipping lines that mark the
/// reference as historical (`DELETED`) or deliberately absent (`[gitignored]`).
fn cited_paths(doc: &str) -> Vec<(usize, String)> {
    read_doc(doc)
        .lines()
        .enumerate()
        .filter(|(_, line)| !line.contains("DELETED") && !line.contains("[gitignored]"))
        // Odd-indexed `split('`')` fragments are the backticked spans.
        .flat_map(|(i, line)| {
            line.split('`')
                .skip(1)
                .step_by(2)
                .filter(|t| looks_like_repo_path(t))
                .map(move |t| (i + 1, t.to_string()))
                .collect::<Vec<_>>()
        })
        .collect()
}

#[test]
fn test_documented_paths_exist() {
    let ws_root = workspace_root();
    let cited: Vec<(&str, usize, String)> = DOCS_WITH_PATHS
        .iter()
        .flat_map(|doc| {
            cited_paths(doc)
                .into_iter()
                .map(move |(line, path)| (*doc, line, path))
        })
        .collect();

    assert!(
        cited.len() >= 20,
        "FALSIFY-DOCS-CLAUDE-001: only {} paths extracted from {DOCS_WITH_PATHS:?} — the \
         extractor stopped matching. A gate that checks nothing is worse than no gate.",
        cited.len()
    );

    let missing: Vec<String> = cited
        .iter()
        .filter(|(_, _, path)| !ws_root.join(path).exists())
        .map(|(doc, line, path)| format!("{doc}:{line}: {path}"))
        .collect();
    assert!(
        missing.is_empty(),
        "FALSIFY-DOCS-CLAUDE-001: documented path(s) do not exist: {missing:#?}\n\
         Fix the doc (APR-MONO moved everything under crates/), or mark a deliberately \
         absent path `[gitignored]` on the same line."
    );
}
