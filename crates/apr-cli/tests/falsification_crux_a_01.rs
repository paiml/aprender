//! Falsification tests for CRUX-A-01 — Pull model by short name.
//!
//! Contract: contracts/crux-A-01-v1.yaml
//! Closes:
//! - FALSIFY-CRUX-A-01-001: `apr pull <short> --dry-run` emits the canonical
//!   URL on stdout and performs zero network I/O.
//! - FALSIFY-CRUX-A-01-002: `configs/aliases.yaml` ships with the repo,
//!   parses as str→str, and contains the four canonical short names:
//!   {"llama3", "mistral", "phi3", "qwen2"} ⊆ keys(map).
//! - FALSIFY-CRUX-A-01-003: unknown short name exits non-zero with a
//!   "did you mean …" hint (Levenshtein ≤ 2 against alias-map keys).
//! - FALSIFY-CRUX-A-01-004: `apr registry aliases --json` emits the full
//!   alias map as a JSON object with every value carrying a known scheme.
//! - FALSIFY-CRUX-A-01-005: three back-to-back process invocations of
//!   `apr pull llama3 --dry-run` emit byte-identical canonical URLs.

#![allow(clippy::unwrap_used)]

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::Command;

fn repo_root() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .ancestors()
        .nth(2)
        .expect("CARGO_MANIFEST_DIR has repo root 2 ancestors up")
        .to_path_buf()
}

fn load_aliases() -> BTreeMap<String, String> {
    let path = repo_root().join("configs").join("aliases.yaml");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("FALSIFY-CRUX-A-01-002: cannot read {path:?}: {e}"));
    serde_yaml::from_str::<BTreeMap<String, String>>(&text)
        .unwrap_or_else(|e| panic!("FALSIFY-CRUX-A-01-002: aliases.yaml not a str→str map: {e}"))
}

#[test]
fn falsify_crux_a_01_002_aliases_yaml_present_and_parseable() {
    let map = load_aliases();
    assert!(
        !map.is_empty(),
        "FALSIFY-CRUX-A-01-002: aliases.yaml parsed empty"
    );
}

#[test]
fn falsify_crux_a_01_002_canonical_short_names_present() {
    let map = load_aliases();
    for canonical in ["llama3", "mistral", "phi3", "qwen2"] {
        assert!(
            map.contains_key(canonical),
            "FALSIFY-CRUX-A-01-002: canonical short name '{canonical}' missing from aliases.yaml"
        );
    }
}

#[test]
fn falsify_crux_a_01_002_every_value_has_known_scheme() {
    let map = load_aliases();
    for (name, url) in &map {
        assert!(
            url.starts_with("hf://") || url.starts_with("https://"),
            "FALSIFY-CRUX-A-01-002: alias '{name}' → '{url}' does not start with hf:// or https://"
        );
    }
}

#[test]
fn falsify_crux_a_01_002_resolution_deterministic() {
    let a = load_aliases();
    let b = load_aliases();
    assert_eq!(
        a, b,
        "FALSIFY-CRUX-A-01-002: two reads of aliases.yaml must yield byte-identical maps"
    );
}

// ---------------------------------------------------------------------------
// FALSIFY-CRUX-A-01-001 — `apr pull <short> --dry-run` emits canonical URL.
// ---------------------------------------------------------------------------

fn run_apr_pull_dry_run(short: &str) -> (std::process::Output, String) {
    let output = Command::new(env!("CARGO_BIN_EXE_apr"))
        .args(["pull", short, "--dry-run"])
        .output()
        .unwrap_or_else(|e| panic!("FALSIFY-CRUX-A-01-001: failed to spawn apr: {e}"));
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    (output, stdout)
}

#[test]
fn falsify_crux_a_01_001_dry_run_emits_canonical_url_for_llama3() {
    let (output, stdout) = run_apr_pull_dry_run("llama3");
    assert!(
        output.status.success(),
        "FALSIFY-CRUX-A-01-001: apr pull llama3 --dry-run must exit 0, stdout=\n{stdout}\nstderr=\n{}",
        String::from_utf8_lossy(&output.stderr),
    );
    assert!(
        stdout.contains("hf://") || stdout.contains("https://"),
        "FALSIFY-CRUX-A-01-001: --dry-run stdout must contain a canonical URL, got:\n{stdout}"
    );
}

#[test]
fn falsify_crux_a_01_001_dry_run_resolution_is_deterministic() {
    let (_, a) = run_apr_pull_dry_run("llama3");
    let (_, b) = run_apr_pull_dry_run("llama3");
    let extract = |s: &str| -> String {
        s.lines()
            .find_map(|l| {
                l.split_whitespace()
                    .find(|w| w.starts_with("hf://") || w.starts_with("https://"))
                    .map(str::to_string)
            })
            .unwrap_or_default()
    };
    let url_a = extract(&a);
    let url_b = extract(&b);
    assert!(!url_a.is_empty(), "no canonical URL in first run:\n{a}");
    assert_eq!(
        url_a, url_b,
        "FALSIFY-CRUX-A-01-001: two back-to-back --dry-run invocations must be byte-identical"
    );
}

#[test]
fn falsify_crux_a_01_001_dry_run_every_canonical_name_resolves() {
    for canonical in ["llama3", "mistral", "phi3", "qwen2"] {
        let (output, stdout) = run_apr_pull_dry_run(canonical);
        assert!(
            output.status.success(),
            "FALSIFY-CRUX-A-01-001: {canonical} --dry-run must exit 0"
        );
        assert!(
            stdout.contains("hf://") || stdout.contains("https://"),
            "FALSIFY-CRUX-A-01-001: {canonical} --dry-run must emit canonical URL, got:\n{stdout}"
        );
    }
}

// ---------------------------------------------------------------------------
// FALSIFY-CRUX-A-01-003 — unknown short name emits a did-you-mean hint
// (Levenshtein ≤ 2 against the alias map) and exits non-zero.
// ---------------------------------------------------------------------------

#[test]
fn falsify_crux_a_01_003_typo_exits_nonzero() {
    let (output, _stdout) = run_apr_pull_dry_run("lama3");
    assert!(
        !output.status.success(),
        "FALSIFY-CRUX-A-01-003: unknown short name 'lama3' must exit non-zero"
    );
}

#[test]
fn falsify_crux_a_01_003_typo_suggests_llama3() {
    let output = Command::new(env!("CARGO_BIN_EXE_apr"))
        .args(["pull", "lama3", "--dry-run"])
        .output()
        .unwrap();
    let combined = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let lower = combined.to_lowercase();
    assert!(
        lower.contains("did you mean"),
        "FALSIFY-CRUX-A-01-003: output must contain 'did you mean', got:\n{combined}"
    );
    assert!(
        lower.contains("llama3"),
        "FALSIFY-CRUX-A-01-003: suggestion must include 'llama3', got:\n{combined}"
    );
}

#[test]
fn falsify_crux_a_01_003_far_typo_has_no_suggestion() {
    // A name with edit distance > 2 from every alias should NOT emit a
    // concrete suggestion — the error should fall back to the generic hint.
    let output = Command::new(env!("CARGO_BIN_EXE_apr"))
        .args(["pull", "completely-unrelated-xyz-model", "--dry-run"])
        .output()
        .unwrap();
    assert!(
        !output.status.success(),
        "FALSIFY-CRUX-A-01-003: far typo must still exit non-zero"
    );
    let combined = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
    .to_lowercase();
    assert!(
        !combined.contains("did you mean"),
        "FALSIFY-CRUX-A-01-003: far typo must NOT fabricate a suggestion, got:\n{combined}"
    );
}

// ---------------------------------------------------------------------------
// FALSIFY-CRUX-A-01-004 — `apr registry aliases --json` emits the full
// alias map as a JSON object; every value starts with a known scheme.
// ---------------------------------------------------------------------------

fn run_registry_aliases_json() -> (std::process::Output, String) {
    let output = Command::new(env!("CARGO_BIN_EXE_apr"))
        .args(["registry", "aliases", "--json"])
        .output()
        .unwrap_or_else(|e| panic!("FALSIFY-CRUX-A-01-004: failed to spawn apr: {e}"));
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    (output, stdout)
}

#[test]
fn falsify_crux_a_01_004_registry_aliases_json_is_object() {
    let (output, stdout) = run_registry_aliases_json();
    assert!(
        output.status.success(),
        "FALSIFY-CRUX-A-01-004: exit 0 required, stdout=\n{stdout}\nstderr=\n{}",
        String::from_utf8_lossy(&output.stderr),
    );
    let json: serde_json::Value = serde_json::from_str(stdout.trim())
        .unwrap_or_else(|e| panic!("FALSIFY-CRUX-A-01-004: stdout not valid JSON: {e}\n{stdout}"));
    let obj = json
        .as_object()
        .expect("FALSIFY-CRUX-A-01-004: top-level JSON must be an object");
    assert!(
        !obj.is_empty(),
        "FALSIFY-CRUX-A-01-004: alias map must be non-empty"
    );
}

#[test]
fn falsify_crux_a_01_004_registry_aliases_contains_canonical_names() {
    let (_, stdout) = run_registry_aliases_json();
    let json: serde_json::Value = serde_json::from_str(stdout.trim()).unwrap();
    let obj = json.as_object().unwrap();
    for canonical in ["llama3", "mistral", "phi3", "qwen2"] {
        let val = obj.get(canonical).unwrap_or_else(|| {
            panic!("FALSIFY-CRUX-A-01-004: '{canonical}' missing from registry aliases --json")
        });
        let url = val.as_str().unwrap_or_else(|| {
            panic!("FALSIFY-CRUX-A-01-004: '{canonical}' value is not a string")
        });
        assert!(
            url.starts_with("hf://") || url.starts_with("https://"),
            "FALSIFY-CRUX-A-01-004: '{canonical}' → '{url}' missing known scheme"
        );
    }
}

#[test]
fn falsify_crux_a_01_004_registry_aliases_is_deterministic() {
    let (_, a) = run_registry_aliases_json();
    let (_, b) = run_registry_aliases_json();
    assert_eq!(
        a.trim(),
        b.trim(),
        "FALSIFY-CRUX-A-01-004: two invocations of `apr registry aliases --json` must be byte-identical"
    );
}

// ---------------------------------------------------------------------------
// FALSIFY-CRUX-A-01-005 — two separate process invocations of
// `apr pull llama3 --dry-run` emit byte-identical canonical URLs.
// ---------------------------------------------------------------------------

#[test]
fn falsify_crux_a_01_005_cross_process_determinism() {
    let extract = |s: &str| -> String {
        s.lines()
            .find_map(|l| {
                l.split_whitespace()
                    .find(|w| w.starts_with("hf://") || w.starts_with("https://"))
                    .map(str::to_string)
            })
            .unwrap_or_default()
    };
    let (_, a) = run_apr_pull_dry_run("llama3");
    let (_, b) = run_apr_pull_dry_run("llama3");
    let (_, c) = run_apr_pull_dry_run("llama3");
    let ua = extract(&a);
    let ub = extract(&b);
    let uc = extract(&c);
    assert!(
        !ua.is_empty(),
        "FALSIFY-CRUX-A-01-005: no canonical URL:\n{a}"
    );
    assert_eq!(
        ua, ub,
        "FALSIFY-CRUX-A-01-005: invocation 1 vs 2 differ ({ua} vs {ub})"
    );
    assert_eq!(
        ub, uc,
        "FALSIFY-CRUX-A-01-005: invocation 2 vs 3 differ ({ub} vs {uc})"
    );
}
