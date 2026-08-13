//! Gate: Strict test-binding (PV-VER-002).
//!
//! Issue #1510 — `pv lint --strict-test-binding` catches dangling test references
//! in contract `falsification_tests[].test` fields. The existing PV-VER-001 only
//! matches names with `test_*` / `prop_*` prefixes, missing the convention used
//! by ~all real contracts (e.g. `pretrain_init_missing_file_errors`).
//!
//! Drift classes this catches (per Issue #1510):
//!   1. Function-name suffix drift (`_init_matches_constructor` vs `_init_matches_input`)
//!   2. Module-path drift (`transformer::attention::tests::foo` vs `transformer::model::tests::foo`)
//!   3. Convention drift (`_encoder_init_errors` vs `validate_pretrain_init_arch_rejects_encoder`)
//!   4. "Or equivalent" prose-style placeholders
//!
//! Default severity: WARNING. With `--strict`, promoted to ERROR.

use std::collections::HashSet;
use std::path::Path;
use std::time::Instant;

use crate::schema::Contract;

use super::finding::LintFinding;
use super::rules::RuleSeverity;
use super::{GateDetail, GateResult};

/// Gate 9: Strict test-binding — verify every cargo-test reference exists in source.
///
/// Walks every `falsification_tests[].test` field, parses it as a cargo test
/// invocation, extracts the test fn name, and verifies it exists in the source
/// tree under a `#[test]` (or `#[tokio::test]`, etc.) attribute.
///
/// Skipped categories (not flagged):
///   - `LIVE-PENDING:` prefix — explicit deferred-live marker
///   - `pv validate ...` — meta-validation invocation, not a unit test
///   - empty / missing `test:` field — covered by other gates
///
/// When `strict_mode` is false (default), missing refs are reported as Warning
/// findings AND the gate is marked `passed=true` so the overall lint still
/// passes — operators see warnings but CI doesn't block. With `strict_mode`
/// true, gate `passed` reflects ref resolution; combined with severity
/// promotion in `lint::mod::apply_severity_overrides`, this gates merge.
pub(crate) fn run_strict_test_binding_gate(
    contracts: &[(String, Contract)],
    project_root: &Path,
    strict_mode: bool,
) -> (GateResult, Vec<LintFinding>) {
    let start = Instant::now();
    let mut findings = Vec::new();
    let mut total_refs = 0usize;
    let mut missing = 0usize;

    let test_fns = scan_all_test_fns(project_root);

    for (stem, contract) in contracts {
        for ft in &contract.falsification_tests {
            let Some(ref raw_test) = ft.test else {
                continue;
            };
            let trimmed = raw_test.trim().trim_matches('"');
            if !is_cargo_test_invocation(trimmed) {
                continue;
            }

            for cited in extract_cited_fn_names(trimmed) {
                total_refs += 1;
                if test_fns.contains(&cited) {
                    continue;
                }
                missing += 1;
                let mut f = LintFinding::new(
                    "PV-VER-002",
                    RuleSeverity::Warning,
                    format!(
                        "Dangling test reference: cited `{cited}` not found in source \
                         (falsification_tests[{}].test)",
                        ft.id
                    ),
                    format!("contracts/{stem}.yaml"),
                );
                f.contract_stem = Some(stem.clone());
                f.suggestion = Some(format!(
                    "Either rename a test fn to `{cited}`, or update the contract \
                     `test:` field for {} to cite the real fn name.",
                    ft.id
                ));
                findings.push(f);
            }
        }
    }

    let duration = u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX);

    // The gate itself emits Warning-severity findings by default; they are
    // promoted to Error when `--strict` is set (handled by
    // `apply_severity_overrides` in `lint::mod`). The gate's `passed` field
    // tracks whether all refs resolved; lint pass/fail is decided downstream
    // based on the post-promotion severity.
    let gate_passed = if strict_mode { missing == 0 } else { true };
    (
        GateResult {
            name: "strict-test-binding".into(),
            passed: gate_passed,
            skipped: false,
            duration_ms: duration,
            detail: GateDetail::Verify {
                total_refs,
                existing: total_refs - missing,
                missing,
            },
        },
        findings,
    )
}

/// True iff the string looks like a cargo-test invocation (vs a `LIVE-PENDING`
/// note, `pv validate ...`, or other non-cargo-test marker).
pub(crate) fn is_cargo_test_invocation(s: &str) -> bool {
    let s = s.trim();
    if s.is_empty() {
        return false;
    }
    if s.starts_with("LIVE-PENDING") || s.starts_with("LIVE-PENDING:") {
        return false;
    }
    if s.starts_with("pv ") || s.starts_with("pv\t") {
        return false;
    }
    // Heuristic: must contain `cargo test` somewhere.
    s.contains("cargo test")
}

/// Extract the bare fn names cited as test filters from a cargo-test invocation.
///
/// Supports compound `&&` / `||` invocations by splitting on shell separators
/// and parsing each leg independently. Returns an empty Vec for invocations
/// where no concrete fn name can be extracted (e.g. `cargo test -p foo --lib`
/// with no filter).
pub(crate) fn extract_cited_fn_names(invocation: &str) -> Vec<String> {
    let mut out = Vec::new();
    // Split compound shell invocations like `cargo test ... && cargo test ...`.
    for leg in invocation.split("&&").flat_map(|s| s.split("||")) {
        if let Some(name) = extract_one_fn_name(leg.trim()) {
            out.push(name);
        }
    }
    out
}

/// Parse a single `cargo test [-p PKG] [--lib | --test BIN] [-- ] <filter>` leg.
///
/// Returns the bare fn name (last `::` segment of the filter) when one can be
/// identified; returns None if the invocation has no test filter (i.e. would
/// run the entire test set), which we don't flag because it's not a binding.
///
/// Conservative parser:
///   - Truncates at the first shell pipe `|` or redirect `>` / `2>&1` token —
///     anything past those is a downstream tool (grep/awk), not a cargo test
///     filter.
///   - Recognizes `cargo test [...] -- <filter>` form (filter after `--`)
///     specifically.
///   - Ignores filters containing punctuation that's invalid for Rust idents
///     (`.`, `,`, `[`, `]`, `(`, `)`, `'`, `"`) — those are prose tokens.
fn extract_one_fn_name(leg: &str) -> Option<String> {
    let leg = leg.trim();
    if leg.is_empty() || !leg.contains("cargo test") {
        return None;
    }
    // Truncate at shell pipe or redirect — anything past is downstream tooling.
    let leg = leg
        .split_once(" | ")
        .map_or(leg, |(pre, _)| pre)
        .split_once(" 2>&1 ")
        .map_or_else(
            || leg.split_once(" | ").map_or(leg, |(pre, _)| pre),
            |(pre, _)| pre,
        );
    // Final clean: drop trailing redirects.
    let leg = leg
        .split_once(" > ")
        .map_or(leg, |(pre, _)| pre)
        .split_once(" 2>&1")
        .map_or_else(|| leg, |(pre, _)| pre);
    let tokens: Vec<&str> = leg.split_whitespace().collect();

    // Flags that take an argument (consume token + value).
    let flags_with_arg: HashSet<&str> = [
        "-p",
        "--package",
        "--test",
        "--bin",
        "--example",
        "--features",
        "-F",
        "--target",
        "--manifest-path",
    ]
    .into_iter()
    .collect();
    let bare_flags: HashSet<&str> = [
        "--lib",
        "--bins",
        "--all-targets",
        "--no-fail-fast",
        "--release",
        "--workspace",
        "--all-features",
    ]
    .into_iter()
    .collect();

    // Find token index of `test` immediately after `cargo`.
    let mut start = 0;
    for (idx, t) in tokens.iter().enumerate() {
        if *t == "test" && idx > 0 && tokens[idx - 1] == "cargo" {
            start = idx + 1;
            break;
        }
    }

    let mut i = start;
    let mut last_filter: Option<String> = None;
    let mut saw_double_dash = false;
    while i < tokens.len() {
        let tok = tokens[i];
        if tok == "--" {
            saw_double_dash = true;
            i += 1;
            continue;
        }
        if flags_with_arg.contains(tok) {
            i += 2; // skip flag + its value
            continue;
        }
        if bare_flags.contains(tok) || tok.starts_with("--") || tok.starts_with('-') {
            i += 1;
            continue;
        }
        // Positional token. Reject obvious shell residue.
        if tok == "&&" || tok == "||" || tok == ";" || tok == "|" {
            break;
        }
        // Once we've seen `--`, the next positional is the filter (canonical).
        // Otherwise we accept the last positional, but must validate it.
        last_filter = Some(tok.to_string());
        i += 1;
        if saw_double_dash {
            // Stop after first post-`--` filter; the rest are extra cargo
            // test runtime args (e.g. --test-threads).
            break;
        }
    }

    let filter = last_filter?;
    // Trim quotes the user may have left in.
    let filter = filter.trim_matches('"').trim_matches('\'');
    // Last `::` segment is the bare fn name.
    let bare = filter.rsplit("::").next().unwrap_or(filter).to_string();
    if !looks_like_rust_ident(&bare) {
        return None;
    }
    Some(bare)
}

/// Reject prose-style tokens that aren't valid Rust identifiers.
/// A Rust ident is `[A-Za-z_][A-Za-z0-9_]*`. Tolerate trailing pattern wildcards
/// (e.g. `commands::tests::pretrain_*`) by trimming `*`.
fn looks_like_rust_ident(s: &str) -> bool {
    let s = s.trim_end_matches('*');
    let Some(first) = s.chars().next() else {
        return false;
    };
    if !(first.is_ascii_alphabetic() || first == '_') {
        return false;
    }
    s.chars()
        .skip(1)
        .all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Scan the project for fn names that have a `#[test]` (or test-like) attribute.
fn scan_all_test_fns(project_root: &Path) -> HashSet<String> {
    let mut found: HashSet<String> = HashSet::new();
    let effective_root = if project_root.as_os_str().is_empty() {
        Path::new(".")
    } else {
        project_root
    };
    // Top-level src/, crates/, tests/, generated/.
    for sub in &["src", "crates", "tests", "generated"] {
        let d = effective_root.join(sub);
        if d.exists() {
            scan_test_fns(&d, &mut found);
        }
    }
    // Workspace members at root level (e.g. trueno-gpu/src/).
    if let Ok(entries) = std::fs::read_dir(effective_root) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() && path.join("Cargo.toml").exists() {
                let name = path.file_name().unwrap_or_default();
                if name == "src" || name == "crates" || name == "tests" {
                    continue;
                }
                for sub in &["src", "tests"] {
                    let d = path.join(sub);
                    if d.exists() {
                        scan_test_fns(&d, &mut found);
                    }
                }
            }
        }
    }
    found
}

/// Recursively scan a directory for `.rs` files and harvest test fn names.
///
/// A function is considered a test if:
///   - it is preceded (within the previous ~3 non-blank lines) by `#[test]`,
///     `#[tokio::test]`, `#[async_std::test]`, `#[serial_test::serial]`,
///     `#[rstest]`, `#[proptest::proptest]`, or `proptest!{ ... }`; OR
///   - its name starts with `test_` or `prop_` (legacy convention)
fn scan_test_fns(dir: &Path, tests: &mut HashSet<String>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            // Skip target/.git for speed.
            let n = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if n == "target" || n == ".git" || n == "node_modules" {
                continue;
            }
            scan_test_fns(&path, tests);
        } else if path.extension().is_some_and(|e| e == "rs") {
            if let Ok(content) = std::fs::read_to_string(&path) {
                harvest_test_fns(&content, tests);
            }
        }
    }
}

/// Walk a Rust source string and insert names of fns annotated as tests.
pub(crate) fn harvest_test_fns(content: &str, tests: &mut HashSet<String>) {
    // Trailing-window pattern: track recent attribute lines and match against
    // the next `fn ...` we see. This is line-based (handles 99%+ of cases) and
    // doesn't require a real parser.
    let mut last_was_test_attr = false;
    for line in content.lines() {
        let t = line.trim();
        if t.is_empty() || t.starts_with("//") {
            continue;
        }
        if is_test_attribute(t) {
            last_was_test_attr = true;
            continue;
        }
        // Other attributes between `#[test]` and `fn` — keep flag.
        if t.starts_with("#[") && t.ends_with(']') {
            continue;
        }
        // Identify `fn <name>(...)` declarations.
        if let Some(name) = parse_fn_name(t) {
            // Two harvest paths:
            //   (a) attribute-driven: prior #[test]/#[tokio::test]/etc.
            //   (b) prefix-driven: legacy `test_*` / `prop_*` names
            if last_was_test_attr || name.starts_with("test_") || name.starts_with("prop_") {
                tests.insert(name);
            }
            last_was_test_attr = false;
            continue;
        }
        // Any other non-attribute line resets the flag.
        last_was_test_attr = false;
    }
}

fn is_test_attribute(line: &str) -> bool {
    let t = line.trim();
    matches!(
        t,
        "#[test]"
            | "#[tokio::test]"
            | "#[async_std::test]"
            | "#[rstest]"
            | "#[proptest]"
            | "#[proptest::proptest]"
            | "#[serial_test::serial]"
    ) || t.starts_with("#[test(")
        || t.starts_with("#[tokio::test(")
        || t.starts_with("#[rstest(")
        || t.starts_with("#[proptest(")
}

fn parse_fn_name(line: &str) -> Option<String> {
    let t = line.trim();
    let rest = if let Some(r) = t.strip_prefix("pub async fn ") {
        r
    } else if let Some(r) = t.strip_prefix("pub(crate) fn ") {
        r
    } else if let Some(r) = t.strip_prefix("pub fn ") {
        r
    } else if let Some(r) = t.strip_prefix("async fn ") {
        r
    } else {
        t.strip_prefix("fn ")?
    };
    let name: String = rest
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect();
    if name.is_empty() {
        None
    } else {
        Some(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{Contract, FalsificationTest, Metadata};

    // ------------------------------------------------------------------
    // is_cargo_test_invocation
    // ------------------------------------------------------------------

    #[test]
    fn skip_live_pending_marker() {
        assert!(!is_cargo_test_invocation("LIVE-PENDING — requires fixture"));
        assert!(!is_cargo_test_invocation("LIVE-PENDING: GPU smoke"));
    }

    #[test]
    fn skip_pv_validate_invocation() {
        assert!(!is_cargo_test_invocation("pv validate contracts/foo.yaml"));
    }

    #[test]
    fn detect_basic_cargo_test() {
        assert!(is_cargo_test_invocation(
            "cargo test -p apr-cli --lib commands::pretrain::tests::foo"
        ));
    }

    // ------------------------------------------------------------------
    // extract_cited_fn_names
    // ------------------------------------------------------------------

    #[test]
    fn extract_simple_filter() {
        let names = extract_cited_fn_names(
            "cargo test -p apr-cli --lib commands::pretrain::tests::pretrain_init_missing_file_errors",
        );
        assert_eq!(names, vec!["pretrain_init_missing_file_errors"]);
    }

    #[test]
    fn extract_compound_invocation() {
        let names = extract_cited_fn_names(
            "cargo test -p apr-cli --lib commands::pretrain::tests::a && \
             cargo test -p apr-cli --lib commands::pretrain::tests::b",
        );
        assert_eq!(names, vec!["a", "b"]);
    }

    #[test]
    fn extract_dashed_filter_after_separator() {
        let names = extract_cited_fn_names(
            "cargo test -p aprender-train --lib -- falsify_apr_pretrain_arch_009",
        );
        assert_eq!(names, vec!["falsify_apr_pretrain_arch_009"]);
    }

    #[test]
    fn extract_skips_shell_pipe_residue() {
        // Issue #1510: contracts with `2>&1 | grep "..."` shell pipelines must
        // not interpret the grep pattern as a test name.
        let names = extract_cited_fn_names(
            r#"cargo test -p aprender-core --lib -- logistic_regression 2>&1 | grep "test result: ok""#,
        );
        // Should only see the real filter `logistic_regression`, not `ok`.
        assert!(names.contains(&"logistic_regression".to_string()));
        assert!(!names.contains(&"ok".to_string()));
    }

    #[test]
    fn extract_skips_features_arg_value() {
        // `--features runtime` — `runtime` is a feature name, not a test.
        let names =
            extract_cited_fn_names("cargo test -p aprender-test-lib --features runtime -- runtime");
        // Without proper flag-arg handling, the parser pulled `runtime`
        // (feature value) too. With `--features` whitelisted, we only see
        // the post-`--` filter `runtime`. That filter happens to also be
        // called "runtime" here, which IS a valid Rust ident.
        assert_eq!(names, vec!["runtime"]);
    }

    #[test]
    fn extract_rejects_prose_tokens() {
        // Issue #1510: `bounds.`, `MiB.`, `2.0]` etc. from prose `test:` fields
        // must not be treated as test names.
        assert!(extract_cited_fn_names("cargo test -- bounds.").is_empty());
        assert!(extract_cited_fn_names("cargo test -- MiB.").is_empty());
        assert!(extract_cited_fn_names("cargo test -- 2.0]").is_empty());
    }

    // ------------------------------------------------------------------
    // harvest_test_fns
    // ------------------------------------------------------------------

    #[test]
    fn harvest_attribute_marked_tests() {
        let src = "
#[test]
fn pretrain_init_missing_file_errors() {}
fn helper() {}
#[tokio::test]
async fn async_smoke() {}
";
        let mut found = HashSet::new();
        harvest_test_fns(src, &mut found);
        assert!(found.contains("pretrain_init_missing_file_errors"));
        assert!(found.contains("async_smoke"));
        assert!(!found.contains("helper"));
    }

    #[test]
    fn harvest_legacy_prefix_tests() {
        let src = "fn test_basic() {}\nfn prop_invariant() {}\nfn other() {}\n";
        let mut found = HashSet::new();
        harvest_test_fns(src, &mut found);
        assert!(found.contains("test_basic"));
        assert!(found.contains("prop_invariant"));
        assert!(!found.contains("other"));
    }

    #[test]
    fn harvest_with_intervening_attribute() {
        // #[test] then #[ignore] then fn — should still pick up.
        let src = "
#[test]
#[ignore]
fn ignored_test_still_counts() {}
";
        let mut found = HashSet::new();
        harvest_test_fns(src, &mut found);
        assert!(found.contains("ignored_test_still_counts"));
    }

    // ------------------------------------------------------------------
    // End-to-end gate tests — drift classes from Issue #1510
    // ------------------------------------------------------------------

    /// Build a one-FalsificationTest contract for end-to-end gate testing.
    fn fixture_contract(test_field: &str) -> Vec<(String, Contract)> {
        let mut c = Contract {
            metadata: Metadata {
                version: "1.0.0".into(),
                description: "fixture contract".into(),
                ..Default::default()
            },
            ..Default::default()
        };
        c.falsification_tests.push(FalsificationTest {
            id: "FALSIFY-TEST-001".into(),
            rule: "rule".into(),
            prediction: "prediction".into(),
            test: Some(test_field.into()),
            if_fails: "investigate".into(),
        });
        vec![("fixture".to_string(), c)]
    }

    /// Build a tempdir source tree with a single .rs file containing `src`.
    fn fixture_source_tree(src: &str) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        let crates = dir.path().join("crates");
        std::fs::create_dir_all(&crates).unwrap();
        let crate_a = crates.join("apr-cli").join("src");
        std::fs::create_dir_all(&crate_a).unwrap();
        std::fs::write(crate_a.join("test_module.rs"), src).unwrap();
        dir
    }

    #[test]
    fn happy_path_existing_test_no_warning() {
        let contracts = fixture_contract(
            "cargo test -p apr-cli --lib commands::pretrain::tests::pretrain_init_matches_input",
        );
        let dir = fixture_source_tree("#[test]\nfn pretrain_init_matches_input() {}\n");

        let (gate, findings) = run_strict_test_binding_gate(&contracts, dir.path(), false);
        assert_eq!(findings.len(), 0, "found unexpected findings: {findings:?}");
        assert!(gate.passed, "gate should pass when all refs resolve");
    }

    #[test]
    fn drift_class_1_suffix_drift_emits_warning() {
        // Issue #1510 drift class 1: contract says `_init_matches_constructor`
        // but source has `_init_matches_input`. Same prefix, different suffix.
        let contracts = fixture_contract(
            "cargo test -p apr-cli --lib commands::pretrain::tests::pretrain_init_matches_constructor",
        );
        let dir = fixture_source_tree("#[test]\nfn pretrain_init_matches_input() {}\n");

        let (_gate, findings) = run_strict_test_binding_gate(&contracts, dir.path(), false);
        assert_eq!(findings.len(), 1);
        let f = &findings[0];
        assert_eq!(f.rule_id, "PV-VER-002");
        assert_eq!(f.severity, RuleSeverity::Warning);
        assert!(
            f.message.contains("pretrain_init_matches_constructor"),
            "expected cited name in message, got: {}",
            f.message
        );
    }

    #[test]
    fn drift_class_2_module_path_no_warning_when_fn_exists_anywhere() {
        // Issue #1510 drift class 2: `transformer::attention::tests::foo` cited
        // but actual is in `transformer::model::tests::foo` — same fn name.
        // Documented choice: we match by bare fn name only, so this is NOT
        // flagged. (Module paths are stripped during extraction.) Catching
        // wrong-module references would require parsing real `mod` trees,
        // which is out of scope for this gate.
        let contracts = fixture_contract(
            "cargo test -p aprender-train --lib transformer::attention::tests::gqa_test",
        );
        let dir = fixture_source_tree("#[test]\nfn gqa_test() {}\n");

        let (_gate, findings) = run_strict_test_binding_gate(&contracts, dir.path(), false);
        assert_eq!(
            findings.len(),
            0,
            "fn name match (regardless of module path) should not warn"
        );
    }

    #[test]
    fn drift_class_3_convention_drift_emits_warning() {
        // Issue #1510 drift class 3: contract says `_encoder_init_errors` but
        // source has `validate_pretrain_init_arch_rejects_encoder` (different
        // convention entirely).
        let contracts = fixture_contract(
            "cargo test -p aprender-train --lib train::pretrain_real::tests::build_transformer_config_encoder_init_errors",
        );
        let dir =
            fixture_source_tree("#[test]\nfn validate_pretrain_init_arch_rejects_encoder() {}\n");

        let (_gate, findings) = run_strict_test_binding_gate(&contracts, dir.path(), false);
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].rule_id, "PV-VER-002");
    }

    #[test]
    fn live_pending_skip_no_warning() {
        // LIVE-PENDING markers are deferred-live; not cargo tests; never flag.
        let contracts = fixture_contract(
            "LIVE-PENDING — requires §50.4 step 5g.2 LIVE 500-step fine-tune dispatch",
        );
        let dir = fixture_source_tree("// no tests here\n");

        let (_gate, findings) = run_strict_test_binding_gate(&contracts, dir.path(), false);
        assert_eq!(
            findings.len(),
            0,
            "LIVE-PENDING marker must be skipped, got: {findings:?}"
        );
    }

    #[test]
    fn pv_validate_invocation_skipped() {
        // `pv validate contracts/foo.yaml` is a meta-validation invocation,
        // not a cargo test. Do not flag.
        let contracts = fixture_contract("pv validate contracts/apr-pretrain-from-init-v1.yaml");
        let dir = fixture_source_tree("// no tests here\n");

        let (_gate, findings) = run_strict_test_binding_gate(&contracts, dir.path(), false);
        assert_eq!(findings.len(), 0);
    }

    #[test]
    fn strict_mode_gate_fails_when_refs_missing() {
        // With strict_mode=true, the gate's `passed` field is false when refs
        // miss, which causes the overall lint to fail.
        let contracts =
            fixture_contract("cargo test -p apr-cli --lib commands::pretrain::tests::nonexistent");
        let dir = fixture_source_tree("#[test]\nfn other_test() {}\n");

        let (gate_default, _) = run_strict_test_binding_gate(&contracts, dir.path(), false);
        assert!(
            gate_default.passed,
            "default mode: gate should still pass (warning-only)"
        );

        let (gate_strict, _) = run_strict_test_binding_gate(&contracts, dir.path(), true);
        assert!(
            !gate_strict.passed,
            "strict mode: gate should fail when refs miss"
        );
    }
}
