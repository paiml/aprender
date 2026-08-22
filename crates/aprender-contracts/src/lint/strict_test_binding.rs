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
//!
//! # #2465 — the gate skipped the entries it was supposed to bind
//!
//! The original implementation read `falsification_tests[].test` and `continue`d
//! on `None`. Two things made that a blind spot rather than a narrow scope:
//!
//!   * `FalsificationTest` had no `test_harness` field at all, and the struct is
//!     not `deny_unknown_fields`, so serde dropped it. 619 of 4206 entries in
//!     `contracts/` name their test ONLY in `test_harness:`/`name:` — including
//!     94 holding a real `cargo test …` invocation. Every one arrived with
//!     `test: None` and was skipped.
//!   * Skipped is indistinguishable from bound in the output: the gate counted
//!     neither a ref nor a miss, so a contract citing a test that does not exist
//!     anywhere read as clean.
//!
//! Reproduced by replacing a real test name in `lora-adapter-trains-base-frozen-v1`
//! with `MUTANT_this_test_fn_does_not_exist_anywhere`: the gate stayed at
//! 253 refs / 51 missing, i.e. it did not notice.
//!
//! The fix resolves a binding source per entry — `test:`, then `test_harness:`,
//! and `name:` only when neither of those exists — and classifies the string
//! before checking it (see [`BindingKind`]). Classification is what keeps the
//! 525 genuine SHELL harnesses (`grep -q 'apr monitor' book/src/cli/monitor.md`,
//! `test -f …`, `bash …`) from being reported as dangling Rust tests: those
//! entries declare a shell mechanism, so their `name:` (`stub_exists`,
//! `module_mentioned`, `runnable_example` — all valid Rust identifiers, none a
//! Rust test) is never consulted.

use std::collections::HashSet;
use std::path::Path;
use std::time::Instant;

use crate::schema::Contract;

use super::finding::LintFinding;
use super::rules::RuleSeverity;
use super::{GateDetail, GateResult};

/// Which field of a `falsification_tests[]` entry a binding claim was read from.
///
/// Rendered into the finding message so an operator knows which line to edit —
/// `.test`, `.test_harness` and `.name` are three different spellings of the
/// same claim and the fix differs per field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BindingField {
    Test,
    TestHarness,
    Name,
}

impl BindingField {
    fn as_yaml_key(self) -> &'static str {
        match self {
            Self::Test => "test",
            Self::TestHarness => "test_harness",
            Self::Name => "name",
        }
    }
}

/// What a binding string actually names.
///
/// #2465: the distinction that matters is [`ShellHarness`](BindingKind::ShellHarness)
/// vs [`BareRustFn`](BindingKind::BareRustFn). `contracts/` holds 525 entries
/// whose `test_harness:` is a shell command and whose `name:` is a slug that
/// happens to be a valid Rust identifier (`stub_exists`, `runnable_example`).
/// Resolving those against `#[test]` fns would manufacture 525 false dangling
/// references, so a shell harness ends the resolution — the entry has declared
/// its mechanism and it is not cargo.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BindingKind {
    /// A `cargo test …` invocation; cited filters are checked against source.
    CargoTest,
    /// A bare Rust identifier naming a test fn directly (`name:` style).
    BareRustFn,
    /// A shell harness — `grep -q …`, `test -f …`, `bash …`, `! grep …`.
    /// Names a shell command, never a Rust test. Never flagged.
    ShellHarness,
    /// `LIVE-PENDING …`, `pv validate …`, prose, or anything else from which
    /// no Rust test name can be resolved.
    Unbindable,
}

/// Shell commands that can legitimately open a `test_harness:`.
///
/// Derived from the actual first-token histogram over `contracts/` (#2465):
/// `grep` 344, `test` 172, `bash` 7, `!` 2 — plus the neighbours that would
/// read identically if someone wrote one.
const SHELL_HARNESS_COMMANDS: &[&str] = &[
    "!", "[", "test", "grep", "rg", "bash", "sh", "zsh", "find", "cat", "ls", "awk", "sed", "jq",
    "diff", "cmp", "wc", "head", "tail", "python", "python3", "make", "git", "curl", "docker",
    "apr", "pmat", "bashrs", "echo", "true", "false",
];

/// Classify a binding string. See [`BindingKind`].
///
/// Order is load-bearing: the shell test runs BEFORE the bare-identifier test,
/// so `test -f book/src/cli/monitor.md` classifies as a shell harness rather
/// than having its leading `test` token mistaken for an identifier.
pub(crate) fn classify_binding(raw: &str) -> BindingKind {
    let s = raw.trim().trim_matches('"').trim();
    if s.is_empty() {
        return BindingKind::Unbindable;
    }
    if s.starts_with("LIVE-PENDING") {
        return BindingKind::Unbindable;
    }
    if s.contains("cargo test") {
        return BindingKind::CargoTest;
    }
    if s.starts_with("pv ") || s.starts_with("pv\t") {
        return BindingKind::Unbindable;
    }
    let first = s.split_whitespace().next().unwrap_or("");
    if SHELL_HARNESS_COMMANDS.contains(&first)
        || first.starts_with("./")
        || first.starts_with('/')
        || first.contains('/')
    {
        return BindingKind::ShellHarness;
    }
    // A single token that is a legal Rust identifier is a direct fn reference.
    // Anything with whitespace left at this point is prose.
    if s.split_whitespace().count() == 1 && looks_like_rust_ident(s) {
        return BindingKind::BareRustFn;
    }
    BindingKind::Unbindable
}

/// Resolve the binding sources for one falsification-test entry, in priority
/// order.
///
/// `test:` and `test_harness:` are both *declarations of how to run the test*,
/// so both are checked when both are present. `name:` is a fallback used ONLY
/// when neither exists — when a harness is present it has already declared the
/// mechanism, and consulting `name:` anyway is exactly what would light up the
/// 525 shell-harness slugs.
pub(crate) fn binding_sources(
    ft: &crate::schema::FalsificationTest,
) -> Vec<(BindingField, String)> {
    let mut out = Vec::new();
    if let Some(t) = ft.test.as_ref().filter(|t| !t.trim().is_empty()) {
        out.push((BindingField::Test, t.clone()));
    }
    if let Some(h) = ft.test_harness.as_ref().filter(|h| !h.trim().is_empty()) {
        out.push((BindingField::TestHarness, h.clone()));
    }
    if out.is_empty() {
        if let Some(n) = ft.name.as_ref().filter(|n| !n.trim().is_empty()) {
            out.push((BindingField::Name, n.clone()));
        }
    }
    out
}

/// Extract the Rust test-fn names a binding string claims, given its kind.
fn cited_names(kind: BindingKind, raw: &str) -> Vec<String> {
    let s = raw.trim().trim_matches('"').trim();
    match kind {
        BindingKind::CargoTest => extract_cited_fn_names(s),
        BindingKind::BareRustFn => vec![s.to_string()],
        BindingKind::ShellHarness | BindingKind::Unbindable => Vec::new(),
    }
}

/// Gate 9: Strict test-binding — verify every cited test reference exists in source.
///
/// Walks every `falsification_tests[]` entry, resolves its binding source
/// (`test:` → `test_harness:` → `name:`), extracts the test fn name(s), and
/// verifies each exists in the source tree under a `#[test]` (or
/// `#[tokio::test]`, etc.) attribute.
///
/// Skipped categories (not flagged):
///   - `LIVE-PENDING:` prefix — explicit deferred-live marker
///   - `pv validate ...` — meta-validation invocation, not a unit test
///   - shell harnesses (`grep -q …`, `test -f …`, `bash …`) — these name a
///     shell command, not a Rust test (#2465)
///   - entries with no binding field at all — covered by other gates
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

    let index = scan_source_index(project_root);

    for (stem, contract) in contracts {
        for ft in &contract.falsification_tests {
            for (field, raw) in binding_sources(ft) {
                let kind = classify_binding(&raw);
                for cited in cited_names(kind, &raw) {
                    total_refs += 1;
                    if index.resolves(&cited) {
                        continue;
                    }
                    missing += 1;
                    let key = field.as_yaml_key();
                    let mut f = LintFinding::new(
                        "PV-VER-002",
                        RuleSeverity::Warning,
                        format!(
                            "Dangling test reference: cited `{cited}` not found in source \
                             (falsification_tests[{}].{key})",
                            ft.id
                        ),
                        format!("contracts/{stem}.yaml"),
                    );
                    f.contract_stem = Some(stem.clone());
                    f.suggestion = Some(format!(
                        "Either rename a test fn to `{cited}`, or update the contract \
                         `{key}:` field for {} to cite the real fn name.",
                        ft.id
                    ));
                    findings.push(f);
                }
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
            extra: None,
        },
        findings,
    )
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
    let leg = truncate_at_shell_plumbing(leg);
    let tokens: Vec<&str> = leg.split_whitespace().collect();
    let filter = last_positional_filter(&tokens)?;
    // Trim quotes the user may have left in.
    let filter = filter.trim_matches('"').trim_matches('\'');
    // Last `::` segment is the bare fn name.
    let bare = filter.rsplit("::").next().unwrap_or(filter).to_string();
    if !looks_like_rust_ident(&bare) {
        return None;
    }
    Some(bare)
}

/// Drop everything from the first shell pipe or redirect onward — past those
/// lies a downstream tool (grep/awk), not a cargo test filter.
fn truncate_at_shell_plumbing(leg: &str) -> &str {
    let leg = leg
        .split_once(" | ")
        .map_or(leg, |(pre, _)| pre)
        .split_once(" 2>&1 ")
        .map_or_else(
            || leg.split_once(" | ").map_or(leg, |(pre, _)| pre),
            |(pre, _)| pre,
        );
    leg.split_once(" > ")
        .map_or(leg, |(pre, _)| pre)
        .split_once(" 2>&1")
        .map_or(leg, |(pre, _)| pre)
}

/// Cargo flags that consume a following value token.
const CARGO_FLAGS_WITH_ARG: &[&str] = &[
    "-p",
    "--package",
    "--test",
    "--bin",
    "--example",
    "--features",
    "-F",
    "--target",
    "--manifest-path",
];

/// Index of the token after `cargo test`, or 0 when that pair is absent.
fn args_start_index(tokens: &[&str]) -> usize {
    tokens
        .iter()
        .enumerate()
        .find(|(idx, t)| **t == "test" && *idx > 0 && tokens[idx - 1] == "cargo")
        .map_or(0, |(idx, _)| idx + 1)
}

/// Walk `cargo test`'s argument tokens and return the filter positional.
///
/// After a `--` separator the very next positional IS the filter and the walk
/// stops (later tokens are libtest runtime args such as `--test-threads`).
/// Without a `--`, the last positional wins.
fn last_positional_filter(tokens: &[&str]) -> Option<String> {
    let mut i = args_start_index(tokens);
    let mut last_filter: Option<String> = None;
    let mut saw_double_dash = false;
    while i < tokens.len() {
        let tok = tokens[i];
        if tok == "--" {
            saw_double_dash = true;
            i += 1;
        } else if CARGO_FLAGS_WITH_ARG.contains(&tok) {
            i += 2; // skip flag + its value
        } else if tok.starts_with('-') {
            i += 1; // any other flag, incl. the bare `--lib` family
        } else if matches!(tok, "&&" | "||" | ";" | "|") {
            break; // shell residue
        } else {
            last_filter = Some(tok.to_string());
            i += 1;
            if saw_double_dash {
                break;
            }
        }
    }
    last_filter
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

/// What the source tree offers as a resolution target for a cited filter.
///
/// `cargo test <filter>` matches the filter as a SUBSTRING of each test's full
/// path, so `cargo test -p apr-cli --lib commands::serve::ollama` is satisfied
/// by the module `ollama` containing tests — no fn is named `ollama`. Resolving
/// against fn names alone therefore reports live, working bindings as dangling:
/// on the real `contracts/` tree that was 12 of the 27 newly-visible refs
/// (`ollama`, `softcap_tests`, `gemma_config_tests`, `pmat754_stop_truncation_tests`,
/// …), every one of them a real `mod` (#2465).
#[derive(Debug, Default)]
pub(crate) struct SourceIndex {
    /// Names of fns carrying a test attribute (or the legacy `test_`/`prop_` prefix).
    pub(crate) test_fns: HashSet<String>,
    /// Names of declared modules — a legal `cargo test` filter segment.
    pub(crate) modules: HashSet<String>,
}

impl SourceIndex {
    /// True iff `cargo test <cited>` would select at least one real test.
    ///
    /// `cargo test` treats its filter as a SUBSTRING of each test's full
    /// `module::path::fn_name`, not as an exact name. Exact matching therefore
    /// reports working bindings as dangling: `accepts_summary` (a real filter
    /// for `accepts_summary_names_the_real_quantize_surface`) and
    /// `streamed_chat_body` (for `streamed_chat_body_carries_the_same_text_…`)
    /// both ran green in CI while the gate called them missing.
    ///
    /// The question a binding gate must answer is "does the command in this
    /// contract run a real test?", so the check follows cargo. A citation that
    /// is a substring of nothing — `MUTANT_this_test_fn_does_not_exist_anywhere`
    /// — still fails, which is the property the gate exists for.
    pub(crate) fn resolves(&self, cited: &str) -> bool {
        if self.test_fns.contains(cited) || self.modules.contains(cited) {
            return true;
        }
        self.test_fns.iter().any(|f| f.contains(cited))
            || self.modules.iter().any(|m| m.contains(cited))
    }
}

/// Scan the project for test fn names and module names.
fn scan_source_index(project_root: &Path) -> SourceIndex {
    let mut found = SourceIndex::default();
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
    scan_root_level_members(effective_root, &mut found);
    found
}

/// Scan workspace members that sit at the checkout root (e.g. `trueno-gpu/src/`)
/// rather than under `crates/`.
fn scan_root_level_members(root: &Path, found: &mut SourceIndex) {
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() || !path.join("Cargo.toml").exists() {
            continue;
        }
        let name = path.file_name().unwrap_or_default();
        if name == "src" || name == "crates" || name == "tests" {
            continue; // already covered by the top-level sweep
        }
        for sub in &["src", "tests"] {
            let d = path.join(sub);
            if d.exists() {
                scan_test_fns(&d, found);
            }
        }
    }
}

/// Recursively scan a directory for `.rs` files and harvest test fn + module names.
///
/// A function is considered a test if:
///   - it is preceded (within the previous ~3 non-blank lines) by `#[test]`,
///     `#[tokio::test]`, `#[async_std::test]`, `#[serial_test::serial]`,
///     `#[rstest]`, `#[proptest::proptest]`, or `proptest!{ ... }`; OR
///   - its name starts with `test_` or `prop_` (legacy convention)
fn scan_test_fns(dir: &Path, index: &mut SourceIndex) {
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
            scan_test_fns(&path, index);
        } else if path.extension().is_some_and(|e| e == "rs") {
            if let Ok(content) = std::fs::read_to_string(&path) {
                harvest_test_fns(&content, &mut index.test_fns);
                harvest_module_names(&content, &mut index.modules);
            }
        }
    }
}

/// Walk a Rust source string and insert the names of declared modules.
///
/// Matches `mod x;`, `mod x {`, and their `pub` / `pub(crate)` forms. A module
/// name is a legal `cargo test` filter segment (see [`SourceIndex`]).
pub(crate) fn harvest_module_names(content: &str, mods: &mut HashSet<String>) {
    for line in content.lines() {
        let t = line.trim();
        if t.starts_with("//") {
            continue;
        }
        let rest = t
            .strip_prefix("pub(crate) mod ")
            .or_else(|| t.strip_prefix("pub(super) mod "))
            .or_else(|| t.strip_prefix("pub mod "))
            .or_else(|| t.strip_prefix("mod "));
        let Some(rest) = rest else { continue };
        let name: String = rest
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        if !name.is_empty() {
            mods.insert(name);
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
            if is_test_fn(&name, last_was_test_attr) {
                tests.insert(name);
            }
        }
        // A fn line, or any other non-attribute line, resets the flag.
        last_was_test_attr = false;
    }
}

/// Two harvest paths:
///   (a) attribute-driven: a prior `#[test]` / `#[tokio::test]` / …
///   (b) prefix-driven: the legacy `test_*` / `prop_*` naming convention
fn is_test_fn(name: &str, preceded_by_test_attr: bool) -> bool {
    preceded_by_test_attr || name.starts_with("test_") || name.starts_with("prop_")
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
    // classify_binding — the decision function (#2465)
    //
    // Every row here is drawn from a real string in `contracts/`. This is the
    // case table: re-run it, do not re-read the classifier.
    // ------------------------------------------------------------------

    #[test]
    fn skip_live_pending_marker() {
        assert_eq!(
            classify_binding("LIVE-PENDING — requires fixture"),
            BindingKind::Unbindable
        );
        assert_eq!(
            classify_binding("LIVE-PENDING: GPU smoke"),
            BindingKind::Unbindable
        );
    }

    #[test]
    fn skip_pv_validate_invocation() {
        assert_eq!(
            classify_binding("pv validate contracts/foo.yaml"),
            BindingKind::Unbindable
        );
    }

    #[test]
    fn detect_basic_cargo_test() {
        assert_eq!(
            classify_binding("cargo test -p apr-cli --lib commands::pretrain::tests::foo"),
            BindingKind::CargoTest
        );
    }

    #[test]
    fn classify_shell_harnesses_from_contracts() {
        // The four shapes that actually occur as `test_harness:` in contracts/.
        // Misclassifying any of these as a Rust fn manufactures a false
        // dangling reference, which is why this table exists.
        for raw in [
            "grep -q 'apr monitor' book/src/cli/monitor.md",
            "test -f book/src/cli/stamp.md",
            "grep -c '^```bash' book/src/cli/showcase.md",
            "bash scripts/check_beats_gated.sh",
            "! grep -rn 'eprintln' crates/aprender-serve/src",
        ] {
            assert_eq!(
                classify_binding(raw),
                BindingKind::ShellHarness,
                "must classify as shell harness: {raw}"
            );
        }
    }

    #[test]
    fn classify_bare_rust_fn_name() {
        assert_eq!(
            classify_binding("falsify_lora_adapter_trains_to_decreasing_loss_base_frozen"),
            BindingKind::BareRustFn
        );
    }

    #[test]
    fn classify_prose_name_is_unbindable() {
        // Real `name:` values from decode-gpu-resident-sampling-v1 and friends.
        // Prose must not be resolved as a fn name.
        assert_eq!(
            classify_binding("Token parity with pre-change decode"),
            BindingKind::Unbindable
        );
        assert_eq!(
            classify_binding("Stop-token latency bounded"),
            BindingKind::Unbindable
        );
    }

    #[test]
    fn classify_leading_test_token_is_shell_not_ident() {
        // Order dependence: `test` is both a shell builtin and a legal Rust
        // identifier prefix. `test -f X` must resolve as shell.
        assert_eq!(
            classify_binding("test -f book/src/lib/text.md"),
            BindingKind::ShellHarness
        );
        // ...while a fn literally named `test_something` stays an identifier.
        assert_eq!(classify_binding("test_something"), BindingKind::BareRustFn);
    }

    // ------------------------------------------------------------------
    // binding_sources — resolution priority (#2465)
    // ------------------------------------------------------------------

    #[test]
    fn name_is_not_consulted_when_a_harness_declares_the_mechanism() {
        // The 525-false-positive case: a shell harness whose `name:` is a slug
        // that happens to be a legal Rust identifier. `name:` must not be
        // reached, so the slug is never looked up as a test fn.
        let ft = FalsificationTest {
            id: "FALSIFY-PAGE-001".into(),
            test_harness: Some("grep -q 'apr monitor' book/src/cli/monitor.md".into()),
            name: Some("module_mentioned".into()),
            ..Default::default()
        };
        let sources = binding_sources(&ft);
        assert_eq!(sources.len(), 1);
        assert_eq!(sources[0].0, BindingField::TestHarness);
    }

    #[test]
    fn name_is_the_fallback_when_no_invocation_field_exists() {
        let ft = FalsificationTest {
            id: "FALSIFY-X-001".into(),
            name: Some("ship_010_full_discharge".into()),
            ..Default::default()
        };
        let sources = binding_sources(&ft);
        assert_eq!(sources.len(), 1);
        assert_eq!(sources[0].0, BindingField::Name);
    }

    #[test]
    fn empty_binding_fields_are_not_sources() {
        let ft = FalsificationTest {
            id: "FALSIFY-X-002".into(),
            test: Some("   ".into()),
            test_harness: Some(String::new()),
            name: Some("  ".into()),
            ..Default::default()
        };
        assert_eq!(binding_sources(&ft).len(), 0);
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
    fn harvest_module_names_covers_every_visibility_form() {
        let src = "
mod plain;
pub mod exported;
pub(crate) mod internal;
pub(super) mod parental;
mod inline_block {
    fn helper() {}
}
// mod commented_out;
struct NotAMod;
";
        let mut found = HashSet::new();
        harvest_module_names(src, &mut found);
        for expected in ["plain", "exported", "internal", "parental", "inline_block"] {
            assert!(found.contains(expected), "missing mod `{expected}`");
        }
        assert!(!found.contains("commented_out"));
        assert!(!found.contains("NotAMod"));
    }

    /// `cargo test commands::serve::ollama` is satisfied by the MODULE
    /// `ollama`; no fn carries that name. Reporting it as dangling is a false
    /// positive, and there were 12 such refs in `contracts/` (#2465).
    #[test]
    fn module_filter_resolves_without_a_matching_fn_name() {
        let contracts = fixture_contract("cargo test -p apr-cli --lib commands::serve::ollama");
        let dir =
            fixture_source_tree("pub mod ollama {\n#[test]\nfn created_at_is_rfc3339() {}\n}\n");

        let (gate, findings) = run_strict_test_binding_gate(&contracts, dir.path(), true);
        assert_eq!(
            findings.len(),
            0,
            "a real module is a resolvable cargo-test filter: {findings:?}"
        );
        assert!(gate.passed);
    }

    /// ...but a module that does NOT exist is still a dangling reference —
    /// the module lookup must not become a blanket amnesty.
    #[test]
    fn module_filter_naming_a_nonexistent_module_is_still_flagged() {
        let contracts =
            fixture_contract("cargo test -p apr-cli --lib commands::serve::MUTANT_no_such_module");
        let dir =
            fixture_source_tree("pub mod ollama {\n#[test]\nfn created_at_is_rfc3339() {}\n}\n");

        let (gate, findings) = run_strict_test_binding_gate(&contracts, dir.path(), true);
        assert_eq!(findings.len(), 1, "got: {findings:?}");
        assert!(!gate.passed);
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
            ..Default::default()
        });
        vec![("fixture".to_string(), c)]
    }

    /// Build a one-entry contract binding via `test_harness:` + `name:` and no
    /// `test:` — the 619-entry shape the gate used to skip wholesale (#2465).
    fn fixture_contract_harness(harness: &str, name: &str) -> Vec<(String, Contract)> {
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
            test: None,
            test_harness: Some(harness.into()),
            name: Some(name.into()),
            if_fails: "investigate".into(),
        });
        vec![("fixture".to_string(), c)]
    }

    /// Build a one-entry contract that names its test ONLY in `name:`.
    fn fixture_name_only(name: &str) -> Vec<(String, Contract)> {
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
            name: Some(name.into()),
            if_fails: "investigate".into(),
            ..Default::default()
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

    // ------------------------------------------------------------------
    // #2465 — the entries the gate used to skip
    // ------------------------------------------------------------------

    /// The verbatim reproduction from #2465, as a unit test.
    ///
    /// `lora-adapter-trains-base-frozen-v1` binds via `test_harness:` only.
    /// Before the fix this contract contributed ZERO refs, so pointing it at
    /// `MUTANT_this_test_fn_does_not_exist_anywhere` produced no finding and
    /// the gate's counts did not move.
    #[test]
    fn test_harness_cargo_invocation_citing_a_nonexistent_fn_is_flagged() {
        let contracts = fixture_contract_harness(
            "cargo test -p aprender-train --lib MUTANT_this_test_fn_does_not_exist_anywhere",
            "MUTANT_this_test_fn_does_not_exist_anywhere",
        );
        let dir = fixture_source_tree(
            "#[test]\nfn falsify_lora_adapter_trains_to_decreasing_loss_base_frozen() {}\n",
        );

        let (gate, findings) = run_strict_test_binding_gate(&contracts, dir.path(), true);
        assert_eq!(findings.len(), 1, "got: {findings:?}");
        assert_eq!(findings[0].rule_id, "PV-VER-002");
        assert!(
            findings[0].message.contains("test_harness"),
            "finding must name the offending YAML field, got: {}",
            findings[0].message
        );
        assert!(
            !gate.passed,
            "strict mode must fail on a dangling harness ref"
        );
    }

    #[test]
    fn test_harness_cargo_invocation_resolving_to_a_real_fn_is_clean() {
        let contracts = fixture_contract_harness(
            "cargo test -p aprender-train --lib lora_forward_backward_reaches_adapter_not_base",
            "lora_forward_backward_reaches_adapter_not_base",
        );
        let dir = fixture_source_tree(
            "#[test]\nfn lora_forward_backward_reaches_adapter_not_base() {}\n",
        );

        let (gate, findings) = run_strict_test_binding_gate(&contracts, dir.path(), true);
        assert_eq!(findings.len(), 0, "got: {findings:?}");
        assert!(gate.passed);
    }

    /// A shell harness names a shell command, not a Rust test. Its `name:`
    /// slug (`module_mentioned`) is a legal Rust identifier and matches no
    /// `#[test]` fn anywhere — resolving it would be a false positive, and
    /// there are 525 such entries in `contracts/`.
    #[test]
    fn shell_harness_is_not_reported_as_a_dangling_rust_test() {
        let contracts = fixture_contract_harness(
            "grep -q 'apr monitor' book/src/cli/monitor.md",
            "module_mentioned",
        );
        let dir = fixture_source_tree("#[test]\nfn something_unrelated() {}\n");

        let (gate, findings) = run_strict_test_binding_gate(&contracts, dir.path(), true);
        assert_eq!(
            findings.len(),
            0,
            "shell harness must not be resolved as a Rust test: {findings:?}"
        );
        assert!(gate.passed);
    }

    /// A `name:`-only entry whose name is a bare identifier IS a binding claim
    /// and is checked; a prose `name:` is not.
    #[test]
    fn name_only_binding_is_checked_when_it_is_an_identifier() {
        let dir = fixture_source_tree("#[test]\nfn something_unrelated() {}\n");

        let ident = fixture_name_only("ship_010_full_discharge_via_live_validate_manifest");
        let (gate, findings) = run_strict_test_binding_gate(&ident, dir.path(), true);
        assert_eq!(findings.len(), 1, "got: {findings:?}");
        assert!(
            findings[0].message.contains(".name)"),
            "finding must name the offending YAML field, got: {}",
            findings[0].message
        );
        assert!(!gate.passed);

        let prose = fixture_name_only("Token parity with pre-change decode");
        let (gate, findings) = run_strict_test_binding_gate(&prose, dir.path(), true);
        assert_eq!(
            findings.len(),
            0,
            "prose name must not be resolved as a fn: {findings:?}"
        );
        assert!(gate.passed);
    }

    // ------------------------------------------------------------------
    // The BLOCKING gate over the real contracts/ tree (#2465).
    //
    // This lives in the lib test suite on purpose. CI's `workspace-test` job
    // is a required status check and runs `--lib` across the workspace, so
    // this cannot go dark; a new `tests/*.rs` target, by contrast, is invisible
    // until someone adds it to the single physical line at ci.yml:317.
    // `scripts/check_contract_test_binding.sh` runs the same comparison through
    // `pv` for operators, and carries the must-flag/must-not-flag case table.
    // ------------------------------------------------------------------

    fn repo_root() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    /// Parse `scripts/contract_test_binding_baseline.txt` (`path<TAB>count`).
    fn read_baseline() -> std::collections::HashMap<String, usize> {
        let path = repo_root().join("scripts/contract_test_binding_baseline.txt");
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
            panic!(
                "read {}: {e}. The ratchet baseline is REQUIRED; a missing file is a \
                 missing measurement, not a pass. Regenerate with \
                 `bash scripts/check_contract_test_binding.sh --update-baseline`.",
                path.display()
            )
        });
        let mut out = std::collections::HashMap::new();
        for line in text.lines() {
            let line = line.trim_end();
            if line.is_empty() {
                continue;
            }
            let (p, c) = line
                .split_once('\t')
                .unwrap_or_else(|| panic!("baseline line is not `path<TAB>count`: {line:?}"));
            let n = c
                .parse::<usize>()
                .unwrap_or_else(|e| panic!("baseline count {c:?} is not a number: {e}"));
            out.insert(p.to_string(), n);
        }
        out
    }

    /// A contract may not cite more nonexistent tests than its baseline allows,
    /// and a contract absent from the baseline may cite none at all.
    #[test]
    fn no_contract_cites_more_nonexistent_tests_than_its_baseline() {
        let root = repo_root();
        let (contracts, parse_errors) =
            super::super::gates::load_contracts(&root.join("contracts"));
        assert!(
            parse_errors.is_empty(),
            "contracts failed to parse: {parse_errors:?}"
        );
        // Vacuity: a scan that finds nothing must not read as clean.
        assert!(
            contracts.len() > 1000,
            "only {} contracts loaded — the scan is broken, not the tree clean",
            contracts.len()
        );

        let (gate, findings) = run_strict_test_binding_gate(&contracts, &root, true);
        let GateDetail::Verify { total_refs, .. } = gate.detail else {
            panic!("strict-test-binding gate did not report Verify detail");
        };
        assert!(
            total_refs >= 250,
            "only {total_refs} test references resolved (floor 250) — the source scan is broken"
        );

        let baseline = read_baseline();
        let mut observed: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        for f in &findings {
            *observed.entry(f.file.clone()).or_default() += 1;
        }

        let mut violations: Vec<String> = observed
            .iter()
            .filter_map(|(file, count)| {
                let allowed = baseline.get(file).copied().unwrap_or(0);
                (*count > allowed).then(|| {
                    format!("{file}: {count} dangling test reference(s), baseline allows {allowed}")
                })
            })
            .collect();
        violations.sort();

        assert!(
            violations.is_empty(),
            "A contract cites a test that no `cargo test` invocation can run.\n\
             Fix the citation (or add the test); do NOT raise the baseline.\n{}",
            violations.join("\n")
        );
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
