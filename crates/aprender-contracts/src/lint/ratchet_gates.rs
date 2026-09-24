//! PVL-001 §EV-11 — two shrink-only ratchets over `contracts/lint-baseline.json`.
//!
//! - `theorem-pairing` (PV-RAT-001): a Lean THEOREM MODULE is a `.lean` file under
//!   `<lean base>/ProvableContracts/Theorems/`, named by its dotted path from the base
//!   (`ProvableContracts.Theorems.Softmax.PartitionOfUnity`). It is PAIRED when that full name appears, on
//!   identifier boundaries, in a `.md` file under `book/` or `crates/aprender-contracts-staging/book/` of the repo
//!   root (the contract dir's parent). The debt `unpaired_theorem_modules` may not rise.
//! - `depends-on-present` (PV-RAT-002): a kernel-kind contract (the parsed kind, registries excluded — the class
//!   the valid-under gate obliges) with an empty `metadata.depends_on`. The debt `contracts_without_depends_on`
//!   may not rise.
//!
//! THE PAIRING RULE IS STRICT ON PURPOSE. The spec row says "module name appears in `book/`"; the full dotted
//! name is the only reading that cannot be satisfied by accident — the book mentions the file STEM of 80 of the
//! 165 `.lean` files (measured 2026-09-24), mostly as ordinary words, and the full module name of 1 of 131. The root module and the lakefile are not
//! theorem modules, so they are not counted either way.
//!
//! THE GATES NEVER WRITE. At or below the baseline they PASS and report the count; lowering the recorded number
//! is `make lint-ratchet`'s job (never in CI). No baseline key is `Unknown(Report)`, never a pass: the whole
//! verdict is the comparison, so the count is REPORTED (that is how `make lint-ratchet` records the first
//! baseline) and not judged, and `--gate` exits 2. A Lean base with no theorem module, a repo with no book, or a
//! corpus with no kernel-kind contract measured nothing and declines outright (ONT R-2: zero is a decline).

use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::ontology::extract::lean::base_under;
use crate::schema::{parse_contract, ContractKind};

use super::finding::LintFinding;
use super::rules::RuleSeverity;
use super::{GateDetail, GateExtra, GateResult, Verdict};
use crate::ontology::verdict::Reason;

/// Top-level `lint-baseline.json` key for the theorem-pairing debt.
pub const UNPAIRED_KEY: &str = "unpaired_theorem_modules";
/// Top-level `lint-baseline.json` key for the depends_on debt.
pub const WITHOUT_DEPENDS_ON_KEY: &str = "contracts_without_depends_on";
/// The book roots a theorem module is paired against, relative to the repo root.
pub const BOOK_ROOTS: [&str; 2] = ["book", "crates/aprender-contracts-staging/book"];

/// What one ratchet gate run answers. Only [`RatchetOutcome::Ran`] is a verdict about the corpus.
#[derive(Debug)]
pub enum RatchetOutcome {
    /// Nothing was measured, or nothing can be compared: the reason, for stderr (ONT R-2: zero is a decline).
    Declined(String),
    /// The gate measured and compared.
    Ran {
        result: Box<GateResult>,
        findings: Vec<LintFinding>,
    },
}

/// The top-level integer `key` of `<contract_dir>/lint-baseline.json`. Absent, unreadable, or not a
/// non-negative integer → `None`.
fn baseline_of(contract_dir: &Path, key: &str) -> Option<usize> {
    let raw = std::fs::read_to_string(contract_dir.join("lint-baseline.json")).ok()?;
    let doc: serde_json::Value = serde_json::from_str(&raw).ok()?;
    doc.get(key)?.as_u64().and_then(|n| usize::try_from(n).ok())
}

/// The repo root the contract dir sits in: its parent, or `.` for a bare relative name.
fn repo_root(contract_dir: &Path) -> PathBuf {
    match contract_dir.parent() {
        Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
        _ => PathBuf::from("."),
    }
}

/// Every file under `dir` with extension `ext`, recursively, in byte order. Unreadable directories are skipped.
fn files_with_ext(dir: &Path, ext: &str, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            files_with_ext(&path, ext, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some(ext) {
            out.push(path);
        }
    }
}

/// `<base>/ProvableContracts/Theorems/A/B.lean` → `ProvableContracts.Theorems.A.B`.
fn module_name(base: &Path, file: &Path) -> Option<String> {
    let rel = file.strip_prefix(base).ok()?.with_extension("");
    let parts: Vec<&str> = rel
        .components()
        .filter_map(|c| c.as_os_str().to_str())
        .collect();
    Some(parts.join("."))
}

fn is_ident_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_' || c == '\''
}

/// Does `text` contain `name` with identifier boundaries on both sides? A trailing `.` counts as a boundary
/// only when no identifier follows it, so `…Softmax.Partition` does not pair `…Softmax.PartitionOfUnity` or a
/// deeper module, and a sentence ending in the module name still pairs it.
#[must_use]
pub fn mentions_module(text: &str, name: &str) -> bool {
    let mut from = 0;
    while let Some(off) = text[from..].find(name) {
        let start = from + off;
        let end = start + name.len();
        let before_ok = text[..start]
            .chars()
            .next_back()
            .is_none_or(|c| !is_ident_char(c) && c != '.');
        let mut after = text[end..].chars();
        let after_ok = match after.next() {
            None => true,
            Some('.') => after.next().is_none_or(|c| !is_ident_char(c)),
            Some(c) => !is_ident_char(c),
        };
        if before_ok && after_ok {
            return true;
        }
        from = start + name.chars().next().map_or(1, char::len_utf8);
    }
    false
}

/// PV-RAT-001 / PV-RAT-002: the debt may fall, never rise. No baseline → no finding (the verdict says why).
fn ratchet_finding(
    rule: &str,
    key: &str,
    baseline: Option<usize>,
    now: usize,
    fix: &str,
) -> Option<LintFinding> {
    let baseline = baseline?;
    if now <= baseline {
        return None;
    }
    let mut f = LintFinding::new(
        rule,
        RuleSeverity::Error,
        format!(
            "{key} rose {baseline} -> {now}: {fix}. The baseline in contracts/lint-baseline.json is shrink-only; `make lint-ratchet` only lowers it"
        ),
        "contracts/lint-baseline.json".to_string(),
    );
    f.contract_stem = Some("lint-baseline".to_string());
    Some(f)
}

fn result_of(
    name: &str,
    checked: usize,
    findings: &[LintFinding],
    baseline: Option<usize>,
    start: Instant,
    extra: GateExtra,
) -> Box<GateResult> {
    let violations = findings.len();
    let judged = baseline.is_some();
    let passed = judged && violations == 0;
    Box::new(GateResult {
        name: name.into(),
        passed,
        skipped: !judged,
        verdict: if judged {
            Verdict::from_gate(passed, false)
        } else {
            Verdict::Unknown(Reason::Report)
        },
        duration_ms: u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX),
        // `GateDetail` is FROZEN at the 0.3.1 variants (see `GateExtra`); the payload rides in `GateExtra`.
        detail: GateDetail::Validate {
            contracts: checked,
            errors: violations,
            warnings: 0,
            error_messages: findings.iter().map(|f| f.message.clone()).collect(),
        },
        extra: Some(extra),
    })
}

/// The `theorem-pairing` gate over `contract_dir`'s repo root.
#[must_use]
pub fn run_theorem_pairing_gate(contract_dir: &Path) -> RatchetOutcome {
    let start = Instant::now();
    let root = repo_root(contract_dir);
    let Some(base) = base_under(&root) else {
        return RatchetOutcome::Declined(format!(
            "no Lean theorem base (ProvableContracts/Theorems) under {}",
            root.display()
        ));
    };
    let mut lean = Vec::new();
    files_with_ext(&base.join("ProvableContracts/Theorems"), "lean", &mut lean);
    lean.sort();
    let modules: Vec<String> = lean.iter().filter_map(|f| module_name(&base, f)).collect();
    if modules.is_empty() {
        return RatchetOutcome::Declined(format!("no theorem module under {}", base.display()));
    }
    let books: Vec<PathBuf> = BOOK_ROOTS
        .iter()
        .map(|b| root.join(b))
        .filter(|p| p.is_dir())
        .collect();
    let mut pages = Vec::new();
    for b in &books {
        files_with_ext(b, "md", &mut pages);
    }
    if pages.is_empty() {
        return RatchetOutcome::Declined(format!(
            "no book page (*.md) under {} of {}: nothing to pair {} theorem module(s) against",
            BOOK_ROOTS.join(" or "),
            root.display(),
            modules.len()
        ));
    }
    let baseline = baseline_of(contract_dir, UNPAIRED_KEY);
    let texts: Vec<String> = pages
        .iter()
        .filter_map(|p| std::fs::read_to_string(p).ok())
        .collect();
    let unpaired: Vec<String> = modules
        .iter()
        .filter(|m| !texts.iter().any(|t| mentions_module(t, m)))
        .cloned()
        .collect();
    let findings: Vec<LintFinding> = ratchet_finding(
        "PV-RAT-001",
        UNPAIRED_KEY,
        baseline,
        unpaired.len(),
        "a Lean theorem module was added that no book page names — cite its full module name in book/",
    )
    .into_iter()
    .collect();
    let extra = GateExtra::TheoremPairing {
        lean_base: base.display().to_string(),
        book_pages: pages.len(),
        theorem_modules: modules.len(),
        paired: modules.len() - unpaired.len(),
        unpaired_theorem_modules: unpaired.len(),
        baseline,
        unpaired,
        violations: findings.len(),
    };
    RatchetOutcome::Ran {
        result: result_of(
            "theorem-pairing",
            modules.len(),
            &findings,
            baseline,
            start,
            extra,
        ),
        findings,
    }
}

/// The `depends-on-present` gate over `contract_dir`.
#[must_use]
pub fn run_depends_on_present_gate(contract_dir: &Path) -> RatchetOutcome {
    let start = Instant::now();
    let mut files = Vec::new();
    super::collect_yaml_files(contract_dir, &mut files);
    files.sort();
    let (mut checked, mut kernels, mut without) = (0usize, 0usize, 0usize);
    for file in &files {
        let Ok(contract) = parse_contract(file) else {
            continue;
        };
        checked += 1;
        if contract.kind() == ContractKind::Kernel && !contract.is_registry() {
            kernels += 1;
            without += usize::from(contract.metadata.depends_on.is_empty());
        }
    }
    if kernels == 0 {
        return RatchetOutcome::Declined(format!(
            "no kernel-kind contract in {checked} contract(s): nothing was measured"
        ));
    }
    let baseline = baseline_of(contract_dir, WITHOUT_DEPENDS_ON_KEY);
    let findings: Vec<LintFinding> = ratchet_finding(
        "PV-RAT-002",
        WITHOUT_DEPENDS_ON_KEY,
        baseline,
        without,
        "a kernel-kind contract with an empty `metadata.depends_on` was added — name what it composes",
    )
    .into_iter()
    .collect();
    let extra = GateExtra::DependsOnPresent {
        contracts_checked: checked,
        kernel_contracts: kernels,
        contracts_without_depends_on: without,
        baseline,
        violations: findings.len(),
    };
    RatchetOutcome::Ran {
        result: result_of(
            "depends-on-present",
            checked,
            &findings,
            baseline,
            start,
            extra,
        ),
        findings,
    }
}

#[cfg(test)]
#[path = "ratchet_gates_tests.rs"]
mod tests;
