//! Proof status report — cross-contract proof level assessment.
//!
//! Computes a hierarchical proof level (L1–L5) for each contract and
//! aggregates them into kernel equivalence classes that mirror the
//! `KernelOp` classification from apr-model-qa-playbook.
//!
//! Output is consumed by `pv proof-status` (text/JSON) and by the
//! playbook's `ProofBonus` MQS integration.

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::binding::{BindingRegistry, ImplStatus};
use crate::schema::Contract;

// ── Proof level hierarchy ─────────────────────────────────────────

/// Hierarchical proof assurance level.
///
/// Each level subsumes the ones below it:
/// - **L1** — Contract YAML exists with equations
/// - **L2** — Property tested (falsification tests cover obligations)
/// - **L3** — Kani bounded-model-checked
/// - **L4** — Lean 4 theorem proved
/// - **L5** — L4 + all bindings verified as implemented
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ProofLevel {
    /// Contract YAML exists with equations
    L1,
    /// Property tested via falsification tests
    L2,
    /// Kani bounded-model-checked
    L3,
    /// Lean 4 theorem proved
    L4,
    /// Lean proved and all bindings verified
    L5,
}

impl fmt::Display for ProofLevel {
    /// Format the proof level as its string label (L1 through L5)
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::L1 => "L1",
            Self::L2 => "L2",
            Self::L3 => "L3",
            Self::L4 => "L4",
            Self::L5 => "L5",
        };
        write!(f, "{s}")
    }
}

// ── Per-contract status ───────────────────────────────────────────

/// Proof status for a single contract.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContractProofStatus {
    /// Contract file stem (e.g. "softmax-kernel-v1")
    pub stem: String,
    /// Computed hierarchical proof level
    pub proof_level: ProofLevel,
    /// Number of proof obligations in the contract
    pub obligations: u32,
    /// Number of falsification tests defined
    pub falsification_tests: u32,
    /// Number of Kani bounded-model-checking harnesses
    pub kani_harnesses: u32,
    /// Number of obligations proved in Lean 4
    pub lean_proved: u32,
    /// Obligations grounded by a sorry-free in-tree Lean theorem the equation names (ONT-2a)
    pub lean_grounded: u32,
    /// The contract claims a Lean proof its tree does not ground: printed `self-declared`, excluded from L4
    pub l4_self_declared: bool,
    /// Number of bindings with `implemented` status
    pub bindings_implemented: u32,
    /// Total number of equation bindings
    pub bindings_total: u32,
}

// ── Kernel class summary ──────────────────────────────────────────

/// Summary of proof status for a kernel equivalence class.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelClassSummary {
    /// Kernel class identifier (A through E)
    pub label: String,
    /// Human-readable description of the kernel combination
    pub description: String,
    /// Contract stems belonging to this class
    pub contract_stems: Vec<String>,
    /// Lowest proof level among class members
    pub min_proof_level: ProofLevel,
    /// Whether all class members have full binding coverage
    pub all_bound: bool,
}

// ── Full report ───────────────────────────────────────────────────

/// Top-level proof status report, serializable to JSON.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofStatusReport {
    /// Report schema version for forward compatibility
    pub schema_version: String,
    /// Unix epoch timestamp when the report was generated
    pub timestamp: String,
    /// Per-contract proof status entries
    pub contracts: Vec<ContractProofStatus>,
    /// Kernel equivalence class summaries
    pub kernel_classes: Vec<KernelClassSummary>,
    /// ONT-2a andon: a self-declared L4 is excluded from the L4 total in this build
    pub l4_self_declared_excluded: bool,
    /// Aggregate totals across all contracts
    pub totals: ProofStatusTotals,
}

/// Aggregate totals across all contracts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofStatusTotals {
    /// Total number of contracts analyzed
    pub contracts: u32,
    /// Sum of proof obligations across all contracts
    pub obligations: u32,
    /// Sum of falsification tests across all contracts
    pub falsification_tests: u32,
    /// Sum of Kani harnesses across all contracts
    pub kani_harnesses: u32,
    /// Sum of Lean-proved obligations across all contracts
    pub lean_proved: u32,
    /// Sum of GROUNDED Lean-proved obligations across all contracts (ONT-2a)
    pub lean_grounded: u32,
    /// Contracts whose L4 claim is self-declared, and therefore excluded from L4 (ONT-2a)
    pub l4_self_declared: u32,
    /// Sum of implemented bindings across all contracts
    pub bindings_implemented: u32,
    /// Sum of total bindings across all contracts
    pub bindings_total: u32,
}

// ── Kernel class → contract stem mapping ──────────────────────────

/// Static mapping from kernel equivalence class to contract stems.
///
/// Mirrors the `KernelOp` classification from `apr-model-qa-playbook`:
/// - **A** — GQA + `RMSNorm` + `SiLU` + `SwiGLU` + `RoPE` (Llama/Mistral)
/// - **B** — MHA + `LayerNorm` + GELU + `AbsPos` (GPT-2/BERT)
/// - **C** — MHA + `LayerNorm` + GELU + `ALiBi` (BLOOM/MPT)
/// - **D** — `LayerNorm` + GELU + `SiLU` + GQA (Gemma)
/// - **E** — `RMSNorm` + `SwiGLU` + GQA (Qwen)
fn kernel_class_map() -> Vec<(&'static str, &'static str, &'static [&'static str])> {
    vec![
        (
            "A",
            "GQA+RMSNorm+SiLU+SwiGLU+RoPE",
            &[
                "rmsnorm-kernel-v1",
                "silu-kernel-v1",
                "swiglu-kernel-v1",
                "rope-kernel-v1",
                "gqa-kernel-v1",
                "softmax-kernel-v1",
                "matmul-kernel-v1",
            ],
        ),
        (
            "B",
            "MHA+LayerNorm+GELU+AbsPos",
            &[
                "layernorm-kernel-v1",
                "gelu-kernel-v1",
                "attention-kernel-v1",
                "softmax-kernel-v1",
                "matmul-kernel-v1",
                "absolute-position-v1",
            ],
        ),
        (
            "C",
            "MHA+LayerNorm+GELU+ALiBi",
            &[
                "layernorm-kernel-v1",
                "gelu-kernel-v1",
                "attention-kernel-v1",
                "softmax-kernel-v1",
                "alibi-kernel-v1",
                "matmul-kernel-v1",
            ],
        ),
        (
            "D",
            "LayerNorm+GELU+SiLU+GQA",
            &[
                "layernorm-kernel-v1",
                "gelu-kernel-v1",
                "silu-kernel-v1",
                "gqa-kernel-v1",
                "softmax-kernel-v1",
                "matmul-kernel-v1",
            ],
        ),
        (
            "E",
            "RMSNorm+SwiGLU+GQA",
            &[
                "rmsnorm-kernel-v1",
                "swiglu-kernel-v1",
                "gqa-kernel-v1",
                "softmax-kernel-v1",
                "matmul-kernel-v1",
            ],
        ),
    ]
}

// ── Core computation ──────────────────────────────────────────────

/// Returns `true` when EVERY proof obligation is discharged in Lean.
///
/// Strict per-obligation semantics — no fuzzy over-promotion. A contract is
/// Lean-proved (L4) only when the number of Lean-proved obligations plus the
/// explicitly not-applicable ones covers ALL proof obligations, and at least
/// one obligation is genuinely proved. The obligation total is
/// `proof_obligations.len()` — the SAME total used for L2/L3 — so a
/// `verification_summary` cannot manufacture L4 by understating the total.
///
/// Proof counts come from the contract's `verification_summary` when it makes a
/// positive claim (`l4_lean_proved > 0`); otherwise from a scan of the in-tree
/// sorry-free `.lean` theorems. Because the scan can only credit obligations it
/// resolves, partial coverage (the old "≥1 resolving ref → L4" over-promotion)
/// now correctly reports L3 instead of a full L4. Contracts with legitimately
/// N/A obligations (e.g. softmax-kernel-v1 = 5 proved + 4 N/A of 9) MUST declare
/// that in `verification_summary` — the scan path grants no N/A credit.
///
/// The grounding count is passed IN rather than scanned here.
///
/// The scan reads the Lean tree from paths relative to the process CWD, so inside a unit test it resolves
/// nothing and every fixture is ungrounded by construction — a suite in which the andon could withdraw
/// ALL credit for ever and no test would notice. Taking the count as a parameter is what keeps both sides
/// covered: `grounded == 0` is the andon, `grounded + not_applicable >= total` is the credit it still
/// grants. Production callers use the wrapper and get the scan.
#[must_use]
pub fn is_lean_proved_with_grounding(contract: &Contract, grounded: u32) -> bool {
    let total = contract.proof_obligations.len() as u32;
    if total == 0 {
        return false;
    }
    // ONT-001 ONT-2a (andon): a `verification_summary` is the contract talking about ITSELF, and until a
    // discharge summary exists to check it against (PVL EV-8b) the only grounding in this tree is a
    // sorry-free Lean theorem an equation names and `lean_theorem_names()` resolves. L4 is therefore
    // granted on the GROUNDED count alone; a claim with nothing under it is reported `self-declared` by
    // `is_l4_self_declared`, excluded from the L4 total, and never counted quietly. The `not_applicable`
    // credit still comes from the summary: it is a claim about APPLICABILITY, not about a proof, and it
    // is ONT-8's evidence block that will give it its own provenance.
    let not_applicable = contract
        .verification_summary
        .as_ref()
        .map_or(0, |vs| vs.l4_not_applicable);
    grounded > 0 && grounded + not_applicable >= total
}

/// Returns `true` when the contract CLAIMS a Lean proof its own tree does not ground.
///
/// The claim is the summary's `l4_lean_proved` plus its `l4_not_applicable` covering every obligation —
/// exactly the test that granted L4 before ONT-2a. The grounding is [`count_lean_theorems_for_contract`].
/// A contract that would have been L4 by its own summary and is not L4 by grounding is SELF-DECLARED: the
/// report prints the word beside it, the L4 total excludes it, and ONT-3b is where the credit is earned.
#[must_use]
pub fn is_l4_self_declared(contract: &Contract) -> bool {
    is_l4_self_declared_with_grounding(contract, count_lean_theorems_for_contract(contract))
}

/// [`is_l4_self_declared`] with the grounding count passed in, for the same reason.
#[must_use]
pub fn is_l4_self_declared_with_grounding(contract: &Contract, grounded: u32) -> bool {
    let total = contract.proof_obligations.len() as u32;
    if total == 0 {
        return false;
    }
    let Some(vs) = contract.verification_summary.as_ref() else {
        return false;
    };
    let claim_covers = vs.l4_lean_proved > 0 && vs.l4_lean_proved + vs.l4_not_applicable >= total;
    claim_covers && !is_lean_proved_with_grounding(contract, grounded)
}

/// Returns `true` when all bindings are implemented.
fn is_fully_bound(binding_status: Option<(u32, u32)>) -> bool {
    binding_status.is_some_and(|(implemented, total)| total > 0 && implemented == total)
}

/// Compute the proof level for a single contract.
///
/// Derivation rules (highest matching level wins):
/// - **L5**: every obligation Lean-proved AND all bindings implemented
/// - **L4**: every obligation Lean-proved — strict per-obligation coverage,
///   `proved + not_applicable >= proof_obligations.len()` with `proved > 0`
///   (partial coverage is NOT L4; see [`is_lean_proved`])
/// - **L3**: has Kani harnesses AND falsification tests cover obligations
/// - **L2**: falsification tests count >= obligations count
/// - **L1**: contract exists with equations
#[allow(clippy::cast_possible_truncation)]
pub fn compute_proof_level(contract: &Contract, binding_status: Option<(u32, u32)>) -> ProofLevel {
    compute_proof_level_with_grounding(
        contract,
        binding_status,
        count_lean_theorems_for_contract(contract),
    )
}

/// [`compute_proof_level`] with the grounding count passed in (ONT-2a).
#[allow(clippy::cast_possible_truncation)]
#[must_use]
pub fn compute_proof_level_with_grounding(
    contract: &Contract,
    binding_status: Option<(u32, u32)>,
    grounded: u32,
) -> ProofLevel {
    let total_obligations = contract.proof_obligations.len() as u32;
    let ft_count = contract.falsification_tests.len() as u32;
    let kani_count = contract.kani_harnesses.len() as u32;

    // Check L4/L5: Lean proved
    if is_lean_proved_with_grounding(contract, grounded) {
        return if is_fully_bound(binding_status) {
            ProofLevel::L5
        } else {
            ProofLevel::L4
        };
    }

    // Check L3: Kani + falsification
    let has_tests = ft_count >= total_obligations && total_obligations > 0;
    if kani_count > 0 && has_tests {
        return ProofLevel::L3;
    }

    // Check L2: falsification tests cover obligations
    if has_tests {
        return ProofLevel::L2;
    }

    // L1: contract exists with equations
    ProofLevel::L1
}

/// Directories scanned (relative to CWD) for sorry-free Lean theorem files,
/// in priority order. The IN-TREE staging tree is FIRST — it is the
/// post-APR-MONO source of truth and a superset of the external sibling, so
/// L4/L5 proof levels are reproducible on a fresh clone / CI without a
/// co-located `../provable-contracts` checkout. The bare `lean` and the
/// external sibling are kept as fallbacks for dev machines that still use them.
pub(crate) const LEAN_THEOREM_BASES: &[&str] = &[
    "crates/aprender-contracts-staging/lean",
    "lean",
    "../provable-contracts/lean",
];

/// Register the three naming forms a single label contributes: namespaced, bare, and lowercased.
fn insert_name_forms(names: &mut std::collections::HashSet<String>, label: &str) {
    names.insert(format!("Theorems.{label}"));
    names.insert(label.to_string());
    names.insert(label.to_lowercase());
}

/// `relu_nonneg` → `ReluNonneg`.
fn camel_case(snake: &str) -> String {
    snake
        .split('_')
        .map(|s| {
            let mut c = s.chars();
            match c.next() {
                None => String::new(),
                Some(f) => f.to_uppercase().chain(c).collect(),
            }
        })
        .collect()
}

/// `ReluNonneg` → `Relu`: the leading word of a CamelCase name.
fn first_camel_word(camel: &str) -> String {
    camel
        .chars()
        .enumerate()
        .take_while(|(i, c)| *i == 0 || !c.is_uppercase())
        .map(|(_, c)| c)
        .collect()
}

/// Register every `theorem <name>` a file declares, in the forms a contract may cite it by.
fn insert_theorem_names_from_content(names: &mut std::collections::HashSet<String>, content: &str) {
    for line in content.lines() {
        let Some(pos) = line.find("theorem ") else {
            continue;
        };
        let tname: String = line[pos + 8..]
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        if tname.is_empty() {
            continue;
        }
        let camel = camel_case(&tname);
        names.insert(format!("Theorems.{camel}"));
        names.insert(camel.clone());
        let first_word = first_camel_word(&camel);
        if first_word.len() >= 3 {
            names.insert(format!("Theorems.{first_word}"));
            names.insert(first_word);
        }
    }
}

/// Register the names contributed by one domain directory's sorry-free `.lean` files.
///
/// A file containing `sorry` contributes NOTHING: an admitted proof grounds no claim, which is the whole
/// reason this scan is the grounding ONT-2a trusts over a contract's own summary.
fn insert_domain_theorems(names: &mut std::collections::HashSet<String>, domain: &std::path::Path) {
    let domain_name = domain
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();
    let Ok(files) = std::fs::read_dir(domain) else {
        return;
    };
    for file in files.flatten() {
        let path = file.path();
        if path.extension().is_none_or(|e| e != "lean") {
            continue;
        }
        let Ok(content) = std::fs::read_to_string(&path) else {
            continue;
        };
        if content.contains("sorry") {
            continue;
        }
        let stem = path
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        insert_name_forms(names, &domain_name);
        insert_name_forms(names, &stem);
        insert_theorem_names_from_content(names, &content);
    }
}

/// Every theorem name one base directory contributes; empty when the base is absent.
fn scan_theorem_base(base: &str) -> std::collections::HashSet<String> {
    let mut names = std::collections::HashSet::new();
    let search_dir = std::path::Path::new(base).join("ProvableContracts/Theorems");
    if !search_dir.exists() {
        return names;
    }
    let Ok(domains) = std::fs::read_dir(&search_dir) else {
        return names;
    };
    for domain_entry in domains.flatten() {
        let path = domain_entry.path();
        if path.is_dir() {
            insert_domain_theorems(&mut names, &path);
        }
    }
    names
}

/// Build a set of all sorry-free Lean theorem names from the Theorems/ directory.
/// Scans once, caches the result in a thread-local for repeated calls.
fn lean_theorem_names() -> &'static std::collections::HashSet<String> {
    use std::sync::OnceLock;
    static CACHE: OnceLock<std::collections::HashSet<String>> = OnceLock::new();
    CACHE.get_or_init(|| {
        for base in LEAN_THEOREM_BASES {
            let names = scan_theorem_base(base);
            if !names.is_empty() {
                return names;
            }
        }
        std::collections::HashSet::new()
    })
}

/// Count Lean theorems for a contract by matching `lean_theorem` refs against
/// sorry-free `.lean` files in the Theorems/ directory.
fn count_lean_theorems_for_contract(contract: &Contract) -> u32 {
    let theorems = lean_theorem_names();
    let mut count = 0u32;
    for eq in contract.equations.values() {
        if let Some(ref theorem_ref) = eq.lean_theorem {
            let name = theorem_ref.trim().trim_matches('"');
            // Try exact match, then without prefix, then lowercase
            if theorems.contains(name)
                || theorems.contains(name.strip_prefix("Theorems.").unwrap_or(name))
                || theorems.contains(&name.to_lowercase())
            {
                count += 1;
            }
        }
    }
    count
}

/// Build a complete proof status report.
///
/// `contracts` is a list of `(stem, &Contract)` pairs.
/// `binding` is an optional binding registry for binding coverage.
/// `include_classes` controls whether kernel class summaries are generated.
#[allow(clippy::cast_possible_truncation)]
pub fn proof_status_report(
    contracts: &[(String, &Contract)],
    binding: Option<&BindingRegistry>,
    include_classes: bool,
) -> ProofStatusReport {
    let mut statuses = Vec::new();
    let mut totals = ProofStatusTotals {
        contracts: contracts.len() as u32,
        obligations: 0,
        falsification_tests: 0,
        kani_harnesses: 0,
        lean_proved: 0,
        lean_grounded: 0,
        l4_self_declared: 0,
        bindings_implemented: 0,
        bindings_total: 0,
    };

    for (stem, contract) in contracts {
        let contract_file = format!("{stem}.yaml");

        let obligations = contract.proof_obligations.len() as u32;
        let ft_count = contract.falsification_tests.len() as u32;
        let kani_count = contract.kani_harnesses.len() as u32;
        // The CLAIM (what the contract says about itself) and the GROUNDING (what the tree shows) are two
        // numbers since ONT-2a; before it, the first stood in for the second whenever it was non-zero.
        let lean_proved = contract
            .verification_summary
            .as_ref()
            .map_or(0, |vs| vs.l4_lean_proved);
        let lean_grounded = count_lean_theorems_for_contract(contract);
        let lean_proved = if lean_proved == 0 {
            lean_grounded
        } else {
            lean_proved
        };
        let l4_self_declared = is_l4_self_declared_with_grounding(contract, lean_grounded);

        // Count bindings for this contract
        let (b_impl, b_total) = if let Some(reg) = binding {
            count_bindings(&contract_file, contract, reg)
        } else {
            (0, contract.equations.len() as u32)
        };

        let binding_status = if binding.is_some() {
            Some((b_impl, b_total))
        } else {
            None
        };

        let proof_level =
            compute_proof_level_with_grounding(contract, binding_status, lean_grounded);

        totals.obligations += obligations;
        totals.falsification_tests += ft_count;
        totals.kani_harnesses += kani_count;
        totals.lean_proved += lean_proved;
        totals.lean_grounded += lean_grounded;
        totals.l4_self_declared += u32::from(l4_self_declared);
        totals.bindings_implemented += b_impl;
        totals.bindings_total += b_total;

        statuses.push(ContractProofStatus {
            stem: stem.clone(),
            proof_level,
            obligations,
            falsification_tests: ft_count,
            kani_harnesses: kani_count,
            lean_proved,
            lean_grounded,
            l4_self_declared,
            bindings_implemented: b_impl,
            bindings_total: b_total,
        });
    }

    // Build kernel class summaries
    let kernel_classes = if include_classes {
        build_kernel_classes(&statuses)
    } else {
        Vec::new()
    };

    let timestamp = current_timestamp();

    ProofStatusReport {
        schema_version: "1.0.0".to_string(),
        l4_self_declared_excluded: true,
        timestamp,
        contracts: statuses,
        kernel_classes,
        totals,
    }
}

/// Format a proof status report as human-readable text.
pub fn format_text(report: &ProofStatusReport) -> String {
    let mut out = String::new();

    out.push_str(&format!(
        "Proof Status ({} contracts)\n\n",
        report.totals.contracts
    ));

    out.push_str(&format!(
        "  {:<35} {:>5} {:>6} {:>5} {:>4} {:>4} {:>9} {:>13}\n",
        "Contract", "Level", "Obligs", "Tests", "Kani", "Lean", "Bindings", "L4 evidence"
    ));
    out.push_str(&format!("  {}\n", "─".repeat(86)));

    for c in &report.contracts {
        // ONT-2a: the andon is a COLUMN, not a footnote. A claim the tree does not ground says so on its
        // own line, beside the level it no longer reaches.
        let l4_evidence = if c.l4_self_declared {
            "self-declared"
        } else if c.lean_grounded > 0 {
            "grounded"
        } else {
            "-"
        };
        out.push_str(&format!(
            "  {:<35} {:>5} {:>6} {:>5} {:>4} {:>4} {:>4}/{:<4} {:>13}\n",
            truncate(&c.stem, 35),
            c.proof_level,
            c.obligations,
            c.falsification_tests,
            c.kani_harnesses,
            c.lean_proved,
            c.bindings_implemented,
            c.bindings_total,
            l4_evidence,
        ));
    }

    if !report.kernel_classes.is_empty() {
        out.push_str("\nKernel Classes:\n");
        for kc in &report.kernel_classes {
            let bound_str = if kc.all_bound { "all bound" } else { "gaps" };
            out.push_str(&format!(
                "  {} ({}): min={}, {} contracts, {}\n",
                kc.label,
                kc.description,
                kc.min_proof_level,
                kc.contract_stems.len(),
                bound_str,
            ));
        }
    }

    out.push_str(&format!(
        "\nTotals: {} obligations, {} tests, {} kani, {} lean claimed ({} grounded), {}/{} bound\n\
         L4 evidence: {} contract(s) self-declared and excluded from L4 (ONT-2a andon); grounded means a \
         sorry-free in-tree Lean theorem the equation names\n",
        report.totals.obligations,
        report.totals.falsification_tests,
        report.totals.kani_harnesses,
        report.totals.lean_proved,
        report.totals.lean_grounded,
        report.totals.bindings_implemented,
        report.totals.bindings_total,
        report.totals.l4_self_declared,
    ));

    out
}

// ── Internal helpers ──────────────────────────────────────────────

/// Count implemented vs total bindings for a contract in the registry
#[allow(clippy::cast_possible_truncation)]
pub(crate) fn count_bindings(
    contract_file: &str,
    contract: &Contract,
    binding: &BindingRegistry,
) -> (u32, u32) {
    let total = contract.equations.len() as u32;
    let implemented = binding
        .bindings_for(contract_file)
        .iter()
        .filter(|b| b.status == ImplStatus::Implemented)
        .count() as u32;
    (implemented, total)
}

/// Build kernel equivalence class summaries from per-contract statuses
fn build_kernel_classes(statuses: &[ContractProofStatus]) -> Vec<KernelClassSummary> {
    let status_map: BTreeMap<&str, &ContractProofStatus> =
        statuses.iter().map(|s| (s.stem.as_str(), s)).collect();

    kernel_class_map()
        .into_iter()
        .map(|(label, desc, stems)| {
            let found_stems: Vec<String> = stems
                .iter()
                .filter(|s| status_map.contains_key(**s))
                .map(|s| (*s).to_string())
                .collect();

            let min_level = found_stems
                .iter()
                .filter_map(|s| status_map.get(s.as_str()))
                .map(|c| c.proof_level)
                .min()
                .unwrap_or(ProofLevel::L1);

            let all_bound = !found_stems.is_empty()
                && found_stems.iter().all(|s| {
                    status_map.get(s.as_str()).is_some_and(|c| {
                        c.bindings_total > 0 && c.bindings_implemented == c.bindings_total
                    })
                });

            KernelClassSummary {
                label: label.to_string(),
                description: desc.to_string(),
                contract_stems: found_stems,
                min_proof_level: min_level,
                all_bound,
            }
        })
        .collect()
}

/// Truncate a string to at most `max` bytes for column alignment
fn truncate(s: &str, max: usize) -> &str {
    if s.len() > max {
        &s[..max]
    } else {
        s
    }
}

/// Generate an ISO-8601-style Unix epoch timestamp string
fn current_timestamp() -> String {
    // Use a simple ISO-8601 timestamp without external deps.
    // In production this would use chrono or time crate.
    // For now we use std::time for a Unix epoch string.
    let duration = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}Z", duration.as_secs())
}

#[cfg(test)]
#[path = "proof_status_tests.rs"]
mod tests;
