//! Lean 4 code generator — Phase 7 of the pipeline.
//!
//! Generates Lean 4 source files from YAML kernel contracts:
//!
//! - **Definition files** with kernel functions as Lean `def` over `ℝ`
//! - **Theorem stubs** with `sorry` for each proof obligation that has
//!   a `lean` block
//! - **Import structure** based on Mathlib dependencies
//!
//! Also provides `lean_status` for reporting proof status across contracts.

use crate::schema::{Contract, LeanStatus, ProofObligation};

/// A generated Lean 4 file.
#[derive(Debug, Clone)]
pub struct LeanFile {
    /// Relative path within the Lean project (e.g. `ProvableContracts/Defs/Softmax.lean`).
    pub path: String,
    /// Lean 4 source content.
    pub content: String,
}

/// Generate Lean 4 source files from a contract.
///
/// Produces:
/// 1. A definitions file with equations as Lean `noncomputable def`s
/// 2. One theorem stub file per proof obligation that has a `lean` block
///
/// Returns an empty vec if the contract has no Lean metadata.
pub fn generate_lean_files(contract: &Contract) -> Vec<LeanFile> {
    let lean_obligations: Vec<&ProofObligation> = contract
        .proof_obligations
        .iter()
        .filter(|ob| ob.lean.is_some())
        .collect();

    if lean_obligations.is_empty() {
        return Vec::new();
    }

    let module_name = derive_module_name(&contract.metadata.description);
    let mut files = Vec::new();

    // 1. Definitions file
    files.push(generate_defs_file(contract, &module_name));

    // 2. Theorem stub files
    for ob in &lean_obligations {
        if let Some(ref lean) = ob.lean {
            files.push(generate_theorem_file(ob, lean, &module_name));
        }
    }

    files
}

/// Report Lean proof status for a contract.
///
/// Returns a `LeanStatusReport` with counts by status.
pub fn lean_status(contract: &Contract) -> LeanStatusReport {
    let mut report = LeanStatusReport {
        contract_description: contract.metadata.description.clone(),
        #[allow(clippy::cast_possible_truncation)]
        total_obligations: contract.proof_obligations.len() as u32,
        with_lean: 0,
        proved: 0,
        sorry: 0,
        wip: 0,
        not_applicable: 0,
        obligations: Vec::new(),
    };

    for ob in &contract.proof_obligations {
        if let Some(ref lean) = ob.lean {
            report.with_lean += 1;
            match lean.status {
                LeanStatus::Proved => report.proved += 1,
                LeanStatus::Sorry => report.sorry += 1,
                LeanStatus::Wip => report.wip += 1,
                LeanStatus::NotApplicable => report.not_applicable += 1,
            }
            report.obligations.push(ObligationStatus {
                property: ob.property.clone(),
                theorem: lean.theorem.clone(),
                status: lean.status,
            });
        }
    }

    report
}

/// Status report for Lean proofs in a single contract.
#[derive(Debug, Clone)]
pub struct LeanStatusReport {
    pub contract_description: String,
    pub total_obligations: u32,
    pub with_lean: u32,
    pub proved: u32,
    pub sorry: u32,
    pub wip: u32,
    pub not_applicable: u32,
    pub obligations: Vec<ObligationStatus>,
}

/// Status of a single obligation's Lean proof.
#[derive(Debug, Clone)]
pub struct ObligationStatus {
    pub property: String,
    pub theorem: String,
    pub status: LeanStatus,
}

/// Format a `LeanStatusReport` as a human-readable table.
pub fn format_status_report(reports: &[LeanStatusReport]) -> String {
    let mut out = String::new();

    out.push_str(&format!(
        "{:<30} {:>5} {:>6} {:>5} {:>3} {:>3}\n",
        "Contract", "Oblgs", "Proved", "Sorry", "WIP", "N/A"
    ));
    out.push_str(&"─".repeat(60));
    out.push('\n');

    let mut total_ob = 0u32;
    let mut total_proved = 0u32;
    let mut total_sorry = 0u32;
    let mut total_wip = 0u32;
    let mut total_na = 0u32;

    for r in reports {
        let name = if r.contract_description.len() > 30 {
            // Truncate at char boundary to avoid UTF-8 panic
            let end = r
                .contract_description
                .char_indices()
                .take_while(|(i, _)| *i < 30)
                .last()
                .map_or(0, |(i, c)| i + c.len_utf8());
            &r.contract_description[..end]
        } else {
            &r.contract_description
        };
        out.push_str(&format!(
            "{:<30} {:>5} {:>6} {:>5} {:>3} {:>3}\n",
            name, r.with_lean, r.proved, r.sorry, r.wip, r.not_applicable
        ));
        total_ob += r.with_lean;
        total_proved += r.proved;
        total_sorry += r.sorry;
        total_wip += r.wip;
        total_na += r.not_applicable;
    }

    out.push_str(&"─".repeat(60));
    out.push('\n');
    out.push_str(&format!(
        "{:<30} {:>5} {:>6} {:>5} {:>3} {:>3}\n",
        "Total", total_ob, total_proved, total_sorry, total_wip, total_na
    ));

    if let Some(pct) = (total_proved * 100).checked_div(total_ob) {
        out.push_str(&format!(
            "L4 Coverage: {pct}% ({total_proved}/{total_ob})   Sorry Debt: {total_sorry}\n"
        ));
    }

    out
}

// ── Internal helpers ──────────────────────────────────────────────

fn derive_module_name(description: &str) -> String {
    let base = description
        .split_whitespace()
        .next()
        .unwrap_or("Unknown")
        .to_string();
    // Capitalize first letter for Lean module convention
    let mut chars = base.chars();
    match chars.next() {
        None => "Unknown".to_string(),
        Some(c) => c.to_uppercase().collect::<String>() + chars.as_str(),
    }
}

fn generate_defs_file(contract: &Contract, module_name: &str) -> LeanFile {
    let mut content = String::new();

    content.push_str(&format!("-- {}\n", contract.metadata.description));
    content.push_str(&format!(
        "-- Generated from contract v{}\n",
        contract.metadata.version
    ));
    content.push_str("-- DO NOT EDIT — regenerate with `pv lean`\n\n");

    // Collect all mathlib imports from obligations
    let mut imports: Vec<&str> = Vec::new();
    for ob in &contract.proof_obligations {
        if let Some(ref lean) = ob.lean {
            for imp in &lean.mathlib_imports {
                if !imports.contains(&imp.as_str()) {
                    imports.push(imp);
                }
            }
        }
    }

    content.push_str("import Mathlib.Data.Real.Basic\n");
    content.push_str("import Mathlib.Data.Finset.Basic\n");
    for imp in &imports {
        content.push_str(&format!("import {imp}\n"));
    }
    content.push('\n');

    content.push_str(&format!("namespace ProvableContracts.{module_name}\n\n"));

    // Generate noncomputable defs from equations
    for (name, eq) in &contract.equations {
        content.push_str(&format!("-- Equation: {name}\n"));
        content.push_str(&format!("-- Formula: {}\n", eq.formula));
        if let Some(ref domain) = eq.domain {
            content.push_str(&format!("-- Domain: {domain}\n"));
        }
        content.push_str(&format!("noncomputable def {name} : sorry := sorry\n\n"));
    }

    content.push_str(&format!("end ProvableContracts.{module_name}\n"));

    LeanFile {
        path: format!("ProvableContracts/Defs/{module_name}.lean"),
        content,
    }
}

fn generate_theorem_file(
    ob: &ProofObligation,
    lean: &crate::schema::LeanProof,
    module_name: &str,
) -> LeanFile {
    let mut content = String::new();

    content.push_str(&format!("-- Theorem: {}\n", lean.theorem));
    content.push_str(&format!("-- Property: {}\n", ob.property));
    content.push_str(&format!("-- Obligation type: {}\n", ob.obligation_type));
    content.push_str("-- Generated with `pv lean`\n\n");

    // Imports
    content.push_str(&format!("import ProvableContracts.Defs.{module_name}\n"));
    for imp in &lean.mathlib_imports {
        content.push_str(&format!("import {imp}\n"));
    }
    content.push('\n');

    // Lean-level dependencies
    if !lean.depends_on.is_empty() {
        content.push_str("-- Dependencies:\n");
        for dep in &lean.depends_on {
            content.push_str(&format!("--   {dep}\n"));
        }
        content.push('\n');
    }

    content.push_str(&format!("namespace ProvableContracts.{module_name}\n\n"));

    // Formal statement if present
    if let Some(ref formal) = ob.formal {
        content.push_str(&format!("-- Formal: {formal}\n"));
    }

    // Theorem stub
    let status_comment = match lean.status {
        LeanStatus::Proved => "-- Status: proved",
        LeanStatus::Sorry => "-- Status: sorry (proof pending)",
        LeanStatus::Wip => "-- Status: work in progress",
        LeanStatus::NotApplicable => "-- Status: not applicable",
    };
    content.push_str(&format!("{status_comment}\n"));
    content.push_str(&format!("theorem {} : sorry := by\n", lean.theorem));
    content.push_str("  sorry\n");

    if let Some(ref notes) = lean.notes {
        content.push_str(&format!("\n-- Note: {notes}\n"));
    }

    content.push_str(&format!("\nend ProvableContracts.{module_name}\n"));

    // Derive file path from theorem name
    let theorem_file = lean.theorem.split('.').next_back().unwrap_or(&lean.theorem);
    LeanFile {
        path: format!("ProvableContracts/Theorems/{module_name}/{theorem_file}.lean"),
        content,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::parse_contract_str;

    #[test]
    fn no_lean_obligations_produces_empty() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Softmax kernel"
  references: ["Paper"]
equations:
  softmax:
    formula: "f(x) = exp(x_i) / sum(exp(x_j))"
proof_obligations:
  - type: invariant
    property: "Output sums to 1"
falsification_tests: []
"#;
        let contract = parse_contract_str(yaml).unwrap();
        let files = generate_lean_files(&contract);
        assert!(files.is_empty());
    }

    #[test]
    fn generates_defs_and_theorem_files() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Softmax kernel"
  references: ["Paper"]
equations:
  softmax:
    formula: "f(x) = exp(x_i) / sum(exp(x_j))"
    domain: "R^n"
proof_obligations:
  - type: invariant
    property: "Output sums to 1"
    formal: "|sum(f(x)) - 1| < eps"
    lean:
      theorem: Softmax.partition_of_unity
      module: ProvableContracts.Softmax
      status: sorry
      depends_on:
        - Real.exp_pos
      mathlib_imports:
        - Mathlib.Analysis.SpecialFunctions.ExpDeriv
      notes: "Proof over reals"
falsification_tests: []
"#;
        let contract = parse_contract_str(yaml).unwrap();
        let files = generate_lean_files(&contract);
        assert_eq!(files.len(), 2); // defs + 1 theorem

        // Check defs file
        let defs = &files[0];
        assert!(defs.path.contains("Defs/Softmax"));
        assert!(defs.content.contains("noncomputable def softmax"));
        assert!(defs.content.contains("namespace ProvableContracts.Softmax"));
        assert!(defs
            .content
            .contains("Mathlib.Analysis.SpecialFunctions.ExpDeriv"));

        // Check theorem file
        let thm = &files[1];
        assert!(thm.path.contains("Theorems/Softmax/partition_of_unity"));
        assert!(thm.content.contains("theorem Softmax.partition_of_unity"));
        assert!(thm.content.contains("sorry"));
        assert!(thm.content.contains("Real.exp_pos"));
        assert!(thm.content.contains("Proof over reals"));
    }

    #[test]
    fn lean_status_counts_correctly() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Test kernel"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: invariant
    property: "P1"
    lean:
      theorem: T1
      status: proved
  - type: bound
    property: "P2"
    lean:
      theorem: T2
      status: sorry
  - type: monotonicity
    property: "P3"
    lean:
      theorem: T3
      status: wip
  - type: equivalence
    property: "P4 no lean"
falsification_tests: []
"#;
        let contract = parse_contract_str(yaml).unwrap();
        let report = lean_status(&contract);
        assert_eq!(report.total_obligations, 4);
        assert_eq!(report.with_lean, 3);
        assert_eq!(report.proved, 1);
        assert_eq!(report.sorry, 1);
        assert_eq!(report.wip, 1);
        assert_eq!(report.not_applicable, 0);
    }

    #[test]
    fn format_status_report_renders_table() {
        let reports = vec![LeanStatusReport {
            contract_description: "Softmax kernel".to_string(),
            total_obligations: 5,
            with_lean: 3,
            proved: 1,
            sorry: 1,
            wip: 1,
            not_applicable: 0,
            obligations: vec![],
        }];
        let table = format_status_report(&reports);
        assert!(table.contains("Softmax kernel"));
        assert!(table.contains("L4 Coverage: 33%"));
        assert!(table.contains("Sorry Debt: 1"));
    }

    #[test]
    fn derive_module_name_capitalizes() {
        assert_eq!(derive_module_name("softmax kernel"), "Softmax");
        assert_eq!(derive_module_name("RMSNorm"), "RMSNorm");
        assert_eq!(derive_module_name(""), "Unknown");
    }

    #[test]
    fn proved_theorem_has_proved_comment() {
        let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Test"
  references: ["P"]
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: invariant
    property: "Always true"
    lean:
      theorem: F.always_true
      status: proved
falsification_tests: []
"#;
        let contract = parse_contract_str(yaml).unwrap();
        let files = generate_lean_files(&contract);
        let thm = &files[1];
        assert!(thm.content.contains("Status: proved"));
    }
}
