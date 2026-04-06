//! Rendering helpers for contract explanation output.
//!
//! Extracted from `explain.rs` to keep each file under 500 lines.

use std::fmt::Write;

use crate::binding::BindingRegistry;
use crate::proof_status::compute_proof_level;
use crate::schema::Contract;

use super::obligation_pattern;
use super::strategy_explanation;

pub(super) fn write_obligations(out: &mut String, contract: &Contract) {
    if contract.proof_obligations.is_empty() {
        return;
    }
    let _ = writeln!(
        out,
        "Proof obligations ({})",
        contract.proof_obligations.len()
    );
    let _ = writeln!(out);

    for (i, ob) in contract.proof_obligations.iter().enumerate() {
        let _ = writeln!(out, "  {}. [{}] {}", i + 1, ob.obligation_type, ob.property);
        let _ = writeln!(
            out,
            "     Pattern: {}",
            obligation_pattern(ob.obligation_type)
        );

        if let Some(ref formal) = ob.formal {
            let _ = writeln!(out, "     Formal:  {formal}");
        }
        if let Some(tol) = ob.tolerance {
            let _ = writeln!(out, "     Tolerance: {tol:e}");
        }

        // Lean proof status
        if let Some(ref lean) = ob.lean {
            let module = lean.module.as_deref().unwrap_or("?");
            let _ = writeln!(
                out,
                "     Lean: {module}.{} ({})",
                lean.theorem, lean.status
            );
            if !lean.depends_on.is_empty() {
                let _ = writeln!(out, "       Depends: {}", lean.depends_on.join(", "));
            }
            if let Some(ref notes) = lean.notes {
                let _ = writeln!(out, "       Note: {notes}");
            }
        }

        // DbC-specific fields
        if let Some(ref req) = ob.requires {
            let _ = writeln!(
                out,
                "     Requires: {req} (postcondition conditional on this precondition)"
            );
        }
        if let Some(ref phase) = ob.applies_to_phase {
            let _ = writeln!(out, "     Phase: {phase} (from kernel_structure)");
        }
        if let Some(ref parent) = ob.parent_contract {
            let _ = writeln!(
                out,
                "     Refines: {parent} (behavioral subtyping — pre weakened, post strengthened)"
            );
        }

        // Cross-reference to falsification tests
        let matching_ft: Vec<&str> = contract
            .falsification_tests
            .iter()
            .filter(|ft| {
                ob.property.to_lowercase().contains(&ft.rule.to_lowercase())
                    || ft.rule.to_lowercase().contains(&ob.property.to_lowercase())
            })
            .map(|ft| ft.id.as_str())
            .collect();

        // Cross-reference to Kani harnesses
        let matching_kh: Vec<&str> = contract
            .kani_harnesses
            .iter()
            .filter(|kh| {
                kh.property
                    .as_ref()
                    .is_some_and(|p| p.to_lowercase().contains(&ob.property.to_lowercase()))
            })
            .map(|kh| kh.id.as_str())
            .collect();

        if !matching_ft.is_empty() || !matching_kh.is_empty() {
            let mut parts = Vec::new();
            for ft in &matching_ft {
                parts.push(format!("L2 ({ft})"));
            }
            for kh in &matching_kh {
                parts.push(format!("L4 ({kh})"));
            }
            if ob
                .lean
                .as_ref()
                .is_some_and(|l| l.status.to_string() == "proved")
            {
                parts.push("L5 (Lean)".to_string());
            }
            if !parts.is_empty() {
                let _ = writeln!(out, "     Verified at: {}", parts.join(", "));
            }
        }

        let _ = writeln!(out);
    }
}

pub(super) fn write_verification_ladder(
    out: &mut String,
    contract: &Contract,
    binding: Option<&BindingRegistry>,
) {
    let total = contract.proof_obligations.len();
    if total == 0 {
        return;
    }

    let lean_proved = contract
        .verification_summary
        .as_ref()
        .map_or(0, |vs| vs.l4_lean_proved as usize);
    let kani_count = contract.kani_harnesses.len();
    let ft_count = contract.falsification_tests.len();
    let level = compute_proof_level(contract, None);

    let _ = writeln!(out, "Verification ladder");
    if lean_proved > 0 {
        let pct = lean_proved * 100 / total;
        let _ = writeln!(out, "  L5 (Lean):  {lean_proved}/{total} proved ({pct}%)");
    }
    if kani_count > 0 {
        // Summarize strategies
        let mut strategies: std::collections::BTreeMap<String, usize> =
            std::collections::BTreeMap::new();
        for kh in &contract.kani_harnesses {
            let s = kh
                .strategy
                .map_or_else(|| "default".to_string(), |s| s.to_string());
            *strategies.entry(s).or_default() += 1;
        }
        let strat_summary: Vec<String> = strategies
            .iter()
            .map(|(s, n)| format!("{n}× {s}"))
            .collect();
        let _ = writeln!(
            out,
            "  L4 (Kani):  {kani_count} harnesses ({})",
            strat_summary.join(", ")
        );
    }
    if ft_count > 0 {
        let _ = writeln!(out, "  L2 (Tests): {ft_count} falsification tests");
    }

    let _ = writeln!(out, "  Level: {level}");

    if let Some(b) = binding {
        let stem_for_binding = format!("{}.yaml", ""); // binding needs contract filename
        let _ = b; // suppress unused — binding status shown in write_binding_status
        let _ = stem_for_binding;
    }
    let _ = writeln!(out);
}

pub(super) fn write_falsification_tests(out: &mut String, contract: &Contract) {
    if contract.falsification_tests.is_empty() {
        return;
    }
    let _ = writeln!(out, "Falsification tests (Popperian)");
    let _ = writeln!(
        out,
        "  Each test tries to refute the contract. Survival = evidence"
    );
    let _ = writeln!(out, "  for, not proof of, correctness.");
    let _ = writeln!(out);

    for ft in &contract.falsification_tests {
        let _ = writeln!(out, "  {}: {}", ft.id, ft.rule);
        let _ = writeln!(out, "    Predicts: {}", ft.prediction);
        if let Some(ref test) = ft.test {
            let _ = writeln!(out, "    Method:   {test}");
        }
        let _ = writeln!(out, "    Catches:  {}", ft.if_fails);
        let _ = writeln!(out);
    }
}

pub(super) fn write_kani_harnesses(out: &mut String, contract: &Contract) {
    if contract.kani_harnesses.is_empty() {
        return;
    }
    let _ = writeln!(out, "Kani bounded model checking");

    for kh in &contract.kani_harnesses {
        let bound = kh.bound.map_or_else(|| "-".to_string(), |b| b.to_string());
        let strategy = kh
            .strategy
            .map_or_else(|| "default".to_string(), |s| s.to_string());
        let harness = kh.harness.as_deref().unwrap_or(&kh.id);

        let _ = writeln!(
            out,
            "  {}: {} (bound: {}, {})",
            kh.id, harness, bound, strategy
        );
        let _ = writeln!(out, "    {}", strategy_explanation(&strategy));
        if let Some(ref prop) = kh.property {
            let _ = writeln!(out, "    Property: {prop}");
        }
        let _ = writeln!(out);
    }
}

pub(super) fn write_kernel_phases(out: &mut String, contract: &Contract) {
    let Some(ref ks) = contract.kernel_structure else {
        return;
    };
    let _ = writeln!(out, "Kernel phases");
    for (i, phase) in ks.phases.iter().enumerate() {
        let inv = phase
            .invariant
            .as_deref()
            .map(|s| format!(" [{s}]"))
            .unwrap_or_default();
        let _ = writeln!(
            out,
            "  {}. {} — {}{}",
            i + 1,
            phase.name,
            phase.description,
            inv
        );
    }
    let _ = writeln!(out);
}

pub(super) fn write_simd_dispatch(out: &mut String, contract: &Contract) {
    if contract.simd_dispatch.is_empty() {
        return;
    }
    let _ = writeln!(out, "SIMD dispatch");
    for (kernel, dispatch) in &contract.simd_dispatch {
        let targets: Vec<String> = dispatch
            .iter()
            .map(|(isa, target)| format!("{isa} → {target}"))
            .collect();
        let _ = writeln!(out, "  {kernel}: {}", targets.join(" | "));
    }
    let _ = writeln!(out);
}

pub(super) fn write_enforcement(out: &mut String, contract: &Contract) {
    if contract.enforcement.is_empty() {
        return;
    }
    let _ = writeln!(out, "Enforcement");
    for (name, rule) in &contract.enforcement {
        let severity = rule.severity.as_deref().unwrap_or("ERROR");
        let check = rule.check.as_deref().unwrap_or("-");
        let _ = writeln!(
            out,
            "  {name} — {} ({severity}) → {check}",
            rule.description
        );
    }
    let _ = writeln!(out);
}

pub(super) fn write_qa_gate(out: &mut String, contract: &Contract) {
    let Some(ref qa) = contract.qa_gate else {
        return;
    };
    let _ = writeln!(out, "Quality gate: {} {}", qa.id, qa.name);
    if let Some(ref desc) = qa.description {
        let _ = writeln!(out, "  {desc}");
    }
    if let Some(ref criteria) = qa.pass_criteria {
        let _ = writeln!(out, "  Pass: {criteria}");
    }
    if let Some(ref falsification) = qa.falsification {
        let _ = writeln!(out, "  Mutation: {falsification}");
    }
    let _ = writeln!(out);
}

pub(super) fn write_binding_status(
    out: &mut String,
    contract: &Contract,
    stem: &str,
    binding: Option<&BindingRegistry>,
) {
    let Some(registry) = binding else {
        return;
    };

    let contract_file = format!("{stem}.yaml");
    let mut found = false;

    for eq_name in contract.equations.keys() {
        let status = registry
            .bindings
            .iter()
            .find(|b| b.contract == contract_file && b.equation == *eq_name)
            .map_or_else(|| "missing".to_string(), |b| b.status.to_string());

        if !found {
            let _ = writeln!(out, "Binding status ({})", registry.target_crate);
            found = true;
        }
        let _ = writeln!(out, "  {eq_name}: {status}");
    }

    if found {
        let _ = writeln!(out);
    }
}
