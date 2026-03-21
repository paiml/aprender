//! JSON and human-readable output formatting.

use super::config::{detect_constraint_mismatches, extract_architecture_display};
use super::family::load_families;
use super::kernel_ops::kernel_ops_for_family;
use super::proof::{proof_status_for_class, ProofLevel};
use super::{ConfigField, Constraints, FamilyInfo, KernelOp};
use serde::Serialize;
use std::collections::BTreeMap;

#[derive(Debug, Serialize)]
pub struct KernelExplainJson {
    pub architecture: String,
    pub kernel_class: String,
    pub kernel_class_label: String,
    pub family: String,
    pub display_name: String,
    pub kernel_ops: Vec<KernelOp>,
    pub constraints: Constraints,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub config_mapping: BTreeMap<String, ConfigField>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub proof_summary: Option<ProofSummary>,
    pub layout: String,
    pub equivalence_class_families: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct ProofSummary {
    pub proven: usize,
    pub tested: usize,
    pub documented: usize,
    pub unknown: usize,
    pub total: usize,
}

pub fn build_json_output(
    family: &FamilyInfo,
    mut config_mapping: BTreeMap<String, ConfigField>,
    show_proof: bool,
) -> KernelExplainJson {
    // Remove internal fields (prefixed with _) from public output
    config_mapping.retain(|k, _| !k.starts_with('_'));
    let ops = kernel_ops_for_family(family.kernel_class, &family.constraints);
    let proofs = if show_proof {
        proof_status_for_class(family.kernel_class)
    } else {
        Vec::new()
    };

    let proven = proofs
        .iter()
        .filter(|p| p.level == ProofLevel::Proven)
        .count();
    let tested = proofs
        .iter()
        .filter(|p| p.level == ProofLevel::Tested)
        .count();
    let documented = proofs
        .iter()
        .filter(|p| p.level == ProofLevel::Documented)
        .count();
    let unknown = proofs
        .iter()
        .filter(|p| p.level == ProofLevel::Unknown)
        .count();

    let families = load_families();
    let equivalence_class_families: Vec<String> = families
        .iter()
        .filter(|f| f.kernel_class == family.kernel_class)
        .map(|f| f.family.clone())
        .collect();

    let arch = extract_architecture_display(family, &config_mapping);

    let warnings = detect_constraint_mismatches(family, &config_mapping);

    KernelExplainJson {
        architecture: arch,
        kernel_class: family.kernel_class.letter().to_string(),
        kernel_class_label: family.kernel_class.label().to_string(),
        family: family.family.clone(),
        display_name: family.display_name.clone(),
        kernel_ops: ops,
        constraints: family.constraints.clone(),
        config_mapping,
        proof_summary: if show_proof {
            Some(ProofSummary {
                proven,
                tested,
                documented,
                unknown,
                total: proofs.len(),
            })
        } else {
            None
        },
        layout: "row_major".to_string(),
        equivalence_class_families,
        warnings,
    }
}

// ── Human-readable output ─────────────────────────────────────────────────

pub fn print_human_output(
    family: &FamilyInfo,
    config_mapping: &BTreeMap<String, ConfigField>,
    verbose: bool,
    show_proof: bool,
) {
    let arch = extract_architecture_display(family, config_mapping);
    let ops = kernel_ops_for_family(family.kernel_class, &family.constraints);

    println!("Kernel Explainability Report: {}", family.display_name);
    println!("{}", "═".repeat(50));
    println!();
    println!("Architecture:  {arch}");
    println!("Kernel Class:  {}", family.kernel_class.label());
    println!("Family:        {}", family.family);
    println!();

    // Kernel pipeline table
    println!("Kernel Pipeline ({} ops)", ops.len());
    println!(
        "┌─────────────────────────┬────────────────────────────────┬──────────────────────────┐"
    );
    println!(
        "│ Operation               │ Kernel                         │ Contract                 │"
    );
    println!(
        "├─────────────────────────┼────────────────────────────────┼──────────────────────────┤"
    );
    for op in &ops {
        println!(
            "│ {:<23} │ {:<30} │ {:<24} │",
            op.op, op.kernel, op.contract
        );
    }
    println!(
        "└─────────────────────────┴────────────────────────────────┴──────────────────────────┘"
    );

    // Config mapping (skip internal fields prefixed with _)
    let visible_fields: Vec<_> = config_mapping
        .iter()
        .filter(|(k, _)| !k.starts_with('_'))
        .collect();
    if !visible_fields.is_empty() {
        println!();
        println!("Config.json → Kernel Mapping:");
        // Calculate alignment width from longest key=value
        let max_kv_len = visible_fields
            .iter()
            .map(|(k, f)| k.len() + 1 + f.value.len())
            .max()
            .unwrap_or(20);
        let pad_to = max_kv_len + 2; // 2 extra spaces before arrow
        for (key, field) in &visible_fields {
            let kv = format!("{key}={}", field.value);
            println!("  {kv:<pad_to$} → {}", field.rationale);
        }
    }

    // Constraints (verbose)
    if verbose {
        println!();
        println!("Constraints (from family contract):");
        println!(
            "  attention_type:      {}",
            family.constraints.attention_type
        );
        println!("  activation:          {}", family.constraints.activation);
        println!("  norm_type:           {}", family.constraints.norm_type);
        println!("  mlp_type:            {}", family.constraints.mlp_type);
        println!(
            "  positional_encoding: {}",
            family.constraints.positional_encoding
        );
        println!("  has_bias:            {}", family.constraints.has_bias);
        println!(
            "  tied_embeddings:     {}",
            family.constraints.tied_embeddings
        );
    }

    // Constraint mismatch warnings (always run — MoE detection works from alias name too)
    let mismatches = detect_constraint_mismatches(family, config_mapping);
    let is_alias = family.display_name.contains(" (via ");
    for warning in &mismatches {
        println!();
        eprintln!("  WARNING: {warning}");
        if is_alias {
            eprintln!("  This model is mapped via alias. Kernel selection may differ from the family contract.");
        }
    }

    // Layout
    println!();
    println!("Layout: Row-major (LAYOUT-002 compliant)");
    println!("  GGUF→APR conversion transposes at import time.");
    println!("  Direct GGUF inference uses column-major kernels.");

    // Equivalence class members
    let families = load_families();
    let class_members: Vec<&str> = families
        .iter()
        .filter(|f| f.kernel_class == family.kernel_class)
        .map(|f| f.family.as_str())
        .collect();
    if !class_members.is_empty() {
        println!();
        println!(
            "Equivalence Class {}: {} {}",
            family.kernel_class.letter(),
            class_members.len(),
            if class_members.len() == 1 {
                "family"
            } else {
                "families"
            }
        );
        println!("  {}", class_members.join(", "));
    }

    // Proof status
    if show_proof {
        let proofs = proof_status_for_class(family.kernel_class);
        println!();
        println!("Proof Status:");
        for proof in &proofs {
            println!(
                "  {} {:<28} {} ({})",
                proof.level.symbol(),
                proof.contract,
                proof.level.label(),
                proof.evidence
            );
        }

        let proven = proofs
            .iter()
            .filter(|p| p.level == ProofLevel::Proven)
            .count();
        let tested = proofs
            .iter()
            .filter(|p| p.level == ProofLevel::Tested)
            .count();
        let total = proofs.len();
        println!();
        println!(
            "Kernel Class {}: {}/{} contracts verified ({} proven, {} tested).",
            family.kernel_class.letter(),
            proven + tested,
            total,
            proven,
            tested
        );
    }
}
