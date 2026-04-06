//! Flux refinement type annotation generation.
//!
//! Generates `#[flux::refined_by]` structs and `#[flux::sig]` function
//! signatures from YAML contracts, enabling compile-time shape verification
//! via SMT solving.

use std::fmt::Write;

use crate::schema::Contract;

/// Generate Flux refinement type annotations from a contract.
///
/// Focuses on shape-related contracts — those with tensor dimensions,
/// matrix shapes, or vector length invariants in their equations.
pub fn generate_flux_annotations(contract: &Contract, stem: &str) -> String {
    let mut out = String::with_capacity(2048);

    let _ = writeln!(
        out,
        "// Auto-generated Flux refinement type annotations from contract: {}",
        contract.metadata.description
    );
    let _ = writeln!(out, "// Regenerate with: pv flux contracts/{stem}.yaml");
    let _ = writeln!(out, "// Requires: cargo flux (Flux nightly toolchain)");
    let _ = writeln!(out);

    // Check if contract has shape-related equations
    let has_shapes = contract.equations.iter().any(|(_, eq)| {
        let formula_lower = eq.formula.to_lowercase();
        formula_lower.contains("shape")
            || formula_lower.contains("dim")
            || formula_lower.contains("len")
            || formula_lower.contains("size")
            || formula_lower.contains("rows")
            || formula_lower.contains("cols")
            || formula_lower.contains("product")
    });

    if has_shapes {
        generate_shape_annotations(&mut out, contract);
    } else {
        generate_generic_annotations(&mut out, contract);
    }

    out
}

fn generate_shape_annotations(out: &mut String, contract: &Contract) {
    // Generate refined struct for tensor types
    let _ = writeln!(
        out,
        "/// Refined tensor type — dimensions tracked at compile time."
    );
    let _ = writeln!(out, "#[flux::refined_by(n: int)]");
    let _ = writeln!(out, "pub struct RefinedVec {{");
    let _ = writeln!(out, "    #[flux::field(Vec<f32>[n])]");
    let _ = writeln!(out, "    data: Vec<f32>,");
    let _ = writeln!(out, "}}");
    let _ = writeln!(out);

    // Generate Flux signatures for each equation
    for (eq_name, eq) in &contract.equations {
        let fn_name = eq_name.replace('-', "_");
        let _ = writeln!(out, "/// Equation: {}", eq.formula);
        if let Some(ref dom) = eq.domain {
            let _ = writeln!(out, "/// Domain: {dom}");
        }

        let _ = writeln!(out, "#[flux::sig(");
        let _ = writeln!(
            out,
            "    fn(input: &RefinedVec[@n]) -> RefinedVec[n] requires n > 0"
        );
        let _ = writeln!(out, ")]");
        let _ = writeln!(out, "pub fn {fn_name}(input: &RefinedVec) -> RefinedVec {{");
        let _ = writeln!(out, "    // TODO: Wire up implementation");
        let _ = writeln!(
            out,
            "    RefinedVec {{ data: vec![0.0; input.data.len()] }}"
        );
        let _ = writeln!(out, "}}");
        let _ = writeln!(out);
    }
}

fn generate_generic_annotations(out: &mut String, contract: &Contract) {
    let _ = writeln!(
        out,
        "// No shape-specific equations found — generating generic length-preserving signatures."
    );
    let _ = writeln!(out);

    for (eq_name, eq) in &contract.equations {
        let fn_name = eq_name.replace('-', "_");
        let _ = writeln!(out, "/// Equation: {}", eq.formula);

        let _ = writeln!(out, "#[flux::sig(");
        let _ = writeln!(
            out,
            "    fn(input: &Vec<f32>[@n]) -> Vec<f32>[n] requires n > 0"
        );
        let _ = writeln!(out, ")]");
        let _ = writeln!(out, "pub fn {fn_name}(input: &[f32]) -> Vec<f32> {{");
        let _ = writeln!(out, "    // TODO: Wire up implementation");
        let _ = writeln!(out, "    vec![0.0; input.len()]");
        let _ = writeln!(out, "}}");
        let _ = writeln!(out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::parse_contract_str;

    #[test]
    fn generates_shape_annotations() {
        let contract = parse_contract_str(
            r#"
metadata:
  version: "1.0.0"
  description: "Tensor shape flow"
  references: ["Test"]
equations:
  reshape:
    formula: "product(old_shape) == product(new_shape)"
    domain: "shape ∈ ℤ^n"
falsification_tests: []
"#,
        )
        .unwrap();

        let output = generate_flux_annotations(&contract, "tensor-shape-flow-v1");
        assert!(output.contains("#[flux::refined_by"));
        assert!(output.contains("#[flux::sig"));
        assert!(output.contains("RefinedVec"));
    }

    #[test]
    fn generates_generic_annotations() {
        let contract = parse_contract_str(
            r#"
metadata:
  version: "1.0.0"
  description: "Simple kernel"
  references: ["Test"]
equations:
  softmax:
    formula: "σ(x) = exp(x) / Σ exp(x)"
falsification_tests: []
"#,
        )
        .unwrap();

        let output = generate_flux_annotations(&contract, "softmax-kernel-v1");
        assert!(output.contains("#[flux::sig"));
        assert!(output.contains("requires n > 0"));
        assert!(!output.contains("RefinedVec"));
    }
}
