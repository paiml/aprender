// ===========================================================================
// Cross-format provability checks
//
// These verify that the mathematical content (equations, invariants, phases,
// proof obligations) is faithfully and consistently represented across all
// output formats for each contract.
// ===========================================================================

/// Helper: parse a YAML contract to extract provability data.
struct ContractFacts {
    equation_ids: Vec<String>,
    formulas: Vec<String>,
    invariants: Vec<String>,
    phase_names: Vec<String>,
    obligation_types: Vec<String>,
    obligation_props: Vec<String>,
}

fn extract_facts(contract: &str) -> ContractFacts {
    let yaml = std::fs::read_to_string(fixture_path(contract)).unwrap();
    let doc: serde_yaml::Value = serde_yaml::from_str(&yaml).unwrap();

    let equations = doc["equations"].as_mapping().unwrap();
    let mut equation_ids = Vec::new();
    let mut formulas = Vec::new();
    let mut invariants = Vec::new();
    for (id, eq) in equations {
        equation_ids.push(id.as_str().unwrap().to_string());
        formulas.push(eq["formula"].as_str().unwrap().to_string());
        if let Some(invs) = eq.get("invariants").and_then(|v| v.as_sequence()) {
            for inv in invs {
                invariants.push(inv.as_str().unwrap().to_string());
            }
        }
    }

    let mut phase_names = Vec::new();
    if let Some(ks) = doc.get("kernel_structure") {
        if let Some(phases) = ks.get("phases").and_then(|v| v.as_sequence()) {
            for phase in phases {
                phase_names.push(phase["name"].as_str().unwrap().to_string());
            }
        }
    }

    let mut obligation_types = Vec::new();
    let mut obligation_props = Vec::new();
    if let Some(obs) = doc.get("proof_obligations").and_then(|v| v.as_sequence()) {
        for ob in obs {
            obligation_types.push(ob["type"].as_str().unwrap().to_string());
            obligation_props.push(ob["property"].as_str().unwrap().to_string());
        }
    }

    ContractFacts {
        equation_ids,
        formulas,
        invariants,
        phase_names,
        obligation_types,
        obligation_props,
    }
}

/// Verify provability: every equation, invariant, phase, and proof
/// obligation from the YAML source appears in all relevant outputs.
fn verify_provability(contract: &str) {
    let facts = extract_facts(contract);
    let text = run_equations(contract, "text");
    let latex = run_equations(contract, "latex");
    let ptx = run_equations(contract, "ptx");
    let asm = run_equations(contract, "asm");

    // 1. Every equation ID appears in text and latex output
    for id in &facts.equation_ids {
        assert!(text.contains(id), "text missing equation ID '{id}'");
        assert!(latex.contains(id), "latex missing equation ID '{id}'");
    }

    // 2. Every formula appears in text output verbatim
    for formula in &facts.formulas {
        assert!(text.contains(formula), "text missing formula '{formula}'");
    }

    // 3. Every invariant appears in text output
    for inv in &facts.invariants {
        assert!(text.contains(inv), "text missing invariant '{inv}'");
    }

    // 4. Every phase name appears in both PTX and ASM
    for (i, name) in facts.phase_names.iter().enumerate() {
        let phase_label = format!("Phase {}: {name}", i + 1);
        assert!(
            ptx.contains(&phase_label),
            "ptx missing phase '{phase_label}'"
        );
        assert!(
            asm.contains(&phase_label),
            "asm missing phase '{phase_label}'"
        );
    }

    // 5. Every proof obligation appears in PTX and ASM
    for prop in &facts.obligation_props {
        assert!(ptx.contains(prop), "ptx missing obligation '{prop}'");
        assert!(asm.contains(prop), "asm missing obligation '{prop}'");
    }

    // 6. Obligation types appear with brackets in PTX and ASM
    for otype in &facts.obligation_types {
        let tag = format!("[{otype}]");
        assert!(ptx.contains(&tag), "ptx missing obligation type '{tag}'");
        assert!(asm.contains(&tag), "asm missing obligation type '{tag}'");
    }

    // 7. PTX and ASM have same number of obligation lines
    let ptx_ob_count = ptx.lines().filter(|l| l.contains("//   [")).count();
    let asm_ob_count = asm.lines().filter(|l| l.contains("//   [")).count();
    assert_eq!(
        ptx_ob_count, asm_ob_count,
        "PTX ({ptx_ob_count}) and ASM ({asm_ob_count}) obligation count mismatch"
    );

    // 8. LaTeX has correct Unicode→LaTeX conversions where applicable
    if latex.contains("\\in") || latex.contains("\\mathbb{R}") {
        // If any equation has ∈ or ℝ, verify the conversion happened
        assert!(
            !latex.contains('∈'),
            "latex still contains raw ∈ (should be \\in)"
        );
        assert!(
            !latex.contains('ℝ'),
            "latex still contains raw ℝ (should be \\mathbb{{R}})"
        );
    }
    if latex.contains("\\geq") || latex.contains("\\leq") {
        assert!(
            !latex.contains('≥'),
            "latex still contains raw ≥ (should be \\geq)"
        );
        assert!(
            !latex.contains('≤'),
            "latex still contains raw ≤ (should be \\leq)"
        );
    }

    // 9. PTX structural validity
    assert!(ptx.contains(".version 8.5"), "ptx missing .version");
    assert!(ptx.contains(".target sm_90"), "ptx missing .target");
    assert!(ptx.contains(".visible .entry"), "ptx missing .entry");
    assert!(ptx.contains("ret;"), "ptx missing ret");

    // 10. ASM structural validity
    assert!(asm.contains(".intel_syntax noprefix"), "asm missing syntax");
    assert!(asm.contains(".globl"), "asm missing .globl");
    assert!(asm.contains("push rbp"), "asm missing prologue");
    assert!(asm.contains("pop rbp"), "asm missing epilogue");
    assert!(asm.contains("ret"), "asm missing ret");
}

#[test]
fn provability_relu() {
    verify_provability("relu-kernel-v1.yaml");
}

#[test]
fn provability_clamp() {
    verify_provability("clamp-kernel-v1.yaml");
}

#[test]
fn provability_dot() {
    verify_provability("dot-kernel-v1.yaml");
}

#[test]
fn provability_scale() {
    verify_provability("scale-kernel-v1.yaml");
}

#[test]
fn provability_l2norm() {
    verify_provability("l2norm-kernel-v1.yaml");
}

// ===========================================================================
// LaTeX-specific math conversion verification
// ===========================================================================

#[test]
fn latex_conversions_relu() {
    let latex = run_equations("relu-kernel-v1.yaml", "latex");
    // ≥ → \geq
    assert!(
        latex.contains("\\geq"),
        "relu latex: ≥ not converted to \\geq"
    );
    // ∈ → \in
    assert!(
        latex.contains("\\in"),
        "relu latex: ∈ not converted to \\in"
    );
    // ℝ → \mathbb{R}
    assert!(latex.contains("\\mathbb{R}"), "relu latex: ℝ not converted");
}

#[test]
fn latex_conversions_clamp() {
    let latex = run_equations("clamp-kernel-v1.yaml", "latex");
    // ≤ → \leq
    assert!(
        latex.contains("\\leq"),
        "clamp latex: ≤ not converted to \\leq"
    );
    // → → \to
    assert!(
        latex.contains("\\to"),
        "clamp latex: → not converted to \\to"
    );
}

#[test]
fn latex_conversions_dot() {
    let latex = run_equations("dot-kernel-v1.yaml", "latex");
    // Σ → \sum
    assert!(
        latex.contains("\\sum"),
        "dot latex: Σ not converted to \\sum"
    );
    // α → \alpha
    assert!(
        latex.contains("\\alpha"),
        "dot latex: α not converted to \\alpha"
    );
}

#[test]
fn latex_conversions_scale() {
    let latex = run_equations("scale-kernel-v1.yaml", "latex");
    // α → \alpha
    assert!(latex.contains("\\alpha"), "scale latex: α not converted");
    // β → \beta
    assert!(latex.contains("\\beta"), "scale latex: β not converted");
}

#[test]
fn latex_conversions_l2norm() {
    let latex = run_equations("l2norm-kernel-v1.yaml", "latex");
    // sqrt(Σ x_i²) → \sqrt{\sum x_i²}
    assert!(
        latex.contains("\\sqrt{"),
        "l2norm latex: sqrt not converted"
    );
    assert!(latex.contains("\\sum"), "l2norm latex: Σ not converted");
    // ≥ → \geq
    assert!(latex.contains("\\geq"), "l2norm latex: ≥ not converted");
    // α → \alpha
    assert!(latex.contains("\\alpha"), "l2norm latex: α not converted");
}

// ===========================================================================
// ISA detection verification
// ===========================================================================

#[test]
fn asm_isa_detection() {
    // All 5 fixtures have simd_dispatch with avx2
    for contract in &[
        "relu-kernel-v1.yaml",
        "clamp-kernel-v1.yaml",
        "dot-kernel-v1.yaml",
        "scale-kernel-v1.yaml",
        "l2norm-kernel-v1.yaml",
    ] {
        let asm = run_equations(contract, "asm");
        assert!(
            asm.contains("AVX2"),
            "{contract}: asm should detect AVX2 ISA"
        );
        assert!(
            asm.contains("ymm"),
            "{contract}: asm should use ymm registers for AVX2"
        );
        assert!(
            asm.contains("_avx2:"),
            "{contract}: asm function label should end with _avx2"
        );
    }
}

// ===========================================================================
// Kernel name derivation verification
// ===========================================================================

#[test]
fn ptx_kernel_names() {
    let cases: &[(&str, &str)] = &[
        ("relu-kernel-v1.yaml", ".visible .entry relu("),
        ("clamp-kernel-v1.yaml", ".visible .entry clamp("),
        ("dot-kernel-v1.yaml", ".visible .entry dot("),
        ("scale-kernel-v1.yaml", ".visible .entry scale("),
        ("l2norm-kernel-v1.yaml", ".visible .entry l2norm("),
    ];
    for &(contract, expected_entry) in cases {
        let ptx = run_equations(contract, "ptx");
        assert!(
            ptx.contains(expected_entry),
            "{contract}: expected PTX entry '{expected_entry}'"
        );
    }
}

#[test]
fn asm_function_labels() {
    let cases: &[(&str, &str)] = &[
        ("relu-kernel-v1.yaml", "relu_avx2:"),
        ("clamp-kernel-v1.yaml", "clamp_avx2:"),
        ("dot-kernel-v1.yaml", "dot_avx2:"),
        ("scale-kernel-v1.yaml", "scale_avx2:"),
        ("l2norm-kernel-v1.yaml", "l2norm_avx2:"),
    ];
    for &(contract, expected_label) in cases {
        let asm = run_equations(contract, "asm");
        assert!(
            asm.contains(expected_label),
            "{contract}: expected ASM label '{expected_label}'"
        );
    }
}

// ===========================================================================
// Phase count consistency: PTX and ASM must show same phases
// ===========================================================================

#[test]
fn phase_count_consistency() {
    let cases: &[(&str, usize)] = &[
        ("relu-kernel-v1.yaml", 2),
        ("clamp-kernel-v1.yaml", 3),
        ("dot-kernel-v1.yaml", 2),
        ("scale-kernel-v1.yaml", 1),
        ("l2norm-kernel-v1.yaml", 3),
    ];
    for &(contract, expected_phases) in cases {
        let ptx = run_equations(contract, "ptx");
        let asm = run_equations(contract, "asm");
        let ptx_phases = ptx.lines().filter(|l| l.contains("// Phase ")).count();
        let asm_phases = asm.lines().filter(|l| l.contains("// Phase ")).count();
        assert_eq!(
            ptx_phases, expected_phases,
            "{contract}: PTX has {ptx_phases} phases, expected {expected_phases}"
        );
        assert_eq!(
            asm_phases, expected_phases,
            "{contract}: ASM has {asm_phases} phases, expected {expected_phases}"
        );
    }
}
