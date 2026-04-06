//! Obligation-type-to-test-pattern body generators.

use crate::schema::ProofObligation;

pub(super) fn tolerance_str(ob: &ProofObligation) -> String {
    match ob.tolerance {
        Some(tol) => format!("{tol:e}"),
        None => "1e-6".to_string(),
    }
}

pub(super) fn generate_invariant_body(out: &mut String, ob: &ProofObligation) {
    let tol = tolerance_str(ob);
    out.push_str(
        "        // Pattern: invariant — property holds \
         for all inputs.\n",
    );
    out.push_str(
        "        // Generate random inputs and check \
         postcondition.\n",
    );
    out.push_str("        for _ in 0..1000 {\n");
    out.push_str("            // let input = generate_random_input();\n");
    out.push_str("            // let output = kernel(&input);\n");
    out.push_str(&format!(
        "            // assert!(postcondition(&output), \
         \"Invariant violated: {}\");\n",
        ob.property
    ));
    out.push_str("        }\n");
    out.push_str(&format!("        let _ = {tol}; // tolerance\n"));
    out.push_str(&format!(
        "        unimplemented!(\"Wire up: {}\")\n",
        ob.property
    ));
}

pub(super) fn generate_equivalence_body(out: &mut String, ob: &ProofObligation) {
    let tol = tolerance_str(ob);
    out.push_str(
        "        // Pattern: equivalence — two implementations \
         must agree.\n",
    );
    out.push_str("        // Compare reference vs optimized within tolerance.\n");
    out.push_str("        for _ in 0..1000 {\n");
    out.push_str("            // let input = generate_random_input();\n");
    out.push_str("            // let ref_out = reference_impl(&input);\n");
    out.push_str("            // let opt_out = optimized_impl(&input);\n");
    out.push_str(&format!(
        "            // assert!(max_ulp_diff(&ref_out, &opt_out) \
         <= {tol});\n"
    ));
    out.push_str("        }\n");
    out.push_str(&format!(
        "        unimplemented!(\"Wire up: {}\")\n",
        ob.property
    ));
}

pub(super) fn generate_bound_body(out: &mut String, ob: &ProofObligation) {
    out.push_str("        // Pattern: bound — all outputs within range.\n");
    out.push_str("        for _ in 0..1000 {\n");
    out.push_str("            // let input = generate_random_input();\n");
    out.push_str("            // let output = kernel(&input);\n");
    out.push_str("            // for val in &output {\n");
    out.push_str("            //     assert!(lo <= *val && *val <= hi);\n");
    out.push_str("            // }\n");
    out.push_str("        }\n");
    out.push_str(&format!(
        "        unimplemented!(\"Wire up: {}\")\n",
        ob.property
    ));
}

pub(super) fn generate_monotonicity_body(out: &mut String, ob: &ProofObligation) {
    out.push_str(
        "        // Pattern: monotonicity — order preserved \
         in output.\n",
    );
    out.push_str("        // Metamorphic: if x_i > x_j then f(x)_i > f(x)_j.\n");
    out.push_str("        for _ in 0..1000 {\n");
    out.push_str("            // let input = generate_random_input();\n");
    out.push_str("            // let output = kernel(&input);\n");
    out.push_str("            // for i in 0..input.len() {\n");
    out.push_str("            //     for j in 0..input.len() {\n");
    out.push_str("            //         if input[i] > input[j] {\n");
    out.push_str("            //             assert!(output[i] > output[j]);\n");
    out.push_str("            //         }\n");
    out.push_str("            //     }\n");
    out.push_str("            // }\n");
    out.push_str("        }\n");
    out.push_str(&format!(
        "        unimplemented!(\"Wire up: {}\")\n",
        ob.property
    ));
}

pub(super) fn generate_idempotency_body(out: &mut String, ob: &ProofObligation) {
    let tol = tolerance_str(ob);
    out.push_str("        // Pattern: idempotency — f(f(x)) == f(x).\n");
    out.push_str("        for _ in 0..1000 {\n");
    out.push_str("            // let input = generate_random_input();\n");
    out.push_str("            // let once = kernel(&input);\n");
    out.push_str("            // let twice = kernel(&once);\n");
    out.push_str(&format!(
        "            // assert!(max_diff(&once, &twice) < {tol});\n"
    ));
    out.push_str("        }\n");
    out.push_str(&format!(
        "        unimplemented!(\"Wire up: {}\")\n",
        ob.property
    ));
}

pub(super) fn generate_linearity_body(out: &mut String, ob: &ProofObligation) {
    let tol = tolerance_str(ob);
    out.push_str("        // Pattern: linearity — f(α·x) == α·f(x).\n");
    out.push_str("        // Metamorphic: scale input, compare scaled output.\n");
    out.push_str("        for _ in 0..1000 {\n");
    out.push_str("            // let input = generate_random_input();\n");
    out.push_str("            // let alpha: f32 = random_nonzero();\n");
    out.push_str(
        "            // let scaled: Vec<f32> = input.iter()\
         .map(|x| alpha * x).collect();\n",
    );
    out.push_str("            // let f_scaled = kernel(&scaled);\n");
    out.push_str(
        "            // let scaled_f: Vec<f32> = kernel(&input).iter()\
         .map(|x| alpha * x).collect();\n",
    );
    out.push_str(&format!(
        "            // assert!(max_diff(&f_scaled, &scaled_f) \
         < {tol});\n"
    ));
    out.push_str("        }\n");
    out.push_str(&format!(
        "        unimplemented!(\"Wire up: {}\")\n",
        ob.property
    ));
}

pub(super) fn generate_symmetry_body(out: &mut String, ob: &ProofObligation) {
    out.push_str("        // Pattern: symmetry — permutation invariance.\n");
    out.push_str("        // f(permute(x)) is related to permute(f(x)).\n");
    out.push_str("        for _ in 0..1000 {\n");
    out.push_str("            // let input = generate_random_input();\n");
    out.push_str("            // let perm = random_permutation(input.len());\n");
    out.push_str("            // let permuted_input = apply_perm(&input, &perm);\n");
    out.push_str("            // let out_orig = kernel(&input);\n");
    out.push_str("            // let out_perm = kernel(&permuted_input);\n");
    out.push_str(
        "            // assert!(check_symmetry(&out_orig, \
         &out_perm, &perm));\n",
    );
    out.push_str("        }\n");
    out.push_str(&format!(
        "        unimplemented!(\"Wire up: {}\")\n",
        ob.property
    ));
}

pub(super) fn generate_associativity_body(out: &mut String, ob: &ProofObligation) {
    let tol = tolerance_str(ob);
    out.push_str(
        "        // Pattern: associativity — \
         (a ⊕ b) ⊕ c == a ⊕ (b ⊕ c).\n",
    );
    out.push_str("        for _ in 0..1000 {\n");
    out.push_str("            // let a = generate_random_input();\n");
    out.push_str("            // let b = generate_random_input();\n");
    out.push_str("            // let c = generate_random_input();\n");
    out.push_str("            // let left = op(&op(&a, &b), &c);\n");
    out.push_str("            // let right = op(&a, &op(&b, &c));\n");
    out.push_str(&format!(
        "            // assert!(max_diff(&left, &right) < {tol});\n"
    ));
    out.push_str("        }\n");
    out.push_str(&format!(
        "        unimplemented!(\"Wire up: {}\")\n",
        ob.property
    ));
}

pub(super) fn generate_ordering_body(out: &mut String, ob: &ProofObligation) {
    out.push_str(
        "        // Pattern: ordering — elements maintain \
         a defined order relation.\n",
    );
    out.push_str("        for _ in 0..1000 {\n");
    out.push_str("            // let items = generate_random_items();\n");
    out.push_str("            // let result = transform(&items);\n");
    out.push_str("            // assert!(is_ordered(&result));\n");
    out.push_str("        }\n");
    out.push_str(&format!(
        "        unimplemented!(\"Wire up: {}\")\n",
        ob.property
    ));
}

pub(super) fn generate_conservation_body(out: &mut String, ob: &ProofObligation) {
    let tol = tolerance_str(ob);
    out.push_str(
        "        // Pattern: conservation — conserved \
         quantity unchanged.\n",
    );
    out.push_str("        // Q(state_before) == Q(state_after).\n");
    out.push_str("        for _ in 0..1000 {\n");
    out.push_str("            // let state = generate_random_state();\n");
    out.push_str("            // let q_before = conserved_quantity(&state);\n");
    out.push_str("            // let new_state = transform(&state);\n");
    out.push_str("            // let q_after = conserved_quantity(&new_state);\n");
    out.push_str(&format!(
        "            // assert!((q_before - q_after).abs() \
         < {tol});\n"
    ));
    out.push_str("        }\n");
    out.push_str(&format!(
        "        unimplemented!(\"Wire up: {}\")\n",
        ob.property
    ));
}
