//! Strategy-specific body generators for Kani harnesses.

use crate::schema::KaniHarness;

pub(super) fn generate_exhaustive_body(out: &mut String, harness: &KaniHarness) {
    let bound = harness.bound.unwrap_or(16);
    out.push_str("        // Strategy: exhaustive — exact verification\n");
    out.push_str(
        "        // Integer/structural arithmetic verified \
         without approximation.\n",
    );
    out.push_str(&format!("        // Bound: {bound} elements\n\n"));

    // Generate symbolic input pattern
    out.push_str("        let n: usize = kani::any();\n");
    out.push_str(&format!(
        "        kani::assume(n >= 1 && n <= {bound});\n\n"
    ));

    out.push_str("        // Symbolic inputs — Kani explores ALL possible values\n");
    out.push_str(
        "        let input: Vec<i32> = (0..n)\
         .map(|_| kani::any()).collect();\n\n",
    );

    // Generate assertion based on property
    if let Some(ref property) = harness.property {
        out.push_str(&format!("        // Verify: {property}\n"));
    }
    out.push_str(&format!("        // Obligation: {}\n", harness.obligation));
    out.push_str(
        "        // TODO: Replace with kernel-specific \
         verification logic\n",
    );
    out.push_str("        //   Example: assert_eq!(precomputed, online);\n");
    out.push_str("        unimplemented!(\"Wire up kernel under test\")\n");
}

pub(super) fn generate_stub_float_body(out: &mut String, harness: &KaniHarness) {
    let bound = harness.bound.unwrap_or(16);
    out.push_str(
        "        // Strategy: stub_float — stub transcendentals, \
         verify structural properties.\n",
    );
    out.push_str(
        "        // Floating-point transcendentals (exp, log, sin, cos) \
         are replaced\n",
    );
    out.push_str(
        "        // with contract stubs that preserve structural \
         invariants.\n\n",
    );

    // Symbolic vector input
    out.push_str("        let n: usize = kani::any();\n");
    out.push_str(&format!(
        "        kani::assume(n >= 1 && n <= {bound});\n\n"
    ));

    out.push_str(
        "        // Symbolic input: every possible finite \
         f32 vector of length n\n",
    );
    out.push_str(
        "        let input: Vec<f32> = (0..n)\
         .map(|_| kani::any()).collect();\n",
    );
    out.push_str(
        "        kani::assume(input.iter()\
         .all(|x| x.is_finite()));\n\n",
    );

    out.push_str("        let mut output = vec![0.0f32; n];\n");

    // Generate postconditions from property
    if let Some(ref property) = harness.property {
        out.push_str(&format!("\n        // Post-condition: {property}\n"));
    }

    out.push_str("        // TODO: Call kernel under test:\n");
    out.push_str("        //   kernel_fn(&input, &mut output);\n");
    out.push_str("        //\n");
    out.push_str("        // Then assert postconditions, e.g.:\n");
    out.push_str("        //   let sum: f32 = output.iter().sum();\n");
    out.push_str("        //   assert!((sum - 1.0).abs() < 1e-5);\n");
    out.push_str("        //   assert!(output.iter().all(|&x| x > 0.0));\n");
    out.push_str("        unimplemented!(\"Wire up kernel under test\")\n");
}

pub(super) fn generate_compositional_body(out: &mut String, harness: &KaniHarness) {
    out.push_str(
        "        // Strategy: compositional — \
         use #[kani::stub_verified] for sub-kernels.\n",
    );
    out.push_str(
        "        // Verified sub-components are replaced with \
         their contract abstractions,\n",
    );
    out.push_str(
        "        // providing free postconditions without \
         re-verifying internals.\n\n",
    );

    let bound = harness.bound.unwrap_or(8);
    out.push_str("        let n: usize = kani::any();\n");
    out.push_str(&format!(
        "        kani::assume(n >= 1 && n <= {bound});\n\n"
    ));

    out.push_str(
        "        let input: Vec<f32> = (0..n)\
         .map(|_| kani::any()).collect();\n",
    );
    out.push_str(
        "        kani::assume(input.iter()\
         .all(|x| x.is_finite()));\n\n",
    );

    if let Some(ref property) = harness.property {
        out.push_str(&format!("        // Verify: {property}\n"));
    }
    out.push_str(
        "        // TODO: Add #[kani::stub_verified(sub_fn)] \
         to harness attribute\n",
    );
    out.push_str(
        "        //   then call composed kernel and assert \
         end-to-end properties.\n",
    );
    out.push_str("        unimplemented!(\"Wire up kernel under test\")\n");
}

pub(super) fn generate_bounded_int_body(out: &mut String, harness: &KaniHarness) {
    let bound = harness.bound.unwrap_or(16);
    out.push_str(
        "        // Strategy: bounded_int — integer-only \
         verification within bounded range.\n",
    );
    out.push_str(
        "        // No floating-point — all inputs are \
         bounded integers or indices.\n\n",
    );

    out.push_str("        let n: usize = kani::any();\n");
    out.push_str(&format!(
        "        kani::assume(n >= 1 && n <= {bound});\n\n"
    ));

    out.push_str(
        "        let input: Vec<i64> = (0..n)\
         .map(|_| kani::any()).collect();\n",
    );
    out.push_str(&format!(
        "        kani::assume(input.iter()\
         .all(|&x| x >= -{bound} as i64 && x <= {bound} as i64));\n\n"
    ));

    if let Some(ref property) = harness.property {
        out.push_str(&format!("        // Verify: {property}\n"));
    }
    out.push_str(&format!("        // Obligation: {}\n", harness.obligation));
    out.push_str("        unimplemented!(\"Wire up kernel under test\")\n");
}

pub(super) fn generate_default_body(out: &mut String, harness: &KaniHarness) {
    let bound = harness.bound.unwrap_or(16);
    out.push_str(
        "        // No strategy specified — using default \
         symbolic verification pattern.\n\n",
    );

    out.push_str("        let n: usize = kani::any();\n");
    out.push_str(&format!(
        "        kani::assume(n >= 1 && n <= {bound});\n\n"
    ));

    out.push_str(
        "        let input: Vec<f32> = (0..n)\
         .map(|_| kani::any()).collect();\n",
    );
    out.push_str(
        "        kani::assume(input.iter()\
         .all(|x| x.is_finite()));\n\n",
    );

    if let Some(ref property) = harness.property {
        out.push_str(&format!("        // Verify: {property}\n"));
    }
    out.push_str(&format!("        // Obligation: {}\n", harness.obligation));
    out.push_str("        unimplemented!(\"Wire up kernel under test\")\n");
}
