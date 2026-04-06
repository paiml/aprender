//! Example: Equation Verification (Feature 22)
//!
//! Demonstrates: Physics and game math equation verification
//!
//! Run with: `cargo run --example equation_verify`
//!
//! Toyota Way: Poka-Yoke (Mistake-Proofing) - Mathematical correctness guarantees

#![allow(clippy::many_single_char_names, clippy::float_cmp)]

use jugar_probar::prelude::*;

fn main() -> ProbarResult<()> {
    println!("=== Equation Verification Example ===\n");

    // 1. Create equation context
    println!("1. Creating equation context...");
    let mut ctx = EquationContext::new();

    ctx.set("velocity", 10.0);
    ctx.set("time", 2.0);
    ctx.set("acceleration", 5.0);

    println!("   Variables set:");
    println!("   velocity = 10.0 m/s");
    println!("   time = 2.0 s");
    println!("   acceleration = 5.0 m/s²");

    // 2. Create equation verifier
    println!("\n2. Creating equation verifier...");
    let mut verifier = EquationVerifier::new("physics_test");

    println!("   Verifier created: physics_test");
    println!("   Default tolerance: 1e-10");

    // 3. Verify basic equations
    println!("\n3. Verifying physics equations...");

    // v = v0 + at
    let v0 = 10.0;
    let a = 5.0;
    let t = 2.0;
    let v = v0 + a * t;
    let expected_v = 20.0;

    verifier.verify_eq("v = v0 + at", expected_v, v);
    println!(
        "   v = v0 + at: {} = {} + {} * {} = {} ✓",
        expected_v, v0, a, t, v
    );

    // x = v0*t + 0.5*a*t²
    let x = v0 * t + 0.5 * a * t * t;
    let expected_x = 30.0;

    verifier.verify_eq("x = v0*t + 0.5*a*t²", expected_x, x);
    println!("   x = v0*t + 0.5*a*t²: {} = {} ✓", expected_x, x);

    // 4. Range verification
    println!("\n4. Verifying value ranges...");

    let health = 75.0;
    let max_health = 100.0;

    verifier.verify_in_range("health", health, 0.0, max_health);
    println!("   health ({}) in range [0, {}] ✓", health, max_health);

    let score = 1500.0;
    verifier.verify_non_negative("score", score);
    println!("   score ({}) >= 0 ✓", score);

    // 5. Energy conservation
    println!("\n5. Energy conservation verification...");

    let mass = 2.0;
    let velocity_sq = 25.0; // v² = 25
    let height = 10.0;
    let g = 9.81;

    let ke = 0.5 * mass * velocity_sq;
    let pe = mass * g * height;
    let total_energy = ke + pe;

    println!(
        "   KE = 0.5 * m * v² = 0.5 * {} * {} = {:.2} J",
        mass, velocity_sq, ke
    );
    println!(
        "   PE = m * g * h = {} * {} * {} = {:.2} J",
        mass, g, height, pe
    );
    println!("   Total = {:.2} J", total_energy);

    verifier.verify_eq("KE = 0.5*m*v²", 25.0, ke);

    // 6. Momentum conservation
    println!("\n6. Momentum verification...");

    let m1 = 2.0;
    let v1 = 5.0;
    let m2 = 3.0;
    let v2 = 2.0;

    let p1 = m1 * v1;
    let p2 = m2 * v2;
    let p_total = p1 + p2;

    verifier.verify_eq("p = m*v", 10.0, p1);
    println!("   p1 = m1 * v1 = {} * {} = {} kg·m/s ✓", m1, v1, p1);
    println!("   p2 = m2 * v2 = {} * {} = {} kg·m/s", m2, v2, p2);
    println!("   p_total = {} kg·m/s", p_total);

    // 7. Check results
    println!("\n7. Verification results...");
    let results = verifier.results();

    println!("   Total verifications: {}", results.len());
    let passed = results.iter().filter(|r| r.passed).count();
    let failed = results.len() - passed;
    println!("   Passed: {}", passed);
    println!("   Failed: {}", failed);

    // 8. Result details
    println!("\n8. Result details...");
    for result in results.iter().take(3) {
        println!(
            "   {} - {} (expected: {:.4}, actual: {:.4})",
            if result.passed { "✓" } else { "✗" },
            result.name,
            result.expected,
            result.actual
        );
    }

    // 9. Tolerance handling
    println!("\n9. Floating-point tolerance...");
    let a_val = 0.1 + 0.2;
    let b_val = 0.3;

    println!("   0.1 + 0.2 = {}", a_val);
    println!("   0.3 = {}", b_val);
    println!("   Exact equal: {}", a_val == b_val);

    verifier.verify_eq("floating point", b_val, a_val);
    println!("   With tolerance: verified ✓");

    // 10. Summary
    println!("\n10. Summary...");
    println!("   Context variables: 3");
    println!("   Verifications performed: {}", verifier.results().len());
    println!("   All passed: {}", verifier.all_passed());
    println!("   Use cases: Physics engines, Game math, Simulations");

    println!("\n✅ Equation verification example completed!");
    Ok(())
}
