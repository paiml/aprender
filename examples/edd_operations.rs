//! Equation-Driven Development: Operations Science Examples
//!
//! This example demonstrates the four governing equations of operations science
//! implemented in the EDD framework:
//!
//! 1. Little's Law: L = λW
//! 2. Kingman's Formula: VUT equation
//! 3. Square Root Law: Safety stock scaling
//! 4. Bullwhip Effect: Variance amplification
//!
//! Run with: cargo run --example edd_operations

use simular::edd::{BullwhipEffect, GoverningEquation, KingmanFormula, LittlesLaw, SquareRootLaw};

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║     Equation-Driven Development: Operations Science           ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    demonstrate_littles_law();
    demonstrate_kingmans_formula();
    demonstrate_square_root_law();
    demonstrate_bullwhip_effect();

    println!("\n✓ All operations science equations demonstrated successfully!");
}

fn demonstrate_littles_law() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("📐 Little's Law: L = λW");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let law = LittlesLaw::new();

    // Display equation info
    println!("Equation: {}", law.latex());
    println!("Citation: {}\n", law.citation());

    // Test cases from Factory Physics
    let test_cases = [
        (5.0, 2.0, "λ=5 items/hr, W=2 hrs"),
        (10.0, 1.5, "λ=10 items/hr, W=1.5 hrs"),
        (2.5, 4.0, "λ=2.5 items/hr, W=4 hrs"),
    ];

    println!("Evaluations:");
    for (lambda, w, desc) in test_cases {
        let l = law.evaluate(lambda, w);
        println!("  {desc} → L = {l:.1} items");
    }

    // Validation example
    println!("\nValidation:");
    let valid = law.validate(10.0, 5.0, 2.0, 0.01);
    println!(
        "  L=10, λ=5, W=2, tol=1%: {}",
        if valid.is_ok() {
            "✓ VALID"
        } else {
            "✗ INVALID"
        }
    );

    let invalid = law.validate(15.0, 5.0, 2.0, 0.01);
    println!(
        "  L=15, λ=5, W=2, tol=1%: {}",
        if invalid.is_ok() {
            "✓ VALID"
        } else {
            "✗ INVALID (expected)"
        }
    );

    println!();
}

fn demonstrate_kingmans_formula() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("📐 Kingman's Formula (VUT Equation)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let formula = KingmanFormula::new();

    println!("Equation: {}", formula.latex());
    println!("Citation: {}\n", formula.citation());

    // Demonstrate the "hockey stick" effect
    println!("The Hockey Stick Effect (c_a=1, c_s=1, t_s=1):");
    println!("┌────────────────┬────────────────┬────────────────┐");
    println!("│  Utilization   │   Wait Time    │    Ratio to    │");
    println!("│      (ρ)       │     (W_q)      │     ρ=50%      │");
    println!("├────────────────┼────────────────┼────────────────┤");

    let base_wait = formula.expected_wait_time(0.5, 1.0, 1.0, 1.0);
    let utilizations = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99];

    for rho in utilizations {
        let wait = formula.expected_wait_time(rho, 1.0, 1.0, 1.0);
        let ratio = wait / base_wait;
        let pct = rho * 100.0;
        println!("│      {pct:>4.0}%     │     {wait:>6.2}     │     {ratio:>5.1}x     │");
    }
    println!("└────────────────┴────────────────┴────────────────┘");

    println!("\n⚠️  Key Insight: At 95% utilization, wait times are 19x higher than at 50%!");
    println!("   This is why Lean/TPS avoids high utilization.\n");
}

fn demonstrate_square_root_law() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("📐 Square Root Law: I_safety = z × σ_D × √L");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let law = SquareRootLaw::new();

    println!("Equation: {}", law.latex());
    println!("Citation: {}\n", law.citation());

    // Demonstrate non-linear scaling
    let sigma_d = 100.0; // Demand std dev
    let z = 1.96; // 97.5% service level

    println!("Safety Stock vs Lead Time (σ_D=100, z=1.96 for 97.5% service):");
    println!("┌────────────────┬────────────────┬────────────────┐");
    println!("│   Lead Time    │  Safety Stock  │  Linear Would  │");
    println!("│      (L)       │    (actual)    │     Be...      │");
    println!("├────────────────┼────────────────┼────────────────┤");

    let base_stock = law.safety_stock(sigma_d, 1.0, z);
    for lead_time in [1.0, 4.0, 9.0, 16.0, 25.0] {
        let stock = law.safety_stock(sigma_d, lead_time, z);
        let linear_would_be = base_stock * lead_time;
        println!("│     {lead_time:>5.0}       │     {stock:>6.0}     │     {linear_would_be:>6.0}     │");
    }
    println!("└────────────────┴────────────────┴────────────────┘");

    println!("\n⚠️  Key Insight: If lead time quadruples (1→4), safety stock only doubles!");
    println!("   Linear thinking would over-invest by 2x.\n");
}

fn demonstrate_bullwhip_effect() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("📐 Bullwhip Effect: Variance Amplification");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let effect = BullwhipEffect::new();

    println!("Equation: {}", effect.latex());
    println!("Citation: {}\n", effect.citation());

    // Demonstrate amplification across supply chain
    println!("Variance Amplification (L=lead time, p=forecast periods):");
    println!("┌──────────┬──────────┬────────────────────┐");
    println!("│    L     │    p     │   Min Amplification │");
    println!("├──────────┼──────────┼────────────────────┤");

    let scenarios = [
        (1, 1, "Worst case"),
        (2, 2, "Typical"),
        (4, 4, "Balanced"),
        (2, 10, "Long MA"),
        (10, 2, "Long lead time"),
    ];

    for (l, p, _desc) in scenarios {
        let amp = effect.amplification_factor(l as f64, p as f64);
        println!("│    {l:>2}    │    {p:>2}    │       {amp:>6.2}x        │");
    }
    println!("└──────────┴──────────┴────────────────────┘");

    // Multi-echelon example
    println!("\nMulti-Echelon Amplification (L=2, p=4):");
    let single_amp = effect.amplification_factor(2.0, 4.0);
    println!("  Retailer → Wholesaler:   {single_amp:.2}x");
    println!(
        "  Wholesaler → Distributor: {:.2}x (cumulative: {:.2}x)",
        single_amp,
        single_amp * single_amp
    );
    println!(
        "  Distributor → Manufacturer: {:.2}x (cumulative: {:.2}x)",
        single_amp,
        single_amp * single_amp * single_amp
    );

    println!("\n⚠️  Key Insight: Heijunka (production leveling) acts as a low-pass filter,");
    println!("   dampening demand variability before it amplifies upstream.\n");
}
