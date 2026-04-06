//! Jidoka Guards Example
//!
//! Demonstrates simular's Jidoka (自働化) anomaly detection:
//! - NaN and Infinity detection
//! - Energy drift monitoring
//! - Severity classification
//! - Automatic halt on anomalies
//!
//! # Running
//! ```bash
//! cargo run --example jidoka_guards
//! ```

use simular::engine::jidoka::{
    JidokaConfig, JidokaGuard, JidokaWarning, SeverityClassifier, ViolationSeverity,
};
use simular::engine::state::{SimState, Vec3};

fn main() {
    println!("=== Simular Jidoka (自働化) Anomaly Detection ===\n");
    println!("\"Stop the line when defects occur\" - Toyota Production System\n");

    // 1. Basic Jidoka Guard Setup
    println!("1. Basic Jidoka Guard Configuration:");

    let config = JidokaConfig {
        energy_tolerance: 1e-6,     // 0.0001% energy drift allowed
        check_finite: true,         // Check for NaN/Inf
        constraint_tolerance: 0.01, // 1% constraint violation allowed
        check_energy: true,         // Enable energy checking
        severity_classifier: SeverityClassifier::default(),
    };

    let mut guard = JidokaGuard::new(config.clone());
    println!("   Energy tolerance: {:.0e}", config.energy_tolerance);
    println!("   Finite check: {}", config.check_finite);
    println!(
        "   Constraint tolerance: {:.2}%\n",
        config.constraint_tolerance * 100.0
    );

    // 2. Valid State Check
    println!("2. Checking Valid State:");

    let mut state = SimState::new();
    state.add_body(1.0, Vec3::new(1.0, 2.0, 3.0), Vec3::new(0.1, 0.2, 0.3));

    match guard.check(&state) {
        Ok(()) => println!("   Valid state: PASSED"),
        Err(e) => println!("   Valid state: FAILED - {:?}", e),
    }

    // 3. NaN Detection Demo
    println!("\n3. NaN Detection (Poka-Yoke):");
    println!("   Injecting NaN into position...\n");

    let mut nan_state = SimState::new();
    nan_state.add_body(1.0, Vec3::new(f64::NAN, 2.0, 3.0), Vec3::new(0.1, 0.2, 0.3));

    let mut guard = JidokaGuard::new(config.clone());
    match guard.check(&nan_state) {
        Ok(()) => println!("   NaN check: MISSED (unexpected!)"),
        Err(e) => println!("   NaN detected: {:?}", e),
    }

    // 4. Infinity Detection Demo
    println!("\n4. Infinity Detection:");
    println!("   Injecting Inf into velocity...\n");

    let mut inf_state = SimState::new();
    inf_state.add_body(
        1.0,
        Vec3::new(1.0, 2.0, 3.0),
        Vec3::new(f64::INFINITY, 0.2, 0.3),
    );

    let mut guard = JidokaGuard::new(config.clone());
    match guard.check(&inf_state) {
        Ok(()) => println!("   Inf check: MISSED (unexpected!)"),
        Err(e) => println!("   Infinity detected: {:?}", e),
    }

    // 5. Severity Classification
    println!("\n5. Severity Classification (Graduated Response):");
    println!("   NASA-style severity levels for anomaly response\n");

    let classifier = SeverityClassifier::new(0.8); // Warn at 80% of tolerance

    let test_cases = [
        (0.0, "No drift"),
        (0.5e-6, "50% of tolerance"),
        (0.85e-6, "85% of tolerance (warning zone)"),
        (1.0e-6, "At tolerance"),
        (1.5e-6, "150% of tolerance (critical)"),
        (f64::NAN, "NaN (fatal)"),
    ];

    for (drift, description) in &test_cases {
        let severity = classifier.classify_energy_drift(*drift, 1e-6);
        let symbol = match severity {
            ViolationSeverity::Acceptable => "+",
            ViolationSeverity::Warning => "!",
            ViolationSeverity::Critical => "X",
            ViolationSeverity::Fatal => "#",
        };
        println!("   {} {:?}: {} ({})", symbol, severity, description, drift);
    }

    // 6. Jidoka Warning Types
    println!("\n6. Jidoka Warning Types:");

    let warnings = [
        JidokaWarning::EnergyDriftApproaching {
            drift: 0.8e-6,
            tolerance: 1e-6,
        },
        JidokaWarning::ConstraintApproaching {
            name: "velocity_magnitude".to_string(),
            violation: 0.008,
            tolerance: 0.01,
        },
    ];

    for warning in &warnings {
        match warning {
            JidokaWarning::EnergyDriftApproaching { drift, tolerance } => {
                println!(
                    "   Energy warning: drift={:.2e}, tolerance={:.2e}",
                    drift, tolerance
                );
            }
            JidokaWarning::ConstraintApproaching {
                name,
                violation,
                tolerance,
            } => {
                println!(
                    "   Constraint warning: {}={:.4}, tolerance={:.4}",
                    name, violation, tolerance
                );
            }
        }
    }

    // 7. Energy Conservation Monitoring
    println!("\n7. Energy Conservation Monitoring:");
    println!("   Simulating physics with energy tracking...\n");

    let config = JidokaConfig {
        energy_tolerance: 1e-6,
        check_finite: true,
        constraint_tolerance: 0.01,
        check_energy: true,
        severity_classifier: SeverityClassifier::default(),
    };

    let mut guard = JidokaGuard::new(config);
    let mut state = SimState::new();
    state.add_body(1.0, Vec3::new(0.0, 0.0, 100.0), Vec3::zero());
    state.set_potential_energy(981.0); // mgh = 1 * 9.81 * 100

    // Initial check establishes baseline
    let _ = guard.check(&state);
    println!(
        "   Initial total energy: {:.2} J",
        state.kinetic_energy() + state.potential_energy()
    );

    // Simulate energy-conserving step (fall halfway)
    state.set_position(0, Vec3::new(0.0, 0.0, 50.0));
    state.set_potential_energy(490.5); // Lost PE becomes KE

    match guard.check(&state) {
        Ok(()) => println!("   After step (conserved): PASSED"),
        Err(e) => println!("   After step: FAILED - {:?}", e),
    }

    // 8. Summary
    println!("\n8. Jidoka Principles Summary:");
    println!("   - Anomaly detection at every step (Poka-Yoke)");
    println!("   - Graduated response (Acceptable -> Warning -> Critical -> Fatal)");
    println!("   - Immediate halt on critical violations");
    println!("   - Root cause preserved for analysis (Genchi Genbutsu)");

    println!("\n=== Jidoka Guards Active ===");
    println!("\"Quality at the source\" - defects caught before propagation");
}
