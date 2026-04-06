//! Example: Playbook State Machine Testing
//!
//! Demonstrates: YAML-driven state machine verification with mutation testing
//!
//! Run with: `cargo run --example playbook_testing`
//!
//! Toyota Way: Genchi Genbutsu - Go and see the state transitions yourself
//!
//! References:
//! - W3C SCXML: https://www.w3.org/TR/scxml/
//! - Fabbri et al., "Mutation Testing Applied to Statecharts" (ISSRE 1999)

use jugar_probar::playbook::{
    calculate_mutation_score, to_dot, to_svg, ComplexityAnalyzer, ComplexityClass, MutantResult,
    MutationClass, MutationGenerator, Playbook, StateMachineValidator,
};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║         Probar Playbook State Machine Testing                ║");
    println!("║         YAML-Driven Verification with Mutation Testing       ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Example: Login Flow State Machine
    let login_playbook_yaml = r##"
version: "1.0"
name: "Login Flow Test"
description: "Verify login state machine behavior"
machine:
  id: "login_flow"
  initial: "logged_out"
  states:
    logged_out:
      id: "logged_out"
      invariants:
        - description: "Login button visible"
          condition: "has_element('#login-btn')"
    authenticating:
      id: "authenticating"
      invariants:
        - description: "Loading spinner visible"
          condition: "has_element('.spinner')"
    logged_in:
      id: "logged_in"
      final_state: true
      invariants:
        - description: "Welcome message visible"
          condition: "has_element('#welcome')"
    error:
      id: "error"
      invariants:
        - description: "Error message visible"
          condition: "has_element('.error')"
  transitions:
    - id: "submit_credentials"
      from: "logged_out"
      to: "authenticating"
      event: "submit"
      actions:
        - type: click
          selector: "#login-btn"
    - id: "auth_success"
      from: "authenticating"
      to: "logged_in"
      event: "auth_ok"
    - id: "auth_failure"
      from: "authenticating"
      to: "error"
      event: "auth_fail"
    - id: "retry_login"
      from: "error"
      to: "logged_out"
      event: "retry"
    - id: "logout"
      from: "logged_in"
      to: "logged_out"
      event: "logout"
      guard: "session.valid"
  forbidden:
    - from: "logged_out"
      to: "logged_in"
      reason: "Cannot skip authentication"
    - from: "error"
      to: "logged_in"
      reason: "Cannot login from error state"
performance:
  max_duration_ms: 5000
  max_memory_mb: 100
"##;

    // 1. Parse the playbook
    println!("1. Parsing YAML Playbook");
    println!("   ─────────────────────────────────────────────────────────");

    let playbook = match Playbook::from_yaml(login_playbook_yaml) {
        Ok(p) => {
            println!("   ✓ Playbook parsed successfully");
            println!("     Name: {}", p.name);
            println!("     Machine ID: {}", p.machine.id);
            println!("     States: {}", p.machine.states.len());
            println!("     Transitions: {}", p.machine.transitions.len());
            println!("     Forbidden: {}", p.machine.forbidden.len());
            p
        }
        Err(e) => {
            eprintln!("   ✗ Parse error: {e}");
            return;
        }
    };

    // 2. Validate State Machine Properties
    println!("\n2. State Machine Validation");
    println!("   ─────────────────────────────────────────────────────────");

    let validator = StateMachineValidator::new(&playbook);
    let validation = validator.validate();

    println!(
        "   Valid: {}",
        if validation.is_valid {
            "✓ Yes"
        } else {
            "✗ No"
        }
    );
    println!(
        "   Reachable states: {:?}",
        validation.reachability.reachable_states
    );
    println!(
        "   Orphaned states: {:?}",
        validation.reachability.orphaned_states
    );
    println!(
        "   Can reach final: {}",
        validation.reachability.can_reach_final
    );
    println!(
        "   Deterministic: {}",
        validation.determinism.is_deterministic
    );

    if !validation.issues.is_empty() {
        println!("   Issues found:");
        for issue in &validation.issues {
            println!("     - {issue:?}");
        }
    }

    // 3. Generate State Diagram
    println!("\n3. State Diagram Generation");
    println!("   ─────────────────────────────────────────────────────────");

    let dot = to_dot(&playbook);
    println!("   DOT format ({} bytes):", dot.len());
    println!("   ┌────────────────────────────────────────────────────────┐");
    for line in dot.lines().take(10) {
        println!("   │ {}", line);
    }
    println!("   │ ...");
    println!("   └────────────────────────────────────────────────────────┘");

    let svg = to_svg(&playbook);
    println!("\n   SVG format ({} bytes):", svg.len());
    println!("   (Use `probar playbook login.yaml --export svg` for full export)");

    // Save SVG to book assets for documentation
    let svg_path = std::path::Path::new("book/src/assets/login_state_machine.svg");
    if svg_path.parent().map(|p| p.exists()).unwrap_or(false) {
        if let Err(e) = std::fs::write(svg_path, &svg) {
            eprintln!("   Warning: Could not save SVG to book: {e}");
        } else {
            println!("   ✓ Saved SVG to book/src/assets/login_state_machine.svg");
        }
    }

    // 4. Mutation Testing (M1-M5 Falsification)
    println!("\n4. Mutation Testing (Falsification Protocol)");
    println!("   ─────────────────────────────────────────────────────────");
    println!("   Reference: Fabbri et al., ISSRE 1999\n");

    let generator = MutationGenerator::new(&playbook);

    println!("   Mutation Classes:");
    println!("   ┌────────┬──────────────────────────────────┬─────────┐");
    println!("   │ Class  │ Description                      │ Mutants │");
    println!("   ├────────┼──────────────────────────────────┼─────────┤");

    let mut all_mutants = Vec::new();
    for class in MutationClass::all() {
        let mutants = generator.generate(class);
        println!(
            "   │ {:6} │ {:32} │ {:7} │",
            class.id(),
            class.description(),
            mutants.len()
        );
        all_mutants.extend(mutants);
    }
    println!("   └────────┴──────────────────────────────────┴─────────┘");
    println!("   Total mutants: {}", all_mutants.len());

    // Simulate mutation test results (in real usage, these come from execution)
    let simulated_results: Vec<MutantResult> = all_mutants
        .iter()
        .enumerate()
        .map(|(i, m)| MutantResult {
            mutant_id: m.id.clone(),
            class: m.class,
            // Simulate 85% kill rate
            killed: i % 7 != 0,
            kill_reason: if i % 7 != 0 {
                Some("Test detected mutation".to_string())
            } else {
                None
            },
        })
        .collect();

    let score = calculate_mutation_score(&simulated_results);
    println!("\n   Mutation Score Results:");
    println!("   ┌────────────────────────────────┬─────────┐");
    println!("   │ Metric                         │ Value   │");
    println!("   ├────────────────────────────────┼─────────┤");
    println!(
        "   │ Total mutants                  │ {:7} │",
        score.total_mutants
    );
    println!("   │ Killed                         │ {:7} │", score.killed);
    println!(
        "   │ Survived                       │ {:7} │",
        score.survived
    );
    println!(
        "   │ Mutation Score                 │ {:6.1}% │",
        score.score * 100.0
    );
    println!("   └────────────────────────────────┴─────────┘");

    println!("\n   Per-Class Scores:");
    for class in MutationClass::all() {
        if let Some(class_score) = score.by_class.get(&class) {
            if class_score.total > 0 {
                println!(
                    "   - {} ({}): {}/{} killed ({:.0}%)",
                    class.id(),
                    class.description(),
                    class_score.killed,
                    class_score.total,
                    class_score.score * 100.0
                );
            }
        }
    }

    // 5. Complexity Analysis
    println!("\n5. Complexity Analysis (O(n) Verification)");
    println!("   ─────────────────────────────────────────────────────────");
    println!("   Reference: Goldsmith et al., ESEC/FSE 2007\n");

    // Simulate timing data for complexity analysis
    let timing_data = vec![
        (10, 100.0),
        (20, 205.0),
        (30, 298.0),
        (40, 410.0),
        (50, 495.0),
        (60, 615.0),
        (70, 698.0),
        (80, 812.0),
    ];

    let analyzer = ComplexityAnalyzer::new(timing_data.clone());
    let result = analyzer.analyze(Some(ComplexityClass::Linear));

    println!("   Input data points: {}", timing_data.len());
    println!("   Detected complexity: {:?}", result.detected_class);
    println!("   Confidence (R\u{00B2}): {:.4}", result.r_squared);
    println!(
        "   Violation detected: {}",
        if result.is_violation { "yes" } else { "no" }
    );

    // 6. Example YAML Playbook Structure
    println!("\n6. Example Playbook YAML Structure");
    println!("   ─────────────────────────────────────────────────────────");
    println!(
        r##"
   version: "1.0"
   name: "My Test Playbook"
   machine:
     id: "my_state_machine"
     initial: "start"
     states:
       start: {{ id: "start" }}
       end: {{ id: "end", final_state: true }}
     transitions:
       - id: "go"
         from: "start"
         to: "end"
         event: "proceed"
     forbidden:
       - from: "end"
         to: "start"
         reason: "No going back"
   playbook:
     setup:
       - type: navigate
         url: "http://example.com"
     steps:
       - name: "Step 1"
         transitions: ["go"]
         capture:
           - var: "result"
             from: "#output"
     teardown:
       - type: screenshot
         path: "./final.png"
   assertions:
     path:
       must_visit: ["start", "end"]
       must_not_visit: ["error"]
     output:
       - var: "result"
         not_empty: true
   "##
    );

    // 7. CLI Usage
    println!("7. CLI Usage");
    println!("   ─────────────────────────────────────────────────────────");
    println!(
        r#"
   # Validate a playbook
   probar playbook login.yaml --validate

   # Export state diagram
   probar playbook login.yaml --export svg --export-output diagram.svg

   # Run mutation testing
   probar playbook login.yaml --mutate

   # Run specific mutation classes
   probar playbook login.yaml --mutate --mutation-classes M1,M2,M3

   # JSON output for CI integration
   probar playbook login.yaml --format json

   # JUnit XML for test reporting
   probar playbook login.yaml --format junit
   "#
    );

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  ✓ Playbook Testing Example Complete                         ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
}
