//! Presentar YAML Support Demo
//!
//! Demonstrates probar's native support for testing presentar TUI configurations
//! with 100-point falsification protocol (F001-F100).
//!
//! Run with: cargo run --example presentar_demo -p jugar-probar

use jugar_probar::{
    generate_falsification_playbook, parse_and_validate_presentar, validate_presentar_config,
    PresentarConfig, TerminalAssertion, TerminalSnapshot, FALSIFICATION_COUNT, SCHEMA_VERSION,
};

fn main() {
    println!("=== Presentar YAML Support Demo ===\n");

    // 1. Schema version and falsification count
    println!("1. Schema Information:");
    println!("   Schema Version: {}", SCHEMA_VERSION);
    println!(
        "   Falsification Checks: {} (F001-F100)",
        FALSIFICATION_COUNT
    );
    println!();

    // 2. Default configuration
    println!("2. Default Configuration:");
    let config = PresentarConfig::default();
    println!("   Refresh Rate: {}ms", config.refresh_ms);
    println!("   Grid Size: {}", config.layout.grid_size);
    println!("   Snap to Grid: {}", config.layout.snap_to_grid);
    println!("   Top Height: {:.0}%", config.layout.top_height * 100.0);
    println!(
        "   Bottom Height: {:.0}%",
        config.layout.bottom_height * 100.0
    );
    println!();

    // 3. YAML parsing and validation
    println!("3. YAML Parsing and Validation:");
    let yaml = r##"
refresh_ms: 500
layout:
  snap_to_grid: true
  grid_size: 8
  top_height: 0.4
  bottom_height: 0.6
panels:
  cpu:
    enabled: true
    sparkline_history: 120
  memory:
    enabled: true
  process:
    enabled: true
    columns: [pid, user, cpu, mem, cmd]
keybindings:
  quit: q
  help: '?'
  filter: /
theme:
  panel_colors:
    cpu: "#64C8FF"
    memory: "#B478FF"
"##;

    match parse_and_validate_presentar(yaml) {
        Ok((parsed_config, result)) => {
            println!("   Parsed successfully!");
            println!("   Refresh Rate: {}ms", parsed_config.refresh_ms);
            println!("   Grid Size: {}", parsed_config.layout.grid_size);
            if result.is_ok() {
                println!("   Validation: PASSED");
            } else {
                println!("   Validation: FAILED");
                for err in &result.errors {
                    println!("     - {}", err);
                }
            }
            if !result.warnings.is_empty() {
                println!("   Warnings:");
                for warn in &result.warnings {
                    println!("     - {}", warn);
                }
            }
        }
        Err(e) => {
            println!("   Parse Error: {}", e);
        }
    }
    println!();

    // 4. Validation errors demo
    println!("4. Validation Error Examples:");
    let bad_config = PresentarConfig {
        refresh_ms: 5, // Too low
        ..PresentarConfig::default()
    };
    let result = validate_presentar_config(&bad_config);
    if result.is_err() {
        for err in &result.errors {
            println!("   - {}", err);
        }
    }
    println!();

    // 5. Terminal snapshot testing
    println!("5. Terminal Snapshot Testing:");
    let snapshot = TerminalSnapshot::from_string(
        "CPU  45% ████████░░░░░░░░ 4 cores\n\
         MEM  60% ██████████░░░░░░ 8GB/16GB\n\
         DISK 75% ████████████░░░░ 500GB/1TB",
        80,
        24,
    );
    println!("   Snapshot size: {:?}", snapshot.dimensions());
    println!("   Contains 'CPU': {}", snapshot.contains("CPU"));
    println!("   Contains 'GPU': {}", snapshot.contains("GPU"));

    // Assertion checks
    let assertions = [
        TerminalAssertion::Contains("CPU".into()),
        TerminalAssertion::Contains("MEM".into()),
        TerminalAssertion::NotContains("GPU".into()),
        TerminalAssertion::CharAt {
            x: 0,
            y: 0,
            expected: 'C',
        },
    ];

    println!("   Assertions:");
    for assertion in &assertions {
        match assertion.check(&snapshot) {
            Ok(()) => println!("     - {:?}: PASS", assertion),
            Err(e) => println!("     - {:?}: FAIL - {}", assertion, e),
        }
    }
    println!();

    // 6. Falsification protocol
    println!("6. Falsification Protocol (F001-F100):");
    let playbook = generate_falsification_playbook(&config);
    let mutations = playbook.falsification.as_ref().map(|f| &f.mutations);

    if let Some(mutations) = mutations {
        println!("   Generated {} mutations", mutations.len());
        println!("   Sample mutations:");
        for mutation in mutations.iter().take(5) {
            println!("     {} - {}", mutation.id, mutation.description);
        }
        println!("     ...");
    }
    println!();

    // 7. Category breakdown
    println!("7. Falsification Categories:");
    let categories = [
        ("F001-F014", "Panel Existence", 14),
        ("F015-F028", "Panel Content", 14),
        ("F029-F042", "Color Consistency", 14),
        ("F043-F056", "Layout Consistency", 14),
        ("F057-F070", "Keybinding Consistency", 14),
        ("F071-F084", "Data Binding", 14),
        ("F085-F092", "Performance", 8),
        ("F093-F100", "Accessibility", 8),
    ];
    for (range, name, count) in categories {
        println!("   {} {}: {} checks", range, name, count);
    }
    println!();

    println!("=== Demo Complete ===");
}
