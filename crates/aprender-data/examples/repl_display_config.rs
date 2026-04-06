//! REPL Display Configuration Demo
//!
//! Demonstrates configurable output formatting.
//! Run: cargo run --example `repl_display_config` --features repl

#[cfg(feature = "repl")]
fn main() {
    use alimentar::repl::DisplayConfig;

    println!("=== REPL Display Configuration Demo ===\n");

    // Default configuration
    println!("--- Default Configuration ---");
    let default_config = DisplayConfig::default();
    println!("  max_rows:         {}", default_config.max_rows);
    println!("  max_column_width: {}", default_config.max_column_width);
    println!("  color_output:     {}", default_config.color_output);

    // Builder pattern customization
    println!("\n--- Custom Configurations ---");

    let compact = DisplayConfig::default()
        .with_max_rows(5)
        .with_max_column_width(30);
    println!(
        "  Compact: rows={}, width={}",
        compact.max_rows, compact.max_column_width
    );

    let detailed = DisplayConfig::default()
        .with_max_rows(100)
        .with_max_column_width(200);
    println!(
        "  Detailed: rows={}, width={}",
        detailed.max_rows, detailed.max_column_width
    );

    let no_color = DisplayConfig::default().with_color(false);
    println!("  No-color: color={}", no_color.color_output);

    let piped = DisplayConfig::default()
        .with_max_rows(1000)
        .with_max_column_width(0) // unlimited
        .with_color(false);
    println!(
        "  Piped: rows={}, width={}, color={}",
        piped.max_rows, piped.max_column_width, piped.color_output
    );

    // Use cases
    println!("\n--- Use Case Examples ---");
    println!("  Interactive terminal: default (10 rows, 50 width, color)");
    println!("  Quick preview:        compact (5 rows, 30 width)");
    println!("  Full analysis:        detailed (100 rows, 200 width)");
    println!("  CI/CD pipeline:       no-color (disable ANSI codes)");
    println!("  Script export:        piped (unlimited, no color)");

    println!("\n=== Demo Complete ===");
}

#[cfg(not(feature = "repl"))]
fn main() {
    eprintln!("Run with: cargo run --example repl_display_config --features repl");
}
