//! TUI Dataset Viewer Demo
//!
//! Demonstrates the interactive TUI viewer for exploring datasets.
//! Run: `cargo run --example tui_viewer --features cli`
//!
//! This example shows:
//! - Loading datasets from various formats
//! - Creating a `DatasetAdapter` for TUI rendering
//! - Using the `DatasetViewer` for navigation
//! - Search functionality

use std::process::Command;

fn main() {
    println!("=== TUI Dataset Viewer Demo ===\n");

    // Show the view command help
    println!("--- View Command Help ---");
    let help_output = Command::new("cargo")
        .args([
            "run",
            "--quiet",
            "--bin",
            "alimentar",
            "--features",
            "cli",
            "--",
            "view",
            "--help",
        ])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output();

    if let Ok(output) = help_output {
        let stdout = String::from_utf8_lossy(&output.stdout);
        for line in stdout.lines() {
            println!("  {line}");
        }
    }

    // Basic usage patterns
    println!("\n--- Basic Usage ---");
    println!("  alimentar view data.parquet          # View Parquet file");
    println!("  alimentar view data.csv              # View CSV file");
    println!("  alimentar view data.json             # View JSON file");
    println!("  alimentar view data.arrow            # View Arrow IPC file");

    // Search functionality
    println!("\n--- Search Functionality ---");
    println!("  alimentar view data.parquet --search \"error\"");
    println!("  # Opens viewer and jumps to first row containing 'error'");

    // Keyboard controls
    println!("\n--- Keyboard Controls ---");
    println!("  Navigation:");
    println!("    ↑/↓ or j/k     Scroll up/down one row");
    println!("    PgUp/PgDn      Scroll up/down one page");
    println!("    Space          Page down");
    println!("    Home/End       Jump to start/end");
    println!("    g/G            Jump to start/end (vim-style)");
    println!();
    println!("  Search:");
    println!("    /              Open search prompt");
    println!("    Enter          Execute search");
    println!("    Esc            Cancel search");
    println!();
    println!("  Exit:");
    println!("    q or Esc       Quit viewer");
    println!("    Ctrl+C         Force quit");

    // Integration with other commands
    println!("\n--- Pipeline Integration ---");
    println!("  # Quick inspection workflow");
    println!("  alimentar info data.parquet     # Check schema first");
    println!("  alimentar head data.parquet     # Preview first rows");
    println!("  alimentar view data.parquet     # Interactive exploration");
    println!();
    println!("  # Quality check then view");
    println!("  alimentar quality check data.csv && alimentar view data.csv");

    // Programmatic usage
    println!("\n--- Programmatic Usage (Library) ---");
    println!("  use alimentar::tui::{{DatasetAdapter, DatasetViewer}};");
    println!("  use alimentar::ArrowDataset;");
    println!();
    println!("  let dataset = ArrowDataset::from_parquet(\"data.parquet\")?;");
    println!("  let adapter = DatasetAdapter::from_dataset(&dataset)?;");
    println!("  let mut viewer = DatasetViewer::new(adapter);");
    println!();
    println!("  // Navigate programmatically");
    println!("  viewer.scroll_down();");
    println!("  viewer.page_down();");
    println!("  viewer.search(\"query\");");
    println!();
    println!("  // Render to strings for custom output");
    println!("  for line in viewer.render_lines() {{");
    println!("      println!(\"{{}}\", line);");
    println!("  }}");

    // Mode information
    println!("\n--- Adapter Modes ---");
    println!("  InMemory Mode:");
    println!("    - Used for datasets < 100,000 rows");
    println!("    - All data loaded upfront");
    println!("    - Fast random access");
    println!();
    println!("  Streaming Mode:");
    println!("    - Used for datasets >= 100,000 rows");
    println!("    - Lazy batch loading");
    println!("    - Memory efficient for large files");

    println!("\n=== Demo Complete ===");
    println!(
        "\nTry it: cargo run --bin alimentar --features cli -- view test_fixtures/data.parquet"
    );
}
