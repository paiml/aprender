//! REPL Session Management Demo
//!
//! Demonstrates session state, history, and export functionality.
//! Run: `cargo run --example repl_session --features repl`

#[cfg(feature = "repl")]
fn main() {
    use alimentar::repl::ReplSession;

    println!("=== REPL Session Demo ===\n");

    // Create a new session
    let mut session = ReplSession::new();
    println!("Created new REPL session");
    println!("  Datasets loaded: {}", session.datasets().len());
    println!("  Active dataset: {:?}", session.active_name());
    println!("  History entries: {}", session.history().len());

    // Simulate command history
    println!("\n--- Building Command History ---");
    let commands = [
        "load data/sales.csv",
        "info",
        "head 10",
        "quality check",
        "quality score --suggest",
        "schema",
        "convert parquet",
    ];

    for cmd in &commands {
        session.add_history(cmd);
        println!("  Added: {cmd}");
    }

    // Show history
    println!("\n--- Session History ---");
    for (i, entry) in session.history().iter().enumerate() {
        println!("  [{}] {}", i + 1, entry);
    }

    // Export as script
    println!("\n--- Exported Script ---");
    let script = session.export_history();
    for line in script.lines().take(15) {
        println!("{line}");
    }

    // Column completion context
    println!("\n--- Completion Context ---");
    println!("  Column names: {:?}", session.column_names());

    println!("\n=== Demo Complete ===");
}

#[cfg(not(feature = "repl"))]
fn main() {
    eprintln!("Run with: cargo run --example repl_session --features repl");
}
