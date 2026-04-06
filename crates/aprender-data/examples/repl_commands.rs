//! REPL Command Parser Demo
//!
//! Demonstrates programmatic parsing of REPL commands.
//! Run: cargo run --example `repl_commands` --features repl

#[cfg(feature = "repl")]
fn main() {
    use alimentar::repl::CommandParser;

    println!("=== REPL Command Parser Demo ===\n");

    // Data loading commands
    let commands = [
        "load data/iris.csv",
        "datasets",
        "use iris",
        "info",
        "head 5",
        "head",
        "schema",
    ];

    println!("--- Data Commands ---");
    for cmd in &commands {
        match CommandParser::parse(cmd) {
            Ok(parsed) => println!("  {cmd:<20} => {parsed:?}"),
            Err(e) => println!("  {cmd:<20} => Error: {e}"),
        }
    }

    // Quality commands
    println!("\n--- Quality Commands ---");
    let quality_cmds = [
        "quality check",
        "quality score",
        "quality score --suggest",
        "quality score --json --badge",
    ];
    for cmd in &quality_cmds {
        match CommandParser::parse(cmd) {
            Ok(parsed) => println!("  {cmd:<30} => {parsed:?}"),
            Err(e) => println!("  {cmd:<30} => Error: {e}"),
        }
    }

    // Export and conversion
    println!("\n--- Export/Convert Commands ---");
    let export_cmds = [
        "convert csv",
        "convert parquet",
        "convert json",
        "export quality",
        "export quality --json",
    ];
    for cmd in &export_cmds {
        match CommandParser::parse(cmd) {
            Ok(parsed) => println!("  {cmd:<25} => {parsed:?}"),
            Err(e) => println!("  {cmd:<25} => Error: {e}"),
        }
    }

    // Session commands
    println!("\n--- Session Commands ---");
    let session_cmds = [
        "history",
        "history --export",
        "help",
        "help quality",
        "?",
        "quit",
        "exit",
        "q",
    ];
    for cmd in &session_cmds {
        match CommandParser::parse(cmd) {
            Ok(parsed) => println!("  {cmd:<20} => {parsed:?}"),
            Err(e) => println!("  {cmd:<20} => Error: {e}"),
        }
    }

    // Error cases
    println!("\n--- Error Handling ---");
    let error_cmds = ["load", "quality", "drift", "unknown", ""];
    for cmd in &error_cmds {
        match CommandParser::parse(cmd) {
            Ok(parsed) => println!("  {cmd:<15} => {parsed:?}"),
            Err(e) => println!("  {cmd:<15} => Error: {e}"),
        }
    }

    println!("\n=== Demo Complete ===");
}

#[cfg(not(feature = "repl"))]
fn main() {
    eprintln!("Run with: cargo run --example repl_commands --features repl");
}
