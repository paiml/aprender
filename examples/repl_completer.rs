//! REPL Tab Completion Demo
//!
//! Demonstrates the schema-aware autocomplete system.
//! Run: `cargo run --example repl_completer --features repl`

#[cfg(feature = "repl")]
fn main() {
    use alimentar::repl::{CommandParser, ReplSession, SchemaAwareCompleter};

    println!("=== REPL Tab Completion Demo ===\n");

    let session = ReplSession::new();
    let completer = SchemaAwareCompleter::new(&session);

    // Command completions
    println!("--- Command Completions ---");
    let prefixes = ["", "q", "he", "qual", "con"];
    for prefix in &prefixes {
        let completions = completer.complete(prefix);
        println!("  '{prefix:<6}' => {completions:?}");
    }

    // Subcommand completions
    println!("\n--- Subcommand Completions ---");
    let sub_prefixes = ["quality ", "drift ", "help ", "convert "];
    for prefix in &sub_prefixes {
        let completions = completer.complete(prefix);
        println!("  '{prefix}' => {completions:?}");
    }

    // Available commands list
    println!("\n--- All Available Commands ---");
    let all_commands = CommandParser::command_names();
    for chunk in all_commands.chunks(5) {
        println!("  {chunk:?}");
    }

    // Subcommand details
    println!("\n--- Subcommand Structure ---");
    for cmd in &["quality", "drift", "export"] {
        let subs = CommandParser::subcommands(cmd);
        let flags = CommandParser::flags(cmd, subs.first().copied());
        println!("  {cmd} => subs: {subs:?}, flags: {flags:?}");
    }

    // Flag completions
    println!("\n--- Flag Details ---");
    let flag_examples = [
        ("quality", Some("score")),
        ("export", None),
        ("validate", None),
        ("history", None),
    ];
    for (cmd, sub) in &flag_examples {
        let flags = CommandParser::flags(cmd, *sub);
        println!("  {cmd} {sub:?} => {flags:?}");
    }

    println!("\n=== Demo Complete ===");
}

#[cfg(not(feature = "repl"))]
fn main() {
    eprintln!("Run with: cargo run --example repl_completer --features repl");
}
