//! entrenar-shell CLI entry point.

use clap::Parser;
use entrenar_shell::{start_with_state, SessionState};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "entrenar-shell")]
#[command(about = "Interactive REPL for HuggingFace model exploration and distillation")]
#[command(version)]
struct Cli {
    /// Load session from file
    #[arg(short, long)]
    session: Option<PathBuf>,

    /// Execute a single command and exit
    #[arg(short, long)]
    command: Option<String>,
}

fn main() {
    let cli = Cli::parse();

    // Load session if provided
    let state = if let Some(ref path) = cli.session {
        match SessionState::load(path) {
            Ok(s) => {
                println!("Loaded session from {}", path.display());
                s
            }
            Err(e) => {
                eprintln!("Failed to load session: {e}");
                SessionState::new()
            }
        }
    } else {
        SessionState::new()
    };

    // Execute single command if provided
    if let Some(cmd) = cli.command {
        let mut state = state;
        match entrenar_shell::commands::parse(&cmd) {
            Ok(parsed) => match entrenar_shell::commands::execute(&parsed, &mut state) {
                Ok(output) => {
                    if !output.is_empty() {
                        println!("{output}");
                    }
                }
                Err(e) => {
                    eprintln!("Error: {e}");
                    std::process::exit(1);
                }
            },
            Err(e) => {
                eprintln!("Parse error: {e}");
                std::process::exit(1);
            }
        }
        return;
    }

    // Start interactive REPL
    if let Err(e) = start_with_state(state) {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}
