//! Completion command implementation

use crate::cli::logging::log;
use crate::cli::LogLevel;
use crate::config::{Cli, CompletionArgs, ShellType};
use clap::CommandFactory;
use clap_complete::{generate, Shell};

pub fn run_completion(args: CompletionArgs, level: LogLevel) -> Result<(), String> {
    log(level, LogLevel::Verbose, &format!("Generating completions for: {}", args.shell));

    let mut cmd = Cli::command();
    let shell = match args.shell {
        ShellType::Bash => Shell::Bash,
        ShellType::Zsh => Shell::Zsh,
        ShellType::Fish => Shell::Fish,
        ShellType::PowerShell => Shell::PowerShell,
    };

    generate(shell, &mut cmd, "entrenar", &mut std::io::stdout());
    Ok(())
}
