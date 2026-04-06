//! CLI command implementations

mod audit;
mod bench;
mod completion;
mod experiments;
mod finetune;
mod info;
mod init;
mod inspect;
mod merge;
mod monitor;
mod publish;
mod quantize;
mod research;
mod train;
mod validate;

#[cfg(test)]
mod tests;

use crate::cli::LogLevel;
use crate::config::{Cli, Command};

/// Execute a CLI command based on the parsed arguments
pub fn run_command(cli: Cli) -> Result<(), String> {
    // Configure output based on verbose/quiet flags
    let log_level = if cli.quiet {
        LogLevel::Quiet
    } else if cli.verbose {
        LogLevel::Verbose
    } else {
        LogLevel::Normal
    };

    match cli.command {
        Command::Train(args) => train::run_train(args, log_level),
        Command::Validate(args) => validate::run_validate(args, log_level),
        Command::Info(args) => info::run_info(args, log_level),
        Command::Init(args) => init::run_init(args, log_level),
        Command::Quantize(args) => quantize::run_quantize(args, log_level),
        Command::Merge(args) => merge::run_merge(args, log_level),
        Command::Research(args) => research::run_research(args, log_level),
        Command::Completion(args) => completion::run_completion(args, log_level),
        Command::Bench(args) => bench::run_bench(args, log_level),
        Command::Inspect(args) => inspect::run_inspect(args, log_level),
        Command::Audit(args) => audit::run_audit(args, log_level),
        Command::Monitor(args) => monitor::run_monitor(args, log_level),
        Command::Publish(args) => publish::run_publish(args, log_level),
        Command::Finetune(args) => finetune::run_finetune(args, log_level),
        Command::Experiments(args) => experiments::run_experiments(args, log_level),
    }
}
