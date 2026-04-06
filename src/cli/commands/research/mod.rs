//! Research command implementation

mod bundle;
mod cite;
mod deposit;
mod export;
mod init;
mod preregister;
mod verify;

use crate::cli::LogLevel;
use crate::config::{ResearchArgs, ResearchCommand};

pub fn run_research(args: ResearchArgs, level: LogLevel) -> Result<(), String> {
    match args.command {
        ResearchCommand::Init(init_args) => init::run_research_init(init_args, level),
        ResearchCommand::Preregister(prereg_args) => {
            preregister::run_research_preregister(prereg_args, level)
        }
        ResearchCommand::Cite(cite_args) => cite::run_research_cite(cite_args, level),
        ResearchCommand::Export(export_args) => export::run_research_export(export_args, level),
        ResearchCommand::Deposit(deposit_args) => {
            deposit::run_research_deposit(deposit_args, level)
        }
        ResearchCommand::Bundle(bundle_args) => bundle::run_research_bundle(bundle_args, level),
        ResearchCommand::Verify(verify_args) => verify::run_research_verify(verify_args, level),
    }
}
