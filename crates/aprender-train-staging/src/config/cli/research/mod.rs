//! Research command types for academic workflows

mod bundle;
mod cite;
mod deposit;
mod export;
mod init;
mod preregister;
mod verify;

#[cfg(test)]
mod tests;

use clap::{Parser, Subcommand};

// Re-export all public types
pub use bundle::BundleArgs;
pub use cite::CiteArgs;
pub use deposit::DepositArgs;
pub use export::ExportArgs;
pub use init::ResearchInitArgs;
pub use preregister::PreregisterArgs;
pub use verify::VerifyArgs;

/// Arguments for the research command
#[derive(Parser, Debug, Clone, PartialEq)]
pub struct ResearchArgs {
    /// Research subcommand to execute
    #[command(subcommand)]
    pub command: ResearchCommand,
}

/// Research subcommands
#[derive(Subcommand, Debug, Clone, PartialEq)]
pub enum ResearchCommand {
    /// Initialize a new research artifact
    Init(ResearchInitArgs),

    /// Create a pre-registration with cryptographic commitment
    Preregister(PreregisterArgs),

    /// Generate citations in various formats
    Cite(CiteArgs),

    /// Export artifacts to various formats
    Export(ExportArgs),

    /// Deposit to academic archives
    Deposit(DepositArgs),

    /// Bundle artifacts into RO-Crate package
    Bundle(BundleArgs),

    /// Verify pre-registration commitments or signatures
    Verify(VerifyArgs),
}
