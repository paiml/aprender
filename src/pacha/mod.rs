//! Pacha Model Registry CLI Module
//!
//! Provides CLI commands for interacting with the Pacha model registry,
//! offering an "ollama-like" experience for model management.
//!
//! ## Commands
//!
//! - `pull` - Download a model from registry or HuggingFace
//! - `list` - List cached models
//! - `rm` - Remove a model from cache
//! - `show` - Show model information
//! - `run` - Run inference on a model (interactive)
//!
//! ## Examples
//!
//! ```bash
//! # Pull a model
//! batuta pacha pull llama3:8b
//!
//! # List cached models
//! batuta pacha list
//!
//! # Show model info
//! batuta pacha show llama3
//!
//! # Remove a model
//! batuta pacha rm llama3:8b
//! ```

// Submodules
mod commands;
pub mod crypto;
mod handlers;
mod helpers;
mod pull;
mod run;

#[cfg(test)]
mod tests;

// Re-exports
pub use commands::PachaCommand;

// ============================================================================
// PACHA-CLI-002: Command Handler
// ============================================================================

/// Execute a pacha command
pub fn cmd_pacha(command: PachaCommand) -> anyhow::Result<()> {
    match command {
        PachaCommand::Pull { model, force, quant } => {
            pull::cmd_pull(&model, force, quant.as_deref())
        }
        PachaCommand::List { verbose, format } => handlers::cmd_list(verbose, &format),
        PachaCommand::Rm { model, all, yes } => handlers::cmd_rm(&model, all, yes),
        PachaCommand::Show { model, full } => handlers::cmd_show(&model, full),
        PachaCommand::Search { query, limit } => handlers::cmd_search(&query, limit),
        PachaCommand::Aliases { pattern } => handlers::cmd_aliases(pattern.as_deref()),
        PachaCommand::Alias { name, target } => handlers::cmd_alias(&name, &target),
        PachaCommand::Stats => handlers::cmd_stats(),
        PachaCommand::Prune { days, dry_run } => handlers::cmd_prune(days, dry_run),
        PachaCommand::Pin { model } => handlers::cmd_pin(&model),
        PachaCommand::Unpin { model } => handlers::cmd_unpin(&model),
        PachaCommand::Run {
            model,
            system,
            modelfile,
            temperature,
            max_tokens,
            context,
            verbose,
        } => run::cmd_run(
            &model,
            system.as_deref(),
            modelfile.as_deref(),
            temperature,
            max_tokens,
            context,
            verbose,
        ),
        PachaCommand::Keygen { output, identity, force } => {
            crypto::cmd_keygen(output.as_deref(), identity.as_deref(), force)
        }
        PachaCommand::Sign { model, key, output, identity } => {
            crypto::cmd_sign(&model, key.as_deref(), output.as_deref(), identity.as_deref())
        }
        PachaCommand::Verify { model, signature, key } => {
            crypto::cmd_verify(&model, signature.as_deref(), key.as_deref())
        }
        PachaCommand::Encrypt { model, output, password_env } => {
            crypto::cmd_encrypt(&model, output.as_deref(), password_env.as_deref())
        }
        PachaCommand::Decrypt { file, output, password_env } => {
            crypto::cmd_decrypt(&file, output.as_deref(), password_env.as_deref())
        }
    }
}
