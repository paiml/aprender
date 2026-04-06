//! Agent command implementations.
//!
//! CLI handlers for `batuta agent run|chat|validate|sign|verify-sig`.
//! Feature-gated behind `agents` (via `cli/mod.rs`).

#[path = "agent_helpers.rs"]
mod agent_helpers;
#[path = "agent_pool_cmds.rs"]
mod agent_pool_cmds;
#[path = "agent_runtime_cmds.rs"]
mod agent_runtime_cmds;

use std::path::PathBuf;

use crate::ansi_colors::Colorize;

use agent_helpers::{
    load_manifest, print_manifest_summary, try_auto_pull, validate_model_file, validate_model_g2,
};

/// Public driver builder for `batuta code` (used by `cli::code`).
pub fn build_driver_pub(
    manifest: &batuta::agent::AgentManifest,
) -> anyhow::Result<Box<dyn batuta::agent::driver::LlmDriver>> {
    agent_helpers::build_driver(manifest)
}
// Re-exports for agent_tests.rs (used via `use super::*`)
#[cfg(test)]
use agent_helpers::{
    build_driver, build_guard, build_memory, build_tool_registry, detect_model_format,
    register_inference_tool, register_spawn_tool,
};
#[cfg(test)]
use agent_pool_cmds::{cmd_agent_pool, cmd_agent_status};
#[cfg(test)]
use agent_runtime_cmds::{cmd_agent_chat, cmd_agent_run};

/// Agent subcommands.
#[derive(Debug, Clone, clap::Subcommand)]
pub enum AgentCommand {
    /// Run an agent with a single prompt (non-interactive).
    Run {
        /// Path to agent manifest (TOML).
        #[arg(long)]
        manifest: PathBuf,

        /// Prompt to send to the agent.
        #[arg(long)]
        prompt: String,

        /// Override max iterations from manifest.
        #[arg(long)]
        max_iterations: Option<u32>,

        /// Run as a long-lived daemon (for forjar deployments).
        #[arg(long)]
        daemon: bool,

        /// Auto-download model via `apr pull` if not cached.
        #[arg(long)]
        auto_pull: bool,

        /// Enable real-time streaming output (token-by-token).
        #[arg(long)]
        stream: bool,
    },

    /// Start an interactive chat session with an agent.
    Chat {
        /// Path to agent manifest (TOML).
        #[arg(long)]
        manifest: PathBuf,

        /// Auto-download model via `apr pull` if not cached.
        #[arg(long)]
        auto_pull: bool,

        /// Enable real-time streaming output (token-by-token).
        #[arg(long)]
        stream: bool,
    },

    /// Validate an agent manifest without running it.
    ///
    /// Checks manifest syntax, consistency, and optionally validates
    /// the model file (BLAKE3 integrity, format detection, inference).
    Validate {
        /// Path to agent manifest (TOML).
        #[arg(long)]
        manifest: PathBuf,

        /// Validate model file (G0: integrity, G1: format).
        #[arg(long)]
        check_model: bool,

        /// Run inference sanity check (G2: probe prompt, entropy).
        /// Requires --check-model. Uses the configured driver.
        #[arg(long)]
        check_inference: bool,
    },

    /// Sign an agent manifest with Ed25519 (via pacha).
    Sign {
        /// Path to agent manifest (TOML).
        #[arg(long)]
        manifest: PathBuf,

        /// Signer identity label (optional).
        #[arg(long)]
        signer: Option<String>,

        /// Path to write the signature sidecar file.
        /// Defaults to `<manifest>.sig`.
        #[arg(long)]
        output: Option<PathBuf>,
    },

    /// Verify an agent manifest signature.
    VerifySig {
        /// Path to agent manifest (TOML).
        #[arg(long)]
        manifest: PathBuf,

        /// Path to the signature sidecar file.
        /// Defaults to `<manifest>.sig`.
        #[arg(long)]
        signature: Option<PathBuf>,

        /// Path to the public key file (hex-encoded).
        #[arg(long)]
        pubkey: PathBuf,
    },

    /// Verify contract invariant bindings against the test suite.
    Contracts,

    /// Show agent manifest status and configuration.
    Status {
        /// Path to agent manifest (TOML).
        #[arg(long)]
        manifest: PathBuf,
    },

    /// Fan-out multiple agents and collect results (multi-agent pool).
    Pool {
        /// Paths to agent manifests (one per agent).
        #[arg(long, required = true, num_args = 1..)]
        manifest: Vec<PathBuf>,

        /// Prompt to send to each agent.
        #[arg(long)]
        prompt: String,

        /// Maximum concurrent agents (default: number of manifests).
        #[arg(long)]
        concurrency: Option<usize>,
    },
}

/// Dispatch an agent subcommand.
pub fn cmd_agent(command: AgentCommand) -> anyhow::Result<()> {
    match command {
        AgentCommand::Run { manifest, prompt, max_iterations, daemon, auto_pull, stream } => {
            agent_runtime_cmds::cmd_agent_run(
                &manifest,
                &prompt,
                max_iterations,
                daemon,
                auto_pull,
                stream,
            )
        }
        AgentCommand::Chat { manifest, auto_pull, stream } => {
            agent_runtime_cmds::cmd_agent_chat(&manifest, auto_pull, stream)
        }
        AgentCommand::Validate { manifest, check_model, check_inference } => {
            cmd_agent_validate(&manifest, check_model, check_inference)
        }
        AgentCommand::Sign { manifest, signer, output } => {
            cmd_agent_sign(&manifest, signer.as_deref(), output)
        }
        AgentCommand::VerifySig { manifest, signature, pubkey } => {
            cmd_agent_verify_sig(&manifest, signature, &pubkey)
        }
        AgentCommand::Contracts => cmd_agent_contracts(),
        AgentCommand::Status { manifest } => agent_pool_cmds::cmd_agent_status(&manifest),
        AgentCommand::Pool { manifest, prompt, concurrency } => {
            agent_pool_cmds::cmd_agent_pool(&manifest, &prompt, concurrency)
        }
    }
}

/// Validate an agent manifest (and optionally the model file).
fn cmd_agent_validate(
    manifest_path: &PathBuf,
    check_model: bool,
    check_inference: bool,
) -> anyhow::Result<()> {
    let manifest = load_manifest(manifest_path)?;

    match manifest.validate() {
        Ok(()) => {
            println!("{} Manifest is valid: {}", "✓".green(), manifest_path.display());
            print_manifest_summary(&manifest);
        }
        Err(errors) => {
            println!("{} Manifest validation failed:", "✗".bright_red());
            for err in &errors {
                println!("  {} {}", "•".bright_red(), err);
            }
            anyhow::bail!("{} validation error(s) in {}", errors.len(), manifest_path.display());
        }
    }

    // Auto-pull check
    if let Some(repo) = manifest.model.needs_pull() {
        println!();
        println!("{} Model needs download: {}", "⚠".bright_yellow(), repo,);
        if let Some(path) = manifest.model.resolve_model_path() {
            println!("  Expected at: {}", path.display());
        }
        println!(
            "  Run: {} {} {}",
            "apr pull".cyan(),
            repo,
            manifest.model.model_quantization.as_deref().unwrap_or("q4_k_m"),
        );
    }

    if check_model {
        validate_model_file(&manifest)?;
    }

    if check_inference {
        validate_model_g2(&manifest)?;
    }

    Ok(())
}

/// Sign an agent manifest with Ed25519 (Jidoka: tamper detection).
fn cmd_agent_sign(
    manifest_path: &PathBuf,
    signer: Option<&str>,
    output: Option<PathBuf>,
) -> anyhow::Result<()> {
    let content = std::fs::read_to_string(manifest_path)
        .map_err(|e| anyhow::anyhow!("Cannot read manifest {}: {e}", manifest_path.display()))?;

    // Generate a new signing key (in production, load from secure store)
    let signing_key = pacha::signing::SigningKey::generate();
    let verifying_key = signing_key.verifying_key();

    let sig = batuta::agent::signing::sign_manifest(&content, &signing_key, signer);

    let sig_path = output.unwrap_or_else(|| {
        let mut p = manifest_path.clone();
        let ext = p
            .extension()
            .map(|e| format!("{}.sig", e.to_string_lossy()))
            .unwrap_or_else(|| "sig".into());
        p.set_extension(ext);
        p
    });

    let sig_toml = batuta::agent::signing::signature_to_toml(&sig);
    std::fs::write(&sig_path, &sig_toml)
        .map_err(|e| anyhow::anyhow!("Cannot write signature to {}: {e}", sig_path.display()))?;

    // Write public key alongside
    let pk_path = sig_path.with_extension("pub");
    std::fs::write(&pk_path, verifying_key.to_hex())
        .map_err(|e| anyhow::anyhow!("Cannot write public key to {}: {e}", pk_path.display()))?;

    println!("{} Manifest signed: {}", "✓".green(), manifest_path.display());
    println!("  Signature: {}", sig_path.display());
    println!("  Public key: {}", pk_path.display());
    println!("  Hash: {}", &sig.content_hash[..16]);
    if let Some(ref s) = sig.signer {
        println!("  Signer: {s}");
    }

    Ok(())
}

/// Verify an agent manifest signature (Jidoka: stop on tampered).
fn cmd_agent_verify_sig(
    manifest_path: &PathBuf,
    signature_path: Option<PathBuf>,
    pubkey_path: &PathBuf,
) -> anyhow::Result<()> {
    let content = std::fs::read_to_string(manifest_path)
        .map_err(|e| anyhow::anyhow!("Cannot read manifest {}: {e}", manifest_path.display()))?;

    let sig_path = signature_path.unwrap_or_else(|| {
        let mut p = manifest_path.clone();
        let ext = p
            .extension()
            .map(|e| format!("{}.sig", e.to_string_lossy()))
            .unwrap_or_else(|| "sig".into());
        p.set_extension(ext);
        p
    });

    let sig_content = std::fs::read_to_string(&sig_path)
        .map_err(|e| anyhow::anyhow!("Cannot read signature {}: {e}", sig_path.display()))?;

    let sig = batuta::agent::signing::signature_from_toml(&sig_content)
        .map_err(|e| anyhow::anyhow!("Invalid signature: {e}"))?;

    let pk_hex = std::fs::read_to_string(pubkey_path)
        .map_err(|e| anyhow::anyhow!("Cannot read public key {}: {e}", pubkey_path.display()))?;

    let vk = pacha::signing::VerifyingKey::from_hex(pk_hex.trim())
        .map_err(|e| anyhow::anyhow!("Invalid public key: {e}"))?;

    match batuta::agent::signing::verify_manifest(&content, &sig, &vk) {
        Ok(()) => {
            println!("{} Signature valid: {}", "✓".green(), manifest_path.display());
            if let Some(ref signer) = sig.signer {
                println!("  Signer: {signer}");
            }
            println!("  Hash: {}", &sig.content_hash[..16]);
            Ok(())
        }
        Err(e) => {
            println!("{} Signature verification FAILED: {e}", "✗".bright_red());
            anyhow::bail!("manifest signature verification failed: {e}");
        }
    }
}

/// Verify contract invariant bindings against the test suite.
fn cmd_agent_contracts() -> anyhow::Result<()> {
    let yaml = include_str!("../../contracts/agent-loop-v1.yaml");
    let contract = batuta::agent::contracts::parse_contract(yaml)
        .map_err(|e| anyhow::anyhow!("contract parse: {e}"))?;

    println!(
        "{} Contract: {} v{}",
        "📋".bright_blue(),
        contract.contract.name.cyan(),
        contract.contract.version,
    );
    println!("  {}", contract.contract.description.dimmed());
    println!();

    println!("{} Invariants ({})", "•".bright_blue(), contract.invariants.len(),);
    for inv in &contract.invariants {
        println!("  {} {} — {}", inv.id.bright_yellow(), inv.name, inv.description.dimmed(),);
        println!("    Test: {}", inv.test_binding.cyan());
    }
    println!();

    println!("{} Verification targets:", "•".bright_blue(),);
    println!(
        "  Coverage: {}%  Mutation: {}%",
        contract.verification.coverage_target, contract.verification.mutation_target,
    );
    println!(
        "  Complexity: cyclomatic {} / cognitive {}",
        contract.verification.complexity_max_cyclomatic,
        contract.verification.complexity_max_cognitive,
    );
    println!();

    println!(
        "{} Run {} to check bindings.",
        "ℹ".bright_blue(),
        "cargo test --features agents -- test_all_contract_bindings_exist".cyan(),
    );

    Ok(())
}

#[cfg(test)]
#[path = "agent_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "agent_tests_extended.rs"]
mod tests_extended;
