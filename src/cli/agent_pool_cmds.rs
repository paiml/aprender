//! Agent pool and status command handlers.
//!
//! Extracted from `agent_runtime_cmds.rs` for QA-002 compliance (≤500 lines).

use std::path::PathBuf;
use std::sync::Arc;

use crate::ansi_colors::Colorize;

use super::agent_helpers::{build_driver, build_memory, load_manifest, print_manifest_summary};

/// Fan-out multiple agents and collect results.
pub(super) fn cmd_agent_pool(
    manifest_paths: &[PathBuf],
    prompt: &str,
    concurrency: Option<usize>,
) -> anyhow::Result<()> {
    use batuta::agent::pool::{AgentPool, SpawnConfig};

    let manifests: Vec<batuta::agent::AgentManifest> =
        manifest_paths.iter().map(load_manifest).collect::<Result<_, _>>()?;

    let max_concurrent = concurrency.unwrap_or(manifests.len());

    println!("{} Multi-Agent Pool", "🔀".bright_cyan().bold());
    println!("{}", "═".repeat(60).dimmed());
    println!("  Agents: {}  Concurrency: {}", manifests.len(), max_concurrent,);
    println!("  Prompt: {}", prompt.bright_yellow());
    println!();

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| anyhow::anyhow!("tokio runtime: {e}"))?;

    let driver =
        build_driver(manifests.first().unwrap_or(&batuta::agent::AgentManifest::default()))?;
    let driver: Arc<dyn batuta::agent::driver::LlmDriver> = Arc::from(driver);
    let memory: Arc<dyn batuta::agent::memory::MemorySubstrate> = Arc::from(build_memory());

    let configs: Vec<SpawnConfig> = manifests
        .iter()
        .map(|m| SpawnConfig { manifest: m.clone(), query: prompt.to_string() })
        .collect();

    let agent_count = configs.len();

    rt.block_on(async {
        let mut pool = AgentPool::new(driver, max_concurrent).with_memory(memory);

        println!("{} Spawning {} agents...", "▶".bright_green(), agent_count,);

        let ids = pool.fan_out(configs).map_err(|e| anyhow::anyhow!("pool fan-out: {e}"))?;
        for id in &ids {
            println!("  {} Agent {id} spawned", "•".bright_blue(),);
        }
        println!();

        println!("{} Waiting for results...", "⏳".bright_blue(),);
        let results = pool.join_all().await;

        println!();
        println!("{} Results ({}/{})", "✓".green(), results.len(), ids.len(),);
        println!("{}", "─".repeat(60).dimmed());

        for (id, result) in &results {
            match result {
                Ok(r) => {
                    println!("  {} Agent {id}: {}", "✓".green(), r.text,);
                    println!(
                        "    {}",
                        format!(
                            "[iter={}, tools={}, tokens={}/{}]",
                            r.iterations, r.tool_calls, r.usage.input_tokens, r.usage.output_tokens,
                        )
                        .dimmed()
                    );
                }
                Err(e) => {
                    println!("  {} Agent {id}: {e}", "✗".bright_red(),);
                }
            }
        }

        Ok::<(), anyhow::Error>(())
    })?;

    Ok(())
}

/// Display agent manifest status and configuration.
pub(super) fn cmd_agent_status(manifest_path: &PathBuf) -> anyhow::Result<()> {
    let manifest = load_manifest(manifest_path)?;
    if let Err(errors) = manifest.validate() {
        println!("{} Manifest has validation errors:", "⚠".bright_yellow());
        for e in &errors {
            println!("  {} {e}", "✗".red());
        }
        println!();
    }
    print_manifest_summary(&manifest);
    println!();

    let r = &manifest.resources;
    println!("{}", "Resource Quotas".bright_cyan().bold());
    println!("{}", "─".repeat(40).dimmed());
    println!("  Max iterations:  {}", r.max_iterations);
    println!("  Max tool calls:  {}", r.max_tool_calls);
    let budget = if r.max_cost_usd > 0.0 {
        format!("${:.4}", r.max_cost_usd)
    } else {
        "unlimited (sovereign)".to_string()
    };
    println!("  Cost budget:     {budget}");
    println!();

    let m = &manifest.model;
    println!("{}", "Model Configuration".bright_cyan().bold());
    println!("{}", "─".repeat(40).dimmed());
    println!("  Max tokens:      {}", m.max_tokens);
    println!("  Temperature:     {}", m.temperature);
    match m.context_window {
        Some(ref ctx) => println!("  Context window:  {ctx}"),
        None => println!("  Context window:  auto-detect"),
    }
    if let Some(ref repo) = m.model_repo {
        let quant = m.model_quantization.as_deref().unwrap_or("q4_k_m");
        println!("  Model repo:      {repo}");
        println!("  Quantization:    {quant}");
        if let Some(resolved) = m.resolve_model_path() {
            let exists = if resolved.exists() { "found" } else { "not found" };
            println!("  Resolved path:   {} ({exists})", resolved.display());
        }
    }
    if let Some(ref remote) = m.remote_model {
        println!("  Remote model:    {remote}");
    }
    println!();

    println!("{}", "Capabilities".bright_cyan().bold());
    println!("{}", "─".repeat(40).dimmed());
    if manifest.capabilities.is_empty() {
        println!("  (none — deny-by-default)");
    }
    for cap in &manifest.capabilities {
        println!("  {} {cap:?}", "•".bright_blue());
    }
    println!();
    println!("{} {}", "✓".green(), "Agent manifest loaded successfully.".green());
    Ok(())
}
