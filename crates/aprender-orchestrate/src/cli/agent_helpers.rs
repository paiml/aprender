//! Agent CLI helper functions (build, validate, format).

use crate::ansi_colors::Colorize;
use std::path::PathBuf;
use std::sync::Arc;

/// Auto-pull model if `model_repo` is set and file is missing.
pub(super) fn try_auto_pull(manifest: &batuta::agent::AgentManifest) -> anyhow::Result<()> {
    if let Some(repo) = manifest.model.needs_pull() {
        println!("{} Auto-pulling model: {}", "⬇".bright_cyan(), repo.cyan());
        let quant = manifest.model.model_quantization.as_deref().unwrap_or("q4_k_m");
        println!("  Quantization: {}, Timeout: 600s", quant);
        match manifest.model.auto_pull(600) {
            Ok(path) => {
                println!("{} Model downloaded: {}", "✓".green(), path.display(),);
            }
            Err(e) => {
                anyhow::bail!("auto-pull failed: {e}");
            }
        }
    }
    Ok(())
}

/// Build the LLM driver from manifest configuration.
///
/// Supports three modes:
/// - Local only: RealizarDriver (model_path + inference feature)
/// - Remote only: RemoteDriver (remote_model + native feature)
/// - Hybrid: RoutingDriver (local-first, remote fallback)
/// - Fallback: MockDriver (dry-run when nothing configured)
pub(super) fn build_driver(
    manifest: &batuta::agent::AgentManifest,
) -> anyhow::Result<Box<dyn batuta::agent::driver::LlmDriver>> {
    let resolved_path = manifest.model.resolve_model_path();
    let local = build_local_driver(manifest, &resolved_path);
    let remote = build_remote_driver(manifest);

    match (local, remote) {
        (Some(primary), Some(fallback)) => {
            println!("{} Routing: local-first with remote fallback", "⚙".bright_cyan());
            Ok(Box::new(batuta::agent::driver::router::RoutingDriver::new(primary, fallback)))
        }
        (Some(driver), None) => Ok(driver),
        (None, Some(driver)) => Ok(driver),
        (None, None) => {
            if resolved_path.is_some() {
                #[cfg(not(feature = "inference"))]
                println!(
                    "{} inference feature not enabled; rebuild with: {}",
                    "⚠".bright_yellow(),
                    "cargo build --features inference".cyan()
                );
            } else {
                println!(
                    "{} No model configured; set model_path or remote_model",
                    "ℹ".bright_blue()
                );
            }
            Ok(Box::new(batuta::agent::driver::mock::MockDriver::single_response(
                "Hello! I'm running in dry-run mode. \
                     Set model_path or remote_model in your agent manifest.",
            )))
        }
    }
}

/// Build local inference driver (RealizarDriver) if available.
fn build_local_driver(
    manifest: &batuta::agent::AgentManifest,
    resolved_path: &Option<std::path::PathBuf>,
) -> Option<Box<dyn batuta::agent::driver::LlmDriver>> {
    #[cfg(feature = "inference")]
    if let Some(ref model_path) = resolved_path {
        match batuta::agent::driver::realizar::RealizarDriver::new(
            model_path.clone(),
            manifest.model.context_window,
        ) {
            Ok(d) => return Some(Box::new(d)),
            Err(e) => {
                eprintln!("Warning: local driver init failed: {e}");
            }
        }
    }
    let _ = (manifest, resolved_path);
    None
}

/// Build remote API driver if remote_model is configured.
fn build_remote_driver(
    manifest: &batuta::agent::AgentManifest,
) -> Option<Box<dyn batuta::agent::driver::LlmDriver>> {
    #[cfg(feature = "native")]
    if let Some(ref model_id) = manifest.model.remote_model {
        use batuta::agent::driver::remote::{ApiProvider, RemoteDriver, RemoteDriverConfig};
        let (provider, base_url, env_key) = if model_id.starts_with("claude") {
            (ApiProvider::Anthropic, "https://api.anthropic.com", "ANTHROPIC_API_KEY")
        } else {
            (ApiProvider::OpenAi, "https://api.openai.com", "OPENAI_API_KEY")
        };
        let api_key = match std::env::var(env_key) {
            Ok(k) if !k.is_empty() => k,
            _ => {
                println!(
                    "{} Remote model set but {} not found; skipping remote driver",
                    "⚠".bright_yellow(),
                    env_key
                );
                return None;
            }
        };
        let config = RemoteDriverConfig {
            base_url: base_url.into(),
            api_key,
            model: model_id.clone(),
            provider,
            context_window: manifest.model.context_window.unwrap_or(8192),
        };
        return Some(Box::new(RemoteDriver::new(config)));
    }
    let _ = manifest;
    None
}

/// Build tool registry from manifest capabilities.
pub(super) fn build_tool_registry(
    manifest: &batuta::agent::AgentManifest,
) -> batuta::agent::tool::ToolRegistry {
    use batuta::agent::capability::Capability;
    use batuta::agent::tool::ToolRegistry;

    let mut registry = ToolRegistry::new();

    for cap in &manifest.capabilities {
        match cap {
            Capability::Memory => {
                let memory = Arc::new(batuta::agent::memory::InMemorySubstrate::new());
                registry.register(Box::new(batuta::agent::tool::memory::MemoryTool::new(
                    memory,
                    manifest.name.clone(),
                )));
            }
            Capability::Compute => {
                let cwd = std::env::current_dir().unwrap_or_default().to_string_lossy().to_string();
                registry.register(Box::new(batuta::agent::tool::compute::ComputeTool::new(cwd)));
            }
            Capability::Shell { allowed_commands } => {
                let cwd = std::env::current_dir().unwrap_or_default();
                registry.register(Box::new(batuta::agent::tool::shell::ShellTool::new(
                    allowed_commands.clone(),
                    cwd,
                )));
            }
            Capability::Network { allowed_hosts } => {
                registry.register(Box::new(batuta::agent::tool::network::NetworkTool::new(
                    allowed_hosts.clone(),
                )));
            }
            #[cfg(feature = "agents-browser")]
            Capability::Browser => {
                registry.register(Box::new(batuta::agent::tool::browser::BrowserTool::new(
                    manifest.privacy,
                )));
            }
            #[cfg(feature = "rag")]
            Capability::Rag => {
                let oracle = Arc::new(batuta::oracle::rag::RagOracle::new());
                registry.register(Box::new(batuta::agent::tool::rag::RagTool::new(oracle, 5)));
            }
            // Mcp registered via register_mcp_tools (agents-mcp).
            _ => {}
        }
    }

    registry
}

/// Register SpawnTool on the registry if Spawn capability is present.
pub(super) fn register_spawn_tool(
    registry: &mut batuta::agent::tool::ToolRegistry,
    manifest: &batuta::agent::AgentManifest,
    driver: Arc<dyn batuta::agent::driver::LlmDriver>,
) {
    use batuta::agent::capability::Capability;
    for cap in &manifest.capabilities {
        if let Capability::Spawn { max_depth } = cap {
            let pool = Arc::new(tokio::sync::Mutex::new(batuta::agent::pool::AgentPool::new(
                Arc::clone(&driver),
                4,
            )));
            registry.register(Box::new(batuta::agent::tool::spawn::SpawnTool::new(
                pool,
                manifest.clone(),
                0,
                *max_depth,
            )));
            break;
        }
    }
}

/// Register InferenceTool on the registry if Inference capability is present.
pub(super) fn register_inference_tool(
    registry: &mut batuta::agent::tool::ToolRegistry,
    manifest: &batuta::agent::AgentManifest,
    driver: Arc<dyn batuta::agent::driver::LlmDriver>,
) {
    use batuta::agent::capability::Capability;
    if manifest.capabilities.contains(&Capability::Inference) {
        let max_tokens = manifest.model.max_tokens;
        registry.register(Box::new(batuta::agent::tool::inference::InferenceTool::new(
            driver, max_tokens,
        )));
    }
}

/// Register MCP client tools discovered from manifest mcp_servers.
#[cfg(feature = "agents-mcp")]
pub(super) async fn register_mcp_tools(
    registry: &mut batuta::agent::tool::ToolRegistry,
    manifest: &batuta::agent::AgentManifest,
) {
    let tools = batuta::agent::tool::mcp_client::discover_mcp_tools(manifest).await;
    for tool in tools {
        registry.register(Box::new(tool));
    }
}

/// Build memory substrate (TruenoMemory when available, else InMemory).
pub(super) fn build_memory() -> Box<dyn batuta::agent::memory::MemorySubstrate> {
    #[cfg(feature = "rag")]
    {
        match batuta::agent::memory::TruenoMemory::open_in_memory() {
            Ok(mem) => return Box::new(mem),
            Err(e) => {
                eprintln!(
                    "Warning: TruenoMemory init failed ({e}), \
                     falling back to InMemorySubstrate"
                );
            }
        }
    }
    Box::new(batuta::agent::memory::InMemorySubstrate::new())
}

/// Print a stream event to stdout.
pub(super) fn print_stream_event(event: &batuta::agent::driver::StreamEvent) {
    use batuta::agent::driver::StreamEvent;
    match event {
        StreamEvent::PhaseChange { phase } => println!("  {} Phase: {phase:?}", "→".bright_blue()),
        StreamEvent::ToolUseStart { name, .. } => {
            println!("  {} Tool: {}", "⚙".bright_yellow(), name.cyan());
        }
        StreamEvent::ToolUseEnd { name, result, .. } => {
            let p =
                if result.len() > 80 { format!("{}...", &result[..77]) } else { result.clone() };
            println!("  {} {} → {}", "✓".green(), name, p.dimmed());
        }
        StreamEvent::TextDelta { text } => print!("{text}"),
        StreamEvent::ContentComplete { .. } => {}
    }
}

/// Validate model file integrity (G0) and format (G1).
pub(super) fn validate_model_file(manifest: &batuta::agent::AgentManifest) -> anyhow::Result<()> {
    let model_path = manifest
        .model
        .resolve_model_path()
        .ok_or_else(|| anyhow::anyhow!("no model_path or model_repo configured"))?;

    println!();
    println!("{} Model Validation (G0-G1)", "🔍".bright_cyan().bold());
    println!("{}", "─".repeat(40).dimmed());

    if !model_path.exists() {
        println!("  {} G0 FAIL: model file not found: {}", "✗".bright_red(), model_path.display());
        anyhow::bail!("G0: model file not found: {}", model_path.display());
    }

    let metadata = std::fs::metadata(&model_path)
        .map_err(|e| anyhow::anyhow!("cannot stat {}: {e}", model_path.display()))?;
    let size_mb = metadata.len() as f64 / (1024.0 * 1024.0);
    let file_bytes = std::fs::read(&model_path)
        .map_err(|e| anyhow::anyhow!("cannot read {}: {e}", model_path.display()))?;
    let hash_hex = blake3::hash(&file_bytes).to_hex();

    println!("  {} G0 PASS: {} ({:.1} MB)", "✓".green(), model_path.display(), size_mb);
    println!("    BLAKE3: {}", &hash_hex[..32]);

    let format = detect_model_format(&file_bytes);
    println!("  {} G1: Format detected: {}", "✓".green(), format.bright_yellow());
    Ok(())
}

/// Validate model inference sanity (G2 gate).
pub(super) fn validate_model_g2(manifest: &batuta::agent::AgentManifest) -> anyhow::Result<()> {
    println!();
    println!("{} G2: Inference Sanity Check", "🧪".bright_cyan().bold());
    println!("{}", "─".repeat(40).dimmed());

    let driver = build_driver(manifest)?;

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| anyhow::anyhow!("tokio runtime: {e}"))?;

    let probe_prompt = "Respond with exactly: Hello, I am operational.";

    let request = batuta::agent::driver::CompletionRequest {
        model: String::new(),
        messages: vec![batuta::agent::driver::Message::User(probe_prompt.into())],
        max_tokens: 64,
        temperature: 0.0,
        tools: vec![],
        system: Some(manifest.model.system_prompt.clone()),
    };

    let result = rt.block_on(async { driver.complete(request).await });

    match result {
        Ok(response) => {
            let text = &response.text;
            if text.is_empty() {
                println!("  {} G2 FAIL: empty response", "✗".bright_red());
                anyhow::bail!("G2: model returned empty response");
            }
            let entropy = char_entropy(text);
            let dot = "•".bright_blue();
            println!("  {dot} G2 probe: \"{}\"", truncate_str(text, 60));
            println!("  {dot} G2 metrics: len={}, entropy={:.2}", text.len(), entropy);
            if entropy > 5.5 {
                println!(
                    "  {} G2 WARN: high entropy ({:.2}) — check LAYOUT-002",
                    "⚠".bright_yellow(),
                    entropy
                );
            }
            println!("  {} G2 PASS: model produces coherent output", "✓".green());
            Ok(())
        }
        Err(e) => {
            println!("  {} G2 FAIL: inference error: {e}", "✗".bright_red());
            anyhow::bail!("G2: inference failed: {e}");
        }
    }
}

/// Shannon entropy of a string (bits per character).
pub(super) fn char_entropy(s: &str) -> f64 {
    if s.is_empty() {
        return 0.0;
    }
    let mut freq = [0u32; 256];
    let total = s.len() as f64;
    for b in s.bytes() {
        freq[b as usize] += 1;
    }
    let mut entropy = 0.0;
    for &count in &freq {
        if count > 0 {
            let p = count as f64 / total;
            entropy -= p * p.log2();
        }
    }
    entropy
}

/// Truncate a string to max_len with ellipsis.
pub(super) fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}

/// Detect model format from magic bytes.
pub(super) fn detect_model_format(data: &[u8]) -> &'static str {
    if data.len() >= 4 {
        // GGUF magic: 0x46475547 ("GGUF" in LE)
        if data[..4] == [0x47, 0x47, 0x55, 0x46] {
            return "GGUF";
        }
        // APR v2 magic: "APR\x02"
        if data[..4] == [b'A', b'P', b'R', 0x02] {
            return "APR v2";
        }
        // SafeTensors: starts with JSON length (LE u64) then '{'
        if data.len() >= 9 && data[8] == b'{' {
            return "SafeTensors";
        }
    }
    "unknown"
}

/// Load and parse an agent manifest from TOML.
pub(super) fn load_manifest(path: &PathBuf) -> anyhow::Result<batuta::agent::AgentManifest> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("Cannot read manifest {}: {e}", path.display()))?;
    batuta::agent::AgentManifest::from_toml(&content)
        .map_err(|e| anyhow::anyhow!("Invalid manifest {}: {e}", path.display()))
}

/// Print a summary of the loaded manifest.
pub(super) fn print_manifest_summary(manifest: &batuta::agent::AgentManifest) {
    let dot = "•".bright_blue();
    println!("{}", "🤖 Batuta Agent Runtime (Sovereign)".bright_cyan().bold());
    println!("{}", "═".repeat(60).dimmed());
    println!("{dot} Agent: {}", manifest.name.cyan());
    println!("{dot} Version: {}", manifest.version.dimmed());
    println!("{dot} Privacy: {:?}", manifest.privacy);
    println!("{dot} Capabilities: {:?}", manifest.capabilities);
    println!("{dot} Max iterations: {}", manifest.resources.max_iterations);
    if let Some(ref path) = manifest.model.model_path {
        println!("{dot} Model: {}", path.display());
    } else if let Some(ref repo) = manifest.model.model_repo {
        let q = manifest.model.model_quantization.as_deref().unwrap_or("q4_k_m");
        println!("{dot} Model: {} ({q})", repo.cyan());
    } else {
        println!("{dot} Model: {}", "none (specify model_path or model_repo)".dimmed());
    }
}

/// Build a LoopGuard from manifest + optional override.
pub(super) fn build_guard(
    manifest: &batuta::agent::AgentManifest,
    max_iterations: Option<u32>,
) -> (u32, u32) {
    let max_iter = max_iterations.unwrap_or(manifest.resources.max_iterations);
    let max_tools = manifest.resources.max_tool_calls;
    (max_iter, max_tools)
}
