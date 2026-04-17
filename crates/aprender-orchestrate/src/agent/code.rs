//! Public entry point for `apr code` / `batuta code`.
//!
//! This module provides the library-level API that both the `batuta` binary
//! and `apr-cli` use to launch the coding assistant. All logic lives here;
//! CLI wrappers are thin dispatchers.
//!
//! PMAT-162: Phase 6 — makes `cmd_code` accessible from the library crate
//! so `apr-cli` can call `batuta::agent::code::cmd_code()` directly.

use std::path::PathBuf;
use std::sync::Arc;

use crate::agent::capability::Capability;
use crate::agent::driver::LlmDriver;
use crate::agent::manifest::{AgentManifest, ModelConfig, ResourceQuota};
use crate::agent::tool::file::{FileEditTool, FileReadTool, FileWriteTool};
use crate::agent::tool::search::{GlobTool, GrepTool};
use crate::agent::tool::shell::ShellTool;
use crate::agent::tool::ToolRegistry;
use crate::serve::backends::PrivacyTier;

/// Entry point for `batuta code` / `apr code`.
///
/// This is the public library API — callable from both the batuta binary
/// and apr-cli (PMAT-162). Handles model discovery, driver selection,
/// tool registration, and REPL launch.
pub fn cmd_code(
    model: Option<PathBuf>,
    project: PathBuf,
    resume: Option<Option<String>>,
    prompt: Vec<String>,
    print: bool,
    max_turns: u32,
    manifest_path: Option<PathBuf>,
) -> anyhow::Result<()> {
    // --project: change working directory for project instructions
    if project.as_os_str() != "." && project.is_dir() {
        std::env::set_current_dir(&project)?;
    }

    // Load manifest or build default
    let mut manifest = match manifest_path {
        Some(ref path) => {
            let content = std::fs::read_to_string(path)
                .map_err(|e| anyhow::anyhow!("cannot read manifest {}: {e}", path.display()))?;
            let m = AgentManifest::from_toml(&content)
                .map_err(|e| anyhow::anyhow!("invalid manifest: {e}"))?;
            eprintln!("✓ Loaded manifest: {}", path.display());
            m
        }
        None => build_default_manifest(),
    };

    // --model flag overrides manifest model_path
    if let Some(ref model_path) = model {
        manifest.model.model_path = Some(model_path.clone());
    }

    // PMAT-150: discover model with Jidoka validation (broken APR → GGUF fallback)
    discover_and_set_model(&mut manifest);

    // PMAT-198: Scale system prompt based on model size.
    // Small models (<2B) degrade with the full tool table + project context.
    if let Some(ref path) = manifest.model.model_path {
        let params_b = estimate_model_params_from_name(path);
        if params_b < 2.0 {
            manifest.model.system_prompt = scale_prompt_for_model(params_b);
        }
    }

    // Contract: no_model_error — never silently use MockDriver
    if manifest.model.resolve_model_path().is_none() && manifest_path.is_none() {
        print_no_model_error();
        std::process::exit(exit_code::NO_MODEL);
    }

    // PMAT-160: Try AprServeDriver first (apr serve has full CUDA/GPU).
    // Falls back to embedded RealizarDriver if `apr` binary not found.
    let driver: Box<dyn LlmDriver> = if let Some(model_path) = manifest.model.resolve_model_path() {
        match crate::agent::driver::apr_serve::AprServeDriver::launch(
            model_path,
            manifest.model.context_window,
        ) {
            Ok(d) => Box::new(d),
            Err(e) => {
                eprintln!("⚠ apr serve unavailable ({e}), using embedded inference");
                build_fallback_driver(&manifest)?
            }
        }
    } else {
        build_fallback_driver(&manifest)?
    };

    // Build tool registry with coding tools
    let tools = build_code_tools(&manifest);

    // Build memory
    let memory = crate::agent::memory::InMemorySubstrate::new();

    // Non-interactive mode: single prompt
    // PMAT-161: Return exit code instead of process::exit() so driver Drop
    // runs and kills the apr serve subprocess (no zombie processes).
    if print || !prompt.is_empty() {
        let prompt_text = if prompt.is_empty() {
            let mut buf = String::new();
            std::io::Read::read_to_string(&mut std::io::stdin(), &mut buf)?;
            buf
        } else {
            prompt.join(" ")
        };
        let code = run_single_prompt(&manifest, driver.as_ref(), &tools, &memory, &prompt_text);
        drop(driver); // Kill apr serve subprocess before exit
        std::process::exit(code);
    }

    // --resume: load previous session
    // PMAT-165: auto-resume prompt when recent session exists (spec §6.3)
    let resume_session_id = match resume {
        Some(Some(id)) => Some(id), // --resume=<session-id>
        Some(None) => {
            // --resume (no ID): find most recent for cwd
            crate::agent::session::SessionStore::find_recent_for_cwd().map(|m| m.id)
        }
        None => {
            // No --resume flag: check for recent session and prompt
            crate::agent::session::offer_auto_resume()
        }
    };

    // Interactive REPL (local inference is free — budget unlimited)
    crate::agent::repl::run_repl(
        &manifest,
        driver.as_ref(),
        &tools,
        &memory,
        max_turns,
        f64::MAX,
        resume_session_id.as_deref(),
    )
}

/// Build fallback driver (embedded RealizarDriver) when AprServeDriver unavailable.
fn build_fallback_driver(manifest: &AgentManifest) -> anyhow::Result<Box<dyn LlmDriver>> {
    #[cfg(feature = "inference")]
    {
        if let Some(model_path) = manifest.model.resolve_model_path() {
            let driver = crate::agent::driver::realizar::RealizarDriver::new(
                model_path,
                manifest.model.context_window,
            )?;
            return Ok(Box::new(driver));
        }
    }
    let _ = manifest;
    // No model or no inference feature — return MockDriver
    Ok(Box::new(crate::agent::driver::mock::MockDriver::single_response(
        "Hello! I'm running in dry-run mode. \
         Set model_path in your agent manifest or install the `apr` binary.",
    )))
}

/// Auto-discover model if none explicitly set (APR preferred over GGUF).
fn discover_and_set_model(manifest: &mut AgentManifest) {
    if manifest.model.model_path.is_some() || manifest.model.model_repo.is_some() {
        return;
    }
    let Some(discovered) = ModelConfig::discover_model() else {
        return;
    };
    eprintln!(
        "Model: {} (auto-discovered)",
        discovered.file_name().unwrap_or_default().to_string_lossy()
    );
    let ext = discovered.extension().and_then(|e| e.to_str()).unwrap_or("");
    if ext == "gguf" && check_invalid_apr_in_search_dirs() {
        eprintln!(
            "⚠ APR model found but invalid (missing tokenizer). Using GGUF fallback: {}",
            discovered.display()
        );
        eprintln!("  Re-convert with: apr convert <source>.gguf -o <output>.apr\n");
    }
    manifest.model.model_path = Some(discovered);
}

/// Print actionable error when no local model is available.
fn print_no_model_error() {
    eprintln!("✗ No local model found. apr code requires a local model.\n");
    if check_invalid_apr_in_search_dirs() {
        eprintln!("  ⚠ APR model(s) found but invalid (missing embedded tokenizer).");
        eprintln!("  Re-convert: apr convert <source>.gguf -o <output>.apr\n");
    }
    eprintln!("  Download a model (APR format preferred):");
    eprintln!("    apr pull qwen3:1.7b-q4k            (default — best tool use at 1.2GB)");
    eprintln!("    apr pull qwen3:8b-q4k              (recommended for complex tasks)");
    eprintln!();
    eprintln!("  Or place a .apr/.gguf file in ~/.apr/models/ (auto-discovered)");
    eprintln!();
    eprintln!("  Then run: apr code or apr code --model <path>");
}

/// Check if any APR files in standard model search dirs are invalid.
fn check_invalid_apr_in_search_dirs() -> bool {
    for dir in &ModelConfig::model_search_dirs() {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().is_some_and(|e| e == "apr")
                    && !crate::agent::driver::validate::is_valid_model_file(&path)
                {
                    return true;
                }
            }
        }
    }
    false
}

/// Load project-level instructions from APR.md or CLAUDE.md.
fn load_project_instructions(max_bytes: usize) -> Option<String> {
    let cwd = std::env::current_dir().ok()?;

    for filename in &["APR.md", "CLAUDE.md"] {
        let path = cwd.join(filename);
        if path.is_file() {
            if let Ok(content) = std::fs::read_to_string(&path) {
                if max_bytes == 0 {
                    return None;
                }
                let truncated = if content.len() > max_bytes {
                    let end = content
                        .char_indices()
                        .take_while(|(i, _)| *i < max_bytes)
                        .last()
                        .map(|(i, c)| i + c.len_utf8())
                        .unwrap_or(max_bytes.min(content.len()));
                    format!("{}...\n(truncated from {} bytes)", &content[..end], content.len())
                } else {
                    content
                };
                return Some(truncated);
            }
        }
    }
    None
}

/// Compute instruction budget based on model context window.
fn instruction_budget(context_window: usize) -> usize {
    if context_window < 4096 {
        return 0;
    }
    let budget = context_window / 4;
    budget.min(4096)
}

/// Gather project context — git info, file stats, language.
fn gather_project_context() -> String {
    let mut ctx = String::new();
    let cwd = std::env::current_dir().unwrap_or_default();
    ctx.push_str(&format!("Working directory: {}\n", cwd.display()));

    if let Ok(output) =
        std::process::Command::new("git").args(["rev-parse", "--abbrev-ref", "HEAD"]).output()
    {
        if output.status.success() {
            let branch = String::from_utf8_lossy(&output.stdout).trim().to_string();
            ctx.push_str(&format!("Git branch: {branch}\n"));
        }
    }
    if let Ok(output) =
        std::process::Command::new("git").args(["diff", "--stat", "--no-color"]).output()
    {
        if output.status.success() {
            let diff = String::from_utf8_lossy(&output.stdout);
            let dirty_count = diff.lines().count().saturating_sub(1);
            if dirty_count > 0 {
                ctx.push_str(&format!("Dirty files: {dirty_count}\n"));
            }
        }
    }

    let mut rs_count = 0u32;
    let mut py_count = 0u32;
    let mut total = 0u32;
    if let Ok(entries) = std::fs::read_dir("src") {
        for e in entries.flatten() {
            total += 1;
            if let Some(ext) = e.path().extension() {
                match ext.to_str() {
                    Some("rs") => rs_count += 1,
                    Some("py") => py_count += 1,
                    _ => {}
                }
            }
        }
    }
    let lang = if rs_count > py_count {
        "Rust"
    } else if py_count > 0 {
        "Python"
    } else {
        "unknown"
    };
    ctx.push_str(&format!("Language: {lang} ({total} files in src/)\n"));

    if PathBuf::from("Cargo.toml").exists() {
        ctx.push_str("Build system: Cargo (Rust)\n");
    } else if PathBuf::from("pyproject.toml").exists() {
        ctx.push_str("Build system: pyproject.toml (Python)\n");
    }

    ctx
}

/// Build a default `AgentManifest` for coding tasks.
fn build_default_manifest() -> AgentManifest {
    let ctx_window = 4096_usize;
    let budget = instruction_budget(ctx_window);
    let project_instructions = load_project_instructions(budget);
    let project_context = gather_project_context();

    let mut system_prompt = CODE_SYSTEM_PROMPT.to_string();
    system_prompt.push_str(&format!("\n\n## Project Context\n\n{project_context}"));
    if let Some(ref instructions) = project_instructions {
        system_prompt.push_str(&format!("\n## Project Instructions\n\n{instructions}"));
    }

    AgentManifest {
        name: "apr-code".to_string(),
        description: "Interactive AI coding assistant".to_string(),
        privacy: PrivacyTier::Sovereign,
        model: ModelConfig {
            system_prompt,
            max_tokens: 4096,
            temperature: 0.0,
            // PMAT-197: Qwen3 supports 32K context. Default 4096 caused
            // truncate_messages to drop user query (9 tool schemas ~4000 tokens
            // consumed the entire window). Set to 32K for Qwen3-class models.
            context_window: Some(32768),
            ..ModelConfig::default()
        },
        resources: ResourceQuota {
            max_iterations: 50,
            max_tool_calls: 200,
            max_cost_usd: 0.0,
            max_tokens_budget: None,
        },
        capabilities: vec![
            Capability::FileRead { allowed_paths: vec!["*".into()] },
            Capability::FileWrite { allowed_paths: vec!["*".into()] },
            Capability::Shell { allowed_commands: vec!["*".into()] },
            Capability::Memory,
            Capability::Rag,
        ],
        ..AgentManifest::default()
    }
}

/// Register all coding tools.
fn build_code_tools(manifest: &AgentManifest) -> ToolRegistry {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));

    let mut tools = ToolRegistry::new();
    tools.register(Box::new(FileReadTool::new(vec!["*".into()])));
    tools.register(Box::new(FileWriteTool::new(vec!["*".into()])));
    tools.register(Box::new(FileEditTool::new(vec!["*".into()])));
    tools.register(Box::new(GlobTool::new(vec!["*".into()])));
    tools.register(Box::new(GrepTool::new(vec!["*".into()])));
    tools.register(Box::new(ShellTool::new(vec!["*".into()], cwd)));

    let memory_sub = Arc::new(crate::agent::memory::InMemorySubstrate::new());
    tools.register(Box::new(crate::agent::tool::memory::MemoryTool::new(
        memory_sub,
        manifest.name.clone(),
    )));

    // PMAT-163: dedicated pmat_query tool
    tools.register(Box::new(crate::agent::tool::pmat_query::PmatQueryTool::new()));

    #[cfg(feature = "rag")]
    {
        let oracle = Arc::new(crate::oracle::rag::RagOracle::new());
        tools.register(Box::new(crate::agent::tool::rag::RagTool::new(oracle, 5)));
    }

    tools
}

pub use super::code_prompts::exit_code;

/// Run a single prompt (non-interactive). PMAT-172: cap iterations at 10.
fn run_single_prompt(
    manifest: &AgentManifest,
    driver: &dyn LlmDriver,
    tools: &ToolRegistry,
    memory: &dyn crate::agent::memory::MemorySubstrate,
    prompt: &str,
) -> i32 {
    let mut single_manifest = manifest.clone();
    single_manifest.resources.max_iterations = single_manifest.resources.max_iterations.min(10);
    // PMAT-197: Use compact system prompt for -p mode.
    // The full CODE_SYSTEM_PROMPT (9-tool table + project context + CLAUDE.md)
    // overwhelms Qwen3 1.7B causing </think> loops. For -p mode, use a minimal
    // prompt that lets the model answer directly. Tools still available if needed.
    single_manifest.model.system_prompt = COMPACT_SYSTEM_PROMPT.to_string();
    // Note: context_window is set at driver launch time (build_default_manifest),
    // not here. See PMAT-197 fix in build_default_manifest.

    let rt = match tokio::runtime::Builder::new_current_thread().enable_all().build() {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("Error: failed to create tokio runtime: {e}");
            return exit_code::AGENT_ERROR;
        }
    };

    // PMAT-197: Use non-nudge loop for -p mode. The nudge ("Use a tool!") forces
    // small models to make tool calls even for simple questions like "What is 2+2?"
    // which causes stuck loops. Let the model decide whether to use tools.
    let result = rt.block_on(crate::agent::runtime::run_agent_loop(
        &single_manifest,
        prompt,
        driver,
        tools,
        memory,
        None,
    ));

    match result {
        Ok(r) => {
            if r.text.is_empty() {
                // PMAT-190: Empty response — model may be emitting only thinking tokens
                // that get stripped by strip_thinking_blocks(). Common with Qwen3 when
                // the serve backend doesn't use Qwen3NoThinkTemplate.
                eprintln!(
                    "⚠ Empty response ({} iterations, {} tool calls). \
                     Model may be in thinking mode — rebuild apr from source for Qwen3NoThinkTemplate fix.",
                    r.iterations, r.tool_calls
                );
            } else {
                println!("{}", r.text);
            }
            exit_code::SUCCESS
        }
        Err(e) => {
            eprintln!("Error: {e}");
            map_error_to_exit_code(&e)
        }
    }
}

// Prompts and exit codes extracted to code_prompts.rs
use super::code_prompts::{
    estimate_model_params_from_name, map_error_to_exit_code, scale_prompt_for_model,
    CODE_SYSTEM_PROMPT, COMPACT_SYSTEM_PROMPT,
};

#[cfg(test)]
#[path = "code_tests.rs"]
mod tests;
