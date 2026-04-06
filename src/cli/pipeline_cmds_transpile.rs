//! Transpile command handler and REPL (QA-002 split).

use crate::ansi_colors::Colorize;
use crate::config::BatutaConfig;
use crate::tools::{self, ToolInfo, ToolRegistry};
use crate::types::{Language, WorkflowPhase, WorkflowState};
use std::path::Path;
use std::path::PathBuf;
use tracing::warn;

pub fn cmd_transpile(
    incremental: bool,
    cache: bool,
    modules: Option<Vec<String>>,
    ruchy: bool,
    repl: bool,
) -> anyhow::Result<()> {
    contract_pre_transpile!(incremental);
    println!("{}", "🔄 Transpiling code...".bright_cyan().bold());
    println!();

    let state_file = crate::cli::get_state_file_path();
    let mut state = WorkflowState::load(&state_file).unwrap_or_else(|e| {
        warn!("Failed to load workflow state, starting fresh: {e}");
        WorkflowState::new()
    });

    // Check prerequisites
    let config = match check_transpile_prerequisites(&state) {
        Ok(c) => c,
        Err(e) => {
            state.fail_phase(WorkflowPhase::Transpilation, e.to_string());
            state.save(&state_file)?;
            return Ok(());
        }
    };

    state.start_phase(WorkflowPhase::Transpilation);
    state.save(&state_file)?;

    // Setup transpiler
    let (_tools, _primary_lang, transpiler) = match setup_transpiler(&config) {
        Ok(t) => t,
        Err(e) => {
            state.fail_phase(WorkflowPhase::Transpilation, e.to_string());
            state.save(&state_file)?;
            return Err(e);
        }
    };

    // Display settings and create directories
    display_transpilation_settings(&config, incremental, cache, ruchy, &modules);
    std::fs::create_dir_all(&config.transpilation.output_dir)?;
    std::fs::create_dir_all(config.transpilation.output_dir.join("src"))?;

    println!("{}", "🚀 Starting transpilation...".bright_green().bold());
    println!();

    // Build and display command
    let owned_args =
        crate::cli::build_transpiler_args(&config, incremental, cache, ruchy, &modules);
    let args: Vec<&str> = owned_args.iter().map(|s| s.as_str()).collect();

    println!("{}", "Executing:".dimmed());
    println!("  {} {} {}", "$".dimmed(), transpiler.name.cyan(), args.join(" ").dimmed());
    println!();
    println!("{}", "Transpiling...".bright_yellow());

    // Run transpiler and handle result
    let result = tools::run_tool(&transpiler.name, &args, Some(&config.source.path));
    handle_transpile_result(result, &mut state, &state_file, &config, &transpiler, repl)
}

fn check_transpile_prerequisites(state: &WorkflowState) -> anyhow::Result<BatutaConfig> {
    // Check if analysis phase is completed
    if !state.is_phase_completed(WorkflowPhase::Analysis) {
        println!("{}", "⚠️  Analysis phase not completed!".yellow().bold());
        println!();
        println!("Run {} first to analyze your project.", "batuta analyze <path>".cyan());
        println!();
        crate::cli::workflow::display_workflow_progress(state);
        anyhow::bail!("Analysis phase not completed");
    }

    // Load configuration
    let config_path = PathBuf::from("batuta.toml");
    if !config_path.exists() {
        println!("{}", "⚠️  No configuration file found!".yellow().bold());
        println!();
        println!("Run {} to create a configuration file.", "batuta init".cyan());
        println!();
        anyhow::bail!("No batuta.toml configuration file");
    }

    BatutaConfig::load(&config_path)
}

fn setup_transpiler(config: &BatutaConfig) -> anyhow::Result<(ToolRegistry, Language, ToolInfo)> {
    // Detect available tools
    let tools = ToolRegistry::detect();

    // Get primary language
    let primary_lang = config
        .project
        .primary_language
        .as_ref()
        .and_then(|s| s.parse().ok())
        .unwrap_or(Language::Other("unknown".to_string()));

    // Get appropriate transpiler
    let transpiler = match tools.get_transpiler_for_language(&primary_lang) {
        Some(t) => t.clone(),
        None => {
            // handle_missing_tools always returns Err, so this branch never completes
            handle_missing_tools(&tools, &primary_lang)?;
            unreachable!("handle_missing_tools always returns Err")
        }
    };

    println!(
        "{} Using transpiler: {} ({})",
        "✓".bright_green(),
        transpiler.name.cyan(),
        transpiler.version.as_ref().unwrap_or(&"unknown".to_string())
    );
    println!();

    Ok((tools, primary_lang, transpiler))
}

fn handle_missing_tools(tools: &ToolRegistry, primary_lang: &Language) -> anyhow::Result<()> {
    println!("{}", "❌ No transpiler available!".red().bold());
    println!();
    println!("{}", "Available tools:".bright_yellow());
    for tool in tools.available_tools() {
        println!("  {} {}", "•".bright_blue(), tool.cyan());
    }
    println!();
    println!(
        "{}: Install the appropriate transpiler for {}",
        "Hint".bold(),
        format!("{}", primary_lang).cyan()
    );
    println!("  {} cargo install depyler", "Python:".dimmed());
    println!("  {} cargo install bashrs", "Shell:".dimmed());
    println!("  {} cargo install decy", "C/C++:".dimmed());
    println!();
    anyhow::bail!("No transpiler available for {}", primary_lang)
}

fn display_transpilation_settings(
    config: &BatutaConfig,
    incremental: bool,
    cache: bool,
    ruchy: bool,
    modules: &Option<Vec<String>>,
) {
    println!("{}", "Transpilation Settings:".bright_yellow().bold());
    println!("  {} Source: {:?}", "•".bright_blue(), config.source.path);
    println!("  {} Output: {:?}", "•".bright_blue(), config.transpilation.output_dir);
    println!(
        "  {} Incremental: {}",
        "•".bright_blue(),
        if incremental || config.transpilation.incremental {
            "enabled".green()
        } else {
            "disabled".dimmed()
        }
    );
    println!(
        "  {} Caching: {}",
        "•".bright_blue(),
        if cache || config.transpilation.cache { "enabled".green() } else { "disabled".dimmed() }
    );

    if let Some(mods) = modules {
        println!("  {} Modules: {}", "•".bright_blue(), mods.join(", ").cyan());
    }

    if ruchy || config.transpilation.use_ruchy {
        println!("  {} Target: {}", "•".bright_blue(), "Ruchy".cyan());
        if let Some(strictness) = &config.transpilation.ruchy_strictness {
            println!("  {} Strictness: {}", "•".bright_blue(), strictness.cyan());
        }
    }
    println!();
}

fn handle_transpile_result(
    result: Result<String, anyhow::Error>,
    state: &mut WorkflowState,
    state_file: &Path,
    config: &BatutaConfig,
    transpiler: &ToolInfo,
    repl: bool,
) -> anyhow::Result<()> {
    match result {
        Ok(output) => handle_transpile_success(output, state, state_file, config, repl),
        Err(e) => handle_transpile_failure(e, state, state_file, config, transpiler),
    }
}

fn handle_transpile_success(
    output: String,
    state: &mut WorkflowState,
    state_file: &Path,
    config: &BatutaConfig,
    repl: bool,
) -> anyhow::Result<()> {
    println!();
    println!("{}", "✅ Transpilation completed successfully!".bright_green().bold());
    println!();

    // Display transpiler output
    if !output.trim().is_empty() {
        println!("{}", "Transpiler output:".bright_yellow());
        println!("{}", "─".repeat(50).dimmed());
        for line in output.lines().take(20) {
            println!("  {}", line.dimmed());
        }
        if output.lines().count() > 20 {
            println!("  {} ... ({} more lines)", "...".dimmed(), output.lines().count() - 20);
        }
        println!("{}", "─".repeat(50).dimmed());
        println!();
    }

    // Complete transpilation phase
    state.complete_phase(WorkflowPhase::Transpilation);
    state.save(state_file)?;

    crate::cli::workflow::display_workflow_progress(state);

    // Show next steps
    println!("{}", "💡 Next Steps:".bright_green().bold());
    println!(
        "  {} Check output directory: {:?}",
        "1.".bright_blue(),
        config.transpilation.output_dir
    );
    println!("  {} Run {} to optimize", "2.".bright_blue(), "batuta optimize".cyan());
    println!("  {} Run {} to validate", "3.".bright_blue(), "batuta validate".cyan());
    println!();

    // Start REPL if requested
    if repl {
        run_ruchy_repl(config)?;
    }

    Ok(())
}

fn run_ruchy_repl(config: &BatutaConfig) -> anyhow::Result<()> {
    // Check if ruchy is available
    let tools = ToolRegistry::detect();
    if tools.ruchy.is_none() {
        println!("{} Ruchy is not installed", "✗".red());
        println!();
        println!("Install Ruchy to use the interactive REPL:");
        println!("  {}", "cargo install ruchy".cyan());
        anyhow::bail!("Ruchy not found in PATH");
    }

    print_repl_banner(config);

    let stdin = std::io::stdin();
    let mut buffer = String::new();

    loop {
        print_repl_prompt(&buffer);

        let mut line = String::new();
        if stdin.read_line(&mut line)? == 0 {
            break;
        }

        match process_repl_line(&line, &mut buffer, config) {
            ReplAction::Continue => {}
            ReplAction::Quit => break,
        }
    }

    println!();
    println!("{}", "Ruchy REPL session ended.".dimmed());
    Ok(())
}

enum ReplAction {
    Continue,
    Quit,
}

fn print_repl_banner(config: &BatutaConfig) {
    println!("{}", "🔬 Ruchy REPL".bright_cyan().bold());
    println!("{}", "─".repeat(50).dimmed());
    println!("  Output dir: {}", config.transpilation.output_dir.display());
    println!(
        "  Strictness: {}",
        config.transpilation.ruchy_strictness.as_deref().unwrap_or("gradual")
    );
    println!();
    println!("  Type Ruchy code, then press Enter twice to execute.");
    println!("  Commands: {} {} {}", ":help".cyan(), ":clear".cyan(), ":quit".cyan());
    println!("{}", "─".repeat(50).dimmed());
    println!();
}

fn print_repl_prompt(buffer: &str) {
    if buffer.is_empty() {
        eprint!("{} ", "ruchy>".bright_blue());
    } else {
        eprint!("{} ", "   ..>".dimmed());
    }
}

fn process_repl_line(line: &str, buffer: &mut String, config: &BatutaConfig) -> ReplAction {
    let trimmed = line.trim();

    // Handle REPL commands (only when buffer is empty)
    if buffer.is_empty() {
        match trimmed {
            ":quit" | ":q" | ":exit" => return ReplAction::Quit,
            ":help" | ":h" => {
                print_repl_help();
                return ReplAction::Continue;
            }
            ":clear" | ":c" => {
                buffer.clear();
                println!("{}", "Buffer cleared.".dimmed());
                return ReplAction::Continue;
            }
            _ => {}
        }
    }

    // Empty line on non-empty buffer = execute
    if trimmed.is_empty() && !buffer.is_empty() {
        execute_repl_snippet(buffer, config);
        buffer.clear();
        return ReplAction::Continue;
    }

    // Accumulate code
    if !trimmed.is_empty() {
        buffer.push_str(line);
    }

    ReplAction::Continue
}

fn print_repl_help() {
    println!();
    println!("{}", "Ruchy REPL Commands:".bright_yellow().bold());
    println!("  {} — Show this help", ":help".cyan());
    println!("  {} — Clear the input buffer", ":clear".cyan());
    println!("  {} — Exit the REPL", ":quit".cyan());
    println!();
    println!("{}", "Usage:".bright_yellow().bold());
    println!("  Type code, press Enter twice to transpile and run.");
    println!("  Multi-line input is supported — keep typing until");
    println!("  you enter a blank line.");
    println!();
}

fn execute_repl_snippet(code: &str, config: &BatutaConfig) {
    // Save code to staging path before transpilation
    let tmp_dir = std::env::temp_dir();
    let snippet_path = tmp_dir.join("batuta_repl_snippet.rcy");
    if let Err(e) = std::fs::write(&snippet_path, code) {
        println!("{} Failed to write snippet: {}", "✗".red(), e);
        return;
    }

    // Build ruchy args
    let snippet_str = snippet_path.to_string_lossy().to_string();
    let strictness = config.transpilation.ruchy_strictness.as_deref().unwrap_or("gradual");
    let args: Vec<&str> = vec!["run", "--strictness", strictness, &snippet_str];

    println!("{}", "─".repeat(50).dimmed());
    match tools::run_tool("ruchy", &args, None) {
        Ok(output) => {
            if output.trim().is_empty() {
                println!("{}", "(no output)".dimmed());
            } else {
                println!("{}", output);
            }
        }
        Err(e) => {
            println!("{} {}", "Error:".red(), e);
        }
    }
    println!("{}", "─".repeat(50).dimmed());
    println!();

    // Clean up
    let _ = std::fs::remove_file(&snippet_path);
}

fn handle_transpile_failure(
    e: anyhow::Error,
    state: &mut WorkflowState,
    state_file: &Path,
    config: &BatutaConfig,
    transpiler: &ToolInfo,
) -> anyhow::Result<()> {
    println!();
    println!("{}", "❌ Transpilation failed!".red().bold());
    println!();
    println!("{}: {}", "Error".bold(), e.to_string().red());
    println!();

    state.fail_phase(WorkflowPhase::Transpilation, e.to_string());
    state.save(state_file)?;

    crate::cli::workflow::display_workflow_progress(state);

    // Provide helpful troubleshooting
    println!("{}", "💡 Troubleshooting:".bright_yellow().bold());
    println!("  {} Verify {} is properly installed", "•".bright_blue(), transpiler.name.cyan());
    println!("  {} Check that source path is correct: {:?}", "•".bright_blue(), config.source.path);
    println!("  {} Try running with {} for more details", "•".bright_blue(), "--verbose".cyan());
    println!(
        "  {} See transpiler docs: {}",
        "•".bright_blue(),
        format!("https://github.com/paiml/{}", transpiler.name).cyan()
    );
    println!();

    Err(e)
}
