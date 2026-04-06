//! Command dispatch for the batuta CLI.
//!
//! Routes each `Commands` variant to its handler in the appropriate
//! `cli::` submodule. Oracle dispatch is in `main_oracle_dispatch.rs`.

use tracing::info;

use crate::cli;
use crate::main_cli::{Commands, McpTransport};
use crate::main_oracle_dispatch::dispatch_oracle;
use crate::pacha;

/// Dispatch CLI command to the appropriate handler.
#[allow(clippy::cognitive_complexity)]
pub(crate) fn dispatch_command(command: Commands) -> anyhow::Result<()> {
    match command {
        Commands::Init { source, output } => {
            info!("Initializing Batuta project from {:?}", source);
            cli::pipeline_cmds::cmd_init(source, output)
        }
        Commands::Analyze { path, tdg, languages, dependencies } => {
            info!("Analyzing project at {:?}", path);
            cli::pipeline_cmds::cmd_analyze(path, tdg, languages, dependencies)
        }
        Commands::Transpile { incremental, cache, modules, ruchy, repl } => {
            info!("Transpiling to {}", if ruchy { "Ruchy" } else { "Rust" });
            cli::pipeline_cmds::cmd_transpile(incremental, cache, modules, ruchy, repl)
        }
        Commands::Optimize { enable_gpu, enable_simd, profile, gpu_threshold } => {
            info!("Optimizing with profile: {:?}", profile);
            cli::pipeline_cmds::cmd_optimize(enable_gpu, enable_simd, profile, gpu_threshold)
        }
        Commands::Validate { trace_syscalls, diff_output, run_original_tests, benchmark } => {
            info!("Validating semantic equivalence");
            cli::pipeline_cmds::cmd_validate(
                trace_syscalls,
                diff_output,
                run_original_tests,
                benchmark,
            )
        }
        Commands::Build { release, target, wasm } => {
            info!("Building Rust project");
            cli::pipeline_cmds::cmd_build(release, target, wasm)
        }
        Commands::Report { output, format } => {
            info!("Generating migration report");
            cli::pipeline_cmds::cmd_report(output, format)
        }
        Commands::Status => {
            info!("Checking workflow status");
            cli::workflow::cmd_status()
        }
        Commands::Reset { yes } => {
            info!("Resetting workflow state");
            cli::workflow::cmd_reset(yes)
        }
        Commands::Parf { path, find, patterns, dependencies, dead_code, format, output } => {
            info!("Running PARF analysis on {:?}", path);
            cli::parf::cmd_parf(
                &path,
                find.as_deref(),
                patterns,
                dependencies,
                dead_code,
                format,
                output.as_deref(),
            )
        }
        Commands::Oracle { args } => dispatch_oracle(args),
        Commands::Stack { command } => {
            info!("Stack Mode");
            cli::stack::cmd_stack(command)
        }
        Commands::Hf { command } => {
            info!("HuggingFace Mode");
            cli::hf::cmd_hf(command)
        }
        Commands::Pacha { command } => {
            info!("Pacha Model Registry Mode");
            pacha::cmd_pacha(command)
        }
        Commands::Data { command } => {
            info!("Data Platforms Mode");
            cli::data::cmd_data(command)
        }
        Commands::Viz { command } => {
            info!("Visualization Frameworks Mode");
            cli::viz::cmd_viz(command)
        }
        Commands::Experiment { command } => {
            info!("Experiment Tracking Frameworks Mode");
            cli::experiment::cmd_experiment(command)
        }
        Commands::Content { command } => {
            info!("Content Creation Tooling Mode");
            cli::content::cmd_content(command)
        }
        Commands::Serve { model, host, port, openai_api, watch, banco } => {
            if banco {
                info!("Starting Banco Workbench API");
                #[cfg(feature = "banco")]
                {
                    let state = batuta::serve::banco::state::BancoStateInner::from_config();
                    // Load model at startup if --model provided
                    if let Some(ref model_path) = model {
                        if let Err(e) = state.model.load(model_path) {
                            eprintln!("[banco] Warning: failed to load model {model_path}: {e}");
                        }
                    }
                    tokio::runtime::Runtime::new()?
                        .block_on(batuta::serve::banco::start_server(&host, port, state))?;
                    return Ok(());
                }
                #[cfg(not(feature = "banco"))]
                {
                    anyhow::bail!(
                        "Banco feature not enabled. Rebuild with: cargo build --features banco"
                    );
                }
            }
            info!("Starting Model Server Mode");
            cli::serve::cmd_serve(model, &host, port, openai_api, watch)
        }
        Commands::Deploy { command } => {
            info!("Deployment Generation Mode");
            cli::deploy::cmd_deploy(command)
        }
        Commands::Falsify { path, critical_only, format, output, min_grade, verbose } => {
            info!("Popperian Falsification Checklist Mode");
            cli::falsify::cmd_falsify(path, critical_only, format, output, &min_grade, verbose)
        }
        Commands::BugHunter { command } => {
            info!("Proactive Bug Hunting Mode");
            cli::bug_hunter::handle_bug_hunter_command(command).map_err(|e| anyhow::anyhow!(e))
        }
        Commands::Mcp { transport } => {
            info!("MCP Server Mode");
            match transport {
                McpTransport::Stdio => {
                    let mut server = batuta::mcp::McpServer::new();
                    server.run_stdio()
                }
            }
        }
        Commands::Playbook { command } => {
            info!("Playbook Mode");
            cli::playbook::cmd_playbook(command)
        }
        #[cfg(feature = "agents")]
        Commands::Agent { command } => {
            info!("Agent Runtime Mode");
            cli::agent::cmd_agent(command)
        }
        #[cfg(feature = "agents")]
        Commands::Code { model, project, resume, prompt, print, max_turns, manifest } => {
            info!("Code Mode (apr code)");
            cli::code::cmd_code(model, project, resume, prompt, print, max_turns, manifest)
        }
    }
}
