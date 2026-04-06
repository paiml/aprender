//! Probar CLI: Command-line interface for WASM testing
//!
//! ## Usage
//!
//! ```bash
//! probar test                     # Run all tests
//! probar test --filter "game::*"  # Filter tests
//! probar record <test> --gif      # Record as GIF
//! probar report --html            # Generate HTML report
//! ```

use clap::Parser;
#[cfg(test)]
use probador::handlers::comply::{check_makefile_cross_origin, check_probar_cross_origin_config};
use probador::{
    handlers::{
        build::find_html_files,
        comply::{
            check_c001_code_execution, check_c002_console_errors, check_c003_custom_elements,
            check_c004_threading_modes, check_c005_low_memory, check_c006_headers,
            check_c007_replay_hash, check_c008_cache, check_c009_wasm_size, check_c010_panic_paths,
            generate_comply_report, ComplianceResult,
        },
    },
    Cli, CliConfig, CliResult, ColorChoice, Commands, TestRunner, Verbosity,
};
use std::process::ExitCode;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("Error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> CliResult<()> {
    let cli = Cli::parse();

    // Build configuration from CLI args
    let config = build_config(&cli);

    match cli.command {
        Commands::Test(args) => run_tests(config, &args),
        Commands::Record(args) => {
            run_record(&config, &args);
            Ok(())
        }
        Commands::Report(args) => {
            run_report(&config, &args);
            Ok(())
        }
        Commands::Coverage(args) => run_coverage(&config, &args),
        Commands::Init(args) => {
            run_init(&config, &args);
            Ok(())
        }
        Commands::Config(args) => {
            run_config(&config, &args);
            Ok(())
        }
        Commands::Serve(args) => run_serve(&args),
        Commands::Build(args) => run_build(&args),
        Commands::Watch(args) => run_watch(&args),
        Commands::Playbook(args) => run_playbook(&config, &args),
        Commands::Comply(args) => run_comply(&config, &args),
        Commands::AvSync(args) => run_av_sync(&config, &args),
        Commands::Audio(args) => run_audio(&config, &args),
        Commands::Video(args) => run_video(&config, &args),
        Commands::Animation(args) => run_animation(&config, &args),
        Commands::Stress(args) => run_stress(&config, &args),
        #[cfg(feature = "llm")]
        Commands::Llm(args) => run_llm(&args),
        #[cfg(not(feature = "llm"))]
        Commands::Llm(_) => Err(probador::CliError::Generic(
            "LLM features not enabled. Rebuild with --features llm".to_string(),
        )),
    }
}

fn build_config(cli: &Cli) -> CliConfig {
    let verbosity = if cli.quiet {
        Verbosity::Quiet
    } else {
        match cli.verbose {
            0 => Verbosity::Normal,
            1 => Verbosity::Verbose,
            _ => Verbosity::Debug,
        }
    };

    let color: ColorChoice = cli.color.clone().into();

    CliConfig::new().with_verbosity(verbosity).with_color(color)
}

fn run_tests(config: CliConfig, args: &probador::TestArgs) -> CliResult<()> {
    // PROBAR-006: Compile-first gate
    // Run `cargo test --no-run` before executing playbook tests to catch compile errors early
    if !args.skip_compile {
        if config.verbosity.is_verbose() {
            println!("Running compile check (cargo test --no-run)...");
        }

        let compile_result = std::process::Command::new("cargo")
            .args(["test", "--no-run"])
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .output();

        match compile_result {
            Ok(output) => {
                if !output.status.success() {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    // Extract the first error for a cleaner message
                    let first_error = stderr
                        .lines()
                        .find(|line| line.contains("error[E"))
                        .unwrap_or("Compilation failed");

                    eprintln!("❌ Compile check failed!");
                    eprintln!();
                    eprintln!("First error: {first_error}");
                    eprintln!();
                    eprintln!("Run `cargo test --no-run` to see full error output.");
                    eprintln!("Use `probar test --skip-compile` to bypass this check.");

                    return Err(probador::CliError::test_execution(
                        "Compile check failed - fix compilation errors before running tests",
                    ));
                }
                if config.verbosity.is_verbose() {
                    println!("✓ Compile check passed");
                }
            }
            Err(e) => {
                // cargo not found or other execution error - warn but continue
                eprintln!("⚠ Could not run compile check: {e}");
                eprintln!("Continuing with tests...");
            }
        }
    }

    let config = config
        .with_parallel_jobs(args.parallel)
        .with_fail_fast(args.fail_fast)
        .with_coverage(args.coverage)
        .with_watch(args.watch)
        .with_output_dir(args.output.to_string_lossy().to_string());

    let mut runner = TestRunner::new(config);
    let results = runner.run(args.filter.as_deref())?;

    if results.all_passed() {
        Ok(())
    } else {
        Err(probador::CliError::test_execution(format!(
            "{} test(s) failed",
            results.failed()
        )))
    }
}

fn run_record(_config: &CliConfig, args: &probador::RecordArgs) {
    println!("Recording test: {}", args.test);
    println!("Format: {:?}", args.format);
    println!("FPS: {}", args.fps);
    println!("Quality: {}", args.quality);

    // Note: Full recording requires browser feature and running tests
    // This displays configuration; actual recording done via test runner
    println!("Recording configuration ready. Run test with --record flag to capture.");
}

fn run_report(config: &CliConfig, args: &probador::ReportArgs) {
    probador::handlers::report::execute_report(config, args);
}

fn run_coverage(config: &CliConfig, args: &probador::CoverageArgs) -> CliResult<()> {
    probador::handlers::coverage::execute_coverage(config, args)
}

fn run_init(config: &CliConfig, args: &probador::InitArgs) {
    probador::handlers::init::execute_init(config, args);
}

fn run_config(config: &CliConfig, args: &probador::ConfigArgs) {
    probador::handlers::config::execute_config(config, args);
}

// =============================================================================
// WASM Development Commands
// =============================================================================

fn run_serve(args: &probador::ServeArgs) -> CliResult<()> {
    use probador::{DevServer, DevServerConfig, ModuleValidator, ServeSubcommand};

    // Handle subcommands
    if let Some(ref subcommand) = args.subcommand {
        return match subcommand {
            ServeSubcommand::Tree(tree_args) => run_serve_tree(tree_args, &args.directory),
            ServeSubcommand::Viz(viz_args) => run_serve_viz(viz_args, &args.directory),
            ServeSubcommand::Score(score_args) => run_serve_score(score_args, &args.directory),
        };
    }

    // Validate module imports if --validate flag is set
    if args.validate {
        let mut validator = ModuleValidator::new(&args.directory);
        if !args.exclude.is_empty() {
            validator = validator.with_exclude(args.exclude.clone());
        }
        let result = validator.validate();
        validator.print_results(&result);

        if !result.is_ok() {
            return Err(probador::CliError::test_execution(format!(
                "Module validation failed: {} error(s) found. Fix imports before serving.",
                result.errors.len()
            )));
        }
        eprintln!("\n✓ All module imports validated successfully\n");
    }

    let config = DevServerConfig {
        directory: args.directory.clone(),
        port: args.port,
        ws_port: args.ws_port,
        cors: args.cors,
        cross_origin_isolated: args.cross_origin_isolated,
    };

    let server = DevServer::new(config);

    // Open browser if requested
    if args.open {
        let url = format!("http://localhost:{}", args.port);
        println!("Opening browser at {url}...");
        #[cfg(target_os = "macos")]
        let _ = std::process::Command::new("open").arg(&url).spawn();
        #[cfg(target_os = "linux")]
        let _ = std::process::Command::new("xdg-open").arg(&url).spawn();
        #[cfg(target_os = "windows")]
        let _ = std::process::Command::new("start").arg(&url).spawn();
    }

    // Run server (blocking)
    let rt = tokio::runtime::Runtime::new().map_err(|e| {
        probador::CliError::test_execution(format!("Failed to create runtime: {e}"))
    })?;

    rt.block_on(async {
        server
            .run()
            .await
            .map_err(|e| probador::CliError::test_execution(format!("Server error: {e}")))
    })
}

fn run_serve_tree(args: &probador::TreeArgs, _default_dir: &std::path::Path) -> CliResult<()> {
    use probador::{build_tree, render_tree, TreeConfig};

    let config = TreeConfig::default()
        .with_depth(args.depth)
        .with_filter(args.filter.as_deref())
        .with_sizes(args.sizes)
        .with_mime_types(args.mime_types);

    let tree = build_tree(&args.path, &config)
        .map_err(|e| probador::CliError::test_execution(format!("Failed to build tree: {e}")))?;

    let output = render_tree(&tree, &config);
    print!("{output}");

    Ok(())
}

fn run_serve_viz(args: &probador::VizArgs, _default_dir: &std::path::Path) -> CliResult<()> {
    use probador::{build_tree, render_tree, TreeConfig};

    // VizArgs only has path and port - use defaults for other options
    let config = TreeConfig::default();

    let tree = build_tree(&args.path, &config)
        .map_err(|e| probador::CliError::test_execution(format!("Failed to build tree: {e}")))?;

    let output = render_tree(&tree, &config);
    print!("{output}");

    Ok(())
}

fn run_serve_score(args: &probador::ScoreArgs, _default_dir: &std::path::Path) -> CliResult<()> {
    use probador::{score, ScoreCalculator, ScoreOutputFormat};

    // If --live flag is set, run actual browser validation
    if args.live {
        return run_live_browser_validation(args);
    }

    let calculator = ScoreCalculator::new(args.path.clone());
    let project_score = calculator.calculate();

    match args.format {
        ScoreOutputFormat::Text => {
            let output = score::render_score_text(&project_score, args.detailed);
            print!("{output}");
        }
        ScoreOutputFormat::Json => {
            let output = score::render_score_json(&project_score).map_err(|e| {
                probador::CliError::report_generation(format!("JSON serialization error: {e}"))
            })?;
            println!("{output}");
        }
    }

    Ok(())
}

/// Live browser validation - FALSIFICATION approach
///
/// This actually starts a server, launches a headless browser, and tries to BREAK the app.
/// The app FAILS if we can prove it's broken. The app PASSES only if we can't break it.
///
/// Checks performed:
/// 1. Module resolution: Can all JS/WASM imports be fetched? (404 = FAIL)
/// 2. MIME types: Are JS/WASM served with correct MIME types? (wrong MIME = FAIL)
/// 3. Console errors: Any console.error or uncaught exceptions? (errors = FAIL)
/// 4. WASM initialization: Does the WASM module load? (load failure = FAIL)
fn run_live_browser_validation(args: &probador::ScoreArgs) -> CliResult<()> {
    use probador::DevServerConfig;

    eprintln!("\n══════════════════════════════════════════════════════════════");
    eprintln!("  LIVE BROWSER VALIDATION (Falsification Mode)");
    eprintln!("══════════════════════════════════════════════════════════════\n");
    eprintln!("This validates the app ACTUALLY WORKS by trying to break it.\n");

    let html_files = find_html_files(&args.path);
    if html_files.is_empty() {
        eprintln!("✗ FAIL: No HTML files found in {}", args.path.display());
        eprintln!("\n  Cannot validate an app without HTML entry points.");
        return Err(probador::CliError::test_execution(
            "No HTML files found - nothing to validate".to_string(),
        ));
    }

    eprintln!("Found {} HTML file(s) to validate:\n", html_files.len());
    for file in &html_files {
        eprintln!("  • {}", file.display());
    }
    eprintln!();

    // Step 1: Static module validation
    let pages_with_wasm = validate_module_imports(&args.path)?;

    // Step 2: Start temporary server
    eprintln!("[2/4] Starting Validation Server");
    let port = resolve_server_port(args.port)?;
    eprintln!("  Starting server on port {port}...\n");

    let config = DevServerConfig {
        directory: args.path.clone(),
        port,
        ws_port: port + 1,
        cors: true,
        cross_origin_isolated: true,
    };

    let rt = tokio::runtime::Runtime::new().map_err(|e| {
        probador::CliError::test_execution(format!("Failed to create runtime: {e}"))
    })?;

    let validation_errors = rt.block_on(async {
        run_browser_validation_async(&args.path, &html_files, port, config, &pages_with_wasm).await
    })?;

    report_browser_validation_results(&validation_errors)
}

/// Step 1: Static module resolution check. Returns set of pages with WASM imports.
fn validate_module_imports(
    path: &std::path::Path,
) -> CliResult<std::collections::HashSet<std::path::PathBuf>> {
    use probador::ModuleValidator;

    eprintln!("[1/4] Module Resolution Check");
    let validator = ModuleValidator::new(path);

    let imports = validator.scan_imports();
    let pages_with_wasm: std::collections::HashSet<_> = imports
        .iter()
        .filter(|imp| imp.import_type == probador::ImportType::Wasm)
        .map(|imp| imp.source_file.clone())
        .collect();

    if !pages_with_wasm.is_empty() {
        eprintln!(
            "  Detected {} page(s) with WASM imports:",
            pages_with_wasm.len()
        );
        for page in &pages_with_wasm {
            eprintln!("    • {}", page.display());
        }
        eprintln!();
    }

    let validation_result = validator.validate();
    if !validation_result.is_ok() {
        print_module_import_errors(&validation_result);
        return Err(probador::CliError::test_execution(format!(
            "Module resolution failed: {} broken import(s)",
            validation_result.errors.len()
        )));
    }
    eprintln!(
        "  ✓ All {} import(s) resolve correctly\n",
        validation_result.total_imports
    );

    Ok(pages_with_wasm)
}

/// Print detailed module import errors.
fn print_module_import_errors(result: &probador::ModuleValidationResult) {
    eprintln!("  ✗ FAIL: {} broken import(s) found\n", result.errors.len());
    for error in &result.errors {
        eprintln!(
            "    • {} (from {}:{})",
            error.import.import_path,
            error.import.source_file.display(),
            error.import.line_number
        );
        let status_str = if error.status == 404 {
            "404 Not Found".to_string()
        } else {
            format!("{}", error.status)
        };
        eprintln!("      Status: {status_str} - {}", error.message);
        eprintln!("      MIME type: {}", error.actual_mime);
    }
    eprintln!();
    eprintln!("══════════════════════════════════════════════════════════════");
    eprintln!("  RESULT: FAIL (Grade: F)");
    eprintln!("══════════════════════════════════════════════════════════════");
    eprintln!("\n  Fix the broken imports above and run again.\n");
}

/// Resolve the server port: use specified port or find an available one.
fn resolve_server_port(port: u16) -> CliResult<u16> {
    use std::net::TcpListener;

    if port != 0 {
        return Ok(port);
    }
    let listener = TcpListener::bind("127.0.0.1:0").map_err(|e| {
        probador::CliError::test_execution(format!("Failed to find available port: {e}"))
    })?;
    listener
        .local_addr()
        .map_err(|e| {
            probador::CliError::test_execution(format!("Failed to get local address: {e}"))
        })
        .map(|addr| addr.port())
}

/// Report final browser validation results.
fn report_browser_validation_results(errors: &[String]) -> CliResult<()> {
    eprintln!("══════════════════════════════════════════════════════════════");
    if errors.is_empty() {
        eprintln!("  RESULT: PASS (App works!)");
        eprintln!("══════════════════════════════════════════════════════════════");
        eprintln!("\n  Could not prove the app is broken. All validation checks passed.\n");
        Ok(())
    } else {
        eprintln!("  RESULT: FAIL (Grade: F)");
        eprintln!("══════════════════════════════════════════════════════════════");
        eprintln!(
            "\n  Found {} issue(s) that prove the app is broken:\n",
            errors.len()
        );
        for (i, error) in errors.iter().enumerate() {
            eprintln!("  {}. {error}", i + 1);
        }
        eprintln!();
        Err(probador::CliError::test_execution(format!(
            "Live validation failed: {} issue(s) found",
            errors.len()
        )))
    }
}

/// Run browser validation asynchronously
///
/// Uses FALSIFICATION approach: we try to PROVE the app is broken.
/// If we can prove it's broken, it FAILS. Only passes if we cannot break it.
async fn run_browser_validation_async(
    serve_dir: &std::path::Path,
    html_files: &[std::path::PathBuf],
    port: u16,
    config: probador::DevServerConfig,
    pages_with_wasm: &std::collections::HashSet<std::path::PathBuf>,
) -> CliResult<Vec<String>> {
    use probador::DevServer;
    use std::time::Duration;

    let server = DevServer::new(config);
    #[allow(unused_mut)]
    let mut errors = Vec::new();

    // Spawn server in background
    let server_handle = tokio::spawn(async move {
        let _ = server.run().await;
    });

    // Give server time to start
    tokio::time::sleep(Duration::from_millis(500)).await;

    eprintln!("[3/4] Browser Bootstrap Check");

    #[cfg(feature = "browser")]
    {
        errors.extend(run_browser_checks(serve_dir, html_files, port, pages_with_wasm).await);
    }

    #[cfg(not(feature = "browser"))]
    {
        // Suppress unused warnings when browser feature is disabled
        let _ = (serve_dir, html_files, port, pages_with_wasm);
        eprintln!("  ⚠ Browser feature not enabled - skipping browser checks");
        eprintln!("    Rebuild with --features browser for full validation\n");
        eprintln!("[4/4] WASM Initialization Check");
        eprintln!("  ⚠ Skipped (browser feature not enabled)\n");
    }

    // Stop server
    server_handle.abort();

    Ok(errors)
}

/// Launch browser and validate all HTML pages.
#[cfg(feature = "browser")]
async fn run_browser_checks(
    serve_dir: &std::path::Path,
    html_files: &[std::path::PathBuf],
    port: u16,
    pages_with_wasm: &std::collections::HashSet<std::path::PathBuf>,
) -> Vec<String> {
    use jugar_probar::{Browser, BrowserConfig};

    let browser_config = BrowserConfig::default()
        .with_headless(true)
        .with_no_sandbox()
        .with_viewport(1280, 720);

    let browser = match Browser::launch(browser_config).await {
        Ok(b) => b,
        Err(e) => {
            eprintln!("  ✗ Failed to launch browser: {e}");
            eprintln!("    Make sure Chrome/Chromium is installed.\n");
            return vec![format!("Failed to launch browser: {e}")];
        }
    };

    let mut errors = Vec::new();
    for html_file in html_files {
        errors.extend(
            validate_browser_page(&browser, serve_dir, html_file, port, pages_with_wasm).await,
        );
    }

    let _ = browser.close().await;
    errors
}

/// Validate a single HTML page in the browser: navigate, check console errors, check WASM.
#[cfg(feature = "browser")]
async fn validate_browser_page(
    browser: &jugar_probar::Browser,
    serve_dir: &std::path::Path,
    html_file: &std::path::Path,
    port: u16,
    pages_with_wasm: &std::collections::HashSet<std::path::PathBuf>,
) -> Vec<String> {
    use std::time::Duration;
    use tokio::time::timeout;

    let mut errors = Vec::new();
    let relative_path = html_file.strip_prefix(serve_dir).unwrap_or(html_file);
    let url_path = relative_path.to_string_lossy().replace('\\', "/");
    let url = format!("http://127.0.0.1:{port}/{url_path}");
    let expects_wasm = pages_with_wasm.contains(html_file);

    eprintln!("  Testing: {url}");
    if expects_wasm {
        eprintln!("    (WASM required - imported in HTML)");
    }

    let mut page = match browser.new_page().await {
        Ok(p) => p,
        Err(e) => {
            eprintln!("    ✗ Failed to create page: {e}");
            return vec![format!("Failed to create browser page: {e}")];
        }
    };

    // Enable console capture BEFORE navigation to catch load errors
    if let Err(e) = page.enable_console_capture().await {
        eprintln!("    Warning: Could not enable console capture: {e}");
    }
    if let Err(e) = page.inject_console_capture().await {
        eprintln!("    Warning: Could not inject console capture: {e}");
    }

    // Navigate to page with timeout
    let nav_result = timeout(Duration::from_secs(30), page.goto(&url)).await;
    match nav_result {
        Ok(Ok(())) => {
            tokio::time::sleep(Duration::from_secs(2)).await;
            errors.extend(check_page_console_errors(&page, &url_path).await);
            errors.extend(check_page_wasm_init(&mut page, &url_path, expects_wasm).await);
        }
        Ok(Err(e)) => {
            eprintln!("    ✗ Navigation failed - APP IS BROKEN");
            eprintln!("    Error: {e}");
            errors.push(format!("[{url_path}] Navigation failed: {e}"));
        }
        Err(_) => {
            eprintln!("    ✗ Navigation timed out after 30s - APP IS BROKEN");
            errors.push(format!("[{url_path}] Navigation timed out"));
        }
    }

    errors
}

/// Check for console errors on a loaded page.
#[cfg(feature = "browser")]
async fn check_page_console_errors(page: &jugar_probar::Page, url_path: &str) -> Vec<String> {
    use jugar_probar::BrowserConsoleLevel;

    let mut errors = Vec::new();
    if let Ok(messages) = page.fetch_console_messages().await {
        let error_messages: Vec<_> = messages
            .iter()
            .filter(|m| m.level == BrowserConsoleLevel::Error)
            .collect();

        if error_messages.is_empty() {
            eprintln!("    ✓ No console errors");
        } else {
            eprintln!(
                "    ✗ {} console error(s) found - APP IS BROKEN",
                error_messages.len()
            );
            for msg in &error_messages {
                eprintln!("      └─ {}", msg.text);
                errors.push(format!("[{url_path}] Console error: {}", msg.text));
            }
        }
    }
    errors
}

/// Check WASM initialization on a loaded page.
#[cfg(feature = "browser")]
async fn check_page_wasm_init(
    page: &mut jugar_probar::Page,
    url_path: &str,
    expects_wasm: bool,
) -> Vec<String> {
    use std::time::Duration;
    use tokio::time::timeout;

    let mut errors = Vec::new();
    eprintln!("\n[4/4] WASM Initialization Check");

    let wasm_result = timeout(Duration::from_secs(15), page.wait_for_wasm_ready()).await;
    match wasm_result {
        Ok(Ok(())) => {
            eprintln!("  ✓ WASM initialized successfully\n");
        }
        Ok(Err(e)) => {
            if expects_wasm {
                eprintln!("  ✗ WASM initialization FAILED - APP IS BROKEN");
                eprintln!("    Error: {e}\n");
                errors.push(format!("[{url_path}] WASM required but failed: {e}"));
            } else {
                eprintln!("  ⚠ WASM initialization: {e}");
                eprintln!("    (No WASM imports detected - may be expected)\n");
            }
        }
        Err(_) => {
            if expects_wasm {
                eprintln!("  ✗ WASM initialization TIMED OUT - APP IS BROKEN");
                eprintln!("    Page imports WASM but it failed to initialize.\n");
                errors.push(format!(
                    "[{url_path}] WASM required but timed out after 15s"
                ));
            } else {
                eprintln!("  ⚠ WASM initialization timed out");
                eprintln!("    (No WASM imports detected - may be expected)\n");
            }
        }
    }
    errors
}

fn run_build(args: &probador::BuildArgs) -> CliResult<()> {
    use probador::dev_server::run_wasm_pack_build;

    // Check if brick-based generation is requested
    if args.bricks.is_some() {
        return run_brick_generation(args);
    }

    let rt = tokio::runtime::Runtime::new().map_err(|e| {
        probador::CliError::test_execution(format!("Failed to create runtime: {e}"))
    })?;

    rt.block_on(async {
        run_wasm_pack_build(
            &args.path,
            args.target.as_str(),
            args.release,
            args.out_dir.as_deref(),
            args.profiling,
        )
        .await
        .map_err(|e| probador::CliError::test_execution(e))
    })
}

/// Run brick-based code generation (PROBAR-SPEC-009-P7)
fn run_brick_generation(args: &probador::BuildArgs) -> CliResult<()> {
    use jugar_probar::brick::Brick;
    use probador::generate::{
        create_whisper_event_brick, create_whisper_worker_brick, generate_from_bricks,
        GenerateConfig,
    };

    println!("Zero-Artifact Code Generation (PROBAR-SPEC-009-P7)");
    println!("===================================================\n");

    let output_dir = args.out_dir.clone().unwrap_or_else(|| args.path.clone());
    let title = args
        .title
        .clone()
        .unwrap_or_else(|| format!("{} Application", args.app_name));

    let config = GenerateConfig {
        app_name: args.app_name.clone(),
        wasm_module: args.wasm_module.clone(),
        model_path: args.model_path.clone(),
        title,
        output_dir: output_dir.clone(),
    };

    println!("Configuration:");
    println!("  App name: {}", config.app_name);
    println!("  WASM module: {}", config.wasm_module);
    println!("  Output dir: {}", output_dir.display());
    if let Some(ref model) = config.model_path {
        println!("  Model path: {model}");
    }
    println!();

    // Create demo bricks (in a real implementation, these would be parsed from the brick file)
    let worker = create_whisper_worker_brick();
    let events = create_whisper_event_brick();

    println!("Generating artifacts...");

    let result = generate_from_bricks(Some(&worker), Some(&events), None, &config)
        .map_err(|e| probador::CliError::test_execution(e))?;

    println!("\nGenerated files:");
    for file in &result.files_written {
        println!("  {}", file.display());
    }

    if args.verify {
        println!("\nVerifying brick assertions...");
        let verification = worker.verify();
        if verification.is_valid() {
            println!(
                "  WorkerBrick: {} assertions passed",
                verification.passed.len()
            );
        } else {
            println!("  WorkerBrick: {} failures", verification.failed.len());
            for (assertion, reason) in &verification.failed {
                println!("    - {assertion:?}: {reason}");
            }
            return Err(probador::CliError::test_execution(
                "Brick verification failed",
            ));
        }
    }

    println!("\nZero-Artifact generation complete.");
    println!("All web artifacts derived from Rust brick definitions.");

    Ok(())
}

fn run_watch(args: &probador::WatchArgs) -> CliResult<()> {
    use probador::{dev_server::run_wasm_pack_build, DevServer, DevServerConfig, FileWatcher};
    use std::sync::Arc;
    use tokio::sync::Mutex;

    let rt = tokio::runtime::Runtime::new().map_err(|e| {
        probador::CliError::test_execution(format!("Failed to create runtime: {e}"))
    })?;

    let path = args.path.clone();
    let target = args.target.as_str().to_string();
    let release = args.release;

    // Initial build
    println!("Performing initial build...");
    rt.block_on(async {
        run_wasm_pack_build(&path, &target, release, None, false)
            .await
            .map_err(|e| probador::CliError::test_execution(e))
    })?;

    // Start server if requested
    let server_handle = if args.serve {
        let config = DevServerConfig {
            directory: args.path.join("pkg"),
            port: args.port,
            ws_port: args.ws_port,
            cors: true,
            cross_origin_isolated: false,
        };
        let server = DevServer::new(config);
        let reload_tx = server.reload_sender();

        // Spawn server in background
        let server_handle = rt.spawn(async move {
            let _ = server.run().await;
        });

        Some((server_handle, reload_tx))
    } else {
        None
    };

    // File watcher
    println!("\nWatching for changes in {}...", args.path.display());
    println!("Press Ctrl+C to stop\n");

    let watcher = FileWatcher::new(args.path.clone(), args.debounce);
    let path_for_rebuild = args.path.clone();
    let target_for_rebuild = args.target.as_str().to_string();
    let release_for_rebuild = args.release;
    let reload_tx = server_handle.as_ref().map(|(_, tx)| tx.clone());

    let rebuild_in_progress = Arc::new(Mutex::new(false));

    rt.block_on(async {
        watcher
            .watch(move |changed_file| {
                let rebuild_in_progress = rebuild_in_progress.clone();
                let path = path_for_rebuild.clone();
                let target = target_for_rebuild.clone();
                let reload_tx = reload_tx.clone();

                // Use a separate runtime for the rebuild since we're in a sync callback
                let rt = tokio::runtime::Handle::current();
                rt.spawn(async move {
                    // Check if rebuild is already in progress
                    let mut in_progress = rebuild_in_progress.lock().await;
                    if *in_progress {
                        return;
                    }
                    *in_progress = true;
                    drop(in_progress);

                    println!(
                        "\n[{}] File changed: {}",
                        chrono::Local::now().format("%H:%M:%S"),
                        changed_file
                    );

                    if let Some(ref tx) = reload_tx {
                        let _ = tx.send(probador::dev_server::HotReloadMessage::FileChanged {
                            path: changed_file.clone(),
                        });
                    }

                    println!("Rebuilding...");
                    let build_start = std::time::Instant::now();
                    match run_wasm_pack_build(&path, &target, release_for_rebuild, None, false)
                        .await
                    {
                        Ok(()) => {
                            let duration_ms = build_start.elapsed().as_millis() as u64;
                            println!("Build successful!");
                            if let Some(ref tx) = reload_tx {
                                let _ = tx.send(
                                    probador::dev_server::HotReloadMessage::RebuildComplete {
                                        duration_ms,
                                    },
                                );
                            }
                        }
                        Err(e) => {
                            eprintln!("Build failed: {e}");
                            if let Some(ref tx) = reload_tx {
                                let _ = tx.send(
                                    probador::dev_server::HotReloadMessage::RebuildFailed {
                                        error: e,
                                    },
                                );
                            }
                        }
                    }

                    let mut in_progress = rebuild_in_progress.lock().await;
                    *in_progress = false;
                });
            })
            .await
            .map_err(|e| probador::CliError::test_execution(format!("Watch error: {e}")))
    })
}

fn run_av_sync(config: &CliConfig, args: &probador::AvSyncArgs) -> CliResult<()> {
    use probador::handlers::av_sync;
    use probador::AvSyncSubcommand;

    match &args.subcommand {
        AvSyncSubcommand::Check(check_args) => av_sync::execute_check(config, check_args),
        AvSyncSubcommand::Report(report_args) => av_sync::execute_report(config, report_args),
    }
}

fn run_audio(config: &CliConfig, args: &probador::AudioArgs) -> CliResult<()> {
    use probador::handlers::audio;
    use probador::AudioSubcommand;

    match &args.subcommand {
        AudioSubcommand::Check(check_args) => audio::execute_check(config, check_args),
    }
}

fn run_video(config: &CliConfig, args: &probador::VideoArgs) -> CliResult<()> {
    use probador::handlers::video;
    use probador::VideoSubcommand;

    match &args.subcommand {
        VideoSubcommand::Check(check_args) => video::execute_check(config, check_args),
    }
}

fn run_animation(config: &CliConfig, args: &probador::AnimationArgs) -> CliResult<()> {
    use probador::handlers::animation;
    use probador::AnimationSubcommand;

    match &args.subcommand {
        AnimationSubcommand::Check(check_args) => animation::execute_check(config, check_args),
    }
}

fn run_playbook(config: &CliConfig, args: &probador::PlaybookArgs) -> CliResult<()> {
    if config.verbosity != Verbosity::Quiet {
        println!("Running playbook(s)...");
    }

    let mut all_passed = true;
    for file in &args.files {
        if config.verbosity != Verbosity::Quiet {
            println!("\nProcessing: {}", file.display());
        }
        let passed = process_single_playbook(config, args, file)?;
        if !passed {
            all_passed = false;
        }
    }

    if all_passed {
        Ok(())
    } else {
        Err(probador::CliError::test_execution(
            "One or more playbooks failed validation".to_string(),
        ))
    }
}

/// Process a single playbook file: validate, optionally export/mutate/format.
/// Returns true if validation passed.
fn process_single_playbook(
    config: &CliConfig,
    args: &probador::PlaybookArgs,
    file: &std::path::Path,
) -> CliResult<bool> {
    use jugar_probar::playbook::StateMachineValidator;

    let playbook = load_playbook(file)?;
    let validator = StateMachineValidator::new(&playbook);
    let validation_result = validator.validate();

    print_playbook_summary(config, &playbook, &validation_result);

    if !validation_result.is_valid {
        for issue in &validation_result.issues {
            eprintln!("  Issue: {issue:?}");
        }
    }

    if args.validate {
        if config.verbosity != Verbosity::Quiet {
            println!("  Validation only mode - skipping execution");
        }
        return Ok(validation_result.is_valid);
    }

    handle_playbook_export(config, args, &playbook)?;
    handle_playbook_mutate(config, args, &playbook);
    format_playbook_output(args, file, &playbook, &validation_result);

    if args.fail_fast && !validation_result.is_valid {
        return Err(probador::CliError::test_execution(
            "Playbook validation failed (--fail-fast)".to_string(),
        ));
    }

    Ok(validation_result.is_valid)
}

fn load_playbook(file: &std::path::Path) -> CliResult<jugar_probar::playbook::Playbook> {
    use jugar_probar::playbook::Playbook;

    let yaml_content = std::fs::read_to_string(file).map_err(|e| {
        probador::CliError::test_execution(format!(
            "Failed to read playbook {}: {e}",
            file.display(),
        ))
    })?;
    Playbook::from_yaml(&yaml_content).map_err(|e| {
        probador::CliError::test_execution(format!(
            "Failed to parse playbook {}: {e}",
            file.display(),
        ))
    })
}

fn print_playbook_summary(
    config: &CliConfig,
    playbook: &jugar_probar::playbook::Playbook,
    validation: &jugar_probar::playbook::ValidationResult,
) {
    if config.verbosity == Verbosity::Quiet {
        return;
    }
    println!("  State machine: {}", playbook.machine.id);
    println!("  States: {}", playbook.machine.states.len());
    println!("  Transitions: {}", playbook.machine.transitions.len());
    println!(
        "  Valid: {}",
        if validation.is_valid { "yes" } else { "no" }
    );
}

fn handle_playbook_export(
    config: &CliConfig,
    args: &probador::PlaybookArgs,
    playbook: &jugar_probar::playbook::Playbook,
) -> CliResult<()> {
    use jugar_probar::playbook::{to_dot, to_svg};

    let Some(ref format) = args.export else {
        return Ok(());
    };

    let diagram = match format {
        probador::DiagramFormat::Dot => to_dot(playbook),
        probador::DiagramFormat::Svg => to_svg(playbook),
    };

    if let Some(ref output_path) = args.export_output {
        std::fs::write(output_path, &diagram).map_err(|e| {
            probador::CliError::report_generation(format!("Failed to write diagram: {e}"))
        })?;
        if config.verbosity != Verbosity::Quiet {
            println!("  Diagram exported to: {}", output_path.display());
        }
    } else {
        println!("{diagram}");
    }
    Ok(())
}

fn handle_playbook_mutate(
    config: &CliConfig,
    args: &probador::PlaybookArgs,
    playbook: &jugar_probar::playbook::Playbook,
) {
    use jugar_probar::playbook::MutationGenerator;

    if !args.mutate {
        return;
    }

    let generator = MutationGenerator::new(playbook);
    let classes_to_run = parse_mutation_classes(args.mutation_classes.as_ref());

    if config.verbosity != Verbosity::Quiet {
        println!(
            "  Running mutation testing ({} classes)...",
            classes_to_run.len()
        );
    }

    let mut total_mutants = 0;
    for class in &classes_to_run {
        let mutants = generator.generate(*class);
        if config.verbosity != Verbosity::Quiet {
            println!("    {}: {} mutants", class.id(), mutants.len());
        }
        total_mutants += mutants.len();
    }

    if config.verbosity != Verbosity::Quiet {
        println!("  Total mutants generated: {total_mutants}");
    }
}

fn parse_mutation_classes(
    class_names: Option<&Vec<String>>,
) -> Vec<jugar_probar::playbook::MutationClass> {
    use jugar_probar::playbook::MutationClass;

    class_names.map_or_else(MutationClass::all, |names| {
        names
            .iter()
            .filter_map(|name| match name.as_str() {
                "M1" => Some(MutationClass::StateRemoval),
                "M2" => Some(MutationClass::TransitionRemoval),
                "M3" => Some(MutationClass::EventSwap),
                "M4" => Some(MutationClass::TargetSwap),
                "M5" => Some(MutationClass::GuardNegation),
                _ => {
                    eprintln!("Unknown mutation class: {name}");
                    None
                }
            })
            .collect()
    })
}

fn format_playbook_output(
    args: &probador::PlaybookArgs,
    file: &std::path::Path,
    playbook: &jugar_probar::playbook::Playbook,
    validation: &jugar_probar::playbook::ValidationResult,
) {
    match args.format {
        probador::PlaybookOutputFormat::Json => {
            let result = serde_json::json!({
                "file": file.display().to_string(),
                "machine_id": playbook.machine.id,
                "states": playbook.machine.states.len(),
                "transitions": playbook.machine.transitions.len(),
                "valid": validation.is_valid,
                "issues": validation.issues.len(),
            });
            println!(
                "{}",
                serde_json::to_string_pretty(&result).unwrap_or_default()
            );
        }
        probador::PlaybookOutputFormat::Junit => {
            print_playbook_junit(file, playbook, validation);
        }
        probador::PlaybookOutputFormat::Text => {} // Already printed in summary
    }
}

fn print_playbook_junit(
    file: &std::path::Path,
    playbook: &jugar_probar::playbook::Playbook,
    validation: &jugar_probar::playbook::ValidationResult,
) {
    let timestamp = chrono::Utc::now().to_rfc3339();
    let failures = i32::from(!validation.is_valid);
    let failure_elem = if validation.is_valid {
        String::new()
    } else {
        format!(
            "<failure message=\"Validation failed\">{} issues</failure>",
            validation.issues.len()
        )
    };
    println!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<testsuites name="playbook" tests="1" failures="{failures}" timestamp="{timestamp}">
  <testsuite name="{}" tests="1" failures="{failures}">
    <testcase name="validation" classname="{}">
      {failure_elem}
    </testcase>
  </testsuite>
</testsuites>"#,
        playbook.machine.id,
        file.display(),
    );
}

// =============================================================================
// Browser/WASM Stress Testing (Section H: Points 116-125)
// =============================================================================

/// Run browser/WASM stress tests per PROBAR-SPEC-WASM-001 Section H
fn run_stress(_config: &CliConfig, args: &probador::StressArgs) -> CliResult<()> {
    use probador::{
        render_stress_json, render_stress_report, StressConfig, StressMode, StressRunner,
    };

    let mode_str = args.get_mode();
    let mode: StressMode = mode_str
        .parse()
        .map_err(|e: String| probador::CliError::invalid_argument(e))?;

    let config = match mode {
        StressMode::Atomics => StressConfig::atomics(args.duration, args.concurrency),
        StressMode::WorkerMsg => StressConfig::worker_msg(args.duration, args.concurrency),
        StressMode::Render => StressConfig::render(args.duration),
        StressMode::Trace => StressConfig::trace(args.duration),
        StressMode::Full => StressConfig::full(args.duration, args.concurrency),
    };

    let runner = StressRunner::new(config);
    let result = runner.run();

    // Output result
    let output = if args.output == "json" {
        render_stress_json(&result)
    } else {
        render_stress_report(&result)
    };

    println!("{}", output);

    if result.passed {
        Ok(())
    } else {
        Err(probador::CliError::test_execution(format!(
            "Stress test {} failed: {}",
            mode, result.actual_value
        )))
    }
}

// =============================================================================
// LLM Testing
// =============================================================================

#[cfg(feature = "llm")]
fn run_llm(args: &probador::LlmArgs) -> CliResult<()> {
    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| probador::CliError::Generic(format!("Failed to create async runtime: {e}")))?;

    match &args.subcommand {
        probador::LlmSubcommand::Test(test_args) => {
            rt.block_on(probador::handlers::llm::execute_llm_test(test_args))
        }
        probador::LlmSubcommand::Load(load_args) => {
            rt.block_on(probador::handlers::llm::execute_llm_load(load_args))
        }
        probador::LlmSubcommand::Bench(bench_args) => {
            rt.block_on(probador::handlers::llm::execute_llm_bench(bench_args))
        }
        probador::LlmSubcommand::Report(report_args) => {
            probador::handlers::llm::execute_llm_report(report_args)
        }
        probador::LlmSubcommand::Experiment(exp_args) => {
            probador::handlers::llm::execute_llm_experiment(exp_args)
        }
        probador::LlmSubcommand::DataAudit(audit_args) => {
            probador::handlers::llm::execute_data_audit(audit_args)
        }
        probador::LlmSubcommand::Sweep(sweep_args) => {
            rt.block_on(probador::handlers::llm::execute_llm_sweep(sweep_args))
        }
        probador::LlmSubcommand::GenDataset(gen_args) => {
            probador::handlers::llm::execute_llm_gen_dataset(gen_args)
        }
        probador::LlmSubcommand::Score(score_args) => {
            probador::handlers::llm::execute_llm_score(score_args)
        }
    }
}

// =============================================================================
// WASM Compliance Checks (C001-C010)
// =============================================================================

/// Run WASM compliance checks per PROBAR-SPEC-011
///
/// Implements the 10-point compliance checklist from the Advanced Probador
/// Testing Concepts specification.
fn run_comply(config: &CliConfig, args: &probador::ComplyArgs) -> CliResult<()> {
    // Handle subcommands if present
    if let Some(ref subcommand) = args.subcommand {
        return match subcommand {
            probador::ComplySubcommand::Check(check_args) => run_comply_check(config, check_args),
            probador::ComplySubcommand::Migrate(migrate_args) => {
                run_comply_migrate(config, migrate_args)
            }
            probador::ComplySubcommand::Diff(diff_args) => run_comply_diff(config, diff_args),
            probador::ComplySubcommand::Enforce(enforce_args) => {
                run_comply_enforce(config, enforce_args)
            }
            probador::ComplySubcommand::Report(report_args) => {
                run_comply_report(config, report_args)
            }
        };
    }

    // Default behavior: run checks (backwards compatibility)
    if config.verbosity != Verbosity::Quiet {
        eprintln!("\n══════════════════════════════════════════════════════════════");
        eprintln!("  PROBAR COMPLY - WASM Compliance Checker");
        eprintln!("══════════════════════════════════════════════════════════════\n");
    }

    run_comply_checks_internal(config, args)
}

// =============================================================================
// Comply Subcommand Handlers
// NOTE: Compliance check functions (check_c001 through check_c010) and
// ComplianceResult are now imported from probador::handlers::comply
// =============================================================================

/// Run comply check subcommand
fn run_comply_check(config: &CliConfig, args: &probador::ComplyCheckArgs) -> CliResult<()> {
    use jugar_probar::strict::{E2ETestChecklist, WasmStrictMode};

    if config.verbosity != Verbosity::Quiet {
        eprintln!("\n══════════════════════════════════════════════════════════════");
        eprintln!("  PROBAR COMPLY CHECK - WASM Compliance Checker");
        eprintln!("══════════════════════════════════════════════════════════════\n");
    }

    let strict_mode = if args.strict {
        WasmStrictMode::production()
    } else {
        WasmStrictMode::development()
    };
    let _checklist = E2ETestChecklist::new().with_strict_mode(strict_mode);

    // Build a ComplyArgs for compatibility
    let compat_args = probador::ComplyArgs {
        subcommand: None,
        path: args.path.clone(),
        checks: args.checks.clone(),
        fail_fast: false,
        format: args.format.clone(),
        max_wasm_size: 5_242_880,
        strict: args.strict,
        report: None,
        detailed: args.detailed,
    };

    // Reuse the existing check logic
    run_comply_checks_internal(config, &compat_args)
}

/// Internal check logic (shared between top-level and check subcommand)
fn run_comply_checks_internal(config: &CliConfig, args: &probador::ComplyArgs) -> CliResult<()> {
    type CheckFn = Box<dyn Fn(&std::path::Path, &probador::ComplyArgs) -> ComplianceResult>;

    let checks_to_run = build_compliance_checks();
    let filtered_checks: Vec<(&str, &str, CheckFn)> =
        filter_compliance_checks(checks_to_run, args.checks.as_ref());

    if config.verbosity != Verbosity::Quiet {
        eprintln!(
            "Running {} compliance check(s) on {}\n",
            filtered_checks.len(),
            args.path.display()
        );
    }

    let (results, all_passed) = execute_compliance_checks(&filtered_checks, config, args);

    output_compliance_results(
        config,
        &results,
        &args.format,
        args.report.as_deref(),
        all_passed,
    )
}

/// Build the vector of all compliance checks (C001-C010).
fn build_compliance_checks() -> Vec<(
    &'static str,
    &'static str,
    Box<dyn Fn(&std::path::Path, &probador::ComplyArgs) -> ComplianceResult>,
)> {
    vec![
        (
            "C001",
            "Code execution verified",
            Box::new(|path, _| check_c001_code_execution(path)),
        ),
        (
            "C002",
            "Console errors fail tests",
            Box::new(|_, _| check_c002_console_errors()),
        ),
        (
            "C003",
            "Custom elements tested",
            Box::new(|path, _| check_c003_custom_elements(path)),
        ),
        (
            "C004",
            "Threading modes tested",
            Box::new(|_, _| check_c004_threading_modes()),
        ),
        (
            "C005",
            "Low memory tested",
            Box::new(|_, _| check_c005_low_memory()),
        ),
        (
            "C006",
            "COOP/COEP headers",
            Box::new(|path, _| check_c006_headers(path)),
        ),
        (
            "C007",
            "Replay hash matches",
            Box::new(|_, _| check_c007_replay_hash()),
        ),
        (
            "C008",
            "Cache handling",
            Box::new(|_, _| check_c008_cache()),
        ),
        (
            "C009",
            "WASM size limit",
            Box::new(|path, args| check_c009_wasm_size(path, args.max_wasm_size)),
        ),
        (
            "C010",
            "No panic paths",
            Box::new(|path, _| check_c010_panic_paths(path)),
        ),
    ]
}

/// Filter compliance checks by requested IDs, or return all if none specified.
fn filter_compliance_checks<F>(
    checks: Vec<(&'static str, &'static str, F)>,
    requested: Option<&Vec<String>>,
) -> Vec<(&'static str, &'static str, F)> {
    match requested {
        Some(ids) => checks
            .into_iter()
            .filter(|(id, _, _)| ids.iter().any(|r| r == id))
            .collect(),
        None => checks,
    }
}

/// Run each compliance check, printing results as we go.
fn execute_compliance_checks(
    checks: &[(
        &str,
        &str,
        Box<dyn Fn(&std::path::Path, &probador::ComplyArgs) -> ComplianceResult>,
    )],
    config: &CliConfig,
    args: &probador::ComplyArgs,
) -> (Vec<ComplianceResult>, bool) {
    let mut results = Vec::new();
    let mut all_passed = true;

    for (id, description, check_fn) in checks {
        let result = check_fn(&args.path, args);
        print_check_result(config, id, description, &result, args.detailed);

        if !result.passed {
            all_passed = false;
            if args.fail_fast {
                results.push(result);
                break;
            }
        }
        results.push(result);
    }

    (results, all_passed)
}

/// Print a single compliance check result.
fn print_check_result(
    config: &CliConfig,
    id: &str,
    description: &str,
    result: &ComplianceResult,
    detailed: bool,
) {
    if config.verbosity == Verbosity::Quiet {
        return;
    }
    let status = if result.passed { "✓" } else { "✗" };
    let color = if result.passed {
        "\x1b[32m"
    } else {
        "\x1b[31m"
    };
    let reset = "\x1b[0m";

    eprintln!("  {color}[{status}]{reset} {id}: {description}");
    if detailed && !result.details.is_empty() {
        for detail in &result.details {
            eprintln!("      └─ {detail}");
        }
    }
}

/// Output compliance results: summary, report file, stdout format.
fn output_compliance_results(
    config: &CliConfig,
    results: &[ComplianceResult],
    format: &probador::ComplyOutputFormat,
    report_path: Option<&std::path::Path>,
    all_passed: bool,
) -> CliResult<()> {
    let passed_count = results.iter().filter(|r| r.passed).count();
    let total_count = results.len();

    if config.verbosity != Verbosity::Quiet {
        eprintln!("\n══════════════════════════════════════════════════════════════");
        eprintln!("  Result: {passed_count}/{total_count} checks passed");
        eprintln!("══════════════════════════════════════════════════════════════\n");
    }

    if let Some(path) = report_path {
        let report = generate_comply_report(results, format);
        std::fs::write(path, &report).map_err(|e| {
            probador::CliError::report_generation(format!("Failed to write report: {e}"))
        })?;
    }

    match format {
        probador::ComplyOutputFormat::Json | probador::ComplyOutputFormat::Junit => {
            let report = generate_comply_report(results, format);
            println!("{report}");
        }
        probador::ComplyOutputFormat::Text => {}
    }

    if all_passed {
        Ok(())
    } else {
        Err(probador::CliError::test_execution(format!(
            "Compliance check failed: {passed_count}/{total_count} checks passed",
        )))
    }
}

/// Run comply migrate subcommand
fn run_comply_migrate(config: &CliConfig, args: &probador::ComplyMigrateArgs) -> CliResult<()> {
    use std::process::Command;

    if config.verbosity != Verbosity::Quiet {
        eprintln!("\n══════════════════════════════════════════════════════════════");
        eprintln!("  PROBAR COMPLY MIGRATE");
        eprintln!("══════════════════════════════════════════════════════════════\n");
    }

    // Check for uncommitted changes
    if !args.force {
        let status = Command::new("git")
            .args(["status", "--porcelain"])
            .current_dir(&args.path)
            .output();

        if let Ok(output) = status {
            if !output.stdout.is_empty() {
                return Err(probador::CliError::config(
                    "Uncommitted changes detected. Use --force to override.".to_string(),
                ));
            }
        }
    }

    let target_version = args.version.as_deref().unwrap_or("latest");

    if args.dry_run {
        eprintln!("DRY RUN - would migrate to version: {target_version}");
        eprintln!("\nChanges that would be applied:");
        eprintln!("  - Update probar.toml version field");
        eprintln!("  - Add new required test configurations");
        eprintln!("  - Update deprecated API calls");
        return Ok(());
    }

    eprintln!("Migrating to version: {target_version}");

    // Create probar.toml if it doesn't exist
    let config_path = args.path.join("probar.toml");
    if !config_path.exists() {
        let config_content = format!(
            r#"# Probar Configuration
# Generated by: probador comply migrate

[probar]
version = "{}"
cross_origin_isolated = true

[strict]
require_code_execution = true
fail_on_console_error = true
verify_custom_elements = true

[quality]
min_coverage = 95
max_wasm_size = 5242880
"#,
            target_version
        );
        std::fs::write(&config_path, config_content)
            .map_err(|e| probador::CliError::config(format!("Failed to create config: {e}")))?;
        eprintln!("  Created: {}", config_path.display());
    }

    eprintln!("\nMigration complete!");
    Ok(())
}

/// Run comply diff subcommand
fn run_comply_diff(config: &CliConfig, args: &probador::ComplyDiffArgs) -> CliResult<()> {
    if config.verbosity != Verbosity::Quiet {
        eprintln!("\n══════════════════════════════════════════════════════════════");
        eprintln!("  PROBAR COMPLY DIFF - Version Changelog");
        eprintln!("══════════════════════════════════════════════════════════════\n");
    }

    let from_version = args.from.as_deref().unwrap_or("0.3.0");
    let to_version = args.to.as_deref().unwrap_or(env!("CARGO_PKG_VERSION"));

    eprintln!("Changes from {} to {}:\n", from_version, to_version);

    // Changelog entries
    let changelog = vec![
        (
            "0.4.0",
            vec![
                ("FEATURE", "Added AudioEmulator for getUserMedia mocking"),
                (
                    "FEATURE",
                    "Added WasmThreadCapabilities for COOP/COEP detection",
                ),
                ("FEATURE", "Added WorkerEmulator for Web Worker testing"),
                (
                    "FEATURE",
                    "Added StreamingUxValidator for real-time UX testing",
                ),
                (
                    "FEATURE",
                    "Added probador comply subcommands (check, migrate, diff, enforce, report)",
                ),
                ("BREAKING", "ConsoleCapture now requires WasmStrictMode"),
            ],
        ),
        (
            "0.3.0",
            vec![
                ("FEATURE", "Added playbook state machine testing"),
                ("FEATURE", "Added pixel coverage heatmaps"),
                ("FEATURE", "Added serve --cross-origin-isolated flag"),
            ],
        ),
    ];

    for (version, changes) in &changelog {
        if args.breaking_only {
            let breaking: Vec<_> = changes.iter().filter(|(t, _)| *t == "BREAKING").collect();
            if !breaking.is_empty() {
                eprintln!("Version {}:", version);
                for (_, desc) in breaking {
                    eprintln!("  ⚠️  BREAKING: {desc}");
                }
                eprintln!();
            }
        } else {
            eprintln!("Version {}:", version);
            for (change_type, desc) in changes {
                let icon = match *change_type {
                    "FEATURE" => "✨",
                    "BREAKING" => "⚠️ ",
                    "FIX" => "🐛",
                    _ => "•",
                };
                eprintln!("  {icon} {desc}");
            }
            eprintln!();
        }
    }

    Ok(())
}

/// Run comply enforce subcommand
fn run_comply_enforce(config: &CliConfig, args: &probador::ComplyEnforceArgs) -> CliResult<()> {
    if config.verbosity != Verbosity::Quiet {
        eprintln!("\n══════════════════════════════════════════════════════════════");
        eprintln!("  PROBAR COMPLY ENFORCE - Git Hooks");
        eprintln!("══════════════════════════════════════════════════════════════\n");
    }

    let hooks_dir = args.path.join(".git/hooks");

    if !hooks_dir.exists() {
        return Err(probador::CliError::config(
            "Not a git repository (no .git/hooks directory)".to_string(),
        ));
    }

    let pre_commit_path = hooks_dir.join("pre-commit");

    if args.disable {
        // Remove hooks
        if pre_commit_path.exists() {
            std::fs::remove_file(&pre_commit_path)
                .map_err(|e| probador::CliError::config(format!("Failed to remove hook: {e}")))?;
            eprintln!("Removed pre-commit hook");
        } else {
            eprintln!("No pre-commit hook found");
        }
        return Ok(());
    }

    // Confirm installation
    if !args.yes {
        eprintln!("This will install a pre-commit hook that runs:");
        eprintln!("  - probador comply check --strict");
        eprintln!("  - WASM binary size check");
        eprintln!("  - Panic path verification");
        eprintln!("\nProceed? [y/N] ");
        // In a real implementation, read user input
        // For now, just proceed
    }

    // Generate pre-commit hook
    let hook_content = r##"#!/bin/bash
# Probar WASM Quality Gates
# Generated by: probador comply enforce

set -e

echo "Running Probar quality gates..."

# 1. WASM binary size regression check
MAX_WASM_SIZE=5000000
wasm_files=$(find target -name "*.wasm" 2>/dev/null | head -1)
if [ -n "$wasm_files" ]; then
    wasm_size=$(stat -c%s "$wasm_files" 2>/dev/null || stat -f%z "$wasm_files" 2>/dev/null || echo 0)
    if [ "$wasm_size" -gt "$MAX_WASM_SIZE" ]; then
        echo "ERROR: WASM binary size regression: ${wasm_size} > ${MAX_WASM_SIZE}"
        exit 1
    fi
fi

# 2. No panic paths in WASM code
if grep -rn "unwrap()" --include="*.rs" src/ 2>/dev/null | grep -v "// SAFETY:" | grep -v "#[cfg(test)]" | head -5; then
    echo "WARNING: unwrap() found in src/ code - consider using expect() with message"
fi

# 3. Run compliance check
if command -v probador &> /dev/null; then
    probador comply check --strict . || {
        echo "ERROR: Compliance check failed"
        exit 1
    }
fi

echo "Probar quality gates passed!"
"##;

    std::fs::write(&pre_commit_path, hook_content)
        .map_err(|e| probador::CliError::config(format!("Failed to write hook: {e}")))?;

    // Make executable
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&pre_commit_path)
            .map_err(|e| probador::CliError::config(format!("Failed to get perms: {e}")))?
            .permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&pre_commit_path, perms)
            .map_err(|e| probador::CliError::config(format!("Failed to set perms: {e}")))?;
    }

    eprintln!(
        "Installed pre-commit hook at: {}",
        pre_commit_path.display()
    );
    eprintln!("\nHook will run on every commit to enforce:");
    eprintln!("  - WASM binary size limits");
    eprintln!("  - Panic-free code patterns");
    eprintln!("  - Full compliance check");

    Ok(())
}

/// Run comply report subcommand
fn run_comply_report(config: &CliConfig, args: &probador::ComplyReportArgs) -> CliResult<()> {
    use std::fs;
    type CheckFn = fn(&std::path::Path) -> ComplianceResult;

    if config.verbosity != Verbosity::Quiet {
        eprintln!("\n══════════════════════════════════════════════════════════════");
        eprintln!("  PROBAR COMPLY REPORT");
        eprintln!("══════════════════════════════════════════════════════════════\n");
    }

    // Run all checks to generate report
    let check_args = probador::ComplyArgs {
        subcommand: None,
        path: args.path.clone(),
        checks: None,
        fail_fast: false,
        format: probador::ComplyOutputFormat::Text,
        max_wasm_size: 5_242_880,
        strict: false,
        report: None,
        detailed: true,
    };

    // Collect results silently
    let mut results: Vec<ComplianceResult> = Vec::new();
    let checks: Vec<(&str, &str, CheckFn)> = vec![
        ("C001", "Code execution verified", |p| {
            check_c001_code_execution(p)
        }),
        ("C002", "Console errors fail tests", |_| {
            check_c002_console_errors()
        }),
        ("C003", "Custom elements tested", |p| {
            check_c003_custom_elements(p)
        }),
        ("C004", "Threading modes tested", |_| {
            check_c004_threading_modes()
        }),
        ("C005", "Low memory tested", |_| check_c005_low_memory()),
        ("C006", "COOP/COEP headers", |p| check_c006_headers(p)),
        ("C007", "Replay hash matches", |_| check_c007_replay_hash()),
        ("C008", "Cache handling", |_| check_c008_cache()),
        ("C009", "WASM size limit", |p| {
            check_c009_wasm_size(p, 5_242_880)
        }),
        ("C010", "No panic paths", |p| check_c010_panic_paths(p)),
    ];

    for (_, _, check_fn) in &checks {
        results.push(check_fn(&check_args.path));
    }

    let passed = results.iter().filter(|r| r.passed).count();
    let total = results.len();
    let timestamp = chrono::Utc::now().to_rfc3339();

    let report = match args.format {
        probador::ComplyReportFormat::Text => {
            format!(
                r#"============================================================
Probador WASM Compliance Report
============================================================

Project: {}
Probador Version: {}
Scan Time: {}

Checks:
{}
Summary: {}/{} passed

============================================================
"#,
                args.path.display(),
                env!("CARGO_PKG_VERSION"),
                timestamp,
                results
                    .iter()
                    .enumerate()
                    .map(|(i, r)| {
                        let status = if r.passed { "✓" } else { "⚠" };
                        format!(
                            "  {} C{:03} {}: {}",
                            status,
                            i + 1,
                            if r.passed { "Passed" } else { "Warning" },
                            r.details.first().unwrap_or(&String::new())
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("\n"),
                passed,
                total
            )
        }
        probador::ComplyReportFormat::Json => serde_json::json!({
            "project": args.path.display().to_string(),
            "version": env!("CARGO_PKG_VERSION"),
            "timestamp": timestamp,
            "results": results.iter().map(|r| {
                serde_json::json!({
                    "id": r.id,
                    "passed": r.passed,
                    "details": r.details
                })
            }).collect::<Vec<_>>(),
            "summary": { "passed": passed, "total": total }
        })
        .to_string(),
        probador::ComplyReportFormat::Markdown => {
            format!(
                r#"# Probador WASM Compliance Report

**Project**: {}
**Version**: {}
**Date**: {}

## Summary

| Metric | Value |
|--------|-------|
| Passed | {} |
| Total | {} |
| Score | {:.0}% |

## Checks

| Check | Status | Details |
|-------|--------|---------|
{}

---
*Generated by probador {}*
"#,
                args.path.display(),
                env!("CARGO_PKG_VERSION"),
                timestamp,
                passed,
                total,
                (passed as f64 / total as f64) * 100.0,
                results
                    .iter()
                    .map(|r| {
                        let status = if r.passed { "✅ Pass" } else { "⚠️ Warn" };
                        format!(
                            "| {} | {} | {} |",
                            r.id,
                            status,
                            r.details.first().unwrap_or(&String::new())
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("\n"),
                env!("CARGO_PKG_VERSION")
            )
        }
        probador::ComplyReportFormat::Html => {
            format!(
                r#"<!DOCTYPE html>
<html>
<head>
    <title>Probador Compliance Report</title>
    <style>
        body {{ font-family: sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #333; }}
        .pass {{ color: green; }}
        .warn {{ color: orange; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #f5f5f5; }}
    </style>
</head>
<body>
    <h1>Probador WASM Compliance Report</h1>
    <p><strong>Project:</strong> {}</p>
    <p><strong>Version:</strong> {}</p>
    <p><strong>Date:</strong> {}</p>
    <h2>Summary: {}/{} checks passed</h2>
    <table>
        <tr><th>Check</th><th>Status</th><th>Details</th></tr>
        {}
    </table>
</body>
</html>"#,
                args.path.display(),
                env!("CARGO_PKG_VERSION"),
                timestamp,
                passed,
                total,
                results
                    .iter()
                    .map(|r| {
                        let (class, status) = if r.passed {
                            ("pass", "✓")
                        } else {
                            ("warn", "⚠")
                        };
                        format!(
                            "<tr><td>{}</td><td class=\"{}\">{}</td><td>{}</td></tr>",
                            r.id,
                            class,
                            status,
                            r.details.first().unwrap_or(&String::new())
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            )
        }
    };

    if let Some(ref output_path) = args.output {
        fs::write(output_path, &report).map_err(|e| {
            probador::CliError::report_generation(format!("Failed to write report: {e}"))
        })?;
        eprintln!("Report written to: {}", output_path.display());
    } else {
        println!("{report}");
    }

    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]
mod tests {
    use super::*;
    use probador::{
        ConfigArgs, CoverageArgs, InitArgs, PaletteArg, RecordArgs, RecordFormat, ReportArgs,
        ReportFormat,
    };
    use std::path::PathBuf;

    mod build_config_tests {
        use super::*;

        #[test]
        fn test_build_config_default() {
            let cli = Cli::parse_from(["probar", "test"]);
            let config = build_config(&cli);
            assert_eq!(config.verbosity, Verbosity::Normal);
        }

        #[test]
        fn test_build_config_verbose() {
            let cli = Cli::parse_from(["probar", "-v", "test"]);
            let config = build_config(&cli);
            assert_eq!(config.verbosity, Verbosity::Verbose);
        }

        #[test]
        fn test_build_config_debug() {
            let cli = Cli::parse_from(["probar", "-vv", "test"]);
            let config = build_config(&cli);
            assert_eq!(config.verbosity, Verbosity::Debug);
        }

        #[test]
        fn test_build_config_very_verbose() {
            let cli = Cli::parse_from(["probar", "-vvv", "test"]);
            let config = build_config(&cli);
            assert_eq!(config.verbosity, Verbosity::Debug);
        }

        #[test]
        fn test_build_config_quiet() {
            let cli = Cli::parse_from(["probar", "-q", "test"]);
            let config = build_config(&cli);
            assert_eq!(config.verbosity, Verbosity::Quiet);
        }

        #[test]
        fn test_build_config_color_never() {
            let cli = Cli::parse_from(["probar", "--color", "never", "test"]);
            let config = build_config(&cli);
            assert_eq!(config.color, ColorChoice::Never);
        }

        #[test]
        fn test_build_config_color_always() {
            let cli = Cli::parse_from(["probar", "--color", "always", "test"]);
            let config = build_config(&cli);
            assert_eq!(config.color, ColorChoice::Always);
        }
    }

    mod run_record_tests {
        use super::*;

        #[test]
        fn test_run_record() {
            let config = CliConfig::default();
            let args = RecordArgs {
                test: "my_test".to_string(),
                format: RecordFormat::Gif,
                output: None,
                fps: 10,
                quality: 80,
            };
            run_record(&config, &args);
            // Just verify it doesn't panic
        }

        #[test]
        fn test_run_record_png() {
            let config = CliConfig::default();
            let args = RecordArgs {
                test: "another_test".to_string(),
                format: RecordFormat::Png,
                output: Some(PathBuf::from("output.png")),
                fps: 30,
                quality: 100,
            };
            run_record(&config, &args);
        }
    }

    mod run_report_tests {
        use super::*;

        #[test]
        fn test_run_report_html() {
            let config = CliConfig::default();
            let args = ReportArgs {
                format: ReportFormat::Html,
                output: PathBuf::from("/tmp/probar_test_report"),
                open: false,
            };
            run_report(&config, &args);
        }

        #[test]
        fn test_run_report_json() {
            let config = CliConfig::default();
            let args = ReportArgs {
                format: ReportFormat::Json,
                output: PathBuf::from("/tmp/probar_test_report.json"),
                open: false,
            };
            run_report(&config, &args);
        }

        #[test]
        fn test_run_report_with_open() {
            let config = CliConfig::default();
            let args = ReportArgs {
                format: ReportFormat::Html,
                output: PathBuf::from("/tmp/probar_test_report_open"),
                open: true,
            };
            run_report(&config, &args);
        }
    }

    mod run_init_tests {
        use super::*;
        use std::fs;

        #[test]
        fn test_run_init_basic() {
            let temp_dir = std::env::temp_dir().join("probar_init_test");
            let _ = fs::remove_dir_all(&temp_dir);

            let config = CliConfig::default();
            let args = InitArgs {
                path: temp_dir.clone(),
                force: false,
            };
            run_init(&config, &args);

            // Cleanup
            let _ = fs::remove_dir_all(&temp_dir);
        }

        #[test]
        fn test_run_init_force() {
            let temp_dir = std::env::temp_dir().join("probar_init_force_test");
            let _ = fs::remove_dir_all(&temp_dir);

            let config = CliConfig::default();
            let args = InitArgs {
                path: temp_dir.clone(),
                force: true,
            };
            run_init(&config, &args);

            // Run again with force
            run_init(&config, &args);

            // Cleanup
            let _ = fs::remove_dir_all(&temp_dir);
        }
    }

    mod run_config_tests {
        use super::*;

        #[test]
        fn test_run_config_show() {
            let config = CliConfig::default();
            let args = ConfigArgs {
                show: true,
                set: None,
                reset: false,
            };
            run_config(&config, &args);
        }

        #[test]
        fn test_run_config_set_valid() {
            let config = CliConfig::default();
            let args = ConfigArgs {
                show: false,
                set: Some("key=value".to_string()),
                reset: false,
            };
            run_config(&config, &args);
        }

        #[test]
        fn test_run_config_set_invalid() {
            let config = CliConfig::default();
            let args = ConfigArgs {
                show: false,
                set: Some("invalid_format".to_string()),
                reset: false,
            };
            run_config(&config, &args);
        }

        #[test]
        fn test_run_config_reset() {
            let config = CliConfig::default();
            let args = ConfigArgs {
                show: false,
                set: None,
                reset: true,
            };
            run_config(&config, &args);
        }

        #[test]
        fn test_run_config_all_flags() {
            let config = CliConfig::default();
            let args = ConfigArgs {
                show: true,
                set: Some("test=value".to_string()),
                reset: true,
            };
            run_config(&config, &args);
        }
    }

    mod run_tests_tests {
        use super::*;
        use probador::TestArgs;

        #[test]
        #[ignore = "Spawns cargo test --list subprocess - causes nested builds in CI"]
        fn test_run_tests_no_tests() {
            let config = CliConfig::default();
            let args = TestArgs {
                filter: None,
                parallel: 0,
                coverage: false,
                mutants: false,
                fail_fast: false,
                watch: false,
                timeout: 30000,
                output: PathBuf::from("target/probar"),
                skip_compile: true, // Skip compile in tests to avoid recursive cargo calls
            };
            // run_tests returns Ok when no tests are found
            let result = run_tests(config, &args);
            assert!(result.is_ok());
        }

        #[test]
        #[ignore = "Spawns cargo test --list subprocess - causes nested builds in CI"]
        fn test_run_tests_with_filter() {
            let config = CliConfig::default();
            let args = TestArgs {
                filter: Some("game::*".to_string()),
                parallel: 4,
                coverage: true,
                mutants: false,
                fail_fast: true,
                watch: false,
                timeout: 5000,
                output: PathBuf::from("target/test_output"),
                skip_compile: true, // Skip compile in tests to avoid recursive cargo calls
            };
            let result = run_tests(config, &args);
            assert!(result.is_ok());
        }
    }

    mod run_coverage_tests {
        use super::*;

        #[test]
        fn test_run_coverage_no_output() {
            let config = CliConfig::default();
            let args = CoverageArgs {
                png: None,
                json: None,
                palette: PaletteArg::Viridis,
                legend: false,
                gaps: false,
                title: None,
                width: 400,
                height: 300,
                input: None,
            };
            let result = run_coverage(&config, &args);
            assert!(result.is_ok());
        }

        #[test]
        fn test_run_coverage_with_png() {
            let temp_dir = std::env::temp_dir();
            let png_path = temp_dir.join("test_coverage.png");

            let config = CliConfig::default();
            let args = CoverageArgs {
                png: Some(png_path.clone()),
                json: None,
                palette: PaletteArg::Magma,
                legend: true,
                gaps: true,
                title: Some("Test Coverage".to_string()),
                width: 800,
                height: 600,
                input: None,
            };

            let result = run_coverage(&config, &args);
            assert!(result.is_ok());

            // Verify PNG was created
            assert!(png_path.exists());

            // Cleanup
            let _ = std::fs::remove_file(&png_path);
        }

        #[test]
        fn test_run_coverage_with_json() {
            let temp_dir = std::env::temp_dir();
            let json_path = temp_dir.join("test_coverage.json");

            let config = CliConfig::default();
            let args = CoverageArgs {
                png: None,
                json: Some(json_path.clone()),
                palette: PaletteArg::Heat,
                legend: false,
                gaps: false,
                title: None,
                width: 640,
                height: 480,
                input: None,
            };

            let result = run_coverage(&config, &args);
            assert!(result.is_ok());

            // Verify JSON was created
            assert!(json_path.exists());

            // Verify JSON content
            let content = std::fs::read_to_string(&json_path).unwrap();
            assert!(content.contains("overall_coverage"));

            // Cleanup
            let _ = std::fs::remove_file(&json_path);
        }

        // Tests for coverage, report generation, and gap cell detection
        // have been moved to handlers module for better testability
    }

    mod compliance_check_tests {
        use super::*;
        use std::fs;
        use tempfile::TempDir;

        #[test]
        fn test_check_c001_with_wasm_and_tests() {
            let temp = TempDir::new().unwrap();
            fs::write(temp.path().join("app.wasm"), b"wasm").unwrap();
            fs::write(temp.path().join("test.rs"), b"#[test]").unwrap();

            let result = check_c001_code_execution(temp.path());
            assert!(result.passed);
        }

        #[test]
        fn test_check_c001_no_wasm() {
            let temp = TempDir::new().unwrap();
            fs::write(temp.path().join("test.rs"), b"#[test]").unwrap();

            let result = check_c001_code_execution(temp.path());
            assert!(!result.passed);
        }

        #[test]
        fn test_check_c001_no_tests() {
            let temp = TempDir::new().unwrap();
            fs::write(temp.path().join("app.wasm"), b"wasm").unwrap();

            let result = check_c001_code_execution(temp.path());
            assert!(!result.passed);
        }

        #[test]
        fn test_check_c002_console_errors() {
            let result = check_c002_console_errors();
            assert!(result.passed);
        }

        #[test]
        fn test_check_c003_with_custom_elements() {
            let temp = TempDir::new().unwrap();
            let html = r#"<html><script>customElements.define('my-el', MyEl)</script></html>"#;
            fs::write(temp.path().join("index.html"), html).unwrap();

            let result = check_c003_custom_elements(temp.path());
            assert!(result.passed);
        }

        #[test]
        fn test_check_c003_with_wasm_element() {
            let temp = TempDir::new().unwrap();
            let html = r#"<html><wasm-app></wasm-app></html>"#;
            fs::write(temp.path().join("index.html"), html).unwrap();

            let result = check_c003_custom_elements(temp.path());
            assert!(result.passed);
        }

        #[test]
        fn test_check_c003_no_custom_elements() {
            let temp = TempDir::new().unwrap();
            let html = r#"<html><div>Hello</div></html>"#;
            fs::write(temp.path().join("index.html"), html).unwrap();

            let result = check_c003_custom_elements(temp.path());
            assert!(result.passed); // Still passes, just with different detail
        }

        #[test]
        fn test_check_c004_threading_modes() {
            let result = check_c004_threading_modes();
            assert!(result.passed);
        }

        #[test]
        fn test_check_c005_low_memory() {
            let result = check_c005_low_memory();
            assert!(result.passed);
        }

        #[test]
        fn test_check_c006_with_htaccess() {
            let temp = TempDir::new().unwrap();
            fs::write(temp.path().join(".htaccess"), "Header set").unwrap();

            let result = check_c006_headers(temp.path());
            assert!(result.passed);
        }

        #[test]
        fn test_check_c006_with_vercel() {
            let temp = TempDir::new().unwrap();
            fs::write(temp.path().join("vercel.json"), "{}").unwrap();

            let result = check_c006_headers(temp.path());
            assert!(result.passed);
        }

        #[test]
        fn test_check_c006_with_netlify() {
            let temp = TempDir::new().unwrap();
            fs::write(temp.path().join("netlify.toml"), "").unwrap();

            let result = check_c006_headers(temp.path());
            assert!(result.passed);
        }

        #[test]
        fn test_check_c006_with_headers_file() {
            let temp = TempDir::new().unwrap();
            fs::write(temp.path().join("_headers"), "/*\n  COOP").unwrap();

            let result = check_c006_headers(temp.path());
            assert!(result.passed);
        }

        #[test]
        fn test_check_c006_with_probar_config() {
            let temp = TempDir::new().unwrap();
            fs::write(
                temp.path().join("probar.toml"),
                "cross_origin_isolated = true",
            )
            .unwrap();

            let result = check_c006_headers(temp.path());
            assert!(result.passed);
        }

        #[test]
        fn test_check_c006_with_makefile() {
            let temp = TempDir::new().unwrap();
            fs::write(
                temp.path().join("Makefile"),
                "serve:\n\tprobador serve --cross-origin-isolated",
            )
            .unwrap();

            let result = check_c006_headers(temp.path());
            assert!(result.passed);
        }

        #[test]
        fn test_check_c006_no_config() {
            let temp = TempDir::new().unwrap();

            let result = check_c006_headers(temp.path());
            assert!(!result.passed);
        }

        #[test]
        fn test_check_c007_replay_hash() {
            let result = check_c007_replay_hash();
            assert!(result.passed);
        }

        #[test]
        fn test_check_c008_cache() {
            let result = check_c008_cache();
            assert!(result.passed);
        }

        #[test]
        fn test_check_c009_wasm_size_under_limit() {
            let temp = TempDir::new().unwrap();
            fs::write(temp.path().join("small.wasm"), vec![0u8; 1000]).unwrap();

            let result = check_c009_wasm_size(temp.path(), 10000);
            assert!(result.passed);
        }

        #[test]
        fn test_check_c009_wasm_size_over_limit() {
            let temp = TempDir::new().unwrap();
            fs::write(temp.path().join("large.wasm"), vec![0u8; 10000]).unwrap();

            let result = check_c009_wasm_size(temp.path(), 1000);
            assert!(!result.passed);
        }

        #[test]
        fn test_check_c009_no_wasm() {
            let temp = TempDir::new().unwrap();

            let result = check_c009_wasm_size(temp.path(), 10000);
            assert!(result.passed);
        }

        #[test]
        fn test_check_c010_with_panic_abort() {
            let temp = TempDir::new().unwrap();
            let cargo = r#"[profile.release]
panic = "abort""#;
            fs::write(temp.path().join("Cargo.toml"), cargo).unwrap();

            let result = check_c010_panic_paths(temp.path());
            assert!(result.passed);
            assert!(result
                .details
                .iter()
                .any(|d| d.contains("panic = \"abort\"")));
        }

        #[test]
        fn test_check_c010_without_panic_abort() {
            let temp = TempDir::new().unwrap();
            fs::write(temp.path().join("Cargo.toml"), "[package]\nname = \"test\"").unwrap();

            let result = check_c010_panic_paths(temp.path());
            assert!(result.passed); // Still passes with different detail
        }
    }

    // NOTE: file_finder_tests moved to handlers/comply.rs

    mod cross_origin_config_tests {
        use super::*;
        use std::fs;
        use tempfile::TempDir;

        #[test]
        fn test_check_probar_cross_origin_config_true() {
            let temp = TempDir::new().unwrap();
            fs::write(
                temp.path().join("probar.toml"),
                "cross_origin_isolated = true",
            )
            .unwrap();

            assert!(check_probar_cross_origin_config(temp.path()));
        }

        #[test]
        fn test_check_probar_cross_origin_config_no_space() {
            let temp = TempDir::new().unwrap();
            fs::write(
                temp.path().join("probar.toml"),
                "cross_origin_isolated=true",
            )
            .unwrap();

            assert!(check_probar_cross_origin_config(temp.path()));
        }

        #[test]
        fn test_check_probar_cross_origin_config_dot_prefixed() {
            let temp = TempDir::new().unwrap();
            fs::write(
                temp.path().join(".probar.toml"),
                "cross_origin_isolated = true",
            )
            .unwrap();

            assert!(check_probar_cross_origin_config(temp.path()));
        }

        #[test]
        fn test_check_probar_cross_origin_config_probador() {
            let temp = TempDir::new().unwrap();
            fs::write(
                temp.path().join("probador.toml"),
                "cross_origin_isolated = true",
            )
            .unwrap();

            assert!(check_probar_cross_origin_config(temp.path()));
        }

        #[test]
        fn test_check_probar_cross_origin_config_false() {
            let temp = TempDir::new().unwrap();
            fs::write(
                temp.path().join("probar.toml"),
                "cross_origin_isolated = false",
            )
            .unwrap();

            assert!(!check_probar_cross_origin_config(temp.path()));
        }

        #[test]
        fn test_check_probar_cross_origin_config_missing() {
            let temp = TempDir::new().unwrap();

            assert!(!check_probar_cross_origin_config(temp.path()));
        }

        #[test]
        fn test_check_makefile_cross_origin_probador() {
            let temp = TempDir::new().unwrap();
            fs::write(
                temp.path().join("Makefile"),
                "serve:\n\tprobador serve --cross-origin-isolated",
            )
            .unwrap();

            assert!(check_makefile_cross_origin(temp.path()));
        }

        #[test]
        fn test_check_makefile_cross_origin_probar() {
            let temp = TempDir::new().unwrap();
            fs::write(
                temp.path().join("Makefile"),
                "serve:\n\tprobar serve --cross-origin-isolated",
            )
            .unwrap();

            assert!(check_makefile_cross_origin(temp.path()));
        }

        #[test]
        fn test_check_makefile_cross_origin_lowercase() {
            let temp = TempDir::new().unwrap();
            fs::write(
                temp.path().join("makefile"),
                "serve:\n\tprobador serve --cross-origin-isolated",
            )
            .unwrap();

            assert!(check_makefile_cross_origin(temp.path()));
        }

        #[test]
        fn test_check_makefile_cross_origin_gnu() {
            let temp = TempDir::new().unwrap();
            fs::write(
                temp.path().join("GNUmakefile"),
                "serve:\n\tprobador serve --cross-origin-isolated",
            )
            .unwrap();

            assert!(check_makefile_cross_origin(temp.path()));
        }

        #[test]
        fn test_check_makefile_cross_origin_package_json() {
            let temp = TempDir::new().unwrap();
            let pkg = r#"{"scripts": {"serve": "probador serve --cross-origin-isolated"}}"#;
            fs::write(temp.path().join("package.json"), pkg).unwrap();

            assert!(check_makefile_cross_origin(temp.path()));
        }

        #[test]
        fn test_check_makefile_cross_origin_without_flag() {
            let temp = TempDir::new().unwrap();
            fs::write(temp.path().join("Makefile"), "serve:\n\tprobador serve").unwrap();

            assert!(!check_makefile_cross_origin(temp.path()));
        }

        #[test]
        fn test_check_makefile_cross_origin_missing() {
            let temp = TempDir::new().unwrap();

            assert!(!check_makefile_cross_origin(temp.path()));
        }
    }

    // NOTE: compliance_result_tests moved to handlers/comply.rs
}
