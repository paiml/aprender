//! CLI command handlers.
//!
//! This module contains the execution logic for each CLI command.
//! Extracted to enable comprehensive testing of command behavior.

use crate::edd::v2::{validate_emc_yaml, validate_experiment_yaml, SchemaValidationError};
use crate::edd::{ExperimentRunner, RunnerConfig};
use std::path::Path;
use std::process::ExitCode;

use super::args::RenderFormat;
use super::output::{
    print_emc_report, print_emc_validation_results, print_experiment_result, print_help,
    print_version,
};
use super::schema::validate_emc_schema;
use super::{Args, Command};

/// Main CLI entry point.
///
/// Dispatches to the appropriate command handler based on parsed arguments.
#[must_use]
pub fn run_cli(args: Args) -> ExitCode {
    match args.command {
        Command::Run {
            experiment_path,
            seed_override,
            verbose,
        } => run_experiment(&experiment_path, seed_override, verbose),
        Command::Render {
            domain,
            format,
            output,
            fps,
            duration,
            seed,
        } => render_svg(&domain, &format, &output, fps, duration, seed),
        Command::Validate { experiment_path } => validate_experiment(&experiment_path),
        Command::Verify {
            experiment_path,
            runs,
        } => verify_reproducibility(&experiment_path, runs),
        Command::EmcCheck { experiment_path } => emc_check(&experiment_path),
        Command::EmcValidate { emc_path } => emc_validate(&emc_path),
        Command::ListEmc => list_emc(),
        Command::Help => {
            print_help();
            ExitCode::SUCCESS
        }
        Command::Version => {
            print_version();
            ExitCode::SUCCESS
        }
        Command::Error(msg) => {
            eprintln!("Error: {msg}");
            print_help();
            ExitCode::FAILURE
        }
    }
}

/// Run an experiment from a YAML file.
///
/// # Arguments
///
/// * `path` - Path to the experiment YAML file
/// * `seed_override` - Optional seed to override the experiment's configured seed
/// * `verbose` - Whether to enable verbose output
#[must_use]
pub fn run_experiment(path: &Path, seed_override: Option<u64>, verbose: bool) -> ExitCode {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║           simular - EDD Experiment Runner                     ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    let config = RunnerConfig {
        seed_override,
        verbose,
        ..RunnerConfig::default()
    };

    let mut runner = ExperimentRunner::with_config(config);

    // Initialize EMC registry
    match runner.initialize() {
        Ok(count) => {
            if verbose {
                println!("Loaded {count} EMCs from library");
            }
        }
        Err(e) => {
            eprintln!("Warning: Failed to scan EMC library: {e}");
        }
    }

    println!("Running experiment: {}\n", path.display());

    match runner.run(path) {
        Ok(result) => {
            print_experiment_result(&result, verbose);
            if result.passed {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(1)
            }
        }
        Err(e) => {
            eprintln!("Error: {e}");
            ExitCode::from(1)
        }
    }
}

/// Validate an experiment YAML file against the EDD v2 schema.
///
/// # Arguments
///
/// * `path` - Path to the experiment YAML file
#[must_use]
pub fn validate_experiment(path: &Path) -> ExitCode {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║       simular - EDD v2 Experiment Schema Validation           ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    println!("Validating: {}\n", path.display());

    // Read file
    let contents = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("✗ Error reading file: {e}");
            return ExitCode::from(1);
        }
    };

    // Validate against EDD v2 schema
    match validate_experiment_yaml(&contents) {
        Ok(()) => {
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!("✓ Schema validation PASSED");
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
            println!("EDD v2 Compliance:");
            println!("  ✓ Required fields present (id, seed, emc_ref, simulation, falsification)");
            println!("  ✓ Falsification criteria defined");
            println!("  ✓ No prohibited custom code fields");
            println!("\nNext steps:");
            println!("  • Run: simular run {}", path.display());
            println!("  • Verify: simular verify {}", path.display());
            ExitCode::SUCCESS
        }
        Err(e) => {
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!("✗ Schema validation FAILED");
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
            match e {
                SchemaValidationError::ValidationFailed(errors) => {
                    println!("Validation errors:");
                    for (i, err) in errors.iter().enumerate() {
                        println!("  {}. {err}", i + 1);
                    }
                }
                other => {
                    println!("Error: {other}");
                }
            }
            println!("\nSee: docs/specifications/EDD-spec-unified.md for schema requirements");
            ExitCode::from(1)
        }
    }
}

/// Verify reproducibility of an experiment across multiple runs.
///
/// # Arguments
///
/// * `path` - Path to the experiment YAML file
/// * `runs` - Number of verification runs to perform
#[must_use]
pub fn verify_reproducibility(path: &Path, runs: usize) -> ExitCode {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║        simular - Reproducibility Verification                 ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    let config = RunnerConfig {
        verify_reproducibility: true,
        reproducibility_runs: runs,
        ..RunnerConfig::default()
    };

    let mut runner = ExperimentRunner::with_config(config);

    if let Err(e) = runner.initialize() {
        eprintln!("Warning: Failed to scan EMC library: {e}");
    }

    println!("Verifying reproducibility: {}", path.display());
    println!("Runs: {runs}\n");

    match runner.verify(path) {
        Ok(summary) => {
            let status = if summary.passed { "PASSED" } else { "FAILED" };
            let sym = if summary.passed { "✓" } else { "✗" };

            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!("Reproducibility Check");
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

            println!("  Runs:      {}", summary.runs);
            println!("  Identical: {}", summary.identical);
            println!("  Platform:  {}", summary.platform);
            println!("\n  Reference Hash: {}", summary.reference_hash);

            if summary.run_hashes.len() > 1 {
                println!("\n  Run Hashes:");
                for (i, hash) in summary.run_hashes.iter().enumerate() {
                    let match_sym = if hash == &summary.reference_hash {
                        "="
                    } else {
                        "!"
                    };
                    println!("    Run {}: {} {}", i + 1, hash, match_sym);
                }
            }

            println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!("{sym} Result: {status}");
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

            if summary.passed {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(1)
            }
        }
        Err(e) => {
            eprintln!("Error: {e}");
            ExitCode::from(1)
        }
    }
}

/// Check EMC compliance for an experiment.
///
/// # Arguments
///
/// * `path` - Path to the experiment YAML file
#[must_use]
pub fn emc_check(path: &Path) -> ExitCode {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║          simular - EMC Compliance Check                       ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    let mut runner = ExperimentRunner::new();

    if let Err(e) = runner.initialize() {
        eprintln!("Warning: Failed to scan EMC library: {e}");
    }

    println!("Checking EMC compliance: {}\n", path.display());

    match runner.emc_check(path) {
        Ok(report) => {
            print_emc_report(&report);
            if report.passed {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(1)
            }
        }
        Err(e) => {
            eprintln!("Error: {e}");
            ExitCode::from(1)
        }
    }
}

/// List all EMCs available in the library.
#[must_use]
pub fn list_emc() -> ExitCode {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║             simular - EMC Library                             ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    let mut runner = ExperimentRunner::new();

    match runner.initialize() {
        Ok(count) => {
            println!("Found {count} EMCs in library:\n");

            let references = runner.registry_mut().list_references();
            let mut sorted: Vec<_> = references.into_iter().collect();
            sorted.sort_unstable();

            let mut current_domain = String::new();

            for reference in &sorted {
                let parts: Vec<&str> = reference.split('/').collect();
                if parts.len() >= 2 {
                    let domain = parts[0];
                    if domain != current_domain {
                        if !current_domain.is_empty() {
                            println!();
                        }
                        println!("{}:", domain.to_uppercase());
                        current_domain = domain.to_string();
                    }
                }
                println!("  - {reference}");
            }

            println!("\nUsage: simular run <experiment.yaml>");
            println!(
                "Reference EMCs using: equation_model_card.emc_ref: \"{}\"",
                if sorted.is_empty() {
                    "domain/name"
                } else {
                    sorted[0]
                }
            );

            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("Error scanning EMC library: {e}");
            ExitCode::from(1)
        }
    }
}

/// Validate an EMC YAML file against the EDD v2 schema.
///
/// # Arguments
///
/// * `path` - Path to the EMC file
#[must_use]
pub fn emc_validate(path: &Path) -> ExitCode {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║        simular - EDD v2 EMC Schema Validation                 ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");
    println!("Validating EMC: {}\n", path.display());

    let contents = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("✗ Error reading file: {e}");
            return ExitCode::from(1);
        }
    };

    // First, validate against EDD v2 JSON schema
    match validate_emc_yaml(&contents) {
        Ok(()) => {
            println!("✓ EDD v2 JSON Schema validation PASSED\n");
        }
        Err(e) => {
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!("✗ EDD v2 JSON Schema validation FAILED");
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
            match e {
                SchemaValidationError::ValidationFailed(errors) => {
                    println!("Validation errors:");
                    for (i, err) in errors.iter().enumerate() {
                        println!("  {}. {err}", i + 1);
                    }
                }
                other => {
                    println!("Error: {other}");
                }
            }
            return ExitCode::from(1);
        }
    }

    // Then run the semantic validation
    let yaml: serde_yaml::Value = match serde_yaml::from_str(&contents) {
        Ok(y) => y,
        Err(e) => {
            eprintln!("YAML Syntax Error: {e}");
            return ExitCode::from(1);
        }
    };

    let (errors, warnings) = validate_emc_schema(&yaml);
    print_emc_validation_results(&yaml, &errors, &warnings);

    if errors.is_empty() {
        println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("✓ All EMC validations PASSED");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}

/// Shared mutable state for frame rendering.
struct RenderCtx<'a> {
    format: &'a RenderFormat,
    output: &'a Path,
    svg: &'a mut crate::renderers::svg::SvgRenderer,
    keyframes: &'a mut crate::renderers::keyframes::KeyframeRecorder,
}

/// Write a single SVG frame or record keyframe data.
fn write_frame(
    frame: usize,
    commands: &[crate::orbit::render::RenderCommand],
    ctx: &mut RenderCtx<'_>,
) -> Result<(), String> {
    match ctx.format {
        RenderFormat::SvgFrames => {
            let svg = ctx.svg.render(commands);
            let path = ctx.output.join(format!("frame_{frame:04}.svg"));
            std::fs::write(&path, &svg).map_err(|e| format!("Error writing frame {frame}: {e}"))
        }
        RenderFormat::SvgKeyframes => {
            ctx.keyframes.record_frame(commands);
            if frame == 0 {
                let svg = ctx.svg.render(commands);
                let path = ctx.output.join("scene.svg");
                std::fs::write(&path, &svg).map_err(|e| format!("Error writing template SVG: {e}"))
            } else {
                Ok(())
            }
        }
    }
}

/// Render a simulation to SVG frames or SVG + keyframes JSON.
pub fn render_svg(
    domain: &str,
    format: &RenderFormat,
    output: &Path,
    fps: u32,
    duration: f64,
    seed: u64,
) -> ExitCode {
    use crate::renderers::keyframes::KeyframeRecorder;
    use crate::renderers::svg::SvgRenderer;

    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║           simular - SVG Render Pipeline                      ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    if let Err(e) = std::fs::create_dir_all(output) {
        eprintln!("Error creating output directory: {e}");
        return ExitCode::from(1);
    }

    let format_name = match format {
        RenderFormat::SvgFrames => "svg-frames",
        RenderFormat::SvgKeyframes => "svg-keyframes",
    };
    println!("Domain:   {domain}");
    println!("Format:   {format_name}");
    println!("Output:   {}", output.display());
    println!("FPS:      {fps}");
    println!("Duration: {duration}s");
    println!("Seed:     {seed}\n");

    let total_frames = (duration * f64::from(fps)) as usize;
    let mut svg_renderer = SvgRenderer::new();
    let mut keyframe_recorder = KeyframeRecorder::new(fps, seed, domain);
    let mut ctx = RenderCtx {
        format,
        output,
        svg: &mut svg_renderer,
        keyframes: &mut keyframe_recorder,
    };

    println!("Rendering {total_frames} frames...");

    let result = match domain {
        "orbit" => render_orbit(fps, total_frames, &mut ctx),
        "bouncing_balls" => render_bouncing_balls(fps, seed, total_frames, &mut ctx),
        _ => {
            eprintln!("Error: unsupported domain '{domain}'. Supported: orbit, bouncing_balls");
            return ExitCode::from(1);
        }
    };

    if let Err(e) = result {
        eprintln!("{e}");
        return ExitCode::from(1);
    }

    if *format == RenderFormat::SvgKeyframes {
        let json = ctx.keyframes.to_json();
        let path = output.join("keyframes.json");
        if let Err(e) = std::fs::write(&path, &json) {
            eprintln!("Error writing keyframes: {e}");
            return ExitCode::from(1);
        }
        println!("\r  Frames:     {total_frames}");
        println!("  Elements:   {}", ctx.keyframes.element_count());
        println!("  Template:   {}", output.join("scene.svg").display());
        println!("  Keyframes:  {}", path.display());
    } else {
        println!("\r  Frames:     {total_frames}");
        println!("  Output:     {}/frame_NNNN.svg", output.display());
    }

    println!("\n✓ Render complete");
    ExitCode::SUCCESS
}

/// Render orbit domain.
fn render_orbit(fps: u32, total_frames: usize, ctx: &mut RenderCtx<'_>) -> Result<(), String> {
    use crate::orbit::physics::YoshidaIntegrator;
    use crate::orbit::render::{render_state, OrbitTrail, RenderConfig};
    use crate::orbit::scenarios::KeplerConfig;
    use crate::orbit::units::OrbitTime;

    let config = KeplerConfig::default();
    let mut state = config.build(1e6);
    let integrator = YoshidaIntegrator::new();
    let dt_seconds = 3600.0;
    let sim_dt_per_frame = 86400.0 / f64::from(fps);
    let steps_per_frame = (sim_dt_per_frame / dt_seconds).max(1.0) as usize;
    let render_config = RenderConfig::default();
    let mut trails = vec![OrbitTrail::new(0), OrbitTrail::new(500)];

    for frame in 0..total_frames {
        for _ in 0..steps_per_frame {
            let dt = OrbitTime::from_seconds(dt_seconds);
            integrator
                .step(&mut state, dt)
                .map_err(|_| format!("Physics error at frame {frame}"))?;
            for (i, body) in state.bodies.iter().enumerate() {
                if i < trails.len() {
                    let (x, y, _) = body.position.as_meters();
                    trails[i].push(x, y);
                }
            }
        }
        let commands = render_state(&state, &render_config, &trails, None, None);
        write_frame(frame, &commands, ctx)?;
        if frame % 60 == 0 {
            print!("\r  Frame {frame}/{total_frames}");
        }
    }
    Ok(())
}

/// Render bouncing balls domain.
fn render_bouncing_balls(
    fps: u32,
    seed: u64,
    total_frames: usize,
    ctx: &mut RenderCtx<'_>,
) -> Result<(), String> {
    use crate::scenarios::bouncing_balls::{BouncingBallsConfig, BouncingBallsState};

    let config = BouncingBallsConfig::default();
    let mut state = BouncingBallsState::new(config, seed);
    let dt = 1.0 / f64::from(fps);

    for frame in 0..total_frames {
        state.step(dt);
        let commands = state.render();
        write_frame(frame, &commands, ctx)?;
        if frame % 60 == 0 {
            print!("\r  Frame {frame}/{total_frames}");
        }
    }
    Ok(())
}
