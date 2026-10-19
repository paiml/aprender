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
use super::{Cli, Commands};

/// Main CLI entry point.
///
/// Dispatches to the appropriate command handler based on parsed arguments.
///
/// There is no parse-failure arm: clap reports a bad argument itself, so a
/// failure never reaches here as a value. The old `Command::Error(String)`
/// variant turned a parse failure INTO a command, which is how unknown input
/// used to reach a handler at all.
#[must_use]
pub fn run_cli(cli: Cli) -> ExitCode {
    match cli.into_command() {
        Commands::Run {
            experiment_path,
            seed_override,
            verbose,
        } => run_experiment(&experiment_path, seed_override, verbose),
        Commands::Render {
            domain,
            format,
            output,
            fps,
            duration,
            seed,
        } => render_svg(&domain, &format, &output, fps, duration, seed),
        Commands::PrFlow {
            design,
            json,
            r_pr,
            r_mq,
            cap,
            builds,
            f,
            n,
            pr_min,
            mq_min,
            p_red,
            pr_setup,
            t_rev,
            rev_concurrent,
            rev_lanes,
            observed_24h: obs24,
            src_runs: runs,
            params,
            seeds,
            hours,
            warmup,
            sha,
        } => match params {
            Some(path) => pr_flow_params(&path, seeds, hours, warmup),
            None => pr_flow(
                &design,
                json,
                &FlowOverrides {
                    r_pr,
                    r_mq,
                    cap,
                    builds,
                    f,
                    n,
                    pr_min,
                    mq_min,
                    p_red,
                    pr_setup,
                    t_rev,
                    rev_concurrent,
                    rev_lanes,
                    obs24,
                    runs,
                },
                &sha,
            ),
        },
        Commands::RepoFlow { case, inputs, json } => repo_flow(&case, inputs.as_deref(), json),
        Commands::CourseFlow { case, inputs, json } => course_flow(&case, inputs.as_deref(), json),
        Commands::Validate { experiment_path } => validate_experiment(&experiment_path),
        Commands::Verify {
            experiment_path,
            runs,
        } => verify_reproducibility(&experiment_path, runs),
        Commands::EmcCheck { experiment_path } => emc_check(&experiment_path),
        Commands::EmcValidate { emc_path } => emc_validate(&emc_path),
        Commands::ListEmc => list_emc(),
        Commands::Help => {
            print_help();
            ExitCode::SUCCESS
        }
        Commands::Version => {
            print_version();
            ExitCode::SUCCESS
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
            println!("  • Run: aprender-simulate run {}", path.display());
            println!("  • Verify: aprender-simulate verify {}", path.display());
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

            println!("\nUsage: aprender-simulate run <experiment.yaml>");
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

/// Measured inputs that replace a design's defaults; each is labelled `measured` in the sim block.
#[derive(Default, Clone)]
struct FlowOverrides {
    r_pr: Option<usize>,
    r_mq: Option<usize>,
    cap: Option<usize>,
    builds: Option<usize>,
    f: Option<f64>,
    n: Option<f64>,
    pr_min: Option<f64>,
    mq_min: Option<f64>,
    p_red: Option<f64>,
    pr_setup: Option<f64>,
    t_rev: Option<f64>,
    rev_concurrent: bool,
    rev_lanes: Option<usize>,
    obs24: Option<f64>,
    runs: Option<String>,
}

impl FlowOverrides {
    fn apply(&self, c: &mut crate::scenarios::PrFlowConfig) {
        c.r_pr = self.r_pr.unwrap_or(c.r_pr);
        c.r_mq = self.r_mq.unwrap_or(c.r_mq);
        c.cap = self.cap.unwrap_or(c.cap);
        c.builds = self.builds.unwrap_or(c.builds);
        c.f = self.f.unwrap_or(c.f);
        c.n = self.n.unwrap_or(c.n);
        c.pr_runner_min = self.pr_min.unwrap_or(c.pr_runner_min);
        c.mq_runner_min = self.mq_min.unwrap_or(c.mq_runner_min);
        c.p_red = self.p_red.unwrap_or(c.p_red);
        c.pr_setup_min = self.pr_setup.unwrap_or(c.pr_setup_min);
        c.t_rev_min = self.t_rev.unwrap_or(c.t_rev_min);
        if self.rev_concurrent {
            c.rev_mode = crate::scenarios::ReviewMode::Concurrent;
        }
        c.rev_lanes = self.rev_lanes.unwrap_or(c.rev_lanes);
    }

    fn cli_flags(&self) -> String {
        let parts = [
            self.r_pr.map(|v| format!(" --r-pr {v}")),
            self.r_mq.map(|v| format!(" --r-mq {v}")),
            self.cap.map(|v| format!(" --cap {v}")),
            self.builds.map(|v| format!(" --builds {v}")),
            self.f.map(|v| format!(" --f {v}")),
            self.n.map(|v| format!(" --n {v}")),
            self.pr_min.map(|v| format!(" --pr-min {v}")),
            self.mq_min.map(|v| format!(" --mq-min {v}")),
            self.p_red.map(|v| format!(" --p-red {v}")),
            self.pr_setup.map(|v| format!(" --pr-setup {v}")),
            self.t_rev.map(|v| format!(" --t-rev {v}")),
            self.rev_concurrent.then(|| " --rev-concurrent".to_string()),
            self.rev_lanes.map(|v| format!(" --rev-lanes {v}")),
            self.obs24.map(|v| format!(" --observed-24h {v}")),
            self.runs.as_ref().map(|v| format!(" --src-runs {v}")),
        ];
        parts.into_iter().flatten().collect()
    }
}

/// Print the reference run (5 seeds x 400 h, 20 h warm-up) of a named PR-flow design.
fn pr_flow(design: &str, json: bool, ov: &FlowOverrides, sha: &str) -> ExitCode {
    let Some(mut cfg) = crate::scenarios::preset(design) else {
        eprintln!("unknown design: {design}");
        return ExitCode::FAILURE;
    };
    ov.apply(&mut cfg);
    if json {
        println!("{}", pr_flow_sim_block(design, &cfg, ov, sha));
        return ExitCode::SUCCESS;
    }
    let r = crate::scenarios::run_reference(&cfg);
    println!(
        "design {design}: merged/h {:.2}  W mean {:.1} min  W p90 {:.1} min  L {:.2}",
        r.merged_per_h, r.w_mean_min, r.w_p90_min, r.l
    );
    println!(
        "PRs/pass {:.2}  PR util {:.0}%  MQ util {:.0}%  runs/PR {:.2}",
        r.prs_per_pass,
        r.pr_util * 100.0,
        r.mq_util * 100.0,
        r.runs_per_pr
    );
    println!(
        "stages: run wall {:.1}  open->green {:.1}  green->armed {:.1}  armed->merged {:.1} min",
        r.run_wall_min, r.checks_min, r.arm_min, r.mq_min
    );
    ExitCode::SUCCESS
}

/// The cycle-record "sim" block: seeds 1..=5, 400 h, 20 h warm-up, one JSON line.
/// `optimum` compares the design's own reference run with this run; `drift_pct_24h` is null (not computed here).
fn pr_flow_sim_block(
    design: &str,
    cfg: &crate::scenarios::PrFlowConfig,
    ov: &FlowOverrides,
    sha: &str,
) -> String {
    let runs: Vec<_> = (1..=5)
        .map(|s| crate::scenarios::run_pr_flow(cfg, s, 400.0, 20.0))
        .collect();
    let n = runs.len() as f64;
    let mean = |f: fn(&crate::scenarios::PrFlowResult) -> f64| runs.iter().map(f).sum::<f64>() / n;
    let m = mean(|r| r.merged_per_h);
    let var = runs
        .iter()
        .map(|r| (r.merged_per_h - m).powi(2))
        .sum::<f64>()
        / (n - 1.0);
    let ci95 = 1.96 * (var / n).sqrt();
    let avg = crate::scenarios::PrFlowResult {
        w_mean_min: mean(|r| r.w_mean_min),
        runs_per_pr: mean(|r| r.runs_per_pr),
        prs_per_pass: mean(|r| r.prs_per_pass),
        run_wall_min: mean(|r| r.run_wall_min),
        pass_wait_min: mean(|r| r.pass_wait_min),
        pass_wall_min: mean(|r| r.pass_wall_min),
        ..Default::default()
    };
    let binding = crate::scenarios::binding_station(cfg, &avg);
    let law = crate::scenarios::eq7_law(cfg, &avg);
    let w_sim = mean(|r| r.w_mean_min);
    let base = crate::scenarios::preset(design)
        .map_or(0.0, |c| crate::scenarios::run_reference(&c).merged_per_h);
    let contract = crate::scenarios::reference_contract_ok();
    let src = |m: bool| if m { "measured" } else { "design_default" };
    serde_json::json!({
        "aprender_sha": sha,
        "scenario": "pr-flow",
        "cli": format!("aprender-simulate pr-flow --design {design} --json{} --sha {sha}", ov.cli_flags()),
        "seeds": [1, 2, 3, 4, 5],
        "hours": 400,
        "inputs": {
            "design": design, "cap": cfg.cap, "n": cfg.n, "t_author_min": cfg.t_author_min,
            "r_pr": cfg.r_pr, "r_mq": cfg.r_mq, "builds": cfg.builds, "p_red": cfg.p_red,
            "pr_runner_min": cfg.pr_runner_min, "mq_runner_min": cfg.mq_runner_min,
            "pr_setup_min": cfg.pr_setup_min, "t_rev_min": cfg.t_rev_min, "f": cfg.f
        },
        "inputs_source": {
            "t_author_min": "design_default", "n": src(ov.n.is_some()), "cap": src(ov.cap.is_some()),
            "r_pr": src(ov.r_pr.is_some()), "r_mq": src(ov.r_mq.is_some()), "p_red": src(ov.p_red.is_some()),
            "pr_runner_min": src(ov.pr_min.is_some()), "mq_runner_min": src(ov.mq_min.is_some()),
            "pr_setup_min": src(ov.pr_setup.is_some()), "f": src(ov.f.is_some()), "builds": src(ov.builds.is_some())
        },
        "predicted": {
            "merged_h": m, "ci95_h": ci95,
            "W_min_mean": w_sim, "L": mean(|r| r.l), "rho_a": mean(|r| r.rho_a), "rho_a_assumed": 0.95
        },
        "law": {
            "eq": 7, "W_law_min": law.w_law_min(), "gap_pct": (w_sim - law.w_law_min()) / law.w_law_min() * 100.0,
            "r": law.r, "w_min": law.w_min, "c_wall_min": law.c_wall_min, "t_fix_min": law.t_fix_min,
            "t_arm_min": law.t_arm_min, "p_dirty": law.p_dirty, "t_reb_min": law.t_reb_min,
            "wait_mq_min": law.wait_mq_min, "t_pass_min": law.t_pass_min, "w_pass_min": law.w_pass_min, "f": law.f,
            "checks_min": law.checks_min(), "rebase_min": law.rebase_min(), "W_mq_min": law.w_mq_min()
        },
        "binding_station": binding,
        "optimum": {
            "design": design, "design_merged_h": base, "this_run_merged_h": m,
            "gap_h": base - m, "gap_pct": if base > 0.0 { (base - m) / base * 100.0 } else { 0.0 }
        },
        "sim3_contract": if contract { "pass" } else { "fail" },
        "observed_24h": ov.obs24, "source_runs": ov.runs,
        "drift_pct_24h": ov.obs24.map(|o| (m - o / 24.0) / (o / 24.0) * 100.0)
    })
    .to_string()
}

/// `pr-flow --params <file>`: one repo's arbiter params in, `{actual, one_change, optimum}` out (one JSON line).
fn pr_flow_params(path: &Path, seeds: u64, hours: f64, warmup: f64) -> ExitCode {
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("params: cannot read {}: {e}", path.display());
            return ExitCode::FAILURE;
        }
    };
    match crate::scenarios::parse_params(&text) {
        Ok(p) => {
            println!("{}", crate::scenarios::evaluate(&p, seeds, hours, warmup));
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}

/// Print the reference run of a repo case, optionally with the ledger's inputs, as text or JSON.
fn repo_flow(case: &str, inputs: Option<&Path>, json: bool) -> ExitCode {
    let Some(mut cfg) = crate::scenarios::repo_preset(case) else {
        eprintln!("unknown case: {case}");
        return ExitCode::FAILURE;
    };
    if let Some(p) = inputs {
        let applied = std::fs::read_to_string(p)
            .map_err(|e| e.to_string())
            .and_then(|s| serde_json::from_str(&s).map_err(|e| e.to_string()))
            .and_then(|v| crate::scenarios::apply_inputs(&mut cfg, &v));
        if let Err(e) = applied {
            eprintln!("inputs: {e}");
            return ExitCode::FAILURE;
        }
    }
    let r = crate::scenarios::run_repo_reference(&cfg);
    if json {
        println!("{}", crate::scenarios::sim_block(case, &cfg, &r));
    } else {
        println!(
            "case {case}: merged/h {:.3}  W mean {:.1} min  L {:.2}  pool {:.0}%",
            r.merged_per_h,
            r.w_mean_min,
            r.l,
            r.pr_util * 100.0
        );
    }
    ExitCode::SUCCESS
}

/// Print the reference run of a course-factory case, optionally with the ledger's inputs.
fn course_flow(case: &str, inputs: Option<&Path>, json: bool) -> ExitCode {
    let Some(mut cfg) = crate::scenarios::course_preset(case) else {
        eprintln!("unknown case: {case}");
        return ExitCode::FAILURE;
    };
    if let Some(p) = inputs {
        let applied = std::fs::read_to_string(p)
            .map_err(|e| e.to_string())
            .and_then(|s| serde_json::from_str(&s).map_err(|e| e.to_string()))
            .and_then(|v| crate::scenarios::apply_course_inputs(&mut cfg, &v));
        if let Err(e) = applied {
            eprintln!("inputs: {e}");
            return ExitCode::FAILURE;
        }
    }
    let r = crate::scenarios::run_course_reference(&cfg);
    if json {
        println!("{}", crate::scenarios::course_sim_block(case, &cfg, &r));
    } else {
        let (st, u) = r.busiest();
        println!(
            "case {case}: courses/day {:.2}  shorts/day {:.2}  lead {:.1} d  busiest {st} {:.0}%",
            r.courses_per_day,
            r.shorts_per_day,
            r.lead_days,
            u * 100.0
        );
    }
    ExitCode::SUCCESS
}
