//! `apr train` — Training commands (plan, apply, watch, sweep).
//!
//! Classification fine-tuning (plan/apply for --task classify) requires
//! entrenar >= 0.8 which has not yet been published. Those subcommands
//! return a clear error. Pre-training (--task pretrain), sweep, and watch
//! work with the current entrenar 0.7.x release.

use colored::Colorize;

use crate::{error::CliError, output};

type Result<T> = std::result::Result<T, CliError>;

/// Error returned when classify fine-tuning APIs are invoked.
fn classify_not_available() -> CliError {
    CliError::ValidationFailed(
        "apr train (classify) requires entrenar >= 0.8 (not yet published). \
         Use --task pretrain for causal LM pre-training."
            .to_string(),
    )
}

/// Run `apr train plan` — generate and display a training plan.
#[allow(clippy::too_many_arguments)]
#[provable_contracts_macros::contract(
    "apr-cli-operations-v1",
    equation = "mutating_output_contract"
)]
pub(crate) fn run_plan(
    _data: Option<&std::path::Path>,
    _model_size: &str,
    _model_path: Option<&std::path::Path>,
    _num_classes: usize,
    task: &str,
    config_path: Option<&std::path::Path>,
    _output_dir: &std::path::Path,
    _strategy: &str,
    _budget: usize,
    _scout: bool,
    _max_epochs: usize,
    _manual_lr: Option<f32>,
    _manual_lora_rank: Option<usize>,
    _manual_batch_size: Option<usize>,
    _val_data: Option<&std::path::Path>,
    _test_data: Option<&std::path::Path>,
    format: &str,
    json_output: bool,
) -> Result<()> {
    contract_pre_training_plan_apply_semantics!();
    // `--format` used to be `_format`: declared, documented, and never read, so
    // text/json/yaml/garbage all produced byte-identical text with exit 0.
    let format = PlanFormat::parse(format)?;
    let format = if json_output {
        PlanFormat::Json
    } else {
        format
    };
    let result = match task {
        "pretrain" | "causal_lm" => run_plan_pretrain(config_path, format),
        "classify" => Err(classify_not_available()),
        _ => Err(CliError::ValidationFailed(format!(
            "Unknown task type: {task}. Supported: classify, pretrain"
        ))),
    };
    if let Ok(ref r) = result {
        contract_post_training_plan_apply_semantics!(r);
    }
    result
}

/// The rendering `apr train plan --format` selects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PlanFormat {
    /// Human-readable table (the default).
    Text,
    /// Pretty-printed JSON manifest.
    Json,
    /// YAML manifest — the same fields as [`PlanFormat::Json`].
    Yaml,
}

impl PlanFormat {
    /// Parse a `--format` value, rejecting anything that is not a real format.
    ///
    /// An unrecognised value used to be accepted silently and rendered as text.
    fn parse(raw: &str) -> Result<Self> {
        match raw.trim().to_lowercase().as_str() {
            "text" | "" => Ok(Self::Text),
            "json" => Ok(Self::Json),
            "yaml" | "yml" => Ok(Self::Yaml),
            other => Err(CliError::ValidationFailed(format!(
                "Unknown --format '{other}'. Supported: text, json, yaml"
            ))),
        }
    }
}

/// Build the serializable plan manifest shared by the JSON and YAML renderings.
fn pretrain_plan_manifest(
    config_path: &std::path::Path,
    spec: &entrenar::config::TrainSpec,
) -> serde_json::Value {
    serde_json::json!({
        "task": "pretrain",
        "config": config_path.display().to_string(),
        "model": {
            "path": spec.model.path.display().to_string(),
            "mode": format!("{:?}", spec.model.mode),
        },
        "data": {
            "train": spec.data.train.display().to_string(),
            "batch_size": spec.data.batch_size,
            "seq_len": spec.data.seq_len,
        },
        "optimizer": {
            "name": spec.optimizer.name,
            "lr": spec.optimizer.lr,
        },
        "training": {
            "epochs": spec.training.epochs,
            "mode": format!("{:?}", spec.training.mode),
            "warmup_steps": spec.training.warmup_steps,
            "gradient_accumulation": spec.training.gradient_accumulation,
            "mixed_precision": spec.training.mixed_precision,
            "output_dir": spec.training.output_dir.display().to_string(),
        },
        "verdict": "ready",
    })
}

/// Plan for causal LM pre-training from YAML config (ALB-009).
fn run_plan_pretrain(config_path: Option<&std::path::Path>, format: PlanFormat) -> Result<()> {
    let config_path = config_path.ok_or_else(|| {
        CliError::ValidationFailed("--config <yaml> is required for --task pretrain".to_string())
    })?;

    if !config_path.exists() {
        return Err(CliError::FileNotFound(config_path.to_path_buf()));
    }

    // Load and validate the training config
    let spec = entrenar::config::load_config(config_path)
        .map_err(|e| CliError::ValidationFailed(format!("Config error: {e}")))?;

    entrenar::config::validate_config(&spec)
        .map_err(|e| CliError::ValidationFailed(format!("Validation error: {e}")))?;

    // Display the pre-training plan
    if format == PlanFormat::Json {
        let plan = pretrain_plan_manifest(config_path, &spec);
        println!(
            "{}",
            serde_json::to_string_pretty(&plan).unwrap_or_default()
        );
    } else if format == PlanFormat::Yaml {
        let plan = pretrain_plan_manifest(config_path, &spec);
        let rendered = serde_yaml::to_string(&plan).map_err(|e| {
            CliError::ValidationFailed(format!("Could not render plan as YAML: {e}"))
        })?;
        print!("{rendered}");
    } else {
        output::header("apr train plan — Pre-training Plan (Causal LM)");
        println!();
        output::kv("  Config", config_path.display().to_string());
        output::kv("  Model path", spec.model.path.display().to_string());
        output::kv("  Model mode", format!("{:?}", spec.model.mode));
        output::kv("  Training mode", format!("{:?}", spec.training.mode));
        println!();
        output::kv("  Train data", spec.data.train.display().to_string());
        output::kv("  Batch size", spec.data.batch_size.to_string());
        if let Some(seq_len) = spec.data.seq_len {
            output::kv("  Sequence length", seq_len.to_string());
        }
        println!();
        output::kv("  Optimizer", &spec.optimizer.name);
        output::kv("  Learning rate", format!("{:.2e}", spec.optimizer.lr));
        output::kv("  Epochs", spec.training.epochs.to_string());
        output::kv("  Warmup steps", spec.training.warmup_steps.to_string());
        if let Some(accum) = spec.training.gradient_accumulation {
            output::kv("  Gradient accumulation", accum.to_string());
        }
        if let Some(ref mp) = spec.training.mixed_precision {
            output::kv("  Mixed precision", mp);
        }
        if let Some(ref lora) = spec.lora {
            println!();
            output::kv("  LoRA rank", lora.rank.to_string());
            output::kv("  LoRA alpha", format!("{:.1}", lora.alpha));
        }

        println!();
        println!(
            "  {} Config validated, ready for apply",
            "READY".green().bold()
        );
    }

    Ok(())
}

/// Run `apr train apply` — execute a training plan.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_apply(
    _plan_file: Option<&std::path::Path>,
    config_path: Option<&std::path::Path>,
    task: &str,
    _data: Option<&std::path::Path>,
    _model_size: &str,
    _model_path: Option<&std::path::Path>,
    _num_classes: usize,
    output_dir: Option<&std::path::Path>,
    _strategy: &str,
    _budget: usize,
    _scout: bool,
    _max_epochs: usize,
    _manual_lr: Option<f32>,
    _manual_lora_rank: Option<usize>,
    _manual_batch_size: Option<usize>,
    json_output: bool,
    distributed: bool,
    world_size: Option<usize>,
    rank: Option<usize>,
    coordinator_addr: Option<&str>,
    deterministic: bool,
    seed: Option<u64>,
) -> Result<()> {
    match task {
        "pretrain" | "causal_lm" => run_apply_pretrain(
            config_path,
            output_dir,
            json_output,
            distributed,
            world_size,
            rank,
            coordinator_addr,
            deterministic,
            seed,
        ),
        "classify" => Err(classify_not_available()),
        _ => Err(CliError::ValidationFailed(format!(
            "Unknown task type: {task}. Supported: classify, pretrain"
        ))),
    }
}

/// Build a YAML distributed config section from CLI flags.
fn build_distributed_yaml(
    world_size: Option<usize>,
    rank: Option<usize>,
    coordinator_addr: Option<&str>,
) -> serde_yaml::Value {
    let mut m = serde_yaml::Mapping::new();
    let ws = world_size.unwrap_or(2);
    m.insert(
        serde_yaml::Value::String("world_size".into()),
        serde_yaml::Value::Number(serde_yaml::Number::from(ws as u64)),
    );
    let r = rank.unwrap_or(0);
    m.insert(
        serde_yaml::Value::String("rank".into()),
        serde_yaml::Value::Number(serde_yaml::Number::from(r as u64)),
    );
    let addr = coordinator_addr.unwrap_or("0.0.0.0:9000");
    m.insert(
        serde_yaml::Value::String("coordinator_addr".into()),
        serde_yaml::Value::String(addr.into()),
    );
    let role = if r == 0 { "coordinator" } else { "worker" };
    m.insert(
        serde_yaml::Value::String("role".into()),
        serde_yaml::Value::String(role.into()),
    );
    serde_yaml::Value::Mapping(m)
}

/// Patch a YAML training config with CLI overrides (distributed, deterministic, seed).
fn patch_yaml_config(
    config_path: &std::path::Path,
    output_dir: Option<&std::path::Path>,
    distributed: bool,
    world_size: Option<usize>,
    rank: Option<usize>,
    coordinator_addr: Option<&str>,
    deterministic: bool,
    seed: Option<u64>,
) -> Result<std::path::PathBuf> {
    let yaml_content = std::fs::read_to_string(config_path)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to read config: {e}")))?;
    let mut yaml_val: serde_yaml::Value = serde_yaml::from_str(&yaml_content)
        .map_err(|e| CliError::ValidationFailed(format!("Invalid YAML: {e}")))?;

    let training = yaml_val
        .get_mut("training")
        .ok_or_else(|| CliError::ValidationFailed("Missing 'training' section".into()))?;

    if let serde_yaml::Value::Mapping(training_map) = training {
        // -o/--output overrides training.output_dir. Until this existed the flag
        // was accepted, documented, and silently discarded.
        if let Some(dir) = output_dir {
            training_map.insert(
                serde_yaml::Value::String("output_dir".into()),
                serde_yaml::Value::String(dir.display().to_string()),
            );
        }
        if distributed {
            let dist = build_distributed_yaml(world_size, rank, coordinator_addr);
            training_map.insert(serde_yaml::Value::String("distributed".into()), dist);
        }
        if deterministic {
            training_map.insert(
                serde_yaml::Value::String("deterministic".into()),
                serde_yaml::Value::Bool(true),
            );
        }
        if let Some(s) = seed {
            training_map.insert(
                serde_yaml::Value::String("seed".into()),
                serde_yaml::Value::Number(serde_yaml::Number::from(s)),
            );
        }
    }

    let temp_path = std::env::temp_dir().join(format!(
        "apr-patched-config-{}-{}.yaml",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    let patched_yaml = serde_yaml::to_string(&yaml_val)
        .map_err(|e| CliError::ValidationFailed(format!("YAML serialize error: {e}")))?;
    std::fs::write(&temp_path, &patched_yaml)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to write temp config: {e}")))?;

    Ok(temp_path)
}

/// Print pretrain CLI summary to stderr.
fn print_pretrain_header(
    config_path: &std::path::Path,
    distributed: bool,
    world_size: Option<usize>,
    rank: Option<usize>,
    coordinator_addr: Option<&str>,
    deterministic: bool,
    seed: Option<u64>,
) {
    if distributed {
        output::header("apr train apply — Distributed Causal LM Pre-training");
    } else {
        output::header("apr train apply — Causal LM Pre-training");
    }
    println!();
    output::kv("  Config", config_path.display().to_string());
    if distributed {
        output::kv("  Mode", "distributed data-parallel (DDP)");
        if let Some(ws) = world_size {
            output::kv("  World size", ws.to_string());
        }
        if let Some(r) = rank {
            output::kv("  Rank", r.to_string());
        }
        if let Some(addr) = coordinator_addr {
            output::kv("  Coordinator", addr.to_string());
        }
    }
    if deterministic {
        output::kv("  Deterministic", "enabled (C-DETERM-001)");
    }
    if let Some(s) = seed {
        output::kv("  Seed", s.to_string());
    }
    println!();
}

/// Execute causal LM pre-training from YAML config (ALB-009).
#[allow(clippy::too_many_arguments)]
fn run_apply_pretrain(
    config_path: Option<&std::path::Path>,
    output_dir: Option<&std::path::Path>,
    json_output: bool,
    distributed: bool,
    world_size: Option<usize>,
    rank: Option<usize>,
    coordinator_addr: Option<&str>,
    deterministic: bool,
    seed: Option<u64>,
) -> Result<()> {
    let config_path = config_path.ok_or_else(|| {
        CliError::ValidationFailed("--config <yaml> is required for --task pretrain".to_string())
    })?;

    if !config_path.exists() {
        return Err(CliError::FileNotFound(config_path.to_path_buf()));
    }

    if !json_output {
        print_pretrain_header(
            config_path,
            distributed,
            world_size,
            rank,
            coordinator_addr,
            deterministic,
            seed,
        );
    }

    let needs_patch = distributed || deterministic || seed.is_some() || output_dir.is_some();

    if needs_patch {
        let temp_path = patch_yaml_config(
            config_path,
            output_dir,
            distributed,
            world_size,
            rank,
            coordinator_addr,
            deterministic,
            seed,
        )?;
        let mode = if distributed {
            "Distributed training"
        } else {
            "Training"
        };
        entrenar::config::train_from_yaml(&temp_path)
            .map_err(|e| CliError::ValidationFailed(format!("{mode} failed: {e}")))?;
        let _ = std::fs::remove_file(&temp_path);
    } else {
        entrenar::config::train_from_yaml(config_path)
            .map_err(|e| CliError::ValidationFailed(format!("Training failed: {e}")))?;
    }

    if json_output {
        let mode = if distributed { "distributed" } else { "single" };
        println!("{{\"status\":\"completed\",\"task\":\"pretrain\",\"mode\":\"{mode}\"}}");
    } else {
        println!();
        let label = if distributed {
            "Distributed pre-training"
        } else {
            "Pre-training"
        };
        println!("  {} {label} completed", "DONE".green().bold());
    }

    Ok(())
}

// ── Watch supervisor ────────────────────────────────────────────────────────

/// Run `apr train watch` — crash-resilient training supervisor.
///
/// Monitors a training process with automatic restart on crash, hang detection
/// via heartbeat staleness, GPU state capture, and exponential backoff.
/// Sovereign Rust replacement for train-guard.sh.
/// Mutable state for the watch supervisor loop.
struct WatchState {
    attempt: usize,
    backoff: u64,
    last_stable: std::time::Instant,
    use_blocking: bool,
}

pub(crate) fn run_watch(
    config_path: &std::path::Path,
    max_restarts: usize,
    heartbeat_timeout: u64,
    backoff_initial: u64,
    backoff_max: u64,
    json_output: bool,
) -> Result<()> {
    // GH-521: Warn that heartbeat timeout monitoring is not yet implemented
    if heartbeat_timeout > 0 && !json_output {
        eprintln!("Warning: --heartbeat-timeout is not yet implemented. Hang detection disabled.");
    }

    if !config_path.exists() {
        return Err(CliError::FileNotFound(config_path.to_path_buf()));
    }

    print_watch_header(
        config_path,
        max_restarts,
        heartbeat_timeout,
        backoff_initial,
        backoff_max,
        json_output,
    );

    let mut state = WatchState {
        attempt: 0,
        backoff: backoff_initial,
        last_stable: std::time::Instant::now(),
        use_blocking: false,
    };

    loop {
        state.attempt += 1;
        if state.attempt > max_restarts + 1 {
            return Err(watch_max_restarts_exceeded(max_restarts, json_output));
        }

        if !json_output {
            println!(
                "  {} Starting training (attempt {}/{})",
                "▶".green(),
                state.attempt,
                max_restarts + 1
            );
        }

        kill_stale_gpu_procs();
        match handle_watch_iteration(
            config_path,
            &mut state,
            backoff_initial,
            backoff_max,
            json_output,
        )? {
            Some(result) => return result,
            None => continue,
        }
    }
}

/// Run one iteration of the watch loop. Returns Some(result) to exit, None to continue.
fn handle_watch_iteration(
    config_path: &std::path::Path,
    state: &mut WatchState,
    backoff_initial: u64,
    backoff_max: u64,
    json_output: bool,
) -> Result<Option<Result<()>>> {
    let exit_status = run_training_process(config_path, state.use_blocking);

    match exit_status {
        Ok(status) if status.success() => Ok(Some(watch_success(state.attempt, json_output))),
        Ok(status) => {
            let action = handle_crash(
                config_path,
                state,
                status,
                backoff_initial,
                backoff_max,
                json_output,
            )?;
            if action == CrashAction::Fatal {
                Ok(Some(Err(CliError::ValidationFailed(
                    "Fatal error".to_string(),
                ))))
            } else {
                Ok(None)
            }
        }
        Err(e) => Ok(Some(Err(watch_spawn_failed(e, json_output)))),
    }
}

/// Print the watch header.
fn print_watch_header(
    config_path: &std::path::Path,
    max_restarts: usize,
    heartbeat_timeout: u64,
    backoff_initial: u64,
    backoff_max: u64,
    json_output: bool,
) {
    if !json_output {
        output::header("apr train watch — Training Supervisor");
        println!();
        output::kv("  Config", config_path.display().to_string());
        output::kv("  Max restarts", max_restarts.to_string());
        output::kv("  Heartbeat timeout", format!("{heartbeat_timeout}s"));
        output::kv("  Backoff", format!("{backoff_initial}s → {backoff_max}s"));
        println!();
    }
}

/// Handle max restarts exceeded.
fn watch_max_restarts_exceeded(max_restarts: usize, json_output: bool) -> CliError {
    let msg = format!("Max restarts ({max_restarts}) exceeded");
    if json_output {
        println!("{{\"status\":\"failed\",\"reason\":\"{msg}\"}}");
    } else {
        println!("  {} {msg}", "FATAL".red().bold());
    }
    CliError::ValidationFailed(msg)
}

/// Handle successful training completion.
fn watch_success(attempt: usize, json_output: bool) -> Result<()> {
    if json_output {
        println!("{{\"status\":\"completed\",\"attempts\":{attempt}}}");
    } else {
        println!();
        println!(
            "  {} Training completed successfully (attempt {attempt})",
            "DONE".green().bold()
        );
    }
    Ok(())
}

/// Handle spawn failure.
fn watch_spawn_failed(e: std::io::Error, json_output: bool) -> CliError {
    if !json_output {
        println!("  {} Failed to start training: {e}", "ERROR".red().bold());
    }
    CliError::ValidationFailed(format!("Cannot start training process: {e}"))
}

#[derive(PartialEq)]
enum CrashAction {
    Restart,
    Fatal,
}

/// Handle a training crash: classify, diagnose, maybe restart.
fn handle_crash(
    config_path: &std::path::Path,
    state: &mut WatchState,
    status: std::process::ExitStatus,
    backoff_initial: u64,
    backoff_max: u64,
    json_output: bool,
) -> Result<CrashAction> {
    let code = status.code().unwrap_or(-1);
    let classification = classify_exit_code(code);

    if !json_output {
        println!();
        println!(
            "  {} Training exited with code {code} ({classification})",
            "CRASH".red().bold()
        );
    }

    write_crash_report(config_path, state.attempt, code, classification);

    if classification == "signal" && state.attempt == 1 {
        state.use_blocking = true;
        if !json_output {
            println!(
                "  {} Enabling CUDA_LAUNCH_BLOCKING for diagnosis",
                "DIAG".yellow()
            );
        }
    }

    if classification == "fatal" {
        let msg = format!("Fatal error (exit code {code}), not restarting");
        if json_output {
            println!("{{\"status\":\"fatal\",\"exit_code\":{code}}}");
        } else {
            println!("  {} {msg}", "FATAL".red().bold());
        }
        return Ok(CrashAction::Fatal);
    }

    if state.last_stable.elapsed().as_secs() > 3600 {
        state.backoff = backoff_initial;
    }
    state.last_stable = std::time::Instant::now();

    if !json_output {
        println!(
            "  {} Waiting {}s before restart...",
            "⏳".dimmed(),
            state.backoff
        );
    }
    std::thread::sleep(std::time::Duration::from_secs(state.backoff));
    state.backoff = (state.backoff * 2).min(backoff_max);

    Ok(CrashAction::Restart)
}

/// Run the training process as a child.
fn run_training_process(
    config_path: &std::path::Path,
    blocking: bool,
) -> std::result::Result<std::process::ExitStatus, std::io::Error> {
    let apr = std::env::current_exe().unwrap_or_else(|_| std::path::PathBuf::from("apr"));
    let mut cmd = std::process::Command::new(&apr);
    cmd.args(["train", "apply", "--task", "pretrain", "--config"]);
    cmd.arg(config_path);
    cmd.env("RUST_BACKTRACE", "1");
    if blocking {
        cmd.env("CUDA_LAUNCH_BLOCKING", "1");
    }
    cmd.status()
}

/// Classify an exit code into a category.
fn classify_exit_code(code: i32) -> &'static str {
    match code {
        0 => "success",
        1 => "error",
        2 => "usage",
        134 => "sigabrt", // CUDA assertion
        135 => "sigbus",  // fatal
        137 => "oom",     // SIGKILL (OOM killer)
        139 => "sigsegv", // CUDA memory corruption
        _ if code > 128 => "signal",
        _ => "unknown",
    }
}

/// Kill any stale GPU processes from previous training runs.
fn kill_stale_gpu_procs() {
    // Best-effort: find processes using the GPU that match our training pattern
    let _ = std::process::Command::new("bash")
        .args([
            "-c",
            "pgrep -f 'apr.*train.*apply' | grep -v $$ | xargs -r kill 2>/dev/null",
        ])
        .status();
}

/// Write a JSON crash report.
fn write_crash_report(
    config_path: &std::path::Path,
    attempt: usize,
    exit_code: i32,
    classification: &str,
) {
    let reports_dir = std::path::Path::new("crash-reports");
    let _ = std::fs::create_dir_all(reports_dir);

    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let report = serde_json::json!({
        "timestamp": timestamp,
        "attempt": attempt,
        "config": config_path.display().to_string(),
        "exit_code": exit_code,
        "classification": classification,
        "gpu_state": capture_gpu_state(),
    });

    let filename = reports_dir.join(format!("crash-{timestamp}-attempt{attempt}.json"));
    let _ = std::fs::write(
        &filename,
        serde_json::to_string_pretty(&report).unwrap_or_default(),
    );
}

/// Capture GPU state via nvidia-smi.
fn capture_gpu_state() -> serde_json::Value {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=gpu_name,memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits",
        ])
        .output();

    match output {
        Ok(out) if out.status.success() => {
            let text = String::from_utf8_lossy(&out.stdout).trim().to_string();
            serde_json::json!({"nvidia_smi": text})
        }
        _ => serde_json::json!({"nvidia_smi": "unavailable"}),
    }
}

// ── Hyperparameter Sweep (R-027 Rust replacement) ───────────────────────────

/// The search `apr train sweep --strategy` selects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SweepStrategy {
    /// Deterministic grid over the hyperparameter axes.
    Grid,
    /// Seeded random sampling.
    Random,
}

impl std::fmt::Display for SweepStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Grid => "grid",
            Self::Random => "random",
        })
    }
}

/// Parse a `--strategy` value, rejecting anything that is not a real strategy.
pub(crate) fn parse_sweep_strategy(raw: &str) -> Result<SweepStrategy> {
    match raw.trim().to_lowercase().as_str() {
        "grid" => Ok(SweepStrategy::Grid),
        "random" => Ok(SweepStrategy::Random),
        other => Err(CliError::ValidationFailed(format!(
            "Unknown sweep strategy '{other}'. Supported: grid, random"
        ))),
    }
}

/// Run `apr train sweep` — generate hyperparameter sweep configs.
///
/// Creates N training configs with varied hyperparameters (grid or random).
/// Each output is a complete YAML that can be passed to `apr train apply`.
pub(crate) fn run_sweep(
    config_path: &std::path::Path,
    strategy: &str,
    num_configs: usize,
    output_dir: &std::path::Path,
    seed: u64,
    json_output: bool,
) -> Result<()> {
    // Reject an unknown strategy BEFORE writing anything. `--strategy gird` used
    // to fall through the `"random" | _` arm to a RANDOM search, print
    // "Strategy: gird" back as though it were real, and write the sweep files —
    // so a user who believed they had a grid search had a random one.
    let strategy = parse_sweep_strategy(strategy)?;

    if !config_path.exists() {
        return Err(CliError::FileNotFound(config_path.to_path_buf()));
    }

    let base_content = std::fs::read_to_string(config_path)
        .map_err(|e| CliError::ValidationFailed(format!("Cannot read config: {e}")))?;
    let base: serde_yaml::Value = serde_yaml::from_str(&base_content)
        .map_err(|e| CliError::ValidationFailed(format!("Invalid YAML: {e}")))?;

    std::fs::create_dir_all(output_dir)
        .map_err(|e| CliError::ValidationFailed(format!("Cannot create output dir: {e}")))?;

    if !json_output {
        output::header("apr train sweep — Hyperparameter Sweep Generator");
        println!();
        output::kv("  Base config", config_path.display().to_string());
        output::kv("  Strategy", strategy.to_string());
        output::kv("  Configs", num_configs.to_string());
        output::kv("  Output", output_dir.display().to_string());
        println!();
    }

    let configs = match strategy {
        SweepStrategy::Grid => generate_grid_configs(&base, num_configs),
        SweepStrategy::Random => generate_random_configs(&base, num_configs, seed),
    };

    let mut results = Vec::new();
    for (i, config) in configs.iter().enumerate() {
        let filename = output_dir.join(format!("sweep-{i:03}.yaml"));
        let yaml_str = serde_yaml::to_string(config)
            .map_err(|e| CliError::ValidationFailed(format!("YAML serialize error: {e}")))?;
        std::fs::write(&filename, &yaml_str).map_err(|e| {
            CliError::ValidationFailed(format!("Cannot write {}: {e}", filename.display()))
        })?;

        let lr = config
            .get("optimizer")
            .and_then(|o| o.get("lr"))
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let bs = config
            .get("data")
            .and_then(|d| d.get("batch_size"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0);

        results.push(serde_json::json!({
            "file": filename.display().to_string(),
            "lr": lr,
            "batch_size": bs,
        }));

        if !json_output {
            println!(
                "  [{}] {} (lr={:.2e}, bs={})",
                i,
                filename.display(),
                lr,
                bs
            );
        }
    }

    if json_output {
        let output = serde_json::json!({
            "strategy": strategy.to_string(),
            "configs_generated": configs.len(),
            "output_dir": output_dir.display().to_string(),
            "configs": results,
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&output).unwrap_or_default()
        );
    } else {
        println!();
        println!(
            "  {} Generated {} configs",
            "DONE".green().bold(),
            configs.len()
        );
    }

    Ok(())
}

/// Generate grid search configs over LR x batch_size x weight_decay.
fn generate_grid_configs(base: &serde_yaml::Value, max_configs: usize) -> Vec<serde_yaml::Value> {
    let lr_values = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3];
    let bs_values: &[u64] = &[2, 4, 8];
    let wd_values = [0.0, 0.01, 0.1];

    let mut configs = Vec::new();
    for &lr in &lr_values {
        for &bs in bs_values {
            for &wd in &wd_values {
                if configs.len() >= max_configs {
                    return configs;
                }
                let mut c = base.clone();
                set_yaml_f64(&mut c, &["optimizer", "lr"], lr);
                set_yaml_u64(&mut c, &["data", "batch_size"], bs);
                set_yaml_f64(&mut c, &["optimizer", "weight_decay"], wd);
                configs.push(c);
            }
        }
    }
    configs
}

/// Generate random search configs using LCG PRNG.
fn generate_random_configs(
    base: &serde_yaml::Value,
    num_configs: usize,
    seed: u64,
) -> Vec<serde_yaml::Value> {
    let mut rng_state = seed;
    let mut configs = Vec::new();

    for _ in 0..num_configs {
        let mut c = base.clone();

        // LR: log-uniform in [1e-5, 1e-2]
        let lr_log = -5.0 + lcg_f64(&mut rng_state) * 3.0; // [-5, -2]
        let lr = 10.0_f64.powf(lr_log);
        set_yaml_f64(&mut c, &["optimizer", "lr"], lr);

        // Batch size: uniform choice from [1, 2, 4, 8, 16]
        let bs_choices: &[u64] = &[1, 2, 4, 8, 16];
        let bs_idx = (lcg_f64(&mut rng_state) * bs_choices.len() as f64) as usize;
        let bs = bs_choices[bs_idx.min(bs_choices.len() - 1)];
        set_yaml_u64(&mut c, &["data", "batch_size"], bs);

        // Weight decay: log-uniform in [1e-3, 0.5]
        let wd_log = -3.0 + lcg_f64(&mut rng_state) * 2.7; // [-3, -0.3]
        let wd = 10.0_f64.powf(wd_log);
        set_yaml_f64(&mut c, &["optimizer", "weight_decay"], wd);

        // Warmup steps: uniform in [50, 2000]
        let warmup = 50 + (lcg_f64(&mut rng_state) * 1950.0) as u64;
        set_yaml_u64(&mut c, &["training", "warmup_steps"], warmup);

        configs.push(c);
    }
    configs
}

/// LCG pseudo-random: returns f64 in [0, 1).
fn lcg_f64(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 33) as f64 / (1u64 << 31) as f64
}

/// Set a nested YAML value (f64).
fn set_yaml_f64(root: &mut serde_yaml::Value, path: &[&str], val: f64) {
    let mut node = root;
    for (i, key) in path.iter().enumerate() {
        if i == path.len() - 1 {
            node[*key] = serde_yaml::Value::Number(serde_yaml::Number::from(val));
        } else {
            if node.get(*key).is_none() {
                node[*key] = serde_yaml::Value::Mapping(serde_yaml::Mapping::new());
            }
            node = &mut node[*key];
        }
    }
}

/// Set a nested YAML value (u64).
fn set_yaml_u64(root: &mut serde_yaml::Value, path: &[&str], val: u64) {
    let mut node = root;
    for (i, key) in path.iter().enumerate() {
        if i == path.len() - 1 {
            node[*key] = serde_yaml::Value::Number(serde_yaml::Number::from(val));
        } else {
            if node.get(*key).is_none() {
                node[*key] = serde_yaml::Value::Mapping(serde_yaml::Mapping::new());
            }
            node = &mut node[*key];
        }
    }
}

// ============================================================================
// Archive command (self-contained, no unpublished API deps)
// ============================================================================

/// Copy checkpoint files to output dir, computing BLAKE3 hashes.
fn copy_checkpoint_files(
    checkpoint_dir: &std::path::Path,
    output_dir: &std::path::Path,
    json_output: bool,
) -> Result<(Vec<serde_json::Value>, u64)> {
    let mut manifest_entries = Vec::new();
    let mut total_bytes: u64 = 0;

    for entry in std::fs::read_dir(checkpoint_dir)
        .map_err(|e| CliError::ValidationFailed(format!("Cannot read dir: {e}")))?
    {
        let entry =
            entry.map_err(|e| CliError::ValidationFailed(format!("Dir entry error: {e}")))?;
        let src = entry.path();
        if !src.is_file() {
            continue;
        }

        let filename = src
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        let dst = output_dir.join(&filename);

        let data = std::fs::read(&src).map_err(|e| {
            CliError::ValidationFailed(format!("Cannot read {}: {e}", src.display()))
        })?;
        let size = data.len() as u64;
        let hash = blake3::hash(&data).to_hex().to_string();

        std::fs::write(&dst, &data).map_err(|e| {
            CliError::ValidationFailed(format!("Cannot write {}: {e}", dst.display()))
        })?;

        manifest_entries.push(serde_json::json!({
            "file": filename,
            "size": size,
            "blake3": hash,
        }));
        total_bytes += size;

        if !json_output {
            println!(
                "  [COPY] {} ({}, BLAKE3: {}...)",
                filename,
                format_archive_size(size),
                &hash[..16]
            );
        }
    }
    Ok((manifest_entries, total_bytes))
}

pub(crate) fn run_archive(
    checkpoint_dir: &std::path::Path,
    output_dir: &std::path::Path,
    version: Option<&str>,
    notes: Option<&str>,
    json_output: bool,
) -> Result<()> {
    if !checkpoint_dir.is_dir() {
        return Err(CliError::ValidationFailed(format!(
            "Not a directory: {}",
            checkpoint_dir.display()
        )));
    }

    std::fs::create_dir_all(output_dir)
        .map_err(|e| CliError::ValidationFailed(format!("Cannot create output dir: {e}")))?;

    if !json_output {
        output::header("apr train archive — Checkpoint Release Bundle");
        println!();
        output::kv("  Source", checkpoint_dir.display().to_string());
        output::kv("  Output", output_dir.display().to_string());
        if let Some(v) = version {
            output::kv("  Version", v);
        }
        println!();
    }

    let (manifest_entries, total_bytes) =
        copy_checkpoint_files(checkpoint_dir, output_dir, json_output)?;

    let manifest = serde_json::json!({
        "format": "albor-checkpoint-archive",
        "version": version.unwrap_or("0.0.0"),
        "created": chrono::Utc::now().to_rfc3339(),
        "notes": notes.unwrap_or(""),
        "source": checkpoint_dir.display().to_string(),
        "files": manifest_entries,
        "total_bytes": total_bytes,
    });

    let manifest_path = output_dir.join("MANIFEST.json");
    std::fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&manifest).unwrap_or_default(),
    )
    .map_err(|e| CliError::ValidationFailed(format!("Cannot write manifest: {e}")))?;

    if json_output {
        println!(
            "{}",
            serde_json::to_string_pretty(&manifest).unwrap_or_default()
        );
    } else {
        println!();
        println!(
            "  [MANIFEST] {} ({} files, {})",
            manifest_path.display(),
            manifest_entries.len(),
            format_archive_size(total_bytes),
        );
        println!();
        println!("  {} Archive created", "DONE".green().bold());
    }

    Ok(())
}

/// Run `apr train submit` — place adapter jobs across a cluster and show launch commands.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_submit(
    cluster_path: &std::path::Path,
    model_path: &std::path::Path,
    adapters: &[String],
    rank: u32,
    epochs: u32,
    budget_mb: u64,
    dry_run: bool,
    json: bool,
) -> Result<()> {
    use entrenar::gpu::cluster::ClusterConfig;
    use entrenar::gpu::coordinator::build_launch_command;
    use entrenar::gpu::placement::{place_adapters, AdapterJob};

    let cluster = ClusterConfig::from_file(cluster_path)
        .map_err(|e| CliError::ValidationFailed(format!("failed to load cluster config: {e}")))?;

    if adapters.is_empty() {
        return Err(CliError::ValidationFailed(
            "at least one --adapter DATA:CHECKPOINT pair is required".to_string(),
        ));
    }

    let jobs: Vec<AdapterJob> = adapters
        .iter()
        .enumerate()
        .map(|(i, spec)| AdapterJob {
            adapter_idx: i,
            budget_mb,
            label: spec.clone(),
        })
        .collect();

    let placements = place_adapters(&cluster, &jobs, &[]);

    if json {
        let entries: Vec<serde_json::Value> = placements
            .iter()
            .map(|p| {
                let parts: Vec<&str> = adapters[p.adapter_idx].splitn(2, ':').collect();
                serde_json::json!({
                    "adapter_idx": p.adapter_idx,
                    "node": p.node_name,
                    "score": p.score,
                    "data": parts.first().unwrap_or(&""),
                    "checkpoint": parts.get(1).unwrap_or(&""),
                })
            })
            .collect();
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "cluster": cluster_path,
                "model": model_path,
                "placements": entries,
                "total_adapters": adapters.len(),
                "placed": placements.len(),
                "dry_run": dry_run,
            }))
            .unwrap_or_default()
        );
        return Ok(());
    }

    println!("{}", cluster);
    println!("--- Placement ---");
    for p in &placements {
        println!(
            "  Adapter {} ({}): -> {} (score: {:.3})",
            p.adapter_idx, adapters[p.adapter_idx], p.node_name, p.score
        );
    }

    let unplaced: Vec<_> = jobs
        .iter()
        .filter(|j| !placements.iter().any(|p| p.adapter_idx == j.adapter_idx))
        .collect();
    for j in &unplaced {
        println!(
            "  Adapter {} ({}): {} (no eligible node)",
            j.adapter_idx,
            j.label,
            "UNPLACED".red()
        );
    }

    println!();
    println!("--- Launch Commands ---");
    for p in &placements {
        if let Some(node) = cluster.find_node(&p.node_name) {
            let parts: Vec<&str> = adapters[p.adapter_idx].splitn(2, ':').collect();
            let data = parts.first().unwrap_or(&"data.jsonl");
            let ckpt = parts.get(1).unwrap_or(&"/tmp/adapter");
            let cmd = build_launch_command(
                node,
                model_path,
                std::path::Path::new(data),
                std::path::Path::new(ckpt),
                rank,
                epochs,
            );
            println!("  [{}] {cmd}", p.node_name);
        }
    }

    if dry_run {
        println!();
        println!(
            "  {} (dry run — no jobs launched)",
            "DRY RUN".yellow().bold()
        );
    }

    Ok(())
}

/// Run `apr train cluster-status` — display cluster node info and capacity.
pub(crate) fn run_cluster_status(cluster_path: &std::path::Path, json: bool) -> Result<()> {
    use entrenar::gpu::cluster::ClusterConfig;

    let cluster = ClusterConfig::from_file(cluster_path)
        .map_err(|e| CliError::ValidationFailed(format!("failed to load cluster config: {e}")))?;

    if json {
        let nodes: Vec<serde_json::Value> = cluster
            .nodes
            .iter()
            .map(|n| {
                serde_json::json!({
                    "name": n.name,
                    "host": n.host,
                    "transport": format!("{}", n.transport),
                    "gpus": n.gpus.iter().map(|g| serde_json::json!({
                        "uuid": g.uuid,
                        "type": g.gpu_type,
                        "vram_mb": g.vram_mb,
                        "usable_vram_mb": g.usable_vram_mb(),
                        "memory_type": format!("{:?}", g.memory_type),
                    })).collect::<Vec<_>>(),
                    "max_adapters": n.max_adapters,
                    "total_vram_mb": n.total_vram_mb(),
                    "usable_vram_mb": n.usable_vram_mb(),
                    "is_local": n.is_local(),
                    "is_cpu_only": n.is_cpu_only(),
                })
            })
            .collect();
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "cluster_config": cluster_path,
                "total_nodes": cluster.nodes.len(),
                "total_adapter_capacity": cluster.total_adapter_capacity(),
                "nodes": nodes,
            }))
            .unwrap_or_default()
        );
        return Ok(());
    }

    println!("{cluster}");
    println!(
        "Total adapter capacity: {}",
        cluster.total_adapter_capacity()
    );

    Ok(())
}

/// Tracked state for a single sweep config during halving.
struct HalvingEntry {
    path: std::path::PathBuf,
    best_ppl: f64,
    lr: f64,
    weight_decay: f64,
    warmup_steps: u64,
    eliminated_round: Option<usize>,
}

/// Run `apr train halving` — successive halving HPO (C-HPO-001).
///
/// Runs sweep configs, ranks by val_ppl, kills worst half each round.
/// Reports winner with μTransfer-scaled LR for target model width.
///
/// References:
/// - Hyperband: https://arxiv.org/abs/1603.06560
/// - μTransfer: https://arxiv.org/abs/2203.03466
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_halving(
    sweep_dir: &std::path::Path,
    rounds: usize,
    steps_per_round: usize,
    source_width: usize,
    target_width: usize,
    output_path: &std::path::Path,
    json_output: bool,
) -> Result<()> {
    if !sweep_dir.exists() {
        return Err(CliError::FileNotFound(sweep_dir.to_path_buf()));
    }

    let configs = discover_sweep_configs(sweep_dir)?;

    if !json_output {
        output::header("apr train halving — Successive Halving HPO (C-HPO-001)");
        println!();
        output::kv("  Configs", configs.len().to_string());
        output::kv("  Rounds", rounds.to_string());
        output::kv(
            "  Steps/round",
            format!("{steps_per_round} (doubles each round)"),
        );
        output::kv("  μTransfer", format!("{source_width} → {target_width}"));
        println!();
    }

    let mut results = parse_sweep_config_params(&configs)?;
    let mut survivors: Vec<usize> = (0..configs.len()).collect();
    let apr_path = std::env::current_exe().unwrap_or_else(|_| std::path::PathBuf::from("apr"));

    for round_idx in 0..rounds {
        survivors = run_halving_round(
            &apr_path,
            &mut results,
            &survivors,
            round_idx,
            steps_per_round * (1 << round_idx),
            json_output,
        );
    }

    emit_halving_results(
        &results,
        &survivors,
        source_width,
        target_width,
        rounds,
        steps_per_round,
        output_path,
        json_output,
    )
}

/// Discover sweep-*.yaml config files in a directory.
fn discover_sweep_configs(sweep_dir: &std::path::Path) -> Result<Vec<std::path::PathBuf>> {
    let mut configs: Vec<std::path::PathBuf> = std::fs::read_dir(sweep_dir)
        .map_err(|e| CliError::ValidationFailed(format!("Cannot read sweep dir: {e}")))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .map(|n| {
                    n.to_string_lossy().starts_with("sweep-")
                        && n.to_string_lossy().ends_with(".yaml")
                })
                .unwrap_or(false)
        })
        .collect();
    configs.sort();

    if configs.is_empty() {
        return Err(CliError::ValidationFailed(
            "No sweep-*.yaml files found in sweep directory".into(),
        ));
    }
    Ok(configs)
}

/// Parse optimizer/training params from each sweep config into `HalvingEntry` structs.
fn parse_sweep_config_params(configs: &[std::path::PathBuf]) -> Result<Vec<HalvingEntry>> {
    let mut entries = Vec::new();
    for c in configs {
        let content = std::fs::read_to_string(c)
            .map_err(|e| CliError::ValidationFailed(format!("Cannot read {}: {e}", c.display())))?;
        let yaml: serde_yaml::Value = serde_yaml::from_str(&content).map_err(|e| {
            CliError::ValidationFailed(format!("Invalid YAML {}: {e}", c.display()))
        })?;
        entries.push(HalvingEntry {
            path: c.clone(),
            best_ppl: f64::INFINITY,
            lr: yaml["optimizer"]["lr"].as_f64().unwrap_or(0.0),
            weight_decay: yaml["optimizer"]["weight_decay"].as_f64().unwrap_or(0.0),
            warmup_steps: yaml["training"]["warmup_steps"].as_u64().unwrap_or(0),
            eliminated_round: None,
        });
    }
    Ok(entries)
}

/// Parse val_ppl from "[eval] step=N val_loss=X val_ppl=Y" lines.
fn parse_best_ppl(output: &str) -> f64 {
    let mut best = f64::INFINITY;
    for line in output.lines() {
        if let Some(idx) = line.find("val_ppl=") {
            let rest = &line[idx + 8..];
            let end = rest
                .find(|c: char| !c.is_ascii_digit() && c != '.')
                .unwrap_or(rest.len());
            if let Ok(ppl) = rest[..end].parse::<f64>() {
                if ppl < best {
                    best = ppl;
                }
            }
        }
    }
    best
}

/// Execute one halving round: update configs, run survivors, rank, eliminate worst half.
/// Returns the surviving indices after elimination.
fn run_halving_round(
    apr_path: &std::path::Path,
    results: &mut [HalvingEntry],
    survivors: &[usize],
    round_idx: usize,
    steps: usize,
    json_output: bool,
) -> Vec<usize> {
    use std::process::Command;

    let n_survive = std::cmp::max(1, survivors.len() / 2);

    if !json_output {
        println!("═══ Round {round_idx} ═══");
        println!(
            "  Survivors: {}, steps: {steps}, keep: {n_survive}",
            survivors.len()
        );
        println!();
    }

    // Update max_steps in each survivor config
    for &idx in survivors {
        let path = &results[idx].path;
        let content = std::fs::read_to_string(path).unwrap_or_default();
        if let Ok(mut yaml) = serde_yaml::from_str::<serde_yaml::Value>(&content) {
            set_yaml_u64(&mut yaml, &["training", "max_steps"], steps as u64);
            let stem = path.file_stem().unwrap_or_default().to_string_lossy();
            let out_dir = format!("./checkpoints/halving-{stem}");
            yaml["training"]["output_dir"] = serde_yaml::Value::String(out_dir);
            if let Ok(s) = serde_yaml::to_string(&yaml) {
                let _ = std::fs::write(path, s);
            }
        }
    }

    // Run each survivor and collect scores
    let mut round_scores: Vec<(usize, f64)> = Vec::new();
    for &idx in survivors {
        let entry = &results[idx];
        let name = entry.path.file_name().unwrap_or_default().to_string_lossy();

        if !json_output {
            print!("  Running {name} (lr={:.2e})...", entry.lr);
            use std::io::Write;
            std::io::stdout().flush().ok();
        }

        let start = std::time::Instant::now();
        let cmd_output = Command::new(apr_path)
            .args(["train", "apply", "--task", "pretrain", "--config"])
            .arg(&entry.path)
            .output();

        let elapsed = start.elapsed().as_secs();
        let mut best_ppl = f64::INFINITY;
        if let Ok(out) = cmd_output {
            let combined = String::from_utf8_lossy(&out.stdout).to_string()
                + &String::from_utf8_lossy(&out.stderr);
            best_ppl = parse_best_ppl(&combined);
        }

        if best_ppl < results[idx].best_ppl {
            results[idx].best_ppl = best_ppl;
        }
        round_scores.push((idx, best_ppl));

        if !json_output {
            if best_ppl.is_finite() {
                println!(" val_ppl={best_ppl:.2} ({elapsed}s)");
            } else {
                println!(" no eval ({elapsed}s)");
            }
        }
    }

    // Rank by best_ppl (lower is better), inf sorts last
    round_scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let kept: Vec<usize> = round_scores
        .iter()
        .take(n_survive)
        .map(|(i, _)| *i)
        .collect();
    for &(idx, _) in round_scores.iter().skip(n_survive) {
        results[idx].eliminated_round = Some(round_idx);
    }

    if !json_output {
        println!();
        let kept_names: Vec<String> = kept
            .iter()
            .map(|i| {
                results[*i]
                    .path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string()
            })
            .collect();
        println!("  Kept: {kept_names:?}");
        println!();
    }

    kept
}

/// Emit halving results: print winner, build JSON, write to output file.
#[allow(clippy::too_many_arguments)]
fn emit_halving_results(
    results: &[HalvingEntry],
    survivors: &[usize],
    source_width: usize,
    target_width: usize,
    rounds: usize,
    steps_per_round: usize,
    output_path: &std::path::Path,
    json_output: bool,
) -> Result<()> {
    let winner = &results[survivors[0]];
    let target_lr = winner.lr * (source_width as f64 / target_width as f64);
    let winner_name = winner
        .path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();

    if !json_output {
        println!("═══ WINNER ═══");
        println!("  Config: {winner_name}");
        println!("  Proxy LR: {:.4e}", winner.lr);
        println!("  Best val_ppl: {:.2}", winner.best_ppl);
        println!("  Weight decay: {:.4}", winner.weight_decay);
        println!("  Warmup steps: {}", winner.warmup_steps);
        println!();
        println!("  μTransfer LR ({source_width}→{target_width}):");
        println!(
            "    lr_target = {:.4e} × ({source_width}/{target_width})",
            winner.lr
        );
        println!("    lr_target = {target_lr:.4e}");
        println!();
    }

    let output_json = build_halving_json(
        results,
        &winner_name,
        winner,
        target_lr,
        source_width,
        target_width,
        rounds,
        steps_per_round,
    );

    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::write(
        output_path,
        serde_json::to_string_pretty(&output_json).unwrap_or_default(),
    )
    .map_err(|e| CliError::ValidationFailed(format!("Cannot write results: {e}")))?;

    if json_output {
        println!(
            "{}",
            serde_json::to_string_pretty(&output_json).unwrap_or_default()
        );
    } else {
        println!("  Results saved to {}", output_path.display());
    }

    Ok(())
}

/// Build the JSON output structure for halving results.
// serde_json::json!() macro uses infallible unwrap internally
#[allow(clippy::too_many_arguments, clippy::disallowed_methods)]
fn build_halving_json(
    results: &[HalvingEntry],
    winner_name: &str,
    winner: &HalvingEntry,
    target_lr: f64,
    source_width: usize,
    target_width: usize,
    rounds: usize,
    steps_per_round: usize,
) -> serde_json::Value {
    let all_results: Vec<serde_json::Value> = results
        .iter()
        .map(|e| {
            serde_json::json!({
                "config": e.path.file_name().unwrap_or_default().to_string_lossy(),
                "lr": e.lr,
                "best_ppl": if e.best_ppl.is_finite() { serde_json::json!(e.best_ppl) } else { serde_json::json!(null) },
                "weight_decay": e.weight_decay,
                "warmup_steps": e.warmup_steps,
                "eliminated_round": e.eliminated_round,
            })
        })
        .collect();

    serde_json::json!({
        "winner": {
            "config": winner_name,
            "proxy_lr": winner.lr,
            "target_lr": target_lr,
            "best_ppl": if winner.best_ppl.is_finite() { serde_json::json!(winner.best_ppl) } else { serde_json::json!(null) },
            "weight_decay": winner.weight_decay,
            "warmup_steps": winner.warmup_steps,
            "source_width": source_width,
            "target_width": target_width,
        },
        "all_results": all_results,
        "settings": {
            "rounds": rounds,
            "steps_per_round": steps_per_round,
        },
    })
}

fn format_archive_size(bytes: u64) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.1} GB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.1} MB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes} B")
    }
}

include!("train_tests.rs");
