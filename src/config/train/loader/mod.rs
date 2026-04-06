#![allow(dead_code)]
//! Main entry points for YAML-based training

// Default model architecture constants (Qwen2.5-Coder-0.5B)
const QWEN_HIDDEN_SIZE: usize = 896;
const QWEN_NUM_ATTENTION_HEADS: usize = 14;
const QWEN_NUM_KV_HEADS: usize = 2;
const QWEN_INTERMEDIATE_SIZE: usize = 4864;
const QWEN_NUM_HIDDEN_LAYERS: usize = 24;
const QWEN_VOCAB_SIZE: usize = 151936;
const QWEN_MAX_POSITION_EMBEDDINGS: usize = 32768;
const QWEN_ROPE_THETA: f64 = 1_000_000.0;

use super::batches::load_training_batches;
use crate::config::schema::{ModelMode, TrainSpec};
use crate::config::validate::validate_config;
use crate::error::{Error, Result};
use crate::monitor::tui::state::{TrainingSnapshot, TrainingState, TrainingStatus};
use crate::storage::{ExperimentStorage, ParameterValue, RunStatus, SqliteBackend};
use crate::tokenizer::HfTokenizer;
use crate::trace::TRACER;
#[cfg(feature = "cuda")]
use crate::train::CudaTransformerTrainer;
use crate::train::{LMBatch, TransformerTrainConfig, TransformerTrainer};
use crate::transformer::{
    load_safetensors_weights, Architecture, ModelArchitecture, Transformer, TransformerConfig,
};
use crate::yaml_mode;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Train a model from YAML configuration file
///
/// This is the main entry point for declarative training. It:
/// 1. Loads and parses the YAML config
/// 2. Validates the configuration
/// 3. Dispatches to appropriate trainer (Tabular or Transformer)
/// 4. Runs the training loop
/// 5. Saves the final model
///
/// # Example
///
/// ```no_run
/// use entrenar::config::train_from_yaml;
///
/// let model = train_from_yaml("config.yaml")?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn train_from_yaml<P: AsRef<Path>>(config_path: P) -> Result<()> {
    let spec = load_config(config_path)?;

    // Dispatch based on model mode
    match spec.model.mode {
        ModelMode::Transformer => train_transformer_from_spec(&spec),
        ModelMode::Tabular => train_tabular_from_spec(&spec),
    }
}

/// Build a TransformerTrainConfig from YAML spec, wiring all hyperparameters.
fn build_train_config(
    model_config: crate::transformer::TransformerConfig,
    spec: &TrainSpec,
) -> TransformerTrainConfig {
    let mut config = TransformerTrainConfig::new(model_config)
        .with_lr(spec.optimizer.lr)
        .with_warmup_steps(spec.training.warmup_steps)
        .with_max_seq_len({
            let seq_len = spec.data.seq_len.unwrap_or_else(|| {
                eprintln!("Warning: seq_len not specified in config, defaulting to 512");
                512
            });
            seq_len
        });

    if let Some(clip) = spec.training.grad_clip {
        config = config.with_grad_clip(clip);
    }

    // Wire optimizer hyperparameters from YAML (ALB-040)
    if let Some(v) = spec.optimizer.params.get("beta2").and_then(serde_json::Value::as_f64) {
        config = config.with_beta2(v as f32);
    }
    if let Some(v) = spec.optimizer.params.get("weight_decay").and_then(serde_json::Value::as_f64) {
        config = config.with_weight_decay(v as f32);
    }

    if let Some(accum) = spec.training.gradient_accumulation {
        config = config.with_accumulation_steps(accum);
        if accum > 1 {
            let eff_batch = spec.data.batch_size * accum * spec.data.seq_len.unwrap_or(1024);
            println!("  Gradient accumulation: {accum} (effective batch: {eff_batch} tokens/step)");
        }
    }

    if let Some(max_steps) = spec.training.max_steps {
        config = config.with_max_steps(max_steps);
    }

    // Enable mixed precision if specified
    if let Some(ref precision) = spec.training.mixed_precision {
        match precision.as_str() {
            "bf16" => config = config.with_bf16(),
            "fp16" => config = config.with_fp16(),
            "fp32" => {}
            other => {
                eprintln!("Warning: unknown mixed_precision value '{other}', defaulting to fp32");
            }
        }
    }

    // R-021: Activation checkpointing (gradient recomputation)
    if let Some(num_segments) = spec.training.checkpoints {
        config = config.with_checkpointing(num_segments);
    }

    // R-084: Bitwise deterministic training (C-DETERM-001)
    if spec.training.deterministic {
        config = config.with_deterministic(true);
    }
    if let Some(seed) = spec.training.seed {
        config = config.with_seed(seed);
    }

    // KAIZEN-047: Step profiler (0 = disabled)
    if spec.training.profile_interval > 0 {
        config = config.with_profile_interval(spec.training.profile_interval);
    }

    // ENT-LoRA-001: Wire LoRA config from YAML spec
    if let Some(ref lora) = spec.lora {
        config = config.with_lora(lora.rank, lora.alpha, lora.target_modules.clone());
        // ENT-LoRA-006: LoRA+ ratio from YAML
        if lora.lora_plus_ratio != 1.0 {
            config = config.with_lora_plus_ratio(lora.lora_plus_ratio);
        }
        // ENT-LoRA-008: Double quantization from YAML
        if lora.double_quantize {
            config = config.with_double_quantize(true);
        }
        // ENT-263: NF4 quantization for QLoRA pretraining
        if lora.quantize_base {
            config = config.with_quantize_nf4(true);
        }
    }

    // Wire distributed config from YAML (#133)
    if let Some(ref dist) = spec.training.distributed {
        use crate::train::{DistributedBackend, DistributedRole, DistributedTrainConfig};

        let role = match dist.role.as_str() {
            "worker" => DistributedRole::Worker,
            _ => DistributedRole::Coordinator,
        };
        let backend = match dist.backend.as_str() {
            "cuda" => DistributedBackend::Cuda,
            "wgpu" => DistributedBackend::Wgpu,
            _ => DistributedBackend::Auto,
        };
        let addr: std::net::SocketAddr =
            dist.coordinator_addr.parse().unwrap_or_else(|_| "0.0.0.0:9000".parse().unwrap());

        config = config.with_distributed(DistributedTrainConfig {
            world_size: dist.world_size,
            rank: dist.rank,
            local_rank: dist.local_rank,
            role,
            coordinator_addr: addr,
            backend,
        });
    }

    config
}

/// Train a transformer model (LLM) from spec
///
/// Uses TransformerTrainer with CausalLMLoss for language modeling.
fn train_transformer_from_spec(spec: &TrainSpec) -> Result<()> {
    println!("✓ Config loaded and validated (Transformer mode)");
    println!("  Model: {}", spec.model.path.display());
    println!("  Optimizer: {} (lr={})", spec.optimizer.name, spec.optimizer.lr);
    println!("  Batch size: {}", spec.data.batch_size);
    println!("  Epochs: {}", spec.training.epochs);
    println!("  Training mode: {:?}", spec.training.mode);

    if let Some(lora) = &spec.lora {
        println!("  LoRA: rank={}, alpha={}", lora.rank, lora.alpha);
        if lora.quantize_base {
            println!("  QLoRA: NF4 quantized base weights (~8x VRAM compression)");
        }
    }
    println!();

    // Build TransformerConfig from spec
    let model_config = build_transformer_config_from_spec(spec)?;

    // Resolve model path (downloads from HF Hub if repo ID)
    let resolved_path = resolve_model_path(&spec.model.path)?;

    // C-INIT-001: Set init seed BEFORE model construction (entrenar#309)
    crate::transformer::init::set_init_seed(spec.training.seed.unwrap_or(42));

    // Try to load model weights if path exists (ENT-117)
    // ALB-097: Check output_dir first for checkpoint resume, then model_path for initial weights
    #[cfg(feature = "cuda")]
    let (transformer, checkpoint_step) =
        load_transformer_model(&resolved_path, &model_config, &spec.training.output_dir)?;
    #[cfg(not(feature = "cuda"))]
    let (transformer, _checkpoint_step) =
        load_transformer_model(&resolved_path, &model_config, &spec.training.output_dir)?;

    // Build TransformerTrainConfig from YAML spec fields
    let train_config = build_train_config(model_config, spec);

    // Apply deterministic settings before any CUDA operations
    train_config.apply_deterministic_settings();

    // Load training data as LMBatches (supports tokenizer + text data)
    println!("Loading training data...");
    let batches = load_lm_batches(spec)?;
    println!("✓ {} LM batches created", batches.len());
    println!();

    // Try CUDA-resident training first (ALB-040), fall back to CPU
    #[cfg(feature = "cuda")]
    if train_config.use_cuda {
        let cuda_config = train_config.clone();
        let cuda_result = match transformer {
            Some(loaded_model) => CudaTransformerTrainer::with_model(loaded_model, cuda_config),
            None => CudaTransformerTrainer::new(cuda_config),
        };

        match cuda_result {
            Ok(mut cuda_trainer) => {
                // Restore step counter from checkpoint for LR schedule + AdamW bias correction
                if checkpoint_step > 0 {
                    cuda_trainer.set_initial_step(checkpoint_step);
                    println!(
                        "  Resumed at step {checkpoint_step} (lr={:.2e})",
                        cuda_trainer.current_lr()
                    );
                    // ALB-096: Try APR optimizer state first, then fall back to JSON
                    let apr_loaded = find_latest_apr_checkpoint(&spec.training.output_dir)
                        .is_some_and(|p| {
                            // ENT-276: Restore LoRA adapter weights from APR checkpoint
                            let (restored, total) = cuda_trainer.restore_lora_from_apr(&p);
                            if restored > 0 {
                                println!("  ✓ LoRA adapters restored ({restored}/{total} layers)");
                            }
                            cuda_trainer.load_optimizer_state_apr(&p)
                        });
                    if apr_loaded {
                        println!("  ✓ Embedding optimizer state restored (APR)");
                    } else if cuda_trainer.load_optimizer_state(&spec.training.output_dir) {
                        println!("  ✓ Embedding optimizer state restored (JSON)");
                    }
                }
                println!("✓ CudaTransformerTrainer initialized (GPU: {})", cuda_trainer.gpu_name());
                // #133: Dispatch to distributed training loop if distributed config present
                if train_config.distributed.is_some() {
                    return train_loop_cuda_distributed(cuda_trainer, &batches, spec);
                }
                return train_loop_cuda(&mut cuda_trainer, &batches, spec);
            }
            Err(e) => {
                eprintln!("Warning: CUDA training failed ({e}), falling back to CPU");
                // transformer was consumed — rebuild from config
                let mut trainer = TransformerTrainer::new(train_config);
                println!("✓ TransformerTrainer initialized (CPU fallback)");
                println!("  Mixed precision: {}", trainer.is_mixed_precision());
                println!("  Checkpointing: {}", trainer.is_checkpointing());
                println!();
                return train_loop_cpu(&mut trainer, &batches, spec);
            }
        }
    }

    // CPU-only path (use_cuda=false or no CUDA feature)
    let mut trainer = if let Some(loaded_model) = transformer {
        TransformerTrainer::with_model(loaded_model, train_config)
    } else {
        TransformerTrainer::new(train_config)
    };
    println!("✓ TransformerTrainer initialized (CPU)");
    println!("  Mixed precision: {}", trainer.is_mixed_precision());
    println!("  Checkpointing: {}", trainer.is_checkpointing());
    println!();

    train_loop_cpu(&mut trainer, &batches, spec)
}

/// Training loop for CPU TransformerTrainer (ALB-045, ALB-055/056)
fn train_loop_cpu(
    trainer: &mut TransformerTrainer,
    batches: &[LMBatch],
    spec: &TrainSpec,
) -> Result<()> {
    println!("Starting transformer training (CPU)...");
    println!();

    TRACER.enable();
    TRACER.clear();

    let num_batches = batches.len();
    let start_time = std::time::Instant::now();
    let log_interval = (num_batches / 100).clamp(1, 100);

    // ALB-045: Initialize training state IPC for `apr monitor`
    let state = TrainingState::new(&spec.training.output_dir);
    let start_ms = now_ms();
    let total_epochs = spec.training.epochs;

    // ALB-055/056: Open SQLite experiment tracking (local + global)
    let mut tracker = PretrainTracker::open(spec, "CPU");

    write_training_snapshot(
        &state,
        start_ms,
        0,
        total_epochs,
        0,
        num_batches,
        0.0,
        &[],
        0.0,
        0.0,
        TrainingStatus::Initializing,
        spec,
        "CPU",
    );

    if let Some(max_steps) = spec.training.max_steps {
        println!("  max_steps: {max_steps} (will stop early when reached)");
    }

    let mut loss_history: Vec<f32> = Vec::new();

    for epoch in 0..spec.training.epochs {
        let epoch_start = std::time::Instant::now();
        let avg_loss =
            trainer.train_epoch_with_callback(batches, |batch_idx, batch_loss, trainer| {
                loss_history.push(batch_loss);
                if loss_history.len() > 100 {
                    loss_history.remove(0);
                }

                if (batch_idx + 1) % log_interval == 0 || batch_idx == 0 {
                    let elapsed = epoch_start.elapsed().as_secs_f64();
                    let batches_done = batch_idx + 1;
                    let seq_len = spec.data.seq_len.unwrap_or(128);
                    let tokens_done = batches_done * spec.data.batch_size * seq_len;
                    let batch_per_sec = batches_done as f64 / elapsed.max(0.001);
                    let remaining = (num_batches - batches_done) as f64 / batch_per_sec.max(0.001);
                    let tok_per_sec = tokens_done as f64 / elapsed.max(0.001);
                    println!(
                        "  [{}/{} batches] step={} loss={:.4} lr={:.2e} tok/s={:.0} eta={:.0}s",
                        batches_done,
                        num_batches,
                        trainer.step(),
                        batch_loss,
                        trainer.current_lr(),
                        tok_per_sec,
                        remaining,
                    );

                    // ALB-045: Write snapshot for `apr monitor`
                    write_training_snapshot(
                        &state,
                        start_ms,
                        epoch + 1,
                        total_epochs,
                        trainer.step(),
                        num_batches,
                        batch_loss,
                        &loss_history,
                        trainer.current_lr(),
                        tok_per_sec as f32,
                        TrainingStatus::Running,
                        spec,
                        "CPU",
                    );

                    // ALB-055/056: Log step metrics to SQLite
                    tracker.log_step(
                        trainer.step() as u64,
                        batch_loss,
                        trainer.current_lr(),
                        tok_per_sec as f32,
                    );
                }
            });
        let ppl = crate::train::perplexity(avg_loss);
        println!(
            "Epoch {}/{}: loss={:.6}, perplexity={:.2}, time={:.1}s",
            epoch + 1,
            spec.training.epochs,
            avg_loss,
            ppl,
            epoch_start.elapsed().as_secs_f64(),
        );

        if trainer.reached_max_steps() {
            println!(
                "Reached max_steps={}, stopping training.",
                spec.training.max_steps.unwrap_or(0)
            );
            break;
        }
    }

    let total_time = start_time.elapsed();
    println!("Total training time: {:.1}s", total_time.as_secs_f64());
    println!("{}", TRACER.report());

    // ALB-045: Write final "Completed" snapshot
    let final_loss = trainer.metrics.losses.last().copied().unwrap_or(0.0);
    write_training_snapshot(
        &state,
        start_ms,
        total_epochs,
        total_epochs,
        trainer.step(),
        num_batches,
        final_loss,
        &loss_history,
        trainer.current_lr(),
        0.0,
        TrainingStatus::Completed,
        spec,
        "CPU",
    );

    // ALB-055/056: Mark run as completed in SQLite
    tracker.complete();

    save_trained_model_cpu(trainer, spec)
}

/// Get current Unix timestamp in milliseconds
fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

/// Query live GPU telemetry via nvidia-smi CLI (ALB-046)
///
/// Shells out to `nvidia-smi --query-gpu` with CSV output and parses
/// the result into GpuTelemetry. Zero-dependency approach — nvidia-smi
/// is always available when CUDA is. Returns None if nvidia-smi fails.
fn query_gpu_telemetry(device_name: &str) -> Option<crate::monitor::tui::state::GpuTelemetry> {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let line = stdout.lines().next()?.trim();
    let fields: Vec<&str> = line.split(',').map(str::trim).collect();
    if fields.len() < 6 {
        return None;
    }

    Some(crate::monitor::tui::state::GpuTelemetry {
        device_name: device_name.to_string(),
        utilization_percent: fields[0].parse().unwrap_or(0.0),
        vram_used_gb: fields[1].parse::<f32>().unwrap_or(0.0) / 1024.0, // MiB → GiB
        vram_total_gb: fields[2].parse::<f32>().unwrap_or(0.0) / 1024.0,
        temperature_celsius: fields[3].parse().unwrap_or(0.0),
        power_watts: fields[4].parse().unwrap_or(0.0),
        power_limit_watts: fields[5].parse().unwrap_or(0.0),
        processes: Vec::new(),
    })
}

/// Write a TrainingSnapshot to training_state.json (ALB-045)
///
/// This is the IPC mechanism that `apr monitor` reads. Called on every
/// log interval so the TUI stays current. Uses atomic write (tmp+rename)
/// via TrainingState::write().
fn write_training_snapshot(
    state: &TrainingState,
    start_ms: u64,
    epoch: usize,
    total_epochs: usize,
    step: usize,
    steps_per_epoch: usize,
    loss: f32,
    loss_history: &[f32],
    lr: f32,
    tokens_per_second: f32,
    status: TrainingStatus,
    spec: &TrainSpec,
    gpu_name: &str,
) {
    let snapshot = TrainingSnapshot {
        timestamp_ms: now_ms(),
        epoch,
        total_epochs,
        step,
        steps_per_epoch,
        loss,
        loss_history: loss_history.to_vec(),
        learning_rate: lr,
        lr_history: Vec::new(),
        gradient_norm: 0.0, // not tracked per-batch in current trainer
        tokens_per_second,
        start_timestamp_ms: start_ms,
        gpu: query_gpu_telemetry(gpu_name).or_else(|| {
            Some(crate::monitor::tui::state::GpuTelemetry {
                device_name: gpu_name.to_string(),
                ..Default::default()
            })
        }),
        sample: None,
        status,
        experiment_id: spec.training.output_dir.display().to_string(),
        model_name: spec.model.path.display().to_string(),
        model_path: spec.model.path.display().to_string(),
        optimizer_name: spec.optimizer.name.clone(),
        batch_size: spec.data.batch_size,
        checkpoint_path: spec.training.output_dir.display().to_string(),
        executable_path: String::new(),
        accuracy: 0.0,
        samples_per_second: 0.0,
    };
    if let Err(e) = state.write(&snapshot) {
        eprintln!("[ALB-045] Failed to write training_state.json: {e}");
    }
}

// =============================================================================
// SQLite Experiment Tracking (ALB-055/056)
// =============================================================================

/// Best-effort experiment tracker for pretrain loops.
///
/// Opens two SQLite databases:
/// - **Local**: `<output_dir>/.entrenar/experiments.db` (per-experiment metrics history)
/// - **Global**: `~/.entrenar/experiments.db` (cross-machine experiment registry)
///
/// All operations are best-effort — storage failures never block training.
struct PretrainTracker {
    local: Option<SqliteBackend>,
    global: Option<SqliteBackend>,
    run_id: Option<String>,
    global_run_id: Option<String>,
}

impl PretrainTracker {
    /// Open both local and global SQLite stores, create experiment + run.
    fn open(spec: &TrainSpec, device: &str) -> Self {
        let exp_name =
            spec.training.output_dir.file_name().and_then(|n| n.to_str()).unwrap_or("pretrain");

        let config_json = serde_json::json!({
            "task": "pretrain",
            "model": spec.model.path.display().to_string(),
            "optimizer": &spec.optimizer.name,
            "lr": spec.optimizer.lr,
            "epochs": spec.training.epochs,
            "batch_size": spec.data.batch_size,
            "seq_len": spec.data.seq_len,
            "max_steps": spec.training.max_steps,
            "device": device,
            "output_dir": spec.training.output_dir.display().to_string(),
        });

        // Local store: in the output/checkpoint directory
        let local = SqliteBackend::open_project(&spec.training.output_dir).ok();

        // Global store: ~/.entrenar/experiments.db
        let global = dirs::home_dir().map(|h| h.join(".entrenar")).and_then(|p| {
            fs::create_dir_all(&p).ok()?;
            SqliteBackend::open(p.join("experiments.db").to_string_lossy().as_ref()).ok()
        });

        let mut tracker = Self { local, global, run_id: None, global_run_id: None };

        // Create experiment + run in local store
        if let Some(store) = tracker.local.as_mut() {
            if let Ok(eid) = store.create_experiment(exp_name, Some(config_json.clone())) {
                if let Ok(rid) = store.create_run(&eid) {
                    let _ = store.start_run(&rid);
                    log_run_params(store, &rid, spec, device);
                    tracker.run_id = Some(rid);
                }
            }
        }

        // Create experiment + run in global store
        if let Some(store) = tracker.global.as_mut() {
            if let Ok(eid) = store.create_experiment(exp_name, Some(config_json)) {
                if let Ok(rid) = store.create_run(&eid) {
                    let _ = store.start_run(&rid);
                    log_run_params(store, &rid, spec, device);
                    tracker.global_run_id = Some(rid);
                }
            }
        }

        tracker
    }

    /// Log a training step's metrics to both local and global stores.
    fn log_step(&mut self, step: u64, loss: f32, lr: f32, tok_per_sec: f32) {
        for (store, run_id) in [
            (self.local.as_mut(), self.run_id.as_deref()),
            (self.global.as_mut(), self.global_run_id.as_deref()),
        ] {
            if let (Some(s), Some(rid)) = (store, run_id) {
                let _ = s.log_metric(rid, "loss", step, f64::from(loss));
                let _ = s.log_metric(rid, "learning_rate", step, f64::from(lr));
                let _ = s.log_metric(rid, "tokens_per_second", step, f64::from(tok_per_sec));
            }
        }
    }

    /// Mark training as completed in both stores.
    fn complete(&mut self) {
        for (store, run_id) in [
            (self.local.as_mut(), self.run_id.as_deref()),
            (self.global.as_mut(), self.global_run_id.as_deref()),
        ] {
            if let (Some(s), Some(rid)) = (store, run_id) {
                let _ = s.complete_run(rid, RunStatus::Success);
            }
        }
    }

    /// Mark training as failed in both stores.
    #[allow(dead_code)]
    fn fail(&mut self) {
        for (store, run_id) in [
            (self.local.as_mut(), self.run_id.as_deref()),
            (self.global.as_mut(), self.global_run_id.as_deref()),
        ] {
            if let (Some(s), Some(rid)) = (store, run_id) {
                let _ = s.complete_run(rid, RunStatus::Failed);
            }
        }
    }
}

/// Log hyperparameters for a pretrain run (ALB-055/056)
fn log_run_params(store: &SqliteBackend, run_id: &str, spec: &TrainSpec, device: &str) {
    let _ = store.log_param(run_id, "task", ParameterValue::String("pretrain".into()));
    let _ = store.log_param(
        run_id,
        "model",
        ParameterValue::String(spec.model.path.display().to_string()),
    );
    let _ =
        store.log_param(run_id, "optimizer", ParameterValue::String(spec.optimizer.name.clone()));
    let _ = store.log_param(
        run_id,
        "learning_rate",
        ParameterValue::Float(f64::from(spec.optimizer.lr)),
    );
    let _ = store.log_param(run_id, "epochs", ParameterValue::Int(spec.training.epochs as i64));
    let _ = store.log_param(run_id, "batch_size", ParameterValue::Int(spec.data.batch_size as i64));
    let _ = store.log_param(run_id, "device", ParameterValue::String(device.to_string()));
    let _ = store.log_param(
        run_id,
        "output_dir",
        ParameterValue::String(spec.training.output_dir.display().to_string()),
    );
    if let Some(seq_len) = spec.data.seq_len {
        let _ = store.log_param(run_id, "seq_len", ParameterValue::Int(seq_len as i64));
    }
    if let Some(max_steps) = spec.training.max_steps {
        let _ = store.log_param(run_id, "max_steps", ParameterValue::Int(max_steps as i64));
    }
}

/// Training loop for GPU CudaTransformerTrainer
///
fn print_max_steps(max_steps: Option<usize>) {
    if let Some(ms) = max_steps {
        println!("  max_steps: {ms} (will stop early when reached)");
    }
}

/// ALB-068: Manual batch loop for intermediate checkpoint saving.
/// R-004: Gradient norm logging. R-008: Graceful shutdown.
/// R-009: Multi-checkpoint retention. R-012: MFU tracking.
/// R-014: JSONL experiment log. R-015: Per-epoch shuffling.
/// R-006/R-007: Training state persistence.
#[cfg(feature = "cuda")]
fn train_loop_cuda(
    trainer: &mut CudaTransformerTrainer,
    batches: &[LMBatch],
    spec: &TrainSpec,
) -> Result<()> {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    println!("Starting transformer training (CUDA GPU-resident)...");
    println!();

    let num_batches = batches.len();

    // ENT-275: Auto-compute max_steps for cosine LR scheduler when not explicit.
    // Without this, current_lr() falls back to constant lr (no decay).
    if spec.training.max_steps.is_none() {
        let total_steps = spec.training.epochs * num_batches;
        trainer.set_max_steps(total_steps);
        println!(
            "  max_steps: {total_steps} (auto: {epochs}×{num_batches})",
            epochs = spec.training.epochs
        );
    }

    let start_time = std::time::Instant::now();
    // Cap log_interval so training_state.json updates at least every 100 steps
    // (enables real-time monitoring via `apr monitor`). Previously num_batches/100
    // gave 12905 for large datasets — too infrequent for a 12-day run.
    let log_interval = (num_batches / 100).clamp(1, 100);
    let save_interval = spec.training.save_interval;
    let max_checkpoints = spec.training.max_checkpoints;

    // ALB-087: Auto eval scheduling — eval_interval defaults to save_interval
    let eval_interval =
        if spec.training.eval_interval > 0 { spec.training.eval_interval } else { save_interval };
    let patience = spec.training.patience;
    let mut best_val_loss: f32 = f32::INFINITY;
    let mut evals_without_improvement: usize = 0;
    let mut last_eval_step: usize = 0;

    // ALB-045: Initialize training state IPC for `apr monitor`
    let state = TrainingState::new(&spec.training.output_dir);
    let start_ms = now_ms();
    let gpu_name = trainer.gpu_name();
    let total_epochs = spec.training.epochs;

    // ALB-055/056: Open SQLite experiment tracking (local + global)
    let mut tracker = PretrainTracker::open(spec, &gpu_name);

    // R-012: MFU calculation setup
    let num_params = trainer.num_params();
    let seq_len = spec.data.seq_len.unwrap_or(128);
    let tokens_per_batch = spec.data.batch_size * seq_len;
    // RTX 4090: 82.6 TFLOPS fp32 (query via cuDeviceGetAttribute when available)
    let gpu_peak_tflops: f64 = 82.58e12;

    // R-014: Open JSONL experiment log
    let jsonl_path = spec.training.output_dir.join("training_log.jsonl");
    std::fs::create_dir_all(&spec.training.output_dir).ok();
    let mut jsonl_file =
        std::fs::OpenOptions::new().create(true).append(true).open(&jsonl_path).ok();
    // Write config header
    write_jsonl_event_json(
        &mut jsonl_file,
        &serde_json::json!({
            "type": "config",
            "num_params": num_params,
            "batch_size": spec.data.batch_size,
            "seq_len": seq_len,
            "max_steps": spec.training.max_steps,
            "epochs": spec.training.epochs,
            "lr": spec.optimizer.lr,
            "gpu": &gpu_name,
            "timestamp": now_ms(),
        }),
    );

    // R-008: Graceful shutdown signal handler
    let shutdown_flag = Arc::new(AtomicBool::new(false));
    {
        let flag = shutdown_flag.clone();
        let _ = ctrlc::set_handler(move || {
            flag.store(true, Ordering::SeqCst);
            eprintln!("\n[SIGINT] Graceful shutdown requested. Saving checkpoint...");
        });
    }

    // Write initial "Initializing" snapshot
    write_training_snapshot(
        &state,
        start_ms,
        0,
        total_epochs,
        0,
        num_batches,
        0.0,
        &[],
        0.0,
        0.0,
        TrainingStatus::Initializing,
        spec,
        &gpu_name,
    );

    print_max_steps(spec.training.max_steps);

    // ALB-087: Print eval scheduling config
    if eval_interval != save_interval {
        println!("  eval_interval: {eval_interval} (decoupled from save_interval={save_interval})");
    }
    if patience > 0 {
        println!("  early_stopping: patience={patience} eval intervals");
    }

    // ALB-082: Scaling law predictor for early convergence ceiling detection
    let mut scaling_predictor = ScalingLawPredictor::new();
    let tokens_per_step = tokens_per_batch * spec.training.gradient_accumulation.unwrap_or(1);

    // Track loss history for TUI sparkline
    let mut loss_history: Vec<f32> = Vec::new();
    let mut last_save_step: usize = 0;

    let model_name = spec
        .model
        .path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("entrenar-model")
        .to_string();

    // R-015: Prepare shuffled batch indices
    let shuffle = spec.training.shuffle;
    let seed = spec.training.seed.unwrap_or(42);

    // R-005: Load validation batches if val path exists
    let val_batches = load_val_batches(spec);

    // R-018: NaN/Inf detection counter
    let mut nan_skips: usize = 0;

    // R-017: ZClip adaptive gradient clipping — EMA of gradient norms
    let mut gnorm_ema: f64 = 0.0;
    let mut gnorm_ema_sq: f64 = 0.0;
    let zclip_alpha: f64 = 0.05; // EMA decay rate
    let zclip_threshold: f64 = 2.0; // z-score threshold for spike detection

    // R-003: Heartbeat file for crash detection
    let heartbeat_path = spec.training.output_dir.join("heartbeat");

    // R-016b: Loss spike rollback — EMA for spike detection
    let mut loss_ema: f64 = 0.0;
    let loss_ema_alpha: f64 = 0.05;
    let loss_spike_threshold: f64 = 3.0; // spike if loss > threshold × EMA
    let mut rollback_count: usize = 0;
    let max_rollbacks: usize = 3;

    // R-029: Gradient noise scale estimation — rolling window of grad norms
    let mut gnorm_window: Vec<f64> = Vec::with_capacity(100);
    let noise_scale_interval: usize = 100;
    let mut last_noise_scale_step: usize = usize::MAX; // Dedup: only log once per optimizer step

    // R-026: Save training config hash to JSONL for diff tracking
    write_config_provenance(&mut jsonl_file, spec);

    // R-023: Curriculum learning — track current stage index
    let mut curriculum_stage: usize = 0;
    let curriculum = spec.training.curriculum.as_deref();
    print_curriculum_stages(curriculum);

    // ALB-120: Skip batches already processed before checkpoint.
    let grad_accum = spec.training.gradient_accumulation.unwrap_or(1);
    let resume_batch_idx = trainer.step() * grad_accum;

    'outer: for epoch in 0..spec.training.epochs {
        let epoch_start = std::time::Instant::now();
        let mut total_loss = 0.0;
        let mut batches_processed = 0;

        // R-015: Generate shuffled indices for this epoch
        let batch_order = shuffled_batch_order(num_batches, shuffle, seed, epoch);

        // ALB-068: Manual batch loop for intermediate checkpoint saving
        for (iter_idx, &batch_idx) in batch_order.iter().enumerate() {
            // ALB-120: Skip batches already processed before checkpoint
            if iter_idx < resume_batch_idx {
                continue;
            }
            // R-008: Check graceful shutdown flag
            if shutdown_flag.load(Ordering::SeqCst) {
                handle_graceful_shutdown(
                    trainer,
                    spec,
                    &state,
                    &mut tracker,
                    start_ms,
                    epoch,
                    iter_idx,
                    total_epochs,
                    num_batches,
                    &loss_history,
                    &model_name,
                    &gpu_name,
                    seed,
                    loss_ema,
                );
                return Ok(());
            }

            // Check max_steps before processing
            if reached_max_steps(spec.training.max_steps, trainer.step()) {
                break 'outer;
            }

            // R-023: Check curriculum stage transition
            curriculum_stage = check_curriculum_transition(
                curriculum,
                curriculum_stage,
                trainer.step(),
                &mut jsonl_file,
            );

            let batch = &batches[batch_idx];
            // R-028: Per-step timing
            let step_start = std::time::Instant::now();
            let batch_loss = trainer.train_batch(batch);
            let step_elapsed = step_start.elapsed();

            // R-018: NaN/Inf detection — skip step if loss is non-finite
            if !batch_loss.is_finite() {
                nan_skips += 1;
                println!(
                    "  [WARN] NaN/Inf loss at step {} (skip #{}) — skipping",
                    trainer.step(),
                    nan_skips
                );
                continue;
            }
            total_loss += batch_loss;
            batches_processed += 1;

            // ENT-283: Seed loss EMA to first observed loss to avoid cold-start false rollbacks
            if loss_ema == 0.0 {
                loss_ema = f64::from(batch_loss);
            }

            // R-016b: Loss spike detection + rollback
            detect_loss_spike(
                batch_loss,
                trainer.step(),
                &mut loss_ema,
                loss_ema_alpha,
                loss_spike_threshold,
                &mut rollback_count,
                max_rollbacks,
                &mut jsonl_file,
            );

            // R-017: ZClip — update EMA and detect gradient spikes
            zclip_update(
                f64::from(trainer.last_grad_norm()),
                trainer.step(),
                &mut gnorm_ema,
                &mut gnorm_ema_sq,
                zclip_alpha,
                zclip_threshold,
            );

            // R-029: Track grad norm for noise scale estimation
            update_noise_scale(
                f64::from(trainer.last_grad_norm()),
                trainer.step(),
                &mut gnorm_window,
                noise_scale_interval,
                &mut last_noise_scale_step,
                &mut jsonl_file,
            );

            // R-003: Write heartbeat for crash detection
            write_heartbeat(&heartbeat_path, trainer.step());

            // Track loss history (keep last 100 for sparkline)
            push_capped(&mut loss_history, batch_loss, 100);

            // Logging at log_interval boundaries
            if should_log(iter_idx, log_interval) {
                log_step_metrics(
                    trainer,
                    &state,
                    &mut tracker,
                    &mut jsonl_file,
                    &epoch_start,
                    &start_time,
                    &step_elapsed,
                    epoch,
                    total_epochs,
                    iter_idx,
                    num_batches,
                    tokens_per_batch,
                    num_params,
                    gpu_peak_tflops,
                    start_ms,
                    batch_loss,
                    &loss_history,
                    spec,
                    &gpu_name,
                );
            }

            // ALB-068/R-009: Intermediate checkpoint saving at save_interval
            let current_step = trainer.step();
            // ALB-132: Skip save on the first step after resume — the checkpoint we just
            // loaded is already at this step. Re-saving would overwrite the original with
            // potentially incomplete GPU optimizer state, corrupting subsequent resumes.
            let is_resume_step =
                resume_batch_idx > 0 && current_step == resume_batch_idx / grad_accum;
            let do_save = !is_resume_step
                && should_save_checkpoint(current_step, last_save_step, save_interval);
            let do_eval = current_step > 0
                && current_step != last_eval_step
                && current_step.is_multiple_of(eval_interval);

            if do_save {
                save_and_validate_checkpoint(
                    trainer,
                    spec,
                    &model_name,
                    current_step,
                    epoch,
                    iter_idx,
                    max_checkpoints,
                    seed,
                    loss_ema,
                );
                last_save_step = current_step;
            }

            // ALB-087: Decoupled eval + best-model tracking + early stopping
            if do_eval {
                last_eval_step = current_step;
                let eval_val_loss = run_validation_eval(
                    trainer,
                    &val_batches,
                    current_step,
                    &mut jsonl_file,
                    &mut scaling_predictor,
                    tokens_per_step,
                    spec.training.max_steps,
                );
                if let Some(val_loss) = eval_val_loss {
                    if val_loss < best_val_loss {
                        best_val_loss = val_loss;
                        evals_without_improvement = 0;
                        save_best_model(trainer, spec, &model_name, current_step);
                    } else {
                        evals_without_improvement += 1;
                    }
                    if patience > 0 && evals_without_improvement >= patience {
                        println!(
                            "  [early-stop] No improvement for {evals_without_improvement} evals (patience={patience}). \
                             Best val_loss={best_val_loss:.4}. Stopping.",
                        );
                        write_jsonl_event_json(
                            &mut jsonl_file,
                            &serde_json::json!({
                                "type": "early_stop",
                                "step": current_step,
                                "best_val_loss": best_val_loss,
                                "evals_without_improvement": evals_without_improvement,
                                "patience": patience,
                                "timestamp": now_ms(),
                            }),
                        );
                        break 'outer;
                    }
                }
            }
        }

        let avg_loss = total_loss / batches_processed.max(1) as f32;
        let ppl = crate::train::perplexity(avg_loss);
        println!(
            "Epoch {}/{}: loss={:.6}, perplexity={:.2}, time={:.1}s",
            epoch + 1,
            spec.training.epochs,
            avg_loss,
            ppl,
            epoch_start.elapsed().as_secs_f64(),
        );

        if reached_max_steps(spec.training.max_steps, trainer.step()) {
            break;
        }
    }

    let total_time = start_time.elapsed();
    println!("Total training time: {:.1}s", total_time.as_secs_f64());

    // KAIZEN-047: Print step profiler report at end of training
    trainer.print_profiler_report();

    // ALB-045: Write final "Completed" snapshot
    let final_loss = trainer.metrics.losses.last().copied().unwrap_or(0.0);
    write_training_snapshot(
        &state,
        start_ms,
        total_epochs,
        total_epochs,
        trainer.step(),
        num_batches,
        final_loss,
        &loss_history,
        trainer.current_lr(),
        0.0,
        TrainingStatus::Completed,
        spec,
        &gpu_name,
    );

    // ALB-055/056: Mark run as completed in SQLite
    tracker.complete();

    // R-014: Write completion entry
    write_jsonl_event_json(
        &mut jsonl_file,
        &serde_json::json!({
            "type": "complete",
            "step": trainer.step(),
            "final_loss": final_loss,
            "total_time_s": total_time.as_secs_f64(),
            "timestamp": now_ms(),
        }),
    );

    save_trained_model_cuda(trainer, spec)
}

/// Distributed CUDA training loop (#133).
///
/// Multi-process DDP: each process runs this function with its own rank.
/// Rank 0 spawns the GradientServer in a background thread. All ranks
/// connect as workers and run the DDP training step in lockstep.
///
/// Data is sharded by rank: worker N processes batches N, N+ws, N+2*ws, ...
#[cfg(feature = "cuda")]
/// Spawn the coordinator (GradientServer) thread for DDP rank 0.
fn spawn_coordinator_thread(
    coord_addr: std::net::SocketAddr,
    world_size: usize,
    num_blocks: usize,
    total_steps: usize,
) -> Result<std::thread::JoinHandle<()>> {
    use crate::finetune::distributed::DistributedConfig;
    use crate::finetune::GradientServer;

    let server_config = DistributedConfig::coordinator(coord_addr, world_size);
    let mut server = GradientServer::bind(server_config)
        .map_err(|e| Error::ConfigError(format!("GradientServer bind failed: {e}")))?;
    println!("  ✓ GradientServer bound on {coord_addr}");

    Ok(std::thread::spawn(move || {
        server.wait_for_workers().unwrap();
        eprintln!("[coordinator] All {world_size} workers connected");

        for _step in 0..total_steps {
            for block_idx in (0..num_blocks).rev() {
                let result =
                    server.collect_and_reduce_block(_step as u64, block_idx as u32).unwrap();
                server.broadcast_averaged_block(_step as u64, &result).unwrap();
            }
            for component in [0u8, 1, 2] {
                let result = server.collect_and_reduce_non_block(_step as u64, component).unwrap();
                server.broadcast_averaged_non_block(_step as u64, &result).unwrap();
            }
        }
        eprintln!("[coordinator] Training complete ({total_steps} steps)");
    }))
}

#[cfg(feature = "cuda")]
fn train_loop_cuda_distributed(
    mut cuda_trainer: CudaTransformerTrainer,
    batches: &[LMBatch],
    spec: &TrainSpec,
) -> Result<()> {
    use crate::finetune::distributed::DistributedConfig;
    use crate::finetune::WorkerClient;
    use crate::train::{shard_batches, DistributedComm, DistributedCudaTrainer};

    let dist_config = cuda_trainer
        .config()
        .distributed
        .clone()
        .ok_or_else(|| Error::ConfigError("missing distributed config".into()))?;

    let rank = dist_config.rank;
    let world_size = dist_config.world_size;
    let coord_addr = dist_config.coordinator_addr;

    println!("Starting distributed training (DDP)...");
    println!("  rank: {rank}/{world_size}");
    println!("  coordinator: {coord_addr}");

    cuda_trainer.ensure_grad_accum();

    let num_blocks = cuda_trainer
        .grad_accum_ref()
        .map_or(0, crate::train::PerBlockGradientAccumulator::num_blocks);

    // Step 1: If rank 0, spawn GradientServer in background thread
    let server_handle = if rank == 0 {
        let max_steps = spec.training.max_steps.unwrap_or(usize::MAX);
        let batches_per_worker = batches.len().div_ceil(world_size);
        let total_steps = std::cmp::min(spec.training.epochs * batches_per_worker, max_steps);
        Some(spawn_coordinator_thread(coord_addr, world_size, num_blocks, total_steps)?)
    } else {
        std::thread::sleep(std::time::Duration::from_millis(100));
        None
    };

    // Step 2: Connect as worker (all ranks, including rank 0)
    let worker_config = DistributedConfig::worker(coord_addr);
    let client = WorkerClient::connect(worker_config, 1, "cuda")
        .map_err(|e| Error::ConfigError(format!("WorkerClient connect failed: {e}")))?;
    println!("  ✓ Connected as worker {} (id={})", rank, client.worker_id());

    // Step 3: Wrap in DistributedCudaTrainer
    let comm = DistributedComm::Remote { client };
    let mut ddp_trainer = DistributedCudaTrainer::new(cuda_trainer, comm, dist_config.clone());

    // Step 4: Training loop with data sharding
    let num_batches = batches.len();
    let start_time = std::time::Instant::now();
    let log_interval = std::cmp::max(num_batches / (world_size * 100).max(1), 1);
    let save_interval = spec.training.save_interval;
    let max_checkpoints = spec.training.max_checkpoints;
    let seed = spec.training.seed.unwrap_or(42);

    // ALB-082: Scaling law predictor for DDP path
    let _scaling_predictor = ScalingLawPredictor::new();
    let seq_len_ddp = spec.data.seq_len.unwrap_or(128);
    let grad_accum_ddp = spec.training.gradient_accumulation.unwrap_or(1);
    let _tokens_per_step_ddp = spec.data.batch_size * seq_len_ddp * grad_accum_ddp;

    let model_name = spec
        .model
        .path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("entrenar-model")
        .to_string();

    // R-005: Load validation batches
    let _val_batches = load_val_batches(spec);

    let mut loss_history: Vec<f32> = Vec::new();
    let mut last_save_step: usize = 0;

    for epoch in 0..spec.training.epochs {
        let epoch_start = std::time::Instant::now();
        let mut total_loss = 0.0;
        let mut batches_processed = 0;

        // Shard batches by rank: worker N gets N, N+ws, N+2*ws, ...
        let my_batch_indices = shard_batches(num_batches, rank, world_size);

        for (iter_idx, &batch_idx) in my_batch_indices.iter().enumerate() {
            if ddp_trainer.reached_max_steps() {
                break;
            }

            let batch = &batches[batch_idx];
            let step_start = std::time::Instant::now();
            let batch_loss = ddp_trainer.train_batch(batch);
            let step_elapsed = step_start.elapsed();

            if !batch_loss.is_finite() {
                continue;
            }
            total_loss += batch_loss;
            batches_processed += 1;
            push_capped(&mut loss_history, batch_loss, 100);

            // Logging (rank 0 only to avoid spam)
            if rank == 0 && should_log(iter_idx, log_interval) {
                let step = ddp_trainer.step();
                let elapsed = epoch_start.elapsed().as_secs_f64();
                let seq_len = spec.data.seq_len.unwrap_or(128);
                let tokens_done = (iter_idx + 1) * spec.data.batch_size * seq_len * world_size;
                let tok_per_sec = tokens_done as f64 / elapsed.max(0.001);
                println!(
                    "  [DDP rank 0] step={} loss={:.4} tok/s={:.0} step_time={:.1}ms",
                    step,
                    batch_loss,
                    tok_per_sec,
                    step_elapsed.as_secs_f64() * 1000.0,
                );
            }

            // Checkpoint (rank 0 only)
            if rank == 0 {
                let current_step = ddp_trainer.step();
                if should_save_checkpoint(current_step, last_save_step, save_interval) {
                    save_and_validate_checkpoint(
                        ddp_trainer.trainer_mut(),
                        spec,
                        &model_name,
                        current_step,
                        epoch,
                        iter_idx,
                        max_checkpoints,
                        seed,
                        0.0,
                    );
                    last_save_step = current_step;
                }
            }
        }

        if batches_processed > 0 {
            let avg_loss = total_loss / batches_processed as f32;
            let ppl = crate::train::perplexity(avg_loss);
            if rank == 0 {
                println!(
                    "Epoch {}/{}: loss={:.6}, perplexity={:.2}, time={:.1}s",
                    epoch + 1,
                    spec.training.epochs,
                    avg_loss,
                    ppl,
                    epoch_start.elapsed().as_secs_f64(),
                );
            }
        }

        if ddp_trainer.reached_max_steps() {
            break;
        }
    }

    let total_time = start_time.elapsed();
    if rank == 0 {
        println!("Total distributed training time: {:.1}s", total_time.as_secs_f64());
    }

    // Save final model (rank 0 only)
    if rank == 0 {
        save_trained_model_cuda(ddp_trainer.trainer_mut(), spec)?;
    }

    // Wait for coordinator thread to finish
    if let Some(handle) = server_handle {
        let _: std::result::Result<(), _> = handle.join();
    }

    Ok(())
}

// Training utility functions: checkpointing, logging, validation, metrics
include!("helpers.rs");

// Model loading, data loading, and config parsing
include!("data.rs");

#[cfg(test)]
mod tests;
