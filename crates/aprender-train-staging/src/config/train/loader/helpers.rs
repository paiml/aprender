/// Check if max_steps has been reached.
/// Check if this iteration should log metrics.
#[allow(clippy::incompatible_msrv)]
fn should_log(iter_idx: usize, interval: usize) -> bool {
    (iter_idx + 1).is_multiple_of(interval) || iter_idx == 0
}

/// Check if this step should trigger a checkpoint save.
#[allow(clippy::incompatible_msrv)]
fn should_save_checkpoint(step: usize, last_save_step: usize, interval: usize) -> bool {
    step > 0 && step != last_save_step && step.is_multiple_of(interval)
}

fn reached_max_steps(max_steps: Option<usize>, current_step: usize) -> bool {
    if let Some(max) = max_steps {
        if current_step >= max {
            println!("Reached max_steps={max}, stopping training.");
            return true;
        }
    }
    false
}

/// R-017: ZClip gradient spike detection — update EMA and log spikes.
fn zclip_update(
    gnorm: f64,
    step: usize,
    ema: &mut f64,
    ema_sq: &mut f64,
    alpha: f64,
    threshold: f64,
) {
    *ema = alpha * gnorm + (1.0 - alpha) * *ema;
    *ema_sq = alpha * gnorm * gnorm + (1.0 - alpha) * *ema_sq;
    let std = (*ema_sq - *ema * *ema).max(0.0).sqrt();
    if std > 1e-8 {
        let z_score = (gnorm - *ema) / std;
        if z_score > threshold {
            println!(
                "  [ZClip] gradient spike at step {}: z={:.1} gnorm={:.2e} ema={:.2e}",
                step, z_score, gnorm, *ema
            );
        }
    }
}

/// Push a value to a capped history buffer, removing oldest if at capacity.
fn push_capped(history: &mut Vec<f32>, value: f32, max_len: usize) {
    history.push(value);
    if history.len() > max_len {
        history.remove(0);
    }
}

/// Push f64 value onto a capped rolling window.
fn push_capped_f64(window: &mut Vec<f64>, value: f64, max_len: usize) {
    window.push(value);
    if window.len() > max_len {
        window.remove(0);
    }
}

/// R-029: Track gradient norm and estimate noise scale at intervals.
/// B_noise = Var(||g||) / Mean(||g||)² — proxy for critical batch size.
#[allow(clippy::incompatible_msrv)]
fn update_noise_scale(
    grad_norm: f64,
    step: usize,
    window: &mut Vec<f64>,
    interval: usize,
    last_logged_step: &mut usize,
    jsonl_file: &mut Option<std::fs::File>,
) {
    push_capped_f64(window, grad_norm, 100);
    if step == 0 || !step.is_multiple_of(interval) || window.len() < 10 || step == *last_logged_step
    {
        return;
    }
    *last_logged_step = step;
    let n = window.len() as f64;
    let mean = window.iter().sum::<f64>() / n;
    if mean < 1e-12 {
        return;
    }
    let variance = window.iter().map(|&g| (g - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let b_noise = variance / (mean * mean);
    println!("  [noise-scale] step={} B_noise={:.4} (window={})", step, b_noise, window.len());
    write_jsonl_event_json(
        jsonl_file,
        &serde_json::json!({
            "type": "noise_scale",
            "step": step,
            "b_noise": b_noise,
            "gnorm_mean": mean,
            "gnorm_var": variance,
            "window_size": window.len(),
            "timestamp": now_ms(),
        }),
    );
}

/// R-016b: Detect loss spikes and log rollback events.
#[allow(clippy::too_many_arguments)]
fn detect_loss_spike(
    loss: f32,
    step: usize,
    ema: &mut f64,
    alpha: f64,
    threshold: f64,
    rollback_count: &mut usize,
    max_rollbacks: usize,
    jsonl_file: &mut Option<std::fs::File>,
) {
    let bl = f64::from(loss);
    if *ema > 0.0 && bl > threshold * *ema && *rollback_count < max_rollbacks {
        *rollback_count += 1;
        println!(
            "  [ROLLBACK] loss spike at step {}: {:.4} > {:.1}×EMA({:.4}), rollback #{}/{}",
            step, loss, threshold, *ema, *rollback_count, max_rollbacks
        );
        write_jsonl_event(jsonl_file, "rollback", step, loss, *ema as f32);
    }
    *ema = alpha * bl + (1.0 - alpha) * *ema;
}

/// Write a generic event entry to the JSONL experiment log.
fn write_jsonl_event(
    jsonl_file: &mut Option<std::fs::File>,
    event_type: &str,
    step: usize,
    loss: f32,
    loss_ema: f32,
) {
    use std::io::Write;
    if let Some(ref mut f) = jsonl_file {
        let entry = serde_json::json!({
            "type": event_type,
            "step": step,
            "loss": loss,
            "loss_ema": loss_ema,
            "timestamp": now_ms(),
        });
        let _ = writeln!(f, "{entry}");
    }
}

/// Write an arbitrary JSON event to the JSONL log (generic version).
fn write_jsonl_event_json(jsonl_file: &mut Option<std::fs::File>, entry: &serde_json::Value) {
    use std::io::Write;
    if let Some(ref mut f) = jsonl_file {
        let _ = writeln!(f, "{entry}");
    }
}

/// R-003: Write heartbeat timestamp for crash detection.
fn write_heartbeat(path: &std::path::Path, step: usize) {
    let data = format!("{}\t{}", now_ms(), step);
    let _ = std::fs::write(path, data);
}

/// R-026: Save training config snapshot + R-024: data provenance to JSONL.
fn write_config_provenance(jsonl_file: &mut Option<std::fs::File>, spec: &TrainSpec) {
    use std::io::Write;
    let Some(ref mut f) = jsonl_file else { return };

    // R-024: Data provenance
    let train_path = spec.data.train.display().to_string();
    let val_path = spec.data.val.as_ref().map(|p| p.display().to_string());
    let provenance = serde_json::json!({
        "type": "provenance",
        "train_path": train_path,
        "val_path": val_path,
        "batch_size": spec.data.batch_size,
        "seq_len": spec.data.seq_len,
        "timestamp": now_ms(),
    });
    let _ = writeln!(f, "{provenance}");

    // R-026: Config snapshot for diff tracking
    let config = serde_json::json!({
        "type": "config_snapshot",
        "optimizer": {
            "name": &spec.optimizer.name,
            "lr": spec.optimizer.lr,
            "params": &spec.optimizer.params,
        },
        "training": {
            "epochs": spec.training.epochs,
            "max_steps": spec.training.max_steps,
            "grad_clip": spec.training.grad_clip,
            "save_interval": spec.training.save_interval,
            "warmup_steps": spec.training.warmup_steps,
            "gradient_accumulation": spec.training.gradient_accumulation,
            "seed": spec.training.seed,
        },
        "model_path": spec.model.path.display().to_string(),
        "timestamp": now_ms(),
    });
    let _ = writeln!(f, "{config}");
}

/// R-005: Load validation batches if val path is configured.
fn load_val_batches(spec: &TrainSpec) -> Vec<LMBatch> {
    let val_path = match &spec.data.val {
        Some(p) if p.exists() => p,
        _ => return Vec::new(),
    };
    let batch_size = spec.data.batch_size;
    let seq_len = spec.data.seq_len.unwrap_or(128);
    let tokenizer = load_tokenizer(spec).ok().flatten();
    let tokenizer_ref = tokenizer.as_ref();

    // Try loading from parquet directory or file
    if val_path.is_dir() {
        if let Some(tok) = tokenizer_ref {
            let column = spec.data.input_column.as_deref().unwrap_or("text");
            if let Ok(batches) =
                load_lm_batches_from_parquet(val_path, tok, batch_size, seq_len, column)
            {
                println!(
                    "  ✓ {} validation batches loaded from {}",
                    batches.len(),
                    val_path.display()
                );
                return batches;
            }
        }
    }
    Vec::new()
}

/// ALB-082: Scaling law predictor — fits L(D) = a - b × ln(D) to eval history.
///
/// After 3+ eval checkpoints, predicts val_ppl at max_steps. Warns if
/// predicted improvement < 10% over current val_ppl.
struct ScalingLawPredictor {
    /// (cumulative_tokens, val_loss) pairs
    history: Vec<(f64, f64)>,
}

impl ScalingLawPredictor {
    fn new() -> Self {
        Self { history: Vec::new() }
    }

    fn record(&mut self, tokens: usize, val_loss: f32) {
        self.history.push((tokens as f64, f64::from(val_loss)));
    }

    /// Fit L(D) = a - b × ln(D) via ordinary least squares.
    /// Returns (a, b) or None if < 3 data points.
    fn fit(&self) -> Option<(f64, f64)> {
        if self.history.len() < 3 {
            return None;
        }
        // OLS: y = a + b*x where x = ln(tokens), y = val_loss, b is negative
        let n = self.history.len() as f64;
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_xx = 0.0;
        for &(tokens, loss) in &self.history {
            let x = tokens.ln();
            sum_x += x;
            sum_y += loss;
            sum_xy += x * loss;
            sum_xx += x * x;
        }
        let denom = n * sum_xx - sum_x * sum_x;
        if denom.abs() < 1e-12 {
            return None;
        }
        let b = (n * sum_xy - sum_x * sum_y) / denom;
        let a = (sum_y - b * sum_x) / n;
        // b should be negative (loss decreases with tokens)
        Some((a, -b)) // Return (a, b) where L(D) = a - b*ln(D), b > 0
    }

    /// Predict val_loss at given token count using fitted parameters.
    fn predict(&self, target_tokens: usize) -> Option<(f64, f64, f64)> {
        let (a, b) = self.fit()?;
        let predicted_loss = a - b * (target_tokens as f64).ln();
        let predicted_ppl = predicted_loss.exp();
        Some((predicted_loss, predicted_ppl, b))
    }
}

/// R-005: Run validation evaluation and log results.
/// ALB-082: Includes scaling law prediction after 3+ eval points.
#[cfg(feature = "cuda")]
fn run_validation_eval(
    trainer: &mut CudaTransformerTrainer,
    val_batches: &[LMBatch],
    step: usize,
    jsonl_file: &mut Option<std::fs::File>,
    predictor: &mut ScalingLawPredictor,
    tokens_per_step: usize,
    max_steps: Option<usize>,
) -> Option<f32> {
    if val_batches.is_empty() {
        return None;
    }
    let mut total_loss = 0.0;
    let mut count = 0;
    for batch in val_batches {
        let loss = trainer.eval_batch(batch);
        if loss > 0.0 {
            total_loss += loss;
            count += 1;
        }
    }
    if count == 0 {
        return None;
    }
    let val_loss = total_loss / count as f32;
    let val_ppl = crate::train::perplexity(val_loss);
    let cumulative_tokens = step * tokens_per_step;

    // ALB-082: Record eval point and predict
    predictor.record(cumulative_tokens, val_loss);

    let target_tokens = max_steps.unwrap_or(step * 2) * tokens_per_step;
    let prediction = predictor.predict(target_tokens);

    // Build JSONL entry (common fields + optional prediction fields)
    let mut entry = serde_json::json!({
        "type": "eval",
        "step": step,
        "val_loss": val_loss,
        "val_ppl": val_ppl,
        "cumulative_tokens": cumulative_tokens,
        "timestamp": now_ms(),
    });

    if let Some((pred_loss, pred_ppl, slope)) = prediction {
        let target_steps = target_tokens / tokens_per_step;
        println!(
            "  [eval] step={step} val_loss={val_loss:.4} val_ppl={val_ppl:.2} ({count} batches) \
             predicted_ppl={pred_ppl:.1} at step {target_steps} (slope={slope:.4})"
        );
        // Warn if predicted improvement < 10%
        let improvement = (f64::from(val_ppl) - pred_ppl) / f64::from(val_ppl);
        if improvement < 0.10 && predictor.history.len() >= 4 {
            println!(
                "  [WARN] Scaling law predicts only {:.1}% improvement by step {} \
                 (val_ppl {:.1} → {:.1}). Consider: more data, larger model, or stopping.",
                improvement * 100.0,
                target_steps,
                val_ppl,
                pred_ppl
            );
        }
        entry["predicted_final_loss"] = serde_json::json!(pred_loss);
        entry["predicted_final_ppl"] = serde_json::json!(pred_ppl);
        entry["scaling_slope"] = serde_json::json!(slope);
        entry["target_steps"] = serde_json::json!(target_steps);
    } else {
        println!(
            "  [eval] step={step} val_loss={val_loss:.4} val_ppl={val_ppl:.2} ({count} batches)"
        );
    }

    use std::io::Write;
    if let Some(ref mut f) = jsonl_file {
        let _ = writeln!(f, "{entry}");
    }
    Some(val_loss)
}

/// ALB-087/ALB-096: Save best model checkpoint (APR format).
#[cfg(feature = "cuda")]
fn save_best_model(
    trainer: &mut CudaTransformerTrainer,
    spec: &TrainSpec,
    model_name: &str,
    step: usize,
) {
    let best_path = spec.training.output_dir.join("model-best.apr");
    let lr = trainer.current_lr();
    let save_fn =
        trainer.prepare_async_apr_save(model_name, "LlamaForCausalLM", step, 0.0, f64::from(lr));
    std::thread::spawn(move || {
        if let Err(e) = save_fn(&best_path) {
            println!("  [WARN] Failed to save model-best: {e}");
        } else {
            println!("  [best-model] step={step} saved to {}", best_path.display());
        }
    });
}

/// ALB-096: Save checkpoint as APR with integrity verification and state persistence.
///
/// Single atomic APR file contains model weights + optimizer state + training metadata.
/// Replaces separate `model-step-N.safetensors` + `optimizer_state.json` files.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn save_and_validate_checkpoint(
    trainer: &mut CudaTransformerTrainer,
    spec: &TrainSpec,
    model_name: &str,
    step: usize,
    epoch: usize,
    batch_idx: usize,
    max_checkpoints: usize,
    seed: u64,
    loss_ema: f64,
) {
    let ckpt_path = checkpoint_path(&spec.training.output_dir, step);
    let lr = trainer.current_lr();
    // ALB-096: Async APR checkpointing — model weights + optimizer state in single atomic file
    let save_fn =
        trainer.prepare_async_apr_save(model_name, "LlamaForCausalLM", step, loss_ema, f64::from(lr));
    let async_path = ckpt_path.clone();
    let async_output_dir = spec.training.output_dir.clone();
    // ENT-269: Save LoRA adapter to checkpoint directory (PEFT format)
    let ckpt_dir = ckpt_path.parent().unwrap_or_else(|| std::path::Path::new("."));
    let lora_ckpt_dir = ckpt_dir.join(format!("checkpoint-{step}"));
    if let Err(e) = trainer.save_cuda_lora_adapter(&lora_ckpt_dir, Some(model_name)) {
        println!("  [WARN] LoRA adapter checkpoint save failed: {e}");
    }

    std::thread::spawn(move || {
        if let Err(e) = save_fn(&async_path) {
            println!("  [WARN] Async APR checkpoint save failed: {e}");
        } else {
            verify_checkpoint(&async_path);
            println!("  [checkpoint] step={} saved to {}", step, async_path.display());
            save_training_state(&async_output_dir, step, epoch, batch_idx, seed, loss_ema);
            prune_checkpoints(&async_output_dir, max_checkpoints);
        }
    });
    // ALB-087: Eval is now decoupled from save — handled by the caller.
}

/// R-010: Verify checkpoint file integrity after save.
fn verify_checkpoint(path: &std::path::Path) {
    match std::fs::metadata(path) {
        Ok(meta) => {
            let size_mb = meta.len() / (1024 * 1024);
            if meta.len() == 0 {
                println!("  [WARN] Checkpoint file is empty: {}", path.display());
            } else {
                println!("  [verify] checkpoint OK ({size_mb} MB)");
            }
        }
        Err(e) => {
            println!("  [WARN] Checkpoint verification failed: {e}");
        }
    }
}

/// R-015: Generate shuffled or sequential batch indices for an epoch.
fn shuffled_batch_order(num_batches: usize, shuffle: bool, seed: u64, epoch: usize) -> Vec<usize> {
    if !shuffle {
        return (0..num_batches).collect();
    }
    let mut indices: Vec<usize> = (0..num_batches).collect();
    // Deterministic Fisher-Yates using LCG PRNG seeded by seed + epoch
    let mut rng_state: u64 = seed
        .wrapping_add(epoch as u64)
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    for i in (1..indices.len()).rev() {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let j = (rng_state >> 33) as usize % (i + 1);
        indices.swap(i, j);
    }
    indices
}

/// R-008: Handle SIGINT/SIGTERM graceful shutdown with emergency checkpoint.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn handle_graceful_shutdown(
    trainer: &mut CudaTransformerTrainer,
    spec: &TrainSpec,
    state: &TrainingState,
    tracker: &mut PretrainTracker,
    start_ms: u64,
    epoch: usize,
    iter_idx: usize,
    total_epochs: usize,
    num_batches: usize,
    loss_history: &[f32],
    model_name: &str,
    gpu_name: &str,
    seed: u64,
    loss_ema: f64,
) {
    println!("[SIGINT] Emergency checkpoint at step {}...", trainer.step());
    let ckpt_path = checkpoint_path(&spec.training.output_dir, trainer.step());
    if let Err(e) = trainer.save_apr(&ckpt_path, model_name, "LlamaForCausalLM") {
        println!("  [WARN] Emergency save failed: {e}");
    } else {
        println!("  [checkpoint] emergency save to {}", ckpt_path.display());
        save_training_state(
            &spec.training.output_dir,
            trainer.step(),
            epoch,
            iter_idx,
            seed,
            loss_ema,
        );
    }
    let final_loss = trainer.metrics.losses.last().copied().unwrap_or(0.0);
    write_training_snapshot(
        state,
        start_ms,
        epoch + 1,
        total_epochs,
        trainer.step(),
        num_batches,
        final_loss,
        loss_history,
        trainer.current_lr(),
        0.0,
        TrainingStatus::Completed,
        spec,
        gpu_name,
    );
    tracker.complete();
    println!("[SIGINT] Shutdown complete.");
}

/// R-023: Print curriculum stage configuration at startup.
fn print_curriculum_stages(curriculum: Option<&[crate::config::schema::CurriculumStage]>) {
    let Some(stages) = curriculum else { return };
    println!("  Curriculum learning: {} stages configured", stages.len());
    for (i, stage) in stages.iter().enumerate() {
        let until = stage.until_step.map_or("end".to_string(), |s| format!("step {s}"));
        println!("    Stage {}: {} (until {})", i, stage.data.display(), until);
    }
}

/// R-023: Check curriculum transition and log if stage changes.
/// Returns the (possibly updated) stage index.
fn check_curriculum_transition(
    curriculum: Option<&[crate::config::schema::CurriculumStage]>,
    current_stage: usize,
    step: usize,
    jsonl_file: &mut Option<std::fs::File>,
) -> usize {
    let Some(stages) = curriculum else { return current_stage };
    let Some(next) = advance_curriculum(stages, current_stage, step) else { return current_stage };
    println!(
        "  [Curriculum] → Stage {} at step {} (data: {})",
        next,
        step,
        stages[next].data.display()
    );
    write_jsonl_event_json(
        jsonl_file,
        &serde_json::json!({
            "type": "curriculum_transition",
            "stage": next,
            "step": step,
            "data": stages[next].data.to_string_lossy(),
            "timestamp": now_ms(),
        }),
    );
    next
}

/// R-023: Check if training should advance to the next curriculum stage.
/// Returns `Some(next_stage)` if a transition is needed, `None` otherwise.
fn advance_curriculum(
    stages: &[crate::config::schema::CurriculumStage],
    current: usize,
    step: usize,
) -> Option<usize> {
    if current >= stages.len() {
        return None;
    }
    let stage = &stages[current];
    if let Some(until) = stage.until_step {
        if step >= until && current + 1 < stages.len() {
            return Some(current + 1);
        }
    }
    None
}

/// R-014: Write a step entry to the JSONL experiment log.
#[allow(clippy::too_many_arguments)]
fn write_jsonl_step(
    jsonl_file: &mut Option<std::fs::File>,
    step: usize,
    loss: f32,
    lr: f32,
    tok_s: f64,
    mfu: f64,
    grad_norm: f32,
    embed_grad_norm: f32,
    epoch: usize,
    elapsed_s: f64,
) {
    use std::io::Write;
    if let Some(ref mut f) = jsonl_file {
        let entry = serde_json::json!({
            "type": "step",
            "step": step,
            "loss": loss,
            "lr": lr,
            "tok_s": tok_s,
            "mfu": mfu,
            "grad_norm": grad_norm,
            "grad_norm_embed": embed_grad_norm,
            "epoch": epoch,
            "elapsed_s": elapsed_s,
            "timestamp": now_ms(),
        });
        let _ = writeln!(f, "{entry}");
    }
}

/// R-009/ALB-096: Generate step-numbered checkpoint path (APR format).
fn checkpoint_path(output_dir: &Path, step: usize) -> PathBuf {
    output_dir.join(format!("model-step-{step}.apr"))
}

/// Parse step number from a checkpoint filename.
/// Supports both APR (`model-step-123.apr`) and legacy SafeTensors (`model-step-123.safetensors`).
fn parse_checkpoint_step(filename: &str) -> Option<usize> {
    filename
        .strip_prefix("model-step-")?
        .strip_suffix(".apr")
        .or_else(|| filename.strip_prefix("model-step-")?.strip_suffix(".safetensors"))
        .and_then(|s| s.parse().ok())
}

/// Log step metrics: console output, IPC snapshot, SQLite, JSONL.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn log_step_metrics(
    trainer: &CudaTransformerTrainer,
    state: &TrainingState,
    tracker: &mut PretrainTracker,
    jsonl_file: &mut Option<std::fs::File>,
    epoch_start: &std::time::Instant,
    start_time: &std::time::Instant,
    step_elapsed: &std::time::Duration,
    epoch: usize,
    total_epochs: usize,
    iter_idx: usize,
    num_batches: usize,
    tokens_per_batch: usize,
    num_params: usize,
    gpu_peak_tflops: f64,
    start_ms: u64,
    batch_loss: f32,
    loss_history: &[f32],
    spec: &TrainSpec,
    gpu_name: &str,
) {
    let elapsed = epoch_start.elapsed().as_secs_f64();
    let batches_done = iter_idx + 1;
    let tokens_done = batches_done * tokens_per_batch;
    let batch_per_sec = batches_done as f64 / elapsed.max(0.001);
    let remaining = (num_batches - batches_done) as f64 / batch_per_sec.max(0.001);
    let tok_per_sec = tokens_done as f64 / elapsed.max(0.001);

    // R-012: Compute MFU
    let flops_per_step = 6.0 * num_params as f64 * tokens_per_batch as f64;
    let step_time = elapsed / batches_done as f64;
    let mfu = (flops_per_step / step_time) / gpu_peak_tflops * 100.0;

    // R-004/R-040: Get per-parameter-group gradient norms
    let (grad_norm, embed_grad_norm) = trainer.param_grad_norms();
    // R-013: GPU memory usage
    let (gpu_used_mb, gpu_total_mb) = trainer.gpu_memory_mb();
    // R-028: Step time in ms
    let step_ms = step_elapsed.as_secs_f64() * 1000.0;

    println!(
        "  [{}/{} batches] step={} loss={:.4} lr={:.2e} tok/s={:.0} mfu={:.1}% gnorm={:.2e} gpu={}/{}MB step={:.0}ms eta={:.0}s",
        batches_done, num_batches,
        trainer.step(), batch_loss, trainer.current_lr(),
        tok_per_sec, mfu, grad_norm, gpu_used_mb, gpu_total_mb, step_ms, remaining,
    );

    // ALB-045: Write snapshot for `apr monitor`
    write_training_snapshot(
        state,
        start_ms,
        epoch + 1,
        total_epochs,
        trainer.step(),
        num_batches,
        batch_loss,
        loss_history,
        trainer.current_lr(),
        tok_per_sec as f32,
        TrainingStatus::Running,
        spec,
        gpu_name,
    );

    // ALB-055/056: Log step metrics to SQLite
    tracker.log_step(trainer.step() as u64, batch_loss, trainer.current_lr(), tok_per_sec as f32);

    // R-014: Write JSONL log entry
    write_jsonl_step(
        jsonl_file,
        trainer.step(),
        batch_loss,
        trainer.current_lr(),
        tok_per_sec,
        mfu,
        grad_norm,
        embed_grad_norm,
        epoch,
        start_time.elapsed().as_secs_f64(),
    );
}

/// R-009: Prune old checkpoints, keeping the most recent `max_keep`.
fn prune_checkpoints(output_dir: &Path, max_keep: usize) {
    if max_keep == 0 {
        return; // 0 = unlimited
    }
    let entries = match std::fs::read_dir(output_dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    let mut ckpts: Vec<(usize, PathBuf)> = entries
        .flatten()
        .filter_map(|e| {
            let step = parse_checkpoint_step(&e.file_name().to_string_lossy())?;
            Some((step, e.path()))
        })
        .collect();
    if ckpts.len() <= max_keep {
        return;
    }
    ckpts.sort_by_key(|(step, _)| *step);
    let to_remove = ckpts.len() - max_keep;
    for (step, path) in ckpts.into_iter().take(to_remove) {
        if std::fs::remove_file(&path).is_ok() {
            println!("  [prune] removed old checkpoint step={step}");
        }
    }
}

/// R-006/R-007: Save training state metadata alongside checkpoint.
fn save_training_state(
    output_dir: &Path,
    step: usize,
    epoch: usize,
    batch_idx: usize,
    seed: u64,
    loss_ema: f64,
) {
    let state = serde_json::json!({
        "step": step,
        "epoch": epoch,
        "batch_index": batch_idx,
        "seed": seed,
        "loss_ema": loss_ema,
        "timestamp": now_ms(),
    });
    let path = output_dir.join("training_state.json");
    if let Ok(json) = serde_json::to_string_pretty(&state) {
        let _ = std::fs::write(path, json);
    }
}

/// Save trained model from CPU trainer
fn save_trained_model_cpu(trainer: &TransformerTrainer, spec: &TrainSpec) -> Result<()> {
    println!();
    println!("✓ Transformer training complete");
    println!("  Final loss: {:.6}", trainer.metrics.losses.last().copied().unwrap_or(0.0));
    println!("  Best loss: {:.6}", trainer.metrics.best_loss().unwrap_or(0.0));
    println!("  Steps completed: {}", trainer.step());
    println!();

    std::fs::create_dir_all(&spec.training.output_dir).ok();

    // ENT-LoRA-015: Save adapter-only checkpoint when LoRA is active
    if trainer.is_lora() {
        let base_model_name =
            spec.model.path.file_name().and_then(|n| n.to_str()).unwrap_or("unknown");
        println!("Saving LoRA adapter to {}...", spec.training.output_dir.display());
        trainer.save_lora_adapter(&spec.training.output_dir, Some(base_model_name))?;
        let adapter_path = spec.training.output_dir.join("adapter_model.safetensors");
        let adapter_size = std::fs::metadata(&adapter_path).map(|m| m.len()).unwrap_or(0);
        println!("✓ LoRA adapter saved ({adapter_size} bytes)");
        println!("  adapter_model.safetensors + adapter_config.json");
    } else {
        let weights_path = spec.training.output_dir.join("model.safetensors");
        let model_name =
            spec.model.path.file_name().and_then(|n| n.to_str()).unwrap_or("entrenar-model");
        println!("Saving model weights to {}...", weights_path.display());
        trainer.save(&weights_path, model_name, "LlamaForCausalLM")?;
        println!(
            "✓ Model weights saved ({} bytes)",
            std::fs::metadata(&weights_path).map(|m| m.len()).unwrap_or(0)
        );
    }

    let weights_path = spec.training.output_dir.join("model.safetensors");
    save_config_and_metadata(
        trainer.model().config(),
        trainer.step(),
        &trainer.metrics,
        &weights_path,
        spec,
    )
}

/// Save trained model from CUDA trainer (syncs GPU→CPU first)
#[cfg(feature = "cuda")]
fn save_trained_model_cuda(trainer: &mut CudaTransformerTrainer, spec: &TrainSpec) -> Result<()> {
    println!();
    println!("✓ Transformer training complete (CUDA)");
    println!("  Final loss: {:.6}", trainer.metrics.losses.last().copied().unwrap_or(0.0));
    println!("  Best loss: {:.6}", trainer.metrics.best_loss().unwrap_or(0.0));
    println!("  Steps completed: {}", trainer.step());
    println!();

    std::fs::create_dir_all(&spec.training.output_dir).ok();

    // ALB-096: Save final model as APR (atomic, includes optimizer state)
    let weights_path = spec.training.output_dir.join("model.apr");
    let model_name =
        spec.model.path.file_name().and_then(|n| n.to_str()).unwrap_or("entrenar-model");
    let final_loss = f64::from(trainer.metrics.losses.last().copied().unwrap_or(0.0));
    let lr = trainer.current_lr();
    println!("Saving model weights to {}...", weights_path.display());
    let save_fn = trainer.prepare_async_apr_save(
        model_name,
        "LlamaForCausalLM",
        trainer.step(),
        final_loss,
        f64::from(lr),
    );
    save_fn(&weights_path)?;
    println!(
        "✓ Model weights saved ({} bytes, APR)",
        std::fs::metadata(&weights_path).map(|m| m.len()).unwrap_or(0)
    );

    // ENT-269: Save LoRA adapter as PEFT-compatible files (adapter_model.safetensors + adapter_config.json)
    // C-QLORA-SAVE-001: NF4 QLoRA training MUST produce adapter_model.safetensors
    trainer.save_cuda_lora_adapter(&spec.training.output_dir, Some(model_name))?;

    save_config_and_metadata(
        trainer.model().config(),
        trainer.step(),
        &trainer.metrics,
        &weights_path,
        spec,
    )
}

/// Save config.json and metadata (shared by CPU and CUDA paths)
fn save_config_and_metadata(
    mc: &TransformerConfig,
    step: usize,
    metrics: &crate::train::MetricsTracker,
    weights_path: &std::path::Path,
    spec: &TrainSpec,
) -> Result<()> {
    let config_json_path = spec.training.output_dir.join("config.json");
    let config_json = serde_json::json!({
        "architectures": [mc.hf_architecture_name()],
        "model_type": mc.hf_model_type_str(),
        "hidden_size": mc.hidden_size,
        "num_hidden_layers": mc.num_hidden_layers,
        "num_attention_heads": mc.num_attention_heads,
        "num_key_value_heads": mc.num_kv_heads,
        "intermediate_size": mc.intermediate_size,
        "vocab_size": mc.vocab_size,
        "max_position_embeddings": mc.max_position_embeddings,
        "rms_norm_eps": mc.rms_norm_eps,
        "rope_theta": mc.rope_theta,
        "tie_word_embeddings": mc.ties_embeddings(),
        "use_cache": true,
    });
    let config_json_str = serde_json::to_string_pretty(&config_json)
        .map_err(|e| Error::ConfigError(format!("Failed to serialize config.json: {e}")))?;
    std::fs::write(&config_json_path, config_json_str)?;
    println!("✓ config.json saved (realizar-compatible)");

    let metadata_path = spec.training.output_dir.join("final_model.json");
    println!("Saving metadata to {}...", metadata_path.display());
    let metadata = serde_json::json!({
        "model_path": spec.model.path,
        "weights_path": weights_path,
        "mode": "transformer",
        "training_mode": format!("{:?}", spec.training.mode),
        "epochs_completed": spec.training.epochs,
        "final_loss": metrics.losses.last().copied().unwrap_or(0.0),
        "best_loss": metrics.best_loss().unwrap_or(0.0),
        "steps": step,
    });
    let metadata_json = serde_json::to_string_pretty(&metadata)
        .map_err(|e| Error::ConfigError(format!("Failed to serialize metadata: {e}")))?;
    std::fs::write(&metadata_path, metadata_json)?;

    println!("✓ Model saved successfully");
    println!();

    Ok(())
}

/// Train a tabular model (regression/classification) from spec
///
/// Uses generic Trainer with MSELoss for regression tasks.
fn train_tabular_from_spec(spec: &TrainSpec) -> Result<()> {
    println!("✓ Config loaded and validated (Tabular mode)");
    println!("  Model: {}", spec.model.path.display());
    println!("  Optimizer: {} (lr={})", spec.optimizer.name, spec.optimizer.lr);
    println!("  Batch size: {}", spec.data.batch_size);
    println!("  Epochs: {}", spec.training.epochs);

    if let Some(lora) = &spec.lora {
        println!("  LoRA: rank={}, alpha={}", lora.rank, lora.alpha);
    }

    if let Some(quant) = &spec.quantize {
        println!("  Quantization: {}-bit", quant.bits);
    }
    println!();

    // Build model and optimizer
    println!("Building model and optimizer...");
    let model = crate::config::build_model(spec)?;
    let optimizer = crate::config::build_optimizer(&spec.optimizer)?;

    // Setup trainer
    use crate::train::{MSELoss, TrainConfig, Trainer};

    let mut train_config = TrainConfig::new().with_log_interval(100);

    if let Some(clip) = spec.training.grad_clip {
        train_config = train_config.with_grad_clip(clip);
    }

    let mut trainer = Trainer::new(
        model.parameters.into_iter().map(|(_, t)| t).collect(),
        optimizer,
        train_config,
    );
    trainer.set_loss(Box::new(MSELoss));

    println!("✓ Trainer initialized");
    println!();

    // Load training data
    println!("Loading training data...");
    let batches = load_training_batches(spec)?;
    println!("✓ {} batches created", batches.len());
    println!();

    // Training loop
    println!("Starting training...");
    println!();

    for epoch in 0..spec.training.epochs {
        let avg_loss = trainer.train_epoch(batches.clone(), Clone::clone);
        println!("Epoch {}/{}: loss={:.6}", epoch + 1, spec.training.epochs, avg_loss);
    }

    println!();
    println!("✓ Training complete");
    println!("  Final loss: {:.6}", trainer.metrics.losses.last().copied().unwrap_or(0.0));
    println!("  Best loss: {:.6}", trainer.metrics.best_loss().unwrap_or(0.0));
    println!();

    // Save the trained model
    let output_path = spec.training.output_dir.join("final_model.json");
    println!("Saving model to {}...", output_path.display());

    // Reconstruct model for saving
    let final_model = crate::io::Model::new(
        model.metadata.clone(),
        trainer
            .params()
            .iter()
            .enumerate()
            .map(|(i, t)| (format!("param_{i}"), t.clone()))
            .collect(),
    );

    use crate::io::{save_model, ModelFormat, SaveConfig};
    let save_config = SaveConfig::new(ModelFormat::Json).with_pretty(true);
    save_model(&final_model, &output_path, &save_config)?;

    println!("✓ Model saved successfully");
    println!();

    Ok(())
}

