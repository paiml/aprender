/// Execute a training plan (the "apply" phase).
///
/// For HPO plans, iterates over trials:
/// 1. `ClassifyTuner.suggest()` → hyperparameters
/// 2. Build `ClassifyConfig` → `ClassifyPipeline::from_pretrained()`
/// 3. Load corpus → `ClassifyTrainer::new()` → `trainer.train()`
/// 4. Record trial results → check scheduler
/// 5. Return `TuneResult` with leaderboard
///
/// For manual plans, executes a single trial with the specified params.
///
/// # Errors
///
/// Returns error if model weights cannot be loaded, data is invalid,
/// or all trials fail.
pub fn execute_plan(
    plan: &TrainingPlan,
    apply: &ApplyConfig,
) -> crate::Result<super::classify_tuner::TuneResult> {
    use super::classify_pipeline::ClassifyConfig;
    use super::classify_tuner::{
        ClassifyTuner, SchedulerKind, TrialSummary, TuneConfig, TuneStrategy,
    };
    use crate::transformer::TransformerConfig;
    use std::collections::HashMap;
    use crate::optim::ParameterValue;

    // ── Verify pre-conditions ──────────────────────────────────────────
    if plan.verdict == PlanVerdict::Blocked {
        return Err(crate::Error::ConfigError(
            "Cannot apply a blocked plan — resolve all failures first".to_string(),
        ));
    }

    if !apply.model_path.is_dir() {
        return Err(crate::Error::ConfigError(format!(
            "Model path does not exist: {}",
            apply.model_path.display()
        )));
    }

    if !apply.data_path.exists() {
        return Err(crate::Error::Io(format!(
            "Training data not found: {}",
            apply.data_path.display()
        )));
    }

    // Create output directory
    std::fs::create_dir_all(&apply.output_dir).map_err(|e| {
        crate::Error::Io(format!(
            "Failed to create output directory {}: {e}",
            apply.output_dir.display()
        ))
    })?;

    // ── Open project-local experiment store ────────────────────────────
    let mut tracker = ExperimentTracker::open(&apply.output_dir, plan);

    // GH-377: Resolve model config — error on unknown instead of silent tiny()
    let model_config = TransformerConfig::from_size_str(&plan.model.size)
        .map_err(|e| crate::Error::ConfigError(e))?;

    let total_start = std::time::Instant::now();

    // Auto-enable NF4 quantization for large models.
    // Full fp32 weights for hidden_size >= 2048 (roughly >= 1B params)
    // exceed RTX 4090 VRAM (24 GB) after scratch + kernel cache overhead.
    let auto_nf4 = model_config.hidden_size >= 2048;
    if auto_nf4 {
        eprintln!("[plan] Auto-enabling NF4 quantization (hidden_size={} >= 2048)", model_config.hidden_size);
    }

    // ── Manual strategy: single trial ──────────────────────────────────
    if plan.hyperparameters.strategy == "manual" {
        return execute_manual_trial(plan, apply, &model_config, auto_nf4, &mut tracker, total_start);
    }

    // ── HPO strategy: multiple trials ──────────────────────────────────
    let strategy: TuneStrategy = plan
        .hyperparameters
        .strategy
        .parse()
        .unwrap_or(TuneStrategy::Tpe);

    let num_classes = plan.data.class_counts.len();

    let tune_config = TuneConfig {
        budget: plan.hyperparameters.budget,
        strategy,
        scheduler: SchedulerKind::Asha,
        scout: plan.hyperparameters.scout,
        max_epochs: plan.hyperparameters.max_epochs,
        num_classes,
        seed: 42,
        time_limit_secs: None,
    };

    let mut tuner = ClassifyTuner::new(tune_config)?;
    let mut searcher = tuner.build_searcher();
    let scheduler = tuner.build_scheduler();

    let budget = plan.hyperparameters.budget;

    // Save plan as YAML in output dir for reproducibility
    let plan_path = apply.output_dir.join("plan.yaml");
    let _ = std::fs::write(&plan_path, plan.to_yaml());

    for trial_idx in 0..budget {
        // ── Suggest hyperparameters ────────────────────────────────────
        let suggestion = match searcher.suggest() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("  Trial {}: searcher exhausted ({e}), stopping", trial_idx + 1);
                break;
            }
        };

        let (lr, rank, alpha, batch_size, warmup, clip, weights_strategy, _targets, lr_min_ratio) =
            super::classify_tuner::extract_trial_params(&suggestion.config);

        // ── Build ClassifyConfig from trial params ─────────────────────
        let class_weights = resolve_class_weights(
            &weights_strategy,
            &plan.data.class_counts,
            num_classes,
        );

        let epochs = if plan.hyperparameters.scout {
            1
        } else {
            plan.hyperparameters.max_epochs
        };

        let classify_config = ClassifyConfig {
            num_classes,
            lora_rank: rank,
            lora_alpha: alpha,
            learning_rate: lr,
            epochs,
            batch_size,
            gradient_clip_norm: Some(clip),
            class_weights,
            quantize_nf4: auto_nf4,
            ..ClassifyConfig::default()
        };

        let trial_dir = apply.output_dir.join(format!("trial_{:03}", trial_idx + 1));
        let trial_start = std::time::Instant::now();

        eprintln!(
            "  Trial {}/{}: lr={:.2e} rank={} alpha={:.1} batch={} warmup={:.2} clip={:.1} weights={}",
            trial_idx + 1, budget, lr, rank, alpha, batch_size, warmup, clip, weights_strategy
        );

        // ── Execute trial ──────────────────────────────────────────────
        let trial_result = run_single_trial_with_warmup(
            &apply.model_path,
            &apply.data_path,
            &trial_dir,
            &model_config,
            classify_config,
            epochs,
            warmup,
            lr_min_ratio,
            &plan.model.size,
        );

        let trial_time_ms = trial_start.elapsed().as_millis() as u64;

        record_hpo_trial_outcome(
            trial_result,
            trial_idx,
            budget,
            trial_time_ms,
            &suggestion,
            &scheduler,
            &mut searcher,
            &mut tuner,
            &mut tracker,
            apply.on_trial_complete,
        );
    }

    let total_time_ms = total_start.elapsed().as_millis() as u64;

    // Save leaderboard
    let result = tuner.into_result(total_time_ms);
    let leaderboard_path = apply.output_dir.join("leaderboard.json");
    let _ = std::fs::write(
        &leaderboard_path,
        serde_json::to_string_pretty(&result).unwrap_or_default(),
    );

    Ok(result)
}

/// Record the outcome of a single HPO trial (success or failure) into the
/// searcher, tuner, and experiment tracker.
fn record_hpo_trial_outcome(
    trial_result: crate::Result<super::classify_trainer::TrainResult>,
    trial_idx: usize,
    budget: usize,
    trial_time_ms: u64,
    suggestion: &crate::optim::hpo::types::trial::Trial,
    scheduler: &dyn super::tune_searchers::TuneScheduler,
    searcher: &mut Box<dyn super::tune_searchers::TuneSearcher>,
    tuner: &mut super::classify_tuner::ClassifyTuner,
    tracker: &mut ExperimentTracker,
    on_trial_complete: Option<fn(usize, usize, &super::classify_tuner::TrialSummary)>,
) {
    use super::classify_tuner::TrialSummary;

    match trial_result {
        Ok(result) => {
            let val_loss = result.best_val_loss as f64;
            let val_accuracy = result
                .epoch_metrics
                .get(result.best_epoch)
                .map_or(0.0, |m| m.val_accuracy as f64);

            let was_pruned = scheduler.should_stop(trial_idx, result.best_epoch, val_loss);

            let status = if was_pruned {
                "pruned"
            } else if result.stopped_early {
                "stopped_early"
            } else {
                "completed"
            };

            let summary = TrialSummary {
                id: trial_idx,
                val_loss,
                val_accuracy,
                train_loss: result.epoch_metrics.last().map_or(0.0, |m| m.train_loss as f64),
                train_accuracy: result.epoch_metrics.last().map_or(0.0, |m| m.train_accuracy as f64),
                epochs_run: result.epoch_metrics.len(),
                time_ms: trial_time_ms,
                config: suggestion.config.clone(),
                status: status.to_string(),
            };

            eprintln!(
                "    => val_loss={:.4} val_acc={:.1}% epochs={} [{}]",
                val_loss, val_accuracy * 100.0, result.epoch_metrics.len(), status,
            );

            tracker.log_hpo_trial(&suggestion.config, &result, was_pruned);
            searcher.record(suggestion.clone(), val_loss, result.epoch_metrics.len());
            tuner.record_trial(summary.clone());

            if let Some(cb) = on_trial_complete {
                cb(trial_idx, budget, &summary);
            }
        }
        Err(e) => {
            eprintln!("    => FAILED: {e}");
            tracker.log_failed_trial();

            let summary = TrialSummary {
                id: trial_idx,
                val_loss: f64::INFINITY,
                val_accuracy: 0.0,
                train_loss: f64::INFINITY,
                train_accuracy: 0.0,
                epochs_run: 0,
                time_ms: trial_time_ms,
                config: suggestion.config.clone(),
                status: "failed".to_string(),
            };
            searcher.record(suggestion.clone(), f64::INFINITY, 0);
            tuner.record_trial(summary);
        }
    }
}

/// Execute the manual strategy (single trial with explicit hyperparameters).
fn execute_manual_trial(
    plan: &TrainingPlan,
    apply: &ApplyConfig,
    model_config: &crate::transformer::TransformerConfig,
    auto_nf4: bool,
    tracker: &mut ExperimentTracker,
    total_start: std::time::Instant,
) -> crate::Result<super::classify_tuner::TuneResult> {
    use super::classify_pipeline::ClassifyConfig;
    use super::classify_tuner::TrialSummary;
    use crate::optim::ParameterValue;
    use std::collections::HashMap;

    let manual = plan.hyperparameters.manual.as_ref().ok_or_else(|| {
        crate::Error::ConfigError(
            "Manual strategy requires manual hyperparameters in plan".to_string(),
        )
    })?;

    // Build ClassifyConfig from contract-grounded defaults.
    // When NF4 is enabled, start from qlora_default() (Dettmers 2023, Lightning AI).
    let model_params = estimate_model_params(model_config);
    let mut classify_config = if auto_nf4 {
        let mut cfg = ClassifyConfig::qlora_default(model_params);
        cfg.num_classes = plan.data.class_counts.len();
        cfg.lora_rank = manual.lora_rank;
        cfg.learning_rate = manual.learning_rate;
        cfg.batch_size = manual.batch_size;
        cfg.epochs = plan.hyperparameters.max_epochs;
        cfg.lora_alpha = manual.lora_alpha.unwrap_or((2 * manual.lora_rank) as f32);
        cfg.gradient_clip_norm = Some(manual.gradient_clip_norm.unwrap_or(1.0));
        let target_eff_batch: usize = 16;
        cfg.accumulation_steps = (target_eff_batch / manual.batch_size.max(1)).max(1);
        cfg
    } else {
        ClassifyConfig {
            num_classes: plan.data.class_counts.len(),
            lora_rank: manual.lora_rank,
            lora_alpha: manual.lora_alpha.unwrap_or(manual.lora_rank as f32),
            learning_rate: manual.learning_rate,
            epochs: plan.hyperparameters.max_epochs,
            batch_size: manual.batch_size,
            gradient_clip_norm: Some(manual.gradient_clip_norm.unwrap_or(1.0)),
            quantize_nf4: false,
            ..ClassifyConfig::default()
        }
    };
    classify_config.quantize_nf4 = auto_nf4;

    let diags = classify_config.validate_hyperparameters(model_params);
    if !diags.items.is_empty() {
        eprintln!("[plan] Hyperparameter contract validation (qlora-hyperparameters-v1.yaml):");
        diags.print_all();
    }
    if diags.has_errors() {
        return Err(crate::Error::ConfigError(
            "Hyperparameter validation failed — see errors above".to_string(),
        ));
    }

    let trial_start = std::time::Instant::now();
    let result = run_single_trial(
        &apply.model_path,
        &apply.data_path,
        &apply.output_dir.join("trial_001"),
        model_config,
        classify_config,
        plan.hyperparameters.max_epochs,
        &plan.model.size,
    )?;

    let mut config_map = HashMap::new();
    config_map.insert(
        "learning_rate".to_string(),
        ParameterValue::Float(manual.learning_rate as f64),
    );
    config_map.insert(
        "lora_rank".to_string(),
        ParameterValue::Int(manual.lora_rank as i64),
    );
    config_map.insert(
        "batch_size".to_string(),
        ParameterValue::Categorical(manual.batch_size.to_string()),
    );

    let summary = TrialSummary {
        id: 0,
        val_loss: result.best_val_loss as f64,
        val_accuracy: result
            .epoch_metrics
            .get(result.best_epoch)
            .map_or(0.0, |m| m.val_accuracy as f64),
        train_loss: result
            .epoch_metrics
            .last()
            .map_or(0.0, |m| m.train_loss as f64),
        train_accuracy: result
            .epoch_metrics
            .last()
            .map_or(0.0, |m| m.train_accuracy as f64),
        epochs_run: result.epoch_metrics.len(),
        time_ms: trial_start.elapsed().as_millis() as u64,
        config: config_map,
        status: if result.stopped_early {
            "stopped_early".to_string()
        } else {
            "completed".to_string()
        },
    };

    tracker.log_manual_trial(manual, &result);

    if let Some(cb) = apply.on_trial_complete {
        cb(0, 1, &summary);
    }

    Ok(super::classify_tuner::TuneResult {
        strategy: "manual".to_string(),
        mode: "manual".to_string(),
        budget: 1,
        trials: vec![summary],
        best_trial_id: 0,
        total_time_ms: total_start.elapsed().as_millis() as u64,
    })
}

/// Execute a single training trial with default warmup/LR settings.
fn run_single_trial(
    model_path: &std::path::Path,
    data_path: &std::path::Path,
    checkpoint_dir: &std::path::Path,
    model_config: &crate::transformer::TransformerConfig,
    classify_config: super::classify_pipeline::ClassifyConfig,
    epochs: usize,
    model_name: &str,
) -> crate::Result<super::classify_trainer::TrainResult> {
    run_single_trial_with_warmup(
        model_path,
        data_path,
        checkpoint_dir,
        model_config,
        classify_config,
        epochs,
        0.1,   // default warmup fraction
        0.01,  // default lr_min_ratio
        model_name,
    )
}

/// Execute a single training trial with explicit warmup/LR min parameters.
fn run_single_trial_with_warmup(
    model_path: &std::path::Path,
    data_path: &std::path::Path,
    checkpoint_dir: &std::path::Path,
    model_config: &crate::transformer::TransformerConfig,
    classify_config: super::classify_pipeline::ClassifyConfig,
    epochs: usize,
    warmup_fraction: f32,
    lr_min_ratio: f32,
    model_name: &str,
) -> crate::Result<super::classify_trainer::TrainResult> {
    use super::classify_pipeline::ClassifyPipeline;
    use super::classify_trainer::{ClassifyTrainer, TrainingConfig};

    // Create checkpoint directory
    std::fs::create_dir_all(checkpoint_dir).map_err(|e| {
        crate::Error::Io(format!(
            "Failed to create checkpoint dir {}: {e}",
            checkpoint_dir.display()
        ))
    })?;

    // Load pipeline with pretrained weights
    let pipeline =
        ClassifyPipeline::from_pretrained(model_path, model_config, classify_config)?;

    // Load corpus
    let samples = pipeline.load_corpus(data_path)?;

    let lr_min = pipeline.config.learning_rate * lr_min_ratio;

    // Build training config
    let training_config = TrainingConfig {
        epochs,
        val_split: 0.2,
        save_every: 1,
        early_stopping_patience: 5,
        checkpoint_dir: checkpoint_dir.to_path_buf(),
        seed: 42,
        log_interval: 1,
        warmup_fraction,
        lr_min,
    };

    // Create trainer
    let mut trainer = ClassifyTrainer::new(pipeline, samples, training_config)?;

    // Attach monitor writer
    let experiment_id = format!(
        "trial-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    );
    let writer = crate::monitor::tui::TrainingStateWriter::new(
        checkpoint_dir,
        &experiment_id,
        model_name,
    )
    .with_console_progress(true);
    trainer.set_monitor_writer(writer);

    // Run training
    Ok(trainer.train())
}

/// Resolve class weights from strategy name and class counts.
fn resolve_class_weights(
    strategy: &str,
    class_counts: &[usize],
    num_classes: usize,
) -> Option<Vec<f32>> {
    use super::classification::{compute_class_weights, ClassWeightStrategy, SafetyCorpusStats};

    match strategy {
        "uniform" => None,
        "inverse_freq" => {
            let stats = SafetyCorpusStats {
                total: class_counts.iter().sum(),
                class_counts: class_counts.to_vec(),
                avg_input_len: 0,
            };
            Some(compute_class_weights(
                &stats,
                ClassWeightStrategy::InverseFreq,
                num_classes,
            ))
        }
        "sqrt_inverse" => {
            let stats = SafetyCorpusStats {
                total: class_counts.iter().sum(),
                class_counts: class_counts.to_vec(),
                avg_input_len: 0,
            };
            Some(compute_class_weights(
                &stats,
                ClassWeightStrategy::SqrtInverse,
                num_classes,
            ))
        }
        _ => None,
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Display helpers (for CLI consumption)
// ═══════════════════════════════════════════════════════════════════════

impl TrainingPlan {
    /// Serialize to pretty JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }

    /// Serialize to YAML.
    pub fn to_yaml(&self) -> String {
        serde_yaml::to_string(self).unwrap_or_default()
    }

    /// Deserialize from a string (auto-detects JSON or YAML).
    pub fn from_str(s: &str) -> crate::Result<Self> {
        // Try JSON first (faster, more common for programmatic use)
        if let Ok(plan) = serde_json::from_str::<TrainingPlan>(s) {
            return Ok(plan);
        }
        // Fall back to YAML
        serde_yaml::from_str::<TrainingPlan>(s).map_err(|e| {
            crate::Error::ConfigError(format!("Failed to parse plan as JSON or YAML: {e}"))
        })
    }

    /// Count pre-flight checks by status.
    pub fn check_counts(&self) -> (usize, usize, usize) {
        let pass = self.pre_flight.iter().filter(|c| c.status == CheckStatus::Pass).count();
        let warn = self.pre_flight.iter().filter(|c| c.status == CheckStatus::Warn).count();
        let fail = self.pre_flight.iter().filter(|c| c.status == CheckStatus::Fail).count();
        (pass, warn, fail)
    }
}

// ── Experiment store integration ──────────────────────────────────────────

/// Thin wrapper around SqliteBackend for logging training experiments.
/// All methods are best-effort (errors silently ignored) so training is
/// never blocked by storage failures.
struct ExperimentTracker {
    store: Option<crate::storage::SqliteBackend>,
    exp_id: Option<String>,
}

impl ExperimentTracker {
    fn open(output_dir: &std::path::Path, plan: &TrainingPlan) -> Self {
        use crate::storage::{ExperimentStorage, SqliteBackend};

        let mut store = SqliteBackend::open_project(output_dir).ok();
        let exp_id = store.as_mut().and_then(|s| {
            let config_json = serde_json::json!({
                "model": &plan.model.architecture,
                "size": &plan.model.size,
                "strategy": &plan.hyperparameters.strategy,
                "budget": plan.hyperparameters.budget,
                "num_classes": plan.data.class_counts.len(),
            });
            s.create_experiment(&plan.model.architecture, Some(config_json)).ok()
        });
        Self { store, exp_id }
    }

    fn log_manual_trial(
        &mut self,
        manual: &ManualConfig,
        result: &super::classify_trainer::TrainResult,
    ) {
        use crate::storage::{ExperimentStorage, ParameterValue as SPV};
        let (store, eid) = match (self.store.as_mut(), self.exp_id.as_ref()) {
            (Some(s), Some(e)) => (s, e),
            _ => return,
        };
        let run_id = match store.create_run(eid) {
            Ok(id) => id,
            Err(_) => return,
        };
        let _ = store.start_run(&run_id);
        let _ = store.log_param(&run_id, "learning_rate", SPV::Float(f64::from(manual.learning_rate)));
        let _ = store.log_param(&run_id, "lora_rank", SPV::Int(manual.lora_rank as i64));
        let _ = store.log_param(&run_id, "batch_size", SPV::Int(manual.batch_size as i64));
        Self::log_epoch_metrics(store, &run_id, &result.epoch_metrics);
        let _ = store.complete_run(&run_id, crate::storage::RunStatus::Success);
    }

    fn log_hpo_trial(
        &mut self,
        config: &std::collections::HashMap<String, crate::optim::ParameterValue>,
        result: &super::classify_trainer::TrainResult,
        was_pruned: bool,
    ) {
        use crate::storage::{ExperimentStorage, ParameterValue as SPV};
        use crate::optim::ParameterValue as OPV;
        let (store, eid) = match (self.store.as_mut(), self.exp_id.as_ref()) {
            (Some(s), Some(e)) => (s, e),
            _ => return,
        };
        let run_id = match store.create_run(eid) {
            Ok(id) => id,
            Err(_) => return,
        };
        let _ = store.start_run(&run_id);
        for (k, v) in config {
            let sv = match v {
                OPV::Float(f) => SPV::Float(*f),
                OPV::Int(i) => SPV::Int(*i),
                OPV::Categorical(s) => SPV::String(s.clone()),
            };
            let _ = store.log_param(&run_id, k, sv);
        }
        Self::log_epoch_metrics(store, &run_id, &result.epoch_metrics);
        let status = if was_pruned {
            crate::storage::RunStatus::Cancelled
        } else {
            crate::storage::RunStatus::Success
        };
        let _ = store.complete_run(&run_id, status);
    }

    fn log_failed_trial(&mut self) {
        use crate::storage::ExperimentStorage;
        let (store, eid) = match (self.store.as_mut(), self.exp_id.as_ref()) {
            (Some(s), Some(e)) => (s, e),
            _ => return,
        };
        if let Ok(run_id) = store.create_run(eid) {
            let _ = store.start_run(&run_id);
            let _ = store.complete_run(&run_id, crate::storage::RunStatus::Failed);
        }
    }

    fn log_epoch_metrics(
        store: &mut crate::storage::SqliteBackend,
        run_id: &str,
        epochs: &[super::classify_trainer::EpochMetrics],
    ) {
        use crate::storage::ExperimentStorage;
        for (i, epoch) in epochs.iter().enumerate() {
            let _ = store.log_metric(run_id, "train_loss", i as u64, f64::from(epoch.train_loss));
            let _ = store.log_metric(run_id, "val_loss", i as u64, f64::from(epoch.val_loss));
            let _ = store.log_metric(run_id, "val_accuracy", i as u64, f64::from(epoch.val_accuracy));
        }
    }
}

/// Estimate total model parameters from config dimensions.
///
/// Rough approximation: 12 * num_layers * hidden_size^2
/// (accounts for Q/K/V/O projections + FFN up/down/gate + embeddings).
///
/// Used by C-HP-001 to select learning rate tier.
fn estimate_model_params(config: &TransformerConfig) -> u64 {
    let h = config.hidden_size as u64;
    let l = config.num_hidden_layers as u64;
    // Transformer parameter estimate:
    // Attention: 4 * h * h per layer (Q, K, V, O)
    // FFN: 3 * h * intermediate per layer (gate, up, down)
    // Norms: 2 * h per layer
    // Embedding: vocab * h
    let intermediate = config.intermediate_size as u64;
    let vocab = config.vocab_size as u64;
    let per_layer = 4 * h * h + 3 * h * intermediate + 2 * h;
    l * per_layer + vocab * h
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_plan_missing_data_file() {
        let config = PlanConfig {
            task: "classify".to_string(),
            data_path: PathBuf::from("/nonexistent/data.jsonl"),
            val_path: None,
            test_path: None,
            model_size: "0.5B".to_string(),
            model_path: None,
            num_classes: 5,
            output_dir: PathBuf::from("/tmp/test-plan-out"),
            strategy: "manual".to_string(),
            budget: 10,
            scout: false,
            max_epochs: 1,
            manual_lr: Some(1e-4),
            manual_lora_rank: Some(16),
            manual_batch_size: Some(32),
        };
        let result = plan(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_plan_manual_strategy_warns() {
        // Create a temp JSONL file
        let dir = tempfile::tempdir().expect("valid");
        let data_path = dir.path().join("train.jsonl");
        let mut lines = Vec::new();
        for i in 0..50 {
            lines.push(format!(
                r#"{{"input": "echo test {i}", "label": {}}}"#,
                i % 5
            ));
        }
        std::fs::write(&data_path, lines.join("\n")).expect("valid");

        let config = PlanConfig {
            task: "classify".to_string(),
            data_path,
            val_path: None,
            test_path: None,
            model_size: "0.5B".to_string(),
            model_path: None,
            num_classes: 5,
            output_dir: dir.path().to_path_buf(),
            strategy: "manual".to_string(),
            budget: 10,
            scout: false,
            max_epochs: 1,
            manual_lr: Some(1e-4),
            manual_lora_rank: Some(16),
            manual_batch_size: Some(32),
        };
        let p = plan(&config).expect("valid");
        assert_eq!(p.hyperparameters.strategy, "manual");
        assert!(p.issues.iter().any(|i| i.category == "Hyperparameters"));
        assert!(p.hyperparameters.recommendation.is_some());
    }

    #[test]
    fn test_plan_tpe_strategy_generates_previews() {
        let dir = tempfile::tempdir().expect("valid");
        let data_path = dir.path().join("train.jsonl");
        let mut lines = Vec::new();
        for i in 0..100 {
            lines.push(format!(
                r#"{{"input": "echo test {i}", "label": {}}}"#,
                i % 5
            ));
        }
        std::fs::write(&data_path, lines.join("\n")).expect("valid");

        let config = PlanConfig {
            task: "classify".to_string(),
            data_path,
            val_path: None,
            test_path: None,
            model_size: "0.5B".to_string(),
            model_path: None,
            num_classes: 5,
            output_dir: dir.path().to_path_buf(),
            strategy: "tpe".to_string(),
            budget: 20,
            scout: true,
            max_epochs: 1,
            manual_lr: None,
            manual_lora_rank: None,
            manual_batch_size: None,
        };
        let p = plan(&config).expect("valid");
        assert_eq!(p.hyperparameters.strategy, "tpe");
        assert_eq!(p.hyperparameters.budget, 20);
        assert!(p.hyperparameters.scout);
        assert!(!p.hyperparameters.sample_configs.is_empty());
        assert_eq!(p.hyperparameters.search_space_params, 9);
    }

    #[test]
    fn test_plan_detects_imbalance() {
        let dir = tempfile::tempdir().expect("valid");
        let data_path = dir.path().join("train.jsonl");
        let mut lines = Vec::new();
        // 80 class 0, 10 each for classes 1-4 = 8:1 imbalance
        for i in 0..80 {
            lines.push(format!(
                r#"{{"input": "safe command {i}", "label": 0}}"#
            ));
        }
        for c in 1..5 {
            for i in 0..10 {
                lines.push(format!(
                    r#"{{"input": "class {c} cmd {i}", "label": {c}}}"#
                ));
            }
        }
        std::fs::write(&data_path, lines.join("\n")).expect("valid");

        let config = PlanConfig {
            task: "classify".to_string(),
            data_path,
            val_path: None,
            test_path: None,
            model_size: "0.5B".to_string(),
            model_path: None,
            num_classes: 5,
            output_dir: dir.path().to_path_buf(),
            strategy: "tpe".to_string(),
            budget: 10,
            scout: true,
            max_epochs: 1,
            manual_lr: None,
            manual_lora_rank: None,
            manual_batch_size: None,
        };
        let p = plan(&config).expect("valid");
        assert!(p.data.imbalance_ratio > 5.0);
        assert!(p.data.auto_class_weights);
        assert!(p.issues.iter().any(|i| i.message.contains("imbalance")));
    }

    #[test]
    fn test_plan_detects_duplicates() {
        let dir = tempfile::tempdir().expect("valid");
        let data_path = dir.path().join("train.jsonl");
        let mut lines = Vec::new();
        for i in 0..50 {
            lines.push(format!(
                r#"{{"input": "echo test {i}", "label": {}}}"#,
                i % 5
            ));
        }
        // Add 5 exact duplicates
        for _ in 0..5 {
            lines.push(r#"{"input": "echo test 0", "label": 0}"#.to_string());
        }
        std::fs::write(&data_path, lines.join("\n")).expect("valid");

        let config = PlanConfig {
            task: "classify".to_string(),
            data_path,
            val_path: None,
            test_path: None,
            model_size: "0.5B".to_string(),
            model_path: None,
            num_classes: 5,
            output_dir: dir.path().to_path_buf(),
            strategy: "manual".to_string(),
            budget: 0,
            scout: false,
            max_epochs: 1,
            manual_lr: Some(1e-4),
            manual_lora_rank: Some(16),
            manual_batch_size: Some(32),
        };
        let p = plan(&config).expect("valid");
        assert!(p.data.duplicates > 0);
        assert!(p.issues.iter().any(|i| i.message.contains("duplicate")));
    }

    #[test]
    fn test_plan_serialization_roundtrip() {
        let dir = tempfile::tempdir().expect("valid");
        let data_path = dir.path().join("train.jsonl");
        let mut lines = Vec::new();
        for i in 0..20 {
            lines.push(format!(
                r#"{{"input": "echo {i}", "label": {}}}"#,
                i % 5
            ));
        }
        std::fs::write(&data_path, lines.join("\n")).expect("valid");

        let config = PlanConfig {
            task: "classify".to_string(),
            data_path,
            val_path: None,
            test_path: None,
            model_size: "0.5B".to_string(),
            model_path: None,
            num_classes: 5,
            output_dir: dir.path().to_path_buf(),
            strategy: "tpe".to_string(),
            budget: 5,
            scout: true,
            max_epochs: 1,
            manual_lr: None,
            manual_lora_rank: None,
            manual_batch_size: None,
        };
        let p = plan(&config).expect("valid");

        // JSON roundtrip
        let json = p.to_json();
        let deserialized: TrainingPlan = serde_json::from_str(&json).expect("valid");
        assert_eq!(deserialized.version, "1.0");
        assert_eq!(deserialized.task, "classify");
        assert_eq!(deserialized.data.train_samples, p.data.train_samples);

        // YAML roundtrip
        let yaml = p.to_yaml();
        let deserialized_yaml: TrainingPlan = serde_yaml::from_str(&yaml).expect("valid");
        assert_eq!(deserialized_yaml.data.train_samples, p.data.train_samples);
    }

    #[test]
    fn test_plan_resource_estimation() {
        let dir = tempfile::tempdir().expect("valid");
        let data_path = dir.path().join("train.jsonl");
        let mut lines = Vec::new();
        for i in 0..100 {
            lines.push(format!(
                r#"{{"input": "echo test {i}", "label": {}}}"#,
                i % 5
            ));
        }
        std::fs::write(&data_path, lines.join("\n")).expect("valid");

        let config = PlanConfig {
            task: "classify".to_string(),
            data_path,
            val_path: None,
            test_path: None,
            model_size: "0.5B".to_string(),
            model_path: None,
            num_classes: 5,
            output_dir: dir.path().to_path_buf(),
            strategy: "manual".to_string(),
            budget: 0,
            scout: false,
            max_epochs: 1,
            manual_lr: Some(1e-4),
            manual_lora_rank: Some(16),
            manual_batch_size: Some(32),
        };
        let p = plan(&config).expect("valid");
        assert!(p.resources.estimated_vram_gb > 0.0);
        assert!(p.resources.steps_per_epoch > 0);
        assert!(p.resources.estimated_checkpoint_mb > 0.0);
    }

    #[test]
    fn test_plan_verdict_ready() {
        let dir = tempfile::tempdir().expect("valid");
        let data_path = dir.path().join("train.jsonl");
        let mut lines = Vec::new();
        for i in 0..100 {
            lines.push(format!(
                r#"{{"input": "echo test {i}", "label": {}}}"#,
                i % 5
            ));
        }
        std::fs::write(&data_path, lines.join("\n")).expect("valid");

        let config = PlanConfig {
            task: "classify".to_string(),
            data_path,
            val_path: None,
            test_path: None,
            model_size: "0.5B".to_string(),
            model_path: None,
            num_classes: 5,
            output_dir: dir.path().to_path_buf(),
            strategy: "tpe".to_string(),
            budget: 20,
            scout: true,
            max_epochs: 1,
            manual_lr: None,
            manual_lora_rank: None,
            manual_batch_size: None,
        };
        let p = plan(&config).expect("valid");
        // Should be WarningsPresent (model_path not specified)
        assert_ne!(p.verdict, PlanVerdict::Blocked);
    }

    #[test]
    fn test_plan_model_info_qwen2() {
        let dir = tempfile::tempdir().expect("valid");
        let data_path = dir.path().join("train.jsonl");
        let mut lines = Vec::new();
        for i in 0..20 {
            lines.push(format!(
                r#"{{"input": "echo {i}", "label": {}}}"#,
                i % 5
            ));
        }
        std::fs::write(&data_path, lines.join("\n")).expect("valid");

        let config = PlanConfig {
            task: "classify".to_string(),
            data_path,
            val_path: None,
            test_path: None,
            model_size: "0.5B".to_string(),
            model_path: None,
            num_classes: 5,
            output_dir: dir.path().to_path_buf(),
            strategy: "manual".to_string(),
            budget: 0,
            scout: false,
            max_epochs: 1,
            manual_lr: Some(1e-4),
            manual_lora_rank: Some(16),
            manual_batch_size: Some(32),
        };
        let p = plan(&config).expect("valid");
        assert_eq!(p.model.hidden_size, 896);
        assert_eq!(p.model.num_layers, 24);
        assert_eq!(p.model.architecture, "qwen2");
        assert!(p.model.lora_trainable_params > 0);
        assert!(p.model.classifier_params > 0);
    }

    #[test]
    fn test_execute_plan_rejects_blocked() {
        let blocked_plan = TrainingPlan {
            version: "1.0".to_string(),
            task: "classify".to_string(),
            data: DataAudit {
                train_path: "/tmp/nonexistent.jsonl".to_string(),
                train_samples: 0,
                avg_input_len: 0,
                class_counts: vec![0; 5],
                imbalance_ratio: 1.0,
                auto_class_weights: false,
                val_samples: None,
                test_samples: None,
                duplicates: 0,
                preamble_count: 0,
            },
            model: ModelInfo {
                size: "0.5B".to_string(),
                hidden_size: 896,
                num_layers: 24,
                architecture: "qwen2".to_string(),
                weights_available: false,
                lora_trainable_params: 0,
                classifier_params: 0,
            },
            hyperparameters: HyperparameterPlan {
                strategy: "manual".to_string(),
                budget: 0,
                scout: false,
                max_epochs: 1,
                search_space_params: 0,
                sample_configs: Vec::new(),
                manual: None,
                recommendation: None,
            },
            resources: ResourceEstimate {
                estimated_vram_gb: 0.0,
                estimated_minutes_per_epoch: 0.0,
                estimated_total_minutes: 0.0,
                estimated_checkpoint_mb: 0.0,
                steps_per_epoch: 0,
                gpu_device: None,
            },
            pre_flight: vec![PreFlightCheck {
                name: "data_file".to_string(),
                status: CheckStatus::Fail,
                detail: "Data not found".to_string(),
            }],
            output_dir: "/tmp/test".to_string(),
            auto_diagnose: false,
            verdict: PlanVerdict::Blocked,
            issues: Vec::new(),
        };

        let apply = ApplyConfig {
            model_path: PathBuf::from("/tmp/nonexistent"),
            data_path: PathBuf::from("/tmp/nonexistent.jsonl"),
            output_dir: PathBuf::from("/tmp/test-apply"),
            on_trial_complete: None,
        };

        let result = execute_plan(&blocked_plan, &apply);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("blocked"), "Error should mention blocked: {err_msg}");
    }

    #[test]
    fn test_resolve_class_weights_uniform() {
        let weights = resolve_class_weights("uniform", &[100, 200, 300], 3);
        assert!(weights.is_none());
    }

    #[test]
    fn test_resolve_class_weights_sqrt_inverse() {
        let weights = resolve_class_weights("sqrt_inverse", &[100, 200, 300], 3);
        assert!(weights.is_some());
        let w = weights.expect("valid");
        assert_eq!(w.len(), 3);
        // Largest class should have smallest weight
        assert!(w[0] > w[2], "class 0 (100 samples) should have higher weight than class 2 (300 samples)");
    }

    #[test]
    fn test_resolve_class_weights_inverse_freq() {
        let weights = resolve_class_weights("inverse_freq", &[100, 200, 300], 3);
        assert!(weights.is_some());
        let w = weights.expect("valid");
        assert_eq!(w.len(), 3);
        assert!(w[0] > w[2]);
    }

    #[test]
    fn test_resolve_class_weights_unknown() {
        let weights = resolve_class_weights("bogus", &[100, 200], 2);
        assert!(weights.is_none());
    }

    #[test]
    fn test_execute_plan_rejects_missing_model_path() {
        let plan = TrainingPlan {
            version: "1.0".to_string(),
            task: "classify".to_string(),
            data: DataAudit {
                train_path: "/tmp/data.jsonl".to_string(),
                train_samples: 100,
                avg_input_len: 50,
                class_counts: vec![50, 50],
                imbalance_ratio: 1.0,
                auto_class_weights: false,
                val_samples: None,
                test_samples: None,
                duplicates: 0,
                preamble_count: 0,
            },
            model: ModelInfo {
                size: "0.5B".to_string(),
                hidden_size: 896,
                num_layers: 24,
                architecture: "qwen2".to_string(),
                weights_available: true,
                lora_trainable_params: 1000,
                classifier_params: 100,
            },
            hyperparameters: HyperparameterPlan {
                strategy: "manual".to_string(),
                budget: 0,
                scout: false,
                max_epochs: 1,
                search_space_params: 0,
                sample_configs: Vec::new(),
                manual: Some(ManualConfig {
                    learning_rate: 1e-4,
                    lora_rank: 16,
                    batch_size: 32,
                }),
                recommendation: None,
            },
            resources: ResourceEstimate {
                estimated_vram_gb: 2.5,
                estimated_minutes_per_epoch: 1.0,
                estimated_total_minutes: 1.0,
                estimated_checkpoint_mb: 1.0,
                steps_per_epoch: 4,
                gpu_device: None,
            },
            pre_flight: Vec::new(),
            output_dir: "/tmp/test".to_string(),
            auto_diagnose: false,
            verdict: PlanVerdict::Ready,
            issues: Vec::new(),
        };

        let apply = ApplyConfig {
            model_path: PathBuf::from("/tmp/definitely-not-a-real-model-dir"),
            data_path: PathBuf::from("/tmp/data.jsonl"),
            output_dir: PathBuf::from("/tmp/test-apply"),
            on_trial_complete: None,
        };

        let result = execute_plan(&plan, &apply);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("Model path") || err_msg.contains("does not exist"),
            "Error should mention model path: {err_msg}"
        );
    }
}
