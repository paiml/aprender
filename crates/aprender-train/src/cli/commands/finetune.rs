//! Finetune command implementation (plan/apply classification training)

use crate::cli::logging::log;
use crate::cli::LogLevel;
use crate::config::{FinetuneArgs, FinetuneCommand};

pub fn run_finetune(args: FinetuneArgs, level: LogLevel) -> Result<(), String> {
    match args.command {
        FinetuneCommand::Plan {
            data,
            model_path,
            model_size,
            num_classes,
            output_dir,
            strategy,
            budget,
            scout,
            max_epochs,
            lr,
            lora_rank,
            batch_size,
            lora_alpha,
            warmup,
            gradient_clip,
            lr_min_ratio,
            class_weights,
            target_modules,
        } => run_plan(
            data,
            model_path,
            model_size,
            num_classes,
            output_dir,
            strategy,
            budget,
            scout,
            max_epochs,
            lr,
            lora_rank,
            batch_size,
            lora_alpha,
            warmup,
            gradient_clip,
            lr_min_ratio,
            class_weights,
            target_modules,
            level,
        ),
        FinetuneCommand::Apply { plan, model_path, data, output_dir } => {
            run_apply(plan, model_path, data, output_dir, level)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_plan(
    data: std::path::PathBuf,
    model_path: Option<std::path::PathBuf>,
    model_size: String,
    num_classes: usize,
    output_dir: std::path::PathBuf,
    strategy: String,
    budget: usize,
    scout: bool,
    max_epochs: usize,
    manual_lr: Option<f32>,
    manual_lora_rank: Option<usize>,
    manual_batch_size: Option<usize>,
    manual_lora_alpha: Option<f32>,
    manual_warmup: Option<f32>,
    manual_gradient_clip: Option<f32>,
    manual_lr_min_ratio: Option<f32>,
    manual_class_weights: Option<String>,
    manual_target_modules: Option<String>,
    level: LogLevel,
) -> Result<(), String> {
    use crate::finetune::training_plan::{plan, PlanConfig};

    log(level, LogLevel::Normal, "Generating training plan...");

    let config = PlanConfig {
        task: "classify".to_string(),
        data_path: data,
        val_path: None,
        test_path: None,
        model_size,
        model_path,
        num_classes,
        output_dir: output_dir.clone(),
        strategy,
        budget,
        scout,
        max_epochs,
        manual_lr,
        manual_lora_rank,
        manual_batch_size,
        manual_lora_alpha,
        manual_warmup,
        manual_gradient_clip,
        manual_lr_min_ratio,
        manual_class_weights,
        manual_target_modules,
    };

    let training_plan = plan(&config).map_err(|e| format!("Plan generation failed: {e}"))?;

    // Print plan summary
    print_plan_summary(&training_plan, level);

    // Save plan to output dir
    std::fs::create_dir_all(&output_dir)
        .map_err(|e| format!("Failed to create output dir: {e}"))?;
    let plan_path = output_dir.join("plan.yaml");
    std::fs::write(&plan_path, training_plan.to_yaml())
        .map_err(|e| format!("Failed to write plan: {e}"))?;

    log(level, LogLevel::Normal, &format!("Plan saved to: {}", plan_path.display()));
    log(
        level,
        LogLevel::Normal,
        &format!(
            "\nTo execute: apr finetune apply --plan {} --model-path <MODEL_DIR> --data <DATA.jsonl> -o {}",
            plan_path.display(),
            output_dir.display()
        ),
    );

    Ok(())
}

fn run_apply(
    plan_path: std::path::PathBuf,
    model_path: std::path::PathBuf,
    data_path: std::path::PathBuf,
    output_dir: std::path::PathBuf,
    level: LogLevel,
) -> Result<(), String> {
    use crate::finetune::training_plan::{execute_plan, ApplyConfig, TrainingPlan};

    log(level, LogLevel::Normal, &format!("Loading plan from: {}", plan_path.display()));

    let plan_str =
        std::fs::read_to_string(&plan_path).map_err(|e| format!("Failed to read plan: {e}"))?;
    let plan =
        TrainingPlan::from_str(&plan_str).map_err(|e| format!("Failed to parse plan: {e}"))?;

    print_plan_summary(&plan, level);

    log(level, LogLevel::Normal, &format!("Model: {}", model_path.display()));
    log(level, LogLevel::Normal, &format!("Data:  {}", data_path.display()));
    log(level, LogLevel::Normal, &format!("Output: {}", output_dir.display()));
    log(level, LogLevel::Normal, "");
    log(level, LogLevel::Normal, "Starting training...");

    let apply = ApplyConfig {
        model_path,
        data_path,
        output_dir,
        on_trial_complete: Some(|trial_id, total, summary| {
            eprintln!(
                "  [{}/{}] val_loss={:.4} val_acc={:.1}% [{}]",
                trial_id + 1,
                total,
                summary.val_loss,
                summary.val_accuracy * 100.0,
                summary.status,
            );
        }),
    };

    let result = execute_plan(&plan, &apply).map_err(|e| format!("Training failed: {e}"))?;

    // Print results
    log(level, LogLevel::Normal, "");
    log(level, LogLevel::Normal, "Training complete!");
    log(
        level,
        LogLevel::Normal,
        &format!(
            "  Strategy: {} | Trials: {} | Time: {:.1}s",
            result.strategy,
            result.trials.len(),
            result.total_time_ms as f64 / 1000.0,
        ),
    );

    if let Some(best) = result.trials.get(result.best_trial_id) {
        log(level, LogLevel::Normal, &format!("  Best trial #{}", result.best_trial_id + 1));
        log(
            level,
            LogLevel::Normal,
            &format!(
                "    val_loss={:.4} val_acc={:.1}% epochs={}",
                best.val_loss,
                best.val_accuracy * 100.0,
                best.epochs_run,
            ),
        );
    }

    Ok(())
}

fn print_plan_summary(plan: &crate::finetune::training_plan::TrainingPlan, level: LogLevel) {
    let (pass, warn, fail) = plan.check_counts();

    log(level, LogLevel::Normal, &format!("Plan: {} v{}", plan.task, plan.version));
    log(
        level,
        LogLevel::Normal,
        &format!(
            "  Data: {} samples, {} classes",
            plan.data.train_samples,
            plan.data.class_counts.len(),
        ),
    );
    if plan.data.imbalance_ratio > 2.0 {
        log(
            level,
            LogLevel::Normal,
            &format!(
                "  Imbalance: {:.1}x (auto class weights: {})",
                plan.data.imbalance_ratio, plan.data.auto_class_weights,
            ),
        );
    }
    log(
        level,
        LogLevel::Normal,
        &format!(
            "  Model: {} ({}, {} layers, hidden={})",
            plan.model.architecture, plan.model.size, plan.model.num_layers, plan.model.hidden_size,
        ),
    );
    log(
        level,
        LogLevel::Normal,
        &format!(
            "  HPO: {} (budget={}, scout={}, max_epochs={})",
            plan.hyperparameters.strategy,
            plan.hyperparameters.budget,
            plan.hyperparameters.scout,
            plan.hyperparameters.max_epochs,
        ),
    );
    log(
        level,
        LogLevel::Normal,
        &format!(
            "  Resources: {:.1} GB VRAM, {:.0} min/epoch, {:.0} min total",
            plan.resources.estimated_vram_gb,
            plan.resources.estimated_minutes_per_epoch,
            plan.resources.estimated_total_minutes,
        ),
    );
    log(level, LogLevel::Normal, &format!("  Pre-flight: {pass} pass, {warn} warn, {fail} fail"));
    log(level, LogLevel::Normal, &format!("  Verdict: {:?}", plan.verdict));
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use crate::finetune::training_plan::{
        CheckStatus, DataAudit, HyperparameterPlan, ModelInfo, PlanVerdict, PreFlightCheck,
        ResourceEstimate, TrainingPlan,
    };

    fn make_plan() -> TrainingPlan {
        TrainingPlan {
            version: "1.0".to_string(),
            task: "classify".to_string(),
            data: DataAudit {
                train_path: "/data/train.jsonl".to_string(),
                train_samples: 1000,
                avg_input_len: 50,
                class_counts: vec![800, 200],
                imbalance_ratio: 4.0,
                auto_class_weights: true,
                val_samples: Some(100),
                test_samples: None,
                duplicates: 0,
                preamble_count: 0,
            },
            model: ModelInfo {
                size: "0.5B".to_string(),
                hidden_size: 896,
                num_layers: 24,
                architecture: "Qwen2".to_string(),
                weights_available: true,
                lora_trainable_params: 1_000_000,
                classifier_params: 1792,
            },
            hyperparameters: HyperparameterPlan {
                strategy: "tpe".to_string(),
                budget: 10,
                scout: false,
                max_epochs: 5,
                search_space_params: 6,
                sample_configs: vec![],
                manual: None,
                recommendation: None,
            },
            resources: ResourceEstimate {
                estimated_vram_gb: 6.5,
                estimated_minutes_per_epoch: 2.0,
                estimated_total_minutes: 100.0,
                estimated_checkpoint_mb: 50.0,
                steps_per_epoch: 32,
                gpu_device: Some("RTX 4090".to_string()),
            },
            pre_flight: vec![
                PreFlightCheck {
                    name: "data_exists".to_string(),
                    status: CheckStatus::Pass,
                    detail: "ok".to_string(),
                },
                PreFlightCheck {
                    name: "vram_check".to_string(),
                    status: CheckStatus::Warn,
                    detail: "tight".to_string(),
                },
            ],
            output_dir: "/tmp/output".to_string(),
            auto_diagnose: true,
            verdict: PlanVerdict::WarningsPresent,
            issues: vec![],
        }
    }

    #[test]
    fn test_print_plan_summary_normal() {
        let plan = make_plan();
        // Should not panic
        print_plan_summary(&plan, LogLevel::Normal);
    }

    #[test]
    fn test_print_plan_summary_verbose() {
        let plan = make_plan();
        print_plan_summary(&plan, LogLevel::Verbose);
    }

    #[test]
    fn test_print_plan_summary_quiet() {
        let plan = make_plan();
        print_plan_summary(&plan, LogLevel::Quiet);
    }

    #[test]
    fn test_print_plan_summary_no_imbalance() {
        let mut plan = make_plan();
        plan.data.imbalance_ratio = 1.0;
        // imbalance_ratio <= 2.0 should skip the imbalance line
        print_plan_summary(&plan, LogLevel::Normal);
    }

    #[test]
    fn test_print_plan_summary_ready() {
        let mut plan = make_plan();
        plan.verdict = PlanVerdict::Ready;
        print_plan_summary(&plan, LogLevel::Normal);
    }

    #[test]
    fn test_print_plan_summary_blocked() {
        let mut plan = make_plan();
        plan.verdict = PlanVerdict::Blocked;
        print_plan_summary(&plan, LogLevel::Normal);
    }

    #[test]
    fn test_check_counts_all_pass() {
        let mut plan = make_plan();
        plan.pre_flight = vec![
            PreFlightCheck { name: "a".into(), status: CheckStatus::Pass, detail: "ok".into() },
            PreFlightCheck { name: "b".into(), status: CheckStatus::Pass, detail: "ok".into() },
        ];
        let (p, w, f) = plan.check_counts();
        assert_eq!(p, 2);
        assert_eq!(w, 0);
        assert_eq!(f, 0);
    }

    #[test]
    fn test_check_counts_mixed() {
        let plan = make_plan();
        let (p, w, f) = plan.check_counts();
        assert_eq!(p, 1);
        assert_eq!(w, 1);
        assert_eq!(f, 0);
    }

    #[test]
    fn test_check_counts_with_fail() {
        let mut plan = make_plan();
        plan.pre_flight.push(PreFlightCheck {
            name: "c".into(),
            status: CheckStatus::Fail,
            detail: "bad".into(),
        });
        let (p, w, f) = plan.check_counts();
        assert_eq!(p, 1);
        assert_eq!(w, 1);
        assert_eq!(f, 1);
    }

    #[test]
    fn test_check_counts_empty() {
        let mut plan = make_plan();
        plan.pre_flight = vec![];
        let (p, w, f) = plan.check_counts();
        assert_eq!(p, 0);
        assert_eq!(w, 0);
        assert_eq!(f, 0);
    }

    #[test]
    fn test_plan_yaml_roundtrip() {
        let plan = make_plan();
        let yaml = plan.to_yaml();
        assert!(!yaml.is_empty());
        let parsed = crate::finetune::training_plan::TrainingPlan::from_str(&yaml).unwrap();
        assert_eq!(parsed.task, "classify");
        assert_eq!(parsed.version, "1.0");
        assert_eq!(parsed.data.train_samples, 1000);
    }

    #[test]
    fn test_plan_json_roundtrip() {
        let plan = make_plan();
        let json = plan.to_json();
        assert!(!json.is_empty());
        let parsed = crate::finetune::training_plan::TrainingPlan::from_str(&json).unwrap();
        assert_eq!(parsed.task, "classify");
    }

    #[test]
    fn test_run_finetune_plan_missing_data() {
        let args = FinetuneArgs {
            command: FinetuneCommand::Plan {
                data: std::path::PathBuf::from("/nonexistent/data.jsonl"),
                model_path: None,
                model_size: "0.5B".to_string(),
                num_classes: 2,
                output_dir: std::path::PathBuf::from("/tmp/ft_test_out"),
                strategy: "manual".to_string(),
                budget: 1,
                scout: false,
                max_epochs: 1,
                lr: Some(1e-4),
                lora_rank: Some(8),
                batch_size: Some(32),
                lora_alpha: None,
                warmup: None,
                gradient_clip: None,
                lr_min_ratio: None,
                class_weights: None,
                target_modules: None,
            },
        };
        // Should fail because data file doesn't exist
        let result = run_finetune(args, LogLevel::Quiet);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_finetune_apply_missing_plan() {
        let args = FinetuneArgs {
            command: FinetuneCommand::Apply {
                plan: std::path::PathBuf::from("/nonexistent/plan.yaml"),
                model_path: std::path::PathBuf::from("/nonexistent/model"),
                data: std::path::PathBuf::from("/nonexistent/data.jsonl"),
                output_dir: std::path::PathBuf::from("/tmp/ft_test_out"),
            },
        };
        let result = run_finetune(args, LogLevel::Quiet);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Failed to read plan"));
    }

    #[test]
    fn test_print_plan_summary_large_data() {
        let mut plan = make_plan();
        plan.data.train_samples = 1_000_000;
        plan.data.class_counts = vec![500_000, 250_000, 200_000, 50_000];
        plan.data.imbalance_ratio = 10.0;
        print_plan_summary(&plan, LogLevel::Normal);
    }

    #[test]
    fn test_print_plan_summary_many_checks() {
        let mut plan = make_plan();
        plan.pre_flight = vec![
            PreFlightCheck { name: "a".into(), status: CheckStatus::Pass, detail: "ok".into() },
            PreFlightCheck { name: "b".into(), status: CheckStatus::Pass, detail: "ok".into() },
            PreFlightCheck { name: "c".into(), status: CheckStatus::Warn, detail: "meh".into() },
            PreFlightCheck { name: "d".into(), status: CheckStatus::Fail, detail: "bad".into() },
            PreFlightCheck { name: "e".into(), status: CheckStatus::Fail, detail: "worse".into() },
        ];
        print_plan_summary(&plan, LogLevel::Normal);
    }

    // ── run_finetune dispatch tests ─────────────────────────────────────

    #[test]
    fn test_run_finetune_plan_with_all_manual_params() {
        let args = FinetuneArgs {
            command: FinetuneCommand::Plan {
                data: std::path::PathBuf::from("/nonexistent/data.jsonl"),
                model_path: Some(std::path::PathBuf::from("/nonexistent/model")),
                model_size: "9B".to_string(),
                num_classes: 5,
                output_dir: std::path::PathBuf::from("/tmp/ft_test_full"),
                strategy: "manual".to_string(),
                budget: 1,
                scout: false,
                max_epochs: 3,
                lr: Some(2e-5),
                lora_rank: Some(8),
                batch_size: Some(16),
                lora_alpha: Some(16.0),
                warmup: Some(0.05),
                gradient_clip: Some(0.5),
                lr_min_ratio: Some(0.001),
                class_weights: Some("sqrt_inverse".to_string()),
                target_modules: Some("qkv".to_string()),
            },
        };
        // Should fail because data file doesn't exist
        let result = run_finetune(args, LogLevel::Quiet);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_finetune_plan_tpe_strategy() {
        let args = FinetuneArgs {
            command: FinetuneCommand::Plan {
                data: std::path::PathBuf::from("/nonexistent/data.jsonl"),
                model_path: None,
                model_size: "0.5B".to_string(),
                num_classes: 2,
                output_dir: std::path::PathBuf::from("/tmp/ft_test_tpe"),
                strategy: "tpe".to_string(),
                budget: 20,
                scout: true,
                max_epochs: 5,
                lr: None,
                lora_rank: None,
                batch_size: None,
                lora_alpha: None,
                warmup: None,
                gradient_clip: None,
                lr_min_ratio: None,
                class_weights: None,
                target_modules: None,
            },
        };
        let result = run_finetune(args, LogLevel::Quiet);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_finetune_apply_invalid_plan_content() {
        // Create a file with invalid YAML/JSON content
        let path = std::env::temp_dir().join("ent_ft_bad_plan.yaml");
        std::fs::write(&path, "not valid plan content {{{{").unwrap();
        let args = FinetuneArgs {
            command: FinetuneCommand::Apply {
                plan: path.clone(),
                model_path: std::path::PathBuf::from("/nonexistent/model"),
                data: std::path::PathBuf::from("/nonexistent/data.jsonl"),
                output_dir: std::path::PathBuf::from("/tmp/ft_test_bad"),
            },
        };
        let result = run_finetune(args, LogLevel::Quiet);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Failed to parse plan"));
        let _ = std::fs::remove_file(&path);
    }

    // ── print_plan_summary edge cases ───────────────────────────────────

    #[test]
    fn test_print_plan_summary_zero_resources() {
        let mut plan = make_plan();
        plan.resources = ResourceEstimate {
            estimated_vram_gb: 0.0,
            estimated_minutes_per_epoch: 0.0,
            estimated_total_minutes: 0.0,
            estimated_checkpoint_mb: 0.0,
            steps_per_epoch: 0,
            gpu_device: None,
        };
        print_plan_summary(&plan, LogLevel::Normal);
    }

    #[test]
    fn test_print_plan_summary_single_class() {
        let mut plan = make_plan();
        plan.data.class_counts = vec![1000];
        plan.data.imbalance_ratio = 1.0;
        print_plan_summary(&plan, LogLevel::Verbose);
    }

    #[test]
    fn test_print_plan_summary_many_classes() {
        let mut plan = make_plan();
        plan.data.class_counts = vec![100, 200, 150, 300, 50, 80, 120, 90, 60, 70];
        plan.data.imbalance_ratio = 6.0;
        print_plan_summary(&plan, LogLevel::Normal);
    }

    #[test]
    fn test_print_plan_summary_high_imbalance() {
        let mut plan = make_plan();
        plan.data.imbalance_ratio = 100.0;
        plan.data.auto_class_weights = true;
        print_plan_summary(&plan, LogLevel::Normal);
    }

    #[test]
    fn test_print_plan_summary_boundary_imbalance() {
        let mut plan = make_plan();
        // Exactly 2.0 should NOT print the imbalance line
        plan.data.imbalance_ratio = 2.0;
        plan.data.auto_class_weights = false;
        print_plan_summary(&plan, LogLevel::Normal);
    }

    #[test]
    fn test_print_plan_summary_just_above_imbalance_threshold() {
        let mut plan = make_plan();
        plan.data.imbalance_ratio = 2.01;
        plan.data.auto_class_weights = true;
        print_plan_summary(&plan, LogLevel::Normal);
    }

    #[test]
    fn test_print_plan_summary_with_hpo_tpe() {
        let mut plan = make_plan();
        plan.hyperparameters.strategy = "tpe".to_string();
        plan.hyperparameters.budget = 20;
        plan.hyperparameters.scout = true;
        plan.hyperparameters.max_epochs = 1;
        plan.hyperparameters.search_space_params = 9;
        print_plan_summary(&plan, LogLevel::Normal);
    }

    // ── check_counts comprehensive ──────────────────────────────────────

    #[test]
    fn test_check_counts_all_fail() {
        let mut plan = make_plan();
        plan.pre_flight = vec![
            PreFlightCheck { name: "a".into(), status: CheckStatus::Fail, detail: "bad".into() },
            PreFlightCheck { name: "b".into(), status: CheckStatus::Fail, detail: "bad".into() },
        ];
        let (p, w, f) = plan.check_counts();
        assert_eq!(p, 0);
        assert_eq!(w, 0);
        assert_eq!(f, 2);
    }

    #[test]
    fn test_check_counts_all_warn() {
        let mut plan = make_plan();
        plan.pre_flight = vec![
            PreFlightCheck { name: "a".into(), status: CheckStatus::Warn, detail: "eh".into() },
            PreFlightCheck { name: "b".into(), status: CheckStatus::Warn, detail: "eh".into() },
            PreFlightCheck { name: "c".into(), status: CheckStatus::Warn, detail: "eh".into() },
        ];
        let (p, w, f) = plan.check_counts();
        assert_eq!(p, 0);
        assert_eq!(w, 3);
        assert_eq!(f, 0);
    }

    // ── make_plan helper verifications ───────────────────────────────────

    #[test]
    fn test_make_plan_defaults() {
        let plan = make_plan();
        assert_eq!(plan.version, "1.0");
        assert_eq!(plan.task, "classify");
        assert_eq!(plan.data.train_samples, 1000);
        assert_eq!(plan.model.hidden_size, 896);
        assert_eq!(plan.verdict, PlanVerdict::WarningsPresent);
        assert!(plan.auto_diagnose);
    }

    #[test]
    fn test_plan_to_yaml_and_back() {
        let plan = make_plan();
        let yaml = plan.to_yaml();
        assert!(yaml.contains("classify"));
        assert!(yaml.contains("1.0"));
        let parsed = crate::finetune::training_plan::TrainingPlan::from_str(&yaml).unwrap();
        assert_eq!(parsed.data.train_samples, 1000);
    }

    #[test]
    fn test_plan_to_json_and_back() {
        let plan = make_plan();
        let json = plan.to_json();
        assert!(json.contains("classify"));
        let parsed = crate::finetune::training_plan::TrainingPlan::from_str(&json).unwrap();
        assert_eq!(parsed.model.architecture, "Qwen2");
    }

    // ── run_finetune with valid plan file ───────────────────────────────

    #[test]
    fn test_run_finetune_apply_valid_plan_file_missing_model() {
        let plan = make_plan();
        let plan_path = std::env::temp_dir().join("ent_ft_valid_plan.yaml");
        std::fs::write(&plan_path, plan.to_yaml()).unwrap();
        let args = FinetuneArgs {
            command: FinetuneCommand::Apply {
                plan: plan_path.clone(),
                model_path: std::path::PathBuf::from("/nonexistent/model"),
                data: std::path::PathBuf::from("/nonexistent/data.jsonl"),
                output_dir: std::path::PathBuf::from("/tmp/ft_test_valid"),
            },
        };
        // Should parse plan but fail on model/data path
        let result = run_finetune(args, LogLevel::Quiet);
        assert!(result.is_err());
        let _ = std::fs::remove_file(&plan_path);
    }
}
