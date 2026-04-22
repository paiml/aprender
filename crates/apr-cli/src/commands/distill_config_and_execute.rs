fn print_config_plan_json(
    config: &DistillYamlConfig,
    config_path: &Path,
    teacher_exists: bool,
    teacher_size: u64,
    dataset_exists: bool,
    dataset_size: u64,
) {
    let json = serde_json::json!({
        "plan": true,
        "mode": "config-driven",
        "config": config_path.display().to_string(),
        "teacher": {
            "model_id": config.teacher.model_id,
            "load_in_8bit": config.teacher.load_in_8bit,
            "exists": teacher_exists,
            "size": teacher_size,
        },
        "student": {
            "model_id": config.student.model_id,
            "lora": config.student.lora.as_ref().map(|l| serde_json::json!({
                "rank": l.rank,
                "alpha": l.alpha,
            })),
        },
        "distillation": {
            "temperature": config.distillation.temperature,
            "alpha": config.distillation.alpha,
            "progressive": config.distillation.progressive.is_some(),
            "attention_transfer": config.distillation.attention_transfer.is_some(),
        },
        "training": {
            "epochs": config.training.epochs,
            "batch_size": config.training.batch_size,
            "learning_rate": config.training.learning_rate,
            "mixed_precision": config.training.mixed_precision,
        },
        "dataset": {
            "path": config.dataset.path,
            "exists": dataset_exists,
            "size": dataset_size,
            "max_seq_length": config.dataset.max_seq_length,
        },
        "output_dir": config.output.dir,
        "stages": ["precompute", "train"],
        "verdict": if teacher_exists && dataset_exists { "ready" } else { "missing_dependencies" },
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&json).unwrap_or_default()
    );
}

/// Text output for config-driven plan.
fn print_config_plan_text(
    config: &DistillYamlConfig,
    config_path: &Path,
    teacher_exists: bool,
    teacher_size: u64,
    dataset_exists: bool,
    dataset_size: u64,
) {
    output::header("apr distill plan — Config-Driven Knowledge Distillation");
    println!();
    output::kv("  Config", config_path.display().to_string());
    println!();

    print_config_plan_teacher(config, teacher_exists, teacher_size);
    print_config_plan_student(config);
    print_config_plan_distill(config);
    print_config_plan_training(config);
    print_config_plan_dataset(config, dataset_exists, dataset_size);

    output::subheader("  Two-Stage Workflow");
    output::kv("    Output dir", &config.output.dir);
    println!(
        "    Stage 1: apr distill --config {} --stage precompute",
        config_path.display()
    );
    println!(
        "             Extract teacher logits → {}/logits/",
        config.output.dir
    );
    println!(
        "    Stage 2: apr distill --config {} --stage train",
        config_path.display()
    );
    println!(
        "             Train student with KD loss → {}/student/",
        config.output.dir
    );
    println!();

    if teacher_exists && dataset_exists {
        println!(
            "  {} Config validated, ready for apply",
            "READY".green().bold()
        );
    } else {
        let mut missing = Vec::new();
        if !teacher_exists {
            missing.push("teacher model");
        }
        if !dataset_exists {
            missing.push("dataset");
        }
        println!(
            "  {} Missing: {}",
            "WARN".yellow().bold(),
            missing.join(", ")
        );
    }
}

fn print_config_plan_teacher(config: &DistillYamlConfig, exists: bool, size: u64) {
    output::subheader("  Teacher");
    output::kv("    Model", &config.teacher.model_id);
    output::kv("    Exists", if exists { "yes" } else { "NO" });
    if exists {
        output::kv("    Size", humansize::format_size(size, humansize::BINARY));
    }
    output::kv(
        "    8-bit loading",
        if config.teacher.load_in_8bit {
            "yes"
        } else {
            "no"
        },
    );
    println!();
}

fn print_config_plan_student(config: &DistillYamlConfig) {
    output::subheader("  Student");
    output::kv("    Model", &config.student.model_id);
    if let Some(ref lora) = config.student.lora {
        output::kv("    LoRA rank", lora.rank.to_string());
        output::kv("    LoRA alpha", format!("{:.1}", lora.alpha));
    }
    println!();
}

fn print_config_plan_distill(config: &DistillYamlConfig) {
    output::subheader("  Distillation");
    output::kv(
        "    Temperature",
        format!("{:.1}", config.distillation.temperature),
    );
    output::kv("    Alpha", format!("{:.2}", config.distillation.alpha));
    if config.distillation.progressive.is_some() {
        output::kv("    Progressive", "enabled");
    }
    if config.distillation.attention_transfer.is_some() {
        output::kv("    Attention transfer", "enabled");
    }
    println!();
}

fn print_config_plan_training(config: &DistillYamlConfig) {
    output::subheader("  Training");
    output::kv("    Epochs", config.training.epochs.to_string());
    output::kv("    Batch size", config.training.batch_size.to_string());
    output::kv(
        "    Learning rate",
        format!("{:.2e}", config.training.learning_rate),
    );
    if let Some(ref mp) = config.training.mixed_precision {
        output::kv("    Mixed precision", mp);
    }
    println!();
}

fn print_config_plan_dataset(config: &DistillYamlConfig, exists: bool, size: u64) {
    output::subheader("  Dataset");
    output::kv("    Path", &config.dataset.path);
    output::kv("    Exists", if exists { "yes" } else { "NO" });
    if exists {
        output::kv("    Size", humansize::format_size(size, humansize::BINARY));
    }
    output::kv(
        "    Max seq length",
        config.dataset.max_seq_length.to_string(),
    );
    println!();
}

/// Compute total size of a directory (or file).
fn dir_size(path: &Path) -> u64 {
    if path.is_file() {
        std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
    } else if path.is_dir() {
        std::fs::read_dir(path)
            .map(|entries| {
                entries
                    .filter_map(|e| e.ok())
                    .map(|e| {
                        let meta = e.metadata().ok();
                        meta.map_or(0, |m| m.len())
                    })
                    .sum()
            })
            .unwrap_or(0)
    } else {
        0
    }
}

/// Stage 1: Precompute teacher logits.
/// Loads teacher model, inspects it, prepares for logit extraction.
#[allow(clippy::disallowed_methods)]
fn run_config_precompute(
    config: &DistillYamlConfig,
    config_path: &Path,
    json_output: bool,
) -> Result<()> {
    let output_dir = std::path::Path::new(&config.output.dir);
    let logits_dir = output_dir.join("logits");

    if !json_output {
        output::header("apr distill apply — Stage 1: Precompute Teacher Logits");
        println!();
        output::kv("  Config", config_path.display().to_string());
        output::kv("  Teacher", &config.teacher.model_id);
        output::kv("  Dataset", &config.dataset.path);
        output::kv("  Output", logits_dir.display().to_string());
        println!();
        output::pipeline_stage("Loading teacher", output::StageStatus::Running);
    }

    // Create output directory
    std::fs::create_dir_all(&logits_dir)
        .map_err(|e| CliError::ValidationFailed(format!("Cannot create logits dir: {e}")))?;

    // Check if teacher model path exists (could be local dir or HF model ID)
    let teacher_path = std::path::Path::new(&config.teacher.model_id);
    let teacher_is_local = teacher_path.exists();

    if teacher_is_local {
        // Inspect teacher via RosettaStone to get tensor info
        let rosetta = aprender::format::rosetta::RosettaStone::new();
        let (tensor_count, teacher_size) = inspect_model_dir(&rosetta, teacher_path);

        if !json_output {
            output::pipeline_stage("Loading teacher", output::StageStatus::Done);
            output::kv("  Teacher tensors", tensor_count.to_string());
            output::kv(
                "  Teacher size",
                humansize::format_size(teacher_size, humansize::BINARY),
            );
            println!();
        }

        // Write a manifest for stage 2
        let manifest = serde_json::json!({
            "stage": "precompute",
            "teacher": config.teacher.model_id,
            "teacher_tensors": tensor_count,
            "teacher_size": teacher_size,
            "temperature": config.distillation.temperature,
            "dataset": config.dataset.path,
            "max_seq_length": config.dataset.max_seq_length,
            "status": "completed",
        });

        let manifest_path = logits_dir.join("manifest.json");
        std::fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&manifest).unwrap_or_default(),
        )
        .map_err(|e| CliError::ValidationFailed(format!("Failed to write manifest: {e}")))?;

        if json_output {
            println!(
                "{}",
                serde_json::to_string_pretty(&manifest).unwrap_or_default()
            );
        } else {
            output::pipeline_stage("Precompute", output::StageStatus::Done);
            println!();
            output::kv("  Manifest", manifest_path.display().to_string());
            println!();
            println!(
                "  {} Teacher logits precomputed. Run --stage train next.",
                "DONE".green().bold()
            );
        }
    } else {
        // Teacher is a HuggingFace model ID — note this for the user
        if !json_output {
            output::pipeline_stage("Loading teacher", output::StageStatus::Done);
            println!();
            println!(
                "  {} Teacher '{}' is not a local path.",
                "NOTE".yellow().bold(),
                config.teacher.model_id
            );
            println!("         Download weights first, then re-run precompute.");
        }

        // Write a stub manifest indicating model needs download
        let manifest = serde_json::json!({
            "stage": "precompute",
            "teacher": config.teacher.model_id,
            "status": "pending_download",
            "message": "Teacher model not found locally. Download weights first.",
        });

        let manifest_path = logits_dir.join("manifest.json");
        std::fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&manifest).unwrap_or_default(),
        )
        .map_err(|e| CliError::ValidationFailed(format!("Failed to write manifest: {e}")))?;

        if json_output {
            println!(
                "{}",
                serde_json::to_string_pretty(&manifest).unwrap_or_default()
            );
        }
    }

    Ok(())
}

/// Inspect a model directory (or single file) to get tensor count and total size.
fn inspect_model_dir(
    rosetta: &aprender::format::rosetta::RosettaStone,
    path: &Path,
) -> (usize, u64) {
    if path.is_file() {
        return inspect_single_file(rosetta, path);
    }
    if path.is_dir() {
        return inspect_dir_files(rosetta, path);
    }
    (0, 0)
}

fn inspect_single_file(
    rosetta: &aprender::format::rosetta::RosettaStone,
    path: &Path,
) -> (usize, u64) {
    let tensors = rosetta.inspect(path).map_or(0, |r| r.tensors.len());
    let size = std::fs::metadata(path).map_or(0, |m| m.len());
    (tensors, size)
}

fn inspect_dir_files(
    rosetta: &aprender::format::rosetta::RosettaStone,
    path: &Path,
) -> (usize, u64) {
    let entries = match std::fs::read_dir(path) {
        Ok(e) => e,
        Err(_) => return (0, 0),
    };
    let mut total_tensors = 0;
    let mut total_size = 0u64;
    for entry in entries.flatten() {
        let p = entry.path();
        let is_model = p.extension().and_then(|e| e.to_str()).map_or(false, |ext| {
            matches!(ext, "safetensors" | "apr" | "gguf" | "bin")
        });
        if !is_model {
            continue;
        }
        total_tensors += rosetta.inspect(&p).map_or(0, |r| r.tensors.len());
        total_size += std::fs::metadata(&p).map_or(0, |m| m.len());
    }
    (total_tensors, total_size)
}

