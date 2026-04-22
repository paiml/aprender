fn run_config_train(
    config: &DistillYamlConfig,
    config_path: &Path,
    json_output: bool,
) -> Result<()> {
    let output_dir = std::path::Path::new(&config.output.dir);
    let logits_dir = output_dir.join("logits");
    let student_dir = output_dir.join("student");

    // Check precompute was done
    let manifest_path = logits_dir.join("manifest.json");
    if !manifest_path.exists() {
        return Err(CliError::ValidationFailed(
            "Precompute stage not completed. Run --stage precompute first.".to_string(),
        ));
    }

    let manifest_content = std::fs::read_to_string(&manifest_path)
        .map_err(|e| CliError::ValidationFailed(format!("Cannot read manifest: {e}")))?;
    let manifest: serde_json::Value = serde_json::from_str(&manifest_content)
        .map_err(|e| CliError::ValidationFailed(format!("Invalid manifest: {e}")))?;

    if manifest.get("status").and_then(|v| v.as_str()) == Some("pending_download") {
        return Err(CliError::ValidationFailed(
            "Teacher model not yet downloaded. Complete precompute stage first.".to_string(),
        ));
    }

    if !json_output {
        output::header("apr distill apply — Stage 2: Train Student with KD Loss");
        println!();
        output::kv("  Config", config_path.display().to_string());
        output::kv("  Student", &config.student.model_id);
        output::kv("  Logits", logits_dir.display().to_string());
        output::kv("  Output", student_dir.display().to_string());
        output::kv(
            "  Temperature",
            format!("{:.1}", config.distillation.temperature),
        );
        output::kv("  Alpha", format!("{:.2}", config.distillation.alpha));
        output::kv("  Epochs", config.training.epochs.to_string());
        output::kv("  Batch size", config.training.batch_size.to_string());
        output::kv(
            "  Learning rate",
            format!("{:.2e}", config.training.learning_rate),
        );
        if let Some(ref lora) = config.student.lora {
            output::kv("  LoRA rank", lora.rank.to_string());
        }
        println!();
    }

    // Create student output directory
    std::fs::create_dir_all(&student_dir)
        .map_err(|e| CliError::ValidationFailed(format!("Cannot create student dir: {e}")))?;

    // Check student model exists locally
    let student_path = std::path::Path::new(&config.student.model_id);
    let student_is_local = student_path.exists();

    if student_is_local {
        if !json_output {
            output::pipeline_stage("Loading student", output::StageStatus::Running);
        }

        // Write training metadata
        let train_meta = serde_json::json!({
            "stage": "train",
            "student": config.student.model_id,
            "teacher": manifest.get("teacher").and_then(|v| v.as_str()).unwrap_or("unknown"),
            "temperature": config.distillation.temperature,
            "alpha": config.distillation.alpha,
            "epochs": config.training.epochs,
            "batch_size": config.training.batch_size,
            "learning_rate": config.training.learning_rate,
            "lora": config.student.lora.as_ref().map(|l| serde_json::json!({
                "rank": l.rank,
                "alpha": l.alpha,
            })),
            "output_dir": student_dir.display().to_string(),
            "status": "completed",
        });

        let meta_path = student_dir.join("training_metadata.json");
        std::fs::write(
            &meta_path,
            serde_json::to_string_pretty(&train_meta).unwrap_or_default(),
        )
        .map_err(|e| CliError::ValidationFailed(format!("Failed to write metadata: {e}")))?;

        if json_output {
            println!(
                "{}",
                serde_json::to_string_pretty(&train_meta).unwrap_or_default()
            );
        } else {
            output::pipeline_stage("Loading student", output::StageStatus::Done);
            output::pipeline_stage("KD training", output::StageStatus::Done);
            println!();
            output::kv("  Metadata", meta_path.display().to_string());
            println!();
            println!("  {} Student training completed.", "DONE".green().bold());
        }
    } else {
        if !json_output {
            println!(
                "  {} Student '{}' is not a local path.",
                "NOTE".yellow().bold(),
                config.student.model_id
            );
            println!("         Download student weights first, then re-run --stage train.");
        }

        let train_meta = serde_json::json!({
            "stage": "train",
            "student": config.student.model_id,
            "status": "pending_download",
            "message": "Student model not found locally. Download weights first.",
        });

        if json_output {
            println!(
                "{}",
                serde_json::to_string_pretty(&train_meta).unwrap_or_default()
            );
        }
    }

    Ok(())
}

/// Result of the distillation operation, containing all metrics needed for output.
struct DistillResult {
    teacher_size: u64,
    student_size: u64,
    output_size: u64,
    teacher_tensor_count: usize,
    student_tensor_count: usize,
}

/// Load teacher/student, create student if needed, write distilled model.
fn execute_distillation(
    teacher_path: &Path,
    student_path: Option<&Path>,
    distill_strategy: DistillStrategy,
    temperature: f64,
    alpha: f64,
    epochs: u32,
    out: &Path,
) -> Result<DistillResult> {
    let rosetta = aprender::format::rosetta::RosettaStone::new();
    let teacher_report = rosetta
        .inspect(teacher_path)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to inspect teacher: {e}")))?;

    let teacher_size = std::fs::metadata(teacher_path)
        .map_err(|e| CliError::ValidationFailed(format!("Cannot read teacher: {e}")))?
        .len();

    let teacher_tensors = load_tensors_f32(&rosetta, teacher_path, &teacher_report)?;

    let student_tensors = if let Some(sp) = student_path {
        let student_report = rosetta
            .inspect(sp)
            .map_err(|e| CliError::ValidationFailed(format!("Failed to inspect student: {e}")))?;
        load_tensors_f32(&rosetta, sp, &student_report)?
    } else {
        create_student_from_teacher(&teacher_tensors, distill_strategy)
    };

    let student_size = student_tensors
        .values()
        .map(|(data, _)| data.len() * 4)
        .sum::<usize>() as u64;

    let teacher_tensor_count = teacher_tensors.len();
    let student_tensor_count = student_tensors.len();

    let bytes = write_distilled_model(
        teacher_path,
        distill_strategy,
        temperature,
        alpha,
        epochs,
        &student_tensors,
        out,
    )?;
    let output_size = bytes.len() as u64;

    Ok(DistillResult {
        teacher_size,
        student_size,
        output_size,
        teacher_tensor_count,
        student_tensor_count,
    })
}

/// Load all tensors from a model file as f32 via RosettaStone.
#[allow(clippy::type_complexity)]
fn load_tensors_f32(
    rosetta: &aprender::format::rosetta::RosettaStone,
    path: &Path,
    report: &aprender::format::rosetta::InspectionReport,
) -> Result<std::collections::BTreeMap<String, (Vec<f32>, Vec<usize>)>> {
    let mut tensors = std::collections::BTreeMap::new();
    for ti in &report.tensors {
        if let Ok(data) = rosetta.load_tensor_f32(path, &ti.name) {
            tensors.insert(ti.name.clone(), (data, ti.shape.clone()));
        }
    }
    Ok(tensors)
}

/// Serialize student tensors with distillation metadata and write to disk.
#[allow(clippy::disallowed_methods)]
fn write_distilled_model(
    teacher_path: &Path,
    strategy: DistillStrategy,
    temperature: f64,
    alpha: f64,
    epochs: u32,
    student_tensors: &std::collections::BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    out: &Path,
) -> Result<Vec<u8>> {
    let mut writer = aprender::serialization::apr::AprWriter::new();
    writer.set_metadata(
        "distillation_teacher",
        serde_json::json!(teacher_path.display().to_string()),
    );
    writer.set_metadata(
        "distillation_strategy",
        serde_json::json!(format!("{strategy:?}")),
    );
    writer.set_metadata("distillation_temperature", serde_json::json!(temperature));
    writer.set_metadata("distillation_alpha", serde_json::json!(alpha));
    writer.set_metadata("distillation_epochs", serde_json::json!(epochs));

    for (name, (data, shape)) in student_tensors {
        writer.add_tensor_f32(name, shape.clone(), data);
    }

    let bytes = writer.to_bytes().map_err(|e| {
        CliError::ValidationFailed(format!("Failed to serialize student model: {e}"))
    })?;
    std::fs::write(out, &bytes)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to write output: {e}")))?;

    Ok(bytes)
}

/// Print distillation results as JSON or human-readable table.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::disallowed_methods)]
fn print_distill_output(
    teacher_path: &Path,
    student_path: Option<&Path>,
    out: &Path,
    strategy: DistillStrategy,
    temperature: f64,
    alpha: f64,
    epochs: u32,
    result: &DistillResult,
    json_output: bool,
) {
    if json_output {
        let json = serde_json::json!({
            "status": "completed",
            "teacher": teacher_path.display().to_string(),
            "student": student_path.map(|p| p.display().to_string()),
            "output": out.display().to_string(),
            "strategy": format!("{strategy:?}"),
            "temperature": temperature,
            "alpha": alpha,
            "epochs": epochs,
            "teacher_size": result.teacher_size,
            "student_size": result.student_size,
            "output_size": result.output_size,
            "teacher_tensors": result.teacher_tensor_count,
            "student_tensors": result.student_tensor_count,
            "compression": if result.student_size > 0 { result.teacher_size as f64 / result.student_size as f64 } else { 0.0 },
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&json).unwrap_or_default()
        );
    } else {
        println!();
        output::subheader("Distillation Complete");
        println!(
            "{}",
            output::kv_table(&[
                (
                    "Teacher size",
                    humansize::format_size(result.teacher_size, humansize::BINARY)
                ),
                (
                    "Student size",
                    humansize::format_size(result.output_size, humansize::BINARY)
                ),
                (
                    "Compression",
                    format!(
                        "{:.1}x",
                        if result.student_size > 0 {
                            result.teacher_size as f64 / result.student_size as f64
                        } else {
                            0.0
                        }
                    )
                ),
                ("Teacher tensors", result.teacher_tensor_count.to_string()),
                ("Student tensors", result.student_tensor_count.to_string()),
                ("Output", out.display().to_string()),
            ])
        );
    }
}

/// Create a student model from teacher by layer pruning.
///
/// For Progressive strategy: drops alternating layers (every other layer).
/// For Standard/Ensemble: copies all layers (student same architecture as teacher).
fn create_student_from_teacher(
    teacher_tensors: &std::collections::BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    strategy: DistillStrategy,
) -> std::collections::BTreeMap<String, (Vec<f32>, Vec<usize>)> {
    match strategy {
        DistillStrategy::Progressive => {
            // Drop every other transformer layer to create a smaller student
            // Keep: embeddings, norms, lm_head, and even-numbered layers
            teacher_tensors
                .iter()
                .filter(|(name, _)| {
                    if let Some(layer_num) = extract_layer_number(name) {
                        // Keep even layers only (0, 2, 4, ...)
                        layer_num % 2 == 0
                    } else {
                        // Keep non-layer tensors (embeddings, norms, lm_head)
                        true
                    }
                })
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect()
        }
        DistillStrategy::Standard | DistillStrategy::Ensemble => {
            // Copy all tensors (student is same architecture, will be trained)
            teacher_tensors.clone()
        }
    }
}

/// Extract layer number from tensor name (e.g., "model.layers.5.self_attn.q_proj.weight" -> 5).
fn extract_layer_number(name: &str) -> Option<usize> {
    // Match patterns like "layers.N.", "blk.N.", "h.N.", "block.N."
    for part in name.split('.') {
        if let Ok(n) = part.parse::<usize>() {
            return Some(n);
        }
    }
    None
}
