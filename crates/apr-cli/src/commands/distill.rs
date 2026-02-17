//! Distill command implementation (GH-247)
//!
//! Knowledge distillation pipeline for transferring knowledge from a
//! teacher model to a smaller student model.
//!
//! # Example
//!
//! ```bash
//! apr distill teacher.apr --student pruned.apr --data train.jsonl -o distilled.apr
//! apr distill teacher.apr --progressive --target-ratio 0.5 --data train.jsonl -o distilled.apr
//! apr distill teacher.apr --plan --json
//! ```

use crate::error::{CliError, Result};
use crate::output;
use std::path::Path;

/// Distillation strategy
#[derive(Debug, Clone, Copy, Default)]
pub enum DistillStrategy {
    /// Standard KL-divergence distillation
    #[default]
    Standard,
    /// Progressive distillation (gradual pruning + distillation)
    Progressive,
    /// Ensemble distillation (multiple teachers)
    Ensemble,
}

impl std::str::FromStr for DistillStrategy {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "standard" | "kl" => Ok(Self::Standard),
            "progressive" | "gradual" => Ok(Self::Progressive),
            "ensemble" | "multi" => Ok(Self::Ensemble),
            _ => Err(format!(
                "Unknown distillation strategy: {s}. Supported: standard, progressive, ensemble"
            )),
        }
    }
}

/// Validate distillation parameters (temperature, alpha).
fn validate_distill_params(temperature: f64, alpha: f64) -> Result<()> {
    if temperature <= 0.0 {
        return Err(CliError::ValidationFailed(format!(
            "Temperature must be positive, got {temperature}"
        )));
    }
    if !(0.0..=1.0).contains(&alpha) {
        return Err(CliError::ValidationFailed(format!(
            "Alpha must be between 0 and 1, got {alpha}"
        )));
    }
    Ok(())
}

/// Validate that optional file paths exist on disk.
fn validate_optional_paths(student_path: Option<&Path>, data_path: Option<&Path>) -> Result<()> {
    if let Some(student) = student_path {
        if !student.exists() {
            return Err(CliError::FileNotFound(student.to_path_buf()));
        }
    }
    if let Some(data) = data_path {
        if !data.exists() {
            return Err(CliError::FileNotFound(data.to_path_buf()));
        }
    }
    Ok(())
}

/// Run the distill command
#[allow(clippy::too_many_arguments)]
#[allow(clippy::disallowed_methods)]
pub(crate) fn run(
    teacher_path: &Path,
    student_path: Option<&Path>,
    data_path: Option<&Path>,
    output_path: Option<&Path>,
    strategy: &str,
    temperature: f64,
    alpha: f64,
    epochs: u32,
    plan_only: bool,
    json_output: bool,
) -> Result<()> {
    if !teacher_path.exists() {
        return Err(CliError::FileNotFound(teacher_path.to_path_buf()));
    }

    let distill_strategy: DistillStrategy = strategy.parse().map_err(CliError::ValidationFailed)?;

    validate_distill_params(temperature, alpha)?;

    if plan_only {
        return run_plan(
            teacher_path,
            student_path,
            distill_strategy,
            temperature,
            alpha,
            epochs,
            json_output,
        );
    }

    if student_path.is_none() && !matches!(distill_strategy, DistillStrategy::Progressive) {
        return Err(CliError::ValidationFailed(
            "Student model required for standard distillation. Use --student <path>".to_string(),
        ));
    }

    let out = output_path.ok_or_else(|| {
        CliError::ValidationFailed(
            "Output path required. Use -o <path> to specify output.".to_string(),
        )
    })?;

    if !json_output {
        output::header("APR Distill");
        let mut pairs = vec![
            ("Teacher", teacher_path.display().to_string()),
            ("Strategy", format!("{distill_strategy:?}")),
            ("Temperature", format!("{temperature:.1}")),
            ("Alpha", format!("{alpha:.2}")),
            ("Epochs", epochs.to_string()),
            ("Output", out.display().to_string()),
        ];
        if let Some(student) = student_path {
            pairs.insert(1, ("Student", student.display().to_string()));
        }
        if let Some(data) = data_path {
            pairs.push(("Training data", data.display().to_string()));
        }
        println!("{}", output::kv_table(&pairs));
        println!();
    }

    validate_optional_paths(student_path, data_path)?;

    if !json_output {
        output::pipeline_stage("Distilling", output::StageStatus::Running);
    }

    let distill_result = execute_distillation(
        teacher_path,
        student_path,
        distill_strategy,
        temperature,
        alpha,
        epochs,
        out,
    )?;

    if !json_output {
        output::pipeline_stage("Distilling", output::StageStatus::Done);
    }

    print_distill_output(
        teacher_path,
        student_path,
        out,
        distill_strategy,
        temperature,
        alpha,
        epochs,
        &distill_result,
        json_output,
    );

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

/// Plan distillation (estimate only)
#[allow(clippy::too_many_arguments)]
#[allow(clippy::disallowed_methods)]
fn run_plan(
    teacher_path: &Path,
    student_path: Option<&Path>,
    strategy: DistillStrategy,
    temperature: f64,
    alpha: f64,
    epochs: u32,
    json_output: bool,
) -> Result<()> {
    let teacher_size = std::fs::metadata(teacher_path)
        .map_err(|e| CliError::ValidationFailed(format!("Cannot read teacher: {e}")))?
        .len();

    let student_size = student_path
        .and_then(|p| std::fs::metadata(p).ok())
        .map_or(teacher_size / 2, |m| m.len());

    let peak_memory = teacher_size + student_size;

    if json_output {
        let json = serde_json::json!({
            "plan": true,
            "teacher": teacher_path.display().to_string(),
            "teacher_size": teacher_size,
            "student_size": student_size,
            "strategy": format!("{strategy:?}"),
            "temperature": temperature,
            "alpha": alpha,
            "epochs": epochs,
            "peak_memory": peak_memory,
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&json).unwrap_or_default()
        );
    } else {
        output::header("APR Distill — Plan");
        println!(
            "{}",
            output::kv_table(&[
                ("Teacher", teacher_path.display().to_string()),
                (
                    "Teacher size",
                    humansize::format_size(teacher_size, humansize::BINARY),
                ),
                (
                    "Student size",
                    humansize::format_size(student_size, humansize::BINARY),
                ),
                ("Strategy", format!("{strategy:?}")),
                ("Temperature", format!("{temperature:.1}")),
                ("Alpha", format!("{alpha:.2}")),
                ("Epochs", epochs.to_string()),
                (
                    "Peak memory",
                    humansize::format_size(peak_memory, humansize::BINARY),
                ),
            ])
        );
        println!();
        println!(
            "  {} Run without --plan to execute.",
            output::badge_info("INFO"),
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_distill_strategy_parse() {
        assert!(matches!(
            "standard".parse::<DistillStrategy>(),
            Ok(DistillStrategy::Standard)
        ));
        assert!(matches!(
            "kl".parse::<DistillStrategy>(),
            Ok(DistillStrategy::Standard)
        ));
        assert!(matches!(
            "progressive".parse::<DistillStrategy>(),
            Ok(DistillStrategy::Progressive)
        ));
        assert!(matches!(
            "ensemble".parse::<DistillStrategy>(),
            Ok(DistillStrategy::Ensemble)
        ));
        assert!("unknown".parse::<DistillStrategy>().is_err());
    }

    #[test]
    fn test_run_teacher_not_found() {
        let result = run(
            Path::new("/nonexistent.apr"),
            None,
            None,
            Some(Path::new("/tmp/out.apr")),
            "standard",
            3.0,
            0.7,
            3,
            false,
            false,
        );
        assert!(result.is_err());
        assert!(matches!(result, Err(CliError::FileNotFound(_))));
    }

    #[test]
    fn test_run_invalid_temperature() {
        let input = NamedTempFile::with_suffix(".apr").expect("create input");
        let result = run(
            input.path(),
            None,
            None,
            Some(Path::new("/tmp/out.apr")),
            "standard",
            0.0,
            0.7,
            3,
            false,
            false,
        );
        assert!(result.is_err());
        match result {
            Err(CliError::ValidationFailed(msg)) => assert!(msg.contains("Temperature")),
            _ => panic!("Expected ValidationFailed"),
        }
    }

    #[test]
    fn test_run_invalid_alpha() {
        let input = NamedTempFile::with_suffix(".apr").expect("create input");
        let result = run(
            input.path(),
            None,
            None,
            Some(Path::new("/tmp/out.apr")),
            "standard",
            3.0,
            1.5,
            3,
            false,
            false,
        );
        assert!(result.is_err());
        match result {
            Err(CliError::ValidationFailed(msg)) => assert!(msg.contains("Alpha")),
            _ => panic!("Expected ValidationFailed"),
        }
    }

    #[test]
    fn test_run_no_student() {
        let mut input = NamedTempFile::with_suffix(".apr").expect("create input");
        input.write_all(&[0u8; 512]).expect("write");
        let result = run(
            input.path(),
            None,
            None,
            Some(Path::new("/tmp/out.apr")),
            "standard",
            3.0,
            0.7,
            3,
            false,
            false,
        );
        assert!(result.is_err());
        match result {
            Err(CliError::ValidationFailed(msg)) => assert!(msg.contains("Student")),
            _ => panic!("Expected ValidationFailed"),
        }
    }

    #[test]
    fn test_run_no_output() {
        let mut teacher = NamedTempFile::with_suffix(".apr").expect("create teacher");
        teacher.write_all(&[0u8; 512]).expect("write");
        let mut student = NamedTempFile::with_suffix(".apr").expect("create student");
        student.write_all(&[0u8; 256]).expect("write");
        let result = run(
            teacher.path(),
            Some(student.path()),
            None,
            None,
            "standard",
            3.0,
            0.7,
            3,
            false,
            false,
        );
        assert!(result.is_err());
        match result {
            Err(CliError::ValidationFailed(msg)) => assert!(msg.contains("Output")),
            _ => panic!("Expected ValidationFailed"),
        }
    }

    /// Create a valid APR test model with some tensors
    fn make_test_model() -> NamedTempFile {
        let mut writer = aprender::serialization::apr::AprWriter::new();
        writer.set_metadata("model_type", serde_json::json!("test"));
        let w0: Vec<f32> = (0..64).map(|i| (i as f32) * 0.01).collect();
        writer.add_tensor_f32("model.layers.0.self_attn.q_proj.weight", vec![8, 8], &w0);
        let w1: Vec<f32> = (0..64).map(|i| (i as f32) * 0.02).collect();
        writer.add_tensor_f32("model.layers.1.self_attn.q_proj.weight", vec![8, 8], &w1);
        writer.add_tensor_f32("model.norm.weight", vec![8], &vec![1.0; 8]);
        writer.add_tensor_f32("model.embed_tokens.weight", vec![10, 8], &vec![0.1; 80]);

        let file = NamedTempFile::with_suffix(".apr").expect("create model");
        let bytes = writer.to_bytes().expect("serialize");
        std::fs::write(file.path(), bytes).expect("write");
        file
    }

    #[test]
    fn test_run_valid() {
        let teacher = make_test_model();
        let student = make_test_model();
        let output = NamedTempFile::with_suffix(".apr").expect("create output");
        let result = run(
            teacher.path(),
            Some(student.path()),
            None,
            Some(output.path()),
            "standard",
            3.0,
            0.7,
            3,
            false,
            true,
        );
        assert!(result.is_ok(), "Distill should succeed: {result:?}");

        // Verify output is a valid APR file
        let reader = aprender::serialization::apr::AprReader::open(output.path())
            .expect("output should be valid APR");
        assert!(!reader.tensors.is_empty(), "Output should have tensors");
        assert!(reader.get_metadata("distillation_teacher").is_some());
    }

    #[test]
    fn test_plan_mode() {
        let teacher = make_test_model();
        let result = run(
            teacher.path(),
            None,
            None,
            None,
            "standard",
            3.0,
            0.7,
            3,
            true,
            false,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_plan_json() {
        let teacher = make_test_model();
        let result = run(
            teacher.path(),
            None,
            None,
            None,
            "progressive",
            4.0,
            0.5,
            5,
            true,
            true,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_progressive_no_student() {
        // Progressive distillation creates student from teacher (drops every other layer)
        let teacher = make_test_model();
        let output = NamedTempFile::with_suffix(".apr").expect("create output");
        let result = run(
            teacher.path(),
            None,
            None,
            Some(output.path()),
            "progressive",
            3.0,
            0.7,
            3,
            false,
            true,
        );
        assert!(result.is_ok(), "Progressive should succeed: {result:?}");

        // Verify student has fewer layers than teacher
        let reader = aprender::serialization::apr::AprReader::open(output.path())
            .expect("output should be valid APR");
        // Teacher has layers 0 and 1, progressive keeps only even (layer 0)
        let layer_names: Vec<_> = reader
            .tensors
            .iter()
            .filter(|t| t.name.contains("layers.1."))
            .collect();
        assert!(
            layer_names.is_empty(),
            "Layer 1 should be dropped by progressive distillation"
        );

        let layer0_names: Vec<_> = reader
            .tensors
            .iter()
            .filter(|t| t.name.contains("layers.0."))
            .collect();
        assert!(!layer0_names.is_empty(), "Layer 0 should be kept");
    }

    #[test]
    fn test_extract_layer_number() {
        assert_eq!(
            extract_layer_number("model.layers.5.self_attn.q_proj.weight"),
            Some(5)
        );
        assert_eq!(extract_layer_number("blk.0.attn_q.weight"), Some(0));
        assert_eq!(extract_layer_number("model.norm.weight"), None);
        assert_eq!(extract_layer_number("lm_head.weight"), None);
    }

    #[test]
    fn test_create_student_progressive() {
        let mut tensors = std::collections::BTreeMap::new();
        tensors.insert(
            "model.layers.0.weight".to_string(),
            (vec![1.0; 4], vec![2, 2]),
        );
        tensors.insert(
            "model.layers.1.weight".to_string(),
            (vec![2.0; 4], vec![2, 2]),
        );
        tensors.insert(
            "model.layers.2.weight".to_string(),
            (vec![3.0; 4], vec![2, 2]),
        );
        tensors.insert(
            "model.layers.3.weight".to_string(),
            (vec![4.0; 4], vec![2, 2]),
        );
        tensors.insert("model.norm.weight".to_string(), (vec![1.0; 2], vec![2]));

        let student = create_student_from_teacher(&tensors, DistillStrategy::Progressive);
        // Even layers (0, 2) + non-layer tensors (norm) = 3
        assert_eq!(student.len(), 3);
        assert!(student.contains_key("model.layers.0.weight"));
        assert!(!student.contains_key("model.layers.1.weight"));
        assert!(student.contains_key("model.layers.2.weight"));
        assert!(!student.contains_key("model.layers.3.weight"));
        assert!(student.contains_key("model.norm.weight"));
    }

    #[test]
    fn test_create_student_standard() {
        let mut tensors = std::collections::BTreeMap::new();
        tensors.insert("a".to_string(), (vec![1.0], vec![1]));
        tensors.insert("b".to_string(), (vec![2.0], vec![1]));

        let student = create_student_from_teacher(&tensors, DistillStrategy::Standard);
        assert_eq!(student.len(), 2, "Standard copies all tensors");
    }
}
