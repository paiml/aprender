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
    // Validate teacher exists
    if !teacher_path.exists() {
        return Err(CliError::FileNotFound(teacher_path.to_path_buf()));
    }

    let distill_strategy: DistillStrategy = strategy
        .parse()
        .map_err(CliError::ValidationFailed)?;

    // Validate temperature and alpha
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

    // Plan mode
    if plan_only {
        return run_plan(teacher_path, student_path, distill_strategy, temperature, alpha, epochs, json_output);
    }

    // Validate required paths for training
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

    // Validate student exists if provided
    if let Some(student) = student_path {
        if !student.exists() {
            return Err(CliError::FileNotFound(student.to_path_buf()));
        }
    }

    // Validate training data exists if provided
    if let Some(data) = data_path {
        if !data.exists() {
            return Err(CliError::FileNotFound(data.to_path_buf()));
        }
    }

    if !json_output {
        output::pipeline_stage("Distilling", output::StageStatus::Running);
    }

    // Distillation execution
    let teacher_size = std::fs::metadata(teacher_path)
        .map_err(|e| CliError::ValidationFailed(format!("Cannot read teacher: {e}")))?
        .len();

    let student_size = student_path
        .and_then(|p| std::fs::metadata(p).ok())
        .map_or(teacher_size / 2, |m| m.len());

    if json_output {
        let json = serde_json::json!({
            "status": "configured",
            "teacher": teacher_path.display().to_string(),
            "student": student_path.map(|p| p.display().to_string()),
            "output": out.display().to_string(),
            "strategy": format!("{distill_strategy:?}"),
            "temperature": temperature,
            "alpha": alpha,
            "epochs": epochs,
            "teacher_size": teacher_size,
            "student_size": student_size,
            "note": "Full distillation execution requires entrenar distillation backend",
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&json).unwrap_or_default()
        );
    } else {
        println!();
        output::subheader("Distillation Configuration");
        println!(
            "{}",
            output::kv_table(&[
                (
                    "Teacher size",
                    humansize::format_size(teacher_size, humansize::BINARY),
                ),
                (
                    "Student size",
                    humansize::format_size(student_size, humansize::BINARY),
                ),
                (
                    "Compression",
                    format!(
                        "{:.1}x",
                        teacher_size as f64 / student_size as f64
                    ),
                ),
            ])
        );
        println!();
        println!(
            "  {} Distillation pipeline configured. Full execution requires entrenar backend.",
            output::badge_info("INFO")
        );
    }

    Ok(())
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
        assert!(matches!("standard".parse::<DistillStrategy>(), Ok(DistillStrategy::Standard)));
        assert!(matches!("kl".parse::<DistillStrategy>(), Ok(DistillStrategy::Standard)));
        assert!(matches!("progressive".parse::<DistillStrategy>(), Ok(DistillStrategy::Progressive)));
        assert!(matches!("ensemble".parse::<DistillStrategy>(), Ok(DistillStrategy::Ensemble)));
        assert!("unknown".parse::<DistillStrategy>().is_err());
    }

    #[test]
    fn test_run_teacher_not_found() {
        let result = run(
            Path::new("/nonexistent.apr"), None, None, Some(Path::new("/tmp/out.apr")),
            "standard", 3.0, 0.7, 3, false, false,
        );
        assert!(result.is_err());
        assert!(matches!(result, Err(CliError::FileNotFound(_))));
    }

    #[test]
    fn test_run_invalid_temperature() {
        let input = NamedTempFile::with_suffix(".apr").expect("create input");
        let result = run(
            input.path(), None, None, Some(Path::new("/tmp/out.apr")),
            "standard", 0.0, 0.7, 3, false, false,
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
            input.path(), None, None, Some(Path::new("/tmp/out.apr")),
            "standard", 3.0, 1.5, 3, false, false,
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
            input.path(), None, None, Some(Path::new("/tmp/out.apr")),
            "standard", 3.0, 0.7, 3, false, false,
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
            teacher.path(), Some(student.path()), None, None,
            "standard", 3.0, 0.7, 3, false, false,
        );
        assert!(result.is_err());
        match result {
            Err(CliError::ValidationFailed(msg)) => assert!(msg.contains("Output")),
            _ => panic!("Expected ValidationFailed"),
        }
    }

    #[test]
    fn test_run_valid() {
        let mut teacher = NamedTempFile::with_suffix(".apr").expect("create teacher");
        teacher.write_all(&[0u8; 1024]).expect("write");
        let mut student = NamedTempFile::with_suffix(".apr").expect("create student");
        student.write_all(&[0u8; 512]).expect("write");
        let result = run(
            teacher.path(), Some(student.path()), None, Some(Path::new("/tmp/distilled.apr")),
            "standard", 3.0, 0.7, 3, false, false,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_plan_mode() {
        let mut teacher = NamedTempFile::with_suffix(".apr").expect("create teacher");
        teacher.write_all(&[0u8; 2048]).expect("write");
        let result = run(
            teacher.path(), None, None, None,
            "standard", 3.0, 0.7, 3, true, false,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_plan_json() {
        let mut teacher = NamedTempFile::with_suffix(".apr").expect("create teacher");
        teacher.write_all(&[0u8; 2048]).expect("write");
        let result = run(
            teacher.path(), None, None, None,
            "progressive", 4.0, 0.5, 5, true, true,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_progressive_no_student() {
        // Progressive distillation doesn't require a student (creates internally)
        let mut teacher = NamedTempFile::with_suffix(".apr").expect("create teacher");
        teacher.write_all(&[0u8; 1024]).expect("write");
        let result = run(
            teacher.path(), None, None, Some(Path::new("/tmp/out.apr")),
            "progressive", 3.0, 0.7, 3, false, false,
        );
        assert!(result.is_ok());
    }
}
