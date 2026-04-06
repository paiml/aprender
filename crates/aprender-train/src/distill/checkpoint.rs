//! Student model checkpoint saving for knowledge distillation
//!
//! Saves student model weights along with distillation metadata including
//! teacher model name, temperature, alpha, and loss.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Distillation checkpoint metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillationCheckpoint {
    /// Teacher model name or path
    pub teacher_model: String,
    /// Distillation temperature
    pub temperature: f32,
    /// KD loss weight (alpha)
    pub alpha: f32,
    /// Final distillation loss
    pub final_loss: Option<f32>,
    /// Training epoch at checkpoint
    pub epoch: usize,
    /// Training step at checkpoint
    pub step: usize,
}

/// Save a student model checkpoint with distillation metadata
///
/// Creates:
/// - Weight file (SafeTensors format)
/// - `distillation_metadata.json` sidecar with teacher info, temperature, alpha, loss
///
/// Returns the path to the weight file.
#[allow(clippy::implicit_hasher)]
pub fn save_student_checkpoint(
    weights: &HashMap<String, Vec<f32>>,
    shapes: &HashMap<String, Vec<usize>>,
    checkpoint: &DistillationCheckpoint,
    output_dir: impl AsRef<Path>,
    filename: &str,
) -> Result<PathBuf, std::io::Error> {
    let output_dir = output_dir.as_ref();
    std::fs::create_dir_all(output_dir)?;

    // Save weights as SafeTensors
    use safetensors::tensor::{Dtype, TensorView};

    let mut sorted_names: Vec<&String> = weights.keys().collect();
    sorted_names.sort();

    let tensor_data: Vec<(String, Vec<u8>, Vec<usize>)> = sorted_names
        .iter()
        .map(|name| {
            let data = &weights[*name];
            let bytes: Vec<u8> = bytemuck::cast_slice(data).to_vec();
            let shape = shapes.get(*name).cloned().unwrap_or_else(|| vec![data.len()]);
            ((*name).clone(), bytes, shape)
        })
        .collect();

    let views: Vec<(&str, TensorView<'_>)> = tensor_data
        .iter()
        .map(|(name, bytes, shape)| {
            let view = TensorView::new(Dtype::F32, shape.clone(), bytes)
                .expect("TensorView construction must not fail for valid F32 data");
            (name.as_str(), view)
        })
        .collect();

    let mut metadata = HashMap::new();
    metadata.insert("teacher_model".to_string(), checkpoint.teacher_model.clone());
    metadata.insert("temperature".to_string(), format!("{}", checkpoint.temperature));
    metadata.insert("alpha".to_string(), format!("{}", checkpoint.alpha));
    metadata.insert("epoch".to_string(), format!("{}", checkpoint.epoch));
    metadata.insert("step".to_string(), format!("{}", checkpoint.step));
    if let Some(loss) = checkpoint.final_loss {
        metadata.insert("final_loss".to_string(), format!("{loss}"));
    }

    let safetensor_bytes = safetensors::serialize(views, Some(metadata))
        .map_err(|e| std::io::Error::other(e.to_string()))?;

    let weights_path = output_dir.join(filename);
    std::fs::write(&weights_path, safetensor_bytes)?;

    // Save distillation metadata sidecar
    let metadata_json = serde_json::to_string_pretty(checkpoint)
        .map_err(|e| std::io::Error::other(e.to_string()))?;
    std::fs::write(output_dir.join("distillation_metadata.json"), metadata_json)?;

    Ok(weights_path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_test_data(
    ) -> (HashMap<String, Vec<f32>>, HashMap<String, Vec<usize>>, DistillationCheckpoint) {
        let mut weights = HashMap::new();
        let mut shapes = HashMap::new();

        weights.insert("student.layer.0.weight".to_string(), vec![1.0; 64]);
        shapes.insert("student.layer.0.weight".to_string(), vec![8, 8]);
        weights.insert("student.layer.0.bias".to_string(), vec![0.1; 8]);
        shapes.insert("student.layer.0.bias".to_string(), vec![8]);

        let checkpoint = DistillationCheckpoint {
            teacher_model: "bert-base-uncased".to_string(),
            temperature: 3.0,
            alpha: 0.5,
            final_loss: Some(1.23),
            epoch: 5,
            step: 10000,
        };

        (weights, shapes, checkpoint)
    }

    #[test]
    fn test_save_checkpoint_creates_files() {
        let (weights, shapes, checkpoint) = make_test_data();
        let tmp = TempDir::new().expect("temp file creation should succeed");

        let path = save_student_checkpoint(
            &weights,
            &shapes,
            &checkpoint,
            tmp.path(),
            "student.safetensors",
        )
        .expect("operation should succeed");

        assert!(path.exists());
        assert!(tmp.path().join("distillation_metadata.json").exists());
    }

    #[test]
    fn test_save_checkpoint_safetensors_valid() {
        let (weights, shapes, checkpoint) = make_test_data();
        let tmp = TempDir::new().expect("temp file creation should succeed");

        let path = save_student_checkpoint(
            &weights,
            &shapes,
            &checkpoint,
            tmp.path(),
            "student.safetensors",
        )
        .expect("operation should succeed");

        let data = std::fs::read(&path).expect("file read should succeed");
        let loaded = safetensors::SafeTensors::deserialize(&data).expect("load should succeed");
        assert_eq!(loaded.len(), 2);

        let names = loaded.names();
        assert!(names.contains(&"student.layer.0.weight"));
        assert!(names.contains(&"student.layer.0.bias"));
    }

    #[test]
    fn test_save_checkpoint_metadata_in_safetensors() {
        let (weights, shapes, checkpoint) = make_test_data();
        let tmp = TempDir::new().expect("temp file creation should succeed");

        let path = save_student_checkpoint(
            &weights,
            &shapes,
            &checkpoint,
            tmp.path(),
            "student.safetensors",
        )
        .expect("operation should succeed");

        let data = std::fs::read(&path).expect("file read should succeed");
        let (_, st_meta) =
            safetensors::SafeTensors::read_metadata(&data).expect("deserialization should succeed");
        let meta = st_meta.metadata().as_ref().expect("operation should succeed");

        assert_eq!(meta.get("teacher_model").expect("key should exist"), "bert-base-uncased");
        assert_eq!(meta.get("temperature").expect("key should exist"), "3");
        assert_eq!(meta.get("alpha").expect("key should exist"), "0.5");
        assert_eq!(meta.get("epoch").expect("key should exist"), "5");
    }

    #[test]
    fn test_save_checkpoint_distillation_metadata() {
        let (weights, shapes, checkpoint) = make_test_data();
        let tmp = TempDir::new().expect("temp file creation should succeed");

        save_student_checkpoint(&weights, &shapes, &checkpoint, tmp.path(), "student.safetensors")
            .expect("operation should succeed");

        let json = std::fs::read_to_string(tmp.path().join("distillation_metadata.json"))
            .expect("file read should succeed");
        let loaded: DistillationCheckpoint =
            serde_json::from_str(&json).expect("JSON deserialization should succeed");

        assert_eq!(loaded.teacher_model, "bert-base-uncased");
        assert_eq!(loaded.temperature, 3.0);
        assert_eq!(loaded.alpha, 0.5);
        assert_eq!(loaded.final_loss, Some(1.23));
        assert_eq!(loaded.epoch, 5);
        assert_eq!(loaded.step, 10000);
    }

    #[test]
    fn test_save_checkpoint_no_loss() {
        let mut weights = HashMap::new();
        let mut shapes = HashMap::new();
        weights.insert("w".to_string(), vec![1.0; 4]);
        shapes.insert("w".to_string(), vec![2, 2]);

        let checkpoint = DistillationCheckpoint {
            teacher_model: "gpt2".to_string(),
            temperature: 2.0,
            alpha: 0.7,
            final_loss: None,
            epoch: 0,
            step: 0,
        };

        let tmp = TempDir::new().expect("temp file creation should succeed");
        let path =
            save_student_checkpoint(&weights, &shapes, &checkpoint, tmp.path(), "ckpt.safetensors")
                .expect("operation should succeed");
        assert!(path.exists());
    }
}
