//! Publish command implementation — upload trained models to HuggingFace Hub

use std::path::Path;

use crate::cli::logging::log;
use crate::cli::LogLevel;
use crate::config::PublishArgs;

pub fn run_publish(args: PublishArgs, level: LogLevel) -> Result<(), String> {
    log(level, LogLevel::Normal, &format!("Publishing to {}", args.repo));

    // Validate model directory
    if !args.model_dir.exists() {
        return Err(format!("Model directory not found: {}", args.model_dir.display()));
    }

    // Find model files to upload
    let files = collect_model_files(&args.model_dir).map_err(|e| format!("File scan: {e}"))?;
    if files.is_empty() {
        return Err(format!("No model files found in {}", args.model_dir.display()));
    }

    log(level, LogLevel::Normal, &format!("  Found {} file(s) to upload", files.len()));
    for (path, remote) in &files {
        log(level, LogLevel::Verbose, &format!("    {} -> {}", path.display(), remote));
    }

    if args.dry_run {
        log(level, LogLevel::Normal, "Dry run — skipping upload");
        return Ok(());
    }

    do_publish(&args, &files, level)
}

#[cfg(feature = "hub-publish")]
fn do_publish(
    args: &PublishArgs,
    files: &[(std::path::PathBuf, String)],
    level: LogLevel,
) -> Result<(), String> {
    use crate::hf_pipeline::publish::config::PublishConfig;

    use crate::hf_pipeline::publish::publisher::HfPublisher;

    let config =
        PublishConfig { repo_id: args.repo.clone(), private: args.private, ..Default::default() };

    let model_card = if args.model_card { Some(build_model_card(args)) } else { None };

    let publisher =
        HfPublisher::new(config).map_err(|e| format!("Publisher initialization: {e}"))?;

    let file_refs: Vec<(&Path, &str)> =
        files.iter().map(|(path, remote)| (path.as_path(), remote.as_str())).collect();

    let result = publisher
        .publish(&file_refs, model_card.as_ref())
        .map_err(|e| format!("Upload failed: {e}"))?;

    log(level, LogLevel::Normal, &format!("Published: {result}"));
    Ok(())
}

#[cfg(not(feature = "hub-publish"))]
fn do_publish(
    _args: &PublishArgs,
    _files: &[(std::path::PathBuf, String)],
    _level: LogLevel,
) -> Result<(), String> {
    Err("Publishing requires the 'hub-publish' feature. Rebuild with: cargo install entrenar --features hub-publish".to_string())
}

/// Collect model files from the output directory for upload.
fn collect_model_files(dir: &Path) -> Result<Vec<(std::path::PathBuf, String)>, std::io::Error> {
    let mut files = Vec::new();

    let extensions = ["safetensors", "gguf", "bin", "json", "yaml", "yml", "txt"];

    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let name = match path.file_name().and_then(|n| n.to_str()) {
            Some(n) => n.to_string(),
            None => continue,
        };

        // Skip hidden files
        if name.starts_with('.') {
            continue;
        }

        // Include files with known extensions
        let include = extensions.iter().any(|ext| name.ends_with(&format!(".{ext}")));

        if include {
            files.push((path, name));
        }
    }

    // Sort for deterministic upload order
    files.sort_by(|a, b| a.1.cmp(&b.1));

    Ok(files)
}

/// Build a model card from publish args and training metadata.
#[cfg(feature = "hub-publish")]
fn build_model_card(args: &PublishArgs) -> crate::hf_pipeline::publish::model_card::ModelCard {
    use crate::hf_pipeline::publish::model_card::ModelCard;

    let model_name = args.repo.rsplit('/').next().unwrap_or(&args.repo).to_string();

    let metadata_path = args.model_dir.join("final_model.json");
    let training_details = read_training_metadata(&metadata_path);

    ModelCard {
        model_name,
        description: format!("Fine-tuned model published via entrenar from {}", args.repo),
        license: Some("apache-2.0".to_string()),
        language: Vec::new(),
        tags: vec!["entrenar".to_string(), "fine-tuned".to_string(), "rust".to_string()],
        metrics: Vec::new(),
        training_details,
        base_model: args.base_model.clone(),
    }
}

/// Read training metadata from final_model.json if it exists.
#[cfg(any(feature = "hub-publish", test))]
fn read_training_metadata(path: &Path) -> Option<String> {
    let content = std::fs::read_to_string(path).ok()?;
    let json: serde_json::Value = serde_json::from_str(&content).ok()?;

    let mut details = String::new();
    if let Some(epochs) = json.get("epochs_completed").and_then(serde_json::Value::as_u64) {
        details.push_str(&format!("- **Epochs:** {epochs}\n"));
    }
    if let Some(loss) = json.get("final_loss").and_then(serde_json::Value::as_f64) {
        details.push_str(&format!("- **Final loss:** {loss:.6}\n"));
    }
    if let Some(mode) = json.get("training_mode").and_then(|v| v.as_str()) {
        details.push_str(&format!("- **Training mode:** {mode}\n"));
    }

    if details.is_empty() {
        None
    } else {
        Some(details)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_collect_model_files_empty_dir() {
        let dir = tempfile::tempdir().expect("temp file creation should succeed");
        let files = collect_model_files(dir.path()).expect("operation should succeed");
        assert!(files.is_empty());
    }

    #[test]
    fn test_collect_model_files_filters_extensions() {
        let dir = tempfile::tempdir().expect("temp file creation should succeed");
        std::fs::write(dir.path().join("model.safetensors"), b"data")
            .expect("file write should succeed");
        std::fs::write(dir.path().join("config.json"), b"{}").expect("file write should succeed");
        std::fs::write(dir.path().join("random.xyz"), b"skip").expect("file write should succeed");
        std::fs::write(dir.path().join(".hidden"), b"skip").expect("file write should succeed");

        let files = collect_model_files(dir.path()).expect("operation should succeed");
        assert_eq!(files.len(), 2);
        let names: Vec<&str> = files.iter().map(|(_, n)| n.as_str()).collect();
        assert!(names.contains(&"model.safetensors"));
        assert!(names.contains(&"config.json"));
    }

    #[test]
    fn test_collect_model_files_sorted() {
        let dir = tempfile::tempdir().expect("temp file creation should succeed");
        std::fs::write(dir.path().join("z_weights.safetensors"), b"w")
            .expect("file write should succeed");
        std::fs::write(dir.path().join("a_config.json"), b"c").expect("file write should succeed");

        let files = collect_model_files(dir.path()).expect("operation should succeed");
        assert_eq!(files[0].1, "a_config.json");
        assert_eq!(files[1].1, "z_weights.safetensors");
    }

    #[test]
    fn test_run_publish_missing_dir() {
        let args = PublishArgs {
            model_dir: PathBuf::from("/tmp/definitely-nonexistent-dir-12345"),
            repo: "user/model".to_string(),
            private: false,
            model_card: true,
            merge_adapters: false,
            base_model: None,
            format: "safetensors".to_string(),
            dry_run: false,
        };
        let result = run_publish(args, LogLevel::Quiet);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not found"));
    }

    #[test]
    fn test_run_publish_empty_dir() {
        let dir = tempfile::tempdir().expect("temp file creation should succeed");
        let args = PublishArgs {
            model_dir: dir.path().to_path_buf(),
            repo: "user/model".to_string(),
            private: false,
            model_card: true,
            merge_adapters: false,
            base_model: None,
            format: "safetensors".to_string(),
            dry_run: false,
        };
        let result = run_publish(args, LogLevel::Quiet);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("No model files"));
    }

    #[test]
    fn test_run_publish_dry_run() {
        let dir = tempfile::tempdir().expect("temp file creation should succeed");
        std::fs::write(dir.path().join("model.safetensors"), b"data")
            .expect("file write should succeed");

        let args = PublishArgs {
            model_dir: dir.path().to_path_buf(),
            repo: "user/model".to_string(),
            private: false,
            model_card: true,
            merge_adapters: false,
            base_model: None,
            format: "safetensors".to_string(),
            dry_run: true,
        };
        let result = run_publish(args, LogLevel::Quiet);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_publish_no_hub_feature() {
        let dir = tempfile::tempdir().expect("temp file creation should succeed");
        std::fs::write(dir.path().join("model.safetensors"), b"data")
            .expect("file write should succeed");

        let args = PublishArgs {
            model_dir: dir.path().to_path_buf(),
            repo: "user/model".to_string(),
            private: false,
            model_card: true,
            merge_adapters: false,
            base_model: None,
            format: "safetensors".to_string(),
            dry_run: false,
        };
        // Without hub-publish feature, this returns an error
        // With hub-publish feature, this would attempt actual upload
        let result = run_publish(args, LogLevel::Quiet);
        #[cfg(not(feature = "hub-publish"))]
        assert!(result.unwrap_err().contains("hub-publish"));
        #[cfg(feature = "hub-publish")]
        let _ = result; // May succeed or fail depending on HF_TOKEN
    }

    #[test]
    fn test_read_training_metadata_missing() {
        let result = read_training_metadata(Path::new("/tmp/nonexistent.json"));
        assert!(result.is_none());
    }

    #[test]
    fn test_read_training_metadata_invalid_json() {
        let dir = tempfile::tempdir().expect("temp file creation should succeed");
        let path = dir.path().join("final_model.json");
        std::fs::write(&path, "not json").expect("file write should succeed");
        let result = read_training_metadata(&path);
        assert!(result.is_none());
    }

    #[test]
    fn test_read_training_metadata_valid() {
        let dir = tempfile::tempdir().expect("temp file creation should succeed");
        let metadata = serde_json::json!({
            "epochs_completed": 3,
            "final_loss": 1.5432,
            "training_mode": "LoRA"
        });
        let path = dir.path().join("final_model.json");
        std::fs::write(&path, serde_json::to_string(&metadata).expect("file write should succeed"))
            .expect("file write should succeed");

        let details = read_training_metadata(&path).expect("operation should succeed");
        assert!(details.contains("Epochs"));
        assert!(details.contains("1.5432"));
        assert!(details.contains("LoRA"));
    }

    #[test]
    fn test_read_training_metadata_partial() {
        let dir = tempfile::tempdir().expect("temp file creation should succeed");
        let metadata = serde_json::json!({
            "epochs_completed": 5
        });
        let path = dir.path().join("final_model.json");
        std::fs::write(&path, serde_json::to_string(&metadata).expect("file write should succeed"))
            .expect("file write should succeed");

        let details = read_training_metadata(&path).expect("operation should succeed");
        assert!(details.contains("Epochs"));
        assert!(details.contains('5'));
    }

    #[test]
    fn test_read_training_metadata_empty_json() {
        let dir = tempfile::tempdir().expect("temp file creation should succeed");
        let path = dir.path().join("final_model.json");
        std::fs::write(&path, "{}").expect("file write should succeed");

        let result = read_training_metadata(&path);
        assert!(result.is_none());
    }

    #[test]
    fn test_collect_model_files_all_extensions() {
        let dir = tempfile::tempdir().expect("temp file creation should succeed");
        for ext in &["safetensors", "gguf", "bin", "json", "yaml", "yml", "txt"] {
            std::fs::write(dir.path().join(format!("file.{ext}")), b"data")
                .expect("file write should succeed");
        }
        let files = collect_model_files(dir.path()).expect("operation should succeed");
        assert_eq!(files.len(), 7);
    }

    #[test]
    fn test_collect_model_files_skips_directories() {
        let dir = tempfile::tempdir().expect("temp file creation should succeed");
        std::fs::write(dir.path().join("model.safetensors"), b"data")
            .expect("file write should succeed");
        std::fs::create_dir(dir.path().join("subdir")).expect("thread join should succeed");

        let files = collect_model_files(dir.path()).expect("operation should succeed");
        assert_eq!(files.len(), 1);
    }
}
