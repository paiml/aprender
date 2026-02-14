//! HuggingFace Hub publish command (APR-PUB-001)
//!
//! Publishes models to HuggingFace Hub with auto-generated model cards.
//! Now uses native Rust HTTP upload instead of shelling out to Python CLI.
//!
//! # Toyota Way Principles
//!
//! - **Jidoka**: Auto-generate model cards to prevent incomplete documentation
//! - **Genchi Genbutsu**: Verify model before publishing
//! - **Muda**: Eliminate manual model card creation
//! - **Andon Cord**: Fail loudly on upload errors (APR-PUB-001)

use crate::error::CliError;
use aprender::format::model_card::ModelCard;
use aprender::hf_hub::{HfHubClient, PushOptions, UploadProgress};
use std::fs;
use std::path::Path;
use std::sync::Arc;

/// Validate publish inputs: repo ID format, directory exists, model files found.
fn validate_publish_inputs(
    directory: &Path,
    repo_id: &str,
) -> Result<Vec<std::path::PathBuf>, CliError> {
    if !repo_id.contains('/') || repo_id.split('/').count() != 2 {
        return Err(CliError::ValidationFailed(format!(
            "Invalid repo ID '{}'. Expected format: org/repo-name",
            repo_id
        )));
    }

    if !directory.exists() {
        return Err(CliError::FileNotFound(directory.to_path_buf()));
    }

    let files = find_model_files(directory)?;
    if files.is_empty() {
        return Err(CliError::ValidationFailed(format!(
            "No model files found in {}. Expected .apr, .safetensors, or .gguf files.",
            directory.display()
        )));
    }

    Ok(files)
}

/// Upload model files and README to HuggingFace Hub.
fn upload_to_hub(
    client: &HfHubClient,
    repo_id: &str,
    files: &[std::path::PathBuf],
    readme_content: &str,
    commit_msg: &str,
    verbose: bool,
) -> Result<(), CliError> {
    let progress_callback: Arc<dyn Fn(UploadProgress) + Send + Sync> = Arc::new(move |progress| {
        if verbose {
            println!(
                "  [{}/{}] {} ({:.1}%)",
                progress.files_completed + 1,
                progress.total_files,
                progress.current_file,
                progress.percentage()
            );
        }
    });

    for file in files {
        let filename = file
            .file_name()
            .ok_or_else(|| CliError::ValidationFailed("Invalid file path".into()))?
            .to_string_lossy()
            .to_string();

        if verbose {
            let size = fs::metadata(file).map(|m| m.len()).unwrap_or(0);
            println!(
                "Uploading {} ({:.1} MB)...",
                filename,
                size as f64 / 1_000_000.0
            );
        }

        let file_data = fs::read(file)?;

        let options = PushOptions::new()
            .with_filename(filename)
            .with_commit_message(commit_msg)
            .with_progress_callback(progress_callback.clone())
            .with_create_repo(true);

        client
            .push_to_hub(repo_id, &file_data, options)
            .map_err(|e| CliError::NetworkError(format!("Upload failed: {e}")))?;
    }

    if verbose {
        println!("Uploading README.md...");
    }

    let readme_options = PushOptions::new()
        .with_filename("README.md")
        .with_commit_message(commit_msg)
        .with_create_repo(false);

    client
        .push_to_hub(repo_id, readme_content.as_bytes(), readme_options)
        .map_err(|e| CliError::NetworkError(format!("README upload failed: {e}")))?;

    Ok(())
}

/// Execute the publish command
pub fn execute(
    directory: &Path,
    repo_id: &str,
    model_name: Option<&str>,
    license: &str,
    pipeline_tag: &str,
    library_name: Option<&str>,
    tags: &[String],
    commit_message: Option<&str>,
    dry_run: bool,
    verbose: bool,
) -> Result<(), CliError> {
    let files = validate_publish_inputs(directory, repo_id)?;

    if verbose {
        println!("Found {} model files:", files.len());
        for f in &files {
            println!("  - {}", f.display());
        }
    }

    let model_card = generate_model_card(
        repo_id,
        model_name,
        license,
        pipeline_tag,
        library_name,
        tags,
        &files,
    );
    let readme_content = model_card.to_huggingface_extended(pipeline_tag, library_name, tags);

    if dry_run {
        println!("=== DRY RUN: Would publish to {} ===\n", repo_id);
        println!("Files to upload:");
        for f in &files {
            let size = fs::metadata(f).map(|m| m.len()).unwrap_or(0);
            println!("  - {} ({:.1} MB)", f.display(), size as f64 / 1_000_000.0);
        }
        println!("\nGenerated README.md:\n");
        println!("{}", readme_content);
        println!("\n=== DRY RUN COMPLETE ===");
        return Ok(());
    }

    let client = HfHubClient::new()
        .map_err(|e| CliError::ValidationFailed(format!("Failed to create HF Hub client: {e}")))?;

    if !client.is_authenticated() {
        return Err(CliError::ValidationFailed(
            "HF_TOKEN environment variable not set. Set it with: export HF_TOKEN=hf_...".into(),
        ));
    }

    let commit_msg = commit_message.unwrap_or("Upload via apr-cli publish");

    println!("Publishing to https://huggingface.co/{}", repo_id);
    let total_size: u64 = files
        .iter()
        .map(|f| fs::metadata(f).map(|m| m.len()).unwrap_or(0))
        .sum::<u64>()
        + readme_content.len() as u64;
    println!(
        "Total upload size: {:.1} MB",
        total_size as f64 / 1_000_000.0
    );

    upload_to_hub(
        &client,
        repo_id,
        &files,
        &readme_content,
        commit_msg,
        verbose,
    )?;

    println!("\n✓ Published to https://huggingface.co/{}", repo_id);
    Ok(())
}

/// Find model files in directory
fn find_model_files(directory: &Path) -> Result<Vec<std::path::PathBuf>, CliError> {
    let mut files = Vec::new();

    let entries = fs::read_dir(directory)?;

    for entry in entries.flatten() {
        let path = entry.path();

        if path.is_file() {
            if let Some(ext) = path.extension() {
                let ext_str = ext.to_string_lossy().to_lowercase();
                if ext_str == "apr" || ext_str == "safetensors" || ext_str == "gguf" {
                    files.push(path);
                }
            }
        }
    }

    // Sort for deterministic order
    files.sort();
    Ok(files)
}

/// Generate model card from parameters
fn generate_model_card(
    repo_id: &str,
    model_name: Option<&str>,
    license: &str,
    _pipeline_tag: &str,
    _library_name: Option<&str>,
    _tags: &[String],
    _files: &[std::path::PathBuf],
) -> ModelCard {
    let name = model_name.unwrap_or_else(|| repo_id.split('/').next_back().unwrap_or(repo_id));

    ModelCard::new(repo_id, "1.0.0")
        .with_name(name)
        .with_license(license)
        .with_description(format!("{} model published via aprender", name))
}

// Note: upload_file function removed in APR-PUB-001
// Now using native aprender::hf_hub::HfHubClient instead of shelling out to huggingface-cli

/// Extended model card generation for HuggingFace format
trait ModelCardExt {
    fn to_huggingface_extended(
        &self,
        pipeline_tag: &str,
        library_name: Option<&str>,
        extra_tags: &[String],
    ) -> String;
}

impl ModelCardExt for ModelCard {
    fn to_huggingface_extended(
        &self,
        pipeline_tag: &str,
        library_name: Option<&str>,
        extra_tags: &[String],
    ) -> String {
        use std::fmt::Write;

        let mut output = String::from("---\n");

        // License
        if let Some(license) = &self.license {
            let _ = writeln!(output, "license: {}", license.to_lowercase());
        }

        // Language (default to multilingual for ASR)
        if pipeline_tag == "automatic-speech-recognition" {
            output.push_str("language:\n");
            output.push_str("  - en\n");
            output.push_str("  - multilingual\n");
        }

        // Pipeline tag
        let _ = writeln!(output, "pipeline_tag: {}", pipeline_tag);

        // Library name
        if let Some(lib) = library_name {
            let _ = writeln!(output, "library_name: {}", lib);
        }

        // Tags
        output.push_str("tags:\n");
        if let Some(arch) = &self.architecture {
            let _ = writeln!(output, "  - {}", arch.to_lowercase());
        }
        output.push_str("  - aprender\n");
        output.push_str("  - rust\n");

        // Extra tags (deduplicated)
        let mut seen_tags = std::collections::HashSet::new();
        seen_tags.insert("aprender");
        seen_tags.insert("rust");

        // Pipeline-specific tags
        if pipeline_tag == "automatic-speech-recognition" {
            if seen_tags.insert("speech-recognition") {
                output.push_str("  - speech-recognition\n");
            }
            if seen_tags.insert("audio") {
                output.push_str("  - audio\n");
            }
        }

        // Extra tags (skip duplicates)
        for tag in extra_tags {
            if seen_tags.insert(tag.as_str()) {
                let _ = writeln!(output, "  - {}", tag);
            }
        }

        // Model index (results, dataset, and metrics are all required by HuggingFace)
        output.push_str("model-index:\n");
        let _ = writeln!(output, "  - name: {}", self.model_id);
        output.push_str("    results:\n");
        output.push_str("      - task:\n");
        let _ = writeln!(output, "          type: {}", pipeline_tag);
        output.push_str("        dataset:\n");
        output.push_str("          name: custom\n");
        output.push_str("          type: custom\n");
        output.push_str("        metrics:\n");
        if self.metrics.is_empty() {
            // Add placeholder metric when none provided (required by HuggingFace)
            output.push_str("          - name: accuracy\n");
            output.push_str("            type: custom\n");
            output.push_str("            value: N/A\n");
        } else {
            for (key, value) in &self.metrics {
                let _ = writeln!(output, "          - name: {}", key);
                output.push_str("            type: custom\n");
                let _ = writeln!(output, "            value: {}", value);
            }
        }

        output.push_str("---\n\n");

        // Title
        let _ = writeln!(output, "# {}\n", self.name);

        // Description
        if let Some(desc) = &self.description {
            let _ = writeln!(output, "{}\n", desc);
        }

        // Formats section
        output.push_str("## Available Formats\n\n");
        output.push_str("| Format | Description |\n");
        output.push_str("|--------|-------------|\n");
        output.push_str("| `model.apr` | Native APR format (streaming, WASM-optimized) |\n");
        output.push_str("| `model.safetensors` | HuggingFace standard format |\n");
        output.push('\n');

        // Usage section
        output.push_str("## Usage\n\n");
        output.push_str("```rust\n");
        output.push_str("use aprender::Model;\n");
        output.push('\n');
        output.push_str("let model = Model::load(\"model.apr\")?;\n");
        output.push_str("let result = model.run(&input)?;\n");
        output.push_str("```\n\n");

        // Framework
        output.push_str("## Framework\n\n");
        let _ = writeln!(output, "- **Version:** {}", self.framework_version);
        if let Some(rust) = &self.rust_version {
            let _ = writeln!(output, "- **Rust:** {}", rust);
        }
        output.push('\n');

        // Citation
        output.push_str("## Citation\n\n");
        output.push_str("```bibtex\n");
        output.push_str("@software{aprender,\n");
        output.push_str("  title = {aprender: Rust ML Library},\n");
        output.push_str("  author = {PAIML},\n");
        output.push_str("  year = {2025},\n");
        output.push_str("  url = {https://github.com/paiml/aprender}\n");
        output.push_str("}\n");
        output.push_str("```\n");

        output
    }
}

#[cfg(test)]
#[path = "publish_tests.rs"]
mod tests;
