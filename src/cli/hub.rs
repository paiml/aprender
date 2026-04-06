//! HuggingFace Hub CLI commands.

use std::path::PathBuf;

use clap::Subcommand;

/// Import source options.
#[derive(Subcommand)]
pub enum ImportSource {
    /// Import from a local file (CSV, JSON, JSONL, Parquet, Arrow)
    Local {
        /// Input file or directory path
        input: PathBuf,
        /// Output file path (format inferred from extension)
        #[arg(short, long)]
        output: PathBuf,
        /// Force output format (csv, json, jsonl, parquet, arrow)
        #[arg(short, long)]
        format: Option<String>,
    },
    /// Import from HuggingFace Hub.
    #[allow(clippy::doc_markdown)]
    #[cfg(feature = "hf-hub")]
    Hf {
        /// Dataset repository ID (e.g., "squad", "openai/gsm8k")
        repo_id: String,
        /// Output path for the downloaded dataset
        #[arg(short, long)]
        output: PathBuf,
        /// Git revision (branch, tag, or commit)
        #[arg(short, long, default_value = "main")]
        revision: String,
        /// Dataset subset/configuration
        #[arg(short, long)]
        subset: Option<String>,
        /// Data split (train, validation, test)
        #[arg(long, default_value = "train")]
        split: String,
    },
}

/// Import a local dataset file, converting between formats.
pub(crate) fn cmd_import_local(
    input: &PathBuf,
    output: &PathBuf,
    format: Option<&str>,
) -> crate::Result<()> {
    use super::basic::{load_dataset, save_dataset};
    use crate::Dataset;

    if !input.exists() {
        return Err(crate::Error::io(
            std::io::Error::new(std::io::ErrorKind::NotFound, "Input file not found"),
            input,
        ));
    }

    println!("Importing {}...", input.display());
    let dataset = load_dataset(input)?;

    // Apply format override by using the target extension for save, then rename
    if let Some(fmt) = format {
        let forced_output = output.with_extension(fmt);
        save_dataset(&dataset, &forced_output)?;
        if forced_output != *output {
            std::fs::rename(&forced_output, output).map_err(|e| crate::Error::io(e, output))?;
        }
    } else {
        save_dataset(&dataset, output)?;
    }

    println!(
        "Imported {} rows: {} -> {}",
        dataset.len(),
        input.display(),
        output.display()
    );

    Ok(())
}

/// HuggingFace Hub commands.
#[cfg(feature = "hf-hub")]
#[derive(Subcommand)]
pub enum HubCommands {
    /// Push (upload) a dataset to HuggingFace Hub
    #[allow(clippy::doc_markdown)]
    Push {
        /// Path to the parquet file to upload
        input: PathBuf,
        /// HuggingFace repository ID (e.g., "paiml/my-dataset")
        repo_id: String,
        /// Path in the repository (e.g., "data/train.parquet")
        #[arg(short, long)]
        path_in_repo: Option<String>,
        /// Commit message for the upload
        #[arg(short, long, default_value = "Upload via alimentar")]
        message: String,
        /// Path to README.md to upload as dataset card
        #[arg(long)]
        readme: Option<PathBuf>,
        /// Make the dataset private
        #[arg(long)]
        private: bool,
    },
}

/// Import from HuggingFace Hub.
#[cfg(feature = "hf-hub")]
pub(crate) fn cmd_import_hf(
    repo_id: &str,
    output: &PathBuf,
    revision: &str,
    subset: Option<&str>,
    split: &str,
) -> crate::Result<()> {
    use crate::{dataset::Dataset, hf_hub::HfDataset};

    println!("Importing {} from HuggingFace Hub...", repo_id);

    let mut builder = HfDataset::builder(repo_id).revision(revision).split(split);

    if let Some(s) = subset {
        builder = builder.subset(s);
    }

    let dataset = builder.build()?;

    println!("Downloading to {}...", output.display());
    let data = dataset.download_to(output)?;

    println!(
        "Successfully imported {} ({} rows) to {}",
        repo_id,
        data.len(),
        output.display()
    );

    Ok(())
}

/// Prints a quality warning for HuggingFace Hub uploads.
#[cfg(feature = "hf-hub")]
fn print_quality_warning() {
    eprintln!();
    eprintln!("WARNING: Data quality is CRITICAL for ML datasets!");
    eprintln!("Publishing low-quality data harms the ML community.");
    eprintln!();
    eprintln!("Before publishing, verify quality with:");
    eprintln!("  alimentar quality score <file.parquet>");
    eprintln!();
    eprintln!("Minimum recommended: Grade B (85%)");
    eprintln!();
    eprintln!("To improve quality, use:");
    eprintln!("  aprender clean <input> -o <output>     # Clean data");
    eprintln!("  entrenar augment <input> -o <output>   # Augment for training");
    eprintln!();
    eprintln!("See: https://paiml.github.io/alimentar/hf-hub/publishing.html");
    eprintln!();
}

/// Push (upload) a dataset to HuggingFace Hub.
#[cfg(feature = "hf-hub")]
pub(crate) fn cmd_hub_push(
    input: &PathBuf,
    repo_id: &str,
    path_in_repo: Option<&str>,
    message: &str,
    readme: Option<&PathBuf>,
    private: bool,
) -> crate::Result<()> {
    use crate::hf_hub::HfPublisher;

    // Display quality warning
    print_quality_warning();

    // Validate input file exists
    if !input.exists() {
        return Err(crate::Error::io(
            std::io::Error::new(std::io::ErrorKind::NotFound, "Input file not found"),
            input,
        ));
    }

    // Derive path_in_repo from filename if not specified
    let path_in_repo = path_in_repo.map(String::from).unwrap_or_else(|| {
        input
            .file_name()
            .map(|f| f.to_string_lossy().into_owned())
            .unwrap_or_else(|| "data.parquet".to_string())
    });

    println!("Pushing {} to {}...", input.display(), repo_id);

    let publisher = HfPublisher::new(repo_id)
        .with_private(private)
        .with_commit_message(message);

    // Create repo (idempotent - succeeds if already exists)
    println!("Creating repository (if needed)...");
    publisher.create_repo_sync()?;

    // Upload parquet file
    println!("Uploading {}...", path_in_repo);
    publisher.upload_parquet_file_sync(input, &path_in_repo)?;

    // Upload README if provided
    if let Some(readme_path) = readme {
        println!("Uploading README.md...");
        let readme_content =
            std::fs::read_to_string(readme_path).map_err(|e| crate::Error::io(e, readme_path))?;
        publisher.upload_readme_validated_sync(&readme_content)?;
    }

    let visibility = if private { "private" } else { "public" };
    println!(
        "Successfully pushed to https://huggingface.co/datasets/{} ({})",
        repo_id, visibility
    );

    Ok(())
}
