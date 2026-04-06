//! alimentar CLI - Data Loading, Distribution and Tooling
//!
//! Command-line interface for alimentar operations.

use std::{path::PathBuf, process::ExitCode};

use clap::{Parser, Subcommand};

mod basic;
mod drift;
mod fed;
mod hub;
mod quality;
mod registry;
mod view;

// Re-export subcommand enums
pub use drift::DriftCommands;
pub use fed::FedCommands;
#[cfg(feature = "hf-hub")]
pub use hub::HubCommands;
pub use hub::ImportSource;
pub use quality::QualityCommands;
pub use registry::RegistryCommands;

/// alimentar - Data Loading, Distribution and Tooling in Pure Rust
#[derive(Parser)]
#[command(name = "alimentar")]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Convert between data formats
    Convert {
        /// Input file path
        input: PathBuf,
        /// Output file path
        output: PathBuf,
    },
    /// Display dataset information
    Info {
        /// Path to dataset file
        path: PathBuf,
    },
    /// Display first N rows of a dataset
    Head {
        /// Path to dataset file
        path: PathBuf,
        /// Number of rows to display
        #[arg(short = 'n', long, default_value = "10")]
        rows: usize,
    },
    /// Display dataset schema
    Schema {
        /// Path to dataset file
        path: PathBuf,
    },
    /// Mix multiple datasets with weighted sampling
    Mix {
        /// Input files with optional weights (file:weight, e.g.,
        /// "data.parquet:0.8")
        #[arg(required = true)]
        inputs: Vec<String>,
        /// Output file path
        #[arg(short, long)]
        output: PathBuf,
        /// Random seed for reproducibility
        #[arg(short, long, default_value = "42")]
        seed: u64,
        /// Maximum total rows in output (0 = sum of all weighted inputs)
        #[arg(short = 'n', long, default_value = "0")]
        max_rows: usize,
    },
    /// Apply Fill-in-the-Middle (FIM) transform for code model training
    #[cfg(feature = "shuffle")]
    Fim {
        /// Input dataset file (Parquet/CSV/JSON)
        input: PathBuf,
        /// Output file path
        #[arg(short, long)]
        output: PathBuf,
        /// Column containing code text
        #[arg(long, default_value = "text")]
        column: String,
        /// FIM application rate (0.0-1.0)
        #[arg(long, default_value = "0.5")]
        rate: f64,
        /// FIM format: psm or spm
        #[arg(long, default_value = "psm")]
        format: String,
        /// Random seed for reproducibility
        #[arg(long, default_value = "42")]
        seed: u64,
    },
    /// Deduplicate dataset by text content (R-019)
    Dedup {
        /// Input dataset file
        input: PathBuf,
        /// Output file path
        #[arg(short, long)]
        output: PathBuf,
        /// Column to dedup on (auto-detected if not specified)
        #[arg(long)]
        column: Option<String>,
    },
    /// Filter dataset by text quality signals (R-022)
    #[command(name = "filter-text")]
    FilterText {
        /// Input dataset file
        input: PathBuf,
        /// Output file path
        #[arg(short, long)]
        output: PathBuf,
        /// Column containing text (auto-detected if not specified)
        #[arg(long)]
        column: Option<String>,
        /// Minimum composite quality score (0.0-1.0)
        #[arg(long, default_value = "0.4")]
        min_score: f64,
        /// Minimum document length in characters
        #[arg(long, default_value = "50")]
        min_length: usize,
        /// Maximum document length in characters
        #[arg(long, default_value = "1000000")]
        max_length: usize,
    },
    /// Interactive TUI viewer for datasets
    View {
        /// Path to dataset file (Parquet/Arrow/CSV/JSON)
        path: PathBuf,
        /// Initial search query
        #[arg(long)]
        search: Option<String>,
    },
    /// Import dataset from local files or HuggingFace Hub
    Import {
        #[command(subcommand)]
        source: ImportSource,
    },
    /// HuggingFace Hub commands (push/upload datasets)
    #[allow(clippy::doc_markdown)]
    #[cfg(feature = "hf-hub")]
    #[command(subcommand)]
    Hub(HubCommands),
    /// Registry commands for dataset sharing and discovery
    #[command(subcommand)]
    Registry(RegistryCommands),
    /// Data drift detection commands
    #[command(subcommand)]
    Drift(DriftCommands),
    /// Data quality checking commands
    #[command(subcommand)]
    Quality(QualityCommands),
    /// Federated split coordination commands
    #[command(subcommand)]
    Fed(FedCommands),
    /// Python doctest extraction commands
    #[cfg(feature = "doctest")]
    #[command(subcommand)]
    Doctest(DoctestCommands),
    /// Interactive REPL for data exploration
    #[cfg(feature = "repl")]
    Repl,
}

/// Python doctest extraction commands
#[cfg(feature = "doctest")]
#[derive(Subcommand)]
pub enum DoctestCommands {
    /// Extract doctests from Python source files
    Extract {
        /// Input directory containing Python source files
        input: PathBuf,
        /// Output parquet file
        #[arg(short, long)]
        output: PathBuf,
        /// Source identifier (e.g., "cpython", "numpy")
        #[arg(short, long, default_value = "unknown")]
        source: String,
        /// Version string or git SHA
        #[arg(short, long, default_value = "unknown")]
        version: String,
    },
    /// Merge multiple doctest corpora into one
    Merge {
        /// Input parquet files to merge
        #[arg(required = true)]
        inputs: Vec<PathBuf>,
        /// Output parquet file
        #[arg(short, long)]
        output: PathBuf,
    },
}

#[allow(clippy::too_many_lines)]
/// Run the alimentar CLI.
pub fn run() -> ExitCode {
    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Convert { input, output } => basic::cmd_convert(&input, &output),
        Commands::Info { path } => basic::cmd_info(&path),
        Commands::Head { path, rows } => basic::cmd_head(&path, rows),
        Commands::Schema { path } => basic::cmd_schema(&path),
        Commands::Mix {
            inputs,
            output,
            seed,
            max_rows,
        } => basic::cmd_mix(&inputs, &output, seed, max_rows),
        #[cfg(feature = "shuffle")]
        Commands::Fim {
            input,
            output,
            column,
            rate,
            format,
            seed,
        } => basic::cmd_fim(&input, &output, &column, rate, &format, seed),
        Commands::Dedup {
            input,
            output,
            column,
        } => basic::cmd_dedup(&input, &output, column.as_deref()),
        Commands::FilterText {
            input,
            output,
            column,
            min_score,
            min_length,
            max_length,
        } => basic::cmd_filter_text(
            &input,
            &output,
            column.as_deref(),
            min_score,
            min_length,
            max_length,
        ),
        Commands::View { path, search } => view::cmd_view(&path, search.as_deref()),
        Commands::Import { source } => match source {
            ImportSource::Local {
                input,
                output,
                format,
            } => hub::cmd_import_local(&input, &output, format.as_deref()),
            #[cfg(feature = "hf-hub")]
            ImportSource::Hf {
                repo_id,
                output,
                revision,
                subset,
                split,
            } => hub::cmd_import_hf(&repo_id, &output, &revision, subset.as_deref(), &split),
        },
        #[cfg(feature = "hf-hub")]
        Commands::Hub(hub_cmd) => match hub_cmd {
            HubCommands::Push {
                input,
                repo_id,
                path_in_repo,
                message,
                readme,
                private,
            } => hub::cmd_hub_push(
                &input,
                &repo_id,
                path_in_repo.as_deref(),
                &message,
                readme.as_ref(),
                private,
            ),
        },
        Commands::Registry(registry_cmd) => dispatch_registry(registry_cmd),
        Commands::Drift(drift_cmd) => dispatch_drift(drift_cmd),
        Commands::Quality(quality_cmd) => dispatch_quality(quality_cmd),
        Commands::Fed(fed_cmd) => dispatch_fed(fed_cmd),
        #[cfg(feature = "doctest")]
        Commands::Doctest(doctest_cmd) => match doctest_cmd {
            DoctestCommands::Extract {
                input,
                output,
                source,
                version,
            } => cmd_doctest_extract(&input, &output, &source, &version),
            DoctestCommands::Merge { inputs, output } => cmd_doctest_merge(&inputs, &output),
        },
        #[cfg(feature = "repl")]
        Commands::Repl => crate::repl::run(),
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("Error: {}", e);
            ExitCode::FAILURE
        }
    }
}

fn dispatch_registry(cmd: RegistryCommands) -> crate::error::Result<()> {
    match cmd {
        RegistryCommands::Init { path } => registry::cmd_registry_init(&path),
        RegistryCommands::List { path } => registry::cmd_registry_list(&path),
        RegistryCommands::Push {
            input,
            name,
            version,
            description,
            license,
            tags,
            registry,
        } => registry::cmd_registry_push(
            &input,
            &name,
            &version,
            &description,
            &license,
            &tags,
            &registry,
        ),
        RegistryCommands::Pull {
            name,
            output,
            version,
            registry,
        } => registry::cmd_registry_pull(&name, &output, version.as_deref(), &registry),
        RegistryCommands::Search { query, path } => registry::cmd_registry_search(&query, &path),
        RegistryCommands::ShowInfo { name, path } => registry::cmd_registry_show_info(&name, &path),
        RegistryCommands::Delete {
            name,
            version,
            path,
        } => registry::cmd_registry_delete(&name, &version, &path),
    }
}

fn dispatch_drift(cmd: DriftCommands) -> crate::error::Result<()> {
    match cmd {
        DriftCommands::Detect {
            reference,
            current,
            tests,
            alpha,
            format,
        } => drift::cmd_drift_detect(&reference, &current, &tests, alpha, &format),
        DriftCommands::Report {
            reference,
            current,
            output,
        } => drift::cmd_drift_report(&reference, &current, output.as_ref()),
        DriftCommands::Sketch {
            input,
            output,
            sketch_type,
            source,
            format,
        } => drift::cmd_drift_sketch(&input, &output, &sketch_type, source.as_deref(), &format),
        DriftCommands::Merge {
            sketches,
            output,
            format,
        } => drift::cmd_drift_merge(&sketches, &output, &format),
        DriftCommands::Compare {
            reference,
            current,
            threshold,
            format,
        } => drift::cmd_drift_compare(&reference, &current, threshold, &format),
    }
}

fn dispatch_quality(cmd: QualityCommands) -> crate::error::Result<()> {
    match cmd {
        QualityCommands::Check {
            path,
            null_threshold,
            duplicate_threshold,
            detect_outliers,
            format,
        } => quality::cmd_quality_check(
            &path,
            null_threshold,
            duplicate_threshold,
            detect_outliers,
            &format,
        ),
        QualityCommands::Report { path, output } => {
            quality::cmd_quality_report(&path, output.as_deref())
        }
        QualityCommands::Score {
            path,
            profile,
            suggest,
            json,
            badge,
        } => quality::cmd_quality_score(&path, &profile, suggest, json, badge),
        QualityCommands::Profiles => quality::cmd_quality_profiles(),
    }
}

fn dispatch_fed(cmd: FedCommands) -> crate::error::Result<()> {
    match cmd {
        FedCommands::Manifest {
            input,
            output,
            node_id,
            train_ratio,
            seed,
            format,
        } => fed::cmd_fed_manifest(&input, &output, &node_id, train_ratio, seed, &format),
        FedCommands::Plan {
            manifests,
            output,
            strategy,
            train_ratio,
            seed,
            stratify_column,
            format,
        } => fed::cmd_fed_plan(
            &manifests,
            &output,
            &strategy,
            train_ratio,
            seed,
            stratify_column.as_deref(),
            &format,
        ),
        FedCommands::Split {
            input,
            plan,
            node_id,
            train_output,
            test_output,
            validation_output,
        } => fed::cmd_fed_split(
            &input,
            &plan,
            &node_id,
            &train_output,
            &test_output,
            validation_output.as_ref(),
        ),
        FedCommands::Verify { manifests, format } => fed::cmd_fed_verify(&manifests, &format),
    }
}

// =============================================================================
// Doctest Commands
// =============================================================================

#[cfg(feature = "doctest")]
fn cmd_doctest_extract(
    input: &std::path::Path,
    output: &std::path::Path,
    source: &str,
    version: &str,
) -> crate::Result<()> {
    use crate::DocTestParser;

    if !input.is_dir() {
        return Err(crate::Error::invalid_config(format!(
            "Input path must be a directory: {}",
            input.display()
        )));
    }

    let parser = DocTestParser::new();
    let corpus = parser.parse_directory(input, source, version)?;

    println!(
        "Extracted {} doctests from {} ({})",
        corpus.len(),
        source,
        version
    );

    if corpus.is_empty() {
        println!("Warning: No doctests found in {}", input.display());
        return Ok(());
    }

    let dataset = corpus.to_dataset()?;
    dataset.to_parquet(output)?;

    println!("Wrote {} to {}", corpus.len(), output.display());
    Ok(())
}

#[cfg(feature = "doctest")]
fn cmd_doctest_merge(inputs: &[PathBuf], output: &std::path::Path) -> crate::Result<()> {
    use crate::{dataset::Dataset, ArrowDataset};

    if inputs.is_empty() {
        return Err(crate::Error::invalid_config("No input files provided"));
    }

    // Load all datasets and concatenate
    let mut all_batches = Vec::new();
    let mut total_rows = 0;

    for input in inputs {
        let dataset = ArrowDataset::from_parquet(input)?;
        total_rows += dataset.len();
        for batch in dataset.iter() {
            all_batches.push(batch.clone());
        }
    }

    if all_batches.is_empty() {
        return Err(crate::Error::invalid_config("No data found in input files"));
    }

    // Create merged dataset
    let merged = ArrowDataset::new(all_batches)?;
    merged.to_parquet(output)?;

    println!(
        "Merged {} doctests from {} files to {}",
        total_rows,
        inputs.len(),
        output.display()
    );
    Ok(())
}
