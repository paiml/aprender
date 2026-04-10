//! Update README.md certification table from models.csv.
//!
//! Usage: apr-qa-readme-sync [--csv PATH] [--readme PATH]

#![forbid(unsafe_code)]

use aprender_qa_certify::{
    generate_summary, generate_table, parse_csv, update_readme, CertifyError, END_MARKER,
    START_MARKER,
};
use chrono::Utc;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

fn find_project_root() -> Option<PathBuf> {
    let mut current = env::current_dir().ok()?;
    loop {
        if current.join("Cargo.toml").exists() && current.join("crates").exists() {
            return Some(current);
        }
        if !current.pop() {
            return None;
        }
    }
}

/// Parsed CLI arguments
struct CliArgs {
    csv_path: Option<PathBuf>,
    readme_path: Option<PathBuf>,
    show_help: bool,
}

fn parse_args() -> CliArgs {
    let args: Vec<String> = env::args().collect();
    let mut result = CliArgs {
        csv_path: None,
        readme_path: None,
        show_help: false,
    };
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--csv" if i + 1 < args.len() => {
                i += 1;
                result.csv_path = Some(PathBuf::from(&args[i]));
            }
            "--readme" if i + 1 < args.len() => {
                i += 1;
                result.readme_path = Some(PathBuf::from(&args[i]));
            }
            "--help" | "-h" => {
                result.show_help = true;
            }
            _ => {}
        }
        i += 1;
    }
    result
}

fn print_help() {
    eprintln!("Usage: apr-qa-readme-sync [--csv PATH] [--readme PATH]");
    eprintln!();
    eprintln!("Updates README.md certification table from models.csv");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --csv PATH     Path to models.csv (default: docs/certifications/models.csv)");
    eprintln!("  --readme PATH  Path to README.md (default: README.md)");
    eprintln!("  --help, -h     Show this help");
}

fn validate_readme_markers(content: &str) -> Result<(), CertifyError> {
    if !content.contains(START_MARKER) {
        return Err(CertifyError::MarkerNotFound(format!(
            "README is missing start marker: {START_MARKER}"
        )));
    }
    if !content.contains(END_MARKER) {
        return Err(CertifyError::MarkerNotFound(format!(
            "README is missing end marker: {END_MARKER}"
        )));
    }
    Ok(())
}

fn run() -> Result<(), CertifyError> {
    let args = parse_args();

    if args.show_help {
        print_help();
        return Ok(());
    }

    // Find project root
    let root = find_project_root().ok_or_else(|| {
        CertifyError::MarkerNotFound(
            "Could not find project root (looking for Cargo.toml + crates/)".to_string(),
        )
    })?;

    let csv_path = args
        .csv_path
        .unwrap_or_else(|| root.join("docs/certifications/models.csv"));
    let readme_path = args.readme_path.unwrap_or_else(|| root.join("README.md"));

    // Read CSV
    eprintln!("Reading CSV from: {}", csv_path.display());
    let csv_content = fs::read_to_string(&csv_path)?;
    let models = parse_csv(&csv_content)?;
    eprintln!("Loaded {} models", models.len());

    // Generate content
    let timestamp = Utc::now().format("%Y-%m-%d %H:%M UTC").to_string();
    let summary = generate_summary(&models, &timestamp);
    let table = generate_table(&models);
    let full_content = format!("{summary}\n\n{table}");

    // Read and update README
    eprintln!("Reading README from: {}", readme_path.display());
    let readme_content = fs::read_to_string(&readme_path)?;
    validate_readme_markers(&readme_content)?;

    let updated_readme = update_readme(&readme_content, &full_content)?;

    // Write updated README
    fs::write(&readme_path, updated_readme)?;
    eprintln!("Updated {}", readme_path.display());
    eprintln!("Done. Commit both README.md and docs/certifications/models.csv together.");

    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("Error: {e}");
            ExitCode::FAILURE
        }
    }
}
