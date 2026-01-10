//! Pull command implementation
//!
//! Downloads GGUF models directly from HuggingFace without conversion.
//! For LLM inference via realizar, models must remain in GGUF format.
//!
//! Usage:
//!   apr pull bartowski/Qwen2.5-Coder-32B-Instruct-GGUF --quant Q4_K_M
//!   apr pull TheBloke/Llama-2-7B-GGUF --quant Q4_K_M -o ./models/

use crate::error::{CliError, Result};
use colored::Colorize;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

/// Quantization variants for GGUF files
#[derive(Debug, Clone, Copy)]
pub enum GgufQuant {
    Q4K,
    Q4KM,
    Q4KS,
    Q5K,
    Q5KM,
    Q5KS,
    Q6K,
    Q8_0,
    F16,
    F32,
}

impl GgufQuant {
    /// Parse quantization string (case-insensitive, with/without underscore)
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_uppercase().replace('-', "_").as_str() {
            "Q4_K" | "Q4K" => Some(Self::Q4K),
            "Q4_K_M" | "Q4KM" => Some(Self::Q4KM),
            "Q4_K_S" | "Q4KS" => Some(Self::Q4KS),
            "Q5_K" | "Q5K" => Some(Self::Q5K),
            "Q5_K_M" | "Q5KM" => Some(Self::Q5KM),
            "Q5_K_S" | "Q5KS" => Some(Self::Q5KS),
            "Q6_K" | "Q6K" => Some(Self::Q6K),
            "Q8_0" | "Q80" => Some(Self::Q8_0),
            "F16" | "FP16" => Some(Self::F16),
            "F32" | "FP32" => Some(Self::F32),
            _ => None,
        }
    }

    /// Get the filename suffix pattern for this quantization
    pub fn filename_pattern(self) -> &'static str {
        match self {
            Self::Q4K => "Q4_K",
            Self::Q4KM => "Q4_K_M",
            Self::Q4KS => "Q4_K_S",
            Self::Q5K => "Q5_K",
            Self::Q5KM => "Q5_K_M",
            Self::Q5KS => "Q5_K_S",
            Self::Q6K => "Q6_K",
            Self::Q8_0 => "Q8_0",
            Self::F16 => "F16",
            Self::F32 => "F32",
        }
    }
}

/// Run the pull command
pub fn run(repo: &str, quant: Option<&str>, output: Option<&Path>, force: bool) -> Result<()> {
    println!("{}", "=== APR Pull (GGUF Download) ===".cyan().bold());
    println!();

    // Parse repository (org/repo format)
    let (org, repo_name) = parse_repo(repo)?;
    println!("Repository: {}/{}", org.cyan(), repo_name.cyan());

    // Parse quantization
    let quant_type = match quant {
        Some(q) => GgufQuant::parse(q).ok_or_else(|| {
            CliError::ValidationFailed(format!(
                "Unknown quantization: {q}. Supported: Q4_K_M, Q4_K_S, Q5_K_M, Q6_K, Q8_0, F16"
            ))
        })?,
        None => GgufQuant::Q4KM, // Default to Q4_K_M
    };
    println!("Quantization: {}", quant_type.filename_pattern().yellow());

    // Determine output path
    let output_dir = output.map_or_else(|| PathBuf::from("./models"), Path::to_path_buf);

    // Create output directory
    if !output_dir.exists() {
        fs::create_dir_all(&output_dir).map_err(|e| {
            CliError::ValidationFailed(format!("Failed to create output directory: {e}"))
        })?;
    }

    // Construct HuggingFace URL
    let filename = format!(
        "{}-{}.gguf",
        repo_name.to_lowercase().replace(' ', "-"),
        quant_type.filename_pattern()
    );
    let output_path = output_dir.join(&filename);

    if output_path.exists() && !force {
        println!();
        println!(
            "{} {} already exists. Use --force to overwrite.",
            "⚠".yellow(),
            output_path.display()
        );
        return Ok(());
    }

    // HuggingFace download URL pattern
    let hf_url = format!("https://huggingface.co/{org}/{repo_name}/resolve/main/{filename}");

    println!("URL: {}", hf_url.dimmed());
    println!("Output: {}", output_path.display());
    println!();

    // Download with progress
    println!("{}", "Downloading...".yellow());
    download_file(&hf_url, &output_path)?;

    println!();
    println!(
        "{} Downloaded to {}",
        "✓".green().bold(),
        output_path.display()
    );

    // Verify GGUF magic
    verify_gguf(&output_path)?;
    println!("{} GGUF format verified", "✓".green());

    println!();
    println!("{}", "Usage:".cyan().bold());
    println!("  apr serve {} --port 8080", output_path.display());
    println!("  apr chat {}", output_path.display());

    Ok(())
}

/// Parse "org/repo" format
fn parse_repo(repo: &str) -> Result<(String, String)> {
    // Handle hf:// prefix
    let repo = repo.strip_prefix("hf://").unwrap_or(repo);

    let parts: Vec<&str> = repo.split('/').collect();
    if parts.len() != 2 {
        return Err(CliError::ValidationFailed(format!(
            "Invalid repository format: {repo}. Expected: org/repo (e.g., bartowski/Qwen2.5-Coder-32B-Instruct-GGUF)"
        )));
    }
    Ok((parts[0].to_string(), parts[1].to_string()))
}

/// Download file with progress (uses ureq for simplicity)
fn download_file(url: &str, output: &Path) -> Result<()> {
    // Use curl for now (available on most systems)
    // TODO: Use hf-hub crate when available
    let status = std::process::Command::new("curl")
        .args([
            "-L", // Follow redirects
            "-#", // Progress bar
            "-o",
            output.to_str().unwrap_or("model.gguf"),
            url,
        ])
        .status()
        .map_err(|e| CliError::NetworkError(format!("Failed to run curl: {e}")))?;

    if !status.success() {
        return Err(CliError::NetworkError(format!(
            "Download failed. Check if the model exists at: {url}"
        )));
    }

    Ok(())
}

/// Verify file is valid GGUF format
fn verify_gguf(path: &Path) -> Result<()> {
    let mut file = fs::File::open(path)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to open file: {e}")))?;

    let mut magic = [0u8; 4];
    file.read_exact(&mut magic)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to read magic: {e}")))?;

    // GGUF magic: "GGUF" (0x46554747)
    if &magic != b"GGUF" {
        return Err(CliError::ValidationFailed(format!(
            "Invalid GGUF file: magic bytes {:?} != GGUF",
            magic
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_repo_valid() {
        let (org, repo) = parse_repo("bartowski/Qwen2.5-Coder-32B-Instruct-GGUF").unwrap();
        assert_eq!(org, "bartowski");
        assert_eq!(repo, "Qwen2.5-Coder-32B-Instruct-GGUF");
    }

    #[test]
    fn test_parse_repo_with_hf_prefix() {
        let (org, repo) = parse_repo("hf://TheBloke/Llama-2-7B-GGUF").unwrap();
        assert_eq!(org, "TheBloke");
        assert_eq!(repo, "Llama-2-7B-GGUF");
    }

    #[test]
    fn test_parse_repo_invalid() {
        assert!(parse_repo("invalid").is_err());
        assert!(parse_repo("too/many/parts").is_err());
    }

    #[test]
    fn test_quant_parse() {
        assert!(matches!(GgufQuant::parse("Q4_K_M"), Some(GgufQuant::Q4KM)));
        assert!(matches!(GgufQuant::parse("q4_k_m"), Some(GgufQuant::Q4KM)));
        assert!(matches!(GgufQuant::parse("Q4KM"), Some(GgufQuant::Q4KM)));
        assert!(matches!(GgufQuant::parse("Q6_K"), Some(GgufQuant::Q6K)));
        assert!(matches!(GgufQuant::parse("F16"), Some(GgufQuant::F16)));
        assert!(GgufQuant::parse("invalid").is_none());
    }

    #[test]
    fn test_quant_filename_pattern() {
        assert_eq!(GgufQuant::Q4KM.filename_pattern(), "Q4_K_M");
        assert_eq!(GgufQuant::Q6K.filename_pattern(), "Q6_K");
    }
}
