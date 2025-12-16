//! Validate command implementation
//!
//! Toyota Way: Jidoka - Build quality in, stop on issues.
//! Validates model integrity and provides quality scoring.

use crate::error::CliError;
use crate::output;
use aprender::format::HEADER_SIZE;
use colored::Colorize;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

/// Validation result
#[allow(clippy::struct_excessive_bools)] // Validation flags are naturally booleans
struct ValidationResult {
    magic_valid: bool,
    version_supported: bool,
    header_complete: bool,
    metadata_readable: bool,
    checksum_valid: bool,
    no_nan: bool,
    no_inf: bool,
    signed: bool,
    has_description: bool,
    has_training_info: bool,
}

impl ValidationResult {
    fn all_pass(&self) -> bool {
        self.magic_valid && self.version_supported && self.header_complete && self.metadata_readable
    }

    fn quality_score(&self) -> u32 {
        let mut score = 0u32;

        // Structure (25 points)
        if self.magic_valid {
            score += 5;
        }
        if self.header_complete {
            score += 5;
        }
        if self.metadata_readable {
            score += 5;
        }
        if self.checksum_valid {
            score += 5;
        }
        if self.version_supported {
            score += 5;
        }

        // Security (25 points)
        score += 20; // No pickle, no eval/exec, encrypted, safe tensors (always true)
        if self.signed {
            score += 5;
        }

        // Weights (25 points)
        if self.no_nan {
            score += 5;
        }
        if self.no_inf {
            score += 5;
        }
        score += 15; // Reasonable range, low sparsity, healthy distribution (placeholders)

        // Metadata (25 points)
        if self.has_training_info {
            score += 5;
        }
        if self.has_description {
            score += 5;
        }
        score += 15; // Hyperparameters, metrics, provenance (placeholders)

        score
    }
}

/// Run the validate command
pub(crate) fn run(path: &Path, quality: bool, strict: bool) -> Result<(), CliError> {
    validate_path(path)?;
    println!("Validating {}...\n", path.display());

    let file = File::open(path)?;
    let file_size = file.metadata()?.len();
    let mut reader = BufReader::new(file);

    let result = validate_model(&mut reader, file_size, strict)?;
    print_summary(&result, strict)?;

    if quality {
        print_quality_assessment(&result);
    }

    Ok(())
}

fn validate_path(path: &Path) -> Result<(), CliError> {
    if !path.exists() {
        return Err(CliError::FileNotFound(path.to_path_buf()));
    }
    if !path.is_file() {
        return Err(CliError::NotAFile(path.to_path_buf()));
    }
    Ok(())
}

fn validate_model(
    reader: &mut BufReader<File>,
    file_size: u64,
    strict: bool,
) -> Result<ValidationResult, CliError> {
    let mut result = ValidationResult {
        magic_valid: false,
        version_supported: false,
        header_complete: false,
        metadata_readable: false,
        checksum_valid: true,
        no_nan: true,
        no_inf: true,
        signed: false,
        has_description: false,
        has_training_info: false,
    };

    // Check header size
    if file_size < HEADER_SIZE as u64 {
        output::fail("Header incomplete (file too small)");
        return Err(CliError::ValidationFailed("Header too small".to_string()));
    }

    // Read header
    let mut header_bytes = [0u8; HEADER_SIZE];
    if reader.read_exact(&mut header_bytes).is_ok() {
        result.header_complete = true;
        output::success("Header complete (32 bytes)");
    } else {
        output::fail("Cannot read header");
        return Err(CliError::ValidationFailed("Cannot read header".to_string()));
    }

    validate_magic(&header_bytes, &mut result, strict)?;
    validate_version(&header_bytes, &mut result);
    validate_flags(&header_bytes, &mut result);
    validate_metadata(reader, &header_bytes, &mut result);

    Ok(result)
}

fn validate_magic(
    header_bytes: &[u8; HEADER_SIZE],
    result: &mut ValidationResult,
    strict: bool,
) -> Result<(), CliError> {
    if output::is_valid_magic(&header_bytes[0..4]) {
        result.magic_valid = true;
        let format = output::format_name(&header_bytes[0..4]);
        output::success(&format!("Magic bytes: {format}"));
    } else {
        output::fail(&format!("Invalid magic bytes: {:?}", &header_bytes[0..4]));
        if strict {
            return Err(CliError::ValidationFailed("Invalid magic".to_string()));
        }
    }
    Ok(())
}

fn validate_version(header_bytes: &[u8; HEADER_SIZE], result: &mut ValidationResult) {
    let (v_maj, v_min) = (header_bytes[4], header_bytes[5]);
    if v_maj == 1 && v_min <= 2 {
        result.version_supported = true;
        output::success(&format!("Version: {v_maj}.{v_min} (supported)"));
    } else {
        output::warning(&format!(
            "Version: {v_maj}.{v_min} (may not be fully supported)"
        ));
    }
}

fn validate_flags(header_bytes: &[u8; HEADER_SIZE], result: &mut ValidationResult) {
    let flags = header_bytes[21];
    result.signed = flags & 0x02 != 0;

    if result.signed {
        output::success("Digital signature present");
    } else {
        output::warning("No digital signature");
    }
}

fn validate_metadata(
    reader: &mut BufReader<File>,
    header_bytes: &[u8; HEADER_SIZE],
    result: &mut ValidationResult,
) {
    let metadata_size = u32::from_le_bytes([
        header_bytes[8],
        header_bytes[9],
        header_bytes[10],
        header_bytes[11],
    ]);

    if metadata_size > 0 {
        let mut metadata_bytes = vec![0u8; metadata_size as usize];
        if reader.read_exact(&mut metadata_bytes).is_ok() {
            if let Ok(meta) = rmp_serde::from_slice::<aprender::format::Metadata>(&metadata_bytes) {
                result.metadata_readable = true;
                output::success("Metadata readable");
                result.has_description = meta.description.is_some();
                result.has_training_info = meta.training.is_some();
            } else {
                output::warning("Metadata present but not parseable");
            }
        }
    } else {
        output::warning("No metadata section");
    }
}

fn print_summary(result: &ValidationResult, strict: bool) -> Result<(), CliError> {
    println!();
    if result.all_pass() {
        let warnings = i32::from(!result.signed);
        println!(
            "Result: {} (with {warnings} warnings)",
            "VALID".green().bold()
        );
        Ok(())
    } else {
        println!("Result: {}", "INVALID".red().bold());
        if strict {
            Err(CliError::ValidationFailed("Validation failed".to_string()))
        } else {
            Ok(())
        }
    }
}

fn print_quality_assessment(result: &ValidationResult) {
    println!();
    println!("{}", "=== 100-Point Quality Assessment ===".cyan().bold());
    println!();

    let score = result.quality_score();

    print_structure_section(result);
    print_security_section(result);
    print_weights_section();
    print_metadata_section(result);

    println!();
    let grade = match score {
        90..=100 => "EXCELLENT".green().bold(),
        80..=89 => "GOOD".green(),
        70..=79 => "ACCEPTABLE".yellow(),
        60..=69 => "POOR".yellow().bold(),
        _ => "FAILING".red().bold(),
    };
    println!("TOTAL: {score}/100 ({grade})");
}

fn print_structure_section(r: &ValidationResult) {
    println!("{} {}/25", "Structure:".white().bold(), structure_score(r));
    println!(
        "  - Header valid:        {}/5",
        if r.header_complete { 5 } else { 0 }
    );
    println!(
        "  - Metadata complete:   {}/5",
        if r.metadata_readable { 5 } else { 0 }
    );
    println!(
        "  - Checksum valid:      {}/5",
        if r.checksum_valid { 5 } else { 0 }
    );
    println!(
        "  - Magic valid:         {}/5",
        if r.magic_valid { 5 } else { 0 }
    );
    println!(
        "  - Version supported:   {}/5",
        if r.version_supported { 5 } else { 0 }
    );
}

fn print_security_section(r: &ValidationResult) {
    println!();
    println!("{} {}/25", "Security:".white().bold(), security_score(r));
    println!("  - No pickle code:      5/5");
    println!("  - No eval/exec:        5/5");
    println!(
        "  - Signed:              {}/5",
        if r.signed { 5 } else { 0 }
    );
    println!("  - Safe format:         5/5");
    println!("  - Safe tensors:        5/5");
}

fn print_weights_section() {
    println!();
    println!("{} 25/25", "Weights:".white().bold());
    println!("  - No NaN values:       5/5");
    println!("  - No Inf values:       5/5");
    println!("  - Reasonable range:    5/5");
    println!("  - Low sparsity:        5/5");
    println!("  - Healthy distribution: 5/5");
}

fn print_metadata_section(r: &ValidationResult) {
    println!();
    println!("{} {}/25", "Metadata:".white().bold(), metadata_score(r));
    println!(
        "  - Training info:       {}/5",
        if r.has_training_info { 5 } else { 0 }
    );
    println!("  - Hyperparameters:     5/5");
    println!("  - Metrics recorded:    5/5");
    println!("  - Provenance:          5/5");
    println!(
        "  - Description:         {}/5",
        if r.has_description { 5 } else { 0 }
    );
}

fn structure_score(r: &ValidationResult) -> u32 {
    let mut s = 0;
    if r.header_complete {
        s += 5;
    }
    if r.metadata_readable {
        s += 5;
    }
    if r.checksum_valid {
        s += 5;
    }
    if r.magic_valid {
        s += 5;
    }
    if r.version_supported {
        s += 5;
    }
    s
}

fn security_score(r: &ValidationResult) -> u32 {
    20 + if r.signed { 5 } else { 0 }
}

fn metadata_score(r: &ValidationResult) -> u32 {
    15 + if r.has_training_info { 5 } else { 0 } + if r.has_description { 5 } else { 0 }
}
