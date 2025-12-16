//! Diff command implementation
//!
//! Toyota Way: Kaizen - Continuous improvement through comparison.
//! Compare two APR models to identify differences in structure, metadata, and weights.

use crate::error::CliError;
use crate::output;
use aprender::format::{self, HEADER_SIZE};
use colored::Colorize;
use serde::Serialize;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

/// Diff result for JSON output
#[derive(Serialize)]
struct DiffResult {
    file1: String,
    file2: String,
    identical: bool,
    differences: Vec<DiffEntry>,
}

#[derive(Serialize)]
struct DiffEntry {
    field: String,
    file1_value: String,
    file2_value: String,
}

/// Model info extracted from header
struct ModelInfo {
    magic_valid: bool,
    version: (u8, u8),
    model_type: u16,
    metadata_size: u32,
    payload_size: u32,
    compression: u8,
    flags: u8,
    metadata: Option<format::Metadata>,
}

/// Run the diff command
pub(crate) fn run(
    path1: &Path,
    path2: &Path,
    show_weights: bool,
    json_output: bool,
) -> Result<(), CliError> {
    validate_paths(path1, path2)?;

    let info1 = read_model_info(path1)?;
    let info2 = read_model_info(path2)?;

    let differences = compute_differences(&info1, &info2);
    let identical = differences.is_empty();

    if json_output {
        output_json(path1, path2, identical, differences);
    } else {
        output_text(path1, path2, identical, &differences, show_weights);
    }

    Ok(())
}

/// Validate both paths exist and are files
fn validate_paths(path1: &Path, path2: &Path) -> Result<(), CliError> {
    for path in [path1, path2] {
        if !path.exists() {
            return Err(CliError::FileNotFound(path.to_path_buf()));
        }
        if !path.is_file() {
            return Err(CliError::NotAFile(path.to_path_buf()));
        }
    }
    Ok(())
}

/// Compute all differences between two models
fn compute_differences(info1: &ModelInfo, info2: &ModelInfo) -> Vec<DiffEntry> {
    let mut differences = Vec::new();

    if info1.magic_valid != info2.magic_valid {
        differences.push(DiffEntry {
            field: "magic_valid".to_string(),
            file1_value: info1.magic_valid.to_string(),
            file2_value: info2.magic_valid.to_string(),
        });
    }

    if info1.version != info2.version {
        let (v1_maj, v1_min) = info1.version;
        let (v2_maj, v2_min) = info2.version;
        differences.push(DiffEntry {
            field: "version".to_string(),
            file1_value: format!("{v1_maj}.{v1_min}"),
            file2_value: format!("{v2_maj}.{v2_min}"),
        });
    }

    if info1.model_type != info2.model_type {
        let t1 = info1.model_type;
        let t2 = info2.model_type;
        differences.push(DiffEntry {
            field: "model_type".to_string(),
            file1_value: format!("0x{t1:04X}"),
            file2_value: format!("0x{t2:04X}"),
        });
    }

    if info1.compression != info2.compression {
        differences.push(DiffEntry {
            field: "compression".to_string(),
            file1_value: format_compression(info1.compression),
            file2_value: format_compression(info2.compression),
        });
    }

    if info1.flags != info2.flags {
        let f1 = info1.flags;
        let f2 = info2.flags;
        differences.push(DiffEntry {
            field: "flags".to_string(),
            file1_value: format!("0x{f1:02X}"),
            file2_value: format!("0x{f2:02X}"),
        });
    }

    if info1.metadata_size != info2.metadata_size {
        differences.push(DiffEntry {
            field: "metadata_size".to_string(),
            file1_value: info1.metadata_size.to_string(),
            file2_value: info2.metadata_size.to_string(),
        });
    }

    if info1.payload_size != info2.payload_size {
        differences.push(DiffEntry {
            field: "payload_size".to_string(),
            file1_value: info1.payload_size.to_string(),
            file2_value: info2.payload_size.to_string(),
        });
    }

    compare_metadata(
        info1.metadata.as_ref(),
        info2.metadata.as_ref(),
        &mut differences,
    );
    differences
}

/// Output results as JSON
fn output_json(path1: &Path, path2: &Path, identical: bool, differences: Vec<DiffEntry>) {
    let result = DiffResult {
        file1: path1.display().to_string(),
        file2: path2.display().to_string(),
        identical,
        differences,
    };
    if let Ok(json) = serde_json::to_string_pretty(&result) {
        println!("{json}");
    }
}

/// Output results as text
fn output_text(
    path1: &Path,
    path2: &Path,
    identical: bool,
    differences: &[DiffEntry],
    show_weights: bool,
) {
    println!(
        "Comparing {} vs {}",
        path1.display().to_string().cyan(),
        path2.display().to_string().cyan()
    );
    println!();

    if identical {
        println!(
            "{}",
            "Models are IDENTICAL in structure and metadata"
                .green()
                .bold()
        );
    } else {
        let count = differences.len();
        println!("{} {count} differences found:", "DIFF:".yellow().bold());
        println!();

        for diff in differences {
            println!(
                "  {}: {} → {}",
                diff.field.white().bold(),
                diff.file1_value.red(),
                diff.file2_value.green()
            );
        }
    }

    if show_weights {
        println!();
        println!("{}", "Weight comparison: (not yet implemented)".yellow());
    }
}

/// Read model info from file
fn read_model_info(path: &Path) -> Result<ModelInfo, CliError> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    let mut header_bytes = [0u8; HEADER_SIZE];
    reader
        .read_exact(&mut header_bytes)
        .map_err(|_| CliError::InvalidFormat("File too small".to_string()))?;

    let magic_valid = output::is_valid_magic(&header_bytes[0..4]);
    let version = (header_bytes[4], header_bytes[5]);
    let model_type = u16::from_le_bytes([header_bytes[6], header_bytes[7]]);
    let metadata_size = u32::from_le_bytes([
        header_bytes[8],
        header_bytes[9],
        header_bytes[10],
        header_bytes[11],
    ]);
    let payload_size = u32::from_le_bytes([
        header_bytes[12],
        header_bytes[13],
        header_bytes[14],
        header_bytes[15],
    ]);
    let compression = header_bytes[20];
    let flags = header_bytes[21];

    let metadata = if metadata_size > 0 {
        let mut metadata_bytes = vec![0u8; metadata_size as usize];
        if reader.read_exact(&mut metadata_bytes).is_ok() {
            rmp_serde::from_slice::<format::Metadata>(&metadata_bytes).ok()
        } else {
            None
        }
    } else {
        None
    };

    Ok(ModelInfo {
        magic_valid,
        version,
        model_type,
        metadata_size,
        payload_size,
        compression,
        flags,
        metadata,
    })
}

/// Compare metadata fields
fn compare_metadata(
    meta1: Option<&format::Metadata>,
    meta2: Option<&format::Metadata>,
    differences: &mut Vec<DiffEntry>,
) {
    match (meta1, meta2) {
        (Some(m1), Some(m2)) => {
            if m1.model_name != m2.model_name {
                differences.push(DiffEntry {
                    field: "model_name".to_string(),
                    file1_value: m1
                        .model_name
                        .clone()
                        .unwrap_or_else(|| "(none)".to_string()),
                    file2_value: m2
                        .model_name
                        .clone()
                        .unwrap_or_else(|| "(none)".to_string()),
                });
            }

            if m1.description != m2.description {
                differences.push(DiffEntry {
                    field: "description".to_string(),
                    file1_value: m1
                        .description
                        .clone()
                        .unwrap_or_else(|| "(none)".to_string()),
                    file2_value: m2
                        .description
                        .clone()
                        .unwrap_or_else(|| "(none)".to_string()),
                });
            }

            if m1.aprender_version != m2.aprender_version {
                differences.push(DiffEntry {
                    field: "aprender_version".to_string(),
                    file1_value: m1.aprender_version.clone(),
                    file2_value: m2.aprender_version.clone(),
                });
            }
        }
        (Some(_), None) => {
            differences.push(DiffEntry {
                field: "metadata".to_string(),
                file1_value: "present".to_string(),
                file2_value: "missing".to_string(),
            });
        }
        (None, Some(_)) => {
            differences.push(DiffEntry {
                field: "metadata".to_string(),
                file1_value: "missing".to_string(),
                file2_value: "present".to_string(),
            });
        }
        (None, None) => {}
    }
}

/// Format compression type as string
fn format_compression(compression: u8) -> String {
    match compression {
        0 => "none".to_string(),
        1 => "zstd".to_string(),
        2 => "lz4".to_string(),
        3 => "snappy".to_string(),
        _ => format!("unknown({compression})"),
    }
}
