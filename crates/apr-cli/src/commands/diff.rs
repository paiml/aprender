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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::{tempdir, NamedTempFile};

    // ========================================================================
    // Path Validation Tests
    // ========================================================================

    #[test]
    fn test_validate_paths_first_not_found() {
        let file2 = NamedTempFile::new().expect("create file");
        let result = validate_paths(
            Path::new("/nonexistent/model1.apr"),
            file2.path(),
        );
        assert!(result.is_err());
        match result {
            Err(CliError::FileNotFound(_)) => {}
            _ => panic!("Expected FileNotFound error"),
        }
    }

    #[test]
    fn test_validate_paths_second_not_found() {
        let file1 = NamedTempFile::new().expect("create file");
        let result = validate_paths(
            file1.path(),
            Path::new("/nonexistent/model2.apr"),
        );
        assert!(result.is_err());
        match result {
            Err(CliError::FileNotFound(_)) => {}
            _ => panic!("Expected FileNotFound error"),
        }
    }

    #[test]
    fn test_validate_paths_first_is_directory() {
        let dir = tempdir().expect("create dir");
        let file2 = NamedTempFile::new().expect("create file");
        let result = validate_paths(dir.path(), file2.path());
        assert!(result.is_err());
        match result {
            Err(CliError::NotAFile(_)) => {}
            _ => panic!("Expected NotAFile error"),
        }
    }

    #[test]
    fn test_validate_paths_valid() {
        let file1 = NamedTempFile::new().expect("create file");
        let file2 = NamedTempFile::new().expect("create file");
        let result = validate_paths(file1.path(), file2.path());
        assert!(result.is_ok());
    }

    // ========================================================================
    // Run Command Tests
    // ========================================================================

    #[test]
    fn test_run_file_not_found() {
        let file = NamedTempFile::new().expect("create file");
        let result = run(
            Path::new("/nonexistent/model.apr"),
            file.path(),
            false,
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_invalid_files() {
        let mut file1 = NamedTempFile::with_suffix(".apr").expect("create file");
        let mut file2 = NamedTempFile::with_suffix(".apr").expect("create file");

        // Write minimal data (less than header size)
        file1.write_all(b"short").expect("write");
        file2.write_all(b"short").expect("write");

        let result = run(file1.path(), file2.path(), false, false);
        // Should fail because files are too small
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_json_output() {
        let mut file1 = NamedTempFile::with_suffix(".apr").expect("create file");
        let mut file2 = NamedTempFile::with_suffix(".apr").expect("create file");

        file1.write_all(b"short").expect("write");
        file2.write_all(b"short").expect("write");

        let result = run(file1.path(), file2.path(), false, true);
        // Should fail but tests json output path
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_show_weights() {
        let mut file1 = NamedTempFile::with_suffix(".apr").expect("create file");
        let mut file2 = NamedTempFile::with_suffix(".apr").expect("create file");

        file1.write_all(b"short").expect("write");
        file2.write_all(b"short").expect("write");

        let result = run(file1.path(), file2.path(), true, false);
        // Should fail but tests show_weights path
        assert!(result.is_err());
    }

    // ========================================================================
    // Format Compression Tests
    // ========================================================================

    #[test]
    fn test_format_compression_none() {
        assert_eq!(format_compression(0), "none");
    }

    #[test]
    fn test_format_compression_zstd() {
        assert_eq!(format_compression(1), "zstd");
    }

    #[test]
    fn test_format_compression_lz4() {
        assert_eq!(format_compression(2), "lz4");
    }

    #[test]
    fn test_format_compression_snappy() {
        assert_eq!(format_compression(3), "snappy");
    }

    #[test]
    fn test_format_compression_unknown() {
        assert_eq!(format_compression(255), "unknown(255)");
    }

    // ========================================================================
    // DiffEntry Tests
    // ========================================================================

    #[test]
    fn test_diff_entry_serialization() {
        let entry = DiffEntry {
            field: "version".to_string(),
            file1_value: "1.0".to_string(),
            file2_value: "2.0".to_string(),
        };
        let json = serde_json::to_string(&entry).expect("serialize");
        assert!(json.contains("version"));
        assert!(json.contains("1.0"));
        assert!(json.contains("2.0"));
    }

    #[test]
    fn test_diff_result_serialization() {
        let result = DiffResult {
            file1: "model1.apr".to_string(),
            file2: "model2.apr".to_string(),
            identical: false,
            differences: vec![DiffEntry {
                field: "payload_size".to_string(),
                file1_value: "1000".to_string(),
                file2_value: "2000".to_string(),
            }],
        };
        let json = serde_json::to_string(&result).expect("serialize");
        assert!(json.contains("model1.apr"));
        assert!(json.contains("model2.apr"));
        assert!(json.contains("payload_size"));
    }

    #[test]
    fn test_diff_result_identical() {
        let result = DiffResult {
            file1: "a.apr".to_string(),
            file2: "b.apr".to_string(),
            identical: true,
            differences: vec![],
        };
        let json = serde_json::to_string(&result).expect("serialize");
        assert!(json.contains("\"identical\":true"));
    }

    // ========================================================================
    // Output Functions Tests
    // ========================================================================

    #[test]
    fn test_output_json() {
        let differences = vec![DiffEntry {
            field: "test".to_string(),
            file1_value: "a".to_string(),
            file2_value: "b".to_string(),
        }];
        // Should not panic
        output_json(Path::new("a.apr"), Path::new("b.apr"), false, differences);
    }

    #[test]
    fn test_output_json_identical() {
        output_json(Path::new("a.apr"), Path::new("b.apr"), true, vec![]);
    }

    #[test]
    fn test_output_text_identical() {
        output_text(Path::new("a.apr"), Path::new("b.apr"), true, &[], false);
    }

    #[test]
    fn test_output_text_with_differences() {
        let differences = vec![
            DiffEntry {
                field: "version".to_string(),
                file1_value: "1.0".to_string(),
                file2_value: "2.0".to_string(),
            },
            DiffEntry {
                field: "compression".to_string(),
                file1_value: "none".to_string(),
                file2_value: "zstd".to_string(),
            },
        ];
        output_text(Path::new("a.apr"), Path::new("b.apr"), false, &differences, false);
    }

    #[test]
    fn test_output_text_with_show_weights() {
        output_text(Path::new("a.apr"), Path::new("b.apr"), true, &[], true);
    }

    // ========================================================================
    // Compute Differences Tests
    // ========================================================================

    #[test]
    fn test_compute_differences_identical() {
        let info = ModelInfo {
            magic_valid: true,
            version: (1, 0),
            model_type: 1,
            metadata_size: 100,
            payload_size: 1000,
            compression: 0,
            flags: 0,
            metadata: None,
        };

        let differences = compute_differences(&info, &info);
        assert!(differences.is_empty());
    }

    #[test]
    fn test_compute_differences_version() {
        let info1 = ModelInfo {
            magic_valid: true,
            version: (1, 0),
            model_type: 1,
            metadata_size: 100,
            payload_size: 1000,
            compression: 0,
            flags: 0,
            metadata: None,
        };

        let info2 = ModelInfo {
            magic_valid: true,
            version: (2, 0),
            model_type: 1,
            metadata_size: 100,
            payload_size: 1000,
            compression: 0,
            flags: 0,
            metadata: None,
        };

        let differences = compute_differences(&info1, &info2);
        assert_eq!(differences.len(), 1);
        assert_eq!(differences[0].field, "version");
    }

    #[test]
    fn test_compute_differences_compression() {
        let info1 = ModelInfo {
            magic_valid: true,
            version: (1, 0),
            model_type: 1,
            metadata_size: 100,
            payload_size: 1000,
            compression: 0,
            flags: 0,
            metadata: None,
        };

        let info2 = ModelInfo {
            magic_valid: true,
            version: (1, 0),
            model_type: 1,
            metadata_size: 100,
            payload_size: 1000,
            compression: 1,
            flags: 0,
            metadata: None,
        };

        let differences = compute_differences(&info1, &info2);
        assert_eq!(differences.len(), 1);
        assert_eq!(differences[0].field, "compression");
    }

    #[test]
    fn test_compute_differences_multiple() {
        let info1 = ModelInfo {
            magic_valid: true,
            version: (1, 0),
            model_type: 1,
            metadata_size: 100,
            payload_size: 1000,
            compression: 0,
            flags: 0,
            metadata: None,
        };

        let info2 = ModelInfo {
            magic_valid: false,
            version: (2, 0),
            model_type: 2,
            metadata_size: 100,
            payload_size: 1000,
            compression: 1,
            flags: 0,
            metadata: None,
        };

        let differences = compute_differences(&info1, &info2);
        assert!(differences.len() >= 4);
    }

    // ========================================================================
    // Metadata Comparison Tests
    // ========================================================================

    #[test]
    fn test_compare_metadata_both_none() {
        let mut differences = Vec::new();
        compare_metadata(None, None, &mut differences);
        assert!(differences.is_empty());
    }

    #[test]
    fn test_compare_metadata_first_present() {
        let meta = format::Metadata::default();
        let mut differences = Vec::new();
        compare_metadata(Some(&meta), None, &mut differences);
        assert_eq!(differences.len(), 1);
        assert_eq!(differences[0].field, "metadata");
    }

    #[test]
    fn test_compare_metadata_second_present() {
        let meta = format::Metadata::default();
        let mut differences = Vec::new();
        compare_metadata(None, Some(&meta), &mut differences);
        assert_eq!(differences.len(), 1);
        assert_eq!(differences[0].field, "metadata");
    }

    #[test]
    fn test_compare_metadata_both_present_same() {
        let meta = format::Metadata::default();
        let mut differences = Vec::new();
        compare_metadata(Some(&meta), Some(&meta), &mut differences);
        // Default metadata should be the same
        assert!(differences.is_empty());
    }
}
