use super::*;
use std::io::Write;
use tempfile::{tempdir, NamedTempFile};

// ========================================================================
// Path Validation Tests
// ========================================================================

#[test]
fn test_validate_path_not_found() {
    let result = validate_path(Path::new("/nonexistent/model.apr"));
    assert!(result.is_err());
    match result {
        Err(CliError::FileNotFound(_)) => {}
        _ => panic!("Expected FileNotFound error"),
    }
}

#[test]
fn test_validate_path_is_directory() {
    let dir = tempdir().expect("create dir");
    let result = validate_path(dir.path());
    assert!(result.is_err());
    match result {
        Err(CliError::NotAFile(_)) => {}
        _ => panic!("Expected NotAFile error"),
    }
}

#[test]
fn test_validate_path_valid() {
    let file = NamedTempFile::new().expect("create file");
    let result = validate_path(file.path());
    assert!(result.is_ok());
}

// ========================================================================
// Model Type Formatting Tests
// ========================================================================

#[test]
fn test_format_model_type_linear_regression() {
    assert_eq!(format_model_type(0x0001), "LinearRegression");
}

#[test]
fn test_format_model_type_logistic_regression() {
    assert_eq!(format_model_type(0x0002), "LogisticRegression");
}

#[test]
fn test_format_model_type_decision_tree() {
    assert_eq!(format_model_type(0x0003), "DecisionTree");
}

#[test]
fn test_format_model_type_random_forest() {
    assert_eq!(format_model_type(0x0004), "RandomForest");
}

#[test]
fn test_format_model_type_gradient_boosting() {
    assert_eq!(format_model_type(0x0005), "GradientBoosting");
}

#[test]
fn test_format_model_type_kmeans() {
    assert_eq!(format_model_type(0x0006), "KMeans");
}

#[test]
fn test_format_model_type_pca() {
    assert_eq!(format_model_type(0x0007), "PCA");
}

#[test]
fn test_format_model_type_naive_bayes() {
    assert_eq!(format_model_type(0x0008), "NaiveBayes");
}

#[test]
fn test_format_model_type_knn() {
    assert_eq!(format_model_type(0x0009), "KNN");
}

#[test]
fn test_format_model_type_svm() {
    assert_eq!(format_model_type(0x000A), "SVM");
}

#[test]
fn test_format_model_type_ngram_lm() {
    assert_eq!(format_model_type(0x0010), "NgramLM");
}

#[test]
fn test_format_model_type_tfidf() {
    assert_eq!(format_model_type(0x0011), "TfIdf");
}

#[test]
fn test_format_model_type_count_vectorizer() {
    assert_eq!(format_model_type(0x0012), "CountVectorizer");
}

#[test]
fn test_format_model_type_neural_sequential() {
    assert_eq!(format_model_type(0x0020), "NeuralSequential");
}

#[test]
fn test_format_model_type_neural_custom() {
    assert_eq!(format_model_type(0x0021), "NeuralCustom");
}

#[test]
fn test_format_model_type_content_recommender() {
    assert_eq!(format_model_type(0x0030), "ContentRecommender");
}

#[test]
fn test_format_model_type_mixture_of_experts() {
    assert_eq!(format_model_type(0x0040), "MixtureOfExperts");
}

#[test]
fn test_format_model_type_custom() {
    assert_eq!(format_model_type(0x00FF), "Custom");
}

#[test]
fn test_format_model_type_unknown() {
    assert_eq!(format_model_type(0xDEAD), "Unknown(0xDEAD)");
}

// ========================================================================
// Run Command Tests
// ========================================================================

#[test]
fn test_run_file_not_found() {
    let result = run(
        Path::new("/nonexistent/model.apr"),
        false,
        false,
        false,
        100,
        false,
    );
    assert!(result.is_err());
}

#[test]
fn test_run_with_small_file() {
    let mut file = NamedTempFile::with_suffix(".apr").expect("create file");
    // Write something smaller than header size
    file.write_all(b"short").expect("write");

    // Should fail because file is too small for header
    let result = run(file.path(), false, false, false, 100, false);
    assert!(result.is_err());
}

// ========================================================================
// Header Parsing Tests
// ========================================================================

#[test]
fn test_parse_header_invalid_magic() {
    let mut header = [0u8; HEADER_SIZE];
    header[0..4].copy_from_slice(b"XXXX");
    let info = parse_header(&header);
    assert!(!info.magic_valid);
    assert_eq!(info.magic_str, "XXXX");
}

#[test]
fn test_parse_header_valid_aprn_magic() {
    let mut header = [0u8; HEADER_SIZE];
    header[0..4].copy_from_slice(b"APRN");
    header[4] = 1; // version major
    header[5] = 0; // version minor
    let info = parse_header(&header);
    assert!(info.magic_valid);
    assert_eq!(info.magic_str, "APRN");
    assert_eq!(info.version, (1, 0));
}

#[test]
fn test_parse_header_valid_apr2_magic() {
    let mut header = [0u8; HEADER_SIZE];
    header[0..4].copy_from_slice(b"APR2");
    header[4] = 2; // version major
    header[5] = 0; // version minor
    let info = parse_header(&header);
    assert!(info.magic_valid);
    assert_eq!(info.magic_str, "APR2");
    assert_eq!(info.version, (2, 0));
}

#[test]
fn test_parse_header_flags() {
    let mut header = [0u8; HEADER_SIZE];
    header[0..4].copy_from_slice(b"APRN");
    header[21] = 0b00000111; // compressed, signed, encrypted
    let info = parse_header(&header);
    assert!(info.compressed);
    assert!(info.signed);
    assert!(info.encrypted);
}

#[test]
fn test_parse_header_model_type() {
    let mut header = [0u8; HEADER_SIZE];
    header[0..4].copy_from_slice(b"APRN");
    header[6] = 0x06; // KMeans (low byte)
    header[7] = 0x00; // (high byte)
    let info = parse_header(&header);
    assert_eq!(info.model_type, 0x0006);
}

// ========================================================================
// collect_flags Tests
// ========================================================================

#[test]
fn collect_flags_returns_empty_when_no_flags_set() {
    let info = HeaderInfo {
        magic_valid: true,
        magic_str: "APRN".to_string(),
        version: (1, 0),
        model_type: 0x0001,
        compressed: false,
        signed: false,
        encrypted: false,
    };
    let flags = collect_flags(&info);
    assert!(flags.is_empty());
}

#[test]
fn collect_flags_returns_compressed_only() {
    let info = HeaderInfo {
        magic_valid: true,
        magic_str: "APRN".to_string(),
        version: (1, 0),
        model_type: 0x0001,
        compressed: true,
        signed: false,
        encrypted: false,
    };
    let flags = collect_flags(&info);
    assert_eq!(flags, vec!["compressed"]);
}

#[test]
fn collect_flags_returns_signed_only() {
    let info = HeaderInfo {
        magic_valid: true,
        magic_str: "APRN".to_string(),
        version: (1, 0),
        model_type: 0x0001,
        compressed: false,
        signed: true,
        encrypted: false,
    };
    let flags = collect_flags(&info);
    assert_eq!(flags, vec!["signed"]);
}

#[test]
fn collect_flags_returns_encrypted_only() {
    let info = HeaderInfo {
        magic_valid: true,
        magic_str: "APRN".to_string(),
        version: (1, 0),
        model_type: 0x0001,
        compressed: false,
        signed: false,
        encrypted: true,
    };
    let flags = collect_flags(&info);
    assert_eq!(flags, vec!["encrypted"]);
}

#[test]
fn collect_flags_returns_all_three_in_order() {
    let info = HeaderInfo {
        magic_valid: true,
        magic_str: "APRN".to_string(),
        version: (1, 0),
        model_type: 0x0001,
        compressed: true,
        signed: true,
        encrypted: true,
    };
    let flags = collect_flags(&info);
    assert_eq!(flags, vec!["compressed", "signed", "encrypted"]);
}

#[test]
fn collect_flags_signed_and_encrypted_without_compressed() {
    let info = HeaderInfo {
        magic_valid: true,
        magic_str: "APRN".to_string(),
        version: (1, 0),
        model_type: 0x0001,
        compressed: false,
        signed: true,
        encrypted: true,
    };
    let flags = collect_flags(&info);
    assert_eq!(flags, vec!["signed", "encrypted"]);
}

// ========================================================================
// parse_header: compression via byte 20 (legacy path)
// ========================================================================

#[test]
fn parse_header_compression_via_legacy_byte20() {
    let mut header = [0u8; HEADER_SIZE];
    header[0..4].copy_from_slice(b"APRN");
    // Flag bits all zero, but legacy byte 20 is nonzero
    header[20] = 0x01;
    header[21] = 0x00;
    let info = parse_header(&header);
    assert!(info.compressed, "byte 20 nonzero should set compressed");
    assert!(!info.signed);
    assert!(!info.encrypted);
}

#[test]
fn parse_header_no_flags_no_legacy() {
    let mut header = [0u8; HEADER_SIZE];
    header[0..4].copy_from_slice(b"APRN");
    header[20] = 0x00;
    header[21] = 0x00;
    let info = parse_header(&header);
    assert!(!info.compressed);
    assert!(!info.signed);
    assert!(!info.encrypted);
}

// ========================================================================
// parse_header: model type little-endian encoding
// ========================================================================

#[test]
fn parse_header_model_type_high_byte_nonzero() {
    let mut header = [0u8; HEADER_SIZE];
    header[0..4].copy_from_slice(b"APRN");
    header[6] = 0xFF;
    header[7] = 0x00;
    let info = parse_header(&header);
    assert_eq!(info.model_type, 0x00FF);
    // Verify this maps to Custom
    assert_eq!(format_model_type(info.model_type), "Custom");
}

#[test]
fn parse_header_model_type_multibyte_le() {
    let mut header = [0u8; HEADER_SIZE];
    header[0..4].copy_from_slice(b"APRN");
    header[6] = 0x40; // low byte
    header[7] = 0x00; // high byte
    let info = parse_header(&header);
    assert_eq!(info.model_type, 0x0040);
    assert_eq!(format_model_type(info.model_type), "MixtureOfExperts");
}

// ========================================================================
// parse_header: version tuple extraction
// ========================================================================

#[test]
fn parse_header_version_zero_zero() {
    let mut header = [0u8; HEADER_SIZE];
    header[0..4].copy_from_slice(b"APRN");
    header[4] = 0;
    header[5] = 0;
    let info = parse_header(&header);
    assert_eq!(info.version, (0, 0));
}

#[test]
fn parse_header_version_high_minor() {
    let mut header = [0u8; HEADER_SIZE];
    header[0..4].copy_from_slice(b"APRN");
    header[4] = 2;
    header[5] = 99;
    let info = parse_header(&header);
    assert_eq!(info.version, (2, 99));
}

// ========================================================================
// parse_header: GGUF magic recognized
// ========================================================================

#[test]
fn parse_header_gguf_magic_is_valid() {
    let mut header = [0u8; HEADER_SIZE];
    header[0..4].copy_from_slice(b"GGUF");
    let info = parse_header(&header);
    assert!(info.magic_valid);
    assert_eq!(info.magic_str, "GGUF");
}

// ========================================================================
// parse_header: non-UTF8 magic bytes
// ========================================================================

#[test]
fn parse_header_non_utf8_magic_lossily_converted() {
    let mut header = [0u8; HEADER_SIZE];
    header[0..4].copy_from_slice(&[0xFF, 0xFE, 0xFD, 0xFC]);
    let info = parse_header(&header);
    assert!(!info.magic_valid);
    // from_utf8_lossy replaces invalid bytes with replacement char
    assert!(info.magic_str.contains('\u{FFFD}'));
}

// ========================================================================
// format_model_type: exhaustive unknown branch
// ========================================================================

#[test]
fn format_model_type_zero_is_unknown() {
    assert_eq!(format_model_type(0x0000), "Unknown(0x0000)");
}

#[test]
fn format_model_type_max_u16_is_unknown() {
    assert_eq!(format_model_type(0xFFFF), "Unknown(0xFFFF)");
}

// ========================================================================
// run: hex and strings modes (file-backed, no model needed)
// ========================================================================

#[test]
fn run_hex_mode_succeeds_on_regular_file() {
    let mut file = NamedTempFile::new().expect("create file");
    file.write_all(b"Hello, hex world! 0123456789ABCDEF")
        .expect("write");
    let result = run(file.path(), false, true, false, 256, false);
    assert!(result.is_ok());
}

include!("debug_tests_part_02.rs");
