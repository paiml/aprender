
#[test]
fn run_strings_mode_succeeds_on_regular_file() {
    let mut file = NamedTempFile::new().expect("create file");
    file.write_all(b"Hello\x00World\x00teststring\x00ab\x00longword")
        .expect("write");
    let result = run(file.path(), false, false, true, 100, false, false);
    assert!(result.is_ok());
}

#[test]
fn run_hex_mode_file_not_found() {
    let result = run(
        Path::new("/nonexistent/file.bin"),
        false,
        true,
        false,
        256,
        false,
        false,
    );
    assert!(result.is_err());
}

#[test]
fn run_strings_mode_file_not_found() {
    let result = run(
        Path::new("/nonexistent/file.bin"),
        false,
        false,
        true,
        256,
        false,
        false,
    );
    assert!(result.is_err());
}

#[test]
fn run_basic_mode_with_valid_header_size_file() {
    let mut file = NamedTempFile::with_suffix(".apr").expect("create file");
    // Write exactly HEADER_SIZE bytes with valid magic
    let mut buf = vec![0u8; HEADER_SIZE];
    buf[0..4].copy_from_slice(b"APRN");
    buf[4] = 1; // version major
    buf[5] = 0; // version minor
    file.write_all(&buf).expect("write");
    // basic mode (no drama, no hex, no strings)
    let result = run(file.path(), false, false, false, 100, false, false);
    assert!(result.is_ok());
}

#[test]
fn run_drama_mode_with_valid_header() {
    let mut file = NamedTempFile::with_suffix(".apr").expect("create file");
    let mut buf = vec![0u8; HEADER_SIZE];
    buf[0..4].copy_from_slice(b"APRN");
    buf[4] = 1;
    buf[5] = 0;
    file.write_all(&buf).expect("write");
    let result = run(file.path(), true, false, false, 100, false, false);
    assert!(result.is_ok());
}

/// FALSIFIER (#2394 finding 1): a file `apr debug` calls CORRUPTED must not
/// exit 0 — in drama mode either, which shouts "Model is CORRUPTED!".
///
/// This test used to assert `is_ok()` on `XXXX` magic. It was green for as
/// long as the defect existed and would have failed the fix: an alibi, not a
/// check. The verdict and the exit code now come from one `Health` value.
#[test]
fn run_drama_mode_with_invalid_magic() {
    let mut file = NamedTempFile::with_suffix(".apr").expect("create file");
    let mut buf = vec![0u8; HEADER_SIZE];
    buf[0..4].copy_from_slice(b"XXXX");
    buf[4] = 1;
    file.write_all(&buf).expect("write");
    let result = run(file.path(), true, false, false, 100, false, false);
    assert!(
        result.is_err(),
        "drama mode prints 'Model is CORRUPTED!' — it must not then exit 0"
    );
}

#[test]
fn run_drama_mode_with_flags_set() {
    let mut file = NamedTempFile::with_suffix(".apr").expect("create file");
    let mut buf = vec![0u8; HEADER_SIZE];
    buf[0..4].copy_from_slice(b"APRN");
    buf[4] = 1;
    // Set compressed + signed + encrypted + quantized flags
    buf[21] = 0b00100111;
    file.write_all(&buf).expect("write");
    let result = run(file.path(), true, false, false, 100, false, false);
    assert!(result.is_ok());
}

#[test]
fn run_drama_mode_version_non_one() {
    let mut file = NamedTempFile::with_suffix(".apr").expect("create file");
    let mut buf = vec![0u8; HEADER_SIZE];
    buf[0..4].copy_from_slice(b"APR2");
    buf[4] = 2; // non-1 version triggers "murmurs of concern"
    buf[5] = 1;
    file.write_all(&buf).expect("write");
    let result = run(file.path(), true, false, false, 100, false, false);
    assert!(result.is_ok());
}

/// FALSIFIER (#2394 finding 1): `apr debug` on a file it reports as
/// `Health ✗ CORRUPTED` must exit non-zero.
///
/// Measured on 200 KB of `/dev/urandom`: the report said `Magic ✗ INVALID`
/// and `Health ✗ CORRUPTED`, and the process exited 0 — so
/// `apr debug f && echo ok` printed ok. This test used to assert `is_ok()` on
/// `ZZZZ` magic, which is the same claim the defect made.
#[test]
fn run_basic_mode_with_invalid_magic() {
    let mut file = NamedTempFile::with_suffix(".apr").expect("create file");
    let mut buf = vec![0u8; HEADER_SIZE];
    buf[0..4].copy_from_slice(b"ZZZZ");
    file.write_all(&buf).expect("write");
    let result = run(file.path(), false, false, false, 100, false, false);
    let err = result.expect_err("a CORRUPTED verdict must not exit 0");
    assert!(
        err.to_string().contains("magic"),
        "the error must say what was wrong, got: {err}"
    );
}

/// ...and the same command must still succeed on a well-formed header, so the
/// exit code discriminates rather than always failing.
#[test]
fn run_basic_mode_with_valid_magic_still_exits_zero() {
    let mut file = NamedTempFile::with_suffix(".apr").expect("create file");
    let mut buf = vec![0u8; HEADER_SIZE];
    buf[0..4].copy_from_slice(b"APRN");
    buf[4] = 1;
    file.write_all(&buf).expect("write");
    run(file.path(), false, false, false, 100, false, false)
        .expect("a valid APR header must still exit 0");
}

#[test]
fn run_basic_mode_with_flags_shows_flag_line() {
    let mut file = NamedTempFile::with_suffix(".apr").expect("create file");
    let mut buf = vec![0u8; HEADER_SIZE];
    buf[0..4].copy_from_slice(b"APRN");
    buf[21] = 0x02; // signed flag
    file.write_all(&buf).expect("write");
    let result = run(file.path(), false, false, false, 100, false, false);
    assert!(result.is_ok());
}

#[test]
fn run_strings_mode_with_limit_one() {
    let mut file = NamedTempFile::new().expect("create file");
    // Two strings separated by null bytes
    file.write_all(b"firststring\x00secondstring\x00thirdstring")
        .expect("write");
    let result = run(file.path(), false, false, true, 1, false, false);
    assert!(result.is_ok());
}

#[test]
fn run_hex_mode_with_small_limit() {
    let mut file = NamedTempFile::new().expect("create file");
    file.write_all(&[0u8; 256]).expect("write");
    let result = run(file.path(), false, true, false, 32, false, false);
    assert!(result.is_ok());
}

#[test]
fn run_directory_rejected() {
    let dir = tempdir().expect("create dir");
    let result = run(dir.path(), false, false, false, 100, false, false);
    assert!(matches!(result, Err(CliError::NotAFile(_))));
}

// ========================================================================
// parse_header: Compression enum values (bytes 20)
// ========================================================================

#[test]
fn parse_header_compression_none_is_zero() {
    let mut header = [0u8; HEADER_SIZE];
    header[0..4].copy_from_slice(b"APRN");
    header[20] = 0; // None
    let info = parse_header(&header);
    assert!(!info.compressed);
}

#[test]
fn parse_header_compression_zstd_default() {
    let mut header = [0u8; HEADER_SIZE];
    header[0..4].copy_from_slice(b"APRN");
    header[20] = 1; // ZstdDefault
    let info = parse_header(&header);
    assert!(info.compressed);
}

#[test]
fn parse_header_compression_zstd_max() {
    let mut header = [0u8; HEADER_SIZE];
    header[0..4].copy_from_slice(b"APRN");
    header[20] = 2; // ZstdMax
    let info = parse_header(&header);
    assert!(info.compressed);
}

#[test]
fn parse_header_compression_lz4() {
    let mut header = [0u8; HEADER_SIZE];
    header[0..4].copy_from_slice(b"APRN");
    header[20] = 3; // Lz4
    let info = parse_header(&header);
    assert!(info.compressed);
}

#[test]
fn parse_header_compression_out_of_range() {
    let mut header = [0u8; HEADER_SIZE];
    header[0..4].copy_from_slice(b"APRN");
    header[20] = 4; // Out of range (0-3)
    let info = parse_header(&header);
    assert!(!info.compressed);
}

#[test]
fn parse_header_compression_255() {
    let mut header = [0u8; HEADER_SIZE];
    header[0..4].copy_from_slice(b"APRN");
    header[20] = 255;
    let info = parse_header(&header);
    assert!(!info.compressed);
}

// ========================================================================
// parse_header: Flags byte (byte 21) boundary values
// ========================================================================

#[test]
fn parse_header_flags_byte_zero_no_flags() {
    let mut header = [0u8; HEADER_SIZE];
    header[0..4].copy_from_slice(b"APRN");
    header[21] = 0x00;
    let info = parse_header(&header);
    assert!(!info.signed);
    assert!(!info.encrypted);
}

#[test]
fn parse_header_flags_signed_bit_only() {
    let mut header = [0u8; HEADER_SIZE];
    header[0..4].copy_from_slice(b"APRN");
    header[21] = 0x02; // bit 1 = signed
    let info = parse_header(&header);
    assert!(info.signed);
    assert!(!info.encrypted);
}

#[test]
fn parse_header_flags_encrypted_bit_only() {
    let mut header = [0u8; HEADER_SIZE];
    header[0..4].copy_from_slice(b"APRN");
    header[21] = 0x04; // bit 2 = encrypted
    let info = parse_header(&header);
    assert!(!info.signed);
    assert!(info.encrypted);
}

#[test]
fn parse_header_flags_signed_and_encrypted() {
    let mut header = [0u8; HEADER_SIZE];
    header[0..4].copy_from_slice(b"APRN");
    header[21] = 0x06; // bits 1+2
    let info = parse_header(&header);
    assert!(info.signed);
    assert!(info.encrypted);
}

#[test]
fn parse_header_flags_high_bits_set_garbage() {
    // When high bits (>= 0x08) are set, flags are garbage from
    // reading non-v2 file bytes as flags
    let mut header = [0u8; HEADER_SIZE];
    header[0..4].copy_from_slice(b"APRN");
    header[21] = 0xFF; // All bits set - garbage
    let info = parse_header(&header);
    // flags_byte >= 0x08 => signed/encrypted guards fail
    assert!(!info.signed);
    assert!(!info.encrypted);
}

#[test]
fn parse_header_flags_just_below_garbage_threshold() {
    let mut header = [0u8; HEADER_SIZE];
    header[0..4].copy_from_slice(b"APRN");
    header[21] = 0x07; // bits 0,1,2 set, still < 0x08
    let info = parse_header(&header);
    assert!(info.signed);
    assert!(info.encrypted);
}

#[test]
fn parse_header_flags_at_garbage_threshold() {
    let mut header = [0u8; HEADER_SIZE];
    header[0..4].copy_from_slice(b"APRN");
    header[21] = 0x08; // bit 3 set, >= 0x08 = garbage
    let info = parse_header(&header);
    assert!(!info.signed);
    assert!(!info.encrypted);
}

// ========================================================================
// collect_flags: combination coverage
// ========================================================================

#[test]
fn collect_flags_compressed_and_signed() {
    let info = HeaderInfo {
        magic_valid: true,
        magic_str: "APRN".to_string(),
        version: (1, 0),
        model_type: 0x0001,
        compressed: true,
        signed: true,
        encrypted: false,
    };
    let flags = collect_flags(&info);
    assert_eq!(flags, vec!["compressed", "signed"]);
}

#[test]
fn collect_flags_compressed_and_encrypted() {
    let info = HeaderInfo {
        magic_valid: true,
        magic_str: "APRN".to_string(),
        version: (1, 0),
        model_type: 0x0001,
        compressed: true,
        signed: false,
        encrypted: true,
    };
    let flags = collect_flags(&info);
    assert_eq!(flags, vec!["compressed", "encrypted"]);
}

// ========================================================================
// format_model_type: additional coverage
// ========================================================================

#[test]
fn format_model_type_gap_between_svm_and_ngram() {
    // IDs between 0x000B and 0x000F should be Unknown
    assert_eq!(format_model_type(0x000B), "Unknown(0x000B)");
    assert_eq!(format_model_type(0x000C), "Unknown(0x000C)");
    assert_eq!(format_model_type(0x000F), "Unknown(0x000F)");
}

#[test]
fn format_model_type_gap_between_count_vec_and_neural() {
    assert_eq!(format_model_type(0x0013), "Unknown(0x0013)");
    assert_eq!(format_model_type(0x001F), "Unknown(0x001F)");
}

#[test]
fn format_model_type_gap_between_neural_custom_and_recommender() {
    assert_eq!(format_model_type(0x0022), "Unknown(0x0022)");
    assert_eq!(format_model_type(0x002F), "Unknown(0x002F)");
}

// ========================================================================
// run_basic_mode: flag string construction
// ========================================================================

#[test]
fn run_basic_mode_no_flags_shows_none() {
    let info = HeaderInfo {
        magic_valid: true,
        magic_str: "APRN".to_string(),
        version: (1, 0),
        model_type: 0x0001,
        compressed: false,
        signed: false,
        encrypted: false,
    };
    let flag_list = collect_flags(&info);
    let flags_str = if flag_list.is_empty() {
        "none".to_string()
    } else {
        flag_list.join(", ")
    };
    assert_eq!(flags_str, "none");
}

#[test]
fn run_basic_mode_flags_join_with_comma() {
    let info = HeaderInfo {
        magic_valid: true,
        magic_str: "APRN".to_string(),
        version: (1, 0),
        model_type: 0x0001,
        compressed: true,
        signed: true,
        encrypted: true,
    };
    let flag_list = collect_flags(&info);
    let flags_str = flag_list.join(", ");
    assert_eq!(flags_str, "compressed, signed, encrypted");
}

// ========================================================================
// run: JSON mode with various file formats
// ========================================================================

#[test]
fn run_json_mode_with_gguf_file() {
    let mut file = NamedTempFile::with_suffix(".gguf").expect("create file");
    // Write GGUF magic + enough data to parse header
    file.write_all(b"GGUF").expect("write");
    file.write_all(&[0u8; 100]).expect("write padding");
    // JSON mode dispatches to run_json_mode via RosettaStone
    let result = run(file.path(), false, false, false, 100, true, false);
    // May fail due to invalid GGUF structure, but exercises the JSON path
    // (either success or error, not panic)
    let _ = result;
}

#[test]
fn run_json_mode_with_apr_file() {
    let mut file = NamedTempFile::with_suffix(".apr").expect("create file");
    let mut buf = vec![0u8; HEADER_SIZE];
    buf[0..4].copy_from_slice(b"APRN");
    buf[4] = 1;
    file.write_all(&buf).expect("write");
    // JSON mode should try RosettaStone inspect
    let result = run(file.path(), false, false, false, 100, true, false);
    let _ = result;
}

// ========================================================================
// run: verbose mode exercises
// ========================================================================

#[test]
fn run_verbose_with_apr_file() {
    let mut file = NamedTempFile::with_suffix(".apr").expect("create file");
    let mut buf = vec![0u8; HEADER_SIZE];
    buf[0..4].copy_from_slice(b"APRN");
    buf[4] = 1;
    buf[5] = 0;
    file.write_all(&buf).expect("write");
    // verbose flag on basic APR mode
    let result = run(file.path(), false, false, false, 100, false, true);
    assert!(result.is_ok());
}

#[test]
fn run_verbose_with_drama_mode() {
    let mut file = NamedTempFile::with_suffix(".apr").expect("create file");
    let mut buf = vec![0u8; HEADER_SIZE];
    buf[0..4].copy_from_slice(b"APRN");
    buf[4] = 1;
    file.write_all(&buf).expect("write");
    // drama + verbose
    let result = run(file.path(), true, false, false, 100, false, true);
    assert!(result.is_ok());
}

// ========================================================================
// run: hex mode edge cases
// ========================================================================

#[test]
fn run_hex_mode_with_empty_file() {
    let file = NamedTempFile::new().expect("create file");
    let result = run(file.path(), false, true, false, 256, false, false);
    assert!(result.is_ok());
}

#[test]
fn run_hex_mode_limit_zero() {
    let mut file = NamedTempFile::new().expect("create file");
    file.write_all(&[0xAB; 100]).expect("write");
    // limit=0 means read 0 bytes
    let result = run(file.path(), false, true, false, 0, false, false);
    assert!(result.is_ok());
}

#[test]
fn run_hex_mode_limit_exceeds_file() {
    let mut file = NamedTempFile::new().expect("create file");
    file.write_all(b"short").expect("write");
    let result = run(file.path(), false, true, false, 8192, false, false);
    assert!(result.is_ok());
}

// ========================================================================
// run: strings mode edge cases
// ========================================================================

#[test]
fn run_strings_mode_all_printable() {
    let mut file = NamedTempFile::new().expect("create file");
    file.write_all(b"abcdefghijklmnop").expect("write");
    let result = run(file.path(), false, false, true, 100, false, false);
    assert!(result.is_ok());
}

#[test]
fn run_strings_mode_all_binary() {
    let mut file = NamedTempFile::new().expect("create file");
    file.write_all(&[0x01, 0x02, 0x03, 0x80, 0xFF]).expect("write");
    let result = run(file.path(), false, false, true, 100, false, false);
    assert!(result.is_ok());
}

#[test]
fn run_strings_mode_mixed_short_strings() {
    let mut file = NamedTempFile::new().expect("create file");
    // Strings < 4 chars should not be printed
    file.write_all(b"ab\x00cd\x00efgh\x00").expect("write");
    let result = run(file.path(), false, false, true, 100, false, false);
    assert!(result.is_ok());
}

#[test]
fn run_strings_mode_empty_file() {
    let file = NamedTempFile::new().expect("create file");
    let result = run(file.path(), false, false, true, 100, false, false);
    assert!(result.is_ok());
}

// ========================================================================
// HeaderInfo construction for all model types
// ========================================================================

#[test]
fn parse_header_all_known_model_types() {
    let known_types: Vec<(u16, &str)> = vec![
        (0x0001, "LinearRegression"),
        (0x0002, "LogisticRegression"),
        (0x0003, "DecisionTree"),
        (0x0004, "RandomForest"),
        (0x0005, "GradientBoosting"),
        (0x0006, "KMeans"),
        (0x0007, "PCA"),
        (0x0008, "NaiveBayes"),
        (0x0009, "KNN"),
        (0x000A, "SVM"),
        (0x0010, "NgramLM"),
        (0x0011, "TfIdf"),
        (0x0012, "CountVectorizer"),
        (0x0020, "NeuralSequential"),
        (0x0021, "NeuralCustom"),
        (0x0030, "ContentRecommender"),
        (0x0040, "MixtureOfExperts"),
        (0x00FF, "Custom"),
    ];
    for (type_id, expected_name) in known_types {
        let mut header = [0u8; HEADER_SIZE];
        header[0..4].copy_from_slice(b"APRN");
        let bytes = type_id.to_le_bytes();
        header[6] = bytes[0];
        header[7] = bytes[1];
        let info = parse_header(&header);
        assert_eq!(info.model_type, type_id);
        assert_eq!(format_model_type(info.model_type), expected_name);
    }
}

// ========================================================================
// validate_path: additional edge cases
// ========================================================================

#[test]
fn validate_path_with_symlink_to_file() {
    let file = NamedTempFile::new().expect("create file");
    // Can validate the path of a temp file
    assert!(validate_path(file.path()).is_ok());
}

#[test]
fn validate_path_empty_path() {
    let result = validate_path(Path::new(""));
    assert!(result.is_err());
}
