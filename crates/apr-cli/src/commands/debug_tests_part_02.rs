
#[test]
fn run_strings_mode_succeeds_on_regular_file() {
    let mut file = NamedTempFile::new().expect("create file");
    file.write_all(b"Hello\x00World\x00teststring\x00ab\x00longword")
        .expect("write");
    let result = run(file.path(), false, false, true, 100, false);
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
    let result = run(file.path(), false, false, false, 100, false);
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
    let result = run(file.path(), true, false, false, 100, false);
    assert!(result.is_ok());
}

#[test]
fn run_drama_mode_with_invalid_magic() {
    let mut file = NamedTempFile::with_suffix(".apr").expect("create file");
    let mut buf = vec![0u8; HEADER_SIZE];
    buf[0..4].copy_from_slice(b"XXXX");
    buf[4] = 1;
    file.write_all(&buf).expect("write");
    let result = run(file.path(), true, false, false, 100, false);
    assert!(result.is_ok());
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
    let result = run(file.path(), true, false, false, 100, false);
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
    let result = run(file.path(), true, false, false, 100, false);
    assert!(result.is_ok());
}

#[test]
fn run_basic_mode_with_invalid_magic() {
    let mut file = NamedTempFile::with_suffix(".apr").expect("create file");
    let mut buf = vec![0u8; HEADER_SIZE];
    buf[0..4].copy_from_slice(b"ZZZZ");
    file.write_all(&buf).expect("write");
    let result = run(file.path(), false, false, false, 100, false);
    assert!(result.is_ok());
}

#[test]
fn run_basic_mode_with_flags_shows_flag_line() {
    let mut file = NamedTempFile::with_suffix(".apr").expect("create file");
    let mut buf = vec![0u8; HEADER_SIZE];
    buf[0..4].copy_from_slice(b"APRN");
    buf[21] = 0x02; // signed flag
    file.write_all(&buf).expect("write");
    let result = run(file.path(), false, false, false, 100, false);
    assert!(result.is_ok());
}

#[test]
fn run_strings_mode_with_limit_one() {
    let mut file = NamedTempFile::new().expect("create file");
    // Two strings separated by null bytes
    file.write_all(b"firststring\x00secondstring\x00thirdstring")
        .expect("write");
    let result = run(file.path(), false, false, true, 1, false);
    assert!(result.is_ok());
}

#[test]
fn run_hex_mode_with_small_limit() {
    let mut file = NamedTempFile::new().expect("create file");
    file.write_all(&[0u8; 256]).expect("write");
    let result = run(file.path(), false, true, false, 32, false);
    assert!(result.is_ok());
}

#[test]
fn run_directory_rejected() {
    let dir = tempdir().expect("create dir");
    let result = run(dir.path(), false, false, false, 100, false);
    assert!(matches!(result, Err(CliError::NotAFile(_))));
}
