//! #3691: `apr hex` and `apr trace` name a GGUF or SafeTensors file by its format.
//!
//! #3661 fixed `apr inspect` and `apr tensors`. Ten more refusals in hex, its
//! SafeTensors header reader, and trace still wrapped their message in
//! `CliError::InvalidFormat`, so a truncated GGUF printed
//! "error: Invalid APR format: Failed to parse GGUF: …". One row per site,
//! rendered exactly as `apr` prints it (`error: {e}`, main.rs), checked the same
//! four ways as #3661's table: the prefix, no "Invalid APR" for a non-APR file,
//! no core "Invalid model format:" leaking under the CLI's own, and exit code 4.
//!
//! This module is a child of `hex`, so it can call the header reader directly.
//! Three sites (the reader's size check, and the "not a JSON object" check in
//! each of its two callers) sit behind `run`'s `detect_format`, which only picks
//! SafeTensors for 9+ bytes with `{` at byte 8. No file reaches them through
//! `run`, so their rows call the private function with the bytes that would.

use super::{parse_safetensors_header, run, run_safetensors, slice_safetensors, HexOptions};
use crate::commands::model_file_error_tests_3661::{
    random_safetensors, truncated_apr, truncated_gguf,
};
use crate::commands::trace;
use crate::error::CliError;
use std::path::{Path, PathBuf};

/// A SafeTensors file: `header_len` as u64 LE, then `header`, then `tail`.
fn safetensors_bytes(header_len: u64, header: &[u8], tail: &[u8]) -> Vec<u8> {
    let mut bytes = header_len.to_le_bytes().to_vec();
    bytes.extend_from_slice(header);
    bytes.extend_from_slice(tail);
    bytes
}

fn write(dir: &Path, name: &str, bytes: &[u8]) -> PathBuf {
    let path = dir.join(name);
    std::fs::write(&path, bytes).expect("write fixture");
    path
}

/// The header claims 1000 bytes; the file holds 16.
fn st_header_overrun() -> Vec<u8> {
    safetensors_bytes(1000, b"{\"a\":1}", b"x")
}

/// A 4-byte header that opens like JSON and is not UTF-8.
fn st_header_not_utf8() -> Vec<u8> {
    safetensors_bytes(4, b"{\xff\xfe}", b"")
}

/// A 5-byte header, `{"a":`, cut off mid-object. It opens with `{"`, so
/// rosetta's magic check (the one trace uses) takes it for SafeTensors too.
fn st_header_not_json() -> Vec<u8> {
    safetensors_bytes(5, b"{\"a\":", b"")
}

/// A header that parses as JSON but is an array, not an object.
fn st_header_array() -> Vec<u8> {
    safetensors_bytes(3, b"[1]", b"")
}

fn hex_err(path: &Path) -> CliError {
    let opts = HexOptions {
        file: path.to_path_buf(),
        ..HexOptions::default()
    };
    run(&opts).expect_err("hex must refuse")
}

/// `apr hex FILE --slice 0:1 --tensor w`
fn hex_slice_err(path: &Path) -> CliError {
    let opts = HexOptions {
        file: path.to_path_buf(),
        tensor: Some("w".to_string()),
        slice: Some("0:1".to_string()),
        ..HexOptions::default()
    };
    run(&opts).expect_err("hex --slice must refuse")
}

fn trace_err(path: &Path) -> CliError {
    trace::run(path, None, None, false, false, false, false, false).expect_err("trace must refuse")
}

#[test]
fn every_hex_and_trace_parse_failure_names_the_format_not_apr() {
    let dir = tempfile::tempdir().expect("tempdir");
    let d = dir.path();
    let gguf = truncated_gguf(d);
    let apr = truncated_apr(d);
    let overrun = write(d, "overrun.safetensors", &st_header_overrun());
    let not_utf8 = write(d, "not_utf8.safetensors", &st_header_not_utf8());
    let not_json = write(d, "not_json.safetensors", &st_header_not_json());
    // #3661's noise file: its first u64 is far past any header length, so no
    // magic matches and trace reaches the SafeTensors reader by extension alone.
    let noise = random_safetensors(d);
    let opts = HexOptions::default();
    let array = st_header_array();

    // (site, the error, the prefix the rendered line must start with)
    let rows: Vec<(&str, CliError, &str)> = vec![
        (
            "hex.rs parse_gguf: hex on a truncated GGUF",
            hex_err(&gguf),
            "error: Invalid GGUF file: Failed to parse GGUF: ",
        ),
        (
            "hex.rs get_gguf_tensor_f32: hex --slice on a truncated GGUF",
            hex_slice_err(&gguf),
            "error: Invalid GGUF file: Failed to parse GGUF: ",
        ),
        (
            "safe_tensors_header.rs size check: 4 bytes, direct",
            parse_safetensors_header(&[0u8; 4])
                .err()
                .expect("4 bytes must refuse"),
            "error: Invalid model file: SafeTensors file too small",
        ),
        (
            "safe_tensors_header.rs length check: hex on an overrunning header",
            hex_err(&overrun),
            "error: Invalid SafeTensors file: SafeTensors header length exceeds file size",
        ),
        (
            "safe_tensors_header.rs UTF-8: hex on a non-UTF-8 header",
            hex_err(&not_utf8),
            "error: Invalid SafeTensors file: Invalid SafeTensors header UTF-8: ",
        ),
        (
            "safe_tensors_header.rs JSON: hex on a cut-off header",
            hex_err(&not_json),
            "error: Invalid SafeTensors file: Invalid SafeTensors JSON: ",
        ),
        (
            "safe_tensors_header.rs JSON, the reader's second caller: hex --slice",
            hex_slice_err(&not_json),
            "error: Invalid SafeTensors file: Invalid SafeTensors JSON: ",
        ),
        (
            "safe_tensors_header.rs run_safetensors: an array header, direct",
            run_safetensors(&opts, &array).expect_err("an array header must refuse"),
            "error: Invalid model file: SafeTensors header is not a JSON object",
        ),
        (
            "safe_tensors_header.rs slice_safetensors: an array header, direct",
            slice_safetensors(&opts, &array, "w", 0, 1).expect_err("an array header must refuse"),
            "error: Invalid model file: SafeTensors header is not a JSON object",
        ),
        (
            "trace trace_gguf: trace on a truncated GGUF",
            trace_err(&gguf),
            "error: Invalid GGUF file: Failed to parse GGUF: ",
        ),
        (
            "trace trace_safetensors: trace on a cut-off header",
            trace_err(&not_json),
            "error: Invalid SafeTensors file: Failed to inspect SafeTensors: ",
        ),
        (
            "trace trace_safetensors: trace on noise named .safetensors",
            trace_err(&noise),
            "error: Invalid model file: Failed to inspect SafeTensors: ",
        ),
        // Control: hex on a truncated .apr still says APR (run_apr, unchanged),
        // so the fix is not "delete the word".
        (
            "control, hex run_apr: hex on a truncated APR",
            hex_err(&apr),
            "error: Invalid APR format: Failed to read APR: ",
        ),
    ];

    let mut failures = Vec::new();
    for (row, err, want_prefix) in &rows {
        let rendered = format!("error: {err}");
        let is_apr = want_prefix.starts_with("error: Invalid APR");
        if !rendered.starts_with(want_prefix) {
            failures.push(format!("{row}: wanted `{want_prefix}…`, got `{rendered}`"));
        }
        if !is_apr && rendered.contains("Invalid APR") {
            failures.push(format!("{row}: calls a non-APR file APR: `{rendered}`"));
        }
        if rendered.contains("Invalid model format") {
            failures.push(format!(
                "{row}: the core prefix leaked under the CLI's: `{rendered}`"
            ));
        }
        if err.exit_code_value() != 4 {
            failures.push(format!("{row}: exit {} (wanted 4)", err.exit_code_value()));
        }
    }
    assert!(
        failures.is_empty(),
        "#3691 rows failed:\n{}",
        failures.join("\n")
    );
}
