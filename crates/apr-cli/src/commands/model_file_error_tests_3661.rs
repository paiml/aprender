//! #3661: a model file that fails to parse is named by what its magic bytes say it is.
//!
//! `CliError::InvalidFormat` Displays "Invalid APR format", and `apr inspect` and
//! `apr tensors` wrapped every parse failure in it, so a truncated GGUF was reported
//! as an invalid APR file. Each row drives the real command function and renders the
//! error exactly as `apr` prints it (`error: {e}`, main.rs), then checks three things:
//! the format the message names, exit code 4, and that the core error's
//! "Invalid model format:" prefix never appears under the CLI's own.

use crate::commands::{inspect, tensors};
use crate::error::CliError;
use std::path::{Path, PathBuf};

/// A GGUF written by aprender's own exporter, cut off inside its metadata.
/// That is the issue's input: a real GGUF truncated mid-header.
pub(super) fn truncated_gguf(dir: &Path) -> PathBuf {
    use aprender::format::gguf::{export_tensors_to_gguf, GgmlType, GgufTensor, GgufValue};
    let tensors = vec![GgufTensor {
        name: "token_embd.weight".to_string(),
        shape: vec![4, 8],
        dtype: GgmlType::F32,
        data: vec![0u8; 4 * 8 * 4],
    }];
    let metadata = vec![
        (
            "general.architecture".to_string(),
            GgufValue::String("llama".to_string()),
        ),
        (
            "general.name".to_string(),
            GgufValue::String("truncated-3661".to_string()),
        ),
    ];
    let mut bytes = Vec::new();
    export_tensors_to_gguf(&mut bytes, &tensors, &metadata).expect("write GGUF");
    // 24 bytes of fixed header (magic, version, two counts), then the first key's u64
    // length: cut 4 bytes into it, so the reader hits the issue's own failure,
    // "Unexpected EOF reading u64".
    let path = dir.join("t.gguf");
    std::fs::write(&path, &bytes[..28]).expect("write truncated GGUF");
    path
}

/// 4 KiB of noise named `.safetensors`, opening with the issue's own magic bytes
/// (`8e44f12f`). Its first u64 is far past any real header length, so no format matches.
pub(super) fn random_safetensors(dir: &Path) -> PathBuf {
    let mut state: u32 = 0x3661;
    let mut bytes: Vec<u8> = vec![0x8e, 0x44, 0xf1, 0x2f, 0x9d, 0x27, 0x6b, 0xb3];
    while bytes.len() < 4096 {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        bytes.push((state >> 24) as u8);
    }
    let path = dir.join("t.safetensors");
    std::fs::write(&path, &bytes).expect("write random bytes");
    path
}

/// A real APR v2 file cut to 32 bytes, less than its 64-byte header.
pub(super) fn truncated_apr(dir: &Path) -> PathBuf {
    use aprender::format::v2::{AprV2Metadata, AprV2Writer};
    let mut writer = AprV2Writer::new(AprV2Metadata::new("truncated-3661"));
    writer.add_f32_tensor("w", vec![2, 2], &[0.0, 1.0, 2.0, 3.0]);
    let bytes = writer.write().expect("write APR v2");
    let path = dir.join("t.apr");
    std::fs::write(&path, &bytes[..32]).expect("write truncated APR");
    path
}

fn inspect_err(path: &Path) -> CliError {
    inspect::run(path, false, false, false, false, false).expect_err("inspect must refuse")
}

fn tensors_err(path: &Path) -> CliError {
    tensors::run(path, false, None, false, 0).expect_err("tensors must refuse")
}

#[test]
fn a_parse_failure_names_the_format_the_magic_bytes_identify() {
    let dir = tempfile::tempdir().expect("tempdir");
    let gguf = truncated_gguf(dir.path());
    let st = random_safetensors(dir.path());
    let apr = truncated_apr(dir.path());

    // (row, the error, the prefix the rendered line must start with)
    let rows: Vec<(&str, CliError, &str)> = vec![
        (
            "inspect truncated GGUF",
            inspect_err(&gguf),
            "error: Invalid GGUF file: ",
        ),
        (
            "tensors truncated GGUF",
            tensors_err(&gguf),
            "error: Invalid GGUF file: ",
        ),
        (
            "inspect random .safetensors",
            inspect_err(&st),
            "error: Invalid model file: ",
        ),
        (
            "tensors random .safetensors",
            tensors_err(&st),
            "error: Invalid model file: ",
        ),
        // A truncated .apr still says APR, so the fix is not "delete the word".
        (
            "inspect truncated APR",
            inspect_err(&apr),
            "error: Invalid APR ",
        ),
        (
            "tensors truncated APR",
            tensors_err(&apr),
            "error: Invalid APR file: ",
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
        "#3661 rows failed:\n{}",
        failures.join("\n")
    );
}
