//! Falsifiers for #2569 — `apr tensors` reported a table that does not fit the file.
//!
//! Before the fix in `check_tensor_table_fits`, every test in this file that
//! expects `Err` got `Ok` with a full, confident tensor table. The defect and
//! its blast radius, measured on the 0.63.0 binary:
//!
//! ```text
//! apr tensors  truncated.gguf          rc=0   "1 tensors 8.0 KB F32"
//! apr tensors  truncated.gguf --json   rc=0   "total_size_bytes": 8192
//! apr validate truncated.gguf          rc=5   "(file is 1 bytes too short)"
//! ```
//!
//! on a 128-byte file. The JSON is what the MCP tool and CI consume, so the
//! surface that said OK is the one that is machine-read.
//!
//! Coverage here is deliberately BOTH formats and BOTH the typed API and the
//! path-dispatch entry point: #2564 recorded that a truncated GGUF passed where
//! a truncated APR did not, so a gate that only exercises one of them proves
//! nothing about the other.

use super::*;
use crate::format::gguf::{export_tensors_to_gguf, GgmlType, GgufTensor, GgufValue};
use crate::format::test_factory::build_pygmy_apr;

/// A valid single-tensor GGUF: 64x32 F32 = 2048 elements = 8192 bytes of data.
fn valid_gguf() -> Vec<u8> {
    let data: Vec<u8> = (0..2048u32)
        .flat_map(|i| (i as f32 + 1.0).to_le_bytes())
        .collect();
    let tensor = GgufTensor {
        name: "test.weight".to_string(),
        shape: vec![64, 32],
        dtype: GgmlType::F32,
        data,
    };
    let metadata = vec![(
        "general.architecture".to_string(),
        GgufValue::String("test".to_string()),
    )];
    let mut bytes = Vec::new();
    export_tensors_to_gguf(&mut bytes, &[tensor], &metadata).expect("export GGUF");
    bytes
}

/// Byte offset where the GGUF data section starts, per the reader itself.
fn gguf_data_offset(bytes: &[u8]) -> usize {
    GgufReader::from_bytes(bytes.to_vec())
        .expect("parse GGUF")
        .data_offset
}

// ========================================================================
// GGUF
// ========================================================================

/// Positive control. The check must not reject a file that IS complete —
/// otherwise every assertion below is satisfied by a function that always
/// errors, and this file would prove nothing.
#[test]
fn gguf_intact_file_still_lists() {
    let bytes = valid_gguf();
    let result = list_tensors_from_bytes(&bytes, TensorListOptions::default())
        .expect("an intact GGUF must still list");
    assert_eq!(result.tensor_count, 1);
    assert_eq!(result.tensors[0].size_bytes, 8192);
    assert_eq!(result.total_size_bytes, 8192);
}

/// The reported defect: the entire data section is missing and the listing
/// still printed 8192 bytes of tensor.
#[test]
fn gguf_missing_data_section_is_refused() {
    let bytes = valid_gguf();
    let header_only = &bytes[..gguf_data_offset(&bytes)];

    let err = list_tensors_from_bytes(header_only, TensorListOptions::default())
        .expect_err("a GGUF with no data section must not list 8192 bytes of tensor");
    let msg = err.to_string();
    assert!(
        msg.contains("Truncated GGUF") && msg.contains("test.weight"),
        "error must name the format and the offending tensor, got: {msg}"
    );
    assert!(
        msg.contains("8192 bytes too short"),
        "error must quantify the shortfall like `apr validate` does, got: {msg}"
    );
}

/// Strictly stronger than the check `apr validate` already had.
///
/// `RosettaStone::validate_gguf` asks only whether the LAST tensor's FIRST byte
/// is inside the file (`data_start + offset + 1`). A file short by one byte
/// satisfies that and is still a lie: the row asserts 8192 bytes exist and 8191
/// do. This case is the reason the check here is on the tensor's full EXTENT.
#[test]
fn gguf_short_by_one_byte_is_refused() {
    let bytes = valid_gguf();
    let one_short = &bytes[..bytes.len() - 1];

    let err = list_tensors_from_bytes(one_short, TensorListOptions::default())
        .expect_err("a GGUF one byte short of its declared extent must be refused");
    assert!(
        err.to_string().contains("1 bytes too short"),
        "got: {err}"
    );
}

/// A `--filter` that selects nothing must not launder a truncated file into a
/// clean exit. The check runs over every tensor in the index, before filtering.
#[test]
fn gguf_filter_cannot_hide_truncation() {
    let bytes = valid_gguf();
    let header_only = &bytes[..gguf_data_offset(&bytes)];

    let options = TensorListOptions::default().with_filter("no-such-tensor");
    let err = list_tensors_from_bytes(header_only, options)
        .expect_err("a filter matching zero tensors must not turn a truncated file into rc=0");
    assert!(err.to_string().contains("Truncated GGUF"), "got: {err}");
}

/// The same refusal through the path-dispatch entry point that `apr tensors`
/// and `apr tune` call, not just the bytes API.
#[test]
fn gguf_truncated_on_disk_is_refused() {
    let dir = std::env::temp_dir().join("apr_tensors_bounds_2569_gguf");
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("truncated.gguf");
    let bytes = valid_gguf();
    std::fs::write(&path, &bytes[..gguf_data_offset(&bytes)]).expect("write fixture");

    let err = list_tensors(&path, TensorListOptions::default())
        .expect_err("list_tensors must refuse a truncated GGUF on disk");
    assert!(err.to_string().contains("Truncated GGUF"), "got: {err}");

    let _ = std::fs::remove_file(&path);
}

/// `--stats` that cannot read a tensor must not render identically to a run
/// that never asked for stats. Before the fix the read was `if let Ok(..)` and
/// the mean/std/range columns printed "—" either way.
#[test]
fn gguf_stats_failure_is_distinguishable_from_stats_not_requested() {
    // Q8_1 (dtype 9): `ggml_dtype_element_size` knows its 1.125 bytes/element so
    // the bounds check is satisfiable, but `GgufReader::get_tensor_f32` has no
    // dequantiser arm for it — exactly the "all the bytes are here and I still
    // cannot give you numbers" case that used to print an em-dash.
    let mut bytes = b"GGUF".to_vec();
    bytes.extend_from_slice(&3u32.to_le_bytes()); // version
    bytes.extend_from_slice(&1u64.to_le_bytes()); // tensor_count
    bytes.extend_from_slice(&0u64.to_le_bytes()); // metadata_kv_count
    let name = b"weird.weight";
    bytes.extend_from_slice(&(name.len() as u64).to_le_bytes());
    bytes.extend_from_slice(name);
    bytes.extend_from_slice(&1u32.to_le_bytes()); // n_dims
    bytes.extend_from_slice(&256u64.to_le_bytes()); // dims[0]
    bytes.extend_from_slice(&9u32.to_le_bytes()); // dtype Q8_1 — no dequantiser
    bytes.extend_from_slice(&0u64.to_le_bytes()); // offset
    while bytes.len() % 32 != 0 {
        bytes.push(0);
    }
    // Q8_1 is 1.125 bytes/element: 256 * 1.125 = 288 bytes, all present.
    bytes.extend_from_slice(&[7u8; 288]);

    // Without --stats the listing is fine: the shape and size are knowable.
    let listed = list_tensors_from_bytes(&bytes, TensorListOptions::default())
        .expect("listing without stats must succeed — the bytes are all there");
    assert_eq!(listed.tensor_count, 1);
    assert!(
        listed.tensors[0].mean.is_none(),
        "no stats requested => no stats"
    );

    // With --stats the failure must be reported, not rendered as an em-dash.
    let err = list_tensors_from_bytes(&bytes, TensorListOptions::default().with_stats())
        .expect_err("--stats that cannot be computed must say so");
    let msg = err.to_string();
    assert!(
        msg.contains("--stats") && msg.contains("weird.weight"),
        "error must name the flag and the tensor, got: {msg}"
    );
}

// ========================================================================
// APR v2
// ========================================================================

/// Positive control for the APR path.
#[test]
fn apr_intact_file_still_lists() {
    let bytes = build_pygmy_apr();
    let result = list_tensors_from_bytes(&bytes, TensorListOptions::default())
        .expect("an intact APR v2 must still list");
    assert!(result.tensor_count > 0, "pygmy APR should have tensors");
}

/// #2564 filed this as an explicit follow-up: a 50 MB head of a 991 MB APR
/// model printed "291 tensors 942.3 MB", 19x the file, because the header
/// offsets it checks genuinely ARE inside the file at that size — the lie had
/// moved to the tensor extent. One byte is the smallest version of that.
#[test]
fn apr_short_by_one_byte_is_refused() {
    let bytes = build_pygmy_apr();
    // Truncate to one byte short of the LAST DECLARED TENSOR BYTE, not one byte
    // off the end of the file: an APR v2 file is 64-byte aligned and can carry
    // trailing padding, so lopping one byte off EOF may still leave every tensor
    // extent inside the file — a fixture that proves nothing.
    let reader = AprV2Reader::from_bytes(&bytes).expect("parse pygmy APR");
    let last_byte = reader.header().data_offset
        + reader
            .tensor_names()
            .iter()
            .filter_map(|n| reader.get_tensor(n))
            .map(|e| e.offset + e.size)
            .max()
            .expect("pygmy APR has tensors");
    let one_short = &bytes[..(last_byte as usize) - 1];

    let err = list_tensors_from_bytes(one_short, TensorListOptions::default())
        .expect_err("an APR v2 one byte short of its declared extent must be refused");
    let msg = err.to_string();
    assert!(
        msg.contains("Truncated APR v2") && msg.contains("bytes too short"),
        "got: {msg}"
    );
}

/// The mmap path — the one `apr tensors <file>.apr` actually takes — must
/// refuse it too, or the sibling check above is theater.
#[test]
fn apr_truncated_on_disk_is_refused() {
    let dir = std::env::temp_dir().join("apr_tensors_bounds_2569_apr");
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("truncated.apr");
    let bytes = build_pygmy_apr();
    // Drop the last quarter of the data section: big enough that a partial
    // download is a realistic way to reach it.
    std::fs::write(&path, &bytes[..bytes.len() * 3 / 4]).expect("write fixture");

    let err = list_tensors(&path, TensorListOptions::default())
        .expect_err("list_tensors must refuse a truncated APR v2 on disk");
    assert!(err.to_string().contains("too short"), "got: {err}");

    let _ = std::fs::remove_file(&path);
}

/// Positive control on the mmap path: an intact APR on disk still lists.
#[test]
fn apr_intact_on_disk_still_lists() {
    let dir = std::env::temp_dir().join("apr_tensors_bounds_2569_apr_ok");
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join("intact.apr");
    std::fs::write(&path, build_pygmy_apr()).expect("write fixture");

    let result = list_tensors(&path, TensorListOptions::default())
        .expect("an intact APR v2 on disk must still list");
    assert!(result.tensor_count > 0);

    let _ = std::fs::remove_file(&path);
}

// ========================================================================
// The check itself
// ========================================================================

#[test]
fn check_tensor_table_fits_accepts_an_exactly_full_file() {
    // end == file_len is fine: the last byte of the tensor is the last byte of
    // the file. An off-by-one here would reject every intact model.
    check_tensor_table_fits("TEST", 100, 20, [("t", 0u64, 80u64)])
        .expect("a tensor ending exactly at EOF fits");
}

#[test]
fn check_tensor_table_fits_rejects_one_byte_over() {
    let err = check_tensor_table_fits("TEST", 100, 20, [("t", 0u64, 81u64)])
        .expect_err("81 bytes at offset 20 do not fit in 100");
    assert!(err.to_string().contains("1 bytes too short"), "got: {err}");
}

#[test]
fn check_tensor_table_fits_reports_the_first_offender_not_the_last() {
    let err = check_tensor_table_fits(
        "TEST",
        100,
        0,
        [("ok", 0u64, 10u64), ("bad", 10u64, 200u64), ("also_bad", 0, 999)],
    )
    .expect_err("must reject");
    assert!(err.to_string().contains("'bad'"), "got: {err}");
}

#[test]
fn check_tensor_table_fits_refuses_an_overflowing_extent() {
    // A crafted header must not be able to wrap u64 into a small end offset.
    let err = check_tensor_table_fits("TEST", 100, 1, [("t", u64::MAX, 1u64)])
        .expect_err("offset overflow must be refused, not wrapped");
    assert!(err.to_string().contains("overflow"), "got: {err}");

    let err = check_tensor_table_fits("TEST", 100, 0, [("t", u64::MAX, 2u64)])
        .expect_err("size overflow must be refused, not wrapped");
    assert!(err.to_string().contains("overflow"), "got: {err}");
}
