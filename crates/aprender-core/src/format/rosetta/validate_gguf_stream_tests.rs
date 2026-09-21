//! #3790: `validate_gguf` reads each tensor's own bytes, not the whole model. It must check
//! exactly what the whole-file read checked: the same values, the same failures on the same
//! tensors, and the same error for a tensor that runs past the end of the file.

use super::*;
use crate::format::gguf::reader::read_tensor_f32;
use crate::format::gguf::{export_tensors_to_gguf, GgmlType, GgufReader, GgufTensor, GgufValue};
use std::io::Write;

/// 256 elements: a whole number of blocks for every block type below.
const N: usize = 256;

/// Deterministic bytes (xorshift), so a quantized tensor holds real-looking blocks.
fn noise(len: usize, mut seed: u64) -> Vec<u8> {
    (0..len)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 32) as u8
        })
        .collect()
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn tensor(name: &str, dtype: GgmlType, data: Vec<u8>) -> GgufTensor {
    GgufTensor {
        name: name.into(),
        shape: vec![N as u64],
        dtype,
        data,
    }
}

fn gguf_file(tensors: &[GgufTensor]) -> tempfile::NamedTempFile {
    let metadata = vec![(
        "general.architecture".to_string(),
        GgufValue::String("llama".into()),
    )];
    let mut bytes = Vec::new();
    export_tensors_to_gguf(&mut bytes, tensors, &metadata).expect("write GGUF");
    let mut f = tempfile::NamedTempFile::with_suffix(".gguf").expect("temp file");
    f.write_all(&bytes).expect("write");
    f
}

/// Every type aprender-core dequantizes, as ggml's block sizes lay them out for 256 elements.
fn every_dequantizable_dtype() -> Vec<GgufTensor> {
    [
        (GgmlType::F32, N * 4),
        (GgmlType::F16, N * 2),
        (GgmlType::Q4_0, N / 32 * 18),
        (GgmlType::Q4_1, N / 32 * 20),
        (GgmlType::Q5_0, N / 32 * 22),
        (GgmlType::Q5_1, N / 32 * 24),
        (GgmlType::Q8_0, N / 32 * 34),
        (GgmlType::Q2K, 84),
        (GgmlType::Q3K, 110),
        (GgmlType::Q4K, 144),
        (GgmlType::Q5K, 176),
        (GgmlType::Q6K, 210),
    ]
    .into_iter()
    .enumerate()
    .map(|(i, (dtype, len))| tensor(&format!("t{i}"), dtype, noise(len, 0x9E37_79B9 + i as u64)))
    .collect()
}

#[test]
fn the_per_tensor_read_equals_the_whole_file_read_for_every_dtype() {
    let f = gguf_file(&every_dequantizable_dtype());
    let whole = GgufReader::from_file(f.path()).expect("the whole file");
    let file = std::fs::File::open(f.path()).expect("open");
    let len = file.metadata().expect("stat").len();
    assert_eq!(whole.tensors.len(), 12);
    for meta in &whole.tensors {
        let (want, want_shape) = whole.get_tensor_f32(&meta.name).expect("whole-file read");
        let (got, got_shape) =
            read_tensor_f32(&file, len, whole.data_offset, meta).expect("per-tensor read");
        assert_eq!(got_shape, want_shape, "{}", meta.name);
        // bit for bit: random blocks may decode to NaN, and NaN != NaN
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(
            bits(&got),
            bits(&want),
            "{} (ggml type {})",
            meta.name,
            meta.dtype
        );
    }
}

#[test]
fn a_tensor_past_the_end_of_the_file_fails_with_the_whole_file_error() {
    let f = gguf_file(&every_dequantizable_dtype());
    let full = std::fs::metadata(f.path()).expect("stat").len();
    // Cut the file inside the last tensor's data
    f.as_file().set_len(full - 40).expect("truncate");
    let whole = GgufReader::from_file(f.path()).expect("the header still parses");
    let file = std::fs::File::open(f.path()).expect("open");
    let len = file.metadata().expect("stat").len();
    let last = whole
        .tensors
        .iter()
        .max_by_key(|t| t.offset)
        .expect("a tensor");
    let want = whole
        .get_tensor_f32(&last.name)
        .expect_err("the whole-file read refuses it");
    let got = read_tensor_f32(&file, len, whole.data_offset, last).expect_err("so does this");
    assert_eq!(got.to_string(), want.to_string());
}

#[test]
fn planted_nan_inf_and_all_zero_tensors_are_each_named_as_the_whole_file_read_named_them() {
    // A spread of values: a constant tensor trips its own data-quality rule
    let ok: Vec<f32> = (0..N).map(|i| (i as f32 - 128.0) / 256.0).collect();
    let mut nan = ok.clone();
    nan[7] = f32::NAN;
    let mut inf = ok.clone();
    inf[9] = f32::INFINITY;
    let f = gguf_file(&[
        tensor("ok.weight", GgmlType::F32, f32_bytes(&ok)),
        tensor("nan.weight", GgmlType::F32, f32_bytes(&nan)),
        tensor("inf.weight", GgmlType::F32, f32_bytes(&inf)),
        tensor("zero.weight", GgmlType::F32, f32_bytes(&vec![0.0f32; N])),
    ]);
    let rosetta = RosettaStone::new();
    let report = rosetta.validate(f.path()).expect("validate");
    assert_eq!(report.tensor_count, 4);
    assert_eq!(report.total_nan_count, 1);
    assert_eq!(report.total_inf_count, 1);
    assert_eq!(report.all_zero_tensors, vec!["zero.weight".to_string()]);
    let failed: Vec<&str> = report
        .tensors
        .iter()
        .filter(|t| !t.is_valid)
        .map(|t| t.name.as_str())
        .collect();
    assert!(
        failed.contains(&"nan.weight") && failed.contains(&"inf.weight"),
        "{failed:?}"
    );
    assert!(!failed.contains(&"ok.weight"), "{failed:?}");

    // The same verdict, tensor by tensor, as the whole-file read gave
    let whole = GgufReader::from_file(f.path()).expect("the whole file");
    for tv in &report.tensors {
        let (data, _) = whole.get_tensor_f32(&tv.name).expect("whole-file read");
        let want = rosetta.compute_tensor_validation(&tv.name, &data);
        assert_eq!(
            (
                tv.is_valid,
                tv.nan_count,
                tv.inf_count,
                tv.zero_count,
                &tv.failures
            ),
            (
                want.is_valid,
                want.nan_count,
                want.inf_count,
                want.zero_count,
                &want.failures
            ),
            "{}",
            tv.name
        );
    }
}

/// The case row #3790's done_when names: validating a 2 GiB GGUF keeps the peak small. Reads,
/// not maps, carry the tensors, so the peak (VmHWM) is heap plus a few mapped pages.
#[cfg(target_os = "linux")]
#[test]
fn validating_a_2_gib_gguf_keeps_peak_rss_small() {
    use crate::format::prefix_rss_tests::{child_peak_kb, gguf_bytes, sparse, PEAK_RSS_BOUND_KB};
    let gguf = sparse(&gguf_bytes(), ".gguf");
    let hwm = child_peak_kb(
        "format::rosetta::validate_gguf_stream_tests::peak_rss_probe",
        &[(PROBE, gguf.path())],
    );
    assert!(
        hwm < PEAK_RSS_BOUND_KB,
        "peak RSS {hwm} KiB validating a 2 GiB GGUF (bound {PEAK_RSS_BOUND_KB} KiB): the whole model is being read again"
    );
}

#[cfg(target_os = "linux")]
const PROBE: &str = "APR_3790_VALIDATE_PROBE";

/// Not a test on its own: with the probe variable unset it does nothing.
#[cfg(target_os = "linux")]
#[test]
fn peak_rss_probe() {
    let Some(path) = std::env::var_os(PROBE) else {
        return;
    };
    let report = RosettaStone::new().validate(std::path::Path::new(&path));
    // Reported before the result check, so a whole-file read fails the row on its peak
    crate::format::prefix_rss_tests::report_peak();
    assert_eq!(report.expect("validate").tensor_count, 1);
}
