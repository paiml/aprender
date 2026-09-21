//! #3761 case rows: aprender-core's header-only readers keep PEAK RSS small on 2 GiB model files.
//!
//! Each fixture is a real file (GGUF, APR v2, APR v1, SafeTensors) extended SPARSELY to 2 GiB.
//! A row runs its readers in a CHILD process (this test binary, re-run on one probe test),
//! because `cargo test` shares one process between tests and its peak would be theirs. The
//! helpers here are shared with the converter's row (`converter::export::rss_tests`).

use std::io::Write;
use std::path::Path;

/// Every row's sparse fixture is this long.
pub(crate) const FIXTURE_LEN: u64 = 2 << 30;

/// Measured (x86-64 debug test binary, 2026-09-21): the format row's peak was 36,916-38,000 KiB over
/// eight runs; the converter row's, 16,200-17,648 KiB. With a whole-file read put back into
/// any one of twelve readers (`list_tensors` on GGUF and on APR v1, lint on SafeTensors and on
/// APR, rosetta `inspect`, `is_onnx_file`, `safetensors_header_prefix`, and the converter's
/// five), its row read 2,110,308-4,218,352 KiB. Put a whole-file read
/// back into any one reader and a row reads over 2 GiB. The bound sits far between.
pub(crate) const PEAK_RSS_BOUND_KB: u64 = 256 * 1024;

/// `bytes`, then extended SPARSELY to [`FIXTURE_LEN`]: a multi-GiB model that costs no disk.
pub(crate) fn sparse(bytes: &[u8], suffix: &str) -> tempfile::NamedTempFile {
    let mut f = tempfile::NamedTempFile::with_suffix(suffix).expect("temp file");
    f.write_all(bytes).expect("write");
    f.as_file().set_len(FIXTURE_LEN).expect("extend sparsely");
    f
}

pub(crate) fn gguf_bytes() -> Vec<u8> {
    use crate::format::gguf::{export_tensors_to_gguf, GgmlType, GgufTensor, GgufValue};
    let tensors = vec![GgufTensor {
        name: "token_embd.weight".into(),
        shape: vec![4, 8],
        dtype: GgmlType::F32,
        data: vec![0u8; 128],
    }];
    let metadata = vec![(
        "general.architecture".to_string(),
        GgufValue::String("llama".into()),
    )];
    let mut bytes = Vec::new();
    export_tensors_to_gguf(&mut bytes, &tensors, &metadata).expect("write GGUF");
    bytes
}

/// An APR v2 file with an architecture and a user `source_metadata` key in its metadata, so
/// the converter's readers each have something to find.
pub(crate) fn apr_bytes() -> Vec<u8> {
    use crate::format::v2::{AprV2Metadata, AprV2Writer, TensorDType};
    let mut metadata = AprV2Metadata::new("test");
    metadata.architecture = Some("llama".to_string());
    metadata.custom.insert(
        "source_metadata".to_string(),
        serde_json::json!({ "origin": "fixture" }),
    );
    let mut writer = AprV2Writer::new(metadata);
    writer.add_tensor("w", TensorDType::F32, vec![4, 4], vec![0u8; 64]);
    let mut bytes = Vec::new();
    writer.write_to(&mut bytes).expect("write APR");
    bytes
}

/// An APR v1 file: the 32-byte header ("APRN", the metadata length at offset 8) and a JSON
/// metadata section naming one tensor shape.
pub(crate) fn apr_v1_bytes() -> Vec<u8> {
    let metadata = br#"{"tensor_shapes":{"w":[4,4]}}"#;
    let mut bytes = vec![0u8; crate::format::HEADER_SIZE];
    bytes[0..4].copy_from_slice(b"APRN");
    bytes[8..12].copy_from_slice(&(metadata.len() as u32).to_le_bytes());
    bytes.extend_from_slice(metadata);
    bytes
}

pub(crate) fn safetensors_bytes() -> Vec<u8> {
    let json = br#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
    let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(json);
    bytes.extend_from_slice(&[0u8; 4]);
    bytes
}

/// Run the probe test `probe` in a child with `envs` set, and return the peak RSS (KiB) it
/// reported. Panics, with the child's output, when it reports none.
pub(crate) fn child_peak_kb(probe: &str, envs: &[(&str, &Path)]) -> u64 {
    let mut cmd = std::process::Command::new(std::env::current_exe().expect("this test binary"));
    cmd.args(["--exact", probe, "--nocapture", "--test-threads=1"]);
    for (k, v) in envs {
        cmd.env(k, v);
    }
    let out = cmd.output().expect("run the probe");
    let text = String::from_utf8_lossy(&out.stdout);
    let hwm = text
        .lines()
        .find_map(|l| l.split("PEAK_RSS_KB=").nth(1)) // libtest prints it after the test name
        .and_then(|v| v.split_whitespace().next()?.parse::<u64>().ok())
        .unwrap_or_else(|| {
            panic!(
                "the probe {probe} reported no peak: {text}{}",
                String::from_utf8_lossy(&out.stderr)
            )
        });
    eprintln!("{probe}: peak RSS {hwm} KiB");
    hwm
}

/// The probe side: print this process's peak RSS where [`child_peak_kb`] reads it.
pub(crate) fn report_peak() {
    let status = std::fs::read_to_string("/proc/self/status").expect("/proc/self/status");
    let hwm = status
        .lines()
        .find_map(|l| l.strip_prefix("VmHWM:"))
        .and_then(|v| v.trim().trim_end_matches("kB").trim().parse::<u64>().ok())
        .expect("VmHWM in /proc/self/status");
    println!("PEAK_RSS_KB={hwm}");
}

const PROBE_GGUF: &str = "APR_3761_CORE_PROBE_GGUF";
const PROBE_APR: &str = "APR_3761_CORE_PROBE_APR";
const PROBE_APR_V1: &str = "APR_3761_CORE_PROBE_APR_V1";
const PROBE_ST: &str = "APR_3761_CORE_PROBE_ST";

#[cfg(target_os = "linux")]
#[test]
fn header_only_readers_keep_peak_rss_small_on_2_gib_models() {
    let gguf = sparse(&gguf_bytes(), ".gguf");
    let apr = sparse(&apr_bytes(), ".apr");
    let apr_v1 = sparse(&apr_v1_bytes(), ".apr");
    let st = sparse(&safetensors_bytes(), ".safetensors");
    let hwm = child_peak_kb(
        "format::prefix_rss_tests::peak_rss_probe",
        &[
            (PROBE_GGUF, gguf.path()),
            (PROBE_APR, apr.path()),
            (PROBE_APR_V1, apr_v1.path()),
            (PROBE_ST, st.path()),
        ],
    );
    assert!(
        hwm < PEAK_RSS_BOUND_KB,
        "peak RSS {hwm} KiB on 2 GiB models (bound {PEAK_RSS_BOUND_KB} KiB): a whole-file read is back"
    );
}

/// Not a test on its own: with the probe variables unset it does nothing.
#[cfg(target_os = "linux")]
#[test]
fn peak_rss_probe() {
    let (Some(gguf), Some(apr), Some(apr_v1), Some(st)) = (
        std::env::var_os(PROBE_GGUF),
        std::env::var_os(PROBE_APR),
        std::env::var_os(PROBE_APR_V1),
        std::env::var_os(PROBE_ST),
    ) else {
        return;
    };
    let (gguf, apr, apr_v1, st) = (
        Path::new(&gguf),
        Path::new(&apr),
        Path::new(&apr_v1),
        Path::new(&st),
    );
    let options = crate::format::TensorListOptions::default;
    let listed = crate::format::list_tensors(gguf, options()).expect("list the GGUF tensors");
    let listed_v1 =
        crate::format::list_tensors(apr_v1, options()).expect("list the APR v1 tensors");
    let st_header = crate::format::prefix::safetensors_header_prefix(st).expect("the header");
    crate::format::lint::lint_model_file(apr).expect("lint the APR");
    crate::format::lint::lint_model_file(st).expect("lint the SafeTensors");
    crate::format::rosetta::RosettaStone::new()
        .inspect(apr)
        .expect("inspect the APR");
    let onnx = crate::format::onnx::is_onnx_file(gguf);
    // Reported before the result checks, so a whole-file read fails the row on its peak
    report_peak();
    assert_eq!(listed.tensor_count, 1);
    assert_eq!(listed_v1.format_version, "v1");
    assert_eq!(listed_v1.tensor_count, 1);
    assert_eq!(&st_header[8..9], b"{");
    assert!(!onnx);
}
