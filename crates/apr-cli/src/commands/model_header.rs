//! Header-only reads of a model file (#3750).
//!
//! `apr qa` asked "is this GGUF?" and "which architecture is it?" by reading the
//! WHOLE model into memory. On Qwen3-30B-A3B that is 17.3 GiB per read: three
//! reads in the capability gate alone, plus a copy. RSS swung past 30 GB, and on
//! GB10's unified memory that competes with the GPU and the CI containers (the
//! 2026-09-21 15:56Z OOM killed 18 of them). None of those questions needs
//! tensor data. Each needs a few bytes or the header, and these readers never
//! touch more.

use std::io::Read;
use std::path::Path;

use aprender::format::gguf::reader::GgufReader;
use aprender::format::v2::{AprV2Header, HEADER_SIZE_V2};

/// The first GGUF header prefix tried. Measured on 21 local GGUFs (2026-09-21): headers are
/// 5.7 MiB (Qwen2.5 / Qwen3) to 10.5 MiB (Qwen3.5, a 248,320-token vocabulary), on files up
/// to 17.3 GiB, so one read covers every model on the ladder.
const GGUF_HEADER_FIRST: usize = 16 << 20;

/// The most a header read ever takes: about 24x the largest header measured. A file whose
/// header does not parse inside it is refused, never read whole.
const HEADER_CAP: usize = 256 << 20;

/// At most the first `n` bytes of `path`.
pub(crate) fn read_prefix(path: &Path, n: usize) -> std::io::Result<Vec<u8>> {
    let mut buf = Vec::with_capacity(n.min(1 << 20));
    std::fs::File::open(path)?
        .take(n as u64)
        .read_to_end(&mut buf)?;
    Ok(buf)
}

/// A GGUF header parsed from a growing prefix of the file: 16 MiB first, doubling, never
/// more than 256 MiB.
///
/// The reader holds ONLY that prefix. Its metadata and tensor table are complete, and its
/// tensor data is absent, so it answers header questions (architecture, the tensor
/// `(name, type)` table, rope/context/eps) and nothing else.
pub(crate) fn gguf_header(path: &Path) -> Result<GgufReader, String> {
    gguf_header_within(path, GGUF_HEADER_FIRST, HEADER_CAP)
}

/// [`gguf_header`] with its two limits as parameters, so the case table can exercise the
/// growth and the cap on small files.
fn gguf_header_within(path: &Path, first: usize, cap: usize) -> Result<GgufReader, String> {
    let len = std::fs::metadata(path)
        .map_err(|e| format!("cannot stat {}: {e}", path.display()))?
        .len();
    let mut n = first.min(cap);
    loop {
        let prefix =
            read_prefix(path, n).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        let whole_file = prefix.len() as u64 >= len;
        match GgufReader::from_bytes(prefix) {
            Ok(reader) => return Ok(reader),
            Err(e) if whole_file => return Err(format!("GGUF header parse failed: {e}")),
            Err(e) if n >= cap => {
                return Err(format!(
                    "no complete GGUF header in the first {cap} bytes of {} ({e}); refused rather than read whole",
                    path.display()
                ))
            },
            Err(_) => n = (n * 2).min(cap),
        }
    }
}

/// The architecture and the tensor `(name, GGML type)` table of a GGUF, from its header.
///
/// `None` when the file is not a GGUF, the header does not parse, or it names no
/// architecture. These are the same values the whole-file read produced, from the same reader.
pub(crate) fn gguf_arch_and_tensors(path: &Path) -> Option<(String, Vec<(String, u32)>)> {
    if read_prefix(path, 4).ok()?.as_slice() != b"GGUF" {
        return None;
    }
    let reader = gguf_header(path).ok()?;
    let arch = reader.architecture()?;
    let tensors = reader
        .tensors
        .into_iter()
        .map(|t| (t.name, t.dtype))
        .collect();
    Some((arch, tensors))
}

/// The bytes of an APR v2 file before its tensor data: the header, the metadata and the
/// tensor index, which is everything `AprV2Reader::from_bytes` parses. The header's own
/// `data_offset` bounds it, and so does `HEADER_CAP`.
pub(crate) fn apr_header_prefix(path: &Path) -> Result<Vec<u8>, String> {
    let head = read_prefix(path, HEADER_SIZE_V2)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let header =
        AprV2Header::from_bytes(&head).map_err(|e| format!("APR header parse failed: {e}"))?;
    let n = usize::try_from(header.data_offset)
        .ok()
        .filter(|&n| n <= HEADER_CAP)
        .ok_or_else(|| {
            format!(
                "APR data_offset {} is past the {} MiB header cap; refused rather than read whole",
                header.data_offset,
                HEADER_CAP >> 20
            )
        })?;
    read_prefix(path, n.max(HEADER_SIZE_V2))
        .map_err(|e| format!("cannot read {}: {e}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// A real two-tensor GGUF (`llama`, F32), optionally with a large metadata string so the
    /// header outgrows a small first read, then extended SPARSELY to `sparse_len` bytes: a
    /// multi-GiB model file that costs no disk.
    fn gguf_fixture(sparse_len: u64, pad_metadata: usize) -> tempfile::NamedTempFile {
        use aprender::format::gguf::{export_tensors_to_gguf, GgmlType, GgufTensor, GgufValue};
        let file = tempfile::NamedTempFile::with_suffix(".gguf").expect("temp file");
        {
            let mut w = std::io::BufWriter::new(file.as_file());
            let tensors = vec![
                GgufTensor {
                    name: "token_embd.weight".into(),
                    shape: vec![4, 8],
                    dtype: GgmlType::F32,
                    data: vec![0u8; 128],
                },
                GgufTensor {
                    name: "blk.0.attn_q.weight".into(),
                    shape: vec![8, 8],
                    dtype: GgmlType::F32,
                    data: vec![0u8; 256],
                },
            ];
            let metadata = vec![
                (
                    "general.architecture".to_string(),
                    GgufValue::String("llama".into()),
                ),
                (
                    "general.name".to_string(),
                    GgufValue::String("x".repeat(pad_metadata)),
                ),
                ("llama.block_count".to_string(), GgufValue::Uint32(1)),
            ];
            export_tensors_to_gguf(&mut w, &tensors, &metadata).expect("write GGUF");
            w.flush().expect("flush");
        }
        if sparse_len > 0 {
            file.as_file().set_len(sparse_len).expect("extend sparsely");
        }
        file
    }

    #[test]
    fn read_prefix_reads_at_most_n_bytes() {
        let mut f = tempfile::NamedTempFile::new().expect("temp file");
        f.write_all(b"0123456789").expect("write");
        assert_eq!(read_prefix(f.path(), 4).expect("read"), b"0123");
        assert_eq!(read_prefix(f.path(), 100).expect("read"), b"0123456789");
    }

    #[test]
    fn a_multi_gib_gguf_answers_from_its_header() {
        let f = gguf_fixture(2 << 30, 0);
        assert_eq!(std::fs::metadata(f.path()).expect("stat").len(), 2 << 30);
        let (arch, tensors) = gguf_arch_and_tensors(f.path()).expect("the header parses");
        assert_eq!(arch, "llama");
        assert_eq!(
            tensors,
            vec![
                ("token_embd.weight".to_string(), 0),
                ("blk.0.attn_q.weight".to_string(), 0)
            ]
        );
    }

    #[test]
    fn a_header_larger_than_the_first_read_is_found_by_growing() {
        let f = gguf_fixture(0, 5_000);
        let reader = gguf_header_within(f.path(), 1024, 1 << 20).expect("grows past 1 KiB");
        assert_eq!(reader.architecture().as_deref(), Some("llama"));
    }

    #[test]
    fn a_header_past_the_cap_is_refused_never_read_whole() {
        let f = gguf_fixture(1 << 30, 5_000);
        let e = gguf_header_within(f.path(), 1024, 4096)
            .expect_err("a 4 KiB cap cannot hold a 5 KB header");
        assert!(e.contains("refused rather than read whole"), "{e}");
    }

    #[test]
    fn a_file_that_is_not_gguf_has_no_gguf_header() {
        let mut f = tempfile::NamedTempFile::new().expect("temp file");
        f.write_all(b"APR\0 not a gguf").expect("write");
        assert!(gguf_arch_and_tensors(f.path()).is_none());
    }

    #[test]
    fn an_apr_prefix_ends_where_tensor_data_starts_and_still_parses() {
        use aprender::format::v2::{AprV2Metadata, AprV2Reader, AprV2Writer, TensorDType};
        let mut writer = AprV2Writer::new(AprV2Metadata::new("test"));
        writer.add_tensor(
            "w",
            TensorDType::F32,
            vec![256, 256],
            vec![0u8; 256 * 256 * 4],
        );
        let mut bytes = Vec::new();
        writer.write_to(&mut bytes).expect("write APR");
        let mut f = tempfile::NamedTempFile::with_suffix(".apr").expect("temp file");
        f.write_all(&bytes).expect("write");

        let prefix = apr_header_prefix(f.path()).expect("the header, metadata and index");
        let header = AprV2Header::from_bytes(&prefix).expect("header");
        assert_eq!(
            prefix.len() as u64,
            header.data_offset,
            "exactly up to the tensor data"
        );
        assert!(
            prefix.len() < bytes.len() / 100,
            "{} of {} bytes",
            prefix.len(),
            bytes.len()
        );
        let reader = AprV2Reader::from_bytes(&prefix).expect("AprV2Reader parses the prefix");
        assert_eq!(reader.tensor_names(), vec!["w"]);
    }

    /// The done_when row: on a sparse 2 GiB GGUF, the header readers' PEAK RSS stays small.
    /// Measured in a CHILD process (this test binary, re-run on the probe below), because
    /// `cargo test` shares one process between tests and its peak would be theirs. The bound
    /// is measured, not guessed: see `PEAK_RSS_BOUND_KB`.
    #[cfg(target_os = "linux")]
    #[test]
    fn header_reads_of_a_2_gib_gguf_keep_peak_rss_small() {
        let f = gguf_fixture(2 << 30, 0);
        let out = std::process::Command::new(std::env::current_exe().expect("this test binary"))
            .args([
                "--exact",
                "commands::model_header::tests::peak_rss_probe",
                "--nocapture",
                "--test-threads=1",
            ])
            .env(PROBE_ENV, f.path())
            .output()
            .expect("run the probe");
        let text = String::from_utf8_lossy(&out.stdout);
        let hwm = text
            .lines()
            .find_map(|l| l.split("PEAK_RSS_KB=").nth(1)) // libtest prints it after the test name
            .and_then(|v| v.split_whitespace().next()?.parse::<u64>().ok())
            .unwrap_or_else(|| panic!("the probe reported no peak: {text}{}", String::from_utf8_lossy(&out.stderr)));
        assert!(
            hwm < PEAK_RSS_BOUND_KB,
            "peak RSS {hwm} KiB reading the header of a 2 GiB GGUF (bound {PEAK_RSS_BOUND_KB} KiB): \
             a whole-file read is back"
        );
    }

    const PROBE_ENV: &str = "APR_3750_PEAK_RSS_PROBE";

    /// Measured on this commit (x86-64 debug test binary, 2026-09-21), the probe's peak RSS was:
    /// 44.4 MiB on this 2 GiB sparse fixture; 59.5-60.3 MiB on a real 0.5B GGUF (5.7 MiB header);
    /// 57.4-59.0 MiB on the real Qwen3-30B-A3B GGUF (17.3 GiB). With a whole-file `std::fs::read`
    /// put back into `gguf_arch_and_tensors`, it was 2,136,776 KiB (2.04 GiB) on this fixture. The
    /// bound sits about 4x above the header path and 8x below the regression.
    const PEAK_RSS_BOUND_KB: u64 = 256 * 1024;

    /// Not a test on its own: with the probe variable unset it does nothing. The test above
    /// re-runs this binary on it, in a fresh process, to read that process's peak RSS.
    #[cfg(target_os = "linux")]
    #[test]
    fn peak_rss_probe() {
        let Some(path) = std::env::var_os(PROBE_ENV) else {
            return;
        };
        let path = std::path::PathBuf::from(path);
        assert!(
            gguf_arch_and_tensors(&path).is_some(),
            "the probe's fixture has a header"
        );
        assert!(gguf_header(&path).is_ok());
        let _ = super::super::qa_capability::cpu_only_architecture(&path);
        let _ = super::super::qa_capability::hybrid_loader_architecture(&path);
        let status = std::fs::read_to_string("/proc/self/status").expect("/proc/self/status");
        let hwm = status
            .lines()
            .find_map(|l| l.strip_prefix("VmHWM:"))
            .and_then(|v| v.trim().trim_end_matches("kB").trim().parse::<u64>().ok())
            .expect("VmHWM in /proc/self/status");
        println!("PEAK_RSS_KB={hwm}");
    }
}
