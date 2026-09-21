//! Bounded reads of a model file's head: the ONE policy for "read the header, never the
//! tensor data" (#3750, #3761).
//!
//! Reading a whole model to learn its format, architecture or metadata costs its full size in
//! RSS: 17.3 GiB for Qwen3-30B-A3B, and on GB10's unified memory that competes with the GPU
//! (the 2026-09-21 OOM killed 18 CI containers). Every reader that needs only the head of a
//! file goes through here, so the limits live in one place.

use std::fmt::Display;
use std::io::Read;
use std::path::Path;

use crate::v2::{AprV2Header, HEADER_SIZE_V2};

/// The first header read. Measured on 21 local GGUFs (2026-09-21): headers are 5.7 MiB
/// (Qwen2.5 / Qwen3) to 10.5 MiB (Qwen3.5, a 248,320-token vocabulary), on files up to
/// 17.3 GiB, so one read covers every model on the ladder.
pub const HEADER_FIRST_READ: usize = 16 << 20;

/// The most a header read ever takes: about 24x the largest header measured. A header that
/// does not fit inside it is refused, and the file is never read whole.
pub const HEADER_READ_CAP: usize = 256 << 20;

/// At most the first `n` bytes of `path`.
pub fn read_prefix(path: &Path, n: usize) -> std::io::Result<Vec<u8>> {
    let mut buf = Vec::with_capacity(n.min(1 << 20));
    std::fs::File::open(path)?
        .take(n as u64)
        .read_to_end(&mut buf)?;
    Ok(buf)
}

/// Parse a header from a growing prefix of `path`: [`HEADER_FIRST_READ`] first, doubling,
/// never more than [`HEADER_READ_CAP`]. `parse` must fail on a truncated header, as every
/// bounds-checked header parser does. A parse that fails once the whole file has been read
/// is that parser's error. A parse that still fails at the cap is refused by name.
pub fn parse_growing_prefix<T, E: Display>(
    path: &Path,
    parse: impl FnMut(Vec<u8>) -> Result<T, E>,
) -> Result<T, String> {
    parse_growing_prefix_within(path, HEADER_FIRST_READ, HEADER_READ_CAP, parse)
}

/// [`parse_growing_prefix`] with its two limits as parameters (the case table uses small ones).
pub fn parse_growing_prefix_within<T, E: Display>(
    path: &Path,
    first: usize,
    cap: usize,
    mut parse: impl FnMut(Vec<u8>) -> Result<T, E>,
) -> Result<T, String> {
    let len = std::fs::metadata(path)
        .map_err(|e| format!("cannot stat {}: {e}", path.display()))?
        .len();
    let mut n = first.min(cap);
    loop {
        let prefix =
            read_prefix(path, n).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        let whole_file = prefix.len() as u64 >= len;
        match parse(prefix) {
            Ok(v) => return Ok(v),
            Err(e) if whole_file => return Err(format!("header parse failed: {e}")),
            Err(e) if n >= cap => {
                return Err(format!(
                    "no complete header in the first {cap} bytes of {} ({e}); refused rather than read whole",
                    path.display()
                ))
            }
            Err(_) => n = (n * 2).min(cap),
        }
    }
}

/// The bytes of an APR v2 file before its tensor data: the header, the metadata and the
/// tensor index, which is everything `AprV2Reader::from_bytes` parses. The header's own
/// `data_offset` bounds it, and so does [`HEADER_READ_CAP`].
pub fn apr_v2_header_prefix(path: &Path) -> Result<Vec<u8>, String> {
    let head = read_prefix(path, HEADER_SIZE_V2)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let header =
        AprV2Header::from_bytes(&head).map_err(|e| format!("APR header parse failed: {e}"))?;
    let n = usize::try_from(header.data_offset)
        .ok()
        .filter(|&n| n <= HEADER_READ_CAP)
        .ok_or_else(|| {
            format!(
                "APR data_offset {} is past the {} MiB header cap; refused rather than read whole",
                header.data_offset,
                HEADER_READ_CAP >> 20
            )
        })?;
    read_prefix(path, n.max(HEADER_SIZE_V2))
        .map_err(|e| format!("cannot read {}: {e}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// A temp file removed on drop (the leaf takes no test dependency for this).
    struct TempFile(std::path::PathBuf);
    impl TempFile {
        fn path(&self) -> &Path {
            &self.0
        }
    }
    impl Drop for TempFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn file_with(bytes: &[u8]) -> TempFile {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static N: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "apr-format-prefix-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::File::create(&path)
            .and_then(|mut f| f.write_all(bytes))
            .expect("write the temp file");
        TempFile(path)
    }

    /// A toy header: its first byte says how long the whole header is. It takes the `Vec` because
    /// `parse_growing_prefix` hands its parser ownership of each prefix.
    #[allow(clippy::needless_pass_by_value)]
    fn toy_parse(prefix: Vec<u8>) -> Result<usize, String> {
        let need = *prefix.first().ok_or("empty")? as usize;
        if prefix.len() >= need {
            Ok(need)
        } else {
            Err(format!("truncated: have {}, need {need}", prefix.len()))
        }
    }

    /// A file of `bytes`, then extended SPARSELY to `len`: a multi-GiB model that costs no disk.
    fn sparse_file_with(bytes: &[u8], len: u64) -> TempFile {
        let f = file_with(bytes);
        std::fs::OpenOptions::new()
            .write(true)
            .open(f.path())
            .and_then(|h| h.set_len(len))
            .expect("extend sparsely");
        f
    }

    fn apr_v2_bytes() -> Vec<u8> {
        use crate::v2::{AprV2Metadata, AprV2Writer, TensorDType};
        let mut writer = AprV2Writer::new(AprV2Metadata::new("test"));
        writer.add_tensor("w", TensorDType::F32, vec![4, 4], vec![0u8; 64]);
        let mut bytes = Vec::new();
        writer.write_to(&mut bytes).expect("write APR");
        bytes
    }

    const PROBE_APR: &str = "APR_3761_PROBE_APR";

    /// Measured (x86-64 debug test binary, 2026-09-21): this probe's peak RSS was 4,608 KiB on three runs.
    /// With a whole-file `std::fs::read` put back into `apr_v2_header_prefix` it was 2,094,336 KiB
    /// on the 2 GiB fixture. The bound sits far between the two. (The SafeTensors header reader
    /// lives in aprender-core, beside its format, and has its row there.)
    const PEAK_RSS_BOUND_KB: u64 = 256 * 1024;

    /// The case row the #3761 done_when names: on a sparse 2 GiB APR v2 file, the header reader
    /// keeps PEAK RSS small. Measured in a CHILD process (this test binary,
    /// re-run on the probe below), because `cargo test` shares one process between tests.
    #[cfg(target_os = "linux")]
    #[test]
    fn header_readers_keep_peak_rss_small_on_2_gib_files() {
        let apr = sparse_file_with(&apr_v2_bytes(), 2 << 30);
        let out = std::process::Command::new(std::env::current_exe().expect("this test binary"))
            .args([
                "--exact",
                "prefix::tests::peak_rss_probe",
                "--nocapture",
                "--test-threads=1",
            ])
            .env(PROBE_APR, apr.path())
            .output()
            .expect("run the probe");
        let text = String::from_utf8_lossy(&out.stdout);
        let hwm = text
            .lines()
            .find_map(|l| l.split("PEAK_RSS_KB=").nth(1))
            .and_then(|v| v.split_whitespace().next()?.parse::<u64>().ok())
            .unwrap_or_else(|| {
                panic!(
                    "the probe reported no peak: {text}{}",
                    String::from_utf8_lossy(&out.stderr)
                )
            });
        eprintln!("prefix::tests::peak_rss_probe: peak RSS {hwm} KiB");
        assert!(
            hwm < PEAK_RSS_BOUND_KB,
            "peak RSS {hwm} KiB reading the header of a 2 GiB APR file (bound {PEAK_RSS_BOUND_KB} KiB): a whole-file read is back"
        );
    }

    /// Not a test on its own: with the probe variable unset it does nothing.
    #[cfg(target_os = "linux")]
    #[test]
    fn peak_rss_probe() {
        let Some(apr) = std::env::var_os(PROBE_APR) else {
            return;
        };
        let apr = apr_v2_header_prefix(Path::new(&apr)).expect("the APR header prefix");
        assert!(crate::v2::AprV2Header::from_bytes(&apr).is_ok());
        let status = std::fs::read_to_string("/proc/self/status").expect("/proc/self/status");
        let hwm = status
            .lines()
            .find_map(|l| l.strip_prefix("VmHWM:"))
            .and_then(|v| v.trim().trim_end_matches("kB").trim().parse::<u64>().ok())
            .expect("VmHWM in /proc/self/status");
        println!("PEAK_RSS_KB={hwm}");
    }

    #[test]
    fn read_prefix_reads_at_most_n_bytes() {
        let f = file_with(b"0123456789");
        assert_eq!(read_prefix(f.path(), 4).expect("read"), b"0123");
        assert_eq!(read_prefix(f.path(), 100).expect("read"), b"0123456789");
    }

    #[test]
    fn a_header_larger_than_the_first_read_is_found_by_growing() {
        let mut bytes = vec![200u8];
        bytes.resize(4096, 0);
        let f = file_with(&bytes);
        assert_eq!(
            parse_growing_prefix_within(f.path(), 16, 1024, toy_parse),
            Ok(200)
        );
    }

    #[test]
    fn a_header_past_the_cap_is_refused_never_read_whole() {
        let mut bytes = vec![200u8];
        bytes.resize(4096, 0);
        let f = file_with(&bytes);
        let e = parse_growing_prefix_within(f.path(), 16, 64, toy_parse).expect_err("200 > 64");
        assert!(e.contains("refused rather than read whole"), "{e}");
    }

    #[test]
    fn a_parse_that_fails_on_the_whole_file_is_the_parsers_error() {
        let f = file_with(&[100u8, 1, 2]);
        let e = parse_growing_prefix_within(f.path(), 16, 1024, toy_parse).expect_err("3 < 100");
        assert!(e.starts_with("header parse failed: truncated"), "{e}");
    }
}
