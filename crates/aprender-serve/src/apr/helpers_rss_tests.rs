//! #3761 case row: `is_apr_file` and `detect_format` (by magic) read 4 bytes of a 2 GiB file,
//! never the whole of it. Each ran `fs::read` on the model to look at its first four bytes.
//!
//! The probe runs in a CHILD process (this test binary, re-run on `peak_rss_probe`), because
//! `cargo test` shares one process between tests and its peak would be theirs.
//!
//! Measured (x86-64 debug, 2026-09-21): 19.9-20.3 MiB over three runs; with `fs::read` put back
//! into `is_apr_file` or `format_from_magic`, 2,113,676 / 2,112,484 KiB.

use std::io::Write;
use std::path::Path;

const PROBE: &str = "APR_3761_SERVE_MAGIC_PROBE";

/// Far above a 4-byte read, far below a 2 GiB one.
const PEAK_RSS_BOUND_KB: u64 = 256 * 1024;

#[test]
fn magic_checks_of_a_2_gib_file_keep_peak_rss_small() {
    // An APR magic, extended sparsely to 2 GiB, with no extension so `detect_format` reads it
    let mut f = tempfile::NamedTempFile::new().expect("temp file");
    f.write_all(b"APR\0").expect("write");
    f.as_file().set_len(2 << 30).expect("extend sparsely");
    let out = std::process::Command::new(std::env::current_exe().expect("this test binary"))
        .args([
            "--exact",
            "apr::helpers::rss_tests::peak_rss_probe",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(PROBE, f.path())
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
    eprintln!("apr::helpers::rss_tests::peak_rss_probe: peak RSS {hwm} KiB");
    assert!(
        hwm < PEAK_RSS_BOUND_KB,
        "peak RSS {hwm} KiB checking the magic of a 2 GiB file (bound {PEAK_RSS_BOUND_KB} KiB): a whole-file read is back"
    );
}

/// Not a test on its own: with the probe variable unset it does nothing.
#[test]
fn peak_rss_probe() {
    let Some(path) = std::env::var_os(PROBE) else {
        return;
    };
    let path = Path::new(&path);
    let is_apr = super::is_apr_file(path);
    let format = super::detect_format(path);
    let status = std::fs::read_to_string("/proc/self/status").expect("/proc/self/status");
    let hwm = status
        .lines()
        .find_map(|l| l.strip_prefix("VmHWM:"))
        .and_then(|v| v.trim().trim_end_matches("kB").trim().parse::<u64>().ok())
        .expect("VmHWM in /proc/self/status");
    println!("PEAK_RSS_KB={hwm}");
    assert!(is_apr, "the fixture carries the APR magic");
    assert_eq!(format, "apr");
}
