//! #3761 case row: `apr bench` on a GGUF detects the format from 8 bytes and maps the model
//! ONCE. The bench runs the model, so the map is its floor: realizar's map pre-faults every page
//! (MAP_POPULATE, PMAT-304), which puts the whole file in RSS. The row proves nothing is read on
//! top of that one map: it read the whole file onto the heap first, and each backend path then
//! mapped it again. The fixture's header parses and its model does not build (one tensor), so
//! the run stops after the reads under test.
//!
//! Measured (x86-64 debug, 2026-09-21): 2,123,680-2,125,672 KiB over three runs, i.e. the
//! 2,097,152 KiB map and about 27 MiB. The whole-file read put back into the format check, or
//! into the tokenizer's parse, gives 4,212,352 / 4,212,152 KiB, and so did the second map this
//! change removed (4,224,696 KiB).

use super::*;
use crate::commands::model_header::rss_probe;

#[test]
fn bench_of_a_2_gib_gguf_keeps_peak_rss_small() {
    let f = rss_probe::sparse(
        &rss_probe::gguf_with_vocab(),
        rss_probe::FIXTURE_LEN,
        ".gguf",
    );
    let hwm = rss_probe::child_peak_kb(
        "commands::bench::rss_tests::bench_peak_rss_probe",
        &[(BENCH_PROBE, f.path())],
    );
    // One map of the file, plus the margin every header row gets.
    let bound = rss_probe::FIXTURE_LEN / 1024 + rss_probe::PEAK_RSS_BOUND_KB;
    assert!(
        hwm < bound,
        "peak RSS {hwm} KiB benchmarking a 2 GiB GGUF (bound {bound} KiB, one map + margin): \
         a whole-file read or a second map is back"
    );
}

const BENCH_PROBE: &str = "APR_3761_BENCH_PROBE";

/// Not a test on its own: with the probe variable unset it does nothing.
#[test]
fn bench_peak_rss_probe() {
    let Some(path) = std::env::var_os(BENCH_PROBE) else {
        return;
    };
    let config = BenchConfig {
        quiet: true,
        ..BenchConfig::default()
    };
    let result = run_realizar_benchmark(Path::new(&path), &config);
    assert!(result.is_err(), "a one-tensor GGUF cannot be benchmarked");
    rss_probe::report_peak();
}
