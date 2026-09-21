//! #3761 case row: the converter's APR readers (`read_apr_metadata`, `extract_user_metadata`,
//! `detect_apr_quantization`, `detect_apr_architecture_for_completeness`,
//! `extract_apr_tokenizer_hint`) read a 2 GiB APR v2 file's header, metadata and tensor index,
//! and never its tensor data. Each read the whole model.
//!
//! Measured (x86-64 debug, 2026-09-21): 16,200-17,648 KiB over eight runs. A whole-file read put
//! back into any one of the five: 2,110,308-4,208,572 KiB.

use super::*;
use crate::format::prefix_rss_tests::{child_peak_kb, report_peak, sparse, PEAK_RSS_BOUND_KB};

const PROBE: &str = "APR_3761_CONVERTER_PROBE";

#[test]
fn converter_apr_readers_keep_peak_rss_small_on_a_2_gib_model() {
    let apr = sparse(&crate::format::prefix_rss_tests::apr_bytes(), ".apr");
    let hwm = child_peak_kb(
        "format::converter::export::rss_tests::peak_rss_probe",
        &[(PROBE, apr.path())],
    );
    assert!(
        hwm < PEAK_RSS_BOUND_KB,
        "peak RSS {hwm} KiB reading a 2 GiB APR's metadata (bound {PEAK_RSS_BOUND_KB} KiB): \
         a whole-file read is back"
    );
}

/// Not a test on its own: with the probe variable unset it does nothing.
#[test]
fn peak_rss_probe() {
    let Some(path) = std::env::var_os(PROBE) else {
        return;
    };
    let path = Path::new(&path);
    let metadata = read_apr_metadata(path);
    let user = extract_user_metadata(path);
    let quantization = detect_apr_quantization(path);
    let arch = detect_apr_architecture_for_completeness(path);
    let hint = extract_apr_tokenizer_hint(path);
    // Reported before the result checks, so a whole-file read fails the row on its peak
    report_peak();
    assert_eq!(
        metadata.and_then(|m| m.architecture).as_deref(),
        Some("llama")
    );
    assert_eq!(user.get("origin").map(String::as_str), Some("fixture"));
    assert!(quantization.is_none(), "an F32 model is not quantized");
    assert_eq!(arch, Some("llama"));
    // The v2 writer zero-pads its metadata: no terminator, so no hint (see the reader)
    assert!(hint.is_none());
}
