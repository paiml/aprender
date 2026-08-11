//! Falsifier for aprender#2376 finding 8 — the startup banner advertised a route
//! the served model could never use.
//!
//! `apr serve run model.gguf` printed
//! `POST /v1/predict     - Model prediction (APR)` before binding. `/v1/predict`
//! is mounted on the GGUF router, but with no `.apr` model resident it can only
//! ever answer `503`, so the banner sent every operator to a dead endpoint. The
//! same banner offered `POST /generate` for `.apr` models, where that route is
//! not mounted at all.

use std::io::Write;

use super::banner_endpoints;

/// Write `bytes` to a temp file with `suffix` and return its path.
fn fixture(suffix: &str, bytes: &[u8]) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!(
        "apr-banner-2376-{}-{}{suffix}",
        std::process::id(),
        bytes.len()
    ));
    let mut f = std::fs::File::create(&path).expect("create fixture");
    f.write_all(bytes).expect("write fixture");
    path
}

/// GGUF magic + enough bytes for the 8-byte sniff the serve path uses.
fn gguf_fixture() -> std::path::PathBuf {
    fixture(".gguf", b"GGUF\x03\x00\x00\x00padding-bytes")
}

/// APR v2 magic (`APR\0`).
fn apr_fixture() -> std::path::PathBuf {
    fixture(".apr", b"APR\x00\x02\x00\x00\x00padding")
}

#[test]
#[cfg(feature = "inference")]
fn gguf_banner_does_not_advertise_the_apr_predictor() {
    let path = gguf_fixture();
    let lines = banner_endpoints(&path, true).join("\n");
    let _ = std::fs::remove_file(&path);

    assert!(
        !lines.contains("/v1/predict"),
        "/v1/predict can only answer 503 for a GGUF model; banner was:\n{lines}"
    );
    assert!(
        lines.contains("/generate"),
        "the GGUF text-generation route must be advertised:\n{lines}"
    );
}

#[test]
#[cfg(feature = "inference")]
fn apr_banner_advertises_the_predictor_and_not_the_gguf_route() {
    let path = apr_fixture();
    let lines = banner_endpoints(&path, true).join("\n");
    let _ = std::fs::remove_file(&path);

    assert!(
        lines.contains("/v1/predict"),
        "an APR model is exactly what /v1/predict serves:\n{lines}"
    );
    assert!(
        !lines.contains("POST /generate"),
        "/generate is not mounted on the APR router:\n{lines}"
    );
}

#[test]
#[cfg(feature = "inference")]
fn banner_omits_metrics_when_metrics_are_disabled() {
    let path = gguf_fixture();
    let with = banner_endpoints(&path, true).join("\n");
    let without = banner_endpoints(&path, false).join("\n");
    let _ = std::fs::remove_file(&path);

    assert!(with.contains("/metrics"), "banner was:\n{with}");
    assert!(
        !without.contains("/metrics"),
        "--no-metrics must not advertise /metrics:\n{without}"
    );
}

/// An unreadable or unrecognised file must not make the banner guess: it prints
/// only what every router mounts.
#[test]
#[cfg(feature = "inference")]
fn unknown_format_advertises_neither_format_specific_route() {
    let path = std::path::Path::new("/nonexistent/model.bin");
    let lines = banner_endpoints(path, false).join("\n");

    assert!(!lines.contains("/v1/predict"), "banner was:\n{lines}");
    assert!(!lines.contains("POST /generate"), "banner was:\n{lines}");
    assert!(
        lines.contains("/health"),
        "/health is mounted by every router:\n{lines}"
    );
}
