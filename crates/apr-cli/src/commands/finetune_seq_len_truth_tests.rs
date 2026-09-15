//! T-2 (PMAT-1009, #2924, PP-066 spec §5): "`--max-seq-len` honoured or
//! refused, never clamped".
//!
//! Root cause: `execute_training_wgpu` built `WgpuInstructPipeline` with the
//! literal `512, // max_seq_len`, silently dropping the CLI flag on the wgpu
//! path while the instruct (`build_instruct_config`) and classify paths
//! already honoured it (#2247). This is a case table over the one function
//! (`effective_max_seq_len`) every path must now route through, so no path
//! can silently diverge from the other two again.
use super::*;

/// Every requested value, on every path, must come back unchanged (none of
/// the three downstream configs — `InstructConfig`, `WgpuInstructPipeline`,
/// `ClassifyConfig` — impose an upper bound; all are plain `usize` fields),
/// or the path must refuse with `CliError::ValidationFailed` (exit code 5,
/// per `crate::error::CliError::exit_code_value`) rather than silently
/// clamp. Today no path refuses, so every row expects `Ok(requested)`.
#[test]
fn effective_max_seq_len_honours_every_requested_value_on_every_path() {
    let requested_values = [256usize, 512, 1024, 2048];
    let paths = [SeqLenPath::Instruct, SeqLenPath::Wgpu, SeqLenPath::Classify];

    for &requested in &requested_values {
        for &path in &paths {
            let effective = effective_max_seq_len(Some(requested), path)
                .unwrap_or_else(|e| panic!("path {path:?} requested {requested}: {e}"));
            assert_eq!(
                effective, requested,
                "path {path:?} must honour --max-seq-len={requested}, got {effective} \
                 (a silent clamp — this is exactly the #2924 defect)"
            );
        }
    }
}

/// `requested = None` must fall back to the path's own documented default —
/// `InstructConfig::default().max_seq_len` for `Instruct`/`Wgpu` (the wgpu
/// pipeline shares the instruct default; there is no separate wgpu literal
/// once #2924 is fixed), `ClassifyConfig::default().max_seq_len` for
/// `Classify`.
#[test]
fn effective_max_seq_len_defaults_when_not_requested() {
    let instruct_default =
        entrenar::finetune::instruct_pipeline::InstructConfig::default().max_seq_len;
    let classify_default =
        entrenar::finetune::classify_pipeline::ClassifyConfig::default().max_seq_len;

    assert_eq!(
        effective_max_seq_len(None, SeqLenPath::Instruct).expect("instruct default"),
        instruct_default
    );
    assert_eq!(
        effective_max_seq_len(None, SeqLenPath::Wgpu).expect("wgpu default"),
        instruct_default,
        "wgpu must share the instruct default, not a separate literal"
    );
    assert_eq!(
        effective_max_seq_len(None, SeqLenPath::Classify).expect("classify default"),
        classify_default
    );
}

/// Registered mutation target for this ticket: if the wgpu pipeline
/// construction reverts to the literal `512` instead of calling
/// `effective_max_seq_len(max_seq_len, SeqLenPath::Wgpu)`, a request of 1024
/// or 2048 would silently train at 512. This test exists so that mutation
/// is caught at the pure-function boundary even before a live wgpu run.
#[test]
fn effective_max_seq_len_wgpu_never_clamps_to_the_old_512_literal() {
    for &requested in &[1024usize, 2048] {
        let effective = effective_max_seq_len(Some(requested), SeqLenPath::Wgpu)
            .expect("wgpu path must not error for a plain usize request");
        assert_ne!(
            effective, 512,
            "wgpu path clamped --max-seq-len={requested} down to the old 512 literal"
        );
        assert_eq!(effective, requested);
    }
}

/// If a path DOES refuse (a future hard upper bound), it must use the exit
/// code `crate::error::CliError` already assigns to `ValidationFailed` — 5 —
/// never invent a smaller/different code. Documents the contract for future
/// changes even though no path refuses today.
#[test]
fn validation_failed_exit_code_is_5_the_documented_refusal_code() {
    let err = CliError::ValidationFailed("max_seq_len too large".to_string());
    assert_eq!(err.exit_code_value(), 5);
}
