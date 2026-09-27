//! #4089: `apr serve run` with no backend flag resolved to the CPU on a cuda build
//! while `apr run` used CUDA — same binary, same file, same host (yoga, Qwen3.5-4B).
//! And an explicit request that could not engage fell back to the CPU silently.
//!
//! The table is the dispatch's own construction (`dispatch_run.rs` builds
//! `gpu_layers` / `gpu_layers_defaulted` exactly this way, pinned by
//! `the_cli_flag_is_actually_wired_into_the_config`).

use super::*;

/// What `dispatch_serve` writes into the config for these flags.
fn from_flags(
    gpu_layers: Option<&str>,
    gpu: bool,
    no_gpu: bool,
    backend: Option<&str>,
) -> ServerConfig {
    ServerConfig {
        gpu,
        no_gpu,
        backend: backend.map(str::to_string),
        gpu_layers: match gpu_layers {
            Some(v) => Some(GpuLayerRequest::parse(v).expect("valid")),
            None if gpu && !no_gpu => Some(GpuLayerRequest::All),
            None => GpuLayerRequest::serve_default(no_gpu, backend),
        },
        gpu_layers_defaulted: gpu_layers.is_none() && !(gpu && !no_gpu),
        ..ServerConfig::default()
    }
}

#[test]
fn no_flag_resolves_as_apr_run_does_and_only_an_explicit_request_refuses() {
    let cuda = cfg!(feature = "cuda");
    // (--gpu-layers, --gpu, --no-gpu, --backend) -> (wants accelerator, explicit)
    #[rustfmt::skip]
    let table: &[(Option<&str>, bool, bool, Option<&str>, bool, bool)] = &[
        // THE DEFECT: no flag. `apr run` uses CUDA on a cuda build; so must serve.
        (None,        false, false, None,         cuda,  false),
        (None,        false, true,  None,         false, false), // --no-gpu
        (None,        false, false, Some("cpu"),  false, false), // --backend cpu
        (None,        false, false, Some("wgpu"), false, false), // no CUDA layers
        (None,        false, false, Some("cuda"), true,  true),  // --backend cuda
        (None,        true,  false, None,         true,  true),  // --gpu
        (Some("all"), false, false, None,         true,  true),
        (Some("8"),   false, false, None,         true,  true),
        (Some("auto"),false, false, None,         true,  false), // fits what fits
        (Some("0"),   false, false, None,         false, false),
        (Some("all"), false, true,  None,         false, false), // --no-gpu wins
    ];
    for &(layers, gpu, no_gpu, backend, wants, explicit) in table {
        let cfg = from_flags(layers, gpu, no_gpu, backend);
        let case =
            format!("--gpu-layers {layers:?} --gpu {gpu} --no-gpu {no_gpu} --backend {backend:?}");
        assert_eq!(cfg.wants_accelerator(), wants, "wants_accelerator: {case}");
        assert_eq!(cfg.accelerator_is_explicit(), explicit, "explicit: {case}");
        let refused = cfg.refuse_unengaged_accelerator("no device").is_err();
        assert_eq!(
            refused, explicit,
            "an unengaged request refuses iff explicit: {case}"
        );
    }
}

#[test]
fn the_refusal_names_the_flag_the_reason_and_the_way_out() {
    let cfg = from_flags(None, false, false, Some("cuda"));
    let err = cfg
        .refuse_unengaged_accelerator("CUDA init failed: no device")
        .expect_err("explicit")
        .to_string();
    for needle in [
        "--backend cuda",
        "CUDA init failed: no device",
        "--no-gpu",
        "#4089",
    ] {
        assert!(err.contains(needle), "{needle:?} missing from: {err}");
    }
}
