//! PP-LLAMA-001 PP-14 / PP-15 / §9 #8: what the loader resolved, as a value.
//!
//! `apr serve run --gpu-layers all model.gguf` computed three numbers —
//! requested, resolved, total — and PRINTED them, next to a `backend=` label
//! that was a `cfg!`. Nothing carried them into the served process, so a
//! receipt's `gpu_layers_resolved` had no producer and its `feature_set` was
//! whatever the operator typed at `--server-feature`.
//!
//! These tests pin the pure function that turns a `ServerConfig` and a resolved
//! layer count into the report the server publishes at
//! `GET /v1/effective-config`.

use realizar::api::pp14_holds;

use super::{cli_build_features, offload_report, types::DEFAULT_CONTEXT_LENGTH};
use crate::commands::serve::{GpuLayerRequest, ServerConfig};

fn config() -> ServerConfig {
    ServerConfig::default()
}

// ---------------------------------------------------------------------------
// PP-15: the resolution is reported, not just printed
// ---------------------------------------------------------------------------

/// `--gpu-layers all` has an observable resolution, and the report carries all
/// three numbers plus the policy that produced them.
#[test]
fn all_resolves_to_every_layer_and_says_so() {
    let mut c = config();
    c.gpu_layers = Some(GpuLayerRequest::All);
    let report = offload_report(&c, 28, 28);
    assert_eq!(report.gpu_layers_requested, "all");
    assert_eq!(report.gpu_layers_resolved, 28);
    assert_eq!(report.gpu_layers_total, 28);
    assert_eq!(
        report.offload_policy, "all_or_nothing",
        "the policy is STATED, because `fits == total_layers` is a limitation \
         and a reader must not read `resolved == total` as a fitting decision"
    );
}

/// `--gpu-layers 0` is an explicit CPU request, and it must be visible as such:
/// this is the case a boolean `--gpu` could never express.
#[test]
fn zero_layers_is_reported_as_an_explicit_cpu_request() {
    let mut c = config();
    c.gpu_layers = Some(GpuLayerRequest::None);
    let report = offload_report(&c, 0, 28);
    assert_eq!(
        report.gpu_layers_requested, "0",
        "the user's own word, quoted back"
    );
    assert_eq!(report.gpu_layers_resolved, 0);
    assert_eq!(report.gpu_layers_total, 28);
    assert!(report.explicit_args.contains(&"gpu_layers".to_string()));
}

/// No flag at all is `"none"`, never `"0"` — "the operator said nothing" and
/// "the operator asked for CPU" are different facts.
#[test]
fn an_absent_flag_is_none_not_zero() {
    let report = offload_report(&config(), 0, 28);
    assert_eq!(report.gpu_layers_requested, "none");
    assert!(
        !report.explicit_args.contains(&"gpu_layers".to_string()),
        "nothing was set, so nothing is explicit: {:?}",
        report.explicit_args
    );
}

// ---------------------------------------------------------------------------
// PP-14: explicit_args is what the operator SET
// ---------------------------------------------------------------------------

/// Each flag the operator set appears; each one left at its default does not.
/// A mutation that reports every argument as explicit, or none, breaks this.
#[test]
fn explicit_args_lists_exactly_the_arguments_that_differ_from_the_default() {
    let report = offload_report(&config(), 0, 28);
    assert!(
        report.explicit_args.is_empty(),
        "a default config set nothing: {:?}",
        report.explicit_args
    );

    let mut c = config();
    c.gpu_layers = Some(GpuLayerRequest::All);
    c.no_gpu = true;
    c.context_length = DEFAULT_CONTEXT_LENGTH * 2;
    c.batch = true;
    c.no_fp8_cache = true;
    c.backend = Some("cuda".to_string());
    let report = offload_report(&c, 0, 28);
    for expected in [
        "gpu_layers",
        "no_gpu",
        "context_length",
        "batch",
        "no_fp8_cache",
        "backend",
    ] {
        assert!(
            report.explicit_args.contains(&expected.to_string()),
            "`{expected}` was set and must be reported: {:?}",
            report.explicit_args
        );
    }
    assert_eq!(report.explicit_args.len(), 6);
}

/// A context length EQUAL to the default is indistinguishable from an unset
/// one, and is reported as not explicit — the conservative direction, since
/// PP-14 is violated by CLAIMING something was set.
#[test]
fn a_typed_default_context_length_is_not_reported_as_explicit() {
    let mut c = config();
    c.context_length = DEFAULT_CONTEXT_LENGTH;
    let report = offload_report(&c, 0, 28);
    assert!(!report.explicit_args.contains(&"context_length".to_string()));
}

/// PP-14 must-not-fire: this loader has no auto-fit, so `autofit_applied` is
/// empty and the invariant holds — for every config, not just a lucky one.
#[test]
fn autofit_ok() {
    for (layers, no_gpu, ctx) in [
        (None, false, DEFAULT_CONTEXT_LENGTH),
        (Some(GpuLayerRequest::All), false, 1024),
        (Some(GpuLayerRequest::None), true, DEFAULT_CONTEXT_LENGTH),
        (Some(GpuLayerRequest::Auto), false, 32768),
    ] {
        let mut c = config();
        c.gpu_layers = layers;
        c.no_gpu = no_gpu;
        c.context_length = ctx;
        let report = offload_report(&c, 28, 28);
        assert!(
            report.autofit_applied.is_empty(),
            "this loader cannot auto-fit: `fits == total_layers` and a partial \
             request is REFUSED, so there is nothing for auto-fit to have changed"
        );
        assert!(pp14_holds(&report), "PP-14 must hold: {report:?}");
    }
}

/// PP-14 must-fire, on the shape this producer emits: a report claiming
/// auto-fit chose the very argument the operator pinned is REFUSED. Built by
/// mutating a real report, so the fixture cannot drift from the producer.
#[test]
fn autofit_override() {
    let mut c = config();
    c.gpu_layers = Some(GpuLayerRequest::All);
    let mut report = offload_report(&c, 28, 28);
    assert!(pp14_holds(&report), "control: the real report holds");

    report.autofit_applied.push("gpu_layers".to_string());
    assert!(
        !pp14_holds(&report),
        "auto-fit and the operator both claiming `gpu_layers` must be REFUSED: \
         the run cannot be reproduced from its own receipt"
    );
}

// ---------------------------------------------------------------------------
// §9 #8: the build's feature set travels with the report
// ---------------------------------------------------------------------------

/// `realizar` cannot see `apr-cli`'s features, so the CLI states them. The list
/// must agree with what this binary was actually built with — a hardcoded list
/// would satisfy every other assertion and record a lie.
#[test]
fn cli_build_features_agree_with_cfg() {
    let features = cli_build_features();
    assert_eq!(
        features.contains(&"inference".to_string()),
        cfg!(feature = "inference")
    );
    assert_eq!(
        features.contains(&"cuda".to_string()),
        cfg!(feature = "cuda")
    );
    assert_eq!(
        features.contains(&"cuda-batch".to_string()),
        cfg!(feature = "cuda-batch"),
        "§2.1 INVALID-BUILD reads this entry, so it must be measured, not assumed"
    );
    assert_eq!(
        features.contains(&"wgpu".to_string()),
        cfg!(feature = "wgpu")
    );
    assert_eq!(
        features.contains(&"training".to_string()),
        cfg!(feature = "training")
    );
}

/// The report carries the feature list and this binary's commit, which is what
/// fills in `server.build_commit` on a server whose own build recorded none.
#[test]
fn the_report_carries_the_build_identity() {
    let report = offload_report(&config(), 0, 28);
    assert_eq!(report.build_features, cli_build_features());
    let commit = report
        .build_commit
        .expect("apr-cli's build records a commit");
    assert!(
        !commit.is_empty() && commit != "unknown",
        "a build commit must be a commit, not the string `unknown`: {commit}"
    );
}
