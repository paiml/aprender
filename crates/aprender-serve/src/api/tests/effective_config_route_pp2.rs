//! PP-LLAMA-001 §12 row 6 / PP-2 falsifiers for `GET /v1/effective-config`.
//!
//! The endpoint exists so that a receipt's `provenance.server_config` is what
//! the SERVER said, not what the operator typed. Every assertion here is
//! something a client can observe over HTTP, because that is the surface the
//! harness reads.
//!
//! The three claims that matter:
//!
//! 1. The route is MOUNTED and ADVERTISED — a harness that gets a 404 records
//!    nothing, and `config_missing` (the PP-2 must-fire) would then be
//!    indistinguishable from "this server predates the field".
//! 2. `compute_class` and `backend_loaded` come from RESIDENCY, never `cfg!`.
//!    On a `--features cuda` build serving a CPU model the honest answer is
//!    `cpu`, and a `cfg!` cannot give it.
//! 3. The JSON KEY SET is identical on every build. `cuda` is `null` here and
//!    an object on a CUDA build; if the key vanished with the feature, a
//!    validator could not assert the shape once.

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use tower::util::ServiceExt;

use crate::api::{create_router, AppState};

async fn get_json(state: AppState, uri: &str) -> (StatusCode, serde_json::Value) {
    let response = create_router(state)
        .oneshot(
            Request::builder()
                .uri(uri)
                .body(Body::empty())
                .expect("build request"),
        )
        .await
        .expect("dispatch");
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read body");
    let parsed = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
    (status, parsed)
}

/// The key set every build must serve. Written out rather than derived, so a
/// field that silently disappears is a test failure and not a shrinking loop.
const REQUIRED_TOP_LEVEL_KEYS: [&str; 12] = [
    "schema_version",
    "server",
    "compute_class",
    "build_features",
    "build_features_cli",
    "backend_loaded",
    "model",
    "offload",
    "scheduler",
    "cuda",
    "kv",
    "lock_contended",
];

// ---------------------------------------------------------------------------
// Claim 1: mounted, advertised, and answering
// ---------------------------------------------------------------------------

/// PP-2 must-not-fire, at the HTTP level: the route answers on the deployment
/// `apr serve run model.gguf` builds.
#[cfg(feature = "gpu")]
#[tokio::test]
async fn effective_config_is_served_on_a_quantized_server() {
    use super::native_routes_2376::quantized_state;

    let (status, body) = get_json(quantized_state(), "/v1/effective-config").await;
    assert_eq!(
        status,
        StatusCode::OK,
        "the route must answer, got {status} with {body}"
    );
    assert_eq!(body["schema_version"].as_u64(), Some(1));
}

/// It is mounted UNCONDITIONALLY, including on a server with no model at all:
/// a harness asks what this process is before it asks it to generate anything,
/// and "no model" is an answer the endpoint has to be able to give.
#[tokio::test]
async fn effective_config_answers_on_a_model_less_server() {
    let state = AppState::demo_mock().expect("model-less AppState");
    let (status, body) = get_json(state, "/v1/effective-config").await;
    assert_eq!(status, StatusCode::OK, "got {status} with {body}");
    assert_eq!(
        body["compute_class"].as_str(),
        Some("unknown"),
        "nothing is resident, and that is a fact — not `cpu`:\n{body}"
    );
    assert_eq!(
        body["backend_loaded"].as_array().map(Vec::len),
        Some(0),
        "no backend is loaded:\n{body}"
    );
    assert_eq!(body["model"]["loaded"].as_bool(), Some(false));
}

/// The route is DERIVED from the same table `create_router` mounts, so it is
/// advertised by `GET /` and by the 404 body. A mounted-but-unadvertised route
/// is the #2376(12) defect: a client following the error message finds nothing.
#[tokio::test]
async fn effective_config_is_advertised_by_the_index_and_the_404_body() {
    let state = AppState::demo_mock().expect("model-less AppState");
    let (_, index) = get_json(state, "/").await;
    let routes = index["routes"].as_array().expect("route index");
    assert!(
        routes
            .iter()
            .any(|r| r.as_str() == Some("GET /v1/effective-config")),
        "the index must advertise the route it mounts:\n{index}"
    );

    let state = AppState::demo_mock().expect("model-less AppState");
    let (status, notfound) = get_json(state, "/v1/no-such-route").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert!(
        notfound["routes"]
            .as_array()
            .expect("404 route list")
            .iter()
            .any(|r| r.as_str() == Some("GET /v1/effective-config")),
        "the 404 body must list it too:\n{notfound}"
    );
}

// ---------------------------------------------------------------------------
// Claim 2: residency, never cfg!
// ---------------------------------------------------------------------------

/// PP-2 must-not-fire ("the CPU cell reports cpu"). This assertion holds on a
/// CUDA build too — that is the point. `compute_class` is derived from what is
/// loaded, so it cannot be right by accident on a CPU-only build and wrong on
/// the build that matters.
#[cfg(feature = "gpu")]
#[tokio::test]
async fn compute_class_is_cpu_on_a_quantized_only_state() {
    use super::native_routes_2376::quantized_state;

    let (_, body) = get_json(quantized_state(), "/v1/effective-config").await;
    assert_eq!(
        body["compute_class"].as_str(),
        Some("cpu"),
        "a quantized CPU model is `cpu` whatever this binary was built with:\n{body}"
    );
    assert_eq!(
        body["backend_loaded"].as_array().map(|v| v.len()),
        Some(1),
        "exactly one backend is resident:\n{body}"
    );
    assert_eq!(body["backend_loaded"][0].as_str(), Some("cpu"));
}

/// `compute_class` and `build_features` are DIFFERENT facts and are reported
/// separately, which is what makes the PP-2 must-fire checkable: a receipt
/// claiming `compute_class: "cuda"` with no `"cuda"` in `build_features` is a
/// build that cannot have done what it says.
#[cfg(feature = "gpu")]
#[tokio::test]
async fn build_features_are_reported_separately_from_compute_class() {
    use super::native_routes_2376::quantized_state;

    let (_, body) = get_json(quantized_state(), "/v1/effective-config").await;
    let features: Vec<&str> = body["build_features"]
        .as_array()
        .expect("build_features")
        .iter()
        .filter_map(|v| v.as_str())
        .collect();
    assert_eq!(
        features.contains(&"cuda"),
        cfg!(feature = "cuda"),
        "build_features must state what the binary CAN do:\n{body}"
    );
    // The cross-check a validator runs: the class must be producible by the build.
    let class = body["compute_class"].as_str().expect("compute_class");
    if class == "cuda" {
        assert!(
            features.contains(&"cuda"),
            "a `cuda` compute_class on a build without the cuda feature is INVALID-BUILD:\n{body}"
        );
    }
}

// ---------------------------------------------------------------------------
// Claim 3: one JSON shape on every build
// ---------------------------------------------------------------------------

/// The key set must not depend on a `cfg!`. `cuda` is present-and-null here;
/// on a CUDA build serving a CUDA model it is an object. A validator asserts
/// this list once.
#[tokio::test]
async fn effective_config_json_shape_is_identical_on_cpu_and_cuda_builds() {
    let state = AppState::demo_mock().expect("model-less AppState");
    let (_, body) = get_json(state, "/v1/effective-config").await;
    let object = body.as_object().expect("a JSON object");
    for key in REQUIRED_TOP_LEVEL_KEYS {
        assert!(
            object.contains_key(key),
            "`{key}` must be present on every build (null is fine, absent is not):\n{body}"
        );
    }
    assert_eq!(
        object.len(),
        REQUIRED_TOP_LEVEL_KEYS.len(),
        "the key set changed; update REQUIRED_TOP_LEVEL_KEYS and the receipt \
         validator together, or a reader learns of the new field from a diff:\n{body}"
    );
    // No CUDA model is resident here, on any build.
    assert!(body["cuda"].is_null(), "cuda must be null with no CUDA model");
    assert!(body["kv"].is_null(), "kv must be null with no KV-owning backend");
    assert_eq!(body["lock_contended"].as_bool(), Some(false));
}

// ---------------------------------------------------------------------------
// PP-30: the timestamp, and the ONE clock behind it
// ---------------------------------------------------------------------------

/// PP-30 must-not-fire (`timestamp_ok`): `started_utc` is RFC 3339 UTC and
/// `clock_source` names the clocks that produced it.
#[tokio::test]
async fn server_block_carries_an_rfc3339_start_and_names_its_clock() {
    let state = AppState::demo_mock().expect("model-less AppState");
    let (_, body) = get_json(state, "/v1/effective-config").await;
    let server = &body["server"];

    let started = server["started_utc"].as_str().expect("started_utc");
    let parsed =
        chrono::DateTime::parse_from_rfc3339(started).expect("started_utc must be RFC 3339");
    assert_eq!(parsed.offset().local_minus_utc(), 0, "must be UTC: {started}");

    let clock = server["clock_source"].as_str().expect("clock_source");
    assert!(
        clock.contains("CLOCK_REALTIME") && clock.contains("CLOCK_MONOTONIC"),
        "clock_source must name both clocks, got {clock}"
    );
    assert!(server["uptime_sec"].as_f64().expect("uptime_sec") >= 0.0);
    assert_eq!(server["pid"].as_u64(), Some(u64::from(std::process::id())));
    assert_eq!(server["version"].as_str(), Some(crate::VERSION));
}

/// The endpoint, `/health` and `/v1/models` read ONE clock. Three "start"
/// clocks that disagree is the CRUX-C-33 doc/mechanism mismatch this replaces:
/// a receipt cannot cite a start time that the health probe contradicts.
#[tokio::test]
async fn health_uptime_and_effective_config_share_one_clock() {
    let state = AppState::demo_mock().expect("model-less AppState");
    let (_, health) = get_json(state.clone(), "/health").await;
    let (_, config) = get_json(state.clone(), "/v1/effective-config").await;
    let (_, models) = get_json(state, "/v1/models").await;

    let health_uptime = health["uptime_sec"].as_f64().expect("health uptime");
    let config_uptime = config["server"]["uptime_sec"].as_f64().expect("config uptime");
    assert!(
        config_uptime >= health_uptime,
        "the later read must not report a smaller uptime ({config_uptime} < {health_uptime})"
    );

    let started_unix = chrono::DateTime::parse_from_rfc3339(
        config["server"]["started_utc"].as_str().expect("started_utc"),
    )
    .expect("rfc3339")
    .timestamp();
    let created = models["data"][0]["created"]
        .as_i64()
        .expect("`created` on the first model");
    assert_eq!(
        created, started_unix,
        "`/v1/models`' `created` must be the same instant the endpoint reports"
    );
}

// ---------------------------------------------------------------------------
// The model block: measured or absent
// ---------------------------------------------------------------------------

/// `ModelSourceInfo`'s rule survives the trip onto this endpoint: a field the
/// loader did not measure is `null`, never a plausible constant.
#[cfg(feature = "gpu")]
#[tokio::test]
async fn unmeasured_model_fields_are_null_not_defaults() {
    use super::native_routes_2376::quantized_state;

    let (_, body) = get_json(quantized_state(), "/v1/effective-config").await;
    let model = &body["model"];
    // This fixture attaches no ModelSourceInfo, so everything file-derived is
    // unknown. The values below are exactly the constants the metadata
    // handlers used to fabricate.
    assert!(model["path"].is_null(), "{model}");
    assert!(model["size_bytes"].is_null(), "size_bytes must not be 0:\n{model}");
    assert!(model["format"].is_null(), "format must not be \"gguf\":\n{model}");
    assert!(
        model["quantization"].is_null(),
        "quantization must not be \"Q4_K_M\":\n{model}"
    );
    assert!(
        model["context_length"].is_null(),
        "context_length must not be 4096:\n{model}"
    );
    assert!(model["content_hash"].is_null(), "{model}");
    // ...but residency IS known, and is reported.
    assert_eq!(model["loaded"].as_bool(), Some(true));
}

/// A measured `ModelSourceInfo` reaches the endpoint verbatim, so the two
/// halves of the claim above are not "everything is always null".
#[cfg(feature = "gpu")]
#[tokio::test]
async fn measured_model_fields_reach_the_endpoint() {
    use super::native_routes_2376::quantized_state;
    use crate::api::ModelSourceInfo;

    let source = ModelSourceInfo::default()
        .with_quantization("Q4_K_M")
        .with_architecture("qwen2")
        .with_context_length(1024)
        .with_model_max_context_length(32768)
        .with_parameter_count(1_500_000_000);
    let state = quantized_state().with_model_source(source);
    let (_, body) = get_json(state, "/v1/effective-config").await;
    let model = &body["model"];
    assert_eq!(model["quantization"].as_str(), Some("Q4_K_M"));
    assert_eq!(model["architecture"].as_str(), Some("qwen2"));
    assert_eq!(model["context_length"].as_u64(), Some(1024));
    assert_eq!(model["model_max_context_length"].as_u64(), Some(32768));
    assert_eq!(model["parameter_count"].as_u64(), Some(1_500_000_000));
}

// ---------------------------------------------------------------------------
// PP-14 / PP-15: the offload block
// ---------------------------------------------------------------------------

/// PP-15 must-not-fire: `--gpu-layers all` has an observable RESOLUTION on the
/// wire. The three numbers used to exist only on stdout.
#[cfg(feature = "gpu")]
#[tokio::test]
async fn offload_report_reaches_the_wire_with_its_resolution() {
    use super::native_routes_2376::quantized_state;
    use crate::api::OffloadReport;

    let state = quantized_state().with_offload_report(OffloadReport {
        gpu_layers_requested: "all".to_string(),
        gpu_layers_resolved: 28,
        gpu_layers_total: 28,
        offload_policy: "all_or_nothing",
        autofit_applied: Vec::new(),
        explicit_args: vec!["gpu_layers".to_string()],
        build_features: vec!["inference".to_string(), "cuda".to_string(), "cuda-batch".to_string()],
        build_commit: Some("abc1234".to_string()),
    });
    let (_, body) = get_json(state, "/v1/effective-config").await;

    let offload = &body["offload"];
    assert_eq!(offload["gpu_layers_requested"].as_str(), Some("all"));
    assert_eq!(offload["gpu_layers_resolved"].as_u64(), Some(28));
    assert_eq!(offload["gpu_layers_total"].as_u64(), Some(28));
    assert_eq!(offload["offload_policy"].as_str(), Some("all_or_nothing"));

    // §9 #8: the CLI's feature list — including `cuda-batch`, which realizar
    // cannot see — reaches the served process's HTTP surface.
    let cli_features: Vec<&str> = body["build_features_cli"]
        .as_array()
        .expect("build_features_cli")
        .iter()
        .filter_map(|v| v.as_str())
        .collect();
    assert!(cli_features.contains(&"cuda-batch"), "{body}");
    assert_eq!(
        body["server"]["build_commit"].as_str(),
        Some("abc1234"),
        "the launching CLI's commit fills in `server.build_commit`, which \
         realizar's own build does not set:\n{body}"
    );

    // ...and the offload block does NOT restate the feature list: one home.
    assert!(offload["build_features"].is_null(), "{offload}");
}

/// PP-14 over HTTP: the reported sets are disjoint, so the receipt-side rule
/// has something real to check.
#[cfg(feature = "gpu")]
#[tokio::test]
async fn reported_offload_satisfies_pp14() {
    use super::native_routes_2376::quantized_state;
    use crate::api::{pp14_holds, OffloadReport};

    let report = OffloadReport {
        gpu_layers_requested: "auto".to_string(),
        gpu_layers_resolved: 28,
        gpu_layers_total: 28,
        offload_policy: "all_or_nothing",
        autofit_applied: Vec::new(),
        explicit_args: vec!["context_length".to_string()],
        build_features: vec!["cuda".to_string()],
        build_commit: None,
    };
    assert!(pp14_holds(&report));

    let state = quantized_state().with_offload_report(report);
    let (_, body) = get_json(state, "/v1/effective-config").await;
    let autofit: Vec<&str> = body["offload"]["autofit_applied"]
        .as_array()
        .expect("autofit_applied")
        .iter()
        .filter_map(|v| v.as_str())
        .collect();
    let explicit: Vec<&str> = body["offload"]["explicit_args"]
        .as_array()
        .expect("explicit_args")
        .iter()
        .filter_map(|v| v.as_str())
        .collect();
    assert!(
        !autofit.iter().any(|a| explicit.contains(a)),
        "auto-fit and the operator must not both claim an argument: {autofit:?} vs {explicit:?}"
    );
}

// ---------------------------------------------------------------------------
// PP-13 / PP-24: the scheduler block
// ---------------------------------------------------------------------------

/// PP-24 needs `slots_admitted` BEFORE a band runs and `peak_in_flight` after
/// it, and it must be able to tell an uninstrumented scheduler from an idle
/// one — hence `null`, not `0`.
#[cfg(feature = "gpu")]
#[tokio::test]
async fn scheduler_report_carries_the_admission_ceiling_and_live_counters() {
    use super::native_routes_2376::quantized_state;
    use crate::api::{InFlightCounter, SchedulerReport};

    let counter = InFlightCounter::new();
    counter.enter();
    counter.enter();
    counter.leave();

    let state = quantized_state().with_scheduler_report(
        SchedulerReport {
            kind: "cuda_batch",
            max_in_flight: 11,
            window_ms: 0,
            prefill_chunk_size: None,
            token_budget: None,
            slots_admitted: 11,
            admission_ceiling_reason: "kv_budget",
            in_flight_now: None,
            peak_in_flight: None,
        },
        Some(counter),
    );
    let (_, body) = get_json(state, "/v1/effective-config").await;
    let scheduler = &body["scheduler"];
    assert_eq!(scheduler["kind"].as_str(), Some("cuda_batch"));
    assert_eq!(scheduler["slots_admitted"].as_u64(), Some(11));
    assert_eq!(scheduler["admission_ceiling_reason"].as_str(), Some("kv_budget"));
    assert_eq!(
        scheduler["in_flight_now"].as_u64(),
        Some(1),
        "the live counter must be read at GET time:\n{scheduler}"
    );
    assert_eq!(
        scheduler["peak_in_flight"].as_u64(),
        Some(2),
        "the peak is a high-water mark, not the current value:\n{scheduler}"
    );
    assert!(
        scheduler["prefill_chunk_size"].is_null(),
        "a scheduler with no chunking says null, not 0:\n{scheduler}"
    );
}

/// An UNINSTRUMENTED scheduler reports `null` counters, which is a different
/// statement from `0` and must stay different.
#[cfg(feature = "gpu")]
#[tokio::test]
async fn an_uninstrumented_scheduler_reports_null_not_zero() {
    use super::native_routes_2376::quantized_state;
    use crate::api::SchedulerReport;

    let state = quantized_state().with_scheduler_report(
        SchedulerReport {
            kind: "iteration",
            max_in_flight: 4,
            window_ms: 0,
            prefill_chunk_size: Some(256),
            token_budget: None,
            slots_admitted: 4,
            admission_ceiling_reason: "default",
            in_flight_now: None,
            peak_in_flight: None,
        },
        None,
    );
    let (_, body) = get_json(state, "/v1/effective-config").await;
    assert!(body["scheduler"]["in_flight_now"].is_null(), "{body}");
    assert!(body["scheduler"]["peak_in_flight"].is_null(), "{body}");
    assert_eq!(body["scheduler"]["prefill_chunk_size"].as_u64(), Some(256));
}
