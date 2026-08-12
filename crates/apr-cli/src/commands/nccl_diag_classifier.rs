//! NCCL failure-diagnostics JSON classifier (CRUX-F-15).
//!
//! Pure, deterministic classifiers that discharge FALSIFY-CRUX-F-15-{001,002,003}
//! at the PARTIAL_ALGORITHM_LEVEL — algorithm-level necessary conditions on
//! the actionable JSON diagnostic emitted on stderr when an NCCL collective
//! fails (e.g. `apr train` under `TORCH_NCCL_ASYNC_ERROR_HANDLING` parity):
//!
//!   * `classify_schema` — JSON object carries all 6 required keys
//!     (`host`, `rank`, `nccl_version`, `last_op`, `code`, `suggest`)
//!     with the expected types.
//!   * `classify_exit_code` — observed exit code is >= 128 (encodes
//!     NCCL error class so schedulers can dispatch).
//!   * `classify_doc_link` — `suggest` field contains an
//!     `nvidia.com` / `github.com/NVIDIA/nccl` URL (actionable
//!     redirect into the canonical NCCL troubleshooting docs).
//!
//! Full discharge of F-15-{001,002,003} requires a live `apr train`
//! actually emitting the diagnostic on NCCL error — tracked as
//! BLOCKER-UPSTREAM-MISSING.

use serde_json::Value;

/// Required top-level keys on the CRUX-F-15 diagnostic JSON.
pub const F15_REQUIRED_KEYS: &[&str] =
    &["host", "rank", "nccl_version", "last_op", "code", "suggest"];

/// Minimum exit code that signals an NCCL-class failure (POSIX:
/// `128 + signal_or_code`). Below this and the scheduler cannot
/// distinguish NCCL faults from generic process failure.
pub const F15_MIN_EXIT_CODE: i32 = 128;

/// Substrings any one of which is sufficient to satisfy the
/// "suggest field cites NCCL doc link" gate.
pub const F15_DOC_LINK_SUBSTRINGS: &[&str] = &[
    "docs.nvidia.com",
    "developer.nvidia.com",
    "github.com/NVIDIA/nccl",
];

/// Outcome of `classify_schema`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum NcclSchemaOutcome {
    Ok,
    NotAnObject,
    MissingKey { key: &'static str },
    HostNotString,
    RankNotNonNegativeInt { got: i64 },
    NcclVersionNotString,
    LastOpNotString,
    CodeNotInt,
    SuggestNotString,
}

/// Outcome of `classify_exit_code`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NcclExitOutcome {
    Ok { code: i32 },
    BelowThreshold { got: i32, threshold: i32 },
}

/// Outcome of `classify_doc_link`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NcclDocLinkOutcome {
    Ok,
    NoDocLink { suggest: String },
}

/// Validate the JSON has all required keys with expected types.
pub fn classify_schema(body: &Value) -> NcclSchemaOutcome {
    let Some(obj) = body.as_object() else {
        return NcclSchemaOutcome::NotAnObject;
    };
    for k in F15_REQUIRED_KEYS {
        if !obj.contains_key(*k) {
            return NcclSchemaOutcome::MissingKey { key: k };
        }
    }
    if obj.get("host").and_then(Value::as_str).is_none() {
        return NcclSchemaOutcome::HostNotString;
    }
    let rank = obj.get("rank").and_then(Value::as_i64).unwrap_or(-1);
    if rank < 0 {
        return NcclSchemaOutcome::RankNotNonNegativeInt { got: rank };
    }
    if obj.get("nccl_version").and_then(Value::as_str).is_none() {
        return NcclSchemaOutcome::NcclVersionNotString;
    }
    if obj.get("last_op").and_then(Value::as_str).is_none() {
        return NcclSchemaOutcome::LastOpNotString;
    }
    if obj.get("code").and_then(Value::as_i64).is_none() {
        return NcclSchemaOutcome::CodeNotInt;
    }
    if obj.get("suggest").and_then(Value::as_str).is_none() {
        return NcclSchemaOutcome::SuggestNotString;
    }
    NcclSchemaOutcome::Ok
}

/// Verify exit code >= `threshold` (default `F15_MIN_EXIT_CODE`).
pub fn classify_exit_code(got: i32, threshold: i32) -> NcclExitOutcome {
    if got >= threshold {
        NcclExitOutcome::Ok { code: got }
    } else {
        NcclExitOutcome::BelowThreshold { got, threshold }
    }
}

/// Verify the `suggest` field contains at least one canonical NCCL doc URL.
pub fn classify_doc_link(body: &Value) -> NcclDocLinkOutcome {
    let suggest = body
        .get("suggest")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    for needle in F15_DOC_LINK_SUBSTRINGS {
        if suggest.contains(*needle) {
            return NcclDocLinkOutcome::Ok;
        }
    }
    NcclDocLinkOutcome::NoDocLink { suggest }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn good_body() -> Value {
        json!({
            "host": "node-0",
            "rank": 0,
            "peer_rank": 1,
            "nccl_version": "2.20.5",
            "cuda_devices": "0,1",
            "fabric": "ib",
            "last_op": "AllReduce",
            "code": 6,
            "suggest": "See https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html"
        })
    }

    #[test]
    fn schema_ok_on_good_body() {
        assert_eq!(classify_schema(&good_body()), NcclSchemaOutcome::Ok);
    }

    #[test]
    fn schema_rejects_not_an_object() {
        assert_eq!(
            classify_schema(&json!([1, 2])),
            NcclSchemaOutcome::NotAnObject
        );
    }

    #[test]
    fn schema_reports_missing_key() {
        let mut body = good_body();
        body.as_object_mut().expect("obj").remove("suggest");
        assert_eq!(
            classify_schema(&body),
            NcclSchemaOutcome::MissingKey { key: "suggest" }
        );
    }

    #[test]
    fn schema_rejects_negative_rank() {
        let mut body = good_body();
        body["rank"] = json!(-1);
        assert!(matches!(
            classify_schema(&body),
            NcclSchemaOutcome::RankNotNonNegativeInt { got: -1 }
        ));
    }

    #[test]
    fn schema_rejects_non_string_host() {
        let mut body = good_body();
        body["host"] = json!(123);
        assert_eq!(classify_schema(&body), NcclSchemaOutcome::HostNotString);
    }

    #[test]
    fn schema_rejects_non_int_code() {
        let mut body = good_body();
        body["code"] = json!("six");
        assert_eq!(classify_schema(&body), NcclSchemaOutcome::CodeNotInt);
    }

    #[test]
    fn exit_code_ok_at_threshold() {
        assert_eq!(
            classify_exit_code(F15_MIN_EXIT_CODE, F15_MIN_EXIT_CODE),
            NcclExitOutcome::Ok { code: 128 }
        );
    }

    #[test]
    fn exit_code_ok_above_threshold() {
        assert_eq!(
            classify_exit_code(134, F15_MIN_EXIT_CODE),
            NcclExitOutcome::Ok { code: 134 }
        );
    }

    #[test]
    fn exit_code_rejects_generic_one() {
        assert_eq!(
            classify_exit_code(1, F15_MIN_EXIT_CODE),
            NcclExitOutcome::BelowThreshold {
                got: 1,
                threshold: 128
            }
        );
    }

    #[test]
    fn doc_link_ok_on_nvidia_url() {
        assert_eq!(classify_doc_link(&good_body()), NcclDocLinkOutcome::Ok);
    }

    #[test]
    fn doc_link_ok_on_nccl_github() {
        let mut body = good_body();
        body["suggest"] = json!("Check github.com/NVIDIA/nccl/issues/123");
        assert_eq!(classify_doc_link(&body), NcclDocLinkOutcome::Ok);
    }

    #[test]
    fn doc_link_rejects_free_text() {
        let mut body = good_body();
        body["suggest"] = json!("Try restarting your training run");
        match classify_doc_link(&body) {
            NcclDocLinkOutcome::NoDocLink { suggest } => {
                assert!(suggest.contains("restarting"));
            }
            other => panic!("expected NoDocLink, got {other:?}"),
        }
    }
}
