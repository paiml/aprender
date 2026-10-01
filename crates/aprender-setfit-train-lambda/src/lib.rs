//! The AWS ends of the training server's seams, plus the configuration both
//! Lambda entry points read.
//!
//! `aprender-mcp-setfit-train` defines three seams and fills each with a
//! process-local implementation. This crate fills the same three with AWS ones
//! and adds nothing else:
//!
//! | Seam | Local | Here |
//! |------|-------|------|
//! | `TaskBackend` | `InMemoryTaskBackend` | [`DynamoDbTaskBackend`] |
//! | `Dispatcher` | `LocalDispatcher` (spawn a child) | [`LambdaDispatcher`] (async invoke) |
//! | terminal write | the spawning process | the worker binary |
//! | dataset in | a directory on the machine | an S3 object the client PUT to a presigned URL |
//! | artifact out | a file on the machine | a presigned GET, minted by `train_status` at read time |
//!
//! The tool surface, the task store, the mint handoff, the status payload and
//! the ONE implementation of "run `apr setfit train`" all stay upstream. If
//! something here starts to look like a second version of one of those, it is
//! wrong.
//!
//! # Why two functions
//!
//! pmcp.run fronts its Lambdas with an API Gateway **HTTP API**, whose 30-second
//! integration timeout cannot be raised, and Lambda freezes the execution
//! environment the moment a handler returns. A 127-second training run
//! therefore cannot happen inside the request — not inline, and not as a
//! background task the handler leaves running. So the request function mints
//! the task, writes the envelope and asynchronously invokes a second function
//! that has 15 minutes and 6 GB; the client polls `tasks/get`.
//!
//! # What is NOT solved here: owner scoping without auth
//!
//! With no auth provider configured, every caller shares ONE owner bucket. On
//! pmcp.run that bucket is `unknown`, not `UNAUTHENTICATED_OWNER` — the
//! platform supplies an auth context whose subject is the literal `"unknown"`
//! even when `[auth] enabled = false` (measured on the deployed server,
//! 2026-09-04), so `resolve_owner` takes its authenticated arm. Task ids are unguessable, so nobody stumbles
//! onto another run's result — but `tasks/list` enumerates the bucket, which
//! means it enumerates everyone. That is acceptable for a single-tenant pilot
//! and NOT acceptable for a shared deployment; the fix is to configure an auth
//! provider, after which both this crate and pmcp's own create gate scope to the
//! authenticated subject with no code change. Stated here rather than left for
//! someone to discover.

mod dataset;
mod dispatch;
mod dynamodb;

pub use dataset::{fetch_dataset, DATASET_PREFIX, DATASET_SUFFIX, MAX_DATASET_BYTES};
pub use dispatch::{
    parse_s3_uri, upload_artifact, LambdaDispatcher, ARTIFACT_PREFIX, DOWNLOAD_URL_TTL,
    UPLOAD_URL_TTL,
};
pub use dynamodb::{DynamoDbTaskBackend, MAX_ITEM_BYTES};

/// The DynamoDB table holding task records. Set by the CDK stack on the worker
/// and by `.pmcp/deploy-train.toml` on the request function; both must name the
/// SAME table, or a client polls a task the worker never sees.
pub const ENV_TASKS_TABLE: &str = "APRENDER_SETFIT_TASKS_TABLE";

/// The S3 bucket trained artifacts are uploaded to.
pub const ENV_ARTIFACT_BUCKET: &str = "APRENDER_SETFIT_ARTIFACT_BUCKET";

/// The worker function the request side invokes. A name or a full ARN — the
/// Lambda API accepts either, and the CDK publishes both to SSM.
pub const ENV_TRAINER_FUNCTION: &str = "APRENDER_SETFIT_TRAINER_FUNCTION";

/// Read a required variable, or say which one is missing.
///
/// Deliberately fatal at STARTUP rather than at the first request: a function
/// that boots without its table name only fails once someone is waiting on it,
/// and the error it produces then names DynamoDB rather than the deployment.
///
/// # Errors
///
/// A message naming the variable.
pub fn require_env(name: &str) -> Result<String, String> {
    std::env::var(name)
        .map_err(|_| format!("{name} is unset; this deployment cannot run without it"))
}

/// The invoke payload the request function sends and the worker receives.
///
/// Deliberately just the two identifiers. The run's CONFIG travels in the
/// task's envelope instead, for the reason chess-mcp records about Step
/// Functions execution history: a dispatch payload is retained and readable by
/// anyone holding the right read permission on the dispatch mechanism, while
/// the envelope is owner-scoped, TTL-bounded task state.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingJob {
    /// The task the worker must finish.
    pub task_id: String,
    /// The owner that task belongs to — every store read is scoped by it.
    pub owner: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_job_payload_carries_no_config() {
        let job = TrainingJob {
            task_id: "t-1".to_string(),
            owner: "local".to_string(),
        };
        let json = serde_json::to_value(&job).expect("serializes");
        // Sorted, because this workspace enables serde_json's `preserve_order`
        // and the wire order is not what is being asserted — the ABSENCE of a
        // third key is.
        let mut keys: Vec<&String> = json.as_object().expect("object").keys().collect();
        keys.sort();
        assert_eq!(keys, ["owner", "task_id"], "{json}");
    }

    #[test]
    fn an_unexpected_payload_field_is_refused() {
        // A config smuggled into the dispatch payload must not be silently
        // accepted — the envelope is the only door for it.
        serde_json::from_str::<TrainingJob>(
            r#"{"task_id":"t-1","owner":"local","config":{"shots":8}}"#,
        )
        .expect_err("deny_unknown_fields must refuse a config in the payload");
    }

    #[test]
    fn a_missing_variable_names_itself() {
        let err = require_env("APRENDER_SETFIT_A_VARIABLE_NOBODY_SETS")
            .expect_err("unset must be an error");
        assert!(
            err.contains("APRENDER_SETFIT_A_VARIABLE_NOBODY_SETS"),
            "{err}"
        );
    }
}
