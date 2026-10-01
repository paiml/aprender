//! [`LambdaDispatcher`] — start the work on another function, and the S3 upload
//! that publishes what it produced.
//!
//! # Why an async invoke and not Step Functions
//!
//! chess-mcp reaches for Step Functions because its work fans out into chunks
//! that must be scheduled, retried and gathered. A SetFit training run is ONE
//! unit of 127 measured seconds that either finishes or does not. A state
//! machine around a single state buys an execution history, an extra IAM
//! surface and a second place for the run to be defined, and pays for it with a
//! second definition of the payload. When training grows a second stage — a
//! calibration sweep, say — that is the moment the state machine earns its
//! keep, not before.
//!
//! # What this dispatcher CANNOT do, and what that costs
//!
//! `LocalDispatcher` runs `apr setfit train --dry-run` before it returns, so a
//! bad config is refused at the MCP boundary as a tool error. This dispatcher
//! cannot: the request function holds no `apr` binary, which is the entire
//! reason it fits in a deployment package. So a bad config is accepted here,
//! and the worker's own pre-flight turns it into a `failed` task a few seconds
//! later. The client sees a failure either way, carrying the same CLI refusal
//! text; what it does not get is a synchronous error. That is the price of the
//! split, and it is paid in seconds rather than in the minutes a full run would
//! take.

use std::time::Duration;

use aprender_mcp_setfit_train::{mint_id, DatasetUpload, Dispatcher, DATASET_FORMAT};
use aws_sdk_lambda::error::DisplayErrorContext;
use aws_sdk_lambda::primitives::Blob;
use aws_sdk_lambda::types::InvocationType;
use aws_sdk_s3::presigning::PresigningConfig;
use pmcp::async_trait;

use crate::dataset::{dataset_key_in, DATASET_PREFIX, DATASET_SUFFIX, MAX_DATASET_BYTES};
use crate::TrainingJob;

/// Where artifacts live in the bucket. The download presign is scoped to it.
pub const ARTIFACT_PREFIX: &str = "tasks/";

/// How long an upload URL stays valid. Long enough to run `tar` and `curl` by
/// hand with a coffee in between; short enough that a URL pasted into a chat
/// is stale by the time anyone else reads it.
pub const UPLOAD_URL_TTL: Duration = Duration::from_secs(15 * 60);

/// How long a download URL stays valid — matched to the task's own TTL, so a
/// client that polled to completion is never handed a link that outlives the
/// record it came from by much.
pub const DOWNLOAD_URL_TTL: Duration = Duration::from_secs(60 * 60);

/// Split an `s3://bucket/key` URI. `None` when it is not one.
///
/// The artifact URI is minted by [`LambdaDispatcher::artifact_uri`] and travels
/// through the envelope to the worker, which has to turn it back into a bucket
/// and a key. Parsing it there rather than passing the two halves separately
/// keeps ONE spelling of where an artifact lives — the same string the client
/// is shown in the status payload.
#[must_use]
pub fn parse_s3_uri(uri: &str) -> Option<(&str, &str)> {
    let rest = uri.strip_prefix("s3://")?;
    let (bucket, key) = rest.split_once('/')?;
    (!bucket.is_empty() && !key.is_empty()).then_some((bucket, key))
}

/// Upload a trained artifact to the URI the task's envelope names.
///
/// # Errors
///
/// A human-readable message: the URI was not an S3 one, the file could not be
/// read, or S3 refused the write.
pub async fn upload_artifact(
    s3: &aws_sdk_s3::Client,
    artifact_uri: &str,
    local_path: &std::path::Path,
) -> Result<(), String> {
    let (bucket, key) = parse_s3_uri(artifact_uri)
        .ok_or_else(|| format!("{artifact_uri} is not an s3://bucket/key URI"))?;
    let body = aws_sdk_s3::primitives::ByteStream::from_path(local_path)
        .await
        .map_err(|e| format!("cannot read {} for upload: {e}", local_path.display()))?;
    s3.put_object()
        .bucket(bucket)
        .key(key)
        .body(body)
        .send()
        .await
        // DisplayErrorContext, not `{e}`: an SdkError's own Display is
        // "service error", which names neither the bucket nor the reason.
        .map_err(|e| format!("cannot upload to {artifact_uri}: {}", DisplayErrorContext(&e)))?;
    Ok(())
}

/// Dispatches training to the worker Lambda, and signs the S3 URLs a client
/// uploads to and downloads from.
#[derive(Debug, Clone)]
pub struct LambdaDispatcher {
    lambda: aws_sdk_lambda::Client,
    s3: aws_sdk_s3::Client,
    function: String,
    bucket: String,
}

impl LambdaDispatcher {
    #[must_use]
    pub fn new(
        lambda: aws_sdk_lambda::Client,
        s3: aws_sdk_s3::Client,
        function: impl Into<String>,
        bucket: impl Into<String>,
    ) -> Self {
        Self {
            lambda,
            s3,
            function: function.into(),
            bucket: bucket.into(),
        }
    }

    /// The URI a freshly minted dataset slot is addressed by.
    fn dataset_uri_for(&self, id: &str) -> String {
        format!("s3://{}/{DATASET_PREFIX}{id}{DATASET_SUFFIX}", self.bucket)
    }

    /// Build one from the ambient AWS configuration.
    ///
    /// # Errors
    ///
    /// When [`crate::ENV_TRAINER_FUNCTION`] or [`crate::ENV_ARTIFACT_BUCKET`]
    /// is unset.
    pub async fn from_env() -> Result<Self, String> {
        let function = crate::require_env(crate::ENV_TRAINER_FUNCTION)?;
        let bucket = crate::require_env(crate::ENV_ARTIFACT_BUCKET)?;
        let config = aws_config::load_defaults(aws_config::BehaviorVersion::latest()).await;
        Ok(Self::new(
            aws_sdk_lambda::Client::new(&config),
            aws_sdk_s3::Client::new(&config),
            function,
            bucket,
        ))
    }
}

#[async_trait]
impl Dispatcher for LambdaDispatcher {
    fn artifact_uri(&self, task_id: &str) -> String {
        // The task id is the whole key: it is a random v4-shaped handle, so two
        // runs cannot collide and knowing the bucket does not let anyone guess
        // an artifact. The owner is deliberately NOT in the path — it is
        // `local` for every unauthenticated caller, so it would add a constant
        // segment while implying a scoping the bucket does not enforce.
        format!("s3://{}/tasks/{task_id}.apr", self.bucket)
    }

    async fn dataset_upload(&self) -> Result<DatasetUpload, String> {
        let id = mint_id();
        let key = format!("{DATASET_PREFIX}{id}{DATASET_SUFFIX}");
        let presign = PresigningConfig::expires_in(UPLOAD_URL_TTL)
            .map_err(|e| format!("cannot build a presigning config: {e}"))?;
        let signed = self
            .s3
            .put_object()
            .bucket(&self.bucket)
            .key(&key)
            .presigned(presign)
            .await
            .map_err(|e| format!("cannot sign an upload URL: {}", DisplayErrorContext(&e)))?;
        Ok(DatasetUpload {
            dataset_uri: self.dataset_uri_for(&id),
            upload_url: signed.uri().to_string(),
            expires_in_secs: UPLOAD_URL_TTL.as_secs(),
            method: "PUT",
            format: DATASET_FORMAT,
        })
    }

    async fn artifact_download_url(&self, artifact_uri: &str) -> Option<String> {
        // Only an artifact THIS deployment wrote: same bucket, artifact prefix.
        // A local path or a foreign bucket gets no link rather than a link that
        // 403s, and never a signature over a key the role could not read anyway.
        let (bucket, key) = crate::parse_s3_uri(artifact_uri)?;
        if bucket != self.bucket || !key.starts_with(ARTIFACT_PREFIX) {
            return None;
        }
        let presign = PresigningConfig::expires_in(DOWNLOAD_URL_TTL).ok()?;
        self.s3
            .get_object()
            .bucket(bucket)
            .key(key)
            .presigned(presign)
            .await
            .ok()
            .map(|signed| signed.uri().to_string())
    }

    async fn dispatch(
        &self,
        task_id: &str,
        owner: &str,
        _config: &serde_json::Value,
        dataset_uri: Option<&str>,
    ) -> Result<(), String> {
        // A dataset is checked HERE, synchronously, for the two things the
        // client can get wrong: a URI this deployment did not issue, and an
        // upload that never happened. Both would otherwise surface from the
        // worker as a failed task — correct, but a minute later and one hop
        // removed from the mistake.
        if let Some(uri) = dataset_uri {
            let key = dataset_key_in(&self.bucket, uri).ok_or_else(|| {
                format!(
                    "dataset_uri must be one this server issued — \
                     s3://{}/{DATASET_PREFIX}<id>{DATASET_SUFFIX} from `dataset_upload_url` — \
                     not {uri}",
                    self.bucket
                )
            })?;
            let head = self
                .s3
                .head_object()
                .bucket(&self.bucket)
                .key(&key)
                .send()
                .await
                .map_err(|e| {
                    // `is_not_found` fires only if the signer also holds
                    // s3:ListBucket on the prefix: without it S3 answers a
                    // missing key with 403, not 404, and this branch never
                    // runs. The CDK policy grants it for that reason.
                    if e.as_service_error().is_some_and(|se| se.is_not_found()) {
                        format!(
                            "nothing has been uploaded to {uri} yet; PUT the archive to the \
                                 upload_url first, then call train"
                        )
                    } else {
                        format!("cannot read {uri}: {}", DisplayErrorContext(&e))
                    }
                })?;
            let size = head.content_length().unwrap_or(0);
            if size > MAX_DATASET_BYTES {
                return Err(format!(
                    "dataset {uri} is {size} bytes, over the {MAX_DATASET_BYTES}-byte ceiling"
                ));
            }
        }

        let job = TrainingJob {
            task_id: task_id.to_string(),
            owner: owner.to_string(),
        };
        let payload = serde_json::to_vec(&job)
            .map_err(|e| format!("cannot serialize the training job: {e}"))?;
        let response = self
            .lambda
            .invoke()
            .function_name(&self.function)
            // Event, not RequestResponse: the caller is a request function
            // behind a 30-second API Gateway and the callee runs for minutes.
            // A synchronous invoke would time out the client AND bill both
            // functions for the whole run.
            .invocation_type(InvocationType::Event)
            .payload(Blob::new(payload))
            .send()
            .await
            .map_err(|e| {
                format!(
                    "cannot start the training worker: {}",
                    DisplayErrorContext(&e)
                )
            })?;
        // An Event invoke answers 202 when the request is queued. Anything else
        // means it was not, and the caller must compensate the task rather than
        // leave a client polling work nobody is doing.
        let status = response.status_code();
        if status != 202 {
            return Err(format!(
                "the training worker was not queued: Lambda answered {status} rather than 202"
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_s3_uri_splits_into_bucket_and_key() {
        assert_eq!(
            parse_s3_uri("s3://my-bucket/tasks/t-1.apr"),
            Some(("my-bucket", "tasks/t-1.apr"))
        );
    }

    #[test]
    fn anything_that_is_not_an_s3_uri_is_refused() {
        // A LOCAL dispatcher's artifact_uri is a filesystem path. Uploading to
        // one would be a silent no-op against a bucket named after a directory.
        assert!(parse_s3_uri("/tmp/setfit-out/t-1.apr").is_none());
        assert!(parse_s3_uri("s3://bucket-with-no-key").is_none());
        assert!(parse_s3_uri("s3:///key-with-no-bucket").is_none());
        assert!(parse_s3_uri("https://bucket.s3.amazonaws.com/key").is_none());
    }

    #[test]
    fn a_dataset_uri_this_deployment_minted_is_one_it_accepts() {
        // The two ends — minted by `dataset_upload`, checked by `dispatch` —
        // written from the same constants, and this is where both are visible.
        let uri = format!(
            "s3://{}/{DATASET_PREFIX}{}{DATASET_SUFFIX}",
            "bkt",
            mint_id()
        );
        assert!(dataset_key_in("bkt", &uri).is_some(), "{uri}");
    }

    #[test]
    fn the_artifact_uri_round_trips_through_the_parser() {
        // The two ends of the S3 path — minted on the request side, parsed on
        // the worker — must agree, and this is the only place both are visible.
        let uri = format!("s3://{}/tasks/{}.apr", "aprender-artifacts-dev", "t-42");
        let (bucket, key) = parse_s3_uri(&uri).expect("round trips");
        assert_eq!(bucket, "aprender-artifacts-dev");
        assert_eq!(key, "tasks/t-42.apr");
    }
}
