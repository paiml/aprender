//! S3-compatible storage backend.
//!
//! Supports AWS S3, MinIO, Ceph, Cloudflare R2, Scaleway, OVH, and other
//! S3-compatible object stores.

use std::sync::Arc;

use aws_config::BehaviorVersion;
use aws_sdk_s3::{
    config::{Credentials, Region},
    primitives::ByteStream,
    Client,
};
use bytes::Bytes;
use tokio::runtime::Runtime;

use super::StorageBackend;
use crate::error::{Error, Result};

/// Configuration for S3 backend credentials.
#[derive(Debug, Clone)]
pub enum CredentialSource {
    /// Use environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY).
    Environment,
    /// Use static credentials.
    Static {
        /// Access key ID.
        access_key: String,
        /// Secret access key.
        secret_key: String,
    },
    /// Anonymous/public access.
    Anonymous,
}

/// A storage backend using S3-compatible object storage.
///
/// This backend supports AWS S3 and any S3-compatible service like MinIO,
/// Ceph, Cloudflare R2, Scaleway, OVH, Wasabi, and Backblaze B2.
///
/// # Example
///
/// ```no_run
/// use alimentar::backend::{CredentialSource, S3Backend};
///
/// // AWS S3
/// let backend = S3Backend::new(
///     "my-bucket",
///     "us-east-1",
///     None, // Use default AWS endpoint
///     CredentialSource::Environment,
/// )
/// .unwrap();
///
/// // MinIO (local)
/// let backend = S3Backend::new(
///     "datasets",
///     "us-east-1",
///     Some("http://localhost:9000".to_string()),
///     CredentialSource::Static {
///         access_key: "minioadmin".to_string(),
///         secret_key: "minioadmin".to_string(),
///     },
/// )
/// .unwrap();
/// ```
pub struct S3Backend {
    client: Client,
    bucket: String,
    runtime: Arc<Runtime>,
}

impl std::fmt::Debug for S3Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("S3Backend")
            .field("bucket", &self.bucket)
            .finish_non_exhaustive()
    }
}

impl S3Backend {
    /// Creates a new S3 backend.
    ///
    /// # Arguments
    ///
    /// * `bucket` - The S3 bucket name
    /// * `region` - AWS region (e.g., "us-east-1")
    /// * `endpoint` - Optional custom endpoint for S3-compatible services
    /// * `credentials` - Credential source for authentication
    ///
    /// # Errors
    ///
    /// Returns an error if the runtime or client cannot be created.
    pub fn new(
        bucket: impl Into<String>,
        region: impl Into<String>,
        endpoint: Option<String>,
        credentials: CredentialSource,
    ) -> Result<Self> {
        let runtime =
            Runtime::new().map_err(|e| Error::storage(format!("Failed to create runtime: {e}")))?;

        let bucket = bucket.into();
        let region = region.into();

        let client = runtime
            .block_on(async { Self::create_client(&region, endpoint, credentials).await })?;

        Ok(Self {
            client,
            bucket,
            runtime: Arc::new(runtime),
        })
    }

    async fn create_client(
        region: &str,
        endpoint: Option<String>,
        credentials: CredentialSource,
    ) -> Result<Client> {
        let region = Region::new(region.to_string());

        let mut config_loader = aws_config::defaults(BehaviorVersion::latest()).region(region);

        // Set credentials based on source
        match credentials {
            CredentialSource::Environment => {
                // Use default credential chain (env vars, config files, etc.)
            }
            CredentialSource::Static {
                access_key,
                secret_key,
            } => {
                let creds = Credentials::new(access_key, secret_key, None, None, "alimentar");
                config_loader = config_loader.credentials_provider(creds);
            }
            CredentialSource::Anonymous => {
                let creds = Credentials::new("", "", None, None, "anonymous");
                config_loader = config_loader.credentials_provider(creds);
            }
        }

        let sdk_config = config_loader.load().await;

        let mut s3_config = aws_sdk_s3::config::Builder::from(&sdk_config);

        // Set custom endpoint for S3-compatible services
        if let Some(endpoint_url) = endpoint {
            s3_config = s3_config.endpoint_url(&endpoint_url).force_path_style(true);
            // Required for MinIO and most S3-compatible services
        }

        Ok(Client::from_conf(s3_config.build()))
    }

    /// Returns the bucket name.
    pub fn bucket(&self) -> &str {
        &self.bucket
    }

    fn block_on<F: std::future::Future>(&self, future: F) -> F::Output {
        self.runtime.block_on(future)
    }
}

impl StorageBackend for S3Backend {
    fn list(&self, prefix: &str) -> Result<Vec<String>> {
        self.block_on(async {
            let mut keys = Vec::new();
            let mut continuation_token: Option<String> = None;

            loop {
                let mut request = self
                    .client
                    .list_objects_v2()
                    .bucket(&self.bucket)
                    .prefix(prefix);

                if let Some(token) = continuation_token.take() {
                    request = request.continuation_token(token);
                }

                let response = request
                    .send()
                    .await
                    .map_err(|e| Error::storage(format!("S3 list error: {e}")))?;

                if let Some(contents) = response.contents {
                    for object in contents {
                        if let Some(key) = object.key {
                            keys.push(key);
                        }
                    }
                }

                if response.is_truncated.unwrap_or(false) {
                    continuation_token = response.next_continuation_token;
                } else {
                    break;
                }
            }

            Ok(keys)
        })
    }

    fn get(&self, key: &str) -> Result<Bytes> {
        self.block_on(async {
            let response = self
                .client
                .get_object()
                .bucket(&self.bucket)
                .key(key)
                .send()
                .await
                .map_err(|e| Error::storage(format!("S3 get error for key '{}': {}", key, e)))?;

            let body = response
                .body
                .collect()
                .await
                .map_err(|e| Error::storage(format!("S3 body read error: {e}")))?;

            Ok(body.into_bytes())
        })
    }

    fn put(&self, key: &str, data: Bytes) -> Result<()> {
        self.block_on(async {
            self.client
                .put_object()
                .bucket(&self.bucket)
                .key(key)
                .body(ByteStream::from(data))
                .send()
                .await
                .map_err(|e| Error::storage(format!("S3 put error for key '{}': {}", key, e)))?;

            Ok(())
        })
    }

    fn delete(&self, key: &str) -> Result<()> {
        self.block_on(async {
            self.client
                .delete_object()
                .bucket(&self.bucket)
                .key(key)
                .send()
                .await
                .map_err(|e| Error::storage(format!("S3 delete error for key '{}': {}", key, e)))?;

            Ok(())
        })
    }

    fn exists(&self, key: &str) -> Result<bool> {
        self.block_on(async {
            match self
                .client
                .head_object()
                .bucket(&self.bucket)
                .key(key)
                .send()
                .await
            {
                Ok(_) => Ok(true),
                Err(e) => {
                    // Check if it's a "not found" error
                    let service_error = e.into_service_error();
                    if service_error.is_not_found() {
                        Ok(false)
                    } else {
                        Err(Error::storage(format!(
                            "S3 exists error for key '{}': {}",
                            key, service_error
                        )))
                    }
                }
            }
        })
    }

    fn size(&self, key: &str) -> Result<u64> {
        self.block_on(async {
            let response = self
                .client
                .head_object()
                .bucket(&self.bucket)
                .key(key)
                .send()
                .await
                .map_err(|e| Error::storage(format!("S3 head error for key '{}': {}", key, e)))?;

            let size = response
                .content_length
                .and_then(|l| u64::try_from(l).ok())
                .unwrap_or(0);
            Ok(size)
        })
    }
}

// S3Backend is automatically Send + Sync because:
// - Client is Send + Sync
// - Arc<Runtime> is Send + Sync
// - String is Send + Sync

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_credential_source_environment() {
        let creds = CredentialSource::Environment;
        assert!(matches!(creds, CredentialSource::Environment));
    }

    #[test]
    fn test_credential_source_static() {
        let creds = CredentialSource::Static {
            access_key: "test".to_string(),
            secret_key: "secret".to_string(),
        };
        assert!(matches!(creds, CredentialSource::Static { .. }));
    }

    #[test]
    fn test_credential_source_anonymous() {
        let creds = CredentialSource::Anonymous;
        assert!(matches!(creds, CredentialSource::Anonymous));
    }

    // Integration tests require a running S3-compatible service
    // See tests/s3_integration.rs for MinIO-based tests
}
