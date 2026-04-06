//! API key authentication for Banco.
//!
//! - Local mode (127.0.0.1): no auth required
//! - LAN mode (0.0.0.0): API key required via `Authorization: Bearer bk_...`
//!
//! Keys are generated at startup and stored in `~/.banco/keys.toml`.

use axum::{
    body::Body,
    http::{Request, Response, StatusCode},
    middleware::Next,
};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::Arc;

/// Authentication mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthMode {
    /// No authentication (localhost only).
    Local,
    /// API key required (LAN/remote access).
    ApiKey,
}

/// API key with scope.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKey {
    pub key: String,
    pub scope: KeyScope,
    pub created: u64,
}

/// Key scope controls which endpoints are accessible.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum KeyScope {
    /// Chat completions, models list, health.
    Chat,
    /// All of Chat + training, data, experiments.
    Train,
    /// All of Train + model load/unload, system config.
    Admin,
}

impl KeyScope {
    /// Check if this scope allows access to the given path.
    #[must_use]
    pub fn allows_path(&self, path: &str) -> bool {
        match self {
            Self::Admin => true,
            Self::Train => {
                !path.contains("/models/load")
                    && !path.contains("/models/unload")
                    && !path.contains("/config")
            }
            Self::Chat => {
                // Chat allows read-only model info but NOT load/unload/merge/pull
                let is_model_admin = path.contains("/models/load")
                    || path.contains("/models/unload")
                    || path.contains("/models/merge")
                    || path.contains("/models/pull")
                    || path.contains("/models/registry");

                !is_model_admin
                    && (path.contains("/chat")
                        || path.contains("/models")
                        || path.contains("/health")
                        || path.contains("/system")
                        || path.contains("/tokenize")
                        || path.contains("/detokenize")
                        || path.contains("/embeddings")
                        || path.contains("/prompts")
                        || path.contains("/conversations")
                        || path.contains("/tags")
                        || path.contains("/show")
                        || path.contains("/completions"))
            }
        }
    }
}

/// API key store.
#[derive(Clone)]
pub struct AuthStore {
    mode: AuthMode,
    keys: Arc<std::sync::RwLock<HashSet<String>>>,
    key_details: Arc<std::sync::RwLock<Vec<ApiKey>>>,
}

impl AuthStore {
    /// Create a store in local mode (no auth).
    #[must_use]
    pub fn local() -> Self {
        Self {
            mode: AuthMode::Local,
            keys: Arc::new(std::sync::RwLock::new(HashSet::new())),
            key_details: Arc::new(std::sync::RwLock::new(Vec::new())),
        }
    }

    /// Create a store in API key mode with a generated key.
    #[must_use]
    pub fn api_key_mode() -> (Self, String) {
        let key = generate_key();
        let mut keys = HashSet::new();
        keys.insert(key.clone());

        let detail = ApiKey { key: key.clone(), scope: KeyScope::Admin, created: epoch_secs() };

        let store = Self {
            mode: AuthMode::ApiKey,
            keys: Arc::new(std::sync::RwLock::new(keys)),
            key_details: Arc::new(std::sync::RwLock::new(vec![detail])),
        };
        (store, key)
    }

    /// Check if auth is required.
    #[must_use]
    pub fn requires_auth(&self) -> bool {
        self.mode == AuthMode::ApiKey
    }

    /// Validate a bearer token.
    #[must_use]
    pub fn validate(&self, token: &str) -> bool {
        if self.mode == AuthMode::Local {
            return true;
        }
        self.keys.read().map(|k| k.contains(token)).unwrap_or(false)
    }

    /// Validate a token against a specific path (scope-aware).
    #[must_use]
    pub fn validate_for_path(&self, token: &str, path: &str) -> bool {
        if self.mode == AuthMode::Local {
            return true;
        }
        if let Ok(details) = self.key_details.read() {
            if let Some(key) = details.iter().find(|k| k.key == token) {
                return key.scope.allows_path(path);
            }
        }
        false
    }

    /// Generate a new scoped API key.
    pub fn generate_scoped_key(&self, scope: KeyScope) -> ApiKey {
        let key_str = generate_key();
        let api_key = ApiKey { key: key_str.clone(), scope, created: epoch_secs() };
        if let Ok(mut keys) = self.keys.write() {
            keys.insert(key_str);
        }
        if let Ok(mut details) = self.key_details.write() {
            details.push(api_key.clone());
        }
        api_key
    }

    /// List all keys (redacted).
    #[must_use]
    pub fn list_keys(&self) -> Vec<ApiKeyInfo> {
        self.key_details
            .read()
            .map(|details| {
                details
                    .iter()
                    .map(|k| ApiKeyInfo {
                        prefix: k.key.chars().take(10).collect::<String>() + "...",
                        scope: k.scope,
                        created: k.created,
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get the auth mode.
    #[must_use]
    pub fn mode(&self) -> AuthMode {
        self.mode
    }

    /// Number of registered keys.
    #[must_use]
    pub fn key_count(&self) -> usize {
        self.keys.read().map(|k| k.len()).unwrap_or(0)
    }
}

/// Redacted key info for listing.
#[derive(Debug, Clone, Serialize)]
pub struct ApiKeyInfo {
    pub prefix: String,
    pub scope: KeyScope,
    pub created: u64,
}

/// Generate a random API key with `bk_` prefix.
fn generate_key() -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let seq = COUNTER.fetch_add(1, Ordering::SeqCst);

    let mut hasher = DefaultHasher::new();
    epoch_secs().hash(&mut hasher);
    std::process::id().hash(&mut hasher);
    seq.hash(&mut hasher);
    let hash = hasher.finish();
    format!("bk_{hash:016x}")
}

fn epoch_secs() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs()
}

/// Axum middleware: enforce API key auth when in ApiKey mode.
pub async fn auth_layer(
    auth_store: AuthStore,
    request: Request<Body>,
    next: Next,
) -> Result<Response<Body>, StatusCode> {
    if !auth_store.requires_auth() {
        return Ok(next.run(request).await);
    }

    // Health/probe endpoints always accessible (for load balancer probes)
    let path_str = request.uri().path();
    if path_str.starts_with("/health") {
        return Ok(next.run(request).await);
    }

    // Extract Bearer token
    let token = request
        .headers()
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "));

    let path = request.uri().path().to_string();
    match token {
        Some(t) if auth_store.validate_for_path(t, &path) => Ok(next.run(request).await),
        Some(_) => Err(StatusCode::FORBIDDEN), // valid key, wrong scope
        _ => Err(StatusCode::UNAUTHORIZED),
    }
}
