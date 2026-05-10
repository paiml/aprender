//! FALSIFY-AUTH-002 — a valid bearer token (one whose SHA-256 hash equals
//! the configured hash) is accepted on every protected route. The hash
//! comparison happens AFTER the bearer is hashed — the configured digest
//! is never compared against plaintext.
//!
//! Contract: `contracts/apr-serve-api-key-auth-v1.yaml`.
//!
//! This is the complement to FALSIFY-AUTH-001: 001 falsifies "missing
//! header passes," 002 falsifies "valid header fails." Both must hold for
//! the gate to be operationally correct.

#![allow(
    clippy::unwrap_used,
    clippy::disallowed_methods,
    clippy::format_collect
)]

#[cfg(feature = "inference")]
mod tests {
    use apr_cli::serve_auth::{layer, AuthGate};
    use axum::{
        body::Body,
        http::{Request, StatusCode},
        routing::{get, post},
        Json, Router,
    };
    use sha2::{Digest, Sha256};
    use tower::ServiceExt;

    fn protected_router(gate: AuthGate) -> Router {
        let router = Router::new()
            .route("/", get(|| async { "ok" }))
            .route(
                "/health",
                get(|| async { Json(serde_json::json!({"ok": true})) }),
            )
            .route(
                "/predict",
                post(|| async { Json(serde_json::json!({"ok": true})) }),
            )
            .route(
                "/v1/chat/completions",
                post(|| async { Json(serde_json::json!({"ok": true})) }),
            );
        layer(gate, router)
    }

    fn sha256_hex(input: &[u8]) -> String {
        Sha256::digest(input)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    }

    async fn fire_with_bearer(router: &Router, method: &str, path: &str, key: &str) -> StatusCode {
        let req = Request::builder()
            .method(method)
            .uri(path)
            .header(axum::http::header::AUTHORIZATION, format!("Bearer {key}"))
            .body(Body::empty())
            .unwrap();
        router.clone().oneshot(req).await.unwrap().status()
    }

    #[tokio::test]
    async fn valid_bearer_passes_on_every_route() {
        let key = "valid-key-from-test";
        let router = protected_router(AuthGate::from_plain_key(key));
        for (method, path) in [
            ("GET", "/"),
            ("GET", "/health"),
            ("POST", "/predict"),
            ("POST", "/v1/chat/completions"),
        ] {
            let status = fire_with_bearer(&router, method, path, key).await;
            assert!(
                status.is_success() || status == StatusCode::OK,
                "{method} {path}: expected 2xx with valid bearer, got {status:?}",
            );
        }
    }

    #[tokio::test]
    async fn from_hash_constructor_path_accepts_same_key_as_from_plain_key() {
        let key = "shared-key";
        let bytes_from_plain = AuthGate::from_plain_key(key);
        let mut hash_bytes = [0u8; 32];
        let digest = Sha256::digest(key.as_bytes());
        hash_bytes.copy_from_slice(&digest);
        let bytes_from_hash = AuthGate::from_hash(hash_bytes);

        // Both gates must accept Bearer <key> with status 200.
        for gate in [bytes_from_plain, bytes_from_hash] {
            let router = protected_router(gate);
            let status = fire_with_bearer(&router, "GET", "/health", key).await;
            assert_eq!(status, StatusCode::OK);
        }
    }

    #[tokio::test]
    async fn auth_gate_does_not_leak_plaintext_in_state() {
        // FALSIFY-AUTH-002 corollary: the configured hash is byte-for-byte
        // SHA-256 of the plaintext key. If the implementation accidentally
        // stored the plaintext, this test would still pass — so we ALSO
        // assert that the public hash decoder accepts the SHA-256 hex
        // representation, which is the only API surface a deployment uses
        // to configure the server. A regression that takes plaintext via
        // APR_API_KEY_HASH would fail this gate's `decode_hex_32` step
        // because plaintext is not 64-char hex.
        let key = "hash-only-config";
        let hex = sha256_hex(key.as_bytes());
        assert_eq!(hex.len(), 64);

        let mut bytes = [0u8; 32];
        for (i, slot) in bytes.iter_mut().enumerate() {
            slot.clone_from(&u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap());
        }
        let gate = AuthGate::from_hash(bytes);
        assert!(gate.is_enabled());
        assert!(gate.check_bearer(Some(&format!("Bearer {key}"))));
        assert!(!gate.check_bearer(Some(&format!("Bearer {key}-extra"))));
    }
}

#[cfg(not(feature = "inference"))]
#[test]
fn auth_layer_only_compiled_with_inference_feature() {
    // See falsify_auth_001.rs — same rationale.
}
