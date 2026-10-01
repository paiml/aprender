//! HELIX-IDEA-009 — single-key bearer-token auth for `apr serve`.
//!
//! Contract: `contracts/apr-serve-api-key-auth-v1.yaml`
//! Pattern source: `helix-db/src/helix_gateway/key_verification.rs`
//! (re-implemented; no code lift).
//!
//! Comparison goes through `subtle::ConstantTimeEq` over fixed-length
//! SHA-256 digests so a probing attacker cannot leak the configured key
//! by timing the response.

use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;

/// Configured authentication state for an `apr serve` instance.
///
/// `expected_hash == None` means auth is disabled — every request passes.
/// The startup path emits a stderr warning when this happens so operators
/// see the open door at boot.
#[derive(Clone, Debug, Default)]
pub struct AuthGate {
    expected_hash: Option<[u8; 32]>,
}

impl AuthGate {
    /// Construct an explicitly-disabled gate. Test fixtures use this; the
    /// production startup path uses [`Self::from_env`].
    #[must_use]
    pub fn disabled() -> Self {
        Self {
            expected_hash: None,
        }
    }

    /// Construct an enabled gate from a SHA-256 digest of the expected
    /// bearer token. Tests use this to avoid touching env vars.
    #[must_use]
    pub fn from_hash(expected_hash: [u8; 32]) -> Self {
        Self {
            expected_hash: Some(expected_hash),
        }
    }

    /// Construct an enabled gate from a plaintext API key.
    #[must_use]
    pub fn from_plain_key(key: &str) -> Self {
        Self::from_hash(sha256_32(key.as_bytes()))
    }

    /// Read configuration from env. `APR_API_KEY_HASH` (hex-encoded
    /// SHA-256) wins over `APR_API_KEY` (plaintext, hashed in-process).
    /// Neither set → disabled with a single stderr warning.
    #[must_use]
    pub fn from_env() -> Self {
        if let Ok(hex) = std::env::var("APR_API_KEY_HASH") {
            match decode_hex_32(&hex) {
                Ok(bytes) => return Self::from_hash(bytes),
                Err(reason) => {
                    eprintln!(
                        "[apr serve] APR_API_KEY_HASH set but {reason}; ignoring (auth disabled)",
                    );
                    return Self::disabled();
                }
            }
        }
        if let Ok(plain) = std::env::var("APR_API_KEY") {
            if !plain.is_empty() {
                return Self::from_plain_key(&plain);
            }
        }
        eprintln!(
            "[apr serve] WARNING: no APR_API_KEY or APR_API_KEY_HASH set; HTTP routes are unauthenticated",
        );
        Self::disabled()
    }

    /// True iff the gate has a configured hash.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.expected_hash.is_some()
    }

    /// Verify a presented `Authorization` header value.
    ///
    /// Returns `true` iff the gate is disabled OR the header is exactly
    /// `Bearer <key>` and `sha256(key) == expected_hash` (constant-time).
    /// All other shapes — missing header, wrong scheme, empty token —
    /// return `false`.
    #[must_use]
    pub fn check_bearer(&self, header: Option<&str>) -> bool {
        let Some(expected) = self.expected_hash.as_ref() else {
            return true;
        };
        let Some(value) = header else {
            return false;
        };
        let Some(token) = value.strip_prefix("Bearer ") else {
            return false;
        };
        let presented = sha256_32(token.as_bytes());
        bool::from(expected.ct_eq(&presented))
    }
}

fn sha256_32(input: &[u8]) -> [u8; 32] {
    let digest = Sha256::digest(input);
    let mut out = [0u8; 32];
    out.copy_from_slice(&digest);
    out
}

fn decode_hex_32(hex: &str) -> Result<[u8; 32], &'static str> {
    if hex.len() != 64 {
        return Err("APR_API_KEY_HASH must be 64 hex chars (SHA-256)");
    }
    let bytes = hex.as_bytes();
    let mut out = [0u8; 32];
    for (i, slot) in out.iter_mut().enumerate() {
        let hi = hex_digit(bytes[i * 2])?;
        let lo = hex_digit(bytes[i * 2 + 1])?;
        *slot = (hi << 4) | lo;
    }
    Ok(out)
}

fn hex_digit(b: u8) -> Result<u8, &'static str> {
    match b {
        b'0'..=b'9' => Ok(b - b'0'),
        b'a'..=b'f' => Ok(b - b'a' + 10),
        b'A'..=b'F' => Ok(b - b'A' + 10),
        _ => Err("APR_API_KEY_HASH must contain only [0-9a-fA-F]"),
    }
}

/// Axum middleware closure that rejects unauthenticated requests with
/// `401 Unauthorized` + `WWW-Authenticate: Bearer` and a JSON envelope.
///
/// Wired into the per-router builder via
/// `axum::middleware::from_fn_with_state(Arc::new(gate), apply)`. The
/// gate is shared (Arc) so all routes on a router observe the same
/// configuration.
#[cfg(feature = "inference")]
pub async fn apply(
    axum::extract::State(gate): axum::extract::State<std::sync::Arc<AuthGate>>,
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    use axum::http::{header, HeaderValue, StatusCode};
    use axum::response::IntoResponse;

    let header_value = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok());

    if gate.check_bearer(header_value) {
        return next.run(req).await;
    }

    let body = axum::Json(serde_json::json!({
        "error": "unauthorized",
        "message": "Missing or invalid Authorization: Bearer <key> header"
    }));
    let mut resp = (StatusCode::UNAUTHORIZED, body).into_response();
    resp.headers_mut()
        .insert(header::WWW_AUTHENTICATE, HeaderValue::from_static("Bearer"));
    resp
}

/// Layer the auth gate onto an axum `Router`, independent of the
/// router's own state type. Callsites in each router builder use this
/// to share one implementation. The gate is wrapped in `Arc` once so
/// every route on the router observes the same snapshot, even after
/// post-startup env-var changes.
#[cfg(feature = "inference")]
#[must_use]
/// Gate everything except CORS preflight and the operational health endpoints.
///
/// [`layer`] gates every route, which is right for the APR server. The SetFit server mounts the
/// SHARED router, and that router carries `/health`, `/health/live` and `/health/ready` — the
/// last of which exists so an orchestrator can admit a classifier — plus it sits behind
/// `CorsLayer::permissive()`. Wrapping all of it produced two operational breaks (WR-07): a
/// browser's `OPTIONS /v1/classify` preflight was answered 401 with no `Access-Control-Allow-Origin`
/// before CORS ever saw it, and every liveness/readiness probe 401'd.
///
/// The two exemptions are narrow and carry no model data:
/// - `OPTIONS` is the CORS preflight. It is defined to travel WITHOUT credentials, so requiring a
///   bearer token on it cannot be satisfied by any conforming client; it returns headers, no body.
/// - `/health*` is the probe surface. Note `/health/ready` reports `classifier_artifact_sha256`
///   and `classifier_verified`; both were fully public before the classify surface was gated at
///   all, so this is not a widening.
///
/// `POST /v1/classify` — the only route that runs a model — is gated exactly as before.
pub fn layer_public_ops<S>(gate: AuthGate, router: axum::Router<S>) -> axum::Router<S>
where
    S: Clone + Send + Sync + 'static,
{
    router.layer(axum::middleware::from_fn_with_state(
        std::sync::Arc::new(gate),
        apply_except_public_ops,
    ))
}

/// [`apply`], with the two exemptions [`layer_public_ops`] documents.
pub async fn apply_except_public_ops(
    state: axum::extract::State<std::sync::Arc<AuthGate>>,
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    let is_preflight = req.method() == axum::http::Method::OPTIONS;
    let is_health = req.uri().path() == "/health" || req.uri().path().starts_with("/health/");
    if is_preflight || is_health {
        return next.run(req).await;
    }
    apply(state, req, next).await
}

pub fn layer<S>(gate: AuthGate, router: axum::Router<S>) -> axum::Router<S>
where
    S: Clone + Send + Sync + 'static,
{
    router.layer(axum::middleware::from_fn_with_state(
        std::sync::Arc::new(gate),
        apply,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_gate_accepts_anything() {
        let g = AuthGate::disabled();
        assert!(g.check_bearer(None));
        assert!(g.check_bearer(Some("Bearer anything")));
        assert!(g.check_bearer(Some("garbage")));
        assert!(!g.is_enabled());
    }

    #[test]
    fn enabled_gate_rejects_missing_header() {
        let g = AuthGate::from_plain_key("s3cr3t");
        assert!(!g.check_bearer(None));
    }

    #[test]
    fn enabled_gate_rejects_wrong_scheme() {
        let g = AuthGate::from_plain_key("s3cr3t");
        assert!(!g.check_bearer(Some("Basic dXNlcjpwYXNz")));
        assert!(!g.check_bearer(Some("Bearer")));
    }

    #[test]
    fn enabled_gate_accepts_correct_bearer() {
        let g = AuthGate::from_plain_key("s3cr3t");
        assert!(g.check_bearer(Some("Bearer s3cr3t")));
    }

    #[test]
    fn enabled_gate_rejects_wrong_bearer() {
        let g = AuthGate::from_plain_key("s3cr3t");
        assert!(!g.check_bearer(Some("Bearer wrong")));
    }

    #[test]
    fn from_hash_matches_from_plain_key_for_same_secret() {
        let plain = "another-secret";
        let g_plain = AuthGate::from_plain_key(plain);
        let g_hash = AuthGate::from_hash(sha256_32(plain.as_bytes()));
        assert!(g_plain.check_bearer(Some(&format!("Bearer {plain}"))));
        assert!(g_hash.check_bearer(Some(&format!("Bearer {plain}"))));
    }

    #[test]
    fn decode_hex_32_round_trip() {
        let bytes = sha256_32(b"hello");
        let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
        let decoded = decode_hex_32(&hex).unwrap();
        assert_eq!(decoded, bytes);
    }

    #[test]
    fn decode_hex_32_rejects_wrong_length() {
        assert!(decode_hex_32("deadbeef").is_err());
        assert!(decode_hex_32(&"a".repeat(63)).is_err());
        assert!(decode_hex_32(&"a".repeat(65)).is_err());
    }

    #[test]
    fn decode_hex_32_rejects_non_hex_char() {
        let mut bad = "0".repeat(64);
        bad.replace_range(0..1, "Z");
        assert!(decode_hex_32(&bad).is_err());
    }

    /// WR-07: the SetFit gate must let CORS preflight and health probes through, and must still
    /// refuse an unauthenticated request to a model route.
    ///
    /// These drive the REAL middleware through a real router, not `check_bearer` in isolation —
    /// the defect being guarded against was entirely about layer position and route scope, which
    /// a unit test on the gate could not have seen.
    mod public_ops {
        use super::super::{layer_public_ops, AuthGate};
        use axum::body::Body;
        use axum::http::{Method, Request, StatusCode};
        use tower::ServiceExt as _;

        fn gated_router() -> axum::Router {
            let router = axum::Router::new()
                .route("/v1/classify", axum::routing::post(|| async { "ok" }))
                .route("/health", axum::routing::get(|| async { "ok" }))
                .route("/health/ready", axum::routing::get(|| async { "ok" }));
            layer_public_ops(AuthGate::from_hash([7_u8; 32]), router)
        }

        async fn status(method: Method, path: &str) -> StatusCode {
            let request = Request::builder()
                .method(method)
                .uri(path)
                .body(Body::empty())
                .expect("request builds");
            gated_router()
                .oneshot(request)
                .await
                .expect("router answers")
                .status()
        }

        #[tokio::test]
        async fn an_unauthenticated_classify_is_still_refused() {
            assert_eq!(
                status(Method::POST, "/v1/classify").await,
                StatusCode::UNAUTHORIZED,
                "the only route that runs a model must stay gated — that is the whole point of \
                 the gate, and the exemptions must not have widened it"
            );
        }

        #[tokio::test]
        async fn a_cors_preflight_reaches_the_cors_layer() {
            assert_ne!(
                status(Method::OPTIONS, "/v1/classify").await,
                StatusCode::UNAUTHORIZED,
                "a preflight travels WITHOUT credentials by definition, so gating it makes the \
                 route unreachable from any conforming browser"
            );
        }

        #[tokio::test]
        async fn health_probes_are_not_gated() {
            for path in ["/health", "/health/ready"] {
                assert_eq!(
                    status(Method::GET, path).await,
                    StatusCode::OK,
                    "{path} must answer an orchestrator: /health/ready exists so a classifier \
                     can be admitted, and a 401 there stalls the rollout"
                );
            }
        }
    }
}
