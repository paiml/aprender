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

/// Digest held by [`AuthGate::deny_all`]: all zeros, which no SHA-256 preimage
/// is known to produce, so no bearer token matches it.
const DENY_ALL_HASH: [u8; 32] = [0u8; 32];

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

    /// A gate that is enabled and rejects every request. Fail-closed stand-in
    /// for a configuration that was set but unusable (#5479): the digest is
    /// not reachable from any bearer token.
    #[must_use]
    pub fn deny_all() -> Self {
        Self {
            expected_hash: Some(DENY_ALL_HASH),
        }
    }

    /// Pure core of [`Self::try_from_env`]. An empty value counts as unset.
    /// A non-empty `APR_API_KEY_HASH` must be 64 hex chars, else `Err` — a
    /// plaintext key never papers over a malformed hash.
    ///
    /// # Errors
    ///
    /// Returns the reason when `hash` is set, non-empty and not a SHA-256 hex digest.
    pub fn from_values(hash: Option<&str>, plain: Option<&str>) -> Result<Self, &'static str> {
        if let Some(hex) = hash.filter(|h| !h.is_empty()) {
            return decode_hex_32(hex).map(Self::from_hash);
        }
        match plain.filter(|p| !p.is_empty()) {
            Some(plain) => Ok(Self::from_plain_key(plain)),
            None => Ok(Self::disabled()),
        }
    }

    /// Read configuration from env. `APR_API_KEY_HASH` (hex-encoded
    /// SHA-256) wins over `APR_API_KEY` (plaintext, hashed in-process).
    /// Neither set → disabled with a single stderr warning.
    ///
    /// # Errors
    ///
    /// A set, non-empty, malformed `APR_API_KEY_HASH` refuses to start.
    pub fn try_from_env() -> Result<Self, &'static str> {
        let hash = std::env::var("APR_API_KEY_HASH").ok();
        let plain = std::env::var("APR_API_KEY").ok();
        let gate = Self::from_values(hash.as_deref(), plain.as_deref())?;
        if !gate.is_enabled() {
            eprintln!(
                "[apr serve] WARNING: no APR_API_KEY or APR_API_KEY_HASH set; HTTP routes are unauthenticated",
            );
        }
        Ok(gate)
    }

    /// [`Self::try_from_env`] for callers that cannot return an error: a
    /// malformed `APR_API_KEY_HASH` yields [`Self::deny_all`], never an open gate.
    #[must_use]
    pub fn from_env() -> Self {
        Self::try_from_env().unwrap_or_else(|reason| {
            eprintln!("[apr serve] {reason}; rejecting all requests (fail closed)");
            Self::deny_all()
        })
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

    // #5479: a bad APR_API_KEY_HASH must never yield an open gate.
    #[test]
    fn from_values_malformed_hash_is_an_error_not_a_disabled_gate() {
        for bad in ["zz", "abc", &"g".repeat(64), &"a".repeat(63), " "] {
            assert!(
                AuthGate::from_values(Some(bad), None).is_err(),
                "malformed hash {bad:?} must be refused"
            );
            assert!(
                AuthGate::from_values(Some(bad), Some("secret")).is_err(),
                "a plaintext key must not paper over malformed hash {bad:?}"
            );
        }
    }

    #[test]
    fn from_values_empty_hash_is_unset_so_the_plain_key_applies() {
        let g = AuthGate::from_values(Some(""), Some("secret")).expect("empty hash is unset");
        assert!(g.is_enabled());
        assert!(g.check_bearer(Some("Bearer secret")));
        assert!(!g.check_bearer(None));
    }

    #[test]
    fn from_values_valid_hash_wins_over_plain_key() {
        let hex = "a".repeat(64);
        let g = AuthGate::from_values(Some(&hex), Some("secret")).expect("valid hash");
        assert!(g.is_enabled());
        assert!(!g.check_bearer(Some("Bearer secret")));
    }

    #[test]
    fn from_values_neither_set_is_disabled() {
        assert!(!AuthGate::from_values(None, None).expect("ok").is_enabled());
        assert!(!AuthGate::from_values(Some(""), Some(""))
            .expect("ok")
            .is_enabled());
    }

    #[test]
    fn deny_all_gate_rejects_every_request() {
        let g = AuthGate::deny_all();
        assert!(g.is_enabled());
        for h in [None, Some("Bearer "), Some("Bearer x"), Some("garbage")] {
            assert!(!g.check_bearer(h));
        }
    }
}
