//! Privacy tier middleware for Banco.
//!
//! Adds `X-Privacy-Tier` header to every response and, in Sovereign mode,
//! rejects requests that hint at external backends.

use axum::{
    body::Body,
    http::{HeaderValue, Request, Response, StatusCode},
    middleware::Next,
};

use crate::serve::backends::PrivacyTier;

use super::types::ErrorResponse;

/// Axum middleware function — inject privacy header and enforce sovereign gate.
pub async fn privacy_layer(
    tier: PrivacyTier,
    request: Request<Body>,
    next: Next,
) -> Result<Response<Body>, StatusCode> {
    // Sovereign mode: reject requests with an external backend hint header
    if tier == PrivacyTier::Sovereign {
        if let Some(val) = request.headers().get("x-banco-backend") {
            let hint = val.to_str().unwrap_or("");
            let is_external = ![
                "realizar",
                "ollama",
                "llamacpp",
                "llamafile",
                "candle",
                "vllm",
                "tgi",
                "localai",
            ]
            .iter()
            .any(|local| hint.eq_ignore_ascii_case(local));

            if is_external {
                let body = serde_json::to_string(&ErrorResponse::new(
                    "External backend not allowed in Sovereign mode",
                    "privacy_violation",
                    403,
                ))
                .unwrap_or_default();

                return Ok(Response::builder()
                    .status(StatusCode::FORBIDDEN)
                    .header("content-type", "application/json")
                    .header("x-privacy-tier", tier_header(tier))
                    .body(Body::from(body))
                    .expect("valid response"));
            }
        }
    }

    // Handle CORS preflight
    if request.method() == axum::http::Method::OPTIONS {
        return Ok(cors_preflight(tier));
    }

    let mut response = next.run(request).await;
    let headers = response.headers_mut();
    headers.insert("x-privacy-tier", tier_header(tier));
    // CORS headers on every response
    headers.insert("access-control-allow-origin", HeaderValue::from_static("*"));
    headers.insert(
        "access-control-allow-methods",
        HeaderValue::from_static("GET, POST, PUT, DELETE, OPTIONS"),
    );
    headers.insert(
        "access-control-allow-headers",
        HeaderValue::from_static("content-type, authorization, x-banco-backend"),
    );
    headers.insert("access-control-expose-headers", HeaderValue::from_static("x-privacy-tier"));
    Ok(response)
}

/// CORS preflight response for OPTIONS requests.
fn cors_preflight(tier: PrivacyTier) -> Response<Body> {
    Response::builder()
        .status(StatusCode::NO_CONTENT)
        .header("access-control-allow-origin", "*")
        .header("access-control-allow-methods", "GET, POST, PUT, DELETE, OPTIONS")
        .header("access-control-allow-headers", "content-type, authorization, x-banco-backend")
        .header("access-control-expose-headers", "x-privacy-tier")
        .header("access-control-max-age", "86400")
        .header("x-privacy-tier", tier_header(tier))
        .body(Body::empty())
        .expect("valid response")
}

fn tier_header(tier: PrivacyTier) -> HeaderValue {
    match tier {
        PrivacyTier::Sovereign => HeaderValue::from_static("sovereign"),
        PrivacyTier::Private => HeaderValue::from_static("private"),
        PrivacyTier::Standard => HeaderValue::from_static("standard"),
    }
}
