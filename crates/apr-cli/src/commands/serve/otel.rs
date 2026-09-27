//! TR-13 (#4568, CRUX-K-08): OTLP request tracing for `apr serve`.
//!
//! Off by default. With `OTEL_EXPORTER_OTLP_ENDPOINT` set, every inference
//! request (`POST` to a generation route) is exported as one `apr.inference`
//! span via renacer's [`RequestSpanExporter`], continuing the caller's W3C
//! `traceparent` when one is sent. `OTEL_EXPORTER_OTLP_PROTOCOL` picks
//! `http/protobuf` (default) or `http/json`.
//!
//! Token counts come only from what the handler put in its response
//! (`usage` for OpenAI routes, `prompt_eval_count`/`eval_count` for Ollama
//! routes). A streamed response carries no counts, so its span has none — the
//! attribute is omitted, never zeroed.
//!
//! Contract: `contracts/crux-K-08-v1.yaml`.

use std::sync::{Arc, OnceLock};
use std::time::SystemTime;

use axum::body::Body;
use axum::extract::{Request, State};
use axum::http::{header, Method, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use futures_util::StreamExt as _;
use renacer::request_span::{InferenceSpan, RequestSpanExporter};
use renacer::trace_context::TraceContext;

/// Largest request body the tracer will buffer to read `model` from.
const MAX_TRACED_REQUEST_BYTES: usize = 16 * 1024 * 1024;

/// Routes that run inference. Everything else (health, metrics, tensors) is
/// not traced.
const INFERENCE_ROUTES: &[&str] = &[
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/batch/completions",
    "/generate",
    "/api/generate",
    "/api/chat",
];

/// Whether a request is one `apr.inference` span.
#[must_use]
pub(crate) fn is_inference_request(method: &Method, path: &str) -> bool {
    method == Method::POST && INFERENCE_ROUTES.contains(&path)
}

static EXPORTER: OnceLock<Option<Arc<RequestSpanExporter>>> = OnceLock::new();

/// The model file this process serves. `apr.model` comes from here, not from
/// the response: the OpenAI handlers echo whatever `model` the client sent.
static SERVED_MODEL: OnceLock<String> = OnceLock::new();

/// Record the served model once, by its file (or directory) name.
pub fn set_served_model(model_path: &std::path::Path) {
    let name = model_path.file_name().map_or_else(
        || model_path.display().to_string(),
        |n| n.to_string_lossy().into_owned(),
    );
    let _ = SERVED_MODEL.set(name);
}

fn exporter() -> Option<Arc<RequestSpanExporter>> {
    EXPORTER
        .get_or_init(|| match RequestSpanExporter::from_env("apr-serve") {
            Ok(Some(e)) => {
                eprintln!(
                    "OTLP tracing: on — one apr.inference span per request to {}",
                    renacer::request_span::endpoint_from_env().unwrap_or_default()
                );
                Some(Arc::new(e))
            }
            Ok(None) => None,
            Err(e) => {
                eprintln!("OTLP tracing: OFF — {e:#}");
                None
            }
        })
        .clone()
}

/// Add request tracing to a server's router when the environment asks for
/// it; otherwise return the router unchanged.
#[must_use]
pub fn layer(router: axum::Router) -> axum::Router {
    match exporter() {
        Some(e) => router.layer(axum::middleware::from_fn_with_state(e, trace_request)),
        None => router,
    }
}

/// What the response body reported about the run.
#[derive(Debug, Default, Clone, PartialEq)]
pub(crate) struct ReportedUsage {
    pub model: Option<String>,
    pub prompt_tokens: Option<i64>,
    pub output_tokens: Option<i64>,
    pub decode_tps: Option<f64>,
}

/// Read usage from an OpenAI (`usage.*`) or Ollama (`*_count`) JSON body.
#[must_use]
pub(crate) fn reported_usage(body: &[u8]) -> ReportedUsage {
    let Ok(v) = serde_json::from_slice::<serde_json::Value>(body) else {
        return ReportedUsage::default();
    };
    let int = |p: &str| v.pointer(p).and_then(serde_json::Value::as_i64);
    let eval_count = int("/eval_count");
    // Ollama reports decode time in ns; only then is a decode rate measured.
    let decode_tps = match (eval_count, int("/eval_duration")) {
        (Some(n), Some(ns)) if ns > 0 => Some(n as f64 / (ns as f64 / 1e9)),
        _ => None,
    };
    ReportedUsage {
        model: v.get("model").and_then(|m| m.as_str()).map(str::to_string),
        prompt_tokens: int("/usage/prompt_tokens").or_else(|| int("/prompt_eval_count")),
        output_tokens: int("/usage/completion_tokens").or(eval_count),
        decode_tps,
    }
}

fn request_model(body: &[u8]) -> Option<String> {
    serde_json::from_slice::<serde_json::Value>(body)
        .ok()?
        .get("model")?
        .as_str()
        .map(str::to_string)
}

fn is_json(resp: &Response) -> bool {
    resp.headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|ct| ct.starts_with("application/json"))
}

/// Records the span when dropped — for streamed bodies, that is when the
/// stream ends or the client goes away.
struct RecordOnDrop {
    exporter: Arc<RequestSpanExporter>,
    span: InferenceSpan,
}

impl Drop for RecordOnDrop {
    fn drop(&mut self) {
        self.span.end = SystemTime::now();
        self.exporter.record(&self.span);
    }
}

async fn trace_request(
    State(exporter): State<Arc<RequestSpanExporter>>,
    req: Request,
    next: Next,
) -> Response {
    if !is_inference_request(req.method(), req.uri().path()) {
        return next.run(req).await;
    }
    let start = SystemTime::now();
    let route = req.uri().path().to_string();
    // An invalid traceparent starts a new trace, per W3C Trace Context §4.
    let parent = req
        .headers()
        .get("traceparent")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| TraceContext::parse(s).ok());
    let (parts, body) = req.into_parts();
    let Ok(bytes) = axum::body::to_bytes(body, MAX_TRACED_REQUEST_BYTES).await else {
        return StatusCode::PAYLOAD_TOO_LARGE.into_response();
    };
    let asked_model = request_model(&bytes);
    let resp = next
        .run(Request::from_parts(parts, Body::from(bytes)))
        .await;

    let mut guard = RecordOnDrop {
        exporter,
        span: InferenceSpan {
            model: SERVED_MODEL
                .get()
                .cloned()
                .or_else(|| asked_model.clone())
                .unwrap_or_else(|| "unknown".to_string()),
            request_model: asked_model,
            route,
            status_code: resp.status().as_u16(),
            prompt_tokens: None,
            output_tokens: None,
            decode_tps: None,
            start,
            end: start,
            parent,
        },
    };

    if !is_json(&resp) {
        // Streamed (SSE) or other: the span ends with the body.
        let (parts, body) = resp.into_parts();
        let stream = body.into_data_stream().map(move |chunk| {
            let _ = &guard;
            chunk
        });
        return Response::from_parts(parts, Body::from_stream(stream));
    }

    let (parts, body) = resp.into_parts();
    let bytes = axum::body::to_bytes(body, usize::MAX)
        .await
        .unwrap_or_default();
    let usage = reported_usage(&bytes);
    if SERVED_MODEL.get().is_none() {
        if let Some(m) = usage.model {
            guard.span.model = m;
        }
    }
    guard.span.prompt_tokens = usage.prompt_tokens;
    guard.span.output_tokens = usage.output_tokens;
    guard.span.decode_tps = usage.decode_tps;
    drop(guard);
    Response::from_parts(parts, Body::from(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_post_generation_routes_are_traced() {
        for p in INFERENCE_ROUTES {
            assert!(is_inference_request(&Method::POST, p), "{p}");
            assert!(!is_inference_request(&Method::GET, p), "GET {p}");
        }
        for p in ["/health", "/metrics", "/tensors", "/v1/models", "/v1/chat"] {
            assert!(!is_inference_request(&Method::POST, p), "{p}");
        }
    }

    #[test]
    fn openai_usage_is_read() {
        let body = br#"{"model":"qwen","usage":{"prompt_tokens":12,"completion_tokens":8,"total_tokens":20}}"#;
        let u = reported_usage(body);
        assert_eq!(u.model.as_deref(), Some("qwen"));
        assert_eq!(u.prompt_tokens, Some(12));
        assert_eq!(u.output_tokens, Some(8));
        assert_eq!(u.decode_tps, None, "OpenAI usage carries no decode time");
    }

    #[test]
    fn ollama_counts_give_a_measured_decode_rate() {
        let body =
            br#"{"model":"m","prompt_eval_count":5,"eval_count":40,"eval_duration":2000000000}"#;
        let u = reported_usage(body);
        assert_eq!(u.prompt_tokens, Some(5));
        assert_eq!(u.output_tokens, Some(40));
        assert_eq!(u.decode_tps, Some(20.0));
    }

    /// No counts in the body → no counts on the span (never zero).
    #[test]
    fn missing_usage_stays_absent() {
        for body in [
            &b"{}"[..],
            b"not json",
            br#"{"eval_count":3,"eval_duration":0}"#,
        ] {
            let u = reported_usage(body);
            assert_eq!(u.prompt_tokens, None);
            assert_eq!(u.decode_tps, None);
        }
    }
}
