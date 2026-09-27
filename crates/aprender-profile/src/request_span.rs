//! TR-13 (#4568, CRUX-K-08): one OTLP span per inference request.
//!
//! [`crate::otlp_exporter::OtlpExporter`] models a traced *process*: one root
//! span, gRPC only, no per-request attributes. An inference server needs the
//! opposite — one `apr.inference` span per request, carrying `gen_ai.*`
//! attributes and continuing the caller's W3C `traceparent`. That is this
//! module. It is off unless `OTEL_EXPORTER_OTLP_ENDPOINT` (or the
//! traces-specific `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`) is set.
//!
//! Attributes are only ever *observed* values: a token count the server did
//! not report is omitted, never written as zero.
//!
//! Contract: `contracts/crux-K-08-v1.yaml`.

use std::time::{Duration, SystemTime};

use anyhow::{anyhow, bail, Context as _, Result};
use opentelemetry::trace::{
    SpanContext, SpanKind, Status, TraceContextExt, TraceFlags, TraceState, Tracer,
    TracerProvider as _,
};
use opentelemetry::{Context, KeyValue};
use opentelemetry_otlp::{Protocol, WithExportConfig};
use opentelemetry_sdk::trace::{
    BatchConfigBuilder, BatchSpanProcessor, SdkTracerProvider, SpanExporter,
};
use opentelemetry_sdk::Resource;

use crate::trace_context::TraceContext;

/// The span name every inference request is exported under.
pub const INFERENCE_SPAN_NAME: &str = "apr.inference";

/// `gen_ai.system` value for spans emitted by `apr`.
pub const GEN_AI_SYSTEM: &str = "apr";

/// Batch delay used unless `OTEL_BSP_SCHEDULE_DELAY` is set. The SDK default
/// (5 s) makes a single curl look like it exported nothing.
const DEFAULT_SCHEDULE_DELAY: Duration = Duration::from_millis(500);

/// What one inference request looked like, as observed by the server.
#[derive(Debug, Clone, PartialEq)]
pub struct InferenceSpan {
    /// Model the server actually ran (`apr.model`).
    pub model: String,
    /// Model the client asked for (`gen_ai.request.model`); falls back to
    /// `model` when the request did not name one.
    pub request_model: Option<String>,
    /// HTTP route that served the request (`http.route`).
    pub route: String,
    /// HTTP status code returned.
    pub status_code: u16,
    /// Prompt tokens, if the server reported them.
    pub prompt_tokens: Option<i64>,
    /// Generated tokens, if the server reported them.
    pub output_tokens: Option<i64>,
    /// Decode throughput, if the server measured it.
    pub decode_tps: Option<f64>,
    /// Wall-clock request start.
    pub start: SystemTime,
    /// Wall-clock request end.
    pub end: SystemTime,
    /// Caller's W3C trace context; `None` starts a new trace.
    pub parent: Option<TraceContext>,
}

/// The span attributes for `span`. Absent observations produce no attribute.
#[must_use]
pub fn inference_attributes(span: &InferenceSpan) -> Vec<KeyValue> {
    let request_model = span.request_model.as_deref().unwrap_or(&span.model);
    let mut attrs = vec![
        KeyValue::new("gen_ai.system", GEN_AI_SYSTEM),
        KeyValue::new("gen_ai.request.model", request_model.to_string()),
        KeyValue::new("apr.model", span.model.clone()),
        KeyValue::new("http.route", span.route.clone()),
        KeyValue::new("http.response.status_code", i64::from(span.status_code)),
    ];
    if let Some(n) = span.prompt_tokens {
        attrs.push(KeyValue::new("apr.tokens.prompt", n));
        attrs.push(KeyValue::new("gen_ai.usage.input_tokens", n));
    }
    if let Some(n) = span.output_tokens {
        attrs.push(KeyValue::new("apr.tokens.output", n));
        attrs.push(KeyValue::new("gen_ai.usage.output_tokens", n));
    }
    if let Some(tps) = span.decode_tps.filter(|t| t.is_finite()) {
        attrs.push(KeyValue::new("apr.decode.tps", tps));
    }
    attrs
}

/// The OTLP endpoint from the environment, if tracing was asked for.
/// An empty value counts as unset.
#[must_use]
pub fn endpoint_from_env() -> Option<String> {
    ["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "OTEL_EXPORTER_OTLP_ENDPOINT"]
        .iter()
        .filter_map(|k| std::env::var(k).ok())
        .map(|v| v.trim().to_string())
        .find(|v| !v.is_empty())
}

/// Map `OTEL_EXPORTER_OTLP_PROTOCOL` to a protocol this exporter speaks.
/// Unset means `http/protobuf`, the OTel spec default for HTTP.
///
/// # Errors
/// `grpc` (this exporter is HTTP-only) and unknown values are refused.
pub fn protocol_from_str(value: Option<&str>) -> Result<Protocol> {
    match value.map(str::trim) {
        None | Some("" | "http/protobuf") => Ok(Protocol::HttpBinary),
        Some("http/json") => Ok(Protocol::HttpJson),
        Some("grpc") => bail!(
            "OTEL_EXPORTER_OTLP_PROTOCOL=grpc is not supported by apr serve; \
             use http/protobuf or http/json"
        ),
        Some(other) => bail!("unknown OTEL_EXPORTER_OTLP_PROTOCOL {other:?}"),
    }
}

/// Exports one [`INFERENCE_SPAN_NAME`] span per recorded request.
pub struct RequestSpanExporter {
    provider: SdkTracerProvider,
    tracer: opentelemetry_sdk::trace::Tracer,
}

impl std::fmt::Debug for RequestSpanExporter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RequestSpanExporter").finish_non_exhaustive()
    }
}

impl RequestSpanExporter {
    /// Build from the standard OTel environment variables. `Ok(None)` when no
    /// endpoint is set — tracing is off by default.
    ///
    /// # Errors
    /// An unsupported protocol, or an exporter that cannot be built.
    pub fn from_env(service_name: &str) -> Result<Option<Self>> {
        let Some(endpoint) = endpoint_from_env() else {
            return Ok(None);
        };
        let protocol =
            protocol_from_str(std::env::var("OTEL_EXPORTER_OTLP_PROTOCOL").ok().as_deref())?;
        let exporter = build_http_exporter(protocol)
            .with_context(|| format!("building OTLP exporter for {endpoint}"))?;
        Ok(Some(Self::with_exporter(exporter, service_name)))
    }

    /// Build on any span exporter (tests use the in-memory one).
    /// `OTEL_SERVICE_NAME`, when set, wins over `service_name`.
    pub fn with_exporter<E: SpanExporter + 'static>(exporter: E, service_name: &str) -> Self {
        let service_name = std::env::var("OTEL_SERVICE_NAME")
            .ok()
            .filter(|s| !s.trim().is_empty())
            .unwrap_or_else(|| service_name.to_string());
        let mut batch = BatchConfigBuilder::default();
        if std::env::var_os("OTEL_BSP_SCHEDULE_DELAY").is_none() {
            batch = batch.with_scheduled_delay(DEFAULT_SCHEDULE_DELAY);
        }
        let processor =
            BatchSpanProcessor::builder(exporter).with_batch_config(batch.build()).build();
        let provider = SdkTracerProvider::builder()
            .with_span_processor(processor)
            .with_resource(Resource::builder().with_service_name(service_name).build())
            .build();
        let tracer = provider.tracer("apr-serve");
        Self { provider, tracer }
    }

    /// Emit one span for `span`.
    pub fn record(&self, span: &InferenceSpan) {
        let parent_cx = span.parent.as_ref().map_or_else(Context::new, |p| {
            let flags = if p.is_sampled() { TraceFlags::SAMPLED } else { TraceFlags::default() };
            Context::new().with_remote_span_context(SpanContext::new(
                p.otel_trace_id(),
                p.otel_parent_id(),
                flags,
                true,
                TraceState::default(),
            ))
        });
        let mut otel_span = self
            .tracer
            .span_builder(INFERENCE_SPAN_NAME)
            .with_kind(SpanKind::Server)
            .with_start_time(span.start)
            .with_attributes(inference_attributes(span))
            .start_with_context(&self.tracer, &parent_cx);
        if span.status_code >= 500 {
            opentelemetry::trace::Span::set_status(
                &mut otel_span,
                Status::error(format!("HTTP {}", span.status_code)),
            );
        }
        opentelemetry::trace::Span::end_with_timestamp(&mut otel_span, span.end);
    }

    /// Flush buffered spans (used by tests and on shutdown).
    ///
    /// # Errors
    /// The exporter failed to flush.
    pub fn force_flush(&self) -> Result<()> {
        self.provider.force_flush().map_err(|e| anyhow!("OTLP flush failed: {e}"))
    }
}

/// The reqwest blocking client owns a runtime and panics if built or dropped
/// inside tokio, which is where `apr serve` calls this. Build it on a plain
/// thread; the batch processor then uses it from its own thread.
fn build_http_exporter(protocol: Protocol) -> Result<opentelemetry_otlp::SpanExporter> {
    std::thread::spawn(move || {
        opentelemetry_otlp::SpanExporter::builder().with_http().with_protocol(protocol).build()
    })
    .join()
    .map_err(|_| anyhow!("OTLP exporter builder thread panicked"))?
    .map_err(|e| anyhow!("{e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use opentelemetry::Value;
    use opentelemetry_sdk::trace::InMemorySpanExporter;

    const TRACEPARENT: &str = "00-0af7651916cd43dd8448eb211c80319c-00f067aa0ba902b7-01";

    fn sample() -> InferenceSpan {
        let start = SystemTime::UNIX_EPOCH + Duration::from_secs(1_700_000_000);
        InferenceSpan {
            model: "qwen2.5-0.5b".into(),
            request_model: Some("qwen".into()),
            route: "/v1/chat/completions".into(),
            status_code: 200,
            prompt_tokens: Some(12),
            output_tokens: Some(8),
            decode_tps: None,
            start,
            end: start + Duration::from_millis(250),
            parent: None,
        }
    }

    fn attr<'a>(attrs: &'a [KeyValue], key: &str) -> Option<&'a Value> {
        attrs.iter().find(|kv| kv.key.as_str() == key).map(|kv| &kv.value)
    }

    /// FALSIFY-CRUX-K-08-002 (unit): the gen_ai / apr.tokens keys are present
    /// with the observed values.
    #[test]
    fn falsify_k08_002_attributes_carry_observed_values() {
        let a = inference_attributes(&sample());
        assert_eq!(attr(&a, "gen_ai.system"), Some(&Value::from("apr")));
        assert_eq!(attr(&a, "gen_ai.request.model"), Some(&Value::from("qwen")));
        assert_eq!(attr(&a, "apr.model"), Some(&Value::from("qwen2.5-0.5b")));
        assert_eq!(attr(&a, "apr.tokens.prompt"), Some(&Value::I64(12)));
        assert_eq!(attr(&a, "apr.tokens.output"), Some(&Value::I64(8)));
        assert_eq!(attr(&a, "gen_ai.usage.input_tokens"), Some(&Value::I64(12)));
    }

    /// Unobserved values are omitted, never zeroed; the request model falls
    /// back to the served model.
    #[test]
    fn falsify_k08_unobserved_values_are_absent_not_zero() {
        let mut s = sample();
        s.request_model = None;
        s.prompt_tokens = None;
        s.output_tokens = None;
        s.decode_tps = Some(f64::NAN);
        let a = inference_attributes(&s);
        assert_eq!(attr(&a, "gen_ai.request.model"), Some(&Value::from("qwen2.5-0.5b")));
        for k in ["apr.tokens.prompt", "apr.tokens.output", "apr.decode.tps"] {
            assert!(attr(&a, k).is_none(), "{k} must be absent");
        }
    }

    #[test]
    fn protocol_env_mapping() {
        assert_eq!(protocol_from_str(None).ok(), Some(Protocol::HttpBinary));
        assert_eq!(protocol_from_str(Some("http/json")).ok(), Some(Protocol::HttpJson));
        assert!(protocol_from_str(Some("grpc")).is_err());
        assert!(protocol_from_str(Some("carrier-pigeon")).is_err());
    }

    /// FALSIFY-CRUX-K-08-001/003 (unit): one `apr.inference` span, and an
    /// incoming traceparent is continued — same trace id, parent span id.
    #[test]
    fn falsify_k08_003_traceparent_is_continued() {
        let mem = InMemorySpanExporter::default();
        let exp = RequestSpanExporter::with_exporter(mem.clone(), "apr-test");
        let mut s = sample();
        s.parent = Some(TraceContext::parse(TRACEPARENT).expect("valid traceparent"));
        exp.record(&s);
        exp.force_flush().expect("flush");
        let spans = mem.get_finished_spans().expect("spans");
        assert_eq!(spans.len(), 1);
        let got = &spans[0];
        assert_eq!(got.name, INFERENCE_SPAN_NAME);
        assert_eq!(got.span_kind, SpanKind::Server);
        assert_eq!(got.span_context.trace_id().to_string(), "0af7651916cd43dd8448eb211c80319c");
        assert_eq!(got.parent_span_id.to_string(), "00f067aa0ba902b7");
        assert_eq!(got.start_time, s.start);
        assert_eq!(got.end_time, s.end);
    }

    /// Without a traceparent the span is a fresh root.
    #[test]
    fn falsify_k08_no_parent_starts_new_trace() {
        let mem = InMemorySpanExporter::default();
        let exp = RequestSpanExporter::with_exporter(mem.clone(), "apr-test");
        exp.record(&sample());
        exp.force_flush().expect("flush");
        let spans = mem.get_finished_spans().expect("spans");
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].parent_span_id, opentelemetry::trace::SpanId::INVALID);
        assert_ne!(
            spans[0].span_context.trace_id().to_string(),
            "0af7651916cd43dd8448eb211c80319c"
        );
    }
}
