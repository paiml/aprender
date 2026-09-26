//! SRV-TIM-001: one structured line per completed generation request.
//!
//! Every chat/completions terminal point (the non-streaming body builder and
//! each SSE terminal chunk) calls [`emit`] with the SAME `Timings` it put on the
//! wire, so the log, the `/metrics` histograms and the optional `--timings-log`
//! JSONL file can never disagree with the response (FALSIFY-SRV-TIM-005/006).
//!
//! The server has no `tracing` subscriber, so the line goes to stderr — where
//! every other serve log line goes and where the fleet units' journald capture
//! reads — prefixed `[request] ` and followed by one JSON object.
//!
//! `ttft_ms` is `total_ms - decode_ms`: queueing, tokenisation and prefill up
//! to the first chosen token, plus the (sub-millisecond) response assembly
//! after the last one, so it is an upper bound, never an estimate. It is absent
//! whenever the split is.
//!
//! APR-OBS-001 OBS-03: a client that sends `X-Request-ID` gets it back in
//! `client_request_id`, so OBS-04 can join its own lane row to this line.
//! [`client_request_id_scope`] holds it in a task-local around the handler AND
//! around the SSE body's polls, because a stream's terminal emit runs while
//! the body is being written, after the handler future has returned.

use std::io::Write as _;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use super::Timings;

/// One request's record. Built only at a terminal point.
#[derive(Debug, Clone, serde::Serialize)]
pub struct RequestRecord {
    /// The response id the client saw.
    pub request_id: String,
    /// The client's `X-Request-ID` header (OBS-03 join key), `null` when the
    /// client sent none or one that was not a short printable token.
    pub client_request_id: Option<String>,
    /// The `model` field of the response.
    pub model: String,
    /// `chat` or `completions`.
    pub endpoint: &'static str,
    /// `gpu` / `cpu` from the response's own `used_gpu`; `unreported` when the
    /// backend records nothing (#3894) — never a guess.
    pub backend: &'static str,
    /// Whether the response was an SSE stream.
    pub stream: bool,
    /// Prompt tokens (== `usage.prompt_tokens`).
    pub prompt_n: usize,
    /// Generated tokens (== `usage.completion_tokens`).
    pub predicted_n: usize,
    /// `timings.prompt_ms`, absent when the response carries no timings.
    pub prefill_ms: Option<f64>,
    /// `timings.predicted_ms`, absent when the response carries no timings.
    pub decode_ms: Option<f64>,
    /// `total_ms - decode_ms` (see the module doc).
    pub ttft_ms: Option<f64>,
    /// Handler entry to the terminal point, wall clock.
    pub total_ms: f64,
    /// The `finish_reason` the client saw.
    pub finish_reason: String,
}

impl RequestRecord {
    /// A record whose phase fields are copied from the response's `timings`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        request_id: &str,
        model: &str,
        endpoint: &'static str,
        used_gpu: Option<bool>,
        stream: bool,
        prompt_n: usize,
        predicted_n: usize,
        timings: Option<&Timings>,
        total: std::time::Duration,
        finish_reason: &str,
    ) -> Self {
        let total_ms = total.as_secs_f64() * 1000.0;
        let prefill_ms = timings.map(|t| t.prompt_ms);
        let decode_ms = timings.map(|t| t.predicted_ms);
        Self {
            request_id: request_id.to_string(),
            client_request_id: current_client_request_id(),
            model: model.to_string(),
            endpoint,
            backend: match used_gpu {
                Some(true) => "gpu",
                Some(false) => "cpu",
                None => "unreported",
            },
            stream,
            prompt_n,
            predicted_n,
            prefill_ms,
            decode_ms,
            ttft_ms: decode_ms.map(|d| (total_ms - d).max(0.0)),
            total_ms,
            finish_reason: finish_reason.to_string(),
        }
    }
}

/// The header OBS-01 clients send and this server echoes.
pub const CLIENT_REQUEST_ID_HEADER: &str = "x-request-id";

/// Longest `X-Request-ID` accepted; a UUIDv7 is 36.
const MAX_CLIENT_REQUEST_ID: usize = 128;

tokio::task_local! {
    static CLIENT_REQUEST_ID: Option<std::sync::Arc<str>>;
}

/// The current request's `X-Request-ID`, if one is in scope.
fn current_client_request_id() -> Option<String> {
    CLIENT_REQUEST_ID
        .try_with(|id| id.as_deref().map(str::to_string))
        .ok()
        .flatten()
}

/// Accept a header value only if it is a short token of visible ASCII.
fn accept_client_request_id(value: &axum::http::HeaderValue) -> Option<std::sync::Arc<str>> {
    let s = value.to_str().ok()?;
    let ok = !s.is_empty()
        && s.len() <= MAX_CLIENT_REQUEST_ID
        && s.bytes().all(|b| b.is_ascii_graphic());
    ok.then(|| std::sync::Arc::from(s))
}

/// Middleware: scope the client's `X-Request-ID` over the handler and, for an
/// SSE response, over every poll of its body; echo it on the response.
/// Mounted INSIDE `cancel_on_disconnect`, whose `tokio::spawn` a task-local
/// does not cross.
pub(crate) async fn client_request_id_scope(
    request: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    use futures::StreamExt as _;
    let Some(id) = request
        .headers()
        .get(CLIENT_REQUEST_ID_HEADER)
        .and_then(accept_client_request_id)
    else {
        return next.run(request).await;
    };
    let mut response = CLIENT_REQUEST_ID
        .scope(Some(id.clone()), next.run(request))
        .await;
    if let Ok(v) = axum::http::HeaderValue::from_str(&id) {
        response.headers_mut().insert(CLIENT_REQUEST_ID_HEADER, v);
    }
    let is_sse = response
        .headers()
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.starts_with("text/event-stream"));
    if !is_sse {
        return response;
    }
    let (parts, body) = response.into_parts();
    let mut inner = body.into_data_stream();
    let scoped = futures::stream::poll_fn(move |cx| {
        CLIENT_REQUEST_ID.sync_scope(Some(id.clone()), || inner.poll_next_unpin(cx))
    });
    axum::response::Response::from_parts(parts, axum::body::Body::from_stream(scoped))
}

/// Prefix of the stderr line; everything after it is one JSON object.
pub const LOG_PREFIX: &str = "[request] ";

/// Where `--timings-log` appends, plus the build identity stamped on each line.
struct TimingsLog {
    file: Mutex<std::fs::File>,
    build: String,
    host: String,
}

static TIMINGS_LOG: OnceLock<TimingsLog> = OnceLock::new();

/// `apr serve --timings-log <path>`: append one JSONL line per request to
/// `path` (created if absent). `build` is the serving binary's `version (sha)`.
/// Off unless called; a second call is refused rather than silently ignored.
///
/// # Errors
/// The file cannot be opened for append, or a log is already configured.
pub fn set_timings_log(path: &std::path::Path, build: &str) -> Result<(), String> {
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| format!("--timings-log {}: {e}", path.display()))?;
    TIMINGS_LOG
        .set(TimingsLog {
            file: Mutex::new(file),
            build: build.to_string(),
            host: host_name(),
        })
        .map_err(|_| "--timings-log is already configured".to_string())
}

fn host_name() -> String {
    std::fs::read_to_string("/proc/sys/kernel/hostname")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .or_else(|| std::env::var("HOSTNAME").ok())
        .unwrap_or_else(|| "unknown".to_string())
}

/// Bucket upper bounds (ms) shared by the three histograms.
pub const BUCKETS_MS: [f64; 12] = [
    5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0, 10000.0, 30000.0,
];

/// A fixed-bucket Prometheus histogram over milliseconds, lock-free.
pub struct MsHistogram {
    buckets: [AtomicU64; 12],
    count: AtomicU64,
    /// Sum in microseconds, so it stays an integer atomic.
    sum_us: AtomicU64,
}

impl MsHistogram {
    const fn new() -> Self {
        Self {
            buckets: [const { AtomicU64::new(0) }; 12],
            count: AtomicU64::new(0),
            sum_us: AtomicU64::new(0),
        }
    }

    fn observe(&self, ms: f64) {
        if let Some(i) = BUCKETS_MS.iter().position(|&b| ms <= b) {
            self.buckets[i].fetch_add(1, Ordering::Relaxed);
        }
        self.count.fetch_add(1, Ordering::Relaxed);
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        self.sum_us
            .fetch_add((ms.max(0.0) * 1000.0) as u64, Ordering::Relaxed);
    }

    fn render(&self, name: &str, help: &str, out: &mut String) {
        use std::fmt::Write as _;
        let _ = writeln!(out, "# HELP {name} {help}");
        let _ = writeln!(out, "# TYPE {name} histogram");
        let mut cumulative = 0u64;
        for (b, n) in BUCKETS_MS.iter().zip(&self.buckets) {
            cumulative += n.load(Ordering::Relaxed);
            let _ = writeln!(out, "{name}_bucket{{le=\"{b}\"}} {cumulative}");
        }
        let count = self.count.load(Ordering::Relaxed);
        let _ = writeln!(out, "{name}_bucket{{le=\"+Inf\"}} {count}");
        #[allow(clippy::cast_precision_loss)]
        let sum_ms = self.sum_us.load(Ordering::Relaxed) as f64 / 1000.0;
        let _ = writeln!(out, "{name}_sum {sum_ms}");
        let _ = writeln!(out, "{name}_count {count}");
    }
}

static PREFILL_MS: MsHistogram = MsHistogram::new();
static DECODE_MS: MsHistogram = MsHistogram::new();
static TTFT_MS: MsHistogram = MsHistogram::new();

/// The three phase histograms in Prometheus text format, appended to
/// `/metrics`. A request without a measured split is not observed: an absent
/// phase is never recorded as a zero.
#[must_use]
pub fn prometheus_histograms() -> String {
    let mut out = String::new();
    PREFILL_MS.render(
        "realizar_prefill_ms",
        "Prompt forward to first chosen token, per request (ms)",
        &mut out,
    );
    DECODE_MS.render(
        "realizar_decode_ms",
        "First chosen token to last token, per request (ms)",
        &mut out,
    );
    TTFT_MS.render(
        "realizar_ttft_ms",
        "Handler entry to first chosen token, upper bound, per request (ms)",
        &mut out,
    );
    out
}

/// Log, observe and (when configured) append one request's record.
pub fn emit(record: &RequestRecord) {
    let Ok(line) = serde_json::to_string(record) else {
        return;
    };
    eprintln!("{LOG_PREFIX}{line}");
    if let (Some(p), Some(d), Some(t)) = (record.prefill_ms, record.decode_ms, record.ttft_ms) {
        PREFILL_MS.observe(p);
        DECODE_MS.observe(d);
        TTFT_MS.observe(t);
    }
    if let Some(log) = TIMINGS_LOG.get() {
        let mut value = serde_json::to_value(record).unwrap_or_default();
        if let Some(obj) = value.as_object_mut() {
            obj.insert("build".into(), log.build.clone().into());
            obj.insert("host".into(), log.host.clone().into());
            obj.insert("unix_ms".into(), unix_ms().into());
        }
        if let Ok(mut f) = log.file.lock() {
            let _ = writeln!(f, "{value}");
        }
    }
}

fn unix_ms() -> u64 {
    #[allow(clippy::cast_possible_truncation)]
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn timings(prompt_ms: f64, predicted_ms: f64) -> Timings {
        Timings {
            prompt_n: 7,
            prompt_ms,
            prompt_per_second: None,
            predicted_n: 3,
            predicted_ms,
            predicted_per_second: None,
            clock: super::super::TIMINGS_CLOCK.to_string(),
        }
    }

    #[test]
    fn record_copies_the_response_timings_exactly() {
        let t = timings(12.5, 40.25);
        let r = RequestRecord::new(
            "id",
            "m",
            "chat",
            Some(true),
            false,
            7,
            3,
            Some(&t),
            std::time::Duration::from_millis(60),
            "stop",
        );
        let v: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&r).expect("ser")).expect("json");
        assert_eq!(v["prefill_ms"].as_f64(), Some(12.5));
        assert_eq!(v["decode_ms"].as_f64(), Some(40.25));
        assert_eq!(v["backend"], "gpu");
        assert!((v["ttft_ms"].as_f64().expect("ttft") - 19.75).abs() < 1e-9);
        for k in [
            "request_id",
            "client_request_id",
            "model",
            "endpoint",
            "backend",
            "stream",
            "prompt_n",
            "predicted_n",
            "prefill_ms",
            "decode_ms",
            "ttft_ms",
            "total_ms",
            "finish_reason",
        ] {
            assert!(v.get(k).is_some(), "missing {k}");
        }
    }

    #[test]
    fn absent_timings_stay_absent_never_zero() {
        let r = RequestRecord::new(
            "id",
            "m",
            "completions",
            None,
            true,
            1,
            1,
            None,
            std::time::Duration::from_millis(5),
            "length",
        );
        assert_eq!(r.prefill_ms, None);
        assert_eq!(r.decode_ms, None);
        assert_eq!(r.ttft_ms, None);
        assert_eq!(r.backend, "unreported");
    }

    fn record_now() -> RequestRecord {
        RequestRecord::new(
            "chatcmpl-x",
            "m",
            "chat",
            None,
            true,
            1,
            1,
            None,
            std::time::Duration::from_millis(1),
            "stop",
        )
    }

    #[test]
    fn obs03_client_request_id_is_null_outside_a_request() {
        assert_eq!(record_now().client_request_id, None);
    }

    #[test]
    fn obs03_header_filter_takes_uuids_and_refuses_junk() {
        use axum::http::HeaderValue;
        let uuid = "01927c3e-8f2a-7b3c-9d4e-5f6a7b8c9d0e";
        assert_eq!(
            accept_client_request_id(&HeaderValue::from_static(uuid)).as_deref(),
            Some(uuid)
        );
        for bad in ["", "has space", &"x".repeat(MAX_CLIENT_REQUEST_ID + 1)] {
            let v = HeaderValue::from_str(bad).expect("header");
            assert_eq!(accept_client_request_id(&v), None, "{bad:?}");
        }
    }

    /// OBS-03: the id reaches a record built in the handler AND one built
    /// while the SSE body is polled, after the handler returned — and the
    /// response echoes the header.
    #[tokio::test]
    async fn obs03_client_request_id_reaches_handler_and_sse_body_records() {
        use axum::response::sse::{Event, Sse};
        use axum::response::IntoResponse as _;
        use tower::ServiceExt as _;
        let seen = std::sync::Arc::new(Mutex::new(Vec::<Option<String>>::new()));
        let (s1, s2) = (seen.clone(), seen.clone());
        let app = axum::Router::new()
            .route(
                "/json",
                axum::routing::get(move || async move {
                    s1.lock()
                        .expect("lock")
                        .push(record_now().client_request_id);
                    "ok"
                }),
            )
            .route(
                "/sse",
                axum::routing::get(move || async move {
                    let s2 = s2.clone();
                    let events = async_stream::stream! {
                        s2.lock().expect("lock").push(record_now().client_request_id);
                        yield Ok::<_, std::convert::Infallible>(Event::default().data("x"));
                    };
                    Sse::new(events).into_response()
                }),
            )
            .layer(axum::middleware::from_fn(client_request_id_scope));
        let id = "01927c3e-8f2a-7b3c-9d4e-5f6a7b8c9d0e";
        for route in ["/json", "/sse"] {
            let req = axum::http::Request::get(route)
                .header(CLIENT_REQUEST_ID_HEADER, id)
                .body(axum::body::Body::empty())
                .expect("req");
            let resp = app.clone().oneshot(req).await.expect("resp");
            assert_eq!(resp.headers()[CLIENT_REQUEST_ID_HEADER], id, "{route}");
            axum::body::to_bytes(resp.into_body(), usize::MAX)
                .await
                .expect("body");
        }
        let seen = seen.lock().expect("lock").clone();
        assert_eq!(seen, vec![Some(id.to_string()), Some(id.to_string())]);
    }

    #[test]
    fn histogram_buckets_are_cumulative_and_count_every_observation() {
        let h = MsHistogram::new();
        h.observe(3.0);
        h.observe(30.0);
        h.observe(99_999.0);
        let mut out = String::new();
        h.render("x", "h", &mut out);
        assert!(out.contains("x_bucket{le=\"5\"} 1"), "{out}");
        assert!(out.contains("x_bucket{le=\"50\"} 2"), "{out}");
        assert!(out.contains("x_bucket{le=\"30000\"} 2"), "{out}");
        assert!(out.contains("x_bucket{le=\"+Inf\"} 3"), "{out}");
        assert!(out.contains("x_count 3"), "{out}");
        assert!(out.contains("x_sum 100032\n"), "{out}");
    }
}
