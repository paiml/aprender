//! APR-PERF-GATE-001 v2.2 §4.4 — the protocol, driven through the existing
//! `apr test llm bench` request path.
//!
//! This is **not a second harness.** It drives [`LlmClient`] — the one HTTP
//! client [`super::loadtest::LoadTest`] drives, and the one
//! `scripts/parity_host_receipt.sh` points at both `apr serve` and
//! `llama-server` (§4.4.8: one client, both servers, or the ratio is refused).
//! What it adds is the part `LoadTest` never had:
//! the §4.4.2 termination rule, the §4.4.2 warmup-then-quiesce, the §4.4.3
//! metric definitions, §4.4.4's interval, §4.4.5 retention, and §4.4.7's
//! `drain_ms`.
//!
//! `LoadTest::run` stays exactly as it is. It answers a different question — "how
//! does this server behave for `duration` seconds" — and a great deal of tooling
//! reads its `LoadTestResult`. Changing its termination rule underneath those
//! readers would silently change every number they have ever recorded.
//!
//! `loadtest::send_one_request` is the sibling adapter: it maps the same client's
//! responses into `LoadTestResult`'s `RequestRecord`. This module maps them into
//! [`RequestSample`] instead, because `RequestRecord` carries no absolute timing
//! and cannot distinguish a timeout from a transport error — both of which §4.4.3
//! requires.

use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use crate::perf_gate::bootstrap::{bootstrap_agg_tok_s_ci, BootstrapCi};
use crate::perf_gate::metrics::{BandMetrics, RequestSample};
use crate::perf_gate::protocol::{BandConfig, ClientModel, Outcome, TokenizationBlock, REPLICATES};
use crate::perf_gate::window::{WindowController, WindowReport};

use super::client::{ChatRequest, LlmClient, LlmClientError};
use super::client::{ChatResponse, Usage};

/// One §4.4-conformant band measurement.
#[derive(Debug, Clone)]
pub struct BandRun {
    /// The protocol parameters actually used.
    pub config: BandConfig,
    /// §4.4.1 — recorded, not assumed.
    pub client_model: ClientModel,
    /// §4.4.6 — required, supplied by the caller because only the caller knows
    /// how its tokens were counted.
    pub tokenization: TokenizationBlock,
    /// §4.4.3 metrics.
    pub metrics: BandMetrics,
    /// §4.4.2/§4.4.7 window and drain accounting.
    pub window: WindowReport,
    /// §4.4.5 raw per-request samples, for retention and resampling.
    pub samples: Vec<RequestSample>,
    /// §4.4.4 bootstrap percentile CI on `agg_tok_s`.
    pub agg_ci: Option<BootstrapCi>,
    /// Warmup requests actually completed (discarded, never in `samples`).
    pub warmup_completed: usize,
    /// Every reason this run is not §4.4-conformant. Empty means conformant.
    /// A run that shrank its window says so here rather than looking identical
    /// to one that did not.
    pub protocol_violations: Vec<String>,
}

impl BandRun {
    /// True when nothing departed from §4.4 and no `SUSPECT` annotation fired.
    #[must_use]
    pub fn is_conformant(&self) -> bool {
        self.protocol_violations.is_empty() && self.window.suspect.is_empty()
    }
}

type Shared = Arc<Mutex<WindowController>>;

/// `(index, in-flight including this request)`, as returned by the admission gate.
type Admitted = (usize, usize);

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// §4.4.2 warmup: every worker completes at least one request, `2 × c` in total,
/// all discarded. Returns how many actually completed.
async fn warmup(
    client: &LlmClient,
    prompts: &[ChatRequest],
    band: &BandConfig,
    stream: bool,
) -> usize {
    let per_worker = band.warmup_requests.div_ceil(band.concurrency).max(1);
    let mut handles = Vec::with_capacity(band.concurrency);
    for worker in 0..band.concurrency {
        let client = client.clone();
        let prompts = prompts.to_vec();
        handles.push(tokio::spawn(async move {
            let mut done = 0_usize;
            for k in 0..per_worker {
                let prompt = &prompts[(worker + k) % prompts.len()];
                if issue(&client, prompt, stream).await.is_some() {
                    done += 1;
                }
            }
            done
        }));
    }
    let mut total = 0;
    for h in handles {
        total += h.await.unwrap_or(0);
    }
    total
}

/// Everything one closed-loop worker needs. A struct rather than eight cloned
/// locals per spawn: the worker body is the part that has to stay readable,
/// because it is where "concurrent" is either true or a comfortable fiction.
struct Worker {
    id: usize,
    client: LlmClient,
    prompts: Vec<ChatRequest>,
    controller: Shared,
    samples: Arc<Mutex<Vec<RequestSample>>>,
    timeout: Duration,
    origin: Instant,
    stream: bool,
}

/// What one request revealed, in the terms §4.4.3 is written in.
///
/// Deliberately NOT `loadtest::RequestRecord`. That type reports a `latency` and
/// a `success` flag; §4.4.3 needs token arrival offsets and a four-valued
/// outcome, and a timeout there is indistinguishable from a transport error.
struct Observed {
    /// Token arrival offsets from THIS request's start. Empty when not streaming.
    token_offsets: Vec<Duration>,
    generated_tokens: u32,
    prompt_tokens: u32,
}

fn usage_counts(usage: Option<&Usage>, fallback_tokens: u32) -> (u32, u32) {
    usage.map_or((fallback_tokens, 0), |u| {
        (u.completion_tokens, u.prompt_tokens)
    })
}

fn observe_blocking(response: &ChatResponse) -> Observed {
    let (generated_tokens, prompt_tokens) = usage_counts(response.usage.as_ref(), 0);
    Observed {
        // A non-streaming response has no per-token arrival information at all.
        // Empty rather than synthesised: §4.4.3's `itl_ms` and `decode_tok_s` are
        // undefined here, and inventing evenly-spaced timestamps would make an
        // unmeasured quantity look measured.
        token_offsets: Vec::new(),
        generated_tokens,
        prompt_tokens,
    }
}

/// Issue one request through the shared [`LlmClient`] — the same client
/// `LoadTest`/`send_one_request` uses, so both drive one transport (§4.4.8).
async fn issue(client: &LlmClient, prompt: &ChatRequest, stream: bool) -> Option<Observed> {
    if stream {
        let streamed = client.chat_completion_stream(prompt).await.ok()?;
        let observed_tokens = u32::try_from(streamed.token_timestamps.len()).unwrap_or(u32::MAX);
        let (generated_tokens, prompt_tokens) =
            usage_counts(streamed.usage.as_ref(), observed_tokens);
        return Some(Observed {
            token_offsets: streamed.token_timestamps,
            generated_tokens,
            prompt_tokens,
        });
    }
    let timed = client.send(prompt).await.ok()?;
    Some(observe_blocking(&timed.response))
}

/// Turn one completed (or timed-out) request into its §4.4.5 retained sample.
fn sample_from(
    slot: Admitted,
    worker: usize,
    span: (f64, f64),
    drained: bool,
    observed: Option<&Observed>,
) -> RequestSample {
    let (start_s, end_s) = span;
    let (index, in_flight_at_start) = slot;
    let Some(observed) = observed else {
        // §4.4.3 — a request that hit the 120 s hard timeout, or failed. Any
        // partial output is discarded: it is not a completed request and must
        // not be credited to `agg_tok_s`'s numerator.
        return RequestSample {
            index,
            worker,
            start_s,
            end_s,
            token_times_s: Vec::new(),
            generated_tokens: 0,
            prompt_tokens: 0,
            outcome: Outcome::Failed,
            in_flight_at_start,
            drained,
        };
    };
    RequestSample {
        index,
        worker,
        start_s,
        end_s,
        // Token offsets are relative to THIS request's start; §4.4.3 needs a
        // band-wide origin so spans are comparable across workers.
        token_times_s: observed
            .token_offsets
            .iter()
            .map(|d| start_s + d.as_secs_f64())
            .collect(),
        generated_tokens: observed.generated_tokens,
        prompt_tokens: observed.prompt_tokens,
        outcome: Outcome::Completed,
        in_flight_at_start,
        drained,
    }
}

/// One closed-loop worker: admit, issue, wait for completion, immediately try to
/// admit again (§4.4.1). Exits when the shared gate closes.
async fn worker_loop(w: Worker) {
    loop {
        // §4.4.2/§4.4.7 — one shared admission decision. The gate closes once,
        // stamping T; no worker issues at or after it.
        let admitted = {
            let mut c = lock(&w.controller);
            c.try_admit_with_in_flight(w.origin.elapsed().as_secs_f64())
        };
        let Some(slot) = admitted else { break };

        let prompt = &w.prompts[slot.0 % w.prompts.len()];
        let start_s = w.origin.elapsed().as_secs_f64();
        // §4.4.3 — the 120 s hard per-request timeout, classified as a TIMEOUT
        // in its own counter rather than folded into a generic failure count.
        let timed_out = tokio::time::timeout(w.timeout, issue(&w.client, prompt, w.stream)).await;
        let end_s = w.origin.elapsed().as_secs_f64();

        let drained = lock(&w.controller).complete(end_s);
        let mut sample = match &timed_out {
            Ok(observed) => sample_from(slot, w.id, (start_s, end_s), drained, observed.as_ref()),
            Err(_elapsed) => sample_from(slot, w.id, (start_s, end_s), drained, None),
        };
        if timed_out.is_err() {
            sample.outcome = Outcome::Timeout;
        }
        lock(&w.samples).push(sample);
    }
}

/// Every reason this run departs from §4.4, for the receipt.
fn violations(
    band: &BandConfig,
    warmup_completed: usize,
    m: &BandMetrics,
    win: &WindowReport,
) -> Vec<String> {
    let mut out = band.conformance_violations();
    if warmup_completed < band.warmup_requests {
        out.push(format!(
            "§4.4.2 warmup completed {warmup_completed} of {} required requests",
            band.warmup_requests
        ));
    }
    if m.completed < band.min_samples {
        out.push(format!(
            "§4.4.2 only {} of {} required sampled requests completed",
            m.completed, band.min_samples
        ));
    }
    out.extend(win.suspect.iter().cloned());
    out
}

/// Run one §4.4-conformant band against `client`.
///
/// The caller supplies `tokenization` because §4.4.6 gives `method` no default:
/// a harness that guessed it would be asserting something it does not know.
///
/// # Errors
/// When `prompts` is empty, which makes the workload undefined rather than
/// merely small, or when the §4.4.6 block does not validate.
pub async fn run_band(
    client: &LlmClient,
    prompts: &[ChatRequest],
    band: &BandConfig,
    tokenization: TokenizationBlock,
    stream: bool,
) -> Result<BandRun, LlmClientError> {
    if prompts.is_empty() {
        return Err(LlmClientError::HealthCheckFailed(
            "run_band: the prompt corpus is empty; §4.3 requires a fixed workload".to_string(),
        ));
    }
    tokenization
        .validate()
        .map_err(LlmClientError::HealthCheckFailed)?;

    // §4.4.2 — warmup, discarded, then a 5 s quiesce before the first sampled
    // request. Without the quiesce the first sampled requests are measured on a
    // server still finishing warmup work, which is the opposite of what warmup
    // was for.
    let warmup_completed = warmup(client, prompts, band, stream).await;
    tokio::time::sleep(band.quiesce).await;

    let controller: Shared = Arc::new(Mutex::new(WindowController::new(band)));
    let samples: Arc<Mutex<Vec<RequestSample>>> = Arc::new(Mutex::new(Vec::new()));
    let origin = Instant::now();

    let mut handles = Vec::with_capacity(band.concurrency);
    for id in 0..band.concurrency {
        handles.push(tokio::spawn(worker_loop(Worker {
            id,
            client: client.clone(),
            prompts: prompts.to_vec(),
            controller: Arc::clone(&controller),
            samples: Arc::clone(&samples),
            timeout: band.request_timeout,
            origin,
            stream,
        })));
    }
    for h in handles {
        let _ = h.await;
    }

    let window = lock(&controller).report();
    let mut collected = std::mem::take(&mut *lock(&samples));
    collected.sort_by_key(|s| s.index);

    let metrics = BandMetrics::from_samples(band.concurrency, &collected);
    let agg_ci = bootstrap_agg_tok_s_ci(&collected, 0.95);
    let protocol_violations = violations(band, warmup_completed, &metrics, &window);

    Ok(BandRun {
        config: band.clone(),
        client_model: band.client_model,
        tokenization,
        metrics,
        window,
        samples: collected,
        agg_ci,
        warmup_completed,
        protocol_violations,
    })
}

/// §4.4.2 — `N = 3` full band replicates per cell.
///
/// Three separate runs, each with its own warmup and quiesce, because a
/// replicate that shares a warmup is not an independent run of the protocol.
///
/// # Errors
/// Propagates the first failing replicate.
pub async fn run_cell(
    client: &LlmClient,
    prompts: &[ChatRequest],
    band: &BandConfig,
    tokenization: &TokenizationBlock,
    stream: bool,
) -> Result<Vec<BandRun>, LlmClientError> {
    let mut runs = Vec::with_capacity(REPLICATES);
    for _ in 0..REPLICATES {
        runs.push(run_band(client, prompts, band, tokenization.clone(), stream).await?);
    }
    Ok(runs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    const BODY: &str = concat!(
        r#"{"id":"c","object":"chat.completion","created":0,"model":"m","#,
        r#""choices":[{"index":0,"message":{"role":"assistant","content":"a b c"},"#,
        r#""finish_reason":"length"}],"#,
        r#""usage":{"prompt_tokens":512,"completion_tokens":128,"total_tokens":640}}"#
    );

    /// A minimal OpenAI-shaped server that holds each request open for
    /// `service_ms` and records the peak number of connections it was serving at
    /// once. That peak is measured on the SERVER side, so it cannot be faked by
    /// the client's own bookkeeping.
    struct Probe {
        url: String,
        peak: Arc<AtomicUsize>,
        served: Arc<AtomicUsize>,
    }

    /// Consume one HTTP request head and its body, so the client is not left
    /// waiting on a half-read socket.
    async fn consume_request(sock: &mut tokio::net::TcpStream) {
        let mut buf = vec![0_u8; 8192];
        let mut seen = Vec::new();
        while let Ok(n) = sock.read(&mut buf).await {
            if n == 0 {
                return;
            }
            seen.extend_from_slice(&buf[..n]);
            let text = String::from_utf8_lossy(&seen).to_string();
            let Some(head_end) = text.find("\r\n\r\n") else {
                continue;
            };
            if seen.len() >= head_end + 4 + content_length(&text) {
                return;
            }
        }
    }

    fn content_length(head: &str) -> usize {
        head.to_lowercase()
            .split("content-length:")
            .nth(1)
            .and_then(|t| t.split("\r\n").next())
            .and_then(|t| t.trim().parse::<usize>().ok())
            .unwrap_or(0)
    }

    /// Serve one connection: count it in, hold it open for `service_ms`, answer.
    async fn serve_one(mut sock: tokio::net::TcpStream, service_ms: u64, counters: Counters) {
        let now = counters.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
        counters.peak.fetch_max(now, Ordering::SeqCst);

        consume_request(&mut sock).await;
        tokio::time::sleep(Duration::from_millis(service_ms)).await;

        let resp = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
             Content-Length: {}\r\nConnection: close\r\n\r\n{BODY}",
            BODY.len()
        );
        let _ = sock.write_all(resp.as_bytes()).await;
        let _ = sock.flush().await;
        let _ = sock.shutdown().await;

        counters.served.fetch_add(1, Ordering::SeqCst);
        counters.in_flight.fetch_sub(1, Ordering::SeqCst);
    }

    #[derive(Clone)]
    struct Counters {
        in_flight: Arc<AtomicUsize>,
        peak: Arc<AtomicUsize>,
        served: Arc<AtomicUsize>,
    }

    async fn spawn_probe(service_ms: u64) -> Probe {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind loopback");
        let addr = listener.local_addr().expect("local addr");
        let counters = Counters {
            in_flight: Arc::new(AtomicUsize::new(0)),
            peak: Arc::new(AtomicUsize::new(0)),
            served: Arc::new(AtomicUsize::new(0)),
        };
        let (peak, served) = (Arc::clone(&counters.peak), Arc::clone(&counters.served));

        tokio::spawn(async move {
            while let Ok((sock, _)) = listener.accept().await {
                tokio::spawn(serve_one(sock, service_ms, counters.clone()));
            }
        });

        Probe {
            url: format!("http://{addr}"),
            peak,
            served,
        }
    }

    /// An SSE server that emits `tokens` chunks `gap_ms` apart, so TTFT and the
    /// inter-token gaps have known values the client must be able to recover.
    async fn serve_sse(mut sock: tokio::net::TcpStream, tokens: usize, gap_ms: u64) {
        consume_request(&mut sock).await;
        let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\
                    Cache-Control: no-cache\r\nConnection: close\r\n\r\n";
        if sock.write_all(head.as_bytes()).await.is_err() {
            return;
        }
        for i in 0..tokens {
            tokio::time::sleep(Duration::from_millis(gap_ms)).await;
            let chunk = format!(
                "data: {{\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"t{i} \"}}}}]}}\n\n"
            );
            if sock.write_all(chunk.as_bytes()).await.is_err() {
                return;
            }
            let _ = sock.flush().await;
        }
        let _ = sock.write_all(b"data: [DONE]\n\n").await;
        let _ = sock.flush().await;
        let _ = sock.shutdown().await;
    }

    async fn spawn_sse_probe(tokens: usize, gap_ms: u64) -> String {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind loopback");
        let addr = listener.local_addr().expect("local addr");
        tokio::spawn(async move {
            while let Ok((sock, _)) = listener.accept().await {
                tokio::spawn(serve_sse(sock, tokens, gap_ms));
            }
        });
        format!("http://{addr}")
    }

    fn prompts() -> Vec<ChatRequest> {
        vec![ChatRequest {
            model: "m".to_string(),
            messages: vec![super::super::client::ChatMessage {
                role: super::super::client::Role::User,
                content: "hello".to_string(),
            }],
            temperature: Some(0.0),
            max_tokens: Some(128),
            stream: Some(false),
            // #2746 added these to ChatRequest after this test module was
            // written on a stacked branch that never saw it. None matches
            // loadtest.rs's convention for a fixture; the PRODUCTION band
            // path never builds a ChatRequest itself -- prompts.rs carries
            // both fields through from the corpus (prompts.rs:206), which is
            // what satisfies the 4.4.4 seed requirement.
            seed: None,
            ignore_eos: None,
        }]
    }

    fn tokenization() -> TokenizationBlock {
        TokenizationBlock::ServerUsage {
            counts_special_tokens: false,
            counts_prompt_echo: false,
        }
    }

    /// A band short enough to run in a test, and it must report that it is not
    /// conformant — the shrunken window is visible in the result, not hidden.
    fn tiny_band(c: usize, samples: usize) -> BandConfig {
        BandConfig::relaxed(c, samples, Duration::ZERO, Duration::ZERO)
    }

    /// THE end-to-end proof: `c = 8` workers, driven through the same
    /// `send_one_request` path `apr test llm bench` uses, over real loopback
    /// HTTP, are concurrent **as observed by the server**.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn eight_workers_are_concurrent_over_real_http() {
        let probe = spawn_probe(60).await;
        let client = LlmClient::new(&probe.url, "m");
        let band = tiny_band(8, 32);

        let run = run_band(&client, &prompts(), &band, tokenization(), false)
            .await
            .expect("band runs");

        let server_peak = probe.peak.load(Ordering::SeqCst);
        let client_peak = run.window.client_peak_in_flight;
        eprintln!(
            "run_band c=8: server_peak={server_peak} client_peak={client_peak} \
             requested={} completed={} window_ms={:.1} drain_ms={:.1} served={}",
            run.window.requested,
            run.metrics.completed,
            run.window.window_ms,
            run.window.drain_ms,
            probe.served.load(Ordering::SeqCst)
        );

        assert_eq!(client_peak, 8, "the client must admit 8 at once");
        assert!(
            server_peak >= 4,
            "the SERVER saw only {server_peak} concurrent connections; a client that \
             says c=8 and is secretly sequential shows 1"
        );
        assert!(run.metrics.completed >= 32, "{:?}", run.metrics);
        // §4.4.3 sanity: 128 tokens per completed request, wall-clock aggregate.
        assert!(run.metrics.agg_tok_s > 0.0);
        assert_eq!(run.metrics.timeouts, 0);
    }

    /// The negative control that makes the assertion above mean something: the
    /// same code at `c = 1` must show a server peak of exactly 1, and must take
    /// several times longer for the same request count.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn c1_is_sequential_and_slower_for_the_same_work() {
        let requests = 16;
        let service_ms = 40;

        let p1 = spawn_probe(service_ms).await;
        let c1 = LlmClient::new(&p1.url, "m");
        let t1 = Instant::now();
        let r1 = run_band(
            &c1,
            &prompts(),
            &tiny_band(1, requests),
            tokenization(),
            false,
        )
        .await
        .expect("c=1 band runs");
        let wall1 = t1.elapsed();

        let p8 = spawn_probe(service_ms).await;
        let c8 = LlmClient::new(&p8.url, "m");
        let t8 = Instant::now();
        let r8 = run_band(
            &c8,
            &prompts(),
            &tiny_band(8, requests),
            tokenization(),
            false,
        )
        .await
        .expect("c=8 band runs");
        let wall8 = t8.elapsed();

        let speedup = wall1.as_secs_f64() / wall8.as_secs_f64();
        eprintln!(
            "same {requests} requests: c=1 wall={wall1:?} server_peak={} | \
             c=8 wall={wall8:?} server_peak={} | speedup={speedup:.2}x",
            p1.peak.load(Ordering::SeqCst),
            p8.peak.load(Ordering::SeqCst)
        );

        assert_eq!(p1.peak.load(Ordering::SeqCst), 1, "c=1 must never overlap");
        assert_eq!(r1.window.client_peak_in_flight, 1);
        assert!(r8.window.client_peak_in_flight > 1);
        assert!(
            speedup > 2.0,
            "c=8 must beat c=1 on the same work; got {speedup:.2}x \
             (c=1 {wall1:?}, c=8 {wall8:?})"
        );
    }

    /// §4.4.3 end to end: TTFT and the pooled inter-token gaps are recovered
    /// from a real SSE stream with known spacing. Without this, `ttft_ms` and
    /// `itl_ms` are only ever tested on synthetic samples, and a wiring bug
    /// between the client and the sample would be invisible.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn ttft_and_itl_are_recovered_from_a_real_sse_stream() {
        // 6 tokens, 25 ms apart: TTFT ~25 ms, five gaps of ~25 ms each.
        let url = spawn_sse_probe(6, 25).await;
        let client = LlmClient::new(&url, "m");
        let run = run_band(&client, &prompts(), &tiny_band(2, 8), tokenization(), true)
            .await
            .expect("band runs");

        eprintln!(
            "sse band: ttft_p50={:.1}ms ttft_p95={:.1}ms itl_p50={:.1}ms itl_p95={:.1}ms \
             decode={:.1}tok/s completed={}",
            run.metrics.ttft_p50_ms,
            run.metrics.ttft_p95_ms,
            run.metrics.itl_p50_ms,
            run.metrics.itl_p95_ms,
            run.metrics.decode_tok_s,
            run.metrics.completed
        );

        assert!(run.metrics.completed >= 8, "{:?}", run.metrics);
        // Generous bounds: loopback plus a loaded runner, but a broken wiring
        // reports 0.0 and a chunk-count bug reports a wildly different spacing.
        assert!(
            (10.0..200.0).contains(&run.metrics.ttft_p50_ms),
            "ttft_p50={} ms, expected ~25",
            run.metrics.ttft_p50_ms
        );
        assert!(
            (10.0..200.0).contains(&run.metrics.itl_p50_ms),
            "itl_p50={} ms, expected ~25",
            run.metrics.itl_p50_ms
        );
        assert!(run.metrics.decode_tok_s > 0.0);
        // Five gaps per request, pooled -- not one mean per request.
        let pooled: usize = run.samples.iter().map(|s| s.itl_gaps_ms().len()).sum();
        assert_eq!(
            pooled,
            5 * run.samples.len(),
            "every gap must be pooled, not one summary per request"
        );
    }

    /// §4.4.2 — a shrunken band must SAY it is shrunken. A relaxed run that
    /// reported itself conformant is the fabricated-baseline shape.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_relaxed_band_reports_its_own_non_conformance() {
        let probe = spawn_probe(5).await;
        let client = LlmClient::new(&probe.url, "m");
        let run = run_band(&client, &prompts(), &tiny_band(2, 8), tokenization(), false)
            .await
            .expect("band runs");
        assert!(!run.is_conformant());
        assert!(
            run.protocol_violations
                .iter()
                .any(|v| v.contains("min_wall_clock")),
            "{:?}",
            run.protocol_violations
        );
        assert!(
            run.protocol_violations
                .iter()
                .any(|v| v.contains("quiesce")),
            "{:?}",
            run.protocol_violations
        );
    }

    /// §4.4.7 — `drain_ms` is produced. It had zero producers in this repo while
    /// `perf_gate.sh` has always failed a receipt that omits it.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn drain_ms_and_window_ms_are_produced() {
        let probe = spawn_probe(30).await;
        let client = LlmClient::new(&probe.url, "m");
        let run = run_band(
            &client,
            &prompts(),
            &tiny_band(4, 16),
            tokenization(),
            false,
        )
        .await
        .expect("band runs");
        assert!(run.window.window_ms > 0.0, "{:?}", run.window);
        assert!(run.window.drain_ms >= 0.0, "{:?}", run.window);
        // The last admissions are still in flight when T is stamped, so a
        // concurrent band always has a non-zero drain.
        assert!(run.window.drain_ms > 0.0, "{:?}", run.window);
    }

    /// §4.4.6 — the tokenization block travels with the run and is validated.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_invalid_tokenization_block_refuses_the_run() {
        let probe = spawn_probe(1).await;
        let client = LlmClient::new(&probe.url, "m");
        // `client_tokenizer` with *no* digest is unrepresentable in
        // `TokenizationBlock`; a malformed one is not, and §4.4.6 rejects it.
        let bad = TokenizationBlock::ClientTokenizer {
            tokenizer_sha256: "deadbeef".to_string(),
            counts_special_tokens: true,
            counts_prompt_echo: false,
        };
        let r = run_band(&client, &prompts(), &tiny_band(1, 1), bad, false).await;
        assert!(
            r.is_err(),
            "a run must not start without a valid §4.4.6 block"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_empty_prompt_corpus_refuses_the_run() {
        let probe = spawn_probe(1).await;
        let client = LlmClient::new(&probe.url, "m");
        let r = run_band(&client, &[], &tiny_band(1, 1), tokenization(), false).await;
        assert!(r.is_err());
    }

    /// §4.4.4/§4.4.5 — the run carries an interval and the raw samples that
    /// re-derive it.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn the_run_carries_samples_and_a_reproducible_interval() {
        let probe = spawn_probe(5).await;
        let client = LlmClient::new(&probe.url, "m");
        let run = run_band(
            &client,
            &prompts(),
            &tiny_band(4, 24),
            tokenization(),
            false,
        )
        .await
        .expect("band runs");

        assert_eq!(run.samples.len(), run.window.requested);
        assert!(run.samples.iter().any(|s| s.in_flight_at_start > 1));
        let ci = run.agg_ci.as_ref().expect("n >= 2");
        assert_eq!(ci.seed, 2026);
        assert_eq!(ci.resamples, 10_000);
        assert_eq!(ci.resampling_unit, "whole_request");
        let again = bootstrap_agg_tok_s_ci(&run.samples, 0.95).expect("n >= 2");
        assert_eq!(&again, ci, "the interval must re-derive from the samples");
    }
}
