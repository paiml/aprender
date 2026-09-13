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
use crate::perf_gate::drain::StreamMode;
use crate::perf_gate::metrics::{BandMetrics, RequestSample};
use crate::perf_gate::protocol::{BandConfig, ClientModel, Outcome, TokenizationBlock, REPLICATES};
use crate::perf_gate::window::{WindowController, WindowReport};

use super::client::ChatResponse;
use super::client::{ChatRequest, LlmClient, LlmClientError};

/// PP-4 / PP-27 / PP-28 — the per-request facts §4.4.5's [`RequestSample`] has
/// no field for.
///
/// `RequestSample` is the §4.4.5 retention shape and is written verbatim into
/// the gzipped samples file that `perf_gate.sh` re-derives the interval from;
/// widening it would change every retained row and invalidate the digests of
/// every sample file already committed. These three facts belong to the same
/// request and are carried beside it, one entry per `samples` entry, in the
/// same order.
///
/// Each is `Option` because "the server did not report it" is a fact this
/// harness records rather than fills in: PP-13 makes a harness-computed
/// substitute for a server-reported field schema-fatal.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct RequestExtra {
    /// PP-28 — the `n_predict` this request was ISSUED with (`max_tokens` on
    /// the wire). Recorded at issue time, so it is known even for a request
    /// that failed, and per-request so a ragged workload (W2) is not counted
    /// short for being ragged.
    pub expected_tokens: Option<u32>,
    /// §3 `prefill` — the server's `timings.prompt_ms` for this request.
    /// `None` when the server reported none; never a client-side estimate.
    pub prefill_ms: Option<f64>,
    /// PP-27 — what the server declared on this request's FIRST SSE chunk.
    pub stream_mode: Option<StreamMode>,
}

/// PP-27 — the mode the band as a whole ran in.
///
/// Read from the COMPLETED requests only: a failed or timed-out request
/// received no first chunk, and letting its absent declaration undeclare the
/// band would mean one transport error silently moved `dec`/`ttft`/`itl` into
/// `unproduced_fields` for a band that streamed perfectly well.
///
/// The rule is deliberately asymmetric. One `replayed` request makes the band
/// replayed — a server that replayed once can replay again, and the latency
/// figures are then a property of the replay. One UNDECLARED request makes the
/// band undeclared, because `live` is a claim and an absent claim is not a
/// weaker version of it.
#[must_use]
fn band_stream_mode(samples: &[RequestSample], extras: &[RequestExtra]) -> Option<StreamMode> {
    let mut declared_live = false;
    for (s, e) in samples.iter().zip(extras.iter()) {
        if s.outcome != Outcome::Completed {
            continue;
        }
        match e.stream_mode {
            None => return None,
            Some(StreamMode::Replayed) => return Some(StreamMode::Replayed),
            Some(StreamMode::Live) => declared_live = true,
        }
    }
    declared_live.then_some(StreamMode::Live)
}

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
    /// PP-4 / PP-27 / PP-28 facts, one per `samples` entry, in the same order.
    pub extras: Vec<RequestExtra>,
    /// PP-27 — what the server declared over this band's completed requests.
    /// `None` on a non-streaming run, and on a server that declared nothing.
    pub stream_mode: Option<StreamMode>,
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
    samples: Arc<Mutex<Vec<(RequestSample, RequestExtra)>>>,
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
    /// §3 — the server's `timings.prompt_ms`, when it reported one.
    prefill_ms: Option<f64>,
    /// PP-27 — what the server declared on the first chunk.
    stream_mode: Option<StreamMode>,
}

fn observe_blocking(response: &ChatResponse) -> Observed {
    // §3: every token count is the SERVER's `usage`. A response without one
    // reports zero, and `BandInput::derive` refuses a completed request with
    // zero generated tokens rather than crediting it to `agg`'s numerator.
    let (generated_tokens, prompt_tokens) = response
        .usage
        .as_ref()
        .map_or((0, 0), |u| (u.completion_tokens, u.prompt_tokens));
    Observed {
        // A non-streaming response has no per-token arrival information at all.
        // Empty rather than synthesised: §4.4.3's `itl_ms` and `decode_tok_s` are
        // undefined here, and inventing evenly-spaced timestamps would make an
        // unmeasured quantity look measured.
        token_offsets: Vec::new(),
        generated_tokens,
        prompt_tokens,
        // A blocking response carries no phase split and declares no mode.
        prefill_ms: None,
        stream_mode: None,
    }
}

/// Issue one request through the shared [`LlmClient`] — the same client
/// `LoadTest`/`send_one_request` uses, so both drive one transport (§4.4.8).
///
/// PP-27: the token counts on the streaming path are the SERVER's, taken from
/// the terminal chunk's `usage`. The chunk-count fallback that used to stand
/// here — `generated_tokens = token_timestamps.len()` — is gone, and cannot
/// come back: [`super::client::StreamedChatResponse::usage`] is not an
/// `Option`, so a stream without terminal `usage` never produces a response to
/// read counts from. It is an `Err`, and this function counts it as a failed
/// request.
async fn issue(client: &LlmClient, prompt: &ChatRequest, stream: bool) -> Option<Observed> {
    if stream {
        let streamed = client.chat_completion_stream(prompt).await.ok()?;
        return Some(Observed {
            token_offsets: streamed.token_timestamps,
            generated_tokens: streamed.usage.completion_tokens,
            prompt_tokens: streamed.usage.prompt_tokens,
            prefill_ms: streamed.timings.and_then(|t| t.prompt_ms),
            stream_mode: streamed.stream_mode,
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
        let observed = match &timed_out {
            Ok(observed) => observed.as_ref(),
            Err(_elapsed) => None,
        };
        let mut sample = sample_from(slot, w.id, (start_s, end_s), drained, observed);
        if timed_out.is_err() {
            sample.outcome = Outcome::Timeout;
        }
        let extra = RequestExtra {
            // PP-28: what this request ASKED for, recorded at issue time from
            // the corpus record itself. Reading it back off the response would
            // make the pin unfalsifiable -- the whole point is to compare what
            // was asked with what came back.
            expected_tokens: prompt.max_tokens,
            prefill_ms: observed.and_then(|o| o.prefill_ms),
            stream_mode: observed.and_then(|o| o.stream_mode),
        };
        lock(&w.samples).push((sample, extra));
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
    let samples: Arc<Mutex<Vec<(RequestSample, RequestExtra)>>> = Arc::new(Mutex::new(Vec::new()));
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
    collected.sort_by_key(|(s, _)| s.index);
    let (collected, extras): (Vec<RequestSample>, Vec<RequestExtra>) =
        collected.into_iter().unzip();

    let metrics = BandMetrics::from_samples(band.concurrency, &collected);
    let agg_ci = bootstrap_agg_tok_s_ci(&collected, 0.95);
    let protocol_violations = violations(band, warmup_completed, &metrics, &window);
    let stream_mode = band_stream_mode(&collected, &extras);

    Ok(BandRun {
        config: band.clone(),
        client_model: band.client_model,
        tokenization,
        metrics,
        window,
        samples: collected,
        extras,
        stream_mode,
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
    ///
    /// PP-27 obliges it to declare `stream_mode` on the first chunk and to
    /// carry `usage` on the terminal one: the client REFUSES a stream that does
    /// neither, so a probe without them would test a refusal path rather than
    /// the recovery this test exists for. `completion_tokens` is deliberately
    /// NOT `tokens` — the server's count and the frame count are different
    /// quantities, and the fixture makes them differ so a regression back to
    /// counting frames is visible.
    async fn serve_sse(mut sock: tokio::net::TcpStream, tokens: usize, gap_ms: u64) {
        consume_request(&mut sock).await;
        let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\
                    Cache-Control: no-cache\r\nConnection: close\r\n\r\n";
        if sock.write_all(head.as_bytes()).await.is_err() {
            return;
        }
        let first = "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"}}],\
                     \"stream_mode\":\"live\"}\n\n";
        if sock.write_all(first.as_bytes()).await.is_err() {
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
        let terminal = format!(
            "data: {{\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"length\"}}],\
             \"usage\":{{\"prompt_tokens\":512,\"completion_tokens\":{SERVER_COMPLETION_TOKENS},\
             \"total_tokens\":{}}},\"timings\":{{\"prompt_n\":512,\"prompt_ms\":40.0,\
             \"predicted_n\":{SERVER_COMPLETION_TOKENS},\"predicted_ms\":200.0}}}}\n\n",
            512 + SERVER_COMPLETION_TOKENS
        );
        let _ = sock.write_all(terminal.as_bytes()).await;
        let _ = sock.write_all(b"data: [DONE]\n\n").await;
        let _ = sock.flush().await;
        let _ = sock.shutdown().await;
    }

    /// What the SSE probe's terminal `usage` declares. Different from the frame
    /// count on purpose (see [`serve_sse`]).
    const SERVER_COMPLETION_TOKENS: u32 = 128;

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
            // PP-28: the fixture carries the §5.1 pin the PRODUCTION corpus
            // carries (`PromptRecord::into_request`, prompts.rs), because the
            // fixture's job is to be the shape the band actually issues. With
            // `None` here no test could notice that the streaming transport
            // dropped both fields -- and for #2746 none did.
            seed: Some(0),
            ignore_eos: Some(true),
            stream_options: None,
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
        // 14 samples over c=4 deliberately does NOT divide evenly. With a
        // multiple of c (the old `16`), all four workers complete their last
        // request together, T is stamped after the final completion, and
        // `drain_ms` is legitimately 0.0 -- so the assertion below was a
        // statement about scheduling, not an invariant. It passed locally on
        // jitter and failed 3/3 in CI with
        // `requested: 16, drain_ms: 0.0, client_peak_in_flight: 4`.
        //
        // With 14, reaching the sample target necessarily leaves workers
        // mid-request, so a drain exists for a reason arithmetic guarantees
        // rather than for a reason the scheduler happens to supply.
        let run = run_band(
            &client,
            &prompts(),
            &tiny_band(4, 14),
            tokenization(),
            false,
        )
        .await
        .expect("band runs");
        assert!(run.window.window_ms > 0.0, "{:?}", run.window);
        assert!(run.window.drain_ms >= 0.0, "{:?}", run.window);
        // Keeps its teeth: `>= 0.0` alone would pass a hardcoded zero, which is
        // exactly the "producer emits nothing" defect this test exists to catch.
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

    // =====================================================================
    // PP-27 / PP-28 — what the band records about the transport it used.
    //
    //  case                                        | band mode  | why
    //  --------------------------------------------|------------|------------
    //  every completed request declared live       | Live       | conformant
    //  one completed request declared replayed     | Replayed   | one replay
    //                                              |            | poisons it
    //  one completed request declared nothing      | None       | `live` is a
    //                                              |            | claim
    //  a FAILED request declared nothing [BOUNDARY]| unchanged  | it got no
    //                                              |            | first chunk
    //  no completed request at all      [BOUNDARY] | None       | nothing to
    //                                              |            | declare
    // =====================================================================

    fn declared(
        index: usize,
        outcome: Outcome,
        mode: Option<StreamMode>,
    ) -> (RequestSample, RequestExtra) {
        (
            RequestSample {
                index,
                worker: 0,
                start_s: 0.0,
                end_s: 1.0,
                token_times_s: Vec::new(),
                generated_tokens: 128,
                prompt_tokens: 512,
                outcome,
                in_flight_at_start: 1,
                drained: false,
            },
            RequestExtra {
                expected_tokens: Some(128),
                prefill_ms: None,
                stream_mode: mode,
            },
        )
    }

    fn band_mode_of(rows: Vec<(RequestSample, RequestExtra)>) -> Option<StreamMode> {
        let (samples, extras): (Vec<_>, Vec<_>) = rows.into_iter().unzip();
        band_stream_mode(&samples, &extras)
    }

    #[test]
    fn a_band_whose_every_request_declared_live_is_live() {
        let rows = (0..4)
            .map(|i| declared(i, Outcome::Completed, Some(StreamMode::Live)))
            .collect();
        assert_eq!(band_mode_of(rows), Some(StreamMode::Live));
    }

    #[test]
    fn one_replayed_request_makes_the_band_replayed() {
        let mut rows: Vec<_> = (0..4)
            .map(|i| declared(i, Outcome::Completed, Some(StreamMode::Live)))
            .collect();
        rows[2] = declared(2, Outcome::Completed, Some(StreamMode::Replayed));
        assert_eq!(band_mode_of(rows), Some(StreamMode::Replayed));
    }

    #[test]
    fn one_undeclared_request_makes_the_band_undeclared() {
        // `live` is a CLAIM. An absent claim is not a weaker version of it, and
        // reading it as one is how a replayed band buys a latency number.
        let mut rows: Vec<_> = (0..4)
            .map(|i| declared(i, Outcome::Completed, Some(StreamMode::Live)))
            .collect();
        rows[1] = declared(1, Outcome::Completed, None);
        assert_eq!(band_mode_of(rows), None);
    }

    #[test]
    fn a_failed_requests_absent_declaration_does_not_undeclare_the_band() {
        // DISCRIMINATION, and the reason the rule reads only completed
        // requests: a request that never got a first chunk declares nothing BY
        // CONSTRUCTION, and letting it undeclare the band would mean one
        // transport error moved dec/ttft/itl into `unproduced_fields` for a
        // band that streamed perfectly well.
        let rows = vec![
            declared(0, Outcome::Completed, Some(StreamMode::Live)),
            declared(1, Outcome::Failed, None),
            declared(2, Outcome::Timeout, None),
            declared(3, Outcome::Completed, Some(StreamMode::Live)),
        ];
        assert_eq!(band_mode_of(rows), Some(StreamMode::Live));
    }

    #[test]
    fn a_band_with_no_completed_request_declares_nothing() {
        let rows = vec![declared(0, Outcome::Failed, Some(StreamMode::Live))];
        assert_eq!(band_mode_of(rows), None);
        assert_eq!(band_mode_of(Vec::new()), None);
    }

    /// PP-27 end to end: the mode the SERVER declared on the first chunk
    /// reaches `BandRun`, and the per-request server prefill duration and the
    /// issued `n_predict` reach `extras`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn the_band_records_what_the_server_declared() {
        let url = spawn_sse_probe(6, 5).await;
        let client = LlmClient::new(&url, "m");
        let run = run_band(&client, &prompts(), &tiny_band(2, 8), tokenization(), true)
            .await
            .expect("band runs");

        assert_eq!(run.stream_mode, Some(StreamMode::Live));
        assert_eq!(run.extras.len(), run.samples.len());
        let completed: Vec<&RequestExtra> = run
            .samples
            .iter()
            .zip(run.extras.iter())
            .filter(|(s, _)| s.outcome == Outcome::Completed)
            .map(|(_, e)| e)
            .collect();
        assert!(!completed.is_empty(), "{:?}", run.metrics);
        for e in completed {
            assert_eq!(e.expected_tokens, Some(128), "PP-28: the issued n_predict");
            assert_eq!(e.prefill_ms, Some(40.0), "§3: the SERVER's prompt_ms");
            assert_eq!(e.stream_mode, Some(StreamMode::Live));
        }
    }

    /// PP-27 — the token count is the SERVER's, not the frame count.
    ///
    /// The probe emits SIX content frames and declares 128 completion tokens.
    /// Before this change `issue` fell back to `token_timestamps.len()` when
    /// `usage` was absent, and nothing distinguished the two numbers because
    /// the probe never sent a `usage` block at all. A regression to counting
    /// frames reports 6 here.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn decode_uses_the_server_count_not_the_chunk_count() {
        let url = spawn_sse_probe(6, 5).await;
        let client = LlmClient::new(&url, "m");
        let run = run_band(&client, &prompts(), &tiny_band(2, 8), tokenization(), true)
            .await
            .expect("band runs");
        let counts: Vec<u32> = run
            .samples
            .iter()
            .filter(|s| s.outcome == Outcome::Completed)
            .map(|s| s.generated_tokens)
            .collect();
        assert!(!counts.is_empty(), "{:?}", run.metrics);
        for n in counts {
            assert_eq!(
                n, SERVER_COMPLETION_TOKENS,
                "the server declared {SERVER_COMPLETION_TOKENS} completion tokens; a frame count \
                 would report 6"
            );
        }
        for s in run
            .samples
            .iter()
            .filter(|s| s.outcome == Outcome::Completed)
        {
            assert_eq!(
                s.token_times_s.len(),
                6,
                "six frames were actually observed"
            );
            assert_eq!(
                s.prompt_tokens, 512,
                "§3: prompt_tokens is the server's too"
            );
        }
    }

    /// A server that answers a stream with no terminal `usage` fails the
    /// REQUEST rather than yielding a band counted from frames (PP-27).
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_stream_without_usage_fails_the_request_rather_than_being_counted() {
        async fn serve_usageless(mut sock: tokio::net::TcpStream) {
            consume_request(&mut sock).await;
            let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\
                        Connection: close\r\n\r\n";
            let _ = sock.write_all(head.as_bytes()).await;
            let _ = sock
                .write_all(
                    b"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"a \"}}]}\n\n",
                )
                .await;
            let _ = sock.write_all(b"data: [DONE]\n\n").await;
            let _ = sock.flush().await;
            let _ = sock.shutdown().await;
        }
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind loopback");
        let addr = listener.local_addr().expect("local addr");
        tokio::spawn(async move {
            while let Ok((sock, _)) = listener.accept().await {
                tokio::spawn(serve_usageless(sock));
            }
        });

        let client = LlmClient::new(&format!("http://{addr}"), "m");
        let run = run_band(&client, &prompts(), &tiny_band(1, 4), tokenization(), true)
            .await
            .expect("the band still runs; the REQUESTS fail");
        assert_eq!(run.metrics.completed, 0, "{:?}", run.metrics);
        assert!(run.metrics.errors > 0, "{:?}", run.metrics);
        assert_eq!(run.stream_mode, None, "no completed request declared one");
    }
}
