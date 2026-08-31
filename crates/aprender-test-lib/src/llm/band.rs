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
use crate::perf_gate::tokenizer::{ClientTokenizer, TokenAccounting};
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
    counts: &PromptCounts,
) -> usize {
    let per_worker = band.warmup_requests.div_ceil(band.concurrency).max(1);
    let mut handles = Vec::with_capacity(band.concurrency);
    for worker in 0..band.concurrency {
        let client = client.clone();
        let prompts = prompts.to_vec();
        // Warmup runs the SAME counting path as the sampled window, so a
        // tokenizer that cannot encode something surfaces in the discarded
        // requests rather than in the ones being measured.
        let counts = counts.clone();
        handles.push(tokio::spawn(async move {
            let mut done = 0_usize;
            for k in 0..per_worker {
                let index = (worker + k) % prompts.len();
                let counting = counts.counting(index);
                if issue(&client, &prompts[index], stream, counting)
                    .await
                    .is_some()
                {
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
    /// §4.4.6 — which side counts, and (under `client_tokenizer`) the prompt
    /// counts computed once before the band opened.
    counts: PromptCounts,
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

/// The text a §4.4.6 client-side count is taken over.
///
/// Every message's `content`, in order, joined by a newline, and **nothing
/// else**: no chat template, no role markers, no BOS, no system message. That
/// is the entire point of `client_tokenizer`. Measured on W1 through one model
/// file, the same 256 prompts were reported as **513** by `apr serve` and
/// **534** by `llama-server`, because each server wraps them in its own
/// template before counting — `apr` in a hardcoded 8-token ChatML wrapper
/// (`aprender-serve/src/api/realize_handlers.rs::format_chat_messages`),
/// `llama` in the GGUF's embedded jinja template, which injects Qwen's default
/// system message. Only the wire text is a number both lanes can be compared
/// on. (505/513/534 were measured at `_meta.body_words = 496`; the raw count is
/// 512 since the corpus was retuned to sit at the centre of §4.3.1's band
/// rather than one token above its floor. The deltas are template overhead and
/// move with it.)
///
/// The join is a stated rule rather than an obvious one: W1 is single-message,
/// so the newline is unreachable for the workload the gate measures, but a
/// multi-turn corpus needs the boundary to be *something*, and silently
/// concatenating would merge the last token of one turn with the first of the
/// next.
fn prompt_text(request: &ChatRequest) -> String {
    request
        .messages
        .iter()
        .map(|m| m.content.as_str())
        .collect::<Vec<_>>()
        .join("\n")
}

/// §4.4.6 — how one request's tokens are counted.
///
/// Two arms rather than an `Option<&ClientTokenizer>`, because the client arm
/// carries something the server arm has no analogue for: the prompt's count,
/// computed once before the band opened rather than per request.
#[derive(Clone, Copy)]
enum Counting<'a> {
    /// Take the numbers out of the server's own `usage` block.
    ServerUsage,
    /// Count client-side, with the model's own tokenizer.
    Client {
        /// The counter, whose digest is the one the receipt carries.
        counter: &'a ClientTokenizer,
        /// This prompt's token count, precomputed in [`PromptCounts::build`].
        prompt_tokens: u32,
    },
}

/// The counter and the per-prompt counts, shared by every worker.
///
/// `prompt_tokens` is parallel to the corpus and is empty under `server_usage`.
/// Computing it once matters for more than speed: it is computed **before** the
/// first request, so a corpus the tokenizer cannot encode refuses the run
/// instead of turning every request into a failed sample halfway through a
/// 14-minute sweep.
#[derive(Clone, Default)]
struct PromptCounts {
    counter: Option<Arc<ClientTokenizer>>,
    prompt_tokens: Arc<Vec<u32>>,
}

impl PromptCounts {
    /// Precompute every prompt's client-side count, or nothing under
    /// `server_usage`.
    ///
    /// # Errors
    /// When the declared §4.4.6 block and the supplied counter disagree, or when
    /// the tokenizer cannot encode one of the prompts.
    fn build(
        accounting: &TokenAccounting,
        prompts: &[ChatRequest],
    ) -> Result<Self, LlmClientError> {
        accounting
            .validate()
            .map_err(LlmClientError::HealthCheckFailed)?;
        let Some(counter) = accounting.counter_handle() else {
            return Ok(Self::default());
        };
        let mut prompt_tokens = Vec::with_capacity(prompts.len());
        for (i, request) in prompts.iter().enumerate() {
            let text = prompt_text(request);
            let n = counter.count(&text).map_err(|e| {
                LlmClientError::HealthCheckFailed(format!(
                    "§4.4.6 client_tokenizer: prompt {i} cannot be counted by {}: {e}",
                    counter.origin()
                ))
            })?;
            prompt_tokens.push(n);
        }
        Ok(Self {
            counter: Some(counter),
            prompt_tokens: Arc::new(prompt_tokens),
        })
    }

    /// How to count the request that used corpus entry `index`.
    fn counting(&self, index: usize) -> Counting<'_> {
        match self.counter.as_deref() {
            Some(counter) if !self.prompt_tokens.is_empty() => Counting::Client {
                counter,
                prompt_tokens: self.prompt_tokens[index % self.prompt_tokens.len()],
            },
            // Unreachable while `build` is the only constructor: a counter is
            // always accompanied by one count per prompt. Stated rather than
            // indexed blindly, because the alternative is a panic inside a
            // spawned worker.
            _ => Counting::ServerUsage,
        }
    }
}

/// `(completion, prompt)` token counts for one response.
///
/// # The defect this closes
///
/// The whole of this function used to be
/// `usage.map_or((fallback_tokens, 0), |u| (u.completion_tokens, u.prompt_tokens))`
/// — the server's own numbers, or, when the server reported none, the count of
/// non-empty SSE content deltas paired with a **`prompt_tokens` of 0**. Measured
/// on `llama-server 39173bcac` and `apr serve 0.64.0`, that zero was not an edge
/// case: **neither server emits `usage` in streaming mode at all**, and §4.5
/// requires streaming. `git grep stream_options` on `origin/main` returns zero
/// hits, so the client never asks `llama` for the usage block it would only send
/// on request, and `apr` has none to send. Every streamed W1 request therefore
/// carried `prompt_tokens = 0` into the receipt.
///
/// Under [`Counting::Client`] neither the server's opinion nor the delta count
/// is consulted: the prompt count was computed before the band opened and the
/// completion is counted from the text that actually came back. That also
/// removes the §4.4.6 `counts_special_tokens` disagreement measured between the
/// two servers, where `llama` counts the stop token and `apr` does not — exactly
/// +1 on every naturally terminated request.
///
/// # The client encode is INSIDE the measured span, and that was measured
///
/// [`worker_loop`] stamps `start_s`, awaits [`issue`], then stamps `end_s`;
/// under [`Counting::Client`] the `counter.count(content)` below happens before
/// `issue` returns. So it is inside every per-request latency and, through the
/// last completed request's `end_s`, inside `agg_tok_s`'s denominator. The
/// prompt side is not: `PromptCounts::build` counts every prompt once, before
/// the band opens.
///
/// Measured, release build, canonical Qwen2.5-Coder tokenizer, on a real
/// 128-token W1 completion, `n = 1000` per thread after 200 warm iterations:
///
/// | concurrent encoders | p50 | p95 | p99 | max |
/// |---|---|---|---|---|
/// | 1  | 0.087 ms | 0.099 ms | 0.136 ms | 0.202 ms |
/// | 8  | 0.088 ms | 0.100 ms | 0.140 ms | 0.215 ms |
/// | 16 | 0.102 ms | 0.186 ms | 0.257 ms | 2.611 ms |
///
/// A W1 request against the campaign's model (`qwen2.5-coder-7b-instruct-
/// q4_k_m.gguf`, RTX 4090) takes **868 ms** end to end. The p50 encode at the
/// widest band the campaign runs is therefore **0.012%** of one request, and the
/// 2.6 ms tail is 0.30%. `agg_tok_s`'s denominator is a band-wide span with a
/// §4.4.2 floor of 60 s, and exactly one encode lands inside it — 0.0002%.
///
/// It is also symmetric: one client, one counter, the same corpus on both lanes,
/// and §4.4.6 makes a `tokenization` mismatch between a lane and its comparator
/// FATAL, so the arm where one side pays the encode and the other does not is
/// not a receipt that can be judged. Moving the encode outside the span would
/// trade a measured 0.012% for an unmeasured gap between `end_s` and the
/// response actually being complete, so it stays where it is.
///
/// Returns `None` when the tokenizer refuses the completion, so the request
/// becomes a `Failed` sample rather than a fabricated zero.
fn usage_counts(
    counting: Counting<'_>,
    usage: Option<&Usage>,
    fallback_tokens: u32,
    content: &str,
) -> Option<(u32, u32)> {
    match counting {
        Counting::ServerUsage => Some(usage.map_or((fallback_tokens, 0), |u| {
            (u.completion_tokens, u.prompt_tokens)
        })),
        Counting::Client {
            counter,
            prompt_tokens,
        } => Some((counter.count(content).ok()?, prompt_tokens)),
    }
}

fn observe_blocking(response: &ChatResponse, counting: Counting<'_>) -> Option<Observed> {
    let content = response
        .choices
        .first()
        .map_or("", |c| c.message.content.as_str());
    let (generated_tokens, prompt_tokens) =
        usage_counts(counting, response.usage.as_ref(), 0, content)?;
    Some(Observed {
        // A non-streaming response has no per-token arrival information at all.
        // Empty rather than synthesised: §4.4.3's `itl_ms` and `decode_tok_s` are
        // undefined here, and inventing evenly-spaced timestamps would make an
        // unmeasured quantity look measured.
        token_offsets: Vec::new(),
        generated_tokens,
        prompt_tokens,
    })
}

/// Issue one request through the shared [`LlmClient`] — the same client
/// `LoadTest`/`send_one_request` uses, so both drive one transport (§4.4.8).
async fn issue(
    client: &LlmClient,
    prompt: &ChatRequest,
    stream: bool,
    counting: Counting<'_>,
) -> Option<Observed> {
    if stream {
        let streamed = client.chat_completion_stream(prompt).await.ok()?;
        let observed_tokens = u32::try_from(streamed.token_timestamps.len()).unwrap_or(u32::MAX);
        let (generated_tokens, prompt_tokens) = usage_counts(
            counting,
            streamed.usage.as_ref(),
            observed_tokens,
            &streamed.content,
        )?;
        return Some(Observed {
            token_offsets: streamed.token_timestamps,
            generated_tokens,
            prompt_tokens,
        });
    }
    let timed = client.send(prompt).await.ok()?;
    observe_blocking(&timed.response, counting)
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

        let index = slot.0 % w.prompts.len();
        let prompt = &w.prompts[index];
        let counting = w.counts.counting(index);
        let start_s = w.origin.elapsed().as_secs_f64();
        // §4.4.3 — the 120 s hard per-request timeout, classified as a TIMEOUT
        // in its own counter rather than folded into a generic failure count.
        let timed_out =
            tokio::time::timeout(w.timeout, issue(&w.client, prompt, w.stream, counting)).await;
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
    accounting: &TokenAccounting,
    stream: bool,
) -> Result<BandRun, LlmClientError> {
    if prompts.is_empty() {
        return Err(LlmClientError::HealthCheckFailed(
            "run_band: the prompt corpus is empty; §4.3 requires a fixed workload".to_string(),
        ));
    }
    // §4.4.6 — validates the block AND, under `client_tokenizer`, checks the
    // declared digest against the digest of the tokenizer this run actually
    // opened, then counts every prompt with it. Both happen before the first
    // request: under I-9 a mis-declared cell is a spent run.
    let counts = PromptCounts::build(accounting, prompts)?;
    let tokenization = accounting.block().clone();

    // §4.4.2 — warmup, discarded, then a 5 s quiesce before the first sampled
    // request. Without the quiesce the first sampled requests are measured on a
    // server still finishing warmup work, which is the opposite of what warmup
    // was for.
    let warmup_completed = warmup(client, prompts, band, stream, &counts).await;
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
            counts: counts.clone(),
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
    accounting: &TokenAccounting,
    stream: bool,
) -> Result<Vec<BandRun>, LlmClientError> {
    let mut runs = Vec::with_capacity(REPLICATES);
    for _ in 0..REPLICATES {
        runs.push(run_band(client, prompts, band, accounting, stream).await?);
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

    fn tokenization() -> TokenAccounting {
        TokenAccounting::server_usage(false, false)
    }

    /// The in-tree MiniLM tokenizer, as a real client-side counter.
    ///
    /// Borrowed from SetFit's fixtures rather than downloaded: the point of
    /// these tests is the WIRING — that `run_band` counts client-side and does
    /// not read the server's `usage` — and any real tokenizer proves that.
    /// `perf_gate::tokenizer`'s own tests pin the counter to W1's 512 against
    /// Qwen's vocabulary.
    fn client_counter() -> TokenAccounting {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../aprender-core/tests/fixtures/setfit/tokenizer.json");
        TokenAccounting::client_tokenizer(
            ClientTokenizer::from_file(&path).expect("the MiniLM fixture must load"),
        )
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

        let run = run_band(&client, &prompts(), &band, &tokenization(), false)
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
            &tokenization(),
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
            &tokenization(),
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
        let run = run_band(&client, &prompts(), &tiny_band(2, 8), &tokenization(), true)
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
        let run = run_band(
            &client,
            &prompts(),
            &tiny_band(2, 8),
            &tokenization(),
            false,
        )
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
            &tokenization(),
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
        // `from_parts` refuses it before a `TokenAccounting` -- and therefore a
        // band -- can exist. Under the old signature this block reached
        // `run_band` and was refused there; now it cannot be carried at all.
        let err = TokenAccounting::from_parts(bad, None).expect_err("malformed digest");
        assert!(err.contains("64 lowercase hex"), "{err}");
        assert_eq!(probe.served.load(Ordering::SeqCst), 0);

        // The well-formed counterpart runs.
        let r = run_band(
            &client,
            &prompts(),
            &tiny_band(1, 1),
            &client_counter(),
            false,
        )
        .await;
        assert!(r.is_ok(), "a well-formed client_tokenizer block must run");
    }

    /// §4.4.6 — under `client_tokenizer` the receipt's counts come from the
    /// CLIENT's tokenizer, not from the server's `usage` block.
    ///
    /// The probe answers a fixed body declaring `prompt_tokens: 512` and
    /// `completion_tokens: 128`, and its message content is `"a b c"`. A run
    /// that still reports 512/128 has not been wired; a run that reports the
    /// MiniLM counts of the request and the reply has.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn client_tokenizer_counts_replace_the_servers_usage_block() {
        let probe = spawn_probe(1).await;
        let client = LlmClient::new(&probe.url, "m");
        let accounting = client_counter();
        let counter = accounting.counter().expect("client counter");
        // What the client's own tokenizer says about the two strings on the wire.
        let want_prompt = counter.count("hello").expect("encodes");
        let want_completion = counter.count("a b c").expect("encodes");

        let run = run_band(&client, &prompts(), &tiny_band(2, 8), &accounting, false)
            .await
            .expect("band runs");

        let completed: Vec<_> = run
            .samples
            .iter()
            .filter(|s| s.outcome == Outcome::Completed)
            .collect();
        assert!(!completed.is_empty(), "{:?}", run.metrics);
        for s in &completed {
            assert_eq!(
                s.prompt_tokens, want_prompt,
                "the SERVER said 512; the client counted {want_prompt}"
            );
            assert_eq!(
                s.generated_tokens, want_completion,
                "the SERVER said 128; the client counted {want_completion}"
            );
        }
        // The negative control: the server's numbers are nowhere in the samples.
        assert!(completed.iter().all(|s| s.prompt_tokens != 512));
        assert!(completed.iter().all(|s| s.generated_tokens != 128));
        // ... and the receipt names the tokenizer that produced them.
        assert_eq!(
            run.tokenization,
            TokenizationBlock::ClientTokenizer {
                tokenizer_sha256: counter.tokenizer_sha256().to_string(),
                counts_special_tokens: crate::perf_gate::COUNTS_SPECIAL_TOKENS,
                counts_prompt_echo: crate::perf_gate::COUNTS_PROMPT_ECHO,
            }
        );
    }

    /// A `client_tokenizer` block whose digest names a file the run did not
    /// open must refuse the band — before a single request is issued.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_borrowed_digest_refuses_the_band_before_it_spends_anything() {
        // The probe exists only so `served` can prove nothing was spent.
        let probe = spawn_probe(1).await;
        let real = client_counter();
        let counter = real.counter().expect("counter");

        // Same tokenizer, one trailing newline: a different digest for what is
        // obviously "the same" vocabulary. This is the shape the guard exists
        // to catch.
        let bytes = std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../aprender-core/tests/fixtures/setfit/tokenizer.json"),
        )
        .expect("fixture");
        let mut other = bytes.clone();
        other.push(b'\n');
        let other = ClientTokenizer::from_bytes(&other, "variant").expect("loads");
        assert_ne!(other.tokenizer_sha256(), counter.tokenizer_sha256());

        let forged = TokenizationBlock::ClientTokenizer {
            tokenizer_sha256: other.tokenizer_sha256().to_string(),
            counts_special_tokens: crate::perf_gate::COUNTS_SPECIAL_TOKENS,
            counts_prompt_echo: crate::perf_gate::COUNTS_PROMPT_ECHO,
        };
        assert!(forged.validate().is_ok(), "well-formed by the OLD rule");

        let err = TokenAccounting::from_parts(
            forged,
            Some(ClientTokenizer::from_bytes(&bytes, "real").expect("loads")),
        )
        .expect_err("a borrowed digest must not reach a band");
        assert!(err.contains("did not open"), "{err}");
        assert_eq!(
            probe.served.load(Ordering::SeqCst),
            0,
            "nothing may be spent before the §4.4.6 block is checked"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_empty_prompt_corpus_refuses_the_run() {
        let probe = spawn_probe(1).await;
        let client = LlmClient::new(&probe.url, "m");
        let r = run_band(&client, &[], &tiny_band(1, 1), &tokenization(), false).await;
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
            &tokenization(),
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
