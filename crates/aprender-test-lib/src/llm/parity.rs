//! #4971 V3-d: the perf041 probe's requests.
//!
//! What a run means is decided in [`crate::perf_gate::parity`], under default
//! features. This module only fires the requests: two `m = 1` references one
//! after the other, then for each `c` of the ladder `c` identical requests at
//! once, reading the server log from the band's own offset to learn the batch
//! it formed. Every request asks for `logprobs` with the sampler's
//! `top_logprobs`, so each band can record the reference's top-2 margin.

use std::path::PathBuf;

use futures::future::join_all;

use super::client::{ChatMessage, ChatRequest, LlmClient, Role};
use crate::perf_gate::parity::{
    batch_formed_since, check_reference, evaluate_band, log_offset, run_verdict, ProbeBand,
    ProbePolicy, ProbeReference, ProbeSample, ProbeSampler, SampleResult,
};
use crate::perf_gate::witness::BatchInvariance;

/// One probe run's settings.
#[derive(Debug, Clone)]
pub struct ParityProbe {
    /// The prompt every request sends.
    pub prompt: String,
    /// The concurrencies to fire, in order.
    pub ladder: Vec<u32>,
    /// What the bands are judged against.
    pub policy: ProbePolicy,
    /// What every request carries.
    pub sampler: ProbeSampler,
    /// The server's log, where the scheduler writes `Batch m=N done`.
    pub server_log: PathBuf,
}

/// What a probe run measured.
#[derive(Debug, Clone, PartialEq)]
pub struct ParityRun {
    /// The two `m = 1` decodes, judged.
    pub reference: ProbeReference,
    /// One per `c` fired. Empty when the reference was not stable.
    pub bands: Vec<ProbeBand>,
    /// [`run_verdict`] over the two.
    pub result: BatchInvariance,
}

impl ParityProbe {
    /// The request every slot sends. The model is left to the client.
    #[must_use]
    pub fn request(&self) -> ChatRequest {
        let prompt = ChatMessage {
            role: Role::User,
            content: self.prompt.clone(),
        };
        ChatRequest {
            temperature: Some(self.sampler.temperature),
            max_tokens: Some(self.sampler.max_tokens),
            seed: Some(self.sampler.seed),
            ignore_eos: Some(self.sampler.ignore_eos),
            logprobs: Some(true),
            top_logprobs: Some(self.sampler.top_logprobs),
            ..ChatRequest::new("", vec![prompt])
        }
    }

    /// The two references, then each band. A reference that is not stable
    /// ends the run with no bands: no batched difference after it could be
    /// laid at batching's door.
    pub async fn run(&self, client: &LlmClient) -> ParityRun {
        let request = self.request();
        let first = fire(client, &request).await;
        let second = fire(client, &request).await;
        let reference = check_reference(&first, &second, &self.policy);
        let m1 = match first {
            Ok(m1) if reference.stable => m1.logprobs,
            _ => {
                return ParityRun {
                    result: run_verdict(&reference, &[]),
                    reference,
                    bands: Vec::new(),
                }
            }
        };
        let mut bands = Vec::with_capacity(self.ladder.len());
        for &c in &self.ladder {
            let offset = log_offset(&self.server_log);
            let samples = join_all((0..c).map(|_| fire(client, &request))).await;
            let m_formed = batch_formed_since(&self.server_log, offset);
            bands.push(evaluate_band(c, m_formed, &m1, &samples, &self.policy));
        }
        ParityRun {
            result: run_verdict(&reference, &bands),
            reference,
            bands,
        }
    }
}

/// One streamed request, as the probe records it.
async fn fire(client: &LlmClient, request: &ChatRequest) -> SampleResult {
    client
        .chat_completion_stream(request)
        .await
        .map(|r| ProbeSample {
            logprobs: r.logprobs,
            completion_tokens: Some(r.usage.completion_tokens),
            finish_reason: r.finish_reason,
        })
        .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use std::io::Write as _;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use serde_json::json;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    use super::*;

    /// What the replay server sends for the request with this global index:
    /// `(token, gap to the runner-up)` per step, and a scheduler line to log.
    type Plan = Arc<dyn Fn(usize) -> (Vec<(String, f64)>, Option<String>) + Send + Sync>;

    const POLICY: ProbePolicy = ProbePolicy {
        declared_min: 4,
        n_predict: 8,
        max_constant_run: 4,
    };

    fn canned(prefix: &str) -> Vec<(String, f64)> {
        (0..8).map(|i| (format!("{prefix}{i} "), 1.0)).collect()
    }

    /// One SSE frame carrying one token and its logprob entry.
    fn token_frame(token: &str, gap: f64) -> String {
        let entry = json!({
            "token": token,
            "logprob": -0.1,
            "bytes": token.as_bytes(),
            "top_logprobs": [
                {"token": token, "logprob": -0.1},
                {"token": "~", "logprob": -0.1 - gap},
            ],
        });
        let chunk =
            json!({"choices": [{"delta": {"content": token}, "logprobs": {"content": [entry]}}]});
        format!("data: {chunk}\n\n")
    }

    fn sse_body(tokens: &[(String, f64)]) -> String {
        let n = tokens.len();
        let usage = json!({"prompt_tokens": 5, "completion_tokens": n, "total_tokens": n + 5});
        let terminal =
            json!({"choices": [{"delta": {}, "finish_reason": "length"}], "usage": usage});
        std::iter::once(
            "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n".to_string(),
        )
        .chain(tokens.iter().map(|(t, gap)| token_frame(t, *gap)))
        .chain([
            format!("data: {terminal}\n\n"),
            "data: [DONE]\n\n".to_string(),
        ])
        .collect()
    }

    /// Read one request through its body, so the client never sees a reset.
    async fn read_request(sock: &mut tokio::net::TcpStream) {
        let mut buf = Vec::new();
        let mut chunk = [0_u8; 4096];
        while let Ok(n @ 1..) = sock.read(&mut chunk).await {
            buf.extend_from_slice(&chunk[..n]);
            let Some(end) = buf.windows(4).position(|w| w == b"\r\n\r\n") else {
                continue;
            };
            let head = String::from_utf8_lossy(&buf[..end]).to_ascii_lowercase();
            let body = head
                .lines()
                .find_map(|l| l.strip_prefix("content-length:"))
                .and_then(|v| v.trim().parse::<usize>().ok())
                .unwrap_or(0);
            if buf.len() >= end + 4 + body {
                return;
            }
        }
    }

    /// The script's `_replay_server`: canned SSE per request index, and the
    /// scheduler's line appended to the log from inside the handler, as a real
    /// server does, so the band offsets are exercised for real.
    async fn replay(plan: Plan, log: PathBuf) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind loopback");
        let addr = listener.local_addr().expect("local addr");
        let counter = Arc::new(AtomicUsize::new(0));
        tokio::spawn(async move {
            while let Ok((mut sock, _)) = listener.accept().await {
                let (plan, log, counter) = (Arc::clone(&plan), log.clone(), Arc::clone(&counter));
                tokio::spawn(async move {
                    read_request(&mut sock).await;
                    let (tokens, line) = plan(counter.fetch_add(1, Ordering::SeqCst));
                    if let Some(line) = line {
                        let mut file = std::fs::OpenOptions::new()
                            .append(true)
                            .open(&log)
                            .expect("open log");
                        writeln!(file, "{line}").expect("append log");
                    }
                    let body = sse_body(&tokens);
                    let head = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\
                         Content-Length: {}\r\nConnection: close\r\n\r\n",
                        body.len()
                    );
                    let _ = sock.write_all((head + &body).as_bytes()).await;
                    let _ = sock.shutdown().await;
                });
            }
        });
        format!("http://{addr}")
    }

    /// Run the probe over the ladder [1, 2] against `plan`. Requests 0 and 1
    /// are the references, 2 is c=1, and 3 and 4 are c=2.
    async fn probe(
        plan: impl Fn(usize) -> (Vec<(String, f64)>, Option<String>) + Send + Sync + 'static,
    ) -> ParityRun {
        let dir = tempfile::tempdir().expect("tempdir");
        let log = dir.path().join("server.log");
        std::fs::write(&log, "server starting\n").expect("seed log");
        let url = replay(Arc::new(plan), log.clone()).await;
        let probe = ParityProbe {
            prompt: "Write an essay on compilers.".to_string(),
            ladder: vec![1, 2],
            policy: POLICY,
            sampler: ProbeSampler {
                temperature: 0.0,
                seed: 0,
                ignore_eos: true,
                max_tokens: POLICY.n_predict,
                top_logprobs: 2,
            },
            server_log: log,
        };
        probe.run(&LlmClient::new(url, "replay")).await
    }

    /// The batch line a request logs: m=1 for the c=1 band, m=2 for c=2.
    fn batch_line(index: usize) -> Option<String> {
        match index {
            2 => Some("[PMAT-044] Batch m=1 done in 1.0ms (9.0 tok/s/slot)".to_string()),
            3.. => Some("[PMAT-044] Batch m=2 done in 1.0ms (9.0 tok/s/slot)".to_string()),
            _ => None,
        }
    }

    #[tokio::test]
    async fn identical_streams_with_batches_formed_pass_every_band() {
        let run = probe(|i| (canned("a"), batch_line(i))).await;
        assert_eq!(run.result, BatchInvariance::Pass, "{run:?}");
        assert!(run.reference.stable);
        let formed: Vec<u32> = run.bands.iter().map(|b| b.m_formed).collect();
        assert_eq!(formed, vec![1, 2], "each band reads only its own lines");
        assert_eq!(run.bands[1].divergence_at, Some(8));
    }

    /// `witness_no_batch_formed_is_unmeasurable`, end to end.
    #[tokio::test]
    async fn a_box_that_never_batches_is_unmeasurable_never_a_pass() {
        let run = probe(|_| (canned("a"), None)).await;
        assert_eq!(run.result, BatchInvariance::Unmeasurable, "{run:?}");
        assert_eq!(run.bands[1].result, BatchInvariance::Unmeasurable);
    }

    /// `witness_constant_token_m3`, end to end: the batched slots freeze.
    #[tokio::test]
    async fn frozen_batched_slots_fail() {
        let frozen = |i: usize| {
            let tokens = if i >= 3 {
                vec![("z".to_string(), 1.0); 8]
            } else {
                canned("a")
            };
            (tokens, batch_line(i))
        };
        let run = probe(frozen).await;
        assert_eq!(run.result, BatchInvariance::Fail, "{run:?}");
        assert_eq!(run.bands[1].result, BatchInvariance::Fail);
    }

    /// The margin, end to end: the batch parts from m=1 at token 1, where the
    /// reference's own choice was a near-tie; the band passes and records it.
    #[tokio::test]
    async fn a_near_tie_flip_is_recorded_with_its_margin() {
        let flip = |i: usize| {
            let mut tokens = canned("a");
            tokens[1] = if i >= 3 {
                ("flip ".to_string(), 0.01)
            } else {
                ("a1 ".to_string(), 0.01)
            };
            (tokens, batch_line(i))
        };
        let run = probe(flip).await;
        assert_eq!(run.result, BatchInvariance::Pass, "{run:?}");
        assert_eq!(run.bands[1].divergence_at, Some(1));
        let margin = run.bands[1].top2_margin_at_divergence.expect("margin");
        assert!((margin - 0.01).abs() < 1e-9, "{margin}");
    }

    /// `witness_short_reference_is_unmeasurable`, and an unstable reference:
    /// either ends the run before any band.
    #[tokio::test]
    async fn a_reference_that_cannot_be_trusted_ends_the_run_with_no_bands() {
        let unstable = |i: usize| {
            let mut tokens = canned("a");
            if i == 1 {
                tokens[2] = ("X ".to_string(), 1.0);
            }
            (tokens, batch_line(i))
        };
        let run = probe(unstable).await;
        assert_eq!(run.result, BatchInvariance::Unmeasurable, "{run:?}");
        assert!(run.bands.is_empty());
        assert_eq!(run.reference.self_divergence_at, Some(2));
        let short = probe(|i| (canned("a")[..5].to_vec(), batch_line(i))).await;
        assert_eq!(short.result, BatchInvariance::Unmeasurable);
        assert!(short.bands.is_empty());
    }

    #[test]
    fn every_request_asks_for_logprobs_and_pins_the_sampler() {
        let probe = ParityProbe {
            prompt: "p".to_string(),
            ladder: vec![1],
            policy: POLICY,
            sampler: ProbeSampler {
                temperature: 0.0,
                seed: 7,
                ignore_eos: true,
                max_tokens: 8,
                top_logprobs: 2,
            },
            server_log: PathBuf::from("server.log"),
        };
        let body = serde_json::to_value(probe.request()).expect("serialize");
        for (key, want) in [
            ("logprobs", json!(true)),
            ("top_logprobs", json!(2)),
            ("seed", json!(7)),
            ("ignore_eos", json!(true)),
            ("temperature", json!(0.0)),
            ("max_tokens", json!(8)),
        ] {
            assert_eq!(body[key], want, "{key}: {body}");
        }
    }
}
