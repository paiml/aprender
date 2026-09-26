//! REX-03 harness: drives a resident `apr serve` (OpenAI-compatible
//! `/v1/chat/completions`) with the fixed prompt, one corpus item at a time,
//! and turns each reply into a `review-experiment-receipt-v1` row.
//!
//! The harness never truncates a diff: an over-long one is sent as is, and a
//! context refusal from the serve becomes `NotRun{ContextOverflow}`.

use crate::corpus::{sha256_hex, Item};
use crate::receipt::{
    parse_verdict, Arm, Decoding, Load, NotRun, Output, Receipt, ServerTimings, Timings, Tokens,
    Verdict, SCHEME,
};
use crate::stats::SplitMix64;
use serde_json::{json, Value};
use std::io::Read;
use std::time::{Duration, Instant};

/// Largest reply body read, in bytes (ureq's `into_string` stops at 10 MB).
const MAX_BODY: u64 = 256 << 20;

/// Fixed decoding (§2.1 `[A]`): greedy, seed = the epic number, 512 tokens
/// (the prompt asks for a verdict line and ≤ 5 short bullets).
pub const DECODING: Decoding = Decoding {
    temperature: 0.0,
    seed: 4354,
    max_tokens: 512,
};

/// Per-request timeout. A review that takes longer is a `ServeError`.
pub const TIMEOUT: Duration = Duration::from_secs(900);

/// The user message: the prompt file verbatim, a blank line, then the diff
/// in a fenced block. Fixed; its sha is part of `request_sha256`.
#[must_use]
pub fn compose(prompt: &str, diff: &str) -> String {
    let nl = if diff.ends_with('\n') { "" } else { "\n" };
    format!("{}\n\n```diff\n{diff}{nl}```\n", prompt.trim_end())
}

/// The exact request body sent for one item.
/// (`json!` expands to an `unwrap` of an infallible `to_value` on these
/// string and number fields, hence the scoped allow.)
#[allow(clippy::disallowed_methods)]
#[must_use]
pub fn request_body(model: &str, prompt: &str, diff: &str) -> Value {
    json!({
        "model": model,
        "messages": [{"role": "user", "content": compose(prompt, diff)}],
        "temperature": DECODING.temperature,
        "seed": DECODING.seed,
        "max_tokens": DECODING.max_tokens,
        "stream": false,
    })
}

/// Run order for one cell (analysis plan §Seeds): Fisher–Yates on the ids,
/// sorted first, driven by SplitMix64(seed).
#[must_use]
pub fn run_order(ids: &[String], seed: u64) -> Vec<String> {
    let mut v = ids.to_vec();
    v.sort();
    let mut rng = SplitMix64::new(seed);
    for i in (1..v.len()).rev() {
        v.swap(i, rng.below(i + 1));
    }
    v
}

/// The determinism re-run subset: the first 10 % (rounded up) of the
/// seed-4354 order of the test items.
#[must_use]
pub fn rerun_subset(test_order: &[String]) -> Vec<String> {
    test_order[..test_order.len().div_ceil(10)].to_vec()
}

/// Host load now: loadavg1 from `/proc/loadavg` (Linux) or `sysctl` (macOS).
#[must_use]
pub fn load_snapshot() -> Option<Load> {
    let text = std::fs::read_to_string("/proc/loadavg").ok().or_else(|| {
        let o = std::process::Command::new("sysctl")
            .args(["-n", "vm.loadavg"])
            .output()
            .ok()?;
        Some(String::from_utf8_lossy(&o.stdout).replace(['{', '}'], ""))
    })?;
    let loadavg1 = text.split_whitespace().next()?.parse().ok()?;
    let cpus = std::thread::available_parallelism().ok()?.get() as u32;
    Some(Load {
        loadavg1,
        cpus,
        co_running: None,
        peak_anon_bytes: None,
    })
}

/// What one HTTP exchange produced.
#[derive(Debug, Clone, PartialEq)]
pub struct Reply {
    pub status: u16,
    pub body: String,
    pub wall_ms: f64,
}

/// POST the body; transport failures are `Err`, HTTP errors are a [`Reply`].
///
/// # Errors
/// The serve could not be reached or the body could not be read.
pub fn post(url: &str, body: &Value) -> Result<Reply, String> {
    let agent = ureq::AgentBuilder::new().timeout(TIMEOUT).build();
    let t = Instant::now();
    let (status, resp) = match agent.post(url).send_json(body) {
        Ok(r) => (r.status(), r),
        Err(ureq::Error::Status(code, r)) => (code, r),
        Err(e) => return Err(e.to_string()),
    };
    // `into_string` refuses bodies over 10 MB; the /tokenize reply for a large
    // candidate diff exceeds that. Read bounded at MAX_BODY instead.
    let body = read_bounded(resp.into_reader(), MAX_BODY)?;
    Ok(Reply {
        status,
        body,
        wall_ms: t.elapsed().as_secs_f64() * 1e3,
    })
}

/// Read `r` to a string of at most `cap` bytes.
///
/// # Errors
/// The read failed, or the body ran past `cap`.
pub fn read_bounded(r: impl Read, cap: u64) -> Result<String, String> {
    let mut body = String::new();
    r.take(cap)
        .read_to_string(&mut body)
        .map_err(|e| e.to_string())?;
    Ok(body)
}

/// The reply's text, token counts, server timings — or why none.
#[derive(Debug, Clone, PartialEq)]
pub struct Parsed {
    pub verdict: Verdict,
    pub text: String,
    pub tokens: Option<Tokens>,
    pub server: Option<ServerTimings>,
}

/// Classify a reply. 200 with a message → parsed verdict; a 4xx naming the
/// context → `ContextOverflow`; anything else → `ServeError`.
#[must_use]
pub fn classify(reply: &Reply) -> Parsed {
    let not_run = |why| Parsed {
        verdict: Verdict::NotRun(why),
        text: reply.body.clone(),
        tokens: None,
        server: None,
    };
    if reply.status != 200 {
        let lower = reply.body.to_lowercase();
        let ctx = (400..500).contains(&reply.status)
            && (lower.contains("context") || lower.contains("too long"));
        return not_run(if ctx {
            NotRun::ContextOverflow
        } else {
            NotRun::ServeError
        });
    }
    let Ok(v) = serde_json::from_str::<Value>(&reply.body) else {
        return not_run(NotRun::ServeError);
    };
    let Some(text) = v["choices"][0]["message"]["content"].as_str() else {
        return not_run(NotRun::ServeError);
    };
    let tokens = Some(Tokens {
        prompt: v["usage"]["prompt_tokens"].as_u64().unwrap_or(0),
        completion: v["usage"]["completion_tokens"].as_u64().unwrap_or(0),
    })
    .filter(|t| v["usage"].is_object() && t.prompt > 0);
    let t = &v["timings"];
    let server = t.is_object().then(|| ServerTimings {
        prompt_ms: t["prompt_ms"].as_f64().unwrap_or(f64::NAN),
        predicted_ms: t["predicted_ms"].as_f64().unwrap_or(f64::NAN),
        prompt_per_second: t["prompt_per_second"].as_f64(),
        predicted_per_second: t["predicted_per_second"].as_f64(),
    });
    Parsed {
        verdict: parse_verdict(text),
        text: text.to_string(),
        tokens,
        server,
    }
}

/// Identity of the (cell, arm) being run; copied into every receipt.
#[derive(Debug, Clone)]
pub struct Run {
    pub cell: String,
    pub host: String,
    pub backend: String,
    pub arm: Arm,
    pub apr_tag: String,
    pub apr_sha256: String,
    pub model_id: String,
    pub weights_sha256: String,
    pub prompt: String,
    pub corpus_version: String,
    pub prereg_sha: String,
}

impl Run {
    /// A receipt for `item` from a classified reply. `raw_path` is where the
    /// caller writes `parsed.text`; its sha is recorded here. A row with no
    /// token counts from the serve is `ServeError` — it cannot be admitted.
    #[must_use]
    pub fn receipt(
        &self,
        item: &Item,
        body: &Value,
        p: &Parsed,
        wall_ms: f64,
        raw_path: &str,
        when: &str,
    ) -> Receipt {
        let mut verdict = p.verdict;
        if verdict.executed() && p.tokens.is_none() {
            verdict = Verdict::NotRun(NotRun::ServeError);
        }
        let ran = verdict.executed();
        Receipt {
            schema: SCHEME.into(),
            item_id: item.id.clone(),
            item_sha256: item.diff_sha256.clone(),
            class: item.class,
            stratum: item.stratum,
            split: item.split,
            cell: self.cell.clone(),
            host: self.host.clone(),
            backend: self.backend.clone(),
            arm: self.arm,
            apr_tag: self.apr_tag.clone(),
            apr_sha256: self.apr_sha256.clone(),
            model_id: self.model_id.clone(),
            weights_sha256: self.weights_sha256.clone(),
            prompt_sha256: sha256_hex(self.prompt.as_bytes()),
            request_sha256: sha256_hex(body.to_string().as_bytes()),
            decoding: DECODING,
            corpus_version: self.corpus_version.clone(),
            prereg_sha: self.prereg_sha.clone(),
            started_at: when.to_string(),
            cold: false,
            rerun: false,
            verdict,
            tokens: p.tokens.filter(|_| ran),
            timings: Some(Timings {
                wall_ms,
                server: p.server,
            }),
            output: Some(Output {
                path: raw_path.to_string(),
                sha256: sha256_hex(p.text.as_bytes()),
            }),
            load: load_snapshot(),
        }
    }

    /// A `NotRun` receipt with no request (e.g. `NoDeclaredExecutor`).
    #[must_use]
    pub fn not_run(&self, item: &Item, why: NotRun, when: &str) -> Receipt {
        let p = Parsed {
            verdict: Verdict::NotRun(why),
            text: String::new(),
            tokens: None,
            server: None,
        };
        let body = request_body(&self.model_id, &self.prompt, "");
        let mut r = self.receipt(item, &body, &p, 0.0, "-", when);
        (r.timings, r.output, r.load) = (None, None, None);
        r
    }
}

/// RFC 3339 UTC for a Unix time (civil-from-days, proleptic Gregorian).
#[must_use]
pub fn rfc3339(unix_secs: u64) -> String {
    let (days, rem) = (unix_secs / 86_400, unix_secs % 86_400);
    let z = days as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        rem % 3600 / 60,
        rem % 60
    )
}

/// Now, RFC 3339 UTC.
#[must_use]
pub fn utc_now() -> String {
    let s = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    rfc3339(s)
}

#[cfg(test)]
#[path = "harness_tests.rs"]
mod tests;
