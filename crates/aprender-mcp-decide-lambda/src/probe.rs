//! A JSON-RPC probe for a live (or loopback) decide endpoint.
//!
//! It proves four things a deploy must show (D-11), each against a value derived from the
//! LOCAL artifact through the decide-apr-v1 ladder: the endpoint advertises exactly one
//! `classify` tool, that tool's description names the artifact's labels in task order,
//! a call answers with the artifact's labels in that order, and the call answers with
//! the served file's sha256 — the identity pinned in the deploy config. [`run_cold_first`] sends the `tools/call` as the FIRST and only
//! POST, the shape a cold container behind the gateway receives when the client's
//! `initialize` landed elsewhere; the server's `x-decide-load` header and its
//! `decide.load` log line (keyed by the probe id) say whether that request was cold.
//!
//! Every request carries `x-decide-probe-id`. A bearer token, when used, is sent and
//! never printed or stored in a report (T-08-07-04).

use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::Serialize;
use serde_json::{json, Value};

use crate::{LOAD_HEADER, PROBE_ID_HEADER};

/// The protocol version the probe sends (header and `initialize`): one pmcp's
/// stateless server supports.
pub const PROBE_PROTOCOL_VERSION: &str = pmcp::DEFAULT_PROTOCOL_VERSION;

/// The text the identity probe classifies (synthetic; never dataset text).
pub const IDENTITY_PROBE_TEXT: &str = "Identity probe: please route this message.";

/// How long one probe request may take (a cold start inside it included).
pub const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);

/// Why a probe could not complete.
#[derive(Debug)]
pub enum ProbeError {
    /// Transport failure (connect, timeout, body).
    Http(String),
    /// Non-2xx status.
    Status {
        /// The JSON-RPC method.
        method: &'static str,
        /// HTTP status.
        status: u16,
        /// The response body (the server's error; never contains the token).
        body: String,
    },
    /// A JSON-RPC error, a tool error, or a body that is not the expected shape.
    Protocol {
        /// The JSON-RPC method.
        method: &'static str,
        /// What was wrong.
        detail: String,
    },
}

impl std::fmt::Display for ProbeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Http(e) => write!(f, "http: {e}"),
            Self::Status {
                method,
                status,
                body,
            } => write!(f, "{method}: HTTP {status}: {body}"),
            Self::Protocol { method, detail } => write!(f, "{method}: {detail}"),
        }
    }
}

impl std::error::Error for ProbeError {}

/// One timed JSON-RPC call.
#[derive(Debug, Clone, Serialize)]
pub struct CallTiming {
    /// The JSON-RPC method.
    pub method: &'static str,
    /// Wall time of the POST, response body included.
    pub elapsed_ms: u128,
    /// The `x-decide-load` header on the response, if any.
    pub load_header: Option<String>,
}

/// The identity probe's result.
#[derive(Debug, Clone, Serialize)]
pub struct ProbeReport {
    /// The probe id sent on every request.
    pub probe_id: String,
    /// `tools/list` returned exactly one tool, named `classify`.
    pub one_tool_named_classify: bool,
    /// The tool description's `The labels, in this order: [...]` segment is exactly the
    /// expected labels (see [`labels_in_order`]).
    pub description_has_labels_in_order: bool,
    /// The `tools/call` response's `labels` array equals the expected labels: same
    /// length, same order (V12-a).
    pub response_labels_match: bool,
    /// The call's `model.artifact_sha256` equals the expected sha256.
    pub identity_matches: bool,
    /// The identity the endpoint reported.
    pub artifact_sha256: String,
    /// The labels the call response carried.
    pub labels: Vec<String>,
    /// Built-row tokens the call was charged.
    pub tokens_total: usize,
    /// `initialize`, `tools/list`, `tools/call`, in order.
    pub calls: Vec<CallTiming>,
}

impl ProbeReport {
    /// Every identity check passed: one `classify` tool, the description's labels, the
    /// response's labels and the sha256. A report field that is deploy evidence and is
    /// not in this AND would be a check the verdict ignores.
    #[must_use]
    pub fn ok(&self) -> bool {
        self.one_tool_named_classify
            && self.description_has_labels_in_order
            && self.response_labels_match
            && self.identity_matches
    }
}

/// The cold-first sample: one `tools/call` with no `initialize` before it.
#[derive(Debug, Clone, Serialize)]
pub struct ColdSample {
    /// The probe id sent.
    pub probe_id: String,
    /// Wall time of the single POST.
    pub elapsed_ms: u128,
    /// Built-row tokens over the request.
    pub tokens_total: usize,
    /// Texts sent.
    pub texts: usize,
    /// Results with `truncated: true`.
    pub truncated: usize,
    /// The identity the endpoint reported.
    pub artifact_sha256: String,
    /// The `x-decide-load` header (`cold;load_ms=<n>` or `warm`).
    pub load_header: Option<String>,
}

/// A fresh probe id: `probe-<time hex>-<pid hex>`, inside the server's accepted
/// `[A-Za-z0-9-]{1,64}` alphabet.
#[must_use]
pub fn new_probe_id() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    format!("probe-{nanos:x}-{:x}", std::process::id())
}

/// A JSON-RPC client over MCP streamable HTTP.
struct Rpc<'a> {
    client: reqwest::Client,
    url: &'a str,
    bearer: Option<&'a str>,
    probe_id: &'a str,
    next_id: u64,
}

impl<'a> Rpc<'a> {
    fn new(url: &'a str, bearer: Option<&'a str>, probe_id: &'a str) -> Result<Self, ProbeError> {
        let mut builder = reqwest::Client::builder().timeout(REQUEST_TIMEOUT);
        if is_loopback(url) {
            builder = builder.no_proxy();
        }
        let client = builder
            .build()
            .map_err(|e| ProbeError::Http(e.to_string()))?;
        Ok(Self {
            client,
            url,
            bearer,
            probe_id,
            next_id: 1,
        })
    }

    #[allow(clippy::disallowed_methods)] // serde_json::json! expands to .unwrap()
    async fn call(
        &mut self,
        method: &'static str,
        params: Value,
    ) -> Result<(Value, CallTiming), ProbeError> {
        let id = self.next_id;
        self.next_id += 1;
        let body = json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params });
        let mut req = self
            .client
            .post(self.url)
            .header("content-type", "application/json")
            .header("accept", "application/json, text/event-stream")
            .header("mcp-protocol-version", PROBE_PROTOCOL_VERSION)
            .header(PROBE_ID_HEADER, self.probe_id)
            .body(body.to_string());
        if let Some(token) = self.bearer {
            req = req.bearer_auth(token);
        }
        let started = Instant::now();
        let resp = req
            .send()
            .await
            .map_err(|e| ProbeError::Http(without_url(&e)))?;
        let status = resp.status();
        let load_header = resp
            .headers()
            .get(LOAD_HEADER)
            .and_then(|v| v.to_str().ok())
            .map(str::to_string);
        let content_type = resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();
        let text = resp
            .text()
            .await
            .map_err(|e| ProbeError::Http(without_url(&e)))?;
        let elapsed_ms = started.elapsed().as_millis();
        if !status.is_success() {
            return Err(ProbeError::Status {
                method,
                status: status.as_u16(),
                body: text,
            });
        }
        let value = parse_rpc_body(&content_type, &text)
            .map_err(|detail| ProbeError::Protocol { method, detail })?;
        if let Some(err) = value.get("error") {
            return Err(ProbeError::Protocol {
                method,
                detail: format!("JSON-RPC error: {err}"),
            });
        }
        let result = value.get("result").cloned().ok_or(ProbeError::Protocol {
            method,
            detail: "response has no result".to_string(),
        })?;
        Ok((
            result,
            CallTiming {
                method,
                elapsed_ms,
                load_header,
            },
        ))
    }
}

/// Describe a transport error WITHOUT its URL: the URL is the caller's own, but a
/// query-string token must never be echoed into a report or a log.
fn without_url(e: &reqwest::Error) -> String {
    let kind = if e.is_timeout() {
        "timeout"
    } else if e.is_connect() {
        "connect"
    } else if e.is_body() || e.is_decode() {
        "body"
    } else {
        "request"
    };
    match e.status() {
        Some(s) => format!("{kind} error (status {s})"),
        None => format!("{kind} error"),
    }
}

fn is_loopback(url: &str) -> bool {
    let rest = url
        .strip_prefix("http://")
        .or_else(|| url.strip_prefix("https://"))
        .unwrap_or(url);
    rest.starts_with("127.0.0.1") || rest.starts_with("localhost") || rest.starts_with("[::1]")
}

/// Parse a JSON-RPC response body, JSON or a single-event SSE stream.
///
/// # Errors
///
/// A description when neither shape parses.
pub fn parse_rpc_body(content_type: &str, body: &str) -> Result<Value, String> {
    if content_type.starts_with("text/event-stream") {
        let data: String = body
            .lines()
            .filter_map(|l| l.strip_prefix("data:"))
            .map(str::trim_start)
            .collect::<Vec<_>>()
            .join("\n");
        return serde_json::from_str(&data).map_err(|e| format!("SSE data is not JSON: {e}"));
    }
    serde_json::from_str(body).map_err(|e| format!("body is not JSON: {e}"))
}

/// The classify payload out of a `tools/call` result (JSON text content).
///
/// # Errors
///
/// A description when the result is a tool error or not the expected shape.
pub fn classify_payload(result: &Value) -> Result<Value, String> {
    if result.get("isError") == Some(&Value::Bool(true)) {
        let text = result["content"][0]["text"].as_str().unwrap_or("");
        return Err(format!("tool error: {text}"));
    }
    let text = result["content"][0]["text"]
        .as_str()
        .ok_or("content[0].text is not a string")?;
    serde_json::from_str(text).map_err(|e| format!("classify text is not JSON: {e}"))
}

/// The literal that opens the labels segment of the served tool description
/// (`aprender_mcp_decide::tool_description`). Plan 08-28 may reword the description's other
/// sentences but keeps this segment, so the check reads ONLY this.
pub const LABELS_SEGMENT: &str = "The labels, in this order: [";

/// The description's `The labels, in this order: [a, b, c]` segment is EXACTLY `labels`:
/// the list between the brackets, split on `, `, equals the expected list in length and
/// order (IN-03). A label that occurs in the question text or inside another word
/// (`none` in `nonetheless`) cannot satisfy it. A missing or unterminated segment, or no
/// expected labels, is false.
#[must_use]
pub fn labels_in_order(description: &str, labels: &[String]) -> bool {
    let Some(start) = description.find(LABELS_SEGMENT) else {
        return false;
    };
    let rest = &description[start + LABELS_SEGMENT.len()..];
    let Some(end) = rest.find(']') else {
        return false;
    };
    !labels.is_empty()
        && rest[..end]
            .split(", ")
            .eq(labels.iter().map(String::as_str))
}

#[allow(clippy::disallowed_methods)] // serde_json::json! expands to .unwrap()
fn classify_params(texts: &[String]) -> Value {
    json!({ "name": aprender_mcp_decide::TOOL_NAME, "arguments": { "texts": texts } })
}

fn tokens_and_truncated(payload: &Value) -> (usize, usize) {
    let results = payload["results"].as_array().map_or(&[][..], Vec::as_slice);
    let tokens = results
        .iter()
        .filter_map(|r| r["tokens"].as_u64())
        .map(|t| usize::try_from(t).unwrap_or(usize::MAX))
        .fold(0usize, usize::saturating_add);
    let truncated = results
        .iter()
        .filter(|r| r["truncated"].as_bool() == Some(true))
        .count();
    (tokens, truncated)
}

/// `initialize` -> `tools/list` -> `tools/call classify`, checking identity and labels.
///
/// # Errors
///
/// [`ProbeError`] for a transport failure, a non-2xx status or a malformed response.
/// A WRONG identity or label list is not an error: it is a false check in the report,
/// and [`ProbeReport::ok`] is false.
#[allow(clippy::disallowed_methods)] // serde_json::json! expands to .unwrap()
pub async fn run_identity_probe(
    url: &str,
    bearer: Option<&str>,
    expected_sha256: &str,
    expected_labels: &[String],
    probe_id: &str,
) -> Result<ProbeReport, ProbeError> {
    let mut rpc = Rpc::new(url, bearer, probe_id)?;
    let (_, init) = rpc
        .call(
            "initialize",
            json!({
                "protocolVersion": PROBE_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": { "name": "aprender-decide-probe", "version": env!("CARGO_PKG_VERSION") }
            }),
        )
        .await?;
    let (listed, list) = rpc.call("tools/list", json!({})).await?;
    let tools = listed["tools"].as_array().cloned().unwrap_or_default();
    let one_tool_named_classify =
        tools.len() == 1 && tools[0]["name"] == aprender_mcp_decide::TOOL_NAME;
    let description = tools
        .first()
        .and_then(|t| t["description"].as_str())
        .unwrap_or("");
    let description_has_labels_in_order = labels_in_order(description, expected_labels);

    let (result, call) = rpc
        .call(
            "tools/call",
            classify_params(&[IDENTITY_PROBE_TEXT.to_string()]),
        )
        .await?;
    let payload = classify_payload(&result).map_err(|detail| ProbeError::Protocol {
        method: "tools/call",
        detail,
    })?;
    let artifact_sha256 = payload["model"]["artifact_sha256"]
        .as_str()
        .unwrap_or("")
        .to_string();
    let labels: Vec<String> = payload["labels"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|l| l.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    let (tokens_total, _) = tokens_and_truncated(&payload);
    Ok(ProbeReport {
        probe_id: probe_id.to_string(),
        one_tool_named_classify,
        description_has_labels_in_order,
        response_labels_match: labels.as_slice() == expected_labels,
        identity_matches: !artifact_sha256.is_empty() && artifact_sha256 == expected_sha256,
        artifact_sha256,
        labels,
        tokens_total,
        calls: vec![init, list, call],
    })
}

/// Send ONE `tools/call classify` as the first and only POST — no `initialize`, no
/// `tools/list` — and time it.
///
/// # Errors
///
/// [`ProbeError`] for a transport failure, a non-2xx status, a tool error or a
/// malformed response.
pub async fn run_cold_first(
    url: &str,
    bearer: Option<&str>,
    texts: &[String],
    probe_id: &str,
) -> Result<ColdSample, ProbeError> {
    let mut rpc = Rpc::new(url, bearer, probe_id)?;
    let (result, call) = rpc.call("tools/call", classify_params(texts)).await?;
    let payload = classify_payload(&result).map_err(|detail| ProbeError::Protocol {
        method: "tools/call",
        detail,
    })?;
    let (tokens_total, truncated) = tokens_and_truncated(&payload);
    Ok(ColdSample {
        probe_id: probe_id.to_string(),
        elapsed_ms: call.elapsed_ms,
        tokens_total,
        texts: texts.len(),
        truncated,
        artifact_sha256: payload["model"]["artifact_sha256"]
            .as_str()
            .unwrap_or("")
            .to_string(),
        load_header: call.load_header,
    })
}

// ===========================================================================
// The maximal legal request (decide-tool-boundary-v1 `accepted_region_cold`)
// ===========================================================================

/// Which maximal legal request to build. Attention cost grows with the square of row
/// length, so two full rows cost more than eight short ones of the same total; the
/// accepted region is claimed for BOTH shapes and neither is inferred from the other.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum MaximalShape {
    /// The fewest texts that reach the token budget, each `max_text_bytes` long and
    /// truncated to a full row: the largest attention cost and tokenizer input.
    Concentrated,
    /// `max_texts` texts whose built rows total the budget: the largest per-row overhead.
    Distributed,
}

impl MaximalShape {
    /// The shape's CLI / report name.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Concentrated => "concentrated",
            Self::Distributed => "distributed",
        }
    }
}

impl std::str::FromStr for MaximalShape {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "concentrated" => Ok(Self::Concentrated),
            "distributed" => Ok(Self::Distributed),
            other => Err(format!(
                "unknown shape {other:?}: use concentrated or distributed"
            )),
        }
    }
}

/// Why a maximal request could not be built.
#[derive(Debug)]
pub enum MaximalError {
    /// The model refused to tokenize the synthetic text.
    Prepare(aprender_decide::DecideError),
    /// The limits admit no request at all (e.g. `max_texts == 0`).
    EmptyLimits,
    /// The server's own budget function (`aprender_mcp_decide::check_token_budget`) refuses
    /// the closest request the builder found: the count bound is not reachable inside the
    /// budget for this task (its shortest built row times `max_texts` exceeds
    /// `max_total_tokens`). A maximal request that is not legal measures nothing
    /// (decide-tool-boundary-v1 `served_task_min_row_tokens`).
    OverBudget {
        /// The shape that could not be built legally.
        shape: MaximalShape,
        /// The server's refusal of that request, verbatim (R6: one budget function).
        refusal: aprender_mcp_decide::ClassifyFailure,
    },
}

impl std::fmt::Display for MaximalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Prepare(e) => write!(f, "the model refused the synthetic text: {e}"),
            Self::EmptyLimits => write!(f, "the limits admit no request"),
            Self::OverBudget { shape, refusal } => write!(
                f,
                "the {} request is refused by the server's own budget: {refusal}; the count \
                 bound is not reachable inside the budget for this task",
                shape.as_str()
            ),
        }
    }
}

impl std::error::Error for MaximalError {}

/// One synthetic unit: plain ASCII (so any byte cut is a char boundary), no dataset text.
const UNIT: &str = "probe ";

/// A synthetic text of `units` repetitions of [`UNIT`], at most `max_bytes` long.
fn synthetic(units: usize, max_bytes: usize) -> String {
    let mut text = UNIT.repeat(units);
    text.truncate(max_bytes);
    text
}

/// A synthetic text exactly `bytes` long (the largest tokenizer input the byte bound
/// admits).
fn synthetic_exact(bytes: usize) -> String {
    let mut text = UNIT.repeat(bytes / UNIT.len() + 1);
    text.truncate(bytes);
    text
}

fn row_tokens(model: &crate::Model, text: &str) -> Result<(usize, bool), MaximalError> {
    let rows = model
        .prepare(&[text.to_string()])
        .map_err(MaximalError::Prepare)?;
    Ok(rows
        .first()
        .map_or((0, false), |r| (r.tokens(), r.truncated())))
}

/// The largest unit count whose built row is at most `target` tokens, and that row's
/// length. Row length is non-decreasing in the unit count, so this is a binary search.
fn largest_within(
    model: &crate::Model,
    target: usize,
    max_bytes: usize,
) -> Result<(usize, usize), MaximalError> {
    let (mut lo, mut lo_tokens) = (0usize, row_tokens(model, "")?.0);
    let mut hi = max_bytes / UNIT.len();
    if lo_tokens > target {
        return Ok((0, lo_tokens));
    }
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        let (tokens, _) = row_tokens(model, &synthetic(mid, max_bytes))?;
        if tokens <= target {
            lo = mid;
            lo_tokens = tokens;
        } else {
            hi = mid - 1;
        }
    }
    Ok((lo, lo_tokens))
}

/// Build the maximal LEGAL request of `shape` under `limits`, sized with the model's
/// own tokenizer and row builder (`prepare`). Returns the texts and their built-row
/// token total. Synthetic text only.
///
/// CONCENTRATED is `min(ceil(max_total_tokens / max_len), max_texts)` texts: all but
/// the last are `max_text_bytes` long and truncated to a full row; the last is too when
/// the budget is a whole number of rows, and otherwise is sized to the remainder, so the
/// total is the largest value not over the budget. For Laya-en (max_len 512) at the
/// contracted 10 240 MB tier (budget 800, v7.0.0) that is 2 texts, one full 512-token row
/// and one sized to the remaining 288; at the superseded 3 008 MB tier (budget 120) it was
/// ONE text of 120 built tokens. DISTRIBUTED is `max_texts` texts whose built rows total
/// the largest value not over the budget.
///
/// # Errors
///
/// [`MaximalError`] if the model refuses the synthetic text, the limits admit nothing,
/// or `aprender_mcp_decide::check_token_budget` — the function the server runs — refuses
/// the closest request of `shape` ([`MaximalError::OverBudget`]: the server would refuse
/// it, so it is never returned).
pub fn build_maximal_request(
    model: &crate::Model,
    limits: &aprender_mcp_decide::ClassifyLimits,
    shape: MaximalShape,
) -> Result<(Vec<String>, usize), MaximalError> {
    if limits.max_texts == 0 || limits.max_total_tokens == 0 {
        return Err(MaximalError::EmptyLimits);
    }
    let budget = limits.max_total_tokens;
    let max_bytes = limits.max_text_bytes;
    let texts = match shape {
        MaximalShape::Concentrated => {
            let max_len = model.manifest().agent.max_len.max(1);
            let count = budget.div_ceil(max_len).min(limits.max_texts);
            let full = synthetic_exact(max_bytes);
            let (full_tokens, _) = row_tokens(model, &full)?;
            let mut texts = vec![full; count];
            let over = (full_tokens * count).saturating_sub(budget);
            if over > 0 {
                // The budget is not a whole number of rows: the last text fills only
                // the remainder.
                let room = full_tokens.saturating_sub(over);
                let (units, _) = largest_within(model, room, max_bytes)?;
                if let Some(last) = texts.last_mut() {
                    *last = synthetic(units, max_bytes);
                }
            }
            texts
        }
        MaximalShape::Distributed => {
            let count = limits.max_texts;
            let mut units = Vec::with_capacity(count);
            let mut tokens = Vec::with_capacity(count);
            for i in 0..count {
                let share = budget / count + usize::from(i < budget % count);
                let (u, t) = largest_within(model, share, max_bytes)?;
                units.push(u);
                tokens.push(t);
            }
            // Second pass: hand any slack (rows land a token or two under their share)
            // to the texts in order, never exceeding the budget.
            for i in 0..count {
                let others: usize = tokens.iter().sum::<usize>() - tokens[i];
                let room = budget.saturating_sub(others);
                let (u, t) = largest_within(model, room, max_bytes)?;
                if t <= room && t >= tokens[i] {
                    units[i] = u;
                    tokens[i] = t;
                }
            }
            units.iter().map(|&u| synthetic(u, max_bytes)).collect()
        }
    };
    let per_text: Vec<usize> = model
        .prepare(&texts)
        .map_err(MaximalError::Prepare)?
        .iter()
        .map(aprender_decide::PreparedRow::tokens)
        .collect();
    // Fit is decided by the SERVER's budget function, not a second summation here (R6).
    aprender_mcp_decide::check_token_budget(limits, &per_text)
        .map_err(|refusal| MaximalError::OverBudget { shape, refusal })?;
    Ok((texts, per_text.iter().sum()))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use aprender_mcp_decide::{check_token_budget, precheck, ClassifyArgs, ClassifyLimits};

    use super::*;
    use crate::tests::{serve, tiny_model};

    /// Limits the tiny fixture (rows of at most 64 tokens) can reach: two full rows.
    const SHRUNK: ClassifyLimits = ClassifyLimits {
        max_texts: 4,
        max_text_bytes: 2_048,
        max_total_tokens: 128,
        ..ClassifyLimits::CONTRACTED
    };

    fn rows(model: &crate::Model, texts: &[String]) -> Vec<aprender_decide::PreparedRow> {
        model.prepare(texts).expect("prepare")
    }

    fn assert_legal(limits: &ClassifyLimits, texts: &[String], total: usize) {
        let args = ClassifyArgs {
            texts: texts.to_vec(),
        };
        precheck(limits, &args).expect("the maximal request passes precheck");
        let per_text: Vec<usize> = rows(&tiny_model(), texts)
            .iter()
            .map(aprender_decide::PreparedRow::tokens)
            .collect();
        assert_eq!(per_text.iter().sum::<usize>(), total);
        check_token_budget(limits, &per_text).expect("the maximal request fits the budget");
    }

    #[test]
    fn maximal_concentrated_is_full_rows_of_max_byte_texts() {
        let model = tiny_model();
        assert_eq!(model.manifest().agent.max_len, 64);
        let (texts, total) =
            build_maximal_request(&model, &SHRUNK, MaximalShape::Concentrated).expect("build");
        assert_eq!(texts.len(), 2, "ceil(128 / 64) texts");
        assert!(texts.iter().all(|t| t.len() == SHRUNK.max_text_bytes));
        assert!(rows(&model, &texts).iter().all(|r| r.truncated()));
        assert!(total <= SHRUNK.max_total_tokens && total + 8 >= SHRUNK.max_total_tokens);
        assert_legal(&SHRUNK, &texts, total);
    }

    #[test]
    fn maximal_concentrated_sizes_the_last_text_to_an_uneven_budget() {
        let model = tiny_model();
        let limits = ClassifyLimits {
            max_total_tokens: 100,
            ..SHRUNK
        };
        let (texts, total) =
            build_maximal_request(&model, &limits, MaximalShape::Concentrated).expect("build");
        assert_eq!(texts.len(), 2);
        assert_eq!(texts[0].len(), limits.max_text_bytes);
        assert!(texts[1].len() < limits.max_text_bytes);
        assert!(total <= 100 && total + 8 >= 100, "total {total}");
        assert_legal(&limits, &texts, total);
    }

    #[test]
    fn maximal_distributed_fills_max_texts_to_the_budget() {
        let model = tiny_model();
        let (texts, total) =
            build_maximal_request(&model, &SHRUNK, MaximalShape::Distributed).expect("build");
        assert_eq!(texts.len(), SHRUNK.max_texts);
        assert!(total <= SHRUNK.max_total_tokens && total + 8 >= SHRUNK.max_total_tokens);
        assert_legal(&SHRUNK, &texts, total);
    }

    /// A count the budget cannot admit at the task's shortest row is refused by the builder,
    /// never returned as a "maximal" request the server would refuse.
    #[test]
    fn maximal_distributed_refuses_a_count_the_budget_cannot_admit() {
        let model = tiny_model();
        let (shortest, _) = row_tokens(&model, "").expect("the empty text builds");
        assert!(
            shortest > 1,
            "the tiny task's shortest row is {shortest} tokens"
        );
        // Four texts need at least 4 x shortest tokens; one fewer can never be met.
        let limits = ClassifyLimits {
            max_texts: 4,
            max_total_tokens: 4 * shortest - 1,
            ..SHRUNK
        };
        match build_maximal_request(&model, &limits, MaximalShape::Distributed) {
            Err(MaximalError::OverBudget {
                shape,
                refusal:
                    aprender_mcp_decide::ClassifyFailure::TokenBudget {
                        total,
                        limit: budget,
                        ..
                    },
            }) => {
                assert_eq!(shape, MaximalShape::Distributed);
                assert_eq!(budget, limits.max_total_tokens);
                assert!(total > budget, "{total} > {budget}");
            }
            other => panic!("expected OverBudget, got {other:?}"),
        }
        // At exactly 4 x shortest the same count is legal again.
        let limits = ClassifyLimits {
            max_total_tokens: 4 * shortest,
            ..limits
        };
        let (texts, total) =
            build_maximal_request(&model, &limits, MaximalShape::Distributed).expect("legal");
        assert_legal(&limits, &texts, total);
    }

    #[test]
    fn shape_names_round_trip() {
        for shape in [MaximalShape::Concentrated, MaximalShape::Distributed] {
            assert_eq!(shape.as_str().parse::<MaximalShape>(), Ok(shape));
        }
        assert!("both".parse::<MaximalShape>().is_err());
    }

    /// V12-a: the right sha with the wrong labels is not the right deploy. The loopback
    /// serves the tiny fixture; the SAME endpoint passes against the artifact's labels and
    /// fails against a reordering of them, on the response labels AND the description.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn probe_fails_when_response_labels_differ() {
        let model = tiny_model();
        let sha = crate::tests::tiny_golden_sha256();
        let labels = model.task().owned_labels();
        let url = serve(Arc::clone(&model)).await;

        let report = run_identity_probe(&url, None, sha, &labels, &new_probe_id())
            .await
            .expect("identity probe completes");
        assert!(report.response_labels_match, "{report:?}");
        assert!(report.ok(), "{report:?}");

        let mut reordered = labels.clone();
        reordered.swap(0, 1);
        assert_ne!(reordered, labels);
        let report = run_identity_probe(&url, None, sha, &reordered, &new_probe_id())
            .await
            .expect("identity probe completes");
        assert!(report.one_tool_named_classify, "{report:?}");
        assert!(report.identity_matches, "{report:?}");
        assert!(!report.response_labels_match, "{report:?}");
        // IN-03 on the real description: the old substring walk found `billing` in the
        // segment, then `shipping` and `account` in the criteria lines after it.
        assert!(!report.description_has_labels_in_order, "{report:?}");
        assert!(!report.ok(), "{report:?}");
    }

    /// The right labels from the wrong artifact are not the right deploy: the endpoint
    /// answers with a sha256 other than the pin, and ONLY the identity check is false.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn probe_fails_when_identity_differs() {
        let model = tiny_model();
        let golden = crate::tests::tiny_golden_sha256();
        let labels = model.task().owned_labels();
        let url = serve(Arc::clone(&model)).await;
        // A pin one hex digit away from what the endpoint serves.
        let flipped = if golden.starts_with('0') { "1" } else { "0" };
        let pin = format!("{flipped}{}", &golden[1..]);
        assert_ne!(pin, golden);

        let report = run_identity_probe(&url, None, &pin, &labels, &new_probe_id())
            .await
            .expect("identity probe completes");
        assert_eq!(
            report.artifact_sha256, golden,
            "the endpoint reports what it serves"
        );
        assert!(report.one_tool_named_classify, "{report:?}");
        assert!(report.description_has_labels_in_order, "{report:?}");
        assert!(report.response_labels_match, "{report:?}");
        assert!(!report.identity_matches, "{report:?}");
        assert!(!report.ok(), "{report:?}");
    }

    /// `ok()` is the AND of every check the report carries: each one alone, false, fails it.
    #[test]
    fn ok_requires_every_check() {
        let all = ProbeReport {
            probe_id: "p".to_string(),
            one_tool_named_classify: true,
            description_has_labels_in_order: true,
            response_labels_match: true,
            identity_matches: true,
            artifact_sha256: String::new(),
            labels: Vec::new(),
            tokens_total: 0,
            calls: Vec::new(),
        };
        assert!(all.ok());
        let flips: [fn(&mut ProbeReport); 4] = [
            |r| r.one_tool_named_classify = false,
            |r| r.description_has_labels_in_order = false,
            |r| r.response_labels_match = false,
            |r| r.identity_matches = false,
        ];
        for (i, flip) in flips.iter().enumerate() {
            let mut report = all.clone();
            flip(&mut report);
            assert!(!report.ok(), "check {i} false must fail ok()");
        }
    }

    /// IN-03: the labels check reads the exact `The labels, in this order: [...]` segment
    /// and compares it as a list, so a label in the question text, or inside another word,
    /// can no longer satisfy it.
    #[test]
    fn labels_segment_is_parsed_exactly() {
        let owned = |l: &[&str]| l.iter().map(|s| (*s).to_string()).collect::<Vec<_>>();
        let expected = owned(&["none", "against", "favor"]);
        // The question contains `none`; the segment lists the labels in ANOTHER order.
        let question_first = "Classify: \"Is there none of it?\" \
             The labels, in this order: [against, none, favor].\n- against\n- none\n- favor";
        assert!(!labels_in_order(question_first, &expected));
        // `none` occurs only inside `nonetheless`.
        let inside_a_word =
            "Classify: \"Q?\" The labels, in this order: [against, favor, nonetheless].";
        assert!(!labels_in_order(
            inside_a_word,
            &owned(&["against", "favor", "none"])
        ));
        // The exact segment, in order.
        let exact = "Classify: \"Is there none of it?\" \
             The labels, in this order: [none, against, favor].\n- none: x";
        assert!(labels_in_order(exact, &expected));
        // A prefix, a superset, a missing segment, an unterminated one and no labels all fail.
        assert!(!labels_in_order(exact, &owned(&["none", "against"])));
        assert!(!labels_in_order(
            exact,
            &owned(&["none", "against", "favor", "x"])
        ));
        assert!(!labels_in_order("none, against, favor", &expected));
        assert!(!labels_in_order(
            "The labels, in this order: [none, against, favor",
            &expected
        ));
        assert!(!labels_in_order(exact, &[]));

        // The served description of the tiny fixture: its labels pass, a reordering fails.
        let model = tiny_model();
        let description = aprender_mcp_decide::tool_description(&model);
        let labels = model.task().owned_labels();
        assert!(labels_in_order(&description, &labels), "{description}");
        let mut reordered = labels.clone();
        reordered.swap(0, 1);
        assert!(!labels_in_order(&description, &reordered), "{description}");
    }

    /// R6: the builder decides fit with the server's own `check_token_budget`, so its refusal
    /// IS the server's refusal of the same rows, word for word. (No request separates the old
    /// plain sum from the server's saturating fold below `usize::MAX`, so the shared function
    /// is observable through its refusal.)
    #[test]
    fn maximal_request_uses_the_server_budget() {
        let model = tiny_model();
        let (shortest, _) = row_tokens(&model, "").expect("the empty text builds");
        let limits = ClassifyLimits {
            max_texts: 4,
            max_total_tokens: 4 * shortest - 1,
            ..SHRUNK
        };
        let err = build_maximal_request(&model, &limits, MaximalShape::Distributed)
            .expect_err("four texts cannot fit 4 x shortest - 1 tokens");
        assert!(matches!(err, MaximalError::OverBudget { .. }), "{err:?}");
        // Every share is under the shortest row, so each text is the empty one.
        let server = check_token_budget(&limits, &[shortest; 4])
            .expect_err("the server refuses the same rows")
            .to_string();
        let message = err.to_string();
        assert!(message.contains("classify_max_total_tokens"), "{message}");
        assert!(message.contains(&server), "{message}\n  server: {server}");
    }

    #[test]
    fn rpc_body_parses_json_and_sse() {
        let json = parse_rpc_body("application/json", r#"{"id":1}"#).expect("json");
        assert_eq!(json["id"], 1);
        let sse = parse_rpc_body("text/event-stream", "event: message\ndata: {\"id\":2}\n\n")
            .expect("sse");
        assert_eq!(sse["id"], 2);
        assert!(parse_rpc_body("application/json", "not json").is_err());
    }

    /// Both maximal shapes, built under the SERVED (contracted) limits, are served as the
    /// first and only POST to a fresh stateless loopback server — the request plan 08-11
    /// times on cold Lambda instances.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn loopback_maximal_requests_are_served_cold_first() {
        let model = tiny_model();
        for shape in [MaximalShape::Concentrated, MaximalShape::Distributed] {
            let (texts, total) =
                build_maximal_request(&model, &ClassifyLimits::CONTRACTED, shape).expect("build");
            let url = serve(Arc::clone(&model)).await;
            let sample = run_cold_first(&url, None, &texts, &new_probe_id())
                .await
                .unwrap_or_else(|e| panic!("{} request refused: {e}", shape.as_str()));
            assert_eq!(sample.tokens_total, total, "{}", shape.as_str());
            assert_eq!(sample.texts, texts.len());
            assert_eq!(sample.artifact_sha256, crate::tests::tiny_golden_sha256());
        }
    }
}
