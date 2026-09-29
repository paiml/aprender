//! TRACE-001 TR-07 / APR-OBS OBS-16: the external serve witness (#4562).
//!
//! `apr serve` measures its own TTFT and wall time (SRV-TIM-001). A server
//! that under-reports itself would pass every gate that reads those numbers,
//! because the only witness is the defendant. This module is the second
//! witness: it rebuilds each connection's timings from the kernel's view of
//! the socket, as renacer recorded it, and compares.
//!
//! Input: `renacer -f -T -e trace=accept4,recvfrom,read,sendto,write,writev
//! --format json -- <apr> serve …`. `-T` is required, because only a timed
//! trace carries `ts_us`. An untimed trace is NOT MEASURED, never a pass.
//!
//! Per connection k (TRACE-001 §2.3):
//! - `wall_ext = t(last write on k) − t(anchor)`
//! - `ttft_ext = t(first write on k after its request bytes were read) − t(anchor)`
//!
//! The anchor is `accept4→k` for a connection's first request. A later request
//! on the same connection (keep-alive) starts at its first read. TTFT is null,
//! with a reason, when the response is not a stream (one write carries the
//! whole body, so the first write is not the first token) or when no write
//! followed the read.
//!
//! Join: the witness runs at c=1, so the n-th witnessed request is the n-th
//! server `[request]` line. A count mismatch is NOT MEASURED.

use serde::{Deserialize, Serialize};

use crate::json_output::{JsonOutput, JsonSyscall};

/// Row schema name, as the TRACE-001 §2.3 ledger rows spell it.
pub const WITNESS_SCHEMA: &str = "apr-serve-witness-v1";

/// Relative tolerance of the witness comparison (5%).
pub const TOLERANCE_REL: f64 = 0.05;

/// Absolute floor of the witness comparison (50 ms).
pub const TOLERANCE_ABS_MS: f64 = 50.0;

/// Why a witness cannot give a verdict. This is never a pass (L25).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NotMeasured {
    /// A socket syscall has no `ts_us`, so the trace was taken without `-T`.
    Untimed { syscall: String },
    /// The trace saw no accepted connection.
    NoConnections,
    /// Witnessed requests and server lines differ in number, so the join is
    /// ambiguous.
    JoinCount { witnessed: usize, server: usize },
}

impl std::fmt::Display for NotMeasured {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Untimed { syscall } => {
                write!(f, "`{syscall}` has no ts_us; trace with -T")
            }
            Self::NoConnections => write!(f, "no accept4 in the trace"),
            Self::JoinCount { witnessed, server } => {
                write!(f, "{witnessed} witnessed request(s) vs {server} server [request] line(s)")
            }
        }
    }
}

/// One socket syscall the witness uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SocketEvent {
    pub kind: EventKind,
    /// For `Accept`, the new fd (the return value). Otherwise the fd argument.
    pub fd: i64,
    /// Bytes moved (return value). Only positive values count.
    pub bytes: i64,
    pub ts_us: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventKind {
    Accept,
    Read,
    Write,
}

fn event_kind(name: &str) -> Option<EventKind> {
    match name {
        "accept4" | "accept" => Some(EventKind::Accept),
        "read" | "recvfrom" => Some(EventKind::Read),
        "write" | "sendto" | "writev" => Some(EventKind::Write),
        _ => None,
    }
}

/// renacer prints a syscall argument as `{:#x}`. This also accepts decimal.
fn parse_fd_arg(arg: &str) -> Option<i64> {
    match arg.strip_prefix("0x") {
        Some(hex) => i64::from_str_radix(hex, 16).ok(),
        None => arg.parse().ok(),
    }
}

fn socket_event(sc: &JsonSyscall) -> Result<Option<SocketEvent>, NotMeasured> {
    let Some(kind) = event_kind(&sc.name) else { return Ok(None) };
    let ts_us = sc.ts_us.ok_or_else(|| NotMeasured::Untimed { syscall: sc.name.clone() })?;
    let fd = match kind {
        EventKind::Accept => sc.result,
        EventKind::Read | EventKind::Write => match sc.args.first().and_then(|a| parse_fd_arg(a)) {
            Some(fd) => fd,
            None => return Ok(None),
        },
    };
    Ok(Some(SocketEvent { kind, fd, bytes: sc.result, ts_us }))
}

/// The witness's events in time order, or NOT MEASURED for an untimed trace.
pub fn socket_events(trace: &JsonOutput) -> Result<Vec<SocketEvent>, NotMeasured> {
    let mut events = Vec::new();
    for sc in &trace.syscalls {
        if let Some(ev) = socket_event(sc)? {
            events.push(ev);
        }
    }
    // Rows are in exit order across threads; ts_us is exit time, so a
    // stable sort keeps equal stamps in their recorded order.
    events.sort_by_key(|e| e.ts_us);
    Ok(events)
}

/// How a witnessed request's clock started.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Anchor {
    /// `accept4` returned the connection's fd.
    Accept4,
    /// A later request on a kept-alive connection: its first read.
    Read,
}

/// One request as the kernel saw it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WitnessedRequest {
    pub fd: i64,
    pub anchor: Anchor,
    pub anchor_us: u64,
    pub request_bytes: u64,
    pub response_bytes: u64,
    /// First write after request bytes were read.
    pub first_write_us: Option<u64>,
    pub last_write_us: Option<u64>,
}

impl WitnessedRequest {
    fn new(fd: i64, anchor: Anchor, anchor_us: u64) -> Self {
        Self {
            fd,
            anchor,
            anchor_us,
            request_bytes: 0,
            response_bytes: 0,
            first_write_us: None,
            last_write_us: None,
        }
    }

    fn on_read(&mut self, bytes: u64) {
        self.request_bytes += bytes;
    }

    fn on_write(&mut self, bytes: u64, ts_us: u64) {
        if self.request_bytes == 0 {
            // Nothing was read yet, so this write answers no request.
            return;
        }
        self.response_bytes += bytes;
        self.first_write_us.get_or_insert(ts_us);
        self.last_write_us = Some(ts_us);
    }

    /// `ttft_ext` in ms: first write after the read, minus the anchor.
    pub fn ttft_ext_ms(&self) -> Option<f64> {
        self.first_write_us.map(|t| us_to_ms(t.saturating_sub(self.anchor_us)))
    }

    /// `wall_ext` in ms: last write, minus the anchor.
    pub fn wall_ext_ms(&self) -> Option<f64> {
        self.last_write_us.map(|t| us_to_ms(t.saturating_sub(self.anchor_us)))
    }
}

fn us_to_ms(us: u64) -> f64 {
    // µs counts in a trace stay far below 2^53; the cast is exact there.
    #[allow(clippy::cast_precision_loss)]
    let ms = us as f64 / 1000.0;
    ms
}

fn bytes_of(ev: &SocketEvent) -> u64 {
    u64::try_from(ev.bytes).unwrap_or(0)
}

/// Group events into requests, in the order their clocks started.
pub fn witnessed_requests(events: &[SocketEvent]) -> Vec<WitnessedRequest> {
    let mut open: std::collections::HashMap<i64, WitnessedRequest> =
        std::collections::HashMap::new();
    let mut done = Vec::new();
    for ev in events {
        if ev.kind == EventKind::Accept {
            if ev.fd >= 0 {
                // A reused fd number closes the connection that held it.
                done.extend(
                    open.insert(ev.fd, WitnessedRequest::new(ev.fd, Anchor::Accept4, ev.ts_us)),
                );
            }
            continue;
        }
        let bytes = bytes_of(ev);
        let Some(req) = open.get_mut(&ev.fd) else { continue };
        if bytes == 0 {
            continue;
        }
        if ev.kind == EventKind::Read && req.last_write_us.is_some() {
            // Keep-alive: the next request starts at its first read.
            let mut next = WitnessedRequest::new(ev.fd, Anchor::Read, ev.ts_us);
            next.on_read(bytes);
            done.extend(open.insert(ev.fd, next));
            continue;
        }
        match ev.kind {
            EventKind::Read => req.on_read(bytes),
            EventKind::Write => req.on_write(bytes, ev.ts_us),
            EventKind::Accept => {}
        }
    }
    done.extend(open.into_values());
    // Drop accepted connections that carried no request (health probes that
    // closed, the listener's own wakeups).
    done.retain(|r| r.request_bytes > 0);
    done.sort_by_key(|r| (r.anchor_us, r.fd));
    done
}

/// The subset of a server `[request]` line the witness compares against.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ServerRequest {
    pub request_id: String,
    #[serde(default)]
    pub client_request_id: Option<String>,
    pub stream: bool,
    #[serde(default)]
    pub ttft_ms: Option<f64>,
    pub total_ms: f64,
}

/// Every `[request] {json}` line in a server log, in order. A line that
/// fails to parse is skipped; the join count then shows it.
pub fn server_requests(log: &str) -> Vec<ServerRequest> {
    log.lines()
        .filter_map(|l| l.split_once("[request] ").map(|(_, json)| json))
        .filter_map(|json| serde_json::from_str(json).ok())
        .collect()
}

/// A metric's verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    Green,
    Red,
    /// No comparison: one side is null. Never a pass.
    NotMeasured,
}

/// The tolerance for an external value: max(5%, 50 ms).
pub fn tolerance_ms(external_ms: f64) -> f64 {
    (external_ms.abs() * TOLERANCE_REL).max(TOLERANCE_ABS_MS)
}

/// |server − external| ≤ max(5%, 50 ms).
pub fn compare(server_ms: Option<f64>, external_ms: Option<f64>) -> Verdict {
    match (server_ms, external_ms) {
        (Some(s), Some(e)) if s.is_finite() && e.is_finite() => {
            if (s - e).abs() <= tolerance_ms(e) {
                Verdict::Green
            } else {
                Verdict::Red
            }
        }
        _ => Verdict::NotMeasured,
    }
}

/// One `apr-serve-witness-v1` row.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WitnessRow {
    pub schema: String,
    /// 0-based position in the join (the c=1 request order).
    pub seq: usize,
    pub request_id: String,
    pub client_request_id: Option<String>,
    pub fd: i64,
    pub anchor: Anchor,
    pub stream: bool,
    pub request_bytes: u64,
    pub response_bytes: u64,
    pub ttft_ext_ms: Option<f64>,
    pub ttft_reason: Option<String>,
    pub ttft_server_ms: Option<f64>,
    pub ttft_verdict: Verdict,
    pub wall_ext_ms: Option<f64>,
    pub wall_server_ms: f64,
    pub wall_verdict: Verdict,
    /// Always true: traced runs are excluded from every perf series.
    pub traced: bool,
}

impl WitnessRow {
    /// RED if either metric is RED. Otherwise GREEN if either is GREEN.
    /// Otherwise (nothing compared) NOT MEASURED.
    pub fn verdict(&self) -> Verdict {
        let both = [self.ttft_verdict, self.wall_verdict];
        if both.contains(&Verdict::Red) {
            Verdict::Red
        } else if both.contains(&Verdict::Green) {
            Verdict::Green
        } else {
            Verdict::NotMeasured
        }
    }
}

fn ttft_external(w: &WitnessedRequest, stream: bool) -> (Option<f64>, Option<String>) {
    if !stream {
        return (None, Some("non-streaming".to_string()));
    }
    match w.ttft_ext_ms() {
        Some(ms) => (Some(ms), None),
        None => (None, Some("no-write-after-read".to_string())),
    }
}

fn row(seq: usize, w: &WitnessedRequest, s: &ServerRequest) -> WitnessRow {
    let (ttft_ext_ms, ttft_reason) = ttft_external(w, s.stream);
    let ttft_server_ms = if s.stream { s.ttft_ms } else { None };
    let wall_ext_ms = w.wall_ext_ms();
    WitnessRow {
        schema: WITNESS_SCHEMA.to_string(),
        seq,
        request_id: s.request_id.clone(),
        client_request_id: s.client_request_id.clone(),
        fd: w.fd,
        anchor: w.anchor,
        stream: s.stream,
        request_bytes: w.request_bytes,
        response_bytes: w.response_bytes,
        ttft_ext_ms,
        ttft_reason,
        ttft_server_ms,
        ttft_verdict: compare(ttft_server_ms, ttft_ext_ms),
        wall_ext_ms,
        wall_server_ms: s.total_ms,
        wall_verdict: compare(Some(s.total_ms), wall_ext_ms),
        traced: true,
    }
}

/// Build the witness rows for one traced c=1 run.
pub fn witness(trace: &JsonOutput, server_log: &str) -> Result<Vec<WitnessRow>, NotMeasured> {
    let requests = witnessed_requests(&socket_events(trace)?);
    if requests.is_empty() {
        return Err(NotMeasured::NoConnections);
    }
    let server = server_requests(server_log);
    if requests.len() != server.len() {
        return Err(NotMeasured::JoinCount { witnessed: requests.len(), server: server.len() });
    }
    Ok(requests.iter().zip(&server).enumerate().map(|(i, (w, s))| row(i, w, s)).collect())
}

/// The run's verdict: RED if any row is RED, GREEN if at least one row
/// compared and none is RED, NOT MEASURED if nothing compared.
pub fn run_verdict(rows: &[WitnessRow]) -> Verdict {
    let verdicts: Vec<Verdict> = rows.iter().map(WitnessRow::verdict).collect();
    if verdicts.contains(&Verdict::Red) {
        Verdict::Red
    } else if verdicts.contains(&Verdict::Green) {
        Verdict::Green
    } else {
        Verdict::NotMeasured
    }
}

#[cfg(test)]
#[path = "serve_witness_tests.rs"]
mod tests;
