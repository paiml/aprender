//! APR-OBS-001 OBS-09: `X-Trace-Level: step|layer` on a Qwen3.5 serve request
//! runs `inference_trace`, and the reply says [`TraceProvenance::Measured`] only
//! when the tracer actually ran (contract `apr-trace-v1`).
//!
//! Before this, every serve backend answered a trace header from
//! [`build_trace_data`]: the request's wall clock, one row, `WallClockTotal`.
//! That stays the answer whenever there is nothing measured to report — a
//! header this path does not trace, a CPU turn asked for layers (the CPU
//! forward times none), or a measurement whose parts add up to MORE than the
//! request's own wall clock, which cannot be a measurement of that request.

use super::{build_trace_data, TraceData, TraceOperation, TraceProvenance};
use crate::inference_trace::{InferenceTracer, TraceConfig, TraceEvent, TraceStep};
use crate::session::LayerTiming;

/// The tracer a request's `X-Trace-Level` asks for, or `None` for a level this
/// path does not trace. Both levels record every forward and every emitted
/// token; `layer` also times each layer of each decode forward.
pub(crate) fn tracer_for(level: Option<&str>) -> Option<InferenceTracer> {
    match level {
        Some("step" | "layer") => {
            let mut config = TraceConfig::enabled();
            config.steps = [TraceStep::TransformerBlock, TraceStep::Decode]
                .into_iter()
                .collect();
            Some(InferenceTracer::new(config))
        },
        _ => None,
    }
}

/// What one traced request measured, and what it cost end to end.
///
/// TR-09 (#4564): `apr run` builds one of these too, so a request traced
/// through `apr run` and through serve goes through the same [`apr_trace`].
pub struct ServeTrace<'a> {
    /// The requested trace level (`brick`, `step`, `layer`).
    pub level: Option<&'a str>,
    /// The tracer's events; empty when it did not run.
    pub events: &'a [TraceEvent],
    /// The backend's per-layer decode tally, when it timed layers.
    pub layers: Option<&'a [LayerTiming]>,
    /// The request's wall clock, microseconds.
    pub wall_us: u64,
    /// Tokens fed to the model.
    pub prompt_tokens: usize,
    /// Tokens generated.
    pub completion_tokens: usize,
    /// Transformer layers in the model.
    pub num_layers: usize,
}

/// The `(brick, step, layer)` traces for the reply: measured where the tracer
/// ran, the wall-clock-only [`build_trace_data`] everywhere else.
pub(crate) fn traces_for(
    t: &ServeTrace<'_>,
) -> (Option<TraceData>, Option<TraceData>, Option<TraceData>) {
    let measured = match t.level {
        Some("step") => step_trace(t.events, t.wall_us).map(|s| (None, Some(s), None)),
        Some("layer") => layer_trace(t.layers, t.wall_us).map(|l| (None, None, Some(l))),
        _ => None,
    };
    measured.unwrap_or_else(|| {
        build_trace_data(
            t.level,
            t.wall_us,
            t.prompt_tokens,
            t.completion_tokens,
            t.num_layers,
        )
    })
}

/// TR-09 (#4564): the ONE `apr-trace-v1` document for `t.level` — what serve
/// puts in its reply and what `apr run --trace-level` renders. `None` for a
/// level that has no trace.
#[must_use]
pub fn apr_trace(t: &ServeTrace<'_>) -> Option<TraceData> {
    let (brick, step, layer) = traces_for(t);
    brick.or(step).or(layer)
}

/// TR-09 (#4564): chrome://tracing is a FORMAT of an `apr-trace-v1` document,
/// never a second measurement. One `request` span holds the whole wall clock;
/// the breakdown rows are laid end to end under it, each as long as its
/// `time_us`. Because apr-trace-v1 caps the rows at the wall clock
/// (C-TRACE-002), they end inside the span; durations are `u64`, so none is
/// negative or NaN (CRUX F-07). Nothing here is invented: an unmeasured trace
/// renders as its one wall-clock row and says `wall_clock_total`.
#[must_use]
pub fn chrome_trace(t: &TraceData) -> serde_json::Value {
    let mut events = vec![serde_json::json!({
        "name": "request",
        "cat": "request",
        "ph": "X",
        "ts": 0,
        "dur": t.total_time_us,
        "pid": 1,
        "tid": 1,
    })];
    let mut ts: u64 = 0;
    for op in &t.breakdown {
        events.push(serde_json::json!({
            "name": op.name,
            "cat": t.level,
            "ph": "X",
            "ts": ts,
            "dur": op.time_us,
            "pid": 1,
            "tid": 2,
            "args": {"details": op.details},
        }));
        ts = ts.saturating_add(op.time_us);
    }
    serde_json::json!({
        "traceEvents": events,
        "displayTimeUnit": "ms",
        "metadata": {
            "schema": "apr-trace-v1",
            "level": t.level,
            "operations": t.operations,
            "provenance": t.provenance,
        },
    })
}

/// Prompt forward, decode forwards, and token emission, summed from the
/// tracer's events. `None` when there are none, or they exceed `wall_us`.
fn step_trace(events: &[TraceEvent], wall_us: u64) -> Option<TraceData> {
    if events.is_empty() {
        return None;
    }
    let tally = |keep: &dyn Fn(&TraceEvent) -> bool| {
        events
            .iter()
            .filter(|e| keep(e))
            .fold((0u64, 0usize), |(us, n), e| {
                (us.saturating_add(e.duration_us), n + 1)
            })
    };
    let forward = |e: &TraceEvent| e.step == TraceStep::TransformerBlock;
    let (prefill_us, prefills) = tally(&|e| forward(e) && e.iteration == 0);
    let (decode_us, decodes) = tally(&|e| forward(e) && e.iteration > 0);
    let (emit_us, emits) = tally(&|e| e.step == TraceStep::Decode);
    let total = prefill_us.saturating_add(decode_us).saturating_add(emit_us);
    if total > wall_us {
        return None;
    }
    let row = |name: &str, time_us: u64, details: String| TraceOperation {
        name: name.to_string(),
        time_us,
        details: Some(details),
    };
    Some(TraceData {
        level: "step".to_string(),
        operations: events.len(),
        total_time_us: wall_us,
        breakdown: vec![
            row(
                "prefill",
                prefill_us,
                format!("{prefills} forward (prompt, first token chosen)"),
            ),
            row(
                "decode",
                decode_us,
                format!("{decodes} forwards (one per later token)"),
            ),
            row(
                "emit",
                emit_us,
                format!("{emits} tokens handed to the response"),
            ),
        ],
        provenance: TraceProvenance::Measured,
    })
}

/// One row per layer, its decode time summed over the turn. `None` when no
/// layer was timed, or the layers sum past `wall_us`.
fn layer_trace(layers: Option<&[LayerTiming]>, wall_us: u64) -> Option<TraceData> {
    let layers = layers.filter(|l| l.iter().any(|x| x.calls > 0))?;
    let total = layers
        .iter()
        .fold(0u64, |acc, l| acc.saturating_add(l.total_us));
    if total > wall_us {
        return None;
    }
    Some(TraceData {
        level: "layer".to_string(),
        operations: layers.len(),
        total_time_us: wall_us,
        breakdown: layers
            .iter()
            .enumerate()
            .map(|(i, l)| TraceOperation {
                name: format!("layer[{i}] {}", l.kind),
                time_us: l.total_us,
                details: Some(format!("{} decode forwards", l.calls)),
            })
            .collect(),
        provenance: TraceProvenance::Measured,
    })
}

#[cfg(test)]
#[path = "serve_trace_tests.rs"]
mod tests;
