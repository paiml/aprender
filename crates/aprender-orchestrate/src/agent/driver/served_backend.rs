//! The device a served completion actually ran on (#3719).
//!
//! `apr code` tells its `apr serve` child `--gpu` or `--no-gpu`, but telling is
//! not running: the child can fall back to the CPU (the F2 divergence guard, a
//! device checkpoint that would not restore) and still answer 200. Each
//! `/v1/chat/completions` body says what that completion ran on in `used_gpu`.
//! [`BackendTally`] counts those answers over a run, so the `apr code` JSON
//! document can say what the run ran on, in the `backend {requested, ran,
//! fell_back}` shape `apr chat --json` already prints.

use std::sync::atomic::{AtomicU32, Ordering};

/// A run's device, as the server reported it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackendReport {
    /// The device the driver asked for: `"gpu"` or `"cpu"`.
    pub requested: &'static str,
    /// What the completions ran on: `"gpu"` when every one reported the GPU,
    /// `"cpu"` when any reported the CPU, `None` when the server did not say
    /// for every completion, or none ran. `None` is not measured, never a pass.
    pub ran: Option<&'static str>,
    /// The GPU was asked for and a completion ran on the CPU. `None` when `ran` is.
    pub fell_back: Option<bool>,
}

impl BackendReport {
    /// The document's `backend` object; an unmeasured field is `null`.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "requested": self.requested,
            "ran": self.ran,
            "fell_back": self.fell_back,
        })
    }
}

/// `used_gpu` from one `/v1/chat/completions` body. `None` when the server left
/// it out, or sent something that is not a boolean.
pub fn served_used_gpu(body: &serde_json::Value) -> Option<bool> {
    body.get("used_gpu").and_then(serde_json::Value::as_bool)
}

/// The per-completion `used_gpu` answers of one run.
#[derive(Debug, Default)]
pub struct BackendTally {
    gpu: AtomicU32,
    cpu: AtomicU32,
    unreported: AtomicU32,
}

impl BackendTally {
    /// Count one completion's answer.
    pub fn record(&self, used_gpu: Option<bool>) {
        let slot = match used_gpu {
            Some(true) => &self.gpu,
            Some(false) => &self.cpu,
            None => &self.unreported,
        };
        slot.fetch_add(1, Ordering::Relaxed);
    }

    /// The run's device, given whether the GPU was asked for.
    pub fn report(&self, requested_gpu: bool) -> BackendReport {
        let gpu = self.gpu.load(Ordering::Relaxed);
        let cpu = self.cpu.load(Ordering::Relaxed);
        let unreported = self.unreported.load(Ordering::Relaxed);
        // One completion on the CPU is a fact whatever the others said. A GPU
        // verdict needs every completion to have said so.
        let ran = if cpu > 0 {
            Some("cpu")
        } else if gpu > 0 && unreported == 0 {
            Some("gpu")
        } else {
            None
        };
        BackendReport {
            requested: if requested_gpu { "gpu" } else { "cpu" },
            ran,
            fell_back: ran.map(|r| requested_gpu && r == "cpu"),
        }
    }
}

#[cfg(test)]
#[path = "served_backend_tests.rs"]
mod tests;
