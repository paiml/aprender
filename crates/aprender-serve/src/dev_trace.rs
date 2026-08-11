//! Gating for developer-only diagnostic traces.
//!
//! A shipped CLI addresses the user in prose, not in ticket numbers. Output that
//! only means something to whoever is holding the ticket — raw tensor dumps,
//! pointer values, per-layer kernel decisions — belongs behind this gate; output
//! the user needs in order to understand what the tool did stays unconditional
//! and is written in English.
//!
//! Set `APR_DEV_TRACE=1` to turn the developer traces on.

use std::sync::OnceLock;

/// Environment variable that enables developer-only diagnostic traces.
pub const DEV_TRACE_ENV: &str = "APR_DEV_TRACE";

/// Pure decision: does `raw` (the value of [`DEV_TRACE_ENV`]) mean "on"?
///
/// Unset, empty, `0`, `false`, `off` and `no` all mean off. Anything else means
/// on, so `APR_DEV_TRACE=1` and `APR_DEV_TRACE=yes` both work.
#[must_use]
pub fn dev_trace_enabled_from(raw: Option<&str>) -> bool {
    match raw {
        None => false,
        Some(value) => {
            let value = value.trim();
            !(value.is_empty()
                || value == "0"
                || value.eq_ignore_ascii_case("false")
                || value.eq_ignore_ascii_case("off")
                || value.eq_ignore_ascii_case("no"))
        },
    }
}

/// Are developer-only diagnostic traces enabled for this process?
///
/// Read once and cached: this is consulted from the decode hot path.
#[must_use]
pub fn dev_trace_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| dev_trace_enabled_from(std::env::var(DEV_TRACE_ENV).ok().as_deref()))
}

/// Should the CUDA attention debug trace run for this incremental-attention call?
///
/// The trace synchronizes the compute stream and copies Q, K, V and both KV
/// caches back to the host, so it is not merely noisy — it is a real cost in the
/// decode path. `trace_enabled` is therefore the FIRST term: with traces off
/// (the default) no synchronize and no device-to-host copy happens at all.
#[must_use]
pub fn should_trace_attention(
    trace_enabled: bool,
    skip_debug: bool,
    layer_idx: usize,
    new_len: usize,
) -> bool {
    trace_enabled && !skip_debug && layer_idx == 0 && new_len <= 3
}

#[cfg(test)]
mod tests {
    use super::{dev_trace_enabled_from, should_trace_attention};

    /// The defect: `apr run` on a GGUF model printed `[PAR-058-ATTN] K cache
    /// head0 pos0 first 16: [...]` — raw tensor floats under a ticket number —
    /// because the trace had no gate at all. With no `APR_DEV_TRACE` in the
    /// environment the trace must not run, for the exact call the defect fired
    /// on: layer 0, first decoded token, not capturing a CUDA graph.
    #[test]
    fn attention_trace_is_off_when_developer_traces_are_not_requested() {
        let trace_enabled = dev_trace_enabled_from(None);
        assert!(
            !should_trace_attention(trace_enabled, false, 0, 1),
            "layer 0 / token 1 must not emit an attention trace without APR_DEV_TRACE"
        );
        for new_len in 1..=3 {
            assert!(
                !should_trace_attention(trace_enabled, false, 0, new_len),
                "no attention trace at new_len={new_len} without APR_DEV_TRACE"
            );
        }
    }

    /// The gate must not have removed the diagnostic — a developer who asks for
    /// it still gets it on exactly the calls it used to fire on.
    #[test]
    fn attention_trace_still_fires_when_developer_traces_are_requested() {
        let trace_enabled = dev_trace_enabled_from(Some("1"));
        assert!(
            trace_enabled,
            "APR_DEV_TRACE=1 must enable developer traces"
        );
        assert!(should_trace_attention(trace_enabled, false, 0, 1));
        assert!(should_trace_attention(trace_enabled, false, 0, 3));
        // Pre-existing narrowing, preserved: layer 0 only, first 3 positions
        // only, and never while a CUDA graph capture is in flight.
        assert!(!should_trace_attention(trace_enabled, false, 1, 1));
        assert!(!should_trace_attention(trace_enabled, false, 0, 4));
        assert!(!should_trace_attention(trace_enabled, true, 0, 1));
    }

    #[test]
    fn off_values_are_treated_as_off() {
        for raw in [
            None,
            Some(""),
            Some("  "),
            Some("0"),
            Some("false"),
            Some("FALSE"),
            Some("off"),
            Some("no"),
        ] {
            assert!(
                !dev_trace_enabled_from(raw),
                "{raw:?} must not enable developer traces"
            );
        }
    }

    #[test]
    fn on_values_are_treated_as_on() {
        for raw in [Some("1"), Some("yes"), Some("true"), Some("trace")] {
            assert!(
                dev_trace_enabled_from(raw),
                "{raw:?} must enable developer traces"
            );
        }
    }
}
