//! TR-01 (#4556): how a trace's timing numbers were obtained.
//!
//! Hoisted from `aprender-serve` (GH-92) so that every tracer in the stack —
//! the serve `TraceData`, renacer's `BrickTracer`, the `apr` bench/qa callers —
//! declares its provenance with one type. A tracer that captured nothing must
//! say `NotInstrumented`; it must never present wall-clock time as a breakdown.
//!
//! Contract: `contracts/brick-tracer-provenance-v1.yaml`.

use serde::{Deserialize, Serialize};

/// Provenance of trace timing data (GH-92: truth-in-reporting).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TraceProvenance {
    /// Real per-operation timing from instrumentation
    Measured,
    /// Only the wall-clock total is real; no per-op breakdown available
    WallClockTotal,
    /// Values are statistical estimates (e.g., from sampling or heuristics)
    #[default]
    Estimated,
    /// The tracer captured no events at all: only the wall-clock duration is
    /// real, and every breakdown component is absent (`null`), not zero
    NotInstrumented,
}

impl TraceProvenance {
    /// The serialized name, for human-readable output.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Measured => "measured",
            Self::WallClockTotal => "wall_clock_total",
            Self::Estimated => "estimated",
            Self::NotInstrumented => "not_instrumented",
        }
    }

    /// Whether per-component numbers under this provenance are real measurements.
    #[must_use]
    pub const fn has_breakdown(self) -> bool {
        matches!(self, Self::Measured)
    }
}

impl std::fmt::Display for TraceProvenance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [TraceProvenance; 4] = [
        TraceProvenance::Measured,
        TraceProvenance::WallClockTotal,
        TraceProvenance::Estimated,
        TraceProvenance::NotInstrumented,
    ];

    /// FALSIFY-BTP-001: `as_str` is exactly the serde name, for every variant.
    #[test]
    fn falsify_btp_001_as_str_matches_serde() {
        for p in ALL {
            let json = serde_json::to_string(&p).expect("serialize");
            assert_eq!(json, format!("\"{}\"", p.as_str()));
            let back: TraceProvenance = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(back, p);
            assert_eq!(p.to_string(), p.as_str());
        }
    }

    /// FALSIFY-BTP-002: only `Measured` claims a breakdown.
    #[test]
    fn falsify_btp_002_only_measured_has_breakdown() {
        let with: Vec<_> = ALL.into_iter().filter(|p| p.has_breakdown()).collect();
        assert_eq!(with, vec![TraceProvenance::Measured]);
    }
}
