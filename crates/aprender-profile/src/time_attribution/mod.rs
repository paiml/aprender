// Time-Weighted Attribution for Syscall Analysis
// (Section 6.2 of single-shot-compile-tooling-spec.md)
//
// Objective: Attribute wall-clock time to syscall categories, not just count.
//
// Key Insight: A syscall appearing 10 times vs 100 times is meaningless without
// time context. A single blocking read() might dominate 1000 fast mmap() calls.
//
// Toyota Way Principle: Genchi Genbutsu (Go and See) - Measure actual wall-clock
// time, don't assume based on counts.

mod attribution;
mod hotspot;

pub use attribution::{calculate_time_attribution, TimeAttribution};
pub use hotspot::{identify_hotspots, Hotspot};

#[cfg(test)]
mod tests;
