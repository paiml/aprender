//! `--timings`: per-phase wall-clock timing of the ttop frame and collector (#4511 item 6).
//!
//! OFF by default and zero-cost when off: every call site holds an `Option<&mut Timings>`,
//! and [`time`] with `None` runs the closure and nothing else — no `Instant::now`, no span.
//! ON, each phase is timed, entered as a `tracing` span named `ttop.phase` (so any
//! subscriber — renacer's OTLP exporter, `tracing-subscriber` fmt — sees the same
//! boundaries), and summarised on exit as one JSON line per phase with p50/p99.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

/// Frame phases, in loop order. `layout_render` is `ui::draw` into the cell buffer,
/// `diff` is the renderer turning cells into escape bytes, `write` is the tty write.
pub const FRAME_PHASES: &[&str] = &["input", "apply", "layout_render", "diff", "write"];

/// Samples kept per phase. A long session keeps the most recent `CAP`, so memory is
/// bounded whatever the run length (the soak gate would catch it otherwise).
pub const CAP: usize = 4096;

/// Per-phase duration samples in microseconds.
#[derive(Debug, Default)]
pub struct Timings {
    phases: BTreeMap<&'static str, Ring>,
}

#[derive(Debug, Default)]
struct Ring {
    samples: Vec<u32>,
    next: usize,
    seen: u64,
}

impl Ring {
    fn push(&mut self, us: u32) {
        if self.samples.len() < CAP {
            self.samples.push(us);
        } else {
            self.samples[self.next] = us;
        }
        self.next = (self.next + 1) % CAP;
        self.seen += 1;
    }
}

/// One phase's summary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhaseStat {
    pub phase: &'static str,
    /// samples seen over the whole run
    pub n: u64,
    pub p50_us: u32,
    pub p99_us: u32,
    pub max_us: u32,
}

impl Timings {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one sample for `phase`.
    pub fn record(&mut self, phase: &'static str, d: Duration) {
        let us = u32::try_from(d.as_micros()).unwrap_or(u32::MAX);
        self.phases.entry(phase).or_default().push(us);
    }

    /// Record samples measured elsewhere (the collector thread), already in µs.
    pub fn record_us(&mut self, phase: &'static str, us: u32) {
        self.phases.entry(phase).or_default().push(us);
    }

    /// p50/p99/max per phase over the retained samples, phases in name order.
    pub fn report(&self) -> Vec<PhaseStat> {
        self.phases
            .iter()
            .filter(|(_, r)| !r.samples.is_empty())
            .map(|(phase, r)| {
                let mut s = r.samples.clone();
                s.sort_unstable();
                PhaseStat {
                    phase,
                    n: r.seen,
                    p50_us: percentile(&s, 50),
                    p99_us: percentile(&s, 99),
                    max_us: *s.last().unwrap_or(&0),
                }
            })
            .collect()
    }

    /// The report as JSON lines: `{"timings":"<phase>","n":N,"p50_us":A,"p99_us":B,"max_us":C}`.
    pub fn to_json_lines(&self) -> String {
        self.report()
            .iter()
            .map(|s| {
                format!(
                    "{{\"timings\":\"{}\",\"n\":{},\"p50_us\":{},\"p99_us\":{},\"max_us\":{}}}\n",
                    s.phase, s.n, s.p50_us, s.p99_us, s.max_us
                )
            })
            .collect()
    }
}

/// Nearest-rank percentile of an ascending slice; 0 for an empty one.
pub fn percentile(sorted: &[u32], p: u32) -> u32 {
    if sorted.is_empty() {
        return 0;
    }
    let rank = (u64::from(p) * sorted.len() as u64).div_ceil(100).max(1);
    sorted[usize::try_from(rank - 1).unwrap_or(0).min(sorted.len() - 1)]
}

/// Run `f` as `phase`. With `None` this is exactly `f()`.
#[inline]
pub fn time<R>(t: Option<&mut Timings>, phase: &'static str, f: impl FnOnce() -> R) -> R {
    match t {
        None => f(),
        Some(t) => {
            let _span = tracing::trace_span!("ttop.phase", phase).entered();
            let start = Instant::now();
            let r = f();
            t.record(phase, start.elapsed());
            r
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_is_nearest_rank() {
        let s: Vec<u32> = (1..=100).collect();
        assert_eq!(percentile(&s, 50), 50);
        assert_eq!(percentile(&s, 99), 99);
        assert_eq!(percentile(&[7], 99), 7);
        assert_eq!(percentile(&[], 50), 0);
        assert_eq!(percentile(&[1, 2], 50), 1);
    }

    #[test]
    fn off_runs_the_closure_and_records_nothing() {
        let mut calls = 0;
        let r = time(None, "diff", || {
            calls += 1;
            42
        });
        assert_eq!((r, calls), (42, 1));
    }

    /// The planted-sleep falsifier at unit level: a sleep inside one phase shows up in
    /// that phase's p50 and in no other. `scripts/ttop_timings_gate.sh --mutant` does the
    /// same through the real binary.
    #[test]
    fn a_planted_sleep_lands_in_its_own_phase() {
        let mut t = Timings::new();
        for _ in 0..5 {
            time(Some(&mut t), "layout_render", || {
                std::thread::sleep(Duration::from_millis(50))
            });
            time(Some(&mut t), "diff", || ());
        }
        let r = t.report();
        let get = |p| {
            r.iter()
                .find(|s| s.phase == p)
                .expect("phase present")
                .clone()
        };
        // sleep never returns early, so the lower bound is exact; the empty `diff` phase
        // would need most of its samples preempted for 50 ms to reach the bound
        assert!(get("layout_render").p50_us >= 50_000, "{r:?}");
        assert!(get("diff").p50_us < 50_000, "{r:?}");
        assert_eq!(get("diff").n, 5);
    }

    #[test]
    fn samples_are_bounded() {
        let mut t = Timings::new();
        for i in 0..(CAP as u32 * 3) {
            t.record_us("cpu", i);
        }
        assert_eq!(t.phases["cpu"].samples.len(), CAP);
        let s = &t.report()[0];
        assert_eq!(s.n, CAP as u64 * 3);
        assert!(
            s.p50_us >= CAP as u32 * 2,
            "old samples were overwritten: {s:?}"
        );
    }

    #[test]
    fn json_lines_have_the_declared_schema() {
        let mut t = Timings::new();
        t.record_us("write", 10);
        assert_eq!(
            t.to_json_lines(),
            "{\"timings\":\"write\",\"n\":1,\"p50_us\":10,\"p99_us\":10,\"max_us\":10}\n"
        );
    }
}
