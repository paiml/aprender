#![allow(dead_code)]
//! Per-step wall-clock profiler for CUDA training (KAIZEN-047).
//!
//! Collects `Instant`-based timings for each phase of `train_step_single()`.
//! Reports per-step breakdown and running statistics after N steps.
//!
//! # Contract (C-STEPPROF-001)
//!
//! - Zero-overhead when disabled: all methods are no-ops
//! - No GPU synchronization added (relies on existing sync points)
//! - Timings include CPU→GPU async dispatch latency (not pure kernel time)
//! - Report interval configurable at construction
//!
//! # Phases measured
//!
//! 1. `embed`    — CPU embedding lookup
//! 2. `h2d`      — Hidden state upload (H2D transfer + padding)
//! 3. `forward`  — Block forward loop (includes D2D layer_input saves)
//! 4. `norm_lm`  — Final RMSNorm + LM head GEMM + logits D2H
//! 5. `loss`     — CPU softmax + cross-entropy + gradient
//! 6. `grad_h2d` — Grad logits upload (H2D transfer)
//! 7. `lm_bwd`   — LM head backward (GEMM_A + GEMM_B + clip)
//! 8. `norm_bwd` — Final RMSNorm backward + clip
//! 9. `blk_bwd`  — Block backward loop (includes recompute + optimizer)
//! 10. `embed_bwd` — Embedding backward (D2H + clip + scatter-add)
//! 11. `opt`      — CPU optimizer step (embedding + bookkeeping)

use std::time::{Duration, Instant};

/// Phase indices (must match `PHASE_NAMES`).
const EMBED: usize = 0;
const H2D: usize = 1;
const FORWARD: usize = 2;
const NORM_LM: usize = 3;
const LOSS: usize = 4;
const GRAD_H2D: usize = 5;
const LM_BWD: usize = 6;
const NORM_BWD: usize = 7;
const BLK_BWD: usize = 8;
const EMBED_BWD: usize = 9;
const OPT: usize = 10;
const NUM_PHASES: usize = 11;

const PHASE_NAMES: [&str; NUM_PHASES] = [
    "embed",
    "h2d",
    "forward",
    "norm_lm",
    "loss",
    "grad_h2d",
    "lm_bwd",
    "norm_bwd",
    "blk_bwd",
    "embed_bwd",
    "opt",
];

/// Maximum number of transformer layers to profile.
const MAX_LAYERS: usize = 64;

/// Per-operation indices for within-layer profiling (PMAT-483/entrenar#328).
/// These track where time goes INSIDE each layer's forward and backward.
const OP_RMSNORM_ATTN: usize = 0;
const OP_QKV_GEMM: usize = 1;
const OP_ATTENTION: usize = 2;
const OP_O_PROJ: usize = 3;
const OP_RMSNORM_FFN: usize = 4;
const OP_GATE_UP_GEMM: usize = 5;
const OP_SILU: usize = 6;
const OP_DOWN_GEMM: usize = 7;
const OP_LORA: usize = 8;
const OP_DOWN_BWD: usize = 9;
const OP_SWIGLU_BWD: usize = 10;
const OP_GATE_UP_BWD: usize = 11;
const OP_ATTN_BWD: usize = 12;
const OP_QKV_BWD: usize = 13;
const OP_NORM_BWD: usize = 14;
const OP_LORA_BWD: usize = 15;
const NUM_OPS: usize = 16;

const OP_NAMES: [&str; NUM_OPS] = [
    "rmsnorm_attn",
    "qkv_gemm",
    "attention",
    "o_proj",
    "rmsnorm_ffn",
    "gate_up_gemm",
    "silu",
    "down_gemm",
    "lora",
    "down_bwd",
    "swiglu_bwd",
    "gate_up_bwd",
    "attn_bwd",
    "qkv_bwd",
    "norm_bwd",
    "lora_bwd",
];

/// Per-step timing accumulator.
///
/// Usage: call `begin(phase)` before each section, `end(phase)` after.
/// Call `finish_step()` to record the step and optionally print a report.
///
/// Per-layer profiling (PMAT-480): call `begin_layer(layer)` / `end_layer_fwd(layer)`
/// and `end_layer_bwd(layer)` inside the forward/backward loops to capture
/// per-layer timing. Reports per-layer breakdown when enabled.
pub struct StepProfiler {
    enabled: bool,
    /// Report every N steps (0 = never auto-report)
    report_interval: usize,
    /// Current step's phase timings
    current: [Duration; NUM_PHASES],
    /// Phase start timestamp (set by `begin`)
    phase_start: Option<Instant>,
    /// Step-level start timestamp
    step_start: Option<Instant>,
    /// Accumulated totals across all steps
    totals: [Duration; NUM_PHASES],
    /// Total wall-clock across all steps
    total_wall: Duration,
    /// Number of completed steps
    step_count: usize,
    /// Per-step wall-clock durations (for percentile analysis)
    step_durations: Vec<Duration>,
    /// Per-layer forward timing (accumulated across steps, PMAT-480)
    layer_fwd_totals: Vec<Duration>,
    /// Per-layer backward timing (accumulated across steps, PMAT-480)
    layer_bwd_totals: Vec<Duration>,
    /// Layer-level start timestamp (set by `begin_layer`)
    layer_start: Option<Instant>,
    /// Number of layers detected (set on first step)
    num_layers: usize,
    /// PMAT-483: Per-operation timing (accumulated across all layers and steps)
    op_totals: [Duration; NUM_OPS],
    /// Per-operation start timestamp
    op_start: Option<Instant>,
}

impl StepProfiler {
    /// Create a new profiler.
    ///
    /// - `enabled`: if false, all methods are no-ops
    /// - `report_interval`: print summary every N steps (0 = manual only)
    pub fn new(enabled: bool, report_interval: usize) -> Self {
        Self {
            enabled,
            report_interval,
            current: [Duration::ZERO; NUM_PHASES],
            phase_start: None,
            step_start: None,
            totals: [Duration::ZERO; NUM_PHASES],
            total_wall: Duration::ZERO,
            step_count: 0,
            step_durations: Vec::new(),
            layer_fwd_totals: vec![Duration::ZERO; MAX_LAYERS],
            layer_bwd_totals: vec![Duration::ZERO; MAX_LAYERS],
            layer_start: None,
            num_layers: 0,
            op_totals: [Duration::ZERO; NUM_OPS],
            op_start: None,
        }
    }

    /// Disabled (no-op) profiler.
    pub fn disabled() -> Self {
        Self::new(false, 0)
    }

    /// Mark the start of a training step.
    #[inline]
    pub fn begin_step(&mut self) {
        if !self.enabled {
            return;
        }
        self.current = [Duration::ZERO; NUM_PHASES];
        self.step_start = Some(Instant::now());
    }

    /// Mark the start of a phase. Must call `end(phase)` to record.
    #[inline]
    pub fn begin(&mut self, _phase: usize) {
        if !self.enabled {
            return;
        }
        self.phase_start = Some(Instant::now());
    }

    /// Record elapsed time for a phase (since last `begin`).
    #[inline]
    pub fn end(&mut self, phase: usize) {
        if !self.enabled {
            return;
        }
        if let Some(start) = self.phase_start.take() {
            self.current[phase] += start.elapsed();
        }
    }

    /// Start per-layer timing (PMAT-480). Call before each layer's forward or backward.
    #[inline]
    pub fn begin_layer(&mut self) {
        if !self.enabled {
            return;
        }
        self.layer_start = Some(Instant::now());
    }

    /// Record per-layer forward time (PMAT-480). Call after layer forward completes.
    #[inline]
    pub fn end_layer_fwd(&mut self, layer: usize) {
        if !self.enabled {
            return;
        }
        if let Some(start) = self.layer_start.take() {
            if layer < MAX_LAYERS {
                self.layer_fwd_totals[layer] += start.elapsed();
                if layer >= self.num_layers {
                    self.num_layers = layer + 1;
                }
            }
        }
    }

    /// Record per-layer backward time (PMAT-480). Call after layer backward completes.
    #[inline]
    pub fn end_layer_bwd(&mut self, layer: usize) {
        if !self.enabled {
            return;
        }
        if let Some(start) = self.layer_start.take() {
            if layer < MAX_LAYERS {
                self.layer_bwd_totals[layer] += start.elapsed();
                if layer >= self.num_layers {
                    self.num_layers = layer + 1;
                }
            }
        }
    }

    /// Finish the current step. Records totals and optionally prints report.
    pub fn finish_step(&mut self) {
        if !self.enabled {
            return;
        }
        let step_wall = self.step_start.take().map_or(Duration::ZERO, |s| s.elapsed());

        for i in 0..NUM_PHASES {
            self.totals[i] += self.current[i];
        }
        self.total_wall += step_wall;
        self.step_count += 1;
        self.step_durations.push(step_wall);

        if self.report_interval > 0 && self.step_count.is_multiple_of(self.report_interval) {
            self.print_report();
        }
    }

    /// Print cumulative profiling report to stdout.
    pub fn print_report(&self) {
        if self.step_count == 0 {
            return;
        }

        let total_us = self.total_wall.as_micros() as f64;
        let avg_step_us = total_us / self.step_count as f64;

        println!(
            "\n┌─ Step Profiler ({} steps, avg {:.1} ms/step) ─┐",
            self.step_count,
            avg_step_us / 1000.0
        );
        println!("│ {:>10} │ {:>8} │ {:>6} │ {:>8} │", "phase", "total_ms", "pct", "avg_ms");
        println!("│ {:-<10} │ {:-<8} │ {:-<6} │ {:-<8} │", "", "", "", "");

        let mut accounted = Duration::ZERO;
        for i in 0..NUM_PHASES {
            let t = self.totals[i];
            accounted += t;
            let ms = t.as_micros() as f64 / 1000.0;
            let pct = if total_us > 0.0 { t.as_micros() as f64 / total_us * 100.0 } else { 0.0 };
            let avg = ms / self.step_count as f64;
            println!("│ {:>10} │ {:>8.1} │ {:>5.1}% │ {:>8.2} │", PHASE_NAMES[i], ms, pct, avg);
        }

        let unaccounted = self.total_wall.saturating_sub(accounted);
        let unaccounted_pct =
            if total_us > 0.0 { unaccounted.as_micros() as f64 / total_us * 100.0 } else { 0.0 };
        println!(
            "│ {:>10} │ {:>8.1} │ {:>5.1}% │ {:>8.2} │",
            "other",
            unaccounted.as_micros() as f64 / 1000.0,
            unaccounted_pct,
            unaccounted.as_micros() as f64 / 1000.0 / self.step_count as f64
        );

        println!(
            "│ {:>10} │ {:>8.1} │ {:>5}  │ {:>8.2} │",
            "TOTAL",
            total_us / 1000.0,
            "100%",
            avg_step_us / 1000.0
        );
        println!("└────────────┴──────────┴────────┴──────────┘");

        // Per-layer breakdown (PMAT-480)
        if self.num_layers > 0 && self.step_count > 0 {
            println!(
                "\n┌─ Per-Layer Profile ({} layers, {} steps) ─┐",
                self.num_layers, self.step_count
            );
            println!(
                "│ {:>5} │ {:>8} │ {:>8} │ {:>8} │ {:>8} │",
                "layer", "fwd_ms", "bwd_ms", "fwd_avg", "bwd_avg"
            );
            println!("│ {:->5} │ {:->8} │ {:->8} │ {:->8} │ {:->8} │", "", "", "", "", "");
            let mut fwd_total = Duration::ZERO;
            let mut bwd_total = Duration::ZERO;
            for i in 0..self.num_layers {
                let fwd = self.layer_fwd_totals[i];
                let bwd = self.layer_bwd_totals[i];
                fwd_total += fwd;
                bwd_total += bwd;
                let fwd_ms = fwd.as_micros() as f64 / 1000.0;
                let bwd_ms = bwd.as_micros() as f64 / 1000.0;
                let fwd_avg = fwd_ms / self.step_count as f64;
                let bwd_avg = bwd_ms / self.step_count as f64;
                println!(
                    "│ {i:>5} │ {fwd_ms:>8.1} │ {bwd_ms:>8.1} │ {fwd_avg:>8.2} │ {bwd_avg:>8.2} │"
                );
            }
            let fwd_total_ms = fwd_total.as_micros() as f64 / 1000.0;
            let bwd_total_ms = bwd_total.as_micros() as f64 / 1000.0;
            println!(
                "│ {:>5} │ {:>8.1} │ {:>8.1} │ {:>8.2} │ {:>8.2} │",
                "TOTAL",
                fwd_total_ms,
                bwd_total_ms,
                fwd_total_ms / self.step_count as f64,
                bwd_total_ms / self.step_count as f64
            );
            println!("└───────┴──────────┴──────────┴──────────┴──────────┘");

            // Identify hotspot layers (>1.5x average)
            let avg_fwd = fwd_total / self.num_layers as u32;
            let avg_bwd = bwd_total / self.num_layers as u32;
            let mut hotspots = Vec::new();
            for i in 0..self.num_layers {
                let fwd_ratio = if avg_fwd.as_nanos() > 0 {
                    self.layer_fwd_totals[i].as_nanos() as f64 / avg_fwd.as_nanos() as f64
                } else {
                    0.0
                };
                let bwd_ratio = if avg_bwd.as_nanos() > 0 {
                    self.layer_bwd_totals[i].as_nanos() as f64 / avg_bwd.as_nanos() as f64
                } else {
                    0.0
                };
                if fwd_ratio > 1.5 || bwd_ratio > 1.5 {
                    hotspots.push((i, fwd_ratio, bwd_ratio));
                }
            }
            if !hotspots.is_empty() {
                println!("  Hotspot layers (>1.5x average):");
                for (layer, fwd_r, bwd_r) in &hotspots {
                    println!("    L{layer}: fwd {fwd_r:.1}x, bwd {bwd_r:.1}x");
                }
            }
        }

        // Percentiles for step wall-clock
        if self.step_durations.len() >= 10 {
            let mut sorted: Vec<u128> =
                self.step_durations.iter().map(std::time::Duration::as_micros).collect();
            sorted.sort_unstable();
            let p50 = sorted[sorted.len() / 2];
            let p95 = sorted[sorted.len() * 95 / 100];
            let p99 = sorted[sorted.len() * 99 / 100];
            println!(
                "  Step latency: p50={:.1}ms p95={:.1}ms p99={:.1}ms",
                p50 as f64 / 1000.0,
                p95 as f64 / 1000.0,
                p99 as f64 / 1000.0
            );
        }
    }

    /// PMAT-483/entrenar#328: Start timing a per-operation phase within a layer.
    #[inline]
    pub fn begin_op(&mut self) {
        if !self.enabled {
            return;
        }
        self.op_start = Some(Instant::now());
    }

    /// PMAT-483/entrenar#328: Record elapsed time for a per-operation phase.
    #[inline]
    pub fn end_op(&mut self, op: usize) {
        if !self.enabled {
            return;
        }
        if let Some(start) = self.op_start.take() {
            if op < NUM_OPS {
                self.op_totals[op] += start.elapsed();
            }
        }
    }

    /// PMAT-483: Feed pre-accumulated microseconds for an operation (from CudaBlockScratch).
    #[inline]
    pub fn end_op_raw(&mut self, op: usize, us: u64) {
        if !self.enabled || op >= NUM_OPS {
            return;
        }
        self.op_totals[op] += Duration::from_micros(us);
    }

    /// Whether the profiler is active.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Get step count.
    pub fn step_count(&self) -> usize {
        self.step_count
    }

    /// PMAT-483: Emit structured JSON profiling report to stderr.
    /// Parseable by canary scripts for scientific analysis.
    pub fn print_json_report(&self) {
        if self.step_count == 0 {
            return;
        }

        let total_us = self.total_wall.as_micros() as f64;
        let avg_step_ms = total_us / self.step_count as f64 / 1000.0;

        let mut accounted_us = 0u128;
        let mut phases = Vec::new();
        for i in 0..NUM_PHASES {
            let t = self.totals[i];
            accounted_us += t.as_micros();
            let ms = t.as_micros() as f64 / 1000.0;
            let pct = if total_us > 0.0 { t.as_micros() as f64 / total_us * 100.0 } else { 0.0 };
            let avg = ms / self.step_count as f64;
            phases.push(format!(
                "\"{}\":{{\"total_ms\":{:.1},\"pct\":{:.1},\"avg_ms\":{:.2}}}",
                PHASE_NAMES[i], ms, pct, avg
            ));
        }

        let wall_coverage = if total_us > 0.0 { accounted_us as f64 / total_us } else { 0.0 };

        let mut layers_json = Vec::new();
        for i in 0..self.num_layers {
            let fwd_ms = self.layer_fwd_totals[i].as_micros() as f64 / 1000.0;
            let bwd_ms = self.layer_bwd_totals[i].as_micros() as f64 / 1000.0;
            layers_json
                .push(format!("{{\"layer\":{i},\"fwd_ms\":{fwd_ms:.1},\"bwd_ms\":{bwd_ms:.1}}}"));
        }

        // Classify bottleneck based on phase distribution
        let forward_pct = if total_us > 0.0 {
            self.totals[FORWARD].as_micros() as f64 / total_us * 100.0
        } else {
            0.0
        };
        let transfer_pct = if total_us > 0.0 {
            (self.totals[H2D].as_micros() + self.totals[GRAD_H2D].as_micros()) as f64 / total_us
                * 100.0
        } else {
            0.0
        };
        let bottleneck = if transfer_pct > 30.0 {
            "transfer"
        } else if forward_pct < 20.0 {
            "launch"
        } else {
            "memory_bw"
        };

        // PMAT-483: Per-operation breakdown
        let mut ops_json = Vec::new();
        let mut total_op_us = 0u128;
        for i in 0..NUM_OPS {
            let t = self.op_totals[i];
            let us = t.as_micros();
            total_op_us += us;
            if us > 0 {
                let ms = us as f64 / 1000.0;
                ops_json.push(format!("\"{}\":{:.1}", OP_NAMES[i], ms));
            }
        }

        // Classify which operation type dominates
        let gemm_us = self.op_totals[OP_QKV_GEMM].as_micros()
            + self.op_totals[OP_O_PROJ].as_micros()
            + self.op_totals[OP_GATE_UP_GEMM].as_micros()
            + self.op_totals[OP_DOWN_GEMM].as_micros();
        let gemm_bwd_us = self.op_totals[OP_QKV_BWD].as_micros()
            + self.op_totals[OP_GATE_UP_BWD].as_micros()
            + self.op_totals[OP_DOWN_BWD].as_micros();
        let total_gemm_us = gemm_us + gemm_bwd_us;
        let gemm_pct =
            if total_op_us > 0 { total_gemm_us as f64 / total_op_us as f64 * 100.0 } else { 0.0 };

        eprintln!(
            "{{\"_profiler\":\"step_profiler_v2\",\"steps\":{},\"avg_step_ms\":{:.2},\"wall_coverage\":{:.3},\"bottleneck\":\"{}\",\"gemm_pct\":{:.1},\"phases\":{{{}}},\"per_layer\":[{}],\"ops\":{{{}}}}}",
            self.step_count,
            avg_step_ms,
            wall_coverage,
            bottleneck,
            gemm_pct,
            phases.join(","),
            layers_json.join(","),
            ops_json.join(","),
        );
    }

    /// PMAT-483: Feed per-layer timing data from InstructGpuTrainingState.
    /// Call once per step after forward+backward complete.
    pub fn record_layer_times(&mut self, fwd_us: &[u64], bwd_us: &[u64]) {
        if !self.enabled {
            return;
        }
        for (i, &us) in fwd_us.iter().enumerate() {
            if i < MAX_LAYERS && us > 0 {
                self.layer_fwd_totals[i] += std::time::Duration::from_micros(us);
                if i >= self.num_layers {
                    self.num_layers = i + 1;
                }
            }
        }
        for (i, &us) in bwd_us.iter().enumerate() {
            if i < MAX_LAYERS && us > 0 {
                self.layer_bwd_totals[i] += std::time::Duration::from_micros(us);
                if i >= self.num_layers {
                    self.num_layers = i + 1;
                }
            }
        }
    }

    // Phase constants for external callers
    pub const EMBED: usize = EMBED;

    // Per-operation constants (PMAT-483/entrenar#328)
    pub const OP_RMSNORM_ATTN: usize = OP_RMSNORM_ATTN;
    pub const OP_QKV_GEMM: usize = OP_QKV_GEMM;
    pub const OP_ATTENTION: usize = OP_ATTENTION;
    pub const OP_O_PROJ: usize = OP_O_PROJ;
    pub const OP_RMSNORM_FFN: usize = OP_RMSNORM_FFN;
    pub const OP_GATE_UP_GEMM: usize = OP_GATE_UP_GEMM;
    pub const OP_SILU: usize = OP_SILU;
    pub const OP_DOWN_GEMM: usize = OP_DOWN_GEMM;
    pub const OP_LORA: usize = OP_LORA;
    pub const OP_DOWN_BWD: usize = OP_DOWN_BWD;
    pub const OP_SWIGLU_BWD: usize = OP_SWIGLU_BWD;
    pub const OP_GATE_UP_BWD: usize = OP_GATE_UP_BWD;
    pub const OP_ATTN_BWD: usize = OP_ATTN_BWD;
    pub const OP_QKV_BWD: usize = OP_QKV_BWD;
    pub const OP_NORM_BWD: usize = OP_NORM_BWD;
    pub const OP_LORA_BWD: usize = OP_LORA_BWD;
    pub const H2D: usize = H2D;
    pub const FORWARD: usize = FORWARD;
    pub const NORM_LM: usize = NORM_LM;
    pub const LOSS: usize = LOSS;
    pub const GRAD_H2D: usize = GRAD_H2D;
    pub const LM_BWD: usize = LM_BWD;
    pub const NORM_BWD: usize = NORM_BWD;
    pub const BLK_BWD: usize = BLK_BWD;
    pub const EMBED_BWD: usize = EMBED_BWD;
    pub const OPT: usize = OPT;
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_disabled_profiler_is_noop() {
        let mut p = StepProfiler::disabled();
        p.begin_step();
        p.begin(StepProfiler::EMBED);
        p.end(StepProfiler::EMBED);
        p.finish_step();
        assert_eq!(p.step_count(), 0);
    }

    #[test]
    fn test_enabled_profiler_counts_steps() {
        let mut p = StepProfiler::new(true, 0);
        p.begin_step();
        p.begin(StepProfiler::EMBED);
        std::thread::sleep(Duration::from_millis(1));
        p.end(StepProfiler::EMBED);
        p.finish_step();
        assert_eq!(p.step_count(), 1);
        assert!(p.totals[EMBED] >= Duration::from_micros(500));
    }

    #[test]
    fn test_multiple_steps_accumulate() {
        let mut p = StepProfiler::new(true, 0);
        for _ in 0..3 {
            p.begin_step();
            p.begin(StepProfiler::LOSS);
            std::thread::sleep(Duration::from_millis(1));
            p.end(StepProfiler::LOSS);
            p.finish_step();
        }
        assert_eq!(p.step_count(), 3);
        assert!(p.totals[LOSS] >= Duration::from_millis(3));
    }

    #[test]
    fn test_unaccounted_time_captured() {
        let mut p = StepProfiler::new(true, 0);
        p.begin_step();
        // Sleep without recording any phase → should appear as "other"
        std::thread::sleep(Duration::from_millis(2));
        p.finish_step();
        assert!(p.total_wall >= Duration::from_millis(2));
        // All phase totals should be zero
        for i in 0..NUM_PHASES {
            assert_eq!(p.totals[i], Duration::ZERO);
        }
    }

    #[test]
    fn test_is_enabled() {
        let p = StepProfiler::new(true, 0);
        assert!(p.is_enabled());
        let p = StepProfiler::disabled();
        assert!(!p.is_enabled());
    }

    #[test]
    fn test_step_count_starts_at_zero() {
        let p = StepProfiler::new(true, 0);
        assert_eq!(p.step_count(), 0);
    }

    #[test]
    fn test_disabled_profiler_step_count_stays_zero() {
        let mut p = StepProfiler::disabled();
        for _ in 0..5 {
            p.begin_step();
            p.begin(StepProfiler::EMBED);
            p.end(StepProfiler::EMBED);
            p.finish_step();
        }
        assert_eq!(p.step_count(), 0);
    }

    #[test]
    fn test_all_phase_constants() {
        // Verify phase constants match expected values
        assert_eq!(StepProfiler::EMBED, 0);
        assert_eq!(StepProfiler::H2D, 1);
        assert_eq!(StepProfiler::FORWARD, 2);
        assert_eq!(StepProfiler::NORM_LM, 3);
        assert_eq!(StepProfiler::LOSS, 4);
        assert_eq!(StepProfiler::GRAD_H2D, 5);
        assert_eq!(StepProfiler::LM_BWD, 6);
        assert_eq!(StepProfiler::NORM_BWD, 7);
        assert_eq!(StepProfiler::BLK_BWD, 8);
        assert_eq!(StepProfiler::EMBED_BWD, 9);
        assert_eq!(StepProfiler::OPT, 10);
    }

    #[test]
    fn test_phase_names_count() {
        assert_eq!(PHASE_NAMES.len(), NUM_PHASES);
        assert_eq!(NUM_PHASES, 11);
    }

    #[test]
    fn test_multiple_phases_in_one_step() {
        let mut p = StepProfiler::new(true, 0);
        p.begin_step();

        p.begin(StepProfiler::EMBED);
        std::thread::sleep(Duration::from_millis(1));
        p.end(StepProfiler::EMBED);

        p.begin(StepProfiler::FORWARD);
        std::thread::sleep(Duration::from_millis(1));
        p.end(StepProfiler::FORWARD);

        p.begin(StepProfiler::LOSS);
        std::thread::sleep(Duration::from_millis(1));
        p.end(StepProfiler::LOSS);

        p.finish_step();

        assert_eq!(p.step_count(), 1);
        assert!(p.totals[EMBED] > Duration::ZERO);
        assert!(p.totals[FORWARD] > Duration::ZERO);
        assert!(p.totals[LOSS] > Duration::ZERO);
        // Unrecorded phases should be zero
        assert_eq!(p.totals[H2D], Duration::ZERO);
        assert_eq!(p.totals[NORM_LM], Duration::ZERO);
    }

    #[test]
    fn test_end_without_begin_is_noop() {
        let mut p = StepProfiler::new(true, 0);
        p.begin_step();
        // end without begin should be a no-op (phase_start is None)
        p.end(StepProfiler::EMBED);
        p.finish_step();
        assert_eq!(p.totals[EMBED], Duration::ZERO);
    }

    #[test]
    fn test_print_report_empty_is_noop() {
        let p = StepProfiler::new(true, 0);
        // No steps recorded — should not panic
        p.print_report();
        assert_eq!(p.step_count(), 0);
    }

    #[test]
    fn test_print_report_with_data() {
        let mut p = StepProfiler::new(true, 0);
        for _ in 0..3 {
            p.begin_step();
            p.begin(StepProfiler::EMBED);
            std::thread::sleep(Duration::from_millis(1));
            p.end(StepProfiler::EMBED);
            p.finish_step();
        }
        // Should print without panic
        p.print_report();
        assert_eq!(p.step_count(), 3);
    }

    #[test]
    fn test_report_interval_auto_print() {
        let mut p = StepProfiler::new(true, 2); // report every 2 steps
        for _ in 0..4 {
            p.begin_step();
            p.begin(StepProfiler::LOSS);
            p.end(StepProfiler::LOSS);
            p.finish_step();
        }
        assert_eq!(p.step_count(), 4);
        // Report should have been triggered at steps 2 and 4
    }

    #[test]
    fn test_step_durations_tracked() {
        let mut p = StepProfiler::new(true, 0);
        for _ in 0..5 {
            p.begin_step();
            std::thread::sleep(Duration::from_millis(1));
            p.finish_step();
        }
        assert_eq!(p.step_durations.len(), 5);
        for d in &p.step_durations {
            assert!(*d >= Duration::from_micros(500));
        }
    }

    #[test]
    fn test_total_wall_accumulates() {
        let mut p = StepProfiler::new(true, 0);
        p.begin_step();
        std::thread::sleep(Duration::from_millis(2));
        p.finish_step();

        p.begin_step();
        std::thread::sleep(Duration::from_millis(2));
        p.finish_step();

        assert!(p.total_wall >= Duration::from_millis(4));
    }

    #[test]
    fn test_percentiles_with_enough_steps() {
        let mut p = StepProfiler::new(true, 0);
        for _ in 0..20 {
            p.begin_step();
            std::thread::sleep(Duration::from_millis(1));
            p.finish_step();
        }
        // print_report will show percentiles (>= 10 steps)
        p.print_report();
        assert_eq!(p.step_durations.len(), 20);
    }

    #[test]
    fn test_finish_step_without_begin_step() {
        let mut p = StepProfiler::new(true, 0);
        // finish_step without begin_step — step_start is None
        p.finish_step();
        // Should record Duration::ZERO for wall time
        assert_eq!(p.step_count(), 1);
        assert_eq!(p.total_wall, Duration::ZERO);
    }

    // --- Per-layer profiling tests (PMAT-480) ---

    #[test]
    fn test_per_layer_fwd_timing() {
        let mut p = StepProfiler::new(true, 0);
        p.begin_step();
        for layer in 0..3 {
            p.begin_layer();
            std::thread::sleep(Duration::from_millis(1));
            p.end_layer_fwd(layer);
        }
        p.finish_step();
        assert_eq!(p.num_layers, 3);
        for layer in 0..3 {
            assert!(p.layer_fwd_totals[layer] >= Duration::from_micros(500));
        }
    }

    #[test]
    fn test_per_layer_bwd_timing() {
        let mut p = StepProfiler::new(true, 0);
        p.begin_step();
        for layer in (0..4).rev() {
            p.begin_layer();
            std::thread::sleep(Duration::from_millis(1));
            p.end_layer_bwd(layer);
        }
        p.finish_step();
        assert_eq!(p.num_layers, 4);
        for layer in 0..4 {
            assert!(p.layer_bwd_totals[layer] >= Duration::from_micros(500));
        }
    }

    #[test]
    fn test_per_layer_accumulates_across_steps() {
        let mut p = StepProfiler::new(true, 0);
        for _ in 0..3 {
            p.begin_step();
            p.begin_layer();
            std::thread::sleep(Duration::from_millis(1));
            p.end_layer_fwd(0);
            p.finish_step();
        }
        assert!(p.layer_fwd_totals[0] >= Duration::from_millis(3));
    }

    #[test]
    fn test_per_layer_disabled_is_noop() {
        let mut p = StepProfiler::disabled();
        p.begin_layer();
        p.end_layer_fwd(0);
        p.begin_layer();
        p.end_layer_bwd(0);
        assert_eq!(p.num_layers, 0);
    }

    #[test]
    fn test_per_layer_out_of_bounds_ignored() {
        let mut p = StepProfiler::new(true, 0);
        p.begin_step();
        p.begin_layer();
        // Layer index >= MAX_LAYERS should be silently ignored
        p.end_layer_fwd(MAX_LAYERS + 1);
        p.finish_step();
        assert_eq!(p.num_layers, 0);
    }

    #[test]
    fn test_per_layer_report_prints() {
        let mut p = StepProfiler::new(true, 0);
        for _ in 0..2 {
            p.begin_step();
            for layer in 0..3 {
                p.begin_layer();
                std::thread::sleep(Duration::from_millis(1));
                p.end_layer_fwd(layer);
            }
            for layer in (0..3).rev() {
                p.begin_layer();
                std::thread::sleep(Duration::from_millis(1));
                p.end_layer_bwd(layer);
            }
            p.finish_step();
        }
        // Should print per-layer breakdown without panic
        p.print_report();
        assert_eq!(p.num_layers, 3);
    }
}
