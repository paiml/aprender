// Contract validation for ProfileReport (P15-06)
//
// Validates profiler output against provable-contracts YAML invariants.
// Contracts 1-3: gpu-decode-profiling-v1.yaml
// Contract 4: per-operation-training-profiling-v1.yaml
//
// Reference: realizr#202, candle-vs-apr P15-06

impl ProfileReport {
    /// Validate profiler output against gpu-decode-profiling-v1 contract invariants.
    ///
    /// Returns a list of (severity, message) for any violated invariants.
    /// Empty list = all contracts satisfied.
    ///
    /// Contract: gpu-decode-profiling-v1.yaml (provable-contracts)
    /// Reference: realizr#202, candle-vs-apr P15-06
    pub fn validate_contracts(&self) -> Vec<(ContractSeverity, String)> {
        let mut violations = Vec::new();

        if !self.is_real_data {
            return violations;
        }

        // CONTRACT 1: wall_coverage >= 0.85
        // "sum(brick_total_ns) / wall_clock_ns >= 0.85"
        // Catches: missing kernel recordings (e.g., 28 SwiGLU kernels in realizr#198)
        if self.total_inference_us > 0.0 {
            let brick_total: f64 = self.operations.values().map(|s| s.total_us).sum();
            let wall_coverage = brick_total / self.total_inference_us;
            if wall_coverage < 0.85 {
                violations.push((
                    ContractSeverity::Error,
                    format!(
                        "gpu-decode-profiling-v1 WALL_COVERAGE: {:.1}% < 85.0% minimum. \
                         {:.0}µs profiled / {:.0}µs wall. Missing bricks likely.",
                        wall_coverage * 100.0,
                        brick_total,
                        self.total_inference_us
                    ),
                ));
            }
        }

        // CONTRACT 2: Immediate sync fidelity — LmHead.avg > 10x RmsNorm.avg
        // "In Immediate mode, GPU-time-dominated bricks must show LmHead >> RmsNorm"
        // Catches: Deferred sync mode reporting CPU launch latency (3.4x error, qcd PMAT-3031)
        let lm_head = self.operations.get("LmHead");
        let rms_norm = self.operations.get("RmsNorm");
        if let (Some(lm), Some(rn)) = (lm_head, rms_norm) {
            if rn.avg_us > 0.0 && lm.avg_us > 0.0 {
                let ratio = lm.avg_us / rn.avg_us;
                if ratio < 10.0 {
                    violations.push((
                        ContractSeverity::Warning,
                        format!(
                            "gpu-decode-profiling-v1 SYNC_FIDELITY: LmHead/RmsNorm ratio = {:.1}x < 10x. \
                             LmHead={:.1}µs, RmsNorm={:.1}µs. Likely using Deferred sync (CPU launch latency, not GPU time).",
                            ratio, lm.avg_us, rn.avg_us
                        ),
                    ));
                }
            }
        }

        // CONTRACT 3: decoded_tokens == LmHead.count
        // "LmHead is invoked exactly once per decoded token"
        // Catches: token accounting errors, profiler miscounting
        if let Some(lm) = lm_head {
            if self.tokens_processed > 0 && lm.count != self.tokens_processed {
                violations.push((
                    ContractSeverity::Warning,
                    format!(
                        "gpu-decode-profiling-v1 TOKEN_ACCOUNTING: LmHead.count={} != tokens_processed={}. \
                         LmHead should fire exactly once per decoded token.",
                        lm.count, self.tokens_processed
                    ),
                ));
            }
        }

        // CONTRACT 4: per-operation-training-profiling-v1 — GEMM >= 50% of compute
        // "gemm_time / layer_fwd >= 0.50"
        // Catches: architecture regressions where non-GEMM ops dominate
        // Reference: provable-contracts/contracts/entrenar/per-operation-training-profiling-v1.yaml
        // Note: For M=1 decode, AttentionScore is memory-bound (not GEMM) — may legitimately
        // fail. Contract fires as Warning, not Error, for decode workloads.
        if self.total_inference_us > 0.0 {
            // GEMM-based operations: projections and LmHead (matmul-dominated)
            let gemm_ops = [
                "QkvProjection", "OutputProjection", "GateProjection",
                "UpProjection", "DownProjection", "LmHead",
                // Legacy/alternate names from BrickProfiler
                "attention_qkv", "mlp_gate_up", "mlp_down",
            ];
            let gemm_total: f64 = self.operations.iter()
                .filter(|(name, _)| gemm_ops.iter().any(|g| name.contains(g)))
                .map(|(_, stats)| stats.total_us)
                .sum();
            let gemm_pct = gemm_total / self.total_inference_us;
            if gemm_pct < 0.50 {
                violations.push((
                    ContractSeverity::Warning,
                    format!(
                        "per-op-training-v1 GEMM_DOMINANCE: GEMM ops = {:.1}% < 50.0% minimum. \
                         {:.0}µs GEMM / {:.0}µs total. Memory-bound ops may dominate (expected for M=1 decode).",
                        gemm_pct * 100.0, gemm_total, self.total_inference_us
                    ),
                ));
            }
        }

        violations
    }
}
