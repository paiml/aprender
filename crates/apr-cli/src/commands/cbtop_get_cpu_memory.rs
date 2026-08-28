
/// Get CPU info (best effort)
fn get_cpu_info() -> String {
    batuta_common::sys::get_cpu_info()
}

/// Get system memory in GB (best effort)
fn get_memory_gb() -> u32 {
    #[cfg(target_os = "linux")]
    {
        let kb = std::fs::read_to_string("/proc/meminfo")
            .ok()
            .and_then(|content| {
                content.lines()
                    .find(|l| l.starts_with("MemTotal:"))
                    .and_then(|l| l.split_whitespace().nth(1))
                    .and_then(|s| s.parse::<u64>().ok())
            });
        if let Some(kb) = kb {
            #[allow(clippy::cast_possible_truncation)]
            return (kb / 1_048_576) as u32;
        }
    }
    64
}

fn score_brick(b: &BrickTiming) -> BrickScore {
    let gap = b.gap_factor();
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let score = if gap <= 1.0 + 1e-9 {
        100
    } else if gap <= 1.2 {
        (100.0 - (gap - 1.0) * 50.0) as u32
    } else {
        (100.0 - (gap - 1.0) * 100.0).max(0.0) as u32
    };
    BrickScore {
        name: b.name.to_string(),
        score,
        grade: score_to_grade(score),
        budget_us: b.budget_us,
        actual_us: b.actual_us,
        gap_factor: gap,
    }
}

fn cv_percent_from_samples(samples: &[f64]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    if mean <= 0.0 || samples.len() <= 1 {
        return 0.0;
    }
    let variance = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (samples.len() - 1) as f64;
    (variance.sqrt() / mean) * 100.0
}

fn percentiles_from_brick(brick: &BrickTiming) -> (f64, f64) {
    let mut sorted = brick.samples.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p50 = sorted.get(sorted.len() / 2).copied().unwrap_or(0.0);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let p99 = sorted
        .get((sorted.len() as f64 * 0.99) as usize)
        .copied()
        .unwrap_or(0.0);
    (p50, p99)
}

fn weighted_brick_score(brick_scores: &[BrickScore]) -> u32 {
    // GH-422 B6b: Equal-weight average across all N bricks.
    // Previous code used 7 hardcoded weights and silently dropped bricks beyond 7.
    if brick_scores.is_empty() {
        return 0;
    }
    let sum: f64 = brick_scores.iter().map(|b| b.score as f64).sum();
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    { (sum / brick_scores.len() as f64) as u32 }
}

/// The report's own verdict, as `(status, ci_result)`.
///
/// #2730: **an empty brick slice is a FAIL, never a pass.**
/// `bricks.iter().all(..)` is vacuously true over the empty set, and both
/// report builders used it bare. So a run in which the profiler collected no
/// per-brick data was scored `"status": "PASS"` / `"ci_result": "green"` inside
/// the same document that said `"brick_scores": []`, `"grade": "F"` and
/// `"falsification": {"passed": 0, "failed": 1}` — the `.max(1)` that produced
/// that `failed: 1` asserts one brick exists so it can be counted failed, while
/// the `.all()` four lines below asserted every brick passed. Identical input,
/// opposite conclusions.
///
/// It is not a display bug: `check_ci_thresholds` delegates to `ci_result`, so
/// `apr cbtop --ci` exited 0 on a grade-F report that measured nothing.
///
/// Zero bricks measured is a failure to MEASURE, not a measurement that passed,
/// so it earns the same verdict as a brick over budget. A non-empty slice is
/// judged exactly as before, so a healthy run is still green.
fn report_verdict(bricks: &[BrickScore]) -> (&'static str, &'static str) {
    if bricks.is_empty() {
        return ("FAIL", "red");
    }
    // 1e-9 epsilon: the budget is derived from the same profiler data, so a
    // healthy gap is ~1.0; without the epsilon, floating-point rounding turns
    // it into 1.0000000000001 and fails a passing brick.
    if bricks.iter().all(|b| b.gap_factor <= 1.0 + 1e-9) {
        ("PASS", "green")
    } else {
        ("FAIL", "red")
    }
}

/// Generate headless report from pipeline state (simulated data)
fn generate_headless_report_simulated(
    model_name: &str,
    pipeline: &PipelineState,
    _config: &CbtopConfig,
) -> HeadlessReport {
    // The date must be the real UTC date. A string-literal date with a
    // computed time-of-day stamped every report ever written — including
    // files persisted with --output for CI provenance — with the wrong day.
    let timestamp = chrono_timestamp();

    let brick_scores: Vec<BrickScore> = pipeline.bricks.iter().map(score_brick).collect();

    let all_samples: Vec<f64> = pipeline
        .bricks
        .iter()
        .flat_map(|b| b.samples.iter().copied())
        .collect();
    let cv_percent = cv_percent_from_samples(&all_samples);
    let (p50, p99) = pipeline.bricks.first().map_or((0.0, 0.0), percentiles_from_brick);

    // #2730: shared with the real-profiling builder so neither can drift back
    // into scoring an empty measurement green.
    let (status, ci_result) = report_verdict(&brick_scores);
    let pmat_brick_score = weighted_brick_score(&brick_scores);

    // GH-425 B14-B18: Derive falsification from brick pass/fail, not hardcoded.
    let n_bricks = brick_scores.len() as u32;
    let brick_passed = brick_scores.iter().filter(|b| b.gap_factor <= 1.0 + 1e-9).count() as u32;
    let brick_failed = n_bricks.saturating_sub(brick_passed);

    HeadlessReport {
        model: model_name.to_string(),
        timestamp,
        hardware: HardwareInfo {
            gpu: "NVIDIA RTX 4090 (simulated)".to_string(),
            cpu: "AMD Ryzen 9 7950X (simulated)".to_string(),
            memory_gb: 64,
        },
        throughput: ThroughputMetrics {
            tokens_per_sec: pipeline.current_tok_s,
            ttft_ms: pipeline.total_actual() * pipeline.total_layers as f64 / 1000.0,
            cv_percent,
            p50_us: p50,
            p99_us: p99,
        },
        brick_scores,
        // GH-425 B14-B16: Report 0 for scores not computed in simulated path.
        pmat_scores: PmatScores {
            rust_project_score: 0.0,
            tdg_score: 0.0,
            cuda_tdg_score: 0.0,
            brick_score: pmat_brick_score,
            grade: score_to_grade(pmat_brick_score),
        },
        // GH-425 B17: Real falsification from brick pass/fail.
        falsification: FalsificationSummary {
            total_points: n_bricks,
            passed: brick_passed,
            failed: brick_failed,
            blocked: 0,
        },
        // GH-425 B18: Status from brick pass/fail only — no hardcoded target.
        // #2730: and FAIL when no brick was measured at all.
        status: status.to_string(),
        ci_result: ci_result.to_string(),
    }
}

