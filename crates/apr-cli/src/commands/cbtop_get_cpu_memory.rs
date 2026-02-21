
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
    let score = if gap <= 1.0 {
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
    let weights = [1.5, 6.0, 1.0, 10.0, 3.5, 1.5, 12.2];
    let weighted_sum: f64 = brick_scores
        .iter()
        .zip(weights.iter())
        .map(|(b, w)| b.score as f64 * w)
        .sum();
    let total_weight: f64 = weights.iter().sum();
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    { (weighted_sum / total_weight) as u32 }
}

/// Generate headless report from pipeline state (simulated data)
fn generate_headless_report_simulated(
    model_name: &str,
    pipeline: &PipelineState,
    _config: &CbtopConfig,
) -> HeadlessReport {
    use std::time::{SystemTime, UNIX_EPOCH};

    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).map_or_else(
        |_| "unknown".to_string(),
        |d| {
            let secs = d.as_secs();
            format!(
                "2026-01-11T{:02}:{:02}:{:02}Z",
                (secs / 3600) % 24,
                (secs / 60) % 60,
                secs % 60
            )
        },
    );

    let brick_scores: Vec<BrickScore> = pipeline.bricks.iter().map(score_brick).collect();

    let all_samples: Vec<f64> = pipeline
        .bricks
        .iter()
        .flat_map(|b| b.samples.iter().copied())
        .collect();
    let cv_percent = cv_percent_from_samples(&all_samples);
    let (p50, p99) = pipeline.bricks.first().map_or((0.0, 0.0), percentiles_from_brick);

    let all_pass = brick_scores.iter().all(|b| b.gap_factor <= 1.0);
    let pmat_brick_score = weighted_brick_score(&brick_scores);

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
        pmat_scores: PmatScores {
            rust_project_score: 173.9,
            tdg_score: 98.1,
            cuda_tdg_score: 95.2,
            brick_score: pmat_brick_score,
            grade: score_to_grade(pmat_brick_score),
        },
        falsification: FalsificationSummary {
            total_points: 137,
            passed: 137,
            failed: 0,
            blocked: 0,
        },
        status: if all_pass { "PASS" } else { "FAIL" }.to_string(),
        ci_result: if all_pass && pipeline.current_tok_s >= pipeline.target_tok_s {
            "green"
        } else {
            "red"
        }
        .to_string(),
    }
}

