
/// IMP-162d: Real-world MAD outlier detection with llama.cpp
#[test]
#[ignore = "Requires running llama.cpp server on port 8082"]
fn test_imp_162d_realworld_mad_outlier_detection() {
    // This test requires: llama-server -m model.gguf --host 127.0.0.1 --port 8082 -ngl 99
    let client = ModelHttpClient::with_timeout(60);
    let request = CompletionRequest {
        model: "default".to_string(),
        prompt: "Hello:".to_string(),
        max_tokens: 5,
        temperature: Some(0.0),
        stream: false,
    };

    // Collect 20 samples
    let mut latencies_ms = Vec::new();
    for _ in 0..20 {
        let start = std::time::Instant::now();
        if client
            .llamacpp_completion("http://127.0.0.1:8082", &request)
            .is_ok()
        {
            latencies_ms.push(start.elapsed().as_secs_f64() * 1000.0);
        }
    }

    if latencies_ms.len() < 10 {
        println!("IMP-162d: Not enough samples collected");
        return;
    }

    // Convert to throughput
    let throughputs: Vec<f64> = latencies_ms
        .iter()
        .map(|lat| if *lat > 0.0 { 5000.0 / lat } else { 0.0 }) // ~5 tokens
        .collect();

    let result = CleanedBenchmarkResult::clean(&throughputs);
    let comparison = MadVsStdComparison::compare(&throughputs, 3.0);

    println!("\nIMP-162d: Real-World MAD Outlier Detection (llama.cpp):");
    println!("  Samples collected: {}", throughputs.len());
    println!(
        "  Raw mean: {:.2} tok/s (CV={:.4})",
        result.raw_stats.mean_tps, result.raw_stats.cv
    );
    println!(
        "  Cleaned mean: {:.2} tok/s (CV={:.4})",
        result.cleaned_stats.mean_tps, result.cleaned_stats.cv
    );
    println!(
        "  Outliers detected: {}",
        result.outlier_result.outlier_count
    );
    if !result.outlier_result.outlier_values.is_empty() {
        println!(
            "  Outlier values: {:?}",
            result.outlier_result.outlier_values
        );
    }
    println!("  CV improvement: {:.2}%", result.cv_improvement_percent);
    println!(
        "  MAD vs Stddev robustness: {:.2}x",
        comparison.robustness_ratio
    );
    println!("  Cleaning beneficial: {}", result.cleaning_beneficial());
}

// =========================================================================
// IMP-163: Real-World A/B Latency Comparison (QA-012, EXTREME TDD)
// =========================================================================
// Per spec QA-012: Latency p99 < 2x p50 (no outliers)
// Compares latency distributions between Realizar and external servers.
// Run with: cargo test test_imp_163 --lib --features bench-http

/// IMP-163a: A/B latency comparison result
#[derive(Debug, Clone)]
pub struct ABLatencyComparison {
    /// Server A name (e.g., "Realizar")
    pub server_a_name: String,
    /// Server B name (e.g., "llama.cpp")
    pub server_b_name: String,
    /// Server A latency percentiles
    pub server_a_latency: LatencyPercentiles,
    /// Server B latency percentiles
    pub server_b_latency: LatencyPercentiles,
    /// Latency ratio (A/B) at p50
    pub p50_ratio: f64,
    /// Latency ratio (A/B) at p99
    pub p99_ratio: f64,
    /// Whether both servers meet QA-012: p99 < 2x p50
    pub both_meet_qa012: bool,
    /// Winner at p50 ("A", "B", or "tie")
    pub p50_winner: String,
    /// Winner at p99
    pub p99_winner: String,
}

impl ABLatencyComparison {
    pub fn compare(
        server_a_name: &str,
        server_a_samples: &[f64],
        server_b_name: &str,
        server_b_samples: &[f64],
    ) -> Self {
        let server_a_latency = LatencyPercentiles::from_samples(server_a_samples);
        let server_b_latency = LatencyPercentiles::from_samples(server_b_samples);

        let p50_ratio = if server_b_latency.p50_ms > 0.0 {
            server_a_latency.p50_ms / server_b_latency.p50_ms
        } else {
            1.0
        };

        let p99_ratio = if server_b_latency.p99_ms > 0.0 {
            server_a_latency.p99_ms / server_b_latency.p99_ms
        } else {
            1.0
        };

        // QA-012: p99 < 2x p50
        let a_meets = server_a_latency.p99_ms < 2.0 * server_a_latency.p50_ms;
        let b_meets = server_b_latency.p99_ms < 2.0 * server_b_latency.p50_ms;

        let p50_winner = if (p50_ratio - 1.0).abs() < 0.05 {
            "tie".to_string()
        } else if p50_ratio < 1.0 {
            server_a_name.to_string()
        } else {
            server_b_name.to_string()
        };

        let p99_winner = if (p99_ratio - 1.0).abs() < 0.05 {
            "tie".to_string()
        } else if p99_ratio < 1.0 {
            server_a_name.to_string()
        } else {
            server_b_name.to_string()
        };

        Self {
            server_a_name: server_a_name.to_string(),
            server_b_name: server_b_name.to_string(),
            server_a_latency,
            server_b_latency,
            p50_ratio,
            p99_ratio,
            both_meet_qa012: a_meets && b_meets,
            p50_winner,
            p99_winner,
        }
    }

    /// Generate summary report
    pub fn summary(&self) -> String {
        format!(
            "{} vs {}: p50 ratio={:.2}x (winner: {}), p99 ratio={:.2}x (winner: {}), QA-012: {}",
            self.server_a_name,
            self.server_b_name,
            self.p50_ratio,
            self.p50_winner,
            self.p99_ratio,
            self.p99_winner,
            if self.both_meet_qa012 { "PASS" } else { "FAIL" }
        )
    }
}

/// IMP-163a: Test A/B latency comparison structure
#[test]
fn test_imp_163a_ab_latency_comparison() {
    // test Realizar latencies (slower)
    let realizar_latencies = vec![
        520.0, 530.0, 510.0, 525.0, 515.0, 540.0, 505.0, 535.0, 520.0, 528.0,
    ];
    // test llama.cpp latencies (faster)
    let llamacpp_latencies = vec![
        160.0, 165.0, 158.0, 162.0, 155.0, 170.0, 152.0, 168.0, 161.0, 164.0,
    ];

    let comparison = ABLatencyComparison::compare(
        "Realizar",
        &realizar_latencies,
        "llama.cpp",
        &llamacpp_latencies,
    );

    // IMP-163a: p50 ratio should be ~3.2x (520/162)
    assert!(
        comparison.p50_ratio > 3.0 && comparison.p50_ratio < 3.5,
        "IMP-163a: p50 ratio should be ~3.2x, got {:.2}x",
        comparison.p50_ratio
    );

    // IMP-163a: llama.cpp should win on p50
    assert_eq!(
        comparison.p50_winner, "llama.cpp",
        "IMP-163a: llama.cpp should win on p50"
    );

    // IMP-163a: Both should meet QA-012 (no outliers in these samples)
    assert!(
        comparison.both_meet_qa012,
        "IMP-163a: Both servers should meet QA-012 (p99 < 2x p50)"
    );

    println!("\nIMP-163a: A/B Latency Comparison:");
    println!("  {}", comparison.summary());
    println!(
        "  Realizar p50: {:.1}ms, p99: {:.1}ms",
        comparison.server_a_latency.p50_ms, comparison.server_a_latency.p99_ms
    );
    println!(
        "  llama.cpp p50: {:.1}ms, p99: {:.1}ms",
        comparison.server_b_latency.p50_ms, comparison.server_b_latency.p99_ms
    );
}

/// IMP-163b: QA-012 violation detection
#[derive(Debug, Clone)]
pub struct QA012Violation {
    /// Server name
    pub server_name: String,
    /// p50 latency
    pub p50_ms: f64,
    /// p99 latency
    pub p99_ms: f64,
    /// Actual ratio (p99/p50)
    pub tail_ratio: f64,
    /// Whether violation occurred (p99 >= 2x p50)
    pub violated: bool,
    /// Severity: "none", "minor" (2-3x), "major" (3-5x), "severe" (>5x)
    pub severity: String,
}

impl QA012Violation {
    pub fn check(server_name: &str, latencies: &[f64]) -> Self {
        let percentiles = LatencyPercentiles::from_samples(latencies);
        let tail_ratio = percentiles.tail_latency_ratio();
        let violated = tail_ratio >= 2.0;

        let severity = if tail_ratio < 2.0 {
            "none"
        } else if tail_ratio < 3.0 {
            "minor"
        } else if tail_ratio < 5.0 {
            "major"
        } else {
            "severe"
        };

        Self {
            server_name: server_name.to_string(),
            p50_ms: percentiles.p50_ms,
            p99_ms: percentiles.p99_ms,
            tail_ratio,
            violated,
            severity: severity.to_string(),
        }
    }
}

/// IMP-163b: Test QA-012 violation detection
#[test]
fn test_imp_163b_qa012_violation_detection() {
    // Good: p99 close to p50
    let good_latencies = vec![
        100.0, 102.0, 98.0, 101.0, 99.0, 105.0, 95.0, 103.0, 100.0, 101.0,
    ];
    let good_check = QA012Violation::check("GoodServer", &good_latencies);
    assert!(
        !good_check.violated,
        "IMP-163b: Good server should not violate QA-012"
    );
    assert_eq!(good_check.severity, "none");

    // Bad: p99 >> p50 (outliers)
    let bad_latencies = vec![
        100.0, 102.0, 98.0, 101.0, 99.0, 100.0, 98.0, 500.0, 600.0, 700.0,
    ];
    let bad_check = QA012Violation::check("BadServer", &bad_latencies);
    assert!(
        bad_check.violated,
        "IMP-163b: Bad server should violate QA-012"
    );
    assert!(
        bad_check.severity == "major" || bad_check.severity == "severe",
        "IMP-163b: Severity should be major or severe, got {}",
        bad_check.severity
    );

    println!("\nIMP-163b: QA-012 Violation Detection:");
    println!(
        "  Good server: p50={:.1}ms, p99={:.1}ms, ratio={:.2}x, violated={}, severity={}",
        good_check.p50_ms,
        good_check.p99_ms,
        good_check.tail_ratio,
        good_check.violated,
        good_check.severity
    );
    println!(
        "  Bad server: p50={:.1}ms, p99={:.1}ms, ratio={:.2}x, violated={}, severity={}",
        bad_check.p50_ms,
        bad_check.p99_ms,
        bad_check.tail_ratio,
        bad_check.violated,
        bad_check.severity
    );
}

/// IMP-163c: Multi-server latency leaderboard
#[derive(Debug, Clone)]
pub struct LatencyLeaderboard {
    /// Server entries sorted by p50 latency
    pub entries: Vec<LeaderboardEntry>,
}

#[derive(Debug, Clone)]
pub struct LeaderboardEntry {
    pub rank: usize,
    pub server_name: String,
    pub p50_ms: f64,
    pub p99_ms: f64,
    pub tail_ratio: f64,
    pub meets_qa012: bool,
}

impl LatencyLeaderboard {
    pub fn from_results(results: Vec<(&str, Vec<f64>)>) -> Self {
        let mut entries: Vec<LeaderboardEntry> = results
            .into_iter()
            .map(|(name, samples)| {
                let percentiles = LatencyPercentiles::from_samples(&samples);
                let tail_ratio = percentiles.tail_latency_ratio();
                LeaderboardEntry {
                    rank: 0, // Set later
                    server_name: name.to_string(),
                    p50_ms: percentiles.p50_ms,
                    p99_ms: percentiles.p99_ms,
                    tail_ratio,
                    meets_qa012: tail_ratio < 2.0,
                }
            })
            .collect();

        // Sort by p50 (lower is better)
        entries.sort_by(|a, b| {
            a.p50_ms
                .partial_cmp(&b.p50_ms)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Assign ranks
        for (i, entry) in entries.iter_mut().enumerate() {
            entry.rank = i + 1;
        }

        Self { entries }
    }

    /// Get winner (rank 1)
    pub fn winner(&self) -> Option<&LeaderboardEntry> {
        self.entries.first()
    }
}

/// IMP-163c: Test multi-server leaderboard
#[test]
fn test_imp_163c_latency_leaderboard() {
    let results = vec![
        ("Realizar", vec![520.0, 530.0, 510.0, 525.0, 515.0]),
        ("llama.cpp", vec![160.0, 165.0, 158.0, 162.0, 155.0]),
        ("Ollama", vec![280.0, 290.0, 275.0, 285.0, 270.0]),
    ];

    let leaderboard = LatencyLeaderboard::from_results(results);

    // IMP-163c: llama.cpp should be rank 1
    assert_eq!(
        leaderboard.winner().map(|e| e.server_name.as_str()),
        Some("llama.cpp"),
        "IMP-163c: llama.cpp should be rank 1"
    );

    // IMP-163c: Realizar should be rank 3
    let realizar_rank = leaderboard
        .entries
        .iter()
        .find(|e| e.server_name == "Realizar")
        .map(|e| e.rank);
    assert_eq!(
        realizar_rank,
        Some(3),
        "IMP-163c: Realizar should be rank 3"
    );

    println!("\nIMP-163c: Latency Leaderboard:");
    for entry in &leaderboard.entries {
        println!(
            "  #{}: {} - p50={:.1}ms, p99={:.1}ms, QA-012={}",
            entry.rank,
            entry.server_name,
            entry.p50_ms,
            entry.p99_ms,
            if entry.meets_qa012 { "PASS" } else { "FAIL" }
        );
    }
}
