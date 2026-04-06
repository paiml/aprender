//! `apr serve loadtest` / `apr serve bench` — LLM load testing via jugar-probar (PMAT-196).
//!
//! Sends concurrent chat completion requests to a running `apr serve` instance
//! and measures TTFT, TPOT, P50/P95/P99 latency, and throughput.
//! Feature-gated behind `llm-loadtest`.

use std::path::Path;
use std::time::Duration;

use colored::Colorize;
use jugar_probar::llm::client::{ChatMessage, ChatRequest, LlmClient, Role};
use jugar_probar::llm::loadtest::{LoadTest, LoadTestConfig, LoadTestResult};

use crate::error::{CliError, Result};

/// Run a single load test against a running `apr serve` instance.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_loadtest(
    url: &str,
    concurrency: usize,
    duration_secs: u64,
    prompts_path: Option<&Path>,
    slo_ttft_p99_ms: Option<f64>,
    slo_tpot_p99_ms: Option<f64>,
    format: &str,
    save_baseline: bool,
) -> Result<()> {
    println!("{}", "=== APR Serve Load Test ===".cyan().bold());
    println!("Target:      {url}");
    println!("Concurrency: {concurrency}");
    println!("Duration:    {duration_secs}s");
    if let Some(ttft) = slo_ttft_p99_ms {
        println!("SLO TTFT P99: {ttft}ms");
    }
    if let Some(tpot) = slo_tpot_p99_ms {
        println!("SLO TPOT P99: {tpot}ms");
    }
    println!();

    let prompts = load_prompts(prompts_path)?;

    let config = LoadTestConfig {
        concurrency,
        duration: Duration::from_secs(duration_secs),
        prompts,
        runtime_name: "apr-serve".to_string(),
        slo_ttft_ms: slo_ttft_p99_ms,
        slo_tpot_ms: slo_tpot_p99_ms,
        ..LoadTestConfig::default()
    };

    let client = LlmClient::new(url, "default");
    let loadtest = LoadTest::new(client, config);

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| CliError::Aprender(format!("tokio runtime: {e}")))?;

    let result = rt
        .block_on(loadtest.run())
        .map_err(|e| CliError::Aprender(format!("loadtest failed: {e}")))?;

    match format {
        "json" => {
            let json = serde_json::to_string_pretty(&result)
                .map_err(|e| CliError::Aprender(format!("serialize: {e}")))?;
            println!("{json}");
        }
        _ => print_result_text(&result),
    }

    if save_baseline {
        save_baseline_file(&result)?;
    }

    // SLO enforcement: fail if thresholds exceeded
    check_slo(&result, slo_ttft_p99_ms, slo_tpot_p99_ms)
}

/// Run multi-run benchmark with warmup and regression detection.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_bench(
    url: &str,
    runs: usize,
    warmup: usize,
    duration_secs: u64,
    baseline_path: Option<&Path>,
    regression_threshold: f64,
    format: &str,
) -> Result<()> {
    println!("{}", "=== APR Serve Benchmark ===".cyan().bold());
    println!("Target:   {url}");
    println!("Runs:     {runs} ({warmup} warmup)");
    println!("Duration: {duration_secs}s per run");
    println!();

    let prompts = load_prompts(None)?;

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| CliError::Aprender(format!("tokio runtime: {e}")))?;

    let mut results: Vec<LoadTestResult> = Vec::new();

    for i in 0..(warmup + runs) {
        let is_warmup = i < warmup;
        let label = if is_warmup {
            format!("warmup {}", i + 1)
        } else {
            format!("run {}", i - warmup + 1)
        };
        eprint!("  {label}...");

        let config = LoadTestConfig {
            concurrency: 4,
            duration: Duration::from_secs(duration_secs),
            prompts: prompts.clone(),
            runtime_name: "apr-serve".to_string(),
            ..LoadTestConfig::default()
        };

        let client = LlmClient::new(url, "default");
        let loadtest = LoadTest::new(client, config);

        let result = rt
            .block_on(loadtest.run())
            .map_err(|e| CliError::Aprender(format!("{label} failed: {e}")))?;

        eprintln!(
            " P99={:.0}ms, tok/s={:.1}",
            result.latency_p99_ms, result.tokens_per_sec
        );

        if !is_warmup {
            results.push(result);
        }
    }

    if results.is_empty() {
        return Err(CliError::Aprender("no measurement runs completed".into()));
    }

    // Aggregate
    let avg_p99 = results.iter().map(|r| r.latency_p99_ms).sum::<f64>() / results.len() as f64;
    let avg_ttft = results.iter().map(|r| r.ttft_p99_ms).sum::<f64>() / results.len() as f64;
    let avg_tps = results.iter().map(|r| r.tokens_per_sec).sum::<f64>() / results.len() as f64;
    let avg_tpot = results.iter().map(|r| r.tpot_p99_ms).sum::<f64>() / results.len() as f64;

    println!();
    println!("{}", "Aggregate (measurement runs):".bold());
    println!("  Latency P99: {avg_p99:.1}ms");
    println!("  TTFT P99:    {avg_ttft:.1}ms");
    println!("  TPOT P99:    {avg_tpot:.1}ms");
    println!("  Throughput:  {avg_tps:.1} tok/s");

    // Regression detection
    if let Some(baseline_path) = baseline_path {
        let baseline = load_baseline(baseline_path)?;
        let regression_pct =
            ((avg_p99 - baseline.latency_p99_ms) / baseline.latency_p99_ms) * 100.0;

        if regression_pct > regression_threshold {
            println!(
                "\n{}",
                format!(
                    "REGRESSION: P99 latency regressed {regression_pct:.1}% (threshold: {regression_threshold}%)"
                )
                .red()
                .bold()
            );
            return Err(CliError::Aprender(format!(
                "P99 latency regression {regression_pct:.1}% exceeds {regression_threshold}% threshold"
            )));
        }
        println!(
            "\n{} P99 delta: {regression_pct:+.1}% (threshold: {regression_threshold}%)",
            "OK:".green().bold()
        );
    }

    if format == "json" {
        let json = serde_json::to_string_pretty(&results)
            .map_err(|e| CliError::Aprender(format!("serialize: {e}")))?;
        println!("{json}");
    }

    Ok(())
}

/// Load prompts from a JSONL file or use a default coding prompt.
fn load_prompts(path: Option<&Path>) -> Result<Vec<ChatRequest>> {
    if let Some(path) = path {
        let content = std::fs::read_to_string(path)
            .map_err(|e| CliError::Aprender(format!("read prompts: {e}")))?;
        let mut prompts = Vec::new();
        for line in content.lines().filter(|l| !l.trim().is_empty()) {
            let v: serde_json::Value = serde_json::from_str(line)
                .map_err(|e| CliError::Aprender(format!("parse prompt JSONL: {e}")))?;
            let text = v["prompt"]
                .as_str()
                .or_else(|| v["content"].as_str())
                .unwrap_or(line.trim())
                .to_string();
            prompts.push(ChatRequest {
                model: "default".into(),
                messages: vec![ChatMessage {
                    role: Role::User,
                    content: text,
                }],
                temperature: Some(0.0),
                max_tokens: Some(128),
                stream: None,
            });
        }
        if prompts.is_empty() {
            return Err(CliError::Aprender("prompt file is empty".into()));
        }
        Ok(prompts)
    } else {
        Ok(vec![ChatRequest {
            model: "default".into(),
            messages: vec![ChatMessage {
                role: Role::User,
                content: "What is 2+2? Answer with just the number.".into(),
            }],
            temperature: Some(0.0),
            max_tokens: Some(32),
            stream: None,
        }])
    }
}

fn print_result_text(r: &LoadTestResult) {
    println!("{}", "Results:".bold());
    println!(
        "  Requests: {} total, {} ok, {} failed ({:.1}% error rate)",
        r.total_requests,
        r.successful,
        r.failed,
        r.error_rate * 100.0
    );
    println!(
        "  Latency:  P50={:.0}ms  P95={:.0}ms  P99={:.0}ms",
        r.latency_p50_ms, r.latency_p95_ms, r.latency_p99_ms
    );
    println!(
        "  TTFT:     P50={:.0}ms  P99={:.0}ms",
        r.ttft_p50_ms, r.ttft_p99_ms
    );
    println!(
        "  TPOT:     P50={:.1}ms  P99={:.1}ms",
        r.tpot_p50_ms, r.tpot_p99_ms
    );
    println!(
        "  Throughput: {:.1} tok/s ({:.1} decode tok/s)",
        r.tokens_per_sec, r.decode_tok_per_sec
    );
    println!(
        "  Duration: {:.1}s  Concurrency: {}",
        r.elapsed_secs, r.concurrency
    );
}

fn check_slo(result: &LoadTestResult, slo_ttft: Option<f64>, slo_tpot: Option<f64>) -> Result<()> {
    let mut violations = Vec::new();

    if let Some(threshold) = slo_ttft {
        if result.ttft_p99_ms > threshold {
            violations.push(format!(
                "TTFT P99 {:.0}ms > {threshold}ms SLO",
                result.ttft_p99_ms
            ));
        }
    }

    if let Some(threshold) = slo_tpot {
        if result.tpot_p99_ms > threshold {
            violations.push(format!(
                "TPOT P99 {:.1}ms > {threshold}ms SLO",
                result.tpot_p99_ms
            ));
        }
    }

    if violations.is_empty() {
        if slo_ttft.is_some() || slo_tpot.is_some() {
            println!("\n{}", "SLO: PASS".green().bold());
        }
        Ok(())
    } else {
        for v in &violations {
            println!("\n{} {v}", "SLO VIOLATION:".red().bold());
        }
        Err(CliError::Aprender(format!(
            "SLO violations: {}",
            violations.join(", ")
        )))
    }
}

fn save_baseline_file(result: &LoadTestResult) -> Result<()> {
    let dir = dirs::home_dir()
        .unwrap_or_default()
        .join(".apr")
        .join("benchmarks");
    std::fs::create_dir_all(&dir)
        .map_err(|e| CliError::Aprender(format!("create benchmark dir: {e}")))?;
    let path = dir.join("baseline.json");
    let json = serde_json::to_string_pretty(result)
        .map_err(|e| CliError::Aprender(format!("serialize: {e}")))?;
    std::fs::write(&path, json).map_err(|e| CliError::Aprender(format!("write baseline: {e}")))?;
    println!(
        "\n{} Baseline saved to {}",
        "OK:".green().bold(),
        path.display()
    );
    Ok(())
}

fn load_baseline(path: &Path) -> Result<LoadTestResult> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| CliError::Aprender(format!("read baseline: {e}")))?;
    serde_json::from_str(&content).map_err(|e| CliError::Aprender(format!("parse baseline: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ═══ Contract: apr-serve-loadtest-v1 enforcement (PMAT-196) ═══

    fn test_result(ttft_p99: f64, tpot_p99: f64) -> LoadTestResult {
        let json = serde_json::json!({
            "total_requests": 100, "successful": 95, "failed": 5,
            "throughput_rps": 10.0, "latency_p50_ms": 200.0,
            "latency_p95_ms": 400.0, "latency_p99_ms": 500.0,
            "ttft_p50_ms": 100.0, "ttft_p99_ms": ttft_p99,
            "tpot_p50_ms": 20.0, "tpot_p99_ms": tpot_p99,
            "tokens_per_sec": 50.0, "timestamp": "2026-04-05T00:00:00Z",
            "runtime_name": "apr-serve", "elapsed_secs": 30.0, "concurrency": 4
        });
        serde_json::from_value(json).unwrap()
    }

    #[test]
    fn falsify_lt_001_slo_ttft_violation() {
        let result = test_result(2000.0, 30.0);
        assert!(
            check_slo(&result, Some(1000.0), None).is_err(),
            "FALSIFY-LT-001"
        );
    }

    #[test]
    fn falsify_lt_002_slo_passes_under_threshold() {
        let result = test_result(500.0, 30.0);
        assert!(
            check_slo(&result, Some(1000.0), Some(50.0)).is_ok(),
            "FALSIFY-LT-002"
        );
    }

    #[test]
    fn falsify_lt_003_no_slo_always_passes() {
        let result = test_result(9999.0, 9999.0);
        assert!(check_slo(&result, None, None).is_ok(), "FALSIFY-LT-003");
    }

    #[test]
    fn falsify_lt_004_default_prompts() {
        let prompts = load_prompts(None).unwrap();
        assert_eq!(prompts.len(), 1, "FALSIFY-LT-004: exactly 1 default prompt");
        assert_eq!(
            prompts[0].messages[0].role,
            Role::User,
            "FALSIFY-LT-004: role=User"
        );
    }

    #[test]
    fn falsify_lt_005_empty_prompts_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.jsonl");
        std::fs::write(&path, "").unwrap();
        assert!(load_prompts(Some(&path)).is_err(), "FALSIFY-LT-005");
    }

    #[test]
    fn falsify_lt_006_jsonl_prompts() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.jsonl");
        std::fs::write(
            &path,
            "{\"prompt\": \"What is Rust?\"}\n{\"prompt\": \"Hello\"}\n",
        )
        .unwrap();
        let prompts = load_prompts(Some(&path)).unwrap();
        assert_eq!(prompts.len(), 2, "FALSIFY-LT-006");
        assert!(prompts[0].messages[0].content.contains("Rust"));
    }

    #[test]
    fn falsify_lt_007_slo_tpot_violation() {
        let result = test_result(500.0, 100.0);
        assert!(
            check_slo(&result, None, Some(50.0)).is_err(),
            "FALSIFY-LT-007"
        );
    }

    #[test]
    fn falsify_lt_008_baseline_roundtrip() {
        let result = test_result(500.0, 30.0);
        let json = serde_json::to_string_pretty(&result).unwrap();
        let parsed: LoadTestResult = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.total_requests, 100, "FALSIFY-LT-008");
        assert!(
            (parsed.ttft_p99_ms - 500.0).abs() < 0.01,
            "FALSIFY-LT-008: ttft"
        );
    }
}
