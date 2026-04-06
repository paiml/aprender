#!/usr/bin/env -S deno run --allow-read --allow-write
/**
 * Regression Detection Tool for Benchmark Results
 *
 * Compares current benchmark results against a baseline and detects performance
 * regressions based on configurable thresholds.
 *
 * Features:
 * - Configurable regression thresholds (percentage slowdown)
 * - Statistical significance testing (Welch's t-test)
 * - Effect size analysis (Cohen's d)
 * - Multiple severity levels (warning, critical)
 * - CI/CD friendly (exit codes and JSON output)
 * - Detailed regression reports
 *
 * Usage:
 *   check_regression.ts --baseline baseline.json --current current.json [OPTIONS]
 *
 * Exit Codes:
 *   0 - No regressions detected
 *   1 - Warning-level regressions detected
 *   2 - Critical regressions detected
 *   3 - Error (file not found, invalid JSON, etc.)
 */

import {
  calculateCohenD,
  calculateMean,
  calculateSpeedup,
  welchTTest,
} from "./statistical_analysis.ts";

// Type definitions matching Rust BenchmarkReport structure
interface Statistics {
  mean_ms: number;
  median_ms: number;
  std_dev_ms: number;
  min_ms: number;
  max_ms: number;
  coefficient_of_variation: number;
  confidence_interval_95: [number, number];
}

interface BenchmarkResult {
  benchmark_name: string;
  benchmark_type: string;
  optimization_level: string;
  warmup_iterations: number;
  measured_iterations: number;
  raw_timings_ms: number[];
  statistics: Statistics;
}

interface BenchmarkReport {
  schema_version: string;
  metadata: {
    benchmark_suite: string;
    timestamp: string;
    git_commit: string;
    git_branch: string;
  };
  benchmarks: BenchmarkResult[];
  summary: {
    total_benchmarks: number;
    successful: number;
    failed: number;
  };
}

/**
 * Regression detection thresholds
 */
interface RegressionThresholds {
  /** Minimum percentage slowdown to report as warning (default: 5%) */
  warningThreshold: number;
  /** Minimum percentage slowdown to report as critical (default: 10%) */
  criticalThreshold: number;
  /** Minimum p-value for statistical significance (default: 0.05) */
  significanceLevel: number;
  /** Minimum effect size (Cohen's d) to consider meaningful (default: 0.2) */
  minEffectSize: number;
}

/**
 * Regression severity levels
 */
enum RegressionSeverity {
  None = "none",
  Warning = "warning",
  Critical = "critical",
}

/**
 * Detected regression
 */
interface Regression {
  benchmarkName: string;
  severity: RegressionSeverity;
  baselineMean: number;
  currentMean: number;
  slowdownPercent: number;
  speedupRatio: number;
  pValue: number;
  cohenD: number;
  statistically_significant: boolean;
}

/**
 * Regression detection result
 */
interface RegressionResult {
  hasRegressions: boolean;
  hasCritical: boolean;
  hasWarnings: boolean;
  regressions: Regression[];
  summary: {
    total_benchmarks: number;
    regressions_found: number;
    critical_count: number;
    warning_count: number;
  };
}

/**
 * Default thresholds
 */
const DEFAULT_THRESHOLDS: RegressionThresholds = {
  warningThreshold: 0.05, // 5% slowdown
  criticalThreshold: 0.10, // 10% slowdown
  significanceLevel: 0.05, // p < 0.05
  minEffectSize: 0.2, // Small effect size
};

/**
 * Load benchmark report from JSON file
 */
async function loadReport(path: string): Promise<BenchmarkReport> {
  try {
    const content = await Deno.readTextFile(path);
    return JSON.parse(content) as BenchmarkReport;
  } catch (error) {
    console.error(`Error loading report from ${path}:`, error);
    throw error;
  }
}

/**
 * Detect regressions by comparing baseline and current results
 */
function detectRegressions(
  baseline: BenchmarkReport,
  current: BenchmarkReport,
  thresholds: RegressionThresholds,
): RegressionResult {
  const regressions: Regression[] = [];

  // Create a map of baseline benchmarks for easy lookup
  const baselineMap = new Map<string, BenchmarkResult>();
  for (const bench of baseline.benchmarks) {
    baselineMap.set(bench.benchmark_name, bench);
  }

  // Compare each current benchmark against baseline
  for (const currentBench of current.benchmarks) {
    const baselineBench = baselineMap.get(currentBench.benchmark_name);

    if (!baselineBench) {
      console.warn(
        `Warning: Benchmark "${currentBench.benchmark_name}" not found in baseline`,
      );
      continue;
    }

    // Calculate comparison metrics
    const baselineMean = baselineBench.statistics.mean_ms;
    const currentMean = currentBench.statistics.mean_ms;

    const speedup = calculateSpeedup(
      baselineBench.raw_timings_ms,
      currentBench.raw_timings_ms,
    );

    const cohenD = calculateCohenD(
      baselineBench.raw_timings_ms,
      currentBench.raw_timings_ms,
    );

    const pValue = welchTTest(
      baselineBench.raw_timings_ms,
      currentBench.raw_timings_ms,
    );

    // Determine if this is a regression
    const slowdownPercent = 1 - speedup.speedupRatio; // Positive means slower
    const isStatSig = pValue < thresholds.significanceLevel &&
      Math.abs(cohenD) >= thresholds.minEffectSize;

    let severity = RegressionSeverity.None;

    if (slowdownPercent > 0) {
      // Performance got worse
      if (isStatSig) {
        if (slowdownPercent >= thresholds.criticalThreshold) {
          severity = RegressionSeverity.Critical;
        } else if (slowdownPercent >= thresholds.warningThreshold) {
          severity = RegressionSeverity.Warning;
        }
      }
    }

    if (severity !== RegressionSeverity.None) {
      regressions.push({
        benchmarkName: currentBench.benchmark_name,
        severity,
        baselineMean,
        currentMean,
        slowdownPercent,
        speedupRatio: speedup.speedupRatio,
        pValue,
        cohenD,
        statistically_significant: isStatSig,
      });
    }
  }

  // Generate summary
  const criticalCount = regressions.filter((r) =>
    r.severity === RegressionSeverity.Critical
  ).length;
  const warningCount = regressions.filter((r) =>
    r.severity === RegressionSeverity.Warning
  ).length;

  return {
    hasRegressions: regressions.length > 0,
    hasCritical: criticalCount > 0,
    hasWarnings: warningCount > 0,
    regressions,
    summary: {
      total_benchmarks: current.benchmarks.length,
      regressions_found: regressions.length,
      critical_count: criticalCount,
      warning_count: warningCount,
    },
  };
}

/**
 * Format regression report for console output
 */
function formatConsoleReport(
  result: RegressionResult,
  baseline: BenchmarkReport,
  current: BenchmarkReport,
): string {
  const lines: string[] = [];

  lines.push("=== Regression Detection Report ===");
  lines.push("");
  lines.push(`Baseline: ${baseline.metadata.git_commit} (${baseline.metadata.timestamp})`);
  lines.push(`Current:  ${current.metadata.git_commit} (${current.metadata.timestamp})`);
  lines.push("");

  lines.push("Summary:");
  lines.push(`  Total benchmarks: ${result.summary.total_benchmarks}`);
  lines.push(`  Regressions found: ${result.summary.regressions_found}`);
  lines.push(`    Critical: ${result.summary.critical_count}`);
  lines.push(`    Warnings: ${result.summary.warning_count}`);
  lines.push("");

  if (!result.hasRegressions) {
    lines.push("✅ No performance regressions detected!");
    return lines.join("\n");
  }

  // Group regressions by severity
  const critical = result.regressions.filter((r) =>
    r.severity === RegressionSeverity.Critical
  );
  const warnings = result.regressions.filter((r) =>
    r.severity === RegressionSeverity.Warning
  );

  // Critical regressions
  if (critical.length > 0) {
    lines.push("🔴 CRITICAL REGRESSIONS:");
    lines.push("");
    for (const reg of critical) {
      lines.push(`  ${reg.benchmarkName}:`);
      lines.push(`    Baseline:  ${reg.baselineMean.toFixed(3)} ms`);
      lines.push(`    Current:   ${reg.currentMean.toFixed(3)} ms`);
      lines.push(
        `    Slowdown:  ${(reg.slowdownPercent * 100).toFixed(1)}% slower`,
      );
      lines.push(`    Cohen's d: ${reg.cohenD.toFixed(3)}`);
      lines.push(`    P-value:   ${reg.pValue.toFixed(6)}`);
      lines.push("");
    }
  }

  // Warnings
  if (warnings.length > 0) {
    lines.push("⚠️  WARNINGS:");
    lines.push("");
    for (const reg of warnings) {
      lines.push(`  ${reg.benchmarkName}:`);
      lines.push(`    Baseline:  ${reg.baselineMean.toFixed(3)} ms`);
      lines.push(`    Current:   ${reg.currentMean.toFixed(3)} ms`);
      lines.push(
        `    Slowdown:  ${(reg.slowdownPercent * 100).toFixed(1)}% slower`,
      );
      lines.push(`    Cohen's d: ${reg.cohenD.toFixed(3)}`);
      lines.push(`    P-value:   ${reg.pValue.toFixed(6)}`);
      lines.push("");
    }
  }

  return lines.join("\n");
}

/**
 * Parse command line arguments
 */
interface Args {
  baseline: string;
  current: string;
  warningThreshold?: number;
  criticalThreshold?: number;
  significanceLevel?: number;
  minEffectSize?: number;
  outputJson?: string;
  quiet?: boolean;
}

function parseArgs(args: string[]): Args {
  const result: Partial<Args> = {};

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];

    switch (arg) {
      case "--baseline":
      case "-b":
        result.baseline = args[++i];
        break;
      case "--current":
      case "-c":
        result.current = args[++i];
        break;
      case "--warning-threshold":
        result.warningThreshold = parseFloat(args[++i]);
        break;
      case "--critical-threshold":
        result.criticalThreshold = parseFloat(args[++i]);
        break;
      case "--significance-level":
        result.significanceLevel = parseFloat(args[++i]);
        break;
      case "--min-effect-size":
        result.minEffectSize = parseFloat(args[++i]);
        break;
      case "--output-json":
      case "-o":
        result.outputJson = args[++i];
        break;
      case "--quiet":
      case "-q":
        result.quiet = true;
        break;
      case "--help":
      case "-h":
        printHelp();
        Deno.exit(0);
        break;
      default:
        console.error(`Unknown argument: ${arg}`);
        Deno.exit(3);
    }
  }

  if (!result.baseline || !result.current) {
    console.error("Error: --baseline and --current are required");
    printHelp();
    Deno.exit(3);
  }

  return result as Args;
}

function printHelp() {
  console.log("Usage: check_regression.ts --baseline <file> --current <file> [OPTIONS]");
  console.log("");
  console.log("Required:");
  console.log("  --baseline, -b <file>      Baseline benchmark results (JSON)");
  console.log("  --current, -c <file>       Current benchmark results (JSON)");
  console.log("");
  console.log("Optional:");
  console.log("  --warning-threshold <n>    Warning threshold (default: 0.05 = 5%)");
  console.log("  --critical-threshold <n>   Critical threshold (default: 0.10 = 10%)");
  console.log("  --significance-level <n>   P-value threshold (default: 0.05)");
  console.log("  --min-effect-size <n>      Min Cohen's d (default: 0.2)");
  console.log("  --output-json, -o <file>   Write JSON report to file");
  console.log("  --quiet, -q                Suppress console output");
  console.log("  --help, -h                 Show this help");
  console.log("");
  console.log("Exit Codes:");
  console.log("  0 - No regressions detected");
  console.log("  1 - Warning-level regressions detected");
  console.log("  2 - Critical regressions detected");
  console.log("  3 - Error (file not found, invalid arguments, etc.)");
  console.log("");
  console.log("Examples:");
  console.log("  # Basic usage");
  console.log("  check_regression.ts -b baseline.json -c current.json");
  console.log("");
  console.log("  # Custom thresholds (15% warning, 25% critical)");
  console.log("  check_regression.ts -b baseline.json -c current.json \\");
  console.log("    --warning-threshold 0.15 --critical-threshold 0.25");
  console.log("");
  console.log("  # CI/CD mode with JSON output");
  console.log("  check_regression.ts -b baseline.json -c current.json \\");
  console.log("    --output-json regression_report.json --quiet");
}

/**
 * Main entry point
 */
async function main() {
  try {
    const args = parseArgs(Deno.args);

    // Load benchmark reports
    const baseline = await loadReport(args.baseline);
    const current = await loadReport(args.current);

    // Build thresholds
    const thresholds: RegressionThresholds = {
      warningThreshold: args.warningThreshold ?? DEFAULT_THRESHOLDS.warningThreshold,
      criticalThreshold: args.criticalThreshold ?? DEFAULT_THRESHOLDS.criticalThreshold,
      significanceLevel: args.significanceLevel ?? DEFAULT_THRESHOLDS.significanceLevel,
      minEffectSize: args.minEffectSize ?? DEFAULT_THRESHOLDS.minEffectSize,
    };

    // Detect regressions
    const result = detectRegressions(baseline, current, thresholds);

    // Output console report (unless quiet)
    if (!args.quiet) {
      const report = formatConsoleReport(result, baseline, current);
      console.log(report);
    }

    // Write JSON report if requested
    if (args.outputJson) {
      const jsonReport = {
        baseline: {
          commit: baseline.metadata.git_commit,
          timestamp: baseline.metadata.timestamp,
        },
        current: {
          commit: current.metadata.git_commit,
          timestamp: current.metadata.timestamp,
        },
        thresholds,
        result,
      };

      await Deno.writeTextFile(
        args.outputJson,
        JSON.stringify(jsonReport, null, 2),
      );

      if (!args.quiet) {
        console.log(`\nJSON report written to: ${args.outputJson}`);
      }
    }

    // Exit with appropriate code
    if (result.hasCritical) {
      Deno.exit(2);
    } else if (result.hasWarnings) {
      Deno.exit(1);
    } else {
      Deno.exit(0);
    }
  } catch (error) {
    console.error("Fatal error:", error);
    Deno.exit(3);
  }
}

// Run if this is the main module
if (import.meta.main) {
  main();
}

// Export for library use
export {
  detectRegressions,
  formatConsoleReport,
  type Regression,
  type RegressionResult,
  RegressionSeverity,
  type RegressionThresholds,
};
