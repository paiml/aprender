#!/usr/bin/env -S deno run --allow-read --allow-write
/**
 * Performance Dashboard Generator
 *
 * Generates an interactive HTML dashboard for visualizing benchmark trends.
 * Features:
 * - Line charts showing performance over time
 * - Bar charts comparing benchmarks
 * - Regression indicators
 * - Statistical summaries
 *
 * Usage:
 *   generate_dashboard.ts --input benchmarks/results/current.json --output dashboard.html
 *   generate_dashboard.ts --history benchmarks/history/ --output trend-dashboard.html
 */

// Type definitions
interface BenchmarkResult {
  benchmark_name: string;
  benchmark_type: string;
  optimization_level: string;
  statistics: {
    mean_ms: number;
    median_ms: number;
    std_dev_ms: number;
    coefficient_of_variation: number;
    confidence_interval_95: [number, number];
  };
}

interface BenchmarkReport {
  metadata: {
    benchmark_suite: string;
    timestamp: string;
    git_commit: string;
    git_branch: string;
  };
  benchmarks: BenchmarkResult[];
}

/**
 * Generate HTML dashboard
 */
function generateDashboard(report: BenchmarkReport): string {
  const benchmarkNames = report.benchmarks.map((b) => b.benchmark_name);
  const means = report.benchmarks.map((b) => b.statistics.mean_ms);
  const stdDevs = report.benchmarks.map((b) => b.statistics.std_dev_ms);
  const cvs = report.benchmarks.map((b) =>
    b.statistics.coefficient_of_variation * 100
  );

  return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Benchmark Dashboard - ${report.metadata.benchmark_suite}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: #0d1117;
            color: #c9d1d9;
            padding: 20px;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
        }

        header {
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 1px solid #30363d;
        }

        h1 {
            color: #58a6ff;
            margin-bottom: 10px;
        }

        .metadata {
            color: #8b949e;
            font-size: 14px;
        }

        .metadata span {
            margin-right: 20px;
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }

        .card {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 20px;
        }

        .card h2 {
            color: #58a6ff;
            font-size: 18px;
            margin-bottom: 15px;
        }

        .chart-container {
            position: relative;
            height: 300px;
        }

        .stats-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }

        .stats-table th {
            text-align: left;
            padding: 8px;
            border-bottom: 1px solid #30363d;
            color: #8b949e;
            font-weight: 600;
        }

        .stats-table td {
            padding: 8px;
            border-bottom: 1px solid #21262d;
        }

        .stats-table tr:hover {
            background: #0d1117;
        }

        .cv-good {
            color: #3fb950;
        }

        .cv-warning {
            color: #d29922;
        }

        .cv-bad {
            color: #f85149;
        }

        footer {
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #30363d;
            text-align: center;
            color: #8b949e;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📊 Benchmark Dashboard</h1>
            <div class="metadata">
                <span><strong>Suite:</strong> ${report.metadata.benchmark_suite}</span>
                <span><strong>Timestamp:</strong> ${report.metadata.timestamp}</span>
                <span><strong>Commit:</strong> <code>${report.metadata.git_commit}</code></span>
                <span><strong>Branch:</strong> ${report.metadata.git_branch}</span>
            </div>
        </header>

        <div class="grid">
            <div class="card">
                <h2>⏱️ Mean Execution Time</h2>
                <div class="chart-container">
                    <canvas id="meanChart"></canvas>
                </div>
            </div>

            <div class="card">
                <h2>📊 Coefficient of Variation</h2>
                <div class="chart-container">
                    <canvas id="cvChart"></canvas>
                </div>
            </div>
        </div>

        <div class="grid">
            <div class="card" style="grid-column: 1 / -1;">
                <h2>📈 Detailed Statistics</h2>
                <table class="stats-table">
                    <thead>
                        <tr>
                            <th>Benchmark</th>
                            <th>Mean (ms)</th>
                            <th>Median (ms)</th>
                            <th>Std Dev</th>
                            <th>CV</th>
                            <th>CI 95%</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${
    report.benchmarks.map((b) => {
      const cv = b.statistics.coefficient_of_variation * 100;
      const cvClass = cv < 5 ? "cv-good" : cv < 10 ? "cv-warning" : "cv-bad";

      return `
                        <tr>
                            <td><strong>${b.benchmark_name}</strong></td>
                            <td>${b.statistics.mean_ms.toFixed(3)}</td>
                            <td>${b.statistics.median_ms.toFixed(3)}</td>
                            <td>${b.statistics.std_dev_ms.toFixed(3)}</td>
                            <td class="${cvClass}">${cv.toFixed(2)}%</td>
                            <td>[${b.statistics.confidence_interval_95[0].toFixed(2)}, ${
        b.statistics.confidence_interval_95[1].toFixed(2)
      }]</td>
                        </tr>`;
    }).join("")
  }
                    </tbody>
                </table>
            </div>
        </div>

        <footer>
            Generated: ${new Date().toISOString()} | Certeza Benchmark Dashboard v1.0
        </footer>
    </div>

    <script>
        // Chart.js configuration
        const chartColors = {
            primary: '#58a6ff',
            success: '#3fb950',
            warning: '#d29922',
            danger: '#f85149',
            grid: '#30363d',
            text: '#c9d1d9'
        };

        const chartDefaults = {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: {
                        color: chartColors.grid
                    },
                    ticks: {
                        color: chartColors.text
                    }
                },
                x: {
                    grid: {
                        color: chartColors.grid
                    },
                    ticks: {
                        color: chartColors.text,
                        maxRotation: 45,
                        minRotation: 45
                    }
                }
            }
        };

        // Mean chart
        new Chart(document.getElementById('meanChart'), {
            type: 'bar',
            data: {
                labels: ${JSON.stringify(benchmarkNames)},
                datasets: [{
                    label: 'Mean (ms)',
                    data: ${JSON.stringify(means)},
                    backgroundColor: chartColors.primary,
                    borderColor: chartColors.primary,
                    borderWidth: 1
                }]
            },
            options: {
                ...chartDefaults,
                scales: {
                    ...chartDefaults.scales,
                    y: {
                        ...chartDefaults.scales.y,
                        title: {
                            display: true,
                            text: 'Time (ms)',
                            color: chartColors.text
                        }
                    }
                }
            }
        });

        // CV chart
        const cvData = ${JSON.stringify(cvs)};
        const cvColors = cvData.map(cv =>
            cv < 5 ? chartColors.success :
            cv < 10 ? chartColors.warning :
            chartColors.danger
        );

        new Chart(document.getElementById('cvChart'), {
            type: 'bar',
            data: {
                labels: ${JSON.stringify(benchmarkNames)},
                datasets: [{
                    label: 'CV (%)',
                    data: cvData,
                    backgroundColor: cvColors,
                    borderColor: cvColors,
                    borderWidth: 1
                }]
            },
            options: {
                ...chartDefaults,
                scales: {
                    ...chartDefaults.scales,
                    y: {
                        ...chartDefaults.scales.y,
                        title: {
                            display: true,
                            text: 'CV (%)',
                            color: chartColors.text
                        }
                    }
                }
            }
        });
    </script>
</body>
</html>`;
}

/**
 * Main entry point
 */
async function main() {
  const args = Deno.args;

  if (args.length < 2 || args.includes("--help")) {
    console.log("Usage: generate_dashboard.ts --input <file> --output <file>");
    console.log("");
    console.log("Options:");
    console.log("  --input <file>   Benchmark results JSON file");
    console.log("  --output <file>  Output HTML dashboard file");
    console.log("  --help           Show this help");
    console.log("");
    console.log("Example:");
    console.log(
      "  deno run --allow-read --allow-write generate_dashboard.ts \\",
    );
    console.log("    --input benchmarks/results/latest.json \\");
    console.log("    --output dashboard.html");
    Deno.exit(args.includes("--help") ? 0 : 1);
  }

  let inputPath = "";
  let outputPath = "";

  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--input") {
      inputPath = args[++i];
    } else if (args[i] === "--output") {
      outputPath = args[++i];
    }
  }

  if (!inputPath || !outputPath) {
    console.error("Error: --input and --output are required");
    Deno.exit(1);
  }

  console.log("=== Performance Dashboard Generator ===");
  console.log(`Input: ${inputPath}`);
  console.log(`Output: ${outputPath}`);
  console.log("");

  // Load benchmark report
  console.log("Loading benchmark report...");
  const content = await Deno.readTextFile(inputPath);
  const report = JSON.parse(content) as BenchmarkReport;
  console.log(`  ✓ Loaded ${report.benchmarks.length} benchmarks`);
  console.log("");

  // Generate dashboard
  console.log("Generating dashboard...");
  const html = generateDashboard(report);

  // Write to file
  await Deno.writeTextFile(outputPath, html);
  console.log(`  ✓ ${outputPath}`);
  console.log("");

  console.log("Dashboard generation complete!");
  console.log(`Open in browser: file://${Deno.cwd()}/${outputPath}`);
}

// Run if main module
if (import.meta.main) {
  main();
}

// Export for library use
export { generateDashboard };
