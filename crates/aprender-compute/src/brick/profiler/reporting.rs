//! Reporting and serialization methods for BrickProfiler.
//!
//! Extracted from mod.rs to keep file sizes manageable.
//! Contains: summary(), to_json(), write_json(), print_category_stats(),
//! tile_summary(), tile_stats_to_json().

use super::BrickProfiler;
use crate::brick::exec_graph::{BrickCategory, BrickId};

impl BrickProfiler {
    /// Print category breakdown to console.
    pub fn print_category_stats(&self) {
        let cats = self.category_stats();
        let total = self.total_ns;

        println!("╭─────────────────────────────────────────────────────────╮");
        println!("│            Category Breakdown (PAR-200)                 │");
        println!("├─────────────────────────────────────────────────────────┤");
        for (i, cat_stats) in cats.iter().enumerate() {
            let cat = BrickCategory::ALL[i];
            if cat_stats.count > 0 {
                println!(
                    "│ {:12} {:8.2}µs avg {:6.1}% [{:5} samples]        │",
                    cat.name(),
                    cat_stats.avg_us(),
                    cat_stats.percentage(total),
                    cat_stats.count
                );
            }
        }
        println!("╰─────────────────────────────────────────────────────────╯");
    }

    /// Generate a summary report.
    #[must_use]
    pub fn summary(&self) -> String {
        let mut report = String::new();
        report.push_str("=== Brick Profiler Summary (PAR-200) ===\n");
        report.push_str(&format!(
            "Total: {} tokens, {:.2}µs, {:.1} tok/s\n",
            self.total_tokens,
            self.total_ns as f64 / 1000.0,
            self.total_throughput()
        ));
        report.push_str("\nPer-Brick Breakdown:\n");

        // Collect all stats (known + dynamic)
        let mut all_stats: Vec<(&str, &crate::brick::exec_graph::BrickStats)> = Vec::new();

        // Add known bricks with non-zero counts
        for (i, stats) in self.brick_stats.iter().enumerate() {
            if stats.count > 0 {
                let brick_id = BrickId::ALL[i];
                all_stats.push((brick_id.name(), stats));
            }
        }

        // Add dynamic bricks
        for (name, stats) in &self.dynamic_stats {
            all_stats.push((name.as_str(), stats));
        }

        // Sort by total time descending
        all_stats.sort_by_key(|b| std::cmp::Reverse(b.1.total_ns));

        for (name, stats) in &all_stats {
            let pct = if self.total_ns > 0 {
                100.0 * stats.total_ns as f64 / self.total_ns as f64
            } else {
                0.0
            };
            report.push_str(&format!(
                "  {:20} {:8.2}µs avg ({:5.1}%) [{} samples]\n",
                name,
                stats.avg_us(),
                pct,
                stats.count
            ));
        }

        // Add category breakdown
        report.push_str("\nCategory Breakdown:\n");
        let cats = self.category_stats();
        for (i, cat_stats) in cats.iter().enumerate() {
            if cat_stats.count > 0 {
                let cat = BrickCategory::ALL[i];
                report.push_str(&format!(
                    "  {:12} {:8.2}µs avg ({:5.1}%)\n",
                    cat.name(),
                    cat_stats.avg_us(),
                    cat_stats.percentage(self.total_ns)
                ));
            }
        }

        report
    }

    /// Export profiling data as JSON for pmat metrics integration.
    ///
    /// Format compatible with `.pmat-metrics/trends/` structure:
    /// ```json
    /// {
    ///   "total_tokens": 1000,
    ///   "total_ns": 5000000,
    ///   "total_throughput": 200000.0,
    ///   "bricks": [
    ///     {
    ///       "name": "RmsNorm",
    ///       "count": 10,
    ///       "total_ns": 1000000,
    ///       "avg_us": 100.0,
    ///       "min_us": 90.0,
    ///       "max_us": 120.0,
    ///       "throughput": 10000.0,
    ///       "pct": 20.0
    ///     }
    ///   ]
    /// }
    /// ```
    #[must_use]
    pub fn to_json(&self) -> String {
        let mut bricks = Vec::new();

        // Collect all stats (known + dynamic)
        let mut all_stats: Vec<(&str, &crate::brick::exec_graph::BrickStats)> = Vec::new();

        // Add known bricks with non-zero counts
        for (i, stats) in self.brick_stats.iter().enumerate() {
            if stats.count > 0 {
                let brick_id = BrickId::ALL[i];
                all_stats.push((brick_id.name(), stats));
            }
        }

        // Add dynamic bricks
        for (name, stats) in &self.dynamic_stats {
            all_stats.push((name.as_str(), stats));
        }

        // Sort by total time descending
        all_stats.sort_by_key(|b| std::cmp::Reverse(b.1.total_ns));

        for (name, stats) in all_stats {
            let pct = if self.total_ns > 0 {
                100.0 * stats.total_ns as f64 / self.total_ns as f64
            } else {
                0.0
            };
            // PMAT-451: Include compression_ratio, throughput_gbps, and bottleneck
            let compression = stats.compression_ratio();
            let throughput_gbps = stats.throughput_gbps();
            let bottleneck = stats.get_bottleneck();
            bricks.push(format!(
                r#"{{"name":"{}","count":{},"total_ns":{},"avg_us":{:.2},"min_us":{:.2},"max_us":{:.2},"throughput":{:.1},"pct":{:.1},"total_bytes":{},"compression_ratio":{:.2},"throughput_gbps":{:.2},"bottleneck":"{}"}}"#,
                name,
                stats.count,
                stats.total_ns,
                stats.avg_us(),
                stats.min_us(),
                stats.max_us(),
                stats.throughput(),
                pct,
                stats.total_bytes,
                compression,
                throughput_gbps,
                bottleneck
            ));
        }

        format!(
            r#"{{"total_tokens":{},"total_ns":{},"total_throughput":{:.1},"bricks":[{}]}}"#,
            self.total_tokens,
            self.total_ns,
            self.total_throughput(),
            bricks.join(",")
        )
    }

    /// Write profiling data to a JSON file for pmat tracking.
    ///
    /// # Errors
    /// Returns error if file cannot be written.
    pub fn write_json(&self, path: &std::path::Path) -> std::io::Result<()> {
        std::fs::write(path, self.to_json())
    }

    /// Generate tile profiling summary report.
    ///
    /// # Example Output
    /// ```text
    /// === Tile Profiling Summary (TILING-SPEC-001) ===
    /// Level       Samples   Avg µs    GFLOP/s   AI      Elements
    /// Macro           128    1234.5     12.34  0.50    1048576
    /// Midi           2048      78.2     45.67  2.00      65536
    /// Micro         32768       4.9     89.12  4.00       4096
    /// ```
    #[must_use]
    pub fn tile_summary(&self) -> String {
        let mut report = String::new();
        report.push_str("=== Tile Profiling Summary (TILING-SPEC-001) ===\n");
        report.push_str("Level       Samples   Avg µs    GFLOP/s   AI      Elements\n");

        for stats in &self.tile_stats {
            if stats.count > 0 {
                report.push_str(&format!(
                    "{:8}  {:9}  {:8.1}  {:8.2}  {:4.2}  {:10}\n",
                    stats.level.name(),
                    stats.count,
                    stats.avg_us(),
                    stats.gflops(),
                    stats.arithmetic_intensity(),
                    stats.total_elements / stats.count.max(1)
                ));
            }
        }

        report
    }

    /// Export tile statistics as JSON.
    ///
    /// Compatible with pmat metrics integration.
    #[must_use]
    pub fn tile_stats_to_json(&self) -> String {
        let tiles: Vec<String> = self
            .tile_stats
            .iter()
            .filter(|s| s.count > 0)
            .map(|s| {
                format!(
                    r#"{{"level":"{}","count":{},"total_ns":{},"avg_us":{:.2},"min_us":{:.2},"max_us":{:.2},"gflops":{:.2},"arithmetic_intensity":{:.2},"total_elements":{},"total_flops":{}}}"#,
                    s.level.name(),
                    s.count,
                    s.total_ns,
                    s.avg_us(),
                    s.min_ns as f64 / 1000.0,
                    s.max_ns as f64 / 1000.0,
                    s.gflops(),
                    s.arithmetic_intensity(),
                    s.total_elements,
                    s.total_flops
                )
            })
            .collect();

        format!(
            r#"{{"tile_profiling_enabled":{},"tiles":[{}]}}"#,
            self.tile_profiling_enabled,
            tiles.join(",")
        )
    }
}
