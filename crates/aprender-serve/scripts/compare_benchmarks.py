#!/usr/bin/env python3
"""
Benchmark Comparison Tool for Realizar vs llama.cpp

This script compares performance benchmarks between Realizar (pure Rust inference engine)
and llama.cpp (reference C++ implementation).

Usage:
    python scripts/compare_benchmarks.py --realizar benchmarks/realizar_results.json \\
                                         --llamacpp benchmarks/llamacpp_results.json \\
                                         --output comparison_report.md

Benchmark Format:
    Both benchmark results should be in JSON format with the following structure:
    {
        "model": "model_name",
        "config": {
            "vocab_size": 100,
            "hidden_dim": 32,
            "num_heads": 1
        },
        "benchmarks": {
            "forward_pass": {
                "seq_len_1": {"mean": 17.5, "std": 0.5, "unit": "µs"},
                "seq_len_5": {"mean": 42.0, "std": 1.2, "unit": "µs"}
            },
            "generation": {
                "tokens_5": {"mean": 1.54, "std": 0.03, "unit": "ms"},
                "tokens_10": {"mean": 3.12, "std": 0.05, "unit": "ms"}
            }
        }
    }
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass
class BenchmarkResult:
    """Single benchmark result with timing information"""
    mean: float
    std: float
    unit: str

    def to_nanoseconds(self) -> float:
        """Convert timing to nanoseconds for consistent comparison"""
        conversions = {
            "ns": 1.0,
            "µs": 1_000.0,
            "us": 1_000.0,
            "ms": 1_000_000.0,
            "s": 1_000_000_000.0,
        }
        return self.mean * conversions.get(self.unit, 1.0)

    def format(self) -> str:
        """Format with appropriate precision"""
        if self.unit in ["ns", "µs", "us"]:
            return f"{self.mean:.2f} {self.unit}"
        else:
            return f"{self.mean:.3f} {self.unit}"


@dataclass
class Comparison:
    """Comparison between two benchmark results"""
    realizar: BenchmarkResult
    llamacpp: Optional[BenchmarkResult]

    def speedup(self) -> Optional[float]:
        """Calculate speedup (positive = realizar faster, negative = slower)"""
        if self.llamacpp is None:
            return None

        realizar_ns = self.realizar.to_nanoseconds()
        llamacpp_ns = self.llamacpp.to_nanoseconds()

        if realizar_ns == 0 or llamacpp_ns == 0:
            return None

        # Return ratio: llama.cpp / realizar
        # > 1.0 means realizar is faster
        # < 1.0 means realizar is slower
        return llamacpp_ns / realizar_ns


def load_benchmark_file(path: Path) -> Dict:
    """Load benchmark results from JSON file"""
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {path}: {e}", file=sys.stderr)
        sys.exit(1)


def parse_benchmark_result(data: Dict) -> BenchmarkResult:
    """Parse benchmark result from JSON data"""
    return BenchmarkResult(
        mean=float(data["mean"]),
        std=float(data["std"]),
        unit=data["unit"]
    )


def generate_markdown_report(
    realizar_data: Dict,
    llamacpp_data: Optional[Dict],
    output_path: Optional[Path]
) -> str:
    """Generate markdown comparison report"""

    report = []
    report.append("# Benchmark Comparison: Realizar vs llama.cpp\n")
    report.append(f"**Model:** {realizar_data.get('model', 'Unknown')}\n")

    # Configuration
    report.append("\n## Configuration\n")
    config = realizar_data.get("config", {})
    report.append("```")
    report.append(f"Vocab Size:      {config.get('vocab_size', 'N/A')}")
    report.append(f"Hidden Dim:      {config.get('hidden_dim', 'N/A')}")
    report.append(f"Num Heads:       {config.get('num_heads', 'N/A')}")
    report.append(f"Num Layers:      {config.get('num_layers', 'N/A')}")
    report.append("```\n")

    # Benchmarks comparison
    report.append("\n## Performance Comparison\n")
    report.append("| Benchmark | Realizar | llama.cpp | Speedup | Winner |")
    report.append("|-----------|----------|-----------|---------|--------|")

    realizar_benches = realizar_data.get("benchmarks", {})
    llamacpp_benches = llamacpp_data.get("benchmarks", {}) if llamacpp_data else {}

    for category, tests in realizar_benches.items():
        for test_name, test_data in tests.items():
            realizar_result = parse_benchmark_result(test_data)
            llamacpp_result = None

            if llamacpp_benches and category in llamacpp_benches:
                if test_name in llamacpp_benches[category]:
                    llamacpp_result = parse_benchmark_result(
                        llamacpp_benches[category][test_name]
                    )

            comparison = Comparison(realizar_result, llamacpp_result)
            speedup = comparison.speedup()

            # Format row
            test_label = f"{category}/{test_name}"
            realizar_str = realizar_result.format()
            llamacpp_str = llamacpp_result.format() if llamacpp_result else "N/A"

            if speedup is None:
                speedup_str = "N/A"
                winner = "-"
            elif speedup > 1.05:  # 5% faster threshold
                speedup_str = f"**{speedup:.2f}x**"
                winner = "✅ **Realizar**"
            elif speedup < 0.95:  # 5% slower threshold
                speedup_str = f"{speedup:.2f}x"
                winner = "❌ llama.cpp"
            else:
                speedup_str = f"{speedup:.2f}x"
                winner = "≈ Tie"

            report.append(
                f"| {test_label} | {realizar_str} | {llamacpp_str} | "
                f"{speedup_str} | {winner} |"
            )

    # Summary
    report.append("\n## Summary\n")
    report.append("**Realizar** is a pure Rust ML inference engine built from scratch:")
    report.append("- 🦀 100% Rust, zero unsafe in public API")
    report.append("- ⚡ SIMD-accelerated via Trueno")
    report.append("- 🎯 EXTREME TDD methodology")
    report.append("- 📦 GGUF and SafeTensors support")
    report.append("- 🌐 Production-ready HTTP API\n")

    report.append("**Comparison Notes:**")
    report.append("- Speedup > 1.0 means Realizar is faster")
    report.append("- Speedup < 1.0 means llama.cpp is faster")
    report.append("- Values within ±5% considered equivalent\n")

    markdown = "\n".join(report)

    # Write to file if output path provided
    if output_path:
        output_path.write_text(markdown)
        print(f"Report written to: {output_path}")

    return markdown


def main():
    parser = argparse.ArgumentParser(
        description="Compare Realizar and llama.cpp benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--realizar",
        type=Path,
        required=True,
        help="Path to Realizar benchmark results (JSON)"
    )

    parser.add_argument(
        "--llamacpp",
        type=Path,
        help="Path to llama.cpp benchmark results (JSON, optional)"
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for markdown report (prints to stdout if not provided)"
    )

    args = parser.parse_args()

    # Load benchmark data
    realizar_data = load_benchmark_file(args.realizar)
    llamacpp_data = load_benchmark_file(args.llamacpp) if args.llamacpp else None

    # Generate report
    report = generate_markdown_report(realizar_data, llamacpp_data, args.output)

    # Print to stdout if no output file
    if not args.output:
        print(report)


if __name__ == "__main__":
    main()
