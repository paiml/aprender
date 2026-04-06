#!/usr/bin/env python3
"""
Compare Trueno (Rust) vs NumPy vs PyTorch benchmark results

This script:
1. Parses Criterion benchmark results (JSON) from Trueno
2. Loads Python benchmark results (NumPy/PyTorch)
3. Compares performance across all three frameworks
4. Generates markdown tables and analysis

Usage:
    python benchmarks/compare_results.py

Output:
    - benchmarks/comparison_report.md (Markdown report)
    - benchmarks/comparison_summary.json (JSON data)
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple


class BenchmarkComparator:
    """Compare benchmark results across Trueno, NumPy, and PyTorch"""

    def __init__(self):
        self.trueno_results = {}
        self.python_results = {}
        self.comparison = {}

    def load_criterion_results(self, criterion_dir: str = "target/criterion"):
        """Load Criterion benchmark results from Trueno"""
        print("Loading Trueno (Criterion) benchmark results...")

        # Criterion stores results in: target/criterion/<group_name>/<benchmark_name>/base/estimates.json
        criterion_path = Path(criterion_dir)

        if not criterion_path.exists():
            print(f"⚠️  Criterion results not found at {criterion_dir}")
            print("   Run: cargo bench --all-features")
            return

        # Find all estimates.json files
        for estimates_file in criterion_path.rglob("estimates.json"):
            # Parse path: <group>/<bench_name>/<backend>/base/estimates.json
            parts = estimates_file.parts
            if len(parts) < 4:
                continue

            # Extract group, backend, and size from path structure
            # Path structure: target/criterion/<operation>/<backend>/<size>/base/estimates.json
            group_idx = parts.index("criterion") + 1
            if group_idx + 2 >= len(parts):
                continue

            group = parts[group_idx]  # e.g., "add", "dot", "relu"
            backend = parts[group_idx + 1]  # e.g., "Scalar", "SSE2", "AVX2", "AVX512"
            size_str = parts[group_idx + 2]  # e.g., "100", "1000", "10000"

            try:
                with open(estimates_file, "r") as f:
                    data = json.load(f)
                    # Criterion stores mean in nanoseconds
                    mean_ns = data["mean"]["point_estimate"]
                    std_ns = data["std_dev"]["point_estimate"]

                    # Store result
                    if group not in self.trueno_results:
                        self.trueno_results[group] = {}
                    if size_str not in self.trueno_results[group]:
                        self.trueno_results[group][size_str] = {}

                    self.trueno_results[group][size_str][backend] = {
                        "mean_ns": mean_ns,
                        "std_ns": std_ns,
                    }
            except Exception as e:
                print(f"⚠️  Failed to parse {estimates_file}: {e}")
                continue

        print(f"✅ Loaded {len(self.trueno_results)} Trueno operation groups")

    def load_python_results(self, python_file: str = "benchmarks/python_results.json"):
        """Load Python (NumPy/PyTorch) benchmark results"""
        print(f"Loading Python benchmark results from {python_file}...")

        if not Path(python_file).exists():
            print(f"⚠️  Python results not found at {python_file}")
            print("   Run: python benchmarks/python_comparison.py")
            return

        with open(python_file, "r") as f:
            self.python_results = json.load(f)

        print(f"✅ Loaded Python results for {len(self.python_results.get('numpy', {}))} operations")

    def _find_best_backend(self, backends_data: Dict) -> Tuple[str, float]:
        """Find the backend with the lowest mean time."""
        best_backend = None
        best_time = float("inf")
        for backend, data in backends_data.items():
            if data["mean_ns"] < best_time:
                best_time = data["mean_ns"]
                best_backend = backend
        return best_backend, best_time

    def _get_python_time(self, framework: str, op_name: str, size: str):
        """Get the mean_ns for a given Python framework, operation, and size."""
        framework_data = self.python_results.get(framework, {})
        if op_name in framework_data and size in framework_data[op_name]:
            return framework_data[op_name][size]["mean_ns"]
        return None

    def _build_comparison_entry(self, best_backend, best_time, numpy_time, pytorch_time) -> Dict:
        """Build a single comparison entry dict."""
        return {
            "trueno_backend": best_backend,
            "trueno_mean_ns": best_time,
            "numpy_mean_ns": numpy_time,
            "pytorch_mean_ns": pytorch_time,
            "trueno_vs_numpy": best_time / numpy_time if numpy_time else None,
            "trueno_vs_pytorch": best_time / pytorch_time if pytorch_time else None,
        }

    def compare_results(self):
        """Compare Trueno vs NumPy vs PyTorch"""
        print("\nComparing results...")

        for op_name in self.trueno_results.keys():
            if op_name not in self.comparison:
                self.comparison[op_name] = {}

            for size in self.trueno_results[op_name].keys():
                backends_data = self.trueno_results[op_name][size]
                best_backend, best_time = self._find_best_backend(backends_data)
                numpy_time = self._get_python_time("numpy", op_name, size)
                pytorch_time = self._get_python_time("pytorch_cpu", op_name, size)
                self.comparison[op_name][size] = self._build_comparison_entry(
                    best_backend, best_time, numpy_time, pytorch_time
                )

        print(f"✅ Compared {len(self.comparison)} operations")

    @staticmethod
    def _classify_entry(data: Dict) -> Tuple[bool, bool, bool, bool]:
        """Classify a single comparison entry.

        Returns (has_numpy, within_20pct, faster_numpy, faster_pytorch).
        """
        numpy_ratio = data.get("trueno_vs_numpy")
        pytorch_ratio = data.get("trueno_vs_pytorch")
        has_numpy = numpy_ratio is not None
        within = has_numpy and 0.8 <= numpy_ratio <= 1.2
        faster_np = has_numpy and numpy_ratio < 1.0
        faster_pt = pytorch_ratio is not None and pytorch_ratio < 1.0
        return has_numpy, within, faster_np, faster_pt

    def _compute_summary_stats(self) -> Dict:
        """Compute aggregate comparison statistics across all operations and sizes."""
        entries = [
            self._classify_entry(data)
            for sizes in self.comparison.values()
            for data in sizes.values()
        ]
        return {
            "total_comparisons": sum(has for has, _, _, _ in entries),
            "within_20_percent": sum(w for _, w, _, _ in entries),
            "faster_than_numpy": sum(fn for _, _, fn, _ in entries),
            "faster_than_pytorch": sum(fp for _, _, _, fp in entries),
        }

    @staticmethod
    def _format_report_header() -> List[str]:
        """Generate the report header lines."""
        return [
            "# Trueno vs NumPy vs PyTorch - Performance Comparison",
            "",
            "**Goal**: Validate that Trueno is within 20% of NumPy/PyTorch for 1D operations",
            "",
            "## Summary",
            "",
        ]

    @staticmethod
    def _format_summary_section(stats: Dict) -> List[str]:
        """Format the summary statistics section of the report."""
        total = stats["total_comparisons"]
        if total == 0:
            return []

        lines = []
        percent_within_20 = (stats["within_20_percent"] / total) * 100
        percent_faster_numpy = (stats["faster_than_numpy"] / total) * 100
        percent_faster_pytorch = (stats["faster_than_pytorch"] / total) * 100

        lines.append(f"- **Within 20% of NumPy**: {stats['within_20_percent']}/{total} ({percent_within_20:.1f}%)")
        lines.append(f"- **Faster than NumPy**: {stats['faster_than_numpy']}/{total} ({percent_faster_numpy:.1f}%)")
        lines.append(f"- **Faster than PyTorch**: {stats['faster_than_pytorch']}/{total} ({percent_faster_pytorch:.1f}%)")
        lines.append("")

        if percent_within_20 >= 80:
            lines.append("✅ **v0.3.0 SUCCESS CRITERIA MET**: >80% of operations within 20% of NumPy")
        else:
            lines.append("❌ **v0.3.0 CRITERIA NOT MET**: Need >80% within 20% (currently {:.1f}%)".format(percent_within_20))
        lines.append("")

        return lines

    @staticmethod
    def _format_time_ns(ns) -> str:
        """Format a time value in nanoseconds to a human-readable string."""
        if ns is None:
            return "-"
        if ns < 1000:
            return f"{ns:.1f} ns"
        elif ns < 1_000_000:
            return f"{ns/1000:.2f} µs"
        else:
            return f"{ns/1_000_000:.2f} ms"

    @staticmethod
    def _format_ratio(ratio) -> str:
        """Format a performance ratio as a human-readable comparison string."""
        if ratio is None:
            return "-"
        if ratio < 1.0:
            return f"✅ {1/ratio:.2f}x faster"
        elif ratio <= 1.2:
            return f"✓ {ratio:.2f}x (within 20%)"
        else:
            return f"⚠️ {ratio:.2f}x slower"

    def _format_operation_table(self, op_name: str) -> List[str]:
        """Format the comparison table for a single operation."""
        lines = [
            f"### {op_name}",
            "",
            "| Size | Trueno (best) | NumPy | PyTorch | Trueno vs NumPy | Trueno vs PyTorch |",
            "|------|---------------|-------|---------|-----------------|-------------------|",
        ]

        for size in ["100", "1000", "10000", "100000", "1000000"]:
            if size not in self.comparison[op_name]:
                continue
            data = self.comparison[op_name][size]
            trueno_str = f"{self._format_time_ns(data['trueno_mean_ns'])} ({data['trueno_backend']})"
            numpy_str = self._format_time_ns(data["numpy_mean_ns"])
            pytorch_str = self._format_time_ns(data["pytorch_mean_ns"])
            ratio_numpy_str = self._format_ratio(data["trueno_vs_numpy"])
            ratio_pytorch_str = self._format_ratio(data["trueno_vs_pytorch"])
            lines.append(f"| {size:>6} | {trueno_str} | {numpy_str} | {pytorch_str} | {ratio_numpy_str} | {ratio_pytorch_str} |")

        lines.append("")
        return lines

    def generate_markdown_report(self, output_file: str = "benchmarks/comparison_report.md"):
        """Generate markdown comparison report"""
        print(f"\nGenerating markdown report: {output_file}...")

        lines = self._format_report_header()
        stats = self._compute_summary_stats()
        lines.extend(self._format_summary_section(stats))

        lines.append("## Detailed Results")
        lines.append("")

        for op_name in sorted(self.comparison.keys()):
            lines.extend(self._format_operation_table(op_name))

        with open(output_file, "w") as f:
            f.write("\n".join(lines))

        print(f"✅ Report saved to: {output_file}")

    def save_comparison_json(self, output_file: str = "benchmarks/comparison_summary.json"):
        """Save comparison data as JSON"""
        output_data = {
            "comparison": self.comparison,
            "summary": {
                "operations_compared": len(self.comparison),
                "sizes": list(set([size for op in self.comparison.values() for size in op.keys()])),
            }
        }

        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"✅ JSON summary saved to: {output_file}")


def main():
    """Main entry point"""
    comparator = BenchmarkComparator()

    # Load results
    comparator.load_criterion_results("target/criterion")
    comparator.load_python_results("benchmarks/python_results.json")

    # Compare
    comparator.compare_results()

    # Generate reports
    comparator.generate_markdown_report()
    comparator.save_comparison_json()

    print("\n" + "=" * 80)
    print("✅ Comparison complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
