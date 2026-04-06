#!/usr/bin/env python3
"""
Benchmark Regression Checker

Parses Criterion benchmark output and compares against baseline.
Fails CI if any benchmark shows >5% regression.

Usage:
    ./scripts/check_regression.py [--baseline FILE] [--current FILE]

    cat bench.txt | ./scripts/check_regression.py --baseline baseline.txt
"""

import argparse
import re
import sys
from typing import Dict, Tuple, Optional


def parse_time(time_str: str) -> float:
    """Convert time string to nanoseconds.

    Examples:
        "52.46 ns" -> 52.46
        "856.3 ns" -> 856.3
        "9.99 µs" -> 9990.0
        "31.95 ms" -> 31950000.0
        "0.65 s" -> 650000000.0
    """
    match = re.match(r'([\d.]+)\s*(ns|µs|us|ms|s)', time_str.strip())
    if not match:
        raise ValueError(f"Cannot parse time: {time_str}")

    value = float(match.group(1))
    unit = match.group(2)

    multipliers = {
        'ns': 1,
        'µs': 1_000,
        'us': 1_000,
        'ms': 1_000_000,
        's': 1_000_000_000
    }

    return value * multipliers[unit]


def parse_criterion_output(text: str) -> Dict[str, float]:
    """Parse Criterion benchmark output and extract mean times.

    Returns dict mapping benchmark name to mean time in nanoseconds.
    """
    results = {}

    # Pattern to match benchmark results
    # Examples:
    # gpu_vec_add/GPU/1000    time:   [31.707 ms 31.950 ms 32.184 ms]
    # gpu_vec_add/Scalar/1000 time:   [52.142 ns 52.462 ns 52.823 ns]
    pattern = r'^(\S+)\s+time:\s+\[([\d.]+\s*(?:ns|µs|us|ms|s))\s+([\d.]+\s*(?:ns|µs|us|ms|s))\s+([\d.]+\s*(?:ns|µs|us|ms|s))\]'

    for line in text.split('\n'):
        line = line.strip()
        match = re.match(pattern, line)
        if match:
            name = match.group(1)
            # Group 3 is the mean (middle value)
            mean_time = parse_time(match.group(3))
            results[name] = mean_time

    return results


def format_time(ns: float) -> str:
    """Format nanoseconds as human-readable string."""
    if ns >= 1_000_000_000:
        return f"{ns / 1_000_000_000:.2f}s"
    elif ns >= 1_000_000:
        return f"{ns / 1_000_000:.2f}ms"
    elif ns >= 1_000:
        return f"{ns / 1_000:.2f}µs"
    else:
        return f"{ns:.2f}ns"


def compare_benchmarks(
    baseline: Dict[str, float],
    current: Dict[str, float],
    regression_threshold: float = 0.05,
    warning_threshold: float = 0.02
) -> Tuple[bool, str]:
    """Compare current benchmarks against baseline.

    Returns:
        (passed, report): Whether check passed and detailed report
    """
    regressions = []
    warnings = []
    improvements = []
    unchanged = []
    missing = []

    for name, baseline_time in sorted(baseline.items()):
        if name not in current:
            missing.append(name)
            continue

        current_time = current[name]
        change = (current_time - baseline_time) / baseline_time

        if change > regression_threshold:
            regressions.append((name, baseline_time, current_time, change))
        elif change > warning_threshold:
            warnings.append((name, baseline_time, current_time, change))
        elif change < -warning_threshold:
            improvements.append((name, baseline_time, current_time, change))
        else:
            unchanged.append(name)

    # New benchmarks (in current but not in baseline)
    new_benchmarks = [name for name in current if name not in baseline]

    # Generate report
    lines = []
    lines.append("=" * 60)
    lines.append("BENCHMARK REGRESSION REPORT")
    lines.append("=" * 60)
    lines.append("")

    if regressions:
        lines.append(f"REGRESSIONS (>{regression_threshold*100:.0f}% slower): {len(regressions)}")
        lines.append("-" * 60)
        for name, base, curr, change in regressions:
            lines.append(f"  {name}")
            lines.append(f"    Baseline: {format_time(base)}")
            lines.append(f"    Current:  {format_time(curr)}")
            lines.append(f"    Change:   +{change*100:.1f}% (FAIL)")
            lines.append("")

    if warnings:
        lines.append(f"WARNINGS ({warning_threshold*100:.0f}%-{regression_threshold*100:.0f}% slower): {len(warnings)}")
        lines.append("-" * 60)
        for name, base, curr, change in warnings:
            lines.append(f"  {name}")
            lines.append(f"    Baseline: {format_time(base)}")
            lines.append(f"    Current:  {format_time(curr)}")
            lines.append(f"    Change:   +{change*100:.1f}%")
            lines.append("")

    if improvements:
        lines.append(f"IMPROVEMENTS (>{warning_threshold*100:.0f}% faster): {len(improvements)}")
        lines.append("-" * 60)
        for name, base, curr, change in improvements:
            lines.append(f"  {name}")
            lines.append(f"    Baseline: {format_time(base)}")
            lines.append(f"    Current:  {format_time(curr)}")
            lines.append(f"    Change:   {change*100:.1f}%")
            lines.append("")

    if missing:
        lines.append(f"MISSING (in baseline, not in current): {len(missing)}")
        lines.append("-" * 60)
        for name in missing:
            lines.append(f"  {name}")
        lines.append("")

    if new_benchmarks:
        lines.append(f"NEW (in current, not in baseline): {len(new_benchmarks)}")
        lines.append("-" * 60)
        for name in new_benchmarks:
            lines.append(f"  {name}")
        lines.append("")

    # Summary
    lines.append("=" * 60)
    lines.append("SUMMARY")
    lines.append("=" * 60)
    total = len(baseline)
    lines.append(f"  Total benchmarks: {total}")
    lines.append(f"  Regressions:      {len(regressions)}")
    lines.append(f"  Warnings:         {len(warnings)}")
    lines.append(f"  Improvements:     {len(improvements)}")
    lines.append(f"  Unchanged:        {len(unchanged)}")
    lines.append(f"  Missing:          {len(missing)}")
    lines.append(f"  New:              {len(new_benchmarks)}")
    lines.append("")

    if regressions:
        lines.append(f"RESULT: FAIL - {len(regressions)} regression(s) detected")
    elif warnings:
        lines.append(f"RESULT: PASS (with warnings) - {len(warnings)} warning(s)")
    else:
        lines.append("RESULT: PASS")

    lines.append("=" * 60)

    passed = len(regressions) == 0
    report = '\n'.join(lines)

    return passed, report


def build_arg_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser for the regression checker."""
    parser = argparse.ArgumentParser(
        description='Check benchmark performance regressions'
    )
    parser.add_argument(
        '--baseline', '-b',
        required=True,
        help='Baseline benchmark output file'
    )
    parser.add_argument(
        '--current', '-c',
        help='Current benchmark output file (default: stdin)'
    )
    parser.add_argument(
        '--regression-threshold', '-r',
        type=float,
        default=0.05,
        help='Regression threshold (default: 0.05 = 5%%)'
    )
    parser.add_argument(
        '--warning-threshold', '-w',
        type=float,
        default=0.02,
        help='Warning threshold (default: 0.02 = 2%%)'
    )
    return parser


def read_file_or_exit(file_path: str, label: str) -> str:
    """Read a file and return its contents, or exit with an error message."""
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: {label} file not found: {file_path}", file=sys.stderr)
        sys.exit(1)


def read_current_input(current_path: Optional[str]) -> str:
    """Read current benchmark data from a file or stdin."""
    if current_path:
        return read_file_or_exit(current_path, "Current")
    return sys.stdin.read()


def parse_and_validate(text: str, label: str) -> Dict[str, float]:
    """Parse criterion output and exit if no benchmarks are found."""
    results = parse_criterion_output(text)
    if not results:
        print(f"Error: No benchmarks found in {label}", file=sys.stderr)
        sys.exit(1)
    return results


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    baseline_text = read_file_or_exit(args.baseline, "Baseline")
    current_text = read_current_input(args.current)

    baseline = parse_and_validate(baseline_text, "baseline")
    current = parse_and_validate(current_text, "current")

    passed, report = compare_benchmarks(
        baseline,
        current,
        args.regression_threshold,
        args.warning_threshold
    )

    print(report)
    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()
