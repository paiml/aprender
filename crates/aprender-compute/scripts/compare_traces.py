#!/usr/bin/env python3
"""Compare OTLP trace data between two commits for performance regression detection."""
import sys
import json
from collections import defaultdict

def analyze(file_path):
    """Analyze a trace file and return syscall statistics."""
    with open(file_path) as f:
        data = json.load(f)

    syscalls = defaultdict(lambda: {"count": 0, "total_us": 0})

    for trace in data.get("data", []):
        for span in trace.get("spans", []):
            op = span.get("operationName", "")
            duration = next(
                (t["value"] for t in span.get("tags", []) if t["key"] == "syscall.duration_us"),
                0
            )
            if op.startswith("syscall:"):
                name = op.split(": ")[1] if ": " in op else op
                syscalls[name]["count"] += 1
                syscalls[name]["total_us"] += duration

    return syscalls

def main():
    if len(sys.argv) != 5:
        print("Usage: compare_traces.py <baseline.json> <current.json> <baseline_tag> <current_tag>", file=sys.stderr)
        sys.exit(1)

    baseline_file = sys.argv[1]
    current_file = sys.argv[2]
    baseline_tag = sys.argv[3]
    current_tag = sys.argv[4]

    try:
        baseline = analyze(baseline_file)
        current = analyze(current_file)
    except FileNotFoundError as e:
        print(f"File not found: {e.filename}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON: {e}", file=sys.stderr)
        sys.exit(1)

    all_syscalls = set(baseline.keys()) | set(current.keys())

    print(f"# Performance Comparison: {baseline_tag} → {current_tag}\n")
    print(f"| Syscall | {baseline_tag} Calls | {current_tag} Calls | Δ Calls | {baseline_tag} Time (μs) | {current_tag} Time (μs) | Δ Time |")
    print("|---------|-----------|----------|---------|------------|----------|--------|")

    for name in sorted(all_syscalls):
        b_count = baseline.get(name, {}).get("count", 0)
        c_count = current.get(name, {}).get("count", 0)
        b_time = baseline.get(name, {}).get("total_us", 0)
        c_time = current.get(name, {}).get("total_us", 0)
        delta_count = c_count - b_count
        delta_time = c_time - b_time
        delta_count_str = f"+{delta_count}" if delta_count > 0 else str(delta_count)
        delta_time_str = f"+{delta_time}" if delta_time > 0 else str(delta_time)

        if b_count > 0 or c_count > 0:
            print(f"| {name:15s} | {b_count:9d} | {c_count:9d} | {delta_count_str:7s} | {b_time:10d} | {c_time:10d} | {delta_time_str:6s} |")

if __name__ == "__main__":
    main()
