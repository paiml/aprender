#!/usr/bin/env python3
"""Analyze exported OTLP trace data for syscall profiling."""
import sys
import json
from collections import defaultdict


def load_trace_file(file_path):
    """Load and parse a JSON trace file, exiting on error."""
    try:
        with open(file_path) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"File not found: {file_path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON: {e}", file=sys.stderr)
        sys.exit(1)


def extract_span_duration(span):
    """Extract the syscall.duration_us tag value from a span."""
    return next(
        (t["value"] for t in span.get("tags", []) if t["key"] == "syscall.duration_us"),
        0,
    )


def extract_syscall_name(operation_name):
    """Extract the syscall name from an operation string like 'syscall: read'."""
    return operation_name.split(": ")[1] if ": " in operation_name else operation_name


def aggregate_syscalls(data):
    """Aggregate syscall statistics from trace data."""
    syscalls = defaultdict(lambda: {"count": 0, "total_us": 0, "max_us": 0})

    for trace in data.get("data", []):
        for span in trace.get("spans", []):
            op = span.get("operationName", "")
            if not op.startswith("syscall:"):
                continue
            duration = extract_span_duration(span)
            name = extract_syscall_name(op)
            syscalls[name]["count"] += 1
            syscalls[name]["total_us"] += duration
            syscalls[name]["max_us"] = max(syscalls[name]["max_us"], duration)

    return syscalls


def print_summary(data, syscalls):
    """Print the trace summary and top syscalls by time."""
    print(f"Traces: {len(data.get('data', []))}")
    print(f"Total syscalls: {sum(s['count'] for s in syscalls.values())}")
    print(f"Total time: {sum(s['total_us'] for s in syscalls.values())}μs\n")
    print("Top syscalls by time:")

    ranked = sorted(syscalls.items(), key=lambda x: x[1]["total_us"], reverse=True)
    for name, stats in ranked[:10]:
        avg = stats["total_us"] / stats["count"] if stats["count"] > 0 else 0
        print(f"  {name:20s} {stats['count']:5d} calls  {stats['total_us']:8d}μs  avg: {avg:6.1f}μs")


def main():
    if len(sys.argv) != 2:
        print("Usage: analyze_traces.py <trace_file.json>", file=sys.stderr)
        sys.exit(1)

    data = load_trace_file(sys.argv[1])
    syscalls = aggregate_syscalls(data)
    print_summary(data, syscalls)


if __name__ == "__main__":
    main()
