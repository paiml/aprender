# Flamegraph Visualization

Flamegraphs are a powerful visualization technique for profiling data, showing call stacks and their time distribution. This chapter shows how to export Renacer's profiling data and visualize it with external flamegraph tools.

> **TDD-Verified:** Export functionality tested in [`tests/sprint22_html_output_tests.rs`](../../../tests/), profiling in [`tests/sprint13_function_profiling_tests.rs`](../../../tests/)

> **Parent Chapter:** See [Function Profiling](./function-profiling.md) for overview and basic usage.

## Overview

**Flamegraphs** visualize call stacks with:
- **X-axis:** Alphabetically sorted function names (not time!)
- **Y-axis:** Stack depth (call hierarchy)
- **Width:** Proportion of total time spent in function
- **Color:** Usually random (for visual distinction) or can indicate metrics

**Why flamegraphs?**
- **Visual bottleneck identification** - Wide bars = hot code paths
- **Call hierarchy understanding** - See parent-child relationships
- **Interactive exploration** - Click to zoom, search functions
- **Shareable analysis** - Export as SVG for reports

**Example flamegraph:**
```text
┌─────────────────────────────────────────────┐ 100% (main)
│                main()                      │
├─────────────┬───────────────────────────────┤
│  process()  │     network_call()           │ 60%
│    40%      │                               │
├─────┬───────┼───────┬───────────────────────┤
│read │write  │connect│  send_request()      │ 35%
│ 20% │ 20%   │  5%   │                       │
└─────┴───────┴───────┴───────────────────────┘
```

**Interpretation:** `send_request()` is the bottleneck (35% of total time).

## Exporting Profiling Data

Renacer supports JSON export for post-processing:

```bash
$ renacer --function-time --source --format json -- ./myapp > profile.json
```

**Tested by:** `test_html_format_flag_accepted` (Sprint 22 - verifies format flag acceptance)

**JSON Output Structure:**
```json
{
  "syscalls": [...],
  "function_profile": [
    {
      "function": "src/main.rs:42",
      "calls": 150,
      "total_time_us": 234567,
      "avg_time_us": 1563,
      "slow_io_count": 148
    },
    ...
  ],
  "summary": {
    "total_syscalls": 500,
    "total_duration_ms": 580.245
  }
}
```

## Generating Flamegraphs

### Method 1: Using Brendan Gregg's Flamegraph Tools

**Prerequisites:**
```bash
$ git clone https://github.com/brendangregg/FlameGraph
$ cd FlameGraph
```

**Step 1: Convert JSON to Folded Stack Format**

Create a conversion script `renacer_to_folded.py`:

```python
#!/usr/bin/env python3
import json
import sys

# Read Renacer JSON output
with open(sys.argv[1]) as f:
    data = json.load(f)

# Convert function profile to folded stacks
for func in data.get('function_profile', []):
    function_name = func['function']
    # Use total time (microseconds) as sample count
    samples = func['total_time_us']

    # Format: function_name samples
    print(f"{function_name} {samples}")
```

**Step 2: Generate Flamegraph**

```bash
# Convert Renacer JSON → folded stacks
$ python3 renacer_to_folded.py profile.json > profile.folded

# Generate flamegraph SVG
$ ./FlameGraph/flamegraph.pl profile.folded > flamegraph.svg

# Open in browser
$ firefox flamegraph.svg
```

**Result:** Interactive SVG flamegraph showing function time distribution!

### Method 2: Using speedscope.app

**Speedscope** is a web-based flamegraph viewer:

**Step 1: Export profiling data**
```bash
$ renacer --function-time --source --format json -- ./myapp > profile.json
```

**Step 2: Visit speedscope.app**
```bash
$ firefox https://www.speedscope.app/
```

**Step 3: Upload `profile.json`**

**Note:** Speedscope expects specific JSON formats (Chrome timeline, Firefox profiler, etc.). Renacer's format may need conversion. Use Method 1 (Brendan Gregg's tools) for simplest workflow.

### Method 3: Manual HTML Visualization

For small datasets, create a simple HTML visualization:

```html
<!DOCTYPE html>
<html>
<head>
    <title>Renacer Flamegraph</title>
    <style>
        .bar { margin: 2px; padding: 5px; background: #f90; }
        .function { font-family: monospace; }
        .time { float: right; color: #666; }
    </style>
</head>
<body>
    <h1>Function Profile</h1>
    <div id="chart"></div>
    <script>
        // Load profile.json (served via http server)
        fetch('profile.json')
            .then(r => r.json())
            .then(data => {
                const total = data.summary.total_duration_ms * 1000; // Convert to μs
                const chart = document.getElementById('chart');

                data.function_profile
                    .sort((a, b) => b.total_time_us - a.total_time_us)
                    .forEach(func => {
                        const pct = (func.total_time_us / total * 100).toFixed(1);
                        const bar = document.createElement('div');
                        bar.className = 'bar';
                        bar.style.width = pct + '%';
                        bar.innerHTML = `
                            <span class="function">${func.function}</span>
                            <span class="time">${func.total_time_us}μs (${pct}%)</span>
                        `;
                        chart.appendChild(bar);
                    });
            });
    </script>
</body>
</html>
```

**Serve and view:**
```bash
$ python3 -m http.server 8080
$ firefox http://localhost:8080/
```

## Practical Examples

### Example 1: Identify Database Bottlenecks

**Scenario:** Web application with database calls

```bash
# Profile application
$ renacer --function-time --source --format json -e trace=network -- ./webapp > profile.json

# Convert to flamegraph
$ python3 renacer_to_folded.py profile.json | ./FlameGraph/flamegraph.pl > db-bottlenecks.svg
```

**Flamegraph shows:**
- Wide bar for `execute_query()` → Database calls dominate execution time
- Narrow bar for `cache_lookup()` → Fast cache hits
- Medium bar for `json_serialize()` → Moderate CPU time

**Action:** Add caching layer to reduce wide `execute_query()` bar.

### Example 2: Build System Analysis

**Scenario:** Slow cargo build

```bash
$ renacer --function-time --source --format json -e trace=file -- cargo build > build-profile.json
$ python3 renacer_to_folded.py build-profile.json | ./FlameGraph/flamegraph.pl > build-flame.svg
```

**Flamegraph shows:**
- Wide bar for `rustc:link` → Linking is the bottleneck
- Narrow bars for `cargo:download_crate` → Dependencies already cached
- Medium bar for `rustc:codegen` → Compilation time is acceptable

**Action:** Enable incremental compilation to reduce linking time.

### Example 3: Comparing Before/After Optimizations

**Before optimization:**
```bash
$ renacer --function-time --source --format json -- ./app-v1 > before.json
$ python3 renacer_to_folded.py before.json | ./FlameGraph/flamegraph.pl > before.svg
```

**After optimization:**
```bash
$ renacer --function-time --source --format json -- ./app-v2 > after.json
$ python3 renacer_to_folded.py after.json | ./FlameGraph/flamegraph.pl > after.svg
```

**Compare side-by-side:**
```bash
$ firefox before.svg after.svg
```

**Look for:**
- Narrower bars (faster functions)
- Shorter stacks (reduced call depth)
- Fewer wide bars (eliminated bottlenecks)

## Advanced Workflows

### Filtering Hot Paths Only

Show only functions consuming >1% of total time:

```python
#!/usr/bin/env python3
import json
import sys

with open(sys.argv[1]) as f:
    data = json.load(f)

total_time = data['summary']['total_duration_ms'] * 1000  # μs
threshold = total_time * 0.01  # 1% threshold

for func in data.get('function_profile', []):
    if func['total_time_us'] > threshold:
        print(f"{func['function']} {func['total_time_us']}")
```

**Usage:**
```bash
$ python3 filter_hot_paths.py profile.json | ./FlameGraph/flamegraph.pl > hot-paths.svg
```

**Result:** Cleaner flamegraph focusing on significant bottlenecks.

### Differential Flamegraphs

Compare two profiles to see what changed:

```bash
# Requires flamegraph-diff tool
$ python3 renacer_to_folded.py before.json > before.folded
$ python3 renacer_to_folded.py after.json > after.folded

$ ./FlameGraph/difffolded.pl before.folded after.folded | ./FlameGraph/flamegraph.pl > diff.svg
```

**Red regions:** Functions slower in after.json
**Blue regions:** Functions faster in after.json
**Gray regions:** No significant change

### Color-Coding by Slow I/O

Customize flamegraph colors based on slow I/O count:

```python
#!/usr/bin/env python3
import json
import sys

with open(sys.argv[1]) as f:
    data = json.load(f)

for func in data.get('function_profile', []):
    slow_io_pct = func['slow_io_count'] / max(func['calls'], 1) * 100
    # Color code: red for >50% slow I/O, orange for >10%, default otherwise
    color = 'red' if slow_io_pct > 50 else ('orange' if slow_io_pct > 10 else 'default')

    print(f"{func['function']} {func['total_time_us']} # {color}")
```

**Result:** Flamegraph visually highlights I/O bottlenecks with color!

## Troubleshooting

### Empty Flamegraph

**Problem:** Flamegraph shows no data or "No function profiling data collected"

**Cause:** Function profiling not enabled or no debug symbols

**Solution:**
```bash
# Ensure both --function-time and --source are used
$ renacer --function-time --source --format json -- ./myapp > profile.json

# Verify debug symbols
$ file ./myapp  # Should show "with debug_info, not stripped"
```

### Incorrect Stack Heights

**Problem:** All functions appear at same level (no hierarchy)

**Cause:** Renacer currently exports immediate callers only, not full stacks

**Workaround:**
- Flamegraphs work best with full stack traces
- Renacer's function profiling shows caller-callee pairs
- For hierarchical visualization, manually construct stacks from profiling data

### Conversion Errors

**Problem:** `renacer_to_folded.py` fails with JSON parse errors

**Cause:** Invalid JSON output or large files

**Solution:**
```bash
# Verify JSON is valid
$ jq '.' profile.json > /dev/null

# For large files, use streaming JSON parser
$ cat profile.json | jq -c '.function_profile[]' | while read line; do
    # Process line by line
done
```

## Limitations

**Current limitations:**
1. **No native flamegraph export** - Requires external tools (Brendan Gregg's scripts)
2. **Immediate callers only** - Full stack traces not yet supported
3. **Time-based only** - Cannot flamegraph by call count or other metrics (without custom scripts)

**Future enhancements:**
- Native folded stack export
- Full stack trace capture (multi-level unwinding)
- Direct SVG flamegraph generation

## Summary

Flamegraph visualization provides:
- ✅ **Visual bottleneck identification** via JSON export + external tools
- ✅ **Interactive exploration** with Brendan Gregg's flamegraph.pl
- ✅ **Comparison workflows** for before/after optimization analysis
- ✅ **Custom color-coding** based on slow I/O metrics
- ✅ **Integration** with existing flamegraph ecosystems

**Workflow:**
1. Profile with `--function-time --source --format json`
2. Convert JSON to folded stack format
3. Generate flamegraph SVG with external tools
4. Analyze and optimize

**All export functionality tested in:** [`tests/sprint22_html_output_tests.rs`](../../../tests/), function profiling in [`tests/sprint13_function_profiling_tests.rs`](../../../tests/)

## Related

- [Function Profiling](./function-profiling.md) - Parent chapter with basic usage
- [Call Graph Analysis](./call-graphs.md) - Understanding function relationships
- [I/O Bottleneck Detection](./io-bottlenecks.md) - Identifying slow I/O for visualization
- [Export to JSON/CSV](../examples/export-data.md) - Data export workflows
