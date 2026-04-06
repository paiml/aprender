# Chaos Testing with Renacer

Renacer includes built-in **chaos engineering** capabilities for robustness testing. Use chaos mode to test how your applications behave under resource pressure, timeouts, and signal interruptions.

## Why Chaos Testing?

Normal testing verifies your application works under ideal conditions. Chaos testing answers harder questions:

- What happens when memory is scarce?
- Does your app handle CPU throttling gracefully?
- How does it respond to unexpected signals?
- Will it timeout properly under load?

## Quick Start

### Using Presets

The easiest way to get started is with presets:

```bash
# Gentle chaos - suitable for CI/CD
renacer --chaos gentle -c -- ./my-app

# Aggressive chaos - stress testing
renacer --chaos aggressive -c -- ./my-app
```

**Preset configurations:**

| Preset | Memory | CPU | Timeout | Signals |
|--------|--------|-----|---------|---------|
| `gentle` | 512MB | 80% | 120s | off |
| `aggressive` | 64MB | 25% | 10s | on |

### Custom Configuration

Fine-tune chaos parameters for your specific needs:

```bash
# Custom memory limit
renacer --chaos-memory-limit 128M -c -- ./my-app

# Custom CPU limit (50% of CPU time)
renacer --chaos-cpu-limit 0.5 -c -- ./my-app

# Custom timeout
renacer --chaos-timeout 30s -c -- ./my-app

# Enable signal injection
renacer --chaos-signals -c -- ./my-app
```

### Combined Options

Mix presets with custom overrides:

```bash
# Start with aggressive preset, but increase memory limit
renacer --chaos aggressive --chaos-memory-limit 128M -c -- ./my-app

# Gentle preset with signals enabled
renacer --chaos gentle --chaos-signals -c -- ./my-app
```

## CLI Reference

### `--chaos <PRESET>`

Use a named chaos preset. Available presets:

- **`gentle`**: Conservative limits for regular testing
  - Memory: 512MB
  - CPU: 80%
  - Timeout: 120 seconds
  - Signals: disabled

- **`aggressive`**: Strict limits for stress testing
  - Memory: 64MB
  - CPU: 25%
  - Timeout: 10 seconds
  - Signals: enabled

### `--chaos-memory-limit <SIZE>`

Set the maximum virtual memory for the traced process.

**Formats supported:**
- Bytes: `67108864`
- Kilobytes: `64K` or `64k`
- Megabytes: `64M` or `64m`
- Gigabytes: `1G` or `1g`

**Examples:**
```bash
renacer --chaos-memory-limit 64M -- ./my-app
renacer --chaos-memory-limit 1G -- ./memory-heavy-app
renacer --chaos-memory-limit 512K -- ./small-app
```

### `--chaos-cpu-limit <FRACTION>`

Limit CPU time as a fraction of real time (0.0 to 1.0).

- `0.5` = 50% of CPU time
- `0.25` = 25% of CPU time
- `1.0` = no limit (100%)

**Examples:**
```bash
# Half CPU speed
renacer --chaos-cpu-limit 0.5 -- ./my-app

# Quarter CPU speed (stress test)
renacer --chaos-cpu-limit 0.25 -- ./my-app
```

### `--chaos-timeout <DURATION>`

Set maximum execution time before termination.

**Formats supported:**
- Seconds: `30` or `30s`
- Minutes: `2m`
- Hours: `1h`

**Examples:**
```bash
renacer --chaos-timeout 10s -- ./quick-app
renacer --chaos-timeout 2m -- ./longer-app
renacer --chaos-timeout 1h -- ./batch-job
```

### `--chaos-signals`

Enable random signal injection. Periodically sends signals to test signal handling:

- `SIGALRM` - alarm timer signal
- `SIGUSR1` - user-defined signal

**Example:**
```bash
renacer --chaos-signals -- ./signal-handler-test
```

## Use Cases

### 1. Memory Leak Detection

Test if your application handles memory pressure gracefully:

```bash
# Strict memory limit
renacer --chaos-memory-limit 32M -c -- ./my-app

# Watch for these patterns:
# - Graceful error messages
# - Proper cleanup on OOM
# - No zombie processes
```

**What to look for:**
- Does the app exit cleanly?
- Are resources properly released?
- Is the error message helpful?

### 2. Timeout Validation

Ensure your application respects timeouts:

```bash
# Short timeout for quick validation
renacer --chaos-timeout 5s -c -- ./network-client

# Expected: App should handle timeout gracefully
# Red flag: Hanging processes, no timeout handling
```

### 3. CI/CD Integration

Add chaos testing to your CI pipeline:

```bash
#!/bin/bash
# ci-chaos-test.sh

echo "Running gentle chaos tests..."
renacer --chaos gentle -c -- ./target/release/my-app
if [ $? -ne 0 ]; then
    echo "Failed gentle chaos test"
    exit 1
fi

echo "Running aggressive chaos tests..."
renacer --chaos aggressive -c -- ./target/release/my-app
# May fail - that's expected for stress testing
# Check for graceful failures, not crashes
```

### 4. Flaky Test Investigation

Reproduce timing-sensitive bugs:

```bash
# Slow down execution to expose race conditions
renacer --chaos-cpu-limit 0.1 --chaos-signals -c -- ./flaky-test

# If test fails under chaos but passes normally,
# you likely have a race condition
```

### 5. Robustness Testing Before Deployment

Final validation before production:

```bash
# Full chaos suite
renacer --chaos aggressive \
    --chaos-memory-limit 64M \
    --chaos-cpu-limit 0.25 \
    --chaos-timeout 30s \
    --chaos-signals \
    -c -- ./production-binary

# If it survives this, it's production-ready!
```

## Example Session

Here's a complete chaos testing session:

```bash
$ renacer --chaos aggressive -c -- aprender-shell suggest "git "
⚠️  Chaos mode enabled: memory=64MB, cpu=25%, timeout=10s, signals=on

git status  1.000

% time     seconds  usecs/call     calls    errors syscall
------ ----------- ----------- --------- --------- ----------------
 45.23    0.012345         123       100           read
 30.12    0.008234          82       100           write
 15.45    0.004234          42       100           mmap
  5.20    0.001423          14       100           close
  4.00    0.001092          10       100           fstat
------ ----------- ----------- --------- --------- ----------------
100.00    0.027328                   500           total
```

## Interpreting Results

### Success Under Chaos

If your application completes successfully under aggressive chaos:
- Memory management is efficient
- Timeouts are handled properly
- Signal handlers are robust

### Graceful Failures

Some failures are expected and acceptable:
- Clean exit with error message
- Proper resource cleanup
- Meaningful exit codes

### Red Flags

Watch out for these warning signs:
- **Segmentation faults** - memory corruption
- **Hanging processes** - missing timeout handling
- **Zombie processes** - improper cleanup
- **Cryptic errors** - poor error handling

## Best Practices

### 1. Start Gentle, Then Aggressive

```bash
# First, validate basic functionality
renacer --chaos gentle -c -- ./my-app

# Then stress test
renacer --chaos aggressive -c -- ./my-app
```

### 2. Test Specific Failure Modes

```bash
# Memory-only stress
renacer --chaos-memory-limit 16M -c -- ./my-app

# CPU-only stress
renacer --chaos-cpu-limit 0.1 -c -- ./my-app

# Timeout-only stress
renacer --chaos-timeout 5s -c -- ./my-app
```

### 3. Combine with Statistics Mode

Always use `-c` (statistics mode) for better insights:

```bash
renacer --chaos aggressive -c --stats-extended -- ./my-app
```

### 4. Use with Anomaly Detection

Combine chaos with ML-based anomaly detection:

```bash
renacer --chaos gentle -c --ml-anomaly -- ./my-app
```

## Troubleshooting

### "Permission denied" Errors

Resource limits require appropriate permissions. Run without sudo first; some limits work without elevated privileges.

### App Killed Immediately

Memory limit may be too low. Start higher and decrease:

```bash
# Start high
renacer --chaos-memory-limit 256M -c -- ./my-app

# Decrease until failure
renacer --chaos-memory-limit 128M -c -- ./my-app
renacer --chaos-memory-limit 64M -c -- ./my-app
```

### Timeout Too Short

Some applications need warm-up time:

```bash
# Increase timeout for initialization
renacer --chaos-timeout 60s --chaos-memory-limit 64M -c -- ./slow-starter
```

## Related Topics

- [Statistical Analysis](./statistical-analysis.md) - Analyze syscall patterns
- [Anomaly Detection](./anomaly-detection.md) - Find outliers automatically
- [Performance Optimization](./performance-optimization.md) - Reduce overhead
- [Chaos Engineering (Contributing)](../contributing/chaos-engineering.md) - Internal chaos API

## Summary

Chaos testing with Renacer helps you:

- **Find hidden bugs** that normal testing misses
- **Validate error handling** under resource pressure
- **Ensure graceful degradation** when things go wrong
- **Build confidence** before production deployment

Start with `--chaos gentle` for regular testing, graduate to `--chaos aggressive` for stress testing, and use custom options for targeted validation.
