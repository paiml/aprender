# Golden Trace Validation

The `renacer validate` subcommand enables regression detection through golden trace comparison. Based on the APR Runtime Model Tracing Support specification (Sprint 50).

## Toyota Way: Jidoka

**Jidoka** means "stop on defect detection." Golden trace validation implements this principle by:

1. **Capturing known-good behavior** as a baseline (golden trace)
2. **Comparing current behavior** against the baseline
3. **Stopping CI/CD pipelines** when regressions are detected

## Quick Start

### Generate a Baseline

```bash
# Create golden trace directory from known-good behavior
renacer validate --generate ./golden -- echo "hello"
```

This creates:
```text
golden/
  manifest.json     # Trace metadata (version, platform, command)
  syscalls.trace    # Binary syscall sequence (RNTR format)
  timing.stats      # Timing statistics per syscall
```

### Validate Against Baseline

```bash
# Compare current behavior against baseline
renacer validate --baseline ./golden -- echo "hello"
```

**Exit code 0**: Validation passed
**Exit code 1**: Regression detected

## Exit Codes

| Code | Constant | Meaning |
|------|----------|---------|
| 0 | `Passed` | Validation passed |
| 1 | `Failed` | Regression detected |
| 2 | `BaselineNotFound` | Baseline directory not found |
| 3 | `InvalidBaseline` | Invalid baseline format (corrupt or incompatible) |
| 4 | `CommandError` | Command execution error |
| 5 | `ConfigError` | Configuration error |

## Configuration Options

### Timing Tolerance

By default, timing variations up to 10% are allowed:

```bash
# Allow 15% timing variance
renacer validate --baseline ./golden --tolerance 15.0 -- ./myapp

# Strict mode: zero tolerance
renacer validate --baseline ./golden --strict -- ./myapp
```

### Ignore Timing

For behavioral-only validation (ignoring performance):

```bash
# Only compare syscall sequences, ignore timing
renacer validate --baseline ./golden --ignore-timing -- ./myapp
```

### Output Formats

```bash
# Human-readable text (default)
renacer validate --baseline ./golden --output text -- ./myapp

# JSON for machine parsing
renacer validate --baseline ./golden --output json -- ./myapp

# JUnit XML for CI integration
renacer validate --baseline ./golden --output junit -- ./myapp > results.xml
```

## Output Format Examples

### Text Output

```text
=== Validation Report ===

Status: FAILED

Summary:
  Total syscalls compared: 42
  Matches: 40
  Mismatches: 1
  Timing regressions: 1

Syscall Mismatches:
  [DIFFERENT] Index 15: expected 'read', found 'write'

Timing Regressions:
  read: 0.10ms -> 0.12ms (+20.0%)
```

### JSON Output

```json
{
  "status": "failed",
  "total_compared": 42,
  "matches": 40,
  "mismatches": 1,
  "timing_regressions": 1,
  "syscall_details": [
    {
      "index": 15,
      "expected": "read",
      "found": "write",
      "mismatch_type": "different"
    }
  ],
  "timing_details": [
    {
      "syscall": "read",
      "baseline_ms": 0.10,
      "actual_ms": 0.12,
      "delta_percent": 20.0
    }
  ]
}
```

### JUnit XML Output

```xml
<?xml version="1.0" encoding="UTF-8"?>
<testsuite name="renacer-validate" tests="3" failures="2" errors="0">
  <testcase name="validation-overall" classname="renacer.validate">
    <failure message="Validation failed: 1 mismatches, 1 timing regressions"/>
  </testcase>
  <testcase name="syscall-15" classname="renacer.validate.syscalls">
    <failure message="Expected 'read', found 'write' (different)"/>
  </testcase>
  <testcase name="timing-read" classname="renacer.validate.timing">
    <failure message="Timing regression: 0.10ms -> 0.12ms (+20.0%)"/>
  </testcase>
</testsuite>
```

## Binary Trace Format

The `syscalls.trace` file uses a binary format for efficient storage:

### Header (32 bytes)

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0 | 4 | Magic | "RNTR" |
| 4 | 4 | Version | Format version (currently 1) |
| 8 | 4 | Flags | Bitfield (compressed, timing, string_args) |
| 12 | 8 | Entry Count | Number of syscall entries |
| 20 | 8 | Checksum | CRC32 of entries |
| 28 | 4 | Reserved | Padding |

### Flags Bitfield

| Bit | Meaning |
|-----|---------|
| 0 | Compressed (LZ4) |
| 1 | Has timing data |
| 2 | Has string arguments |

### Footer

| Size | Field | Description |
|------|-------|-------------|
| 4 | Magic | "RTNR" (reversed) |

## Manifest Format

The `manifest.json` contains trace metadata:

```json
{
  "version": "1.0.0",
  "renacer_version": "0.5.0",
  "created_at": "1734355200",
  "platform": {
    "os": "linux",
    "arch": "x86_64",
    "kernel": "6.8.0-87-generic"
  },
  "command": ["echo", "hello"],
  "statistics": {
    "total_syscalls": 42,
    "total_duration_ms": 1.234,
    "unique_syscalls": 12
  },
  "tolerance": {
    "timing_percent": 10.0,
    "syscall_sequence": "exact",
    "argument_match": "fuzzy"
  }
}
```

## CI/CD Integration

### GitHub Actions

```yaml
name: Regression Detection

on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install Renacer
        run: cargo install renacer

      - name: Validate against golden trace
        run: |
          renacer validate --baseline tests/golden/ --output junit -- ./target/release/myapp > results.xml

      - name: Upload test results
        uses: actions/upload-artifact@v4
        with:
          name: validation-results
          path: results.xml
```

### Jenkins Pipeline

```groovy
pipeline {
    stages {
        stage('Validate') {
            steps {
                sh 'renacer validate --baseline golden/ --output junit -- ./myapp > validation.xml'
            }
            post {
                always {
                    junit 'validation.xml'
                }
            }
        }
    }
}
```

## Comparison Logic

### Syscall Sequence Comparison

The validator compares syscall sequences index-by-index:

1. **Length check**: Different lengths = mismatch
2. **Syscall number check**: Different syscall at same index = mismatch
3. **Order sensitivity**: Strict ordering by default

Mismatch types:
- `DIFFERENT`: Different syscall at same position
- `EXTRA`: Syscall in actual but not in baseline
- `MISSING`: Syscall in baseline but not in actual

### Timing Regression Detection

For each syscall type, the validator checks:

```text
delta_percent = (actual_mean - baseline_mean) / baseline_mean * 100

if delta_percent > tolerance_percent:
    return TimingRegression
```

## API Usage

```rust,ignore
use renacer::validate::{
    compare_syscalls, compare_timing, format_json_report,
    ComparisonResult, SyscallTimingStats, TraceSyscallEntry,
    ValidateConfig, ValidateExitCode,
};
use std::collections::HashMap;

// Compare syscall sequences
let baseline = vec![
    TraceSyscallEntry::simple(257, "openat", 50_000),
    TraceSyscallEntry::simple(0, "read", 100_000),
];
let actual = baseline.clone();

let result = compare_syscalls(&baseline, &actual, false)?;
assert!(result.passed);

// Compare timing with tolerance
let mut baseline_stats = HashMap::new();
baseline_stats.insert("read".to_string(), SyscallTimingStats {
    count: 100,
    mean_ns: 100_000,  // 100us baseline
    ..Default::default()
});

let mut actual_stats = HashMap::new();
actual_stats.insert("read".to_string(), SyscallTimingStats {
    count: 100,
    mean_ns: 105_000,  // 5% slower
    ..Default::default()
});

let regressions = compare_timing(&baseline_stats, &actual_stats, 10.0);
assert!(regressions.is_empty());  // Within 10% tolerance
```

## Best Practices

### 1. Multiple Baseline Runs

Generate baselines from multiple runs to capture natural variance:

```bash
for i in {1..5}; do
  renacer validate --generate ./golden-run-$i -- ./myapp
done
# Then merge statistics
```

### 2. Platform-Specific Baselines

Kernel and hardware differences affect syscall behavior:

```bash
# Store baselines per platform
renacer validate --generate ./golden/linux-x86_64/ -- ./myapp
renacer validate --generate ./golden/linux-arm64/ -- ./myapp
```

### 3. Lenient Tolerance for CI

Use higher tolerance in CI to reduce false positives from runner variance:

```bash
renacer validate --baseline ./golden --tolerance 25.0 -- ./myapp
```

### 4. Strict Mode for Security

For security-critical validation, use strict mode:

```bash
renacer validate --baseline ./golden --strict -- ./security-critical-app
```

## Related

- [Regression Detection](./regression-detection.md) - Statistical hypothesis testing
- [Semantic Equivalence](./semantic-equivalence.md) - Behavioral equivalence checking
- [Syscall Clustering](./syscall-clustering.md) - Group related syscalls

## Toyota Way Principles Applied

| Principle | Application |
|-----------|-------------|
| **Jidoka** | Stop CI/CD on regression detection |
| **Muda** | Eliminate waste from repeated manual testing |
| **Poka-Yoke** | Prevent defects from reaching production |
| **Kaizen** | Continuously improve baseline accuracy |
