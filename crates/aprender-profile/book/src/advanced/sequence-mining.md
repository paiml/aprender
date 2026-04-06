# Sequence Mining

N-gram grammar detection for identifying unexpected syscall patterns.

## Overview

Sequence mining analyzes the **order** of syscalls to detect behavioral anomalies. Based on Forrest et al.'s (1996) seminal work on intrusion detection, this technique identifies "grammar violations" - syscall sequences that deviate from baseline behavior.

## Key Concept: Syscall Grammar

Every program has an implicit "grammar" - expected patterns of syscall sequences:

```text
Normal transpiler grammar:
  open → read → mmap → write → close

Anomalous grammar (telemetry leak):
  open → read → socket → connect → send → mmap → write → close
                ^^^^^^^^^^^^^^^^^^^^^^^^
                      NEW PATTERN
```

## N-gram Extraction

Renacer extracts N-grams (sliding windows) from syscall sequences:

### 2-grams (bigrams)
```rust,ignore
Sequence: ["open", "read", "mmap", "write", "close"]

2-grams:
  - ["open", "read"]
  - ["read", "mmap"]
  - ["mmap", "write"]
  - ["write", "close"]
```

### 3-grams (trigrams)
```rust,ignore
Sequence: ["open", "read", "mmap", "write", "close"]

3-grams:
  - ["open", "read", "mmap"]
  - ["read", "mmap", "write"]
  - ["mmap", "write", "close"]
```

## Anomaly Detection

Compare baseline N-grams with current N-grams to find new patterns:

```rust,ignore
use renacer::sequence::{extract_ngrams, detect_sequence_anomalies};

// Baseline (known-good)
let baseline_syscalls = vec!["open", "read", "write", "close"];
let baseline_ngrams = extract_ngrams(&baseline_syscalls, 3);

// Current (test version)
let current_syscalls = vec!["open", "read", "socket", "connect", "send", "write", "close"];
let current_ngrams = extract_ngrams(&current_syscalls, 3);

// Detect anomalies (30% frequency threshold)
let anomalies = detect_sequence_anomalies(&baseline_ngrams, &current_ngrams, 0.30);

for anomaly in anomalies {
    println!("New pattern: {:?}", anomaly.ngram);
    println!("Frequency: {} times", anomaly.frequency);
}
```

## Real-World Example: depyler Telemetry Leak

**Baseline Grammar** (v3.19.0):
```text
open → read → mmap → write → close
```

**Current Grammar** (v3.20.0 with Sentry):
```text
open → read → socket → connect → send → mmap → write → close
```

**Detected Anomalies**:
- `["read", "socket", "connect"]` (NEW)
- `["socket", "connect", "send"]` (NEW)
- `["connect", "send", "mmap"]` (NEW)

**Root Cause**: Sentry-rs telemetry library added networking syscalls.

## Frequency Thresholding

Not all new patterns are bugs! Use frequency thresholds to filter noise:

```rust,ignore
// Only report patterns that occur in >30% of executions
let anomalies = detect_sequence_anomalies(&baseline, &current, 0.30);
```

**Rationale**: Rare patterns may be legitimate edge cases.

## N-gram Size Selection

| N-gram Size | Coverage | Noise |
|-------------|----------|-------|
| 2-grams | High | High (many false positives) |
| 3-grams | **Optimal** | Low (good signal-to-noise) |
| 4-grams | Low | Very low (may miss patterns) |

**Recommendation**: Use **3-grams** (trigrams) for best results.

## Implementation

### Extract N-grams
```rust,ignore
use renacer::sequence::extract_ngrams;

let syscalls = vec!["open", "read", "write", "close"];
let ngrams = extract_ngrams(&syscalls, 3);

// Result: {"open,read,write": 1, "read,write,close": 1}
```

### Detect Anomalies
```rust,ignore
use renacer::sequence::{detect_sequence_anomalies, SequenceAnomaly};

let anomalies = detect_sequence_anomalies(&baseline_ngrams, &current_ngrams, 0.30);

for anomaly in anomalies {
    println!("Ngram: {:?}", anomaly.ngram);        // ["socket", "connect", "send"]
    println!("Frequency: {}", anomaly.frequency); // 24
    println!("Severity: {:?}", anomaly.severity); // High
}
```

## Toyota Way: Andon (Stop the Line)

Sequence anomalies trigger **build-time assertions** that fail CI:

```rust,ignore
#[test]
fn test_no_networking_in_transpiler() {
    let ngrams = extract_ngrams_from_trace("test.trace");

    // FAIL if any networking patterns detected
    assert!(!ngrams.iter().any(|ng|
        ng.contains(&"socket") || ng.contains(&"connect")
    ), "Networking detected in single-shot compile!");
}
```

## Peer-Reviewed Foundation

Based on **Forrest et al. (1996)** "A Sense of Self for Unix Processes" (IEEE S&P):
- N-gram approach for intrusion detection
- Validated on real Unix programs
- 98% detection rate with low false positives

## Testing

13 passing tests covering:
- N-gram extraction (2-grams, 3-grams, 4-grams)
- Anomaly detection with frequency thresholds
- Empty sequence handling
- Performance benchmarks

## Next Steps

- Use [Time-Weighted Attribution](./time-attribution.md) to quantify impact
- Combine with [Syscall Clustering](./syscall-clustering.md) for semantic analysis
- Enable [Regression Detection](./regression-detection.md) for CI/CD
