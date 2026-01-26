# Case Study: QA Falsification Protocol (PMAT-098)

This chapter documents the Popperian falsification methodology used in the aprender QA infrastructure. The key insight: **a test that cannot fail provides no information**.

## Overview

The QA protocol implements a 21-cell test matrix that systematically validates model inference across:

- **3 Modalities**: `run`, `chat`, `serve`
- **3 Formats**: GGUF, SafeTensors, APR
- **2 Backends**: CPU, GPU
- **Trace variants**: With and without tracing enabled

## The Falsification Methodology

Following Karl Popper's philosophy of science, each test is designed to be **falsifiable**—it must be possible for the test to fail if the system is broken.

### Principle 1: Hang Detection (§7.6)

**Hypothesis**: A command that doesn't complete within 60 seconds is hung.

```rust
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(60);

let output = Command::new("timeout")
    .args(["60", "apr", "run", &model, "--prompt", prompt])
    .output()?;
```

**Falsification**: If a model legitimately requires >60s for a simple prompt, this test produces a false positive. The timeout is tuned for the canonical model (Qwen2.5-Coder-1.5B).

### Principle 2: Garbage Detection (§7.3)

**Hypothesis**: Valid model output has specific characteristics that garbage lacks.

```rust
fn is_garbage_output(output: &str) -> bool {
    // 1. High non-ASCII ratio (>30%)
    let non_ascii = output.chars().filter(|c| !c.is_ascii()).count();
    if non_ascii as f64 / output.len() as f64 > 0.3 {
        return true;
    }

    // 2. Repetition patterns (same char 10+ times)
    if has_repetition_pattern(output, 10) {
        return true;
    }

    // 3. Known garbage patterns
    let garbage_patterns = [
        "�", "\0", "\x00",  // Mojibake, null bytes
        "ÄÄÄÄ", "ÃÃÃÃ",     // Common encoding failures
    ];

    garbage_patterns.iter().any(|p| output.contains(p))
}
```

**Falsification**: Non-English text may trigger false positives. The 30% threshold balances sensitivity vs specificity.

### Principle 3: Answer Verification with Word Boundaries

**Hypothesis**: The model's answer contains the expected value as a complete word.

**Bug Found**: Naive substring matching caused false positives.

```rust
// BUG: "four" matches in "fourteen"
assert!(output.contains("4") || output.contains("four"));

// FIX: Word boundary checking
fn contains_as_word(haystack: &str, needle: &str) -> bool {
    let mut search_start = 0;
    while let Some(pos) = haystack[search_start..].find(needle) {
        let abs_pos = search_start + pos;
        let end_pos = abs_pos + needle.len();

        let left_ok = abs_pos == 0 || {
            let prev_char = haystack[..abs_pos].chars().last().unwrap();
            !prev_char.is_alphanumeric()
        };

        let right_ok = end_pos >= haystack.len() || {
            let next_char = haystack[end_pos..].chars().next().unwrap();
            !next_char.is_alphanumeric()
        };

        if left_ok && right_ok {
            return true;
        }
        search_start = abs_pos + 1;
    }
    false
}
```

## SIGINT Resiliency (PMAT-098-PF)

**Problem**: When users press Ctrl+C during QA tests, orphaned `apr serve` processes remain running.

**Solution**: Layered cleanup with Jidoka-style messaging.

### Layer 1: Process Registry

```rust
static PROCESS_REGISTRY: OnceLock<Arc<Mutex<Vec<u32>>>> = OnceLock::new();

fn register_process(pid: u32) {
    if let Ok(mut registry) = get_registry().lock() {
        registry.push(pid);
    }
}

fn unregister_process(pid: u32) {
    if let Ok(mut registry) = get_registry().lock() {
        registry.retain(|&p| p != pid);
    }
}
```

### Layer 2: ProcessGuard RAII

```rust
struct ProcessGuard {
    child: Option<Child>,
    pid: u32,
}

impl Drop for ProcessGuard {
    fn drop(&mut self) {
        if let Some(ref mut child) = self.child {
            let _ = child.kill();
            let _ = child.wait();
            unregister_process(self.pid);
        }
    }
}
```

### Layer 3: Signal Handler

```rust
fn setup_signal_handler() {
    ctrlc::set_handler(move || {
        let count = kill_all_registered();
        eprintln!(
            "\n[JIDOKA] SIGINT received. Reaping {} active child process(es)...",
            count
        );
        std::process::exit(130);
    }).expect("Signal handler setup");
}
```

The Jidoka message references Toyota's "autonomation" principle—the system stops itself when a problem is detected and signals for human attention.

## Running the QA Suite

### Full Matrix

```bash
cargo run --example qa_run -- --full-matrix
```

Output:
```
╔═════════════════════════════════════════════════════════════╗
║      APR RUN QA - Matrix Falsification Suite                ║
║      PMAT-QA-RUST-001 + PMAT-QA-MATRIX-001                   ║
╚═════════════════════════════════════════════════════════════╝

Testing 21 cell(s):
  R1 apr run × CPU × GGUF → ...
  R2 apr run × CPU × SafeTensors → ...
  ...
```

### Falsification Tests

```bash
cargo run --example qa_falsify
```

Output:
```
=== QA Infrastructure Falsification Suite ===
Testing hang detection...     ✓ PASS
Testing garbage detection...  ✓ PASS
Testing answer verification... ✓ PASS
Testing matrix integrity...   ✓ PASS
Testing SIGINT handler...     ✓ PASS
```

### Ollama Comparison

```bash
cargo run --example qa_run -- --with-ollama
```

## Lessons Learned

1. **Substring matching is insufficient** - Word boundaries matter for answer verification
2. **Documentation drift** - The matrix was documented as 27 cells but was actually 21
3. **Process cleanup is critical** - SIGINT handlers prevent resource leaks
4. **Jidoka messaging** - Clear error messages help debugging

## References

- Karl Popper, "The Logic of Scientific Discovery" (1934)
- Toyota Production System: Jidoka (autonomation)
- PMAT-QA-PROTOCOL-001 specification
