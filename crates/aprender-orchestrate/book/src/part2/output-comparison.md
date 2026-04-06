# Output Comparison

Output comparison runs both the original and transpiled programs with identical input and verifies that their stdout output matches. This is the most intuitive validation method: if both programs print the same thing, they likely compute the same result.

## Comparison Process

```
Input data ------> Original program ------> Capture stdout A
     |
     +-----------> Transpiled program ----> Capture stdout B
                                                   |
                                            Compare A vs B
                                                   |
                                            Pass / Fail
```

## Byte-Level Comparison

The default comparison mode is byte-level exact match. Each line of stdout from the original program must be identical to the corresponding line from the transpiled program.

Differences are displayed in unified diff format, truncated to 20 lines:

```
--- original output
+++ transpiled output
@@ -3,4 +3,4 @@
 Processing batch 1...
 Processing batch 2...
-Total: 42.0
+Total: 42.00000000000001
 Done.
```

## Numerical Tolerance

Floating-point computations may produce slightly different results due to instruction ordering differences between Python and Rust. Batuta supports configurable tolerance:

| Mode | Tolerance | Use Case |
|------|----------|----------|
| Exact | 0 | Integer output, string output |
| Relative | 1e-6 | Scientific computing, ML inference |
| Absolute | 1e-9 | Financial calculations |
| Custom | User-defined | Domain-specific requirements |

```bash
# Exact comparison (default)
batuta validate --diff-output

# With floating-point tolerance
batuta validate --diff-output --tolerance 1e-6
```

## Structured Output Comparison

For programs that produce structured output (JSON, CSV, XML), Batuta can perform semantic comparison rather than byte-level diff:

```bash
# JSON comparison (ignores key ordering)
batuta validate --diff-output --format json

# CSV comparison (ignores column ordering)
batuta validate --diff-output --format csv
```

## CLI Usage

```bash
# Basic output comparison
batuta validate --diff-output

# With specific input file
batuta validate --diff-output --input test-data.txt

# Compare specific binaries
batuta validate --diff-output \
    --original ./run_original.sh \
    --transpiled ./rust-output/target/release/app
```

## Handling Non-Determinism

Some programs produce non-deterministic output (timestamps, random numbers, process IDs). Strategies for handling this:

1. **Seed random generators** -- pass `--seed 42` to both programs
2. **Filter timestamps** -- `--ignore-pattern '\d{4}-\d{2}-\d{2}'`
3. **Sort output** -- `--sort-lines` for set-like output

If the original program binary is not available, the comparison is skipped with a warning rather than a failure.

---

**Navigate:** [Table of Contents](../SUMMARY.md)
