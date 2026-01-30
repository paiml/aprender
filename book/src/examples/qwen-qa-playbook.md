# Case Study: Qwen2.5-Coder QA Playbook Results (2026-01-30)

This chapter documents the qualification testing of Qwen2.5-Coder-1.5B-Instruct using the **apr-model-qa-playbook** framework, which implements Popperian falsification methodology with Toyota Way quality principles.

## Test Summary

| Metric | Result | Status |
|--------|--------|--------|
| **Tool Coverage** | 12/12 (100%) | ✅ PASS |
| **Conversion Tests** | 0/7 (0%) | ❌ BLOCKED |
| **MQS Score** | N/A | ⚠️ Cannot compute (blocked) |
| **Certification** | NOT QUALIFIED | Blocked by GH-185 |
| **APR Version** | 0.2.12 | |
| **Last Requalification** | 2026-01-30 16:55 UTC | GH-185 still open |

## Tool Coverage Testing (F-TOOL-*)

All 12 APR tools verified and passing:

```
Tool                 Status     Exit       Duration
------------------------------------------------------------
inspect              ✅ PASS     0          1352ms
validate             ✅ PASS     0          768ms
check                ✅ PASS     0          2147ms
bench                ✅ PASS     0          594ms
trace-none           ✅ PASS     0          5250ms
trace-basic          ✅ PASS     0          4434ms
trace-layer          ✅ PASS     0          4707ms
trace-payload        ✅ PASS     0          4559ms
profile              ✅ PASS     0          4110ms
profile-ci           ✅ PASS     0          2654ms
profile-ci-assertion ✅ PASS     1          2373ms
profile-ci-p99       ✅ PASS     0          2303ms
------------------------------------------------------------
Total: 12 passed, 0 failed
```

### New Profile CI Features (F-PROFILE-006/007/008)

The `apr profile` command now supports CI mode with assertion checking:

```bash
# CI mode with throughput assertion
apr profile model.gguf --ci --assert-throughput 10.0 --warmup 3 --measure 10

# Output:
CI PROFILE REPORT (PMAT-192)
════════════════════════════════════════════════════════════
  Throughput:  12.8 tok/s
  Latency p50: 156.51 ms
  Latency p99: 156.51 ms

ASSERTIONS
  ✅ PASS throughput: 12.8 tok/s (expected >= 10.0 tok/s)
```

**Available Flags:**
- `--ci` - Enable assertion checking mode
- `--assert-throughput N` - Fail if throughput < N tok/s (exit code 1)
- `--assert-p99 N` - Fail if p99 latency > N ms
- `--assert-p50 N` - Fail if p50 latency > N ms
- `--warmup N` - Warmup passes before measurement
- `--measure N` - Measurement passes for statistics

## Format Conversion Testing (F-CONV-*)

**Status: BLOCKED by GH-185**

All 7 conversion tests failing due to missing embedded tokenizer in APR format:

| Gate | Conversion | Observed Diff | Required | Status |
|------|------------|---------------|----------|--------|
| F-CONV-G-A | GGUF → APR | 0.746 | < 1e-6 | ❌ FAIL |
| F-CONV-A-G | APR → GGUF | 0.560 | < 1e-6 | ❌ FAIL |
| F-CONV-G-S | GGUF → SafeTensors | NaN | < 1e-6 | ❌ FAIL |
| F-CONV-S-G | SafeTensors → GGUF | 0.560 | < 1e-6 | ❌ FAIL |
| F-CONV-A-S | APR → SafeTensors | NaN | < 1e-6 | ❌ FAIL |
| F-CONV-S-A | SafeTensors → APR | 0.748 | < 1e-6 | ❌ FAIL |
| F-CONV-RT-001 | Round-trip | NaN | < 1e-6 | ❌ FAIL |

### Root Cause: GH-185

```bash
# GGUF inference - CORRECT
apr run model.gguf -p "What is 2+2?" --max-tokens 8 --no-gpu
# Output: "4"

# APR inference - WRONG (missing tokenizer)
apr rosetta convert model.gguf model.apr
apr run model.apr -p "What is 2+2?" --max-tokens 8 --no-gpu
# Error: [PMAT-172] APR file missing embedded tokenizer.
# Output: "1. What is the difference between a"
```

**Five-Whys Analysis:**
1. **Why** wrong output? → Tokenizer missing from APR file
2. **Why** missing? → Conversion only copies tensor data
3. **Why** only tensors? → GGUF stores tokenizer in metadata fields
4. **Why** not extracted? → `tokenizer.ggml.*` fields not parsed
5. **ROOT CAUSE:** Converter focuses on weights, not model packaging

## Upstream Issue Status

| Issue | Title | Severity | Status |
|-------|-------|----------|--------|
| #185 | APR missing embedded tokenizer | **P0** | ⏳ OPEN |
| #184 | CI exit code on failure | P2 | ✅ CLOSED |
| #183 | GGUF v3 validation messages | P2 | ✅ FIXED |
| #182 | SafeTensors companion files | P1 | ✅ FIXED |
| #181 | Q4_K_M block alignment | P0 | ✅ FIXED |

## Requalification History

| Date | APR Version | Tool Tests | Conversion | Result |
|------|-------------|------------|------------|--------|
| 2026-01-30 16:55 | 0.2.12 | 12/12 ✅ | 0/7 ❌ | BLOCKED (GH-185) |
| 2026-01-30 (initial) | 0.2.11 | 12/12 ✅ | 0/7 ❌ | BLOCKED (GH-185) |

**Next Steps:** Requalify after GH-185 is merged and `apr` version >= 0.2.13 is released.

## Running the QA Playbook

### Install apr-qa CLI

```bash
git clone https://github.com/paiml/apr-model-qa-playbook
cd apr-model-qa-playbook
cargo build --release
```

### Run Tool Tests

```bash
apr-qa tools /path/to/model.gguf --no-gpu
```

### Run Full Playbook

```bash
apr-qa run playbooks/models/qwen2.5-coder-1.5b.playbook.yaml \
  --subprocess --model-path /path/to/model.gguf --no-gpu
```

### Generate Reports

```bash
apr-qa report output/evidence.json -o output/ --formats all --model "Qwen/Qwen2.5-Coder-1.5B-Instruct"
```

## References

- [apr-model-qa-playbook Specification](https://github.com/paiml/apr-model-qa-playbook/blob/main/docs/specifications/apr-playbook-spec.md)
- [Qwen2.5-Coder Showcase Demo](../../docs/specifications/qwen2.5-coder-showcase-demo.md)
- Karl Popper, "The Logic of Scientific Discovery" (1934)
- Toyota Production System: Jidoka + Poka-Yoke
