# Quality Philosophy: The Toyota Way

> Archived from qwen2.5-coder-showcase-demo.md (lines 605-688)

## Quality Philosophy: The Toyota Way

> "Stop the line. Fix it now. Never pass a defect to the next process."
> — Taiichi Ohno, Father of the Toyota Production System

This specification follows the **Toyota Way** quality philosophy. Unlike traditional software development where technical debt is "managed" and defects are "prioritized," we practice **zero tolerance for defects**.

### Core Principles

| Principle | Traditional Approach | Toyota Way |
|-----------|---------------------|------------|
| **SATD** | "We'll fix it later" (TODO/FIXME/HACK) | **FORBIDDEN.** SATD is a defect. Stop the line. |
| **Defects** | Log, triage, prioritize, schedule | **STOP THE LINE.** Fix immediately or mark FALSIFIED. |
| **Failures** | Hide, minimize, spin as "known issues" | **CELEBRATE.** Falsifications demarcate real capabilities. |
| **Metrics** | Optimize for green dashboards | **Genchi Genbutsu.** Go see the real data. |
| **Testing** | Confirm what works | **Falsify.** Actively try to break the system. |

### The Andon Cord: How We Stop the Line

When a defect is discovered, we do NOT:
- Add a TODO comment and continue
- Create a "low priority" ticket for later
- Ship with "known issues" documentation
- Derive metrics that hide the problem

We DO:
- **Mark it FALSIFIED** immediately (public acknowledgment)
- **Run 5-Whys** to find root cause
- **Fix it** before any new feature work
- **Add regression test** to prevent recurrence

### SATD (Self-Admitted Technical Debt) Policy

**SATD markers are defects, not placeholders.**

```rust
// ❌ FORBIDDEN - This is a defect in the codebase
// TODO: Handle edge case for empty input
// FIXME: This will break for large models
// HACK: Workaround for issue #123

// ✅ REQUIRED - Either fix it or mark the feature FALSIFIED
fn process_input(input: &[u8]) -> Result<Output, Error> {
    if input.is_empty() {
        return Err(Error::EmptyInput);  // Handle it NOW
    }
    // ...
}
```

**SATD Scan Enforcement:**
- CI blocks merge if SATD count > 0
- PMAT quality gates enforce zero SATD
- Every PMAT ticket requires falsification audit

### Falsification is Honesty, Not Failure

The ❌ **FALSIFIED** status is **valuable**, not shameful. It:
- Tells users exactly what doesn't work
- Prevents wasted time on broken paths
- Focuses engineering effort on real problems
- Builds trust through transparency

Compare:
- **Dishonest:** "GPU inference: ⚠️ Experimental" (vague, covers up)
- **Honest:** "APR GGUF GPU: ❌ FALSIFIED (Q5_0 dequantization garbage)" (precise, actionable)

---

**Honest QA Assessment (Popperian Falsification) - Updated 2026-02-01 (Round 23 Audit):**
- GGUF CPU: ⚠️ **SUSPECT** (tested with pre-baked HF GGUF, not self-converted)
- GGUF GPU: ⚠️ **SUSPECT** (276.9 tok/s was pre-baked HF GGUF, needs retest with `apr export` output)
- SafeTensors CPU: ✅ **CORROBORATED** (T200: Real Qwen2-0.5B, argmax=262)
- SafeTensors GPU: ✅ **CORROBORATED** (PMAT-120 Fix: QKV bias loading + weight transpose)
- APR CPU (from SafeTensors): ✅ **VERIFIED** (Phase 4.1: "2+2" → "4", matches SafeTensors ground truth)
- APR GPU (from SafeTensors): ✅ **VERIFIED** (2026-01-29: CUDA path verified, argmax=17)
- APR (from GGUF): ⚠️ **SUSPECT** (source GGUF was pre-baked, not self-converted)
- Cross-format parity: ❌ **NOT TESTED** (never compared self-converted GGUF against SafeTensors ground truth)
- `apr check` (10-stage): ⚠️ **FALSE POSITIVE** (GH-190: 10/10 PASS on corrupted model — needs gate improvement)
- `apr profile`: ✅ **VERIFIED** (Real BrickProfiler telemetry)
- `apr chat`: ✅ Verified (Modality Matrix - CPU and GPU)
- **SafeTensors→APR conversion:** ✅ **VERIFIED** (Phase 4.1: identical output)
- **SafeTensors→GGUF conversion:** ❌ **NOT TESTED** (used pre-baked HF GGUF instead)

