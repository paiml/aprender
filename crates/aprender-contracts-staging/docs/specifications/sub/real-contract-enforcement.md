# 30. Real Contract Enforcement

## The Diagnosis (v2.3.0 Falsification)

v2.3.0 deployed `pv codegen` macros to 18 repos with 27 call sites.
Every call site was then falsified:

| Metric | Value | Assessment |
|--------|-------|------------|
| Generated macros | 530 | Large number, but all identical |
| Unique assertion body | **1** | `!_contract_input.is_empty()` |
| Domain-specific checks | **0** | No finite check, no shape check, no range check |
| Postcondition checks | **0** | No return-value validation at all |
| Call sites / bindings | 27 / 612 = **4.4%** | 95.6% of bindings have zero enforcement |
| Bugs catchable | **0** | An `is_empty` check catches no numerical, shape, or logic bugs |

**Root cause (five-whys), corrected after falsification:**

1. Every generated macro asserts `!input.is_empty()` regardless of
   what the YAML says.
2. `codegen.rs` line 60 hardcodes `!_contract_input.is_empty()` —
   it never reads the `equation.preconditions` vector content.
3. This was a scaffolding shortcut: the precondition loop counts
   items (`pre_count += equation.preconditions.len()`) but never
   emits the actual expressions.
4. **The YAML already has real preconditions** for core kernels
   (e.g., `x.iter().all(|v| v.is_finite())` in softmax). The data
   is there — the codegen just ignores it.
5. Postcondition codegen (lines 70-88) DOES emit YAML content — so
   the pattern exists, it was just never applied to preconditions.

**The fix is ~10 lines in `codegen.rs`**: loop over
`equation.preconditions` and emit each as a `debug_assert!`, the
same way postconditions already work on lines 78-84.

## Three-Layer Fix

### Layer 1: Fix Codegen to Emit Actual YAML Preconditions

The YAML already contains domain-specific preconditions for core kernels.
The codegen bug (line 60) hardcodes `!is_empty()` instead of emitting them.
Fix: loop over `equation.preconditions` and emit each expression, matching
the existing postcondition pattern (lines 78-84).

Remaining YAML work: audit the ~156 non-core contracts and replace any
remaining `!input.is_empty()` placeholders with real expressions.

**Core kernel examples (9 contracts, highest impact):**

```yaml
# softmax-kernel-v1.yaml
equations:
  softmax:
    preconditions:
      - 'x.iter().all(|v| v.is_finite())'
      - 'x.len() > 0'
    postconditions:
      - '(result.iter().map(|v| *v as f64).sum::<f64>() - 1.0).abs() < 1e-6'
      - 'result.iter().all(|v| *v >= 0.0)'
      - 'result.len() == x.len()'

# matmul-kernel-v1.yaml
equations:
  matmul:
    preconditions:
      - 'a.len() == m * k'
      - 'b.len() == k * n'
      - 'm > 0 && k > 0 && n > 0'
    postconditions:
      - 'result.len() == m * n'

# rmsnorm-kernel-v1.yaml
equations:
  rmsnorm:
    preconditions:
      - 'input.iter().all(|v| v.is_finite())'
      - 'weight.len() == input.len()'
      - 'eps > 0.0'
    postconditions:
      - 'result.iter().all(|v| v.is_finite())'
      - 'result.len() == input.len()'

# cross-entropy-kernel-v1.yaml
equations:
  cross_entropy:
    preconditions:
      - 'logits.len() == targets.len()'
      - 'logits.iter().all(|v| v.is_finite())'
    postconditions:
      - 'result.is_finite()'
      - 'result >= 0.0'

# rope-kernel-v1.yaml
equations:
  rope:
    preconditions:
      - 'x.len() % 2 == 0'
      - 'freqs.len() == x.len() / 2'
    postconditions:
      - 'result.len() == x.len()'

# layernorm-kernel-v1.yaml
equations:
  layernorm:
    preconditions:
      - 'input.iter().all(|v| v.is_finite())'
      - 'gamma.len() == input.len()'
      - 'beta.len() == input.len()'
      - 'eps > 0.0'
    postconditions:
      - 'result.iter().all(|v| v.is_finite())'

# swiglu-kernel-v1.yaml
equations:
  swiglu:
    preconditions:
      - 'gate.len() == up.len()'
    postconditions:
      - 'result.len() == gate.len()'

# attention-kernel-v1.yaml
equations:
  attention:
    preconditions:
      - 'q.len() == k.len()'
      - 'scale > 0.0'
    postconditions:
      - 'result.len() == q.len()'

# embedding-lookup-v1.yaml
equations:
  embedding_lookup:
    preconditions:
      - 'indices.iter().all(|&i| i < vocab_size)'
      - 'table.len() == vocab_size * dim'
    postconditions:
      - 'result.len() == indices.len() * dim'
```

**Precondition taxonomy (what replaces `!input.is_empty()`):**

| Category | Expression Pattern | Contracts |
|----------|-------------------|-----------|
| Finiteness | `x.iter().all(\|v\| v.is_finite())` | softmax, rmsnorm, layernorm, attention |
| Shape match | `a.len() == b.len()` | swiglu, cross_entropy, layernorm |
| Dimension | `a.len() == m * k` | matmul, embedding_lookup |
| Positivity | `eps > 0.0`, `scale > 0.0` | rmsnorm, layernorm, attention |
| Parity | `x.len() % 2 == 0` | rope |
| Bounds | `indices.iter().all(\|&i\| i < vocab_size)` | embedding_lookup |
| Non-empty | `x.len() > 0` | all (retained but not sole check) |

### Layer 2: Postcondition Codegen

`pv codegen` currently generates precondition-only macros. New format
wraps the function body to check both pre and post:

```rust
// CURRENT (v2.3.0): precondition only, generic
macro_rules! contract_pre_softmax {
    ($input:expr) => {{
        debug_assert!(!$input.is_empty());
    }};
}

// NEW (v2.4.0): pre + post, domain-specific
macro_rules! contract_softmax {
    (pre: $x:expr) => {{
        debug_assert!($x.iter().all(|v| v.is_finite()),
            "softmax: all inputs must be finite");
        debug_assert!(!$x.is_empty(),
            "softmax: input must not be empty");
    }};
    (post: $x:expr, $result:expr) => {{
        debug_assert!($result.len() == $x.len(),
            "softmax: output length must match input");
        debug_assert!($result.iter().all(|v| *v >= 0.0),
            "softmax: all outputs must be non-negative");
        debug_assert!(
            ($result.iter().map(|v| *v as f64).sum::<f64>() - 1.0).abs() < 1e-6,
            "softmax: output must sum to 1.0");
    }};
}
```

Usage at call site:

```rust
pub fn softmax(x: &[f32]) -> Vec<f32> {
    contract_softmax!(pre: x);
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = x.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let result: Vec<f32> = exps.iter().map(|v| v / sum).collect();
    contract_softmax!(post: x, &result);
    result
}
```

Zero cost in release builds (`debug_assert!` compiles to nothing).

### Layer 3: Enforcement Quality Metric

Replace the current binary metric ("does a call site exist?") with a
three-level quality scale:

| Level | Name | Criteria | Example |
|-------|------|----------|---------|
| **E0** | Stub | `!input.is_empty()` only | Current state |
| **E1** | Precondition | Domain-specific pre checks | `x.iter().all(\|v\| v.is_finite())` |
| **E2** | Full DbC | Pre + post + real expressions | Pre: finite check, Post: sums to 1.0 |

**Enforcement score** = weighted average across all bindings:

```
enforcement = sum(level_weight * binding_count_at_level) / total_bindings
  where E0 = 0.1, E1 = 0.5, E2 = 1.0
```

v2.3.0 state: 27 bindings at E0, 585 at nothing = **0.004** (0.4%).
v2.4.0 state: 4 bindings at E1 (softmax, rmsnorm, cross_entropy), 23 at E0 = **0.008**.
Target: 9 core kernel contracts at E2, 156 others at E1 = **0.52** (52%).

**v2.4.0 progress:** Codegen now emits real YAML preconditions (32 domain-specific
assertions). 18/18 repos compile, 0 contract-caused test failures across 84,000+ tests.
ruchy: all 20,319 tests pass after fixing opcode table UB + variable binding.

## Implementation Priority (by bug-detection ROI)

| Priority | Contracts | Why | Bindings |
|----------|-----------|-----|----------|
| **P0** | softmax, matmul, rmsnorm | Most call sites, clearest math | ~30 |
| **P1** | cross_entropy, attention, rope, layernorm, swiglu | Core forward pass | ~25 |
| **P2** | embedding_lookup, sampling | Input/output boundaries | ~15 |
| **P3** | All others | Replace `!input.is_empty()` with finiteness/shape | ~540 |

## What Changes in pv codegen

| Feature | v2.3.0 (current) | v2.4.0 (target) |
|---------|-------------------|------------------|
| Reads | `preconditions` only | `preconditions` + `postconditions` |
| Macro format | `contract_pre_<name>!($input)` | `contract_<name>!(pre: $args)` + `contract_<name>!(post: $args, $result)` |
| Assertion body | `!$input.is_empty()` verbatim | Real Rust expressions from YAML |
| Per-binding | No (per-equation) | Yes — reads binding.yaml for param names |
| Quality metric | Binary (exists/not) | E0/E1/E2 scale |

## Falsification Criteria (Section 30)

This section will be falsified by:

1. **Detection test**: Write a function with a known bug (e.g., softmax
   that doesn't subtract max), run with contract. Does the postcondition
   catch it? If not, the contract is useless.
2. **Coverage audit**: After implementation, re-measure enforcement rate.
   Must exceed 50% of bindings at E1+ for core kernel contracts.
3. **False positive rate**: Run full test suite with E2 contracts. If
   >1% of tests fail due to contract noise (not real bugs), the
   assertions are too tight.

## References (Section 30)

- Meyer (1992). "Applying Design by Contract." IEEE Computer 25(10).
  The canonical source on pre/postcondition methodology.
- Findler & Felleisen (2002). "Contracts for Higher-Order Functions."
  ICFP 2002. Blame assignment for contract violations.
- Dimoulas et al. (2011). "Complete Monitors for Behavioral Contracts."
  ESOP 2011. Completeness of runtime monitoring.
- Logozzo & Fahndrich (2012). "On the Relative Completeness of
  Bytecode Analysis Versus Source Code Analysis." CC 2012.
  Why assertion placement matters for detection.
