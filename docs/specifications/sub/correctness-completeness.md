# 28. Correctness + Completeness

## The Problem

Grade A (v2.2.0) measures **correctness** — "did you keep the promises
you made?" A repo with 4 bindings that all resolve gets A. A repo with
400 bindings where 390 resolve gets B. The one with 4 is "better" by
the metric but covers almost nothing.

Nobody asks: **what functions SHOULD have contracts but don't?**

This is the distinction between:
- **Correctness**: the contracts you wrote are right
- **Completeness**: you wrote contracts for everything that needs them

Both are required. One without the other is insufficient.

## Meyer's Insight

Meyer never mandated "every routine must have a contract." DbC is
prescriptive about HOW to write contracts, not WHEN. But:

> "One cannot expect large-scale reuse without a precise documentation
> of what every component expects (precondition), what it guarantees
> in return (postcondition) and what general conditions it maintains
> (invariant)."
> — Meyer, *Object-Oriented Software Construction* (1997), Ch. 11

The **class invariant** is Meyer's completeness mechanism — it applies
to ALL exported features of a class, not just the ones the developer
chose to annotate. If you have a class invariant, every routine
implicitly has at least that contract.

Our analog: a repo's `binding.yaml` declares which functions have
contracts (correctness). The **completeness gap** is the set of
`pub fn` declarations in source code that have no binding at all.

## Research Support

**VeriEquivBench (arXiv:2510.06296, 2025)** introduces an "equivalence
score" measuring bidirectional implication between code and spec —
both soundness (spec implies code) AND completeness (code implies spec).

> "Without [a completeness metric], there is no way to guarantee that
> verified code truly aligns with its intended behaviour."

**VERINA (arXiv:2505.23135, 2025)** measures "soundness AND
completeness" of specifications against ground truth, finding the best
model achieves only 52.3% specification completeness.

**CLEVER (arXiv:2505.13938, 2025)** notes:

> "Automatically generated specifications can be incomplete or leaky."

**Coverage metrics for formal verification (Springer, 2003):**

> "Even when a system is proven correct, there is still a question of
> how complete the specification is, and whether it really covers all
> the behaviors of the system."

## The Two Dimensions

```
┌────────────────────────────┬──────────────────────────────────────┐
│     CORRECTNESS            │         COMPLETENESS                 │
│  "contracts are right"     │  "everything has a contract"         │
├────────────────────────────┼──────────────────────────────────────┤
│ Does code match contract?  │ Does every significant fn have one?  │
├────────────────────────────┼──────────────────────────────────────┤
│ Measured by:               │ Measured by:                         │
│ - CD1: declared/resolved   │ - CD2: bound_pub_fns / total_pub_fns│
│ - CB-1203: macros present  │ - CB-1211: pub fn coverage gap (NEW)│
│ - CB-1208: bindings exist  │ - PV-05: completeness gate (NEW)    │
│ - PV-01: pv lint passes    │                                     │
├────────────────────────────┼──────────────────────────────────────┤
│ Eiffel parallel:           │ Eiffel parallel:                     │
│ require/ensure on routines │ class invariant on ALL features      │
├────────────────────────────┼──────────────────────────────────────┤
│ Tool:                      │ Tool:                                │
│ pv score --binding         │ pv score --binding --crate-dir (NEW) │
│ pv verify-bindings         │ pv infer (existing, not in score)    │
│                            │ reverse_coverage.rs (existing)       │
└────────────────────────────┴──────────────────────────────────────┘
```

## How pmat Already Integrates (and the Gap)

pmat has three systems that check provable-contracts:

**TDG (per-file quality):** Computes `provability_factor` from
contract presence — files with contracts get lower tech-debt scores.
But TDG does not flag files WITHOUT contracts. The provability factor
only rewards; it doesn't penalize absence.

**pmat comply (CB-1200..1210):**
- CB-1200: Contracts validate
- CB-1203: Contract-bound fns have `#[contract]` macros
- CB-1208: Bindings resolve in source (ghost detection)
- CB-1209: Trait enforcement
- CB-1210: Precondition quality

All five check the **contracts side** — are the contracts correct?
None check the **code side** — what code lacks contracts?

**pmat infra-score (PV-01..04):**
- PV-01: `pv lint` passes (3pts)
- PV-02: `pv score >= 0.5` (3pts)
- PV-03: Proof level L2+ (2pts)
- PV-04: contracts/ exists (2pts)

All four check contract quality. None check code coverage.

## Falsification of CD2 Ratio Metric (2026-03-28)

The proposed `bound_pub_fns / total_pub_fns` ratio was **falsified**
against all sovereign stack repos before implementation:

| Repo | pub fns | Bindings | Ratio | Impact |
|------|---------|----------|-------|--------|
| aprender | 4,545 | 233 | 5.1% | A → F |
| trueno | 1,084 | 31 | 2.8% | A → F |
| entrenar | 3,266 | 50 | 1.5% | A → F |
| realizar | 3,350 | 58 | 1.7% | A → F |
| forjar | 1,020 | 13 | 1.2% | A → F |

Even with kernel-pattern filtering (denominator = ML-keyword fns only):
aprender 29%, trueno 22%, entrenar 13%, realizar 6%. Still far below
any useful threshold.

**Root cause:** `pub fn` count includes Display/From/Default impls,
builder patterns, trait methods, and thousands of non-kernel functions.
A ratio against this denominator is meaningless.

**Meyer's answer:** Meyer did not solve completeness with metrics.
He solved it with **culture** — in Eiffel, contracts ARE the language
syntax. There is no separate "contract coverage" metric because
contracts are written naturally as part of every routine.

## The Right Model: Critical Path Coverage (CB-1202)

`pmat comply check` already has **CB-1202: Contract Coverage** which
asks the right question: "do you have contracts for the important
stuff?" It checks 16 critical keywords:

```
forward, backward, optimizer, checkpoint, loss, gradient,
sampling, kv_cache, tokenize, quantize, kernel, dispatch,
softmax, matmul, gemm, batch
```

For each keyword, CB-1202 checks: does a `pub fn` containing this
keyword exist in src/ AND does a matching contract exist? This is
**critical path coverage** — not a ratio, but a checklist.

## CD2: Developer-Declared Critical Path (v4, converged)

**Three rounds of falsification** killed three designs:
1. `bound_pub_fns / total_pub_fns` → 1-5% for all repos (F1)
2. ML-only keywords → vacuous 100% for non-ML repos (F2)
3. Domain keyword registry → 8+ domains, keeps growing (F3)

**Converged design:** The developer declares their critical path.
No global keywords. No domain heuristic. The repo says what matters.

```yaml
# contracts/whisper.apr/binding.yaml
version: "1.0.0"
target_crate: whisper-apr
critical_path:              # ← developer declares what matters
  - mel_spectrogram         # audio preprocessing
  - whisper_forward         # model forward pass
  - segment_audio           # VAD segmentation
  - decode_tokens           # beam search decoding
  - vad_detect              # voice activity detection
bindings:
  - contract: ...
```

**CD2 = critical_path entries with matching bindings / len(critical_path)**

- whisper.apr declares 5 critical fns, has contracts for 4 → CD2 = 80%
- aprender declares 15 critical kernels, has contracts for 12 → CD2 = 80%
- presentar declares 8 critical render fns, has contracts for 6 → CD2 = 75%
- Repo with no `critical_path` → CD2 = 0% (no completeness credit)

**Why this works:**
- No vacuous truth — you must declare to get credit
- No domain classification — each repo is its own domain
- No global keyword maintenance — scales to any repo type
- Developer ownership — the person who knows the code decides
- Meyer's class invariant: the developer declares "these are my
  invariants" — the tool verifies they exist

## Implementation (v4)

### Schema: `critical_path` field in `BindingRegistry`

```rust
// In binding.rs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BindingRegistry {
    pub version: String,
    pub target_crate: String,
    #[serde(default)]
    pub critical_path: Vec<String>,  // NEW
    #[serde(default)]
    pub bindings: Vec<KernelBinding>,
}
```

### CD2 computation in `score_codebase_full()`

```rust
// CD2: Critical path completeness
let critical_path_coverage = if binding.critical_path.is_empty() {
    0.0  // no declaration = no completeness credit
} else {
    let covered = binding.critical_path.iter()
        .filter(|cp| binding.bindings.iter()
            .any(|b| b.function.as_deref()
                .is_some_and(|f| f.contains(cp.as_str()))))
        .count();
    covered as f64 / binding.critical_path.len() as f64
};
```

### Grade Definition (v2.3.0)

```
Grade A: correctness >= 90% AND critical_path_coverage >= 75%
Grade B: correctness >= 80% AND critical_path_coverage >= 50%
Grade C: correctness >= 70%
Grade F: correctness < 50%
```

The AND in Grade A is critical. A repo cannot achieve A by being
perfectly correct on 4 functions while ignoring the other 496.

## Self-Falsification (4 rounds, 3 designs killed)

| Round | Proposed | Tested Against | Finding | Resolution |
|-------|----------|----------------|---------|------------|
| R1 | bound/total pub fns | aprender (4,545 fns) | 5.1% — every repo F | **KILLED** |
| R1 | 50% threshold | all repos | impossible | **KILLED** |
| R2 | ML-only keywords | presentar, batuta, forjar | vacuous 100% for 11/15 repos | **KILLED** |
| R3 | 4-domain keywords | whisper.apr, simular, renacer, probar | 8+ domains, keeps growing | **KILLED** |
| R4 | Developer-declared `critical_path` | — | No upstream data yet | **CONVERGED** |

**Key lesson:** Completeness cannot be inferred — it must be declared.
Only the developer knows which functions need contracts. Meyer solved
this with culture (contracts ARE the language), not metrics.

## CLI

```bash
# Correctness + completeness (v2.3.0):
pv score contracts/ --binding contracts/whisper.apr/binding.yaml

# CD1 checks: do declared bindings resolve?
# CD2 checks: do critical_path entries have matching bindings?
# Both require developer to declare in binding.yaml

# pmat comply (enhanced):
pmat comply check
# CB-1208: 38/58 bindings verified in source (correctness)
# CB-1211: 4/5 critical_path fns have contracts (80% completeness) ← NEW
```

## References (Section 28)

- Meyer (1997). *Object-Oriented Software Construction.* Ch. 11:
  Design by Contract. Class invariants as completeness mechanism.
- Meyer (1992). "Applying Design by Contract." IEEE Computer 25(10).
- VERINA (arXiv:2505.23135, 2025). Specification soundness AND
  completeness benchmark.
- VeriEquivBench (arXiv:2510.06296, 2025). Bidirectional equivalence
  score — code implies spec AND spec implies code.
- CLEVER (arXiv:2505.13938, 2025). Human-curated specs for
  completeness; auto-generated specs are "incomplete or leaky."
- Coverage Metrics for Formal Verification (Springer, 2003). "Even
  when proven correct, how complete is the specification?"
