# backend-dispatch-v1

**Version:** 1.0.0

Backend dispatch thresholds, garbage oracle, and BPE roundtrip

## References

- Qwen2.5-Coder Showcase Spec — backend dispatch
- Qwen3 Performance Parity Spec — QK norm score bound

## Dependency Graph

```mermaid
graph LR
    gpu_multi_backend_parity_v1["gpu-multi-backend-parity-v1"] --> backend_dispatch_v1["backend-dispatch-v1"]
```

## Equations

### garbage_oracle

$$
is_garbage(text) = repetition_ratio > 0.3 OR unique_chars < 10
$$

**Domain:** $text \in String$

**Invariants:**

- $Highly repetitive text is garbage$
- $Very low character diversity is garbage$

### gpu_threshold

$$
dispatch(n) = GPU if n >= 100_000 else CPU
$$

**Domain:** $n = tensor element count$

**Invariants:**

- $Threshold is monotonic: GPU-eligible implies all larger tensors GPU-eligible$

### qk_norm_score_bound

```
|pre_softmax_score| <= sqrt(head_dim)
```

**Domain:** $After QK normalization$

**Invariants:**

- $Bounded by sqrt of head dimension$
- $Prevents attention score explosion$

### simd_only_threshold

$$
dispatch(n) = SIMD_only if n < 1_000 else SIMD+threading
$$

**Domain:** $n = tensor element count$

**Invariants:**

- $Small tensors avoid threading overhead$

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | monotonicity | GPU threshold monotonic | $n1 >= threshold AND n2 > n1 => n2 >= threshold$ |
| 2 | invariant | Garbage oracle detects repetition | $repetition_ratio > 0.3 => is_garbage$ |
| 3 | bound | QK norm score bound | $\|score\| <= \sqrt{d_k} after L2 normalization$ |
| 4 | equivalence | BPE roundtrip | `decode(encode(text)) == text for representable strings` |
| 5 | equivalence | SIMD dispatch equivalence |  |

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-BD-001 | GPU threshold | Threshold is monotonic step function | Non-monotonic dispatch decision |
| FALSIFY-BD-002 | Garbage oracle | Repeated characters flagged as garbage | Oracle misses degenerate text |
| FALSIFY-BD-003 | QK norm bound | Dot product of unit vectors bounded by sqrt(d) | L2 normalization not applied |
| FALSIFY-BD-004 | BPE roundtrip | encode(decode(tokens)) == tokens | Lossy tokenization roundtrip |
| FALSIFY-BD-005 | SIMD dispatch equivalence | SIMD backend produces same results as scalar | SIMD backend diverges from scalar |

## Kani Harnesses

| ID | Obligation | Bound | Strategy |
|----|------------|-------|----------|
| KANI-BD-001 | BD-MON-001 | 4 | bounded_int |
| KANI-BACKEN-002 | GPU threshold monotonic | 8 | exhaustive |
| KANI-BACKEN-003 | Garbage oracle detects repetition | 8 | exhaustive |
| KANI-BACKEN-004 | QK norm score bound | 8 | stub_float |
| KANI-BACKEN-005 | BPE roundtrip | 8 | exhaustive |
| KANI-BACKEN-006 | SIMD dispatch equivalence | 8 | exhaustive |

## QA Gate

**Backend Dispatch Contract** (F-BD-001)

Dispatch threshold and oracle quality gate

**Checks:** gpu_threshold_monotonic, garbage_oracle, qk_norm_bound, bpe_roundtrip

**Pass criteria:** All 5 falsification tests pass

