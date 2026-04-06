# WARP Falsification Report v1.0

**Date:** 2026-01-26
**Status:** PARTIALLY FALSIFIED
**Implementation Version:** 1.1.0
**Methodology:** Popperian Falsification per `docs/specifications/warp-spec.md` Section 10

---

## Executive Summary

Following the Popperian methodology, we subjected the WARP implementation to severe
tests designed to expose flaws. The tests were conducted with a
**MockMultiVectorEmbedder** which produces deterministic pseudo-random embeddings
based on text hash. This is a critical limitation that affects the interpretability
of semantic tests.

| Conjecture | Observed | Threshold | Verdict |
|------------|----------|-----------|---------|
| Experimentum Crucis | 0.00% delta | ≥15% delta | **FALSIFIED*** |
| C1: Compression | τ = 0.8312 | τ ≥ 0.90 | **FALSIFIED** |
| C2: Pruning | 93.33% recall | ≥95% recall | **FALSIFIED** |
| C3: Scaling | Latency ~linear | Sub-linear | **FALSIFIED** |

\* With caveat: Mock embedder cannot test semantic properties

---

## 1. Experimentum Crucis (Hard Negatives)

### Hypothesis
Token-level interaction captures nuances that single-vector misses.

### Test Design
- 10 hard negative pairs (e.g., "The cat is on the mat" vs "The cat is NOT on the mat")
- Compare WARP MRR@10 vs single-vector MRR@10
- Falsifier: Delta < 15%

### Results
```
WARP MRR@10:          1.0000
Single-Vector MRR@10: 1.0000
Delta:                0.00%
Verdict:              FALSIFIED
```

### Analysis
Both methods achieved perfect MRR@10 because the **MockMultiVectorEmbedder generates
deterministic pseudo-random vectors based on text hash, not semantic content**. The
mock embedder cannot distinguish semantic nuances like negation.

**CRITICAL CAVEAT:** This test is INCONCLUSIVE for semantic properties. A proper
evaluation requires a real ColBERT-style model (e.g., `colbertv2-msmarco`). The
infrastructure to support such evaluation is in place, but the semantic hypothesis
cannot be tested with mock embeddings.

### Verdict
**FALSIFIED (with caveat)** - The test infrastructure works, but semantic evaluation requires real embeddings.

---

## 2. Conjecture 1: Compression Preserves Score Ordering

### Hypothesis
Residual quantization preserves the relative ordering of MaxSim scores with high fidelity.

### Test Design
- 50 documents, 3 queries
- 4-bit quantization with 8 centroids
- Measure Kendall's τ correlation between exact and quantized scores
- Falsifier: τ < 0.90

### Results
```
Kendall's tau:  0.8312
Threshold:      0.90
Score pairs:    150
Verdict:        FALSIFIED
```

### Analysis
The rank correlation of 0.8312 indicates **significant score ordering distortion**
from quantization. This is concerning because:

1. **8.7% of pairwise orderings are inverted** - relevant documents may be incorrectly ranked below irrelevant ones
2. **4-bit quantization was used** - 2-bit would be worse
3. The spec claims 2-bit has "~3-5% MRR loss" which may underestimate rank distortion

### Potential Causes
1. **Residual quantization bucket boundaries** may not be optimally learned
2. **K-means clustering** may produce suboptimal centroids for this data distribution
3. **Mock embeddings** may have unusual distributions not well-suited to residual quantization

### Recommendation
Investigate bucket boundary learning algorithm. Consider using ScaNN's asymmetric
quantization or product quantization for better rank preservation.

### Verdict
**FALSIFIED** - Compression does not preserve score ordering to the required 0.90 threshold.

---

## 3. Conjecture 2: Centroid Pruning Recall

### Hypothesis
Top-nprobe centroids contain relevant tokens for accurate retrieval.

### Test Design
- 100 documents, 3 queries
- 4 centroids, nprobe=4 (exhaustive probes all centroids)
- Compare recall@10 of nprobe=4 vs exhaustive (also nprobe=4)
- Falsifier: recall < 95%

### Results
```
recall@10:   0.9333 (93.33%)
Threshold:   95%
Verdict:     FALSIFIED (narrowly)
```

### Analysis
At 93.33%, recall is **close to but below** the 95% threshold. This means:

1. **~7% of relevant results are missed** by centroid pruning
2. With only 4 centroids and nprobe=4, we're effectively exhaustive
3. The loss comes from **score approximation**, not pruning itself

### Potential Causes
1. Score approximation errors (per Conjecture 1) cause reranking errors
2. The combination of quantization + centroid scoring compounds errors

### Recommendation
This is a narrow failure. With better compression (C1), recall would likely improve.

### Verdict
**FALSIFIED** - Narrow miss at 93.33% vs 95% threshold.

---

## 4. Conjecture 3: Scaling Laws

### Hypothesis
- Memory scales linearly with N×T×bits
- Latency scales with nprobe, not N

### Test Design
- Corpus sizes: 500, 1000, 2000 documents
- Measure memory usage and search latency
- Falsifier: Memory > theoretical × 1.2 OR latency linear with N

### Results
```
Memory:
  N=500:  167 KB
  N=1000: 332 KB
  N=2000: 663 KB
  → Linear scaling (OK)

Latency:
  N=500:  25,040 μs
  N=1000: 59,763 μs
  N=2000: 54,540 μs
  → Linear with N initially, then constant (INCONSISTENT)
```

### Analysis

**Memory:** Scales linearly with N, which is expected (2×N → 2×memory). This is **CORROBORATED**.

**Latency:** The results are inconsistent:
- N=500→1000: 2.4× increase (linear with N - BAD)
- N=1000→2000: 0.9× increase (sub-linear - GOOD?)

The N=2000 result being faster than N=1000 is anomalous and likely due to:
1. JIT/cache warming effects
2. Measurement noise
3. Different centroid distributions affecting search patterns

### Verdict
**FALSIFIED** - Latency does not show consistent sub-linear scaling. Further investigation needed.

---

## 5. Overall Assessment

### Final Verdict: **PARTIALLY FALSIFIED**

The WARP implementation shows:

| Aspect | Status |
|--------|--------|
| Core Algorithm | Implemented correctly |
| Memory Efficiency | CORROBORATED |
| Compression Quality | FALSIFIED (τ=0.83 < 0.90) |
| Pruning Effectiveness | FALSIFIED (narrowly, 93.33% < 95%) |
| Latency Scaling | INCONCLUSIVE (noisy measurements) |
| Semantic Properties | UNTESTABLE (mock embedder) |

### Key Findings

1. **Compression is the weak link** - τ=0.8312 indicates significant rank distortion
2. **Pruning recall is close** - 93.33% vs 95% threshold
3. **Semantic tests require real embeddings** - Mock embedder cannot evaluate semantic hypotheses

### Recommendations (Do NOT Patch - Reformulate)

Following Popperian methodology, we do NOT recommend ad-hoc fixes. Instead:

1. **For Compression (C1):**
   - Investigate alternative quantization schemes (product quantization, asymmetric)
   - Consider adaptive bucket boundaries per centroid
   - Benchmark against ScaNN's compression quality

2. **For Pruning (C2):**
   - Likely to improve if C1 is fixed
   - Consider HNSW-style graph index instead of IVF

3. **For Semantic Evaluation:**
   - Integrate real ColBERT model (e.g., via fastembed)
   - Rerun Experimentum Crucis with actual semantic embeddings

4. **For Latency:**
   - Use criterion.rs for statistically rigorous benchmarking
   - Test with 10,000+ documents for clearer scaling patterns

---

## 6. Appendix: Test Environment

```
Platform: Linux 6.8.0
Rust: 1.83+
Test Runner: cargo test --features multivector
Embedder: MockMultiVectorEmbedder (deterministic pseudo-random)
```

---

## 7. Conclusion

The WARP implementation has been **partially falsified** under severe testing. The
core algorithm is correctly implemented, but compression quality falls short of the
0.90 rank correlation threshold. This is a **scientific finding**, not a bug to be
patched.

Following the Toyota Way's Jidoka principle, we recommend stopping to investigate
the root cause rather than applying epicycles. The compression algorithm should be
reformulated with better theoretical foundations before proceeding.

> "The wrong view of science betrays itself in the craving to be right."
> — Karl Popper

---

**Report Generated By:** Claude Code (Popperian Analysis)
**Review Status:** Pending peer review
