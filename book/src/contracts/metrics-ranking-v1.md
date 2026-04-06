# metrics-ranking-v1

**Version:** 1.0.0

Ranking metrics -- Hit@K, Reciprocal Rank, MRR, and NDCG@K

## References

- Manning, Raghavan, Schutze (2008) Introduction to Information Retrieval, Ch. 8
- Jarvelin & Kekalainen (2002) Cumulated Gain-Based Evaluation of IR, TOIS

## Equations

### hit_at_k

$$
hit@k = 1 if relevant item in top-k results, 0 otherwise
$$

**Domain:** $ranked list of items, relevant item set, k >= 1$

**Codomain:** $hit@k in {0, 1}$

**Invariants:**

- $hit@k is binary: exactly 0 or 1$
- $hit@k is monotone non-decreasing in k (hit@k <= hit@(k+1))$

### mrr

$$
MRR = (1/|Q|) * sum_{q=1}^{|Q|} RR_q
$$

**Domain:** $Q queries, each with ranked list and relevant items$

**Codomain:** $MRR in [0, 1]$

**Invariants:**

- $MRR in [0, 1] (average of values in [0,1])$
- $MRR = 1 iff all queries have first item relevant$

### ndcg_at_k

$$
NDCG@k = DCG@k / IDCG@k, where DCG@k = sum_{i=1}^{k} rel_i / log2(i+1)
$$

**Domain:** $ranked list with relevance scores, k >= 1$

**Codomain:** $NDCG@k in [0, 1]$

**Invariants:**

- $NDCG@k in [0, 1]$
- $NDCG@k = 1 for perfect ranking (items sorted by relevance)$
- $NDCG@k = 0 when all items have zero relevance$

### reciprocal_rank

```
RR = 1 / rank_of_first_relevant_item, or 0 if none relevant
```

**Domain:** $ranked list, relevant item set$

**Codomain:** $RR in [0, 1]$

**Invariants:**

- $RR in [0, 1]$
- $RR = 1 iff first item is relevant$
- $RR = 0 iff no relevant item in list$

## Proof Obligations

| # | Type | Property | Formal |
|---|------|----------|--------|
| 1 | bound | All metrics in [0, 1] | $hit@k in {0,1}, RR in [0,1], MRR in [0,1], NDCG@k in [0,1]$ |
| 2 | invariant | NDCG perfect ranking | $NDCG@k = 1.0 when items are sorted by decreasing relevance$ |
| 3 | invariant | hit@k binary | $hit@k in {0, 1} for all k and all ranked lists$ |
| 4 | invariant | MRR bounded | $0 <= MRR <= 1 for any set of queries$ |

## Falsification Tests

| ID | Rule | Prediction | If Fails |
|----|------|------------|----------|
| FALSIFY-RANK-001 | hit@k binary | hit@k returns exactly 0 or 1 | hit@k returning fractional or unbounded value |
| FALSIFY-RANK-002 | MRR in [0, 1] | MRR in [0, 1] for any query set | Division error or reciprocal rank exceeding 1 |
| FALSIFY-RANK-003 | NDCG perfect ranking | NDCG@k = 1.0 for perfectly sorted ranking | DCG/IDCG computation mismatch or log base error |
| FALSIFY-RANK-004 | NDCG in [0, 1] | NDCG@k in [0, 1] for any ranking and relevance scores | IDCG computed incorrectly or division by zero when all relevance = 0 |

## Kani Harnesses

| ID | Obligation | Bound | Strategy |
|----|------------|-------|----------|
| KANI-RANK-001 | RANK-BND-001 | 8 | stub_float |
| KANI-RANK-002 | RANK-INV-001 | 8 | stub_float |
| KANI-METRIC-003 | All metrics in [0, 1] | 8 | exhaustive |
| KANI-METRIC-004 | NDCG perfect ranking | 8 | exhaustive |
| KANI-METRIC-005 | hit@k binary | 8 | exhaustive |
| KANI-METRIC-006 | MRR bounded | 8 | stub_float |

## QA Gate

**Ranking Metrics Contract** (F-RANK-001)

Ranking metrics correctness quality gate

**Checks:** hit_binary, mrr_bounded, ndcg_perfect, ndcg_bounded

**Pass criteria:** All 4 falsification tests pass + Kani harnesses verify

