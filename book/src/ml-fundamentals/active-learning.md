# Active Learning Theory

Active learning optimizes labeling budgets by selecting the most informative samples for human annotation.

## The Active Learning Loop

```
┌──────────────────────────────────────────────┐
│                                              │
▼                                              │
Unlabeled Pool → Query Strategy → Oracle → Labeled Set
                      │              │           │
                      │         (Human)          │
                      │                          │
                      └─────────────────────────┘
                           Train Model
```

## Why Active Learning?

| Approach | Samples | Accuracy | Cost |
|----------|---------|----------|------|
| Random sampling | 10,000 | 85% | $10,000 |
| Active learning | 2,000 | 85% | $2,000 |

Same accuracy with 80% fewer labels.

## Query Strategies

### 1. Uncertainty Sampling

Select samples where model is most uncertain:

**Least Confidence:**
```
x* = argmax_x (1 - P(ŷ|x))
```

**Margin Sampling:**
```
x* = argmin_x (P(ŷ₁|x) - P(ŷ₂|x))
```

**Entropy:**
```
x* = argmax_x H(P(y|x)) = argmax_x (-Σ P(y|x) log P(y|x))
```

### 2. Query-by-Committee (QBC)

Train multiple models, select where they disagree:

```
Models: M₁, M₂, ..., Mₙ
Vote entropy: x* = argmax_x H(votes)
```

### 3. Expected Model Change

Select samples that would change model most:

```
x* = argmax_x ||∇L(x)||
```

Gradient magnitude indicates influence.

### 4. Diversity Sampling

Ensure selected samples cover feature space:

```
Cluster unlabeled data
Select one sample per cluster
```

### 5. Hybrid Strategies

Combine uncertainty and diversity:

```
Score(x) = α · Uncertainty(x) + (1-α) · Diversity(x)
```

## Batch Active Learning

Select multiple samples per round:

**Greedy:** Select top-k by score
**Diverse:** Cluster-based selection
**Batch-mode:** Joint optimization over batch

## Cold Start Problem

Initial model has no training data:

**Solutions:**
1. Random initial batch
2. Diversity-based selection
3. Transfer from related task
4. Self-supervised pre-training

## Stopping Criteria

When to stop querying:

| Criterion | Description |
|-----------|-------------|
| Budget exhausted | Fixed label budget |
| Performance plateau | Accuracy stops improving |
| Uncertainty threshold | All samples below threshold |
| Committee agreement | Models converge |

## Pool-Based vs Stream-Based

**Pool-Based:**
- Access to entire unlabeled pool
- Can compare and rank samples
- Common in research

**Stream-Based:**
- Samples arrive sequentially
- Must decide immediately
- Common in production

## References

- Settles, B. (2012). "Active Learning." Morgan & Claypool.
- Sener, O., & Savarese, S. (2018). "Active Learning for Convolutional Neural Networks: A Core-Set Approach." ICLR.
