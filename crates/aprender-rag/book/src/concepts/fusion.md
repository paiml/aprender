# Fusion Strategies

Fusion combines results from dense and sparse retrieval.

## Available Strategies

### Reciprocal Rank Fusion (RRF)

Default strategy. Combines ranks rather than scores:

```rust
FusionStrategy::RRF { k: 60.0 }
```

Formula: `score = 1 / (k + rank)`

Best for: General-purpose retrieval

### Linear Combination

Weighted sum of normalized scores:

```rust
FusionStrategy::Linear { dense_weight: 0.7 }
// sparse_weight = 1.0 - dense_weight
```

Best for: When you know one retriever is more reliable

### Distribution-Based Score Fusion (DBSF)

Normalizes scores using z-score before combining:

```rust
FusionStrategy::DBSF
```

Best for: When score distributions differ significantly

### Convex Combination

Similar to linear but with explicit weights:

```rust
FusionStrategy::Convex { dense_weight: 0.6 }
```

### Union

Returns results from either retriever:

```rust
FusionStrategy::Union
```

Best for: Maximum recall

### Intersection

Returns only results found by both retrievers:

```rust
FusionStrategy::Intersection
```

Best for: High precision, lower recall

## Choosing a Strategy

| Scenario | Recommended |
|----------|-------------|
| General use | RRF |
| Known strong retriever | Linear |
| Different score scales | DBSF |
| High recall needed | Union |
| High precision needed | Intersection |

## Custom Fusion

Implement custom fusion logic:

```rust
let dense_results = dense_retriever.retrieve(&query, k)?;
let sparse_results = sparse_retriever.retrieve(&query, k)?;

// Your custom fusion logic
let fused = my_custom_fusion(dense_results, sparse_results);
```
