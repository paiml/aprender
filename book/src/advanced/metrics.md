# Evaluation Metrics

Measure retrieval quality with standard IR metrics.

## Available Metrics

### Recall@k

Fraction of relevant items retrieved in top k:

```rust
let recall = RetrievalMetrics::recall_at_k(&retrieved, &relevant, k);
```

### Precision@k

Fraction of top k items that are relevant:

```rust
let precision = RetrievalMetrics::precision_at_k(&retrieved, &relevant, k);
```

### Mean Reciprocal Rank (MRR)

Position of first relevant result:

```rust
let mrr = RetrievalMetrics::mean_reciprocal_rank(&retrieved, &relevant);
```

### NDCG@k

Normalized Discounted Cumulative Gain:

```rust
let ndcg = RetrievalMetrics::ndcg_at_k(&retrieved, &relevant, k);
```

### Average Precision (AP)

Area under precision-recall curve:

```rust
let ap = RetrievalMetrics::average_precision(&retrieved, &relevant);
```

## Computing All Metrics

```rust
use trueno_rag::metrics::RetrievalMetrics;
use std::collections::HashSet;

let retrieved: Vec<ChunkId> = results.iter().map(|r| r.chunk.id).collect();
let relevant: HashSet<ChunkId> = ground_truth.into_iter().collect();
let k_values = vec![1, 5, 10, 20];

let metrics = RetrievalMetrics::compute(&retrieved, &relevant, &k_values);

println!("MRR: {:.3}", metrics.mrr);
println!("MAP: {:.3}", metrics.map);
println!("Recall@5: {:.3}", metrics.recall.get(&5).unwrap_or(&0.0));
println!("NDCG@10: {:.3}", metrics.ndcg.get(&10).unwrap_or(&0.0));
```

## Aggregating Across Queries

```rust
use trueno_rag::metrics::AggregatedMetrics;

let all_metrics: Vec<RetrievalMetrics> = queries
    .iter()
    .map(|q| evaluate_query(q))
    .collect();

let aggregated = AggregatedMetrics::aggregate(&all_metrics);

println!("Mean MRR: {:.3}", aggregated.mean_mrr);
println!("MAP: {:.3}", aggregated.map);
println!("Queries evaluated: {}", aggregated.query_count);
```

## Best Practices

1. Use multiple metrics for a complete picture
2. Evaluate at multiple k values (1, 5, 10, 20)
3. Track metrics over time for regression detection
4. Compare against baselines (BM25-only, dense-only)
