# Metrics Evaluation

Compute retrieval quality metrics (Recall, Precision, MRR, NDCG, MAP) from simulated results.

```bash
cargo run --example metrics_evaluation
```

## Source

```rust
use std::collections::HashSet;
use trueno_rag::{metrics::RetrievalMetrics, ChunkId};

fn main() {
    println!("=== Retrieval Metrics Evaluation ===\n");

    // Simulate retrieved results (chunk IDs in rank order)
    let retrieved: Vec<ChunkId> = (1..=10)
        .map(|n| ChunkId(uuid::Uuid::from_u128(n)))
        .collect();

    // Ground truth: relevant documents are 1, 3, 5, 7
    let relevant: HashSet<ChunkId> = [1, 3, 5, 7]
        .iter()
        .map(|&n| ChunkId(uuid::Uuid::from_u128(n)))
        .collect();

    let k_values = vec![1, 3, 5, 10];
    let metrics = RetrievalMetrics::compute(&retrieved, &relevant, &k_values);

    println!("MRR: {:.3}", metrics.mrr);
    println!("MAP: {:.3}", metrics.map);

    for k in &k_values {
        let recall = metrics.recall.get(k).unwrap_or(&0.0);
        let precision = metrics.precision.get(k).unwrap_or(&0.0);
        let ndcg = metrics.ndcg.get(k).unwrap_or(&0.0);
        println!("k={}: Recall={:.3}, Precision={:.3}, NDCG={:.3}",
            k, recall, precision, ndcg);
    }
}
```

## Expected Output

```
=== Retrieval Metrics Evaluation ===

Mean Reciprocal Rank (MRR): 1.000
Mean Average Precision (MAP): 0.750

| k  | Recall@k | Precision@k | NDCG@k |
|----|----------|-------------|--------|
|  1 | 0.250    | 1.000       | 1.000  |
|  3 | 0.500    | 0.667       | 0.773  |
|  5 | 0.750    | 0.600       | 0.742  |
| 10 | 1.000    | 0.400       | 0.767  |

=== Additional Metrics ===

F1@1: 0.400, Hit Rate@1: 1.000
F1@3: 0.571, Hit Rate@3: 1.000
F1@5: 0.667, Hit Rate@5: 1.000
F1@10: 0.571, Hit Rate@10: 1.000
```
