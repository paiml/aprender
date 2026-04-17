//! Retrieval Metrics Evaluation Example
//!
//! Run with: cargo run --example metrics_evaluation

use std::collections::HashSet;
use trueno_rag::{metrics::RetrievalMetrics, ChunkId};

fn main() {
    println!("=== Retrieval Metrics Evaluation ===\n");

    // Simulate retrieved results (chunk IDs in rank order)
    let retrieved: Vec<ChunkId> = (1..=10).map(|n| ChunkId(uuid::Uuid::from_u128(n))).collect();

    // Ground truth: relevant documents are 1, 3, 5, 7
    let relevant: HashSet<ChunkId> =
        [1, 3, 5, 7].iter().map(|&n| ChunkId(uuid::Uuid::from_u128(n))).collect();

    println!("Retrieved order: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10");
    println!("Relevant items: 1, 3, 5, 7");
    println!();

    // Compute metrics at different k values
    let k_values = vec![1, 3, 5, 10];

    let metrics = RetrievalMetrics::compute(&retrieved, &relevant, &k_values);

    println!("=== Metrics ===\n");

    println!("Mean Reciprocal Rank (MRR): {:.3}", metrics.mrr);
    println!("Mean Average Precision (MAP): {:.3}", metrics.map);
    println!();

    println!("| k  | Recall@k | Precision@k | NDCG@k |");
    println!("|----|----------|-------------|--------|");

    for k in &k_values {
        let recall = metrics.recall.get(k).unwrap_or(&0.0);
        let precision = metrics.precision.get(k).unwrap_or(&0.0);
        let ndcg = metrics.ndcg.get(k).unwrap_or(&0.0);

        println!("| {:2} | {:.3}    | {:.3}       | {:.3}  |", k, recall, precision, ndcg);
    }

    println!();

    // Additional metrics
    println!("=== Additional Metrics ===\n");

    for k in &[1, 3, 5, 10] {
        let f1 = RetrievalMetrics::f1_at_k(&retrieved, &relevant, *k);
        let hit_rate = RetrievalMetrics::hit_rate_at_k(&retrieved, &relevant, *k);

        println!("F1@{}: {:.3}, Hit Rate@{}: {:.3}", k, f1, k, hit_rate);
    }

    println!();
    println!("Interpretation:");
    println!("- MRR = 1.0 means the first relevant result is at position 1");
    println!("- High Recall means we found most relevant items");
    println!("- High Precision means most retrieved items are relevant");
    println!("- NDCG rewards having relevant items ranked higher");
}
