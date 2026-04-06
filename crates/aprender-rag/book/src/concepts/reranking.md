# Reranking

Reranking improves result ordering after initial retrieval.

## Why Rerank?

Initial retrieval is fast but may not perfectly order results. Reranking uses more sophisticated scoring:

1. Considers query-document interaction
2. Uses additional signals (lexical overlap, coverage)
3. Can use cross-encoder models

## Available Rerankers

### NoOpReranker

Passes results through unchanged:

```rust
let reranker = NoOpReranker::new();
```

### LexicalReranker

Scores based on lexical features:

```rust
let reranker = LexicalReranker::new()
    .with_weights(0.5, 0.3, 0.2);  // exact, partial, coverage
```

Features:
- Exact phrase matching
- Partial term overlap
- Query term coverage

### MockCrossEncoderReranker

For testing cross-encoder patterns:

```rust
let reranker = MockCrossEncoderReranker::new();
```

### CompositeReranker

Combines multiple rerankers:

```rust
let reranker = CompositeReranker::new()
    .add(LexicalReranker::new(), 0.6)
    .add(MyCustomReranker::new(), 0.4);
```

## Reranker Trait

```rust
pub trait Reranker: Send + Sync {
    fn rerank(
        &self,
        query: &str,
        candidates: &[RetrievalResult],
        top_k: usize,
    ) -> Result<Vec<RetrievalResult>>;
}
```

## Custom Reranker

```rust
struct MyReranker {
    model: MyModel,
}

impl Reranker for MyReranker {
    fn rerank(
        &self,
        query: &str,
        candidates: &[RetrievalResult],
        top_k: usize,
    ) -> Result<Vec<RetrievalResult>> {
        let mut results = candidates.to_vec();

        for result in &mut results {
            let score = self.model.score(query, &result.chunk.content);
            result.rerank_score = Some(score);
        }

        results.sort_by(|a, b|
            b.rerank_score.partial_cmp(&a.rerank_score).unwrap()
        );

        Ok(results.into_iter().take(top_k).collect())
    }
}
```

## Performance Considerations

- Reranking is slower than retrieval
- Typically rerank top 20-50 candidates to get top 5-10
- Consider batching for cross-encoder models
