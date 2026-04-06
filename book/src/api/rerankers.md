# Rerankers API

## Reranker Trait

```rust
pub trait Reranker: Send + Sync {
    fn rerank(
        &self,
        query: &str,
        candidates: &[RetrievalResult],
        top_k: usize
    ) -> Result<Vec<RetrievalResult>>;
}
```

## NoOpReranker

Passes through without modification:

```rust
pub struct NoOpReranker { ... }

impl NoOpReranker {
    pub fn new() -> Self;
}
```

## LexicalReranker

Reranks based on lexical features:

```rust
pub struct LexicalReranker { ... }

impl LexicalReranker {
    pub fn new() -> Self;
    pub fn with_weights(
        self,
        exact_weight: f32,
        partial_weight: f32,
        coverage_weight: f32
    ) -> Self;
}
```

## MockCrossEncoderReranker

For testing cross-encoder patterns:

```rust
pub struct MockCrossEncoderReranker { ... }

impl MockCrossEncoderReranker {
    pub fn new() -> Self;
}
```

## CompositeReranker

Combines multiple rerankers:

```rust
pub struct CompositeReranker { ... }

impl CompositeReranker {
    pub fn new() -> Self;
    pub fn add<R: Reranker + 'static>(self, reranker: R, weight: f32) -> Self;
}
```

## Scoring Details

### LexicalReranker Features

1. **Exact Match** (`exact_weight`): Full query phrase found in document
2. **Partial Match** (`partial_weight`): Individual query terms found
3. **Coverage** (`coverage_weight`): Fraction of query terms present

### Default Weights

```rust
exact_weight: 0.4
partial_weight: 0.35
coverage_weight: 0.25
```

### Custom Weights Example

```rust
let reranker = LexicalReranker::new()
    .with_weights(0.5, 0.3, 0.2);  // Favor exact matches
```
