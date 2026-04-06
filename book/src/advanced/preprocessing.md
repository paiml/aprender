# Query Preprocessing

Improve retrieval by transforming queries before search.

## Available Preprocessors

### PassthroughPreprocessor

No transformation (default):

```rust
let preprocessor = PassthroughPreprocessor;
```

### HyDE (Hypothetical Document Embeddings)

Generates a hypothetical answer for better matching:

```rust
let hyde = HydePreprocessor::new(MockHypotheticalGenerator::new())
    .with_original_query(true);

let queries = hyde.preprocess("What is machine learning?")?;
// ["What is machine learning?", "Machine learning is..."]
```

### Multi-Query Expansion

Expands to multiple related queries:

```rust
let expander = KeywordExpander::new();
let multi = MultiQueryPreprocessor::new(expander)
    .with_max_queries(5)
    .with_original_query(true);

let queries = multi.preprocess("rust programming")?;
```

### Synonym Expansion

Replaces terms with synonyms:

```rust
let expander = SynonymExpander::with_technical_synonyms();
let queries = expander.expand("create a function")?;
// ["make a function", "build a function", ...]
```

### Chained Preprocessor

Combines multiple preprocessors:

```rust
let chained = ChainedPreprocessor::new()
    .add(HydePreprocessor::new(generator))
    .add(MultiQueryPreprocessor::new(expander))
    .with_max_total(10)
    .with_deduplicate(true);
```

## Query Analysis

Understand query intent:

```rust
let analyzer = QueryAnalyzer::new();
let analysis = analyzer.analyze("how to fix error in rust");

println!("Intent: {:?}", analysis.intent);      // Troubleshooting
println!("Keywords: {:?}", analysis.keywords);  // ["fix", "error", "rust"]
println!("Confidence: {}", analysis.confidence);
```

## Query Intents

- `Informational`: Looking for information
- `HowTo`: Looking for instructions
- `Definition`: Looking for definitions
- `Troubleshooting`: Looking to fix problems
- `Comparison`: Comparing options
