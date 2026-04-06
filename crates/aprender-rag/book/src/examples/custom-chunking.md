# Custom Chunking

Examples of using different chunking strategies.

## Structural Chunking for Markdown

Best for documentation and structured content:

```rust
use trueno_rag::chunk::StructuralChunker;

let chunker = StructuralChunker::new(true, 500);

let doc = Document::new(r#"
# Introduction

Welcome to our guide.

## Getting Started

Follow these steps to begin.

## Advanced Usage

For power users, consider these options.
"#);

let chunks = chunker.chunk(&doc)?;

for chunk in &chunks {
    println!("Headers: {:?}", chunk.metadata.headers);
    println!("Content: {}...\n", &chunk.content[..50.min(chunk.content.len())]);
}
```

## Semantic Chunking for Topic Detection

Groups content by meaning:

```rust
use trueno_rag::chunk::SemanticChunker;
use trueno_rag::embed::MockEmbedder;

let embedder = MockEmbedder::new(384);
let chunker = SemanticChunker::new(embedder, 0.5, 1000);

let doc = Document::new(
    "Rust is a systems programming language. \
     It focuses on safety and performance. \
     Python is great for data science. \
     It has many libraries for machine learning."
);

// Sentences about Rust should cluster together,
// sentences about Python should cluster together
let chunks = chunker.chunk(&doc)?;
```

## Paragraph Chunking for Essays

Groups paragraphs together:

```rust
use trueno_rag::chunk::ParagraphChunker;

let chunker = ParagraphChunker::new(2); // 2 paragraphs per chunk

let doc = Document::new(
    "First paragraph here.\n\n\
     Second paragraph follows.\n\n\
     Third paragraph continues.\n\n\
     Fourth paragraph ends."
);

let chunks = chunker.chunk(&doc)?;
assert_eq!(chunks.len(), 2);
```

## Sentence Chunking for Fine-Grained Retrieval

Groups sentences:

```rust
use trueno_rag::chunk::SentenceChunker;

let chunker = SentenceChunker::new(3, 1); // 3 sentences, 1 overlap

let doc = Document::new(
    "First. Second. Third. Fourth. Fifth. Sixth."
);

let chunks = chunker.chunk(&doc)?;
```

## Combining Strategies

Use structural chunking first, then refine:

```rust
let structural = StructuralChunker::new(true, 2000);
let sentence = SentenceChunker::new(5, 1);

// First pass: structural
let sections = structural.chunk(&doc)?;

// Second pass: sentence-level for large sections
let mut final_chunks = Vec::new();
for section in sections {
    if section.content.len() > 500 {
        let sub_doc = Document::new(&section.content);
        let sub_chunks = sentence.chunk(&sub_doc)?;
        final_chunks.extend(sub_chunks);
    } else {
        final_chunks.push(section);
    }
}
```
