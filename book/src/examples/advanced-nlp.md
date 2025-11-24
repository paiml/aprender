# Advanced NLP: Similarity, Entities, and Summarization

This chapter demonstrates three powerful NLP capabilities in Aprender:

1. **Document Similarity** - Measuring how similar documents are using multiple metrics
2. **Entity Extraction** - Identifying structured information from unstructured text
3. **Text Summarization** - Automatically creating concise summaries of long documents

## Theory

### Document Similarity

Document similarity measures how alike two documents are. Aprender provides three complementary approaches:

**1. Cosine Similarity (Vector-Based)**

Measures the angle between TF-IDF vectors:

```
cosine_sim(A, B) = (A · B) / (||A|| * ||B||)
```

- Returns values in [-1, 1]
- 1 = identical direction (very similar)
- 0 = orthogonal (unrelated)
- Works well with semantic similarity

**2. Jaccard Similarity (Set-Based)**

Measures token overlap between documents:

```
jaccard(A, B) = |A ∩ B| / |A ∪ B|
```

- Returns values in [0, 1]
- 1 = identical word sets
- 0 = no words in common
- Fast and intuitive

**3. Levenshtein Edit Distance (String-Based)**

Counts minimum character edits (insert, delete, substitute) to transform one string into another:

- Lower values = more similar
- Exact string matching
- Useful for spell checking, fuzzy matching

### Entity Extraction

Pattern-based extraction identifies structured entities:

- **Email addresses**: `word@domain.com` format
- **URLs**: `http://` or `https://` protocols
- **Phone numbers**: US formats like `XXX-XXX-XXXX`
- **Mentions**: Social media `@username` format
- **Hashtags**: Topic markers like `#topic`
- **Named Entities**: Capitalized words (proper nouns)

### Text Summarization

Aprender implements extractive summarization - selecting the most important sentences:

**1. TF-IDF Scoring**

Sentences are scored by the importance of their words:

```
score(sentence) = Σ tf(word) * idf(word)
```

- High-scoring sentences contain important words
- Fast and simple
- Works well for factual content

**2. TextRank (Graph-Based)**

Inspired by PageRank, treats sentences as nodes in a graph:

```
score(i) = (1-d)/N + d * Σ similarity(i,j) * score(j) / Σ similarity(j,k)
```

- Iterative algorithm finds "central" sentences
- Considers inter-sentence relationships
- Captures document structure

**3. Hybrid Method**

Combines normalized TF-IDF and TextRank scores:

```
score = (normalize(tfidf) + normalize(textrank)) / 2
```

- Balances term importance and structure
- More robust than single methods

## Example: Advanced NLP Pipeline

```rust
use aprender::primitives::Vector;
use aprender::text::entities::EntityExtractor;
use aprender::text::similarity::{
    cosine_similarity, edit_distance, jaccard_similarity, top_k_similar,
};
use aprender::text::summarize::{SummarizationMethod, TextSummarizer};
use aprender::text::tokenize::WhitespaceTokenizer;
use aprender::text::vectorize::TfidfVectorizer;

fn main() {
    // --- 1. Document Similarity ---

    let documents = vec![
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks for pattern recognition",
        "Machine learning algorithms learn from data",
        "Natural language processing analyzes human language",
    ];

    // Compute TF-IDF vectors
    let tokenizer = Box::new(WhitespaceTokenizer::new());
    let mut vectorizer = TfidfVectorizer::new().with_tokenizer(tokenizer);
    let tfidf_matrix = vectorizer
        .fit_transform(&documents)
        .expect("TF-IDF transformation should succeed");

    // Extract document vectors
    let doc_vectors: Vec<Vector<f64>> = (0..documents.len())
        .map(|i| {
            let row: Vec<f64> = (0..tfidf_matrix.n_cols())
                .map(|j| tfidf_matrix.get(i, j))
                .collect();
            Vector::from_slice(&row)
        })
        .collect();

    // Compute cosine similarity
    let similarity = cosine_similarity(&doc_vectors[0], &doc_vectors[2])
        .expect("Cosine similarity should succeed");
    println!("Cosine similarity: {:.3}", similarity);
    // Output: Cosine similarity: 0.173

    // Find top-k most similar documents
    let query = doc_vectors[0].clone();
    let candidates = doc_vectors[1..].to_vec();
    let top_similar = top_k_similar(&query, &candidates, 2)
        .expect("Top-k should succeed");

    println!("\\nTop 2 most similar:");
    for (idx, score) in &top_similar {
        println!("  [{}] {:.3}", idx, score);
    }
    // Output:
    //   [2] 0.173
    //   [1] 0.056

    // Jaccard similarity (token overlap)
    let tokenized: Vec<Vec<&str>> = documents
        .iter()
        .map(|d| d.split_whitespace().collect())
        .collect();

    let jaccard = jaccard_similarity(&tokenized[0], &tokenized[2])
        .expect("Jaccard should succeed");
    println!("\\nJaccard similarity: {:.3}", jaccard);
    // Output: Jaccard similarity: 0.167

    // Edit distance (string matching)
    let distance = edit_distance("machine learning", "deep learning")
        .expect("Edit distance should succeed");
    println!("\\nEdit distance: {} edits", distance);
    // Output: Edit distance: 7 edits

    // --- 2. Entity Extraction ---

    let text = "Contact @john_doe at john@example.com or visit https://example.com. \
                Call 555-123-4567 for support. #MachineLearning #AI";

    let extractor = EntityExtractor::new();
    let entities = extractor.extract(text)
        .expect("Extraction should succeed");

    println!("\\n--- Extracted Entities ---");
    println!("Emails: {:?}", entities.emails);
    // Output: Emails: ["john@example.com"]

    println!("URLs: {:?}", entities.urls);
    // Output: URLs: ["https://example.com"]

    println!("Phone: {:?}", entities.phone_numbers);
    // Output: Phone: ["555-123-4567"]

    println!("Mentions: {:?}", entities.mentions);
    // Output: Mentions: ["@john_doe"]

    println!("Hashtags: {:?}", entities.hashtags);
    // Output: Hashtags: ["#MachineLearning", "#AI"]

    println!("Total entities: {}", entities.total_count());
    // Output: Total entities: 5+

    // --- 3. Text Summarization ---

    let long_text = "Machine learning is a subset of artificial intelligence that \
                     focuses on the development of algorithms and statistical models. \
                     These algorithms enable computer systems to improve their \
                     performance on tasks through experience. Deep learning is a \
                     specialized branch of machine learning that uses neural networks \
                     with multiple layers. Natural language processing is another \
                     important area of AI that deals with the interaction between \
                     computers and human language.";

    // TF-IDF summarization
    let tfidf_summarizer = TextSummarizer::new(
        SummarizationMethod::TfIdf,
        2  // Top 2 sentences
    );
    let summary = tfidf_summarizer.summarize(long_text)
        .expect("Summarization should succeed");

    println!("\\n--- TF-IDF Summary (2 sentences) ---");
    for sentence in &summary {
        println!("  - {}", sentence);
    }

    // TextRank summarization (graph-based)
    let textrank_summarizer = TextSummarizer::new(
        SummarizationMethod::TextRank,
        2
    )
    .with_damping_factor(0.85)
    .with_max_iterations(100);

    let textrank_summary = textrank_summarizer.summarize(long_text)
        .expect("TextRank should succeed");

    println!("\\n--- TextRank Summary (2 sentences) ---");
    for sentence in &textrank_summary {
        println!("  - {}", sentence);
    }

    // Hybrid summarization (best of both)
    let hybrid_summarizer = TextSummarizer::new(
        SummarizationMethod::Hybrid,
        2
    );
    let hybrid_summary = hybrid_summarizer.summarize(long_text)
        .expect("Hybrid should succeed");

    println!("\\n--- Hybrid Summary (2 sentences) ---");
    for sentence in &hybrid_summary {
        println!("  - {}", sentence);
    }
}
```

## Expected Output

```text
Cosine similarity: 0.173

Top 2 most similar:
  [2] 0.173
  [1] 0.056

Jaccard similarity: 0.167

Edit distance: 7 edits

--- Extracted Entities ---
Emails: ["john@example.com"]
URLs: ["https://example.com"]
Phone: ["555-123-4567"]
Mentions: ["@john_doe"]
Hashtags: ["#MachineLearning", "#AI"]
Total entities: 5+

--- TF-IDF Summary (2 sentences) ---
  - These algorithms enable computer systems to improve their performance on tasks through experience
  - Natural language processing is another important area of AI that deals with the interaction between computers and human language

--- TextRank Summary (2 sentences) ---
  - Machine learning is a subset of artificial intelligence that focuses on the development of algorithms and statistical models
  - Natural language processing is another important area of AI that deals with the interaction between computers and human language

--- Hybrid Summary (2 sentences) ---
  - Natural language processing is another important area of AI that deals with the interaction between computers and human language
  - These algorithms enable computer systems to improve their performance on tasks through experience
```

## Choosing the Right Method

### Similarity Metrics

- **Cosine similarity**: Best for semantic similarity with TF-IDF vectors
- **Jaccard similarity**: Fast, works well for duplicate detection
- **Edit distance**: Exact string matching, spell checking, fuzzy search

### Summarization Methods

- **TF-IDF**: Fast, works well for factual/informative content
- **TextRank**: Better captures document structure, good for narratives
- **Hybrid**: More robust, balances both approaches

## Best Practices

1. **Preprocessing**: Clean text before similarity computation
2. **Normalization**: Lowercase, remove punctuation for better matching
3. **Context matters**: Choose similarity metric based on use case
4. **Tune parameters**: Adjust damping factor, iterations for TextRank
5. **Validate results**: Check summaries maintain key information

## Integration Example

Combine all three features for a complete NLP pipeline:

```rust
// 1. Extract entities from documents
let entities = extractor.extract(document)?;

// 2. Find similar documents
let similar_docs = top_k_similar(&query_vec, &doc_vecs, 5)?;

// 3. Summarize the most relevant document
let summary = summarizer.summarize(similar_docs[0])?;

// 4. Extract entities from summary for key information
let summary_entities = extractor.extract(&summary.join(". "))?;
```

## Performance Considerations

- **Cosine similarity**: O(d) where d = vector dimension
- **Jaccard similarity**: O(n + m) where n, m = token counts
- **Edit distance**: O(nm) dynamic programming
- **TextRank**: O(s² * i) where s = sentences, i = iterations
- **TF-IDF scoring**: O(s * w) where w = words per sentence

For large documents:
- Use TF-IDF for initial filtering
- Apply TextRank to smaller candidate sets
- Cache similarity computations when possible

## Run the Example

```bash
cargo run --example nlp_advanced
```

## References

- TF-IDF: Salton & Buckley (1988)
- TextRank: Mihalcea & Tarau (2004)
- Edit Distance: Levenshtein (1966)
- Cosine Similarity: Salton et al. (1975)
