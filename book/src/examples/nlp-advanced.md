# Case Study: Advanced NLP Features

Document similarity, entity extraction, and text summarization.

## Features

1. **Similarity**: Cosine, Jaccard, edit distance
2. **Entity Extraction**: Emails, URLs, mentions, hashtags
3. **Summarization**: TextRank, TF-IDF extractive

## Document Similarity

```rust,ignore
use aprender::text::similarity::{cosine_similarity, jaccard_similarity, edit_distance};
use aprender::text::vectorize::TfidfVectorizer;
use aprender::text::tokenize::WhitespaceTokenizer;

fn main() {
    let docs = vec![
        "machine learning is fascinating",
        "deep learning uses neural networks",
        "cooking recipes are delicious",
    ];

    // TF-IDF vectorization
    let mut vectorizer = TfidfVectorizer::new()
        .with_tokenizer(Box::new(WhitespaceTokenizer::new()));
    let matrix = vectorizer.fit_transform(&docs).unwrap();

    // Cosine similarity
    let vec1 = matrix.row(0);
    let vec2 = matrix.row(1);
    let vec3 = matrix.row(2);

    println!("ML vs DL: {:.3}", cosine_similarity(&vec1, &vec2));  // High
    println!("ML vs Cooking: {:.3}", cosine_similarity(&vec1, &vec3));  // Low

    // Jaccard similarity (token overlap)
    let tokens1: Vec<&str> = docs[0].split_whitespace().collect();
    let tokens2: Vec<&str> = docs[1].split_whitespace().collect();
    println!("Jaccard: {:.3}", jaccard_similarity(&tokens1, &tokens2));

    // Edit distance
    println!("Edit distance: {}", edit_distance("learning", "learner"));
}
```

## Entity Extraction

```rust,ignore
use aprender::text::entities::EntityExtractor;

fn main() {
    let text = "Contact @john at john@example.com or visit https://example.com #rust";

    let extractor = EntityExtractor::new();

    println!("Emails: {:?}", extractor.extract_emails(text));
    println!("URLs: {:?}", extractor.extract_urls(text));
    println!("Mentions: {:?}", extractor.extract_mentions(text));
    println!("Hashtags: {:?}", extractor.extract_hashtags(text));
}
```

Output:
```text
Emails: ["john@example.com"]
URLs: ["https://example.com"]
Mentions: ["@john"]
Hashtags: ["#rust"]
```

## Text Summarization

```rust,ignore
use aprender::text::summarize::{TextSummarizer, SummarizationMethod};

fn main() {
    let article = "Machine learning is transforming industries. \
        Companies use ML for prediction and automation. \
        Deep learning enables image recognition. \
        Natural language processing understands text. \
        The future of AI is promising.";

    let summarizer = TextSummarizer::new(SummarizationMethod::TfIdf);

    // Extract top 2 sentences
    let summary = summarizer.summarize(article, 2).unwrap();
    println!("Summary:\n{}", summary.join(" "));
}
```

## Run

```bash
cargo run --example nlp_advanced
```
