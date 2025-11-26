# Case Study: Topic Modeling & Sentiment Analysis

Discover topics in documents and analyze sentiment.

## Features

1. **LDA Topic Modeling**: Find hidden topics in corpus
2. **Sentiment Analysis**: Lexicon-based polarity scoring
3. **Combined Analysis**: Topics + sentiment per document

## Sentiment Analysis

```rust,ignore
use aprender::text::sentiment::{SentimentAnalyzer, Polarity};

fn main() {
    let analyzer = SentimentAnalyzer::new();

    let reviews = vec![
        "This product is amazing! Absolutely love it!",
        "Terrible experience. Complete waste of money.",
        "It's okay, nothing special but works fine.",
    ];

    for review in &reviews {
        let result = analyzer.analyze(review);
        let emoji = match result.polarity {
            Polarity::Positive => "😊",
            Polarity::Negative => "😞",
            Polarity::Neutral => "😐",
        };
        println!("{} Score: {:.2} - {}", emoji, result.score, review);
    }
}
```

Output:
```text
😊 Score: 0.85 - This product is amazing! Absolutely love it!
😞 Score: -0.72 - Terrible experience. Complete waste of money.
😐 Score: 0.12 - It's okay, nothing special but works fine.
```

## Topic Modeling with LDA

```rust,ignore
use aprender::text::topic::LatentDirichletAllocation;
use aprender::text::vectorize::CountVectorizer;
use aprender::text::tokenize::WhitespaceTokenizer;

fn main() {
    let documents = vec![
        "machine learning algorithms data science",
        "neural networks deep learning training",
        "cooking recipes kitchen ingredients",
        "baking bread flour yeast oven",
        "stocks market trading investment",
        "bonds portfolio financial returns",
    ];

    // Vectorize
    let mut vectorizer = CountVectorizer::new()
        .with_tokenizer(Box::new(WhitespaceTokenizer::new()));
    let doc_term_matrix = vectorizer.fit_transform(&documents).unwrap();

    // Find 3 topics
    let mut lda = LatentDirichletAllocation::new(3)
        .with_max_iter(100)
        .with_random_state(42);

    lda.fit(&doc_term_matrix).unwrap();

    // Print top words per topic
    let vocab: Vec<&str> = vectorizer.vocabulary()
        .iter()
        .map(|(k, _)| k.as_str())
        .collect();

    for (i, topic) in lda.topics().iter().enumerate() {
        let top_words = lda.top_words(topic, &vocab, 5);
        println!("Topic {}: {:?}", i, top_words);
    }
}
```

Output:
```text
Topic 0: ["learning", "machine", "neural", "deep", "data"]
Topic 1: ["cooking", "recipes", "baking", "bread", "flour"]
Topic 2: ["stocks", "market", "trading", "financial", "bonds"]
```

## Combined Analysis

Analyze both topic and sentiment per document:

```rust,ignore
for doc in &documents {
    let sentiment = analyzer.analyze(doc);
    let topic_dist = lda.transform_single(doc);
    let dominant_topic = topic_dist.argmax();

    println!("Doc: '{}...'", &doc[..30.min(doc.len())]);
    println!("  Topic: {} | Sentiment: {:.2}", dominant_topic, sentiment.score);
}
```

## Run

```bash
cargo run --example topic_sentiment_analysis
```
