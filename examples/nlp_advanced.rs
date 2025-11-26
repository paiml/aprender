//! Advanced NLP features: Similarity, Entity Extraction, and Summarization.
//!
//! This example demonstrates three key NLP capabilities:
//! 1. Document similarity measurement (cosine, Jaccard, edit distance)
//! 2. Pattern-based entity extraction (emails, URLs, mentions, hashtags)
//! 3. Extractive text summarization (TextRank, TF-IDF)
//!
//! Run with: cargo run --example nlp_advanced

use aprender::primitives::Vector;
use aprender::text::entities::EntityExtractor;
use aprender::text::similarity::{
    cosine_similarity, edit_distance, jaccard_similarity, top_k_similar,
};
use aprender::text::summarize::{SummarizationMethod, TextSummarizer};
use aprender::text::tokenize::WhitespaceTokenizer;
use aprender::text::vectorize::TfidfVectorizer;

fn main() {
    println!("=== Advanced NLP Features Demo ===\n");

    example1_document_similarity();
    example2_entity_extraction();
    example3_text_summarization();
}

/// Example 1: Document Similarity
///
/// Demonstrates computing similarity between documents using:
/// - Cosine similarity (TF-IDF vectors)
/// - Jaccard similarity (token overlap)
/// - Levenshtein edit distance (character-level)
fn example1_document_similarity() {
    println!("--- Example 1: Document Similarity ---\n");

    let documents = vec![
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks for pattern recognition",
        "Machine learning algorithms learn from data",
        "Natural language processing analyzes human language",
    ];

    // 1. TF-IDF Cosine Similarity
    println!("1. TF-IDF Cosine Similarity:");

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

    // Compute similarity between first document and all others
    println!("\nSimilarity to '{}':", documents[0]);
    for (i, doc) in documents.iter().enumerate().skip(1) {
        let similarity = cosine_similarity(&doc_vectors[0], &doc_vectors[i])
            .expect("Cosine similarity should succeed");
        println!("  vs '{doc}': {similarity:.3}");
    }

    // Find top-k most similar documents to a query
    let query = doc_vectors[0].clone();
    let candidates = doc_vectors[1..].to_vec();
    let top_similar = top_k_similar(&query, &candidates, 2).expect("Top-k should succeed");

    println!("\nTop 2 most similar documents:");
    for (idx, score) in &top_similar {
        println!("  [{}] {:.3}: {}", idx + 1, score, documents[*idx + 1]);
    }

    // 2. Jaccard Similarity (Token Overlap)
    println!("\n2. Jaccard Similarity (Token Overlap):");

    let tokenized: Vec<Vec<&str>> = documents
        .iter()
        .map(|d| d.split_whitespace().collect())
        .collect();

    let jaccard = jaccard_similarity(&tokenized[0], &tokenized[2])
        .expect("Jaccard similarity should succeed");
    println!("  '{}' vs '{}': {:.3}", documents[0], documents[2], jaccard);

    // 3. Edit Distance (String Similarity)
    println!("\n3. Levenshtein Edit Distance:");

    let str1 = "machine learning";
    let str2 = "deep learning";
    let distance = edit_distance(str1, str2).expect("Edit distance should succeed");
    println!("  '{str1}' vs '{str2}': {distance} edits\n");
}

/// Example 2: Entity Extraction
///
/// Demonstrates extracting structured entities from unstructured text:
/// - Email addresses
/// - URLs
/// - Phone numbers
/// - Social media mentions (@username)
/// - Hashtags (#topic)
/// - Named entities (capitalized words)
fn example2_entity_extraction() {
    println!("--- Example 2: Entity Extraction ---\n");

    let text = "Contact @john_doe at john@example.com or visit https://example.com for more info. \
                Call 555-123-4567 for support. Join the discussion with #MachineLearning and #AI. \
                Professor Smith from Stanford University will present at the conference.";

    let extractor = EntityExtractor::new();
    let entities = extractor.extract(text).expect("Extraction should succeed");

    println!("Extracted entities from text:\n");

    if !entities.emails.is_empty() {
        println!("Emails:");
        for email in &entities.emails {
            println!("  - {email}");
        }
        println!();
    }

    if !entities.urls.is_empty() {
        println!("URLs:");
        for url in &entities.urls {
            println!("  - {url}");
        }
        println!();
    }

    if !entities.phone_numbers.is_empty() {
        println!("Phone Numbers:");
        for phone in &entities.phone_numbers {
            println!("  - {phone}");
        }
        println!();
    }

    if !entities.mentions.is_empty() {
        println!("Mentions:");
        for mention in &entities.mentions {
            println!("  - {mention}");
        }
        println!();
    }

    if !entities.hashtags.is_empty() {
        println!("Hashtags:");
        for hashtag in &entities.hashtags {
            println!("  - {hashtag}");
        }
        println!();
    }

    if !entities.named_entities.is_empty() {
        println!("Named Entities (Capitalized Words):");
        for entity in &entities.named_entities {
            println!("  - {entity}");
        }
        println!();
    }

    println!("Total entities found: {}\n", entities.total_count());
}

/// Example 3: Text Summarization
///
/// Demonstrates extractive summarization using:
/// - TF-IDF sentence scoring
/// - TextRank algorithm (graph-based)
/// - Hybrid approach (combining multiple methods)
fn example3_text_summarization() {
    println!("--- Example 3: Text Summarization ---\n");

    let long_text = "Machine learning is a subset of artificial intelligence that focuses on \
                     the development of algorithms and statistical models. These algorithms enable \
                     computer systems to improve their performance on tasks through experience. \
                     Deep learning is a specialized branch of machine learning that uses neural \
                     networks with multiple layers. Neural networks are inspired by the structure \
                     and function of the human brain. They consist of interconnected nodes that \
                     process and transmit information. Natural language processing is another \
                     important area of AI that deals with the interaction between computers and \
                     human language. It enables machines to understand, interpret, and generate \
                     human language in a valuable way. Applications of NLP include machine \
                     translation, sentiment analysis, and chatbots. The field of AI is rapidly \
                     evolving with new breakthroughs happening regularly. Researchers continue to \
                     push the boundaries of what machines can accomplish.";

    // 1. TF-IDF Summarization
    println!("1. TF-IDF Summarization (Top 3 Sentences):\n");

    let tfidf_summarizer = TextSummarizer::new(SummarizationMethod::TfIdf, 3);
    let tfidf_summary = tfidf_summarizer
        .summarize(long_text)
        .expect("TF-IDF summarization should succeed");

    for (i, sentence) in tfidf_summary.iter().enumerate() {
        println!("  [{}] {}", i + 1, sentence);
    }
    println!();

    // 2. TextRank Summarization
    println!("2. TextRank Summarization (Top 3 Sentences):\n");

    let textrank_summarizer = TextSummarizer::new(SummarizationMethod::TextRank, 3)
        .with_damping_factor(0.85)
        .with_max_iterations(100);

    let textrank_summary = textrank_summarizer
        .summarize(long_text)
        .expect("TextRank summarization should succeed");

    for (i, sentence) in textrank_summary.iter().enumerate() {
        println!("  [{}] {}", i + 1, sentence);
    }
    println!();

    // 3. Hybrid Summarization
    println!("3. Hybrid Summarization (Top 2 Sentences):\n");

    let hybrid_summarizer = TextSummarizer::new(SummarizationMethod::Hybrid, 2);
    let hybrid_summary = hybrid_summarizer
        .summarize(long_text)
        .expect("Hybrid summarization should succeed");

    for (i, sentence) in hybrid_summary.iter().enumerate() {
        println!("  [{}] {}", i + 1, sentence);
    }
    println!();

    // Summary Comparison
    println!("Summary Statistics:");
    println!("  Original text: {} sentences", count_sentences(long_text));
    println!("  TF-IDF summary: {} sentences", tfidf_summary.len());
    println!("  TextRank summary: {} sentences", textrank_summary.len());
    println!("  Hybrid summary: {} sentences", hybrid_summary.len());
    println!(
        "  Compression ratio (hybrid): {:.1}%",
        (hybrid_summary.len() as f64 / count_sentences(long_text) as f64) * 100.0
    );
}

/// Helper function to count sentences in text.
fn count_sentences(text: &str) -> usize {
    text.split(['.', '!', '?'])
        .filter(|s| !s.trim().is_empty())
        .count()
}
