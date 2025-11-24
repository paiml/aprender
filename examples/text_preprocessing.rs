//! Text Preprocessing and NLP
//!
//! Demonstrates text preprocessing pipeline for Natural Language Processing:
//! - Tokenization (whitespace, word, character)
//! - Stop words filtering
//! - Stemming (Porter algorithm)
//!
//! # Run
//!
//! ```bash
//! cargo run --example text_preprocessing
//! ```

use aprender::text::stem::{PorterStemmer, Stemmer};
use aprender::text::stopwords::StopWordsFilter;
use aprender::text::tokenize::{CharTokenizer, WhitespaceTokenizer, WordTokenizer};
use aprender::text::Tokenizer;

fn main() {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║              Text Preprocessing & NLP Examples               ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    // Example 1: Tokenization strategies
    example_1_tokenization();

    println!("\n{}", "═".repeat(64));

    // Example 2: Stop words filtering
    example_2_stopwords();

    println!("\n{}", "═".repeat(64));

    // Example 3: Stemming
    example_3_stemming();

    println!("\n{}", "═".repeat(64));

    // Example 4: Complete preprocessing pipeline
    example_4_full_pipeline();
}

/// Example 1: Tokenization Strategies
///
/// Demonstrates different tokenization approaches for text splitting.
fn example_1_tokenization() {
    println!("EXAMPLE 1: Tokenization Strategies");
    println!("{}", "─".repeat(64));

    let text = "Hello, world! Natural Language Processing is amazing.";

    println!("\n📝 Input Text:");
    println!("   \"{text}\"");

    // Whitespace tokenization
    println!("\n🔤 Whitespace Tokenization:");
    println!("   Splits on spaces, preserves punctuation");
    let whitespace_tokenizer = WhitespaceTokenizer::new();
    let whitespace_tokens = whitespace_tokenizer
        .tokenize(text)
        .expect("Tokenization should succeed");
    println!("   Tokens: {whitespace_tokens:?}");
    println!("   Count: {} tokens", whitespace_tokens.len());

    // Word tokenization
    println!("\n🔤 Word Tokenization:");
    println!("   Splits on spaces, separates punctuation");
    let word_tokenizer = WordTokenizer::new();
    let word_tokens = word_tokenizer
        .tokenize(text)
        .expect("Tokenization should succeed");
    println!("   Tokens: {word_tokens:?}");
    println!("   Count: {} tokens", word_tokens.len());

    // Character tokenization
    println!("\n🔤 Character Tokenization:");
    println!("   Splits into individual characters");
    let char_tokenizer = CharTokenizer::new();
    let char_tokens = char_tokenizer
        .tokenize("NLP")
        .expect("Tokenization should succeed");
    println!("   Input: \"NLP\"");
    println!("   Tokens: {char_tokens:?}");
    println!("   Count: {} characters", char_tokens.len());

    println!("\n💡 Analysis:");
    println!("   • Whitespace: Simple, fast, preserves punctuation");
    println!("   • Word: Better for analysis, separates punctuation");
    println!("   • Character: Useful for character-level models");
}

/// Example 2: Stop Words Filtering
///
/// Demonstrates removing common words that add little semantic value.
fn example_2_stopwords() {
    println!("EXAMPLE 2: Stop Words Filtering");
    println!("{}", "─".repeat(64));

    let text = "The quick brown fox jumps over the lazy dog in the garden";

    println!("\n📝 Input Text:");
    println!("   \"{text}\"");

    // Tokenize first
    let tokenizer = WhitespaceTokenizer::new();
    let tokens = tokenizer
        .tokenize(text)
        .expect("Tokenization should succeed");

    println!("\n🔤 Original Tokens ({} words):", tokens.len());
    println!("   {tokens:?}");

    // Filter stop words
    println!("\n🚫 Stop Words Filtering:");
    println!("   Removing common words (the, in, over, etc.)");
    let filter = StopWordsFilter::english();
    let filtered_tokens = filter.filter(&tokens).expect("Filter should succeed");

    println!("\n✅ Filtered Tokens ({} words):", filtered_tokens.len());
    println!("   {filtered_tokens:?}");

    println!("\n💡 Analysis:");
    let reduction = 100.0 * (1.0 - filtered_tokens.len() as f64 / tokens.len() as f64);
    println!("   Token reduction: {reduction:.1}%");
    println!("   Original: {} words", tokens.len());
    println!("   Filtered: {} words", filtered_tokens.len());
    println!(
        "   Removed: {} stop words",
        tokens.len() - filtered_tokens.len()
    );

    // Custom stop words
    println!("\n🎯 Custom Stop Words:");
    println!("   Filtering domain-specific words");
    let custom_filter = StopWordsFilter::new(vec!["fox", "dog", "garden"]);
    let custom_filtered = custom_filter
        .filter(&filtered_tokens)
        .expect("Filter should succeed");
    println!("   Custom list: [\"fox\", \"dog\", \"garden\"]");
    println!("   Result: {custom_filtered:?}");
}

/// Example 3: Stemming
///
/// Demonstrates word normalization using Porter stemmer.
fn example_3_stemming() {
    println!("EXAMPLE 3: Stemming (Word Normalization)");
    println!("{}", "─".repeat(64));

    println!("\n📐 Porter Stemming Algorithm:");
    println!("   Reduces words to their root form");

    let stemmer = PorterStemmer::new();

    // Single word examples
    println!("\n🔍 Single Word Stemming:");
    let examples = vec![
        ("running", "Verb gerund → root"),
        ("studies", "Plural noun → singular"),
        ("better", "Comparative → base"),
        ("flying", "Verb gerund → root"),
        ("happiness", "Noun with suffix → root"),
        ("easily", "Adverb → adjective"),
    ];

    for (word, description) in &examples {
        let stemmed = stemmer.stem(word).expect("Stem should succeed");
        println!("   {word} → {stemmed} ({description})");
    }

    // Multiple word stemming
    println!("\n🔍 Batch Stemming:");
    let words = vec!["running", "jumped", "flying", "studies", "cats", "quickly"];
    let stemmed = stemmer.stem_tokens(&words).expect("Stem should succeed");

    println!("   Original: {words:?}");
    println!("   Stemmed:  {stemmed:?}");

    println!("\n💡 Analysis:");
    println!("   • Normalizes word variations to common root");
    println!("   • Reduces vocabulary size for ML models");
    println!("   • Groups semantically similar words");
    println!("   • Trade-off: May over-stem (\"better\" → \"better\")");
}

/// Example 4: Complete Preprocessing Pipeline
///
/// Demonstrates a full NLP preprocessing pipeline combining all techniques.
fn example_4_full_pipeline() {
    println!("EXAMPLE 4: Complete NLP Preprocessing Pipeline");
    println!("{}", "─".repeat(64));

    let document = "The students are studying machine learning algorithms. \
                    They're analyzing different classification models and \
                    comparing their performances on various datasets.";

    println!("\n📄 Input Document:");
    println!("   \"{document}\"");

    // Step 1: Tokenization
    println!("\n📍 Step 1: Tokenization");
    let tokenizer = WordTokenizer::new();
    let tokens = tokenizer
        .tokenize(document)
        .expect("Tokenization should succeed");
    println!("   Tokenizer: WordTokenizer (separates punctuation)");
    println!("   Tokens: {} items", tokens.len());
    println!("   Sample: {:?}...", &tokens[..10]);

    // Step 2: Lowercase normalization
    println!("\n📍 Step 2: Lowercase Normalization");
    let lowercase_tokens: Vec<String> = tokens.iter().map(|t| t.to_lowercase()).collect();
    println!("   Converted all tokens to lowercase");
    println!("   Sample: {:?}...", &lowercase_tokens[..10]);

    // Step 3: Stop words filtering
    println!("\n📍 Step 3: Stop Words Filtering");
    let filter = StopWordsFilter::english();
    let filtered_tokens = filter
        .filter(&lowercase_tokens)
        .expect("Filter should succeed");
    println!("   Removed common English stop words");
    println!(
        "   Tokens: {} items (removed {})",
        filtered_tokens.len(),
        lowercase_tokens.len() - filtered_tokens.len()
    );
    println!(
        "   Sample: {:?}...",
        &filtered_tokens[..8.min(filtered_tokens.len())]
    );

    // Step 4: Stemming
    println!("\n📍 Step 4: Stemming");
    let stemmer = PorterStemmer::new();
    let stemmed_tokens = stemmer
        .stem_tokens(&filtered_tokens)
        .expect("Stem should succeed");
    println!("   Applied Porter stemmer");
    println!("   Sample transformations:");
    for (original, stemmed) in filtered_tokens.iter().zip(stemmed_tokens.iter()).take(8) {
        if original != stemmed {
            println!("      {original} → {stemmed}");
        }
    }

    // Final output
    println!("\n✅ Final Processed Tokens:");
    println!("   {stemmed_tokens:?}");

    // Statistics
    println!("\n📊 Pipeline Statistics:");
    println!("   Original tokens:      {}", tokens.len());
    println!("   After lowercasing:    {}", lowercase_tokens.len());
    println!("   After stop words:     {}", filtered_tokens.len());
    println!("   After stemming:       {}", stemmed_tokens.len());
    let reduction = 100.0 * (1.0 - stemmed_tokens.len() as f64 / tokens.len() as f64);
    println!("   Total reduction:      {reduction:.1}%");

    println!("\n💡 Pipeline Benefits:");
    println!("   • Normalized text representation");
    println!("   • Reduced vocabulary size");
    println!("   • Ready for feature extraction (TF-IDF, word embeddings)");
    println!("   • Improved ML model performance");

    println!("\n🎯 Next Steps:");
    println!("   • Vectorization (Bag of Words, TF-IDF)");
    println!("   • Feature extraction");
    println!("   • Model training (classification, clustering)");
}
