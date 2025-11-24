//! Text Classification with TF-IDF
//!
//! Demonstrates end-to-end text classification pipeline:
//! - Text preprocessing (tokenization, stop words, stemming)
//! - Feature extraction (Bag of Words, TF-IDF)
//! - Model training (Logistic Regression, Naive Bayes)
//! - Prediction and evaluation
//!
//! # Run
//!
//! ```bash
//! cargo run --example text_classification
//! ```

#![allow(non_snake_case)]

use aprender::classification::{GaussianNB, LogisticRegression};
use aprender::text::stem::{PorterStemmer, Stemmer};
use aprender::text::stopwords::StopWordsFilter;
use aprender::text::tokenize::WhitespaceTokenizer;
use aprender::text::vectorize::{CountVectorizer, TfidfVectorizer};
use aprender::text::Tokenizer;

fn main() {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║              Text Classification with TF-IDF                 ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    // Example 1: Bag of Words classification
    example_1_bag_of_words();

    println!("\n{}", "═".repeat(64));

    // Example 2: TF-IDF classification
    example_2_tfidf_classification();

    println!("\n{}", "═".repeat(64));

    // Example 3: Full preprocessing pipeline
    example_3_full_pipeline();
}

/// Example 1: Sentiment Classification with Bag of Words
///
/// Simple sentiment analysis using word counts.
fn example_1_bag_of_words() {
    println!("EXAMPLE 1: Sentiment Classification (Bag of Words)");
    println!("{}", "─".repeat(64));

    // Training data: movie reviews
    let train_docs = vec![
        "this movie was excellent and amazing", // Positive
        "great film with wonderful acting",     // Positive
        "fantastic movie loved every minute",   // Positive
        "terrible movie waste of time",         // Negative
        "awful film boring and disappointing",  // Negative
        "horrible acting very bad movie",       // Negative
    ];

    let train_labels = vec![1, 1, 1, 0, 0, 0]; // 1 = positive, 0 = negative

    println!("\n📚 Training Data: {} movie reviews", train_docs.len());
    println!("   Positive: \"{}\"", train_docs[0]);
    println!("   Negative: \"{}\"", train_docs[3]);

    // Create CountVectorizer
    println!("\n🔧 Vectorization: Bag of Words");
    let mut vectorizer = CountVectorizer::new()
        .with_tokenizer(Box::new(WhitespaceTokenizer::new()))
        .with_max_features(20);

    let X_train = vectorizer
        .fit_transform(&train_docs)
        .expect("Vectorization should succeed");

    println!("   Vocabulary size: {}", vectorizer.vocabulary_size());
    println!(
        "   Feature matrix: {} × {}",
        X_train.n_rows(),
        X_train.n_cols()
    );

    // Show vocabulary
    let mut vocab: Vec<_> = vectorizer.vocabulary().iter().collect();
    vocab.sort_by_key(|&(_, idx)| idx);
    println!(
        "\n   Top words: {:?}",
        vocab
            .iter()
            .take(10)
            .map(|(word, _)| *word)
            .collect::<Vec<_>>()
    );

    // Convert to f32 for classifier
    let X_train_f32 = convert_matrix_to_f32(&X_train);

    // Train Gaussian Naive Bayes classifier
    println!("\n🤖 Training: Gaussian Naive Bayes Classifier");
    let mut classifier = GaussianNB::new();
    classifier
        .fit(&X_train_f32, &train_labels)
        .expect("Training should succeed");

    // Test data
    let test_docs = vec![
        "excellent movie great acting", // Should predict positive
        "terrible film very bad",       // Should predict negative
    ];

    let X_test = vectorizer
        .transform(&test_docs)
        .expect("Transform should succeed");
    let X_test_f32 = convert_matrix_to_f32(&X_test);

    let predictions = classifier
        .predict(&X_test_f32)
        .expect("Prediction should succeed");

    println!("\n🔮 Predictions:");
    for (i, doc) in test_docs.iter().enumerate() {
        let sentiment = if predictions[i] == 1 {
            "Positive ✅"
        } else {
            "Negative ❌"
        };
        println!("   \"{doc}\"");
        println!("   → {sentiment}");
    }

    // Training accuracy
    let train_pred = classifier
        .predict(&X_train_f32)
        .expect("Prediction should succeed");
    let accuracy = train_pred
        .iter()
        .zip(&train_labels)
        .filter(|(p, l)| p == l)
        .count() as f64
        / train_labels.len() as f64;

    println!("\n📊 Training Accuracy: {:.1}%", accuracy * 100.0);
}

/// Example 2: Document Classification with TF-IDF
///
/// Topic classification using TF-IDF weighting.
fn example_2_tfidf_classification() {
    println!("EXAMPLE 2: Topic Classification (TF-IDF)");
    println!("{}", "─".repeat(64));

    // Training data: tech vs sports articles
    let train_docs = vec![
        "python programming language machine learning", // Tech
        "artificial intelligence neural networks deep", // Tech
        "software development code rust programming",   // Tech
        "basketball game score team championship",      // Sports
        "football soccer match goal tournament",        // Sports
        "tennis player serves match competition",       // Sports
    ];

    let train_labels = vec![0, 0, 0, 1, 1, 1]; // 0 = tech, 1 = sports

    println!("\n📚 Training Data: {} articles", train_docs.len());
    println!("   Tech:   \"{}\"", train_docs[0]);
    println!("   Sports: \"{}\"", train_docs[3]);

    // Create TF-IDF vectorizer
    println!("\n🔧 Vectorization: TF-IDF");
    let mut vectorizer =
        TfidfVectorizer::new().with_tokenizer(Box::new(WhitespaceTokenizer::new()));

    let X_train = vectorizer
        .fit_transform(&train_docs)
        .expect("Vectorization should succeed");

    println!("   Vocabulary size: {}", vectorizer.vocabulary_size());
    println!(
        "   TF-IDF matrix: {} × {}",
        X_train.n_rows(),
        X_train.n_cols()
    );

    // Show IDF values
    println!("\n   IDF values (sample):");
    let mut vocab: Vec<_> = vectorizer.vocabulary().iter().collect();
    vocab.sort_by_key(|&(_, idx)| idx);
    for (word, &idx) in vocab.iter().take(5) {
        println!("      {}: {:.3}", word, vectorizer.idf_values()[idx]);
    }

    // Convert f64 matrix to f32 for Logistic Regression
    let X_train_f32 = convert_matrix_to_f32(&X_train);

    // Train Logistic Regression
    println!("\n🤖 Training: Logistic Regression");
    let mut classifier = LogisticRegression::new()
        .with_learning_rate(0.1)
        .with_max_iter(100);

    classifier
        .fit(&X_train_f32, &train_labels)
        .expect("Training should succeed");

    // Test data
    let test_docs = vec![
        "programming code algorithm", // Should predict tech
        "basketball score game",      // Should predict sports
        "neural network learning",    // Should predict tech
    ];

    let X_test = vectorizer
        .transform(&test_docs)
        .expect("Transform should succeed");
    let X_test_f32 = convert_matrix_to_f32(&X_test);

    let predictions = classifier.predict(&X_test_f32);

    println!("\n🔮 Predictions:");
    for (i, doc) in test_docs.iter().enumerate() {
        let topic = if predictions[i] == 0 {
            "Tech 💻"
        } else {
            "Sports ⚽"
        };
        println!("   \"{doc}\"");
        println!("   → {topic}");
    }

    // Training accuracy
    let train_pred = classifier.predict(&X_train_f32);
    let accuracy = train_pred
        .iter()
        .zip(&train_labels)
        .filter(|(p, l)| p == l)
        .count() as f64
        / train_labels.len() as f64;

    println!("\n📊 Training Accuracy: {:.1}%", accuracy * 100.0);
}

/// Example 3: Full Preprocessing + Classification Pipeline
///
/// Complete workflow with tokenization, stop words, stemming, TF-IDF, and classification.
fn example_3_full_pipeline() {
    println!("EXAMPLE 3: Full Text Classification Pipeline");
    println!("{}", "─".repeat(64));

    // Raw documents
    let raw_docs = vec![
        "The machine learning algorithms are improving rapidly with deep neural networks",
        "The team scored three goals in the championship football match yesterday",
        "Scientists developed new artificial intelligence systems using transformers",
        "The basketball players trained hard for the upcoming tournament games",
    ];

    let labels = vec![0, 1, 0, 1]; // 0 = tech, 1 = sports

    println!("\n📄 Raw Documents: {} articles", raw_docs.len());

    // Step 1: Tokenization
    println!("\n📍 Step 1: Tokenization");
    let tokenizer = WhitespaceTokenizer::new();
    let mut tokenized_docs = Vec::new();
    for doc in &raw_docs {
        let tokens = tokenizer.tokenize(doc).expect("Tokenize should succeed");
        tokenized_docs.push(tokens);
    }
    println!("   Tokens (doc 1): {} words", tokenized_docs[0].len());

    // Step 2: Lowercase + Stop words
    println!("\n📍 Step 2: Lowercase + Stop Words Filtering");
    let filter = StopWordsFilter::english();
    let mut filtered_docs = Vec::new();
    for tokens in &tokenized_docs {
        let lower: Vec<String> = tokens.iter().map(|t| t.to_lowercase()).collect();
        let filtered = filter.filter(&lower).expect("Filter should succeed");
        filtered_docs.push(filtered);
    }
    println!("   Removed stop words");
    println!(
        "   Before: {} words → After: {} words",
        tokenized_docs[0].len(),
        filtered_docs[0].len()
    );

    // Step 3: Stemming
    println!("\n📍 Step 3: Stemming");
    let stemmer = PorterStemmer::new();
    let mut stemmed_docs = Vec::new();
    for tokens in &filtered_docs {
        let stemmed = stemmer
            .stem_tokens(tokens)
            .expect("Stemming should succeed");
        stemmed_docs.push(stemmed);
    }
    println!("   Applied Porter stemmer");
    println!("   Sample: {:?}", &stemmed_docs[0][..3]);

    // Step 4: Rejoin for vectorization
    let processed_docs: Vec<String> = stemmed_docs.iter().map(|tokens| tokens.join(" ")).collect();

    println!("\n📍 Step 4: TF-IDF Vectorization");
    let mut vectorizer =
        TfidfVectorizer::new().with_tokenizer(Box::new(WhitespaceTokenizer::new()));

    let X = vectorizer
        .fit_transform(&processed_docs)
        .expect("Vectorization should succeed");

    println!(
        "   Vocabulary: {} unique stems",
        vectorizer.vocabulary_size()
    );
    println!("   TF-IDF matrix: {} × {}", X.n_rows(), X.n_cols());

    // Step 5: Classification
    println!("\n📍 Step 5: Classification (Gaussian Naive Bayes)");
    let X_f32 = convert_matrix_to_f32(&X);
    let mut classifier = GaussianNB::new();
    classifier
        .fit(&X_f32, &labels)
        .expect("Training should succeed");

    // Predictions
    let predictions = classifier
        .predict(&X_f32)
        .expect("Prediction should succeed");

    println!("\n🔮 Results:");
    for (i, doc) in raw_docs.iter().enumerate() {
        let topic = if predictions[i] == 0 {
            "Tech"
        } else {
            "Sports"
        };
        let correct = if predictions[i] == labels[i] {
            "✅"
        } else {
            "❌"
        };
        println!("\n   Doc {}: {correct}", i + 1);
        println!("   Text: \"{}...\"", &doc[..50]);
        println!("   Predicted: {topic}");
    }

    // Accuracy
    let accuracy = predictions
        .iter()
        .zip(&labels)
        .filter(|(p, l)| p == l)
        .count() as f64
        / labels.len() as f64;

    println!("\n📊 Pipeline Performance:");
    println!("   Accuracy: {:.1}%", accuracy * 100.0);
    println!(
        "   Vocabulary reduction: {:.1}%",
        (1.0 - vectorizer.vocabulary_size() as f64 / 20.0) * 100.0
    );

    println!("\n💡 Pipeline Summary:");
    println!("   • Tokenization → Stop words → Stemming");
    println!("   • TF-IDF vectorization");
    println!("   • Naive Bayes classification");
    println!("   • Ready for production use!");
}

/// Helper function to convert Matrix<f64> to Matrix<f32>
fn convert_matrix_to_f32(
    m: &aprender::primitives::Matrix<f64>,
) -> aprender::primitives::Matrix<f32> {
    let data: Vec<f32> = m.as_slice().iter().map(|&x| x as f32).collect();
    aprender::primitives::Matrix::from_vec(m.n_rows(), m.n_cols(), data)
        .expect("Conversion should succeed")
}
