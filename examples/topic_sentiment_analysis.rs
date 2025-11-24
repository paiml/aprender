//! Topic Modeling and Sentiment Analysis
//!
//! Demonstrates advanced NLP techniques:
//! - Topic modeling with LDA (Latent Dirichlet Allocation)
//! - Sentiment analysis with lexicon-based scoring
//! - Combined topic + sentiment analysis
//!
//! # Run
//!
//! ```bash
//! cargo run --example topic_sentiment_analysis
//! ```

#![allow(non_snake_case)]

use aprender::primitives::Matrix;
use aprender::text::sentiment::{Polarity, SentimentAnalyzer};
use aprender::text::tokenize::WhitespaceTokenizer;
use aprender::text::topic::LatentDirichletAllocation;
use aprender::text::vectorize::CountVectorizer;
use aprender::text::Tokenizer;

fn main() {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║          Topic Modeling & Sentiment Analysis                 ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    // Example 1: Sentiment analysis
    example_1_sentiment_analysis();

    println!("\n{}", "═".repeat(64));

    // Example 2: Topic modeling with LDA
    example_2_topic_modeling();

    println!("\n{}", "═".repeat(64));

    // Example 3: Combined topic + sentiment
    example_3_topic_sentiment();
}

/// Example 1: Lexicon-Based Sentiment Analysis
///
/// Analyzes sentiment using a dictionary of positive/negative words.
fn example_1_sentiment_analysis() {
    println!("EXAMPLE 1: Sentiment Analysis");
    println!("{}", "─".repeat(64));

    let analyzer = SentimentAnalyzer::default();

    println!("\n📊 Sentiment Lexicon:");
    println!("   Size: {} words", analyzer.lexicon_size());
    println!("   Positive examples: excellent (+3.0), good (+2.0), ok (+1.0)");
    println!("   Negative examples: terrible (-3.0), bad (-2.0), weak (-1.0)");

    // Test reviews
    let reviews = vec![
        (
            "This product is amazing and wonderful! Love it!",
            "Review 1",
        ),
        ("Terrible quality, very disappointed and upset.", "Review 2"),
        ("The item arrived on time. No issues.", "Review 3"),
        ("Absolutely fantastic! Best purchase ever!", "Review 4"),
        ("Awful experience. Would not recommend at all.", "Review 5"),
    ];

    println!("\n🔮 Sentiment Scores:");
    for (text, label) in &reviews {
        let score = analyzer.score(text).expect("Score should succeed");
        let polarity = analyzer.classify(text).expect("Classify should succeed");

        let polarity_str = match polarity {
            Polarity::Positive => "Positive ✅",
            Polarity::Negative => "Negative ❌",
            Polarity::Neutral => "Neutral  ⚪",
        };

        println!("\n   {}:", label);
        println!("   Text: \"{}...\"", &text[..40.min(text.len())]);
        println!("   Score: {:.3}", score);
        println!("   → {}", polarity_str);
    }

    // Statistics
    let scores: Vec<f64> = reviews
        .iter()
        .map(|(text, _)| analyzer.score(text).unwrap_or(0.0))
        .collect();

    let pos_count = scores.iter().filter(|&&s| s > 0.05).count();
    let neg_count = scores.iter().filter(|&&s| s < -0.05).count();
    let neu_count = scores.len() - pos_count - neg_count;

    println!("\n📈 Sentiment Distribution:");
    println!("   Positive: {} reviews", pos_count);
    println!("   Negative: {} reviews", neg_count);
    println!("   Neutral:  {} reviews", neu_count);
}

/// Example 2: Topic Modeling with LDA
///
/// Discovers latent topics in a document collection.
fn example_2_topic_modeling() {
    println!("EXAMPLE 2: Topic Modeling (LDA)");
    println!("{}", "─".repeat(64));

    // Document collection (product reviews by category)
    let documents = vec![
        "laptop computer fast performance excellent screen display quality",
        "phone mobile battery camera picture great apps software",
        "computer gaming graphics card processor speed powerful",
        "camera lens zoom photo quality professional digital image",
        "laptop keyboard touchpad battery life portable lightweight",
        "phone screen display resolution touch responsive smooth",
    ];

    println!("\n📚 Document Collection: {} documents", documents.len());
    println!("   Topics: Electronics (laptop, phone, camera)");

    // Create document-term matrix
    println!("\n🔧 Vectorization: Creating document-term matrix");
    let mut vectorizer =
        CountVectorizer::new().with_tokenizer(Box::new(WhitespaceTokenizer::new()));

    let dtm = vectorizer
        .fit_transform(&documents)
        .expect("Vectorization should succeed");

    println!("   Matrix: {} docs × {} terms", dtm.n_rows(), dtm.n_cols());
    println!("   Vocabulary size: {}", vectorizer.vocabulary_size());

    // Train LDA model
    println!("\n🤖 Training: Latent Dirichlet Allocation");
    let n_topics = 3;
    let max_iter = 50;

    let mut lda = LatentDirichletAllocation::new(n_topics).with_random_seed(42);

    lda.fit(&dtm, max_iter).expect("LDA fit should succeed");

    println!("   Topics: {}", n_topics);
    println!("   Iterations: {}", max_iter);

    // Get vocabulary for display
    let mut vocab_pairs: Vec<_> = vectorizer.vocabulary().iter().collect();
    vocab_pairs.sort_by_key(|(_, &idx)| idx);
    let vocabulary: Vec<String> = vocab_pairs
        .iter()
        .map(|(word, _)| (*word).clone())
        .collect();

    // Display top words per topic
    println!("\n📊 Discovered Topics:");
    let top_words = lda
        .top_words(&vocabulary, 5)
        .expect("Top words should succeed");

    for (topic_idx, words) in top_words.iter().enumerate() {
        println!("\n   Topic {} (top 5 words):", topic_idx + 1);
        for (word, score) in words {
            println!("      {}: {:.3}", word, score);
        }
    }

    // Document-topic distribution
    println!("\n🔍 Document-Topic Distribution:");
    let doc_topics = lda.document_topics().expect("Should have doc topics");

    for doc_idx in 0..documents.len() {
        let doc_text = &documents[doc_idx][..40.min(documents[doc_idx].len())];
        println!("\n   Doc {}: \"{}...\"", doc_idx + 1, doc_text);

        let mut topic_probs: Vec<(usize, f64)> = (0..n_topics)
            .map(|topic| (topic, doc_topics.get(doc_idx, topic)))
            .collect();

        topic_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        for (topic_idx, prob) in topic_probs.iter().take(2) {
            println!("      Topic {}: {:.1}%", topic_idx + 1, prob * 100.0);
        }
    }
}

/// Example 3: Combined Topic + Sentiment Analysis
///
/// Analyzes topics and sentiment together for deeper insights.
fn example_3_topic_sentiment() {
    println!("EXAMPLE 3: Combined Topic + Sentiment Analysis");
    println!("{}", "─".repeat(64));

    // Product reviews with different topics and sentiments
    let reviews = vec![
        "laptop excellent performance great battery life wonderful",
        "phone terrible battery awful camera quality disappointing",
        "camera amazing photo quality perfect lens fantastic",
        "laptop slow performance poor screen bad experience",
        "phone great display awesome apps smooth experience",
    ];

    println!("\n📚 Product Reviews: {} items", reviews.len());

    // Step 1: Topic modeling
    println!("\n📍 Step 1: Topic Discovery (LDA)");

    let mut vectorizer =
        CountVectorizer::new().with_tokenizer(Box::new(WhitespaceTokenizer::new()));

    let dtm = vectorizer
        .fit_transform(&reviews)
        .expect("Vectorization should succeed");

    let n_topics = 2; // Positive vs negative aspects
    let mut lda = LatentDirichletAllocation::new(n_topics).with_random_seed(123);

    lda.fit(&dtm, 30).expect("LDA fit should succeed");

    let mut vocab_pairs: Vec<_> = vectorizer.vocabulary().iter().collect();
    vocab_pairs.sort_by_key(|(_, &idx)| idx);
    let vocabulary: Vec<String> = vocab_pairs
        .iter()
        .map(|(word, _)| (*word).clone())
        .collect();

    let top_words = lda
        .top_words(&vocabulary, 4)
        .expect("Top words should succeed");

    println!("   Discovered {} topics:", n_topics);
    for (topic_idx, words) in top_words.iter().enumerate() {
        let words_str: Vec<String> = words
            .iter()
            .map(|(word, score)| format!("{}({:.2})", word, score))
            .collect();
        println!("      Topic {}: {}", topic_idx + 1, words_str.join(", "));
    }

    // Step 2: Sentiment analysis
    println!("\n📍 Step 2: Sentiment Scoring");
    let sentiment_analyzer = SentimentAnalyzer::default();

    let sentiments: Vec<f64> = reviews
        .iter()
        .map(|text| sentiment_analyzer.score(text).unwrap_or(0.0))
        .collect();

    println!("   Computed sentiment scores for all reviews");

    // Step 3: Combined analysis
    println!("\n🔍 Combined Analysis:");

    let doc_topics = lda.document_topics().expect("Should have doc topics");

    for (idx, review) in reviews.iter().enumerate() {
        let review_preview = &review[..45.min(review.len())];
        let sentiment = sentiments[idx];
        let polarity = sentiment_analyzer
            .classify(review)
            .expect("Classify should succeed");

        // Get dominant topic
        let mut topic_probs: Vec<(usize, f64)> = (0..n_topics)
            .map(|topic| (topic, doc_topics.get(idx, topic)))
            .collect();

        topic_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let (dominant_topic, topic_prob) = topic_probs[0];

        let sentiment_label = match polarity {
            Polarity::Positive => format!("Positive ({:.2})", sentiment),
            Polarity::Negative => format!("Negative ({:.2})", sentiment),
            Polarity::Neutral => format!("Neutral ({:.2})", sentiment),
        };

        println!("\n   Review {}:", idx + 1);
        println!("      Text: \"{}...\"", review_preview);
        println!(
            "      Topic: {} ({:.1}%)",
            dominant_topic + 1,
            topic_prob * 100.0
        );
        println!("      Sentiment: {}", sentiment_label);
    }

    // Topic-sentiment correlation
    println!("\n📊 Topic-Sentiment Correlation:");
    for topic_idx in 0..n_topics {
        let mut topic_sentiments = Vec::new();

        for doc_idx in 0..reviews.len() {
            let topic_prob = doc_topics.get(doc_idx, topic_idx);
            if topic_prob > 0.4 {
                // Documents where this topic is dominant
                topic_sentiments.push(sentiments[doc_idx]);
            }
        }

        if !topic_sentiments.is_empty() {
            let avg_sentiment: f64 =
                topic_sentiments.iter().sum::<f64>() / topic_sentiments.len() as f64;

            let sentiment_label = if avg_sentiment > 0.1 {
                "Positive ✅"
            } else if avg_sentiment < -0.1 {
                "Negative ❌"
            } else {
                "Neutral ⚪"
            };

            println!(
                "   Topic {}: {} docs, avg sentiment {:.3} ({})",
                topic_idx + 1,
                topic_sentiments.len(),
                avg_sentiment,
                sentiment_label
            );
        }
    }

    println!("\n💡 Insights:");
    println!("   • Topics capture different product aspects");
    println!("   • Sentiment reveals customer satisfaction per topic");
    println!("   • Combined analysis identifies specific pain points");
    println!("   • Actionable for product improvement prioritization");
}
