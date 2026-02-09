//! Recommendation systems.
//!
//! This module provides collaborative filtering and content-based recommendation algorithms
//! optimized for production cold-start scenarios.
//!
//! # Algorithms
//!
//! - **Content-Based**: Item-to-item similarity using TF-IDF + HNSW
//!
//! # Quick Start
//!
//! ```
//! use aprender::recommend::ContentRecommender;
//!
//! let mut recommender = ContentRecommender::new(16, 200, 0.95);
//!
//! // Add items with text content
//! recommender.add_item("doc1", "machine learning algorithms");
//! recommender.add_item("doc2", "deep learning neural networks");
//! recommender.add_item("doc3", "machine learning applications");
//!
//! // Get recommendations for similar items
//! let recommendations = recommender.recommend("doc1", 2).expect("item exists");
//!
//! assert_eq!(recommendations.len(), 2);
//! // doc3 should be most similar (shares "machine learning")
//! ```

pub mod content_based;

pub use content_based::ContentRecommender;
