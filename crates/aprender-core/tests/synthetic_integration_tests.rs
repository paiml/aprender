#![allow(clippy::disallowed_methods)]
//! Integration tests for Synthetic Data Generation module.
//!
//! These tests verify end-to-end workflows for synthetic data generation,
//! including EDA, Template, Shell, MixUp, WeakSupervision, and Caching.

use aprender::synthetic::andon::AndonConfig;
use aprender::synthetic::cache::SyntheticCache;
use aprender::synthetic::eda::{EdaConfig, EdaGenerator};
use aprender::synthetic::mixup::{Embeddable, MixUpConfig, MixUpGenerator};
use aprender::synthetic::shell::{ShellGrammar, ShellSample, ShellSyntheticGenerator};
use aprender::synthetic::template::{Template, TemplateGenerator};
use aprender::synthetic::weak_supervision::{
    AggregationStrategy, KeywordLF, LabelVote, WeakSupervisionConfig, WeakSupervisionGenerator,
};
use aprender::synthetic::{SyntheticConfig, SyntheticGenerator};

// ============================================================================
// Test Fixtures
// ============================================================================

/// Sample type implementing Embeddable for MixUp tests.
#[derive(Debug, Clone, PartialEq)]
struct TextSample {
    text: String,
    embedding: Vec<f32>,
}

impl TextSample {
    fn new(text: &str, embedding: Vec<f32>) -> Self {
        Self {
            text: text.to_string(),
            embedding,
        }
    }
}

impl Embeddable for TextSample {
    fn embedding(&self) -> &[f32] {
        &self.embedding
    }

    fn from_embedding(embedding: Vec<f32>, reference: &Self) -> Self {
        Self {
            text: format!("mixed_{}", reference.text),
            embedding,
        }
    }
}

include!("includes/synthetic_integration_eda.rs");
include!("includes/synthetic_integration_cache.rs");
