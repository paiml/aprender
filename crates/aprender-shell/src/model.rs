//! N-gram Markov model for command prediction
//!
//! Uses the .apr binary format for efficient model persistence.

use aprender::format::{self, ModelType, SaveOptions};
use aprender::metrics::ranking::RankingMetrics;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

use crate::trie::Trie;

/// N-gram Markov model for shell command prediction
#[derive(Serialize, Deserialize)]
pub struct MarkovModel {
    /// N-gram size
    n: usize,
    /// N-gram counts: context -> (next_token -> count)
    ngrams: HashMap<String, HashMap<String, u32>>,
    /// Command frequency
    command_freq: HashMap<String, u32>,
    /// Prefix trie for fast lookup
    #[serde(skip)]
    trie: Option<Trie>,
    /// Total commands trained on
    total_commands: usize,
    /// Last trained position in history (for incremental updates)
    #[serde(default)]
    last_trained_pos: usize,
}

include!("model_markov.rs");
include!("validation_result.rs");
include!("model_encrypted_roundtrip.rs");
include!("model_corpus_path_load.rs");
