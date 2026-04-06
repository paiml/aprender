//! Utility functions for transformer training

/// Calculate perplexity from cross-entropy loss
pub fn perplexity(loss: f32) -> f32 {
    loss.exp()
}

/// Calculate tokens per second
pub fn tokens_per_second(num_tokens: usize, elapsed_secs: f64) -> f64 {
    num_tokens as f64 / elapsed_secs
}
