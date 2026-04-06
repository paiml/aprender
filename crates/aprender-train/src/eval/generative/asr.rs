//! Automatic Speech Recognition (ASR) evaluation metrics
//!
//! Provides Word Error Rate (WER) and Real-Time Factor inverse (RTFx)
//! for evaluating speech recognition and audio processing models.

/// Compute Word Error Rate via word-level Levenshtein edit distance.
///
/// WER = (Substitutions + Deletions + Insertions) / Reference Length
///
/// Returns 0.0 for identical strings. Can exceed 1.0 when hypothesis is
/// much longer than reference.
///
/// # Panics
///
/// Returns `f64::INFINITY` if reference is empty and hypothesis is non-empty.
pub fn word_error_rate(reference: &str, hypothesis: &str) -> f64 {
    let ref_words: Vec<&str> = reference.split_whitespace().collect();
    let hyp_words: Vec<&str> = hypothesis.split_whitespace().collect();

    let n = ref_words.len();
    let m = hyp_words.len();

    if n == 0 && m == 0 {
        return 0.0;
    }
    if n == 0 {
        return f64::INFINITY;
    }

    // Dynamic programming table for edit distance
    let mut dp = vec![vec![0usize; m + 1]; n + 1];

    for i in 0..=n {
        dp[i][0] = i;
    }
    for j in 0..=m {
        dp[0][j] = j;
    }

    for i in 1..=n {
        for j in 1..=m {
            let cost = usize::from(ref_words[i - 1] != hyp_words[j - 1]);
            dp[i][j] = (dp[i - 1][j] + 1) // deletion
                .min(dp[i][j - 1] + 1) // insertion
                .min(dp[i - 1][j - 1] + cost); // substitution
        }
    }

    dp[n][m] as f64 / n as f64
}

/// Compute inverse Real-Time Factor (RTFx).
///
/// RTFx = audio_duration / processing_time
///
/// Higher is better: RTFx=100 means the model processes audio 100x faster
/// than real-time.
///
/// Returns 0.0 if `processing_secs` is zero or negative.
pub fn real_time_factor_inverse(processing_secs: f64, audio_duration_secs: f64) -> f64 {
    if processing_secs <= 0.0 {
        return 0.0;
    }
    audio_duration_secs / processing_secs
}
