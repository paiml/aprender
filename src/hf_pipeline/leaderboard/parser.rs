//! Leaderboard-to-EvalResult parser
//!
//! Converts HuggingFace leaderboard data into the native evaluation types,
//! enabling comparison of your models against published leaderboard entries.

use super::types::{HfLeaderboard, LeaderboardKind};
use crate::eval::evaluator::{EvalResult, Leaderboard, Metric};
use crate::eval::RougeVariant;

/// Convert an `HfLeaderboard` into a native `Leaderboard` for comparison.
///
/// Maps leaderboard columns to `Metric` variants using kind-specific mappings.
pub fn to_leaderboard(hf: &HfLeaderboard) -> Leaderboard {
    let primary = hf.kind.primary_metric();
    let mut leaderboard = Leaderboard::new(primary);

    for entry in &hf.entries {
        let mut result = EvalResult::new(&entry.model_id);

        for (column, &value) in &entry.scores {
            if let Some(metric) = column_to_metric(&hf.kind, column) {
                result.add_score(metric, value);
            }
        }

        leaderboard.add(result);
    }

    leaderboard
}

/// Map a leaderboard column name to a `Metric` variant.
///
/// Each leaderboard kind has its own column naming conventions.
#[must_use]
pub fn column_to_metric(kind: &LeaderboardKind, column: &str) -> Option<Metric> {
    let col_lower = column.to_lowercase();

    match kind {
        LeaderboardKind::OpenASR => match col_lower.as_str() {
            "wer" | "average_wer" | "word_error_rate" => Some(Metric::WER),
            "rtfx" | "rtf" | "real_time_factor" => Some(Metric::RTFx),
            _ => None,
        },
        LeaderboardKind::OpenLLMv2 => match col_lower.as_str() {
            "mmlu" | "mmlu_pro" | "mmlu_accuracy" => Some(Metric::MMLUAccuracy),
            "accuracy" | "average" | "avg" => Some(Metric::Accuracy),
            _ => None,
        },
        LeaderboardKind::MTEB => match col_lower.as_str() {
            "ndcg@10" | "ndcg_at_10" => Some(Metric::NDCGAtK(10)),
            "accuracy" => Some(Metric::Accuracy),
            _ => None,
        },
        LeaderboardKind::BigCodeBench => match col_lower.as_str() {
            "pass@1" | "pass_at_1" => Some(Metric::PassAtK(1)),
            "pass@10" | "pass_at_10" => Some(Metric::PassAtK(10)),
            _ => None,
        },
        LeaderboardKind::Custom(_) => generic_column_to_metric(&col_lower),
    }
}

/// Best-effort column name → Metric mapping for custom leaderboards
fn generic_column_to_metric(column: &str) -> Option<Metric> {
    match column {
        "accuracy" | "acc" => Some(Metric::Accuracy),
        "wer" | "word_error_rate" => Some(Metric::WER),
        "bleu" => Some(Metric::BLEU),
        "rouge1" | "rouge_1" => Some(Metric::ROUGE(RougeVariant::Rouge1)),
        "rouge2" | "rouge_2" => Some(Metric::ROUGE(RougeVariant::Rouge2)),
        "rougel" | "rouge_l" => Some(Metric::ROUGE(RougeVariant::RougeL)),
        "perplexity" | "ppl" => Some(Metric::Perplexity),
        "mmlu" => Some(Metric::MMLUAccuracy),
        "pass@1" | "pass_at_1" => Some(Metric::PassAtK(1)),
        "ndcg@10" | "ndcg_at_10" => Some(Metric::NDCGAtK(10)),
        _ => None,
    }
}

/// Compare your model's `EvalResult` against a HuggingFace leaderboard.
///
/// Inserts your result into the leaderboard for ranking, returning
/// a sorted `Leaderboard` with your model included.
pub fn compare_with_leaderboard(my_result: &EvalResult, hf: &HfLeaderboard) -> Leaderboard {
    let mut leaderboard = to_leaderboard(hf);
    leaderboard.add(my_result.clone());
    leaderboard
}
