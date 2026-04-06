//! Falsification tests for HF leaderboard and publishing integration
//!
//! Tests the contracts between leaderboard parsing, model card generation,
//! and submission formatting against HuggingFace Hub specifications.

use crate::eval::evaluator::{EvalResult, Leaderboard, Metric};
use crate::eval::RougeVariant;

use super::parser::{column_to_metric, compare_with_leaderboard, to_leaderboard};
use super::types::{HfLeaderboard, LeaderboardEntry, LeaderboardKind};

// ═══════════════════════════════════════════════════════════════════════
// CLAIM: Leaderboard ranking respects higher_is_better semantics
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn falsify_wer_leaderboard_ranks_lower_first() {
    // WER is lower-is-better: model with WER=0.05 should beat WER=0.10
    let mut hf = HfLeaderboard::new(LeaderboardKind::OpenASR);

    let mut bad = LeaderboardEntry::new("bad-model");
    bad.scores.insert("wer".into(), 0.20);
    hf.entries.push(bad);

    let mut good = LeaderboardEntry::new("good-model");
    good.scores.insert("wer".into(), 0.05);
    hf.entries.push(good);

    let lb = to_leaderboard(&hf);
    assert_eq!(
        lb.best().expect("operation should succeed").model_name,
        "good-model",
        "WER leaderboard should rank lower WER first"
    );
}

#[test]
fn falsify_bleu_leaderboard_ranks_higher_first() {
    let mut lb = Leaderboard::new(Metric::BLEU);

    let mut low = EvalResult::new("low-bleu");
    low.add_score(Metric::BLEU, 0.2);
    lb.add(low);

    let mut high = EvalResult::new("high-bleu");
    high.add_score(Metric::BLEU, 0.8);
    lb.add(high);

    assert_eq!(
        lb.best().expect("operation should succeed").model_name,
        "high-bleu",
        "BLEU leaderboard should rank higher BLEU first"
    );
}

#[test]
fn falsify_perplexity_leaderboard_ranks_lower_first() {
    let mut lb = Leaderboard::new(Metric::Perplexity);

    let mut high_ppl = EvalResult::new("high-ppl");
    high_ppl.add_score(Metric::Perplexity, 100.0);
    lb.add(high_ppl);

    let mut low_ppl = EvalResult::new("low-ppl");
    low_ppl.add_score(Metric::Perplexity, 5.0);
    lb.add(low_ppl);

    assert_eq!(
        lb.best().expect("operation should succeed").model_name,
        "low-ppl",
        "Perplexity leaderboard should rank lower PPL first"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// CLAIM: compare_with_leaderboard correctly inserts user model
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn falsify_compare_preserves_all_entries() {
    let mut hf = HfLeaderboard::new(LeaderboardKind::OpenASR);
    for i in 0..5 {
        let mut entry = LeaderboardEntry::new(format!("model-{i}"));
        entry.scores.insert("wer".into(), 0.1 * f64::from(i + 1));
        hf.entries.push(entry);
    }

    let mut my = EvalResult::new("my-model");
    my.add_score(Metric::WER, 0.07);

    let lb = compare_with_leaderboard(&my, &hf);
    assert_eq!(lb.results.len(), 6, "Should have 5 HF entries + 1 user entry");
    assert_eq!(
        lb.best().expect("operation should succeed").model_name,
        "my-model",
        "User model with WER=0.07 should be best"
    );
}

#[test]
fn falsify_compare_user_model_worse_than_all() {
    let mut hf = HfLeaderboard::new(LeaderboardKind::OpenASR);
    let mut entry = LeaderboardEntry::new("sota-model");
    entry.scores.insert("wer".into(), 0.01);
    hf.entries.push(entry);

    let mut my = EvalResult::new("my-bad-model");
    my.add_score(Metric::WER, 0.50);

    let lb = compare_with_leaderboard(&my, &hf);
    assert_eq!(
        lb.best().expect("operation should succeed").model_name,
        "sota-model",
        "SOTA model should remain best"
    );
    // User model should be last
    assert_eq!(
        lb.results.last().expect("collection should not be empty").model_name,
        "my-bad-model"
    );
}

// ═══════════════════════════════════════════════════════════════════════
// CLAIM: column_to_metric handles case insensitivity
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn falsify_column_mapping_case_insensitive() {
    // The spec says column names should be lowercased before matching
    assert_eq!(column_to_metric(&LeaderboardKind::OpenASR, "WER"), Some(Metric::WER));
    assert_eq!(column_to_metric(&LeaderboardKind::OpenASR, "Wer"), Some(Metric::WER));
    assert_eq!(column_to_metric(&LeaderboardKind::OpenLLMv2, "MMLU"), Some(Metric::MMLUAccuracy));
    assert_eq!(
        column_to_metric(&LeaderboardKind::BigCodeBench, "Pass@1"),
        Some(Metric::PassAtK(1))
    );
}

#[test]
fn falsify_column_mapping_unknown_returns_none() {
    assert_eq!(column_to_metric(&LeaderboardKind::OpenASR, "completely_unknown_metric"), None);
    assert_eq!(column_to_metric(&LeaderboardKind::OpenLLMv2, ""), None);
}

// ═══════════════════════════════════════════════════════════════════════
// CLAIM: Custom leaderboard kind accepts arbitrary dataset IDs
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn falsify_custom_leaderboard_passthrough() {
    let kind = LeaderboardKind::Custom("my-org/my-leaderboard".into());
    assert_eq!(kind.dataset_repo_id(), "my-org/my-leaderboard");
    // Custom defaults to Accuracy
    assert_eq!(kind.primary_metric(), Metric::Accuracy);
}

#[test]
fn falsify_custom_leaderboard_generic_mapping() {
    let kind = LeaderboardKind::Custom("anything".into());
    // Generic mapper should handle common metric names
    assert_eq!(column_to_metric(&kind, "accuracy"), Some(Metric::Accuracy));
    assert_eq!(column_to_metric(&kind, "wer"), Some(Metric::WER));
    assert_eq!(column_to_metric(&kind, "bleu"), Some(Metric::BLEU));
    assert_eq!(column_to_metric(&kind, "rouge1"), Some(Metric::ROUGE(RougeVariant::Rouge1)));
    assert_eq!(column_to_metric(&kind, "rougel"), Some(Metric::ROUGE(RougeVariant::RougeL)));
    assert_eq!(column_to_metric(&kind, "perplexity"), Some(Metric::Perplexity));
    assert_eq!(column_to_metric(&kind, "ppl"), Some(Metric::Perplexity));
    assert_eq!(column_to_metric(&kind, "mmlu"), Some(Metric::MMLUAccuracy));
    assert_eq!(column_to_metric(&kind, "pass@1"), Some(Metric::PassAtK(1)));
    assert_eq!(column_to_metric(&kind, "ndcg@10"), Some(Metric::NDCGAtK(10)));
}

// ═══════════════════════════════════════════════════════════════════════
// CLAIM: HfLeaderboard.find_model returns correct entry
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn falsify_find_model_exact_match() {
    let mut lb = HfLeaderboard::new(LeaderboardKind::OpenASR);
    let mut e = LeaderboardEntry::new("openai/whisper-large-v3");
    e.scores.insert("wer".into(), 0.05);
    lb.entries.push(e);
    lb.entries.push(LeaderboardEntry::new("other/model"));

    let found = lb.find_model("openai/whisper-large-v3");
    assert!(found.is_some());
    assert_eq!(found.expect("operation should succeed").get_score("wer"), Some(0.05));
}

#[test]
fn falsify_find_model_no_partial_match() {
    let mut lb = HfLeaderboard::new(LeaderboardKind::OpenASR);
    lb.entries.push(LeaderboardEntry::new("openai/whisper-large-v3"));

    // Partial match should NOT work
    assert!(lb.find_model("openai/whisper").is_none());
    assert!(lb.find_model("whisper-large-v3").is_none());
}

// ═══════════════════════════════════════════════════════════════════════
// CLAIM: Empty leaderboard produces empty Leaderboard
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn falsify_empty_hf_leaderboard() {
    let hf = HfLeaderboard::new(LeaderboardKind::MTEB);
    let lb = to_leaderboard(&hf);
    assert!(lb.results.is_empty());
    assert!(lb.best().is_none());
}

// ═══════════════════════════════════════════════════════════════════════
// CLAIM: Scores with no matching metric columns are silently dropped
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn falsify_unrecognized_columns_dropped() {
    let mut hf = HfLeaderboard::new(LeaderboardKind::OpenASR);
    let mut entry = LeaderboardEntry::new("model-a");
    entry.scores.insert("wer".into(), 0.1);
    entry.scores.insert("completely_unknown_column".into(), 42.0);
    entry.scores.insert("another_unknown".into(), 99.0);
    hf.entries.push(entry);

    let lb = to_leaderboard(&hf);
    let result = &lb.results[0];
    // Only WER should be present
    assert_eq!(result.get_score(Metric::WER), Some(0.1));
    // Unknown columns should not create spurious metrics
    assert_eq!(result.scores.len(), 1);
}
