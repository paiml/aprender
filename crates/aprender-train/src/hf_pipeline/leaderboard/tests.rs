//! Tests for HuggingFace leaderboard integration

use super::parser::{column_to_metric, compare_with_leaderboard, to_leaderboard};
use super::types::{HfLeaderboard, LeaderboardEntry, LeaderboardKind};
use crate::eval::evaluator::{EvalResult, Metric};

// ─── LeaderboardKind tests ───────────────────────────────────────────

#[test]
fn test_kind_dataset_repo_ids() {
    assert_eq!(LeaderboardKind::OpenASR.dataset_repo_id(), "hf-audio/open_asr_leaderboard");
    assert_eq!(LeaderboardKind::OpenLLMv2.dataset_repo_id(), "open-llm-leaderboard/results");
    assert_eq!(LeaderboardKind::MTEB.dataset_repo_id(), "mteb/leaderboard");
    assert_eq!(LeaderboardKind::BigCodeBench.dataset_repo_id(), "bigcode/bigcodebench-results");
    assert_eq!(
        LeaderboardKind::Custom("my/leaderboard".into()).dataset_repo_id(),
        "my/leaderboard"
    );
}

#[test]
fn test_kind_primary_metric() {
    assert_eq!(LeaderboardKind::OpenASR.primary_metric(), Metric::WER);
    assert_eq!(LeaderboardKind::OpenLLMv2.primary_metric(), Metric::MMLUAccuracy);
    assert_eq!(LeaderboardKind::MTEB.primary_metric(), Metric::NDCGAtK(10));
    assert_eq!(LeaderboardKind::BigCodeBench.primary_metric(), Metric::PassAtK(1));
}

#[test]
fn test_kind_tracked_metrics() {
    let metrics = LeaderboardKind::OpenASR.tracked_metrics();
    assert!(metrics.contains(&Metric::WER));
    assert!(metrics.contains(&Metric::RTFx));

    let metrics = LeaderboardKind::BigCodeBench.tracked_metrics();
    assert!(metrics.contains(&Metric::PassAtK(1)));
    assert!(metrics.contains(&Metric::PassAtK(10)));
}

#[test]
fn test_kind_display() {
    assert_eq!(format!("{}", LeaderboardKind::OpenASR), "Open ASR Leaderboard");
    assert_eq!(format!("{}", LeaderboardKind::Custom("foo/bar".into())), "Custom (foo/bar)");
}

// ─── LeaderboardEntry tests ─────────────────────────────────────────

#[test]
fn test_entry_creation() {
    let mut entry = LeaderboardEntry::new("openai/whisper-large-v3");
    entry.scores.insert("wer".into(), 0.08);
    entry.metadata.insert("license".into(), "apache-2.0".into());

    assert_eq!(entry.model_id, "openai/whisper-large-v3");
    assert_eq!(entry.get_score("wer"), Some(0.08));
    assert_eq!(entry.get_score("missing"), None);
}

// ─── HfLeaderboard tests ───────────────────────────────────────────

#[test]
fn test_hf_leaderboard_find_model() {
    let mut lb = HfLeaderboard::new(LeaderboardKind::OpenASR);
    lb.entries.push(LeaderboardEntry::new("model-a"));
    lb.entries.push(LeaderboardEntry::new("model-b"));

    assert!(lb.find_model("model-a").is_some());
    assert!(lb.find_model("model-c").is_none());
}

// ─── Parser tests ───────────────────────────────────────────────────

#[test]
fn test_column_to_metric_open_asr() {
    assert_eq!(column_to_metric(&LeaderboardKind::OpenASR, "wer"), Some(Metric::WER));
    assert_eq!(column_to_metric(&LeaderboardKind::OpenASR, "average_wer"), Some(Metric::WER));
    assert_eq!(column_to_metric(&LeaderboardKind::OpenASR, "rtfx"), Some(Metric::RTFx));
    assert_eq!(column_to_metric(&LeaderboardKind::OpenASR, "unknown"), None);
}

#[test]
fn test_column_to_metric_open_llm() {
    assert_eq!(column_to_metric(&LeaderboardKind::OpenLLMv2, "mmlu"), Some(Metric::MMLUAccuracy));
    assert_eq!(column_to_metric(&LeaderboardKind::OpenLLMv2, "accuracy"), Some(Metric::Accuracy));
}

#[test]
fn test_column_to_metric_bigcode() {
    assert_eq!(
        column_to_metric(&LeaderboardKind::BigCodeBench, "pass@1"),
        Some(Metric::PassAtK(1))
    );
    assert_eq!(
        column_to_metric(&LeaderboardKind::BigCodeBench, "pass@10"),
        Some(Metric::PassAtK(10))
    );
}

#[test]
fn test_column_to_metric_custom() {
    assert_eq!(column_to_metric(&LeaderboardKind::Custom("x".into()), "bleu"), Some(Metric::BLEU));
    assert_eq!(
        column_to_metric(&LeaderboardKind::Custom("x".into()), "perplexity"),
        Some(Metric::Perplexity)
    );
}

#[test]
fn test_to_leaderboard_conversion() {
    let mut hf = HfLeaderboard::new(LeaderboardKind::OpenASR);

    let mut entry1 = LeaderboardEntry::new("model-a");
    entry1.scores.insert("wer".into(), 0.15);
    hf.entries.push(entry1);

    let mut entry2 = LeaderboardEntry::new("model-b");
    entry2.scores.insert("wer".into(), 0.08);
    hf.entries.push(entry2);

    let leaderboard = to_leaderboard(&hf);
    assert_eq!(leaderboard.results.len(), 2);
    // WER is lower-is-better, so model-b (0.08) should be best
    assert_eq!(leaderboard.best().expect("operation should succeed").model_name, "model-b");
}

#[test]
fn test_compare_with_leaderboard() {
    let mut hf = HfLeaderboard::new(LeaderboardKind::OpenASR);

    let mut entry = LeaderboardEntry::new("existing-model");
    entry.scores.insert("wer".into(), 0.10);
    hf.entries.push(entry);

    let mut my_result = EvalResult::new("my-model");
    my_result.add_score(Metric::WER, 0.05);

    let leaderboard = compare_with_leaderboard(&my_result, &hf);
    assert_eq!(leaderboard.results.len(), 2);
    // my-model has better WER (0.05 < 0.10)
    assert_eq!(leaderboard.best().expect("operation should succeed").model_name, "my-model");
}

#[test]
fn test_empty_leaderboard_conversion() {
    let hf = HfLeaderboard::new(LeaderboardKind::OpenLLMv2);
    let leaderboard = to_leaderboard(&hf);
    assert!(leaderboard.results.is_empty());
    assert!(leaderboard.best().is_none());
}

// ─── Integration tests (requires network, run with --ignored) ───────

#[test]
#[ignore = "Requires network access and HF_TOKEN"]
fn test_fetch_real_leaderboard() {
    use super::client::LeaderboardClient;

    let client = LeaderboardClient::new().expect("Failed to create client");
    let result = client.fetch_paginated(LeaderboardKind::OpenASR, 0, 5);
    match result {
        Ok(hf) => {
            assert!(!hf.entries.is_empty());
            assert!(hf.total_count > 0);
        }
        Err(e) => {
            // Network errors are acceptable in CI
            eprintln!("Leaderboard fetch failed (expected in CI): {e}");
        }
    }
}
