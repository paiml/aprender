#![allow(clippy::unwrap_used, clippy::disallowed_methods)]
//! `ShellAutocomplete::suggest` ranking and filtering contract (#5553).
//!
//! The model is built from bytes through `load_from_bytes` (public API): a
//! 32-byte APR header followed by an uncompressed bincode payload that mirrors
//! the private `MarkovModelData` layout.

use aprender_present_lib::browser::ShellAutocomplete;
use serde::Serialize;
use std::collections::HashMap;

#[derive(Serialize)]
struct ModelData {
    n: usize,
    ngrams: HashMap<String, HashMap<String, u32>>,
    command_freq: HashMap<String, u32>,
    total_commands: usize,
    last_trained_pos: usize,
}

fn row(pairs: &[(&str, u32)]) -> HashMap<String, u32> {
    pairs.iter().map(|(t, c)| ((*t).to_string(), *c)).collect()
}

fn load(data: &ModelData) -> ShellAutocomplete {
    let payload = bincode::serialize(data).unwrap();
    let mut bytes = vec![0u8; 32];
    bytes[0..4].copy_from_slice(b"APRN");
    bytes[8..12].copy_from_slice(&0u32.to_le_bytes());
    bytes[12..16].copy_from_slice(&u32::try_from(payload.len()).unwrap().to_le_bytes());
    bytes[20] = 0; // no compression
    bytes.extend_from_slice(&payload);
    ShellAutocomplete::load_from_bytes(&bytes).unwrap()
}

fn freq(pairs: &[(&str, u32)]) -> HashMap<String, u32> {
    row(pairs)
}

fn model(n: usize) -> ShellAutocomplete {
    let mut ngrams = HashMap::new();
    ngrams.insert(
        "git".to_string(),
        row(&[("commit", 3), ("config", 1), ("commit-m", 4)]),
    );
    ngrams.insert("commit".to_string(), row(&[("-m", 2), ("--amend", 1)]));
    load(&ModelData {
        n,
        ngrams,
        command_freq: freq(&[
            ("git commit", 60),
            ("git status", 30),
            ("ls", 10),
            ("git  push", 5),
        ]),
        total_commands: 100,
        last_trained_pos: 0,
    })
}

fn score_of(s: &[(String, f32)], text: &str) -> Option<f32> {
    s.iter().find(|(t, _)| t == text).map(|(_, x)| *x)
}

#[test]
fn trie_matches_rank_by_frequency_score() {
    let s = model(3).suggest("git", 10);
    assert_eq!(s[0].0, "git commit");
    assert!((s[0].1 - 0.6).abs() < 1e-6);
    assert!(score_of(&s, "git status").is_some());
}

#[test]
fn corrupted_trie_entries_are_dropped() {
    let s = model(3).suggest("git", 10);
    assert!(s.iter().all(|(t, _)| t != "git  push"));
}

#[test]
fn count_truncates_after_sorting() {
    let s = model(3).suggest("git", 1);
    assert_eq!(s.len(), 1);
    assert_eq!(s[0].0, "git commit");
    assert!(model(3).suggest("git", 0).is_empty());
}

#[test]
fn unknown_prefix_yields_nothing() {
    assert!(model(3).suggest("zzz", 5).is_empty());
}

#[test]
fn prefix_is_trimmed_so_trailing_space_adds_no_ngram_completions() {
    let m = model(3);
    assert_eq!(m.suggest("git ", 10), m.suggest("git", 10));
    assert!(score_of(&m.suggest("git ", 10), "git config").is_none());
}

#[test]
fn partial_token_uses_ngram_context_at_090_weight() {
    let s = model(3).suggest("git co", 10);
    let cfg = score_of(&s, "git config").expect("ngram completion present");
    assert!((cfg - 0.9 * 1.0 / 8.0).abs() < 1e-6, "{cfg}");
}

#[test]
fn ngram_completion_does_not_duplicate_a_trie_hit() {
    let s = model(3).suggest("git co", 10);
    assert_eq!(s.iter().filter(|(t, _)| t == "git commit").count(), 1);
    assert!((score_of(&s, "git commit").expect("trie hit") - 0.6).abs() < 1e-6);
}

#[test]
fn corrupted_ngram_tokens_are_filtered() {
    let s = model(3).suggest("git co", 10);
    assert!(score_of(&s, "git commit-m").is_none());
}

#[test]
fn context_window_is_n_minus_one_tokens() {
    let s = model(2).suggest("git commit -", 10);
    let m = score_of(&s, "git commit -m").expect("bigram context");
    assert!((m - 0.9 * 2.0 / 3.0).abs() < 1e-6, "{m}");
    assert!(score_of(&s, "git commit --amend").is_some());
    assert!(model(3).suggest("git commit -", 10).is_empty());
}

#[test]
fn single_token_prefix_skips_ngram_strategy() {
    assert!(model(3).suggest("c", 10).is_empty());
}

#[test]
fn empty_prefix_lists_commands_by_frequency() {
    let s = model(3).suggest("", 2);
    assert_eq!(s.len(), 2);
    assert_eq!(s[0].0, "git commit");
    assert_eq!(s[1].0, "git status");
}

#[test]
fn empty_prefix_falls_back_to_top_commands_when_all_trie_hits_are_corrupted() {
    let m = load(&ModelData {
        n: 3,
        ngrams: HashMap::new(),
        command_freq: freq(&[("a  b", 7)]),
        total_commands: 14,
        last_trained_pos: 0,
    });
    let s = m.suggest("", 5);
    assert_eq!(s.len(), 1);
    assert_eq!(s[0].0, "a  b");
    assert!((s[0].1 - 0.5).abs() < 1e-6);
}

#[test]
fn zero_total_commands_does_not_divide_by_zero() {
    let m = load(&ModelData {
        n: 3,
        ngrams: HashMap::new(),
        command_freq: freq(&[("ls", 10)]),
        total_commands: 0,
        last_trained_pos: 0,
    });
    let s = m.suggest("ls", 5);
    assert_eq!(s[0].0, "ls");
    assert!(s[0].1.is_finite());
}
