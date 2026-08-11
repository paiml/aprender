// PMAT-125 B2: CPU-only unit tests for pure helpers in eval/mod.rs.

use super::*;
use std::str::FromStr;

// ── Dataset::from_str ──────────────────────────────────────────────────────

#[test]
fn dataset_parses_wikitext_aliases() {
    assert_eq!(Dataset::from_str("wikitext-2").unwrap(), Dataset::WikiText2);
    assert_eq!(Dataset::from_str("wikitext2").unwrap(), Dataset::WikiText2);
    assert_eq!(Dataset::from_str("WikiText-2").unwrap(), Dataset::WikiText2);
}

#[test]
fn dataset_parses_lambada_and_custom() {
    assert_eq!(Dataset::from_str("lambada").unwrap(), Dataset::Lambada);
    assert_eq!(Dataset::from_str("LAMBADA").unwrap(), Dataset::Lambada);
    assert_eq!(Dataset::from_str("custom").unwrap(), Dataset::Custom);
}

#[test]
fn dataset_rejects_unknown() {
    let err = Dataset::from_str("squad").unwrap_err();
    assert!(err.contains("Unknown dataset"));
}

// ── extract_ngrams + compute_ngram_overlap ─────────────────────────────────

#[test]
fn extract_ngrams_basic() {
    let grams = extract_ngrams("abcd", 2);
    // "ab", "bc", "cd"
    assert_eq!(grams.len(), 3);
    assert!(grams.contains("ab"));
    assert!(grams.contains("cd"));
}

#[test]
fn extract_ngrams_dedups_repeats() {
    // "aaaa" with n=2 ⇒ only "aa".
    let grams = extract_ngrams("aaaa", 2);
    assert_eq!(grams.len(), 1);
    assert!(grams.contains("aa"));
}

#[test]
fn extract_ngrams_too_short_is_empty() {
    assert!(extract_ngrams("ab", 5).is_empty());
    assert!(extract_ngrams("", 1).is_empty());
}

#[test]
fn ngram_overlap_identical_is_one() {
    let a = extract_ngrams("hello world", 3);
    let b = extract_ngrams("hello world", 3);
    assert!((compute_ngram_overlap(&a, &b) - 1.0).abs() < 1e-6);
}

#[test]
fn ngram_overlap_disjoint_is_zero() {
    let a = extract_ngrams("aaaa", 2);
    let b = extract_ngrams("bbbb", 2);
    assert_eq!(compute_ngram_overlap(&a, &b), 0.0);
}

#[test]
fn ngram_overlap_empty_is_zero() {
    let a = extract_ngrams("abc", 2);
    let empty = std::collections::HashSet::new();
    assert_eq!(compute_ngram_overlap(&a, &empty), 0.0);
    assert_eq!(compute_ngram_overlap(&empty, &a), 0.0);
}

#[test]
fn ngram_overlap_partial_is_jaccard() {
    // a = {ab, bc}, b = {bc, cd} ⇒ intersection 1, union 3 ⇒ 1/3.
    let a = extract_ngrams("abc", 2);
    let b = extract_ngrams("bcd", 2);
    let j = compute_ngram_overlap(&a, &b);
    assert!((j - 1.0 / 3.0).abs() < 1e-6, "got {j}");
}

// ── format_bytes ───────────────────────────────────────────────────────────

#[test]
fn format_bytes_units() {
    assert_eq!(format_bytes(512), "512 B");
    assert_eq!(format_bytes(2048), "2.0 KiB");
    assert_eq!(format_bytes(1_048_576), "1.0 MiB");
    assert_eq!(format_bytes(1_073_741_824), "1.00 GiB");
}

#[test]
fn format_bytes_boundaries() {
    assert_eq!(format_bytes(1023), "1023 B");
    assert_eq!(format_bytes(1024), "1.0 KiB");
    assert_eq!(format_bytes(1_048_575), "1024.0 KiB");
}

// ── format_archive_size (GB/MB/KB variant) ─────────────────────────────────

#[test]
fn format_archive_size_units() {
    assert_eq!(format_archive_size(100), "100 B");
    assert_eq!(format_archive_size(1536), "1.5 KB");
    assert_eq!(format_archive_size(1_572_864), "1.5 MB");
    assert_eq!(format_archive_size(1_610_612_736), "1.5 GB");
}

// ── compute_file_hash (FNV-1a) ─────────────────────────────────────────────

#[test]
fn file_hash_is_deterministic() {
    let h1 = compute_file_hash(b"hello");
    let h2 = compute_file_hash(b"hello");
    assert_eq!(h1, h2);
    assert_eq!(h1.len(), 16); // 64-bit hex
}

#[test]
fn file_hash_differs_on_content() {
    assert_ne!(compute_file_hash(b"hello"), compute_file_hash(b"world"));
}

#[test]
fn file_hash_empty_input() {
    // FNV-1a offset basis for empty input.
    assert_eq!(compute_file_hash(b""), "cbf29ce484222325");
}

// ── pearson_correlation ────────────────────────────────────────────────────

#[test]
fn pearson_perfect_positive() {
    let x = [1.0, 2.0, 3.0, 4.0];
    let y = [2.0, 4.0, 6.0, 8.0];
    assert!((pearson_correlation(&x, &y) - 1.0).abs() < 1e-9);
}

#[test]
fn pearson_perfect_negative() {
    let x = [1.0, 2.0, 3.0, 4.0];
    let y = [4.0, 3.0, 2.0, 1.0];
    assert!((pearson_correlation(&x, &y) + 1.0).abs() < 1e-9);
}

#[test]
fn pearson_too_few_points_is_zero() {
    assert_eq!(pearson_correlation(&[1.0], &[2.0]), 0.0);
    assert_eq!(pearson_correlation(&[], &[]), 0.0);
}

#[test]
fn pearson_zero_variance_is_zero() {
    // Constant x ⇒ undefined correlation, guarded to 0.
    let x = [5.0, 5.0, 5.0];
    let y = [1.0, 2.0, 3.0];
    assert_eq!(pearson_correlation(&x, &y), 0.0);
}

// ── compute_ranks + spearman_correlation ───────────────────────────────────

#[test]
fn compute_ranks_no_ties() {
    // values 30,10,20 ⇒ ranks 3,1,2
    let ranks = compute_ranks(&[30.0, 10.0, 20.0]);
    assert_eq!(ranks, vec![3.0, 1.0, 2.0]);
}

#[test]
fn compute_ranks_with_ties_averages() {
    // values 10,10,20 ⇒ first two tied at ranks (1+2)/2=1.5, third = 3.
    let ranks = compute_ranks(&[10.0, 10.0, 20.0]);
    assert_eq!(ranks, vec![1.5, 1.5, 3.0]);
}

#[test]
fn spearman_monotonic_is_one() {
    // Strictly monotonic (non-linear) ⇒ Spearman = 1 even though Pearson < 1.
    let x = [1.0, 2.0, 3.0, 4.0];
    let y = [1.0, 4.0, 9.0, 16.0];
    assert!((spearman_correlation(&x, &y) - 1.0).abs() < 1e-9);
}

// ── interpret_correlation ──────────────────────────────────────────────────

#[test]
fn interpret_correlation_strength_buckets() {
    assert!(interpret_correlation(0.95).contains("Very strong"));
    assert!(interpret_correlation(0.8).contains("Strong"));
    assert!(interpret_correlation(0.6).contains("Moderate"));
    assert!(interpret_correlation(0.4).contains("Weak"));
    assert!(interpret_correlation(0.1).contains("Very weak"));
}

#[test]
fn interpret_correlation_direction() {
    assert!(interpret_correlation(-0.8).contains("negative"));
    assert!(interpret_correlation(0.8).contains("positive"));
}

// ── encryption primitives: apply_keystream + compute_mac ───────────────────

#[test]
fn keystream_roundtrips_to_plaintext() {
    let key = [7u8; 32];
    let nonce = [9u8; 32];
    let plaintext = b"the quick brown fox jumps over the lazy dog repeatedly".to_vec();
    let ct = apply_keystream(&key, &nonce, &plaintext);
    assert_ne!(ct, plaintext); // actually encrypted
    let pt = apply_keystream(&key, &nonce, &ct);
    assert_eq!(pt, plaintext); // XOR cipher is its own inverse
}

#[test]
fn keystream_spans_multiple_blocks() {
    // >64 bytes forces multiple keystream blocks.
    let key = [1u8; 32];
    let nonce = [2u8; 32];
    let plaintext = vec![0xABu8; 200];
    let ct = apply_keystream(&key, &nonce, &plaintext);
    assert_eq!(ct.len(), 200);
    let pt = apply_keystream(&key, &nonce, &ct);
    assert_eq!(pt, plaintext);
}

#[test]
fn keystream_different_nonce_differs() {
    let key = [3u8; 32];
    let data = b"same plaintext".to_vec();
    let a = apply_keystream(&key, &[0u8; 32], &data);
    let b = apply_keystream(&key, &[1u8; 32], &data);
    assert_ne!(a, b);
}

#[test]
fn mac_is_deterministic_and_content_sensitive() {
    let key = [4u8; 32];
    let nonce = [5u8; 32];
    let m1 = compute_mac(&key, &nonce, b"data");
    let m2 = compute_mac(&key, &nonce, b"data");
    assert_eq!(m1.as_bytes(), m2.as_bytes());
    let m3 = compute_mac(&key, &nonce, b"datb");
    assert_ne!(m1.as_bytes(), m3.as_bytes());
}

// ── derive_encryption_key ──────────────────────────────────────────────────

#[test]
fn derive_key_from_32_byte_file_uses_raw_bytes() {
    let dir = std::env::temp_dir().join(format!("apr_key_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let kf = dir.join("key.bin");
    let raw = [0x5Au8; 32];
    std::fs::write(&kf, raw).unwrap();
    let key = derive_encryption_key(Some(&kf)).unwrap();
    assert_eq!(key, raw);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn derive_key_from_short_file_derives() {
    let dir = std::env::temp_dir().join(format!("apr_key_short_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let kf = dir.join("key.bin");
    std::fs::write(&kf, b"short").unwrap();
    let key = derive_encryption_key(Some(&kf)).unwrap();
    // Derived key is a BLAKE3 derive_key of the content — deterministic.
    let expected = blake3::derive_key("albor model encryption 2026", b"short");
    assert_eq!(key, expected);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn derive_key_missing_file_errors() {
    let r = derive_encryption_key(Some(Path::new("/nonexistent/key.bin")));
    assert!(r.is_err());
}

// ── extract_ppl_from_jsonl ─────────────────────────────────────────────────

#[test]
fn extract_ppl_normalizes_steps() {
    let dir = std::env::temp_dir().join(format!("apr_ppl_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let f = dir.join("log.jsonl");
    std::fs::write(
        &f,
        "{\"step\":0,\"val_ppl\":10.0}\n{\"step\":100,\"val_ppl\":5.0}\n",
    )
    .unwrap();
    let pairs = extract_ppl_from_jsonl(&f);
    assert_eq!(pairs.len(), 2);
    // (ppl, step/max_step): (10.0, 0.0), (5.0, 1.0)
    assert_eq!(pairs[0], (10.0, 0.0));
    assert_eq!(pairs[1], (5.0, 1.0));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn extract_ppl_single_entry_is_empty() {
    let dir = std::env::temp_dir().join(format!("apr_ppl_one_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let f = dir.join("log.jsonl");
    std::fs::write(&f, "{\"step\":0,\"val_ppl\":10.0}\n").unwrap();
    assert!(extract_ppl_from_jsonl(&f).is_empty());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn extract_ppl_missing_file_is_empty() {
    assert!(extract_ppl_from_jsonl(Path::new("/nonexistent/log.jsonl")).is_empty());
}

// ── read_json_f64 ──────────────────────────────────────────────────────────

#[test]
fn read_json_f64_reads_value() {
    let dir = std::env::temp_dir().join(format!("apr_jf64_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let f = dir.join("eval.json");
    std::fs::write(&f, "{\"perplexity\": 3.5, \"name\": \"x\"}").unwrap();
    assert_eq!(read_json_f64(&f, "perplexity"), Some(3.5));
    assert_eq!(read_json_f64(&f, "name"), None); // not an f64
    assert_eq!(read_json_f64(&f, "missing"), None);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn read_json_f64_missing_file_is_none() {
    assert_eq!(read_json_f64(Path::new("/no/such.json"), "k"), None);
}

// ── extract_loss_history_pairs ─────────────────────────────────────────────

#[test]
fn loss_history_maps_to_exp_and_progress() {
    let dir = std::env::temp_dir().join(format!("apr_lh_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let f = dir.join("training_state.json");
    std::fs::write(&f, "{\"loss_history\": [0.0, 1.0]}").unwrap();
    let pairs = extract_loss_history_pairs(&f);
    assert_eq!(pairs.len(), 2);
    // (exp(loss), (i+1)/n): (1.0, 0.5), (e, 1.0)
    assert!((pairs[0].0 - 1.0).abs() < 1e-9);
    assert!((pairs[0].1 - 0.5).abs() < 1e-9);
    assert!((pairs[1].0 - std::f64::consts::E).abs() < 1e-9);
    assert!((pairs[1].1 - 1.0).abs() < 1e-9);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn loss_history_too_short_is_empty() {
    let dir = std::env::temp_dir().join(format!("apr_lh_short_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let f = dir.join("training_state.json");
    std::fs::write(&f, "{\"loss_history\": [1.0]}").unwrap();
    assert!(extract_loss_history_pairs(&f).is_empty());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn loss_history_no_field_is_empty() {
    let dir = std::env::temp_dir().join(format!("apr_lh_none_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let f = dir.join("training_state.json");
    std::fs::write(&f, "{\"other\": 1}").unwrap();
    assert!(extract_loss_history_pairs(&f).is_empty());
    let _ = std::fs::remove_dir_all(&dir);
}

// ── count_safetensors_keys ─────────────────────────────────────────────────

#[test]
fn count_safetensors_keys_excludes_metadata() {
    let dir = std::env::temp_dir().join(format!("apr_st_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let f = dir.join("model.safetensors");
    // Build a minimal safetensors file: [8-byte header_len][header json].
    let header = r#"{"__metadata__":{"k":"v"},"w1":{"dtype":"F32"},"w2":{"dtype":"F32"}}"#;
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header.as_bytes());
    std::fs::write(&f, &bytes).unwrap();
    // Two real tensors, __metadata__ excluded.
    assert_eq!(count_safetensors_keys(&f), 2);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn count_safetensors_keys_truncated_is_zero() {
    let dir = std::env::temp_dir().join(format!("apr_st_bad_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let f = dir.join("bad.safetensors");
    std::fs::write(&f, b"abc").unwrap(); // < 8 bytes
    assert_eq!(count_safetensors_keys(&f), 0);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn count_safetensors_keys_missing_is_zero() {
    assert_eq!(
        count_safetensors_keys(Path::new("/no/model.safetensors")),
        0
    );
}

// ── load_eval_prompts + default_code_eval_prompts ──────────────────────────

#[test]
fn load_eval_prompts_from_jsonl() {
    let dir = std::env::temp_dir().join(format!("apr_prompts_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let f = dir.join("prompts.jsonl");
    std::fs::write(&f, "{\"prompt\":\"def a():\"}\n{\"prompt\":\"def b():\"}\n").unwrap();
    let prompts = load_eval_prompts(&f).unwrap();
    assert_eq!(
        prompts,
        vec!["def a():".to_string(), "def b():".to_string()]
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn load_eval_prompts_plaintext_fallback() {
    let dir = std::env::temp_dir().join(format!("apr_prompts_txt_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let f = dir.join("prompts.txt");
    std::fs::write(&f, "line one\nline two\n").unwrap();
    let prompts = load_eval_prompts(&f).unwrap();
    assert_eq!(
        prompts,
        vec!["line one".to_string(), "line two".to_string()]
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn load_eval_prompts_missing_file_errors() {
    assert!(load_eval_prompts(Path::new("/no/prompts.jsonl")).is_err());
}

// ── --device in perplexity mode ────────────────────────────────────────────
//
// `--device cuda` and `--device bogus` used to produce byte-identical output to
// `--device cpu`, with no mention of a device anywhere — the flag was accepted,
// never consulted, and never reported.

#[test]
fn perplexity_device_is_always_cpu() {
    assert_eq!(perplexity_device("cpu"), "cpu");
    assert_eq!(
        perplexity_device("cuda"),
        "cpu",
        "perplexity has no GPU path; the report must say cpu"
    );
}

#[test]
fn perplexity_device_notice_fires_only_when_the_request_is_not_honoured() {
    assert!(
        perplexity_device_notice("cpu").is_none(),
        "no notice when the request is honoured"
    );

    let notice = perplexity_device_notice("cuda").expect("cuda must be called out");
    assert!(notice.contains("cuda"), "notice must quote the request");
    assert!(
        notice.contains("CPU-only") && notice.contains("cpu"),
        "notice must state what actually ran, got: {notice}"
    );
}

#[test]
fn default_prompts_are_nonempty_python() {
    let prompts = default_code_eval_prompts();
    assert_eq!(prompts.len(), 10);
    for p in &prompts {
        assert!(p.contains("def ") || p.contains("class "));
    }
}
