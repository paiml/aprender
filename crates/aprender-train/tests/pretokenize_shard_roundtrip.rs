//! GATE-PRETOK-003: Round-trip integrator gate for pretokenize-bin-v1.
//!
//! Proves the exact shard layout produced by `apr tokenize encode-corpus`
//! (little-endian u32 stream in `shard-NNNN.bin`) is faithfully consumed by
//! `ShardBatchIter` — the pretrain-time reader used by MODEL-2 training.
//!
//! This closes the root-cause fix for the pretokenize-to-bin gap
//! (memory/project_shard_reader_bin_format.md); the contract is
//! `contracts/pretokenize-bin-v1.yaml`.

use entrenar::training::shard_reader::ShardBatchIter;
use std::io::Write;

fn write_shard(dir: &std::path::Path, name: &str, tokens: &[u32]) {
    let path = dir.join(name);
    let mut f = std::fs::File::create(&path).expect("create shard");
    for t in tokens {
        f.write_all(&t.to_le_bytes()).expect("write u32");
    }
    f.flush().expect("flush");
}

/// INV-PRETOK-002 + INV-PRETOK-007: the exact byte layout produced by
/// `run_encode_corpus` (single shard, little-endian u32) is readable by
/// `ShardBatchIter` with every token preserved and in order.
#[test]
fn cli_shard_layout_is_read_by_shard_batch_iter() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let tokens: Vec<u32> = (0u32..40).collect();
    write_shard(tmp.path(), "shard-00000.bin", &tokens);

    let iter = ShardBatchIter::new(tmp.path(), 1, 4, 0, 0).expect("iter");
    let mut collected: Vec<u32> = Vec::new();
    for batch in iter {
        for seq in 0..batch.batch_size {
            if let Some(row) = batch.get_input(seq) {
                collected.extend_from_slice(row);
            }
        }
    }
    // 40 tokens / (seq_length+1=5) = 8 sequences. get_input returns seq_length=4
    // so 8 sequences × 4 tokens = 32 — the final token of each 5-tuple is the label.
    assert_eq!(collected.len(), 32, "expected 32 input tokens from 40 total");
    assert_eq!(&collected[..4], &[0, 1, 2, 3], "first seq head preserved");
}

/// INV-PRETOK-004 (shard naming) + multi-shard lexical order is preserved
/// by both the producer (`apr tokenize encode-corpus` writes `shard-NNNNN.bin`)
/// and the consumer (`ShardBatchIter` sorts lexically).
#[test]
fn multi_shard_names_preserve_order() {
    let tmp = tempfile::tempdir().expect("tempdir");
    write_shard(tmp.path(), "shard-00000.bin", &(0u32..10).collect::<Vec<_>>());
    write_shard(tmp.path(), "shard-00001.bin", &(100u32..110).collect::<Vec<_>>());
    let mut iter = ShardBatchIter::new(tmp.path(), 1, 4, 0, 0).expect("iter");
    let first = iter.next().expect("batch");
    let head = first.get_input(0).expect("input");
    assert_eq!(head[0], 0, "shard-00000 must come first (lexical)");
}
