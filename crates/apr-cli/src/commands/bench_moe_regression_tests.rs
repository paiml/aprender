// #1749 regression fixture — `apr bench` on a Qwen3-MoE GGUF used to take
// the dense path and panic in `matmul_fused.rs:211` (index out of bounds).
// Fix: `benchmark.rs`'s `run_gguf_benchmark` checks `is_moe_gguf(&gguf)`
// and routes to `run_gguf_moe_benchmark` (`bench_moe.rs`) instead. Closed
// by 541c2477d / PR #1751. This file is the missing regression test
// (measured: no test covered this path before PMAT-3429).
//
// The fixture tensor table (names/dims/metadata keys) mirrors
// `crates/aprender-serve/src/gguf/regression_fixtures.rs::build_qwen3_moe_gguf`
// (aprender-serve's own #2535/#3091 pin), rebuilt here with apr-cli's own
// in-tree GGUF writer (`aprender::format::gguf::export_tensors_to_gguf`)
// rather than importing aprender-serve's internal `test_factory::GGUFBuilder`.
//
// Two wrinkles from doing it with the OTHER writer:
//
// 1. `export_tensors_to_gguf` writes `GgufTensor.shape` verbatim (it does
//    NOT reverse dims), whereas aprender-serve's `GGUFBuilder::build()`
//    reverses row-major dims into GGML `ne[]` order before writing, and
//    the realizar reader (`reading.rs`) reverses back on load. To land on
//    the same row-major shape the realizar loader expects, every tensor
//    below is declared with dims already in GGML order — i.e. the REVERSE
//    of `regression_fixtures.rs`'s row-major dims.
// 2. `bench_moe.rs::run_gguf_moe_benchmark` additionally requires
//    `{arch}.expert_feed_forward_length`, which the aprender-serve fixture
//    doesn't set (it only exercises the load path, not the bench path) —
//    added here. A `tokenizer.ggml.tokens` array is also added so
//    `GGUFModel::encode` succeeds instead of falling back to a fixed
//    token list that can exceed this fixture's small vocab.

#[cfg(test)]
mod regression_1749_tests {
    use aprender::format::gguf::{export_tensors_to_gguf, GgmlType, GgufTensor, GgufValue};
    use sha2::{Digest, Sha256};
    use std::io::{BufWriter, Write};
    use tempfile::NamedTempFile;

    const HIDDEN: usize = 256;
    const INTERMEDIATE: usize = 256;
    const VOCAB: usize = 32;
    const NUM_EXPERTS: usize = 2;
    const HEADS: usize = 4;

    /// All-zero Q4_K bytes for a flat (non-2D) tensor of `n` elements —
    /// mirrors `aprender-serve`'s `test_factory::create_q4_k_data`.
    fn q4k_flat(n: usize) -> Vec<u8> {
        vec![0u8; n.div_ceil(256) * 144]
    }

    /// All-zero, row-padded Q4_K bytes for a 2D `rows x cols` tensor —
    /// mirrors `test_factory::create_q4_k_data_2d`.
    fn q4k_2d(rows: usize, cols: usize) -> Vec<u8> {
        vec![0u8; rows * cols.div_ceil(256) * 144]
    }

    fn f32_bytes(data: &[f32]) -> Vec<u8> {
        data.iter().flat_map(|f| f.to_le_bytes()).collect()
    }

    fn embedding_data(vocab: usize, hidden: usize) -> Vec<f32> {
        (0..vocab * hidden)
            .map(|i| ((i % 1000) as f32 - 500.0) / 5000.0)
            .collect()
    }

    fn norm_data(dim: usize) -> Vec<f32> {
        vec![1.0f32; dim]
    }

    /// Build the #1749 regression fixture: a minimal single-layer `qwen3moe`
    /// GGUF whose expert tensors are Q4_K (a supported expert qtype). Byte
    /// layout is deliberately pinned below (`regression_fixture_bytes_are_pinned`).
    fn build_qwen3_moe_gguf() -> Vec<u8> {
        let arch = "qwen3moe";

        // NOTE: `shape` below is GGML (`ne[]`) order — the REVERSE of the
        // row-major dims `regression_fixtures.rs` passes to `GGUFBuilder`
        // (see module doc). Square 2D tensors reverse to themselves.
        let tensors = vec![
            GgufTensor {
                name: "token_embd.weight".to_string(),
                shape: vec![HIDDEN as u64, VOCAB as u64], // rev([VOCAB, HIDDEN])
                dtype: GgmlType::F32,
                data: f32_bytes(&embedding_data(VOCAB, HIDDEN)),
            },
            GgufTensor {
                name: "blk.0.attn_norm.weight".to_string(),
                shape: vec![HIDDEN as u64],
                dtype: GgmlType::F32,
                data: f32_bytes(&norm_data(HIDDEN)),
            },
            GgufTensor {
                name: "blk.0.attn_q.weight".to_string(),
                shape: vec![HIDDEN as u64, HIDDEN as u64],
                dtype: GgmlType::Q4K,
                data: q4k_2d(HIDDEN, HIDDEN),
            },
            GgufTensor {
                name: "blk.0.attn_k.weight".to_string(),
                shape: vec![HIDDEN as u64, HIDDEN as u64],
                dtype: GgmlType::Q4K,
                data: q4k_2d(HIDDEN, HIDDEN),
            },
            GgufTensor {
                name: "blk.0.attn_v.weight".to_string(),
                shape: vec![HIDDEN as u64, HIDDEN as u64],
                dtype: GgmlType::Q4K,
                data: q4k_2d(HIDDEN, HIDDEN),
            },
            GgufTensor {
                name: "blk.0.attn_output.weight".to_string(),
                shape: vec![HIDDEN as u64, HIDDEN as u64],
                dtype: GgmlType::Q4K,
                data: q4k_2d(HIDDEN, HIDDEN),
            },
            GgufTensor {
                name: "blk.0.ffn_norm.weight".to_string(),
                shape: vec![HIDDEN as u64],
                dtype: GgmlType::F32,
                data: f32_bytes(&norm_data(HIDDEN)),
            },
            GgufTensor {
                name: "blk.0.ffn_gate_inp.weight".to_string(),
                shape: vec![HIDDEN as u64, NUM_EXPERTS as u64], // rev([NUM_EXPERTS, HIDDEN])
                dtype: GgmlType::F32,
                data: f32_bytes(&vec![0.0f32; NUM_EXPERTS * HIDDEN]),
            },
            GgufTensor {
                name: "blk.0.ffn_gate_exps.weight".to_string(),
                // rev([NUM_EXPERTS, INTERMEDIATE, HIDDEN])
                shape: vec![HIDDEN as u64, INTERMEDIATE as u64, NUM_EXPERTS as u64],
                dtype: GgmlType::Q4K,
                data: q4k_flat(NUM_EXPERTS * INTERMEDIATE * HIDDEN),
            },
            GgufTensor {
                name: "blk.0.ffn_up_exps.weight".to_string(),
                shape: vec![HIDDEN as u64, INTERMEDIATE as u64, NUM_EXPERTS as u64],
                dtype: GgmlType::Q4K,
                data: q4k_flat(NUM_EXPERTS * INTERMEDIATE * HIDDEN),
            },
            GgufTensor {
                name: "blk.0.ffn_down_exps.weight".to_string(),
                shape: vec![HIDDEN as u64, INTERMEDIATE as u64, NUM_EXPERTS as u64],
                dtype: GgmlType::Q4K,
                data: q4k_flat(NUM_EXPERTS * INTERMEDIATE * HIDDEN),
            },
            GgufTensor {
                name: "output_norm.weight".to_string(),
                shape: vec![HIDDEN as u64],
                dtype: GgmlType::F32,
                data: f32_bytes(&norm_data(HIDDEN)),
            },
        ];

        // Minimal vocab so `GGUFModel::encode` succeeds (`Some`) instead of
        // falling back to a fixed token list that can exceed VOCAB — every
        // unmatched byte in the prompt maps to token id 0, which is always
        // in-range as long as the array is non-empty.
        let tokens: Vec<String> = (0..VOCAB).map(|i| format!("<tok{i}>")).collect();

        let metadata = vec![
            (
                "general.architecture".to_string(),
                GgufValue::String(arch.to_string()),
            ),
            (
                format!("{arch}.embedding_length"),
                GgufValue::Uint32(HIDDEN as u32),
            ),
            (format!("{arch}.block_count"), GgufValue::Uint32(1)),
            (
                format!("{arch}.attention.head_count"),
                GgufValue::Uint32(HEADS as u32),
            ),
            (
                format!("{arch}.attention.head_count_kv"),
                GgufValue::Uint32(HEADS as u32),
            ),
            (format!("{arch}.context_length"), GgufValue::Uint32(256)),
            (
                format!("{arch}.rope.freq_base"),
                GgufValue::Float32(10000.0),
            ),
            (
                format!("{arch}.attention.layer_norm_rms_epsilon"),
                GgufValue::Float32(1e-6),
            ),
            (
                format!("{arch}.feed_forward_length"),
                GgufValue::Uint32(INTERMEDIATE as u32),
            ),
            (
                format!("{arch}.expert_count"),
                GgufValue::Uint32(NUM_EXPERTS as u32),
            ),
            (format!("{arch}.expert_used_count"), GgufValue::Uint32(1)),
            // Required by bench_moe.rs::run_gguf_moe_benchmark, unlike the
            // aprender-serve load-only fixture this table mirrors.
            (
                format!("{arch}.expert_feed_forward_length"),
                GgufValue::Uint32(INTERMEDIATE as u32),
            ),
            (
                "tokenizer.ggml.tokens".to_string(),
                GgufValue::ArrayString(tokens),
            ),
        ];

        let mut bytes = Vec::new();
        {
            let mut writer = BufWriter::new(&mut bytes);
            export_tensors_to_gguf(&mut writer, &tensors, &metadata).expect("write GGUF fixture");
        }
        bytes
    }

    /// Fixture change must be deliberate (mirrors the `#3091`/`#2535` pin in
    /// `regression_fixtures.rs::regression_fixture_bytes_are_pinned`).
    #[test]
    fn regression_1749_fixture_bytes_are_pinned() {
        let bytes = build_qwen3_moe_gguf();
        let hash = format!("{:x}", Sha256::digest(&bytes));
        assert_eq!(
            hash, "627a7edf665b2748d0cc6ee902b1be3c0a01521f9cc6391497d4ee6d8c679ae0",
            "#1749: fixture bytes changed — update the pin deliberately, got {hash}"
        );
    }

    /// The dispatch premise: this GGUF must actually be seen as MoE by the
    /// same predicate `run_gguf_benchmark` guards on (#1749).
    #[test]
    fn regression_1749_fixture_is_moe_gguf() {
        let bytes = build_qwen3_moe_gguf();
        let gguf = realizar::gguf::GGUFModel::from_bytes(&bytes)
            .expect("#1749: fixture GGUF must parse");
        assert!(
            gguf.expert_count().unwrap_or(0) > 0,
            "#1749: fixture must report expert_count > 0 so is_moe_gguf() routes it to the MoE bench path"
        );
    }

    /// THE regression (#1749): `apr bench` on a Qwen3-MoE GGUF must route
    /// through `run_gguf_moe_benchmark` and GENERATE. Asserted on the token
    /// count, never on throughput: `run`'s `passed` flag is a 10 tok/s
    /// wall-clock floor (spec H12), and a required check must not depend on
    /// how loaded the runner is. Measured under the inverse of the fix (the
    /// `is_moe_gguf` dispatch deleted): the dense path returns Ok with 0
    /// tokens, so `total_tokens == iterations * max_tokens` is the
    /// discriminator.
    #[test]
    fn regression_1749_moe_gguf_bench_takes_the_moe_route() {
        let bytes = build_qwen3_moe_gguf();
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(&bytes).expect("write fixture to disk");

        let config = super::BenchConfig {
            warmup: 0,
            iterations: 1,
            max_tokens: 1,
            prompt: "2+2=".to_string(),
            quiet: true,
        };
        let result = super::run_realizar_benchmark(file.path(), &config);

        match result {
            Ok(r) => assert_eq!(
                r.total_tokens, 1,
                "#1749: apr bench on a Qwen3-MoE GGUF generated {} tokens, want 1 — the MoE \
                 dispatch to run_gguf_moe_benchmark did not run",
                r.total_tokens
            ),
            Err(e) => panic!(
                "#1749: apr bench on a Qwen3-MoE GGUF must route through run_gguf_moe_benchmark, \
                 got: {e:?}"
            ),
        }
    }
}
