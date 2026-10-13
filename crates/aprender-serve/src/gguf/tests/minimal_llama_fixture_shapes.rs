//! #4985: `build_minimal_llama_gguf` shapes every 2-D weight [out, in] and sizes it out rows
//! of in columns, so a non-square FFN (and a GQA K/V) fixture loads and runs a forward.
//!
//! Before the fix the fixture declared ffn_up/ffn_gate [hidden, intermediate] and sized them
//! hidden rows, half the bytes the intermediate-row matmul needs, and the forward refused it
//! (`validate_matmul_weight_shape`). Callers worked around it with intermediate = hidden.

use crate::gguf::test_factory::build_minimal_llama_gguf;
use crate::gguf::{GGUFModel, MappedGGUFModel, OwnedQuantizedModel};
use std::io::Write;

const VOCAB: usize = 32;
const HIDDEN: usize = 64;
const INTERMEDIATE: usize = 128;
const HEADS: usize = 4;
const KV_HEADS: usize = 2;
const KV_DIM: usize = KV_HEADS * (HIDDEN / HEADS);

fn non_square_fixture() -> Vec<u8> {
    build_minimal_llama_gguf(VOCAB, HIDDEN, INTERMEDIATE, HEADS, KV_HEADS)
}

#[test]
fn every_2d_weight_reads_back_out_by_in() {
    let bytes = non_square_fixture();
    let model = GGUFModel::from_bytes(&bytes).expect("parse the fixture");
    let want: [(&str, [u64; 2]); 7] = [
        ("blk.0.attn_q.weight", [HIDDEN as u64, HIDDEN as u64]),
        ("blk.0.attn_k.weight", [KV_DIM as u64, HIDDEN as u64]),
        ("blk.0.attn_v.weight", [KV_DIM as u64, HIDDEN as u64]),
        ("blk.0.attn_output.weight", [HIDDEN as u64, HIDDEN as u64]),
        ("blk.0.ffn_up.weight", [INTERMEDIATE as u64, HIDDEN as u64]),
        (
            "blk.0.ffn_gate.weight",
            [INTERMEDIATE as u64, HIDDEN as u64],
        ),
        (
            "blk.0.ffn_down.weight",
            [HIDDEN as u64, INTERMEDIATE as u64],
        ),
    ];
    for (name, dims) in want {
        let t = model
            .tensors
            .iter()
            .find(|t| t.name == name)
            .unwrap_or_else(|| panic!("{name} is in the fixture"));
        assert_eq!(t.dims, dims, "{name} is [out, in]");
    }
}

#[test]
fn a_non_square_ffn_fixture_runs_a_forward() {
    let mut f = tempfile::NamedTempFile::with_suffix(".gguf").expect("temp");
    f.write_all(&non_square_fixture()).expect("write");
    let mapped = MappedGGUFModel::from_path(f.path()).expect("map");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("load");
    let logits = model
        .forward(&[2])
        .expect("every weight holds the bytes its out rows need");
    assert_eq!(logits.len(), VOCAB);
    assert!(logits.iter().all(|x| x.is_finite()));
}
