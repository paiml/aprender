//! #7175 (master #7169): qwen4_exp indexer, indexed attention and hc residual
//! against goldens from the transformers reference (see goldens/MANIFEST.md).
use realizar::qwen4_exp_attn::*;
use serde_json::Value;

const TOL: f32 = 1e-5;

fn golden(name: &str) -> Value {
    let path = format!(
        "{}/tests/fixtures/qwen4_exp/goldens/{name}.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(&path).expect("read golden");
    serde_json::from_str(&text).expect("parse golden")
}

fn data(v: &Value) -> &Vec<Value> {
    v["data"].as_array().expect("tensor data")
}

fn f32s(v: &Value) -> Vec<f32> {
    data(v)
        .iter()
        .map(|x| x.as_f64().expect("num") as f32)
        .collect()
}

fn bools(v: &Value) -> Vec<bool> {
    data(v)
        .iter()
        .map(|x| x.as_f64().expect("num") != 0.0)
        .collect()
}

fn usizes(v: &Value) -> Vec<usize> {
    data(v)
        .iter()
        .map(|x| x.as_u64().expect("int") as usize)
        .collect()
}

fn max_err(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len(), "length");
    got.iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}

fn cfg_usize(g: &Value, k: &str) -> usize {
    g["config"][k].as_u64().expect(k) as usize
}

fn indexer_shape(g: &Value) -> IndexerShape {
    IndexerShape {
        hidden: cfg_usize(g, "hidden_size"),
        n_heads: cfg_usize(g, "indexer_n_heads"),
        head_dim: cfg_usize(g, "indexer_head_dim"),
        budget: cfg_usize(g, "indexer_budget"),
        ratio: cfg_usize(g, "indexer_compress_ratio"),
        eps: g["config"]["rms_norm_eps"].as_f64().expect("eps") as f32,
    }
}

fn seq_len(g: &Value) -> usize {
    g["inputs"]["hidden_states"]["shape"][1]
        .as_u64()
        .expect("seq") as usize
}

#[test]
fn indexer_selection_matches_reference_exactly() {
    let g = golden("indexer");
    let (qkw, qn, kn) = (
        f32s(&g["params"]["index_qk_proj.weight"]),
        f32s(&g["params"]["q_layernorm.weight"]),
        f32s(&g["params"]["k_layernorm.weight"]),
    );
    let p = IndexerParams {
        qk_proj: &qkw,
        q_norm: &qn,
        k_norm: &kn,
    };
    let s = indexer_shape(&g);
    let (hid, cos, sin) = (
        f32s(&g["inputs"]["hidden_states"]),
        f32s(&g["inputs"]["cos"]),
        f32s(&g["inputs"]["sin"]),
    );
    let (vis, want) = (
        bools(&g["inputs"]["visible_mask"]),
        bools(&g["output"]["selected_token_mask"]),
    );
    let seq = seq_len(&g);
    let (hs, rs, ms) = (seq * s.hidden, cos.len() / 2, seq * seq);
    let mut mismatches = 0;
    for b in 0..2 {
        let got = indexer_select(
            &p,
            &s,
            &hid[b * hs..(b + 1) * hs],
            &cos[b * rs..(b + 1) * rs],
            &sin[b * rs..(b + 1) * rs],
            &vis[b * ms..(b + 1) * ms],
        )
        .expect("indexer");
        mismatches += got
            .iter()
            .zip(&want[b * ms..(b + 1) * ms])
            .filter(|(a, b)| a != b)
            .count();
    }
    assert_eq!(
        mismatches, 0,
        "selected-token mask differs from the reference"
    );
}

#[test]
fn indexer_golden_selection_is_not_tie_ambiguous() {
    // batch 0, query S-1: selected_blocks must be the top-4 of block_scores, and the
    // selection boundary is untied.
    let g = golden("indexer");
    let scores = f32s(&g["intermediates"]["block_scores"]);
    let sel = usizes(&g["intermediates"]["selected_blocks"]);
    // Scores of unselected blocks may tie (relu gives 0.0 twice); the selection is unambiguous
    // iff the weakest selected score is strictly above the strongest unselected one.
    let weakest = sel.iter().map(|&i| scores[i]).fold(f32::INFINITY, f32::min);
    let strongest_out = (0..scores.len())
        .filter(|i| !sel.contains(i))
        .map(|i| scores[i])
        .fold(f32::NEG_INFINITY, f32::max);
    assert!(weakest > strongest_out, "tie at the selection boundary");
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
    assert_eq!(idx[..sel.len()].to_vec(), sel);
}

#[test]
fn indexed_attention_matches_reference() {
    let g = golden("indexed_attention");
    let w = |k: &str| f32s(&g["params"][k]);
    let (qp, kp, vp, op, qn, kn) = (
        w("q_proj.weight"),
        w("k_proj.weight"),
        w("v_proj.weight"),
        w("o_proj.weight"),
        w("q_norm.weight"),
        w("k_norm.weight"),
    );
    let (iq, iqn, ikn) = (
        w("indexer.index_qk_proj.weight"),
        w("indexer.q_layernorm.weight"),
        w("indexer.k_layernorm.weight"),
    );
    let p = AttentionParams {
        q_proj: &qp,
        k_proj: &kp,
        v_proj: &vp,
        o_proj: &op,
        q_norm: &qn,
        k_norm: &kn,
        indexer: IndexerParams {
            qk_proj: &iq,
            q_norm: &iqn,
            k_norm: &ikn,
        },
    };
    let a = AttentionShape {
        n_heads: cfg_usize(&g, "num_attention_heads"),
        n_kv: cfg_usize(&g, "num_key_value_heads"),
        head_dim: cfg_usize(&g, "head_dim"),
        indexer: indexer_shape(&g),
    };
    let (hid, cos, sin, vis, want) = (
        f32s(&g["inputs"]["hidden_states"]),
        f32s(&g["inputs"]["cos"]),
        f32s(&g["inputs"]["sin"]),
        bools(&g["inputs"]["visible_mask"]),
        f32s(&g["output"]["attn_output"]),
    );
    let seq = seq_len(&g);
    let (hs, rs, ms) = (seq * a.indexer.hidden, cos.len() / 2, seq * seq);
    let mut got = Vec::new();
    for b in 0..2 {
        got.extend(
            indexed_attention(
                &p,
                &a,
                &hid[b * hs..(b + 1) * hs],
                &cos[b * rs..(b + 1) * rs],
                &sin[b * rs..(b + 1) * rs],
                &vis[b * ms..(b + 1) * ms],
            )
            .expect("attention"),
        );
    }
    let e = max_err(&got, &want);
    println!("indexed_attention max_abs_err = {e:e}");
    assert!(e <= TOL, "max abs err {e}");
}

fn hc_weights(p: &Value) -> Vec<Vec<f32>> {
    let mut w = vec![
        f32s(&p["hc_norm.weight"]),
        f32s(&p["input_mix_weight_down.weight"]),
        f32s(&p["input_mix_weight_up.weight"]),
    ];
    if p.get("block_inject_weight.weight").is_some() {
        w.push(f32s(&p["block_inject_weight.weight"]));
    }
    w
}

fn hc_params(w: &[Vec<f32>]) -> HcParams<'_> {
    HcParams {
        norm: &w[0],
        down: &w[1],
        up: &w[2],
        inject: w.get(3).map(|x| x.as_slice()),
    }
}

#[test]
fn hc_residual_matches_reference() {
    let g = golden("hc_residual");
    let s = HcShape {
        hidden: cfg_usize(&g, "hidden_size"),
        count: cfg_usize(&g, "hc_count"),
        lowrank: cfg_usize(&g, "hc_lowrank"),
        eps: g["config"]["rms_norm_eps"].as_f64().expect("eps") as f32,
    };
    let x = f32s(&g["inputs"]["hyper_input"]);
    let blk = f32s(&g["inputs"]["block_output"]);
    let w = hc_weights(&g["params_with_combine"]);
    let out = hc_gated_residual(&hc_params(&w), &s, &x).expect("hc");
    let o = &g["output_with_combine"];
    let e_mixed = max_err(&out.mixed, &f32s(&o["mixed_input"]));
    let e_inj = max_err(&out.inject, &f32s(&o["injection_weights"]));
    let comb = hc_combine(&s, &x, &blk, &out.inject).expect("combine");
    let e_comb = max_err(&comb, &f32s(&o["combined_next_streams"]));
    let wm = hc_weights(&g["final_mixer"]["params"]);
    let fm = hc_final_mixer(&hc_params(&wm), &s, &x).expect("mixer");
    let e_fin = max_err(&fm, &f32s(&g["final_mixer"]["output"]));
    let si = &g["stream_init"];
    let init = hc_stream_init(&s, &f32s(&si["input_embeds"]));
    let e_init = max_err(&init, &f32s(&si["repeated_streams"]));
    println!("hc max_abs_err mixed={e_mixed:e} inj={e_inj:e} combined={e_comb:e} final={e_fin:e} init={e_init:e}");
    for e in [e_mixed, e_inj, e_comb, e_fin, e_init] {
        assert!(e <= TOL, "max abs err {e}");
    }
}
