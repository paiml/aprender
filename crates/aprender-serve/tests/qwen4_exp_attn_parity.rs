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
    assert!(got.iter().all(|v| v.is_finite()), "non-finite output");
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

fn indexer_params<'a>(w: &'a [Vec<f32>]) -> IndexerParams<'a> {
    IndexerParams {
        qk_proj: &w[0],
        q_norm: &w[1],
        k_norm: &w[2],
    }
}

/// Run `indexer_select` over both batches; returns the concatenated `[2, seq, seq]` mask.
fn select_all(p: &IndexerParams, s: &IndexerShape, g: &Value) -> Vec<bool> {
    let (hid, cos, sin) = (
        f32s(&g["inputs"]["hidden_states"]),
        f32s(&g["inputs"]["cos"]),
        f32s(&g["inputs"]["sin"]),
    );
    let vis = bools(&g["inputs"]["visible_mask"]);
    let seq = seq_len(g);
    let (hs, rs, ms) = (seq * s.hidden, cos.len() / 2, seq * seq);
    let mut out = Vec::new();
    for b in 0..2 {
        out.extend(
            indexer_select(
                p,
                s,
                &hid[b * hs..(b + 1) * hs],
                &cos[b * rs..(b + 1) * rs],
                &sin[b * rs..(b + 1) * rs],
                &vis[b * ms..(b + 1) * ms],
            )
            .expect("indexer"),
        );
    }
    out
}

fn row_matches(got: &[bool], want: &[bool], seq: usize) -> usize {
    (0..got.len() / seq)
        .filter(|&r| got[r * seq..(r + 1) * seq] == want[r * seq..(r + 1) * seq])
        .count()
}

#[test]
fn indexer_selection_matches_reference() {
    let g = golden("indexer");
    let w = [
        f32s(&g["params"]["index_qk_proj.weight"]),
        f32s(&g["params"]["q_layernorm.weight"]),
        f32s(&g["params"]["k_layernorm.weight"]),
    ];
    let s = indexer_shape(&g);
    let got = select_all(&indexer_params(&w), &s, &g);
    let want = bools(&g["output"]["selected_token_mask"]);
    let seq = seq_len(&g);
    let rows = row_matches(&got, &want, seq);
    println!("indexer rows matched {rows}/{}", 2 * seq);
    assert_eq!(
        rows,
        2 * seq,
        "selected-token mask differs from the reference"
    );
}

/// The golden must stay tie-free: a future tied golden fails here instead of being accepted.
#[test]
fn golden_topk_margin_is_at_least_1e_3() {
    for name in ["indexer", "indexed_attention"] {
        let g = golden(name);
        let m = &g["topk_margin"];
        let gap = m["min_kth_vs_k1th_block_score_gap"].as_f64().expect("gap");
        let req = m["required"].as_f64().expect("required");
        assert!(req >= 1e-3, "{name}: golden requires only {req}");
        assert!(
            gap >= 1e-3,
            "{name}: top-k margin {gap} < 1e-3 (tied golden)"
        );
    }
    // The dumped block-score case: weakest selected block strictly beats the strongest unselected.
    let g = golden("indexer");
    let scores = f32s(&g["intermediates"]["block_scores"]);
    let sel = usizes(&g["intermediates"]["selected_blocks"]);
    let weakest = sel.iter().map(|&i| scores[i]).fold(f32::INFINITY, f32::min);
    let strongest_out = (0..scores.len())
        .filter(|i| !sel.contains(i))
        .map(|i| scores[i])
        .fold(f32::NEG_INFINITY, f32::max);
    assert!(weakest - strongest_out >= 1e-3, "dumped case is tied");
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
    let iw = [
        w("indexer.index_qk_proj.weight"),
        w("indexer.q_layernorm.weight"),
        w("indexer.k_layernorm.weight"),
    ];
    let p = AttentionParams {
        q_proj: &qp,
        k_proj: &kp,
        v_proj: &vp,
        o_proj: &op,
        q_norm: &qn,
        k_norm: &kn,
        indexer: indexer_params(&iw),
    };
    let a = AttentionShape {
        n_heads: cfg_usize(&g, "num_attention_heads"),
        n_kv: cfg_usize(&g, "num_key_value_heads"),
        head_dim: cfg_usize(&g, "head_dim"),
        indexer: indexer_shape(&g),
    };
    let seq = seq_len(&g);
    // The attention's own selection must equal the reference mask on every row.
    let sel = select_all(&p.indexer, &a.indexer, &g);
    let want_sel = bools(&g["output"]["selection_mask"]);
    let rows = row_matches(&sel, &want_sel, seq);
    println!("attention selection rows matched {rows}/{}", 2 * seq);
    assert_eq!(rows, 2 * seq, "selection differs from selection_mask");

    let (hid, cos, sin, vis, want) = (
        f32s(&g["inputs"]["hidden_states"]),
        f32s(&g["inputs"]["cos"]),
        f32s(&g["inputs"]["sin"]),
        bools(&g["inputs"]["visible_mask"]),
        f32s(&g["output"]["attn_output"]),
    );
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

#[test]
fn query_with_no_visible_key_yields_zero_row_not_panic() {
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
    let iw = [
        w("indexer.index_qk_proj.weight"),
        w("indexer.q_layernorm.weight"),
        w("indexer.k_layernorm.weight"),
    ];
    let p = AttentionParams {
        q_proj: &qp,
        k_proj: &kp,
        v_proj: &vp,
        o_proj: &op,
        q_norm: &qn,
        k_norm: &kn,
        indexer: indexer_params(&iw),
    };
    let a = AttentionShape {
        n_heads: cfg_usize(&g, "num_attention_heads"),
        n_kv: cfg_usize(&g, "num_key_value_heads"),
        head_dim: cfg_usize(&g, "head_dim"),
        indexer: indexer_shape(&g),
    };
    let seq = seq_len(&g);
    let h = a.indexer.hidden;
    let hid = f32s(&g["inputs"]["hidden_states"]);
    let cos = f32s(&g["inputs"]["cos"]);
    let sin = f32s(&g["inputs"]["sin"]);
    let mut vis = bools(&g["inputs"]["visible_mask"])[..seq * seq].to_vec();
    let want = f32s(&g["output"]["attn_output"]);
    // A left-pad query: its row sees no key at all (not a hand-edited diagonal).
    vis[..seq].fill(false);
    let (hs, rs) = (seq * h, cos.len() / 2);
    let got = indexed_attention(&p, &a, &hid[..hs], &cos[..rs], &sin[..rs], &vis)
        .expect("empty row must not error or panic");
    assert!(got[..h].iter().all(|v| *v == 0.0), "empty row is not zero");
    // Other rows do not depend on row 0's visibility (queries equal keys).
    let e = max_err(&got[h..], &want[h..hs]);
    println!("rows after the empty one: max_abs_err = {e:e}");
    assert!(e <= TOL, "max abs err {e}");
    // Mis-sized weights are reported, not panicked on.
    let short = &kp[..kp.len() - 1];
    let bad = AttentionParams { k_proj: short, ..p };
    assert!(indexed_attention(&bad, &a, &hid[..hs], &cos[..rs], &sin[..rs], &vis).is_err());
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
