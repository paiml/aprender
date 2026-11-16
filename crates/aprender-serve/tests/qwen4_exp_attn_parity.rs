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

/// True when the k-th and (k+1)-th best block scores are equal: `torch.topk` documents no
/// tie order, so the reference's choice among the tied blocks cannot be pinned by a golden.
fn boundary_tie(scores: &[f32], k: usize) -> bool {
    let k = k.min(scores.len());
    if k == 0 || k >= scores.len() {
        return false;
    }
    let mut s = scores.to_vec();
    s.sort_by(|a, b| b.total_cmp(a));
    s[k - 1] == s[k]
}

/// Sorted scores of the blocks a mask row selects (a block counts when its first token is set).
fn chosen_scores(row: &[bool], visible: &[bool], ratio: usize, scores: &[f32]) -> Vec<f32> {
    let vis: Vec<usize> = (0..visible.len()).filter(|&i| visible[i]).collect();
    let mut out: Vec<f32> = (0..scores.len())
        .filter(|&b| row[vis[b * ratio]])
        .map(|b| scores[b])
        .collect();
    out.sort_by(|a, b| a.total_cmp(b));
    out
}

struct IndexerCase {
    qkw: Vec<f32>,
    qn: Vec<f32>,
    kn: Vec<f32>,
    s: IndexerShape,
    seq: usize,
}

impl IndexerCase {
    fn params(&self) -> IndexerParams<'_> {
        IndexerParams {
            qk_proj: &self.qkw,
            q_norm: &self.qn,
            k_norm: &self.kn,
        }
    }
}

#[test]
fn indexer_selection_matches_reference() {
    let g = golden("indexer");
    let c = IndexerCase {
        qkw: f32s(&g["params"]["index_qk_proj.weight"]),
        qn: f32s(&g["params"]["q_layernorm.weight"]),
        kn: f32s(&g["params"]["k_layernorm.weight"]),
        s: indexer_shape(&g),
        seq: seq_len(&g),
    };
    let (hid, cos, sin) = (
        f32s(&g["inputs"]["hidden_states"]),
        f32s(&g["inputs"]["cos"]),
        f32s(&g["inputs"]["sin"]),
    );
    let (vis, want) = (
        bools(&g["inputs"]["visible_mask"]),
        bools(&g["output"]["selected_token_mask"]),
    );
    let (seq, s) = (c.seq, &c.s);
    let (hs, rs, ms) = (seq * s.hidden, cos.len() / 2, seq * seq);
    let (mut exact, mut tied) = (0, Vec::new());
    for b in 0..2 {
        let (h, co, si, v) = (
            &hid[b * hs..(b + 1) * hs],
            &cos[b * rs..(b + 1) * rs],
            &sin[b * rs..(b + 1) * rs],
            &vis[b * ms..(b + 1) * ms],
        );
        let got = indexer_select(&c.params(), s, h, co, si, v).expect("indexer");
        for t in 0..seq {
            let (gr, wr) = (
                &got[t * seq..(t + 1) * seq],
                &want[b * ms + t * seq..][..seq],
            );
            if gr == wr {
                exact += 1;
                continue;
            }
            // A differing row is only acceptable if the top-k boundary is tied and the
            // reference's choice is an equally valid top-k (same multiset of block scores).
            let sc = indexer_block_scores_row(&c.params(), s, h, co, si, v, t).expect("scores");
            assert!(
                boundary_tie(&sc, s.budget / s.ratio),
                "b={b} q={t}: differs without a tie"
            );
            let vr = &v[t * seq..(t + 1) * seq];
            assert_eq!(
                chosen_scores(gr, vr, s.ratio, &sc),
                chosen_scores(wr, vr, s.ratio, &sc),
                "b={b} q={t}: reference selection is not an equal-score top-k"
            );
            tied.push((b, t));
        }
    }
    println!("indexer rows exact={exact} tie-ambiguous={tied:?}");
    assert_eq!(exact + tied.len(), 2 * seq);
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
    // Rows whose indexer top-k boundary is tied (relu-dead blocks, score exactly 0) are
    // ambiguous vs torch.topk's unspecified tie order; every other row must match.
    let (h, s) = (a.indexer.hidden, &a.indexer);
    let (mut worst, mut worst_tied, mut tied) = (0f32, 0f32, Vec::new());
    for b in 0..2 {
        for t in 0..seq {
            let lo = (b * seq + t) * h;
            let e = max_err(&got[lo..lo + h], &want[lo..lo + h]);
            let sc = indexer_block_scores_row(
                &p.indexer,
                s,
                &hid[b * hs..(b + 1) * hs],
                &cos[b * rs..(b + 1) * rs],
                &sin[b * rs..(b + 1) * rs],
                &vis[b * ms..(b + 1) * ms],
                t,
            )
            .expect("scores");
            if boundary_tie(&sc, s.budget / s.ratio) {
                tied.push((b, t));
                worst_tied = worst_tied.max(e);
            } else {
                worst = worst.max(e);
            }
        }
    }
    println!("indexed_attention max_abs_err (untied rows) = {worst:e}; tied rows {tied:?} err {worst_tied:e}");
    assert!(worst <= TOL, "max abs err {worst}");
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
