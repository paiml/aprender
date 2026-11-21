//! #7175 S2: gated norm, n-gram embedding and PLE vs transformers goldens (max-abs <= 1e-5);
//! #7332: their decode paths (n-gram carry-over, PLE conv state) vs the same goldens.
use realizar::qwen4_exp_embed::{
    gated_rms_norm, ngram_embed, ngram_ids, ple_forward, GateAct, NgramConfig, NgramState,
    PleParams, PleState,
};
use serde_json::Value;

const TOL: f32 = 1e-5;

fn golden(name: &str) -> Value {
    let p = format!(
        "{}/tests/fixtures/qwen4_exp/goldens/{name}.json",
        env!("CARGO_MANIFEST_DIR")
    );
    serde_json::from_str(&std::fs::read_to_string(p).expect("read golden")).expect("json")
}

fn f32s(v: &Value) -> Vec<f32> {
    v["data"]
        .as_array()
        .expect("data")
        .iter()
        .map(|x| x.as_f64().expect("num") as f32)
        .collect()
}

fn i64s(v: &Value) -> Vec<i64> {
    v["data"]
        .as_array()
        .expect("data")
        .iter()
        .map(|x| x.as_i64().expect("int"))
        .collect()
}

fn max_err(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn ngram_cfg(c: &Value, ple_layer_index: usize) -> NgramConfig {
    NgramConfig {
        vocab_size: c["vocab_size"].as_u64().expect("vocab"),
        ngram_size: c["ngram_size"].as_u64().expect("n") as usize,
        heads_per_ngram: c["heads_per_ngram"].as_u64().expect("h") as usize,
        ngram_vocab_size_base: c["ngram_vocab_size_base"].as_u64().expect("b"),
        seed: c["seed"].as_u64().expect("seed"),
        eos_token_id: c["eos_token_id"].as_i64().expect("eos"),
        embed_dim: c["ple_embed_dim"].as_u64().expect("e") as usize,
        ple_layer_index,
    }
}

#[test]
fn output_gate_matches_reference() {
    let g = golden("output_gate");
    let w = f32s(&g["params"]["weight"]);
    let x = f32s(&g["inputs"]["hidden_states"]);
    let z = f32s(&g["inputs"]["gate"]);
    let eps = g["config"]["rms_norm_eps"].as_f64().expect("eps") as f32;
    let got = gated_rms_norm(&x, &z, &w, eps, GateAct::Sigmoid);
    let e = max_err(&got, &f32s(&g["output"]["out"]));
    eprintln!("output_gate max_abs_err = {e:e}");
    assert!(e <= TOL, "err {e}");
}

#[test]
fn ngram_embedding_matches_reference() {
    let g = golden("ngram");
    let cfg = ngram_cfg(
        &g["config"],
        g["ple_layer_index"].as_u64().expect("i") as usize,
    );
    let ids = i64s(&g["inputs"]["input_ids"]);
    let want_ids = i64s(&g["intermediates"]["ngram_ids"]);
    let table = f32s(&g["params"]["ngram_embedding.weight"]);
    let want = f32s(&g["output"]["embeddings"]);
    let (b, s) = (2, ids.len() / 2);
    let heads = want_ids.len() / ids.len();
    let mut got_ids = Vec::new();
    let mut got = Vec::new();
    for r in 0..b {
        let row = &ids[r * s..(r + 1) * s];
        got_ids.extend(ngram_ids(&cfg, row, None));
        got.extend(ngram_embed(&cfg, &table, row));
    }
    assert_eq!(got_ids.len(), heads * ids.len());
    assert_eq!(got_ids, want_ids, "hashed ids differ");
    let e = max_err(&got, &want);
    eprintln!("ngram max_abs_err = {e:e}");
    assert!(e <= TOL, "err {e}");
}

#[test]
fn ple_matches_reference() {
    let g = golden("ple");
    let c = &g["config"];
    let cfg = ngram_cfg(c, c["ple_layer_index"].as_u64().expect("i") as usize);
    let p = &g["params"];
    let table = f32s(&p["ple_embedding.ngram_embedding.weight"]);
    let (kp, vp) = (f32s(&p["key_proj.weight"]), f32s(&p["value_proj.weight"]));
    let (nk, nq, nc) = (
        f32s(&p["norm_key.weight"]),
        f32s(&p["norm_query.weight"]),
        f32s(&p["norm_conv.weight"]),
    );
    let cw = f32s(&p["conv1d.weight"]);
    let params = PleParams {
        ngram: &cfg,
        hidden_size: c["hidden_size"].as_u64().expect("h") as usize,
        hc_count: c["hc_count"].as_u64().expect("c") as usize,
        kernel: c["ple_conv_kernel_size"].as_u64().expect("k") as usize,
        eps: c["rms_norm_eps"].as_f64().expect("eps") as f32,
        table: &table,
        key_proj: &kp,
        value_proj: &vp,
        norm_key: &nk,
        norm_query: &nq,
        norm_conv: &nc,
        conv_w: &cw,
    };
    let hs = f32s(&g["inputs"]["hidden_states"]);
    let ids = i64s(&g["inputs"]["input_ids"]);
    let want = f32s(&g["output"]["ple_output"]);
    let (b, s) = (2, ids.len() / 2);
    let w = hs.len() / ids.len();
    let mut got = Vec::new();
    for r in 0..b {
        got.extend(ple_forward(
            &params,
            &hs[r * s * w..(r + 1) * s * w],
            &ids[r * s..(r + 1) * s],
        ));
    }
    let e = max_err(&got, &want);
    eprintln!("ple max_abs_err = {e:e}");
    assert!(e <= TOL, "err {e}");
}

// ---- #7332 decode goldens -------------------------------------------------
// Basis: the transformers prefill goldens above (ngram.json, ple.json). The
// reference's decode contract is that a cached step reproduces the no-cache
// forward row for that position, so feeding the same tokens one at a time (and
// in uneven chunks) through the carried state must reproduce the reference rows
// at TOL. No value produced by this crate is pinned as truth.

/// Chunkings of a `len`-token sequence: one token per step, then an uneven split.
fn chunkings(len: usize) -> Vec<Vec<usize>> {
    vec![vec![1; len], vec![1, 4, 2, len - 7]]
}

#[test]
fn qwen4_exp_decode_ngram_context_carry_over_matches_reference() {
    let g = golden("ngram");
    let cfg = ngram_cfg(
        &g["config"],
        g["ple_layer_index"].as_u64().expect("i") as usize,
    );
    let ids = i64s(&g["inputs"]["input_ids"]);
    let want_ids = i64s(&g["intermediates"]["ngram_ids"]);
    let table = f32s(&g["params"]["ngram_embedding.weight"]);
    let want = f32s(&g["output"]["embeddings"]);
    let s = ids.len() / 2;
    for steps in chunkings(s) {
        let (mut got_ids, mut got) = (Vec::new(), Vec::new());
        for r in 0..2 {
            let mut st = NgramState::new(&cfg);
            let mut at = r * s;
            for &n in &steps {
                let chunk = &ids[at..at + n];
                got_ids.extend(st.ids(&cfg, chunk));
                at += n;
            }
            let mut st = NgramState::new(&cfg);
            let mut at = r * s;
            for &n in &steps {
                got.extend(st.embed(&cfg, &table, &ids[at..at + n]));
                at += n;
            }
            assert_eq!(st.context(), &ids[(r + 1) * s - 2..(r + 1) * s]);
        }
        assert_eq!(got_ids, want_ids, "decode n-gram ids differ ({steps:?})");
        let e = max_err(&got, &want);
        eprintln!("decode ngram {steps:?} max_abs_err = {e:e}");
        assert!(e <= TOL, "err {e}");
    }
}

#[test]
fn qwen4_exp_decode_ple_conv_state_matches_reference() {
    let g = golden("ple");
    let c = &g["config"];
    let cfg = ngram_cfg(c, c["ple_layer_index"].as_u64().expect("i") as usize);
    let p = &g["params"];
    let table = f32s(&p["ple_embedding.ngram_embedding.weight"]);
    let (kp, vp) = (f32s(&p["key_proj.weight"]), f32s(&p["value_proj.weight"]));
    let (nk, nq, nc) = (
        f32s(&p["norm_key.weight"]),
        f32s(&p["norm_query.weight"]),
        f32s(&p["norm_conv.weight"]),
    );
    let cw = f32s(&p["conv1d.weight"]);
    let params = PleParams {
        ngram: &cfg,
        hidden_size: c["hidden_size"].as_u64().expect("h") as usize,
        hc_count: c["hc_count"].as_u64().expect("c") as usize,
        kernel: c["ple_conv_kernel_size"].as_u64().expect("k") as usize,
        eps: c["rms_norm_eps"].as_f64().expect("eps") as f32,
        table: &table,
        key_proj: &kp,
        value_proj: &vp,
        norm_key: &nk,
        norm_query: &nq,
        norm_conv: &nc,
        conv_w: &cw,
    };
    let hs = f32s(&g["inputs"]["hidden_states"]);
    let ids = i64s(&g["inputs"]["input_ids"]);
    let want = f32s(&g["output"]["ple_output"]);
    let s = ids.len() / 2;
    let w = hs.len() / ids.len();
    // The conv reaches back (K-1)*dilation = 9 rows; 12 tokens step past it.
    assert!(s > (params.kernel - 1) * cfg.ngram_size, "golden too short");
    for steps in chunkings(s) {
        let mut got = Vec::new();
        for r in 0..2 {
            let mut st = PleState::new(&params);
            let mut at = r * s;
            for &n in &steps {
                got.extend(st.step(&params, &hs[at * w..(at + n) * w], &ids[at..at + n]));
                at += n;
            }
        }
        let e = max_err(&got, &want);
        eprintln!("decode ple {steps:?} max_abs_err = {e:e}");
        assert!(e <= TOL, "err {e}");
    }
}
