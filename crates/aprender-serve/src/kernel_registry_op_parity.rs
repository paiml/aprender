//! #4539 KREG-001 AC-3 for `ops[]` rows: parity receipts for the per-forward op kernels.
//!
//! An op kernel runs on f32 activations, so there is no block format to decode: the oracle is the
//! op's own definition evaluated in f64 on the same inputs (`f64_definition`), written here from
//! the formula, not from the kernel. It shares no code with the kernel, so the receipt says
//! `oracle_independent: true`.
//!
//! Most error models these rows declare (EM-RED for a norm's reduction followed by a rsqrt and a
//! scale, EM-ELEM for tanh) have no implemented bound in `aprender-kernel-oracle` yet, so the
//! receipt's margin block says `not_modelled` and decides nothing. EM-ROPE is modelled here
//! ([`modelled_bound`]): RoPE forms its angle in f32 and drifts ~1e-3 by position 32k, so an
//! f32-exact ceiling is the wrong test for it. What the receipt DOES claim is checked: the
//! committed receipt re-measures within its `tolerance_rel` on this host, and `precision=f32`
//! holds below [`F32_CEILING`] — or below the row's modelled bound, when it has one.
//!
//! `emit_op_parity_receipts` (ignored) writes them, like `emit_parity_receipts`:
//! `KREG_RECEIPT_OUT=evidence/kreg/parity KREG_GIT_SHA=<sha> cargo test -p aprender-serve --lib
//! emit_op_parity_receipts -- --ignored`.

use super::*;

pub(super) const OP_ORACLE: &str = "f64_definition";

/// One `ops[]` row the harness measures.
pub(super) struct Op {
    id: &'static str,
    source_fn: &'static str,
    /// The definition the f64 oracle evaluates; part of the receipt's input set.
    formula: &'static str,
    workload: OpWorkload,
    /// Draw the inputs from `rng`, run the kernel, and return its output with the f64 oracle's.
    run: fn(&mut rand::rngs::StdRng, usize) -> (Vec<f32>, Vec<f64>),
}

#[derive(Debug, Clone, Copy)]
struct OpWorkload {
    n: usize,
    seed: u64,
    trials: usize,
}

const OP_WORKLOAD: OpWorkload = OpWorkload {
    n: 896,
    seed: 0x4539_0f00,
    trials: 8,
};

const EPS: f32 = 1e-6;

fn activations(rng: &mut rand::rngs::StdRng, n: usize, r: f32) -> Vec<f32> {
    (0..n).map(|_| rng.random_range(-r..r)).collect()
}

fn gains(rng: &mut rand::rngs::StdRng, n: usize) -> Vec<f32> {
    (0..n).map(|_| rng.random_range(0.5..1.5)).collect()
}

fn run_rmsnorm(rng: &mut rand::rngs::StdRng, n: usize) -> (Vec<f32>, Vec<f64>) {
    let (x, w) = (activations(rng, n, 2.0), gains(rng, n));
    let mut got = vec![0.0f32; n];
    crate::gguf::ops::rms_norm_into(&x, &w, EPS, &mut got);
    let ms = x.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / n as f64;
    let inv = 1.0 / (ms + f64::from(EPS)).sqrt();
    let want = x
        .iter()
        .zip(&w)
        .map(|(a, g)| f64::from(*a) * inv * f64::from(*g))
        .collect();
    (got, want)
}

fn run_layernorm(rng: &mut rand::rngs::StdRng, n: usize) -> (Vec<f32>, Vec<f64>) {
    // An offset mean, so the centring the kernel does is exercised, not a no-op on ~0-mean input.
    let x: Vec<f32> = activations(rng, n, 2.0).iter().map(|v| v + 0.75).collect();
    let (w, b) = (gains(rng, n), activations(rng, n, 0.5));
    let mut got = vec![0.0f32; n];
    crate::gguf::ops::layer_norm_into(&x, &w, Some(&b), EPS, &mut got);
    let mean = x.iter().map(|v| f64::from(*v)).sum::<f64>() / n as f64;
    let var = x
        .iter()
        .map(|v| (f64::from(*v) - mean).powi(2))
        .sum::<f64>()
        / n as f64;
    let inv = 1.0 / (var + f64::from(EPS)).sqrt();
    let want = x
        .iter()
        .zip(w.iter().zip(&b))
        .map(|(a, (g, c))| (f64::from(*a) - mean) * inv * f64::from(*g) + f64::from(*c))
        .collect();
    (got, want)
}

fn run_gelu(rng: &mut rand::rngs::StdRng, n: usize) -> (Vec<f32>, Vec<f64>) {
    let x = activations(rng, n, 6.0);
    let mut got = x.clone();
    crate::gguf::ops::gelu(&mut got);
    let c = (2.0 / std::f64::consts::PI).sqrt();
    let want = x
        .iter()
        .map(|v| {
            let v = f64::from(*v);
            0.5 * v * (1.0 + (c * (v + 0.044_715 * v.powi(3))).tanh())
        })
        .collect();
    (got, want)
}

/// Argmax as `[max, first index of max]`: the logits are random, then a strict maximum is planted
/// TWICE, so the definition's tie-break (the FIRST index, as `>` keeps it) is exercised on every
/// trial. A kernel that took the last maximum (`>=`) is off by the gap between the two indices.
fn run_argmax(rng: &mut rand::rngs::StdRng, n: usize) -> (Vec<f32>, Vec<f64>) {
    let mut x = activations(rng, n, 8.0);
    let first = rng.random_range(0..n / 2);
    let second = rng.random_range(n / 2..n);
    let top = 9.0 + rng.random_range(0.0f32..1.0);
    x[first] = top;
    x[second] = top;
    let idx = crate::gguf::ops::argmax(&x) as usize;
    let got = vec![x.get(idx).copied().unwrap_or(f32::NAN), idx as f32];
    let max = x
        .iter()
        .fold(f64::NEG_INFINITY, |a, v| a.max(f64::from(*v)));
    let want_idx = x
        .iter()
        .position(|v| f64::from(*v) == max)
        .expect("a maximum exists");
    (got, vec![max, want_idx as f64])
}

/// RoPE's workload: Qwen2-shaped heads (head_dim 128, theta 1e6), both rotation layouts, and
/// positions across a 32k context. The kernel computes each angle `pos * theta^(-2i/d)` in f32, so
/// its error grows with the position; the formula names the position range it was measured over.
const ROPE_HEAD_DIM: usize = 128;
const ROPE_THETA: f32 = 1.0e6;
const ROPE_POS_MAX: usize = 32_768;

/// A weight-free model shell whose config the per-head ops read (`head_dim`, GQA grouping, RoPE).
fn model_shell(
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    rope_type: u32,
    vocab: usize,
) -> crate::gguf::OwnedQuantizedModel {
    let config = crate::gguf::GGUFConfig {
        architecture: "test".to_string(),
        constraints: crate::gguf::ArchConstraints::from_architecture("test"),
        hidden_dim: heads * head_dim,
        intermediate_dim: 256,
        num_layers: 1,
        num_heads: heads,
        num_kv_heads: kv_heads,
        vocab_size: vocab,
        context_length: ROPE_POS_MAX,
        eps: 1e-5,
        rope_type,
        rope_theta: ROPE_THETA,
        explicit_head_dim: None,
        query_pre_attn_scalar: None,
        bos_token_id: None,
        eos_token_id: None,
    };
    crate::gguf::test_helpers::create_test_model_with_config(&config)
}

fn rope_model(rope_type: u32) -> crate::gguf::OwnedQuantizedModel {
    model_shell(2, 2, ROPE_HEAD_DIM, rope_type, 16)
}

/// Qwen2.5-0.5B's attention shape: 14 query heads over 2 KV heads (GQA 7:1), head_dim 64.
const ATT_HEADS: usize = 14;
const ATT_KV_HEADS: usize = 2;
const ATT_HEAD_DIM: usize = 64;
const ATT_CACHE_MAX: usize = 512;

/// Decode-step attention over a cache of random length plus the current position, against
/// `softmax(q·k / sqrt(d)) · v` in f64 with query head `h` reading KV head `h / (H / KV)`.
fn run_attention(rng: &mut rand::rngs::StdRng, n: usize) -> (Vec<f32>, Vec<f64>) {
    let (d, kv_dim) = (ATT_HEAD_DIM, ATT_KV_HEADS * ATT_HEAD_DIM);
    assert_eq!(n, ATT_HEADS * d, "attention: n is the q dim");
    let model = model_shell(ATT_HEADS, ATT_KV_HEADS, d, 0, 16);
    let cache_len = rng.random_range(1..ATT_CACHE_MAX);
    let q = activations(rng, n, 1.0);
    let (k_cache, v_cache) = (
        activations(rng, cache_len * kv_dim, 1.0),
        activations(rng, cache_len * kv_dim, 2.0),
    );
    let (k_cur, v_cur) = (activations(rng, kv_dim, 1.0), activations(rng, kv_dim, 2.0));
    let mut got = vec![0.0f32; n];
    model.attention_with_cache_gqa_into(&q, &k_cache, &v_cache, &k_cur, &v_cur, &mut got);
    let row = |cache: &[f32], cur: &[f32], pos: usize, kv: usize| -> Vec<f64> {
        let src = if pos < cache_len {
            &cache[pos * kv_dim..(pos + 1) * kv_dim]
        } else {
            cur
        };
        src[kv * d..(kv + 1) * d]
            .iter()
            .map(|v| f64::from(*v))
            .collect()
    };
    let scale = 1.0 / (d as f64).sqrt();
    let mut want = Vec::with_capacity(n);
    for h in 0..ATT_HEADS {
        let kv = h / (ATT_HEADS / ATT_KV_HEADS);
        let qh: Vec<f64> = q[h * d..(h + 1) * d]
            .iter()
            .map(|v| f64::from(*v))
            .collect();
        let scores: Vec<f64> = (0..=cache_len)
            .map(|p| {
                let k = row(&k_cache, &k_cur, p, kv);
                qh.iter().zip(&k).map(|(a, b)| a * b).sum::<f64>() * scale
            })
            .collect();
        let max = scores.iter().fold(f64::NEG_INFINITY, |a, s| a.max(*s));
        let e: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
        let z = e.iter().sum::<f64>();
        let mut out = vec![0.0f64; d];
        for (p, w) in e.iter().enumerate() {
            for (o, v) in out.iter_mut().zip(row(&v_cache, &v_cur, p, kv)) {
                *o += w / z * v;
            }
        }
        want.extend(out);
    }
    (got, want)
}

/// Embedding lookup: row `token` of the table, for tokens across the whole vocabulary.
fn run_embed(rng: &mut rand::rngs::StdRng, n: usize) -> (Vec<f32>, Vec<f64>) {
    let vocab = 64;
    let mut model = model_shell(n / ATT_HEAD_DIM, n / ATT_HEAD_DIM, ATT_HEAD_DIM, 0, vocab);
    model.token_embedding = activations(rng, vocab * n, 1.0);
    let token = rng.random_range(0..vocab);
    let mut got = vec![0.0f32; n];
    model.embed_into(token as u32, &mut got);
    let want = model.token_embedding[token * n..(token + 1) * n]
        .iter()
        .map(|v| f64::from(*v))
        .collect();
    (got, want)
}

/// KV-cache append: each position's K and V land at `cache[layer][pos * kv_dim..]`, per layer.
fn run_kv_write(rng: &mut rand::rngs::StdRng, n: usize) -> (Vec<f32>, Vec<f64>) {
    let (layers, positions) = (2, 5);
    let mut cache = crate::gguf::OwnedQuantizedKVCache::new(layers, n, positions);
    let mut want: Vec<Vec<f64>> = vec![Vec::new(); 2 * layers];
    for _ in 0..positions {
        for l in 0..layers {
            let (k, v) = (activations(rng, n, 1.0), activations(rng, n, 1.0));
            cache.append(l, &k, &v);
            want[2 * l].extend(k.iter().map(|x| f64::from(*x)));
            want[2 * l + 1].extend(v.iter().map(|x| f64::from(*x)));
        }
        cache.advance();
    }
    let got = (0..layers)
        .flat_map(|l| [cache.get_k(l), cache.get_v(l)])
        .flat_map(|c| c.iter().copied())
        .collect();
    (got, want.concat())
}

/// Both layouts on each draw: NORM (type 0) rotates adjacent pairs `(2i, 2i+1)`, NEOX (type 2)
/// rotates `(i, i + d/2)`. The f64 oracle writes each from the definition, at a random position.
fn run_rope(rng: &mut rand::rngs::StdRng, n: usize) -> (Vec<f32>, Vec<f64>) {
    let (d, half) = (ROPE_HEAD_DIM, ROPE_HEAD_DIM / 2);
    let heads = n / (2 * d);
    assert!(
        heads > 0 && n == 2 * heads * d,
        "rope: n must be 2 * heads * {d}"
    );
    let pos = rng.random_range(0..ROPE_POS_MAX);
    let (mut got, mut want) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for rope_type in [0u32, 2] {
        let x = activations(rng, heads * d, 2.0);
        let mut y = x.clone();
        rope_model(rope_type).apply_rope(&mut y, pos, heads);
        got.extend_from_slice(&y);
        let mut r: Vec<f64> = x.iter().map(|v| f64::from(*v)).collect();
        for h in r.chunks_exact_mut(d) {
            for i in 0..half {
                let freq = f64::from(ROPE_THETA).powf(-2.0 * i as f64 / d as f64);
                let (sin, cos) = (pos as f64 * freq).sin_cos();
                let (a, b) = if rope_type == 2 {
                    (i, i + half)
                } else {
                    (2 * i, 2 * i + 1)
                };
                let (x0, x1) = (h[a], h[b]);
                h[a] = x0 * cos - x1 * sin;
                h[b] = x0 * sin + x1 * cos;
            }
        }
        want.extend(r);
    }
    (got, want)
}

/// SwiGLU's elementwise stage, `silu(g) * u`, against `g / (1 + e^-g) * u` in f64. Gates span
/// ±8 so both tails of the sigmoid are exercised.
fn run_swiglu(rng: &mut rand::rngs::StdRng, n: usize) -> (Vec<f32>, Vec<f64>) {
    let (g, u) = (activations(rng, n, 8.0), activations(rng, n, 4.0));
    let mut got = g.clone();
    crate::gguf::OwnedQuantizedModel::swiglu_gate_into(&mut got, &u);
    let want = g
        .iter()
        .zip(&u)
        .map(|(g, u)| {
            let g = f64::from(*g);
            g / (1.0 + (-g).exp()) * f64::from(*u)
        })
        .collect();
    (got, want)
}

/// Residual add against the correctly rounded f32 sum `fl32(h + d)`. f32 addition is defined
/// as exactly that, so the row is bitwise. The f64 sum it is rounded from is asserted exact
/// (Fast2Sum error 0), so no double rounding can hide in the oracle.
fn run_residual_add(rng: &mut rand::rngs::StdRng, n: usize) -> (Vec<f32>, Vec<f64>) {
    let (h, d) = (activations(rng, n, 4.0), activations(rng, n, 4.0));
    let mut got = h.clone();
    crate::gguf::OwnedQuantizedModel::residual_add_into(&mut got, &d);
    let want = h
        .iter()
        .zip(&d)
        .map(|(h, d)| {
            let (a, b) = (f64::from(*h), f64::from(*d));
            let (big, small) = if a.abs() >= b.abs() { (a, b) } else { (b, a) };
            let sum = big + small;
            assert_eq!(
                small - (sum - big),
                0.0,
                "f64 sum of {a} + {b} is not exact"
            );
            f64::from(sum as f32)
        })
        .collect();
    (got, want)
}

/// The repeat penalty's workload: a 64-token history of which the last 32 are the window.
const PEN_P: f32 = 1.3;
const PEN_LAST_N: usize = 32;

/// Repeat penalty against its definition: each DISTINCT in-vocabulary token of the last `last_n`
/// gets `x * p` if `x <= 0`, else `x / p`, once. f64 holds an f32 product exactly, and its 53
/// bits make the f32 rounding of an f64 quotient the correctly rounded f32 quotient, so the row is
/// bitwise. Every trial plants a token repeated inside the window (compounding would hit it
/// twice), one only before the window, and one past the vocabulary (it must be skipped).
fn run_repeat_penalty(rng: &mut rand::rngs::StdRng, n: usize) -> (Vec<f32>, Vec<f64>) {
    let x = activations(rng, n, 8.0);
    let mut recent: Vec<u32> = (0..2 * PEN_LAST_N)
        .map(|_| rng.random_range(0..n as u32))
        .collect();
    let outside = rng.random_range(0..n as u32);
    let twice = (outside + rng.random_range(1..n as u32)) % n as u32;
    recent.retain(|&t| t != outside);
    recent.insert(0, outside);
    let w = recent.len() - PEN_LAST_N;
    recent[w + 1] = twice;
    recent[w + 3] = twice;
    recent[w + 5] = n as u32 + 7;
    let mut got = x.clone();
    crate::gguf::OwnedQuantizedModel::apply_repeat_penalty(&mut got, &recent, PEN_P, PEN_LAST_N);
    let p = f64::from(PEN_P);
    let mut want: Vec<f64> = x.iter().map(|v| f64::from(*v)).collect();
    let mut done = vec![false; n];
    for &t in &recent[w..] {
        let i = t as usize;
        if i < n && !done[i] {
            done[i] = true;
            let v = want[i];
            want[i] = f64::from((if v <= 0.0 { v * p } else { v / p }) as f32);
        }
    }
    (got, want)
}

/// The sampler's workload: a 256-token vocabulary, T = 0.7, top-k 40.
const SAMPLE_T: f32 = 0.7;
const SAMPLE_K: usize = 40;
/// The smallest gap, in the f64 definition, the inputs keep between neighbouring scaled logits of
/// the top k + 1 and between a nucleus cumulative and `top_p`. Nearer than that, the f32 kernel may
/// order or cut the set differently for a reason that is rounding, not a defect; such a draw is
/// redrawn, so every set membership the receipt measures is decided by the definition alone.
const SAMPLE_GAP: f64 = 1e-4;

/// The f64 definition of the draw: the kept tokens in order and their CDF, or `None` when the
/// inputs sit within [`SAMPLE_GAP`] of a set boundary.
fn sample_definition(x: &[f32], top_p: f32) -> Option<(Vec<usize>, Vec<f64>)> {
    let t = f64::from(SAMPLE_T);
    let mut order: Vec<usize> = (0..x.len()).collect();
    order.sort_by(|&a, &b| f64::from(x[b]).total_cmp(&f64::from(x[a])));
    let s = |i: usize| f64::from(x[order[i]]) / t;
    let near = (1..=SAMPLE_K.min(x.len() - 1)).any(|i| s(i - 1) - s(i) < SAMPLE_GAP);
    if near {
        return None;
    }
    order.truncate(SAMPLE_K);
    let softmax = |kept: &[usize]| -> Vec<f64> {
        let e: Vec<f64> = kept
            .iter()
            .map(|&i| (f64::from(x[i]) / t - f64::from(x[kept[0]]) / t).exp())
            .collect();
        let z: f64 = e.iter().sum();
        e.iter().map(|v| v / z).collect()
    };
    if top_p > 0.0 && top_p < 1.0 {
        let (p, mut c) = (f64::from(top_p), 0.0);
        let mut cut = order.len();
        for (i, q) in softmax(&order).iter().enumerate() {
            c += q;
            if (c - p).abs() < SAMPLE_GAP {
                return None;
            }
            if c >= p {
                cut = i + 1;
                break;
            }
        }
        order.truncate(cut);
    }
    let cdf = softmax(&order)
        .iter()
        .scan(0.0, |c, q| {
            *c += q;
            Some(*c)
        })
        .collect();
    Some((order, cdf))
}

/// The draw's inverse-CDF breakpoints against the definition's CDF. The draw returns the token of
/// rank `j` for `r` up to the j-th cumulative probability and a later rank above it, so the
/// smallest f32 `r` that yields a rank above `j` — found by bisecting `r`'s bits, which order like
/// the values on `[0, 1)` — is the kernel's own `C_j`. A token outside the definition's set is
/// NaN; a set cut short reads 1.0 where the definition is below it.
fn run_sample(rng: &mut rand::rngs::StdRng, n: usize, top_p: f32) -> (Vec<f32>, Vec<f64>) {
    let (x, (kept, cdf)) = loop {
        let x = activations(rng, n, 8.0);
        if let Some(d) = sample_definition(&x, top_p) {
            break (x, d);
        }
    };
    let rank = |r: f32| {
        let tok = crate::sampling::draw(&x, SAMPLE_T, SAMPLE_K, top_p, r) as usize;
        kept.iter().position(|&k| k == tok)
    };
    let mut got = Vec::with_capacity(kept.len() - 1);
    for j in 0..kept.len() - 1 {
        let (mut lo, mut hi) = (0u32, 1.0f32.to_bits());
        let mut foreign = false;
        while hi - lo > 1 {
            let mid = lo + (hi - lo) / 2;
            match rank(f32::from_bits(mid)) {
                Some(k) if k > j => hi = mid,
                Some(_) => lo = mid,
                None => {
                    foreign = true;
                    break;
                },
            }
        }
        got.push(if foreign {
            f32::NAN
        } else {
            f32::from_bits(hi)
        });
    }
    (got, cdf[..kept.len() - 1].to_vec())
}

/// `sample_topk`'s draw: top-k only (it always passes `top_p = 1.0`).
fn run_sample_topk(rng: &mut rand::rngs::StdRng, n: usize) -> (Vec<f32>, Vec<f64>) {
    run_sample(rng, n, 1.0)
}

/// `sample_topk_seeded`'s draw: its callers pass the request's `top_p` through, so the nucleus
/// cut is measured here, at 0.9.
fn run_sample_nucleus(rng: &mut rand::rngs::StdRng, n: usize) -> (Vec<f32>, Vec<f64>) {
    run_sample(rng, n, 0.9)
}

pub(super) const OPS: &[Op] = &[
    Op {
        id: "cpu.rmsnorm.f32",
        source_fn: "rms_norm_into",
        formula: "rmsnorm:x*w/sqrt(mean(x^2)+eps)",
        workload: OP_WORKLOAD,
        run: run_rmsnorm,
    },
    Op {
        id: "cpu.layernorm.f32",
        source_fn: "layer_norm_into",
        formula: "layernorm:(x-mean)*w/sqrt(var+eps)+b",
        workload: OP_WORKLOAD,
        run: run_layernorm,
    },
    Op {
        id: "cpu.gelu.f32",
        source_fn: "gelu",
        formula: "gelu_tanh:0.5x(1+tanh(sqrt(2/pi)(x+0.044715x^3)))",
        workload: OP_WORKLOAD,
        run: run_gelu,
    },
    Op {
        id: "cpu.argmax.f32",
        source_fn: "argmax",
        formula: "argmax:[max(x),min{i:x_i=max(x)}]",
        workload: OP_WORKLOAD,
        run: run_argmax,
    },
    Op {
        id: "cpu.rope.f32",
        source_fn: "apply_rope",
        formula: "rope:norm+neox,d=128,theta=1e6,pos<32768,angle=pos*theta^(-2i/d)",
        workload: OpWorkload {
            n: 2 * 2 * ROPE_HEAD_DIM,
            ..OP_WORKLOAD
        },
        run: run_rope,
    },
    Op {
        id: "cpu.attention.f32",
        source_fn: "attention_with_cache_gqa_into",
        formula: "attention_gqa:softmax(q.k/sqrt(d))v,kv=h/(H/KV),H=14,KV=2,d=64,cache<512",
        workload: OpWorkload {
            n: ATT_HEADS * ATT_HEAD_DIM,
            ..OP_WORKLOAD
        },
        run: run_attention,
    },
    Op {
        id: "cpu.embed.f32",
        source_fn: "embed_into",
        formula: "embed:E[token*n..(token+1)*n]",
        workload: OP_WORKLOAD,
        run: run_embed,
    },
    Op {
        id: "cpu.kv_write.f32",
        source_fn: "append",
        formula: "kv_write:cache[l][p*kv..(p+1)*kv]=k_p|v_p",
        workload: OpWorkload {
            n: ATT_KV_HEADS * ATT_HEAD_DIM,
            ..OP_WORKLOAD
        },
        run: run_kv_write,
    },
    Op {
        id: "cpu.swiglu.f32",
        source_fn: "swiglu_gate_into",
        formula: "swiglu:g/(1+exp(-g))*u,|g|<8",
        workload: OP_WORKLOAD,
        run: run_swiglu,
    },
    Op {
        id: "cpu.residual_add.f32",
        source_fn: "residual_add_into",
        formula: "residual_add:fl32(h+d),h+d exact in f64",
        workload: OP_WORKLOAD,
        run: run_residual_add,
    },
    Op {
        id: "cpu.repeat_penalty.f32",
        source_fn: "apply_repeat_penalty",
        formula: "repeat_penalty:once per distinct t<n in last_n,x<=0?fl32(x*p):fl32(x/p),p=1.3,last_n=32",
        workload: OP_WORKLOAD,
        run: run_repeat_penalty,
    },
    Op {
        id: "cpu.sample.topk.f32",
        source_fn: "draw",
        formula: "sample:inverse_cdf(softmax(x/T) over top-k),breakpoints,T=0.7,k=40,top_p=1,gap>=1e-4",
        workload: OpWorkload {
            n: 256,
            ..OP_WORKLOAD
        },
        run: run_sample_topk,
    },
    Op {
        id: "cpu.sample.seeded.f32",
        source_fn: "draw",
        formula: "sample:inverse_cdf(softmax(x/T) over nucleus(top_p) of top-k),breakpoints,T=0.7,k=40,top_p=0.9,gap>=1e-4",
        workload: OpWorkload {
            n: 256,
            ..OP_WORKLOAD
        },
        run: run_sample_nucleus,
    },
];

/// The error model's bound on `max_rel_err`, for the error models this harness implements.
///
/// EM-ROPE: the angle `pos * theta^(-2i/d)` is formed in f32 — `powf` within 2 ulp, the product
/// 1 more — so it is off by at most `3u * pos` radians (`u` the f32 unit roundoff, `freq ≤ 1`).
/// Rotating a pair by an angle off by δ moves each output by at most `|pair| * δ ≤ √2 * scale * δ`,
/// and sin/cos plus the rotation's own multiply-add add a few `u`. Nothing about the kernel is
/// assumed but that it forms the angle in f32; a kernel that got the angle wrong misses by O(1).
fn modelled_bound(row: &OpRow) -> Option<(f64, &'static str)> {
    let u = f64::from(f32::EPSILON) / 2.0;
    match row.error_model.as_str() {
        "EM-ROPE" => Some((
            std::f64::consts::SQRT_2 * 3.0 * u * ROPE_POS_MAX as f64 + 4.0 * u,
            "sqrt2*3u*pos_max+4u: f32 angle pos*theta^(-2i/d), powf 2ulp + mul 1ulp",
        )),
        _ => None,
    }
}

pub(super) fn op(id: &str) -> &'static Op {
    OPS.iter()
        .find(|o| o.id == id)
        .unwrap_or_else(|| panic!("{id}: no op harness entry"))
}

/// The oracle part of an op receipt's input set: the oracle and the definition it evaluates.
pub(super) fn op_oracle(o: &Op) -> String {
    format!("{OP_ORACLE}:{}", o.formula)
}

#[derive(Debug, Default)]
struct OpMeasured {
    max_abs_err: f64,
    max_rel_err: f64,
}

fn measure_op(o: &Op, w: OpWorkload) -> OpMeasured {
    let mut rng = rand::rngs::StdRng::seed_from_u64(w.seed);
    let mut m = OpMeasured::default();
    for _ in 0..w.trials {
        let (got, want) = (o.run)(&mut rng, w.n);
        assert_eq!(got.len(), want.len(), "{}: output length", o.id);
        assert!(
            !got.is_empty(),
            "{}: an empty output measures nothing",
            o.id
        );
        let scale = want.iter().fold(0.0f64, |a, r| a.max(r.abs()));
        assert!(
            scale > 0.0,
            "{}: an all-zero reference measures nothing",
            o.id
        );
        for (g, r) in got.iter().zip(&want) {
            let abs = (f64::from(*g) - r).abs();
            assert!(abs.is_finite(), "{}: non-finite output", o.id);
            m.max_abs_err = m.max_abs_err.max(abs);
            m.max_rel_err = m.max_rel_err.max(abs / scale);
        }
    }
    m
}

fn op_row<'a>(r: &'a Registry, id: &str) -> &'a OpRow {
    r.ops()
        .iter()
        .find(|row| row.kernel_id == id)
        .unwrap_or_else(|| panic!("{id} is not an ops[] row"))
}

// serde_json::json!() macro uses infallible unwrap internally
#[allow(clippy::disallowed_methods)]
fn op_receipt(o: &Op, row: &OpRow) -> serde_json::Value {
    let served = measure_op(o, o.workload);
    let set = InputSet::from_op_tree(&repo_root(), row, "none", &device(), &op_oracle(o))
        .unwrap_or_else(|e| panic!("{}: {e}", o.id));
    let mut doc = receipt_header_for(o.id, o.source_fn, &row.precision, &set);
    doc["oracle"] = OP_ORACLE.into();
    doc["oracle_formula"] = o.formula.into();
    doc["oracle_independent"] = true.into();
    doc["workload"] = serde_json::json!({
        "n": o.workload.n,
        "seed": o.workload.seed,
        "trials": o.workload.trials,
    });
    doc["served"] =
        serde_json::json!({"max_abs_err": served.max_abs_err, "max_rel_err": served.max_rel_err});
    doc["tolerance_rel"] = tolerance_from(served.max_rel_err).into();
    doc["margin"] = match modelled_bound(row) {
        Some((bound, why)) => serde_json::json!({
            "error_model": row.error_model,
            "verdict": if served.max_rel_err <= bound { "within" } else { "exceeds" },
            "bound_rel": bound,
            "why": why,
        }),
        None => serde_json::json!({
            "error_model": row.error_model,
            "verdict": "not_modelled",
            "why": "aprender-kernel-oracle has no bound for this error model yet; tolerance_rel decides",
        }),
    };
    doc
}

#[test]
#[ignore = "writes receipts; run on the host whose arch it admits (see module doc)"]
fn emit_op_parity_receipts() {
    let out = repo_root().join(std::env::var("KREG_RECEIPT_OUT").expect("KREG_RECEIPT_OUT"));
    std::fs::create_dir_all(&out).expect("receipt dir");
    let r = registry().expect("registry");
    let only = std::env::var("KREG_EMIT").ok();
    for o in OPS.iter().filter(|o| {
        only.as_deref()
            .is_none_or(|w| w.split(',').any(|w| w == o.id))
    }) {
        let row = op_row(&r, o.id);
        assert_eq!(
            row.source_fn, o.source_fn,
            "{}: harness and row name different fns",
            o.id
        );
        let doc = op_receipt(o, row);
        let path = out.join(format!("{}.json", o.id));
        let text = serde_json::to_string_pretty(&doc).expect("receipt json");
        std::fs::write(&path, text + "\n").expect("write receipt");
        eprintln!("{}: {}", path.display(), doc["served"]);
    }
}

/// Every `op_receipts` entry of the ratchet file with the receipt it points at.
pub(super) fn committed_ops() -> Vec<(serde_json::Value, serde_json::Value)> {
    let doc: serde_json::Value =
        serde_json::from_str(include_str!("../kernel-registry-receipts.json"))
            .expect("receipts json");
    doc["op_receipts"]
        .as_array()
        .expect("op_receipts")
        .iter()
        .map(|entry| {
            let path = entry["receipt"].as_str().expect("receipt path");
            let text = std::fs::read_to_string(repo_root().join(path))
                .unwrap_or_else(|e| panic!("{path}: {e}"));
            let receipt = serde_json::from_str(&text).unwrap_or_else(|e| panic!("{path}: {e}"));
            (entry.clone(), receipt)
        })
        .collect()
}

/// FALSIFY-KREG-009 for `ops[]`: a committed op receipt names its row and oracle, and still holds
/// on this host — its own seed and workload, re-measured, stay within `tolerance_rel` — and a
/// `precision=f32` row is f32-exact against the f64 definition.
#[test]
fn committed_op_parity_receipts_hold_on_this_host() {
    let r = registry().expect("registry");
    let all = committed_ops();
    assert!(
        !all.is_empty(),
        "no op receipts: this test would pass measuring nothing"
    );
    for (entry, rc) in all {
        let id = rc["kernel_id"].as_str().expect("kernel_id");
        let path = entry["receipt"].as_str().expect("receipt path");
        assert_eq!(rc["schema"], SCHEMA, "{path}: schema");
        assert_eq!(
            entry["kernel_id"], id,
            "{path}: ratchet entry names another op"
        );
        assert_eq!(
            entry["host_arch"], rc["host_arch"],
            "{path}: ratchet entry arch"
        );
        let row = op_row(&r, id);
        let o = op(id);
        assert_eq!(
            row.tolerance, path,
            "{id}: tolerance must point at its receipt"
        );
        assert_eq!(
            rc["registry_precision"],
            row.precision.as_str(),
            "{id}: precision drifted"
        );
        assert_eq!(rc["oracle"], OP_ORACLE, "{path}: oracle");
        assert_eq!(
            rc["oracle_formula"], o.formula,
            "{path}: the oracle's definition changed"
        );
        assert_eq!(rc["oracle_independent"], true, "{path}: oracle_independent");
        let wl = &rc["workload"];
        let w = OpWorkload {
            n: wl["n"].as_u64().expect("n") as usize,
            seed: wl["seed"].as_u64().expect("seed"),
            trials: wl["trials"].as_u64().expect("trials") as usize,
        };
        let bound = rc["tolerance_rel"].as_f64().expect("tolerance_rel");
        let now = measure_op(o, w);
        assert!(
            now.max_rel_err <= bound,
            "{id}: max_rel_err {} > receipt tolerance {bound} ({path})",
            now.max_rel_err
        );
        if row.determinism == "bitwise" {
            assert!(
                now.max_abs_err == 0.0,
                "{id}: determinism=bitwise, yet it differs from its definition by {}",
                now.max_abs_err
            );
        }
        match (row.precision.as_str(), modelled_bound(row)) {
            // An error model that bounds the f32 kernel's own drift (RoPE's f32 angle grows with
            // the position) replaces the f32-exact ceiling; the receipt must agree it held.
            ("f32", Some((bound, _))) => {
                assert!(
                    now.max_rel_err <= bound,
                    "{id}: max_rel_err {} > the {} bound {bound}",
                    now.max_rel_err,
                    row.error_model
                );
                assert_eq!(rc["margin"]["verdict"], "within", "{path}: margin verdict");
                assert_eq!(
                    rc["margin"]["bound_rel"], bound,
                    "{path}: the modelled bound changed"
                );
            },
            ("f32", None) => assert!(
                now.max_rel_err < F32_CEILING,
                "{id}: precision=f32, yet the error against f64 is {} ≥ {F32_CEILING}",
                now.max_rel_err
            ),
            (p, _) => panic!("{id}: no precision check for {p}"),
        }
    }
}

#[test]
fn every_op_harness_entry_names_its_row() {
    let r = registry().expect("registry");
    for o in OPS {
        assert_eq!(op_row(&r, o.id).source_fn, o.source_fn, "{}", o.id);
    }
}
