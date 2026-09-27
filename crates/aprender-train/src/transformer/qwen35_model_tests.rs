//! QTG-001 at model scale: the training-side Qwen3.5 forward equals serve's
//! (`realizar::gguf::forward_qwen35::Qwen35Model::forward_single_qwen35`, token by
//! token) on the same GGUF, logit for logit. The two sides read the file
//! independently (see the module doc of [`super`]).

use std::path::Path;

use aprender::format::gguf::{export_tensors_to_gguf, GgmlType, GgufTensor, GgufValue};
use realizar::gguf::forward_qwen35::Qwen35Model as ServeModel;
use realizar::gguf::MappedGGUFModel;

use super::Qwen35Model;

struct Lcg(u64);

impl Lcg {
    fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|_| {
                self.0 = self
                    .0
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                (((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0) * scale
            })
            .collect()
    }
}

/// Serve's logits for `tokens`, one position at a time from a fresh state, under
/// `with_fp32_activations`: serve's exact-activation reference (#3714). By default its
/// Q4_K matvecs quantise the activation to Q8_K first, which moves the real 0.8B's
/// logits to cos 0.998 of the f32 forward; this is the arithmetic, not the model.
fn serve_logits(path: &Path, tokens: &[u32]) -> Vec<Vec<f32>> {
    realizar::quantize::with_fp32_activations(|| {
        let mapped = MappedGGUFModel::from_path(path).expect("serve maps the file");
        let base = ServeModel::create_base_model(&mapped.model, mapped.data()).expect("serve base");
        let qwen = ServeModel::from_model_and_layers(&base, &mapped.model, mapped.data())
            .expect("serve layers");
        let mut state = qwen.new_state(tokens.len() + 1);
        tokens
            .iter()
            .enumerate()
            .map(|(pos, &t)| qwen.forward_single_qwen35(t, &mut state, pos).expect("serve forward"))
            .collect()
    })
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().fold(0, |best, (i, x)| if *x > v[best] { i } else { best })
}

/// Per position: (max |ours − serve| / max |serve|, cosine, argmax equal).
fn compare(ours: &[f32], theirs: &[Vec<f32>], vocab: usize) -> Vec<(f32, f32, bool)> {
    ours.chunks_exact(vocab)
        .zip(theirs)
        .map(|(o, s)| {
            let err = o.iter().zip(s).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
            let scale = s.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
            (err / scale.max(1e-6), cosine(o, s), argmax(o) == argmax(s))
        })
        .collect()
}

// ---- A small random Qwen3.5 written as an f32 GGUF, so CI runs the whole model. ----

const HIDDEN: usize = 16;
const VOCAB: usize = 40;
const LAYERS: usize = 5; // 0,1,2 GDN · 3 attention · 4 GDN (full_attention_interval 4)
const INTER: usize = 24;
const HEADS: usize = 4;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 8; // q_dim 32 ≠ hidden: the head width is not hidden/heads
const K_HEADS: usize = 2;
const V_HEADS: usize = 4;
const STATE: usize = 4; // head_k_dim = head_v_dim, as serve assumes
const KERNEL: usize = 4;

fn tensor(name: &str, data: &[f32], shape: &[usize]) -> GgufTensor {
    assert_eq!(data.len(), shape.iter().product::<usize>(), "{name}");
    GgufTensor {
        name: name.to_string(),
        shape: shape.iter().map(|&d| d as u64).collect(),
        dtype: GgmlType::F32,
        data: data.iter().flat_map(|v| v.to_le_bytes()).collect(),
    }
}

/// `[out × in]` weight in GGUF's `[ne0 = in, ne1 = out]` order.
fn weight(r: &mut Lcg, name: &str, d_in: usize, d_out: usize) -> GgufTensor {
    tensor(name, &r.vec(d_in * d_out, 0.5 / (d_in as f32).sqrt() * 2.0), &[d_in, d_out])
}

fn norm(r: &mut Lcg, name: &str, n: usize) -> GgufTensor {
    let w: Vec<f32> = r.vec(n, 0.2).iter().map(|x| 1.0 + x).collect();
    tensor(name, &w, &[n])
}

fn write_tiny_qwen35(path: &Path, seed: u64) {
    let mut r = Lcg(seed);
    let (kd, vd) = (K_HEADS * STATE, V_HEADS * STATE);
    let conv_dim = 2 * kd + vd;
    let mut t = vec![
        tensor("token_embd.weight", &r.vec(VOCAB * HIDDEN, 1.0), &[HIDDEN, VOCAB]),
        norm(&mut r, "output_norm.weight", HIDDEN),
    ];
    for i in 0..LAYERS {
        let b = |s: &str| format!("blk.{i}.{s}");
        t.push(norm(&mut r, &b("attn_norm.weight"), HIDDEN));
        t.push(norm(&mut r, &b("post_attention_norm.weight"), HIDDEN));
        t.push(weight(&mut r, &b("ffn_gate.weight"), HIDDEN, INTER));
        t.push(weight(&mut r, &b("ffn_up.weight"), HIDDEN, INTER));
        t.push(weight(&mut r, &b("ffn_down.weight"), INTER, HIDDEN));
        if (i + 1) % 4 == 0 {
            t.push(weight(&mut r, &b("attn_q.weight"), HIDDEN, 2 * HEADS * HEAD_DIM));
            t.push(weight(&mut r, &b("attn_k.weight"), HIDDEN, KV_HEADS * HEAD_DIM));
            t.push(weight(&mut r, &b("attn_v.weight"), HIDDEN, KV_HEADS * HEAD_DIM));
            t.push(norm(&mut r, &b("attn_q_norm.weight"), HEAD_DIM));
            t.push(norm(&mut r, &b("attn_k_norm.weight"), HEAD_DIM));
            t.push(weight(&mut r, &b("attn_output.weight"), HEADS * HEAD_DIM, HIDDEN));
        } else {
            t.push(weight(&mut r, &b("attn_qkv.weight"), HIDDEN, conv_dim));
            t.push(weight(&mut r, &b("attn_gate.weight"), HIDDEN, vd));
            t.push(weight(&mut r, &b("ssm_alpha.weight"), HIDDEN, V_HEADS));
            t.push(weight(&mut r, &b("ssm_beta.weight"), HIDDEN, V_HEADS));
            let a: Vec<f32> = r.vec(V_HEADS, 1.0).iter().map(|x| -(x.exp())).collect();
            t.push(tensor(&b("ssm_a"), &a, &[V_HEADS]));
            t.push(tensor(&b("ssm_dt.bias"), &r.vec(V_HEADS, 0.5), &[V_HEADS]));
            t.push(tensor(
                &b("ssm_conv1d.weight"),
                &r.vec(conv_dim * KERNEL, 0.5),
                &[KERNEL, conv_dim],
            ));
            t.push(norm(&mut r, &b("ssm_norm.weight"), STATE));
            t.push(weight(&mut r, &b("ssm_out.weight"), vd, HIDDEN));
        }
    }
    let u = |k: &str, v: usize| (format!("qwen35.{k}"), GgufValue::Uint32(v as u32));
    let metadata = vec![
        ("general.architecture".to_string(), GgufValue::String("qwen35".to_string())),
        u("embedding_length", HIDDEN),
        u("block_count", LAYERS),
        u("feed_forward_length", INTER),
        u("context_length", 64),
        u("attention.head_count", HEADS),
        u("attention.head_count_kv", KV_HEADS),
        u("attention.key_length", HEAD_DIM),
        u("attention.value_length", HEAD_DIM),
        u("ssm.state_size", STATE),
        u("ssm.group_count", K_HEADS),
        u("ssm.time_step_rank", V_HEADS),
        u("ssm.conv_kernel", KERNEL),
        u("ssm.inner_size", vd),
        u("full_attention_interval", 4),
        ("qwen35.rope.freq_base".to_string(), GgufValue::Float32(10_000.0)),
        ("qwen35.attention.layer_norm_rms_epsilon".to_string(), GgufValue::Float32(1e-6)),
        // n_rot = 2 · Σ = 4 of each 8-wide head: a partial rope, as on the real model.
        ("qwen35.rope.dimension_sections".to_string(), GgufValue::ArrayUint32(vec![1, 1, 0, 0])),
    ];
    let mut f = std::fs::File::create(path).expect("create gguf");
    export_tensors_to_gguf(&mut f, &t, &metadata).expect("write gguf");
}

/// FALSIFY-QTG-001 (model): on a random 5-layer Qwen3.5 (GDN, attention and a GDN
/// after it; tied `lm_head`), our logits equal serve's at every position.
#[test]
fn falsify_qtg_001_tiny_model_logits_equal_serve() {
    let dir = tempfile::tempdir().expect("tempdir");
    for seed in [1_u64, 2, 3] {
        let path = dir.path().join(format!("tiny-{seed}.gguf"));
        write_tiny_qwen35(&path, seed);
        let ours = Qwen35Model::from_gguf(&path).expect("train loads");
        assert_eq!(ours.attention_schedule(), [false, false, false, true, false]);
        let tokens = [3_u32, 17, 0, 39, 22, 5, 11];
        let logits = ours.forward(&tokens);
        let theirs = serve_logits(&path, &tokens);
        for (pos, (err, cos, same)) in compare(&logits, &theirs, VOCAB).into_iter().enumerate() {
            assert!(err <= 1e-4 && same, "seed {seed} pos {pos}: rel err {err}, cos {cos}");
        }
    }
}

/// FALSIFY-QTG-001 (model, real weights): Qwen3.5 GGUF named by `QWEN35_GGUF`
/// (e.g. Qwen3.5-0.8B-Q4_K_M) — per position cos ≥ 0.99999, rel err ≤ 5e-3, argmax equal (measured 1.000000 and ≤ 1.8e-3 on the 0.8B).
/// Ignored in CI: it needs the file and ~5 GB of RAM.
#[test]
#[ignore = "needs a Qwen3.5 GGUF: QWEN35_GGUF=/path/to/Qwen3.5-0.8B-Q4_K_M.gguf"]
fn falsify_qtg_001_real_model_logits_equal_serve() {
    let path = std::env::var("QWEN35_GGUF").expect("set QWEN35_GGUF");
    let ours = Qwen35Model::from_gguf(&path).expect("train loads");
    let schedule = ours.attention_schedule();
    assert!(schedule.contains(&true) && schedule.contains(&false), "{schedule:?}");
    let tokens = [9_707_u32, 11, 1_879, 374, 264, 1_273];
    let logits = ours.forward(&tokens);
    let theirs = serve_logits(Path::new(&path), &tokens);
    let stats = compare(&logits, &theirs, ours.vocab_size());
    for (pos, (err, cos, same)) in stats.iter().enumerate() {
        eprintln!("pos {pos}: rel err {err:.2e}, cos {cos:.6}, argmax equal {same}");
    }
    for (pos, (err, cos, same)) in stats.iter().enumerate() {
        assert!(*cos >= 0.99999 && *err <= 5e-3 && *same, "pos {pos}: rel err {err}, cos {cos}");
    }
}

/// The two readers of the file agree: every tensor aprender's `GgufReader` dequantises
/// equals realizar's own dequantisation of it. With this, any logit gap left between
/// train and serve on real weights is serve's matmul numerics, not a misread weight.
#[test]
#[ignore = "needs a Qwen3.5 GGUF: QWEN35_GGUF=/path/to/Qwen3.5-0.8B-Q4_K_M.gguf"]
fn real_model_dequant_equals_serve_dequant() {
    let path = std::env::var("QWEN35_GGUF").expect("set QWEN35_GGUF");
    let reader = aprender::format::gguf::GgufReader::from_file_full(&path).expect("aprender reads");
    let mapped = MappedGGUFModel::from_path(&path).expect("serve maps the file");
    let mut worst = (0.0_f32, String::new());
    for info in &mapped.model.tensors {
        let (ours, _) = reader.get_tensor_f32(&info.name).expect("aprender dequant");
        let theirs = mapped.model.get_tensor_f32(&info.name, mapped.data()).expect("serve dequant");
        assert_eq!(ours.len(), theirs.len(), "{}", info.name);
        let err = ours.iter().zip(&theirs).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
        if err > worst.0 {
            worst = (err, format!("{} (qtype {})", info.name, info.qtype));
        }
    }
    eprintln!(
        "{} tensors, worst |aprender - realizar| = {:.3e} at {}",
        mapped.model.tensors.len(),
        worst.0,
        worst.1
    );
    assert!(worst.0 <= 1e-6, "dequant disagrees: {} at {}", worst.0, worst.1);
}

/// R3 on the real 0.8B: the backward runs at full shape (24 layers, the real vocab),
/// its loss is the cross-entropy of `forward`'s own logits, and every gradient is
/// finite with a non-zero embedding/head gradient.
#[test]
#[ignore = "needs a Qwen3.5 GGUF: QWEN35_GGUF=/path/to/Qwen3.5-0.8B-Q4_K_M.gguf"]
fn real_model_loss_and_grads_are_consistent_and_finite() {
    let path = std::env::var("QWEN35_GGUF").expect("set QWEN35_GGUF");
    let model = Qwen35Model::from_gguf(&path).expect("train loads");
    let (tokens, targets) = ([9_707_u32, 11, 1_879, 374, 264], [11_u32, 1_879, 374, 264, 1_273]);
    let (loss, g) = model.loss_and_grads(&tokens, &targets);
    let vocab = model.vocab_size();
    let want: f64 = model
        .forward(&tokens)
        .chunks_exact(vocab)
        .zip(&targets)
        .map(|(row, &t)| {
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let lse =
                f64::from(max) + row.iter().map(|&x| f64::from(x - max).exp()).sum::<f64>().ln();
            lse - f64::from(row[t as usize])
        })
        .sum::<f64>()
        / tokens.len() as f64;
    eprintln!("loss {loss} (independent {want:.6}), ln vocab {:.3}", (vocab as f64).ln());
    assert!((f64::from(loss) - want).abs() <= 1e-4 * want, "loss {loss} vs {want}");
    assert!(want < (vocab as f64).ln(), "a trained model beats uniform on real text");
    assert_eq!(g.layers.len(), model.num_layers());
    let finite = |v: &[f32]| v.iter().all(|x| x.is_finite());
    assert!(finite(&g.embed) && finite(&g.final_norm), "embed/final_norm grads finite");
    assert!(g.embed.iter().any(|&x| x != 0.0), "embedding gradient is zero");
    for (i, l) in g.layers.iter().enumerate() {
        let all = [&l.attn_norm, &l.post_norm, &l.ffn_gate, &l.ffn_up, &l.ffn_down];
        assert!(all.iter().all(|v| finite(v)), "layer {i} grads finite");
    }
}
