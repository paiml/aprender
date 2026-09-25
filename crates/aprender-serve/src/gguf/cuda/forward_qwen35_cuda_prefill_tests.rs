//! PMAT-3596: the batched prefill against the per-token path it replaces, on real
//! Qwen3.5 files.
//!
//! The per-token path is `T` calls of [`Qwen35CudaModel::forward_single`]; the batched
//! one is [`Qwen35CudaModel::prefill`]. They run the same layer programs, so they
//! differ only by the order in which floats are summed: a GEMM tiles its dot products
//! differently from a GEMV, and cuBLAS attention reduces differently from
//! `DecodeAttention256Kernel`. What is asserted, at 64 and at a two-chunk length:
//!
//! 1. the last position's **argmax is identical** and its logits are within a cosine
//!    floor and a relative L∞ budget;
//! 2. every piece of state the prefill leaves behind — each layer's conv window,
//!    recurrent state and KV rows — is within a relative L∞ budget of the per-token
//!    state, so a decode that continues from it continues from the same place;
//! 3. one decode step from each state agrees (argmax + cosine);
//! 4. a prefill split across two calls (`pos0 > 0`, which exercises the attention
//!    base offset and the KV rows written by the first call) agrees with one call.
//!
//! Every comparison prints its reading; the constants are measured budgets. Parity of
//! apr against llama.cpp is NOT claimed here — #3693 measured apr's CPU and CUDA
//! forwards both diverging from llama.cpp at copy positions — it is the separate 20k
//! evidence run with a llama.cpp column.

use super::super::{Qwen35CudaModel, Qwen35CudaState};
use crate::gguf::forward_qwen35::Qwen35Model;

const MODEL_0_8B: &str = "/home/noah/models/Qwen3.5-0.8B-Q4_K_M.gguf";

/// Budgets for one attention path.
#[derive(Clone, Copy)]
struct Budget {
    /// Cosine floor on the logits.
    cosine: f64,
    /// Relative L∞ on the logits.
    logits: f32,
    /// Relative L∞ on any state buffer (conv window, recurrent state, KV rows).
    state: f32,
}

/// The f32 cuBLAS attention path. MEASURED on the 0.8B (sm_89 and sm_121,
/// 2026-09-21): cosine 1.0000000, logits 4.6e-7 .. 1.6e-6, states <= 3.8e-6.
const F32_BUDGET: Budget = Budget {
    cosine: 0.99999,
    logits: 1e-4,
    state: 1e-4,
};

/// The flash path (f16 inputs, f32 accumulation — the #3596 ruling). MEASURED on
/// the 0.8B (sm_89, 2026-09-21): identical argmax and cosine 1.0000000 everywhere,
/// logits 8.2e-5 .. 1.5e-4, states <= 5.4e-4 (layer-23 K rows at n=64) — ~100x the
/// f32 path, the price of f16 inputs; the budgets are ~10x the reading.
const FLASH_BUDGET: Budget = Budget {
    cosine: 0.99999,
    logits: 2e-3,
    state: 5e-3,
};

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut ab, mut aa, mut bb) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        ab += f64::from(*x) * f64::from(*y);
        aa += f64::from(*x) * f64::from(*x);
        bb += f64::from(*y) * f64::from(*y);
    }
    ab / (aa.sqrt() * bb.sqrt()).max(1e-30)
}

/// `max|got - want| / max|want|`, refusing a reference that is all ~zero.
fn rel_linf(got: &[f32], want: &[f32], what: &str) -> f32 {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    assert!(scale > 1e-6, "{what}: the per-token reference is all ~zero");
    got.iter()
        .zip(want)
        .fold(0.0f32, |m, (g, w)| m.max((g - w).abs()))
        / scale
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
            if x > bv {
                (i, x)
            } else {
                (bi, bv)
            }
        })
        .0
}

fn assert_logits_agree(batched: &[f32], per_token: &[f32], what: &str, b: Budget) {
    let cos = cosine(batched, per_token);
    let linf = rel_linf(batched, per_token, what);
    let (ab, ap) = (argmax(batched), argmax(per_token));
    println!(
        "[3596] {what}: argmax batched {ab} / per-token {ap}, cosine {cos:.7}, rel L∞ {linf:.3e}"
    );
    assert_eq!(ab, ap, "{what}: argmax differs");
    assert!(cos >= b.cosine, "{what}: cosine {cos} < {}", b.cosine);
    assert!(linf <= b.logits, "{what}: rel L∞ {linf} > {}", b.logits);
}

fn download(buf: &trueno_gpu::driver::GpuBuffer<f32>, elems: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; buf.len()];
    buf.copy_to_host(&mut v).expect("download state");
    v.truncate(elems);
    v
}

/// Every layer's conv window, recurrent state and written KV rows, compared.
fn assert_states_agree(
    gpu: &mut Qwen35CudaModel<'_>,
    batched: &Qwen35CudaState,
    per_token: &Qwen35CudaState,
    what: &str,
    budget: Budget,
) {
    gpu.executor_mut().sync_stream().expect("sync");
    assert_eq!(batched.kv_len, per_token.kv_len, "{what}: kv_len");
    let mut worst = (0.0f32, String::new());
    for il in 0..batched.conv.len() {
        let mut pairs = Vec::new();
        if let (Some((kb, vb)), Some((kp, vp))) = (&batched.kv[il], &per_token.kv[il]) {
            let n = per_token.kv_len * per_token.kv_row;
            pairs.push(("k", download(kb, n), download(kp, n)));
            pairs.push(("v", download(vb, n), download(vp, n)));
        } else {
            pairs.push((
                "conv",
                download(&batched.conv[il], batched.conv_len),
                download(&per_token.conv[il], per_token.conv_len),
            ));
            pairs.push((
                "ssm",
                download(&batched.ssm[il], batched.ssm_len),
                download(&per_token.ssm[il], per_token.ssm_len),
            ));
        }
        for (kind, b, p) in pairs {
            let linf = rel_linf(&b, &p, &format!("{what} layer {il} {kind}"));
            if linf > worst.0 {
                worst = (linf, format!("layer {il} {kind}"));
            }
        }
    }
    println!(
        "[3596] {what}: worst state rel L∞ {:.3e} at {}",
        worst.0, worst.1
    );
    assert!(
        worst.0 <= budget.state,
        "{what}: state rel L∞ {} at {} > {}",
        worst.0,
        worst.1,
        budget.state
    );
}

/// Deterministic token ids inside the embedding table, away from the special tokens.
fn tokens(n: usize, vocab: usize, seed: u32) -> Vec<u32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            1000 + (s >> 8) % (vocab as u32 - 2000)
        })
        .collect()
}

fn batched_equals_per_token(model_path: &str, n: usize, attention: super::PrefillAttention) {
    batched_equals_per_token_rows(model_path, n, attention, None);
}

fn batched_equals_per_token_rows(
    model_path: &str,
    n: usize,
    attention: super::PrefillAttention,
    chunk_rows: Option<usize>,
) {
    super::ATTENTION_OVERRIDE.with(|c| c.set(Some(attention)));
    let b = match attention {
        super::PrefillAttention::CublasF32 => F32_BUDGET,
        super::PrefillAttention::FlashF16In => FLASH_BUDGET,
    };
    if !std::path::Path::new(model_path).exists() {
        eprintln!("SKIP: {model_path} is absent");
        return;
    }
    let executor = crate::cuda_executor_or_skip!(0);
    let mapped = crate::gguf::MappedGGUFModel::from_path(model_path).expect("map the GGUF");
    let base = Qwen35Model::create_base_model(&mapped.model, mapped.data()).expect("base");
    let qwen =
        Qwen35Model::from_model_and_layers(&base, &mapped.model, mapped.data()).expect("qwen35");
    let mut gpu = Qwen35CudaModel::with_max_seq_len(&qwen, executor, n + 2).expect("gpu model");
    if let Some(rows) = chunk_rows {
        gpu.set_prefill_chunk_rows(rows);
    }
    // The path under test is the path that runs — never the default by accident.
    gpu.set_prefill_attention(attention);
    assert_eq!(gpu.prefill_attention_mode(), attention);
    let vocab = base.config.vocab_size;
    let prompt = tokens(n, vocab, 0x3596_0100 ^ n as u32);

    // Per-token: the path qwen35_gpu_decode used to take.
    let mut per_token = gpu.new_state().expect("state");
    let mut want = Vec::new();
    for (pos, &t) in prompt.iter().enumerate() {
        want = gpu
            .forward_single(t, &mut per_token, pos)
            .expect("forward_single");
    }

    // Batched, one call.
    let mut batched = gpu.new_state().expect("state");
    let rows = gpu.prefill_chunk_rows(n);
    let passes = super::attention_rows_for(gpu.dims, n, gpu.prefill_rows);
    let got = gpu.prefill(&prompt, &mut batched, 0).expect("prefill");
    let what = format!(
        "{model_path} n={n} (chunk rows {rows}, attention {}, rows/pass {passes})",
        attention.as_str()
    );
    assert_logits_agree(&got, &want, &format!("{what} last logits"), b);
    assert_states_agree(&mut gpu, &batched, &per_token, &what, b);

    // Split across two calls: pos0 > 0 reads the first call's KV rows. Compared
    // before either state decodes, so both hold exactly the prompt.
    let cut = n / 3;
    let mut split = gpu.new_state().expect("state");
    let _ = gpu
        .prefill(&prompt[..cut], &mut split, 0)
        .expect("prefill part 1");
    let got_split = gpu
        .prefill(&prompt[cut..], &mut split, cut)
        .expect("prefill part 2");
    assert_logits_agree(&got_split, &want, &format!("{what} split at {cut}"), b);
    assert_states_agree(
        &mut gpu,
        &split,
        &per_token,
        &format!("{what} split at {cut}"),
        b,
    );

    // One decode step continued from each state.
    let next = argmax(&want) as u32;
    let step_b = gpu
        .forward_single(next, &mut batched, n)
        .expect("decode from batched");
    let step_p = gpu
        .forward_single(next, &mut per_token, n)
        .expect("decode from per-token");
    assert_logits_agree(&step_b, &step_p, &format!("{what} decode step"), b);
    super::ATTENTION_OVERRIDE.with(|c| c.set(None));
}

#[test]
#[serial_test::serial]
fn qwen35_prefill_equals_per_token_at_64_positions_0_8b() {
    batched_equals_per_token(MODEL_0_8B, 64, super::PrefillAttention::CublasF32);
}

#[test]
#[serial_test::serial]
fn qwen35_flash_prefill_equals_per_token_at_64_positions_0_8b() {
    batched_equals_per_token(MODEL_0_8B, 64, super::PrefillAttention::FlashF16In);
}

#[test]
#[serial_test::serial]
fn qwen35_flash_prefill_equals_per_token_across_a_chunk_boundary_0_8b() {
    batched_equals_per_token(MODEL_0_8B, 600, super::PrefillAttention::FlashF16In);
}

#[test]
#[serial_test::serial]
fn qwen35_prefill_equals_per_token_across_a_chunk_boundary_0_8b() {
    // 600 > PREFILL_MAX_CHUNK_ROWS: two chunks, the second reading the first's KV.
    batched_equals_per_token(MODEL_0_8B, 600, super::PrefillAttention::CublasF32);
}

#[test]
#[serial_test::serial]
fn qwen35_prefill_equals_per_token_with_many_attention_passes_0_8b() {
    // Budget for exactly 37 query rows per pass at 600 positions (0.8B: 4 heads per
    // KV head): 14 passes over the first chunk, 3 over the second, none aligned.
    let rows = 37usize;
    let budget = 4 * 4 * rows * 600;
    super::SCORES_BUDGET_OVERRIDE.with(|c| c.set(Some(budget)));
    batched_equals_per_token(MODEL_0_8B, 600, super::PrefillAttention::CublasF32);
    super::SCORES_BUDGET_OVERRIDE.with(|c| c.set(None));
}

#[test]
#[serial_test::serial]
fn qwen35_prefill_equals_per_token_with_unified_memory_chunk_rows_0_8b() {
    // The unified-memory chunk (2048 rows) covers all 600 positions in one GEMM chunk;
    // 64 rows cuts them into ten. Both must land where the per-token path does.
    batched_equals_per_token_rows(
        MODEL_0_8B,
        600,
        super::PrefillAttention::FlashF16In,
        Some(super::UNIFIED_PREFILL_CHUNK_ROWS),
    );
    batched_equals_per_token_rows(
        MODEL_0_8B,
        600,
        super::PrefillAttention::FlashF16In,
        Some(64),
    );
}

#[test]
#[serial_test::serial]
fn qwen35_prefill_refuses_an_empty_prompt_and_positions_past_the_cache() {
    if !std::path::Path::new(MODEL_0_8B).exists() {
        eprintln!("SKIP: {MODEL_0_8B} is absent");
        return;
    }
    let executor = crate::cuda_executor_or_skip!(0);
    let mapped = crate::gguf::MappedGGUFModel::from_path(MODEL_0_8B).expect("map");
    let base = Qwen35Model::create_base_model(&mapped.model, mapped.data()).expect("base");
    let qwen =
        Qwen35Model::from_model_and_layers(&base, &mapped.model, mapped.data()).expect("qwen35");
    let mut gpu = Qwen35CudaModel::with_max_seq_len(&qwen, executor, 8).expect("gpu model");
    let mut state = gpu.new_state().expect("state");
    assert!(
        gpu.prefill(&[], &mut state, 0).is_err(),
        "an empty prompt must be refused"
    );
    assert!(
        gpu.prefill(&[1000; 9], &mut state, 0).is_err(),
        "9 positions into an 8-row cache must be refused, not truncated"
    );
    assert!(
        gpu.prefill(&[u32::MAX], &mut state, 0).is_err(),
        "a token outside the vocabulary must be refused"
    );
    // prefill_logits_at: positions must ascend inside pos0..pos0+len.
    assert!(
        gpu.prefill_logits_at(&[1000; 4], &mut state, 0, &[2, 1])
            .is_err(),
        "descending positions must be refused"
    );
    assert!(
        gpu.prefill_logits_at(&[1000; 4], &mut state, 0, &[4])
            .is_err(),
        "a position past the prompt must be refused"
    );
    let mut fresh = gpu.new_state().expect("state");
    let got = gpu
        .prefill_logits_at(&[1000, 1001, 1002, 1003], &mut fresh, 0, &[0, 3])
        .expect("two requested rows");
    assert_eq!(got.len(), 2, "one logits vector per requested position");
    // pos0 must be where the state stands (0.69.3, `apr serve` extends a state it
    // keeps across turns): `fresh` holds 4 rows now.
    assert!(
        gpu.prefill(&[1000], &mut fresh, 0).is_err(),
        "rewinding to pos0 0 over 4 written rows must be refused"
    );
    assert!(
        gpu.prefill(&[1000], &mut fresh, 5).is_err(),
        "skipping to pos0 5 past 4 written rows must be refused"
    );
    let mut unused = gpu.new_state().expect("state");
    assert!(
        gpu.prefill(&[1000], &mut unused, 1).is_err(),
        "a fresh state starts at 0"
    );
    gpu.prefill(&[1000], &mut fresh, 4)
        .expect("pos0 == kv_len continues the state");
}

/// The cop's 2026-09-21 ruling on the default: cuBLAS f32 first (exact, and faster
/// than flash on sm_89 from 20k to 148k), flash second where it can run; the
/// environment pins one. Pure — no device.
#[test]
fn qwen35_prefill_attention_prefers_f32_then_flash_and_the_environment_pins_one() {
    use super::{
        attention_candidates,
        PrefillAttention::{CublasF32, FlashF16In},
    };
    let rows: [(Option<&str>, bool, &[super::PrefillAttention]); 7] = [
        (None, true, &[CublasF32, FlashF16In]),
        (None, false, &[CublasF32]),
        (Some("f32"), true, &[CublasF32]),
        (Some("flash"), true, &[FlashF16In]),
        // Flash asked for where it cannot run: said so, and f32 — never nothing.
        (Some("flash"), false, &[CublasF32]),
        // An unrecognised value is the default, and printed.
        (Some("fast"), true, &[CublasF32, FlashF16In]),
        (Some(""), false, &[CublasF32]),
    ];
    for (forced, flash, want) in rows {
        assert_eq!(
            attention_candidates(forced, flash),
            want,
            "{forced:?}, flash supported {flash}"
        );
    }
}

/// #4313 GB10 bisect (diagnostic, `--ignored`): prefill a REAL probe through the
/// batched path and per-token, then print per-position logit cosine and every
/// layer's state divergence. The first layer whose state leaves the f32 noise
/// floor names where the batched prefill goes wrong. Probe: `APR_BISECT_IDS`
/// (comma-separated ids) or the last `APR_BISECT_N` (64) ids of `APR_BISECT_TEXT`.
#[test]
#[ignore = "diagnostic: needs APR_BISECT_MODEL and a probe"]
#[serial_test::serial]
fn qwen35_bisect_real_probe_per_layer() {
    let Ok(model_path) = std::env::var("APR_BISECT_MODEL") else {
        eprintln!("SKIP: APR_BISECT_MODEL unset");
        return;
    };
    super::ATTENTION_OVERRIDE.with(|c| c.set(Some(super::PrefillAttention::CublasF32)));
    crate::cuda::QWEN35_PREFILL_GEMM_OVERRIDE
        .with(|c| c.set(Some(crate::cuda::Qwen35PrefillGemm::F32)));
    let mapped = crate::gguf::MappedGGUFModel::from_path(&model_path).expect("map the GGUF");
    let probe: Vec<u32> = if let Ok(ids) = std::env::var("APR_BISECT_IDS") {
        ids.split(',').map(|s| s.trim().parse().expect("id")).collect()
    } else {
        let path = std::env::var("APR_BISECT_TEXT").expect("APR_BISECT_TEXT or APR_BISECT_IDS");
        let text = std::fs::read_to_string(&path).expect("read the probe text");
        let text = if std::env::var("APR_BISECT_CHAT").is_ok_and(|v| v == "1") {
            format!("<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n")
        } else {
            text
        };
        let all = mapped.model.encode(&text).expect("encode");
        println!("[4313] encoded {} ids, tail {:?}", all.len(), &all[all.len().saturating_sub(12)..]);
        let n: usize = std::env::var("APR_BISECT_N").map_or(64, |v| v.parse().expect("N"));
        all[all.len().saturating_sub(n)..].to_vec()
    };
    let n = probe.len();
    println!("[4313] probe n={n} ids={probe:?}");
    let base = Qwen35Model::create_base_model(&mapped.model, mapped.data()).expect("base");
    let qwen =
        Qwen35Model::from_model_and_layers(&base, &mapped.model, mapped.data()).expect("qwen35");
    let mut cpu_state = qwen.new_state(n + 2);
    let cpu: Vec<Vec<f32>> = probe
        .iter()
        .enumerate()
        .map(|(pos, &t)| qwen.forward_single_qwen35(t, &mut cpu_state, pos).expect("cpu"))
        .collect();
    if std::env::var("APR_BISECT_CPU_ONLY").is_ok_and(|v| v == "1") {
        let dir = std::env::var("APR_BISECT_DUMP").expect("APR_BISECT_DUMP");
        std::fs::create_dir_all(&dir).expect("dump dir");
        let bytes: Vec<u8> = cpu.iter().flatten().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(format!("{dir}/cpu.f32"), bytes).expect("dump");
        return;
    }
    let executor = crate::cuda_executor_or_skip!(0);
    // The F2 shape: a model sized for the probe (QWEN35_F2_PROBE_MAX + 2), a state
    // sized for probe + 1 decode step, and the session's planned chunk rows.
    let mut gpu = Qwen35CudaModel::with_max_seq_len(&qwen, executor, 66).expect("gpu model");
    if let Ok(rows) = std::env::var("APR_BISECT_ROWS") {
        gpu.set_prefill_chunk_rows(rows.parse().expect("rows"));
    }
    gpu.set_prefill_attention(super::PrefillAttention::CublasF32);
    println!("[4313] chunk rows {}", gpu.prefill_chunk_rows(n));

    let mut per_token = gpu.new_state_with_len(n + 1).expect("state");
    let mut want = Vec::with_capacity(n);
    for (pos, &t) in probe.iter().enumerate() {
        want.push(gpu.forward_single(t, &mut per_token, pos).expect("forward_single"));
    }
    let mut batched = gpu.new_state_with_len(n + 1).expect("state");
    let every: Vec<usize> = (0..n).collect();
    let got = gpu
        .prefill_logits_at(&probe, &mut batched, 0, &every)
        .expect("prefill_logits_at");
    let mut first_bad = None;
    for (pos, ((g, w), c)) in got.iter().zip(&want).zip(&cpu).enumerate() {
        let cos = cosine(g, w);
        println!(
            "[4313] pos {pos:3} batched~per_token {cos:.6} batched~cpu {:.6} per_token~cpu {:.6} argmax b{} p{} c{}",
            cosine(g, c),
            cosine(w, c),
            argmax(g),
            argmax(w),
            argmax(c)
        );
        if (cos < 0.99 || cosine(g, c) < 0.99) && first_bad.is_none() {
            first_bad = Some(pos);
        }
    }
    println!("[4313] first position with cos < 0.99: {first_bad:?}");
    if let Ok(dir) = std::env::var("APR_BISECT_DUMP") {
        std::fs::create_dir_all(&dir).expect("dump dir");
        for (tag, rows) in [("cpu", &cpu), ("gpu_batched", &got), ("gpu_per_token", &want)] {
            let bytes: Vec<u8> = rows.iter().flatten().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(format!("{dir}/{tag}.f32"), bytes).expect("dump");
        }
    }

    gpu.executor_mut().sync_stream().expect("sync");
    for il in 0..batched.conv.len() {
        let mut pairs = Vec::new();
        if let (Some((kb, vb)), Some((kp, vp))) = (&batched.kv[il], &per_token.kv[il]) {
            let len = per_token.kv_len * per_token.kv_row;
            pairs.push(("k", download(kb, len), download(kp, len)));
            pairs.push(("v", download(vb, len), download(vp, len)));
        } else {
            pairs.push((
                "conv",
                download(&batched.conv[il], batched.conv_len),
                download(&per_token.conv[il], per_token.conv_len),
            ));
            pairs.push((
                "ssm",
                download(&batched.ssm[il], batched.ssm_len),
                download(&per_token.ssm[il], per_token.ssm_len),
            ));
        }
        for (kind, b, p) in pairs {
            let scale = p.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let (arg, diff) = b.iter().zip(&p).enumerate().fold((0, 0.0f32), |acc, (i, (x, y))| {
                let d = (x - y).abs();
                if d > acc.1 { (i, d) } else { acc }
            });
            let nonfinite = b.iter().filter(|v| !v.is_finite()).count();
            println!(
                "[4313] layer {il:2} {kind:4} rel L∞ {:.3e} (abs {diff:.3e} at {arg}, scale {scale:.3e}, cos {:.6}, nonfinite {nonfinite})",
                if scale > 0.0 { diff / scale } else { diff },
                cosine(&b, &p)
            );
        }
    }
    super::ATTENTION_OVERRIDE.with(|c| c.set(None));
    crate::cuda::QWEN35_PREFILL_GEMM_OVERRIDE.with(|c| c.set(None));
}
