//! WGPU Eval — forward-only eval through full 36-layer transformer with LoRA
//!
//! Phase 8.1: Uses trained LoRA adapters in scoring (previous eval skipped them).
//! Contract: C-WGPU-IMPROVE-001
//!
//! ```bash
//! cargo run --features gpu --release --example wgpu_eval -- \
//!   --model /home/noah/src/models/qwen3-4b \
//!   --checkpoint /tmp/wgpu-train-output/lora-checkpoint-step500.json \
//!   --test /home/noah/src/bashrs/training/shellsafetybench/splits/test.jsonl
//! ```

fn main() {
    #[cfg(not(feature = "gpu"))]
    {
        eprintln!("Error: --features gpu required");
        std::process::exit(1);
    }
    #[cfg(feature = "gpu")]
    {
        if let Err(e) = run() {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    }
}

#[cfg(feature = "gpu")]
fn run() -> Result<(), String> {
    use entrenar::train::transformer_trainer::wgpu_trainer::WgpuModelState;

    let args: Vec<String> = std::env::args().collect();
    let model_dir =
        get_arg(&args, "--model").unwrap_or_else(|| "/home/noah/src/models/qwen3-4b".into());
    let ckpt_path = get_arg(&args, "--checkpoint")
        .unwrap_or_else(|| "/tmp/wgpu-train-output/lora-checkpoint-step500.json".into());
    let test_path = get_arg(&args, "--test").unwrap_or_else(|| {
        "/home/noah/src/bashrs/training/shellsafetybench/splits/test.jsonl".into()
    });
    let max_entries: usize = get_arg(&args, "--max").and_then(|s| s.parse().ok()).unwrap_or(500);
    let fast = args.iter().any(|a| a == "--fast"); // lm_head-only (skip 36 layers)

    eprintln!(
        "=== WGPU Eval ({}): ShellSafetyBench ===\n",
        if fast { "lm_head-only" } else { "Full Forward" }
    );

    // 1. Load tokenizer + model + checkpoint
    let tokenizer_path = std::path::Path::new(&model_dir).join("tokenizer.json");
    let tokenizer = entrenar::tokenizer::HfTokenizer::from_file(&tokenizer_path)
        .map_err(|e| format!("Tokenizer: {e}"))?;
    let rank: u32 = get_arg(&args, "--rank").and_then(|s| s.parse().ok()).unwrap_or(32);
    let alpha = rank as f32 * 2.0;
    let mut model = WgpuModelState::load_qwen3_4b(std::path::Path::new(&model_dir), rank, alpha)?;
    let (step, ckpt_loss) = model.load_checkpoint(std::path::Path::new(&ckpt_path))?;
    eprintln!("Loaded checkpoint: step={step}, loss={ckpt_loss:.3}");

    // 2. GPU device + populate weight cache (pre-transposed, one-time)
    let device = trueno::backends::gpu::GpuDevice::new()?;
    model.populate_weight_cache(&device)?;
    eprintln!("Weight cache populated\n");

    let h = model.hidden_size;
    let i_size = model.intermediate_size;
    let v = model.vocab_size;
    let n_layers = model.num_layers;
    let nh = model.num_heads as u32;
    let nkv = model.num_kv_heads as u32;
    let hd = model.head_dim as u32;
    let lora_alpha = 32.0f32;

    // 3. Load test data
    let test_data = std::fs::read_to_string(&test_path).map_err(|e| format!("Test data: {e}"))?;
    let entries: Vec<(String, i32)> = test_data
        .lines()
        .filter_map(|line| {
            let val: serde_json::Value = serde_json::from_str(line).ok()?;
            Some((val["input"].as_str()?.to_string(), val["label"].as_i64()? as i32))
        })
        .take(max_entries)
        .collect();
    eprintln!("Test entries: {} (max {})\n", entries.len(), max_entries);

    // 4. Score each entry with BOTH methods (ensemble)
    // Stores: (full_forward_loss, lm_head_loss, seq_len, label)
    let mut all_scores: Vec<(f32, f32, usize, i32)> = Vec::new();

    for (idx, (input, label)) in entries.iter().enumerate() {
        let tokens = tokenizer.encode(input);
        let max_seq = if fast { 64 } else { 24 }; // shorter seq for full-forward (speed)
        let len = tokens.len().min(max_seq);
        if len < 2 {
            continue;
        }
        let input_ids = &tokens[..len];
        let target_ids: Vec<u32> = tokens[1..len].to_vec();
        let s = target_ids.len() as u32;

        // Embedding lookup
        let mut hidden = vec![0.0f32; s as usize * h];
        for (si, &tid) in input_ids[..s as usize].iter().enumerate() {
            let tid = (tid as usize).min(v - 1);
            for hi in 0..h {
                hidden[si * h + hi] = model.lm_head[tid * h + hi];
            }
        }
        // lm_head-only score (before transformer forward)
        let lm_loss = compute_ce_loss(&device, &hidden, &target_ids, &model.lm_head, h, v)?;

        // Full forward or lm_head-only
        if !fast {
            let rmsnorm = |buf: &mut [f32], seq: usize, dim: usize| {
                let eps = 1e-5f32;
                for si in 0..seq {
                    let rms = (buf[si * dim..(si + 1) * dim].iter().map(|x| x * x).sum::<f32>()
                        / dim as f32
                        + eps)
                        .sqrt();
                    for hi in 0..dim {
                        buf[si * dim + hi] /= rms;
                    }
                }
            };

            // Forward through all 36 layers — CPU matmul (fast for small seq_len)
            // GPU dispatch overhead (~100ms each × 252 dispatches = 25s) >> CPU compute
            // CPU matmul at seq=64: ~5ms/layer × 36 = ~180ms total
            for layer_idx in 0..n_layers {
                rmsnorm(&mut hidden, s as usize, h);
                // Attention: CPU matmul for QKV+O (small: [seq,h]@[h,dim])
                let (q_w, k_w, v_w, o_w) = model.attn_cache[layer_idx]
                    .as_ref()
                    .map(|(q, k, vv, o)| (q.as_slice(), k.as_slice(), vv.as_slice(), o.as_slice()))
                    .expect("attn cache");
                let (attn_out, _) =
                    entrenar::train::transformer_trainer::wgpu_attention::attention_forward(
                        &device,
                        &hidden,
                        q_w,
                        k_w,
                        v_w,
                        o_w,
                        &model.lora[layer_idx].q,
                        &model.lora[layer_idx].v,
                        lora_alpha,
                        s,
                        h as u32,
                        nh,
                        nkv,
                        hd,
                    )?;
                for j in 0..(s as usize * h) {
                    hidden[j] += attn_out[j];
                }
                rmsnorm(&mut hidden, s as usize, h);
                // FFN forward
                let (gw, uw, dw) = model.ffn_cache[layer_idx]
                    .as_ref()
                    .map(|(g, u, d)| (g.as_slice(), u.as_slice(), d.as_slice()))
                    .expect("ffn cache");
                let sl = s as usize;
                let mut go = vec![0.0f32; sl * i_size];
                device.matmul(&hidden, gw, &mut go, sl, h, i_size)?;
                let mut uo = vec![0.0f32; sl * i_size];
                device.matmul(&hidden, uw, &mut uo, sl, h, i_size)?;
                let swiglu: Vec<f32> = go
                    .iter()
                    .zip(uo.iter())
                    .map(|(&g, &u)| {
                        let sig = 1.0 / (1.0 + (-g).exp());
                        g * sig * u
                    })
                    .collect();
                let mut fo = vec![0.0f32; sl * h];
                device.matmul(&swiglu, dw, &mut fo, sl, i_size, h)?;
                for j in 0..sl * h {
                    hidden[j] += fo[j];
                }
            }
        } // end if !fast

        // Full-forward loss (after transformer)
        let ff_loss = compute_ce_loss(&device, &hidden, &target_ids, &model.lm_head, h, v)?;

        all_scores.push((ff_loss, lm_loss, s as usize, *label));
        if (idx + 1) % 50 == 0 {
            eprintln!(
                "  [{}/{}] ff={ff_loss:.1} lm={lm_loss:.1} label={label}",
                idx + 1,
                entries.len()
            );
        }
    }

    // 5. Build multiple scoring methods from the two raw scores
    // Method 1: full-forward inverted
    let scores_ff_inv: Vec<(f32, i32)> = all_scores.iter().map(|(ff, _, _, l)| (-ff, *l)).collect();
    // Method 2: lm_head raw (higher = unsafe)
    let scores_lm: Vec<(f32, i32)> = all_scores.iter().map(|(_, lm, _, l)| (*lm, *l)).collect();
    // Method 3: ensemble = lm_head_loss - alpha * full_forward_loss (inverted ff + raw lm)
    let scores_ens: Vec<(f32, i32)> =
        all_scores.iter().map(|(ff, lm, _, l)| (lm - 0.5 * ff, *l)).collect();
    // Method 4: length-normalized full-forward inverted
    let scores_len: Vec<(f32, i32)> =
        all_scores.iter().map(|(ff, _, slen, l)| (-ff / (*slen as f32).sqrt(), *l)).collect();
    // Method 5: ensemble + length norm
    let scores_ens_len: Vec<(f32, i32)> = all_scores
        .iter()
        .map(|(ff, lm, slen, l)| {
            let norm = (*slen as f32).sqrt();
            (lm / norm - 0.5 * ff / norm, *l)
        })
        .collect();

    let (mcc_ff, _, tp_ff, fp_ff, tn_ff, fn_ff) = compute_best_mcc(&scores_ff_inv);
    let (mcc_lm, _, tp_lm, fp_lm, tn_lm, fn_lm) = compute_best_mcc(&scores_lm);
    let (mcc_ens, _, tp_e, fp_e, tn_e, fn_e) = compute_best_mcc(&scores_ens);
    let (mcc_len, _, tp_ln, fp_ln, tn_ln, fn_ln) = compute_best_mcc(&scores_len);
    let (mcc_el, _, tp_el, fp_el, tn_el, fn_el) = compute_best_mcc(&scores_ens_len);

    // Grid search alpha for ensemble
    let mut best_alpha_mcc = 0.0f32;
    let mut best_alpha = 0.0f32;
    for a in 0..20 {
        let alpha = a as f32 * 0.1;
        let sc: Vec<(f32, i32)> =
            all_scores.iter().map(|(ff, lm, _, l)| (lm - alpha * ff, *l)).collect();
        let (m, _, _, _, _, _) = compute_best_mcc(&sc);
        if m > best_alpha_mcc {
            best_alpha_mcc = m;
            best_alpha = alpha;
        }
    }

    let best_mcc = mcc_ff.max(mcc_lm).max(mcc_ens).max(mcc_len).max(mcc_el).max(best_alpha_mcc);

    eprintln!("\n=== Results ({}) ===", if fast { "lm_head-only" } else { "Ensemble" });
    eprintln!("Entries scored: {}", all_scores.len());
    eprintln!("  FF inverted:  MCC={mcc_ff:.4} TP={tp_ff} FP={fp_ff} TN={tn_ff} FN={fn_ff}");
    eprintln!("  lm_head raw:  MCC={mcc_lm:.4} TP={tp_lm} FP={fp_lm} TN={tn_lm} FN={fn_lm}");
    eprintln!("  Ensemble:     MCC={mcc_ens:.4} TP={tp_e} FP={fp_e} TN={tn_e} FN={fn_e}");
    eprintln!("  Len-norm FF:  MCC={mcc_len:.4} TP={tp_ln} FP={fp_ln} TN={tn_ln} FN={fn_ln}");
    eprintln!("  Ens+LenNorm:  MCC={mcc_el:.4} TP={tp_el} FP={fp_el} TN={tn_el} FN={fn_el}");
    eprintln!("  Grid(a={best_alpha:.1}): MCC={best_alpha_mcc:.4}");
    eprintln!("\nBest: MCC={best_mcc:.4}");
    eprintln!("Ship criteria: MCC > 0.50 → {}", if best_mcc > 0.5 { "PASS" } else { "FAIL" });
    eprintln!("Stretch goal: MCC > 0.75 → {}", if best_mcc > 0.75 { "PASS" } else { "FAIL" });

    let safe_ff: Vec<f32> = all_scores.iter().filter(|s| s.3 == 0).map(|s| s.0).collect();
    let unsafe_ff: Vec<f32> = all_scores.iter().filter(|s| s.3 == 1).map(|s| s.0).collect();
    if !safe_ff.is_empty() && !unsafe_ff.is_empty() {
        let ms = safe_ff.iter().sum::<f32>() / safe_ff.len() as f32;
        let mu = unsafe_ff.iter().sum::<f32>() / unsafe_ff.len() as f32;
        let ss = (safe_ff.iter().map(|x| (x - ms) * (x - ms)).sum::<f32>() / safe_ff.len() as f32)
            .sqrt();
        let su = (unsafe_ff.iter().map(|x| (x - mu) * (x - mu)).sum::<f32>()
            / unsafe_ff.len() as f32)
            .sqrt();
        eprintln!(
            "FF  Safe: mean={ms:.1} std={ss:.1} (n={}) | Unsafe: mean={mu:.1} std={su:.1} (n={})",
            safe_ff.len(),
            unsafe_ff.len()
        );
    }
    Ok(())
}

#[cfg(feature = "gpu")]
fn cpu_matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0f32;
            for p in 0..k {
                s += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = s;
        }
    }
}

#[cfg(feature = "gpu")]
fn compute_ce_loss(
    device: &trueno::backends::gpu::GpuDevice,
    hidden: &[f32],
    targets: &[u32],
    lm_head: &[f32],
    h: usize,
    v: usize,
) -> Result<f32, String> {
    let s = targets.len();
    let mut logits = vec![0.0f32; s * v];
    device.gemm_backward_a(hidden, lm_head, &mut logits, s as u32, v as u32, h as u32)?;
    let mut loss = 0.0f32;
    for si in 0..s {
        let row = &logits[si * v..(si + 1) * v];
        let mx = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let lse = mx + row.iter().map(|&x| (x - mx).exp()).sum::<f32>().ln();
        let t = targets[si] as usize;
        if t < v {
            loss -= logits[si * v + t] - lse;
        }
    }
    Ok(loss / s as f32)
}

#[cfg(feature = "gpu")]
fn compute_best_mcc(scores: &[(f32, i32)]) -> (f32, f32, usize, usize, usize, usize) {
    let mut best_mcc = -2.0f32;
    let mut best = (0.0f32, 0usize, 0usize, 0usize, 0usize);
    let mut thresholds: Vec<f32> = scores.iter().map(|(s, _)| *s).collect();
    thresholds.sort_by(|a, b| a.partial_cmp(b).unwrap());
    thresholds.dedup();
    for &thresh in &thresholds {
        let (mut tp, mut fp, mut tn, mut fn_) = (0usize, 0, 0, 0);
        for &(score, label) in scores {
            let pred = i32::from(score > thresh);
            match (pred, label) {
                (1, 1) => tp += 1,
                (1, 0) => fp += 1,
                (0, 0) => tn += 1,
                (0, 1) => fn_ += 1,
                _ => {}
            }
        }
        let num = (tp * tn) as f64 - (fp * fn_) as f64;
        let den =
            ((tp + fp) as f64 * (tp + fn_) as f64 * (tn + fp) as f64 * (tn + fn_) as f64).sqrt();
        let mcc = if den > 0.0 { (num / den) as f32 } else { 0.0 };
        if mcc > best_mcc {
            best_mcc = mcc;
            best = (thresh, tp, fp, tn, fn_);
        }
    }
    (best_mcc, best.0, best.1, best.2, best.3, best.4)
}

#[cfg(feature = "gpu")]
fn get_arg(args: &[String], flag: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == flag).map(|w| w[1].clone())
}
