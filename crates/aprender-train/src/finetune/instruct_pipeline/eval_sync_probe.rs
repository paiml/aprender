//! FALSIFY-CUDA-EVAL-ADAPTER-SYNC-001 — `evaluate()` must reflect the current
//! GPU-trained LoRA adapters, not a stale never-synced copy.
//!
//! Defect. The NF4 QLoRA epoch loop (`instruct_trainer.rs`) calls
//! `InstructPipeline::evaluate()` once per epoch to compute `val_loss`.
//! `evaluate()` computes logits with the CPU path
//! `self.model.forward_with_lora(&full_ids, &self.lora_layers)`, but GPU QLoRA
//! training writes adapter deltas into the GPU-resident `cuda_blocks`, and
//! `self.lora_layers` is only ever refreshed by `sync_lora_to_cpu()`, which is
//! invoked exclusively inside `save_checkpoint` — never before `evaluate()` in
//! the loop. So `evaluate()` reads adapters that lag (or entirely ignore) the
//! GPU training state, and per-epoch `val_loss` is constant across epochs and
//! runs. Downstream, `best_val_loss`/best-epoch selection freezes at epoch 0
//! and early stopping fires on a phantom plateau.
//!
//! Falsifier. We inject a GPU-only adapter change — the exact situation the
//! epoch loop creates — WITHOUT going through the optimizer: evaluate a fixed
//! sample, then `download`→(set B nonzero)→`upload` the block-0 LoRA weights so
//! the GPU adapters diverge from the still-zero CPU `lora_layers`, and evaluate
//! the SAME sample again. Doing it by direct upload keeps the probe independent
//! of the (separately-tracked) GPU optimizer/clip-kernel path and runs on a
//! fresh, uncorrupted CUDA context.
//!   RED  (buggy `evaluate`): `val_after == val_before` byte-for-byte — the CPU
//!        `lora_layers` never saw the GPU change, so the two forwards match.
//!   GREEN (fixed `evaluate`): `evaluate` syncs the GPU adapters into
//!        `lora_layers` first, so `val_after` reflects the injected delta and
//!        differs from `val_before`.
//!
//! Env-gated on `APR_PARITY_MODEL` (a Qwen2-family `.apr` with an embedded
//! tokenizer) + a CUDA GPU. Run:
//!
//! ```text
//! APR_PARITY_MODEL=~/models/qwen2.5-coder-1.5b-instruct-q4k.apr \
//!   cargo test -p aprender-train --features cuda --lib \
//!   falsify_cuda_eval_adapter_sync_001 -- --ignored --nocapture
//! ```

#[allow(clippy::wildcard_imports)]
use super::*;

#[test]
#[ignore = "requires CUDA GPU + APR_PARITY_MODEL pointing at a Qwen2-family .apr"]
fn falsify_cuda_eval_adapter_sync_001() {
    if !trueno_gpu::driver::cuda_available() {
        eprintln!("[eval-sync] SKIP: no CUDA device");
        return;
    }
    let Ok(model_path) = std::env::var("APR_PARITY_MODEL") else {
        eprintln!("[eval-sync] SKIP: APR_PARITY_MODEL unset");
        return;
    };
    let model_path = std::path::PathBuf::from(model_path);
    if !model_path.exists() {
        eprintln!("[eval-sync] SKIP: {model_path:?} does not exist");
        return;
    }

    // Inlined (not a helper fn) so cargo-mutants generates no mutant for a
    // function only reachable from this `#[ignore]` GPU test.
    let model_config = TransformerConfig::from_apr_metadata(
        Some(1536),
        Some(12),
        Some(2),
        Some(8960),
        Some(28),
        Some(151_936),
        Some(32_768),
        Some(1e-6),
        Some(1_000_000.0),
        Some("qwen2"),
    )
    .expect("qwen2 config from metadata");
    let instruct_config = InstructConfig {
        lora_rank: 8,
        lora_alpha: 16.0,
        learning_rate: 2e-5,
        epochs: 1,
        max_seq_len: 128,
        gradient_clip_norm: None,
        quantize_nf4: true,
    };
    let mut p = InstructPipeline::from_apr(&model_path, &model_config, instruct_config)
        .expect("pipeline from_apr");
    assert!(p.cuda_blocks.is_some(), "CUDA blocks must initialize (quantize_nf4=true)");

    let val_prompt = p.tokenize("What is two plus two in ordinary arithmetic?\n");
    let val_response = p.tokenize("The answer is four.");

    // Baseline: adapters are at init (B = 0 on both the GPU blocks and the CPU
    // `lora_layers`), so this value is the same on the buggy and fixed paths.
    let val_before =
        p.evaluate(std::slice::from_ref(&val_prompt), std::slice::from_ref(&val_response)).avg_loss;
    eprintln!("[eval-sync] val_loss BEFORE injection = {val_before:.8}");
    assert!(val_before.is_finite(), "baseline val_loss must be finite, got {val_before}");

    // Inject a GPU-only adapter change: keep A, set B to a nonzero constant, and
    // upload it back to block 0. This diverges the GPU adapters from the
    // still-zero CPU `lora_layers` exactly as a completed GPU training step
    // would — the state `evaluate()` is responsible for reconciling.
    {
        let blocks = p.cuda_blocks.as_mut().expect("cuda blocks");
        let (a_q, b_q, a_v, b_v) =
            blocks[0].download_lora_weights().expect("download block-0 LoRA (fresh context)");
        let b_q_hot = vec![0.05f32; b_q.len()];
        let b_v_hot = vec![0.05f32; b_v.len()];
        blocks[0]
            .upload_lora_weights(&a_q, &b_q_hot, &a_v, &b_v_hot)
            .expect("upload nonzero block-0 B");
    }

    // Fixed evaluate() syncs the GPU adapters into `lora_layers` before the CPU
    // forward, so this reflects the injected delta; buggy evaluate() reads the
    // untouched CPU adapters and returns the baseline byte-for-byte.
    let val_after =
        p.evaluate(std::slice::from_ref(&val_prompt), std::slice::from_ref(&val_response)).avg_loss;
    eprintln!("[eval-sync] val_loss AFTER  injection = {val_after:.8}");
    assert!(val_after.is_finite(), "post-injection val_loss must be finite, got {val_after}");

    let delta = (val_after - val_before).abs();
    eprintln!("[eval-sync] |Δ val_loss| = {delta:.8}");

    assert!(
        delta > 1e-6,
        "FALSIFY-CUDA-EVAL-ADAPTER-SYNC-001: evaluate() returned a byte-identical \
         val_loss ({val_before:.8} == {val_after:.8}) before and after a GPU-only \
         adapter change — evaluate() is reading stale CPU `lora_layers` that were \
         never synced from the GPU `cuda_blocks`. Per-epoch val_loss, best-epoch \
         selection, and early stopping are all driven by a constant.",
    );
}
