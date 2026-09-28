//! #3715: the device KV length for a chat session, whose turn length is unknown at load.
//!
//! `apr chat` built its CUDA model with `OwnedQuantizedModelCuda::new`, a flat 2048
//! positions. The dense session refuses a turn longer than the device KV, so a
//! 26k-token chat turn on Qwen3-1.7B left the GPU for the CPU and hit the 600 s cell
//! timeout. `apr run` sizes the KV to the turn (#4268); chat cannot, because it loads
//! before the first turn is read. So chat takes the model's whole context when the
//! device has room for it, after the weights, the FP16/FP8 prefill cache and a
//! reserve, and keeps the old 2048 floor when it does not.

/// The positions a session model's device KV holds. Pure; see the module doc.
///
/// * `context_length` — the model's trained context (the ceiling).
/// * `kv_bytes_per_position` — f32 K+V over every layer for one position.
/// * `free_vram` — free device bytes before the KV and the weights are allocated.
/// * `resident_bytes` — the weights plus the prefill weight cache that load will upload.
/// * `reserve_bytes` — workspace, prefill score buffers and runtime headroom.
pub(crate) fn session_kv_len(
    context_length: usize,
    kv_bytes_per_position: usize,
    free_vram: usize,
    resident_bytes: usize,
    reserve_bytes: usize,
) -> usize {
    const FLOOR: usize = 2048;
    let floor = FLOOR.min(context_length.max(1));
    let room = free_vram.saturating_sub(resident_bytes.saturating_add(reserve_bytes));
    (room / kv_bytes_per_position.max(1)).clamp(floor, context_length.max(floor))
}

/// Elements the FP16/FP8 prefill weight cache holds: the seven projection matrices of
/// every layer plus the LM head (the "197 matrices" a 28-layer model logs).
pub(crate) fn prefill_cache_elements(
    num_layers: usize,
    hidden: usize,
    q_dim: usize,
    kv_dim: usize,
    intermediate: usize,
    vocab: usize,
) -> usize {
    let per_layer = hidden * (q_dim + 2 * kv_dim) + q_dim * hidden + 3 * hidden * intermediate;
    num_layers * per_layer + vocab * hidden
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: usize = 1_000_000_000;
    // f32 K+V per position: 2 * kv_heads * head_dim * 4 * layers
    const QWEN3_1_7B_KV: usize = 2 * 8 * 128 * 4 * 28;
    const QWEN3_8B_KV: usize = 2 * 8 * 128 * 4 * 36;

    #[test]
    fn small_model_gets_its_whole_context() {
        // Qwen3-1.7B on a 24 GB card with 1.7 GB used elsewhere: weights ~1.1 GB,
        // FP16 cache ~3.4 GB. The 26k-token chat turn that timed out must fit.
        let fp16 = 2 * prefill_cache_elements(28, 2048, 2048, 1024, 6144, 151_936);
        let len = session_kv_len(40_960, QWEN3_1_7B_KV, 22 * GB, GB + fp16, 4_600_000_000);
        assert_eq!(len, 40_960);
        assert!(len > 26_147);
    }

    #[test]
    fn a_model_that_fills_the_card_keeps_the_old_floor() {
        // Qwen3-8B: 4.8 GB weights + ~15 GB FP16 cache leave no room for more KV,
        // so the session keeps the 2048 it always had rather than starving the cache.
        let fp16 = 2 * prefill_cache_elements(36, 4096, 4096, 1024, 12_288, 151_936);
        assert!(fp16 > 14 * GB, "fp16 cache {fp16}");
        let len = session_kv_len(
            40_960,
            QWEN3_8B_KV,
            22 * GB,
            4_800_000_000 + fp16,
            4_600_000_000,
        );
        assert_eq!(len, 2048);
    }

    #[test]
    fn partial_room_is_used_and_never_exceeds_it() {
        let room = 3 * GB;
        let len = session_kv_len(1 << 20, QWEN3_1_7B_KV, 10 * GB, 5 * GB, 2 * GB);
        assert!(len * QWEN3_1_7B_KV <= room);
        assert!((len + 1) * QWEN3_1_7B_KV > room);
        assert!(len > 2048);
    }

    #[test]
    fn floor_never_exceeds_a_short_context() {
        assert_eq!(session_kv_len(512, QWEN3_1_7B_KV, 0, 0, 0), 512);
        assert_eq!(session_kv_len(0, 0, 0, 0, 0), 1);
    }

    #[test]
    fn fp16_cache_estimate_matches_the_logged_qwen3_1_7b_cache() {
        // [PMAT-037] FP16 weight cache: 197 matrices cached (3281.5 MB), measured on lambda.
        let bytes = 2 * prefill_cache_elements(28, 2048, 2048, 1024, 6144, 151_936);
        let logged = 3281.5 * 1024.0 * 1024.0;
        assert!(
            (bytes as f64 - logged).abs() / logged < 0.02,
            "{bytes} vs {logged}"
        );
    }
}
