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

/// K1 (#4603): batched-KV slots an `apr serve` KV budget holds beside the session KV.
/// The continuous-batching scheduler allocates slots of the same length for concurrent
/// requests (`fit_batched_kv_alloc`); two keep c=2 at full context.
pub(crate) const SERVE_BATCH_SLOTS: usize = 2;

/// K1 (#4603): serve's fixed KV length before K1. A server never gets less, so a
/// model that fills the card serves exactly what it served before.
pub(crate) const SERVE_FLOOR: usize = 4096;

/// K1 (#4603): the device KV length for `apr serve`: [`session_kv_len`] with each
/// position also paying for [`SERVE_BATCH_SLOTS`] batched slots, floored at
/// [`SERVE_FLOOR`] (or the model's context when shorter).
pub(crate) fn serving_kv_len(
    context_length: usize,
    kv_bytes_per_position: usize,
    free_vram: usize,
    resident_bytes: usize,
    reserve_bytes: usize,
) -> usize {
    let fit = session_kv_len(
        context_length,
        kv_bytes_per_position.saturating_mul(1 + SERVE_BATCH_SLOTS),
        free_vram,
        resident_bytes,
        reserve_bytes,
    );
    fit.max(SERVE_FLOOR.min(context_length.max(1)))
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: usize = 1_000_000_000;
    // f32 K+V per position: 2 * kv_heads * head_dim * 4 * layers
    const QWEN3_1_7B_KV: usize = 2 * 8 * 128 * 4 * 28;
    const QWEN3_8B_KV: usize = 2 * 8 * 128 * 4 * 36;
    // FP16 prefill caches as logged on lambda ([PMAT-037]): 3281.5 MB for 1.7B,
    // ~15.1 GB for 8B (`resident::projection_sizes` computes them from the tensors).
    const QWEN3_1_7B_FP16: usize = 3_441_000_000;
    const QWEN3_8B_FP16: usize = 15_100_000_000;

    #[test]
    fn small_model_gets_its_whole_context() {
        // Qwen3-1.7B on a 24 GB card with 1.7 GB used elsewhere: weights ~1.1 GB.
        // The 26k-token chat turn that timed out must fit.
        let len = session_kv_len(
            40_960,
            QWEN3_1_7B_KV,
            22 * GB,
            GB + QWEN3_1_7B_FP16,
            4_600_000_000,
        );
        assert_eq!(len, 40_960);
        assert!(len > 26_147);
    }

    #[test]
    fn a_model_that_fills_the_card_keeps_the_old_floor() {
        // Qwen3-8B: 4.8 GB weights + ~15 GB FP16 cache leave no room for more KV,
        // so the session keeps the 2048 it always had rather than starving the cache.
        let len = session_kv_len(
            40_960,
            QWEN3_8B_KV,
            22 * GB,
            4_800_000_000 + QWEN3_8B_FP16,
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

    /// K1: Qwen3-1.7B on a 24 GB card serves its whole 40960 context, where serve
    /// used a fixed 4096. RED before K1: the > 4096 serve cells overflowed the KV.
    #[test]
    fn serve_gives_a_small_model_its_whole_context() {
        let len = serving_kv_len(
            40_960,
            QWEN3_1_7B_KV,
            22 * GB,
            GB + QWEN3_1_7B_FP16,
            4_600_000_000,
        );
        assert_eq!(len, 40_960);
    }

    /// K1: the budget pays for the batched slots too, so the session KV plus
    /// `SERVE_BATCH_SLOTS` slots of the same length fit in the room.
    #[test]
    fn serve_budget_holds_the_batched_slots() {
        let room = 3 * GB;
        let len = serving_kv_len(1 << 20, QWEN3_1_7B_KV, 10 * GB, 5 * GB, 2 * GB);
        let per_pos = QWEN3_1_7B_KV * (1 + SERVE_BATCH_SLOTS);
        assert!(len * per_pos <= room);
        assert!((len + 1) * per_pos > room);
        assert!(len < session_kv_len(1 << 20, QWEN3_1_7B_KV, 10 * GB, 5 * GB, 2 * GB));
    }

    /// K1: a model that fills the card (Qwen3-8B + FP16 cache on 24 GB) keeps serve's
    /// old 4096, never the session floor of 2048 -- no model serves less than before.
    #[test]
    fn serve_never_drops_below_its_old_length() {
        let len = serving_kv_len(
            40_960,
            QWEN3_8B_KV,
            22 * GB,
            4_800_000_000 + QWEN3_8B_FP16,
            4_600_000_000,
        );
        assert_eq!(len, SERVE_FLOOR);
        assert_eq!(serving_kv_len(2048, QWEN3_8B_KV, 0, 0, 0), 2048);
    }
}
