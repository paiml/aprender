/// FALSIFY-CB-009 (`contracts/continuous-batching-v1.yaml`): "KV cache populated for all slots
/// after prefill — `batched_kv_lengths[i] == prefill_len` for all i in 0..M".
///
/// This is the last untested link in the chain PERF-050 bisected. The attention kernel and its
/// decode-step wiring are both proven correct, but both proofs SEED THE BATCHED CACHE
/// THEMSELVES. Neither says anything about the handoff: whether prefill's bytes actually arrive
/// in the batched cache, at the right slot, at the right positions, for every layer.
///
/// `scatter_single_kv_to_batched` is that handoff for the sequential prefill path — the one a
/// forced-batched m=1 request takes, which is the configuration where aprender#2753 reproduces
/// with a single request and no batch at all. It is pure memory movement between two layouts,
/// so its correct output is computable without a model: the batched slot must equal the single
/// cache for the first `seq_len` positions of every KV head and every layer, and nothing
/// outside that slot may be touched.
#[cfg(test)]
#[cfg(feature = "cuda")]
mod cb009_kv_handoff {
    use super::*;

    const LAYERS: usize = 3;
    const NUM_HEADS: usize = 12;
    const NUM_KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 128;
    const MAX_LEN: usize = 256;
    const KV_SLOTS: usize = 4;
    const SEQ_LEN: usize = 17;
    const TARGET_SLOT: usize = 2; // not 0: a dropped slot offset must be visible

    /// Distinct per (layer, head, pos, dim) so any transposition shows up.
    fn val(layer: usize, head: usize, pos: usize, d: usize, tag: f32) -> f32 {
        tag + layer as f32 * 1000.0 + head as f32 * 100.0 + pos as f32 + d as f32 * 0.001
    }

    fn fill_single_cache(exec: &mut CudaExecutor, tag: f32, is_k: bool) -> Vec<Vec<f32>> {
        let per_layer = NUM_KV_HEADS * MAX_LEN * HEAD_DIM;
        let mut all = Vec::new();
        for layer in 0..LAYERS {
            let mut host = vec![0.0f32; per_layer];
            for head in 0..NUM_KV_HEADS {
                for pos in 0..SEQ_LEN {
                    let base = (head * MAX_LEN + pos) * HEAD_DIM;
                    for d in 0..HEAD_DIM {
                        host[base + d] = val(layer, head, pos, d, tag);
                    }
                }
            }
            let key = if is_k {
                format!("kv_{layer}_k")
            } else {
                format!("kv_{layer}_v")
            };
            exec.kv_cache_gpu
                .get_mut(&key)
                .expect("single cache present")
                .copy_from_host(&host)
                .expect("upload single cache");
            all.push(host);
        }
        all
    }

    /// The slot must equal the single cache, element for element, compared by bits so a
    /// transposition cannot alias through float equality.
    fn assert_slot_contents(got: &[f32], want_layer: &[f32], layer: usize, which: &str) {
        let slot_stride = NUM_KV_HEADS * MAX_LEN * HEAD_DIM;
        for head in 0..NUM_KV_HEADS {
            for pos in 0..SEQ_LEN {
                let src = (head * MAX_LEN + pos) * HEAD_DIM;
                let dst = TARGET_SLOT * slot_stride + src;
                let ok = (0..HEAD_DIM).all(|d| got[dst + d].to_bits() == want_layer[src + d].to_bits());
                assert!(
                    ok,
                    "FALSIFY-CB-009: {which} layer {layer} head {head} pos {pos} did not arrive \
                     in batched slot {TARGET_SLOT}. scatter_single_kv_to_batched is the \
                     prefill->decode handoff; if it drops a head, a layer, or the slot offset, \
                     decode attends over the wrong bytes and every proof about the attention \
                     kernel and its wiring is beside the point. See aprender#2753."
                );
            }
        }
    }

    /// Nothing outside the target slot may be written. A stride error here corrupts a PEER
    /// request, which is the failure mode that matters at c>1 and that an equality-only test
    /// would miss entirely.
    fn assert_peers_untouched(got: &[f32], layer: usize, which: &str) {
        let slot_stride = NUM_KV_HEADS * MAX_LEN * HEAD_DIM;
        let live = NUM_KV_HEADS * SEQ_LEN * HEAD_DIM;
        for slot in 0..KV_SLOTS {
            if slot == TARGET_SLOT {
                continue;
            }
            let base = slot * slot_stride;
            let clean = got[base..base + live].iter().all(|&x| x == 0.0);
            assert!(
                clean,
                "FALSIFY-CB-009: {which} layer {layer} scatter wrote into slot {slot} while \
                 targeting slot {TARGET_SLOT} — a stride error that corrupts a PEER request"
            );
        }
    }

    #[test]
    fn scatter_single_kv_to_batched_moves_every_layer_and_head_to_the_right_slot() {
        let Ok(mut exec) = CudaExecutor::new(0) else {
            println!(
                "cb009_kv_handoff: no CUDA device — SKIPPED. This covers the prefill->decode KV \
                 handoff, which is device memory movement and cannot be checked on the host."
            );
            return;
        };
        exec.init_kv_cache_gpu(LAYERS, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, MAX_LEN)
            .expect("single kv cache");
        exec.init_batched_kv_cache_gpu(LAYERS, KV_SLOTS)
            .expect("batched kv cache");

        let k_want = fill_single_cache(&mut exec, 1.0, true);
        let v_want = fill_single_cache(&mut exec, 2.0, false);

        exec.scatter_single_kv_to_batched(TARGET_SLOT, SEQ_LEN)
            .expect("scatter");
        exec.stream.synchronize().expect("sync");

        assert_eq!(
            exec.batched_kv_lengths[TARGET_SLOT], SEQ_LEN,
            "FALSIFY-CB-009: batched_kv_lengths[{TARGET_SLOT}] must equal the prefilled length"
        );

        let slot_stride = NUM_KV_HEADS * MAX_LEN * HEAD_DIM;
        let mut got = vec![0.0f32; KV_SLOTS * slot_stride];
        for layer in 0..LAYERS {
            exec.batched_kv_k_caches
                .get(&layer)
                .expect("batched k cache")
                .copy_to_host(&mut got)
                .expect("download k");
            assert_slot_contents(&got, &k_want[layer], layer, "K");
            assert_peers_untouched(&got, layer, "K");
            exec.batched_kv_v_caches
                .get(&layer)
                .expect("batched v cache")
                .copy_to_host(&mut got)
                .expect("download v");
            assert_slot_contents(&got, &v_want[layer], layer, "V");
            assert_peers_untouched(&got, layer, "V");
        }
    }
}
