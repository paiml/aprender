/// PERF-050 / FALSIFY-CB-008, one layer out from the kernel: the decode-step attention
/// WIRING.
///
/// `crates/aprender-gpu/.../paged/batched.rs` now proves the attention KERNEL is right at
/// production shape. That is a negative result about `batched_incremental_attention_into`,
/// not a positive one: this function also owns three things the kernel never sees, and all
/// three are reachable only from batched decode — prefill's attention comes from
/// `prefill_attention_from_packed` / `prefill_attention_cublas` and never calls this.
///
///   1. the `batched_kv_cache_scatter` PTX that writes THIS step's K/V into the batched
///      cache at `positions[slot]`,
///   2. the `k_ptrs` / `v_ptrs` slot-base arrays, which are padded to
///      `batched_kv_lengths.len()` — 32 on this box, where only `m` are live,
///   3. `seq_lens`, which is per slot and is zeroed for slots in `batched_done_mask`.
///
/// A bug in any of those hands a correct kernel the wrong bytes. The oracle is the same one
/// that settled the kernel: a CPU softmax over the K/V this test itself placed, including the
/// position the scatter is supposed to have just written. If the scatter drops or misplaces
/// the current step's K/V, the last position is missing or wrong and the reference diverges.
#[cfg(test)]
#[cfg(feature = "cuda")]
mod cb008_decode_attention_wiring {
    use super::*;

    const LAYERS: usize = 1;
    const NUM_HEADS: usize = 12;
    const NUM_KV_HEADS: usize = 2; // GQA 6:1, as Qwen2.5-Coder-1.5B
    const HEAD_DIM: usize = 128;
    const MAX_LEN: usize = 2048;
    const M: usize = 3;
    const KV_SLOTS: usize = 8; // deliberately > M: the padded-pointer-array case
    const PRIOR: [usize; M] = [17, 11, 5]; // positions already in cache, per slot

    fn kv_group_of(head: usize) -> usize {
        head * NUM_KV_HEADS / NUM_HEADS
    }
    fn k_base(slot: usize, group: usize, d: usize) -> f32 {
        0.02 * (1 + (slot * 3 + group * 5 + d) % 11) as f32
    }
    /// Strictly increasing in `pos`, so the online-softmax rescale is exercised and the LAST
    /// position — the one the decode scatter writes — dominates the output. A wiring bug that
    /// drops it is therefore maximally visible.
    fn k_at(slot: usize, group: usize, pos: usize, d: usize) -> f32 {
        k_base(slot, group, d) * (1.0 + 0.6 * pos as f32)
    }
    fn v_at(slot: usize, group: usize, pos: usize, d: usize) -> f32 {
        (pos as f32 + 1.0) + 0.01 * d as f32 + 0.5 * slot as f32 + 0.25 * group as f32
    }
    fn q_at(slot: usize, head: usize, d: usize) -> f32 {
        0.5 + 0.01 * ((slot * 5 + head * 3 + d) % 7) as f32
    }

    /// CPU softmax over PRIOR[slot] + 1 positions: the cache contents plus the token this
    /// decode step is scattering.
    fn reference(slot: usize, head: usize) -> Vec<f32> {
        let group = kv_group_of(head);
        let total = PRIOR[slot] + 1;
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();
        let scores: Vec<f32> = (0..total)
            .map(|p| {
                let dot: f32 = (0..HEAD_DIM)
                    .map(|d| q_at(slot, head, d) * k_at(slot, group, p, d))
                    .sum();
                dot * scale
            })
            .collect();
        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
        let denom: f32 = exps.iter().sum();
        (0..HEAD_DIM)
            .map(|d| {
                let acc: f32 = (0..total).map(|p| exps[p] * v_at(slot, group, p, d)).sum();
                acc / denom
            })
            .collect()
    }

    /// Fill the batched KV cache with PRIOR[slot] positions, exactly as a correct
    /// prefill + scatter would have left it. Layout: [slot][kv_head][max_len][dim].
    fn seed_cache(exec: &mut CudaExecutor) {
        let slot_stride = NUM_KV_HEADS * MAX_LEN * HEAD_DIM;
        let mut k = vec![0.0f32; KV_SLOTS * slot_stride];
        let mut v = vec![0.0f32; KV_SLOTS * slot_stride];
        for slot in 0..M {
            for group in 0..NUM_KV_HEADS {
                for pos in 0..PRIOR[slot] {
                    let base = slot * slot_stride + (group * MAX_LEN + pos) * HEAD_DIM;
                    for d in 0..HEAD_DIM {
                        k[base + d] = k_at(slot, group, pos, d);
                        v[base + d] = v_at(slot, group, pos, d);
                    }
                }
            }
        }
        exec.batched_kv_k_caches
            .get_mut(&0)
            .expect("k cache layer 0")
            .copy_from_host(&k)
            .expect("upload k");
        exec.batched_kv_v_caches
            .get_mut(&0)
            .expect("v cache layer 0")
            .copy_from_host(&v)
            .expect("upload v");
        for slot in 0..M {
            exec.batched_kv_lengths[slot] = PRIOR[slot];
        }
        exec.batched_done_mask = vec![false; M];
    }

    /// This decode step's Q, and the K/V the scatter must place at positions[slot].
    fn step_inputs() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let q_dim = NUM_HEADS * HEAD_DIM;
        let kv_dim = NUM_KV_HEADS * HEAD_DIM;
        let mut q = vec![0.0f32; M * q_dim];
        let mut k = vec![0.0f32; M * kv_dim];
        let mut v = vec![0.0f32; M * kv_dim];
        for slot in 0..M {
            for head in 0..NUM_HEADS {
                for d in 0..HEAD_DIM {
                    q[slot * q_dim + head * HEAD_DIM + d] = q_at(slot, head, d);
                }
            }
            for group in 0..NUM_KV_HEADS {
                let p = PRIOR[slot];
                for d in 0..HEAD_DIM {
                    k[slot * kv_dim + group * HEAD_DIM + d] = k_at(slot, group, p, d);
                    v[slot * kv_dim + group * HEAD_DIM + d] = v_at(slot, group, p, d);
                }
            }
        }
        (q, k, v)
    }

    fn assert_slot_head(got: &[f32], slot: usize, head: usize) {
        let q_dim = NUM_HEADS * HEAD_DIM;
        let want = reference(slot, head);
        let base = slot * q_dim + head * HEAD_DIM;
        for d in 0..HEAD_DIM {
            let g = got[base + d];
            let w = want[d];
            assert!(
                (g - w).abs() <= 2e-3 * w.abs().max(1.0),
                "FALSIFY-CB-008 (wiring): slot {slot} head {head} (kv group {group}, prior \
                 {prior} + 1 scattered) dim {d} = {g}, CPU softmax reference = {w}. Scores \
                 increase with position, so the token this step scattered dominates: a scatter \
                 writing the wrong offset, a k_ptrs/v_ptrs slot base off by a stride, or a \
                 seq_lens that excludes the new position all land here. KV_SLOTS ({slots}) > M \
                 ({m}) on purpose -- the pointer arrays are padded, and reading the padding is \
                 one of the faults this is looking for. See aprender#2753.",
                group = kv_group_of(head),
                prior = PRIOR[slot],
                slots = KV_SLOTS,
                m = M
            );
        }
    }

    #[test]
    fn decode_step_attention_wiring_matches_cpu_softmax() {
        let Ok(mut exec) = CudaExecutor::new(0) else {
            println!(
                "cb008_decode_attention_wiring: no CUDA device -- SKIPPED. This test covers the \
                 decode-only scatter/pointer-array/seq_lens wiring; the always-on guard for the \
                 kernel itself is aprender-gpu's cb008_online_softmax_rescale (PTX codegen)."
            );
            return;
        };
        let hidden_dim = NUM_HEADS * HEAD_DIM;
        let q_dim = hidden_dim;

        exec.init_kv_cache_gpu(LAYERS, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, MAX_LEN)
            .expect("single kv cache");
        exec.init_batched_kv_cache_gpu(LAYERS, KV_SLOTS)
            .expect("batched kv cache");
        exec.init_batched_workspace(hidden_dim, hidden_dim, M)
            .expect("workspace");
        seed_cache(&mut exec);

        let (q_host, k_step, v_step) = step_inputs();
        let q_buf = GpuBuffer::from_host(&exec.context, &q_host).expect("q");
        let k_buf = GpuBuffer::from_host(&exec.context, &k_step).expect("k");
        let v_buf = GpuBuffer::from_host(&exec.context, &v_step).expect("v");
        let out_buf = GpuBuffer::<f32>::new(&exec.context, M * q_dim).expect("out");

        let positions: Vec<u32> = PRIOR.iter().map(|&p| p as u32).collect();
        exec.batched_incremental_attention_into(0, &q_buf, &k_buf, &v_buf, &out_buf, M, &positions)
            .expect("batched_incremental_attention_into");
        exec.stream.synchronize().expect("sync");

        let mut got = vec![0.0f32; M * q_dim];
        out_buf.copy_to_host(&mut got).expect("download");

        // The wiring is also supposed to have advanced the per-slot lengths.
        for slot in 0..M {
            assert_eq!(
                exec.batched_kv_lengths[slot],
                PRIOR[slot] + 1,
                "FALSIFY-CB-008: slot {slot} length was not advanced past the token this decode \
                 step scattered; the next step will attend over a short cache"
            );
        }

        for slot in 0..M {
            for head in 0..NUM_HEADS {
                assert_slot_head(&got, slot, head);
            }
        }
    }
}
