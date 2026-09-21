//! PMAT-084 FP8 activation reuse, scoped to GEMMs that share an input (#3727).
//!
//! `cublas_prefill_fp8_gemm` converts its f32 input to E4M3 once and lets the next GEMM on the
//! same input reuse it: K and V after Q, up after gate (3 conversions saved per layer). The key
//! used to be `(input_ptr, element_count)`, cleared only when the scratch was reallocated. The
//! contents behind an unchanged `(ptr, count)` change every layer, so any two FP8 GEMMs on the
//! same buffer with no other FP8 GEMM between them reused a stale activation. On a mixed-quant
//! model that is every layer: qwen2.5-coder-0.5b's Q/K/V/O/gate/up are Q5_0/Q8_0 (non-FP8
//! routes) and only `ffn_down` is FP8, so layers 1..23 multiplied layer 0's activation
//! (`PREFILL_DETAIL_TRACE`: 23 hits at one ptr, count 49 x 4864), which is #3602's cosine 0.4153.
//!
//! Reuse is now opt-in. Every batched GEMM dispatch calls [`Fp8ActivationCache::begin_dispatch`],
//! which drops the cached conversion unless the caller armed [`Fp8ActivationCache::share_next`]
//! immediately before it. So a hit needs a caller that knows the input is shared. A missing
//! `share_next` costs one extra conversion; it can no longer cost a wrong result.

/// The one cached FP8 activation conversion, and whether the next dispatch may reuse it.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Fp8ActivationCache {
    /// `(input_ptr, element_count)` of the conversion held in the FP8 activation scratch.
    key: Option<(u64, u32)>,
    /// Set by `share_next`, consumed by the next `begin_dispatch`.
    share_armed: bool,
}

impl Fp8ActivationCache {
    /// The next GEMM dispatch reads the same input, unmodified, as the one before it.
    pub(crate) fn share_next(&mut self) {
        self.share_armed = true;
    }

    /// Start of every batched GEMM dispatch, whatever route it takes: unless this dispatch was
    /// armed with `share_next`, the held conversion may be stale, so it is dropped.
    pub(crate) fn begin_dispatch(&mut self) {
        if !std::mem::take(&mut self.share_armed) {
            self.key = None;
        }
    }

    /// May the held conversion be reused for this input?
    pub(crate) fn hit(&self, input_ptr: u64, count: u32) -> bool {
        self.key == Some((input_ptr, count))
    }

    /// The FP8 scratch now holds the conversion of this input.
    pub(crate) fn record(&mut self, input_ptr: u64, count: u32) {
        self.key = Some((input_ptr, count));
    }

    /// The scratch was reallocated: whatever it held is gone.
    pub(crate) fn invalidate(&mut self) {
        self.key = None;
    }
}

#[cfg(test)]
mod tests {
    use super::Fp8ActivationCache;

    /// One batched GEMM, in the order the executor runs it: `batched_gemv_or_gemm` calls
    /// `begin_dispatch`; an FP8 route then consults `hit` and, on a miss, converts and records.
    /// Returns whether the FP8 route reused the held conversion.
    fn gemm(
        cache: &mut Fp8ActivationCache,
        shares_previous_input: bool,
        fp8: Option<(u64, u32)>,
    ) -> bool {
        if shares_previous_input {
            cache.share_next();
        }
        cache.begin_dispatch();
        match fp8 {
            Some((ptr, count)) if cache.hit(ptr, count) => true,
            Some((ptr, count)) => {
                cache.record(ptr, count);
                false
            },
            None => false,
        }
    }

    const NORM: u64 = 0x7000_0000; // hidden_buf1: attn/ffn norm output, and O/down's output
    const ATTN: u64 = 0x7100_0000; // attn_out_buf
    const ACT: u64 = 0x7200_0000; // ffn_act_buf (SwiGLU output)
    const M: u32 = 49;

    /// qwen2.5-coder-0.5b's shape: Q/K/V/O/gate/up take non-FP8 routes, only ffn_down is FP8,
    /// always on the same buffer with the same count.
    fn mixed_quant_hits(cache: &mut Fp8ActivationCache, layers: usize) -> usize {
        let mut hits = 0;
        for _ in 0..layers {
            gemm(cache, false, None); // Q
            gemm(cache, true, None); // K
            gemm(cache, true, None); // V
            gemm(cache, false, None); // O
            gemm(cache, false, None); // gate
            gemm(cache, true, None); // up
            hits += usize::from(gemm(cache, false, Some((ACT, M * 4864)))); // down
        }
        hits
    }

    #[test]
    fn a_mixed_quant_model_never_reuses_an_earlier_layers_activation() {
        let mut cache = Fp8ActivationCache::default();
        assert_eq!(
            mixed_quant_hits(&mut cache, 24),
            0,
            "each layer's ffn_down must convert its own SwiGLU output; the unscoped (ptr, count) \
             key reused layer 0's for layers 1..23 (#3727)"
        );
    }

    #[test]
    fn an_all_fp8_model_keeps_its_three_shared_input_hits_per_layer() {
        let mut cache = Fp8ActivationCache::default();
        let (hidden, inter) = (3584, 18944); // qwen2.5-coder-7b
        let mut hits = 0;
        for _ in 0..28 {
            let norm = Some((NORM, M * hidden));
            hits += usize::from(gemm(&mut cache, false, norm)); // Q: converts
            hits += usize::from(gemm(&mut cache, true, norm)); // K: reuses Q's
            hits += usize::from(gemm(&mut cache, true, norm)); // V: reuses Q's
            hits += usize::from(gemm(&mut cache, false, Some((ATTN, M * hidden)))); // O
            hits += usize::from(gemm(&mut cache, false, norm)); // gate: the norm buffer was rewritten
            hits += usize::from(gemm(&mut cache, true, norm)); // up: reuses gate's
            let act = Some((ACT, M * inter));
            hits += usize::from(gemm(&mut cache, false, act)); // down
        }
        assert_eq!(
            hits, 84,
            "K, V and up reuse their group leader's conversion: 3 x 28 layers"
        );
    }

    #[test]
    fn an_unarmed_dispatch_misses_even_on_the_same_ptr_and_count() {
        // gate reads hidden_buf1 with the same (ptr, count) Q did, after RMSNorm rewrote it.
        let mut cache = Fp8ActivationCache::default();
        assert!(!gemm(&mut cache, false, Some((NORM, M * 896))));
        assert!(
            !gemm(&mut cache, false, Some((NORM, M * 896))),
            "same key, not armed: stale"
        );
    }

    #[test]
    fn an_armed_dispatch_on_a_different_input_misses() {
        let mut cache = Fp8ActivationCache::default();
        gemm(&mut cache, false, Some((NORM, M * 896)));
        assert!(!gemm(&mut cache, true, Some((ATTN, M * 896))));
    }

    #[test]
    fn sharing_is_armed_for_one_dispatch_only() {
        let mut cache = Fp8ActivationCache::default();
        gemm(&mut cache, false, Some((NORM, M * 896)));
        assert!(gemm(&mut cache, true, Some((NORM, M * 896))));
        assert!(
            !gemm(&mut cache, false, Some((NORM, M * 896))),
            "the arm was consumed"
        );
    }

    #[test]
    fn a_non_fp8_group_leader_still_lets_its_group_share_one_conversion() {
        // Q on a non-FP8 route, K and V FP8 on the same input: K converts, V reuses K's.
        let mut cache = Fp8ActivationCache::default();
        gemm(&mut cache, false, None);
        assert!(!gemm(&mut cache, true, Some((NORM, M * 896))));
        assert!(gemm(&mut cache, true, Some((NORM, M * 896))));
    }

    #[test]
    fn a_reallocated_scratch_holds_nothing() {
        let mut cache = Fp8ActivationCache::default();
        gemm(&mut cache, false, Some((NORM, M * 896)));
        cache.invalidate();
        assert!(!gemm(&mut cache, true, Some((NORM, M * 896))));
    }
}
