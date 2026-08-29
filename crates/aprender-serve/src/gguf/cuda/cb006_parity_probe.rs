// PERF-050 round 3: the in-process A/B that CB-006's residue needs.
//
// Four components of batched decode are individually proven correct (attention kernel,
// decode-step attention wiring, the prefill->decode KV handoff, prefill itself) and the output
// is still garbage. That is a contradiction, and no further component-level test can resolve
// it: each one seeds its own state, so none of them observes the state the real system is in.
//
// This probe observes it. At m == 1 the batched decode and the M=1 fast path compute the SAME
// function of the SAME prompt: one token at one position against a KV cache that the same
// prefill populated. `cb009_kv_handoff` proves the batched cache is a bit-exact copy of the
// single cache, so `forward_all_layers_gpu_to_logits` — the M=1 path that produces coherent
// English — is a true oracle for `forward_batched_to_logits`. Their logits must match.
//
// `APR_PARITY_PROBE=1` runs it once, at the first decode step of any m == 1 batch, and prints
// three comparisons rather than one, because a bare oracle comparison has two silent failure
// modes and this repo has been bitten by both:
//
//   SELF   batched vs batched, re-run. Must be IDENTICAL. If this diverges the batched path is
//          not deterministic and no comparison against it means anything; if it "passes" while
//          the comparator is broken, the comparator is reporting agreement it never computed.
//   ORACLE batched vs M=1. This is the measurement.
//   PERTURBED batched vs M=1 fed a deliberately corrupted embedding. Must DIVERGE. This is the
//          can-it-go-red control: an oracle comparison that accidentally compares a buffer with
//          itself, or compares nothing, passes SELF and ORACLE and fails only here.
//
// The probe is read-only with respect to the batch it observes: it restores `batched_kv_stride`
// and the batched workspace before returning, exactly as `add_slot_to_batch` does around its own
// M=1 prefill.

#[cfg(feature = "cuda")]
fn parity_probe_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("APR_PARITY_PROBE").as_deref() == Ok("1"))
}

/// Summary of one logits-vs-logits comparison.
#[cfg(feature = "cuda")]
struct LogitsDelta {
    argmax_a: usize,
    argmax_b: usize,
    max_abs: f32,
    cosine: f32,
    nonfinite_a: usize,
    nonfinite_b: usize,
    zeros_a: usize,
    zeros_b: usize,
}

#[cfg(feature = "cuda")]
fn compare_logits(a: &[f32], b: &[f32]) -> LogitsDelta {
    let argmax = |v: &[f32]| {
        v.iter()
            .enumerate()
            .max_by(|(_, x), (_, y)| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(i, _)| i)
    };
    let mut max_abs = 0.0f32;
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b.iter()) {
        max_abs = max_abs.max((x - y).abs());
        dot += f64::from(x) * f64::from(y);
        na += f64::from(x) * f64::from(x);
        nb += f64::from(y) * f64::from(y);
    }
    let denom = (na.sqrt() * nb.sqrt()).max(f64::MIN_POSITIVE);
    LogitsDelta {
        argmax_a: argmax(a),
        argmax_b: argmax(b),
        max_abs,
        cosine: (dot / denom) as f32,
        nonfinite_a: a.iter().filter(|x| !x.is_finite()).count(),
        nonfinite_b: b.iter().filter(|x| !x.is_finite()).count(),
        zeros_a: a.iter().filter(|x| **x == 0.0).count(),
        zeros_b: b.iter().filter(|x| **x == 0.0).count(),
    }
}

#[cfg(feature = "cuda")]
fn report(tag: &str, d: &LogitsDelta, expect: &str) {
    eprintln!(
        "[CB-006-PARITY] {tag:<9} argmax {:>6} vs {:>6} | cosine {:.6} | max|d| {:.4} \
         | nonfinite {}/{} | zeros {}/{} | expect {expect}",
        d.argmax_a, d.argmax_b, d.cosine, d.max_abs, d.nonfinite_a, d.nonfinite_b,
        d.zeros_a, d.zeros_b
    );
}

#[cfg(feature = "cuda")]
impl OwnedQuantizedModelCuda {
    /// Run `forward_batched_to_logits` and the M=1 oracle on the same post-prefill state.
    ///
    /// Only meaningful at m == 1, where the two paths are computing the same thing. Errors are
    /// reported and swallowed: this is a diagnostic and must never change whether a request
    /// succeeds.
    pub(crate) fn cb006_parity_probe(&mut self, state: &BatchedDecodeState) {
        if !parity_probe_enabled() || state.m != 1 || state.gen_idx != 0 {
            return;
        }
        let (nl, hd, id, vs, eps) = (
            state.num_layers,
            state.hidden_dim as u32,
            state.intermediate_dim as u32,
            state.vocab_size as u32,
            state.eps,
        );
        let position = state.positions[0] as u32;

        // FIXTURE VALIDITY. The first version of this probe ran at the TOP of
        // batched_decode_step, before the embedding is written, so it fed an all-zero vector to
        // both paths. RMSNorm on a zero vector is rsqrt(0) = inf and 0 * inf = NaN, so every
        // layer came back all-NaN and the "oracle divergence" it reported (cosine 0.803758) was
        // a comparison of two different ways of mishandling a degenerate input, not a
        // measurement of the defect.
        //
        // Neither the SELF nor the PERTURBED control could catch that: both pass whether or not
        // the input means anything. An input check is a THIRD kind of control, and it belongs
        // here rather than in the report, because a probe that cannot tell it is being fed
        // garbage will keep producing confident numbers.
        if state.embed_buf.iter().all(|v| *v == 0.0) {
            eprintln!(
                "[CB-006-PARITY] REFUSING TO REPORT: embed_buf is all zeros, so both paths would \
                 be fed a degenerate input. The probe is being called before the embedding is \
                 written. Nothing below would be a measurement of the batched decode."
            );
            return;
        }
        eprintln!(
            "[CB-006-PARITY] m=1 step 0: token {} at position {position}, \
             batched_kv_lengths[0]={:?}",
            state.last_tokens[0],
            self.executor.batched_kv_lengths().first().copied()
        );

        // The dead-slot mask is process-global and is refreshed LATER in batched_decode_step,
        // so a probe that runs before that point inherits the PREVIOUS batch's mask. On the
        // second request that mask is [true], which sets seq_lens[0] = 0, and the attention
        // kernel's online softmax then divides by a sum_exp that never accumulated: 1/0 = inf,
        // 0 * inf = NaN, and all 151936 logits come back non-finite. That was observed, and it
        // was the PROBE's artifact rather than a production fault -- batched_decode_step always
        // refreshes the mask before its own forward. The probe now sets it too, so its reading
        // does not depend on where in the step it is called from.
        //
        // Worth recording even so: seq_len == 0 is undefended in that kernel, so any future
        // caller that runs batched attention without refreshing the mask gets NaN rather than
        // an error.
        self.executor.set_batched_done_mask(&state.done);

        // 1. The batched path, twice — SELF control.
        let batched = match self
            .executor
            .forward_batched_to_logits(&state.embed_buf, &state.pos_buf, nl, hd, id, vs, eps)
        {
            Ok(v) => v,
            Err(e) => {
                eprintln!("[CB-006-PARITY] batched forward failed: {e}");
                return;
            },
        };
        match self
            .executor
            .forward_batched_to_logits(&state.embed_buf, &state.pos_buf, nl, hd, id, vs, eps)
        {
            Ok(again) => report("SELF", &compare_logits(&batched, &again), "IDENTICAL"),
            Err(e) => eprintln!("[CB-006-PARITY] batched re-run failed: {e}"),
        }

        // 1b. SENSITIVITY control: the batched path on a PERTURBED embedding. Must DIVERGE.
        //
        // Across two runs the batched argmax was 15 whether the embedding was the real one or
        // the all-zero buffer of the withdrawn measurement, while the oracle's argmax moved
        // (19415 -> 13552) as an input-dependent function must. If a forward pass does not
        // depend on its input it is not reading it, which is what the allocation-size lead
        // predicts: a read of memory that was never written. SELF cannot see this — it feeds
        // the same input twice and identical output is exactly what it wants.
        let mut sens_in = state.embed_buf.clone();
        for v in sens_in.iter_mut() {
            *v = -*v;
        }
        match self
            .executor
            .forward_batched_to_logits(&sens_in, &state.pos_buf, nl, hd, id, vs, eps)
        {
            Ok(flipped) => report(
                "SENSITIVE",
                &compare_logits(&batched, &flipped),
                "DIVERGE(else the batched forward ignores its input)",
            ),
            Err(e) => eprintln!("[CB-006-PARITY] sensitivity forward failed: {e}"),
        }

        // 2. The M=1 oracle, on the single KV cache the same prefill populated. Stride is
        //    zeroed around the call so the M=1 path does not take the batched attention branch,
        //    exactly as add_slot_to_batch does.
        let mut oracle = vec![0.0f32; state.vocab_size];
        let saved = self.executor.batched_kv_stride;
        self.executor.batched_kv_stride = 0;
        let ok = self.executor.forward_all_layers_gpu_to_logits(
            &state.embed_buf, &mut oracle, position, nl, hd, id, vs, eps,
        );
        self.executor.batched_kv_stride = saved;
        match ok {
            Ok(()) => report("ORACLE", &compare_logits(&batched, &oracle), "IDENTICAL"),
            Err(e) => eprintln!("[CB-006-PARITY] oracle forward failed: {e}"),
        }

        // 3. PERTURBED control: the same oracle on a deliberately corrupted embedding. If this
        //    does not diverge, the comparison above is not measuring what it claims to.
        let mut perturbed_in = state.embed_buf.clone();
        perturbed_in[0] += 10.0;
        let mut perturbed = vec![0.0f32; state.vocab_size];
        let saved = self.executor.batched_kv_stride;
        self.executor.batched_kv_stride = 0;
        let ok = self.executor.forward_all_layers_gpu_to_logits(
            &perturbed_in, &mut perturbed, position, nl, hd, id, vs, eps,
        );
        self.executor.batched_kv_stride = saved;
        match ok {
            Ok(()) => report("PERTURBED", &compare_logits(&batched, &perturbed), "DIVERGE"),
            Err(e) => eprintln!("[CB-006-PARITY] perturbed forward failed: {e}"),
        }

        // Leave the batch exactly as it was found.
        if let Err(e) = self.executor.init_batched_workspace(
            state.hidden_dim,
            state.intermediate_dim,
            state.m,
        ) {
            eprintln!("[CB-006-PARITY] workspace restore failed: {e}");
        }
        self.executor.clear_decode_graph();
    }
}
