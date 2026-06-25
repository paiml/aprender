//! End-to-end LoRA training proof (PMAT-931).
//!
//! This is the P3 / Unsloth-pillar capability proof, extending the PMAT-921
//! end-to-end-training-proof methodology (a real model trained to a decreasing
//! loss) from a full transformer to the LoRA fine-tuning path.
//!
//! The existing LoRA gradient tests (`gradient_tests.rs`) only check the *static*
//! `requires_grad` flags and *manually-injected* gradients — they never run a
//! real backward pass through `LoRALayer::forward`. A composition of
//! individually-correct ops can still SEVER the autograd graph on the
//! integration path (exactly the PMAT-921/922 graph-severing class), freezing
//! the adapter while every per-flag test stays green.
//!
//! This module trains a tiny LoRA adapter on a frozen base weight for a few
//! deterministic steps and asserts the LoRA invariant directly:
//!
//!   (a) the loss DECREASES substantially, and
//!   (b) the LoRA adapter params A and B genuinely UPDATE from init AND received
//!       a finite non-zero gradient, while
//!   (c) the BASE weight stays EXACTLY FROZEN.
//!
//! Falsifier (OBLIG-LORA-ADAPTER-TRAINS-BASE-FROZEN):
//!   - RED if the adapter does not train (graph severed / frozen adapter), OR
//!   - RED if the base weight moves (frozen-base violated).
//!   - GREEN only on correct LoRA training.
//!
//! Everything is LCG-seeded / closed-form deterministic for CI stability and
//! runs as a fast per-PR CPU test (tiny dims, bounded steps).

#[cfg(test)]
mod tests {
    use crate::autograd::backward;
    use crate::autograd::ops::{add, mul, scale, sum};
    use crate::lora::LoRALayer;
    use crate::Tensor;
    use ndarray::Array1;

    /// Tiny deterministic LCG so the test is seeded and CI-stable (no `rand`
    /// dependence, no clock, fully reproducible).
    struct Lcg(u64);
    impl Lcg {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        /// Next f32 in roughly [-0.5, 0.5].
        fn next_f32(&mut self) -> f32 {
            // Numerical Recipes LCG constants.
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let bits = (self.0 >> 33) as u32; // top 31 bits
            (bits as f32 / u32::MAX as f32) - 0.5
        }
    }

    /// Squared-error loss `Σ (y - target)^2` built entirely from autograd-aware
    /// ops so the backward pass flows from the scalar loss all the way back into
    /// the LoRA adapter. `target` is a constant (no grad).
    fn squared_error(y: &Tensor, target: &Tensor) -> Tensor {
        // diff = y + (-1)*target   (target is a no-grad constant)
        let neg_target = scale(target, -1.0);
        let diff = add(y, &neg_target);
        let sq = mul(&diff, &diff);
        sum(&sq)
    }

    /// PMAT-931 / OBLIG-LORA-ADAPTER-TRAINS-BASE-FROZEN.
    ///
    /// Trains a tiny LoRA adapter on a fixed deterministic regression target and
    /// asserts the full LoRA invariant: adapter trains (loss decreases, A and B
    /// move from init with finite non-zero gradients) while the base stays frozen.
    #[test]
    fn falsify_lora_adapter_trains_to_decreasing_loss_base_frozen() {
        // ---- Tiny deterministic problem -----------------------------------
        let d_out = 4;
        let d_in = 4;
        let rank = 2;
        // alpha == rank → scale = alpha/rank = 1.0. A larger scale amplifies the
        // effective adapter learning rate (it multiplies both the forward LoRA
        // branch and its backward gradients), which makes a fixed lr diverge; a
        // unit scale keeps this tiny CPU test numerically stable.
        let alpha = 2.0;
        let steps = 400;
        let lr = 0.05;

        let mut rng = Lcg::new(0xC0FF_EE13);

        // Frozen base weight W (d_out x d_in). Some structure so base@x != 0.
        let base_data: Vec<f32> = (0..d_out * d_in).map(|_| rng.next_f32()).collect();
        let base_snapshot = base_data.clone();
        let base = Tensor::from_vec(base_data, false);

        let mut lora = LoRALayer::new(base, d_out, d_in, rank, alpha);

        // Initialise B with small non-zero values so the LoRA branch is alive
        // from step 0 (canonical LoRA inits B=0, which would make the FIRST
        // gradient-to-A vanish; we want a tiny live signal both directions).
        let a_init: Vec<f32> = (0..rank * d_in).map(|_| 0.10 * rng.next_f32()).collect();
        let b_init: Vec<f32> = (0..d_out * rank).map(|_| 0.10 * rng.next_f32()).collect();
        *lora.lora_a_mut().data_mut() = Array1::from(a_init.clone());
        *lora.lora_b_mut().data_mut() = Array1::from(b_init.clone());

        // Fixed input and a target the base alone cannot reach, so the adapter
        // MUST move to reduce the loss.
        let x = Tensor::from_vec(vec![0.5, -0.3, 0.8, 0.2], false);
        let target = Tensor::from_vec(vec![1.0, -1.0, 0.5, -0.5], false);

        // ---- Snapshots / accumulators -------------------------------------
        let initial_loss = {
            let y = lora.forward(&x);
            squared_error(&y, &target).data()[0]
        };

        // Track whether A and B ever received a finite NON-ZERO gradient during
        // training. If the forward graph is severed, these stay false → RED.
        let mut a_saw_nonzero_grad = false;
        let mut b_saw_nonzero_grad = false;
        let mut last_loss = initial_loss;

        for _ in 0..steps {
            // Fresh gradients each step.
            lora.lora_a().zero_grad();
            lora.lora_b().zero_grad();

            // Forward + scalar loss.
            let y = lora.forward(&x);
            let mut loss = squared_error(&y, &target);
            last_loss = loss.data()[0];

            // Backward from the scalar loss through the whole LoRA graph.
            backward(&mut loss, None);

            // The adapter MUST have received gradients here. Record finiteness
            // and non-zero magnitude (the falsifier signal).
            let ga = lora.lora_a().grad().expect(
                "OBLIG-LORA-ADAPTER-TRAINS: lora_a received NO gradient — \
                 LoRA forward severed the autograd graph (frozen adapter)",
            );
            let gb = lora.lora_b().grad().expect(
                "OBLIG-LORA-ADAPTER-TRAINS: lora_b received NO gradient — \
                 LoRA forward severed the autograd graph (frozen adapter)",
            );
            assert!(ga.iter().all(|v| v.is_finite()), "grad_A must be finite");
            assert!(gb.iter().all(|v| v.is_finite()), "grad_B must be finite");
            if ga.iter().any(|&v| v.abs() > 1e-9) {
                a_saw_nonzero_grad = true;
            }
            if gb.iter().any(|&v| v.abs() > 1e-9) {
                b_saw_nonzero_grad = true;
            }

            // Manual SGD step on the trainable adapter ONLY (A and B). The base
            // is deliberately never touched — that is the frozen-base invariant.
            let ga = lora.lora_a().grad().expect("grad_A present");
            let gb = lora.lora_b().grad().expect("grad_B present");
            {
                let a_data = lora.lora_a_mut().data_mut();
                *a_data = &*a_data - &(&ga * lr);
            }
            {
                let b_data = lora.lora_b_mut().data_mut();
                *b_data = &*b_data - &(&gb * lr);
            }
        }

        let final_loss = last_loss;

        // ---- (a) loss decreased substantially -----------------------------
        assert!(
            final_loss < initial_loss * 0.5,
            "OBLIG-LORA-ADAPTER-TRAINS: loss did not decrease substantially \
             (initial={initial_loss:.6}, final={final_loss:.6}); the adapter is \
             not training through LoRALayer::forward"
        );

        // ---- (b) the adapter genuinely UPDATED + got real gradients -------
        assert!(
            a_saw_nonzero_grad,
            "OBLIG-LORA-ADAPTER-TRAINS: lora_a never received a non-zero gradient \
             (graph severed / adapter frozen)"
        );
        assert!(
            b_saw_nonzero_grad,
            "OBLIG-LORA-ADAPTER-TRAINS: lora_b never received a non-zero gradient \
             (graph severed / adapter frozen)"
        );

        let a_final = lora.lora_a().data().to_vec();
        let b_final = lora.lora_b().data().to_vec();
        let a_moved: f32 = a_final.iter().zip(&a_init).map(|(f, i)| (f - i).abs()).sum();
        let b_moved: f32 = b_final.iter().zip(&b_init).map(|(f, i)| (f - i).abs()).sum();
        assert!(
            a_moved > 1e-4,
            "OBLIG-LORA-ADAPTER-TRAINS: lora_a did not move from init (Σ|Δ|={a_moved:.2e})"
        );
        assert!(
            b_moved > 1e-4,
            "OBLIG-LORA-ADAPTER-TRAINS: lora_b did not move from init (Σ|Δ|={b_moved:.2e})"
        );

        // ---- (c) the BASE weight stayed EXACTLY frozen --------------------
        let base_final = lora.base_weight().data().to_vec();
        assert_eq!(base_final.len(), base_snapshot.len(), "base weight length changed");
        for (i, (got, exp)) in base_final.iter().zip(&base_snapshot).enumerate() {
            assert!(
                (got - exp).abs() < 1e-12,
                "OBLIG-LORA-BASE-FROZEN: base weight[{i}] moved during training \
                 (got={got}, expected frozen {exp}); LoRA must NEVER update the base"
            );
        }
        // Base must also remain ungraded — it never participates as a trainable.
        assert!(
            !lora.base_weight().requires_grad(),
            "OBLIG-LORA-BASE-FROZEN: base weight must stay requires_grad=false"
        );
    }

    /// Single-step structural guard for the same obligation, kept minimal so a
    /// mutation that re-severs `LoRALayer::forward` is caught fast: one forward,
    /// one backward, the adapter must receive gradient and the base must not.
    #[test]
    fn lora_forward_backward_reaches_adapter_not_base() {
        let base = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], false);
        let mut lora = LoRALayer::new(base, 2, 2, 2, 4.0);
        *lora.lora_a_mut().data_mut() = Array1::from(vec![0.1, 0.2, 0.3, 0.4]);
        *lora.lora_b_mut().data_mut() = Array1::from(vec![0.5, 0.3, 0.2, 0.7]);

        let x = Tensor::from_vec(vec![1.0, 1.0], false);
        let y = lora.forward(&x);

        // The forward output MUST keep a live autograd graph.
        assert!(y.requires_grad(), "LoRA forward output severed requires_grad");
        assert!(y.backward_op().is_some(), "LoRA forward output has no backward op");

        let mut loss = sum(&y);
        backward(&mut loss, None);

        assert!(lora.lora_a().grad().is_some(), "adapter A got no gradient");
        assert!(lora.lora_b().grad().is_some(), "adapter B got no gradient");
        // Base is frozen: it never accumulates a gradient.
        assert!(lora.base_weight().grad().is_none(), "frozen base must not accumulate gradient");
    }
}
