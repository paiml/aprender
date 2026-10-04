//! The layout the CUDA NF4 block keeps a LoRA adapter in
//! (contract `cuda-nf4-train-loss-parity-v1`, FALSIFY-CUDA-NF4-TRAIN-LOSS-PARITY-003).
//!
//! `LoRALayer` holds A as `[rank, d_in]` and B as `[d_out, rank]`, the PEFT layout, and the
//! CPU forward computes the delta as `x·Aᵀ·Bᵀ`. The NF4 block computes it as `(x·A)·B` with
//! `gemm_forward` (`C = A @ B`, A `[M, K]`, B `[K, N]`, row-major), so the device needs
//! `Aᵀ` (`[d_in, rank]`) and `Bᵀ` (`[rank, d_out]`). A buffer copied across unchanged is
//! reshaped, not transposed: the device then trains one adapter while evaluation,
//! checkpoints, merge and PEFT export read another.

use super::LoRALayer;
use crate::autograd::transpose;
use crate::Tensor;

impl LoRALayer {
    /// A and B in the CUDA NF4 block's layout: `(Aᵀ [d_in, rank], Bᵀ [rank, d_out])`, row-major.
    pub fn device_layout(&self) -> (Vec<f32>, Vec<f32>) {
        let a = self.lora_a().data().to_vec();
        let b = self.lora_b().data().to_vec();
        (transpose(&a, self.rank(), self.d_in()), transpose(&b, self.d_out(), self.rank()))
    }

    /// Set A and B from buffers in the CUDA NF4 block's layout, the inverse of
    /// [`LoRALayer::device_layout`]. Both stay trainable, as [`LoRALayer::new`] makes them.
    ///
    /// # Panics
    /// If a buffer's length is not its matrix's.
    pub fn set_from_device_layout(&mut self, a_t: &[f32], b_t: &[f32]) {
        let (rank, d_in, d_out) = (self.rank(), self.d_in(), self.d_out());
        assert_eq!(a_t.len(), rank * d_in, "device-layout A must be [d_in {d_in}, rank {rank}]");
        assert_eq!(b_t.len(), rank * d_out, "device-layout B must be [rank {rank}, d_out {d_out}]");
        *self.lora_a_mut() = Tensor::from_vec(transpose(a_t, d_in, rank), true);
        *self.lora_b_mut() = Tensor::from_vec(transpose(b_t, rank, d_out), true);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::matmul_nt;

    // Non-square in every pair, so a transpose with its dimensions swapped is a different map.
    const D_OUT: usize = 8;
    const D_IN: usize = 12;
    const RANK: usize = 3;

    /// A layer whose A and B entries are all distinct, so any wrong index map shows.
    fn layer(offset: f32) -> LoRALayer {
        let mut l = LoRALayer::new(Tensor::zeros(D_OUT * D_IN, false), D_OUT, D_IN, RANK, 6.0);
        let a = (0..RANK * D_IN).map(|i| offset + 0.01 * (i as f32 + 1.0)).collect();
        let b = (0..D_OUT * RANK).map(|i| offset - 0.02 * (i as f32 + 1.0)).collect();
        *l.lora_a_mut() = Tensor::from_vec(a, true);
        *l.lora_b_mut() = Tensor::from_vec(b, true);
        l
    }

    /// `gemm_forward`'s documented contract: `C = A @ B`, A `[m, k]`, B `[k, n]`, row-major.
    fn gemm(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for p in 0..k {
                for j in 0..n {
                    c[i * n + j] += a[i * k + p] * b[p * n + j];
                }
            }
        }
        c
    }

    fn rel_l2(x: &[f32], reference: &[f32]) -> f32 {
        let diff: f32 = x.iter().zip(reference).map(|(a, b)| (a - b) * (a - b)).sum();
        let norm: f32 = reference.iter().map(|b| b * b).sum();
        (diff / norm).sqrt()
    }

    #[test]
    fn falsify_cuda_nf4_train_loss_parity_003_device_layout_is_the_transpose() {
        let l = layer(0.0);
        let (a, b) = (l.lora_a().data().to_vec(), l.lora_b().data().to_vec());
        let (a_t, b_t) = l.device_layout();
        assert_eq!((a_t.len(), b_t.len()), (RANK * D_IN, D_OUT * RANK));
        for k in 0..RANK {
            for j in 0..D_IN {
                assert_eq!(a_t[j * RANK + k], a[k * D_IN + j], "Aᵀ[{j}, {k}]");
            }
            for o in 0..D_OUT {
                assert_eq!(b_t[k * D_OUT + o], b[o * RANK + k], "Bᵀ[{k}, {o}]");
            }
        }
    }

    #[test]
    fn falsify_cuda_nf4_train_loss_parity_003_device_gemm_computes_the_cpu_adapter() {
        const SEQ: usize = 5;
        let l = layer(0.0);
        let x: Vec<f32> = (0..SEQ * D_IN).map(|i| ((i * 7 % 11) as f32 - 5.0) * 0.1).collect();
        let x_t = Tensor::from_vec(x.clone(), false);
        let mid = matmul_nt(&x_t, l.lora_a(), SEQ, D_IN, RANK);
        let cpu = matmul_nt(&mid, l.lora_b(), SEQ, RANK, D_OUT).data().to_vec();
        assert!(cpu.iter().any(|v| v.abs() > 1e-2), "the CPU delta must be nonzero");

        let (a_t, b_t) = l.device_layout();
        let device = gemm(&gemm(&x, &a_t, SEQ, D_IN, RANK), &b_t, SEQ, RANK, D_OUT);
        let err = rel_l2(&device, &cpu);
        assert!(err < 1e-5, "device-layout GEMMs vs CPU x·Aᵀ·Bᵀ: rel L2 {err}");

        // Power: the PEFT buffers copied raw compute another adapter.
        let (a, b) = (l.lora_a().data().to_vec(), l.lora_b().data().to_vec());
        let raw = gemm(&gemm(&x, &a, SEQ, D_IN, RANK), &b, SEQ, RANK, D_OUT);
        let raw_err = rel_l2(&raw, &cpu);
        assert!(raw_err > 0.1, "raw buffers must not match the CPU adapter: rel L2 {raw_err}");
    }

    #[test]
    fn falsify_cuda_nf4_train_loss_parity_003_set_from_device_layout_inverts() {
        let src = layer(0.0);
        let (a_t, b_t) = src.device_layout();
        let mut dst = layer(1.0);
        assert_ne!(dst.lora_a().data().to_vec(), src.lora_a().data().to_vec());
        assert_ne!(dst.lora_b().data().to_vec(), src.lora_b().data().to_vec());

        dst.set_from_device_layout(&a_t, &b_t);
        assert_eq!(dst.lora_a().data().to_vec(), src.lora_a().data().to_vec());
        assert_eq!(dst.lora_b().data().to_vec(), src.lora_b().data().to_vec());
        assert!(dst.lora_a().requires_grad() && dst.lora_b().requires_grad());
    }

    // Each buffer is one entry too long: a transpose alone would drop the extra entry silently.
    #[test]
    #[should_panic(expected = "device-layout A")]
    fn falsify_cuda_nf4_train_loss_parity_003_wrong_length_a_panics() {
        let mut l = layer(0.0);
        let (mut a_t, b_t) = l.device_layout();
        a_t.push(0.5);
        l.set_from_device_layout(&a_t, &b_t);
    }

    #[test]
    #[should_panic(expected = "device-layout B")]
    fn falsify_cuda_nf4_train_loss_parity_003_wrong_length_b_panics() {
        let mut l = layer(0.0);
        let (a_t, mut b_t) = l.device_layout();
        b_t.push(0.5);
        l.set_from_device_layout(&a_t, &b_t);
    }
}
