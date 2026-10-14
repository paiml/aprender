//! Q6_K super-block layout against GGML's `dequantize_row_q6_K` (ggml-quants.c), transcribed here as
//! the loop it is: the scalar, AVX2, parallel and column-major kernels must read every element where
//! GGML writes it. A linear `ql[idx/2]`/`qh[idx/4]` packing passed the old scalar==dispatch goldens,
//! because both sides shared it, and disagreed with realizar's production Q6_K kernel by up to 9×.

use super::super::*;

/// GGML's dequantization of one 210-byte super-block, structure kept: two halves of 128, each with
/// four 32-wide lanes of `ql`/`qh` bits and scales `sc[is + 0/2/4/6]`.
fn ggml_dequant(sb: &[u8]) -> [f32; 256] {
    let d = f16_to_f32(u16::from_le_bytes([sb[208], sb[209]]));
    let mut y = [0.0f32; 256];
    for n in 0..2 {
        let (ql, qh, sc) = (&sb[64 * n..], &sb[128 + 32 * n..], &sb[192 + 8 * n..]);
        for l in 0..32 {
            let is = l / 16;
            let q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i8 - 32;
            let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i8 - 32;
            let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i8 - 32;
            let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i8 - 32;
            let y = &mut y[128 * n..];
            y[l] = d * f32::from(sc[is] as i8) * f32::from(q1);
            y[l + 32] = d * f32::from(sc[is + 2] as i8) * f32::from(q2);
            y[l + 64] = d * f32::from(sc[is + 4] as i8) * f32::from(q3);
            y[l + 96] = d * f32::from(sc[is + 6] as i8) * f32::from(q4);
        }
    }
    y
}

/// `rows` pseudo-random super-blocks (one per row) with a small finite `d`.
fn blocks(rows: usize, seed: u64) -> Vec<u8> {
    let mut s = seed;
    let mut v = Vec::with_capacity(rows * SUPER_BLOCK_BYTES);
    for _ in 0..rows {
        for i in 0..SUPER_BLOCK_BYTES {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            v.push(match i {
                208 => 0x66,
                209 => 0x2E,
                _ => (s >> 33) as u8,
            });
        }
    }
    v
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn close(got: f32, want: f32) -> bool {
    (got - want).abs() <= 1e-3 * want.abs().max(1.0)
}

/// Every element, one-hot: the scalar kernel returns exactly GGML's value for each index.
#[test]
fn scalar_reads_every_element_where_ggml_writes_it() {
    for seed in [1, 7, 42] {
        let w = blocks(1, seed);
        let want = ggml_dequant(&w);
        for i in 0..256 {
            let mut e = vec![0.0f32; 256];
            e[i] = 1.0;
            let got = matmul_q6k_f32_scalar(&w, &e, 1, 256)[0];
            assert_eq!(got, want[i], "seed {seed}: element {i}");
        }
    }
}

/// The SIMD dispatch and the parallel path (≥ 8M ops) agree with GGML, not only with the scalar.
#[test]
fn dispatch_and_parallel_match_ggml() {
    for (rows, seed) in [(8usize, 3u64), (32_768, 5)] {
        let w = blocks(rows, seed);
        let x: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.019).sin() * 0.4).collect();
        let got = matmul_q6k_f32_dispatch(&w, &x, rows, 256);
        for r in (0..rows).step_by((rows / 8).max(1)) {
            let want = dot(&ggml_dequant(&w[r * SUPER_BLOCK_BYTES..]), &x);
            assert!(close(got[r], want), "rows {rows} row {r}: {} vs ggml {want}", got[r]);
        }
    }
}

/// The deprecated column-major twin reads the super-block the same way: one column, x = 1.
#[test]
#[allow(deprecated)]
fn colmajor_reads_the_ggml_layout() {
    let w = blocks(1, 11);
    let got = matmul_q6k_f32_colmajor(&w, &[1.0], 256, 1);
    assert_eq!(got[..], ggml_dequant(&w)[..]);
}
