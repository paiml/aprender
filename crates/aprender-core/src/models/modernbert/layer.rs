//! The ModernBERT layer and its reusable primitives: `layer_norm`, `gelu_exact`,
//! `rope_rotate_half`, `attention`.
//!
//! These are public because aprender-decide's Laya head and scorer reuse them (one
//! implementation per operation, OPS-03) rather than copying them. Every function
//! validates its buffer lengths and returns a typed [`ModernBertError`] instead of
//! panicking on an inconsistent call. Numerics and their order are spike 025's.

use super::{check_len, Linear, ModernBertConfig, ModernBertError};
use rayon::prelude::*;

/// Row-wise LayerNorm over `[rows, d]` with f64 mean/variance accumulation.
///
/// `eps` is the caller's configured epsilon (`norm_eps` for the encoder). It is
/// rounded to f32 before use, exactly as torch's CPU LayerNorm casts its double
/// `eps` to the f32 accumulation type, so `1e-5` here is torch's `1e-5f`.
///
/// # Errors
///
/// [`ModernBertError::InputShape`] when `d == 0`, `x` is not a whole number of rows,
/// or `w` / `b` are not `[d]`.
pub fn layer_norm(
    x: &[f32],
    d: usize,
    w: &[f32],
    b: Option<&[f32]>,
    eps: f64,
) -> Result<Vec<f32>, ModernBertError> {
    if d == 0 || x.len() % d != 0 {
        return Err(ModernBertError::InputShape {
            what: "layer_norm.x",
            expected: None,
            observed: x.len(),
        });
    }
    check_len("layer_norm.w", w.len(), &[d])?;
    if let Some(b) = b {
        check_len("layer_norm.b", b.len(), &[d])?;
    }
    let eps = f64::from(eps as f32);
    let mut y = vec![0.0f32; x.len()];
    y.par_chunks_mut(d).zip(x.par_chunks(d)).for_each(|(o, r)| {
        let mean = r.iter().map(|&v| f64::from(v)).sum::<f64>() / d as f64;
        let var = r
            .iter()
            .map(|&v| (f64::from(v) - mean).powi(2))
            .sum::<f64>()
            / d as f64;
        let inv = 1.0 / (var + eps).sqrt();
        for (i, (oi, &ri)) in o.iter_mut().zip(r).enumerate() {
            let v = ((f64::from(ri) - mean) * inv) as f32 * w[i];
            *oi = b.map_or(v, |b| v + b[i]);
        }
    });
    Ok(y)
}

/// Exact (erf) GELU: `0.5 * x * (1 + erf(x / sqrt 2))`, evaluated in f64 as
/// `0.5 * x * erfc(-x / sqrt 2)` through the house `erfc_precise` (the erfc form
/// avoids the negative-tail cancellation).
pub fn gelu_exact(x: f32) -> f32 {
    let x = f64::from(x);
    (0.5 * x * batuta_common::math::erfc_precise(-x / std::f64::consts::SQRT_2)) as f32
}

/// RoPE `inv_freq[p]` for `p < hd / 2`, bit for bit as transformers computes it in f32:
/// `1.0 / (theta ** (arange(0, hd, 2).float() / hd))` — the power rounded to f32 FIRST,
/// then an f32 reciprocal. Rounding `1 / theta^(2p / hd)` once from f64 is off by one ULP
/// on about a quarter of the frequencies, and `pos * inv_freq` carries that into every
/// sin / cos of the row (debug session laya-rescore-drift).
pub(crate) fn rope_inv_freq(theta: f64, hd: usize) -> Vec<f32> {
    (0..hd / 2)
        .map(|p| {
            let e = (2 * p) as f32 / hd as f32;
            let pw = f64::from(theta as f32).powf(f64::from(e)) as f32;
            1.0f32 / pw
        })
        .collect()
}

/// The `[l, hd / 2]` sin / cos table of one RoPE theta: `angle = pos * inv_freq` in f32,
/// then `sin_cos` — the exact values a per-head evaluation computes, built once.
#[derive(Debug, Clone)]
pub(crate) struct RopeTable {
    half: usize,
    sin: Vec<f32>,
    cos: Vec<f32>,
}

#[cfg(test)]
thread_local! {
    /// How many RoPE tables this thread has started to build — lets a test prove a
    /// guard ran BEFORE `RopeTable::new` sized anything from a caller's `l`.
    static ROPE_TABLES_STARTED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Test hook: RoPE tables this thread has started to build.
#[cfg(test)]
pub(crate) fn rope_tables_started() -> usize {
    ROPE_TABLES_STARTED.with(std::cell::Cell::get)
}

impl RopeTable {
    pub(crate) fn new(inv_freq: &[f32], l: usize) -> Self {
        #[cfg(test)]
        ROPE_TABLES_STARTED.with(|c| c.set(c.get() + 1));
        let half = inv_freq.len();
        let (mut sin, mut cos) = (Vec::with_capacity(l * half), Vec::with_capacity(l * half));
        for pos in 0..l {
            for &f in inv_freq {
                let (s, c) = (pos as f32 * f).sin_cos();
                sin.push(s);
                cos.push(c);
            }
        }
        Self { half, sin, cos }
    }

    /// Rotate every `2 * half`-wide head of `row` (position `pos`) in place:
    /// `(x_p, x_{p+hd/2}) -> (x_p cos - x_{p+hd/2} sin, x_{p+hd/2} cos + x_p sin)`.
    pub(crate) fn rotate_heads(&self, row: &mut [f32], pos: usize) {
        let half = self.half;
        let s = &self.sin[pos * half..(pos + 1) * half];
        let c = &self.cos[pos * half..(pos + 1) * half];
        for v in row.chunks_exact_mut(2 * half) {
            for p in 0..half {
                let (a, b) = (v[p], v[p + half]);
                v[p] = a * c[p] - b * s[p];
                v[p + half] = b * c[p] + a * s[p];
            }
        }
    }
}

/// Rotate-half RoPE in place on `[l, heads * hd]`, position = row index.
///
/// torch order: `inv_freq = 1 / f32(theta^(2p / hd))` in f32 ([`rope_inv_freq`]),
/// `angle = pos * inv_freq` in f32,
/// `(x_p, x_{p+hd/2}) -> (x_p cos - x_{p+hd/2} sin, x_{p+hd/2} cos + x_p sin)`.
///
/// # Errors
///
/// [`ModernBertError::InputShape`] when `hd` is 0 or odd, or `x` is not `[l, heads * hd]`.
pub fn rope_rotate_half(
    x: &mut [f32],
    l: usize,
    heads: usize,
    hd: usize,
    theta: f64,
) -> Result<(), ModernBertError> {
    check_rope_geometry(heads, hd)?;
    check_len("rope.x", x.len(), &[l, heads, hd])?;
    let table = RopeTable::new(&rope_inv_freq(theta, hd), l);
    x.par_chunks_mut(heads * hd)
        .enumerate()
        .for_each(|(pos, row)| table.rotate_heads(row, pos));
    Ok(())
}

fn check_rope_geometry(heads: usize, hd: usize) -> Result<(), ModernBertError> {
    if hd == 0 || hd % 2 != 0 || heads == 0 {
        return Err(ModernBertError::InputShape {
            what: "rope.head_dim",
            expected: None,
            observed: hd,
        });
    }
    Ok(())
}

/// Key range `[lo, hi)` row `i` attends to: all `l` keys, or `|i - j| <= window`.
fn key_range(i: usize, l: usize, window: Option<usize>) -> (usize, usize) {
    match window {
        Some(w) => (
            i.saturating_sub(w),
            i.saturating_add(w).saturating_add(1).min(l),
        ),
        None => (0, l),
    }
}

/// Bidirectional multi-head scaled-dot-product attention over one row.
///
/// `q`, `k`, `v` are `[l, heads * hd]`; `window = Some(w)` keeps `|i - j| <= w`
/// (inclusive), `None` is global. Returns `[l, heads * hd]`.
///
/// # Errors
///
/// [`ModernBertError::InputShape`] when `heads` or `hd` is 0 or any operand is not
/// `[l, heads * hd]`.
pub fn attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    l: usize,
    heads: usize,
    hd: usize,
    window: Option<usize>,
) -> Result<Vec<f32>, ModernBertError> {
    if heads == 0 || hd == 0 {
        return Err(ModernBertError::InputShape {
            what: "attention.heads_x_hd",
            expected: None,
            observed: heads * hd,
        });
    }
    check_len("attention.q", q.len(), &[l, heads, hd])?;
    check_len("attention.k", k.len(), &[l, heads, hd])?;
    check_len("attention.v", v.len(), &[l, heads, hd])?;
    Ok(attention_strided(q, k, v, heads * hd, l, heads, hd, window))
}

/// [`attention`] over operands whose row `i` starts at `i * stride` (`stride >= heads *
/// hd`), so q / k / v can be read in place from a fused `[l, 3d]` projection. Callers
/// have checked every operand holds `(l - 1) * stride + heads * hd` values.
#[allow(clippy::too_many_arguments)]
fn attention_strided(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    stride: usize,
    l: usize,
    heads: usize,
    hd: usize,
    window: Option<usize>,
) -> Vec<f32> {
    let d = heads * hd;
    let scale = 1.0 / (hd as f32).sqrt();
    let mut out = vec![0.0f32; l * d];
    out.par_chunks_mut(d).enumerate().for_each(|(i, o)| {
        let (lo, hi) = key_range(i, l, window);
        let mut s = vec![0.0f32; hi - lo];
        for h in 0..heads {
            let qi = &q[i * stride + h * hd..i * stride + (h + 1) * hd];
            let mut mx = f32::NEG_INFINITY;
            for (jj, sv) in s.iter_mut().enumerate() {
                let j = lo + jj;
                let kj = &k[j * stride + h * hd..j * stride + (h + 1) * hd];
                *sv = qi.iter().zip(kj).map(|(a, b)| a * b).sum::<f32>() * scale;
                mx = mx.max(*sv);
            }
            let mut z = 0.0f32;
            for sv in &mut s {
                *sv = (*sv - mx).exp();
                z += *sv;
            }
            let oh = &mut o[h * hd..(h + 1) * hd];
            for (jj, &w) in s.iter().enumerate() {
                let j = lo + jj;
                let vj = &v[j * stride + h * hd..j * stride + (h + 1) * hd];
                let w = w / z;
                oh.iter_mut().zip(vj).for_each(|(a, b)| *a += w * b);
            }
        }
    });
    out
}

/// One ModernBERT encoder layer: pre-norm attention (layer 0 has no `attn_norm`) and
/// a GeGLU MLP, both residual. Constructed only by the loader, so its weights agree
/// with the config.
#[derive(Debug, Clone)]
pub struct ModernBertLayer {
    attn_norm: Option<Vec<f32>>,
    wqkv: Linear,
    wo: Linear,
    mlp_norm: Vec<f32>,
    wi: Linear,
    wo_mlp: Linear,
    global: bool,
}

impl ModernBertLayer {
    pub(crate) fn from_parts(
        attn_norm: Option<Vec<f32>>,
        wqkv: Linear,
        wo: Linear,
        mlp_norm: Vec<f32>,
        wi: Linear,
        wo_mlp: Linear,
        global: bool,
    ) -> Self {
        Self {
            attn_norm,
            wqkv,
            wo,
            mlp_norm,
            wi,
            wo_mlp,
            global,
        }
    }

    /// Whether this layer attends globally (else within the local window).
    pub fn is_global(&self) -> bool {
        self.global
    }

    /// Apply the layer in place to `x` (`[l, d]`); `window` is the local half-window
    /// (ignored on a global layer).
    ///
    /// `l` and `x` are caller-controlled, so both are checked BEFORE the RoPE table is
    /// sized from `l` (WR-04 / V10-b, plan 08-20): an empty row or an `l` that disagrees
    /// with `x` is a typed error, never a slice panic, an overflow or an allocation
    /// proportional to a claimed length.
    ///
    /// # Errors
    ///
    /// [`ModernBertError::EmptyInput`] when `l == 0`; [`ModernBertError::InputShape`]
    /// when `x` is not `[l, d]` (including an `l * d` that overflows); any other
    /// [`ModernBertError`] from the primitives (shape or GEMM).
    pub fn forward(
        &self,
        x: &mut [f32],
        l: usize,
        config: &ModernBertConfig,
        window: usize,
    ) -> Result<(), ModernBertError> {
        if l == 0 {
            return Err(ModernBertError::EmptyInput);
        }
        check_len("layer.x", x.len(), &[l, config.hidden_size()])?;
        let hd = config.head_dim();
        check_rope_geometry(config.num_attention_heads(), hd)?;
        let theta = if self.global {
            config.rope_theta_global()
        } else {
            config.rope_theta_local()
        };
        let rope = RopeTable::new(&rope_inv_freq(theta, hd), l);
        self.forward_with_rope(x, l, config, window, &rope)
    }

    /// [`Self::forward`] with this layer's RoPE table (its theta, `l` positions) built
    /// by the caller, so an encoder builds each theta's table once per row.
    pub(crate) fn forward_with_rope(
        &self,
        x: &mut [f32],
        l: usize,
        config: &ModernBertConfig,
        window: usize,
        rope: &RopeTable,
    ) -> Result<(), ModernBertError> {
        let d = config.hidden_size();
        let heads = config.num_attention_heads();
        let hd = config.head_dim();
        let eps = config.norm_eps();
        // With l == 0 every length check below passes (0 == 0) and `&qkv[d..]` panics.
        if l == 0 {
            return Err(ModernBertError::EmptyInput);
        }
        check_len("layer.x", x.len(), &[l, d])?;
        check_rope_geometry(heads, hd)?;
        check_len("rope.x", heads * hd, &[d])?;
        let normed;
        let xn: &[f32] = match &self.attn_norm {
            Some(w) => {
                normed = layer_norm(x, d, w, None, eps)?;
                &normed
            }
            None => x,
        };
        let mut qkv = self.wqkv.forward(xn, l)?;
        check_len("layer.qkv", qkv.len(), &[l, 3, d])?;
        // q = qkv[.., 0..d], k = qkv[.., d..2d], v = qkv[.., 2d..3d]: RoPE on q and k in
        // place (they are adjacent, so one pass over each row's first 2d values), then
        // attention reads all three with row stride 3d — no split copies.
        qkv.par_chunks_mut(3 * d)
            .enumerate()
            .for_each(|(pos, row)| rope.rotate_heads(&mut row[..2 * d], pos));
        let a = attention_strided(
            &qkv,
            &qkv[d..],
            &qkv[2 * d..],
            3 * d,
            l,
            heads,
            hd,
            if self.global { None } else { Some(window) },
        );
        let a = self.wo.forward(&a, l)?;
        check_len("layer.attn_out", a.len(), &[l, d])?;
        x.par_iter_mut()
            .zip(a.par_iter())
            .for_each(|(h, o)| *h += o);
        let xn = layer_norm(x, d, &self.mlp_norm, None, eps)?;
        let h = self.wi.forward(&xn, l)?;
        let inter = self.wi.out / 2;
        check_len("layer.wi_out", h.len(), &[l, 2, inter])?;
        let mut g = vec![0.0f32; l * inter];
        if inter > 0 {
            g.par_chunks_mut(inter)
                .zip(h.par_chunks(2 * inter))
                .for_each(|(o, r)| {
                    for (j, oj) in o.iter_mut().enumerate() {
                        *oj = gelu_exact(r[j]) * r[inter + j];
                    }
                });
        }
        let m = self.wo_mlp.forward(&g, l)?;
        check_len("layer.mlp_out", m.len(), &[l, d])?;
        x.par_iter_mut()
            .zip(m.par_iter())
            .for_each(|(h, o)| *h += o);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        attention, gelu_exact, key_range, layer_norm, rope_inv_freq, rope_tables_started, RopeTable,
    };
    use crate::models::modernbert::test_support::fixture_encoder;
    use crate::models::modernbert::ModernBertError;

    /// `gelu_exact` matches `0.5 * x * (1 + erf(x / sqrt 2))` to 1e-7. Reference values
    /// are Python `math.erf` in f64 (the f32 result's own rounding is <= 8.4e-8 here).
    #[test]
    fn gelu_exact_reference() {
        let cases = [
            (-3.0f32, -0.004_049_694_094_890_31_f64),
            (-1.0, -0.158_655_253_931_457_07),
            (-0.5, -0.154_268_769_362_993_47),
            (0.0, 0.0),
            (0.5, 0.345_731_230_637_006_5),
            (1.0, 0.841_344_746_068_542_9),
            (3.0, 2.995_950_305_905_11),
        ];
        for (x, want) in cases {
            let got = f64::from(gelu_exact(x));
            let d = (got - want).abs();
            assert!(
                d <= 1e-7,
                "gelu_exact({x}) = {got}, reference {want}, |d| = {d:e}"
            );
        }
    }

    /// transformers 5.17's `inv_freq = 1.0 / (base ** (arange(0, dim, 2).float() / dim))`
    /// on torch 2.14 CPU, as f32 bit patterns (captured 2026-09-26, debug session
    /// laya-rescore-drift). torch rounds the power to f32 and then takes an f32
    /// reciprocal; the single-rounding `f32(1 / theta^e)` differs from these by one
    /// ULP at 8 of 32 (theta 160000) and 10 of 32 (theta 10000) frequencies, which
    /// puts up to ~32 ULP into the sin / cos tables of a 128-token row.
    #[test]
    fn rope_inv_freq_is_torch_bitwise() {
        #[rustfmt::skip]
        const CASES: [(f64, usize, &[u32]); 4] = [
            (
                160_000.0,
                64,
                &[
                    0x3f80_0000, 0x3f30_0a3a, 0x3ef2_1c1f, 0x3ea6_7d01, 0x3e64_f92e, 0x3e1d_7475,
                    0x3dd8_8cb4, 0x3d94_e963, 0x3d4c_cccd, 0x3d0c_d4fb, 0x3cc1_b019, 0x3c85_30ce,
                    0x3c37_2dbf, 0x3bfb_ed88, 0x3bad_3d5e, 0x3b6e_4237, 0x3b23_d70a, 0x3ae1_54c5,
                    0x3a9a_f347, 0x3a55_1ae3, 0x3a12_8aff, 0x39c9_8ad3, 0x398a_977e, 0x393e_9b60,
                    0x3903_126f, 0x38b4_43d0, 0x3877_eba6, 0x382a_7be8, 0x37ea_77ff, 0x37a1_3bdc,
                    0x375d_bf30, 0x3718_7c4d,
                ],
            ),
            (
                10_000.0,
                64,
                &[
                    0x3f80_0000, 0x3f3f_f911, 0x3f0f_f59a, 0x3ed7_e89b, 0x3ea1_e89b, 0x3e72_d423,
                    0x3e36_1887, 0x3e08_8d77, 0x3dcc_cccd, 0x3d99_940d, 0x3d66_55c2, 0x3d2c_ba15,
                    0x3d01_86e3, 0x3cc2_434f, 0x3c91_ad39, 0x3c5a_7bf2, 0x3c23_d70a, 0x3bf5_b9b0,
                    0x3bb8_449c, 0x3b8a_2e77, 0x3b4f_3e38, 0x3b1b_690d, 0x3ae9_1528, 0x3aae_c98e,
                    0x3a83_126f, 0x3a44_948c, 0x3a13_6a16, 0x39dd_1725, 0x39a5_cb60, 0x3978_a815,
                    0x393a_7753, 0x390b_d472,
                ],
            ),
            (
                160_000.0,
                16,
                &[
                    0x3f80_0000, 0x3e64_f92e, 0x3d4c_cccd, 0x3c37_2dbf, 0x3b23_d70a, 0x3a12_8aff,
                    0x3903_126f, 0x37ea_77ff,
                ],
            ),
            (
                10_000.0,
                16,
                &[
                    0x3f80_0000, 0x3ea1_e89b, 0x3dcc_cccd, 0x3d01_86e3, 0x3c23_d70a, 0x3b4f_3e38,
                    0x3a83_126f, 0x39a5_cb60,
                ],
            ),
        ];
        for (theta, hd, want) in CASES {
            let got: Vec<u32> = rope_inv_freq(theta, hd)
                .iter()
                .map(|f| f.to_bits())
                .collect();
            let bad: Vec<usize> = (0..want.len())
                .filter(|&p| got.get(p) != Some(&want[p]))
                .collect();
            assert!(
                got.len() == want.len() && bad.is_empty(),
                "theta {theta} hd {hd}: inv_freq differs from torch at p = {bad:?}"
            );
        }
    }

    /// KANI-LAYA-PARITY-002's evidence: exhaustive over i, j < 512 with the Laya
    /// half-window 64, the attended key range is exactly `|i - j| <= 64`, inclusive.
    #[test]
    fn window_predicate_exhaustive_512() {
        let (l, w) = (512usize, 64usize);
        for i in 0..l {
            let (lo, hi) = key_range(i, l, Some(w));
            for j in 0..l {
                assert_eq!(lo <= j && j < hi, i.abs_diff(j) <= w, "i={i} j={j}");
            }
            assert_eq!(key_range(i, l, None), (0, l));
        }
    }

    /// WR-04 / V10-b (plan 08-20): the public layer refuses an empty row with
    /// `EmptyInput` BEFORE it builds a RoPE table, instead of passing every length
    /// check with `l == 0` and panicking on `&qkv[d..]`.
    #[test]
    fn layer_forward_empty_row_is_refused() {
        let enc = fixture_encoder();
        let (cfg, layer) = (enc.config(), &enc.layers()[0]);
        let before = rope_tables_started();
        let mut x: Vec<f32> = Vec::new();
        assert!(matches!(
            layer.forward(&mut x, 0, cfg, 64),
            Err(ModernBertError::EmptyInput)
        ));
        assert_eq!(
            rope_tables_started(),
            before,
            "the empty row was refused before any RoPE table was built"
        );
    }

    /// The crate-internal entry point the encoder uses carries the same guard, so a
    /// caller that builds its own (empty) table cannot reach the slicing either.
    #[test]
    fn layer_forward_with_rope_empty_row_is_refused() {
        let enc = fixture_encoder();
        let (cfg, layer) = (enc.config(), &enc.layers()[0]);
        let rope = RopeTable::new(&rope_inv_freq(cfg.rope_theta_global(), cfg.head_dim()), 0);
        let mut x: Vec<f32> = Vec::new();
        assert!(matches!(
            layer.forward_with_rope(&mut x, 0, cfg, 64, &rope),
            Err(ModernBertError::EmptyInput)
        ));
    }

    /// A caller `l` that disagrees with `x` — here one so large that `l * half` would
    /// overflow the RoPE table's capacity — is a typed shape error returned BEFORE the
    /// table is sized from it: no panic, no allocation proportional to `l`.
    #[test]
    fn layer_forward_huge_l_is_refused_before_rope() {
        let enc = fixture_encoder();
        let (cfg, layer) = (enc.config(), &enc.layers()[0]);
        let before = rope_tables_started();
        let mut x = vec![0.0f32; cfg.hidden_size()];
        let got = layer.forward(&mut x, usize::MAX / 2, cfg, 64);
        assert!(
            matches!(
                got,
                Err(ModernBertError::InputShape {
                    what: "layer.x",
                    ..
                })
            ),
            "expected InputShape(layer.x), got {got:?}"
        );
        assert_eq!(
            rope_tables_started(),
            before,
            "the mis-sized row was refused before any RoPE table was built"
        );
    }

    /// The reusable primitives refuse inconsistent buffers with a typed error.
    #[test]
    fn primitives_refuse_bad_shapes() {
        assert!(matches!(
            layer_norm(&[1.0, 2.0, 3.0], 2, &[1.0, 1.0], None, 1e-5),
            Err(ModernBertError::InputShape { .. })
        ));
        assert!(matches!(
            layer_norm(&[1.0, 2.0], 0, &[], None, 1e-5),
            Err(ModernBertError::InputShape { .. })
        ));
        let q = vec![0.0f32; 8];
        assert!(matches!(
            attention(&q, &q, &q[..6], 2, 2, 2, None),
            Err(ModernBertError::InputShape {
                what: "attention.v",
                ..
            })
        ));
    }
}
