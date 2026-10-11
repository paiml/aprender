//! The Qwen3.5 token embedding kept in its GGUF encoding (#4991).
//!
//! The base model used to hold `token_embd.weight` as f32: about 4.9 times its
//! Q6_K bytes and 8 times its Q4_K bytes, 4.07 GB for the 9B, which made the host
//! build cost well over twice the file. Decode and prefill only ever read whole
//! rows, so the table stays quantized and one row is dequantized per lookup.

use crate::error::{RealizarError, Result};
use crate::gguf::types::{
    GGUF_TYPE_F16, GGUF_TYPE_F32, GGUF_TYPE_Q4_K, GGUF_TYPE_Q6_K, GGUF_TYPE_Q8_0,
};
use crate::gguf::GGUFModel;
use crate::quantize::dequant::{
    dequantize_q4_k_into, dequantize_q6_k_into, dequantize_q8_0_into, f16_to_f32,
};

/// A `[rows, row_len]` table in its GGUF encoding; [`Self::row`] dequantizes one row.
pub(crate) struct Qwen35Embedding {
    data: Vec<u8>,
    qtype: u32,
    rows: usize,
    row_len: usize,
    row_bytes: usize,
}

impl Qwen35Embedding {
    /// Copy `token_embd.weight`, still quantized, out of the mapped file.
    ///
    /// # Errors
    /// The tensor is missing, lies outside the file, or its rows are not
    /// `hidden_dim` wide and byte-aligned.
    pub(crate) fn load(model: &GGUFModel, data: &[u8], hidden_dim: usize) -> Result<Self> {
        let r = crate::gguf::QuantizedGGUFTransformer::get_tensor_ref(
            model,
            data,
            "token_embd.weight",
        )?;
        let end = r.offset.checked_add(r.byte_size);
        let bytes = end.and_then(|end| data.get(r.offset..end)).ok_or_else(|| {
            RealizarError::FormatError {
                reason: format!(
                    "qwen35: token_embd.weight at byte {} (+{}) lies outside the file",
                    r.offset, r.byte_size
                ),
            }
        })?;
        Self::from_bytes(bytes.to_vec(), r.qtype, r.num_elements, hidden_dim)
    }

    /// A table of `num_elements` values of GGUF type `qtype`, `row_len` per row.
    ///
    /// # Errors
    /// The values do not split into whole rows, the bytes do not split into
    /// equal rows, or row 0 does not dequantize to `row_len` values.
    pub(crate) fn from_bytes(
        data: Vec<u8>,
        qtype: u32,
        num_elements: usize,
        row_len: usize,
    ) -> Result<Self> {
        let rows = if row_len == 0 {
            0
        } else {
            num_elements / row_len
        };
        if rows == 0 || rows * row_len != num_elements || data.len() % rows != 0 {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "qwen35: token_embd.weight ({num_elements} values, {} bytes, type {qtype}) \
                     does not split into rows of {row_len}",
                    data.len()
                ),
            });
        }
        let table = Self {
            row_bytes: data.len() / rows,
            data,
            qtype,
            rows,
            row_len,
        };
        // Refuse at load, not at the first token: an unsupported type or a row
        // narrower than its encoding fails here.
        if table.row(0)?.len() != row_len {
            return Err(RealizarError::InvalidShape {
                reason: format!("qwen35: token_embd.weight row 0 is not {row_len} values wide"),
            });
        }
        Ok(table)
    }

    /// An F32 table over `values`, `row_len` per row, for tests that build a model
    /// by hand.
    #[cfg(test)]
    pub(crate) fn from_f32(values: &[f32], row_len: usize) -> Result<Self> {
        let data = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        Self::from_bytes(
            data,
            crate::gguf::types::GGUF_TYPE_F32,
            values.len(),
            row_len,
        )
    }

    /// The number of rows (the vocabulary the table covers).
    pub(crate) fn rows(&self) -> usize {
        self.rows
    }

    /// The number of values the table holds, `rows * row_len`.
    pub(crate) fn num_elements(&self) -> usize {
        self.rows * self.row_len
    }

    /// The width of one row.
    pub(crate) fn row_len(&self) -> usize {
        self.row_len
    }

    /// Row `token`, dequantized.
    ///
    /// # Errors
    /// `token` is outside the table, or the row fails to dequantize.
    pub(crate) fn row(&self, token: usize) -> Result<Vec<f32>> {
        let mut out = vec![0.0; self.row_len];
        self.row_into(token, &mut out)?;
        Ok(out)
    }

    /// Row `token`, dequantized into `out`. F32, F16, Q8_0, Q4_K and Q6_K, the
    /// types an embedding ships in, write in place, so a decode step that reuses
    /// `out` allocates nothing; any other type goes through a whole-row copy.
    ///
    /// # Errors
    /// `token` is outside the table, `out` is not one row wide, or the row
    /// fails to dequantize.
    pub(crate) fn row_into(&self, token: usize, out: &mut [f32]) -> Result<()> {
        if token >= self.rows || out.len() != self.row_len {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "qwen35: row {token} into {} values; the embedding table is {} rows of {}",
                    out.len(),
                    self.rows,
                    self.row_len
                ),
            });
        }
        let start = token * self.row_bytes;
        let bytes = &self.data[start..start + self.row_bytes];
        match self.qtype {
            GGUF_TYPE_F32 if bytes.len() == 4 * out.len() => {
                for (o, b) in out.iter_mut().zip(bytes.as_chunks::<4>().0) {
                    *o = f32::from_le_bytes(*b);
                }
            },
            GGUF_TYPE_F16 if bytes.len() == 2 * out.len() => {
                for (o, b) in out.iter_mut().zip(bytes.as_chunks::<2>().0) {
                    *o = f16_to_f32(u16::from_le_bytes(*b));
                }
            },
            GGUF_TYPE_Q8_0 => dequantize_q8_0_into(bytes, out)?,
            GGUF_TYPE_Q4_K => dequantize_q4_k_into(bytes, out)?,
            GGUF_TYPE_Q6_K => dequantize_q6_k_into(bytes, out)?,
            _ => out.copy_from_slice(&GGUFModel::dequantize_at(
                self.qtype,
                bytes,
                0,
                self.row_len,
            )?),
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::Qwen35Embedding;
    use crate::gguf::types::{
        GGUF_TYPE_F16, GGUF_TYPE_F32, GGUF_TYPE_Q4_0, GGUF_TYPE_Q4_K, GGUF_TYPE_Q6_K,
        GGUF_TYPE_Q8_0,
    };
    use crate::gguf::GGUFModel;

    /// Bytes for `n` Q4_K super-blocks, each one different, with finite scales.
    fn q4k_bytes(n: usize) -> Vec<u8> {
        let mut data = vec![0u8; n * 144];
        for (sb, block) in data.chunks_exact_mut(144).enumerate() {
            block[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // d = 1.0
            block[2..4].copy_from_slice(&0x3800u16.to_le_bytes()); // dmin = 0.5
            for (i, b) in block[4..].iter_mut().enumerate() {
                *b = (sb * 29 + i * 11) as u8;
            }
        }
        data
    }

    /// Every row equals the same slice of the whole-table dequant, for a
    /// K-quant (two super-blocks per row) and a 32-block type.
    #[test]
    fn each_row_matches_the_whole_table() {
        let (rows, row_len) = (5, 512);
        let q4k = q4k_bytes(rows * 2);
        let mut q8 = vec![0u8; rows * (row_len / 32) * 34];
        for (b, block) in q8.chunks_exact_mut(34).enumerate() {
            block[0..2].copy_from_slice(&0x3C00u16.to_le_bytes());
            for (i, v) in block[2..].iter_mut().enumerate() {
                *v = (b * 13 + i * 5) as u8;
            }
        }
        // Q6_K: d = 1.0 in the last two bytes of each 210-byte super-block.
        let mut q6k: Vec<u8> = (0..rows * 2 * 210).map(|i| (i * 7) as u8).collect();
        for block in q6k.chunks_exact_mut(210) {
            block[208..].copy_from_slice(&0x3C00u16.to_le_bytes());
        }
        let f16: Vec<u8> = (0..rows * row_len)
            .flat_map(|i| (0x3C00u16 + (i % 512) as u16).to_le_bytes())
            .collect();
        // Q4_0 has no in-place path: it goes through the whole-row copy.
        let mut q4_0 = vec![0u8; rows * (row_len / 32) * 18];
        for (b, block) in q4_0.chunks_exact_mut(18).enumerate() {
            block[0..2].copy_from_slice(&0x3800u16.to_le_bytes());
            block[2..]
                .iter_mut()
                .enumerate()
                .for_each(|(i, v)| *v = (b + i * 3) as u8);
        }
        for (qtype, data) in [
            (GGUF_TYPE_Q4_K, q4k),
            (GGUF_TYPE_Q8_0, q8),
            (GGUF_TYPE_Q6_K, q6k),
            (GGUF_TYPE_F16, f16),
            (GGUF_TYPE_Q4_0, q4_0),
        ] {
            let whole =
                GGUFModel::dequantize_at(qtype, &data, 0, rows * row_len).expect("whole table");
            let table =
                Qwen35Embedding::from_bytes(data, qtype, rows * row_len, row_len).expect("table");
            assert_eq!(table.rows(), rows);
            for t in 0..rows {
                assert!(whole.iter().all(|v| v.is_finite()), "type {qtype}");
                // The row paths are scalar and the whole-table path may be SIMD (with
                // FMA), so the two agree to rounding, not bit for bit.
                let row = table.row(t).expect("row");
                for (i, (a, b)) in row
                    .iter()
                    .zip(&whole[t * row_len..(t + 1) * row_len])
                    .enumerate()
                {
                    let tol = 4.0 * f32::EPSILON * a.abs().max(b.abs()).max(1.0);
                    assert!(
                        (a - b).abs() <= tol,
                        "type {qtype} row {t} value {i}: {a} vs {b}"
                    );
                }
            }
        }
    }

    #[test]
    fn a_token_past_the_table_is_refused() {
        let data: Vec<u8> = (0..12).flat_map(|i| (i as f32).to_le_bytes()).collect();
        let table = Qwen35Embedding::from_bytes(data, GGUF_TYPE_F32, 12, 4).expect("table");
        assert_eq!(table.row(2).expect("last row"), vec![8.0, 9.0, 10.0, 11.0]);
        assert!(table.row(3).is_err());
        // An output that is not one row wide is refused, not truncated.
        assert!(table.row_into(0, &mut [0.0; 3]).is_err());
    }

    #[test]
    fn a_table_that_does_not_split_into_rows_is_refused() {
        // 10 values cannot be rows of 4.
        let data: Vec<u8> = (0..10).flat_map(|i| (i as f32).to_le_bytes()).collect();
        assert!(Qwen35Embedding::from_bytes(data, GGUF_TYPE_F32, 10, 4).is_err());
        // An unsupported type is refused at load, not at the first token.
        assert!(Qwen35Embedding::from_bytes(vec![0u8; 64], 9999, 16, 8).is_err());
    }
}
