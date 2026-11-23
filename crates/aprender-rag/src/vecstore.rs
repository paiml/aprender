//! Flat int8 vector store kept next to `index.sqlite` (`vectors.bin`).
//!
//! Record = `rowid: i64 LE | scale: f32 LE | dim x i8`. Vectors are L2-normalised then quantised
//! with `scale = max|x| / 127`. Brute-force dot-product scan; append-only so an embed job resumes.

use crate::{Error, Result};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::Path;

const MAGIC: &[u8; 8] = b"RAGV1\0\0\0";
const HEADER: usize = 16;

fn rec_len(dim: usize) -> usize {
    12 + dim
}

/// L2-normalise then quantise to int8; returns `(scale, codes)`.
#[must_use]
pub fn quantize(v: &[f32]) -> (f32, Vec<i8>) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    let max = v.iter().fold(0.0_f32, |m, x| m.max((x / norm).abs())).max(1e-12);
    let scale = max / 127.0;
    let codes = v.iter().map(|x| ((x / norm) / scale).round().clamp(-127.0, 127.0) as i8).collect();
    (scale, codes)
}

/// Append-only writer.
#[derive(Debug)]
pub struct VecWriter {
    file: File,
    dim: usize,
    last_rowid: i64,
}

impl VecWriter {
    /// Open (creating if absent) a store of dimension `dim`; a torn trailing record is truncated.
    ///
    /// # Errors
    /// I/O failure or a dimension/magic mismatch with an existing file.
    pub fn open(path: &Path, dim: usize) -> Result<Self> {
        let io = |e: std::io::Error| Error::Query(format!("vecstore {}: {e}", path.display()));
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)
            .map_err(io)?;
        let len = file.metadata().map_err(io)?.len() as usize;
        if len == 0 {
            let mut h = [0u8; HEADER];
            h[..8].copy_from_slice(MAGIC);
            h[8..12].copy_from_slice(&(dim as u32).to_le_bytes());
            file.write_all(&h).map_err(io)?;
            return Ok(Self { file, dim, last_rowid: 0 });
        }
        let mut h = [0u8; HEADER];
        file.read_exact(&mut h).map_err(io)?;
        let d = u32::from_le_bytes([h[8], h[9], h[10], h[11]]) as usize;
        if &h[..8] != MAGIC || d != dim {
            return Err(Error::Query(format!(
                "vecstore {}: bad magic or dim {d} != {dim}",
                path.display()
            )));
        }
        let n = (len - HEADER) / rec_len(dim);
        file.set_len((HEADER + n * rec_len(dim)) as u64).map_err(io)?;
        let mut last = 0_i64;
        if n > 0 {
            use std::io::{Seek, SeekFrom};
            file.seek(SeekFrom::Start((HEADER + (n - 1) * rec_len(dim)) as u64)).map_err(io)?;
            let mut b = [0u8; 8];
            file.read_exact(&mut b).map_err(io)?;
            last = i64::from_le_bytes(b);
        }
        use std::io::{Seek, SeekFrom};
        file.seek(SeekFrom::End(0)).map_err(io)?;
        Ok(Self { file, dim, last_rowid: last })
    }

    /// Highest rowid stored (0 when empty); the embed job resumes after it.
    #[must_use]
    pub fn last_rowid(&self) -> i64 {
        self.last_rowid
    }

    /// Append a batch of `(rowid, vector)` in one write.
    ///
    /// # Errors
    /// I/O failure or a wrong vector length.
    pub fn append(&mut self, batch: &[(i64, Vec<f32>)]) -> Result<()> {
        let mut buf = Vec::with_capacity(batch.len() * rec_len(self.dim));
        for (rid, v) in batch {
            if v.len() != self.dim {
                return Err(Error::Query(format!("vector len {} != dim {}", v.len(), self.dim)));
            }
            let (s, codes) = quantize(v);
            buf.extend_from_slice(&rid.to_le_bytes());
            buf.extend_from_slice(&s.to_le_bytes());
            buf.extend(codes.iter().map(|c| *c as u8));
            self.last_rowid = *rid;
        }
        self.file.write_all(&buf).map_err(|e| Error::Query(format!("vecstore write: {e}")))
    }
}

/// In-memory reader doing brute-force scans.
#[derive(Debug)]
pub struct VecReader {
    data: Vec<u8>,
    dim: usize,
}

impl VecReader {
    /// Load the whole file (about 400 bytes per chunk at dim 384).
    ///
    /// # Errors
    /// I/O failure or bad header.
    pub fn open(path: &Path) -> Result<Self> {
        let data = std::fs::read(path)
            .map_err(|e| Error::Query(format!("vecstore {}: {e}", path.display())))?;
        if data.len() < HEADER || &data[..8] != MAGIC {
            return Err(Error::Query(format!("vecstore {}: bad header", path.display())));
        }
        let dim = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
        Ok(Self { data, dim })
    }

    /// Number of stored vectors.
    #[must_use]
    pub fn len(&self) -> usize {
        (self.data.len() - HEADER) / rec_len(self.dim)
    }

    /// True when empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Vector dimension.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Top-`k` `(rowid, cosine)` for query `q`, best first.
    #[must_use]
    pub fn top_k(&self, q: &[f32], k: usize) -> Vec<(i64, f32)> {
        if q.len() != self.dim || k == 0 {
            return Vec::new();
        }
        let (qs, qc) = quantize(q);
        let rl = rec_len(self.dim);
        let mut scored: Vec<(i64, f32)> = self.data[HEADER..]
            .chunks_exact(rl)
            .map(|r| {
                let rid = i64::from_le_bytes(r[..8].try_into().unwrap_or([0; 8]));
                let s = f32::from_le_bytes(r[8..12].try_into().unwrap_or([0; 4]));
                let dot: i32 =
                    r[12..].iter().zip(&qc).map(|(a, b)| i32::from(*a as i8) * i32::from(*b)).sum();
                (rid, dot as f32 * s * qs)
            })
            .collect();
        let k = k.min(scored.len());
        if k == 0 {
            return Vec::new();
        }
        scored.select_nth_unstable_by(k - 1, |a, b| b.1.total_cmp(&a.1));
        scored.truncate(k);
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        scored
    }
}

/// Reciprocal-rank fusion of ranked id lists: `sum 1/(k0 + rank)`; returns ids best first.
#[must_use]
pub fn rrf(lists: &[Vec<String>], k0: f64) -> Vec<(String, f64)> {
    let mut acc: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
    for l in lists {
        for (i, id) in l.iter().enumerate() {
            *acc.entry(id.clone()).or_default() += 1.0 / (k0 + (i + 1) as f64);
        }
    }
    let mut v: Vec<_> = acc.into_iter().collect();
    v.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> std::path::PathBuf {
        let d = std::env::var("TMPDIR").unwrap_or_else(|_| "/tmp".into());
        let p = std::path::PathBuf::from(d).join(format!("vecstore-{name}-{}", std::process::id()));
        let _ = std::fs::remove_file(&p);
        p
    }

    #[test]
    fn roundtrip_topk_and_resume() {
        let p = tmp("rt");
        let mut w = VecWriter::open(&p, 4).unwrap();
        assert_eq!(w.last_rowid(), 0);
        w.append(&[(1, vec![1.0, 0.0, 0.0, 0.0]), (2, vec![0.0, 1.0, 0.0, 0.0])]).unwrap();
        drop(w);
        let mut w = VecWriter::open(&p, 4).unwrap();
        assert_eq!(w.last_rowid(), 2);
        w.append(&[(7, vec![0.9, 0.1, 0.0, 0.0])]).unwrap();
        assert!(w.append(&[(8, vec![1.0])]).is_err());
        let r = VecReader::open(&p).unwrap();
        assert_eq!((r.len(), r.dim(), r.is_empty()), (3, 4, false));
        let top = r.top_k(&[1.0, 0.0, 0.0, 0.0], 2);
        assert_eq!(top[0].0, 1);
        assert_eq!(top[1].0, 7);
        assert!(top[0].1 > 0.99 && top[0].1 >= top[1].1);
        assert!(r.top_k(&[1.0], 2).is_empty() && r.top_k(&[1.0, 0.0, 0.0, 0.0], 0).is_empty());
        assert!(VecWriter::open(&p, 5).is_err());
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn torn_tail_is_truncated() {
        let p = tmp("torn");
        let mut w = VecWriter::open(&p, 2).unwrap();
        w.append(&[(3, vec![1.0, 1.0])]).unwrap();
        drop(w);
        let mut f = OpenOptions::new().append(true).open(&p).unwrap();
        f.write_all(&[1, 2, 3]).unwrap();
        drop(f);
        let w = VecWriter::open(&p, 2).unwrap();
        assert_eq!(w.last_rowid(), 3);
        assert_eq!(VecReader::open(&p).unwrap().len(), 1);
        let _ = std::fs::remove_file(&p);
        assert!(VecReader::open(&p).is_err());
    }

    #[test]
    fn rrf_orders_by_combined_rank() {
        let out = rrf(&[vec!["a".into(), "b".into()], vec!["b".into(), "c".into()]], 60.0);
        assert_eq!(out[0].0, "b");
        assert_eq!(out.len(), 3);
    }
}
