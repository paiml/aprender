//! Dense embeddings for the SQLite content index: `embed` job (resumable) and query-time vectors.
//! Vectors live in `<index>/vectors.bin` (see `aprender_rag::vecstore`).

use anyhow::{Context, Result};
use aprender_rag::embed::{Embedder, EmbeddingModelType, FastEmbedder};
use aprender_rag::vecstore::VecWriter;
use std::path::Path;

/// Dimension of bge-small-en-v1.5.
pub const DIM: usize = 384;
/// bge-small query instruction (documents are embedded raw).
const QUERY_PREFIX: &str = "Represent this sentence for searching relevant passages: ";

fn model() -> Result<FastEmbedder> {
    if std::env::var_os("FASTEMBED_CACHE_DIR").is_none() {
        std::env::set_var(
            "FASTEMBED_CACHE_DIR",
            "/mnt/nvme-raid0/rag/models/fastembed",
        );
    }
    FastEmbedder::new(EmbeddingModelType::BgeSmallEnV15).context("loading bge-small-en-v1.5")
}

/// Embed a query string with the bge query prefix.
///
/// # Errors
/// Fails when the model cannot load or embedding fails.
pub fn embed_query_vec(query: &str) -> Result<Vec<f32>> {
    let m = model()?;
    m.embed(&format!("{QUERY_PREFIX}{query}"))
        .context("embedding query")
}

/// Truncate to at most `max` characters on a char boundary.
pub fn clip(s: &str, max: usize) -> &str {
    match s.char_indices().nth(max) {
        Some((i, _)) => &s[..i],
        None => s,
    }
}

/// Embed every chunk of `<index>/index.sqlite` not yet in `vectors.bin`; `limit` 0 = all.
///
/// # Errors
/// Fails on sqlite, model or file errors.
pub fn run_embed(index: &str, batch: usize, limit: usize) -> Result<()> {
    let dir = Path::new(index);
    let idx = aprender_rag::SqliteIndex::open(dir.join("index.sqlite"))
        .map_err(|e| anyhow::anyhow!("open index.sqlite: {e}"))?;
    let mut w = VecWriter::open(&dir.join("vectors.bin"), DIM)
        .map_err(|e| anyhow::anyhow!("open vectors.bin: {e}"))?;
    let m = model()?;
    let total = idx.chunk_count().map_err(|e| anyhow::anyhow!("{e}"))?;
    let (mut done, mut after) = (0usize, w.last_rowid());
    let t0 = std::time::Instant::now();
    loop {
        let want = if limit > 0 {
            batch.min(limit - done)
        } else {
            batch
        };
        if want == 0 {
            break;
        }
        let rows = idx
            .chunks_after(after, want)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        if rows.is_empty() {
            break;
        }
        let texts: Vec<&str> = rows.iter().map(|(_, c)| clip(c, 1500)).collect();
        let vecs = m
            .embed_batch(&texts)
            .map_err(|e| anyhow::anyhow!("embed batch: {e}"))?;
        anyhow::ensure!(
            vecs.len() == rows.len(),
            "empty chunk text in batch near rowid {after}"
        );
        let out: Vec<(i64, Vec<f32>)> = rows.iter().map(|(r, _)| *r).zip(vecs).collect();
        w.append(&out).map_err(|e| anyhow::anyhow!("append: {e}"))?;
        after = rows.last().map_or(after, |(r, _)| *r);
        done += rows.len();
        if (done / batch) % 20 == 0 {
            eprintln!(
                "embed: {done} chunks this run (last rowid {after} of ~{total}), {:.1}/s",
                done as f64 / t0.elapsed().as_secs_f64()
            );
        }
    }
    println!(
        "embed: {done} chunks embedded this run, last rowid {after}, {:.1}s",
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::clip;

    #[test]
    fn clip_respects_char_boundaries() {
        assert_eq!(clip("abc", 5), "abc");
        assert_eq!(clip("abcdef", 3), "abc");
        assert_eq!(clip("aeioé", 4), "aeio");
        assert_eq!(clip("日本語", 2), "日本");
    }
}
