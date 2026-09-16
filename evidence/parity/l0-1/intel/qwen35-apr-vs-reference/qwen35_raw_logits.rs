//! apr (realizar) Qwen3.5 CPU forward -> raw float32 logits for EVERY position (PMAT-3091).
//!
//! Subject harness for the apr-vs-llama.cpp measurement. It calls the same
//! `Qwen35Model::forward_single_qwen35` loop that `run_qwen35_generate` (the `apr run` /
//! `apr chat` Qwen3.5 path in #3114) uses for prefill, one token per call, and keeps the
//! logits row after EVERY position instead of only the last. Token ids are read directly;
//! the tokenizer is never involved.
//!
//! Output layout is byte-compatible with the llama.cpp reference producer
//! (evidence/parity/l0-1/intel/qwen35-cpu-reference/producer/apr_raw_logits.cpp):
//!   char[8] "APRRAWLG" | uint32 version=1 | int32 n_pos | int32 n_vocab |
//!   int32 token_ids[n_pos] | float32 logits[n_pos*n_vocab]   (little-endian)
//!
//! Build: copied (uncommitted) into crates/aprender-serve/examples/ of the #3114 tree.
//! Usage: qwen35_raw_logits <model.gguf> <ids.txt: comma/space separated> <out.bin>

use realizar::gguf::forward_qwen35::Qwen35Model;
use realizar::gguf::MappedGGUFModel;
use std::io::Write;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [model_path, ids_path, out_path] = args.as_slice() else {
        return Err("usage: qwen35_raw_logits <model.gguf> <ids.txt> <out.bin>".into());
    };
    let ids: Vec<u32> = std::fs::read_to_string(ids_path)?
        .split(|c: char| c == ',' || c.is_whitespace() || c == '[' || c == ']')
        .filter(|t| !t.is_empty())
        .map(str::parse::<u32>)
        .collect::<Result<_, _>>()?;
    if ids.is_empty() {
        return Err("no token ids".into());
    }

    let mapped = MappedGGUFModel::from_path(model_path)?;
    let base = Qwen35Model::create_base_model(&mapped.model, mapped.data())?;
    let qwen = Qwen35Model::from_model_and_layers(&base, &mapped.model, mapped.data())?;
    let mut state = qwen.new_state(ids.len() + 1);

    let mut rows: Vec<Vec<f32>> = Vec::with_capacity(ids.len());
    for (pos, &token) in ids.iter().enumerate() {
        rows.push(qwen.forward_single_qwen35(token, &mut state, pos)?);
    }
    let n_vocab = rows.first().map_or(0, Vec::len);
    if rows.iter().any(|r| r.len() != n_vocab) {
        return Err("ragged logits rows".into());
    }

    let tmp = format!("{out_path}.partial");
    let mut out = std::io::BufWriter::new(std::fs::File::create(&tmp)?);
    out.write_all(b"APRRAWLG")?;
    out.write_all(&1u32.to_le_bytes())?;
    out.write_all(&i32::try_from(ids.len())?.to_le_bytes())?;
    out.write_all(&i32::try_from(n_vocab)?.to_le_bytes())?;
    for &id in &ids {
        out.write_all(&i32::try_from(id)?.to_le_bytes())?;
    }
    for row in &rows {
        for &v in row {
            out.write_all(&v.to_le_bytes())?;
        }
    }
    out.into_inner().map_err(|e| e.into_error())?.sync_all()?;
    std::fs::rename(&tmp, out_path)?;
    println!("qwen35_raw_logits: n_pos={} n_vocab={n_vocab} out={out_path}", ids.len());
    Ok(())
}
