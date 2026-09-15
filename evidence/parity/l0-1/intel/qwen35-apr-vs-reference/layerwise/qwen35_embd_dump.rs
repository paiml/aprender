//! apr (realizar) Qwen3.5: dump the token-embedding row for each given id (PMAT-3091 layerwise).
//!
//! This is the ONLY per-layer quantity reachable through realizar's public API at #3114 `31448f6c3`:
//! `Qwen35Model::create_base_model` + `OwnedQuantizedModel::token_embedding()` is exactly the slice
//! `forward_single_qwen35` copies into `hidden` before layer 0. Layer outputs are not reachable:
//! `Qwen35Model.layers`, `Qwen35OwnedLayer`, `forward_deltanet` and `forward_attention` are
//! `pub(crate)` / private.
//!
//! Build: copied (uncommitted) into crates/aprender-serve/examples/ of the #3114 tree.
//! Usage: qwen35_embd_dump <model.gguf> <ids.txt> <positions: comma list> <out_dir>
//! Writes <out_dir>/pos<P>/embd.f32 (hidden_dim little-endian f32) and prints "pos id hidden_dim" lines.

use realizar::gguf::forward_qwen35::Qwen35Model;
use realizar::gguf::MappedGGUFModel;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [model_path, ids_path, positions, out_dir] = args.as_slice() else {
        return Err("usage: qwen35_embd_dump <model.gguf> <ids.txt> <positions> <out_dir>".into());
    };
    let ids: Vec<u32> = std::fs::read_to_string(ids_path)?
        .split(|c: char| c == ',' || c.is_whitespace())
        .filter(|t| !t.is_empty())
        .map(str::parse::<u32>)
        .collect::<Result<_, _>>()?;
    let mapped = MappedGGUFModel::from_path(model_path)?;
    let base = Qwen35Model::create_base_model(&mapped.model, mapped.data())?;
    let emb = base.token_embedding();
    // `config` is pub(crate); the final-norm weight length is hidden_dim through the public API.
    let hidden = base.output_norm_weight().len();
    for p in positions.split(',').filter(|t| !t.is_empty()) {
        let pos: usize = p.parse()?;
        let id = *ids.get(pos).ok_or("position past the id list")? as usize;
        let row = emb.get(id * hidden..(id + 1) * hidden).ok_or("id past the embedding table")?;
        let dir = format!("{out_dir}/pos{pos}");
        std::fs::create_dir_all(&dir)?;
        let bytes: Vec<u8> = row.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(format!("{dir}/embd.f32"), bytes)?;
        println!("qwen35_embd_dump: pos={pos} id={id} hidden_dim={hidden}");
    }
    Ok(())
}
