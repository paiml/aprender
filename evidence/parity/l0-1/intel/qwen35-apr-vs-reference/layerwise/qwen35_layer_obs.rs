//! apr (realizar) Qwen3.5 CPU forward through `forward_single_qwen35_observed` (PMAT-3091 layer observer).
//!
//! Two modes, both writing the APRRAWLG logits file of `qwen35_raw_logits.rs` (same layout):
//!   noop  <model.gguf> <ids.txt> <out.bin>
//!         observed path with a NO-OP closure; the invariance proof compares <out.bin> bytes to the
//!         plain-path subject (orig f6f79264..., p4 92f1b54d...).
//!   dump  <model.gguf> <ids.txt> <out.bin> <positions: comma list> <out_dir>
//!         observed path with a closure that, at the listed positions only, writes each observation to
//!         <out_dir>/pos<P>/<name>-<il>.f32 (layer points) or <out_dir>/pos<P>/<name>.f32 (global
//!         points), the llama.cpp dump naming. A manifest.tsv lists name, layer, pos, n, file.
//!
//! Build: copied (uncommitted) into crates/aprender-serve/examples/ of the obs-3091 tree
//! (branch PMAT-3091-layer-observer, stacked on #3114 31448f6c3).

use realizar::gguf::forward_qwen35::{Qwen35Model, QWEN35_OBS_NO_LAYER};
use realizar::gguf::MappedGGUFModel;
use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::io::Write;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

fn read_ids(path: &str) -> Res<Vec<u32>> {
    let ids: Vec<u32> = std::fs::read_to_string(path)?
        .split(|c: char| c == ',' || c.is_whitespace() || c == '[' || c == ']')
        .filter(|t| !t.is_empty())
        .map(str::parse::<u32>)
        .collect::<Result<_, _>>()?;
    if ids.is_empty() {
        return Err("no token ids".into());
    }
    Ok(ids)
}

fn write_rawlogits(out_path: &str, ids: &[u32], rows: &[Vec<f32>]) -> Res<()> {
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
    for &id in ids {
        out.write_all(&i32::try_from(id)?.to_le_bytes())?;
    }
    for row in rows {
        for &v in row {
            out.write_all(&v.to_le_bytes())?;
        }
    }
    out.into_inner().map_err(|e| e.into_error())?.sync_all()?;
    std::fs::rename(&tmp, out_path)?;
    println!(
        "qwen35_layer_obs: n_pos={} n_vocab={n_vocab} out={out_path}",
        ids.len()
    );
    Ok(())
}

/// Observer state for `dump` mode. Write errors are kept (the closure cannot return one).
struct Dumper {
    dir: String,
    positions: BTreeSet<usize>,
    pos: usize,
    manifest: String,
    written: usize,
    error: Option<String>,
}

impl Dumper {
    fn observe(&mut self, name: &str, il: usize, v: &[f32]) {
        if !self.positions.contains(&self.pos) || self.error.is_some() {
            return;
        }
        let rel = if il == QWEN35_OBS_NO_LAYER {
            format!("pos{}/{name}.f32", self.pos)
        } else {
            format!("pos{}/{name}-{il}.f32", self.pos)
        };
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        let res = std::fs::create_dir_all(format!("{}/pos{}", self.dir, self.pos))
            .and_then(|()| std::fs::write(format!("{}/{rel}", self.dir), bytes));
        if let Err(e) = res {
            self.error = Some(format!("{rel}: {e}"));
            return;
        }
        let layer = if il == QWEN35_OBS_NO_LAYER {
            -1
        } else {
            i64::try_from(il).unwrap_or(-2)
        };
        let _ = writeln!(
            self.manifest,
            "{name}\t{layer}\t{}\t{}\t{rel}",
            self.pos,
            v.len()
        );
        self.written += 1;
    }
}

fn main() -> Res<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (mode, rest) = args
        .split_first()
        .ok_or("usage: qwen35_layer_obs noop|dump ...")?;
    let (model_path, ids_path, out_path) = match rest {
        [m, i, o, ..] => (m, i, o),
        _ => {
            return Err(
                "usage: qwen35_layer_obs noop|dump <model> <ids> <out.bin> [positions out_dir]"
                    .into(),
            )
        },
    };
    let ids = read_ids(ids_path)?;
    let mapped = MappedGGUFModel::from_path(model_path)?;
    let base = Qwen35Model::create_base_model(&mapped.model, mapped.data())?;
    let qwen = Qwen35Model::from_model_and_layers(&base, &mapped.model, mapped.data())?;
    let mut state = qwen.new_state(ids.len() + 1);
    let mut rows: Vec<Vec<f32>> = Vec::with_capacity(ids.len());

    match (mode.as_str(), rest) {
        ("noop", [_, _, _]) => {
            for (pos, &token) in ids.iter().enumerate() {
                rows.push(qwen.forward_single_qwen35_observed(token, &mut state, pos, &mut |_, _, _| {})?);
            }
        },
        ("dump", [_, _, _, positions, dir]) => {
            let positions = positions
                .split(',')
                .filter(|t| !t.is_empty())
                .map(str::parse::<usize>)
                .collect::<Result<BTreeSet<_>, _>>()?;
            let mut d = Dumper { dir: dir.clone(), positions, pos: 0, manifest: String::new(), written: 0, error: None };
            for (pos, &token) in ids.iter().enumerate() {
                d.pos = pos;
                rows.push(qwen.forward_single_qwen35_observed(token, &mut state, pos, &mut |n, l, v| d.observe(n, l, v))?);
            }
            if let Some(e) = d.error {
                return Err(format!("dump write failed: {e}").into());
            }
            std::fs::create_dir_all(dir)?;
            std::fs::write(format!("{dir}/manifest.tsv"), format!("name\tlayer\tpos\tn\tfile\n{}", d.manifest))?;
            println!("qwen35_layer_obs: dump dir={dir} written={}", d.written);
        },
        _ => return Err("usage: qwen35_layer_obs noop <model> <ids> <out.bin> | dump <model> <ids> <out.bin> <positions> <out_dir>".into()),
    }
    write_rawlogits(out_path, &ids, &rows)
}
