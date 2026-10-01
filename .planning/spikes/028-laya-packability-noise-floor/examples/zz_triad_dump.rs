//! THROWAWAY (spike 028) — copied into crates/aprender-decide/examples/ only while measuring,
//! removed before any commit. Derived from the laya-rescore-drift debug session's zz_rescore_dump.
//!
//! zz_triad_dump CKPT_DIR DATA_DIR OUT_DIR [EXTRA_T,...]
//!
//! Scores every row of DATA_DIR/eval.jsonl with the Rust builder's OWN ids (what pack does) and
//! writes OUT_DIR/all.json (ids, markers, logits z, probs p at the model's applied T, and p at each
//! EXTRA_T via the same softmax_t) plus OUT_DIR/final.f32: the encoder's final-norm output for every
//! row, concatenated row by row (len_r * hidden f32 LE each).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::disallowed_methods)]

use aprender_decide::laya::QType;
use aprender_decide::{DecisionMethod, Task};
use serde_json::json;
use std::io::Write;
use std::path::PathBuf;

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let ck = PathBuf::from(&a[0]);
    let data = PathBuf::from(&a[1]);
    let out = PathBuf::from(&a[2]);
    let extra: Vec<f32> = a
        .get(3)
        .map(|s| s.split(',').filter(|x| !x.is_empty()).map(|x| x.parse().unwrap()).collect())
        .unwrap_or_default();
    std::fs::create_dir_all(&out).unwrap();
    let task = Task::from_slice(&std::fs::read(data.join("task.json")).unwrap()).unwrap();
    let texts: Vec<String> = std::fs::read_to_string(data.join("eval.jsonl"))
        .unwrap()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            serde_json::from_str::<serde_json::Value>(l).unwrap()["text"]
                .as_str()
                .unwrap()
                .to_string()
        })
        .collect();
    let laya = aprender_decide::pack::load_checkpoint_for_scoring(&ck, task).unwrap();
    let t = laya.temperature();
    eprintln!("T = {t} arch = {}", std::env::consts::ARCH);
    let prepared = laya.prepare(&texts).unwrap();
    let mut fin = std::io::BufWriter::new(std::fs::File::create(out.join("final.f32")).unwrap());
    let mut rows = Vec::new();
    for (i, pr) in prepared.iter().enumerate() {
        let mut fb: Vec<f32> = Vec::new();
        let z = laya
            .forward_row(pr.ids(), pr.markers(), QType::Choice, |name, block| {
                if name == "final" {
                    fb = block.to_vec();
                }
            })
            .unwrap();
        let bytes: Vec<u8> = fb.iter().flat_map(|v| v.to_le_bytes()).collect();
        fin.write_all(&bytes).unwrap();
        let p = aprender_decide::laya::temperature::softmax_t(&z, t);
        let alt: Vec<Vec<f64>> = extra
            .iter()
            .map(|&te| {
                aprender_decide::laya::temperature::softmax_t(&z, te)
                    .iter()
                    .map(|&v| f64::from(v))
                    .collect()
            })
            .collect();
        rows.push(json!({
            "row": i, "len": pr.ids().len(), "final_len": fb.len(),
            "ids": pr.ids(), "markers": pr.markers(),
            "z": z.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
            "p": p.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
            "p_alt": alt,
        }));
    }
    fin.flush().unwrap();
    std::fs::write(
        out.join("all.json"),
        serde_json::to_vec(&json!({"t": t, "extra_t": extra, "arch": std::env::consts::ARCH, "rows": rows})).unwrap(),
    )
    .unwrap();
    eprintln!("wrote {} ({} rows)", out.display(), texts.len());
}
