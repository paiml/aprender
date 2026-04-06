//! Check GGUF metadata
use realizar::gguf::GGUFModel;
use std::fs;

fn main() {
    let path = "../aprender/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf";
    let data = fs::read(path).expect("read file");
    let model = GGUFModel::from_bytes(&data).expect("parse");

    eprintln!("=== All metadata keys ===");
    for (k, v) in &model.metadata {
        eprintln!("{}: {:?}", k, v);
    }
}
