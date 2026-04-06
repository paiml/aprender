use realizar::gguf::GGUFModel;
use std::{env, fs};

fn main() {
    let args: Vec<String> = env::args().collect();
    let path = args.get(1).map_or(
        "/home/noah/src/aprender/tinyllama-1.1b-chat-v1.0.Q4_0.gguf",
        |s| s.as_str(),
    );
    let data = fs::read(path).expect("Failed to read file");
    let model = GGUFModel::from_bytes(&data).expect("test");

    println!("Key tensor dimensions (after dims.reverse()):");
    for t in &model.tensors {
        if t.name == "token_embd.weight"
            || t.name == "output.weight"
            || t.name == "output_norm.weight"
            || t.name.contains("blk.0.attn_q")
            || t.name.contains("blk.0.attn_k")
            || t.name.contains("blk.0.attn_v")
            || t.name.contains("blk.0.attn_output")
            || t.name.contains("blk.0.attn_norm")
            || t.name.contains("blk.0.ffn")
        {
            println!("  {}: dims={:?}, qtype={}", t.name, t.dims, t.qtype);
        }
    }
}
