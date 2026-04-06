use realizar::gguf::GGUFModel;
use std::fs;

fn main() {
    let data =
        fs::read("/home/noah/src/aprender/tinyllama-1.1b-chat-v1.0.Q4_0.gguf").expect("test");
    let model = GGUFModel::from_bytes(&data).expect("test");

    println!("Tensor dimensions (first layer Q/K/V and embedding):");
    for t in &model.tensors {
        if t.name.contains("token_embd")
            || t.name.contains("output.weight")
            || t.name.contains("blk.0.")
        {
            println!("  {}: dims={:?}, qtype={}", t.name, t.dims, t.qtype);
        }
    }
}
