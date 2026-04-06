//! Check tensor names in Qwen2 GGUF file
use realizar::gguf::MappedGGUFModel;

fn main() {
    let path = "../aprender/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(path).expect("load");

    eprintln!("=== All tensor names containing 'blk.0' ===");
    for tensor in &mapped.model.tensors {
        if tensor.name.contains("blk.0.") {
            eprintln!(
                "  {} : {:?} bytes={}",
                tensor.name, tensor.dims, tensor.qtype
            );
        }
    }

    eprintln!("\n=== Checking specific QKV tensor names ===");
    for suffix in [
        "attn_qkv.weight",
        "attn_q.weight",
        "attn_k.weight",
        "attn_v.weight",
    ] {
        let name = format!("blk.0.{}", suffix);
        if let Some(tensor) = mapped.model.tensors.iter().find(|t| t.name == name) {
            eprintln!(
                "  FOUND: {} : {:?} bytes={}",
                tensor.name, tensor.dims, tensor.qtype
            );
        } else {
            eprintln!("  NOT FOUND: {}", name);
        }
    }

    // Also check token embedding
    eprintln!("\n=== Token embedding ===");
    if let Some(tensor) = mapped
        .model
        .tensors
        .iter()
        .find(|t| t.name == "token_embd.weight")
    {
        eprintln!(
            "  {} : {:?} bytes={}",
            tensor.name, tensor.dims, tensor.qtype
        );
    }
}
