use realizar::gguf::MappedGGUFModel;

fn main() {
    let model_path = std::env::args().nth(1).unwrap();
    let mapped = MappedGGUFModel::from_path(&model_path).unwrap();

    println!("Layer 4 tensor types:");
    for tensor in &mapped.model.tensors {
        if tensor.name.contains(".4.") || tensor.name.contains("blk.4") {
            println!(
                "  {} -> {:?} ({:?})",
                tensor.name, tensor.qtype, tensor.dims
            );
        }
    }

    println!("\nAll unique qtypes in model:");
    let mut qtypes: std::collections::HashSet<_> = std::collections::HashSet::new();
    for tensor in &mapped.model.tensors {
        qtypes.insert(tensor.qtype);
    }
    for qtype in qtypes {
        println!("  {:?}", qtype);
    }
}
