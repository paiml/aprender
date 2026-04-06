//! Check raw GGUF bias tensors
use realizar::gguf::MappedGGUFModel;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = "/home/noah/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct-GGUF/snapshots/198f08841147e5196a6a69bd0053690fb1fd3857/qwen2-0_5b-instruct-q4_0.gguf";
    let mapped = MappedGGUFModel::from_path(path)?;

    println!("=== Raw GGUF Bias Tensors ===\n");

    // Find Q, K, V bias tensors for layer 0
    for tensor in &mapped.model.tensors {
        if tensor.name.contains("blk.0") && tensor.name.contains("bias") {
            println!(
                "{}: dims={:?}, qtype={}",
                tensor.name, tensor.dims, tensor.qtype
            );

            // Load the raw data
            let data = mapped.model.get_tensor_f32(&tensor.name, mapped.data())?;

            let norm: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
            let sum: f32 = data.iter().sum();
            let max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let min = data.iter().copied().fold(f32::INFINITY, f32::min);

            println!("  len: {}, norm: {:.4}, sum: {:.4}", data.len(), norm, sum);
            println!("  max: {:.4}, min: {:.4}", max, min);
            println!("  first 8: {:?}", &data[..8.min(data.len())]);
            println!();
        }
    }

    Ok(())
}
