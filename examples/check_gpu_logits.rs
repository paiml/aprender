//! Check GPU logits at specific positions
#[cfg(feature = "cuda")]
use realizar::gguf::OwnedQuantizedModelCuda;
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, QuantizedGenerateConfig};

fn main() {
    let path = "/home/noah/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(path).expect("load");

    // CPU reference
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("parse");
    let tokens = vec![791u32];
    let cpu_logits = model.forward(&tokens).expect("CPU forward");

    println!("CPU reference:");
    println!("  logit[16] = {:.4}", cpu_logits[16]);
    println!("  logit[74403] = {:.4}", cpu_logits[74403]);

    let cpu_argmax = cpu_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    println!("  argmax = {} (logit: {:.4})", cpu_argmax.0, cpu_argmax.1);

    #[cfg(feature = "cuda")]
    {
        // Get GPU logits directly - need to expose a method for this
        // For now, let's use the workspace path with debug
        std::env::set_var("GPU_DEBUG", "1");
        std::env::set_var("CUDA_GRAPH_DISABLE", "1");

        println!("\nGPU path (with debug):");
        let model = OwnedQuantizedModel::from_mapped(&mapped).expect("parse");
        let mut cuda = OwnedQuantizedModelCuda::new(model, 0).expect("cuda");

        let config = QuantizedGenerateConfig {
            max_tokens: 1,
            temperature: 0.0,
            top_k: 1,
            ..Default::default()
        };
        let output = cuda
            .generate_gpu_resident(&tokens, &config)
            .expect("GPU gen");
        println!("GPU token: {:?}", output.last());
    }
}
