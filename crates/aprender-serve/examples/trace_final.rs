//! Trace final forward path
#[cfg(feature = "cuda")]
use realizar::gguf::OwnedQuantizedModelCuda;
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, QuantizedGenerateConfig};

fn main() {
    let path = "/home/noah/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(path).expect("load");

    // Get CPU results
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("parse");
    let config = QuantizedGenerateConfig {
        max_tokens: 1,
        temperature: 0.0,
        top_k: 1,
        ..Default::default()
    };

    // Run full CPU forward
    let tokens = vec![791u32];
    let cpu_logits = model.forward(&tokens).expect("CPU forward");

    println!("CPU logits stats:");
    let cpu_argmax = cpu_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    println!(
        "  argmax: {} (logit: {:.4})",
        cpu_argmax, cpu_logits[cpu_argmax]
    );
    println!("  sum: {:.4}", cpu_logits.iter().sum::<f32>());
    println!("  first 5: {:?}", &cpu_logits[..5]);

    // Check specific tokens
    println!("  logit[16]: {:.4}", cpu_logits[16]);
    println!("  logit[74403]: {:.4}", cpu_logits[74403]);

    // Look at tokens near 74403 to understand the pattern
    println!("\nLogits near 74403:");
    for i in [74400, 74401, 74402, 74403, 74404, 74405] {
        println!("  logit[{}]: {:.4}", i, cpu_logits[i]);
    }

    #[cfg(feature = "cuda")]
    {
        println!("\n--- GPU Path ---");
        let model = OwnedQuantizedModel::from_mapped(&mapped).expect("parse");
        let mut cuda = OwnedQuantizedModelCuda::new(model, 0).expect("cuda");

        let gpu_out = cuda
            .generate_gpu_resident(&tokens, &config)
            .expect("GPU gen");
        println!("GPU generated token: {:?}", gpu_out.last());
    }
}
