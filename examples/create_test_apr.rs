//! Generate test APR v2 file for CLI testing
use aprender::format::v2::{AprV2Metadata, AprV2Writer, TensorDType};

fn main() {
    let mut metadata = AprV2Metadata::new("whisper");
    metadata.name = Some("whisper-tiny-test".to_string());
    metadata.description = Some("Test Whisper model for APR v2 serving".to_string());

    let mut writer = AprV2Writer::new(metadata);

    // Helper to create f32 tensor bytes
    fn f32_bytes(vals: &[f32]) -> Vec<u8> {
        vals.iter().flat_map(|f| f.to_le_bytes()).collect()
    }

    // Encoder layers
    for i in 0..2 {
        let prefix = format!("encoder.layers.{}", i);
        writer.add_tensor(
            &format!("{}.self_attn.q_proj.weight", prefix),
            TensorDType::F32,
            vec![384, 384],
            f32_bytes(&vec![0.01; 384 * 384]),
        );
        writer.add_tensor(
            &format!("{}.self_attn.k_proj.weight", prefix),
            TensorDType::F32,
            vec![384, 384],
            f32_bytes(&vec![0.01; 384 * 384]),
        );
        writer.add_tensor(
            &format!("{}.self_attn.v_proj.weight", prefix),
            TensorDType::F32,
            vec![384, 384],
            f32_bytes(&vec![0.01; 384 * 384]),
        );
        writer.add_tensor(
            &format!("{}.self_attn.out_proj.weight", prefix),
            TensorDType::F32,
            vec![384, 384],
            f32_bytes(&vec![0.01; 384 * 384]),
        );
        writer.add_tensor(
            &format!("{}.mlp.fc1.weight", prefix),
            TensorDType::F32,
            vec![1536, 384],
            f32_bytes(&vec![0.01; 1536 * 384]),
        );
        writer.add_tensor(
            &format!("{}.mlp.fc2.weight", prefix),
            TensorDType::F32,
            vec![384, 1536],
            f32_bytes(&vec![0.01; 384 * 1536]),
        );
    }

    // Decoder layers
    for i in 0..2 {
        let prefix = format!("decoder.layers.{}", i);
        // Self attention
        writer.add_tensor(
            &format!("{}.self_attn.q_proj.weight", prefix),
            TensorDType::F32,
            vec![384, 384],
            f32_bytes(&vec![0.02; 384 * 384]),
        );
        writer.add_tensor(
            &format!("{}.self_attn.k_proj.weight", prefix),
            TensorDType::F32,
            vec![384, 384],
            f32_bytes(&vec![0.02; 384 * 384]),
        );
        writer.add_tensor(
            &format!("{}.self_attn.v_proj.weight", prefix),
            TensorDType::F32,
            vec![384, 384],
            f32_bytes(&vec![0.02; 384 * 384]),
        );
        writer.add_tensor(
            &format!("{}.self_attn.out_proj.weight", prefix),
            TensorDType::F32,
            vec![384, 384],
            f32_bytes(&vec![0.02; 384 * 384]),
        );
        // Cross attention
        writer.add_tensor(
            &format!("{}.cross_attn.q_proj.weight", prefix),
            TensorDType::F32,
            vec![384, 384],
            f32_bytes(&vec![0.03; 384 * 384]),
        );
        writer.add_tensor(
            &format!("{}.cross_attn.k_proj.weight", prefix),
            TensorDType::F32,
            vec![384, 384],
            f32_bytes(&vec![0.03; 384 * 384]),
        );
        writer.add_tensor(
            &format!("{}.cross_attn.v_proj.weight", prefix),
            TensorDType::F32,
            vec![384, 384],
            f32_bytes(&vec![0.03; 384 * 384]),
        );
        writer.add_tensor(
            &format!("{}.cross_attn.out_proj.weight", prefix),
            TensorDType::F32,
            vec![384, 384],
            f32_bytes(&vec![0.03; 384 * 384]),
        );
        // FFN
        writer.add_tensor(
            &format!("{}.mlp.fc1.weight", prefix),
            TensorDType::F32,
            vec![1536, 384],
            f32_bytes(&vec![0.02; 1536 * 384]),
        );
        writer.add_tensor(
            &format!("{}.mlp.fc2.weight", prefix),
            TensorDType::F32,
            vec![384, 1536],
            f32_bytes(&vec![0.02; 384 * 1536]),
        );
    }

    let apr_bytes = writer.write().expect("Failed to write APR");
    std::fs::write("/tmp/test-whisper-v2.apr", &apr_bytes).expect("Failed to save");
    println!("Created /tmp/test-whisper-v2.apr ({} bytes)", apr_bytes.len());
}
