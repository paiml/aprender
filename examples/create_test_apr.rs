//! Generate test APR1 file for CLI testing
use aprender::serialization::apr::AprWriter;
use serde_json::json;

fn main() {
    let mut writer = AprWriter::new();

    // Model metadata
    writer.set_metadata("model_type", json!("whisper"));
    writer.set_metadata("model_size", json!("tiny"));
    writer.set_metadata("n_layers", json!(4));
    writer.set_metadata("n_heads", json!(6));
    writer.set_metadata("n_mels", json!(80));

    // Encoder layers
    for i in 0..2 {
        let prefix = format!("encoder.layers.{}", i);
        writer.add_tensor_f32(
            &format!("{}.self_attn.q_proj.weight", prefix),
            vec![384, 384],
            &vec![0.01; 384 * 384],
        );
        writer.add_tensor_f32(
            &format!("{}.self_attn.k_proj.weight", prefix),
            vec![384, 384],
            &vec![0.01; 384 * 384],
        );
        writer.add_tensor_f32(
            &format!("{}.self_attn.v_proj.weight", prefix),
            vec![384, 384],
            &vec![0.01; 384 * 384],
        );
        writer.add_tensor_f32(
            &format!("{}.self_attn.out_proj.weight", prefix),
            vec![384, 384],
            &vec![0.01; 384 * 384],
        );
        writer.add_tensor_f32(
            &format!("{}.mlp.fc1.weight", prefix),
            vec![1536, 384],
            &vec![0.01; 1536 * 384],
        );
        writer.add_tensor_f32(
            &format!("{}.mlp.fc2.weight", prefix),
            vec![384, 1536],
            &vec![0.01; 384 * 1536],
        );
    }

    // Decoder layers
    for i in 0..2 {
        let prefix = format!("decoder.layers.{}", i);
        // Self attention
        writer.add_tensor_f32(
            &format!("{}.self_attn.q_proj.weight", prefix),
            vec![384, 384],
            &vec![0.02; 384 * 384],
        );
        writer.add_tensor_f32(
            &format!("{}.self_attn.k_proj.weight", prefix),
            vec![384, 384],
            &vec![0.02; 384 * 384],
        );
        writer.add_tensor_f32(
            &format!("{}.self_attn.v_proj.weight", prefix),
            vec![384, 384],
            &vec![0.02; 384 * 384],
        );
        writer.add_tensor_f32(
            &format!("{}.self_attn.out_proj.weight", prefix),
            vec![384, 384],
            &vec![0.02; 384 * 384],
        );
        // Cross attention
        writer.add_tensor_f32(
            &format!("{}.cross_attn.q_proj.weight", prefix),
            vec![384, 384],
            &vec![0.03; 384 * 384],
        );
        writer.add_tensor_f32(
            &format!("{}.cross_attn.k_proj.weight", prefix),
            vec![384, 384],
            &vec![0.03; 384 * 384],
        );
        writer.add_tensor_f32(
            &format!("{}.cross_attn.v_proj.weight", prefix),
            vec![384, 384],
            &vec![0.03; 384 * 384],
        );
        writer.add_tensor_f32(
            &format!("{}.cross_attn.out_proj.weight", prefix),
            vec![384, 384],
            &vec![0.03; 384 * 384],
        );
        // FFN
        writer.add_tensor_f32(
            &format!("{}.mlp.fc1.weight", prefix),
            vec![1536, 384],
            &vec![0.02; 1536 * 384],
        );
        writer.add_tensor_f32(
            &format!("{}.mlp.fc2.weight", prefix),
            vec![384, 1536],
            &vec![0.02; 384 * 1536],
        );
    }

    writer
        .write("/tmp/test-whisper.apr")
        .expect("Failed to write APR file");
    println!("Created /tmp/test-whisper.apr");
}
