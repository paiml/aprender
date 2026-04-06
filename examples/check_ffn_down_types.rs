use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn main() {
    let path = "/tmp/parity-bench/tinyllama-1.1b-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(path).expect("Failed");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("test");

    for layer_idx in 0..5 {
        let layer = &model.layers()[layer_idx];
        println!(
            "Layer {} ffn_down: in_dim={}, out_dim={}, qtype={}, data_len={}",
            layer_idx,
            layer.ffn_down_weight.in_dim,
            layer.ffn_down_weight.out_dim,
            layer.ffn_down_weight.qtype,
            layer.ffn_down_weight.data.len()
        );
    }
}
