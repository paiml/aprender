//! Check V weight structure

use realizar::gguf::{MappedGGUFModel, OwnedQKVWeights, OwnedQuantizedModel};

fn main() {
    let path = "/tmp/parity-bench/tinyllama-1.1b-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(path).expect("Failed");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("test");

    let layer = &model.layers()[0];

    let OwnedQKVWeights::Separate {
        q: q_weight,
        k: k_weight,
        v: v_weight,
    } = &layer.qkv_weight
    else {
        panic!("Expected separate QKV")
    };

    println!(
        "Q weight: in_dim={}, out_dim={}, qtype={}, data_len={}",
        q_weight.in_dim,
        q_weight.out_dim,
        q_weight.qtype,
        q_weight.data.len()
    );
    println!(
        "K weight: in_dim={}, out_dim={}, qtype={}, data_len={}",
        k_weight.in_dim,
        k_weight.out_dim,
        k_weight.qtype,
        k_weight.data.len()
    );
    println!(
        "V weight: in_dim={}, out_dim={}, qtype={}, data_len={}",
        v_weight.in_dim,
        v_weight.out_dim,
        v_weight.qtype,
        v_weight.data.len()
    );

    // Check expected sizes
    // Q4_K: 144 bytes per 256 elements
    println!("\nExpected data sizes:");
    println!("  Q (2048x2048): {} bytes", (2048 * 2048 / 256) * 144);
    println!("  K (2048x256): {} bytes", (2048 * 256 / 256) * 144);
    println!("  V (2048x256): {} bytes", (2048 * 256 / 256) * 144);

    // Check Q6_K size
    // Q6_K: 210 bytes per 256 elements
    println!("\nQ6_K data sizes:");
    println!("  Q (2048x2048): {} bytes", (2048 * 2048 / 256) * 210);
    println!("  K (2048x256): {} bytes", (2048 * 256 / 256) * 210);
    println!("  V (2048x256): {} bytes", (2048 * 256 / 256) * 210);

    // Check first bytes of each weight's data
    println!("\nV weight first 20 bytes: {:02x?}", &v_weight.data[..20]);
    println!("K weight first 20 bytes: {:02x?}", &k_weight.data[..20]);
}
