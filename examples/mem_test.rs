//! Memory usage test for Qwen2 model
use aprender::demo::Qwen2Config;
use aprender::models::Qwen2Model;

fn get_mem_mb() -> f64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in status.lines() {
        if line.starts_with("VmRSS:") {
            let kb: f64 = line.split_whitespace().nth(1).and_then(|s| s.parse().ok()).unwrap_or(0.0);
            return kb / 1024.0;
        }
    }
    0.0
}

fn main() {
    println!("Initial memory: {:.1} MB", get_mem_mb());
    
    let config = Qwen2Config {
        hidden_size: 896,
        num_attention_heads: 14,
        num_kv_heads: 2,
        num_layers: 24,
        vocab_size: 151936,
        max_seq_len: 32768,
        intermediate_size: 4864,
        rope_theta: 1_000_000.0,
    };
    
    println!("Creating model with config: {:?}", config);
    println!("Expected ~2.5 GB...");
    
    let _model = Qwen2Model::new(&config);
    println!("After model creation: {:.1} MB", get_mem_mb());
    
    println!("Done!");
}
