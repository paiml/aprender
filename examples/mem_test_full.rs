//! Full memory trace for Qwen2 chat session
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
    println!("=== Memory Trace ===\n");
    println!("1. Initial:              {:.1} MB", get_mem_mb());
    
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
    
    println!("2. After config:         {:.1} MB", get_mem_mb());
    
    let mut model = Qwen2Model::new(&config);
    println!("3. After model creation: {:.1} MB", get_mem_mb());
    
    // Simulate what chat does - forward pass with small input
    let input_ids: Vec<u32> = vec![1, 2, 3, 4, 5];
    let position_ids: Vec<usize> = (0..5).collect();
    
    println!("4. Before forward pass:  {:.1} MB", get_mem_mb());
    
    let _logits = model.forward(&input_ids, &position_ids);
    println!("5. After forward pass:   {:.1} MB", get_mem_mb());
    
    // Try generate
    println!("6. Before generate:      {:.1} MB", get_mem_mb());
    let _output = model.generate(&input_ids, 10, 0.7, 0.9);
    println!("7. After generate:       {:.1} MB", get_mem_mb());
    
    println!("\n=== Five Whys Analysis ===");
    println!("If OOM occurs, check which step caused the spike.");
}
