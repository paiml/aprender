//! Check GGUF file metadata
use realizar::gguf::GGUFModel;
use std::{env, fs};

fn main() {
    let args: Vec<String> = env::args().collect();
    let path = args.get(1).map_or(
        "/home/noah/src/aprender/tinyllama-1.1b-chat-v1.0.Q4_0.gguf",
        |s| s.as_str(),
    );
    let data = fs::read(path).expect("Failed to read file");
    let model = GGUFModel::from_bytes(&data).expect("test");

    println!("GGUF Metadata:");
    for (key, value) in &model.metadata {
        // Print up to 100 chars of each value
        let value_str = format!("{:?}", value);
        let truncated = if value_str.len() > 100 {
            format!("{}...", &value_str[..100])
        } else {
            value_str
        };
        println!("  {}: {}", key, truncated);
    }
}
