// Quick check: where are non-zero values in the embedding matrix?
use realizar::safetensors::MappedSafeTensorsModel;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = Path::new("/home/noah/models/qwen2.5-coder-0.5b-instruct/model.safetensors");
    let st = MappedSafeTensorsModel::load(path)?;

    // Get raw embedding data
    let emb = st.get_tensor_auto("model.embed_tokens.weight")?;

    println!("Data len: {} floats", emb.len());
    // Expected: 151936 * 896 = 136,134,656

    // Find first non-zero value
    for (i, &v) in emb.iter().enumerate() {
        if v.abs() > 0.01 {
            let hidden_dim = 896;
            let row = i / hidden_dim;
            let col = i % hidden_dim;
            println!(
                "First non-zero at index {}: value={:.4}, row={}, col={}",
                i, v, row, col
            );
            break;
        }
    }

    // Find where the data actually starts being non-zero
    let mut zero_count = 0;
    for &v in &emb {
        if v.abs() < 0.0001 {
            zero_count += 1;
        } else {
            break;
        }
    }
    println!(
        "Leading zeros: {} ({:.1}% of total)",
        zero_count,
        100.0 * zero_count as f64 / emb.len() as f64
    );

    // Check middle of matrix
    let mid = emb.len() / 2;
    println!("Middle 10 values (at {}): {:?}", mid, &emb[mid..mid + 10]);

    // Check end of matrix
    let end = emb.len() - 10;
    println!("Last 10 values: {:?}", &emb[end..]);

    Ok(())
}
