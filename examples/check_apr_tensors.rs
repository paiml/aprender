//! Check APR tensor names and dtypes
use realizar::apr::AprV2Model;
use std::env;

fn main() {
    let path = env::args().nth(1).unwrap_or_else(|| {
        eprintln!("Usage: check_apr_tensors <model.apr>");
        std::process::exit(1);
    });

    let model = AprV2Model::load(&path).expect("Failed to load APR model");
    println!("Tensors: {}", model.tensor_count());
    println!("Metadata: {:?}", model.metadata());
    println!("\nFirst 50 tensors:");
    for name in model.tensor_names().iter().take(50) {
        if let Some(t) = model.get_tensor(name) {
            println!("  {}: dtype={} shape={:?}", name, t.dtype, t.shape);
        }
    }
}
