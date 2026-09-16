//! Classification witness: `needs-feature`.
//!
//! The real rows this stands for, from examples-nightly run 34826552378:
//!   aprender-db::compressed_kv        This example requires the 'compression' feature.
//!   aprender-rag::semantic_embeddings This example requires the 'embeddings' feature.
//!   aprender-distribute::tensor_example  Error: This example requires the 'tensor' feature.
fn main() {
    eprintln!("Error: This example requires the 'compression' feature.");
    std::process::exit(1);
}
