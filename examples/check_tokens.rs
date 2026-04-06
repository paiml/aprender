use realizar::gguf::MappedGGUFModel;

fn main() {
    let path = "/home/noah/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf";
    let mapped = MappedGGUFModel::from_path(path).expect("load");

    let prompt = "Hello";
    let tokens = mapped.model.encode(prompt).unwrap_or_default();
    println!("'{}' -> {:?}", prompt, tokens);

    // Also check token 791
    let decode_791 = mapped.model.decode(&[791]);
    println!("791 -> '{}'", decode_791);

    // Check what forward of [791] vs forward of Hello tokens produces
}
