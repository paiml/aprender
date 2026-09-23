// usage: coredec <gguf> <outdir> <name>...   — decodes each named tensor through
// aprender-core's GgufReader::get_tensor_f32 (the path tensor_contract uses).
// Writes <outdir>/<i>.f32 and prints "i name shape" or "i name ERR msg".
use std::io::Write;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let r = aprender::format::gguf::GgufReader::from_file(&a[1]).expect("open");
    for (i, name) in a[3..].iter().enumerate() {
        match r.get_tensor_f32(name) {
            Ok((data, shape)) => {
                let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
                std::fs::File::create(format!("{}/{i}.f32", a[2])).unwrap().write_all(&bytes).unwrap();
                println!("{i}\t{name}\t{shape:?}");
            }
            Err(e) => println!("{i}\t{name}\tERR {e}"),
        }
    }
}
