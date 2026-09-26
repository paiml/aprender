//! Classification witness: `fail`. Declared long-running, but dies at once. A
//! server that cannot stay up is the defect the declaration must not excuse.
fn main() {
    eprintln!("Error: Os {{ code: 98, kind: AddrInUse, message: \"Address already in use\" }}");
    std::process::exit(1);
}
