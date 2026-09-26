//! Classification witness: declared `needs-args`. A lowercase `usage:` line, in
//! the example's own words, that the clap-anchored pattern does not match.
fn main() {
    eprintln!("usage: decl_args <model.gguf> <budget>");
    std::process::exit(2);
}
