//! Classification witness: `fail`. Exit code 3 is not 124 and not 137, and the
//! first stderr line is what the TSV must cite.
fn main() {
    eprintln!("dogfood-examples fixture: deliberate failure");
    std::process::exit(3);
}
