//! Classification witness: `pass`. Declared needs-data, yet exits 0 -- a
//! declaration never turns a success into a skip.
fn main() {
    println!("ran without the data after all");
}
