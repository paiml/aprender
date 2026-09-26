//! Classification witness: `fail` at the BUILD stage. Declared needs-data; a
//! declaration never excuses an example that does not compile.
fn main() {
    let n: u32 = "not a number";
    println!("{n}");
}
