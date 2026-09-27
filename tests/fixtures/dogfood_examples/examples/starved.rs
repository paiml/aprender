//! Classification witness: `fail` whose first line is not the cause. The cite
//! must be the resource line, as when a full disk kills rustc mid-build.
fn main() {
    eprintln!("error: could not compile `fixture` (lib)");
    eprintln!("error: No space left on device (os error 28)");
    std::process::exit(101);
}
