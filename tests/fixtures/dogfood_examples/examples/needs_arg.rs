//! Classification witness: `needs-args`. Clap-free on purpose (the fixture has
//! no dependencies) but emits the same `Usage:` first line clap does, because
//! the classifier keys on that wording and not on a crate.
fn main() {
    if std::env::args().count() < 2 {
        eprintln!("Usage: needs_arg <FILE>");
        std::process::exit(2);
    }
}
