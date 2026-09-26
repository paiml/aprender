//! Classification witness: declared `needs-tty`. stdin is /dev/null, so raw mode
//! fails with ENXIO, the message crossterm surfaces.
fn main() {
    eprintln!(
        "Error: Os {{ code: 6, kind: Uncategorized, message: \"No such device or address\" }}"
    );
    std::process::exit(1);
}
