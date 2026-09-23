//! #3931: the codebooks embedded in PTX are transcriptions of Rust constants,
//! and nothing compared them.
//!
//! `generate_iq3_s_gemv_ptx` carries `iq3s_grid_g[512]` with the comment
//! "IQ3S_GRID: quantize::iq_grids::IQ3S_GRID, 512 packed 4-byte codebook
//! entries." That is 512 numbers copied by hand, IN A DIFFERENT BASE — the Rust
//! constant is hex, the PTX is decimal — with no test relating them.
//!
//! They agree today; this was measured before the test was written, so this
//! pins a fact rather than fixing a defect. But a wrong codebook entry does not
//! fail to assemble and does not crash: it decodes one weight index to the wrong
//! eight magnitudes, which is a plausible-looking wrong answer in exactly the
//! place the device A/B is least able to localise it.
//!
//! WHY THIS LIVES IN `quantize/` AND NOT NEXT TO THE PTX IT GUARDS.
//! `pub mod cuda` is `#[cfg(feature = "cuda")]`, and per #3902 no gate runs
//! `aprender-serve` under that feature — so a test placed beside `layout.rs`
//! compiles only on a developer's cuda build and is dark in CI. This guard needs
//! no device and no cuda feature: it reads `layout.rs` as TEXT and compares it to
//! a Rust constant. Both work in every build, so it lives where it runs.
//!
//! Measured, not assumed: appending invalid Rust to a file under `cuda/` did not
//! fail `cargo test -p aprender-serve --lib`, while the same edit to a file under
//! `quantize/` failed it immediately.
//!
//! The parser refuses to pass vacuously — it asserts it found 512 entries on
//! both sides, so a renamed symbol or a reformatted table fails loudly instead
//! of comparing two empty lists.

/// Every integer literal inside `.global ... <symbol>[<n>] = { ... };` in the
/// PTX-generating source.
fn ptx_table(symbol: &str) -> Vec<u64> {
    let src = include_str!("../cuda/layout.rs");
    let needle = format!("{symbol}[");
    let start = src.find(&needle).unwrap_or_else(|| {
        panic!("PTX symbol {symbol} not found in layout.rs — if it was renamed, rename it here")
    });
    let open = src[start..].find('{').expect("the table's opening brace") + start;
    let close = src[open..].find('}').expect("the table's closing brace") + open;
    src[open + 1..close]
        .split(|c: char| !c.is_ascii_digit())
        .filter(|t| !t.is_empty())
        .map(|t| t.parse::<u64>().expect("a decimal PTX literal"))
        .collect()
}

#[test]
fn the_ptx_iq3s_grid_equals_the_rust_constant_3931() {
    let ptx = ptx_table("iq3s_grid_g");
    let rust: Vec<u64> = super::iq_grids::IQ3S_GRID
        .iter()
        .map(|&v| u64::from(v))
        .collect();
    assert_eq!(ptx.len(), 512, "parsed {} PTX entries, not 512 — the table's shape changed and this test would compare the wrong thing", ptx.len());
    assert_eq!(rust.len(), 512);
    let diff: Vec<String> = ptx
        .iter()
        .zip(rust.iter())
        .enumerate()
        .filter(|(_, (a, b))| a != b)
        .map(|(i, (a, b))| format!("\n  - index {i}: PTX {a} (0x{a:08x}) vs IQ3S_GRID 0x{b:08x}"))
        .collect();
    assert!(
        diff.is_empty(),
        "{} codebook entr(y/ies) in the PTX disagree with quantize::iq_grids::IQ3S_GRID. \
         A wrong entry assembles cleanly and decodes one index to the wrong eight \
         magnitudes:{}",
        diff.len(),
        diff.join("")
    );
}

/// The comparison must be able to SEE a difference, or the test above counts
/// nothing. A single perturbed entry must be reported, and reported by index.
#[test]
fn the_codebook_comparison_can_detect_one_wrong_entry_3931() {
    let ptx = ptx_table("iq3s_grid_g");
    let mut tampered = ptx.clone();
    tampered[7] ^= 1;
    let diff: Vec<usize> = ptx
        .iter()
        .zip(tampered.iter())
        .enumerate()
        .filter(|(_, (a, b))| a != b)
        .map(|(i, _)| i)
        .collect();
    assert_eq!(
        diff,
        vec![7],
        "the comparison must localise a single wrong entry"
    );
    assert!(!ptx.is_empty(), "and it must have had entries to compare");
}
