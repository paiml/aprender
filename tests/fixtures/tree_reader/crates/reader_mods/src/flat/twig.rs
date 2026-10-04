//! A reader under a 2018-layout parent (`src/flat.rs` declares it) -> the row's
//! module is `flat::twig`, not the whole crate.
pub fn twig_reader() -> bool {
    std::fs::read_to_string("scripts/twig_baseline.txt").is_ok()
}

#[cfg(test)]
mod tests {
    #[test]
    fn twig_module_reads_the_tree() {
        let _ = std::fs::read_to_string("scripts/twig_baseline.txt");
    }
}
