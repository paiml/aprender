//! A reader whose parent is the 2018-layout sibling src/sib.rs -> `sib::kid`.
pub fn kid_reader() -> bool {
    std::fs::read_to_string("scripts/kid_baseline.txt").is_ok()
}

#[cfg(test)]
mod tests {
    #[test]
    fn kid_module_reads_the_tree() {
        let _ = std::fs::read_to_string("scripts/kid_baseline.txt");
    }
}
