//! An include!()-pulled reader: no `mod part;` declares this file, so its row
//! must name the includer's module (`inc`), not `gen::part`.
pub fn included_reader() -> bool {
    std::fs::read_to_string("scripts/part_baseline.txt").is_ok()
}

#[cfg(test)]
mod part_tests {
    #[test]
    fn included_module_reads_the_tree() {
        let _ = std::fs::read_to_string("scripts/part_baseline.txt");
    }
}
