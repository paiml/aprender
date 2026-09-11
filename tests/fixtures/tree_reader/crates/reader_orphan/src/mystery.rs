//! No `mod mystery;` declares this file and no include!() names it: the module
//! is UNRESOLVABLE, so the row falls back to the whole crate and the
//! derivation prints WARN unresolved-include.
pub fn mystery_reader() -> bool {
    std::fs::read_to_string("scripts/mystery_baseline.txt").is_ok()
}

#[cfg(test)]
mod tests {
    #[test]
    fn mystery_module_reads_the_tree() {
        let _ = std::fs::read_to_string("scripts/mystery_baseline.txt");
    }
}
