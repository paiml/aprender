//! A reader under a 2018-layout parent -> the row's module is `flat::child`
//! (the declaring file is src/flat.rs, a sibling of the src/flat/ directory).
pub fn child_reader() -> bool {
    std::fs::read_to_string("scripts/child_baseline.txt").is_ok()
}

#[cfg(test)]
mod tests {
    #[test]
    fn child_module_reads_the_tree() {
        let _ = std::fs::read_to_string("scripts/child_baseline.txt");
    }
}
