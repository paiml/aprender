//! A reader in a leaf file -> the row's module is `deep::leaf`.
pub fn leaf_reader() -> bool {
    std::fs::read_to_string("scripts/leaf_baseline.txt").is_ok()
}

#[cfg(test)]
mod tests {
    #[test]
    fn leaf_module_reads_the_tree() {
        let _ = std::fs::read_to_string("scripts/leaf_baseline.txt");
    }
}
