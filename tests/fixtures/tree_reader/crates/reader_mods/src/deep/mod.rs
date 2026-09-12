//! A reader in a nested `mod.rs` -> the row's module is `deep`.
mod leaf;

#[cfg(test)]
#[path = "../attached.rs"]
mod bolted;

pub fn dir_reader() -> bool {
    std::fs::read_to_string("scripts/deep_baseline.txt").is_ok()
}

#[cfg(test)]
mod tests {
    #[test]
    fn dir_module_reads_the_tree() {
        let _ = std::fs::read_to_string("scripts/deep_baseline.txt");
    }
}
