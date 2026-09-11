//! Attached by `#[path = "../attached.rs"] mod bolted;` from src/deep/mod.rs,
//! so its row must name `deep::bolted` — the DECLARED name, not the file name.
pub fn attached_reader() -> bool {
    std::fs::read_to_string("scripts/attached_baseline.txt").is_ok()
}

#[cfg(test)]
mod tests {
    #[test]
    fn attached_module_reads_the_tree() {
        let _ = std::fs::read_to_string("scripts/attached_baseline.txt");
    }
}
