//! PP-QUANT-001 Phase 0 gate — binds `contracts/quant-dispatch-completeness-v1.yaml`
//! to the tree (docs/specifications/PP-QUANT-001-MASTER.md §2.2, §3).
//!
//! PO-QDC-002 (removed ids rejected), PO-QDC-003 (block-format consts equal their
//! rows) and PO-QDC-005 (the dispatch-site oracle agrees with the inventory) are
//! checkable before the table exists and are enforced here. PO-QDC-001 and
//! PO-QDC-004 need the Phase 1 table and grids; the contract lists them as owed.

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::path::{Path, PathBuf};

    use serde::Deserialize;

    use crate::gguf::GgmlQuantType;
    use crate::quantize::format_trait::{Q4_0Fmt, Q8_0Fmt, QuantBlockFormat, Q4K, Q5K, Q6K};

    const CONTRACT_REL: &str = "contracts/quant-dispatch-completeness-v1.yaml";
    const CRATE_SRC_REL: &str = "crates/aprender-serve/src";

    #[derive(Deserialize)]
    struct Contract {
        type_count: u32,
        removed_ids: Vec<u32>,
        types: Vec<TypeRow>,
        dispatch_sites: Vec<SiteRow>,
    }

    #[derive(Deserialize)]
    struct TypeRow {
        id: u32,
        name: String,
        block: usize,
        bytes: usize,
    }

    #[derive(Deserialize)]
    struct SiteRow {
        file: String,
        #[serde(default)]
        migrated: bool,
    }

    fn repo_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .canonicalize()
            .expect("repo root resolves")
    }

    fn load_contract() -> Contract {
        let path = repo_root().join(CONTRACT_REL);
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        serde_yaml_ng::from_str(&text)
            .unwrap_or_else(|e| panic!("{CONTRACT_REL} does not parse: {e}"))
    }

    #[test]
    fn qdc_enumeration_is_gapless() {
        let c = load_contract();
        let live: BTreeSet<u32> = c.types.iter().map(|t| t.id).collect();
        let removed: BTreeSet<u32> = c.removed_ids.iter().copied().collect();
        assert_eq!(live.len(), c.types.len(), "duplicate ids in `types`");
        assert!(live.is_disjoint(&removed), "an id is both live and removed");
        let all: BTreeSet<u32> = (0..c.type_count).collect();
        let covered: BTreeSet<u32> = live.union(&removed).copied().collect();
        assert_eq!(
            covered, all,
            "every id in [0, type_count) must be a `types` row or a `removed_ids` entry"
        );
        let names: BTreeSet<&str> = c.types.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names.len(), c.types.len(), "duplicate names in `types`");
    }

    /// PO-QDC-002 — today's `GgmlQuantType::from_id` must already refuse the removed ids.
    #[test]
    fn qdc_removed_ids_rejected() {
        let c = load_contract();
        for id in c
            .removed_ids
            .iter()
            .copied()
            .chain(std::iter::once(c.type_count))
        {
            assert!(
                GgmlQuantType::from_id(id).is_none(),
                "GgmlQuantType::from_id({id}) must be None (removed or out of range)"
            );
        }
    }

    /// Every id `GgmlQuantType` already knows must spell its name the way the contract does.
    #[test]
    fn qdc_named_rows_agree_with_ggml_quant_type() {
        let c = load_contract();
        for row in &c.types {
            if let Some(t) = GgmlQuantType::from_id(row.id) {
                assert_eq!(
                    t.as_str(),
                    row.name,
                    "GgmlQuantType names id {} `{}`, the contract names it `{}`",
                    row.id,
                    t.as_str(),
                    row.name
                );
            }
        }
    }

    fn assert_format_matches_row<F: QuantBlockFormat>(c: &Contract) {
        let row = c
            .types
            .iter()
            .find(|t| t.name == F::FORMAT_ID)
            .unwrap_or_else(|| panic!("no contract row named {}", F::FORMAT_ID));
        assert_eq!(
            F::SUPERBLOCK_BYTES,
            row.bytes,
            "{}: SUPERBLOCK_BYTES {} != contract bytes {}",
            F::FORMAT_ID,
            F::SUPERBLOCK_BYTES,
            row.bytes
        );
        assert_eq!(
            F::ELEMENTS_PER_SUPERBLOCK,
            row.block,
            "{}: ELEMENTS_PER_SUPERBLOCK {} != contract block {}",
            F::FORMAT_ID,
            F::ELEMENTS_PER_SUPERBLOCK,
            row.block
        );
    }

    /// PO-QDC-003 — every `QuantBlockFormat` impl's geometry equals its contract row.
    #[test]
    fn qdc_block_format_consts_match_rows() {
        let c = load_contract();
        assert_format_matches_row::<Q4K>(&c);
        assert_format_matches_row::<Q5K>(&c);
        assert_format_matches_row::<Q6K>(&c);
        assert_format_matches_row::<Q4_0Fmt>(&c);
        assert_format_matches_row::<Q8_0Fmt>(&c);
    }

    /// The oracle written beside `dispatch_sites` in the contract, line by line:
    /// `GGUF_TYPE_Q4_K|GGUF_TYPE_Q4_0|match qtype|match.*\.qtype`.
    fn oracle_hits_line(line: &str) -> bool {
        line.contains("GGUF_TYPE_Q4_K")
            || line.contains("GGUF_TYPE_Q4_0")
            || line.contains("match qtype")
            || line
                .find("match")
                .is_some_and(|i| line[i..].contains(".qtype"))
    }

    fn collect_rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
        let entries =
            std::fs::read_dir(dir).unwrap_or_else(|e| panic!("read_dir {}: {e}", dir.display()));
        for entry in entries {
            let path = entry.expect("dir entry").path();
            if path.is_dir() {
                collect_rs_files(&path, out);
            } else if path.extension().is_some_and(|x| x == "rs") {
                out.push(path);
            }
        }
    }

    fn oracle_sites(root: &Path) -> BTreeSet<String> {
        let mut files = Vec::new();
        collect_rs_files(&root.join(CRATE_SRC_REL), &mut files);
        files
            .into_iter()
            .filter_map(|p| {
                let rel = p
                    .strip_prefix(root)
                    .expect("under repo root")
                    .to_string_lossy()
                    .into_owned();
                // `grep -v test` in the oracle drops any path containing "test".
                if rel.contains("test") {
                    return None;
                }
                let src = std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("read {rel}: {e}"));
                src.lines().any(oracle_hits_line).then_some(rel)
            })
            .collect()
    }

    /// PO-QDC-005 — the tree's raw-qtype dispatch sites are exactly the contract's
    /// `migrated: false` rows. A new site fails until it is listed; a row flipped to
    /// `migrated: true` fails until the site really stopped matching on a raw qtype.
    #[test]
    fn qdc_dispatch_sites_match_oracle() {
        let root = repo_root();
        let c = load_contract();
        for s in &c.dispatch_sites {
            assert!(
                root.join(&s.file).is_file(),
                "dispatch_sites lists a missing file: {}",
                s.file
            );
        }
        let listed_open: BTreeSet<String> = c
            .dispatch_sites
            .iter()
            .filter(|s| !s.migrated)
            .map(|s| s.file.clone())
            .collect();
        let found = oracle_sites(&root);

        let unlisted: Vec<&String> = found.difference(&listed_open).collect();
        let stale: Vec<&String> = listed_open.difference(&found).collect();
        assert!(
            unlisted.is_empty() && stale.is_empty(),
            "PO-QDC-005 FAILED — {CONTRACT_REL} `dispatch_sites` and the tree disagree.\n\
             Matching on a raw qtype but NOT listed (a new dispatch site — migrate it to \
             quant_type_traits() or add a row): {unlisted:?}\n\
             Listed as open (migrated: false) but no longer matching (flip the row to \
             migrated: true): {stale:?}"
        );
    }
}
