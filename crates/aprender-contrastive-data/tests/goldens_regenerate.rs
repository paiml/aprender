//! Re-baseline the committed selection goldens under `tests/goldens/`.
//!
//! ```text
//! cargo test -p aprender-contrastive-data --test goldens_regenerate -- --ignored
//! ```
//!
//! IGNORED BY DEFAULT, AND WHY IT IS A TEST AT ALL
//! -----------------------------------------------
//! A golden whose regeneration procedure lives only in a summary is a golden nobody can
//! re-baseline reviewably; the next person hand-edits the file instead, and the diff stops
//! meaning anything. Keeping the generator in-tree makes a re-baseline a reviewed diff
//! produced by a named command. It is `#[ignore]`d so an ordinary `cargo test` can never
//! overwrite the artifacts it is supposed to be checking.
//!
//! THIS IS NOT A D-04 BOUNDARY VIOLATION
//! -------------------------------------
//! `make contrastive-data-boundary` bans `std::fs` under `src/`, with no `cfg(test)`
//! exemption. This file is under `tests/`, outside that scan and outside the library
//! consumers link against. The library's own verifier — `manifest.rs`'s `golden_tests` —
//! reads the same bytes through `include_bytes!` and touches no filesystem at all.
//!
//! THE GENERATOR AND THE VERIFIER MUST AGREE
//! -----------------------------------------
//! Both build the dataset from the SAME three committed `golden_corpus_*.jsonl` files
//! through the same public API, so the only thing duplicated here is the declaration
//! block (three class-count vectors and the label map), which the verifier asserts against
//! the corpus in `golden_corpus_has_the_shape_the_goldens_were_derived_from`.

use std::fs;
use std::path::{Path, PathBuf};

use aprender_contrastive_data::ledger::AccessLedger;
use aprender_contrastive_data::manifest::dump_pairs;
use aprender_contrastive_data::pairs::{PairConfig, PairSampler};
use aprender_contrastive_data::prepared::{Canonical, CanonicalDeclarations, PreparedDataset};
use aprender_contrastive_data::schema::parse_jsonl_bytes;
use aprender_contrastive_data::select::{FewShotSelector, SelectionConfig};
use aprender_contrastive_data::split::SplitDeclaration;
use sha2::{Digest, Sha256};

/// `(root_seed, shots_per_class)` for every committed golden.
const CASES: [(u64, u32); 4] = [(13, 8), (13, 16), (17, 8), (17, 16)];

/// `(root_seed, shots_per_class)` for every committed PAIR golden (plan 02-07 Task 3).
///
/// These must be regenerated alongside the selection goldens, not separately: the manifest
/// this file writes covers every name in `names`, so omitting the pair goldens here would
/// silently drop them from `manifest.sha256` on the next re-baseline and turn the
/// `include_bytes!` verifier red for a reason that has nothing to do with drift.
const PAIR_CASES: [(u64, u32); 2] = [(13, 8), (17, 8)];

/// How many pairs each committed pair golden holds. Must equal `PAIR_GOLDEN_PREFIX` in
/// `src/manifest.rs`'s verifier.
const PAIR_PREFIX: u64 = 32;

const CORPUS_FILES: [&str; 3] = [
    "golden_corpus_train.jsonl",
    "golden_corpus_validation.jsonl",
    "golden_corpus_test.jsonl",
];

fn goldens_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("goldens")
}

fn declarations() -> CanonicalDeclarations {
    let label_names: Vec<String> = ["none", "against", "favor"]
        .iter()
        .map(|name| (*name).to_string())
        .collect();
    let decl = |counts: Vec<usize>| SplitDeclaration {
        expected_class_counts: counts,
        label_names: label_names.clone(),
    };
    CanonicalDeclarations {
        train: decl(vec![20, 20, 20]),
        validation: decl(vec![1, 1, 1]),
        test: decl(vec![1, 1, 1]),
        label_names,
    }
}

fn dataset(ledger: &mut AccessLedger) -> PreparedDataset<Canonical> {
    let dir = goldens_dir();
    let read = |name: &str, role: &str| {
        let bytes = fs::read(dir.join(name)).unwrap_or_else(|e| panic!("read {name}: {e}"));
        parse_jsonl_bytes(&bytes, role).unwrap_or_else(|e| panic!("parse {name}: {e}"))
    };
    PreparedDataset::<Canonical>::from_labeled_rows(
        read(CORPUS_FILES[0], "train"),
        read(CORPUS_FILES[1], "validation"),
        read(CORPUS_FILES[2], "test"),
        &declarations(),
        ledger,
    )
    .expect("the golden corpus must be a valid canonical dataset")
}

fn golden_name(seed: u64, shots: u32) -> String {
    format!("selection_seed{seed}_shots{shots}.payload.json")
}

fn pair_golden_name(seed: u64, shots: u32) -> String {
    format!("pairs_seed{seed}_shots{shots}_first{PAIR_PREFIX}.jsonl")
}

#[test]
#[ignore = "writes the committed goldens; run explicitly with --ignored to re-baseline"]
fn regenerate_selection_goldens() {
    let dir = goldens_dir();
    let mut names: Vec<String> = CORPUS_FILES.iter().map(|n| (*n).to_string()).collect();

    for (seed, shots) in CASES {
        let mut ledger = AccessLedger::new();
        let prepared = dataset(&mut ledger);
        let selection = FewShotSelector::select(
            &prepared,
            &SelectionConfig {
                root_seed: seed,
                shots_per_class: shots,
            },
            &mut ledger,
        )
        .expect("the golden corpus must support this selection");
        let bytes = selection
            .payload()
            .to_canonical_bytes()
            .expect("payload serializes");
        let name = golden_name(seed, shots);
        fs::write(dir.join(&name), &bytes).unwrap_or_else(|e| panic!("write {name}: {e}"));
        println!("wrote {name}: {} bytes", bytes.len());
        names.push(name);
    }

    for (seed, shots) in PAIR_CASES {
        let mut ledger = AccessLedger::new();
        let prepared = dataset(&mut ledger);
        let selection = FewShotSelector::select(
            &prepared,
            &SelectionConfig {
                root_seed: seed,
                shots_per_class: shots,
            },
            &mut ledger,
        )
        .expect("the golden corpus must support this selection");
        let cfg = PairConfig {
            budget: Some(PAIR_PREFIX),
            ..PairConfig::new(seed)
        };
        let sampler =
            PairSampler::new(&selection, &cfg).expect("the golden corpus supports 32 pairs");
        let mut bytes = Vec::new();
        dump_pairs(&sampler, &mut bytes).expect("dumping to a Vec cannot fail");
        let name = pair_golden_name(seed, shots);
        fs::write(dir.join(&name), &bytes).unwrap_or_else(|e| panic!("write {name}: {e}"));
        println!("wrote {name}: {} bytes", bytes.len());
        names.push(name);
    }

    // The manifest covers every file in the directory EXCEPT itself.
    let mut manifest = String::new();
    names.sort();
    for name in &names {
        let bytes = fs::read(dir.join(name)).unwrap_or_else(|e| panic!("read {name}: {e}"));
        let digest = Sha256::digest(&bytes);
        manifest.push_str(&format!("{digest:x}  {name}\n"));
    }
    fs::write(dir.join("manifest.sha256"), manifest.as_bytes()).expect("write manifest.sha256");
    println!("manifest.sha256 covers {} files", names.len());
}
