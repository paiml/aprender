//! The deterministic, network-free trainer fixture every Phase 3 trainer test builds on.
//!
//! Contract: `setfit-train-lifecycle-v1` (authored in plan 03-06). Requirements: TRN-03,
//! TRN-06.
//!
//! # `#[cfg(test)]`, and it stays that way
//!
//! This module is compiled only for test targets. Plan 03-10's acceptance criteria reject a
//! `#[doc(hidden)]` "test support" door on the shipped surface, because such a door is
//! indistinguishable at the type level from a supported entry point and is exactly how an
//! ungated tuning path reaches a caller. Every consumer here lives in the same crate's test
//! configuration, so no widening is needed and none is taken.
//!
//! # What is REAL here and what is SYNTHETIC — the distinction the calibration turns on
//!
//! * The **encoder** is a real slice of the pinned `all-MiniLM-L6-v2`
//!   (`source_revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41`). Its DIMENSIONS are
//!   reduced — hidden 64 instead of 384, 2 layers instead of 6, 2 heads instead of 12,
//!   intermediate 256 instead of 1536, a 97-row vocabulary carved out of 30522 — but its
//!   WEIGHT VALUES are the upstream pretrained values for the rows and columns retained. It
//!   is **not** a randomly initialized toy. That matters because plan 03-06 freezes a
//!   per-class epsilon from relative-delta distributions measured HERE and applies it to the
//!   full encoder: a randomly initialized fixture would have a different gradient scale and
//!   a different initial-norm scale, and the transfer argument would be unfounded.
//! * The **text** is entirely synthetic. No content from the phase's source corpus, and no
//!   dataset text of any kind, is committed here (T-3-18). The provenance gate that keeps it
//!   that way greps this file for the corpus's characteristic terms, so this paragraph
//!   DESCRIBES them rather than quoting them — a doc comment is source text, and a gate that
//!   turns red on its own prose is a gate nobody will keep (the same trap 03-02 and 03-03
//!   each hit once).
//!
//! # Why the synthetic text reads the way it does
//!
//! The slice's vocabulary is 97 rows, and `BertSentenceEncoder` maps canonical token ids
//! through the fixture's `vocab_remap.json`, returning `SetFitError::VocabOutOfSlice` for
//! anything outside it (`setfit/import.rs:199`). So the corpus cannot spell arbitrary words
//! — not even digits, which are absent from the slice. Every sentence below is assembled
//! from the retained token strings. This is a property of the fixture, not a style choice,
//! and `fixture_every_row_encodes_within_the_slice_vocabulary` is what keeps it true.

use std::path::PathBuf;

use aprender::setfit::FreezeGroup;
use aprender::setfit::SetFitMiniLm;
use aprender_contrastive_data::ledger::AccessLedger;
use aprender_contrastive_data::pairs::PairConfig;
use aprender_contrastive_data::prepared::{Canonical, CanonicalDeclarations, PreparedDataset};
use aprender_contrastive_data::schema::LabeledExample;
use aprender_contrastive_data::select::{FewShotSelector, Selection, SelectionConfig};
use aprender_contrastive_data::split::SplitDeclaration;

use super::config::{SetFitTrainConfig, SetFitTrainRequest};
use super::verify::SerdeJsonCodec;
use super::{ArtifactReloadedAndVerified, HeadFitted, Prepared, SetFitRun};

/// The seed `aprender-core`'s own slice tests build with. Reused so a divergence between
/// this crate's fixture and that one is a divergence in the trainer, not in the seed.
pub(crate) const FIXTURE_SEED: u64 = 0x0107_5E7F_1701;

/// Training rows per class in the synthetic corpus.
///
/// Exactly the largest shot count the calibration matrix asks for, so the 16-shot cell draws
/// the whole pool and the 8-shot cell performs a real partial Fisher-Yates.
pub(crate) const TRAIN_PER_CLASS: usize = 16;

/// Declared classes.
pub(crate) const CLASSES: usize = 3;

/// The declared label map. These strings are METADATA — they are never encoded, so they are
/// under no vocabulary constraint.
const LABEL_NAMES: [&str; CLASSES] = ["alpha", "beta", "gamma"];

/// Per-class sentence material, all drawn from the slice's retained token strings.
///
/// One subject and one verb per class, so same-class pairs are genuinely more similar than
/// cross-class pairs and the contrastive objective has a real signal to follow. Four
/// modifiers times four objects gives the sixteen distinct rows each class needs.
struct ClassWords {
    subject: &'static str,
    verb: &'static str,
    modifiers: [&'static str; 4],
    objects: [&'static str; 4],
}

const CLASS_WORDS: [ClassWords; CLASSES] = [
    ClassWords {
        subject: "cat",
        verb: "sat",
        modifiers: ["quick", "brown", "lazy", "warm"],
        objects: ["mat", "rug", "line", "pad"],
    },
    ClassWords {
        subject: "stock",
        verb: "fell",
        modifiers: ["sunny", "tiny", "short", "good"],
        objects: ["text", "case", "rows", "batch"],
    },
    ClassWords {
        subject: "fox",
        verb: "jumps",
        modifiers: ["mixed", "lower", "longer", "different"],
        objects: ["cafe", "weather", "markets", "dog"],
    },
];

/// Held-out material for the validation and test splits, disjoint from every train row.
///
/// The ingest ladder coalesces cross-split duplicates, and a duplicate would silently shrink
/// a class pool — turning a capacity measurement into a measurement of the dedup pass.
const HELDOUT_MODIFIERS: [&str; 2] = ["naive", "shorter"];

/// The synthetic text for one row.
fn row_text(role_index: usize, label: usize, index: usize) -> String {
    let words = &CLASS_WORDS[label];
    if role_index == 0 {
        let modifier = words.modifiers[index % words.modifiers.len()];
        let object = words.objects[(index / words.modifiers.len()) % words.objects.len()];
        format!("the {modifier} {} {} over the {object} .", words.subject, words.verb)
    } else {
        let modifier = HELDOUT_MODIFIERS[(role_index - 1) % HELDOUT_MODIFIERS.len()];
        format!("the {modifier} {} {} again today .", words.subject, words.verb)
    }
}

/// One synthetic row, in the `{split}:{label}-{index}` id convention.
fn synthetic_row(role: &str, role_index: usize, label: usize, index: usize) -> LabeledExample {
    LabeledExample {
        id: format!("{role}:{label}-{index}"),
        input: row_text(role_index, label, index),
        label,
        label_text: LABEL_NAMES[label].to_string(),
        source_split: role.to_string(),
    }
}

/// The synthetic canonical dataset, built through the real `from_labeled_rows` ingest ladder.
pub(crate) fn synthetic_dataset(ledger: &mut AccessLedger) -> PreparedDataset<Canonical> {
    let label_names: Vec<String> = LABEL_NAMES.iter().map(|n| (*n).to_string()).collect();
    let rows = |role: &str, role_index: usize, per_class: usize| -> Vec<LabeledExample> {
        (0..CLASSES)
            .flat_map(|label| {
                (0..per_class).map(move |index| synthetic_row(role, role_index, label, index))
            })
            .collect()
    };
    let decl = |per_class: usize| SplitDeclaration {
        expected_class_counts: vec![per_class; CLASSES],
        label_names: label_names.clone(),
    };
    PreparedDataset::<Canonical>::from_labeled_rows(
        rows("train", 0, TRAIN_PER_CLASS),
        rows("validation", 1, 1),
        rows("test", 2, 1),
        &CanonicalDeclarations {
            train: decl(TRAIN_PER_CLASS),
            validation: decl(1),
            test: decl(1),
            label_names,
        },
        ledger,
    )
    .expect("the synthetic corpus must be a valid canonical dataset")
}

/// Resolve the conformance fixture directory the same way `aprender-core` does: the
/// `APRENDER_SETFIT_FIXTURES` override when it names a directory, otherwise the in-repo path.
///
/// `CARGO_MANIFEST_DIR` here is `crates/aprender-train`, so the default is the sibling
/// crate's `tests/fixtures/setfit`.
pub(crate) fn fixtures_dir() -> PathBuf {
    if let Ok(p) = std::env::var("APRENDER_SETFIT_FIXTURES") {
        let p = PathBuf::from(p);
        if p.is_dir() {
            return p;
        }
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../aprender-core/tests/fixtures/setfit")
}

/// The real-weight MiniLM slice encoder, in eval mode, seeded at `root_seed`.
pub(crate) fn slice_encoder(root_seed: u64) -> SetFitMiniLm {
    SetFitMiniLm::from_slice_fixture(&fixtures_dir(), root_seed)
        .expect("the frozen MiniLM slice fixture must load")
}

/// One cell of the calibration matrix: a seed crossed with a shot/epoch boundary
/// configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CalibrationVariant {
    /// A human-readable cell name, used as the `calibration_regime_id` component.
    pub(crate) label: &'static str,
    /// The root seed for every RNG domain of the run.
    pub(crate) root_seed: u64,
    /// Shots per class — one of Phase 2's `{8, 16, 32, 64}`.
    pub(crate) shots_per_class: u32,
    /// Contrastive epochs.
    pub(crate) epochs: u32,
    /// Pair batch size.
    pub(crate) batch_size: u32,
    /// The explicit pair budget. Explicit rather than default because the default closed
    /// form is 384 pairs at 8 shots and 1536 at 16, which is a training run rather than a
    /// test.
    pub(crate) budget: u64,
}

impl CalibrationVariant {
    /// Total optimizer steps this cell performs.
    pub(crate) fn total_steps(self) -> u64 {
        let per_epoch = self.budget.div_ceil(u64::from(self.batch_size));
        u64::from(self.epochs) * per_epoch
    }
}

/// The seeds the matrix sweeps.
const CALIBRATION_SEEDS: [u64; 3] = [1, 7, 42];

/// The two shot/epoch BOUNDARY configurations.
///
/// The second is the one that stresses the endpoint statistic (two epochs means the pair
/// order is reshuffled and the first-k/last-k window straddles an epoch boundary) and the
/// sparse-embedding support fraction (twice the selected rows touch more embedding rows).
const CALIBRATION_BOUNDARIES: [(&str, u32, u32, u32, u64); 2] =
    [("s8e1b4", 8, 1, 4, 12), ("s16e2b8", 16, 2, 8, 16)];

/// The calibration matrix: every seed crossed with every boundary configuration.
///
/// Six cells. Plan 03-05 runs each with a `1e-30`-learning-rate control alongside, so the
/// matrix is twelve complete `run_tuning` passes.
pub(crate) fn calibration_variants() -> Vec<CalibrationVariant> {
    let mut out = Vec::with_capacity(CALIBRATION_SEEDS.len() * CALIBRATION_BOUNDARIES.len());
    for &root_seed in &CALIBRATION_SEEDS {
        for &(label, shots_per_class, epochs, batch_size, budget) in &CALIBRATION_BOUNDARIES {
            out.push(CalibrationVariant {
                label,
                root_seed,
                shots_per_class,
                epochs,
                batch_size,
                budget,
            });
        }
    }
    out
}

/// The default cell: 8 shots, 1 epoch, batch 4, budget 12 — a three-step run.
///
/// Its SEED is [`FIXTURE_SEED`], which the calibration matrix never swept, so a run built from
/// it is OUTSIDE the calibrated regime and the evidence gate refuses it. That is deliberate and
/// load-bearing: it is the fixture the seed-negative below is written against. Tests that need
/// to reach a threshold comparison use [`calibrated_variant`] instead.
pub(crate) fn default_variant() -> CalibrationVariant {
    CalibrationVariant {
        label: "s8e1b4",
        root_seed: FIXTURE_SEED,
        shots_per_class: 8,
        epochs: 1,
        batch_size: 4,
        budget: 12,
    }
}

/// The same cell at a seed the calibration matrix DID sweep — the gate-crossing fixture.
///
/// Every test that needs `tune_encoder` to get past the regime check and actually compare
/// evidence against the frozen epsilons builds from this. The seed is taken from
/// [`CALIBRATION_SEEDS`] rather than written out, so a change to the swept set moves this
/// fixture with it instead of leaving a stale literal that silently stops being calibrated.
pub(crate) fn calibrated_variant() -> CalibrationVariant {
    CalibrationVariant { root_seed: CALIBRATION_SEEDS[0], ..default_variant() }
}

/// A CALIBRATED seed in an UNMEASURED cell: batch 3 where the matrix measured batch 4.
///
/// The cell-negative's fixture. It differs from [`calibrated_variant`] in the batch size and
/// in nothing else, so a refusal is attributable to the cell alone — the seed, the encoder and
/// every other knob are the calibrated ones.
pub(crate) fn uncalibrated_cell_variant() -> CalibrationVariant {
    CalibrationVariant { label: "s8e1b3", batch_size: 3, ..calibrated_variant() }
}

/// The validated configuration for a cell, derived from the reference recipe and shrunk.
///
/// `encoder_lr_override` is how the `1e-30` control is expressed: the control differs from
/// its cell in the learning rate and in NOTHING else, so a difference in the measured
/// relative deltas is attributable to the learning rate alone.
pub(crate) fn config_for(
    variant: CalibrationVariant,
    encoder_lr_override: Option<f64>,
) -> SetFitTrainConfig {
    let reference = SetFitTrainConfig::reference_defaults(variant.root_seed);
    let mut pair_config = PairConfig::new(variant.root_seed);
    pair_config.budget = Some(variant.budget);
    SetFitTrainConfig::new(SetFitTrainRequest {
        encoder_lr: encoder_lr_override.unwrap_or_else(|| reference.encoder_lr()),
        epochs: variant.epochs,
        batch_size: variant.batch_size,
        warmup_ratio: reference.warmup_ratio(),
        grad_clip_max_norm: reference.grad_clip_max_norm(),
        max_length: reference.max_length(),
        pair_config,
        freeze_policy: Vec::new(),
        head_regularization: reference.head_regularization(),
        root_seed: variant.root_seed,
        device: "cpu".to_string(),
        lr_schedule: reference.lr_schedule(),
    })
    .expect("the fixture configuration satisfies the twelve-knob table")
}

/// A complete, prepared run for one cell.
///
/// Built through the SHIPPED doors only — `from_labeled_rows`, `FewShotSelector::select`,
/// `SetFitTrainConfig::new`, `SetFitRun::prepare` — so a test written against it is a test
/// against the path production takes.
pub(crate) fn prepared_run(
    variant: CalibrationVariant,
    encoder_lr_override: Option<f64>,
) -> SetFitRun<Prepared> {
    let mut ledger = AccessLedger::new();
    let dataset = synthetic_dataset(&mut ledger);
    let selection = FewShotSelector::select(
        &dataset,
        &SelectionConfig { root_seed: variant.root_seed, shots_per_class: variant.shots_per_class },
        &mut ledger,
    )
    .expect("the synthetic corpus must support this selection");
    let encoder = slice_encoder(variant.root_seed);
    SetFitRun::prepare(encoder, dataset, selection, config_for(variant, encoder_lr_override))
        .expect("the fixture run must prepare")
}

/// A complete calibrated pipeline: `prepare` -> `tune_encoder` -> `fit_head`.
///
/// Lives here rather than in each test module because it was written out THREE times —
/// `bundle_tests.rs`, `verify_tests.rs` and `mod.rs`'s own test module — identical down to
/// both `expect` strings, the `mod.rs` copy differing only by taking the variant instead of
/// hardcoding [`calibrated_variant`]. Three copies of a pipeline is three places for the
/// pipeline's shape to drift apart, in fixtures whose whole job is that every test starts
/// from the same run.
///
/// # Panics
///
/// If the run misses the evidence gate or the head fails to fit — both are fixture defects,
/// not test failures, and both are the reason the messages name the measured cell.
pub(crate) fn head_fitted_run(variant: CalibrationVariant) -> SetFitRun<HeadFitted> {
    prepared_run(variant, None)
        .tune_encoder()
        .expect("a run at a measured seed and cell must pass the evidence gate")
        .fit_head()
        .expect("the head must fit on the fixture's 24 encode-once rows")
}

/// A complete calibrated pipeline carried one step further, through the shipped codec.
///
/// Here for the same reason as [`head_fitted_run`] directly above, and recorded because the
/// three-copy state that doc describes had reappeared one stage later: `verify_tests.rs`,
/// `evaluate_tests.rs` and `lock_tests.rs` each spelled this out identically, down to the
/// `expect` string. The codec choice is the part that must not drift — a test that reloads
/// through a DIFFERENT codec than its neighbours is comparing two things nobody declared were
/// the same.
///
/// # Panics
///
/// If the round trip fails, which is a fixture defect rather than a test failure: the shipped
/// codec is faithful by construction, and `EchoCodec` is the door for the negative case.
pub(crate) fn verified_run(variant: CalibrationVariant) -> SetFitRun<ArtifactReloadedAndVerified> {
    head_fitted_run(variant)
        .verify_artifact(&SerdeJsonCodec::new())
        .expect("a faithful codec must complete the round trip")
}

/// The fixture dataset, rebuilt INDEPENDENTLY of any run's copy.
///
/// Deliberately not `run.dataset()`: passing a run's own dataset back into an evaluator would
/// make its agreement check pass by IDENTITY, and the property being relied on is that it
/// passes by FINGERPRINT.
pub(crate) fn fixture_dataset() -> PreparedDataset<Canonical> {
    let mut ledger = AccessLedger::new();
    synthetic_dataset(&mut ledger)
}

/// The fixture corpus with ONE TEST row's bytes changed — a DIFFERENT canonical dataset.
///
/// Same ids, same label map, same declarations, same train and validation splits, so the only
/// thing that separates it from [`fixture_dataset`] is the corpus digest. That is what makes it
/// the right probe for "the model is the locked one but the DATA is not": every other reason a
/// door might refuse is held constant, so a refusal can only be about the dataset identity.
///
/// # Panics
///
/// If the rebuilt corpus is not a valid canonical dataset, which is a fixture defect.
pub(crate) fn dataset_with_altered_test_row() -> PreparedDataset<Canonical> {
    let base = fixture_dataset();
    let label_names = base.label_names().to_vec();
    let classes = label_names.len();
    let train = base.train().rows().to_vec();
    let validation = base.validation().rows().to_vec();
    let mut test = base.test().rows().to_vec();
    test[0].input = format!("{} and again .", test[0].input);

    let train_per_class = train.len() / classes;
    let decl = |per_class: usize| SplitDeclaration {
        expected_class_counts: vec![per_class; classes],
        label_names: label_names.clone(),
    };
    let mut ledger = AccessLedger::new();
    PreparedDataset::<Canonical>::from_labeled_rows(
        train,
        validation,
        test,
        &CanonicalDeclarations {
            train: decl(train_per_class),
            validation: decl(1),
            test: decl(1),
            label_names: label_names.clone(),
        },
        &mut ledger,
    )
    .expect("altering one test row must still yield a valid canonical dataset")
}

/// A prepared run with an explicit freeze policy.
///
/// The all-frozen negative needs a run whose trainable set is EMPTY, and `config_for` hard-codes
/// an empty policy because every other cell wants the D-20 all-trainable default.
pub(crate) fn prepared_run_with_freeze(
    variant: CalibrationVariant,
    freeze_policy: Vec<FreezeGroup>,
) -> SetFitRun<Prepared> {
    let mut ledger = AccessLedger::new();
    let dataset = synthetic_dataset(&mut ledger);
    let selection = FewShotSelector::select(
        &dataset,
        &SelectionConfig { root_seed: variant.root_seed, shots_per_class: variant.shots_per_class },
        &mut ledger,
    )
    .expect("the synthetic corpus must support this selection");
    let encoder = slice_encoder(variant.root_seed);
    let reference = SetFitTrainConfig::reference_defaults(variant.root_seed);
    let mut pair_config = PairConfig::new(variant.root_seed);
    pair_config.budget = Some(variant.budget);
    let config = SetFitTrainConfig::new(SetFitTrainRequest {
        encoder_lr: reference.encoder_lr(),
        epochs: variant.epochs,
        batch_size: variant.batch_size,
        warmup_ratio: reference.warmup_ratio(),
        grad_clip_max_norm: reference.grad_clip_max_norm(),
        max_length: reference.max_length(),
        pair_config,
        freeze_policy,
        head_regularization: reference.head_regularization(),
        root_seed: variant.root_seed,
        device: "cpu".to_string(),
        lr_schedule: reference.lr_schedule(),
    })
    .expect("the fixture configuration satisfies the twelve-knob table");
    SetFitRun::prepare(encoder, dataset, selection, config).expect("the fixture run must prepare")
}

/// The text from `header` up to the first line that closes a block at column zero.
///
/// # Why a source scan lives in the fixture module
///
/// Several phase-3 modules assert what does NOT exist — no public constructor, no float
/// parameter, no removal API — and a property of that shape cannot be witnessed by calling
/// something. `evaluate_tests.rs` and `lock_tests.rs` each spelled this scanner out, character
/// for character, including the sibling below. Two copies of a scanner is two places for "what
/// counts as the end of a block" to drift apart, in guards whose entire value is that they mean
/// the same thing in every module that runs them.
///
/// # Panics
///
/// If `header` does not appear, or opens no block closing at column zero. Both are guard
/// defects rather than test failures: a scan that silently matched nothing would be green for
/// a module that had deleted the very shape the guard describes.
pub(crate) fn source_block_after(src: &str, header: &str) -> String {
    let start =
        src.find(header).unwrap_or_else(|| panic!("`{header}` must appear in the scanned source"));
    let rest = &src[start..];
    let end = rest
        .find("\n}")
        .unwrap_or_else(|| panic!("`{header}` must open a block that closes at column zero"));
    rest[..end].to_string()
}

/// The signature text of `header`, up to the opening brace of its body.
///
/// Reads the WHOLE signature rather than the header's line, so an assertion about a parameter
/// survives a signature rustfmt has wrapped across several lines.
///
/// # Panics
///
/// If `header` does not appear, or is not followed by a body — see [`source_block_after`].
pub(crate) fn source_signature_after(src: &str, header: &str) -> String {
    let start =
        src.find(header).unwrap_or_else(|| panic!("`{header}` must appear in the scanned source"));
    let rest = &src[start..];
    let end = rest.find(" {").unwrap_or_else(|| panic!("`{header}` must be followed by a body"));
    rest[..end].to_string()
}

/// The selection alone, for tests that do not need an encoder.
pub(crate) fn fixture_selection(root_seed: u64, shots_per_class: u32) -> Selection {
    let mut ledger = AccessLedger::new();
    let dataset = synthetic_dataset(&mut ledger);
    FewShotSelector::select(&dataset, &SelectionConfig { root_seed, shots_per_class }, &mut ledger)
        .expect("the synthetic corpus must support this selection")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Purity: two independent builds must produce structurally identical objects.
    ///
    /// The selection's semantic hash covers the ordered ids, the label map and both content
    /// hashes, so equality here is stronger than comparing the id lists.
    #[test]
    fn fixture_two_builds_agree_on_the_selection_semantic_hash() {
        let first = fixture_selection(FIXTURE_SEED, 8);
        let second = fixture_selection(FIXTURE_SEED, 8);
        assert_eq!(
            first.semantic_hash(),
            second.semantic_hash(),
            "the fixture must be a pure function of its inputs",
        );
        assert_eq!(first.ordered_ids(), second.ordered_ids());
    }

    /// A different seed must produce a different draw, or the purity assertion above would
    /// hold vacuously for a fixture that ignored its seed entirely.
    #[test]
    fn fixture_a_different_seed_draws_a_different_selection() {
        let a = fixture_selection(1, 8);
        let b = fixture_selection(7, 8);
        assert_ne!(a.semantic_hash(), b.semantic_hash(), "the seed must reach the draw",);
    }

    #[test]
    fn fixture_corpus_declares_three_classes_and_sixteen_rows_each() {
        let dataset = fixture_dataset();
        assert_eq!(dataset.label_names().len(), CLASSES);
        assert_eq!(dataset.train().rows().len(), CLASSES * TRAIN_PER_CLASS);
        assert_eq!(dataset.train().class_counts(), &[TRAIN_PER_CLASS as u64; CLASSES]);
    }

    /// Every train row is distinct. A duplicate would be coalesced by the ingest ladder and
    /// silently shrink a class pool.
    #[test]
    fn fixture_every_row_text_is_distinct() {
        let dataset = fixture_dataset();
        let mut seen: Vec<&str> = Vec::new();
        for row in dataset.train().rows() {
            assert!(!seen.contains(&row.input.as_str()), "duplicate synthetic text: {}", row.input);
            seen.push(&row.input);
        }
        assert_eq!(seen.len(), CLASSES * TRAIN_PER_CLASS);
    }

    /// The vocabulary constraint, enforced rather than trusted.
    ///
    /// `encode_texts` returns `SetFitError::VocabOutOfSlice` for any canonical token id the
    /// 97-row remap does not carry, so a successful encode of every row IS the proof that
    /// the corpus stayed inside the slice.
    #[test]
    fn fixture_every_row_encodes_within_the_slice_vocabulary() {
        let dataset = fixture_dataset();
        let encoder = slice_encoder(FIXTURE_SEED);
        let texts: Vec<&str> = dataset.train().rows().iter().map(|r| r.input.as_str()).collect();
        let embeddings = encoder
            .encode_texts(&texts)
            .expect("every synthetic row must tokenize inside the slice vocabulary");
        assert_eq!(embeddings.shape(), &[texts.len(), 64]);

        for role_rows in [dataset.validation().rows(), dataset.test().rows()] {
            let texts: Vec<&str> = role_rows.iter().map(|r| r.input.as_str()).collect();
            encoder
                .encode_texts(&texts)
                .expect("held-out rows must also tokenize inside the slice");
        }
    }

    /// The slice's dimensions, asserted here so the SUMMARY's recorded numbers are checked
    /// by a test rather than transcribed from a JSON file by hand.
    #[test]
    fn fixture_slice_encoder_has_the_recorded_dimensions() {
        let encoder = slice_encoder(FIXTURE_SEED);
        assert_eq!(encoder.num_layers(), 2, "slice_config.json num_layers");
        assert!(!encoder.training(), "from_slice_fixture returns eval mode");
        let mut encoder = encoder;
        let names: Vec<String> =
            encoder.trainable_parameters_mut().into_iter().map(|(n, _)| n).collect();
        assert_eq!(
            names.len(),
            37,
            "3 embedding tables + 1 embedding LayerNorm pair + 2 layers x 16",
        );
        assert!(names.contains(&"embeddings.word_embeddings.weight".to_string()));
        assert!(names.contains(&"encoder.layer.1.output.LayerNorm.bias".to_string()));
    }

    /// The matrix has at least the six cells plan 03-05 contracts, and the two boundary
    /// configurations really do differ in shots, epochs AND batch size.
    #[test]
    fn fixture_calibration_variants_cover_the_matrix() {
        let variants = calibration_variants();
        assert!(
            variants.len() >= 6,
            "the calibration matrix needs >= 3 seeds x >= 2 boundary configurations, got {}",
            variants.len(),
        );

        let mut seeds: Vec<u64> = variants.iter().map(|v| v.root_seed).collect();
        seeds.sort_unstable();
        seeds.dedup();
        assert!(seeds.len() >= 3, "at least three distinct seeds");

        let mut shapes: Vec<(u32, u32, u32)> =
            variants.iter().map(|v| (v.shots_per_class, v.epochs, v.batch_size)).collect();
        shapes.sort_unstable();
        shapes.dedup();
        assert!(shapes.len() >= 2, "at least two distinct shot/epoch boundary configurations",);
        assert!(
            shapes.windows(2).all(|w| w[0].0 != w[1].0 && w[0].1 != w[1].1 && w[0].2 != w[1].2),
            "the boundary configurations must differ in shots, epochs AND batch size",
        );

        for v in &variants {
            assert!(v.total_steps() >= 2, "{} must take at least two steps", v.label);
        }
    }

    /// The three gate-facing variants really are what their names claim.
    ///
    /// Asserted here rather than trusted, because the regime negatives elsewhere are only
    /// evidence if their fixtures differ from the calibrated one in EXACTLY the component the
    /// test blames. A `default_variant` that happened to sit on a swept seed would make the
    /// seed-negative green for the wrong reason.
    #[test]
    fn fixture_gate_facing_variants_isolate_one_component_each() {
        let calibrated = calibrated_variant();
        assert!(
            CALIBRATION_SEEDS.contains(&calibrated.root_seed),
            "the calibrated fixture must sit on a swept seed",
        );
        assert!(
            CALIBRATION_BOUNDARIES.iter().any(|&(label, shots, epochs, batch, _)| {
                label == calibrated.label
                    && (shots, epochs, batch)
                        == (calibrated.shots_per_class, calibrated.epochs, calibrated.batch_size)
            }),
            "the calibrated fixture must sit in a measured cell",
        );

        // The seed-negative: an unswept seed, everything else the calibrated cell.
        let seed_negative = default_variant();
        assert!(
            !CALIBRATION_SEEDS.contains(&seed_negative.root_seed),
            "FIXTURE_SEED must NOT be a swept seed, or the seed-negative proves nothing",
        );
        assert_eq!(
            (
                seed_negative.label,
                seed_negative.shots_per_class,
                seed_negative.epochs,
                seed_negative.batch_size
            ),
            (
                calibrated.label,
                calibrated.shots_per_class,
                calibrated.epochs,
                calibrated.batch_size
            ),
            "the seed-negative must differ from the calibrated fixture in the SEED alone",
        );

        // The cell-negative: a swept seed, a cell the matrix never measured.
        let cell_negative = uncalibrated_cell_variant();
        assert_eq!(
            cell_negative.root_seed, calibrated.root_seed,
            "the cell-negative must keep the calibrated seed",
        );
        assert_ne!(cell_negative.batch_size, calibrated.batch_size);
        assert!(
            !CALIBRATION_BOUNDARIES.iter().any(|&(_, shots, epochs, batch, _)| {
                (shots, epochs, batch)
                    == (
                        cell_negative.shots_per_class,
                        cell_negative.epochs,
                        cell_negative.batch_size,
                    )
            }),
            "the cell-negative's shot/epoch/batch triple must be one the matrix never measured",
        );
    }

    /// The control differs from its cell in the learning rate and in nothing else.
    #[test]
    fn fixture_control_config_differs_only_in_the_learning_rate() {
        let v = default_variant();
        let real = config_for(v, None);
        let control = config_for(v, Some(1e-30));
        assert_ne!(real.encoder_lr(), control.encoder_lr());
        assert_eq!(real.epochs(), control.epochs());
        assert_eq!(real.batch_size(), control.batch_size());
        assert_eq!(real.warmup_ratio(), control.warmup_ratio());
        assert_eq!(real.root_seed(), control.root_seed());
        assert_eq!(real.pair_config(), control.pair_config());
        assert_eq!(real.max_length(), control.max_length());
    }

    /// The fixture run prepares through the shipped door.
    #[test]
    fn fixture_prepared_run_reaches_the_prepared_state() {
        let run = prepared_run(default_variant(), None);
        assert_eq!(run.state_name(), "prepared");
        assert_eq!(run.selection().len(), CLASSES * 8);
    }
}
