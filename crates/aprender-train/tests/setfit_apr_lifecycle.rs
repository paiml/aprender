//! OPS-01 — how far a Rust caller gets through the SetFit lifecycle using PUBLIC APIs only.
//!
//! # What OPS-01 asks for, and what this file can currently witness
//!
//! `train -> save -> load -> embed -> classify -> inspect`, with zero types from the
//! command-line crate. This is an out-of-crate integration test, so `pub(crate)` internals
//! are invisible by construction — the file compiling at all is part of the claim.
//!
//! **Neither spelling of that crate's name appears anywhere in this file, including in this
//! comment and including in a path literal.** That is deliberate:
//! `lifecycle_source_names_no_command_line_crate` scans the WHOLE source with no
//! comment-filter, and a filter is the part of such a guard that goes wrong silently
//! (CLAUDE.md verification discipline 7). Keeping the file clean of the string is what lets
//! the guard stay filter-free.
//!
//! | rung | reachable out-of-crate today | witness |
//! |------|------------------------------|---------|
//! | train | **yes** | `lifecycle_train_then_save_hands_the_caller_the_hashed_artifact_bytes` |
//! | save (bytes) | **yes**, since 04-17's `into_artifact_bytes` | same test — the bytes are RE-HASHED against the recorded digest |
//! | save (as `setfit-apr-v1`) | **NO — with the SLICE encoder** | `lifecycle_the_apr_save_rung_is_refused_by_the_conformance_slice_encoder` |
//! | load / embed / classify / inspect | **NO** — they have no input HERE | `lifecycle_the_load_rung_requires_setfit_apr_v1_bytes` |
//!
//! # The blocker, restated after Phase 5 (what changed and what did not)
//!
//! **Before Phase 5's 05-03 calibration edit (commit `a63bb130b`), no user-reachable path
//! produced a `setfit-apr-v1`.** Two facts held at the same time and were jointly fatal:
//!
//! 1. `tune_encoder` refuses any run whose regime coordinates are outside
//!    `CALIBRATED_REGIMES`, and that set had exactly ONE entry, whose architecture component
//!    is compared for EXACT equality:
//!    `minilm-slice-h64-l2-a2-i256-v97@1110a243|seeds=1,42,7|cells=s16e2b8,s8e1b4`. So the
//!    phase-3 MiniLM slice was the only encoder that could reach `HeadFitted` through the
//!    shipped transitions.
//! 2. That same slice provably cannot compute the `setfit-apr-v1` contract's six embedded
//!    probes: it is a 97-row VOCABULARY CLOSURE, and `probe_unicode` needs canonical id
//!    5915. Every route to APR bytes ends in `write_setfit_apr`, which RECOMPUTES the
//!    probes from the view's own tensors, so the refusal is a property of the ENCODER and
//!    not of any particular door.
//!
//! **Fact 1 no longer holds.** `a63bb130b` added a SECOND, measured regime entry for the
//! production `all-MiniLM-L6-v2` checkout — additively, with the fixture entry byte-untouched
//! — so a run on the production encoder now resolves a threshold table instead of failing
//! closed with `UncalibratedRegime`. The end-to-end chain that unblocks is proven at the
//! spawned-binary tier by 05-07
//! (`setfit_cli_production_chain_completes_after_the_calibration_edit`).
//!
//! **Fact 2 is untouched, and it is the fact this file measures.** The conformance slice is
//! still a 97-row vocabulary closure and still cannot compute `probe_unicode`. Every refusal
//! below is therefore about the SLICE ENCODER, not about the calibrated set — which is why the
//! calibration edit did not turn any of them red. What this file can witness out-of-crate is
//! bounded by that: the production checkout is an 86.7 MB offline prerequisite, not a
//! committed fixture, so an out-of-crate test that trained on it would be measuring the host.
//!
//! # What this file deliberately does NOT do
//!
//! It does not hand-build a `SetFitArtifactView` and call core's public `write_setfit_apr`
//! with a synthetic APR-capable encoder to manufacture an artifact. That would compile, and
//! it would let this file call `.embed(` and `.classify(` — but the model classified would
//! not be the one the `train` rung produced, so the green light would mean "a test-local
//! writer round-trips" while reading as "OPS-01 holds". The whole point of OPS-01 is the
//! JOIN between the rungs.
//!
//! Nor does it reach for a `pub(crate)` door. 04-05 solved the same problem in-crate by
//! substituting the encoder and head into a run through a struct literal; that shape is
//! compiler-closed here (`E0451`, both `SetFitRun` and `HeadFittedEvidence`), which is
//! itself the finding: the in-crate remedy is not available at the tier where OPS-01 is
//! claimed.
//!
//! Every refusal below is asserted as a TYPED value naming its cause, so the day the SLICE
//! stops being refused these tests go red and point a reader at this module rather than
//! silently continuing to pass.

#![cfg(feature = "setfit")]

use aprender::setfit::{load_setfit_apr, SetFitArtifactError, SetFitMiniLm, MAX_SEQUENCE_LENGTH};
use aprender_contrastive_data::ledger::AccessLedger;
use aprender_contrastive_data::pairs::PairConfig;
use aprender_contrastive_data::prepared::{Canonical, CanonicalDeclarations, PreparedDataset};
use aprender_contrastive_data::schema::LabeledExample;
use aprender_contrastive_data::select::{FewShotSelector, Selection, SelectionConfig};
use aprender_contrastive_data::split::SplitDeclaration;
use entrenar::train::setfit::apr_codec::AprCodec;
use entrenar::train::setfit::config::{SetFitTrainConfig, SetFitTrainRequest};
use entrenar::train::setfit::verify::{CodecError, SerdeJsonCodec, SetFitCodec};
use entrenar::train::setfit::{
    ArtifactReloadedAndVerified, HeadFitted, SetFitRun, SetFitTrainError,
};
use sha2::{Digest, Sha256};

// ===========================================================================================
// The cell. CALIBRATED, and the seed is why — the same coordinates `setfit_repro.rs` uses.
// ===========================================================================================

/// The root seed. One of the three the 03-05 calibration matrix swept, so `tune_encoder`
/// reaches the evidence-gate COMPARISON instead of being refused for being outside the
/// calibrated regime.
const ROOT_SEED: u64 = 1;
/// Shots per class, epochs, pair batch size, explicit pair budget — the `s8e1b4` cell.
const SHOTS_PER_CLASS: u32 = 8;
const EPOCHS: u32 = 1;
const BATCH_SIZE: u32 = 4;
const BUDGET: u64 = 12;

/// A seed OUTSIDE `seeds=1,42,7`, for the fail-closed witness.
const UNCALIBRATED_SEED: u64 = 2;
/// A batch size that renders a cell OUTSIDE `cells=s16e2b8,s8e1b4` (it makes `s8e1b8`).
const UNCALIBRATED_BATCH_SIZE: u32 = 8;

/// Training rows per class in the synthetic corpus.
const TRAIN_PER_CLASS: usize = 16;
/// Declared classes.
const CLASSES: usize = 3;
const LABEL_NAMES: [&str; CLASSES] = ["alpha", "beta", "gamma"];

const SUBJECTS: [&str; CLASSES] = ["cat", "stock", "fox"];
const VERBS: [&str; CLASSES] = ["sat", "fell", "jumps"];
const MODIFIERS: [[&str; 4]; CLASSES] = [
    ["quick", "brown", "lazy", "warm"],
    ["sunny", "tiny", "short", "good"],
    ["mixed", "lower", "longer", "different"],
];
const OBJECTS: [[&str; 4]; CLASSES] = [
    ["mat", "rug", "line", "pad"],
    ["text", "case", "rows", "batch"],
    ["cafe", "weather", "markets", "dog"],
];
/// Held-out material, disjoint from every train row — a cross-split duplicate is coalesced
/// by the ingest ladder and would silently shrink a class pool.
const HELDOUT_MODIFIERS: [&str; 2] = ["naive", "shorter"];

// ===========================================================================================
// The pipeline, built through shipped PUBLIC doors only
// ===========================================================================================

fn row_text(role_index: usize, label: usize, index: usize) -> String {
    let subject = SUBJECTS[label];
    let verb = VERBS[label];
    if role_index == 0 {
        let modifier = MODIFIERS[label][index % 4];
        let object = OBJECTS[label][(index / 4) % 4];
        format!("the {modifier} {subject} {verb} over the {object} .")
    } else {
        let modifier = HELDOUT_MODIFIERS[(role_index - 1) % HELDOUT_MODIFIERS.len()];
        format!("the {modifier} {subject} {verb} again today .")
    }
}

fn synthetic_row(role: &str, role_index: usize, label: usize, index: usize) -> LabeledExample {
    LabeledExample {
        id: format!("{role}:{label}-{index}"),
        input: row_text(role_index, label, index),
        label,
        label_text: LABEL_NAMES[label].to_string(),
        source_split: role.to_string(),
    }
}

/// The synthetic corpus, re-spelled from the public API.
///
/// It must stay inside the MiniLM slice's 97-row vocabulary — the slice's encoder returns
/// `VocabOutOfSlice` for anything outside it, digits included — which is why the sentences
/// read as they do. That same 97-row closure is what the refusals in section (2) are about.
fn synthetic_dataset() -> PreparedDataset<Canonical> {
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
    let mut ledger = AccessLedger::new();
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
        &mut ledger,
    )
    .expect("the synthetic corpus must be a valid canonical dataset")
}

/// The fixture directory, resolved the way `aprender-core` resolves it.
fn fixtures_dir() -> std::path::PathBuf {
    if let Ok(p) = std::env::var("APRENDER_SETFIT_FIXTURES") {
        let p = std::path::PathBuf::from(p);
        if p.is_dir() {
            return p;
        }
    }
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../aprender-core/tests/fixtures/setfit")
}

fn config_at(seed: u64, batch_size: u32) -> SetFitTrainConfig {
    let reference = SetFitTrainConfig::reference_defaults(seed);
    let mut pair_config = PairConfig::new(seed);
    pair_config.budget = Some(BUDGET);
    SetFitTrainConfig::new(SetFitTrainRequest {
        encoder_lr: reference.encoder_lr(),
        epochs: EPOCHS,
        batch_size,
        warmup_ratio: reference.warmup_ratio(),
        grad_clip_max_norm: reference.grad_clip_max_norm(),
        max_length: reference.max_length(),
        pair_config,
        freeze_policy: Vec::new(),
        head_regularization: reference.head_regularization(),
        root_seed: seed,
        device: "cpu".to_string(),
        lr_schedule: reference.lr_schedule(),
    })
    .expect("the fixture configuration satisfies the twelve-knob table")
}

fn selection_at(dataset: &PreparedDataset<Canonical>, seed: u64) -> Selection {
    let mut ledger = AccessLedger::new();
    FewShotSelector::select(
        dataset,
        &SelectionConfig { root_seed: seed, shots_per_class: SHOTS_PER_CLASS },
        &mut ledger,
    )
    .expect("the synthetic corpus must support this selection")
}

/// The ONLY encoder an out-of-crate caller can drive through the shipped transitions.
///
/// `SetFitMiniLm::from_bundle_parts` is also `pub`, so a synthetic encoder is
/// CONSTRUCTIBLE here — it just cannot be TRAINED, because `tune_encoder`'s regime gate
/// compares the architecture fingerprint for exact equality against a one-entry set.
/// `lifecycle_no_second_encoder_can_reach_the_save_rung` is the measurement of that.
fn slice_encoder(seed: u64) -> SetFitMiniLm {
    SetFitMiniLm::from_slice_fixture(&fixtures_dir(), seed)
        .expect("the frozen MiniLM slice fixture must load")
}

/// `prepare -> tune_encoder -> fit_head`, all public doors, at the calibrated coordinates.
fn head_fitted_run() -> SetFitRun<HeadFitted> {
    let dataset = synthetic_dataset();
    let selection = selection_at(&dataset, ROOT_SEED);
    SetFitRun::prepare(
        slice_encoder(ROOT_SEED),
        dataset,
        selection,
        config_at(ROOT_SEED, BATCH_SIZE),
    )
    .expect("the fixture run must prepare")
    .tune_encoder()
    .expect("a run at a measured seed and cell must pass the evidence gate")
    .fit_head()
    .expect("the head must fit on the fixture's unique encode-once rows")
}

/// The same run carried through the trusted verify policy with phase 3's debug codec.
///
/// `SerdeJsonCodec` and not `AprCodec`, for the reason this module's header gives: the APR
/// codec cannot close over the slice's vocabulary. What this door proves is the SAVE rung's
/// mechanics — a real serialize, a real hash, a real reload and a real byte-canonical
/// closure check, all inside `run_verify_policy` — which is what makes the bytes taken from
/// it below the bytes that policy hashed.
fn verified_run() -> SetFitRun<ArtifactReloadedAndVerified> {
    head_fitted_run()
        .verify_artifact(&SerdeJsonCodec::new())
        .expect("a faithful codec must complete the trusted round trip")
}

/// Lowercase-hex SHA-256, computed HERE rather than read off the run.
///
/// The whole value of the re-hash below is that it is an INDEPENDENT computation over the
/// value a caller actually receives. Asking the run for its own digest twice would compare
/// a field with itself.
fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex::encode(hasher.finalize())
}

// ===========================================================================================
// (1) train -> save: the rungs that DO work out-of-crate
// ===========================================================================================

/// A caller trains, verifies, and walks away holding the artifact's bytes.
///
/// # This is the OUT-OF-CRATE witness for 04-17's G1 door
///
/// `verify_into_artifact_bytes_are_the_hashed_bytes` makes the same re-hash claim, but it
/// runs `--lib`, so it sees `pub(crate)` and cannot distinguish "the door exists" from "the
/// door is reachable by a downstream caller". This file is a different crate. Before 04-17
/// this test could not have been written at all: `run_verify_policy` dropped the buffer and
/// kept `bytes.len()`, and `VerifyReport::artifact_bytes()` — a `usize` — was the trap,
/// because it compiles and reads as if the payload were in hand.
///
/// # The ORDER of the reads is load-bearing
///
/// `into_artifact_bytes` CONSUMES the run, so every accessor value has to be taken first.
/// That is not a wart: a caller that needs both the file and the lock/token chain must
/// create the lock BEFORE writing the file, which is the correct order anyway. The borrow
/// checker enforces it, so no source assertion is needed for it.
#[test]
fn lifecycle_train_then_save_hands_the_caller_the_hashed_artifact_bytes() {
    let run = verified_run();

    // Read EVERYTHING off the run before the consuming door.
    let recorded_hash = run.artifact_hash();
    let format_id = run.artifact_format_id().to_string();
    let evidence_table_hash = run.evidence_table_hash().to_string();
    let selection_hash = run.selection_semantic_hash();
    let report = run.evidence().verify_report();
    let declared_len = report.artifact_bytes();
    let round_trip_closed = report.round_trip_closed();
    let probe_rows = report.probe_rows();

    // Non-vacuity FIRST: a run that produced nothing would make every claim below trivially
    // true, and a re-hash of an empty buffer is a perfectly reproducible constant.
    assert!(
        round_trip_closed,
        "the trusted policy's byte-canonical closure check must have passed"
    );
    assert!(probe_rows > 0, "the verification must have compared at least one probe row");
    assert!(declared_len > 0, "the policy must have serialized a non-empty artifact");
    assert_eq!(recorded_hash.len(), 64, "a SHA-256 renders as 64 hex characters");
    assert!(
        recorded_hash.chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
        "the recorded digest must be lowercase hex, got {recorded_hash}",
    );
    assert_eq!(evidence_table_hash.len(), 64, "the evidence table hash is a SHA-256");
    assert_eq!(selection_hash.len(), 64, "the selection semantic hash is a SHA-256");

    // THE DOOR. It consumes the run; nothing above may be read after this line.
    let bytes = run.into_artifact_bytes();

    assert_eq!(
        bytes.len(),
        declared_len,
        "the payload and the length the report published describe different artifacts; before \
         04-17 only the LENGTH survived the policy, and this is the assertion that the two are \
         now the same object",
    );
    assert_eq!(
        sha256_hex(&bytes),
        recorded_hash,
        "the bytes the door returned do not hash to the digest the run recorded, so they are a \
         RE-SERIALIZATION rather than the buffer the trusted policy hashed and closed — which is \
         precisely the substitute this door exists to make unnecessary",
    );
    assert_eq!(
        format_id, "setfit-serde-json-v1",
        "this run was closed with phase 3's debug codec; if it ever reports the APR format here, \
         the SLICE encoder has become able to carry an artifact and the rest of this file must \
         be revisited",
    );
}

// ===========================================================================================
// (2) THE FINDING, kept executable: the SLICE encoder cannot produce `setfit-apr-v1`
// ===========================================================================================

/// The OPS-01 tier: the CONFORMANCE SLICE encoder cannot carry an APR artifact.
///
/// Renamed from `..._by_the_only_calibrated_encoder` by 05-07. That name asserted a fact that
/// stopped holding at `a63bb130b`: `CALIBRATED_REGIMES` now has TWO entries, so the slice is no
/// longer the only calibrated encoder. **Nothing else about this test changed** — the slice is
/// still a 97-row vocabulary closure, `probe_unicode` still needs canonical id 5915, and both
/// routes below still refuse. The old name would have made a true measurement carry a false
/// claim.
///
/// 04-05 records the same measurement from INSIDE the crate
/// (`round_trip_the_phase_three_slice_fixture_cannot_carry_an_apr_artifact`). This is the
/// stronger claim: in-crate, the phase has a remedy — substitute the encoder and head into
/// the run through a struct literal. Out here that shape is `E0451`, so the refusal below is
/// terminal for a downstream caller rather than an inconvenience.
///
/// Both structural gaps are asserted, not only the one that fires first, so the record does
/// not silently narrow to whichever probe the encoder reaches soonest.
#[test]
fn lifecycle_the_apr_save_rung_is_refused_by_the_conformance_slice_encoder() {
    let arch = slice_encoder(ROOT_SEED).architecture();
    assert!(
        arch.vocab_remap.is_some(),
        "gap 1: the slice is a VOCABULARY CLOSURE, so a probe token outside it has no id",
    );
    assert!(
        arch.positions < MAX_SEQUENCE_LENGTH,
        "gap 2: the slice declares {} position rows, below the {}-token truncation boundary the \
         contract's probes reach",
        arch.positions,
        MAX_SEQUENCE_LENGTH,
    );

    // Route 1: the shipped door. `SetFitRun::<HeadFitted>::verify_artifact(&AprCodec)`.
    let Err(via_policy) = head_fitted_run().verify_artifact(&AprCodec::new()) else {
        panic!(
            "the SLICE fixture produced a setfit-apr-v1 artifact. This is NOT the F-10 signal \
             (F-10 was closed for the production encoder at a63bb130b): it means the 97-row \
             vocabulary closure has become able to compute probe_unicode. Restore the full \
             OPS-01 chain in this file: load_setfit_apr -> embed -> classify -> doc_view.",
        );
    };
    assert_probe_unicode_refusal(&via_policy);

    // Route 2: the public codec seam, from the SAME run's real verified bytes. A bundle is
    // recovered through `SetFitCodec::deserialize` — no `pub(crate)` door, no hand-built
    // view — and handed to the APR codec directly, bypassing the verify policy entirely.
    //
    // Two routes, one refusal, and that is the point (CLAUDE.md verification discipline 6:
    // one failing input is an anecdote). It shows the cause is the ENCODER's vocabulary and
    // not the particular door, because every route ends in `write_setfit_apr`, which
    // RECOMPUTES the six probes from the view's own tensors.
    let bundle = SerdeJsonCodec::new()
        .deserialize(&verified_run().into_artifact_bytes())
        .expect("phase 3's codec must read back the bytes it just wrote");
    let Err(via_seam) = AprCodec::new().serialize(&bundle) else {
        panic!(
            "the codec seam produced a setfit-apr-v1 artifact FROM THE SLICE's bundle; see above"
        );
    };
    assert_matches_probe_unicode(&via_seam, "the codec seam");
}

/// The train-error form of the refusal, unwrapped to its codec cause.
fn assert_probe_unicode_refusal(err: &SetFitTrainError) {
    let SetFitTrainError::Codec(codec_err) = err else {
        panic!("expected a codec refusal from the APR door, got {err:?}");
    };
    assert_matches_probe_unicode(codec_err, "the shipped verify_artifact door");
}

/// The refusal must be TYPED and must name the probe — never matched on message text.
///
/// A `is_err()` check would pass for a refusal with any cause at all, including one that
/// arrived because the fixture directory was missing. Naming `probe_unicode` is what makes
/// this a record of the SLICE's vocabulary closure rather than a record of something going
/// wrong.
fn assert_matches_probe_unicode(err: &CodecError, route: &str) {
    assert!(
        matches!(
            err,
            CodecError::Artifact {
                source: SetFitArtifactError::ProbeComputation { probe, .. },
                ..
            } if probe == "probe_unicode",
        ),
        "{route}: expected a typed probe-computation refusal naming probe_unicode, got {err:?}",
    );
}

/// The escape route is closed: an OUT-OF-ENVELOPE run cannot reach the save rung.
///
/// Renamed from `lifecycle_no_second_encoder_can_reach_the_save_rung` by 05-07. That name
/// asserted a fact that stopped holding at `a63bb130b`: there IS now a second calibrated
/// architecture, the production all-MiniLM-L6-v2 checkout. What this test measures was never
/// "only one architecture exists" — it is that the gate compares the run's coordinates
/// COMPONENT-WISE and fails closed on any coordinate the calibrated set does not cover, which
/// is exactly as true with two entries as with one.
///
/// # Why this test is what makes the finding complete
///
/// "The slice cannot carry an artifact" invites the obvious answer: use a different encoder.
/// `SetFitMiniLm::from_bundle_parts` is `pub`, so one is constructible out here. It cannot be
/// TRAINED at coordinates nobody measured: `tune_encoder` renders the run's own coordinates and
/// requires the frozen thresholds' calibrated set to COVER them, with the architecture
/// component compared for exact equality.
///
/// The refusal PUBLISHES the calibrated set, so this test reads the constant out of the error
/// rather than restating it — a copy here could drift from `CALIBRATED_REGIMES` silently, and
/// `CALIBRATED_REGIMES` is `pub(crate)` so it cannot be named directly from out here anyway.
///
/// Two INDEPENDENT coordinates are varied. One would leave open that the gate happened to
/// fire for an unrelated reason; both firing, each naming its own varied coordinate in the
/// observed id, shows the comparison is component-wise and fail-closed.
#[test]
fn lifecycle_an_out_of_envelope_run_cannot_reach_the_save_rung() {
    let cases = [
        ("seed", UNCALIBRATED_SEED, BATCH_SIZE, "seeds=2"),
        ("cell", ROOT_SEED, UNCALIBRATED_BATCH_SIZE, "cells=s8e1b8"),
    ];

    for (what, seed, batch_size, expected_component) in cases {
        let dataset = synthetic_dataset();
        let selection = selection_at(&dataset, seed);
        let prepared = SetFitRun::prepare(
            slice_encoder(seed),
            dataset,
            selection,
            config_at(seed, batch_size),
        )
        .expect("the run must prepare — the gate under test is downstream of prepare");

        let Err(err) = prepared.tune_encoder() else {
            panic!("the {what} variation was ACCEPTED; the calibration gate is not fail-closed");
        };
        let SetFitTrainError::UncalibratedRegime { observed, calibrated } = err else {
            panic!("expected UncalibratedRegime for the {what} variation, got {err:?}");
        };

        assert!(
            observed.contains(expected_component),
            "the {what} variation should have rendered `{expected_component}` into its regime id, \
             got `{observed}` — this run is not varying the coordinate it claims to vary",
        );
        // THE STATE CHANGE, ASSERTED POSITIVELY (05-07).
        //
        // This read `calibrated.len() == 1` with the note "if it has grown, an APR-capable
        // encoder may now be trainable and this whole file must be revisited". It grew:
        // Phase 5's 05-03 calibration edit (commit `a63bb130b`) added the production
        // all-MiniLM-L6-v2 regime, and the file WAS revisited — that is this plan. The old
        // literal was left behind by 05-03 and made `make setfit-lifecycle-tests` RED; it is
        // migrated here rather than deleted, so the transition is recorded by an assertion
        // instead of by an assertion's disappearance (05-03's own deviation-3 pattern).
        //
        // Exactly TWO, and both named, because a bare `>= 1` would pass a set that had
        // silently gained a third architecture — which is precisely the widening the
        // regime gate exists to make visible.
        assert_eq!(
            calibrated.len(),
            2,
            "the calibrated set has exactly two entries since a63bb130b — the phase-3 slice \
             and the production checkout. If it has grown again, a THIRD architecture became \
             trainable and this whole file must be revisited: got {calibrated:?}",
        );
        let fixture_entry = calibrated
            .iter()
            .find(|entry| entry.starts_with("minilm-slice-h64-l2-a2-i256-v97@"))
            .unwrap_or_else(|| {
                panic!("the phase-3 slice regime must still be calibrated, got {calibrated:?}")
            });
        assert!(
            calibrated
                .iter()
                .any(|entry| entry.starts_with("minilm-slice-h384-l6-a12-i1536-v30522@")),
            "and the PRODUCTION regime a63bb130b added must be present — its absence would \
             mean the F-10 unblock was reverted, got {calibrated:?}",
        );
        assert_ne!(
            &observed, fixture_entry,
            "non-vacuity: the observed id must actually differ from the fixture entry",
        );
        assert!(
            !calibrated.contains(&observed),
            "non-vacuity, restated for two entries: the observed id must differ from EVERY \
             calibrated entry, or the refusal above proves nothing about the coordinate this \
             case varied. observed `{observed}` against {calibrated:?}",
        );
    }
}

// ===========================================================================================
// (3) load: the rung with no reachable input
// ===========================================================================================

/// `load_setfit_apr` genuinely requires `setfit-apr-v1` bytes — which rung 2 cannot produce.
///
/// This is what closes the chain's remaining half. `embed`, `classify` and `doc_view` are
/// all methods on `VerifiedSetFitModel`, and `load_setfit_apr` is its ONLY constructor
/// (there is no public constructor, no `Default`, no `Deserialize`, and
/// `tests/ui/setfit_verified_model_constructed.rs` pins that as a compile error). So a
/// loader with no admissible input is four unreachable rungs, not one.
///
/// TWO different non-APR inputs, drawing TWO different typed refusals from two different
/// rungs of the ladder — a single input could not distinguish "this input is wrong" from
/// "the loader refuses everything".
#[test]
fn lifecycle_the_load_rung_requires_setfit_apr_v1_bytes() {
    // (a) The real artifact bytes this file's own train->save rung produces. They are a
    //     genuine, policy-verified artifact — in the WRONG container.
    let bytes = verified_run().into_artifact_bytes();
    assert!(!bytes.is_empty(), "non-vacuity: the save rung must have produced bytes");
    let Err(container) = load_setfit_apr(&bytes) else {
        panic!(
            "the production loader accepted phase 3's debug encoding; the container check is not \
             doing its job",
        );
    };
    assert!(
        matches!(container, SetFitArtifactError::ContainerIntegrity { .. }),
        "expected a typed rung-3 container refusal, got {container:?}",
    );

    // (b) The committed `.apr` fixture, which IS a well-formed APR container and is NOT a
    //     SetFit artifact. It reaches a LATER rung, which is what makes the pair informative.
    let fixture = fixtures_dir().join("slice_model.apr");
    let fixture_bytes = std::fs::read(&fixture).unwrap_or_else(|e| {
        panic!("the committed slice fixture must be readable at {fixture:?}: {e}")
    });
    let Err(not_setfit) = load_setfit_apr(&fixture_bytes) else {
        panic!("a plain Bert .apr is not a setfit-apr-v1 artifact and must not load as one");
    };
    assert!(
        matches!(not_setfit, SetFitArtifactError::NotASetFitArtifact { .. }),
        "expected a typed rung-4 refusal naming the model type, got {not_setfit:?}",
    );
    assert!(
        !matches!(not_setfit, SetFitArtifactError::ContainerIntegrity { .. }),
        "non-vacuity: the two inputs must be refused by DIFFERENT rungs, or this pair proves \
         only that the loader says no to everything",
    );
}

// ===========================================================================================
// (4) T-04-36: zero command-line-crate types, asserted with a two-sided control
// ===========================================================================================

/// This file's own source names no type from the command-line crate — OPS-01's import half.
///
/// # The needles are BUILT, never written
///
/// A self-scan for a literal needle would find its own needle and could never report zero.
/// Both spellings — underscored and dashed — are assembled at runtime from pieces, and the
/// control file is opened through a path assembled the same way, because `include_str!`
/// takes a literal and that literal would itself be an occurrence.
///
/// # The two-sided control is the reason this is a measurement
///
/// A counter that can only return zero reports success on a tree where the thing it counts
/// has been deleted (the Phase 3 CR-02 lesson). That crate's own manifest is scanned with
/// the same counter and must return a POSITIVE count; only then does zero here mean anything.
#[test]
fn lifecycle_source_names_no_command_line_crate() {
    let this_file = include_str!("setfit_apr_lifecycle.rs");

    let underscored = format!("apr{}cli", '_');
    let dashed = format!("apr{}cli", '-');

    let control_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join(&dashed)
        .join("Cargo.toml");
    let cli_manifest = std::fs::read_to_string(&control_path).unwrap_or_else(|e| {
        panic!("the control manifest must be readable at {control_path:?}: {e}")
    });

    // POSITIVE CONTROL first: prove the counter can return non-zero.
    let control = cli_manifest.matches(dashed.as_str()).count();
    assert!(
        control > 0,
        "the control manifest does not contain the dashed spelling, so a zero below would be \
         vacuous",
    );

    assert_eq!(
        this_file.matches(underscored.as_str()).count(),
        0,
        "this file names the `{underscored}` crate; OPS-01 claims the lifecycle needs no CLI",
    );
    assert_eq!(
        this_file.matches(dashed.as_str()).count(),
        0,
        "this file names the `{dashed}` crate; OPS-01 claims the lifecycle needs no CLI",
    );

    // Non-vacuity on the SCAN ITSELF: assert we read the file we think we did. Without this,
    // an `include_str!` that resolved to an empty or wrong file would report two clean zeros.
    assert!(
        this_file.contains("load_setfit_apr"),
        "the scanned source does not name the production loader, so include_str! did not read \
         this file",
    );
    assert!(
        this_file.contains("into_artifact_bytes"),
        "the scanned source does not name the save-rung door, so include_str! did not read this \
         file",
    );
}
