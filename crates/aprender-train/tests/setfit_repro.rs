//! TRN-06 — two clean runs of the whole pipeline produce the same artifact. (03-10 T2, D-16.)
//!
//! # Three tests, three distinct claims, deliberately not fused
//!
//! | Test | Claim | What it cannot see |
//! |------|-------|--------------------|
//! | `setfit_repro_in_process_two_runs_agree` | the pipeline is a function of its inputs | shares the pool, allocator and lazily-initialized statics with itself |
//! | `setfit_repro_cross_process` | AUTHORITATIVE: two fresh processes at DIFFERENT fixed pool sizes agree | whether the agreed-on execution was the intended one |
//! | `setfit_repro_recorded_matches_expected_replay` | the run consumed the order it was supposed to | nothing about reproducibility |
//!
//! D-16 rejects the in-process form as the authoritative claim, and the reason is structural
//! rather than cautious: two runs in one process share the rayon pool, the allocator's free
//! lists and every `OnceLock`, which is precisely the class of nondeterminism a "clean run"
//! exists to expose. It is kept because it is the fast signal (tier2), and because a failure
//! there localizes the defect to the pipeline rather than to the environment.
//!
//! The third test is separate for the reason the phase-3 review gave: a single fused check
//! cannot distinguish "reproducible" from "correct". A run that consumed its pairs in the
//! wrong order reproduces that wrong order perfectly, so cross-process equality is silent
//! about it. This file therefore compares the RECORDED digests across processes, and
//! separately compares those recorded digests against an INDEPENDENTLY recomputed replay
//! derived from the seed and the selection. Two assertions, two failure meanings: the first
//! going red means the pipeline is not a function of its inputs; the second going red means
//! the pipeline is a faithful function of the wrong thing.
//!
//! # Every value read here comes from a PUBLIC accessor
//!
//! This is an out-of-crate integration test, so `pub(crate)` internals — including
//! `test_fixtures` — are invisible by construction. That is the point: the reproducibility
//! surface 03-08 landed is exercised through exactly the doors a downstream caller has. No
//! `#[doc(hidden)]` door was added and no visibility was widened for this file. One accessor
//! WAS added in 03-10 T2 (`batch_boundary_digest`), as a public read-only accessor in the
//! same counted block, because the recorded boundary digest had no public reader at all —
//! per plan, that is an 03-08 omission to fix in the open, not a reason for a backdoor.
//!
//! `SetFitMiniLm::from_slice_fixture` is reached through an existing `pub fn`. Its
//! reachability comes from the `conformance-fixtures` DEV-dependency edge 03-05 added to
//! this crate's `Cargo.toml`; no production feature was widened (see that manifest comment).
//!
//! # The fixture is re-derived here, not imported
//!
//! `test_fixtures::synthetic_dataset` is `pub(crate)`. The corpus below is the same
//! synthetic text under the same construction rules, re-spelled from the public API. It must
//! stay inside the MiniLM slice's 97-row vocabulary — the slice's `BertSentenceEncoder`
//! returns `VocabOutOfSlice` for anything outside it, digits included — which is why the
//! sentences read as they do. If this file and `test_fixtures` ever disagree, the cross-
//! process claim is still sound (both children build from THIS file); what would be lost is
//! the correspondence to the lib suite's numbers, so the two are kept in step deliberately.

#![cfg(feature = "setfit")]

use std::process::Command;

use aprender::setfit::SetFitMiniLm;
use aprender_contrastive_data::ledger::AccessLedger;
use aprender_contrastive_data::pairs::{PairConfig, PairSampler};
use aprender_contrastive_data::prepared::{Canonical, CanonicalDeclarations, PreparedDataset};
use aprender_contrastive_data::schema::LabeledExample;
use aprender_contrastive_data::select::{FewShotSelector, Selection, SelectionConfig};
use aprender_contrastive_data::split::SplitDeclaration;
use entrenar::train::setfit::config::{SetFitTrainConfig, SetFitTrainRequest};
use entrenar::train::setfit::epoch::epoch_pair_order;
use entrenar::train::setfit::verify::SerdeJsonCodec;
use entrenar::train::setfit::{ArtifactReloadedAndVerified, SetFitRun};
use sha2::{Digest, Sha256};

// ===========================================================================================
// The cell. CALIBRATED, and the seed is why.
// ===========================================================================================

/// The root seed. It is one of the three the 03-05 calibration matrix swept, so
/// `tune_encoder` reaches the evidence-gate COMPARISON instead of being refused for being
/// outside the calibrated regime. A run refused at the regime check would never produce a
/// digest, and this whole file would compare two absences.
const ROOT_SEED: u64 = 1;
/// Shots per class, epochs, pair batch size, explicit pair budget — the `s8e1b4` boundary
/// cell. Three optimizer steps, which is a test rather than a training run.
const SHOTS_PER_CLASS: u32 = 8;
const EPOCHS: u32 = 1;
const BATCH_SIZE: u32 = 4;
const BUDGET: u64 = 12;

/// Training rows per class in the synthetic corpus.
const TRAIN_PER_CLASS: usize = 16;
/// Declared classes.
const CLASSES: usize = 3;
const LABEL_NAMES: [&str; CLASSES] = ["alpha", "beta", "gamma"];

/// Per-class sentence material, all drawn from the slice's retained token strings.
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
/// Held-out material for validation and test, disjoint from every train row — a cross-split
/// duplicate is coalesced by the ingest ladder and would silently shrink a class pool.
const HELDOUT_MODIFIERS: [&str; 2] = ["naive", "shorter"];

// ===========================================================================================
// The pipeline, built through shipped doors only
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
///
/// `CARGO_MANIFEST_DIR` here is `crates/aprender-train`, so the default is the sibling
/// crate's fixture tree.
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

fn config() -> SetFitTrainConfig {
    let reference = SetFitTrainConfig::reference_defaults(ROOT_SEED);
    let mut pair_config = PairConfig::new(ROOT_SEED);
    pair_config.budget = Some(BUDGET);
    SetFitTrainConfig::new(SetFitTrainRequest {
        encoder_lr: reference.encoder_lr(),
        epochs: EPOCHS,
        batch_size: BATCH_SIZE,
        warmup_ratio: reference.warmup_ratio(),
        grad_clip_max_norm: reference.grad_clip_max_norm(),
        max_length: reference.max_length(),
        pair_config,
        freeze_policy: Vec::new(),
        head_regularization: reference.head_regularization(),
        root_seed: ROOT_SEED,
        device: "cpu".to_string(),
        lr_schedule: reference.lr_schedule(),
    })
    .expect("the fixture configuration satisfies the twelve-knob table")
}

fn selection(dataset: &PreparedDataset<Canonical>) -> Selection {
    let mut ledger = AccessLedger::new();
    FewShotSelector::select(
        dataset,
        &SelectionConfig { root_seed: ROOT_SEED, shots_per_class: SHOTS_PER_CLASS },
        &mut ledger,
    )
    .expect("the synthetic corpus must support this selection")
}

/// `prepare` -> `tune_encoder` -> `fit_head` -> `verify_artifact`, all public doors.
fn verified_run() -> SetFitRun<ArtifactReloadedAndVerified> {
    let dataset = synthetic_dataset();
    let selection = selection(&dataset);
    let encoder = SetFitMiniLm::from_slice_fixture(&fixtures_dir(), ROOT_SEED)
        .expect("the frozen MiniLM slice fixture must load");
    SetFitRun::prepare(encoder, dataset, selection, config())
        .expect("the fixture run must prepare")
        .tune_encoder()
        .expect("a run at a measured seed and cell must pass the evidence gate")
        .fit_head()
        .expect("the head must fit on the fixture's unique encode-once rows")
        .verify_artifact(&SerdeJsonCodec::new())
        .expect("a faithful codec must complete the round trip")
}

// ===========================================================================================
// The composite report
// ===========================================================================================

/// The ten components of the composite hash, plus the observed pool size.
///
/// Every field is sourced from a PUBLIC accessor on the verified run, and every hash-shaped
/// one is a digest the run RECORDED rather than one recomputed from configuration. That
/// distinction is the whole reason this comparison means anything: an accessor that rebuilt
/// its value from the config would make two runs that consumed the same WRONG order agree
/// perfectly, and the equality below would be a tautology about the config file.
#[derive(Debug, PartialEq, Eq)]
struct Report {
    selection: String,
    pair_order: String,
    batches: String,
    steps: u64,
    loss_trace: String,
    evidence_table: String,
    ledger: String,
    registry: String,
    artifact: String,
    predictions: String,
}

/// SHA-256 over the reloaded model's answers on the probe rows.
///
/// Ids and labels as length-prefixed bytes, embeddings and probabilities as LE BIT PATTERNS —
/// a rendered float would round away exactly the low-bit divergence this gate exists to
/// detect, and a bare concatenation of ids would let `["ab","c"]` and `["a","bc"]` collide.
fn probe_digest(run: &SetFitRun<ArtifactReloadedAndVerified>) -> String {
    let probe = run.probe_predictions();
    let mut h = Sha256::new();
    for id in probe.ids() {
        h.update((id.len() as u64).to_le_bytes());
        h.update(id.as_bytes());
    }
    for label in probe.labels() {
        h.update((label.len() as u64).to_le_bytes());
        h.update(label.as_bytes());
    }
    for row in probe.embeddings() {
        h.update((row.len() as u64).to_le_bytes());
        for v in row {
            h.update(v.to_bits().to_le_bytes());
        }
    }
    for row in probe.probabilities() {
        h.update((row.len() as u64).to_le_bytes());
        for v in row {
            h.update(v.to_bits().to_le_bytes());
        }
    }
    hex::encode(h.finalize())
}

fn report_of(run: &SetFitRun<ArtifactReloadedAndVerified>) -> Report {
    Report {
        selection: run.selection_semantic_hash(),
        pair_order: run.pair_order_digest().to_string(),
        batches: format!("{}:{}", run.batch_boundaries().len(), run.batch_boundary_digest()),
        steps: run.step_count(),
        loss_trace: run.loss_trace_hash().to_string(),
        evidence_table: run.evidence_table_hash().to_string(),
        ledger: run.encode_ledger_hash(),
        registry: run.parameter_registry_hash().to_string(),
        artifact: run.artifact_hash(),
        predictions: probe_digest(run),
    }
}

/// The labeled lines a child prints, in a fixed order.
fn print_report(report: &Report, threads: usize) {
    println!("SELECTION={}", report.selection);
    println!("PAIRORDER={}", report.pair_order);
    println!("BATCHES={}", report.batches);
    println!("STEPS={}", report.steps);
    println!("LOSSTRACE={}", report.loss_trace);
    println!("EVIDENCE={}", report.evidence_table);
    println!("LEDGER={}", report.ledger);
    println!("REGISTRY={}", report.registry);
    println!("ARTIFACT={}", report.artifact);
    println!("PREDICTIONS={}", report.predictions);
    println!("THREADS={threads}");
}

// ===========================================================================================
// (1) The child
// ===========================================================================================

/// One child process: run the pipeline once and print the composite report.
///
/// A no-op unless `SETFIT_REPRO_CHILD=1`, so an ordinary `cargo test` run executes it as a
/// trivial pass and only the parent drives the expensive path.
#[test]
fn setfit_repro_child() {
    if std::env::var("SETFIT_REPRO_CHILD").as_deref() != Ok("1") {
        return;
    }
    let run = verified_run();
    // The pool size this process ACTUALLY got. `RAYON_NUM_THREADS` says what was requested;
    // CLAUDE.md verification discipline 2 — never label a run by intent.
    print_report(&report_of(&run), rayon::current_num_threads());
}

// ===========================================================================================
// (2) In-process: the fast, structurally blind half of D-16
// ===========================================================================================

/// Two runs in ONE process agree on every component.
///
/// This is D-16's tier2 signal and it is NOT the authoritative claim: both runs share the
/// rayon pool, the allocator's state and every lazily-initialized static, so it passes for
/// the wrong reason exactly when the wrong reason is present. `setfit_repro_cross_process`
/// below is the claim; this one localizes a failure to the pipeline when it fires.
#[test]
fn setfit_repro_in_process_two_runs_agree() {
    let first = report_of(&verified_run());
    let second = report_of(&verified_run());

    // Non-vacuity: a run that produced nothing would make every comparison below trivially
    // true. Assert the components are populated BEFORE asserting they agree.
    assert!(!first.artifact.is_empty(), "the run must have produced an artifact hash");
    assert!(first.steps > 0, "the tuning loop must have taken at least one step");
    assert!(!first.pair_order.is_empty(), "the loop must have recorded a pair digest");

    assert_eq!(first.selection, second.selection, "selection semantic hash");
    assert_eq!(first.pair_order, second.pair_order, "recorded consumed-pair digest");
    assert_eq!(first.batches, second.batches, "recorded batch-boundary count and digest");
    assert_eq!(first.steps, second.steps, "step count");
    assert_eq!(first.loss_trace, second.loss_trace, "loss-trace hash");
    assert_eq!(first.evidence_table, second.evidence_table, "evidence-table hash");
    assert_eq!(first.ledger, second.ledger, "encode-ledger hash");
    assert_eq!(first.registry, second.registry, "parameter-registry hash");
    assert_eq!(first.artifact, second.artifact, "artifact hash");
    assert_eq!(first.predictions, second.predictions, "probe predictions");
    assert_eq!(first, second, "every component of the composite report");
}

// ===========================================================================================
// (3) Cross-process: the AUTHORITATIVE claim
// ===========================================================================================

/// Fixed pool sizes, never derived from the host.
///
/// A constrained runner's maximum can be 1, so a size taken from what the machine offers
/// would make a "the pools differed" assertion unsatisfiable rather than skipped — the
/// anti-pattern 03-02's gate records by name and this file avoids by construction.
const POOL_SIZES: [usize; 2] = [1, 3];

struct ChildReport {
    requested: usize,
    threads: usize,
    report: Report,
}

fn field(stdout: &str, key: &str) -> String {
    stdout
        .lines()
        .find_map(|line| line.trim().strip_prefix(key))
        .unwrap_or_else(|| panic!("child printed no {key} line:\n{stdout}"))
        .to_string()
}

fn run_child(requested: usize, exe: &std::path::Path) -> ChildReport {
    let output = Command::new(exe)
        // The child's EXACT test name plus `--exact --nocapture`. A libtest binary runs the
        // tests its FILTER selects; it does not run arbitrary code because an env var
        // happens to be set, so an env-var-only spawn would execute nothing at all.
        .args(["setfit_repro_child", "--exact", "--nocapture"])
        .env("SETFIT_REPRO_CHILD", "1")
        .env("RAYON_NUM_THREADS", requested.to_string())
        .output()
        .expect("re-invoking the test binary must succeed");

    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    assert!(
        output.status.success(),
        "child at RAYON_NUM_THREADS={requested} failed: {stdout}\n{}",
        String::from_utf8_lossy(&output.stderr)
    );

    ChildReport {
        requested,
        threads: field(&stdout, "THREADS=").parse().expect("THREADS must be a number"),
        report: Report {
            selection: field(&stdout, "SELECTION="),
            pair_order: field(&stdout, "PAIRORDER="),
            batches: field(&stdout, "BATCHES="),
            steps: field(&stdout, "STEPS=").parse().expect("STEPS must be a number"),
            loss_trace: field(&stdout, "LOSSTRACE="),
            evidence_table: field(&stdout, "EVIDENCE="),
            ledger: field(&stdout, "LEDGER="),
            registry: field(&stdout, "REGISTRY="),
            artifact: field(&stdout, "ARTIFACT="),
            predictions: field(&stdout, "PREDICTIONS="),
        },
    }
}

/// TWO SEPARATE PROCESSES at DIFFERENT fixed pool sizes report identical composite hashes.
///
/// This is TRN-06's authoritative claim (D-16). Each child gets a fresh rayon pool, a fresh
/// allocator and fresh statics, so an agreement here cannot be an artifact of shared process
/// state — and the pool sizes differ, so it cannot be an artifact of a single partitioning
/// either.
#[test]
fn setfit_repro_cross_process() {
    let exe = std::env::current_exe().expect("the test binary must know its own path");
    let children: Vec<ChildReport> = POOL_SIZES.iter().map(|n| run_child(*n, &exe)).collect();

    for c in &children {
        println!(
            "child RAYON_NUM_THREADS={} -> THREADS={} ARTIFACT={}",
            c.requested, c.threads, c.report.artifact
        );
    }

    // (a) MECHANISM ENGAGED. Setting the env var says what was requested; this asserts what
    //     rayon actually built. Without it, both children could have run on one thread and
    //     the equality below would measure one pool size twice.
    for c in &children {
        assert_eq!(
            c.threads, c.requested,
            "child asked for RAYON_NUM_THREADS={} but rayon built a pool of {}; the env var \
             did not reach the pool, so this run measures one pool size twice",
            c.requested, c.threads
        );
    }
    let mut observed: Vec<usize> = children.iter().map(|c| c.threads).collect();
    observed.sort_unstable();
    observed.dedup();
    assert_eq!(
        observed,
        POOL_SIZES.to_vec(),
        "the two children did not run at two DISTINCT pool sizes, so this comparison is \
         blind to exactly the thread-count dependence it exists to detect"
    );

    // (b) Every component of the composite hash agrees.
    let (a, b) = (&children[0], &children[1]);
    let named: [(&str, &String, &String); 9] = [
        ("selection semantic hash", &a.report.selection, &b.report.selection),
        ("recorded consumed-pair digest", &a.report.pair_order, &b.report.pair_order),
        ("recorded batch-boundary count:digest", &a.report.batches, &b.report.batches),
        ("loss-trace hash", &a.report.loss_trace, &b.report.loss_trace),
        ("evidence-table hash", &a.report.evidence_table, &b.report.evidence_table),
        ("encode-ledger hash", &a.report.ledger, &b.report.ledger),
        ("parameter-registry hash", &a.report.registry, &b.report.registry),
        ("artifact hash", &a.report.artifact, &b.report.artifact),
        ("probe predictions", &a.report.predictions, &b.report.predictions),
    ];
    for (what, left, right) in named {
        assert_eq!(
            left, right,
            "{what}: THREADS={} gave {left} but THREADS={} gave {right} — the pipeline \
             DEPENDS ON THE RAYON POOL SIZE, so TRN-06's two-clean-runs guarantee does not \
             hold on this host",
            a.threads, b.threads
        );
    }
    assert_eq!(a.report.steps, b.report.steps, "step count");
    assert_eq!(a.report, b.report, "every component of the composite report");
}

// ===========================================================================================
// (4) Recorded vs. independently recomputed — a SEPARATE claim
// ===========================================================================================

/// The RECORDED digests equal an independently recomputed expected replay.
///
/// # Why this is a separate test and not another assertion in the one above
///
/// `setfit_repro_cross_process` proves the run is REPRODUCIBLE: two fresh processes agree.
/// It is structurally silent about whether the agreed-on execution was the INTENDED one — a
/// loop that consumed its pairs in the wrong order reproduces that wrong order exactly, in
/// every process, forever. This test proves the run consumed the order the protocol
/// specifies, and says nothing about reproducibility.
///
/// Fusing them would produce one green light with two meanings and no way to tell
/// "reproducibly wrong" from "correct" when it goes red. So: distinct tests, distinct
/// failure messages, and neither can pass on the other's behalf.
///
/// The recomputation is INDEPENDENT of the tuning loop: the order comes from the public
/// `epoch_pair_order(seed, epoch, n)`, the batches from the consecutive-window rule, the
/// endpoints from a freshly constructed `PairSampler`, and the absorption format is spelled
/// out here in the same field order the contract fixes. Nothing is read back out of the run
/// except the two digests under test.
#[test]
fn setfit_repro_recorded_matches_expected_replay() {
    let run = verified_run();

    // The sampler is rebuilt from the run's own selection and pair config, both public.
    let sampler = PairSampler::new(run.selection(), run.config().requested().pair_config())
        .expect("the fixture selection must support a sampler");
    let n_pairs = sampler.budget();
    let batch_size = usize::try_from(BATCH_SIZE).expect("the batch size fits a usize");

    let mut pair_hasher = Sha256::new();
    let mut boundary_hasher = Sha256::new();
    let mut expected_boundaries: Vec<(u32, u64, u32)> = Vec::new();
    let mut global_step: u64 = 0;

    for epoch in 0..EPOCHS {
        let order = epoch_pair_order(ROOT_SEED, epoch, n_pairs);
        for (batch_index, chunk) in order.chunks(batch_size).enumerate() {
            let batch_index = u32::try_from(batch_index).expect("the batch index fits a u32");
            let start = chunk.first().copied().unwrap_or(0);
            let len = u32::try_from(chunk.len()).expect("the batch length fits a u32");

            // Boundary absorption, in the contracted field order.
            boundary_hasher.update(epoch.to_le_bytes());
            boundary_hasher.update(batch_index.to_le_bytes());
            boundary_hasher.update(global_step.to_le_bytes());
            boundary_hasher.update(start.to_le_bytes());
            boundary_hasher.update(len.to_le_bytes());
            expected_boundaries.push((epoch, start, len));

            // Pair absorption, once per pair, at the position the loop draws it.
            for (position, &ordinal) in chunk.iter().enumerate() {
                let position = u32::try_from(position).expect("the position fits a u32");
                let labeled = sampler.pair_at(ordinal).expect("every ordinal must resolve");
                pair_hasher.update(epoch.to_le_bytes());
                pair_hasher.update(batch_index.to_le_bytes());
                pair_hasher.update(position.to_le_bytes());
                pair_hasher.update(ordinal.to_le_bytes());
                pair_hasher.update(labeled.pair.lo().ordinal().to_le_bytes());
                pair_hasher.update(labeled.pair.hi().ordinal().to_le_bytes());
                pair_hasher.update(labeled.target.to_bits().to_le_bytes());
            }
            global_step += 1;
        }
    }

    // Non-vacuity: an empty replay would agree with an empty recording.
    assert!(n_pairs > 0, "the replay must cover at least one pair");
    assert!(!expected_boundaries.is_empty(), "the replay must open at least one batch");
    assert_eq!(
        global_step,
        run.step_count(),
        "the replay opened {global_step} batches but the run recorded {} steps; the two are \
         describing different executions and neither digest below would mean anything",
        run.step_count(),
    );

    assert_eq!(
        run.pair_order_digest(),
        hex::encode(pair_hasher.finalize()),
        "the RECORDED consumed-pair digest does not match the order the protocol specifies: \
         the run is a faithful function of the WRONG consumption order (cross-process \
         equality cannot see this)",
    );
    assert_eq!(
        run.batch_boundary_digest(),
        hex::encode(boundary_hasher.finalize()),
        "the RECORDED batch-boundary digest does not match the consecutive-window rule",
    );
    assert_eq!(
        run.batch_boundaries(),
        expected_boundaries.as_slice(),
        "the recorded (epoch, start_ordinal, len) triples do not match the expected windows",
    );
}
