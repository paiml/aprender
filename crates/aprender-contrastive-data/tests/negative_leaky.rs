//! D-25 — the leakage gate is not theater.
//!
//! Obligation: `OBLIG-CPP-LEAKY-RED`. Requirement: DATA-06. This is the PF-002 discipline
//! `crates/aprender-core/tests/setfit_conformance/detach_negative.rs` established in
//! Phase 1 (D-24), applied to the pair boundary.
//!
//! A leakage gate that has only ever been observed PASSING is not evidence. Every positive
//! assertion in this crate about "no validation row reaches the pair stream" is compatible
//! with a membership check that was never written, because the honest sampler never
//! produces one to catch. This file builds the poisoned input on purpose and requires the
//! SAME public call that accepts the honest dump to REJECT it, naming the offender.
//!
//! # Why the poison is an `UntrustedPairRecord` and not a `LabeledPair`
//!
//! A trusted [`LabeledPair`](aprender_contrastive_data::pairs::LabeledPair) CANNOT
//! REPRESENT this attack: its endpoints are `SelectedId`s whose constructor is private to
//! `select.rs`, so an id that was never selected has no `SelectedId` to be. That is the
//! structural half of `split_span_fail_closed`, and it is why the poisoning surface has to
//! be the untrusted DTO — the bytes that arrive from a replayed or hand-edited dump. If
//! this file could construct the leak from trusted types, the typestate would be broken and
//! THAT would be the finding.
//!
//! # The three elements, all mandatory
//!
//! 1. **The negative** — the poisoned list is rejected, and the message NAMES the offending
//!    identifier, because that is what makes a real failure diagnosable rather than red.
//! 2. **The control** — the same list with the poisoned record REMOVED validates `Ok`.
//!    Without it, "the poisoned list is rejected" is equally satisfied by a gate that
//!    rejects everything, or by a dump that was malformed for some unrelated reason.
//! 3. **The mirror** — the identical call over the untouched dump returns `Ok` and yields
//!    exactly the sampler's own pairs.
//!
//! Run with `cargo test -p aprender-contrastive-data --test negative_leaky`.

// `tests/common/mod.rs` is compiled into EVERY integration-test crate that names it, so
// the fixture loaders this file does not call are dead code HERE while being the whole
// point of `reference_fixtures.rs`. The allow is on the module so the shared module stays
// a single definition.
#[allow(dead_code)]
mod common;

use aprender_contrastive_data::manifest::dump_pairs;
use aprender_contrastive_data::pairs::{
    parse_pair_dump, validate_pair_records, LabeledPair, PairConfig, PairSampler,
    UntrustedPairRecord,
};
use aprender_contrastive_data::select::Selection;
use aprender_contrastive_data::ContrastiveDataError;

/// Three classes, twenty training rows each, eight shots — 24 selected examples.
const CLASSES: usize = 3;
const TRAIN_PER_CLASS: usize = 20;
const ROOT_SEED: u64 = 13;
const SHOTS: u32 = 8;
/// Small enough to read in a failure message, large enough to hold both target values.
const BUDGET: u64 = 32;

/// The honest artifacts: a selection, the sampler's own pairs, and the parsed dump.
struct Honest {
    selection: Selection,
    sampler_pairs: Vec<LabeledPair>,
    records: Vec<UntrustedPairRecord>,
}

fn honest() -> Honest {
    let selection = common::synthetic_selection(CLASSES, TRAIN_PER_CLASS, ROOT_SEED, SHOTS);
    let cfg = PairConfig {
        budget: Some(BUDGET),
        ..PairConfig::new(ROOT_SEED)
    };
    let (sampler_pairs, bytes) = {
        let sampler =
            PairSampler::new(&selection, &cfg).expect("24 examples support a 32-pair budget");
        let pairs: Vec<LabeledPair> = sampler
            .iter_from(0)
            .expect("offset 0 is always within the budget")
            .collect();
        let mut bytes = Vec::new();
        dump_pairs(&sampler, &mut bytes).expect("dumping to a Vec cannot fail");
        (pairs, bytes)
    };
    let records = parse_pair_dump(&bytes).expect("the crate's own dump must parse");

    // Vacuity guard. Two empty vectors satisfy almost any relation between them, and an
    // empty record list validates `Ok` trivially — so the population is pinned BEFORE any
    // assertion is made over it (plan 02-04's lesson, which recurred three times in 02-07).
    assert_eq!(
        records.len(),
        BUDGET as usize,
        "the dump must hold one record per budgeted ordinal before anything is asserted \
         over it"
    );
    assert_eq!(sampler_pairs.len(), BUDGET as usize);

    Honest {
        selection,
        sampler_pairs,
        records,
    }
}

/// The index of the first record whose declared target is `want`.
fn first_record_with_target(records: &[UntrustedPairRecord], want: f32) -> usize {
    records
        .iter()
        .position(|record| record.target == want)
        .unwrap_or_else(|| panic!("the honest dump contains no record with target {want}"))
}

// ===============================================================================
// The mirror. Stated first, because every negative below is only meaningful
// relative to it.
// ===============================================================================

#[test]
fn mirror_the_same_call_accepts_the_honest_dump_and_reproduces_the_sampler() {
    let h = honest();
    let validated = validate_pair_records(&h.records, &h.selection)
        .expect("the identical call must ACCEPT the crate's own honest dump");

    // Not merely `Ok`: the round trip must recover the sampler's own stream. A validator
    // that accepted everything and returned garbage would satisfy an `is_ok()` assertion.
    assert_eq!(
        validated, h.sampler_pairs,
        "the validated dump must be pair-for-pair the stream the sampler emitted"
    );
}

// ===============================================================================
// Negative 1 — an endpoint that names a row outside the selection.
// ===============================================================================

#[test]
fn a_dump_endpoint_naming_a_validation_row_is_rejected_and_the_id_is_named() {
    let h = honest();

    // `validation:1-0` exists in the dataset and is deliberately NOT in the selection:
    // selection draws from the TRAIN split only. This is the leak PF-002 describes — a
    // held-out row reaching the contrastive objective through the pair channel.
    let leaked_id = common::synthetic_id("validation", 1, 0);
    assert!(
        h.selection.selected_id(&leaked_id).is_none(),
        "the poison must name a row the selection genuinely does not contain, or this \
         test proves nothing about membership"
    );

    let poisoned_index = 7;
    let mut leaky = h.records.clone();
    leaky[poisoned_index].lo = leaked_id.clone();

    // --- THE CONTROL, run BEFORE the rejection assertion. ---------------------------
    // The same list with the poisoned record REMOVED must validate `Ok`. Without this,
    // "the leaky list is rejected" would also hold against a gate that rejects everything,
    // or against a list that had become malformed for an unrelated reason.
    let mut without_poison = leaky.clone();
    without_poison.remove(poisoned_index);
    validate_pair_records(&without_poison, &h.selection).expect(
        "removing the single poisoned record must make the list valid again — the leak is \
         the ONLY defect in it",
    );

    // --- THE NEGATIVE. ---------------------------------------------------------------
    let error = validate_pair_records(&leaky, &h.selection).expect_err(
        "the membership gate ACCEPTED a pair endpoint naming a validation row. Every \
         positive leakage result in this crate is worthless.",
    );

    assert!(
        matches!(error, ContrastiveDataError::EndpointNotInSelection { .. }),
        "the rejection must be the MEMBERSHIP error, not some other refusal that happens \
         to be red: {error}"
    );
    let rendered = error.to_string();
    assert!(
        rendered.contains(&leaked_id),
        "the failure does not NAME the offending identifier `{leaked_id}`, so a real \
         failure would not be diagnosable: {rendered}"
    );
}

// ===============================================================================
// Negative 2 — a record that lies about its target.
// ===============================================================================

#[test]
fn a_dump_record_declaring_the_wrong_target_is_rejected_naming_both_targets() {
    let h = honest();

    // A membership-only gate would wave this through: both endpoints ARE in the selection.
    // The pair is simply relabeled — a cross-class pair claiming to be a positive, which is
    // the supervision signal itself being poisoned rather than the population.
    let poisoned_index = first_record_with_target(&h.records, 0.0);
    let mut poisoned = h.records.clone();
    poisoned[poisoned_index].target = 1.0;

    // --- THE CONTROL. ----------------------------------------------------------------
    // Restoring only the target makes the identical list valid, so the mislabeling is the
    // only defect.
    let mut restored = poisoned.clone();
    restored[poisoned_index].target = 0.0;
    validate_pair_records(&restored, &h.selection)
        .expect("restoring the target must make the identical list valid again");

    // --- THE NEGATIVE. ---------------------------------------------------------------
    let error = validate_pair_records(&poisoned, &h.selection).expect_err(
        "the validator ACCEPTED a cross-class pair declaring target 1.0. The target is \
         supposed to be DERIVED from the endpoints, never trusted from bytes.",
    );

    assert!(
        matches!(error, ContrastiveDataError::PairTargetMismatch { .. }),
        "a mislabeled target must be its own typed refusal, not a membership error: {error}"
    );
    let rendered = error.to_string();
    for probe in ["declares target 1", "derive 0"] {
        assert!(
            rendered.contains(probe),
            "the failure does not report both the declared and the derived target \
             (`{probe}` missing): {rendered}"
        );
    }
    assert!(
        rendered.contains(&poisoned[poisoned_index].lo)
            && rendered.contains(&poisoned[poisoned_index].hi),
        "the failure does not NAME the offending endpoints: {rendered}"
    );
}

// ===============================================================================
// The gate is ONE call, not two implementations.
// ===============================================================================

#[test]
fn every_leakage_assertion_here_goes_through_the_same_public_entry_point() {
    // The identity is the whole point of D-25: a second validator could be wrong in exactly
    // the way that lets both the honest and the poisoned list pass. The needle is assembled
    // at runtime so this scan cannot trip on itself.
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/negative_leaky.rs");
    let text = std::fs::read_to_string(&path).expect("this file is readable");
    let call = format!("{}{}", "validate_pair_", "records(&");
    let hits = text
        .lines()
        .filter(|line| line.split("//").next().unwrap_or("").contains(&call))
        .count();
    assert!(
        hits >= 5,
        "expected the honest mirror, both controls and both negatives to call the SAME \
         public entry point; found {hits} call sites"
    );
}
