//! Canonical integrity and schema gate over the committed SetFit pair-count reference
//! fixtures (plan 02-04, `FALSIFY-CPP-016` / `FALSIFY-CPP-017`).
//!
//! Run with `cargo test -p aprender-contrastive-data --test reference_fixtures`. The
//! result is identical from the repository root and from `crates/aprender-contrastive-data/`
//! because every path resolves against `CARGO_MANIFEST_DIR` rather than the process
//! working directory — see `tests/common/mod.rs`.
//!
//! The fixture models live in `tests/common/mod.rs`, NOT here: each file under `tests/`
//! is its own crate, so plans 02-07 and 02-08 could not import them from this one.

// `tests/common/mod.rs` is compiled into EVERY integration-test crate that names it, so an
// item this file does not call is dead code HERE while being live in `pair_counts.rs` (and,
// from plan 02-08, in the negative gates). The allow is on the module so the shared
// definitions stay in one place instead of being duplicated per consumer.
#[allow(dead_code)]
mod common;

use common::{ContractedFixture, MeasuredFixture};

/// The six layouts the phase reasons about. Named here so a fixture that quietly stops
/// being emitted fails a test instead of shrinking the evidence base unnoticed.
const EXPECTED_IDS: [&str; 6] = [
    "4_1",
    "64_64_64",
    "8_4_8",
    "8_4_8_maxpairs100",
    "8_8_8",
    "singletons_32",
];

#[test]
fn manifest_covers_every_fixture_and_every_digest_matches() {
    let drift = common::manifest_drift();
    assert!(
        drift.is_empty(),
        "fixture integrity failed:\n  {}",
        drift.join("\n  ")
    );
    assert_eq!(
        common::manifest_names().len(),
        2 * EXPECTED_IDS.len(),
        "manifest should cover one measured and one contracted fixture per layout"
    );
}

#[test]
fn no_fixture_on_disk_is_missing_from_the_manifest() {
    // The digest check above proves that everything LISTED is intact. It says nothing
    // about a file that was added to the directory and never listed — which is how an
    // unprotected fixture would enter the evidence base.
    let listed = common::manifest_names();
    // VACUITY GUARD. With no files and no manifest entries the orphan set is empty and
    // this test is green while checking nothing — observed for real when the loaders
    // were still stubs. Pin the population first.
    assert_eq!(
        common::fixture_files().len(),
        2 * EXPECTED_IDS.len(),
        "expected one measured and one contracted fixture per layout on disk"
    );
    let orphans: Vec<String> = common::fixture_files()
        .into_iter()
        .filter(|name| !listed.contains(name))
        .collect();
    assert!(
        orphans.is_empty(),
        "fixture files present on disk but absent from manifest.sha256: {orphans:?}"
    );
    assert_eq!(
        common::fixture_files().len(),
        listed.len(),
        "manifest and directory disagree on how many fixtures exist"
    );
}

fn assert_measured_shape(f: &MeasuredFixture) {
    let id = &f.fixture_id;
    assert_eq!(f.fixture_family, "setfit_measured", "{id}: wrong family");
    assert_eq!(f.sampling_strategy, "oversampling", "{id}: wrong strategy");
    assert!(!f.multilabel, "{id}: fixtures are single-label");
    assert_eq!(
        f.n_examples,
        f.layout.iter().sum::<u64>(),
        "{id}: n_examples disagrees with the layout"
    );
    assert_eq!(f.n_classes, f.layout.len() as u64, "{id}: n_classes wrong");
}

fn assert_measured_counts(f: &MeasuredFixture) {
    let id = &f.fixture_id;
    // Oversampling balances both lists to their maximum, so the epoch length is exactly
    // twice that maximum. Asserted rather than trusted: this is the relation that makes
    // `total` meaningful.
    assert_eq!(f.len_pos, f.len_neg, "{id}: oversampling must balance");
    assert_eq!(
        f.len_pos,
        f.stored_pos.max(f.stored_neg),
        "{id}: len != max"
    );
    assert_eq!(
        f.total,
        f.len_pos + f.len_neg,
        "{id}: total != len_pos+len_neg"
    );
    // `np.triu_indices` walks i <= j, so one unordered pair cannot appear both ways.
    assert_eq!(f.orientation_duplicate_count, 0, "{id}: orientation dup");
    assert!(
        f.self_pair_count <= f.stored_pos,
        "{id}: more self-pairs than stored positives"
    );
    // Uncapped, every example contributes exactly one diagonal entry. Under a cap, WHICH
    // positives survive depends on the reference's hardcoded permutation, so the fixture
    // must say so instead of pretending the number is layout-derived.
    if f.max_pairs == -1 {
        assert!(
            f.rng_dependent_fields.is_empty(),
            "{id}: unexpected rng dep"
        );
        assert_eq!(f.self_pair_count, f.n_examples, "{id}: diagonal incomplete");
    } else {
        assert_eq!(
            f.rng_dependent_fields,
            vec!["self_pair_count".to_string()],
            "{id}: a capped fixture must declare its RNG-dependent field"
        );
    }
}

fn assert_measured_provenance(f: &MeasuredFixture) {
    let id = &f.fixture_id;
    assert!(
        f.derivation.contains("MEASURED"),
        "{id}: a measured fixture must say it was measured, not computed"
    );
    assert!(
        f.reference_notes.len() >= 4,
        "{id}: reference notes missing"
    );
    assert!(!f.why_this_layout.is_empty(), "{id}: layout unexplained");
    assert_eq!(f.setfit_version, "1.1.3", "{id}: wrong reference pin");
    assert_eq!(f.uv_lock_sha256.len(), 64, "{id}: uv.lock digest malformed");
    assert!(
        f.uv_version.starts_with("uv "),
        "{id}: uv version malformed"
    );
}

#[test]
fn every_measured_fixture_deserializes_and_is_internally_consistent() {
    let measured = common::load_measured();
    for id in EXPECTED_IDS {
        assert!(measured.contains_key(id), "missing measured fixture `{id}`");
    }
    assert_eq!(measured.len(), EXPECTED_IDS.len());

    for f in measured.values() {
        assert_measured_shape(f);
        assert_measured_counts(f);
        assert_measured_provenance(f);
    }

    // SetFit's own documented worked example. The docs claim 62 positives; the pinned
    // implementation stores 82, because it includes the 20-entry diagonal.
    let worked = &measured["8_4_8"];
    assert_eq!(worked.stored_pos, 82, "measured [8,4,8] stored positives");
    assert_eq!(worked.stored_neg, 128, "measured [8,4,8] stored negatives");
    assert_eq!(worked.self_pair_count, 20, "measured [8,4,8] self-pairs");
    assert_eq!(worked.total, 256, "measured [8,4,8] epoch length");
    // The cap is per LIST (max_pairs // 2), not a total.
    assert_eq!(measured["8_4_8_maxpairs100"].stored_pos, 50);
    assert_eq!(measured["8_4_8_maxpairs100"].stored_neg, 50);
}

fn assert_contracted_shape(f: &ContractedFixture) {
    let id = &f.fixture_id;
    assert_eq!(f.fixture_family, "aprender_contracted", "{id}: family");
    assert!(
        f.self_pairs_excluded,
        "{id}: self-pairs are excluded (D-14)"
    );
    assert_eq!(
        f.n_examples,
        f.layout.iter().sum::<u64>(),
        "{id}: n_examples disagrees with the layout"
    );
    assert_eq!(f.n_classes, f.layout.len() as u64, "{id}: n_classes wrong");
    assert!(!f.why_this_layout.is_empty(), "{id}: layout unexplained");
    assert_eq!(
        f.measured_counterpart,
        format!("setfit_measured_{id}.json"),
        "{id}: counterpart filename must resolve"
    );
    assert!(
        !f.divergence_note.is_empty(),
        "{id}: divergence unexplained"
    );
    assert!(
        f.derivation.contains("COMPUTED"),
        "{id}: a contracted fixture is computed from closed forms, never measured"
    );
}

fn assert_contracted_budget(f: &ContractedFixture) {
    let id = &f.fixture_id;
    assert_eq!(
        f.closed_form_budget,
        2 * f.positive_capacity.max(f.negative_capacity),
        "{id}: closed form must be 2*max(pos, neg)"
    );
    assert_eq!(
        f.default_epoch_budget,
        f.closed_form_budget.min(f.hard_cap),
        "{id}: default budget must be the clamped closed form"
    );
    assert_eq!(
        f.clamp_engaged,
        f.closed_form_budget > f.hard_cap,
        "{id}: clamp flag must record whether the cap actually bound"
    );
    assert_eq!(
        f.resolved_budget,
        f.explicit_budget.unwrap_or(f.default_epoch_budget),
        "{id}: an explicit budget overrides the default and is never silently clamped"
    );
    assert_eq!(
        f.resolved_pos_count + f.resolved_neg_count,
        f.resolved_budget,
        "{id}: the emitted split must exhaust the resolved budget"
    );
}

fn assert_contracted_deviation(f: &ContractedFixture) {
    let id = &f.fixture_id;
    // PF-008: the exclusion is OURS. Attributing it to SetFit would be a false claim
    // about code that does the opposite.
    assert_eq!(f.deviation_attribution, "aprender", "{id}: attribution");
    assert_eq!(
        f.deviation_clauses.len(),
        3,
        "{id}: OBLIG-CPP-DEVIATION-DECLARED has exactly three clauses"
    );
    let ids: Vec<&str> = f
        .deviation_clauses
        .iter()
        .map(|c| c.clause_id.as_str())
        .collect();
    assert_eq!(
        ids,
        vec!["sampled_identities", "capped_count", "self_pairs_excluded"],
        "{id}: clause identities and order are contracted"
    );
    assert!(
        f.deviation_clauses
            .iter()
            .all(|c| !c.statement.trim().is_empty()),
        "{id}: an empty clause statement declares nothing"
    );
}

#[test]
fn every_contracted_fixture_deserializes_and_is_internally_consistent() {
    let contracted = common::load_contracted();
    for id in EXPECTED_IDS {
        assert!(contracted.contains_key(id), "missing contracted `{id}`");
    }
    assert_eq!(contracted.len(), EXPECTED_IDS.len());

    for f in contracted.values() {
        assert_contracted_shape(f);
        assert_contracted_budget(f);
        assert_contracted_deviation(f);
    }

    let worked = &contracted["8_4_8"];
    assert_eq!(worked.positive_capacity, 62, "contracted [8,4,8] positives");
    assert_eq!(
        worked.negative_capacity, 128,
        "contracted [8,4,8] negatives"
    );
    assert_eq!(
        worked.default_epoch_budget, 256,
        "contracted [8,4,8] budget"
    );
    assert_eq!(contracted["8_8_8"].default_epoch_budget, 384, "8-shot D-14");
    assert_eq!(
        contracted["64_64_64"].default_epoch_budget, 24576,
        "64-shot D-14"
    );
}

#[test]
fn layout_4_1_divergence_is_recorded_in_both_families() {
    // FALSIFY-CPP-017. Asserting either number alone would let the next reader "fix" the
    // correct side; the divergence itself is the artifact.
    let measured = common::load_measured();
    let contracted = common::load_contracted();
    let m = &measured["4_1"];
    let c = &contracted["4_1"];

    assert_eq!(m.total, 22, "pinned reference epoch length on [4,1]");
    assert_eq!(m.stored_pos, 11, "reference stores 11 positives on [4,1]");
    assert_eq!(m.self_pair_count, 5, "five of which are self-pairs");
    assert_eq!(c.default_epoch_budget, 12, "Aprender's contracted budget");
    assert_eq!(c.positive_capacity, 6, "C(4,2) — the singleton adds none");
    assert_eq!(c.negative_capacity, 4, "4 * 1");
    assert_ne!(
        m.total, c.default_epoch_budget,
        "the [4,1] families must DISAGREE; if they now agree, either the reference \
         fixture was regenerated from the docs or Aprender started emitting self-pairs"
    );
    assert_eq!(
        c.measured_total, m.total,
        "the contracted row must quote the \
         measured number it diverges from, so the divergence is visible in one file"
    );
}

#[test]
fn k_equals_n_adversarial_layout_has_a_contracted_reference_row() {
    // Plan 02-08's capacity gate reads THIS row rather than a hand-typed constant. A
    // three-class fixture set could never expose an O(K^2) sampler.
    let contracted = common::load_contracted();
    let f = &contracted["singletons_32"];
    assert_eq!(f.layout.len(), 32, "K = N = 32");
    assert!(f.layout.iter().all(|&n| n == 1), "every class a singleton");
    assert_eq!(
        f.positive_capacity, 0,
        "a singleton contributes no positives"
    );
    assert_eq!(f.negative_capacity, 496, "C(32, 2)");
    assert_eq!(f.default_epoch_budget, 992, "2 * 496");
    assert_eq!(
        f.degenerate_case.as_deref(),
        Some("negatives_only"),
        "pos == 0 and neg > 0 is a DEFINED degenerate case, not an error"
    );
    assert_eq!(f.resolved_pos_count, 0, "no positives can be emitted");
    assert_eq!(
        f.resolved_neg_count, 992,
        "so the whole budget is negatives"
    );
}

#[test]
fn loaders_pair_up_every_fixture_and_carry_the_environment_attestation() {
    // Test 7 proves the loaders are a usable surface by USING them, which is the only
    // thing that keeps them honest once 02-07 and 02-08 depend on them.
    let measured = common::load_measured();
    let contracted = common::load_contracted();
    // VACUITY GUARD, as above: two empty maps have equal key sets and an empty loop body
    // asserts nothing. This test was green against the stub loaders until this line.
    assert_eq!(
        measured.len(),
        EXPECTED_IDS.len(),
        "the loaders must actually load something"
    );
    assert_eq!(
        measured.keys().collect::<Vec<_>>(),
        contracted.keys().collect::<Vec<_>>(),
        "every measured layout needs its contracted counterpart, and vice versa"
    );

    for (id, m) in &measured {
        let c = &contracted[id];
        assert_eq!(
            m.layout, c.layout,
            "{id}: the two families must share a layout"
        );
        // A version string alone does not identify an environment: the same `setfit
        // 1.1.3` resolves differently under a different lockfile or resolver.
        assert_eq!(m.setfit_version, c.setfit_version, "{id}: setfit pin");
        assert_eq!(m.uv_lock_sha256, c.uv_lock_sha256, "{id}: uv.lock digest");
        assert_eq!(m.uv_version, c.uv_version, "{id}: uv version");
        assert!(!m.uv_lock_sha256.is_empty(), "{id}: unattested environment");
    }
}
