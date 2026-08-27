//! Fixture-driven pair-count reproduction (plan 02-07, `FALSIFY-CPP-007` / `-009` / `-010`).
//!
//! WHY THIS FILE EXISTS RATHER THAN A `src/` UNIT TEST
//! ---------------------------------------------------
//! Every assertion here READS plan 02-04's committed fixtures, and two cargo facts make
//! that impossible anywhere else:
//!
//! * `tests/common/mod.rs` is a MODULE of each integration-test crate, so a lib unit test
//!   under `src/` cannot see `load_contracted()` at all; and
//! * `make contrastive-data-boundary` bans `std::fs` under `src/` outright, with no
//!   `cfg(test)` exemption, so a fixture-reading unit test could not open the file even if
//!   it could name the loader.
//!
//! Integration tests are outside the D-04 library boundary, which is why reading committed
//! fixtures here is correct and intended. The pure closed-form assertions over hand-typed
//! layouts live in `src/pairs.rs`; these are the ones that must not be hand-typed, because
//! a constant typed into the same file as the implementation proves only that someone typed
//! it twice.
//!
//! Run with `cargo test -p aprender-contrastive-data --test pair_counts`.

// `tests/common/mod.rs` is compiled into EVERY integration-test crate that names it, so
// the loaders this file does not call (`manifest_drift`, `fixture_files`, ...) are dead
// code HERE while being the whole point of `reference_fixtures.rs`. The allow is on the
// module rather than on individual items so the shared module stays a single definition.
#[allow(dead_code)]
mod common;

use aprender_contrastive_data::pairs::{
    classify_degenerate, default_epoch_budget, effective_default_budget, negative_capacity,
    positive_capacity, resolve_budget, EmittedKinds, PairConfig, DEFAULT_HARD_CAP,
};

/// The six contracted fixtures plan 02-04 committed. Pinned so a fixture that quietly
/// stops being emitted fails here rather than shrinking the evidence base unnoticed
/// (02-04's own vacuity lesson).
const EXPECTED_FIXTURE_COUNT: usize = 6;

#[test]
fn every_contracted_fixture_capacity_triple_is_reproduced() {
    let contracted = common::load_contracted();
    assert_eq!(
        contracted.len(),
        EXPECTED_FIXTURE_COUNT,
        "the fixture population must be pinned before a relation is asserted over it"
    );

    for (id, fixture) in &contracted {
        assert_eq!(
            positive_capacity(&fixture.layout).expect("a contracted layout never overflows"),
            fixture.positive_capacity,
            "{id}: positive_capacity"
        );
        assert_eq!(
            negative_capacity(&fixture.layout).expect("a contracted layout never overflows"),
            fixture.negative_capacity,
            "{id}: negative_capacity"
        );
        assert_eq!(
            default_epoch_budget(&fixture.layout).expect("a contracted layout never overflows"),
            fixture.closed_form_budget,
            "{id}: closed-form budget"
        );
        assert_eq!(
            effective_default_budget(&fixture.layout, fixture.hard_cap)
                .expect("a contracted layout never overflows"),
            fixture.default_epoch_budget,
            "{id}: effective default budget"
        );
        assert_eq!(
            fixture.hard_cap, DEFAULT_HARD_CAP,
            "{id}: the fixture's cap must be the contract-resident constant"
        );
    }
}

/// The K = N adversarial row, read from the fixture rather than typed here.
///
/// A three-class fixture set could never expose an O(K²) sampler — at K = 3, K² = 9 and
/// K = 3 are indistinguishable. This row is the one that can.
#[test]
fn the_k_equals_n_adversarial_row_comes_from_the_fixture_not_a_literal() {
    let contracted = common::load_contracted();
    let fixture = contracted
        .get("singletons_32")
        .expect("plan 02-04 commits the [1]*32 contracted fixture");

    assert_eq!(fixture.layout.len(), 32, "K = N = 32");
    assert!(
        fixture.layout.iter().all(|n| *n == 1),
        "every class must be a singleton or this row is not the adversarial one"
    );
    assert_eq!(fixture.n_examples, 32);
    assert_eq!(fixture.n_classes, 32);
    // The shared builder plan 02-08's capacity gate uses must agree with the fixture, or
    // the two artifacts would be measuring different layouts while both looking green.
    assert_eq!(common::all_singleton_layout(32), fixture.layout);
    assert_eq!(common::contracted_layout("singletons_32"), fixture.layout);
    assert!(common::ADVERSARIAL_BUDGET < fixture.resolved_budget);

    assert_eq!(
        positive_capacity(&fixture.layout).expect("no overflow"),
        fixture.positive_capacity
    );
    assert_eq!(
        negative_capacity(&fixture.layout).expect("no overflow"),
        fixture.negative_capacity
    );
    assert_eq!(
        classify_degenerate(fixture.positive_capacity, fixture.negative_capacity)
            .expect("a legal layout"),
        EmittedKinds::NegativesOnly
    );
    assert_eq!(
        fixture.degenerate_case.as_deref(),
        Some("negatives_only"),
        "the fixture and the policy must agree on the name, not only on the behaviour"
    );
}

/// The `[4, 1]` divergence row: a singleton class contributes zero positive capacity, so
/// Aprender's budget (12) is not the reference's (22). Asserting only the balanced layouts
/// would pass against a sampler that includes self-pairs.
#[test]
fn the_singleton_divergence_row_comes_from_the_fixture_not_a_literal() {
    let contracted = common::load_contracted();
    let fixture = contracted
        .get("4_1")
        .expect("plan 02-04 commits the [4, 1] contracted fixture");
    let measured = common::load_measured();
    let reference = measured
        .get("4_1")
        .expect("plan 02-04 commits the [4, 1] measured fixture");

    assert_eq!(fixture.layout, vec![4, 1]);
    assert_eq!(
        positive_capacity(&fixture.layout).expect("no overflow"),
        fixture.positive_capacity
    );
    assert_eq!(
        default_epoch_budget(&fixture.layout).expect("no overflow"),
        fixture.closed_form_budget
    );
    // Vacuity guard: this row is only evidence if the two families genuinely disagree.
    assert_ne!(
        fixture.closed_form_budget, reference.total,
        "the [4, 1] row must DIVERGE from the pinned reference or it proves nothing"
    );
    assert!(
        reference.self_pair_count > 0,
        "the divergence is caused by the reference's self-pairs; if it has none, the \
         fixture no longer records what this test claims"
    );
}

/// Budget RESOLUTION, not just the closed form: every fixture's `resolved_budget` and
/// clamp flag are reproduced through the shipped resolver.
#[test]
fn contracted_budget_resolution_matches_every_fixture_row() {
    let contracted = common::load_contracted();
    assert_eq!(contracted.len(), EXPECTED_FIXTURE_COUNT);

    let mut explicit_rows = 0;
    for (id, fixture) in &contracted {
        let cfg = PairConfig {
            budget: fixture.explicit_budget,
            hard_cap: Some(fixture.hard_cap),
            ..PairConfig::new(13)
        };
        if fixture.explicit_budget.is_some() {
            explicit_rows += 1;
        }
        let (budget, clamped) =
            resolve_budget(&cfg, &fixture.layout).expect("every contracted row resolves");
        assert_eq!(budget, fixture.resolved_budget, "{id}: resolved budget");
        assert_eq!(clamped, fixture.clamp_engaged, "{id}: clamp flag");
        assert_eq!(
            budget,
            fixture.resolved_pos_count + fixture.resolved_neg_count,
            "{id}: the fixture's own composition must sum to its resolved budget"
        );
    }
    assert_eq!(
        explicit_rows, 1,
        "exactly one contracted row (8_4_8_maxpairs100) carries an explicit budget; \
         without it this test would never exercise the explicit branch"
    );
}
