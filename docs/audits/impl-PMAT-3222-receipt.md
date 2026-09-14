# PMAT-3222 — ONT-2a: L4 and L5 print `self-declared` until the tree grounds them

## What changed, in one sentence

A contract's `verification_summary` is the contract talking about ITSELF. Before
this change it was enough to reach L4: `l4_lean_proved + l4_not_applicable >=
total` granted the level on the claim alone. Now L4 is granted on GROUNDING — a
sorry-free Lean theorem that an equation names and `lean_theorem_names()`
resolves — and a claim with nothing under it is reported `self-declared`,
excluded from the L4 total, and never counted quietly.

## This is a STRENGTHENING, and here is the measurement that settles it

A review lane read it the other way: that granting `l4_not_applicable` credit
without also requiring `vs.l4_lean_proved > 0` *lowers* the bar, and predicted
that `level_strict_summary_cannot_understate_total` would fail. Measured:

```
$ cargo test -p aprender-contracts --lib level_strict_summary_cannot_understate_total
test proof_status::tests::level_strict_summary_cannot_understate_total ... ok
```

and the whole crate:

```
$ cargo test -p aprender-contracts --lib
test result: ok. 1505 passed; 0 failed; 5 ignored
```

The lane also read that test's fixture as `l4_lean_proved: 2`; it is `3`
(`proof_status_tests.rs:211`). The fixture understates `total_obligations` as 3
against six real `proof_obligations`, passes `grounded = 3`, and asserts **L3** —
which is what the code returns, because `total` comes from
`proof_obligations.len()` and never from the summary.

Why the reading was wrong: the binding constraint MOVED. It used to be "the
summary says so"; it is now `grounded > 0 && grounded + not_applicable >= total`,
where `grounded` is counted out of the Lean tree. A summary can no longer buy L4
at any value. `not_applicable` still comes from the summary because it is a claim
about APPLICABILITY, not about a proof — and `is_l4_self_declared` exists so that
a contract which WOULD have been L4 under the old rule and is not L4 under this
one is named in the report rather than silently demoted.

## Why the Lean-name scan was split

Not a refactor for taste. `lean_theorem_names()` was one function at cognitive
complexity **132**; this repository's gate refuses that. It is now
`insert_name_forms`, `camel_case`, `first_camel_word`,
`insert_theorem_names_from_content`, `insert_domain_theorems` and
`scan_theorem_base`, and the file's worst function is **13**:

```
$ pmat analyze complexity --path crates/aprender-contracts/src/proof_status.rs
Max Cognitive: 13   Median Cognitive: 1.5   Functions: 22
1. compute_proof_level_with_grounding  Cyclomatic: 7, Cognitive: 13
```

The split is a precondition of landing the change at all, not a second change
riding along with it. It is recorded here because a lane asked for the reason and
the diff did not carry one.

## Why the grounding count is a PARAMETER

`count_lean_theorems_for_contract` reads the Lean tree from paths relative to the
process CWD. Inside a unit test it resolves nothing, so every fixture would be
ungrounded by construction — a suite in which the andon could withdraw ALL credit
for ever and no test would notice. `*_with_grounding` takes the count as an
argument, so both sides are covered: `grounded == 0` is the andon and
`grounded + not_applicable >= total` is the credit it still grants. Production
callers use the wrapper and get the scan.

## Also in this diff

`ci / lint` refused an earlier head on two clippy errors, both artefacts of the
split: an orphaned doc block left standing above a blank line with no item under
it (and opening with an intra-doc link to `is_lean_proved`, a name this file no
longer defines), and `!path.extension().is_some_and(|e| e == "lean")`, which is
`is_none_or(|e| e != "lean")`. Both fixed; `cargo clippy -p aprender-contracts
--lib --all-features -- -D warnings` is clean.

IMPL-PMAT-3222-RECEIPT-END
