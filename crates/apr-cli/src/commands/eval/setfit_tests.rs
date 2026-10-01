//! Tests for the `apr eval` SetFit branch.
//!
//! # The measured boundary of what these can reach
//!
//! These UNIT tests cannot construct a `setfit-apr-v1` artifact that passes the load ladder, and
//! therefore cannot construct a `SelectionLock` either: `SelectionCandidate::from_evaluation`
//! needs a `ValidationEvaluation`, whose only producers take a verified model. The blockers are
//! structural, not gaps in effort — the conformance slice's 97-row vocabulary closure cannot
//! compute two of the six contract-resident probes, and core's APR-capable view fixture is
//! `#[cfg(test)] pub(crate)`. Producing a real artifact needs the 86.7 MB production checkout,
//! which a default-suite module may not require.
//!
//! So the lock->token->grant chain is proven where an artifact exists: in `aprender-train`, by
//! `apr_evaluate_the_lock_travels_between_two_invocations_as_a_file` (TRN-07, IN-PROCESS,
//! file-mediated). The SPAWNED cross-process form is 04-15's ladder, and its POSITIVE half —
//! `--split validation --lock-out` in one process, `--split test --selection-lock` in the next,
//! same lock hash — is walked by 05-07's
//! `setfit_cli_production_chain_completes_after_the_calibration_edit`. **Before Phase 5's 05-03
//! calibration edit (commit `a63bb130b`) no user-reachable path produced a `setfit-apr-v1`
//! (finding F-10), so that positive half was unreachable; it is not any more.**
//!
//! What IS reachable here — and is what this adapter is — is every refusal, every ordering
//! claim, and the structural guarantee that no path reaches the test split around `grant`.

use super::*;
use tempfile::TempDir;

/// This branch's own source, for the source assertions below.
const EVAL_SETFIT_SOURCE: &str = include_str!("setfit.rs");

/// Assemble a needle at RUNTIME so it cannot match the scan's own source.
fn needle(fragments: &[&str]) -> String {
    fragments.concat()
}

/// A minimal argument set; each test overrides the field it is about.
fn args<'a>(artifact: &'a Path, data: &'a Path, selection: &'a Path) -> SetFitEvalArgs<'a> {
    SetFitEvalArgs {
        artifact,
        data: Some(data),
        selection: Some(selection),
        split: Split::Validation,
        lock_out: None,
        selection_lock: None,
        candidates: &[],
        force: false,
        json: true,
    }
}

// ===========================================================================
// The split, and the flags that belong to each half
// ===========================================================================

#[test]
fn eval_setfit_split_is_a_closed_set_and_names_why_train_is_absent() {
    assert_eq!(
        "validation".parse::<Split>().expect("valid"),
        Split::Validation
    );
    assert_eq!("test".parse::<Split>().expect("valid"), Split::Test);
    let error = "train"
        .parse::<Split>()
        .expect_err("there is no third split");
    let rendered = error.to_string();
    assert!(
        rendered.contains("validation") && rendered.contains("test"),
        "the refusal must name the two legal values; got: {rendered}"
    );
    assert!(
        rendered.contains("memorisation"),
        "and it must say WHY the train split is not one of them; got: {rendered}"
    );
}

#[test]
fn eval_setfit_test_split_requires_a_lock_and_names_the_exact_prior_command() {
    // THE B4 REFUSAL. A test run cannot create the lock it needs — that is the whole point —
    // so the message must hand the operator the command that writes one.
    let temp = TempDir::new().expect("tempdir");
    let artifact = temp.path().join("model.apr");
    let data = temp.path().join("data");
    let selection = temp.path().join("selection-manifest.json");
    let mut a = args(&artifact, &data, &selection);
    a.split = Split::Test;

    let error = run(&a).expect_err("canonical test access requires a committed lock");
    assert!(
        matches!(error, CliError::ValidationFailed(_)),
        "an absent lock is a request error (exit 5); got: {error}"
    );
    let rendered = error.to_string();
    assert!(
        rendered.contains("--selection-lock"),
        "the refusal must name the missing flag; got: {rendered}"
    );
    assert!(
        rendered.contains("--split validation") && rendered.contains("--lock-out"),
        "the remedy must name the EXACT prior command that writes a lock; got: {rendered}"
    );
    // ORDERING: none of these paths exist. A run that opened `--data` first would have
    // reported a missing directory, sending the operator to look at a file that is fine.
    assert!(
        !rendered.contains("not found"),
        "the flag refusal must come BEFORE any file is opened; got: {rendered}"
    );
}

#[test]
fn eval_setfit_refuses_lock_out_on_a_test_run() {
    let temp = TempDir::new().expect("tempdir");
    let artifact = temp.path().join("model.apr");
    let data = temp.path().join("data");
    let selection = temp.path().join("sel.json");
    let lock_out = temp.path().join("lock.json");
    let mut a = args(&artifact, &data, &selection);
    a.split = Split::Test;
    a.lock_out = Some(&lock_out);

    let rendered = run(&a)
        .expect_err("a test run must not write its own lock")
        .to_string();
    assert!(
        rendered.contains("--lock-out") && rendered.contains("BEFORE"),
        "the refusal must say that a self-written lock defeats the ordering the lock exists \
         for; got: {rendered}"
    );
}

#[test]
fn eval_setfit_refuses_candidates_on_a_test_run() {
    let temp = TempDir::new().expect("tempdir");
    let artifact = temp.path().join("model.apr");
    let data = temp.path().join("data");
    let selection = temp.path().join("sel.json");
    let candidates = vec![temp.path().join("other.apr")];
    let mut a = args(&artifact, &data, &selection);
    a.split = Split::Test;
    a.selection_lock = Some(&artifact);
    a.candidates = &candidates;

    let rendered = run(&a)
        .expect_err("the candidate set was fixed when the lock was written")
        .to_string();
    assert!(rendered.contains("--candidate"), "got: {rendered}");
}

#[test]
fn eval_setfit_refuses_a_selection_lock_on_a_validation_run() {
    let temp = TempDir::new().expect("tempdir");
    let artifact = temp.path().join("model.apr");
    let data = temp.path().join("data");
    let selection = temp.path().join("sel.json");
    let lock = temp.path().join("lock.json");
    let mut a = args(&artifact, &data, &selection);
    a.selection_lock = Some(&lock);

    let rendered = run(&a)
        .expect_err("a validation run commits a selection, it does not consume one")
        .to_string();
    assert!(rendered.contains("--selection-lock") && rendered.contains("--lock-out"));
}

#[test]
fn eval_setfit_requires_data_and_selection_before_anything_is_read() {
    let temp = TempDir::new().expect("tempdir");
    let artifact = temp.path().join("model.apr");
    let selection = temp.path().join("sel.json");

    let mut a = args(&artifact, &artifact, &selection);
    a.data = None;
    let rendered = run(&a).expect_err("--data is required").to_string();
    assert!(rendered.contains("--data"), "got: {rendered}");

    let mut a = args(&artifact, &artifact, &selection);
    a.selection = None;
    let rendered = run(&a).expect_err("--selection is required").to_string();
    assert!(
        rendered.contains("--selection") && rendered.contains("reload"),
        "the refusal must say what the selection is compared against; got: {rendered}"
    );
}

// ===========================================================================
// The lock file: bounded, typed, and never invented
// ===========================================================================

#[test]
fn eval_setfit_reports_an_absent_lock_by_naming_the_command_that_writes_one() {
    let temp = TempDir::new().expect("tempdir");
    let error =
        read_lock(&temp.path().join("absent.json")).expect_err("there is no lock at that path");
    let rendered = error.to_string();
    assert!(
        matches!(error, CliError::ValidationFailed(_)),
        "got: {rendered}"
    );
    assert!(
        rendered.contains("--lock-out"),
        "an absent lock must point at the command that writes one; got: {rendered}"
    );
}

#[test]
fn eval_setfit_refuses_an_over_cap_lock_file_before_reading_it() {
    // T-04-50 at this surface. Sparse, so the test costs no disk: an attacker does not have to
    // spend a megabyte to make a reader spend one.
    let temp = TempDir::new().expect("tempdir");
    let path = temp.path().join("huge.json");
    let file = std::fs::File::create(&path).expect("fixture is creatable");
    file.set_len(MAX_SELECTION_LOCK_BYTES + 1)
        .expect("a sparse over-cap file is creatable");
    drop(file);

    let error = read_lock(&path).expect_err("a lock past the contracted bound must be refused");
    assert!(
        matches!(error, CliError::InvalidFormat(_)),
        "an over-cap lock is a format refusal (exit 4); got: {error}"
    );
    let rendered = error.to_string();
    assert!(
        rendered.contains(&MAX_SELECTION_LOCK_BYTES.to_string())
            && rendered.contains(&(MAX_SELECTION_LOCK_BYTES + 1).to_string()),
        "the refusal must name BOTH the bound and the observed length; got: {rendered}"
    );
}

#[test]
fn eval_setfit_refuses_a_lock_file_that_is_not_a_lock() {
    let temp = TempDir::new().expect("tempdir");
    let path = temp.path().join("not-a-lock.json");
    std::fs::write(&path, br#"{"schema_version":99}"#).expect("fixture is writable");
    let error = read_lock(&path).expect_err("an arbitrary JSON object is not a selection lock");
    assert!(
        matches!(error, CliError::ValidationFailed(_)),
        "the library's typed refusal must surface, not a panic; got: {error}"
    );
    assert!(
        error.to_string().contains("not a usable selection lock"),
        "the message must name the path and carry the library's diagnosis"
    );
}

#[test]
fn eval_setfit_refuses_a_directory_as_a_lock() {
    let temp = TempDir::new().expect("tempdir");
    let error = read_lock(temp.path()).expect_err("a directory is not a lock");
    assert!(matches!(error, CliError::NotAFile(_)), "got: {error}");
}

// ===========================================================================
// WR-10: the no-clobber gate runs BEFORE the work, not after it
//
// # What these four tests are, as a group
//
// `setfit_train.rs:12-20` states the ordering discipline this command was supposed to share:
// the request-level checks run "BEFORE a single row of `--data` is read", and "refusing the
// output path last of the four is what stops a twenty-minute run from ending in 'I will not
// overwrite that file'". `refuse_existing_output` was factored out as a standalone function
// precisely so it could be called early. `apr eval --lock-out` never called it early: it ran
// the whole multi-candidate sweep — a bounded read of the corpus, the eight-rung load ladder
// and a classify pass over the whole validation split for EVERY candidate — and only then
// discovered that the file it was asked to write already existed.
//
// # Why there are TWO ordering tests and not one
//
// CLAUDE.md rule 6: one failing input is an anecdote. A single test showing the lock refusal
// winning proves only that it beat ONE particular later error. Tests 1 and 2 hold everything
// constant except HOW the later input is broken, and produce two DIFFERENT pre-fix
// diagnoses — a missing corpus directory and a corpus whose manifest is not JSON. The lock
// refusal has to beat both.
//
// # The measured limit of the in-process pair, stated rather than glossed
//
// The plan asked the second witness to fail at step (3) — the artifact reload — rather than
// at a second point inside step (2). That is NOT reachable from this file: getting past step
// (2) needs a VALID attested prepared dataset, and the only fixture that writes one is
// `#[cfg(test)]`-private to `data_contrastive`, which is outside this change's file scope.
// Widening it would put a dataset fixture in a second place. So the second witness varies the
// later input WITHIN step (2), where the two diagnoses are still demonstrably different, and
// the process-tier witness in `tests/setfit_cli_lifecycle.rs` covers the same property against
// the shipped binary.
// ===========================================================================

/// A `--lock-out` destination that ALREADY holds something.
///
/// The contents are deliberately not a valid lock: the gate under test is `exists()`, and a
/// gate that had to parse the file first would be doing the expensive work this plan moved.
fn occupied_lock_destination(dir: &Path) -> PathBuf {
    let path = dir.join("lock.json");
    std::fs::write(&path, br#"{"committed":"by a prior validation run"}"#)
        .expect("the pre-existing lock destination is writable");
    path
}

/// A `--data` directory that EXISTS and whose manifest is unreadable.
///
/// Distinct from an absent directory on purpose: it makes step (2) fail with a PARSE diagnosis
/// (`benchmark-manifest.json is not valid JSON`) instead of a NOT-FOUND one, which is what
/// makes the second ordering witness a different measurement rather than a copy of the first.
fn corpus_with_an_unparseable_manifest(dir: &Path) -> PathBuf {
    let path = dir.join("broken-corpus");
    std::fs::create_dir_all(&path).expect("the corpus directory is creatable");
    std::fs::write(
        path.join(crate::commands::data_tweeteval::MANIFEST_FILE),
        b"this is not JSON",
    )
    .expect("the broken manifest is writable");
    path
}

#[test]
fn lock_out_refuses_an_existing_destination_before_the_dataset_is_read() {
    let temp = TempDir::new().expect("tempdir");
    let artifact = temp.path().join("model.apr");
    // ABSENT on purpose. The only way to tell "the corpus was never opened" from "the corpus
    // was opened and was fine" is to make opening it fail loudly, then check whether that
    // failure is what the operator was told about.
    let data = temp.path().join("no-such-corpus");
    let selection = temp.path().join("no-such-selection.json");
    let lock_out = occupied_lock_destination(temp.path());
    assert!(
        !data.exists() && lock_out.exists(),
        "the fixture must be an absent corpus and an OCCUPIED destination, or this test \
         measures nothing"
    );

    let mut a = args(&artifact, &data, &selection);
    a.lock_out = Some(&lock_out);
    assert!(
        !a.force,
        "the pre-flight is gated on --force; a forced run is Test 3's subject, not this one"
    );

    let error = run(&a).expect_err("an occupied --lock-out destination must be refused");
    assert!(
        matches!(error, CliError::ValidationFailed(_)),
        "refusing to clobber a commitment is a request error (exit 5); got: {error}"
    );
    let rendered = error.to_string();
    // POSITIVE: the bespoke wording, not the generic no-clobber message. "Refusing to replace
    // existing file X" does not tell an operator what replacing a selection lock COSTS.
    assert!(
        rendered.contains("A selection lock is a COMMITMENT"),
        "the refusal must carry the domain wording that says what replacing a lock costs; \
         got: {rendered}"
    );
    assert!(
        rendered.contains(&lock_out.display().to_string()),
        "and it must name the destination the operator supplied; got: {rendered}"
    );
    // NEGATIVE, and this is the ORDER assertion. Pre-fix, step (2) ran first and the operator
    // was sent to look at the corpus — a true statement about the wrong problem, arrived at
    // after the whole sweep. A one-sided assertion would pass on a run that failed for a third
    // reason entirely.
    assert!(
        !rendered.contains(&data.display().to_string()),
        "the lock refusal must come BEFORE the dataset is read, so the corpus must not be \
         named at all; got: {rendered}"
    );
}

#[test]
fn lock_out_refuses_an_existing_destination_before_a_broken_corpus_is_diagnosed() {
    // THE SECOND WITNESS. Same occupied destination, a DIFFERENT later-stage failure: the
    // corpus directory exists and its manifest is unparseable, so step (2) fails with a parse
    // diagnosis rather than a not-found one. Two different later failures both losing to the
    // pre-flight is what distinguishes "the check moved earlier" from "the check happened to
    // beat one particular error".
    let temp = TempDir::new().expect("tempdir");
    let artifact = temp.path().join("model.apr");
    let data = corpus_with_an_unparseable_manifest(temp.path());
    let selection = temp.path().join("no-such-selection.json");
    let lock_out = occupied_lock_destination(temp.path());

    let mut a = args(&artifact, &data, &selection);
    a.lock_out = Some(&lock_out);

    let error = run(&a).expect_err("an occupied --lock-out destination must be refused");
    assert!(
        matches!(error, CliError::ValidationFailed(_)),
        "got: {error}"
    );
    let rendered = error.to_string();
    assert!(
        rendered.contains("A selection lock is a COMMITMENT"),
        "the same bespoke refusal must win here too; got: {rendered}"
    );
    // NEGATIVE: the pre-fix path reached `attestation_bytes_from_manifest` and said so.
    assert!(
        !rendered.contains("is not valid JSON"),
        "the lock refusal must come BEFORE the manifest is parsed — a run that diagnosed the \
         manifest has already done the read this plan moved the gate above; got: {rendered}"
    );
    assert!(
        !rendered.contains(crate::commands::data_tweeteval::MANIFEST_FILE),
        "and it must not name the manifest file at all; got: {rendered}"
    );
}

#[test]
fn lock_out_force_proceeds_past_the_preflight() {
    // The pre-flight must be gated on `force` exactly like the two checks it joins. An
    // operator who passed --force asked for the replacement; the run must proceed to the work
    // and fail (or succeed) on the LATER input.
    let temp = TempDir::new().expect("tempdir");
    let artifact = temp.path().join("model.apr");
    let data = temp.path().join("no-such-corpus");
    let selection = temp.path().join("no-such-selection.json");
    let lock_out = occupied_lock_destination(temp.path());

    let mut a = args(&artifact, &data, &selection);
    a.lock_out = Some(&lock_out);
    a.force = true;

    let rendered = run(&a)
        .expect_err("the corpus really is absent, so a forced run must still fail — but LATER")
        .to_string();
    assert!(
        rendered.contains(&data.display().to_string()),
        "--force must carry the run PAST the pre-flight and into the ingest, which is what \
         names the corpus. If this run is silent about the corpus, --force did not override \
         the gate and Test 1 proved nothing about --force; got: {rendered}"
    );
    assert!(
        !rendered.contains("A selection lock is a COMMITMENT"),
        "and the no-clobber refusal must NOT have fired: --force is the operator saying they \
         intend to replace the committed decision; got: {rendered}"
    );
}

#[test]
fn write_lock_still_refuses_a_destination_that_appeared_mid_run() {
    // THE THIRD CHECK, and why this test drives the writer rather than `write_lock`.
    //
    // There are three ordered no-clobber checks once the pre-flight lands: the pre-flight
    // (before any input is read), `write_lock`'s bespoke check (before bytes are produced),
    // and `atomic_write`'s `refuse_existing_output` (immediately before the rename). Only the
    // third covers a file that appeared WHILE the run was going, and the risk this test exists
    // for is someone deleting it as "now redundant" after the pre-flight moved earlier.
    //
    // `write_lock` itself cannot be called from here: it needs a `SelectionLock`, and this
    // file's header records the measured reason none can be built in this crate's unit suite —
    // `SelectionCandidate::from_evaluation` needs a `ValidationEvaluation`, whose only
    // producers take a verified model, and no `setfit-apr-v1` artifact is available here
    // without the 86.7 MB production checkout (before commit a63bb130b none was producible at
    // all — F-10). `SelectionLock::from_canonical_bytes` exists, but hand-forging canonical
    // lock bytes here would be a SECOND implementation of the lock's wire form living in a CLI
    // test — the exact drift this phase removes elsewhere. So the property is pinned where it
    // actually lives: at the writer `write_lock` delegates to, plus a source anchor tying the
    // two together so the delegation cannot be quietly replaced.
    let temp = TempDir::new().expect("tempdir");

    // The source anchor: this is the write path the behaviour below is about.
    assert!(
        production_code_lines().contains("setfit_train::atomic_write("),
        "the lock write must still delegate to the crate's ONE atomic writer, or the check \
         exercised below is not the check on the lock's write path"
    );

    // A destination that did NOT exist at pre-flight time and DOES exist by write time.
    let destination = temp.path().join("appeared-mid-run.json");
    assert!(
        !destination.exists(),
        "the pre-flight would have passed on this path — that is the premise"
    );
    std::fs::write(&destination, b"a concurrent run committed here")
        .expect("the mid-run file is writable");

    let error = crate::commands::setfit_train::atomic_write(&destination, b"{}", false)
        .expect_err("a file that appeared mid-run must still not be clobbered");
    assert!(
        matches!(error, CliError::ValidationFailed(_)),
        "the write-time refusal is typed, not an IO error; got: {error}"
    );
    assert!(
        error
            .to_string()
            .contains(&destination.display().to_string()),
        "and it names the path it refused to replace; got: {error}"
    );
    // Non-vacuity: the bytes on disk are untouched, so the refusal happened BEFORE the rename
    // rather than after a clobber that then reported an error.
    assert_eq!(
        std::fs::read(&destination).expect("the destination is readable"),
        b"a concurrent run committed here",
        "a refused write must leave the existing file exactly as it was"
    );
    // And the control: --force still replaces it, so the assertion above is about the gate
    // rather than about the writer being broken.
    crate::commands::setfit_train::atomic_write(&destination, b"{}", true)
        .expect("--force replaces the destination");
    assert_eq!(
        std::fs::read(&destination).expect("the destination is readable"),
        b"{}",
        "the forced write must actually have replaced the file"
    );
}

// ===========================================================================
// Structural: this adapter gates nothing itself, and bypasses nothing
// ===========================================================================

/// The CODE LINES of this branch, so the scans below do not fail on their own explanations.
fn production_code_lines() -> String {
    let code: String = EVAL_SETFIT_SOURCE
        .lines()
        .filter(|line| {
            let t = line.trim_start();
            !(t.starts_with("//") || t.starts_with("/// ") || t.starts_with("///"))
        })
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        code.contains("fn run(args: &SetFitEvalArgs"),
        "non-vacuity: the comment filter must not have eaten the module"
    );
    code
}

#[test]
fn eval_setfit_drives_the_three_library_doors_and_defines_no_gating_type() {
    let code = production_code_lines();
    for door in [
        "reload_verified_run_from_apr(",
        "create_selection_lock(",
        "SelectionLock::from_canonical_bytes(",
        "mint_test_token(",
        "CanonicalTestAccess::grant(",
    ] {
        assert!(
            code.contains(door),
            "the durable-lock workflow must drive `{door}` — this adapter re-implements none \
             of them"
        );
    }
    // A gating type defined HERE would be a second answer to "may this process read the test
    // split", and the second answer is the one nobody audits.
    for forged in [
        "struct CanonicalTestToken",
        "struct CanonicalTestGrant",
        "struct SelectionLock",
        "struct SelectionCandidate",
        "impl SetFitCredential",
    ] {
        assert!(
            !code.contains(forged),
            "`{forged}` here would be a CLI-side gate; every gate lives in aprender-train"
        );
    }
}

#[test]
fn eval_setfit_has_no_path_to_the_test_split_that_bypasses_the_grant() {
    // The T-04-21 assertion. `grant.test()` is the ONLY way this file names the test rows;
    // a direct `dataset.test()` would read them with the lock chain merely alongside.
    let code = production_code_lines();
    let direct = needle(&["dataset.", "test()"]);
    assert_eq!(
        code.matches(&direct).count(),
        0,
        "the test split must come OUT of the grant, never off the dataset directly"
    );
    assert_eq!(
        code.matches("grant.test()").count(),
        1,
        "exactly one route to the canonical test rows"
    );
    // And the reverse control: the grant is obtained from a token, which is obtained from a
    // lock READ OFF DISK. A run that minted from a lock it had just built in memory would
    // satisfy every needle above and prove nothing about ordering.
    let mint = code
        .find("mint_test_token(")
        .expect("the mint call must exist");
    let read = code
        .find("let lock = read_lock(")
        .expect("the lock read must exist");
    assert!(
        read < mint,
        "the lock must be READ from disk before a token is minted from it; in `--split test` \
         there is no other source for one"
    );
}

#[test]
fn eval_setfit_reads_artifacts_only_through_the_one_bounded_door() {
    let code = production_code_lines();
    let banned = needle(&["fs::", "read("]);
    assert_eq!(
        code.matches(&banned).count(),
        0,
        "no unbounded whole-file read may appear in this branch"
    );
    assert!(
        code.contains("setfit_io::read_setfit_apr_file_bounded"),
        "artifact bytes come through the ONE bounded door (review B5)"
    );
    assert!(
        code.contains("MAX_SELECTION_LOCK_BYTES"),
        "the lock file carries its own contracted bound, applied before the read"
    );
}

#[test]
fn eval_setfit_writes_the_lock_atomically_through_exactly_one_rename() {
    let code = production_code_lines();

    // The atomicity property is DELEGATED now, not hand-rolled here. This module used to carry
    // its own rename/sync/no-clobber sequence — a third copy of the crate's atomic writer, and
    // the only one with a colliding temp name and `create(true)` instead of `create_new(true)`.
    // So the guard asserts the delegation and the ABSENCE of a re-hand-roll, which is what can
    // actually regress; `setfit_train`'s own tests prove the write is atomic.
    assert!(
        code.contains("setfit_train::atomic_write("),
        "the lock write must go through the crate's ONE atomic writer, so the sync-then-rename \
         guarantee is proven in one place instead of three"
    );
    assert_eq!(
        code.matches("fs::rename(").count(),
        0,
        "no rename site may reappear here: a fourth hand-rolled writer is exactly the drift \
         this delegation removed"
    );
    assert!(
        !code.contains("sync_all()"),
        "no sync site may reappear here either — if one does, the write has been re-inlined"
    );

    // The bespoke no-clobber refusal STAYS local: "a lock is a COMMITMENT" says more than the
    // generic message, and it must fire before any bytes are produced.
    assert!(
        code.contains("already exists"),
        "an existing lock must not be clobbered without --force"
    );
    assert!(
        code.contains("is a COMMITMENT"),
        "the domain-specific refusal must survive delegation — the generic no-clobber message \
         does not tell an operator what replacing a lock costs"
    );
}

#[test]
fn eval_setfit_uses_the_librarys_evaluator_and_computes_no_validation_metric_itself() {
    let code = production_code_lines();
    assert!(
        code.contains("evaluate_validation_from_artifact("),
        "the validation metric must be the library's, so the number the lock records is the \
         one trusted code computed"
    );
    for banned in ["fn accuracy(", "fn macro_f1(", "ValidationEvaluation {"] {
        assert!(
            !code.contains(banned),
            "`{banned}` here would be a CLI-side validation metric — the caller-asserted number \
             the library removed"
        );
    }
}

/// CR-01 (04-REVIEW): the label-map gate must stand between the three doors and the scoring.
///
/// `evaluate_test` scores `position(result.label())` — an index into the ARTIFACT's head —
/// against `row.label`, an index into the DATASET. None of the three doors compares those two
/// orderings: `mint_test_token` compares the artifact hash, `grant` compares artifact + dataset
/// fingerprint, and the reload gate compares corpus/ledger/draw. Without the fourth gate a
/// mismatched corpus yields a confidently wrong accuracy instead of an error — which is exactly
/// what the validation evaluator refuses by name (`apr_evaluate.rs`, `LabelMapMismatch`).
///
/// The behaviour is not reachable from this adapter's unit tests: reaching it needs a real
/// `ReloadedSetFitCredential`, and building one needs the production checkout this module may
/// not require (before commit a63bb130b no artifact was producible at all — F-10). So
/// this is a SOURCE guard, in the same shape this phase uses elsewhere for blocked rungs. It
/// asserts ORDER, not mere presence — a gate placed after the scoring would be no gate at all.
#[test]
fn eval_setfit_test_split_gates_the_label_map_before_it_scores() {
    let src = include_str!("setfit.rs");

    let gate = src
        .find("artifact_labels != dataset_labels")
        .expect("the label-map gate must exist in the test path (CR-01)");
    let scoring = src
        .find("let evaluation = evaluate_test(")
        .expect("the test path must still call evaluate_test — this guard is anchored to it");

    assert!(
        gate < scoring,
        "the label-map gate must run BEFORE evaluate_test, not after: a gate downstream of the \
         scoring cannot stop a confidently wrong number from being computed"
    );

    // Non-vacuity: both sides must be read from real sources, or the comparison above could be
    // between two empty vecs and would pass while gating nothing.
    assert!(
        src.contains("credential.model().ordered_labels()"),
        "the artifact side must be read off the REBUILT HEAD, which is what a classification \
         actually indexes into — not the document's copy, which can drift"
    );
    assert!(
        src.contains("dataset.label_names()"),
        "the dataset side must be read off the prepared dataset"
    );
}
