//! Phase 5 D-10: the `--selection-manifest` door on `apr finetune --task classify`.
//!
//! Two things are proven here and nothing else:
//!
//! 1. The REFUSALS fire before any data is read — a missing `--seed`, a missing `--data`,
//!    a manifest whose envelope digest does not verify, and a selection whose ids have no
//!    row in the dataset.
//! 2. The resolved `TrainingConfig` is byte-identical to the pre-D-10 hardcode block when
//!    none of the new flags is passed. That is the "defaults preserved" claim, asserted
//!    against the literal values the old block contained rather than against the constants
//!    the code now reads — a test that quoted the constants would pass no matter what they
//!    were changed to.

use super::{
    resolve_classify_selection, resolve_selected_samples, resolve_training_config,
    ClassifyOverrides, DEFAULT_CLASSIFY_SAVE_EVERY,
};
use std::path::{Path, PathBuf};

fn overrides<'a>(selection_manifest: Option<&'a Path>, seed: Option<u64>) -> ClassifyOverrides<'a> {
    ClassifyOverrides {
        selection_manifest,
        seed,
        val_split: None,
        early_stopping_patience: None,
    }
}

fn message(error: &crate::error::CliError) -> String {
    error.to_string()
}

// ------------------------------------------------------------------------------------
// Flag validation: refused before a byte of data is read
// ------------------------------------------------------------------------------------

#[test]
fn selection_manifest_without_seed_refuses_naming_both_flags() {
    // A path that does NOT exist: if the seed check did not come first, this would fail
    // with a file error instead, and the test would be asserting the wrong ordering.
    let manifest = PathBuf::from("/nonexistent/selection-manifest.json");
    let error = resolve_classify_selection(&overrides(Some(&manifest), None), None)
        .expect_err("a selection-manifest run without --seed is refused");

    let text = message(&error);
    assert!(
        text.contains("--selection-manifest"),
        "the refusal must name the flag that triggered it: {text}"
    );
    assert!(
        text.contains("--seed"),
        "the refusal must name the flag that fixes it: {text}"
    );
}

#[test]
fn selection_manifest_without_data_refuses_naming_the_attested_directory() {
    let manifest = PathBuf::from("/nonexistent/selection-manifest.json");
    let error = resolve_classify_selection(&overrides(Some(&manifest), Some(13)), None)
        .expect_err("a selection-manifest run without --data is refused");

    let text = message(&error);
    assert!(
        text.contains("--data"),
        "the refusal must name --data: {text}"
    );
    assert!(
        text.contains("ATTESTED CANONICAL"),
        "the refusal must say WHICH kind of --data is meant, because the corpus .jsonl \
         form is the one every other classify run uses: {text}"
    );
}

#[test]
fn absent_selection_manifest_is_not_an_error_and_resolves_to_nothing() {
    let resolved = resolve_classify_selection(&overrides(None, None), None)
        .expect("no --selection-manifest is the historical path, not a refusal");
    assert!(
        resolved.is_none(),
        "without the flag there is no selection to resolve"
    );
}

// ------------------------------------------------------------------------------------
// Defaults preserved: the resolved config equals the old hardcode block
// ------------------------------------------------------------------------------------

#[test]
fn absent_flags_resolve_to_the_pre_d10_hardcode_block() {
    let config = resolve_training_config(
        &overrides(None, None),
        7,
        PathBuf::from("checkpoints"),
        None,
    );

    // The literals from the block this change replaced:
    //   { epochs, val_split: 0.2, save_every: 5, early_stopping_patience: 10, seed: 42 }
    assert_eq!(config.epochs, 7);
    assert!(
        (config.val_split - 0.2).abs() < f32::EPSILON,
        "val_split default moved: {}",
        config.val_split
    );
    assert_eq!(config.save_every, 5);
    assert_eq!(config.early_stopping_patience, 10);
    assert_eq!(config.seed, 42);
    assert_eq!(config.log_interval, 1);
    assert_eq!(config.checkpoint_dir, PathBuf::from("checkpoints"));
}

#[test]
fn each_flag_overrides_exactly_its_own_field() {
    let config = resolve_training_config(
        &ClassifyOverrides {
            selection_manifest: None,
            seed: Some(13),
            val_split: Some(0.0),
            early_stopping_patience: Some(0),
        },
        4,
        PathBuf::from("out"),
        None,
    );

    assert_eq!(config.seed, 13);
    assert!(config.val_split.abs() < f32::EPSILON);
    assert_eq!(config.early_stopping_patience, 0);
}

#[test]
fn disabling_early_stopping_pushes_save_every_out_to_the_final_epoch() {
    // The frozen-defaults regime: nothing may select an epoch, so the periodic checkpoint
    // cadence stops producing intermediate candidates.
    let disabled = resolve_training_config(
        &ClassifyOverrides {
            selection_manifest: None,
            seed: None,
            val_split: None,
            early_stopping_patience: Some(0),
        },
        12,
        PathBuf::from("out"),
        None,
    );
    assert_eq!(disabled.save_every, 12);

    // ...and leaving early stopping ON keeps the historical cadence.
    let enabled = resolve_training_config(&overrides(None, None), 12, PathBuf::from("out"), None);
    assert_eq!(enabled.save_every, DEFAULT_CLASSIFY_SAVE_EVERY);
}

// ------------------------------------------------------------------------------------
// Row resolution: selection order is preserved, a missing id is a counted refusal
// ------------------------------------------------------------------------------------

fn row(id: &str, input: &str, label: usize) -> aprender_contrastive_data::schema::LabeledExample {
    aprender_contrastive_data::schema::LabeledExample {
        id: id.to_string(),
        input: input.to_string(),
        label,
        label_text: format!("class-{label}"),
        source_split: "train".to_string(),
    }
}

#[test]
fn resolved_samples_follow_the_selection_order_not_the_file_order() {
    let rows = vec![
        row("train:0", "alpha", 0),
        row("train:1", "beta", 1),
        row("train:2", "gamma", 2),
    ];
    // Deliberately NOT the file order: the selection's order is class-ascending, draw
    // order within a class, and a resolver that iterated `rows` would silently produce a
    // different training sequence for the same manifest.
    let ordered = [("train:2", 2), ("train:0", 0), ("train:1", 1)];

    let samples = resolve_selected_samples(&ordered, &rows).expect("every id has a row");
    let inputs: Vec<&str> = samples.iter().map(|s| s.input.as_str()).collect();
    assert_eq!(inputs, vec!["gamma", "alpha", "beta"]);
    let labels: Vec<usize> = samples.iter().map(|s| s.label).collect();
    assert_eq!(labels, vec![2, 0, 1]);
}

#[test]
fn a_selection_id_with_no_row_refuses_with_the_missing_count() {
    let rows = vec![row("train:0", "alpha", 0)];
    let ordered = [("train:0", 0), ("train:7", 1), ("train:9", 2)];

    let error = resolve_selected_samples(&ordered, &rows)
        .expect_err("ids absent from --data are a refusal, never a silently shorter run");

    let text = message(&error);
    assert!(
        text.contains('2'),
        "the refusal must state HOW MANY ids were missing — a shortened training set is \
         invisible without it: {text}"
    );
    assert!(
        text.contains("train:7"),
        "and name at least one of them: {text}"
    );
}

// ------------------------------------------------------------------------------------
// End-to-end refusals against a REAL attested directory and a REAL selection manifest
//
// Nothing here fabricates an attestation: the benchmark directory is produced by running
// `apr data tweet-eval-stance` over the synthetic source tree, and the manifest by running
// `apr data select` over that directory — the same discipline `data_contrastive`'s own
// test module states. A hand-built manifest would only prove this module agrees with
// itself.
// ------------------------------------------------------------------------------------

mod attested {
    use super::super::{resolve_classify_selection, ClassifyOverrides};
    use crate::commands::data_contrastive::{run_select, SELECTION_MANIFEST_FILE};
    use crate::commands::data_tweeteval::{self, fixtures, CANONICAL_REVISION};
    use crate::TweetEvalStanceProfile;
    use std::fs;
    use std::path::{Path, PathBuf};
    use tempfile::TempDir;

    /// Prepare an attested canonical directory and write a selection manifest into it.
    fn attested_dir_with_selection(root: &Path, name: &str, tag: &str, seed: u64) -> PathBuf {
        let source = root.join(format!("{name}-source"));
        fs::create_dir_all(&source).expect("fixture source directory is creatable");
        fixtures::write_canonical_fixture_tagged(&source, tag);
        let out = root.join(name);
        data_tweeteval::run(
            &out,
            TweetEvalStanceProfile::Canonical,
            Some(&source),
            CANONICAL_REVISION,
            false,
            true,
            true,
        )
        .expect("the synthetic canonical fixture prepares cleanly");
        run_select(&out, 8, seed, false, None, false, true).expect("select succeeds");
        out
    }

    fn overrides<'a>(manifest: &'a Path, seed: u64) -> ClassifyOverrides<'a> {
        ClassifyOverrides {
            selection_manifest: Some(manifest),
            seed: Some(seed),
            val_split: None,
            early_stopping_patience: None,
        }
    }

    /// The happy path, so the two refusals below are known to be refusing something that
    /// would otherwise have worked (a refusal test whose positive control is missing
    /// proves only that the function returns errors).
    #[test]
    fn a_valid_manifest_resolves_the_selected_rows_in_selection_order() {
        let temp = TempDir::new().expect("tempdir");
        let data = attested_dir_with_selection(temp.path(), "canonical", fixtures::DEFAULT_TAG, 13);
        let manifest = data.join(SELECTION_MANIFEST_FILE);

        let resolved = resolve_classify_selection(&overrides(&manifest, 13), Some(&data))
            .expect("a manifest written from THIS directory replays against it")
            .expect("--selection-manifest was passed, so a selection is resolved");

        // 8 shots for each of the three declared stance classes.
        assert_eq!(resolved.samples.len(), 24);
        assert_eq!(
            resolved.semantic_hash.len(),
            64,
            "the row records a hex SHA-256: {}",
            resolved.semantic_hash
        );
        // Selection order is class-ascending, so the first row is class 0 and the last is
        // class 2 — not the dataset's file order.
        assert_eq!(resolved.samples[0].label, 0);
        assert_eq!(resolved.samples[23].label, 2);
    }

    /// T-05-06-01: a doctored manifest byte is refused by the envelope digest, BEFORE any
    /// training work. The mutation targets the hashed payload, not the volatile block.
    #[test]
    fn a_doctored_manifest_byte_refuses_on_the_envelope_digest() {
        let temp = TempDir::new().expect("tempdir");
        let data = attested_dir_with_selection(temp.path(), "canonical", fixtures::DEFAULT_TAG, 13);
        let manifest = data.join(SELECTION_MANIFEST_FILE);

        let text = fs::read_to_string(&manifest).expect("manifest is readable");
        let mut value: serde_json::Value = serde_json::from_str(&text).expect("manifest is JSON");
        let first = value["payload"]["ordered_examples"][0]["id"]
            .as_str()
            .expect("the manifest lists ordered examples with ids")
            .to_string();
        // One byte: replace the last character of the first selected id with a digit it
        // is not. The digest over the canonical payload no longer matches the envelope's
        // recorded digest.
        let mut doctored = first.clone();
        let last = doctored.pop().expect("row ids are never empty");
        doctored.push(if last == '0' { '1' } else { '0' });
        assert_ne!(doctored, first);
        value["payload"]["ordered_examples"][0]["id"] = serde_json::json!(doctored);
        fs::write(
            &manifest,
            serde_json::to_vec_pretty(&value).expect("re-encodes"),
        )
        .expect("manifest is writable");

        let error = resolve_classify_selection(&overrides(&manifest, 13), Some(&data))
            .expect_err("a manifest whose payload was edited must be refused");
        // The refusal comes from the manifest's OWN digest check inside
        // `SelectionManifest::from_bytes` — i.e. before the dataset is even consulted and
        // long before a batch is built. Asserting the mechanism, not just "some error":
        // a replay-stage rejection would also have turned this test green while meaning
        // something quite different about when the run stops.
        let text = error.to_string();
        assert!(
            text.contains("semantic_hash mismatch"),
            "the refusal must come from the manifest digest check, and must say so: {text}"
        );
    }

    /// T-05-06-01 (second face): a manifest that is internally valid but was written
    /// against a DIFFERENT preparation cannot be replayed here. Its ids and hashes belong
    /// to another dataset, so the run refuses instead of training on whatever happens to
    /// match.
    #[test]
    fn a_manifest_from_another_preparation_refuses_against_this_data() {
        let temp = TempDir::new().expect("tempdir");
        let mine = attested_dir_with_selection(temp.path(), "mine", fixtures::DEFAULT_TAG, 13);
        let theirs =
            attested_dir_with_selection(temp.path(), "theirs", "a different preparation", 17);

        let foreign_manifest = theirs.join(SELECTION_MANIFEST_FILE);
        assert_ne!(
            fs::read(&foreign_manifest).expect("their manifest"),
            fs::read(mine.join(SELECTION_MANIFEST_FILE)).expect("my manifest"),
            "the two preparations must differ, or this fixture proves nothing"
        );

        let error = resolve_classify_selection(&overrides(&foreign_manifest, 17), Some(&mine))
            .expect_err("a manifest from another dataset must not resolve against this one");

        let text = error.to_string();
        assert!(
            text.contains("--selection-manifest") || text.contains("contrastive data"),
            "the refusal must say which input it is rejecting: {text}"
        );
    }

    /// A missing manifest file is a refusal that names the command which writes one.
    #[test]
    fn a_missing_manifest_file_refuses_with_the_command_that_writes_one() {
        let temp = TempDir::new().expect("tempdir");
        let data = attested_dir_with_selection(temp.path(), "canonical", fixtures::DEFAULT_TAG, 13);
        let absent = data.join("no-such-selection-manifest.json");

        let error = resolve_classify_selection(&overrides(&absent, 13), Some(&data))
            .expect_err("an absent manifest is a refusal");
        let text = error.to_string();
        assert!(
            text.contains("apr data select"),
            "the refusal must name the command that produces the missing file: {text}"
        );
    }
}
