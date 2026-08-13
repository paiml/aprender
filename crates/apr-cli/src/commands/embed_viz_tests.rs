//! Tests for the `apr debug embed-viz` producer (aprender#2377 finding 3).
//!
//! The load-bearing test is `round_trip_*`: the CSV the producer writes is fed
//! to `apr embed-viz-lint` and the lint must ACCEPT it — schema, row count and
//! determinism. The negative half corrupts the CSV and requires the lint to
//! reject it, so the round trip cannot pass vacuously.

use super::*;
use crate::commands::{embed_viz_classifier, embed_viz_lint};

const VOCAB: usize = 24;
const HIDDEN: usize = 8;

/// GGUF fixture extents. Deliberately UNEQUAL and different from the APR
/// fixture's, so an axis mixup cannot hide behind a coincidence.
const G_VOCAB: usize = 24;
const G_HIDDEN: usize = 6;

/// A minimal APR v2 model carrying a real `model.embed_tokens.weight` table.
fn model_fixture() -> tempfile::NamedTempFile {
    use aprender::format::v2::{AprV2Metadata, AprV2Writer};

    let file = tempfile::NamedTempFile::with_suffix(".apr").expect("tempfile");
    let mut metadata = AprV2Metadata::new("embed-viz-fixture");
    metadata.architecture = Some("llama".to_string());
    metadata.hidden_size = Some(HIDDEN);
    metadata.vocab_size = Some(VOCAB);

    let mut writer = AprV2Writer::new(metadata);
    // A table with structure, so PCA has a real principal direction to find.
    let data: Vec<f32> = (0..VOCAB * HIDDEN)
        .map(|i| {
            let row = (i / HIDDEN) as f32;
            let col = (i % HIDDEN) as f32;
            (row * 0.1).mul_add(col + 1.0, (col * 0.37).sin())
        })
        .collect();
    writer.add_f32_tensor("model.embed_tokens.weight", vec![VOCAB, HIDDEN], &data);
    let bytes = writer.write().expect("write APR v2");
    std::fs::write(file.path(), bytes).expect("write file");
    file
}

fn args(model: &Path, out: &Path, projection: Projection) -> EmbedVizArgs {
    EmbedVizArgs {
        model: model.to_path_buf(),
        tensor: None,
        projection,
        seed: 42,
        limit: None,
        tokens: None,
        output: Some(out.to_path_buf()),
        force: false,
    }
}

// ── ROUND TRIP ───────────────────────────────────────────────────────────

#[test]
fn round_trip_producer_csv_is_accepted_by_embed_viz_lint() {
    let model = model_fixture();
    let dir = tempfile::tempdir().expect("tempdir");

    for projection in [Projection::Pca, Projection::Random] {
        let csv = dir
            .path()
            .join(format!("{}.csv", projection_label(projection)));
        run(&args(model.path(), &csv, projection)).expect("producer must project the table");

        embed_viz_lint::run(&csv, Some(VOCAB), None, false).unwrap_or_else(|e| {
            panic!(
                "embed-viz-lint must accept the producer's own {} CSV: {e}",
                projection_label(projection)
            )
        });
    }
}

/// FALSIFY-CRUX-F-18-003: two runs under the same seed must be byte-identical,
/// which is exactly what the lint's `--csv-file-b` gate checks.
#[test]
fn round_trip_two_seeded_runs_pass_the_determinism_gate() {
    let model = model_fixture();
    let dir = tempfile::tempdir().expect("tempdir");
    let a = dir.path().join("a.csv");
    let b = dir.path().join("b.csv");

    run(&args(model.path(), &a, Projection::Random)).expect("run a");
    run(&args(model.path(), &b, Projection::Random)).expect("run b");

    embed_viz_lint::run(&a, Some(VOCAB), Some(&b), false)
        .expect("two runs at seed 42 must be byte-identical");
}

#[test]
fn a_different_seed_moves_the_projection() {
    let model = model_fixture();
    let dir = tempfile::tempdir().expect("tempdir");
    let a = dir.path().join("a.csv");
    let b = dir.path().join("b.csv");

    run(&args(model.path(), &a, Projection::Random)).expect("run a");
    let mut other = args(model.path(), &b, Projection::Random);
    other.seed = 43;
    run(&other).expect("run b");

    let (ta, tb) = (
        std::fs::read_to_string(&a).expect("a"),
        std::fs::read_to_string(&b).expect("b"),
    );
    assert_ne!(
        ta, tb,
        "if seed 42 and 43 give identical output the seed is not wired to the draw"
    );
}

#[test]
fn round_trip_cannot_pass_vacuously_when_the_csv_is_corrupted() {
    let model = model_fixture();
    let dir = tempfile::tempdir().expect("tempdir");
    let csv = dir.path().join("good.csv");
    run(&args(model.path(), &csv, Projection::Random)).expect("producer");
    let good = std::fs::read_to_string(&csv).expect("read");

    // Replace the x coordinate of the first data row with `nan`.
    let poisoned = {
        let mut lines: Vec<String> = good.lines().map(str::to_string).collect();
        let mut fields: Vec<String> = lines[1].split(',').map(str::to_string).collect();
        fields[2] = "nan".to_string();
        lines[1] = fields.join(",");
        lines.join("\n")
    };

    let cases: [(&str, String); 4] = [
        ("a non-finite coordinate", poisoned),
        (
            "a dropped row",
            good.lines()
                .take(good.lines().count() - 1)
                .collect::<Vec<_>>()
                .join("\n"),
        ),
        (
            "a renamed header column",
            good.replacen("token_id", "id", 1),
        ),
        ("a negative token id", good.replacen("\n0,", "\n-1,", 1)),
    ];

    for (label, body) in cases {
        let bad = dir.path().join("bad.csv");
        std::fs::write(&bad, &body).expect("write");
        let err = embed_viz_lint::run(&bad, Some(VOCAB), None, false)
            .expect_err(&format!("lint must reject: {label}"));
        assert!(
            matches!(err, CliError::ValidationFailed(_)),
            "{label}: expected a validation refusal, got {err:?}"
        );
    }
}

/// The determinism gate must be able to fail, or the byte-identity test above
/// would prove nothing.
#[test]
fn the_determinism_gate_rejects_two_different_csvs() {
    let model = model_fixture();
    let dir = tempfile::tempdir().expect("tempdir");
    let a = dir.path().join("a.csv");
    let b = dir.path().join("b.csv");
    run(&args(model.path(), &a, Projection::Random)).expect("run a");
    let mut other = args(model.path(), &b, Projection::Random);
    other.seed = 43;
    run(&other).expect("run b");

    let err = embed_viz_lint::run(&a, Some(VOCAB), Some(&b), false)
        .expect_err("two different projections must not pass a determinism gate");
    assert!(matches!(err, CliError::ValidationFailed(_)), "{err:?}");
}

// ── honest refusals ──────────────────────────────────────────────────────

#[test]
fn umap_is_refused_rather_than_silently_substituted() {
    let model = model_fixture();
    let dir = tempfile::tempdir().expect("tempdir");
    let out = dir.path().join("umap.csv");
    let err = run(&args(model.path(), &out, Projection::Umap))
        .expect_err("an algorithm this binary does not implement must not be labelled as run");
    assert!(matches!(err, CliError::NotImplemented(_)), "{err:?}");
    assert!(
        !out.exists(),
        "a refused projection must not leave a CSV behind"
    );
}

#[test]
fn a_missing_model_is_refused() {
    let dir = tempfile::tempdir().expect("tempdir");
    let out = dir.path().join("x.csv");
    let err = run(&args(
        Path::new("/no/such/model.apr"),
        &out,
        Projection::Pca,
    ))
    .expect_err("a missing model must not produce coordinates");
    assert!(matches!(err, CliError::FileNotFound(_)), "{err:?}");
}

#[test]
fn an_unknown_tensor_name_is_refused_and_names_what_was_asked_for() {
    let model = model_fixture();
    let dir = tempfile::tempdir().expect("tempdir");
    let out = dir.path().join("x.csv");
    let mut a = args(model.path(), &out, Projection::Random);
    a.tensor = Some("does.not.exist".to_string());
    let err = run(&a).expect_err("an absent tensor cannot be projected");
    assert!(err.to_string().contains("does.not.exist"), "got: {err}");
}

#[test]
fn a_one_dimensional_tensor_is_refused_as_an_embedding_table() {
    use aprender::format::v2::{AprV2Metadata, AprV2Writer};
    let file = tempfile::NamedTempFile::with_suffix(".apr").expect("tempfile");
    let mut writer = AprV2Writer::new(AprV2Metadata::new("1d"));
    writer.add_f32_tensor("model.norm.weight", vec![8], &[1.0f32; 8]);
    std::fs::write(file.path(), writer.write().expect("write")).expect("write file");

    let dir = tempfile::tempdir().expect("tempdir");
    let out = dir.path().join("x.csv");
    let mut a = args(file.path(), &out, Projection::Random);
    a.tensor = Some("model.norm.weight".to_string());
    let err = run(&a).expect_err("a 1-D tensor is not an embedding table");
    assert!(
        err.to_string().contains("2-D [vocab, hidden]"),
        "got: {err}"
    );
}

#[test]
fn a_zero_row_limit_is_refused() {
    let model = model_fixture();
    let dir = tempfile::tempdir().expect("tempdir");
    let out = dir.path().join("x.csv");
    let mut a = args(model.path(), &out, Projection::Random);
    a.limit = Some(0);
    let err = run(&a).expect_err("0 rows is not a projection");
    assert!(err.to_string().contains("0 rows"), "got: {err}");
}

#[test]
fn refuses_to_clobber_an_existing_csv_without_force() {
    let model = model_fixture();
    let dir = tempfile::tempdir().expect("tempdir");
    let out = dir.path().join("existing.csv");
    std::fs::write(&out, "precious").expect("write");
    let err = run(&args(model.path(), &out, Projection::Random))
        .expect_err("an existing CSV must not be overwritten silently");
    assert!(err.to_string().contains("--force"), "got: {err}");
}

#[test]
fn a_short_tokens_file_is_refused_rather_than_padded() {
    let model = model_fixture();
    let dir = tempfile::tempdir().expect("tempdir");
    let tokens = dir.path().join("tokens.txt");
    std::fs::write(&tokens, "a\nb\nc\n").expect("write");
    let out = dir.path().join("x.csv");
    let mut a = args(model.path(), &out, Projection::Random);
    a.tokens = Some(tokens);
    let err = run(&a).expect_err("3 tokens cannot label 24 rows");
    assert!(err.to_string().contains("3 lines"), "got: {err}");
}

// ── limits, tokens, escaping ─────────────────────────────────────────────

#[test]
fn limit_selects_the_first_n_rows_and_the_row_count_gate_sees_it() {
    let model = model_fixture();
    let dir = tempfile::tempdir().expect("tempdir");
    let out = dir.path().join("x.csv");
    let mut a = args(model.path(), &out, Projection::Random);
    a.limit = Some(5);
    run(&a).expect("producer");

    embed_viz_lint::run(&out, Some(5), None, false).expect("5 rows were requested and written");
    let err = embed_viz_lint::run(&out, Some(VOCAB), None, false)
        .expect_err("the row-count gate must notice the limit");
    assert!(matches!(err, CliError::ValidationFailed(_)), "{err:?}");
}

#[test]
fn tokens_from_a_file_land_in_the_csv() {
    let model = model_fixture();
    let dir = tempfile::tempdir().expect("tempdir");
    let tokens = dir.path().join("tokens.txt");
    let body: String = (0..VOCAB).map(|i| format!("tok{i}\n")).collect();
    std::fs::write(&tokens, body).expect("write");

    let out = dir.path().join("x.csv");
    let mut a = args(model.path(), &out, Projection::Random);
    a.tokens = Some(tokens);
    run(&a).expect("producer");

    let csv = std::fs::read_to_string(&out).expect("read");
    assert!(csv.contains(",tok0,"), "got: {csv}");
    embed_viz_lint::run(&out, Some(VOCAB), None, false).expect("lint");
}

/// A token containing a comma would shift the column count the F-18 classifier
/// counts, so the producer escapes it. This asserts the CLASSIFIER accepts the
/// escaped row — the property that matters — not merely that a byte changed.
#[test]
fn a_token_containing_a_comma_does_not_shift_the_column_count() {
    let tokens = ResolvedTokens {
        strings: vec![
            "a,b".to_string(),
            "quote\"here".to_string(),
            "back\\slash".to_string(),
            "line\nbreak".to_string(),
        ],
        source: "test".to_string(),
    };
    let csv = render_csv(&[(0.0, 1.0), (2.0, 3.0), (4.0, 5.0), (6.0, 7.0)], &tokens);
    assert_eq!(
        embed_viz_classifier::classify_schema(&csv),
        embed_viz_classifier::EmbedSchemaOutcome::Ok { rows: 4 },
        "escaped token text must keep 4 columns per row:\n{csv}"
    );
    assert_eq!(csv.lines().count(), 5, "header + 4 rows:\n{csv}");
}

#[test]
fn escape_token_is_reversible_in_the_characters_it_touches() {
    assert_eq!(escape_token("a,b"), "a\\x2cb");
    assert_eq!(escape_token("a\\b"), "a\\\\b");
    assert_eq!(escape_token("a\"b"), "a\\x22b");
    assert_eq!(escape_token("a\nb"), "a\\nb");
    assert_eq!(escape_token("plain"), "plain");
}

#[test]
fn unresolved_tokens_are_marked_not_invented() {
    let model = model_fixture();
    let dir = tempfile::tempdir().expect("tempdir");
    let out = dir.path().join("x.csv");
    run(&args(model.path(), &out, Projection::Random)).expect("producer");
    let csv = std::fs::read_to_string(&out).expect("read");
    assert!(
        csv.contains("<unresolved>"),
        "an APR fixture carries no vocabulary; token_str must say so: {csv}"
    );
}

// ── projection maths ─────────────────────────────────────────────────────

#[test]
fn the_projection_is_not_constant() {
    let model = model_fixture();
    let dir = tempfile::tempdir().expect("tempdir");
    for projection in [Projection::Pca, Projection::Random] {
        let out = dir.path().join("p.csv");
        let mut a = args(model.path(), &out, projection);
        a.force = true;
        run(&a).expect("producer");
        let csv = std::fs::read_to_string(&out).expect("read");
        let xs: Vec<&str> = csv
            .lines()
            .skip(1)
            .filter_map(|l| l.split(',').nth(2))
            .collect();
        assert!(
            xs.windows(2).any(|w| w[0] != w[1]),
            "{} collapsed every row onto one x: {csv}",
            projection_label(projection)
        );
    }
}

// ── GGUF: the first-class advertised path (aprender#2377-3 blockers 1-3) ─
//
// `token_embd.weight` is the FIRST entry in `EMBEDDING_TENSOR_CANDIDATES`, yet
// the only fixture in this file was an APR v2 table written as
// `[VOCAB, HIDDEN]` — the single layout where `shape[0]` really is the vocab
// axis. That made the GGUF axis inversion invisible: the producer emitted 1024
// rows for `Qwen3.5-0.8B-Q4_K_M.gguf`'s 248320-token vocabulary and
// `embed-viz-lint --expected-vocab-size 248320` exited 5 on its own producer's
// output.
//
// THE TRAP these tests are built to avoid: `token_str` is resolved from the
// vocabulary list by row index, so it reads correctly no matter which axis
// produced the coordinates. Asserting on `token_str` CANNOT detect this bug.
// What follows asserts the ROW COUNT against the real vocab size and the
// COORDINATES against a `hidden`-length slice.

/// The value at (token `t`, dim `d`) of the GGUF fixture's embedding table.
///
/// Rank-1 by construction — row `t` is `(t+1)` times a fixed profile — because
/// that makes the seeded JL projection EXACTLY linear in `t+1`, which is the
/// property `gguf_coordinates_come_from_the_hidden_axis` checks without having
/// to re-derive the RNG stream.
fn g_value(t: usize, d: usize) -> f32 {
    (t as f32 + 1.0) * (0.25f32.mul_add(d as f32, 1.0))
}

/// A minimal GGUF carrying `token_embd.weight` in real GGML `ne` order.
///
/// `ne` is `[hidden, vocab]` — the CONTIGUOUS dimension first — while the
/// payload is `[vocab][hidden]` rows, exactly as llama.cpp writes it and as
/// measured against a real model in both formats.
fn gguf_fixture(vocab: usize, hidden: usize, declared_tokens: usize) -> tempfile::NamedTempFile {
    use aprender::format::gguf::{export_tensors_to_gguf, GgmlType, GgufTensor, GgufValue};

    let mut bytes = Vec::new();
    for t in 0..vocab {
        for d in 0..hidden {
            bytes.extend_from_slice(&g_value(t, d).to_le_bytes());
        }
    }
    let tensor = GgufTensor {
        name: "token_embd.weight".to_string(),
        // GGML `ne` order: ne[0] is contiguous, so [hidden, vocab].
        shape: vec![hidden as u64, vocab as u64],
        dtype: GgmlType::F32,
        data: bytes,
    };
    let tokens: Vec<String> = (0..declared_tokens).map(|i| format!("tok{i}")).collect();
    let metadata = vec![
        (
            "general.architecture".to_string(),
            GgufValue::String("llama".to_string()),
        ),
        (
            "tokenizer.ggml.model".to_string(),
            GgufValue::String("gpt2".to_string()),
        ),
        (
            "tokenizer.ggml.tokens".to_string(),
            GgufValue::ArrayString(tokens),
        ),
    ];

    let file = tempfile::NamedTempFile::with_suffix(".gguf").expect("tempfile");
    let mut buf = Vec::new();
    export_tensors_to_gguf(&mut buf, &[tensor], &metadata).expect("write GGUF");
    std::fs::write(file.path(), buf).expect("write file");
    file
}

/// The per-format axis rule, as a case table. GGUF is `[hidden, vocab]`;
/// APR and SafeTensors are `[vocab, hidden]`.
#[test]
fn the_vocab_axis_is_chosen_per_format_not_assumed_to_be_axis_zero() {
    use aprender::format::rosetta::FormatType;

    // A real measurement: Qwen3.5-0.8B-Q4_K_M.gguf reports [1024, 248320] for a
    // 248320-token vocabulary at hidden size 1024.
    assert_eq!(
        embedding_axes(FormatType::Gguf, &[1024, 248_320]),
        (248_320, 1024),
        "GGUF ne order is [hidden, vocab]"
    );
    assert_eq!(
        embedding_axes(FormatType::SafeTensors, &[151_936, 896]),
        (151_936, 896),
        "SafeTensors is row-major [vocab, hidden]"
    );
    assert_eq!(
        embedding_axes(FormatType::Apr, &[VOCAB, HIDDEN]),
        (VOCAB, HIDDEN),
        "APR is row-major [vocab, hidden]"
    );
}

/// `locate_embedding` must hand back `(vocab, hidden)` in APR order whatever the
/// container said. This is the assertion that the projected slice is `hidden`
/// long: `run` slices `data[i * hidden .. (i + 1) * hidden]`.
#[test]
fn gguf_locate_embedding_returns_vocab_and_hidden_in_apr_order() {
    let model = gguf_fixture(G_VOCAB, G_HIDDEN, G_VOCAB);
    let (name, vocab, hidden) = locate_embedding(model.path(), None).expect("locate");
    assert_eq!(name, "token_embd.weight");
    assert_eq!(
        (vocab, hidden),
        (G_VOCAB, G_HIDDEN),
        "GGUF ne [{G_HIDDEN}, {G_VOCAB}] must be read as vocab={G_VOCAB} hidden={G_HIDDEN}"
    );
}

/// The round trip the audit found missing: producer -> lint, on GGUF.
#[test]
fn round_trip_gguf_producer_csv_is_accepted_by_embed_viz_lint() {
    let model = gguf_fixture(G_VOCAB, G_HIDDEN, G_VOCAB);
    let dir = tempfile::tempdir().expect("tempdir");

    for projection in [Projection::Pca, Projection::Random] {
        let csv = dir
            .path()
            .join(format!("g-{}.csv", projection_label(projection)));
        run(&args(model.path(), &csv, projection)).expect("producer must project a GGUF table");

        let body = std::fs::read_to_string(&csv).expect("read");
        assert_eq!(
            body.lines().count() - 1,
            G_VOCAB,
            "one row per TOKEN, not one per hidden dim:\n{body}"
        );

        embed_viz_lint::run(&csv, Some(G_VOCAB), None, false).unwrap_or_else(|e| {
            panic!(
                "embed-viz-lint must accept the producer's own GGUF {} CSV: {e}",
                projection_label(projection)
            )
        });
        // Pinned in both directions: the row count is G_VOCAB and nothing else.
        let err = embed_viz_lint::run(&csv, Some(G_HIDDEN), None, false)
            .expect_err("the row-count gate must reject the hidden dim as a vocab size");
        assert!(matches!(err, CliError::ValidationFailed(_)), "{err:?}");
    }
}

/// The coordinates must be the projection of `hidden`-long TOKEN rows.
///
/// The fixture is rank-1 — row `t` is `(t+1)` times a fixed profile — so the
/// seeded JL projection is exactly linear in `t+1`: `x_t / x_0 == t+1`. Slicing
/// the other axis mixes several tokens into each row and destroys that ratio.
/// This is the assertion `token_str` cannot make.
#[test]
fn gguf_coordinates_come_from_the_hidden_axis() {
    let model = gguf_fixture(G_VOCAB, G_HIDDEN, G_VOCAB);
    let dir = tempfile::tempdir().expect("tempdir");
    let csv = dir.path().join("g.csv");
    run(&args(model.path(), &csv, Projection::Random)).expect("producer");

    let body = std::fs::read_to_string(&csv).expect("read");
    let xs: Vec<f64> = body
        .lines()
        .skip(1)
        .map(|l| {
            l.split(',')
                .nth(2)
                .and_then(|f| f.parse::<f64>().ok())
                .unwrap_or_else(|| panic!("unparseable x in row `{l}`"))
        })
        .collect();
    assert_eq!(xs.len(), G_VOCAB);
    assert!(
        xs[0].abs() > 1e-6,
        "the fixture must not project token 0 onto the origin: {xs:?}"
    );
    for (t, x) in xs.iter().enumerate() {
        let expected = xs[0] * (t as f64 + 1.0);
        assert!(
            (x - expected).abs() <= 1e-4 * expected.abs().max(1.0),
            "row {t}: x={x} but a hidden-axis projection of a rank-1 table must give \
             {expected} (x_0 * (t+1)); the coordinates were projected from the wrong axis"
        );
    }
}

/// A GGUF whose vocabulary is LARGER than its embedding table has been read the
/// wrong way round — every token must have a row. Padding goes the other way
/// (more rows than tokens), so this one-sided check cannot fire spuriously.
#[test]
fn a_vocabulary_larger_than_the_embedding_table_is_refused() {
    let model = gguf_fixture(G_VOCAB, G_HIDDEN, G_VOCAB + 6);
    let dir = tempfile::tempdir().expect("tempdir");
    let out = dir.path().join("x.csv");
    let err = run(&args(model.path(), &out, Projection::Random))
        .expect_err("30 tokens cannot be embedded by a 24-row table");
    assert!(matches!(err, CliError::ValidationFailed(_)), "{err:?}");
    assert!(
        err.to_string().contains("wrong way round"),
        "the refusal must name the axis as the suspect: {err}"
    );
}

/// A table padded ABOVE the token list is normal and must still be projected.
#[test]
fn an_embedding_table_padded_above_the_token_list_is_accepted() {
    let model = gguf_fixture(G_VOCAB, G_HIDDEN, G_VOCAB - 4);
    let dir = tempfile::tempdir().expect("tempdir");
    let out = dir.path().join("x.csv");
    run(&args(model.path(), &out, Projection::Random))
        .expect("more rows than tokens is padding, not an inverted axis");
    embed_viz_lint::run(&out, Some(G_VOCAB), None, false).expect("lint");
}

#[test]
fn pca_needs_at_least_two_rows() {
    let model = model_fixture();
    let dir = tempfile::tempdir().expect("tempdir");
    let out = dir.path().join("x.csv");
    let mut a = args(model.path(), &out, Projection::Pca);
    a.limit = Some(1);
    let err = run(&a).expect_err("PCA on one sample has no variance to decompose");
    assert!(err.to_string().contains("at least 2 rows"), "got: {err}");
}
