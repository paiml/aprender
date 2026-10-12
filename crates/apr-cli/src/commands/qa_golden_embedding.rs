// APR-EMBED-001 EG-5 (#4904): the golden gate for an embedding model.
//
// An embedding model generates no text, so `golden_output` cannot judge it and every gate built on
// a token stream (throughput, Ollama parity, GPU speedup, format parity, PTX parity, GPU state
// isolation) has no instrument for it. `golden_embedding` judges the vectors instead: the width
// the file declares, finite unit-length values, a bit-equal repeat, and on each golden triple the
// paraphrase scoring nearer the anchor than the unrelated sentence.

/// Skip reason for a token-stream gate on an embedding model.
const EMBEDDING_KIND_SKIP: &str =
    "Embedding model (ModelKind::Embedding): it generates no text, so golden_embedding judges it";

/// Skip reason for `golden_embedding` on a model that is not an embedding model.
const GENERATIVE_KIND_SKIP: &str =
    "Not an embedding model: golden_output judges a generative model";

/// Allowed distance of an output vector's L2 norm from 1.
const GOLDEN_EMBEDDING_NORM_TOL: f64 = 1e-3;

/// Skip decision for a token-stream gate on an embedding model.
fn embedding_gate_skip(embedding: bool, skip: bool, reason: &'static str) -> (bool, &'static str) {
    if embedding {
        (true, EMBEDDING_KIND_SKIP)
    } else {
        (skip, reason)
    }
}

/// Skip decision for `golden_embedding`: only an embedding model runs it.
fn golden_embedding_skip(embedding: bool, skip: bool, reason: &'static str) -> (bool, &'static str) {
    if embedding {
        (skip, reason)
    } else {
        (true, GENERATIVE_KIND_SKIP)
    }
}

/// Golden triples: (anchor, paraphrase, unrelated). The paraphrase must score nearer the anchor.
fn golden_embedding_triples() -> [(&'static str, &'static str, &'static str); 3] {
    [
        (
            "A cat is sitting on the mat.",
            "A kitten rests on a rug.",
            "The stock market fell sharply today.",
        ),
        (
            "How do I reset my password?",
            "I forgot my password and need to change it.",
            "What is the boiling point of water?",
        ),
        (
            "Water boils at 100 degrees Celsius at sea level.",
            "At sea level, water reaches its boiling point at 100 degrees Celsius.",
            "The football match ended in a draw.",
        ),
    ]
}

/// Cosine of two vectors (the encoder's vectors are unit length, but the gate does not assume it).
fn golden_cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| f64::from(*x) * f64::from(*y)).sum();
    let na: f64 = a.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    dot / (na * nb)
}

/// The first defect of one output vector, if any: its width, a non-finite value, or its norm.
fn golden_vector_defect(text: &str, vector: &[f32], width: usize) -> Option<String> {
    if vector.len() != width {
        return Some(format!("{text:?}: {} values, the file declares {width}", vector.len()));
    }
    if let Some(i) = vector.iter().position(|v| !v.is_finite()) {
        return Some(format!("{text:?}: value {i} is {}", vector[i]));
    }
    let norm = vector.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    if (norm - 1.0).abs() > GOLDEN_EMBEDDING_NORM_TOL {
        return Some(format!("{text:?}: norm {norm:.6}, not unit length"));
    }
    None
}

/// Judge the triples' vectors: `Err` names the first defect, `Ok` is the smallest gap
/// cos(anchor, paraphrase) − cos(anchor, unrelated).
fn judge_golden_embeddings(
    triples: &[(&str, &str, &str)],
    vectors: &[[Vec<f32>; 3]],
    width: usize,
) -> std::result::Result<f64, String> {
    let mut min_gap = f64::INFINITY;
    for ((anchor, near, far), [va, vn, vf]) in triples.iter().zip(vectors) {
        for (text, v) in [(anchor, va), (near, vn), (far, vf)] {
            if let Some(defect) = golden_vector_defect(text, v, width) {
                return Err(defect);
            }
        }
        let (cn, cf) = (golden_cosine(va, vn), golden_cosine(va, vf));
        if cn <= cf {
            return Err(format!(
                "{anchor:?}: paraphrase cosine {cn:.4} <= unrelated cosine {cf:.4}"
            ));
        }
        min_gap = min_gap.min(cn - cf);
    }
    Ok(min_gap)
}

/// `golden_embedding`: embed the golden triples with the file's own tokenizer, through the call
/// `apr embed` and `apr serve` make (EG-3), and judge the vectors.
#[cfg(feature = "inference")]
fn run_golden_embedding_gate(path: &Path, config: &QaConfig) -> Result<GateResult> {
    use realizar::gguf::embedding_gemma2_input::{EmbedTokenizer, Prompt, Task};
    use realizar::gguf::forward_embedding_gemma2::{embed_text, EmbeddingGemma2};
    use realizar::gguf::MappedGGUFModel;

    let start = Instant::now();
    if !config.json && config.verbose {
        println!("{}", "Running golden embedding test...".yellow());
    }
    let load = |e: realizar::RealizarError| CliError::ValidationFailed(format!("Model failed: {e}"));
    let mapped = MappedGGUFModel::from_path(path).map_err(load)?;
    let encoder = EmbeddingGemma2::from_mapped(&mapped).map_err(load)?;
    let tokenizer = EmbedTokenizer::from_gguf(&mapped.model).map_err(load)?;
    let prompt = Prompt::Query(Task::SentenceSimilarity);
    let embed = |text: &str| -> Result<Vec<f32>> {
        embed_text(&encoder, &tokenizer, &prompt, text, None)
            .map(|(_, v)| v)
            .map_err(|e| CliError::ValidationFailed(format!("{text:?}: {e}")))
    };
    let triples = golden_embedding_triples();
    let mut vectors = Vec::with_capacity(triples.len());
    for (a, n, f) in &triples {
        vectors.push([embed(a)?, embed(n)?, embed(f)?]);
    }
    let width = encoder.output_dim();
    let fail = |msg: &str, value| GateResult::failed("golden_embedding", msg, value, Some(0.0), start.elapsed());
    // Determinism: the same text gives the same bits.
    if embed(triples[0].0)? != vectors[0][0] {
        return Ok(fail("Repeat of the same text gave a different vector", None));
    }
    Ok(match judge_golden_embeddings(&triples, &vectors, width) {
        Ok(gap) => GateResult::passed(
            "golden_embedding",
            &format!(
                "{} triples, {width} wide, unit length, repeat bit-equal; min gap {gap:.4}",
                triples.len()
            ),
            Some(gap),
            Some(0.0),
            start.elapsed(),
        ),
        Err(defect) => fail(&defect, None),
    })
}

#[cfg(not(feature = "inference"))]
fn run_golden_embedding_gate(_path: &Path, _config: &QaConfig) -> Result<GateResult> {
    Ok(GateResult::skipped(
        "golden_embedding",
        "Requires the 'inference' feature",
    ))
}

#[cfg(test)]
mod golden_embedding_tests {
    use super::*;

    fn unit(v: &[f32]) -> Vec<f32> {
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter().map(|x| x / n).collect()
    }

    fn triple() -> [(&'static str, &'static str, &'static str); 1] {
        [("a", "b", "c")]
    }

    #[test]
    fn falsify_eg5_golden_embedding_passes_an_ordered_triple() {
        let v = [[unit(&[1.0, 0.0]), unit(&[0.9, 0.1]), unit(&[0.0, 1.0])]];
        let gap = judge_golden_embeddings(&triple(), &v, 2).expect("ordered");
        assert!(gap > 0.9, "{gap}");
    }

    #[test]
    fn falsify_eg5_golden_embedding_fails_a_reversed_triple() {
        let v = [[unit(&[1.0, 0.0]), unit(&[0.0, 1.0]), unit(&[0.9, 0.1])]];
        let err = judge_golden_embeddings(&triple(), &v, 2).expect_err("reversed");
        assert!(err.contains("paraphrase cosine"), "{err}");
    }

    #[test]
    fn falsify_eg5_golden_embedding_fails_width_nan_and_norm() {
        let good = unit(&[1.0, 0.0]);
        let cases = [
            (vec![1.0_f32, 0.0, 0.0], "declares 2"),
            (vec![f32::NAN, 0.0], "is NaN"),
            (vec![2.0_f32, 0.0], "not unit length"),
        ];
        for (bad, want) in cases {
            let v = [[bad, good.clone(), unit(&[0.0, 1.0])]];
            let err = judge_golden_embeddings(&triple(), &v, 2).expect_err(want);
            assert!(err.contains(want), "{err} lacks {want}");
        }
    }

    #[test]
    fn falsify_eg5_embedding_gate_skip_names_the_kind() {
        assert_eq!(embedding_gate_skip(true, false, "x"), (true, EMBEDDING_KIND_SKIP));
        assert_eq!(embedding_gate_skip(false, false, "x"), (false, "x"));
    }

    #[test]
    fn falsify_eg5_golden_embedding_skips_a_generative_model() {
        assert_eq!(golden_embedding_skip(false, false, "x"), (true, GENERATIVE_KIND_SKIP));
        assert_eq!(golden_embedding_skip(true, true, "x"), (true, "x"));
        assert_eq!(golden_embedding_skip(true, false, "x"), (false, "x"));
    }
}
