//! `apr debug embed-viz` — the PRODUCER whose output `apr embed-viz-lint` reads
//! (aprender#2377 finding 3).
//!
//! CRUX-F-18 shipped a consumer with no producer: `embed-viz-lint`'s help
//! documented `apr debug embed-viz --seed N -o emb.csv` and `apr debug` had no
//! such subcommand, so the schema, row-count and determinism gates had never
//! run on real data and could not.
//!
//! ## What this claims
//!
//! It reads a REAL token-embedding matrix out of a GGUF / APR / SafeTensors
//! model (dequantising as needed, via `RosettaStone::load_tensor_f32`) and
//! projects it to 2-D with a named, deterministic method:
//!
//!   * `--projection pca` — exact PCA onto the top 2 principal components
//!     (`aprender::preprocessing::PCA`). Deterministic; cost is O(hidden²)
//!     memory for the covariance eigendecomposition.
//!   * `--projection random` — seeded Johnson–Lindenstrauss random projection.
//!     Deterministic in `--seed`, cheap at any hidden size.
//!
//! It does **not** implement UMAP, and `--projection umap` is REFUSED with a
//! non-zero exit rather than silently substituting a different algorithm and
//! labelling the output "umap". The CSV header and the `projection` note the
//! producer prints name the method that actually ran.
//!
//! `token_str` is resolved from the model's own GGUF vocabulary, or from
//! `--tokens`. When neither is available every row carries the literal
//! `<unresolved>` — a marker that claims nothing — and the producer says so on
//! stderr. Token text is escaped (`\` → `\\`, `,` → `\x2c`, `"` → `\x22`,
//! CR/LF → `\r`/`\n`) so a token containing a comma cannot silently shift the
//! column count the F-18 classifier counts.

use std::path::{Path, PathBuf};

use aprender::format::rosetta::RosettaStone;
use aprender::format::tensors::{list_tensors, TensorListOptions};

use crate::error::{refuse_overwrite, CliError, Result};

/// `--projection` is declared at the crate root (`extended_commands.rs`) because
/// `ExtendedCommands` is public and `mod commands` is not.
pub(crate) use crate::EmbedProjection as Projection;

/// Name the projection that actually ran — this string goes in the report, so
/// it must never say `umap` for something else.
pub(crate) fn projection_label(p: Projection) -> &'static str {
    match p {
        Projection::Pca => "pca",
        Projection::Random => "random",
        Projection::Umap => "umap",
    }
}

/// Tensor names that hold token embeddings across the architectures apr reads.
pub(crate) const EMBEDDING_TENSOR_CANDIDATES: [&str; 6] = [
    "token_embd.weight",
    "model.embed_tokens.weight",
    "tok_embeddings.weight",
    "transformer.wte.weight",
    "wte.weight",
    "embeddings.word_embeddings.weight",
];

/// Options for one `embed-viz` run.
#[derive(Debug, Clone)]
pub(crate) struct EmbedVizArgs {
    pub model: PathBuf,
    pub tensor: Option<String>,
    pub projection: Projection,
    pub seed: u64,
    pub limit: Option<usize>,
    pub tokens: Option<PathBuf>,
    pub output: Option<PathBuf>,
    pub force: bool,
}

/// Run the producer.
pub(crate) fn run(args: &EmbedVizArgs) -> Result<()> {
    if args.projection == Projection::Umap {
        return Err(CliError::NotImplemented(
            "apr debug embed-viz: --projection umap is not implemented in this binary. \
             Refusing rather than labelling a different algorithm's output `umap`. \
             Use --projection pca (exact) or --projection random (seeded JL)."
                .to_string(),
        ));
    }
    if !args.model.exists() {
        return Err(CliError::FileNotFound(args.model.clone()));
    }
    if let Some(out) = &args.output {
        refuse_overwrite(out, args.force)?;
    }

    let (name, vocab, hidden) = locate_embedding(&args.model, args.tensor.as_deref())?;
    let data = RosettaStone::new()
        .load_tensor_f32(&args.model, &name)
        .map_err(|e| {
            CliError::ValidationFailed(format!(
                "apr debug embed-viz: cannot read tensor `{name}` from {}: {e}",
                args.model.display()
            ))
        })?;
    if data.len() != vocab * hidden {
        return Err(CliError::ValidationFailed(format!(
            "apr debug embed-viz: tensor `{name}` says {vocab}x{hidden} but decoded to {} values",
            data.len()
        )));
    }

    let rows = args.limit.map_or(vocab, |n| n.min(vocab));
    if rows == 0 {
        return Err(CliError::ValidationFailed(
            "apr debug embed-viz: 0 rows selected, so there is nothing to project".to_string(),
        ));
    }
    let coords = project(&data[..rows * hidden], rows, hidden, args)?;
    let tokens = resolve_tokens(args, rows)?;
    let csv = render_csv(&coords, &tokens);

    match &args.output {
        Some(out) => std::fs::write(out, &csv)?,
        None => print!("{csv}"),
    }
    eprintln!(
        "embed-viz: {name} [{vocab}x{hidden}] -> {rows} rows, projection={}, seed={}, token_str={}",
        projection_label(args.projection),
        args.seed,
        tokens.source
    );
    Ok(())
}

/// Find the embedding tensor and its 2-D shape.
fn locate_embedding(model: &Path, requested: Option<&str>) -> Result<(String, usize, usize)> {
    let listing = list_tensors(model, TensorListOptions::default()).map_err(|e| {
        CliError::ValidationFailed(format!(
            "apr debug embed-viz: cannot list tensors in {}: {e}",
            model.display()
        ))
    })?;
    let info = match requested {
        Some(want) => listing
            .tensors
            .iter()
            .find(|t| t.name == want)
            .ok_or_else(|| {
                CliError::ValidationFailed(format!(
                    "apr debug embed-viz: {} has no tensor named `{want}`",
                    model.display()
                ))
            })?,
        None => listing
            .tensors
            .iter()
            .find(|t| EMBEDDING_TENSOR_CANDIDATES.contains(&t.name.as_str()))
            .ok_or_else(|| {
                CliError::ValidationFailed(format!(
                    "apr debug embed-viz: {} has none of the known embedding tensors {:?}; \
                     name one explicitly with --tensor",
                    model.display(),
                    EMBEDDING_TENSOR_CANDIDATES
                ))
            })?,
    };
    if info.shape.len() != 2 || info.shape[0] == 0 || info.shape[1] == 0 {
        return Err(CliError::ValidationFailed(format!(
            "apr debug embed-viz: tensor `{}` has shape {:?}; an embedding table must be \
             2-D [vocab, hidden]",
            info.name, info.shape
        )));
    }
    Ok((info.name.clone(), info.shape[0], info.shape[1]))
}

// ── projection ───────────────────────────────────────────────────────────

fn project(
    data: &[f32],
    rows: usize,
    hidden: usize,
    args: &EmbedVizArgs,
) -> Result<Vec<(f64, f64)>> {
    let coords = match args.projection {
        Projection::Pca => project_pca(data, rows, hidden)?,
        Projection::Random => project_random(data, rows, hidden, args.seed),
        // `run` refuses Umap before reaching here.
        Projection::Umap => unreachable!("umap is refused at entry"),
    };
    if let Some((i, bad)) = coords
        .iter()
        .enumerate()
        .find(|(_, (x, y))| !x.is_finite() || !y.is_finite())
    {
        return Err(CliError::ValidationFailed(format!(
            "apr debug embed-viz: row {i} projected to a non-finite coordinate {bad:?}; \
             refusing to write a CSV a consumer would have to reject"
        )));
    }
    Ok(coords)
}

fn project_pca(data: &[f32], rows: usize, hidden: usize) -> Result<Vec<(f64, f64)>> {
    use aprender::preprocessing::PCA;
    use aprender::primitives::Matrix;
    use aprender::traits::Transformer;

    if rows < 2 {
        return Err(CliError::ValidationFailed(
            "apr debug embed-viz: --projection pca needs at least 2 rows; \
             use --projection random for a single row"
                .to_string(),
        ));
    }
    let x = Matrix::from_vec(rows, hidden, data.to_vec())
        .map_err(|e| CliError::ValidationFailed(format!("apr debug embed-viz: {e}")))?;
    let mut pca = PCA::new(2);
    pca.fit(&x)
        .map_err(|e| CliError::ValidationFailed(format!("apr debug embed-viz: PCA fit: {e}")))?;
    let y = pca.transform(&x).map_err(|e| {
        CliError::ValidationFailed(format!("apr debug embed-viz: PCA transform: {e}"))
    })?;
    Ok((0..rows)
        .map(|i| (f64::from(y.get(i, 0)), f64::from(y.get(i, 1))))
        .collect())
}

/// Seeded Johnson–Lindenstrauss projection: X · R / sqrt(hidden), R ~ U[-1, 1).
fn project_random(data: &[f32], rows: usize, hidden: usize, seed: u64) -> Vec<(f64, f64)> {
    let mut rng = super::kernel_parity::SplitMix64::new(seed ^ 0xF18_F18_F18);
    let r: Vec<f32> = (0..hidden * 2).map(|_| rng.next_unit()).collect();
    let norm = 1.0 / (hidden as f64).sqrt();
    (0..rows)
        .map(|i| {
            let row = &data[i * hidden..(i + 1) * hidden];
            let mut x = 0.0f64;
            let mut y = 0.0f64;
            for (d, value) in row.iter().enumerate() {
                x += f64::from(*value) * f64::from(r[d * 2]);
                y += f64::from(*value) * f64::from(r[d * 2 + 1]);
            }
            (x * norm, y * norm)
        })
        .collect()
}

// ── token text ───────────────────────────────────────────────────────────

/// Resolved token strings plus a note saying where they came from.
pub(crate) struct ResolvedTokens {
    pub strings: Vec<String>,
    pub source: String,
}

const UNRESOLVED: &str = "<unresolved>";

fn resolve_tokens(args: &EmbedVizArgs, rows: usize) -> Result<ResolvedTokens> {
    if let Some(path) = &args.tokens {
        let text = std::fs::read_to_string(path)?;
        let mut strings: Vec<String> = text.lines().map(str::to_string).collect();
        if strings.len() < rows {
            return Err(CliError::ValidationFailed(format!(
                "apr debug embed-viz: --tokens {} holds {} lines but {rows} rows were projected",
                path.display(),
                strings.len()
            )));
        }
        strings.truncate(rows);
        return Ok(ResolvedTokens {
            strings,
            source: format!("--tokens {}", path.display()),
        });
    }
    if let Some(vocab) = gguf_vocab(&args.model, rows) {
        return Ok(ResolvedTokens {
            strings: vocab,
            source: "gguf tokenizer.ggml.tokens".to_string(),
        });
    }
    eprintln!(
        "embed-viz: no token text available for {} — every token_str is `{UNRESOLVED}`. \
         Pass --tokens FILE to resolve them.",
        args.model.display()
    );
    Ok(ResolvedTokens {
        strings: vec![UNRESOLVED.to_string(); rows],
        source: UNRESOLVED.to_string(),
    })
}

/// Read `tokenizer.ggml.tokens` out of a GGUF, when the model is one.
fn gguf_vocab(model: &Path, rows: usize) -> Option<Vec<String>> {
    use aprender::text::llama_tokenizer::LlamaTokenizer;
    let bytes = std::fs::read(model).ok()?;
    if !bytes.starts_with(b"GGUF") {
        return None;
    }
    let tokenizer = LlamaTokenizer::from_gguf_bytes(&bytes).ok()?;
    let mut out = Vec::with_capacity(rows);
    for id in 0..rows {
        out.push(tokenizer.id_to_token(u32::try_from(id).ok()?)?.to_string());
    }
    Some(out)
}

/// Escape token text so it can never change the CSV column count.
pub(crate) fn escape_token(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    for ch in raw.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            ',' => out.push_str("\\x2c"),
            '"' => out.push_str("\\x22"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            c => out.push(c),
        }
    }
    out
}

/// Render the `token_id,token_str,x,y` CSV the F-18 classifier parses.
pub(crate) fn render_csv(coords: &[(f64, f64)], tokens: &ResolvedTokens) -> String {
    let mut out = String::from("token_id,token_str,x,y\n");
    for (id, (x, y)) in coords.iter().enumerate() {
        let token = tokens.strings.get(id).map_or(UNRESOLVED, String::as_str);
        out.push_str(&format!("{id},{},{x:.6},{y:.6}\n", escape_token(token)));
    }
    out
}

#[cfg(test)]
#[path = "embed_viz_tests.rs"]
mod tests;
