// ════════════════════════════════════════════════════════════════════
// APR-EMBED-001 EG-1: `apr flow` on an embedding model draws the embedding path by kind:
// bidirectional layers, pooling, projection, L2 normalisation. Never a decoder with an LM head
// and a next-token output (the EG-0 probe's silent wrong answer).
// Contract: contracts/embeddinggemma2-load-v1.yaml (model_kind, layer_attention_kind).
// ════════════════════════════════════════════════════════════════════

use realizar::model_kind::{layer_attention_kinds, model_kind, LayerAttention, ModelKind};

/// What `apr flow` draws for an embedding model. Every field is read from the file (I-9).
#[derive(Debug, Clone, PartialEq, Eq)]
struct EmbeddingFlow {
    /// `general.architecture`, as the file carries it.
    architecture: String,
    /// `<arch>.embedding_length`.
    hidden: usize,
    /// One entry per layer, from the output width of `blk.<l>.attn_q.weight`.
    layers: Vec<LayerAttention>,
    /// `<arch>.attention.sliding_window`, the local layers' window.
    sliding_window: Option<usize>,
    /// A `blk.0.ffn_gate.weight` tensor is present.
    gated_ffn: bool,
    /// `<arch>.pooling_type`, named.
    pooling: String,
    /// `<arch>.embedding_length_out`, the projected width.
    output_dim: Option<usize>,
}

/// The embedding path of `report`, or `None` for a file that is not an embedding model.
///
/// # Errors
/// An embedding architecture whose `attention.causal` flag contradicts it, a missing
/// `embedding_length` or `block_count`, or layer attention kinds that cannot be read.
fn embedding_flow(
    report: &aprender::format::rosetta::InspectionReport,
) -> Result<Option<EmbeddingFlow>, CliError> {
    let Some(arch) = report.metadata.get("general.architecture") else {
        return Ok(None);
    };
    let causal = report
        .metadata
        .get(&format!("{arch}.attention.causal"))
        .map(|v| v == "true");
    if model_kind(arch, causal).map_err(CliError::InvalidFormat)? == ModelKind::Generative {
        return Ok(None);
    }
    let required = |key: &str| {
        meta_usize(report, arch, key)
            .ok_or_else(|| CliError::InvalidFormat(format!("embedding model without {arch}.{key}")))
    };
    let hidden = required("embedding_length")?;
    let widths = (0..required("block_count")?)
        .map(|l| attn_q_width(report, l, hidden))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Some(EmbeddingFlow {
        architecture: arch.clone(),
        hidden,
        layers: layer_attention_kinds(&widths).map_err(CliError::InvalidFormat)?,
        sliding_window: meta_usize(report, arch, "attention.sliding_window"),
        gated_ffn: report
            .tensors
            .iter()
            .any(|t| t.name == "blk.0.ffn_gate.weight"),
        pooling: pooling_name(report.metadata.get(&format!("{arch}.pooling_type"))),
        output_dim: meta_usize(report, arch, "embedding_length_out"),
    }))
}

/// `<arch>.<key>` as an unsigned integer.
fn meta_usize(
    report: &aprender::format::rosetta::InspectionReport,
    arch: &str,
    key: &str,
) -> Option<usize> {
    report.metadata.get(&format!("{arch}.{key}"))?.parse().ok()
}

/// The output width of layer `l`'s query projection: its element count over the model width.
/// Independent of the order the format lists the two dimensions in.
fn attn_q_width(
    report: &aprender::format::rosetta::InspectionReport,
    l: usize,
    hidden: usize,
) -> Result<usize, CliError> {
    let name = format!("blk.{l}.attn_q.weight");
    let numel: usize = report
        .tensors
        .iter()
        .find(|t| t.name == name)
        .ok_or_else(|| CliError::InvalidFormat(format!("embedding model without {name}")))?
        .shape
        .iter()
        .product();
    if hidden == 0 || numel % hidden != 0 {
        return Err(CliError::InvalidFormat(format!(
            "{name} has {numel} elements, not a multiple of embedding_length {hidden}"
        )));
    }
    Ok(numel / hidden)
}

/// The GGUF `pooling_type` value, named (llama.cpp's `llama_pooling_type`).
fn pooling_name(value: Option<&String>) -> String {
    match value.map(String::as_str) {
        None => "unspecified".to_string(),
        Some("0") => "none".to_string(),
        Some("1") => "mean".to_string(),
        Some("2") => "cls".to_string(),
        Some("3") => "last".to_string(),
        Some("4") => "rank".to_string(),
        Some(other) => format!("pooling_type {other}"),
    }
}

/// Indices of the global-attention layers.
fn global_layers(flow: &EmbeddingFlow) -> Vec<usize> {
    flow.layers
        .iter()
        .enumerate()
        .filter(|(_, k)| **k == LayerAttention::Global)
        .map(|(l, _)| l)
        .collect()
}

/// The embedding path as plain lines, one stage per box.
fn embedding_flow_lines(flow: &EmbeddingFlow) -> Vec<String> {
    let global = global_layers(flow);
    let local = flow.layers.len() - global.len();
    let window = flow
        .sliding_window
        .map_or_else(String::new, |w| format!(", window {w}"));
    let global_list = global
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    let ffn = if flow.gated_ffn {
        "gated GELU FFN"
    } else {
        "GELU FFN"
    };
    let out = flow
        .output_dim
        .map_or_else(|| "out_dim".to_string(), |d| d.to_string());
    let stages: [(String, Vec<String>); 6] = [
        (
            "Token Embed".to_string(),
            vec![format!(
                "token_embd.weight → [batch, seq_len, {}]",
                flow.hidden
            )],
        ),
        (
            format!(
                "Layers × {} (bidirectional, no causal mask)",
                flow.layers.len()
            ),
            vec![
                format!(
                    "local × {local}{window}; global × {}: {global_list}",
                    global.len()
                ),
                "RMSNorm → self-attention → + residual".to_string(),
                format!("RMSNorm → {ffn} → + residual"),
            ],
        ),
        ("Final RMSNorm".to_string(), vec![]),
        (
            format!("Pool: {}", flow.pooling),
            vec![format!("over real tokens → [batch, {}]", flow.hidden)],
        ),
        (
            "Projection".to_string(),
            vec![format!("[batch, {}] → [batch, {out}]", flow.hidden)],
        ),
        ("L2 normalise".to_string(), vec![]),
    ];
    let mut lines = vec!["INPUT  tokens [batch, seq_len]".to_string()];
    for (title, body) in &stages {
        lines.extend(["   │".to_string(), "   ▼".to_string()]);
        lines.extend(boxed(title, body));
    }
    lines.extend([
        "   │".to_string(),
        "   ▼".to_string(),
        format!("OUTPUT  embedding [batch, {out}]"),
    ]);
    lines
}

/// One stage in a box sized to its widest line.
fn boxed(title: &str, body: &[String]) -> Vec<String> {
    let rows: Vec<&str> = std::iter::once(title)
        .chain(body.iter().map(String::as_str))
        .collect();
    let width = rows.iter().map(|r| r.chars().count()).max().unwrap_or(0);
    let mut out = vec![format!("┌{}┐", "─".repeat(width + 2))];
    for row in rows {
        let pad = width - row.chars().count();
        out.push(format!("│ {row}{} │", " ".repeat(pad)));
    }
    out.push(format!("└{}┘", "─".repeat(width + 2)));
    out
}

/// The embedding path as JSON.
fn embedding_flow_json(
    path: &Path,
    report: &aprender::format::rosetta::InspectionReport,
    flow: &EmbeddingFlow,
) -> serde_json::Value {
    serde_json::json!({
        "file": path.file_name().and_then(|s| s.to_str()).unwrap_or("model"),
        "format": report.format.to_string(),
        "architecture": flow.architecture,
        "kind": "embedding",
        "component": "full",
        "total_tensors": report.tensors.len(),
        "hidden": flow.hidden,
        "layers": flow.layers.len(),
        "global_layers": global_layers(flow),
        "sliding_window": flow.sliding_window,
        "gated_ffn": flow.gated_ffn,
        "pooling": flow.pooling,
        "output_dim": flow.output_dim,
        "normalize": "l2",
    })
}

/// Draw an embedding model whole. Only `--component full` applies: it has no decoder, encoder
/// block or cross-attention, and drawing one would be a wrong answer.
fn run_embedding_flow(
    path: &Path,
    report: &aprender::format::rosetta::InspectionReport,
    flow: &EmbeddingFlow,
    component: FlowComponent,
    json_output: bool,
) -> Result<(), CliError> {
    if component != FlowComponent::Full {
        return Err(CliError::InvalidInput(format!(
            "'{}' is an embedding model (ModelKind::Embedding): `apr flow` draws it whole; \
             --component {component:?} applies to generative and encoder-decoder models. \
             Use --component full.",
            flow.architecture
        )));
    }
    if json_output {
        let json = embedding_flow_json(path, report, flow);
        println!(
            "{}",
            serde_json::to_string_pretty(&json).unwrap_or_else(|_| json.to_string())
        );
        return Ok(());
    }
    println!(
        "{} {} ({}, {}, {})",
        "Model:".bold(),
        path.file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("model")
            .cyan(),
        "embedding".yellow(),
        flow.architecture,
        format!("{}", report.format).cyan()
    );
    println!();
    for line in embedding_flow_lines(flow) {
        println!("{line}");
    }
    Ok(())
}
