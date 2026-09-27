//! #3852 probe: do `layout_contract_enforce`'s UNWIRED guards fire on models we
//! currently report green?
//!
//! `enforce_load_contract`, `enforce_embedding_contract`,
//! `enforce_matmul_contract` and `validate_ffn_shape_symmetry` have **zero
//! production callers** (aprender-5e, proven by enumeration). Two of them are
//! documented MANDATORY and one says in its own assert that a violation "will
//! cause garbage inference output". CLAUDE.md says this defect class "has
//! occurred 100+ times". So their silence has never been evidence of anything.
//!
//! This CALLS them — it does not re-implement their predicates, because a
//! re-implementation would answer a question about my copy rather than theirs.
//! It does NOT wire them: that changes a load path for every model and is not
//! a release-night edit.
//!
//! The comparison is between two INDEPENDENT sources, which is the only way
//! these guards can say anything: the tensor table's shapes on one side, and
//! the GGUF metadata's `embedding_length` / `feed_forward_length` on the other.
//! Feeding both sides from the same shape would be vacuous.
//!
//! Input on stdin, one record per line:
//! ```text
//! M|<model>|<hidden>|<ffn>|<vocab>|<attn_in>
//! T|<tensor>|<d0>,<d1>          (dims as the tensor table reports them)
//! ```
use std::panic::{catch_unwind, AssertUnwindSafe};

fn fired<F: FnOnce()>(f: F) -> Option<String> {
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let r = catch_unwind(AssertUnwindSafe(f));
    std::panic::set_hook(prev);
    r.err().map(|e| {
        e.downcast_ref::<String>()
            .cloned()
            .or_else(|| e.downcast_ref::<&str>().map(|s| (*s).to_string()))
            .unwrap_or_else(|| "<non-string panic>".to_string())
            .lines()
            .next()
            .unwrap_or("")
            .to_string()
    })
}

struct Model {
    name: String,
    hidden: usize,
    ffn: usize,
    vocab: usize,
    /// `attn_output`'s in_dim = head_count * value_length, which is NOT
    /// hidden_dim on a model whose value_length is not hidden/head_count.
    /// Assuming it was cost me a false release-critical finding: every
    /// qwen3.5 hybrid "violated" the contract at blk.3/7/11/15 —
    /// `full_attention_interval = 4` — and 16 heads x 256 value_length = 4096
    /// against hidden 2560. The files were right and the expectation was mine.
    attn_in: usize,
    arch: String,
    kv_heads_key: usize,
    qkv_width: usize,
    ts_rank: usize,
    conv_kernel: usize,
    experts: usize,
    tensors: Vec<(String, Vec<usize>)>,
}

impl Model {
    /// The tensor table reports `ne` (reversed), so `[in, out]` there.
    fn dims(&self, name: &str) -> Option<(usize, usize)> {
        self.tensors
            .iter()
            .find(|(n, _)| n == name)
            .filter(|(_, d)| d.len() == 2)
            .map(|(_, d)| (d[0], d[1]))
    }

    /// Row-major `[out, in]`, the order the contract is written in.
    fn row_major(&self, name: &str) -> Option<Vec<usize>> {
        self.dims(name).map(|(i, o)| vec![o, i])
    }
}

/// Parse the probe's line format into models.
fn read_models(input: impl std::io::BufRead) -> Vec<Model> {
    let mut models: Vec<Model> = Vec::new();
    for line in input.lines().map_while(Result::ok) {
        let p: Vec<&str> = line.trim().split('|').collect();
        match p.as_slice() {
            ["M", name, h, f, v, a, arch, kvk, qkv, tsr, ck, ex] => models.push(Model {
                name: (*name).to_string(),
                hidden: h.parse().unwrap_or(0),
                ffn: f.parse().unwrap_or(0),
                vocab: v.parse().unwrap_or(0),
                attn_in: a.parse().unwrap_or(0),
                arch: (*arch).to_string(),
                kv_heads_key: kvk.parse().unwrap_or(0),
                qkv_width: qkv.parse().unwrap_or(0),
                ts_rank: tsr.parse().unwrap_or(0),
                conv_kernel: ck.parse().unwrap_or(0),
                experts: ex.parse().unwrap_or(0),
                tensors: Vec::new(),
            }),
            ["T", name, dims] => push_tensor(&mut models, name, dims),
            _ => {}
        }
    }
    models
}

fn push_tensor(models: &mut [Model], name: &str, dims: &str) {
    if let Some(m) = models.last_mut() {
        let d: Vec<usize> = dims.split(',').filter_map(|x| x.parse().ok()).collect();
        m.tensors.push((name.to_string(), d));
    }
}

/// `enforce_embedding_contract`: tensor element count vs METADATA dims.
fn probe_embedding(m: &Model) -> Vec<String> {
    let Some((i, o)) = m.dims("token_embd.weight") else {
        return Vec::new();
    };
    let elems = i * o;
    fired(|| {
        aprender::format::layout_contract::enforce_embedding_contract(elems, m.vocab, m.hidden);
    })
    .map(|why| format!("enforce_embedding_contract: {why}"))
    .into_iter()
    .collect()
}

/// #3863: the per-architecture expectation table, DERIVED from the inventory
/// rather than assumed.
///
/// The naive version of this — `attn_output` expected as `[hidden, hidden]` —
/// fires on every qwen3.5 hybrid at blk.3/7/11/15 (`full_attention_interval`)
/// because that in_dim is `head_count * key_length`, 4096 against hidden 2560.
/// So a wrong table is worse than no table, and the table below was obtained by
/// fitting candidate metadata expressions against every model of each
/// architecture and keeping only formulas that hold for ALL of them.
///
/// EVIDENCE PER ARCHITECTURE (lambda inventory, 21 models):
/// ```text
///   qwen35     n=8   zero ambiguity — every stem has exactly one fitting formula
///   qwen2      n=7   unambiguous once `heads*head_dim` is read as `hidden`
///   qwen3      n=3   same
///   qwen3moe   n=2   same
///   qwen35moe  n=1   NOT DETERMINED — see below
/// ```
///
/// `heads*head_dim` and `hidden` are the same number BY CONSTRUCTION when
/// head_dim is derived as hidden/heads, so a tie between them is definitional,
/// not evidence of a choice.
///
/// REFUSE, NEVER DEFAULT. An architecture absent from this table returns
/// `Unknown`, and the caller reports that rather than falling back to
/// `hidden`. Falling back is exactly `resolve_qtype`'s
/// `.unwrap_or(WeightQuantType::Q4K)` (#3850) one layer up: a plausible
/// default silently applied to something it does not describe.
enum Expect {
    /// (out_dim, in_dim) the metadata says this tensor must have.
    Dims(usize, usize),
    /// This architecture, or this stem within it, has no derived expectation.
    Unknown(String),
    /// Not a tensor this table speaks for.
    NotCovered,
}

/// Metadata-derived quantities, named as the table refers to them.
struct Dims {
    hidden: usize,
    ffn: usize,
    vocab: usize,
    heads_key: usize,
    kv_heads_key: usize,
    qkv_width: usize,
    ts_rank: usize,
    conv_kernel: usize,
    experts: usize,
}

fn expected_dims(m: &Model, arch: &str, name: &str) -> Expect {
    let d = Dims {
        hidden: m.hidden,
        ffn: m.ffn,
        vocab: m.vocab,
        heads_key: m.attn_in,
        kv_heads_key: m.kv_heads_key,
        qkv_width: m.qkv_width,
        ts_rank: m.ts_rank,
        conv_kernel: m.conv_kernel,
        experts: m.experts,
    };
    let stem = name.split('.').nth_back(1).unwrap_or(name);
    match arch {
        "qwen2" => qwen2_dims(&d, stem),
        "qwen3" | "qwen3moe" => qwen3_dims(&d, stem),
        "qwen35" => qwen35_dims(&d, stem),
        // n=1. `attn_qkv` out fits BOTH `2*heads*key_len` and `qkv_width`, and
        // `ffn_*_shexp` fits three different expressions, on the single model
        // available. One model is an anecdote; this refuses until a second
        // qwen35moe exists to separate them.
        "qwen35moe" => Expect::Unknown(
            "qwen35moe: derived from ONE model, so its formulas are coincidences".to_string(),
        ),
        other => Expect::Unknown(format!("architecture {other} is not in the derived table")),
    }
}

fn qwen2_dims(d: &Dims, stem: &str) -> Expect {
    match stem {
        "attn_q" | "attn_output" => Expect::Dims(d.hidden, d.hidden),
        "attn_k" | "attn_v" => Expect::Dims(d.kv_heads_key, d.hidden),
        "ffn_gate" | "ffn_up" => Expect::Dims(d.ffn, d.hidden),
        "ffn_down" => Expect::Dims(d.hidden, d.ffn),
        "token_embd" | "output" => Expect::Dims(d.vocab, d.hidden),
        _ => Expect::NotCovered,
    }
}

fn qwen3_dims(d: &Dims, stem: &str) -> Expect {
    match stem {
        "attn_q" => Expect::Dims(d.heads_key, d.hidden),
        "attn_output" => Expect::Dims(d.hidden, d.heads_key),
        "attn_k" | "attn_v" => Expect::Dims(d.kv_heads_key, d.hidden),
        "ffn_gate" | "ffn_up" => Expect::Dims(d.ffn, d.hidden),
        "ffn_down" => Expect::Dims(d.hidden, d.ffn),
        "ffn_gate_inp" => Expect::Dims(d.experts, d.hidden),
        "token_embd" | "output" => Expect::Dims(d.vocab, d.hidden),
        _ => Expect::NotCovered,
    }
}

fn qwen35_dims(d: &Dims, stem: &str) -> Expect {
    match stem {
        "attn_q" => Expect::Dims(2 * d.heads_key, d.hidden),
        "attn_gate" => Expect::Dims(d.heads_key, d.hidden),
        "attn_output" | "ssm_out" => Expect::Dims(d.hidden, d.heads_key),
        "attn_k" | "attn_v" => Expect::Dims(d.kv_heads_key, d.hidden),
        "attn_qkv" => Expect::Dims(d.qkv_width, d.hidden),
        "ssm_conv1d" => Expect::Dims(d.qkv_width, d.conv_kernel),
        "ssm_alpha" | "ssm_beta" => Expect::Dims(d.ts_rank, d.hidden),
        "ffn_gate" | "ffn_up" => Expect::Dims(d.ffn, d.hidden),
        "ffn_down" => Expect::Dims(d.hidden, d.ffn),
        "token_embd" | "output" => Expect::Dims(d.vocab, d.hidden),
        _ => Expect::NotCovered,
    }
}

/// `enforce_matmul_contract` on every 2D projection the metadata can speak for.
fn probe_matmul(m: &Model) -> Vec<String> {
    m.tensors
        .iter()
        .filter(|(_, d)| d.len() == 2)
        .filter_map(|(name, d)| check_one(m, name, d))
        .collect()
}

/// One tensor against its architecture's expectation. An architecture with no
/// derived expectation REFUSES and says so; it never falls back to `hidden`.
fn check_one(m: &Model, name: &str, d: &[usize]) -> Option<String> {
    match expected_dims(m, &m.arch, name) {
        Expect::NotCovered => None,
        Expect::Unknown(why) => Some(format!(
            "NO EXPECTATION for {name}: {why} — refusing rather than assuming"
        )),
        Expect::Dims(eo, ei) => {
            let shape = vec![d[1], d[0]];
            fired(|| {
                aprender::format::layout_contract::enforce_matmul_contract(name, &shape, eo, ei);
            })
            .map(|why| format!("enforce_matmul_contract {name}: {why}"))
        }
    }
}

/// `validate_ffn_shape_symmetry` on layer 0.
fn probe_ffn_symmetry(m: &Model) -> Vec<String> {
    let (Some(g), Some(u), Some(dn)) = (
        m.row_major("blk.0.ffn_gate.weight"),
        m.row_major("blk.0.ffn_up.weight"),
        m.row_major("blk.0.ffn_down.weight"),
    ) else {
        return Vec::new();
    };
    aprender::format::layout_contract::validate_ffn_shape_symmetry(&g, &u, &dn)
        .err()
        .map(|e| format!("validate_ffn_shape_symmetry: {e}"))
        .into_iter()
        .collect()
}

/// `enforce_load_contract`.
///
/// The registry is keyed on APR names (`lm_head.weight`,
/// `model.layers.{n}.self_attn.q_proj.weight`), NOT GGUF names, and exactly ONE
/// of its twelve contracts is `is_critical: true` — `lm_head.weight`, the
/// GH-202 root cause. It validates only critical ones, so it is a no-op for
/// every other name and for every GGUF name.
///
/// Feeding it GGUF names (which is what I did first) makes its silence vacuous:
/// it returned Ok for a deliberately corrupted embedding. The only way to
/// exercise it is the APR name with the APR shape, which for lm_head is
/// [vocab, hidden] — the GGUF side is [hidden, vocab] and is transposed at
/// import. Note 11 of 21 models tie their embeddings and have no
/// `output.weight` at all, so this guard is silent for them by construction.
fn probe_load_contract(m: &Model) -> Vec<String> {
    let Some(apr_shape) = m.row_major("output.weight") else {
        return Vec::new();
    };
    aprender::format::layout_contract::enforce_load_contract(
        "lm_head.weight",
        &apr_shape,
        m.vocab,
        m.hidden,
    )
    .err()
    .map(|e| format!("enforce_load_contract lm_head.weight: {e}"))
    .into_iter()
    .collect()
}

fn report(m: &Model, fires: &[String]) {
    if fires.is_empty() {
        println!("  [clean] {}", m.name);
        return;
    }
    println!("  [FIRES] {} ({} violations)", m.name, fires.len());
    for f in fires.iter().take(4) {
        println!("       {f}");
    }
}

fn main() {
    let models = read_models(std::io::stdin().lock());
    let mut total = 0usize;
    for m in &models {
        let mut fires = probe_embedding(m);
        fires.extend(probe_matmul(m));
        fires.extend(probe_ffn_symmetry(m));
        fires.extend(probe_load_contract(m));
        total += fires.len();
        report(m, &fires);
    }
    println!(
        "\n  models probed: {}  total guard fires: {total}",
        models.len()
    );
}
