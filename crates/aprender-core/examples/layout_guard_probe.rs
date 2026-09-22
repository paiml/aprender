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
use std::io::BufRead as _;
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
    tensors: Vec<(String, Vec<usize>)>,
}

fn main() {
    let mut models: Vec<Model> = Vec::new();
    for line in std::io::stdin().lock().lines().map_while(Result::ok) {
        let p: Vec<&str> = line.trim().split('|').collect();
        match p.as_slice() {
            ["M", name, h, f, v, a] => models.push(Model {
                name: (*name).to_string(),
                hidden: h.parse().unwrap_or(0),
                ffn: f.parse().unwrap_or(0),
                vocab: v.parse().unwrap_or(0),
                attn_in: a.parse().unwrap_or(0),
                tensors: Vec::new(),
            }),
            ["T", name, dims] => {
                if let Some(m) = models.last_mut() {
                    let d: Vec<usize> = dims.split(',').filter_map(|x| x.parse().ok()).collect();
                    m.tensors.push(((*name).to_string(), d));
                }
            }
            _ => {}
        }
    }

    let mut total_fires = 0usize;
    for m in &models {
        let mut fires: Vec<String> = Vec::new();

        // --- enforce_embedding_contract: tensor element count vs metadata dims
        if let Some((_, d)) = m.tensors.iter().find(|(n, _)| n == "token_embd.weight") {
            let elems: usize = d.iter().product();
            if let Some(why) = fired(|| {
                aprender::format::layout_contract::enforce_embedding_contract(
                    elems, m.vocab, m.hidden,
                );
            }) {
                fires.push(format!("enforce_embedding_contract: {why}"));
            }
        }

        // --- enforce_matmul_contract on every 2D projection, expected dims
        // from METADATA, actual from the tensor table. `apr tensors` reports
        // `ne` (reversed), so [in, out] there is [out, in] here.
        for (name, d) in &m.tensors {
            if d.len() != 2 {
                continue;
            }
            let (in_d, out_d) = (d[0], d[1]);
            let expect = |o: usize, i: usize| (o, i);
            let e = if name.ends_with("ffn_gate.weight") || name.ends_with("ffn_up.weight") {
                Some(expect(m.ffn, m.hidden))
            } else if name.ends_with("ffn_down.weight") {
                Some(expect(m.hidden, m.ffn))
            } else if name.ends_with("attn_output.weight") {
                Some(expect(m.hidden, m.attn_in))
            } else {
                None
            };
            if let Some((eo, ei)) = e {
                if let Some(why) = fired(|| {
                    aprender::format::layout_contract::enforce_matmul_contract(
                        name,
                        &[out_d, in_d],
                        eo,
                        ei,
                    );
                }) {
                    fires.push(format!("enforce_matmul_contract {name}: {why}"));
                }
            }
        }

        // --- validate_ffn_shape_symmetry on layer 0
        let get = |suffix: &str| {
            m.tensors
                .iter()
                .find(|(n, _)| n == &format!("blk.0.{suffix}.weight"))
                .map(|(_, d)| vec![d[1], d[0]])
        };
        if let (Some(g), Some(u), Some(dn)) = (get("ffn_gate"), get("ffn_up"), get("ffn_down")) {
            if let Err(e) =
                aprender::format::layout_contract::validate_ffn_shape_symmetry(&g, &u, &dn)
            {
                fires.push(format!("validate_ffn_shape_symmetry: {e}"));
            }
        }

        // --- enforce_load_contract
        //
        // The registry is keyed on APR names (`lm_head.weight`,
        // `model.layers.{n}.self_attn.q_proj.weight`), NOT GGUF names, and
        // exactly ONE of its twelve contracts is `is_critical: true` —
        // `lm_head.weight`, the GH-202 root cause. `enforce_load_contract`
        // validates only critical ones, so it is a no-op for every other name
        // and for every GGUF name.
        //
        // Feeding it GGUF names (which is what I did first) makes its silence
        // vacuous: it returned Ok for a deliberately corrupted embedding. The
        // only way to exercise it is the APR name with the APR shape, which for
        // lm_head is [vocab, hidden] (the GGUF side is [hidden, vocab] and is
        // transposed at import).
        if let Some((_, d)) = m.tensors.iter().find(|(n, _)| n == "output.weight") {
            let apr_shape = vec![d[1], d[0]];
            if let Err(e) = aprender::format::layout_contract::enforce_load_contract(
                "lm_head.weight",
                &apr_shape,
                m.vocab,
                m.hidden,
            ) {
                fires.push(format!("enforce_load_contract lm_head.weight: {e}"));
            }
        }

        total_fires += fires.len();
        if fires.is_empty() {
            println!("  [clean] {}", m.name);
        } else {
            println!("  [FIRES] {} ({} violations)", m.name, fires.len());
            for f in fires.iter().take(4) {
                println!("       {f}");
            }
        }
    }
    println!(
        "\n  models probed: {}  total guard fires: {total_fires}",
        models.len()
    );
}
