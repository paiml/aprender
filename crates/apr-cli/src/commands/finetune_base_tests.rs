//! Base-format tests for `apr finetune` (PMAT-7241, FALSIFY-APR-RECIPE-011).

use super::*;
use std::collections::BTreeMap;

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("apr-ft-base-{tag}-{}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir).expect("scratch dir");
    dir
}

/// A GGUF v3 file with no metadata and one tensor of `dtype` over `dims`.
fn gguf_bytes(dtype: u32, dims: &[u64], data_len: usize) -> Vec<u8> {
    let mut b = b"GGUF".to_vec();
    b.extend_from_slice(&3u32.to_le_bytes());
    b.extend_from_slice(&1u64.to_le_bytes());
    b.extend_from_slice(&0u64.to_le_bytes());
    let name = b"blk.0.attn_q.weight";
    b.extend_from_slice(&(name.len() as u64).to_le_bytes());
    b.extend_from_slice(name);
    b.extend_from_slice(&(dims.len() as u32).to_le_bytes());
    for d in dims {
        b.extend_from_slice(&d.to_le_bytes());
    }
    b.extend_from_slice(&dtype.to_le_bytes());
    b.extend_from_slice(&0u64.to_le_bytes());
    while b.len() % 32 != 0 {
        b.push(0);
    }
    b.extend(std::iter::repeat(0u8).take(data_len));
    b
}

fn tiny_llama_tensors() -> BTreeMap<String, (Vec<f32>, Vec<usize>)> {
    let (h, inter, v) = (HIDDEN, INTER, VOCAB);
    let mut t = BTreeMap::new();
    let mut put = |name: &str, shape: Vec<usize>| {
        let n: usize = shape.iter().product();
        let data = (0..n)
            .map(|i| ((i * 31 + name.len()) % 17) as f32 * 0.01 - 0.08)
            .collect();
        t.insert(name.to_string(), (data, shape));
    };
    put("model.embed_tokens.weight", vec![v, h]);
    for p in ["q", "k", "v", "o"] {
        put(
            &format!("model.layers.0.self_attn.{p}_proj.weight"),
            vec![h, h],
        );
    }
    put("model.layers.0.mlp.gate_proj.weight", vec![inter, h]);
    put("model.layers.0.mlp.up_proj.weight", vec![inter, h]);
    put("model.layers.0.mlp.down_proj.weight", vec![h, inter]);
    put("model.layers.0.input_layernorm.weight", vec![h]);
    put("model.layers.0.post_attention_layernorm.weight", vec![h]);
    put("model.norm.weight", vec![h]);
    put("lm_head.weight", vec![v, h]);
    t
}

const HIDDEN: usize = 8;
const INTER: usize = 16;
/// 256 byte-level tokens plus `<s>`, `</s>` and `<pad>`.
const VOCAB: usize = 259;

const TINY_CONFIG: &str = r#"{"architectures":["LlamaForCausalLM"],"model_type":"llama",
"hidden_size":8,"intermediate_size":16,"num_hidden_layers":1,"num_attention_heads":2,
"num_key_value_heads":2,"vocab_size":259,"max_position_embeddings":512,"rms_norm_eps":1e-6,
"rope_theta":10000.0}"#;

/// The byte-level vocabulary, in id order, so a prompt of any text encodes inside the model.
fn byte_level_tokens() -> Vec<String> {
    let (encoder, _) = aprender::text::bpe::bytes_to_unicode();
    let mut tokens: Vec<String> = (0..=255u8).map(|b| encoder[&b].to_string()).collect();
    tokens.extend(["<s>", "</s>", "<pad>"].map(String::from));
    assert_eq!(tokens.len(), VOCAB);
    tokens
}

fn tokenizer_json() -> String {
    let vocab: serde_json::Map<String, serde_json::Value> = byte_level_tokens()
        .into_iter()
        .enumerate()
        .map(|(id, tok)| (tok, serde_json::json!(id)))
        .collect();
    serde_json::json!({"model": {"type": "BPE", "vocab": vocab, "merges": []}, "added_tokens": []})
        .to_string()
}

fn write_tiny_safetensors(dir: &Path, file: &str) -> PathBuf {
    let path = dir.join(file);
    aprender::serialization::safetensors::save_safetensors(&path, &tiny_llama_tensors())
        .expect("save safetensors");
    std::fs::write(dir.join("config.json"), TINY_CONFIG).expect("config");
    std::fs::write(dir.join("tokenizer.json"), tokenizer_json()).expect("tokenizer");
    path
}

const GGUF_NAMES: [(&str, &str); 12] = [
    ("model.embed_tokens.weight", "token_embd.weight"),
    (
        "model.layers.0.self_attn.q_proj.weight",
        "blk.0.attn_q.weight",
    ),
    (
        "model.layers.0.self_attn.k_proj.weight",
        "blk.0.attn_k.weight",
    ),
    (
        "model.layers.0.self_attn.v_proj.weight",
        "blk.0.attn_v.weight",
    ),
    (
        "model.layers.0.self_attn.o_proj.weight",
        "blk.0.attn_output.weight",
    ),
    (
        "model.layers.0.mlp.gate_proj.weight",
        "blk.0.ffn_gate.weight",
    ),
    ("model.layers.0.mlp.up_proj.weight", "blk.0.ffn_up.weight"),
    (
        "model.layers.0.mlp.down_proj.weight",
        "blk.0.ffn_down.weight",
    ),
    (
        "model.layers.0.input_layernorm.weight",
        "blk.0.attn_norm.weight",
    ),
    (
        "model.layers.0.post_attention_layernorm.weight",
        "blk.0.ffn_norm.weight",
    ),
    ("model.norm.weight", "output_norm.weight"),
    ("lm_head.weight", "output.weight"),
];

fn gguf_str(b: &mut Vec<u8>, s: &str) {
    b.extend_from_slice(&(s.len() as u64).to_le_bytes());
    b.extend_from_slice(s.as_bytes());
}

fn gguf_kv_u32(b: &mut Vec<u8>, key: &str, v: u32) {
    gguf_str(b, key);
    b.extend_from_slice(&4u32.to_le_bytes());
    b.extend_from_slice(&v.to_le_bytes());
}

fn gguf_kv_f32(b: &mut Vec<u8>, key: &str, v: f32) {
    gguf_str(b, key);
    b.extend_from_slice(&6u32.to_le_bytes());
    b.extend_from_slice(&v.to_le_bytes());
}

fn gguf_kv_str(b: &mut Vec<u8>, key: &str, v: &str) {
    gguf_str(b, key);
    b.extend_from_slice(&8u32.to_le_bytes());
    gguf_str(b, v);
}

fn gguf_metadata(b: &mut Vec<u8>) {
    gguf_kv_str(b, "general.architecture", "llama");
    gguf_kv_u32(b, "llama.embedding_length", HIDDEN as u32);
    gguf_kv_u32(b, "llama.block_count", 1);
    gguf_kv_u32(b, "llama.attention.head_count", 2);
    gguf_kv_u32(b, "llama.attention.head_count_kv", 2);
    gguf_kv_u32(b, "llama.feed_forward_length", INTER as u32);
    gguf_kv_u32(b, "llama.context_length", 512);
    gguf_kv_f32(b, "llama.rope.freq_base", 10000.0);
    gguf_kv_f32(b, "llama.attention.layer_norm_rms_epsilon", 1e-6);
    gguf_str(b, "tokenizer.ggml.tokens");
    b.extend_from_slice(&9u32.to_le_bytes());
    b.extend_from_slice(&8u32.to_le_bytes());
    let tokens = byte_level_tokens();
    b.extend_from_slice(&(tokens.len() as u64).to_le_bytes());
    for t in &tokens {
        gguf_str(b, t);
    }
}

/// The tiny llama as an exact-F32 GGUF: GGUF tensor names, and dims in GGUF order
/// (`[ne0, ne1]`, innermost first), i.e. the REVERSE of the row-major shape.
fn tiny_gguf_bytes() -> Vec<u8> {
    let tensors = tiny_llama_tensors();
    let entries: Vec<(&str, &(Vec<f32>, Vec<usize>))> = GGUF_NAMES
        .iter()
        .map(|(hf, g)| (*g, &tensors[*hf]))
        .collect();
    let mut b = b"GGUF".to_vec();
    b.extend_from_slice(&3u32.to_le_bytes());
    b.extend_from_slice(&(entries.len() as u64).to_le_bytes());
    b.extend_from_slice(&10u64.to_le_bytes());
    gguf_metadata(&mut b);
    let mut offset = 0u64;
    let mut offsets = Vec::new();
    for (name, (data, shape)) in &entries {
        gguf_str(&mut b, name);
        b.extend_from_slice(&(shape.len() as u32).to_le_bytes());
        for d in shape.iter().rev() {
            b.extend_from_slice(&(*d as u64).to_le_bytes());
        }
        b.extend_from_slice(&0u32.to_le_bytes());
        b.extend_from_slice(&offset.to_le_bytes());
        offsets.push(offset);
        offset += (data.len() * 4).next_multiple_of(32) as u64;
    }
    while b.len() % 32 != 0 {
        b.push(0);
    }
    for (_, (data, _)) in &entries {
        for v in data {
            b.extend_from_slice(&v.to_le_bytes());
        }
        while b.len() % 32 != 0 {
            b.push(0);
        }
    }
    b
}

fn read_apr(path: &Path) -> aprender::serialization::apr::AprReader {
    aprender::serialization::apr::AprReader::open(path).expect("open apr")
}

#[test]
fn a_gguf_base_lands_row_major_with_data_unchanged_after_import() {
    let dir = scratch("gguf-layout");
    let gguf = dir.join("base.gguf");
    std::fs::write(&gguf, tiny_gguf_bytes()).expect("write gguf");
    let st = write_tiny_safetensors(&dir, "base.safetensors");

    let from_gguf = prepare_base(&gguf).expect("gguf base prepares");
    assert_eq!(from_gguf.format, BaseFormat::Gguf);
    let from_st = prepare_base(&st).expect("st base prepares");
    let (g, s) = (read_apr(from_gguf.path()), read_apr(from_st.path()));

    // The row-major shapes the safetensors source declares. GGUF stored them reversed;
    // skipping the reversal would leave gate_proj as [8, 16].
    let shape_of = |r: &aprender::serialization::apr::AprReader, name: &str| {
        r.tensors
            .iter()
            .find(|t| t.name == name)
            .unwrap_or_else(|| {
                panic!(
                    "{name} missing; have {:?}",
                    r.tensors.iter().map(|t| &t.name).collect::<Vec<_>>()
                )
            })
            .shape
            .clone()
    };
    // The transformer loader accepts the GGUF tensor names as imported (`token_embd.weight`).
    let gate = "blk.0.ffn_gate.weight";
    let down = "blk.0.ffn_down.weight";
    for (name, shape) in [(gate, [INTER, HIDDEN]), (down, [HIDDEN, INTER])] {
        assert_eq!(
            shape_of(&g, name),
            shape,
            "{name} must be row-major [out, in]"
        );
    }
    assert_eq!(g.tensors.len(), s.tensors.len(), "same tensor set");
    // Every tensor: the safetensors import's shape and values, under the GGUF name.
    for (hf, gg) in GGUF_NAMES.iter() {
        assert_eq!(shape_of(&g, gg), shape_of(&s, hf), "{gg} shape");
        let gv = g.read_tensor_as_f32(gg).expect("gguf values");
        let sv = s.read_tensor_as_f32(hf).expect("st values");
        assert_eq!(gv, sv, "{gg} values must be unchanged by import");
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn format_comes_from_magic_bytes_not_the_extension() {
    let dir = scratch("magic");
    let st = write_tiny_safetensors(&dir, "model.bin");
    assert_eq!(
        detect_base_format(&st).expect("st"),
        BaseFormat::SafeTensors
    );

    let gguf = dir.join("model.safetensors");
    std::fs::write(&gguf, gguf_bytes(0, &[4], 16)).expect("write");
    assert_eq!(detect_base_format(&gguf).expect("gguf"), BaseFormat::Gguf);

    let apr = dir.join("model.gguf");
    let mut bytes = b"APR\0".to_vec();
    bytes.resize(128, 0);
    std::fs::write(&apr, &bytes).expect("write");
    assert_eq!(detect_base_format(&apr).expect("apr"), BaseFormat::Apr);

    let v1 = dir.join("legacy");
    let mut bytes = b"APRN".to_vec();
    bytes.resize(128, 0);
    std::fs::write(&v1, &bytes).expect("write");
    assert_eq!(detect_base_format(&v1).expect("apr v1"), BaseFormat::Apr);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn unsupported_or_corrupt_files_are_refused_with_a_clear_error() {
    let dir = scratch("corrupt");
    let junk = dir.join("junk.apr");
    std::fs::write(&junk, b"this is not a model file at all").expect("write");
    let msg = detect_base_format(&junk).expect_err("junk").to_string();
    assert!(
        msg.contains("junk.apr") && msg.contains("unsupported or corrupt"),
        "{msg}"
    );

    let short = dir.join("short.gguf");
    std::fs::write(&short, b"GG").expect("write");
    assert!(detect_base_format(&short).is_err());

    let empty = dir.join("empty.safetensors");
    std::fs::write(&empty, b"").expect("write");
    assert!(detect_base_format(&empty).is_err());

    // A SafeTensors header that claims more bytes than the file holds is truncated.
    let trunc = dir.join("trunc.safetensors");
    let mut bytes = 1_000_000u64.to_le_bytes().to_vec();
    bytes.extend_from_slice(b"{\"a\":1}");
    std::fs::write(&trunc, &bytes).expect("write");
    assert!(detect_base_format(&trunc).is_err());

    let missing = dir.join("nope.apr");
    assert!(detect_base_format(&missing).is_err());

    // Corrupt GGUF: right magic, nothing else.
    let bad = dir.join("bad.gguf");
    let mut bytes = b"GGUF".to_vec();
    bytes.resize(30, 0xFF);
    std::fs::write(&bad, &bytes).expect("write");
    let msg = prepare_base(&bad).err().expect("refused").to_string();
    assert!(msg.contains("bad.gguf") && msg.contains("corrupt"), "{msg}");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_quantized_gguf_is_refused_with_its_reason_and_an_exact_one_is_not() {
    let dir = scratch("quant");
    let q = dir.join("q4k.gguf");
    std::fs::write(&q, gguf_bytes(12, &[256], 144)).expect("write");
    let msg = preflight_base(&q).expect_err("quantized").to_string();
    assert!(
        msg.contains("quantized") && msg.contains("silently"),
        "{msg}"
    );
    assert!(msg.contains("blk.0.attn_q.weight"), "{msg}");

    for (dtype, bytes) in [(0u32, 16usize), (1, 8), (30, 8)] {
        let ok = dir.join(format!("exact{dtype}.gguf"));
        std::fs::write(&ok, gguf_bytes(dtype, &[4], bytes)).expect("write");
        assert_eq!(preflight_base(&ok).expect("exact"), BaseFormat::Gguf);
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_safetensors_base_is_imported_to_row_major_apr_and_the_format_is_named() {
    let dir = scratch("st-import");
    let st = write_tiny_safetensors(&dir, "base.safetensors");
    let prepared = prepare_base(&st).expect("safetensors base prepares");
    assert_eq!(prepared.format, BaseFormat::SafeTensors);
    assert!(crate::commands::model_config::is_apr_file(prepared.path()));
    assert_ne!(prepared.path(), st.as_path());

    // The same source twice gives the same tensors: the base is deterministic.
    let again = prepare_base(&st).expect("again");
    let (a, b) = (read_apr(prepared.path()), read_apr(again.path()));
    assert_eq!(a.tensors.len(), 12);
    for t in &a.tensors {
        assert_eq!(
            a.read_tensor_as_f32(&t.name).expect("a"),
            b.read_tensor_as_f32(&t.name).expect("b"),
            "{} differs between two imports",
            t.name
        );
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn an_apr_base_is_used_in_place() {
    let dir = scratch("apr-inplace");
    let st = write_tiny_safetensors(&dir, "base.safetensors");
    let converted = prepare_base(&st).expect("convert");
    let apr = dir.join("base.apr");
    std::fs::copy(converted.path(), &apr).expect("copy");
    let prepared = prepare_base(&apr).expect("apr prepares");
    assert_eq!(prepared.format, BaseFormat::Apr);
    assert_eq!(prepared.path(), apr.as_path());
    std::fs::remove_dir_all(&dir).ok();
}

const E2E_TRAIN: &str = "\
{\"instruction\":\"say a\",\"response\":\"a a\"}
{\"instruction\":\"say b\",\"response\":\"b b\"}
{\"instruction\":\"say c\",\"response\":\"c c\"}
{\"instruction\":\"say d\",\"response\":\"d d\"}
{\"instruction\":\"say e\",\"response\":\"e e\"}
{\"instruction\":\"say f\",\"response\":\"f f\"}
";

fn e2e_recipe(base: &str, out: &str) -> String {
    format!(
        "recipe: apr-recipe-v1\nname: base-formats\nbase: {base}\nmethod: lora\n\
         data:\n  train: train.jsonl\neval:\n  heldout: heldout.jsonl\nseed: 42\n\
         training:\n  rank: 4\n  epochs: 1\n  learning_rate: 0.01\n  max_seq_len: 256\n\
         output:\n  dir: {out}\n"
    )
}

fn e2e_flags() -> crate::commands::finetune::RecipeRunFlags<'static> {
    crate::commands::finetune::RecipeRunFlags {
        vram_gb: 16.0,
        plan_only: false,
        output_path: None,
        model_size: None,
        checkpoint_format: "apr,safetensors",
        gpus: None,
        gpu_backend: "cpu",
        wait_gpu: 0,
        json_output: true,
        experimental_mps: false,
        gpu_share: 50,
    }
}

type AdapterWeights = BTreeMap<String, (Vec<usize>, Vec<f32>)>;

/// Adapter tensors by name. Compared as values: the SafeTensors header's JSON key order is
/// not stable between two writes of the same weights, so the raw bytes are not the claim.
fn read_adapter(path: &Path) -> AdapterWeights {
    use aprender::serialization::safetensors::{extract_tensor, load_safetensors};
    let (meta, raw) = load_safetensors(path).expect("load adapter");
    meta.iter()
        .map(|(name, m)| {
            let values = extract_tensor(&raw, m).expect("extract");
            (name.clone(), (m.shape.clone(), values))
        })
        .collect()
}

/// What a run leaves behind that does not depend on the clock: the LoRA adapter
/// weights and the loss/hash fields of the checkpoint metadata.
fn run_fingerprint(dir: &Path, base: &str, tag: &str) -> (AdapterWeights, serde_json::Value) {
    let out = format!("out-{tag}");
    let recipe = dir.join(format!("{tag}.yaml"));
    std::fs::write(&recipe, e2e_recipe(base, &out)).expect("recipe");
    crate::commands::finetune::run_from_recipe(&recipe, &e2e_flags())
        .unwrap_or_else(|e| panic!("{tag} run failed: {e}"));
    let ckpt = dir.join(&out).join("checkpoints").join("epoch-0");
    let weights = read_adapter(&ckpt.join("model.safetensors"));
    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(ckpt.join("metadata.json")).expect("metadata"))
            .expect("metadata json");
    let keep = ["train_loss", "val_loss", "lora_rank", "data_hash"];
    let stable = keep
        .iter()
        .map(|k| (k.to_string(), meta[*k].clone()))
        .collect();
    (weights, serde_json::Value::Object(stable))
}

/// FALSIFY-APR-RECIPE-011, end to end: the same tiny model as SafeTensors, as the APR
/// that import produces, and as an exact-F32 GGUF, run through one recipe shape and
/// one seed. The run seed is the trainer's own fixed 42 (the recipe `seed` is recorded,
/// not applied), so it is the same for all three.
#[test]
fn one_recipe_trains_the_same_model_from_safetensors_apr_and_gguf() {
    let dir = scratch("e2e");
    let st = write_tiny_safetensors(&dir, "base.safetensors");
    std::fs::write(dir.join("base.gguf"), tiny_gguf_bytes()).expect("gguf");
    std::fs::copy(
        prepare_base(&st).expect("convert").path(),
        dir.join("base.apr"),
    )
    .expect("apr copy");
    std::fs::write(dir.join("train.jsonl"), E2E_TRAIN).expect("train");
    std::fs::write(dir.join("heldout.jsonl"), E2E_TRAIN).expect("heldout");

    let (w_st, m_st) = run_fingerprint(&dir, "base.safetensors", "st");
    let (w_apr, m_apr) = run_fingerprint(&dir, "base.apr", "apr");
    let (w_apr2, m_apr2) = run_fingerprint(&dir, "base.apr", "apr2");
    let (w_gguf, m_gguf) = run_fingerprint(&dir, "base.gguf", "gguf");

    assert!(!w_st.is_empty());
    assert!(
        w_st.values().any(|(_, v)| v.iter().any(|x| *x != 0.0)),
        "the adapter is all zeros: no training happened"
    );
    let loss = m_st["train_loss"].as_f64().expect("train_loss");
    assert!(loss.is_finite() && loss > 0.0, "loss {loss}");
    // The run is deterministic: the same APR twice is byte-identical.
    assert_eq!(w_apr, w_apr2, "apr base run is not repeatable");
    assert_eq!(m_apr, m_apr2);
    // .apr and .safetensors are the same model after import: identical results.
    assert_eq!(w_st, w_apr, ".safetensors and .apr adapters differ");
    assert_eq!(m_st, m_apr, ".safetensors and .apr metrics differ");
    // The GGUF carries the same tensors (row-major after import) and the same byte-level
    // vocabulary, so it trains to the same adapter.
    assert_eq!(w_st, w_gguf, ".safetensors and .gguf adapters differ");
    assert_eq!(m_st, m_gguf, ".safetensors and .gguf metrics differ");
    std::fs::remove_dir_all(&dir).ok();
}
