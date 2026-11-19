//! Base-format tests for `apr finetune` (PMAT-7241, FALSIFY-APR-RECIPE-006).

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
    let (h, inter, v) = (8usize, 16usize, 16usize);
    let mut t = BTreeMap::new();
    let mut put = |name: &str, shape: Vec<usize>| {
        let n: usize = shape.iter().product();
        let data = (0..n).map(|i| ((i % 7) as f32 - 3.0) * 0.01).collect();
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

const TINY_CONFIG: &str = r#"{"architectures":["LlamaForCausalLM"],"model_type":"llama",
"hidden_size":8,"intermediate_size":16,"num_hidden_layers":1,"num_attention_heads":2,
"num_key_value_heads":2,"vocab_size":16,"max_position_embeddings":64,"rms_norm_eps":1e-6,
"rope_theta":10000.0}"#;

fn write_tiny_safetensors(dir: &Path, file: &str) -> PathBuf {
    let path = dir.join(file);
    aprender::serialization::safetensors::save_safetensors(&path, &tiny_llama_tensors())
        .expect("save safetensors");
    std::fs::write(dir.join("config.json"), TINY_CONFIG).expect("config");
    path
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

    // The same source twice gives the same APR bytes: the base is deterministic.
    let again = prepare_base(&st).expect("again");
    assert_eq!(
        std::fs::read(prepared.path()).expect("read"),
        std::fs::read(again.path()).expect("read")
    );
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
