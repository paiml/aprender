// APR-EMBED-001 EG-6b: verbatim GGUF header passthrough.
//
// A raw GGUF import of an architecture whose forward stays in realizar's GGUF
// path (EmbeddingGemma 2) records every GGUF header key, typed, under the APR
// custom key `gguf.kv`. Export writes those keys back unchanged and copies each
// tensor's bytes under its own name and dtype, so GGUF → APR → GGUF keeps the
// names, shapes, dtypes and header keys of the source file.

/// APR `custom` key holding the source GGUF header, typed.
const GGUF_KV_CUSTOM_KEY: &str = "gguf.kv";

/// Tokenizer arrays the import already stores under `tokenizer.*` custom keys.
/// `gguf.kv` records only a reference and the length; export rebuilds them.
/// Columns: GGUF key, APR custom key, value tag.
const GGUF_KV_FROM_TOKENIZER: [(&str, &str, &str); 4] = [
    ("tokenizer.ggml.tokens", "tokenizer.vocabulary", "arr_str"),
    ("tokenizer.ggml.merges", "tokenizer.merges", "arr_str"),
    ("tokenizer.ggml.scores", "tokenizer.scores", "arr_f32"),
    (
        "tokenizer.ggml.token_type",
        "tokenizer.token_type",
        "arr_i32",
    ),
];

fn json_f32(x: f32) -> Option<serde_json::Value> {
    serde_json::Number::from_f64(f64::from(x)).map(serde_json::Value::Number)
}

/// Encode one GGUF value as `{"t": tag, "v": value}`. `None` for a NaN or
/// infinite float, which JSON cannot hold.
fn gguf_value_to_json(value: &crate::format::gguf::GgufValue) -> Option<serde_json::Value> {
    use crate::format::gguf::GgufValue;
    use serde_json::Value;
    let (tag, v) = match value {
        GgufValue::Uint8(x) => ("u8", Value::from(*x)),
        GgufValue::Int8(x) => ("i8", Value::from(*x)),
        GgufValue::Uint16(x) => ("u16", Value::from(*x)),
        GgufValue::Int16(x) => ("i16", Value::from(*x)),
        GgufValue::Uint32(x) => ("u32", Value::from(*x)),
        GgufValue::Int32(x) => ("i32", Value::from(*x)),
        GgufValue::Float32(x) => ("f32", json_f32(*x)?),
        GgufValue::Bool(x) => ("bool", Value::from(*x)),
        GgufValue::String(x) => ("str", Value::from(x.as_str())),
        GgufValue::Uint64(x) => ("u64", Value::from(*x)),
        GgufValue::Int64(x) => ("i64", Value::from(*x)),
        GgufValue::Float64(x) => ("f64", Value::Number(serde_json::Number::from_f64(*x)?)),
        GgufValue::ArrayUint32(a) => ("arr_u32", Value::from(a.clone())),
        GgufValue::ArrayInt32(a) => ("arr_i32", Value::from(a.clone())),
        GgufValue::ArrayFloat32(a) => (
            "arr_f32",
            Value::Array(a.iter().map(|&x| json_f32(x)).collect::<Option<_>>()?),
        ),
        GgufValue::ArrayString(a) => ("arr_str", Value::from(a.clone())),
        GgufValue::ArrayBool(a) => ("arr_bool", Value::from(a.clone())),
    };
    let mut entry = serde_json::Map::new();
    entry.insert("t".to_string(), Value::from(tag));
    entry.insert("v".to_string(), v);
    Some(Value::Object(entry))
}

fn json_vec<T>(
    v: &serde_json::Value,
    f: impl Fn(&serde_json::Value) -> Option<T>,
) -> Option<Vec<T>> {
    v.as_array()?.iter().map(f).collect()
}

/// Decode a value written by `gguf_value_to_json`. `None` when the tag is
/// unknown or the value does not fit it.
#[allow(clippy::cast_possible_truncation)]
fn gguf_value_from_json(
    tag: &str,
    v: &serde_json::Value,
) -> Option<crate::format::gguf::GgufValue> {
    use crate::format::gguf::GgufValue;
    Some(match tag {
        "u8" => GgufValue::Uint8(u8::try_from(v.as_u64()?).ok()?),
        "i8" => GgufValue::Int8(i8::try_from(v.as_i64()?).ok()?),
        "u16" => GgufValue::Uint16(u16::try_from(v.as_u64()?).ok()?),
        "i16" => GgufValue::Int16(i16::try_from(v.as_i64()?).ok()?),
        "u32" => GgufValue::Uint32(u32::try_from(v.as_u64()?).ok()?),
        "i32" => GgufValue::Int32(i32::try_from(v.as_i64()?).ok()?),
        // An f32 widened to f64 narrows back to the same f32.
        "f32" => GgufValue::Float32(v.as_f64()? as f32),
        "bool" => GgufValue::Bool(v.as_bool()?),
        "str" => GgufValue::String(v.as_str()?.to_string()),
        "u64" => GgufValue::Uint64(v.as_u64()?),
        "i64" => GgufValue::Int64(v.as_i64()?),
        "f64" => GgufValue::Float64(v.as_f64()?),
        "arr_u32" => GgufValue::ArrayUint32(json_vec(v, |x| u32::try_from(x.as_u64()?).ok())?),
        "arr_i32" => GgufValue::ArrayInt32(json_vec(v, |x| i32::try_from(x.as_i64()?).ok())?),
        "arr_f32" => GgufValue::ArrayFloat32(json_vec(v, |x| Some(x.as_f64()? as f32))?),
        "arr_str" => GgufValue::ArrayString(json_vec(v, |x| Some(x.as_str()?.to_string()))?),
        "arr_bool" => GgufValue::ArrayBool(json_vec(v, serde_json::Value::as_bool)?),
        _ => return None,
    })
}

fn gguf_value_len(value: &crate::format::gguf::GgufValue) -> usize {
    use crate::format::gguf::GgufValue;
    match value {
        GgufValue::ArrayUint32(a) => a.len(),
        GgufValue::ArrayInt32(a) => a.len(),
        GgufValue::ArrayFloat32(a) => a.len(),
        GgufValue::ArrayString(a) => a.len(),
        GgufValue::ArrayBool(a) => a.len(),
        _ => 1,
    }
}

/// A by-reference record for a tokenizer array, only when the custom key
/// already holds exactly the source value. Anything else is stored inline.
fn gguf_kv_reference(
    key: &str,
    value: &crate::format::gguf::GgufValue,
    custom: &std::collections::HashMap<String, serde_json::Value>,
) -> Option<serde_json::Value> {
    let (_, custom_key, _) = GGUF_KV_FROM_TOKENIZER.iter().find(|(k, _, _)| *k == key)?;
    let mut entry = serde_json::Map::new();
    entry.insert("t".to_string(), serde_json::Value::from("custom"));
    entry.insert("custom".to_string(), serde_json::Value::from(*custom_key));
    entry.insert(
        "len".to_string(),
        serde_json::Value::from(gguf_value_len(value)),
    );
    let entry = serde_json::Value::Object(entry);
    let rebuilt = gguf_kv_from_tokenizer(key, &entry, custom).ok()?;
    (rebuilt == *value).then_some(entry)
}

/// Record every GGUF header key, typed, under `custom["gguf.kv"]`. A key whose
/// float value JSON cannot hold is left out with a warning.
pub(crate) fn insert_gguf_kv_metadata(
    kv: &BTreeMap<String, crate::format::gguf::GgufValue>,
    custom: &mut std::collections::HashMap<String, serde_json::Value>,
) {
    let mut record = serde_json::Map::new();
    for (key, value) in kv {
        let Some(entry) =
            gguf_kv_reference(key, value, custom).or_else(|| gguf_value_to_json(value))
        else {
            eprintln!("[EG-6] GGUF key '{key}' holds a non-finite float; not recorded");
            continue;
        };
        record.insert(key.clone(), entry);
    }
    custom.insert(
        GGUF_KV_CUSTOM_KEY.to_string(),
        serde_json::Value::Object(record),
    );
}

fn gguf_kv_err(key: &str, why: &str) -> AprenderError {
    AprenderError::FormatError {
        message: format!("APR `{GGUF_KV_CUSTOM_KEY}` entry '{key}' {why}"),
    }
}

/// Rebuild a tokenizer array that `gguf.kv` records by reference.
fn gguf_kv_from_tokenizer(
    key: &str,
    entry: &serde_json::Value,
    custom: &std::collections::HashMap<String, serde_json::Value>,
) -> Result<crate::format::gguf::GgufValue> {
    let (_, custom_key, tag) = GGUF_KV_FROM_TOKENIZER
        .iter()
        .find(|(k, _, _)| *k == key)
        .ok_or_else(|| gguf_kv_err(key, "is not a tokenizer array"))?;
    let source = custom
        .get(*custom_key)
        .ok_or_else(|| gguf_kv_err(key, &format!("needs custom key '{custom_key}', absent")))?;
    let value = gguf_value_from_json(tag, source)
        .ok_or_else(|| gguf_kv_err(key, &format!("cannot decode '{custom_key}' as {tag}")))?;
    let want = entry.get("len").and_then(serde_json::Value::as_u64);
    if want != Some(gguf_value_len(&value) as u64) {
        return Err(gguf_kv_err(
            key,
            &format!(
                "length {want:?} != {} in '{custom_key}'",
                gguf_value_len(&value)
            ),
        ));
    }
    Ok(value)
}

/// The GGUF header recorded at import, or `None` when the APR file has none.
fn gguf_kv_from_custom(
    custom: &std::collections::HashMap<String, serde_json::Value>,
) -> Option<Result<Vec<(String, crate::format::gguf::GgufValue)>>> {
    let record = custom.get(GGUF_KV_CUSTOM_KEY)?.as_object()?;
    Some(
        record
            .iter()
            .map(|(key, entry)| {
                let tag = entry.get("t").and_then(serde_json::Value::as_str);
                let value = match tag {
                    Some("custom") => gguf_kv_from_tokenizer(key, entry, custom)?,
                    Some(tag) => entry
                        .get("v")
                        .and_then(|v| gguf_value_from_json(tag, v))
                        .ok_or_else(|| gguf_kv_err(key, &format!("does not decode as {tag}")))?,
                    None => return Err(gguf_kv_err(key, "has no type tag")),
                };
                Ok((key.clone(), value))
            })
            .collect(),
    )
}

/// Inverse of the raw import's `map_gguf_dtype`.
fn gguf_dtype_for_verbatim(
    name: &str,
    dtype: crate::format::v2::TensorDType,
) -> Result<crate::format::gguf::GgmlType> {
    use crate::format::gguf::GgmlType;
    use crate::format::v2::TensorDType;
    match dtype {
        TensorDType::F32 => Ok(GgmlType::F32),
        TensorDType::F16 => Ok(GgmlType::F16),
        TensorDType::BF16 => Ok(GgmlType::BF16),
        TensorDType::Q4K => Ok(GgmlType::Q4K),
        TensorDType::Q6K => Ok(GgmlType::Q6K),
        other => Err(AprenderError::FormatError {
            message: format!(
                "Tensor '{name}' has dtype {other:?}, which a GGUF raw import never writes; \
                 verbatim GGUF export copies only F32/F16/BF16/Q4_K/Q6_K"
            ),
        }),
    }
}

/// Undo `enforce_import_contract`'s shape rule for a tensor kept under its
/// GGUF name: a 2D shape was reversed unless its contract entry says no
/// transpose; every other shape passed through.
fn gguf_shape_for_verbatim(
    layout: &crate::format::layout_contract::LayoutContract,
    name: &str,
    apr_shape: &[usize],
) -> Vec<u64> {
    let kept = layout
        .get_gguf_contract(name)
        .or_else(|| layout.get_apr_contract(name))
        .is_some_and(|tc| !tc.should_transpose);
    let mut shape: Vec<u64> = apr_shape.iter().map(|&d| d as u64).collect();
    if shape.len() == 2 && !kept {
        shape.reverse();
    }
    shape
}

/// APR-EMBED-001 EG-6b: when the APR file carries `gguf.kv`, write the GGUF
/// back verbatim. `Ok(None)` when it does not, so the caller falls through.
fn try_gguf_kv_verbatim_export(input: &Path, output: &Path) -> Result<Option<ExportReport>> {
    use crate::format::gguf::{export_tensors_to_gguf, GgufTensor};
    use crate::format::v2::AprV2Reader;

    let data = fs::read(input).map_err(|e| AprenderError::FormatError {
        message: format!("Failed to read APR file: {e}"),
    })?;
    let Ok(reader) = AprV2Reader::from_bytes(&data) else {
        return Ok(None);
    };
    let Some(metadata) = gguf_kv_from_custom(&reader.metadata().custom) else {
        return Ok(None);
    };
    let metadata = metadata?;

    let layout = crate::format::layout_contract::contract();
    let mut tensors = Vec::new();
    for name in reader.tensor_names() {
        let entry = reader
            .get_tensor(name)
            .ok_or_else(|| AprenderError::FormatError {
                message: format!("Tensor '{name}' missing from index"),
            })?;
        let bytes = reader
            .get_tensor_data(name)
            .ok_or_else(|| AprenderError::FormatError {
                message: format!("Tensor '{name}' data not found"),
            })?;
        tensors.push(GgufTensor {
            name: name.to_string(),
            shape: gguf_shape_for_verbatim(&layout, name, &entry.shape),
            dtype: gguf_dtype_for_verbatim(name, entry.dtype)?,
            data: bytes.to_vec(),
        });
    }
    eprintln!(
        "[EG-6] Verbatim GGUF export: {} header keys, {} tensors, names and dtypes as imported",
        metadata.len(),
        tensors.len()
    );

    let file = fs::File::create(output).map_err(|e| AprenderError::FormatError {
        message: format!("Failed to create output file: {e}"),
    })?;
    let mut writer = std::io::BufWriter::new(file);
    export_tensors_to_gguf(&mut writer, &tensors, &metadata)?;
    std::io::Write::flush(&mut writer).map_err(|e| AprenderError::FormatError {
        message: format!("Failed to flush output file: {e}"),
    })?;

    Ok(Some(ExportReport {
        original_size: data.len(),
        exported_size: fs::metadata(output).map_or(0, |m| m.len() as usize),
        tensor_count: tensors.len(),
        format: ExportFormat::Gguf,
        quantization: None,
    }))
}

#[cfg(test)]
mod gguf_kv_passthrough_tests {
    use super::*;
    use crate::format::gguf::{
        export_tensors_to_gguf, GgmlType, GgufReader, GgufTensor, GgufValue,
    };

    fn kv(key: &str, value: GgufValue) -> (String, GgufValue) {
        (key.to_string(), value)
    }

    fn header_json(pairs: &[(String, GgufValue)]) -> BTreeMap<String, serde_json::Value> {
        pairs
            .iter()
            .map(|(k, v)| (k.clone(), gguf_value_to_json(v).expect("finite")))
            .collect()
    }

    fn reader_header(reader: &GgufReader) -> BTreeMap<String, serde_json::Value> {
        reader
            .metadata
            .iter()
            .chain(&reader.display_only_metadata)
            .map(|(k, v)| (k.clone(), gguf_value_to_json(v).expect("finite")))
            .collect()
    }

    #[test]
    fn eg6b_every_gguf_value_type_survives_json() {
        let values = [
            GgufValue::Uint8(200),
            GgufValue::Int8(-7),
            GgufValue::Uint16(60_000),
            GgufValue::Int16(-30_000),
            GgufValue::Uint32(4_000_000_000),
            GgufValue::Int32(-2_000_000_000),
            GgufValue::Float32(1e-6),
            GgufValue::Bool(true),
            GgufValue::String("gemma-embedding2".to_string()),
            GgufValue::Uint64(u64::MAX),
            GgufValue::Int64(i64::MIN),
            GgufValue::Float64(0.1),
            GgufValue::ArrayUint32(vec![1, 2]),
            GgufValue::ArrayInt32(vec![-1, 4]),
            GgufValue::ArrayFloat32(vec![0.1, -3.5]),
            GgufValue::ArrayString(vec!["a".to_string(), String::new()]),
            GgufValue::ArrayBool(vec![true, false, true]),
        ];
        for value in values {
            let json = gguf_value_to_json(&value).expect("finite value encodes");
            let tag = json["t"].as_str().expect("tag");
            let back = gguf_value_from_json(tag, &json["v"]).expect("decodes");
            assert_eq!(format!("{back:?}"), format!("{value:?}"));
        }
        assert!(gguf_value_to_json(&GgufValue::Float32(f32::NAN)).is_none());
        assert!(gguf_value_from_json("u8", &serde_json::json!(256)).is_none());
        assert!(gguf_value_from_json("nope", &serde_json::json!(1)).is_none());
    }

    /// GGUF → APR → GGUF on a small EmbeddingGemma 2 file keeps every header
    /// key with its type and value, and every tensor's name, shape, dtype and
    /// bytes, BF16 included.
    #[test]
    fn eg6b_embedding_gemma2_round_trips_gguf_verbatim() {
        let dir = tempfile::tempdir().expect("tempdir");
        let src = dir.path().join("eg2.gguf");
        let apr = dir.path().join("eg2.apr");
        let out = dir.path().join("eg2-rt.gguf");

        let header = vec![
            kv(
                "general.architecture",
                GgufValue::String("gemma-embedding2".into()),
            ),
            kv("general.name", GgufValue::String("tiny".into())),
            kv("gemma-embedding2.block_count", GgufValue::Uint32(1)),
            kv("gemma-embedding2.context_length", GgufValue::Uint32(64)),
            kv("gemma-embedding2.embedding_length", GgufValue::Uint32(8)),
            kv(
                "gemma-embedding2.feed_forward_length",
                GgufValue::Uint32(16),
            ),
            kv(
                "gemma-embedding2.attention.head_count",
                GgufValue::Uint32(2),
            ),
            kv(
                "gemma-embedding2.attention.head_count_kv",
                GgufValue::ArrayInt32(vec![1]),
            ),
            kv(
                "gemma-embedding2.attention.sliding_window_pattern",
                GgufValue::ArrayBool(vec![true]),
            ),
            kv("gemma-embedding2.attention.causal", GgufValue::Bool(false)),
            kv(
                "gemma-embedding2.rope.freq_base_swa",
                GgufValue::Float32(10_000.0),
            ),
            kv("gemma-embedding2.pooling_type", GgufValue::Uint32(1)),
            kv("tokenizer.ggml.model", GgufValue::String("llama".into())),
            kv(
                "tokenizer.ggml.tokens",
                GgufValue::ArrayString(vec![
                    "<pad>".into(),
                    "<eos>".into(),
                    "<bos>".into(),
                    "a".into(),
                ]),
            ),
            kv(
                "tokenizer.ggml.scores",
                GgufValue::ArrayFloat32(vec![0.0, 0.0, 0.0, -1.5]),
            ),
            kv(
                "tokenizer.ggml.token_type",
                GgufValue::ArrayInt32(vec![3, 3, 3, 1]),
            ),
            kv("tokenizer.ggml.bos_token_id", GgufValue::Uint32(2)),
            kv("tokenizer.ggml.eos_token_id", GgufValue::Uint32(1)),
            kv("tokenizer.ggml.add_eos_token", GgufValue::Bool(true)),
        ];
        let tensors = vec![
            GgufTensor {
                name: "token_embd.weight".into(),
                shape: vec![8, 4],
                dtype: GgmlType::BF16,
                data: (0..64u8).collect(),
            },
            GgufTensor {
                name: "blk.0.attn_q_norm.weight".into(),
                shape: vec![4],
                dtype: GgmlType::F32,
                data: (0..16u8).collect(),
            },
            GgufTensor {
                name: "output_norm.weight".into(),
                shape: vec![8],
                dtype: GgmlType::F32,
                data: (0..32u8).collect(),
            },
        ];
        let mut f = std::io::BufWriter::new(fs::File::create(&src).expect("create"));
        export_tensors_to_gguf(&mut f, &tensors, &header).expect("write source GGUF");
        drop(f);

        crate::format::converter::import::apr_import_gguf_raw(
            &src,
            &apr,
            &crate::format::converter::ImportOptions::default(),
        )
        .expect("import");
        let report = try_gguf_kv_verbatim_export(&apr, &out)
            .expect("export")
            .expect("APR carries gguf.kv");
        assert_eq!(report.tensor_count, tensors.len(), "no lm_head synthesized");

        let back = GgufReader::from_file(&out).expect("read round-tripped GGUF");
        assert_eq!(reader_header(&back), header_json(&header));
        let source = GgufReader::from_file(&src).expect("read source GGUF");
        let want = source.get_all_tensors_raw().expect("source tensors");
        let got = back.get_all_tensors_raw().expect("round-tripped tensors");
        assert_eq!(got.len(), tensors.len());
        assert_eq!(
            got, want,
            "names, shapes, dtypes and bytes as in the source"
        );
    }

    #[test]
    fn eg6b_apr_without_gguf_kv_falls_through() {
        let custom = std::collections::HashMap::new();
        assert!(gguf_kv_from_custom(&custom).is_none());
    }

    #[test]
    fn eg6b_tokenizer_reference_length_mismatch_is_refused() {
        let mut custom = std::collections::HashMap::new();
        custom.insert(
            "tokenizer.vocabulary".to_string(),
            serde_json::json!(["a", "b"]),
        );
        custom.insert(
            GGUF_KV_CUSTOM_KEY.to_string(),
            serde_json::json!({ "tokenizer.ggml.tokens": { "t": "custom", "custom": "tokenizer.vocabulary", "len": 3 } }),
        );
        let err = gguf_kv_from_custom(&custom)
            .expect("present")
            .expect_err("length mismatch");
        assert!(err.to_string().contains("length"), "{err}");
    }
}
