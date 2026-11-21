//! Deterministic SafeTensors serialization (#7293).

/// Serialize tensors to SafeTensors bytes with `__metadata__` keys in sorted order.
///
/// `safetensors::serialize` takes a `HashMap`, whose iteration order is random per
/// instance, so two same-seed runs wrote different header bytes (#7255). The tensors
/// are serialized without metadata and the sorted `__metadata__` object is spliced in
/// as the first header entry; the header length is recomputed and re-padded to 8 bytes.
pub(crate) fn serialize_with_ordered_metadata<V>(
    views: Vec<(&str, V)>,
    metadata: &std::collections::BTreeMap<String, String>,
) -> crate::Result<Vec<u8>>
where
    V: safetensors::tensor::View,
{
    let ser_err =
        |m: String| crate::Error::Serialization(format!("SafeTensors serialization failed: {m}"));
    let bytes = safetensors::serialize(views, None).map_err(|e| ser_err(e.to_string()))?;
    let n = u64::from_le_bytes(bytes[..8].try_into().map_err(|_| ser_err("short header".into()))?)
        as usize;
    let header = std::str::from_utf8(&bytes[8..8 + n]).map_err(|e| ser_err(e.to_string()))?;
    let body = header
        .trim_end()
        .strip_prefix('{')
        .ok_or_else(|| ser_err("header is not an object".into()))?;
    let meta = serde_json::to_string(metadata).map_err(|e| ser_err(e.to_string()))?;
    let sep = if body.trim_start().starts_with('}') { "" } else { "," };
    let mut new_header = format!("{{\"__metadata__\":{meta}{sep}{body}");
    while new_header.len() % 8 != 0 {
        new_header.push(' ');
    }
    let mut out = Vec::with_capacity(8 + new_header.len() + bytes.len() - 8 - n);
    out.extend_from_slice(&(new_header.len() as u64).to_le_bytes());
    out.extend_from_slice(new_header.as_bytes());
    out.extend_from_slice(&bytes[8 + n..]);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordered_metadata_is_sorted_and_roundtrips() {
        use safetensors::tensor::{Dtype, TensorView};
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let bytes: &[u8] = bytemuck::cast_slice(&data);
        let mut meta = std::collections::BTreeMap::new();
        meta.insert("val_loss".to_string(), "6.743455".to_string());
        meta.insert("epoch".to_string(), "1".to_string());
        let mk = || vec![("t", TensorView::new(Dtype::F32, vec![2, 2], bytes).unwrap())];
        let a = serialize_with_ordered_metadata(mk(), &meta).unwrap();
        for _ in 0..32 {
            assert_eq!(a, serialize_with_ordered_metadata(mk(), &meta).unwrap());
        }
        let n = u64::from_le_bytes(a[..8].try_into().unwrap()) as usize;
        assert_eq!(n % 8, 0);
        let hdr = std::str::from_utf8(&a[8..8 + n]).unwrap();
        assert!(
            hdr.starts_with(r#"{"__metadata__":{"epoch":"1","val_loss":"6.743455"},"#),
            "{hdr}"
        );
        let st = safetensors::SafeTensors::deserialize(&a).unwrap();
        assert_eq!(st.tensor("t").unwrap().data(), bytes);
    }

    #[test]
    fn ordered_metadata_with_no_tensors_is_valid() {
        let mut meta = std::collections::BTreeMap::new();
        meta.insert("a".to_string(), "b".to_string());
        let none: Vec<(&str, safetensors::tensor::TensorView<'_>)> = vec![];
        let out = serialize_with_ordered_metadata(none, &meta).unwrap();
        assert!(safetensors::SafeTensors::deserialize(&out).is_ok());
    }
}
