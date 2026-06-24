
// PMAT-918: `apr convert` must synthesize a tied lm_head for tied-embedding
// models so the converted `.apr` is runnable.
//
// Contract: contracts/tied-embeddings-v1.yaml
//   OBLIG-CONVERT-TIED-EMBEDDING-LMHEAD
//
// RED (pre-fix): for a tied-embedding source (only `model.embed_tokens.weight`,
// NO separate `lm_head.weight`/`output.weight`), every `apr_convert` save path
// (f32/int8/int4/fp16/Q4K) emitted an APR whose tensor index contained NO output
// projection. The artifact was non-self-describing and the Q4K path produced a
// model with no LM head at all.
//
// GREEN (post-fix): the synthesized `lm_head.weight` is present in EVERY quant
// path, bit-for-bit identical to the embedding, row-major `[vocab, hidden]`.
#[cfg(test)]
mod tests_convert_tied_lmhead {
    use super::*;
    use crate::format::v2::AprV2Reader;
    use std::collections::BTreeMap;

    /// Build a tiny tied-embedding SafeTensors fixture: one embedding tensor,
    /// NO `lm_head.weight`/`output.weight`. vocab=8, hidden=4.
    fn tied_embedding_safetensors(path: &std::path::Path) {
        let vocab = 8usize;
        let hidden = 4usize;
        let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
        let embed: Vec<f32> = (0..vocab * hidden).map(|i| (i as f32) * 0.01 - 0.1).collect();
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            (embed, vec![vocab, hidden]),
        );
        save_safetensors(path, &tensors).expect("write tied fixture safetensors");
    }

    /// Return the list of tensor names in an APR file on disk.
    fn apr_tensor_names(path: &std::path::Path) -> Vec<String> {
        let bytes = std::fs::read(path).expect("read apr");
        let reader = AprV2Reader::from_bytes(&bytes).expect("parse apr v2");
        reader.tensor_names().iter().map(|s| s.to_string()).collect()
    }

    fn convert_with(quant: Option<QuantizationType>) -> Vec<String> {
        let dir = tempfile::tempdir().expect("tempdir");
        let input = dir.path().join("tied.safetensors");
        let output = dir.path().join("out.apr");
        tied_embedding_safetensors(&input);

        let options = ConvertOptions {
            quantize: quant,
            compress: if quant.is_none() {
                Some(Compression::None)
            } else {
                None
            },
            ..Default::default()
        };
        apr_convert(
            input.to_string_lossy().as_ref(),
            output.to_string_lossy().as_ref(),
            options,
        )
        .expect("apr_convert must succeed");
        apr_tensor_names(&output)
    }

    fn has_lm_head(names: &[String]) -> bool {
        names.iter().any(|n| n == "lm_head.weight" || n == "output.weight")
    }

    /// FALSIFY-CONVERT-TIED-LMHEAD-001 (f32 path):
    /// converting a tied-embedding model MUST produce an APR with a resolvable
    /// output projection. RED before PMAT-918 (NONE present), GREEN after.
    #[test]
    fn test_convert_tied_synthesizes_lm_head_f32() {
        let names = convert_with(None);
        assert!(
            has_lm_head(&names),
            "FALSIFY-CONVERT-TIED-LMHEAD-001 (f32): tied-embedding `apr convert` \
             produced a non-runnable APR with NO lm_head.weight/output.weight. \
             Tensors present: {names:?}"
        );
    }

    /// FALSIFY-CONVERT-TIED-LMHEAD-001 (int8 path).
    #[test]
    fn test_convert_tied_synthesizes_lm_head_int8() {
        let names = convert_with(Some(QuantizationType::Int8));
        assert!(
            has_lm_head(&names),
            "FALSIFY-CONVERT-TIED-LMHEAD-001 (int8): tied-embedding `apr convert \
             --quantize int8` produced an APR with NO output projection. \
             Tensors present: {names:?}"
        );
    }

    /// FALSIFY-CONVERT-TIED-LMHEAD-001 (Q4K path): the Q4K save path previously
    /// had no tied-LM-head synthesis at all, so the converted model literally
    /// had no way to project to vocab logits.
    #[test]
    fn test_convert_tied_synthesizes_lm_head_q4k() {
        let names = convert_with(Some(QuantizationType::Q4K));
        assert!(
            has_lm_head(&names),
            "FALSIFY-CONVERT-TIED-LMHEAD-001 (Q4K): tied-embedding `apr convert \
             --quantize q4k` produced an APR with NO output projection. \
             Tensors present: {names:?}"
        );
    }

    /// FALSIFY-CONVERT-TIED-LMHEAD-002: the synthesized lm_head must keep the
    /// embedding shape `[vocab, hidden]` (row-major, LAYOUT-001) — NOT transposed.
    #[test]
    fn test_convert_tied_lm_head_shape_matches_embedding() {
        let dir = tempfile::tempdir().expect("tempdir");
        let input = dir.path().join("tied.safetensors");
        let output = dir.path().join("out.apr");
        tied_embedding_safetensors(&input);

        apr_convert(
            input.to_string_lossy().as_ref(),
            output.to_string_lossy().as_ref(),
            ConvertOptions {
                compress: Some(Compression::None),
                ..Default::default()
            },
        )
        .expect("apr_convert ok");

        let bytes = std::fs::read(&output).expect("read apr");
        let reader = AprV2Reader::from_bytes(&bytes).expect("parse apr");
        let embed = reader
            .get_tensor("model.embed_tokens.weight")
            .expect("embedding tensor present")
            .shape
            .clone();
        let lm_head = reader
            .get_tensor("lm_head.weight")
            .expect("synthesized lm_head present")
            .shape
            .clone();
        assert_eq!(
            lm_head, embed,
            "FALSIFY-CONVERT-TIED-LMHEAD-002: synthesized lm_head shape {lm_head:?} \
             must equal embedding shape {embed:?} (row-major [vocab, hidden])"
        );
    }
}
