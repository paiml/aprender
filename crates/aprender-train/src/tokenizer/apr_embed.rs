//! #3803: the ONE reader of a HuggingFace tokenizer.json for the tokenizer an exported `.apr`
//! embeds (`tokenizer.vocabulary`, `tokenizer.merges`).
//!
//! The CUDA trainer's two `.apr` export sites each read the tokenizer.json by hand. Both read
//! merges in the `"a b"` form only, so every merge of a file `tokenizers` 0.20+ wrote
//! (`["a", "b"]`) was dropped without a word and the `.apr` encoded every text byte by byte.
//! Both built the vocabulary from `model.vocab` alone, so the added tokens (`<|im_start|>`,
//! `<|im_end|>`, …) never reached it, and listed it sorted by id, so a gap in the ids would
//! have shifted every later token onto the wrong id.

/// A tokenizer.json's tables as an `.apr` embeds them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbeddedTokenizerTables {
    /// Token at index `id`: `model.vocab`, then `added_tokens` at their own ids. An id neither
    /// names is an empty string, so every later id stays where the file put it.
    pub vocabulary: Vec<String>,
    /// `model.merges` as `"a b"`, whichever of the two forms the file uses, in rank order.
    pub merges: Vec<String>,
}

/// The largest token id accepted: far above any real vocabulary (~1M), far below what a
/// hostile id would make `resize` allocate.
const MAX_TOKEN_ID: usize = 1 << 24;

/// The tables of a parsed tokenizer.json, or `None` when it has no `model.vocab` object or
/// names an id above [`MAX_TOKEN_ID`].
#[must_use]
pub fn embedded_tokenizer_tables(tok: &serde_json::Value) -> Option<EmbeddedTokenizerTables> {
    let model = tok.get("model")?;
    let base = model.get("vocab")?.as_object()?;
    let added = tok
        .get("added_tokens")
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|t| Some((t.get("content")?.as_str()?, t.get("id")?.as_u64()?)));

    let mut vocabulary: Vec<String> = Vec::with_capacity(base.len());
    for (token, id) in base.iter().filter_map(|(k, v)| Some((k.as_str(), v.as_u64()?))).chain(added)
    {
        let id = usize::try_from(id).ok().filter(|&id| id <= MAX_TOKEN_ID)?;
        if vocabulary.len() <= id {
            vocabulary.resize(id + 1, String::new());
        }
        vocabulary[id] = token.to_string();
    }

    let merges = model
        .get("merges")
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|m| match m {
            serde_json::Value::String(s) => Some(s.clone()),
            serde_json::Value::Array(pair) => match pair.as_slice() {
                [a, b] => Some(format!("{} {}", a.as_str()?, b.as_str()?)),
                _ => None,
            },
            _ => None,
        })
        .collect();

    Some(EmbeddedTokenizerTables { vocabulary, merges })
}

#[cfg(test)]
#[path = "apr_embed_tests.rs"]
mod tests;
