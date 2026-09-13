// OBLIG-CPP-LEAKAGE-NOT-CONSTRUCTIBLE / DATA-06 / D-16.
//
// Both `Split` constructors are `pub(crate)`. A downstream caller therefore cannot MINT a
// typed split at all, which is what stops a train/validation pair being assembled from two
// unrelated datasets and handed to `PreparedDataset::from_validated_splits`. The only doors
// in are `PreparedDataset::from_labeled_rows` and `from_attested_bytes`, and both run the
// full five-gate ingest ladder plus the fingerprint.
//
// `Split<Train>` itself is public — accessors like `rows()` and `class_counts()` are part of
// the API — so this case is specifically about the CONSTRUCTOR's visibility, not about the
// type being hidden.
//
// Expected diagnostic: `associated function 'from_jsonl_bytes' is private`.

use aprender_contrastive_data::split::{Split, SplitDeclaration, Train};

fn mint_a_split(bytes: &[u8], decl: &SplitDeclaration) {
    let _ = Split::<Train>::from_jsonl_bytes(bytes, decl);
}

fn main() {}
