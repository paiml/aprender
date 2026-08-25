// OBLIG-CPP-LEAKAGE-NOT-CONSTRUCTIBLE / DATA-06 / D-27.
//
// `PairSampler::new` takes `&Selection` and nothing else. There is no constructor that
// accepts a bare list of row identifiers, which is what makes `split_span_fail_closed`'s
// STRUCTURAL half real: every endpoint the sampler can emit came out of the selection it
// borrows, so an id from the validation split has no way in. The typed half — for bytes
// that arrive from a dump — is `validate_pair_records`, and its negative lives in
// `tests/negative_leaky.rs`.
//
// Expected diagnostic: a type mismatch naming `Selection` against `Vec<String>`.

use aprender_contrastive_data::pairs::{PairConfig, PairSampler};

fn sample_from_raw_ids(ids: &Vec<String>) {
    let cfg = PairConfig::new(13);
    let _ = PairSampler::new(ids, &cfg);
}

fn main() {}
