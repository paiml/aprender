// OBLIG-CPP-LEAKAGE-NOT-CONSTRUCTIBLE / DATA-06 / D-19.
//
// This is the core of D-19: the compatibility dataset does not merely leave its validation
// split empty, it HAS NO PLACE TO PUT ONE. `DatasetProfile::Splits` selects which splits
// exist, and `validation_witness` is implemented only in `impl PreparedDataset<Canonical>`.
// Not rejected — unrepresentable.
//
// Expected diagnostic: `no method named 'validation_witness' found for reference
// '&PreparedDataset<Compatibility>'`.

use aprender_contrastive_data::prepared::{Compatibility, PreparedDataset};

fn take_witness(dataset: &PreparedDataset<Compatibility>) {
    let _ = dataset.validation_witness();
}

fn main() {}
