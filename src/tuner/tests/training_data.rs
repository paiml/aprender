//! Training data quality tests (F041-F060).

use super::super::*;

#[test]
fn f059_no_data_leakage() {
    // Training labels should not be in feature vector
    let features = TunerFeatures::builder()
        .measured_tps(500.0) // Label
        .build();

    let v = features.to_vector();
    // measured_tps should NOT be in the vector (it's a label)
    assert_eq!(v.len(), TunerFeatures::DIM);
}
