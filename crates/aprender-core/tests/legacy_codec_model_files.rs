//! FALSIFY-LEGACY-CODEC-005: model files written by the bincode 1.x build still load, and
//! saving the loaded model reproduces those files byte-for-byte (the on-disk format did not move).
//! Fixtures were written by `LinearRegression::save` / `KMeans::save` on origin/main
//! (bincode 1.3.3) from the data below.
use aprender::cluster::KMeans;
use aprender::linear_model::LinearRegression;
use aprender::primitives::Matrix;
use aprender::traits::{Estimator, UnsupervisedEstimator};

const LR: &[u8] = include_bytes!("fixtures/legacy_codec/linear_regression.bin");
const KM: &[u8] = include_bytes!("fixtures/legacy_codec/kmeans.bin");

fn x() -> Matrix<f32> {
    Matrix::from_vec(
        6,
        2,
        vec![1.0, 2.0, 2.0, 1.0, 3.0, 5.0, 4.0, 3.0, 5.0, 8.0, 6.0, 7.0],
    )
    .expect("matrix")
}

fn tmp(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("legacy_codec_{}_{name}", std::process::id()))
}

#[test]
fn linear_regression_file_loads_and_resaves_identically() {
    let p = tmp("lr.bin");
    std::fs::write(&p, LR).expect("write fixture");
    let model = LinearRegression::load(&p).expect("old file must load");
    assert!(model.is_fitted());
    assert_eq!(model.predict(&x()).len(), 6);
    let q = tmp("lr_out.bin");
    model.save(&q).expect("save");
    assert_eq!(
        std::fs::read(&q).expect("read"),
        LR,
        "re-saved bytes differ from the 1.x file"
    );
}

#[test]
fn kmeans_file_loads_and_resaves_identically() {
    let p = tmp("km.bin");
    std::fs::write(&p, KM).expect("write fixture");
    let model = KMeans::load(&p).expect("old file must load");
    assert_eq!(model.predict(&x()).len(), 6);
    let q = tmp("km_out.bin");
    model.save(&q).expect("save");
    assert_eq!(
        std::fs::read(&q).expect("read"),
        KM,
        "re-saved bytes differ from the 1.x file"
    );
}
