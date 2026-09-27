//! E9 falsifier for `KMeans::with_n_init` (#3146).
//!
//! Fits k = 2..=30 with `n_init = 1` and `n_init = 10` (seed 42) on a dense
//! matrix file (one row per line, whitespace-separated floats; the E9 corpus
//! is sklearn's TF-IDF of 232 issues x 2665 terms). Prints one TSV row per k
//! and exits 1 unless `n_init = 10` is never worse at any k and strictly
//! better at one k at least.
//!
//! `cargo run -p aprender-core --example kmeans_n_init_e9 --release -- <matrix.txt>`
use aprender::cluster::KMeans;
use aprender::primitives::Matrix;
use aprender::traits::UnsupervisedEstimator;

fn fit(x: &Matrix<f32>, k: usize, n_init: usize) -> f32 {
    let mut m = KMeans::new(k)
        .with_random_state(42)
        .with_max_iter(300)
        .with_tol(1e-4)
        .with_n_init(n_init);
    m.fit(x).expect("fit");
    m.inertia()
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: kmeans_n_init_e9 <matrix.txt>");
    let text = std::fs::read_to_string(&path).expect("read matrix");
    let rows: Vec<Vec<f32>> = text
        .lines()
        .map(|l| {
            l.split_whitespace()
                .map(|t| t.parse().expect("float"))
                .collect()
        })
        .collect();
    let (r, c) = (rows.len(), rows[0].len());
    let x = Matrix::from_vec(r, c, rows.into_iter().flatten().collect()).expect("matrix");

    println!("k\tn_init_1\tn_init_10\tdelta_pct");
    let (mut worse, mut better) = (0, 0);
    for k in 2..=30 {
        let (one, ten) = (fit(&x, k, 1), fit(&x, k, 10));
        worse += usize::from(ten > one);
        better += usize::from(ten < one);
        println!(
            "{k}\t{one}\t{ten}\t{:.4}",
            100.0 * f64::from(ten - one) / f64::from(one)
        );
    }
    println!("# {r}x{c}: n_init=10 better at {better}/29 k, worse at {worse}/29 k");
    if worse > 0 || better == 0 {
        eprintln!("FALSIFIED: n_init=10 must never lose to n_init=1 and must win somewhere");
        std::process::exit(1);
    }
}
