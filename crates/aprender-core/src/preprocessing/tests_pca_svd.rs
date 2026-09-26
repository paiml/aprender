//! #3148 — PCA by SVD of the centered data, and TruncatedSVD.
//! Oracle: `tests/fixtures/pca_sklearn_v1.json` (sklearn.decomposition.PCA, full solver;
//! generator `gen_pca_sklearn_v1.py` beside it).

use super::*;

#[derive(serde::Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}

#[derive(serde::Deserialize)]
struct Case {
    name: String,
    rows: usize,
    cols: usize,
    x: Vec<f64>,
    n_components: f64,
    k: usize,
    vectors: usize,
    components: Vec<f64>,
    explained_variance: Vec<f64>,
    explained_variance_ratio: Vec<f64>,
    singular_values: Vec<f64>,
}

fn fixture() -> Vec<Case> {
    let f: Fixture = serde_json::from_str(include_str!("../../tests/fixtures/pca_sklearn_v1.json"))
        .expect("pca_sklearn_v1.json parses");
    f.cases
}

fn case(name: &str) -> Case {
    fixture()
        .into_iter()
        .find(|c| c.name == name)
        .unwrap_or_else(|| panic!("fixture case {name} missing"))
}

fn matrix_f32(c: &Case) -> Matrix<f32> {
    // the generator rounded every value to f32, so this cast is exact
    Matrix::from_vec(c.rows, c.cols, c.x.iter().map(|&v| v as f32).collect())
        .expect("fixture shape")
}

fn pca_for(c: &Case) -> PCA {
    let p = if c.n_components < 1.0 {
        PCA::with_variance_ratio(c.n_components)
    } else {
        PCA::new(c.n_components as usize)
    };
    p.with_svd_solver(SvdSolver::Full)
}

/// Largest |a - s·b| over the first `rows` rows of two `_ x cols` matrices, best sign per row.
fn max_row_err_up_to_sign(a: &[f64], b: &[f64], rows: usize, cols: usize) -> f64 {
    (0..rows)
        .map(|r| {
            let (ra, rb) = (&a[r * cols..(r + 1) * cols], &b[r * cols..(r + 1) * cols]);
            let err = |s: f64| ra.iter().zip(rb).map(|(x, y)| (x - s * y).abs()).fold(0.0, f64::max);
            err(1.0).min(err(-1.0))
        })
        .fold(0.0, f64::max)
}

// ---- no covariance path -------------------------------------------------------------

/// Does this (comment-stripped) source line build a covariance / normal-equations matrix?
/// Anchored on the three shapes the old path used: an eigensolver on a symmetric matrix,
/// a `d x d` buffer sized `n_features * n_features`, and a binding named `cov…`.
fn forms_covariance(line: &str) -> bool {
    let code = line.split("//").next().unwrap_or("");
    let squashed: String = code.chars().filter(|c| !c.is_whitespace()).collect();
    let idents: Vec<&str> = code
        .split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .filter(|t| !t.is_empty())
        .collect();
    let binds_cov = idents.windows(2).enumerate().any(|(i, w)| {
        let name = if w[1] == "mut" { idents.get(i + 2).copied().unwrap_or("") } else { w[1] };
        w[0] == "let" && (name == "cov" || name.starts_with("cov_") || name.starts_with("covariance"))
    });
    squashed.contains("SymmetricEigen")
        || squashed.contains("n_features*n_features")
        || squashed.contains("d*d]")
        || binds_cov
}

#[test]
fn covariance_guard_case_table() {
    let must_match = [
        "let mut cov = vec![0.0_f32; n_features * n_features];",
        "        let cov_matrix = trueno::Matrix::from_vec(n_features, n_features, cov)",
        "let eigen = SymmetricEigen::new(&cov_matrix)",
        "use trueno::SymmetricEigen;",
        "let cov: Vec<f64> = vec![0.0; d * d];",
        "let mut covariance = xt.matmul(&x);",
    ];
    let must_not_match = [
        "// forming the covariance matrix squares the condition number",
        "/// The covariance matrix is never formed.",
        "let centered: Vec<f64> = raw.iter().map(|v| v - m).collect();",
        "let (sigma, vt) = top_k_svd(&centered, n, d, k_svd, randomized)?;",
        "let total = sigma.iter().map(|s| s * s).sum::<f64>() / denom;",
        "let covered = 3; // not a covariance",
        "let mut coverage = vec![0.0; n];",
        "let recov = cov_free_path(x);",
    ];
    for l in must_match {
        assert!(forms_covariance(l), "guard missed: {l}");
    }
    for l in must_not_match {
        assert!(!forms_covariance(l), "guard false positive: {l}");
    }
}

#[test]
fn falsify_pca_svd_001_pca_source_forms_no_covariance_matrix() {
    let src = include_str!("pca.rs");
    let hits: Vec<&str> = src.lines().filter(|l| forms_covariance(l)).collect();
    assert!(hits.is_empty(), "pca.rs forms a covariance matrix again: {hits:?}");
}

// ---- ill-conditioning ---------------------------------------------------------------

/// The pre-#3148 fit, verbatim in substance: f64 cross-products stored as an f32
/// covariance, then trueno's f32 SymmetricEigen. Kept here, and only here, as the foil.
fn covariance_path_components(x: &Matrix<f32>, k: usize) -> Vec<f64> {
    let (n, d) = x.shape();
    let raw: Vec<f64> = x.as_slice().iter().map(|&v| f64::from(v)).collect();
    let mean: Vec<f32> = (0..d)
        .map(|j| ((0..n).map(|i| raw[i * d + j]).sum::<f64>() / n as f64) as f32)
        .collect();
    let c: Vec<f64> = raw.iter().enumerate().map(|(i, v)| v - f64::from(mean[i % d])).collect();
    let mut cv = vec![0.0_f32; d * d];
    for i in 0..d {
        for j in 0..d {
            let s: f64 = (0..n).map(|r| c[r * d + i] * c[r * d + j]).sum();
            cv[i * d + j] = (s / (n - 1) as f64) as f32;
        }
    }
    let m = trueno::Matrix::from_vec(d, d, cv).expect("square");
    let eig = trueno::SymmetricEigen::new(&m).expect("eigen");
    let vecs = eig.eigenvectors();
    let mut out = Vec::with_capacity(k * d);
    for i in 0..k {
        for j in 0..d {
            out.push(f64::from(*vecs.get(j, i).expect("in range")));
        }
    }
    out
}

#[test]
fn falsify_pca_svd_002_ill_conditioned_components_match_exact_svd_where_covariance_fails() {
    // singular values 1e0 .. 1e-9 (fixture `illcond_200x12_k10`)
    let c = case("illcond_200x12_k10");
    let x = matrix_f32(&c);
    let mut pca = pca_for(&c);
    pca.fit(&x).expect("fit");
    let got = pca.components_f64().expect("fitted");
    let svd_err = max_row_err_up_to_sign(got, &c.components, c.vectors, c.cols);
    assert!(svd_err <= 1e-7, "SVD path: {svd_err:e} > 1e-7 on the first {} components", c.vectors);

    let foil = covariance_path_components(&x, c.k);
    let cov_err = max_row_err_up_to_sign(&foil, &c.components, c.vectors, c.cols);
    assert!(
        cov_err > 1e-7,
        "the covariance foil met 1e-7 ({cov_err:e}) — this test no longer distinguishes the paths"
    );
}

// ---- explained variance -------------------------------------------------------------

#[test]
fn falsify_pca_svd_003_ratio_sums_to_one_at_full_rank() {
    for (name, solver) in [("gauss_50x8_k8", SvdSolver::Full), ("gauss_50x8_k8", SvdSolver::Auto)] {
        let c = case(name);
        let mut pca = PCA::new(c.rows.min(c.cols)).with_svd_solver(solver);
        pca.fit(&matrix_f32(&c)).expect("fit");
        let sum: f64 = pca.explained_variance_ratio_f64().expect("fitted").iter().sum();
        assert!((sum - 1.0).abs() <= 1e-12, "{name} {solver:?}: ratio sum {sum:.17}");
    }
    // wide: k = min(n, d) = n
    let c = case("wide_30x60_k5");
    let mut pca = PCA::new(c.rows);
    // n - 1 non-zero components after centering; the last is rounding noise but counts
    pca.fit(&matrix_f32(&c)).expect("fit");
    let sum: f64 = pca.explained_variance_ratio_f64().expect("fitted").iter().sum();
    assert!((sum - 1.0).abs() <= 1e-12, "wide: ratio sum {sum:.17}");
}

// ---- determinism --------------------------------------------------------------------

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn falsify_pca_svd_004_two_fits_are_byte_identical() {
    let x = spectrum_matrix(600, 40, 5);
    for solver in [SvdSolver::Full, SvdSolver::Randomized] {
        let mut a = PCA::new(5).with_svd_solver(solver);
        let mut b = PCA::new(5).with_svd_solver(solver);
        a.fit(&x).expect("fit a");
        b.fit(&x).expect("fit b");
        assert_eq!(bits(a.components_f64().expect("a")), bits(b.components_f64().expect("b")), "{solver:?}");
        assert_eq!(
            a.components().expect("a").as_slice().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            b.components().expect("b").as_slice().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        );
    }
}

#[test]
fn sign_rule_first_max_abs_entry_of_every_component_is_positive() {
    let c = case("gauss_50x8_k8");
    let mut pca = pca_for(&c);
    pca.fit(&matrix_f32(&c)).expect("fit");
    for row in pca.components_f64().expect("fitted").chunks_exact(c.cols) {
        let best = row.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let first = row.iter().find(|v| v.abs() == best).expect("non-empty");
        assert!(*first > 0.0, "{row:?}");
    }
}

// ---- sklearn oracle -----------------------------------------------------------------

#[test]
fn falsify_pca_svd_005_sklearn_parity() {
    for c in fixture() {
        let mut pca = pca_for(&c);
        pca.fit(&matrix_f32(&c)).expect("fit");
        assert_eq!(pca.n_components_fitted(), Some(c.k), "{}: k", c.name);
        let err = max_row_err_up_to_sign(pca.components_f64().expect("fitted"), &c.components, c.vectors, c.cols);
        assert!(err <= 1e-8, "{}: components {err:e} > 1e-8", c.name);
        for (what, got, want) in [
            ("explained_variance", pca.explained_variance_f64(), &c.explained_variance),
            ("explained_variance_ratio", pca.explained_variance_ratio_f64(), &c.explained_variance_ratio),
            ("singular_values", pca.singular_values(), &c.singular_values),
        ] {
            let got = got.expect("fitted");
            assert_eq!(got.len(), want.len(), "{}: {what} length", c.name);
            for (g, w) in got.iter().zip(want) {
                assert!((g - w).abs() <= 1e-10 * w.abs().max(1e-300) + 1e-15, "{}: {what} {g} vs {w}", c.name);
            }
        }
    }
}

// ---- solver selection ---------------------------------------------------------------

/// `n x d` with singular values 2^-i: a clean low-rank spectrum for the randomized solver.
fn spectrum_matrix(n: usize, d: usize, seed: u64) -> Matrix<f32> {
    let mut s = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    let mut next = move || {
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        ((s >> 11) as f64 / (1u64 << 53) as f64) - 0.5
    };
    let r = d.min(n);
    let u: Vec<f64> = (0..n * r).map(|_| next()).collect();
    let v: Vec<f64> = (0..r * d).map(|_| next()).collect();
    let mut x = vec![0.0_f32; n * d];
    for i in 0..n {
        for j in 0..d {
            let val: f64 = (0..r).map(|t| u[i * r + t] * 0.5_f64.powi(t as i32) * v[t * d + j]).sum();
            x[i * d + j] = val as f32;
        }
    }
    Matrix::from_vec(n, d, x).expect("shape")
}

#[test]
fn auto_picks_randomized_for_large_low_rank_and_it_agrees_with_full() {
    let x = spectrum_matrix(600, 40, 7);
    let mut auto = PCA::new(5);
    let mut rand = PCA::new(5).with_svd_solver(SvdSolver::Randomized);
    let mut full = PCA::new(5).with_svd_solver(SvdSolver::Full);
    for p in [&mut auto, &mut rand, &mut full] {
        p.fit(&x).expect("fit");
    }
    // auto resolved to randomized: bit-identical to the explicit choice, not to full
    assert_eq!(bits(auto.components_f64().expect("a")), bits(rand.components_f64().expect("r")));
    assert_ne!(bits(auto.components_f64().expect("a")), bits(full.components_f64().expect("f")));
    let err = max_row_err_up_to_sign(rand.components_f64().expect("r"), full.components_f64().expect("f"), 5, 40);
    assert!(err <= 1e-6, "randomized vs full components {err:e}");
    let (rr, fr) = (rand.explained_variance_ratio_f64().expect("r"), full.explained_variance_ratio_f64().expect("f"));
    for (a, b) in rr.iter().zip(fr) {
        assert!((a - b).abs() <= 1e-9, "ratio {a} vs {b}");
    }
}

#[test]
fn transform_then_inverse_reconstructs_at_full_rank() {
    let c = case("gauss_50x8_k8");
    let x = matrix_f32(&c);
    let mut pca = pca_for(&c);
    let z = pca.fit_transform(&x).expect("fit_transform");
    let back = pca.inverse_transform(&z).expect("inverse");
    for (a, b) in back.as_slice().iter().zip(x.as_slice()) {
        assert!((a - b).abs() <= 1e-4 * b.abs().max(1.0), "{a} vs {b}");
    }
}

#[test]
fn invalid_requests_are_errors() {
    let x = spectrum_matrix(10, 4, 1);
    let one_row = Matrix::from_vec(1, 4, vec![1.0_f32; 4]).expect("shape");
    let wide = spectrum_matrix(3, 6, 2);
    let cases: [(PCA, &Matrix<f32>, &str); 6] = [
        (PCA::new(5), &x, "n_components cannot exceed number of features"),
        (PCA::new(0), &x, "n_components must be in"),
        (PCA::new(4), &wide, "n_components must be in"),
        (PCA::new(1), &one_row, "at least 2 samples"),
        (PCA::with_variance_ratio(1.0), &x, "(0, 1)"),
        (PCA::with_variance_ratio(0.5).with_svd_solver(SvdSolver::Randomized), &x, "full spectrum"),
    ];
    for (mut p, m, want) in cases {
        let err = p.fit(m).expect_err(want).to_string();
        assert!(err.contains(want), "{err:?} lacks {want:?}");
    }
}

#[test]
fn components_for_ratio_matches_searchsorted_right_plus_one() {
    let r = [0.5, 0.3, 0.2];
    // (target, sklearn k)
    for (t, k) in [(0.1, 1), (0.5, 2), (0.79, 2), (0.8, 3), (0.99, 3)] {
        assert_eq!(components_for_ratio(&r, t), k, "ratio {t}");
    }
}

// ---- TruncatedSVD -------------------------------------------------------------------

#[test]
fn truncated_svd_randomized_matches_exact_and_does_not_center() {
    let x32 = spectrum_matrix(80, 30, 11);
    let x = Matrix::from_vec(80, 30, x32.as_slice().iter().map(|&v| f64::from(v) + 3.0).collect())
        .expect("shape");
    let mut exact = TruncatedSVD::new(4).with_svd_solver(SvdSolver::Full);
    let mut rand = TruncatedSVD::new(4);
    let ze = exact.fit_transform(&x).expect("exact");
    let zr = rand.fit_transform(&x).expect("randomized");
    let err = max_row_err_up_to_sign(exact.components().expect("e"), rand.components().expect("r"), 4, 30);
    assert!(err <= 1e-6, "randomized vs exact {err:e}");
    // uncentered: the +3 offset dominates the first singular value (PCA would remove it)
    let s0 = exact.singular_values().expect("e")[0];
    assert!(s0 > 3.0 * (80.0_f64 * 30.0).sqrt() * 0.9, "σ0 {s0} does not carry the offset");
    // explained_variance_ is the ddof=0 variance of each transformed column
    let col0: Vec<f64> = (0..80).map(|i| ze.get(i, 0)).collect();
    let m = col0.iter().sum::<f64>() / 80.0;
    let v = col0.iter().map(|z| (z - m) * (z - m)).sum::<f64>() / 80.0;
    assert!((exact.explained_variance().expect("e")[0] - v).abs() <= 1e-9 * v);
    assert_eq!(zr.shape(), (80, 4));
    let again = rand.transform(&x).expect("transform");
    assert_eq!(bits(again.as_slice()), bits(zr.as_slice()));
}

/// LSA retrieval: three topics with disjoint core vocabularies, sharing filler words.
/// Nearest neighbour (cosine) in the 3-d TruncatedSVD space must share the topic.
#[test]
fn falsify_pca_svd_006_tfidf_lsa_nearest_neighbour_beats_random() {
    use crate::text::tokenize::WhitespaceTokenizer;
    use crate::text::vectorize::TfidfVectorizer;
    let topics: [&[&str]; 3] = [
        &["goal", "striker", "keeper", "league", "match", "penalty", "coach", "stadium"],
        &["loan", "interest", "bank", "credit", "mortgage", "deposit", "rate", "savings"],
        &["virus", "vaccine", "doctor", "fever", "clinic", "patient", "dose", "symptom"],
    ];
    let filler = ["the", "a", "report", "today", "new", "said", "week", "people"];
    let mut s = 3148_u64;
    let mut pick = |n: usize| {
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        (s >> 33) as usize % n
    };
    let (mut docs, mut label) = (Vec::new(), Vec::new());
    for doc in 0..60 {
        let t = doc % 3;
        // 3 topic words among 6 filler words: a single doc is mostly noise
        let mut words: Vec<&str> = (0..3).map(|_| topics[t][pick(8)]).collect();
        words.extend((0..6).map(|_| filler[pick(8)]));
        docs.push(words.join(" "));
        label.push(t);
    }
    let mut tfidf = TfidfVectorizer::new().with_tokenizer(Box::new(WhitespaceTokenizer::new()));
    let x = tfidf.fit_transform(&docs).expect("tfidf");
    let mut lsa = TruncatedSVD::new(3);
    let z = lsa.fit_transform(&x).expect("lsa");
    let row = |i: usize| (0..3).map(|j| z.get(i, j)).collect::<Vec<f64>>();
    let cos = |a: &[f64], b: &[f64]| {
        let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        dot / (a.iter().map(|v| v * v).sum::<f64>().sqrt() * b.iter().map(|v| v * v).sum::<f64>().sqrt() + 1e-300)
    };
    let hits = (0..docs.len())
        .filter(|&i| {
            let nn = (0..docs.len())
                .filter(|&j| j != i)
                .max_by(|&a, &b| cos(&row(i), &row(a)).total_cmp(&cos(&row(i), &row(b))))
                .expect("neighbours");
            label[nn] == label[i]
        })
        .count();
    let accuracy = hits as f64 / docs.len() as f64;
    // random neighbour: 19 of 59 others share the topic
    let baseline = 19.0 / 59.0;
    assert!(accuracy >= baseline + 0.5, "LSA nn accuracy {accuracy:.3} vs random {baseline:.3}");
}
