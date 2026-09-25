use super::*;

const SHA: &str = "70dffe5625e1849661303b82bb1fdf5254ab1d4560e2fe3eacc147e5d2588bba";

fn header(k: u8, n: u32) -> Header {
    Header {
        schema: SCHEME.into(),
        mode: Mode::Topk,
        model_sha256: SHA.into(),
        tokenizer_sha256: SHA.into(),
        apr_tag: "v0.70.0-rc.1".into(),
        backend: "gx10-cuda".into(),
        temperature_of_record: 0.7,
        k,
        n_tokens: n,
    }
}

/// A deterministic vocab row bigger than 2^16 so id width is exercised.
fn row(seed: u32, vocab: usize) -> Vec<f32> {
    (0..vocab)
        .map(|i| {
            let x = (i as u32)
                .wrapping_mul(2_654_435_761)
                .wrapping_add(seed.wrapping_mul(40_503));
            f32::from((x >> 20) as u16) / 256.0
        })
        .collect()
}

fn tokens(n: u32, vocab: usize) -> Vec<TokenLogits> {
    (0..n)
        .map(|s| {
            TokenLogits::from_full_logits(&row(s, vocab), DEFAULT_K, s * 7).expect("finite row")
        })
        .collect()
}

/// FALSIFY-SPL-001: header and every column survive a round trip exactly.
#[test]
fn falsify_spl_001_round_trip_is_exact() {
    let t = tokens(16, 70_000);
    let h = header(DEFAULT_K, 16);
    let (h2, t2) = decode(&encode(&h, &t).expect("encode")).expect("decode");
    assert_eq!(h2, h);
    assert_eq!(t2, t);
    assert!(
        t.iter().flat_map(|x| &x.ids).any(|&id| id > 65_535),
        "fixture must carry ids > u16"
    );
}

/// FALSIFY-SPL-002: the stored mass is Σexp(logprobs) of the top-k of the
/// full softmax, and 1 − M is the true tail mass.
#[test]
fn falsify_spl_002_residual_mass_is_the_true_tail() {
    let logits = [3.0_f32, 1.0, 0.5, 0.0, -1.0, -2.0];
    let t = TokenLogits::from_full_logits(&logits, 2, 0).expect("row");
    let z: f64 = logits.iter().map(|&x| f64::from(x).exp()).sum();
    let tail = (f64::from(0.5_f32).exp() + 1.0 + (-1.0_f64).exp() + (-2.0_f64).exp()) / z;
    assert_eq!(t.ids, vec![0, 1]);
    assert!(
        (f64::from(t.residual_mass()) - tail).abs() < 1e-3,
        "{} vs {tail}",
        t.residual_mass()
    );
    assert!((f64::from(t.logsumexp_full) - z.ln()).abs() < 1e-5);
    assert!(t.logprobs.iter().all(|lp| lp.to_f32() <= 0.0));
    let (_, back) = decode(&encode(&header(2, 1), &[t.clone()]).expect("encode")).expect("decode");
    assert_eq!(back[0].residual_mass(), t.residual_mass());
}

/// FALSIFY-SPL-003: a mass that disagrees with its own logprobs is refused
/// at BOTH ends — a producer cannot write it, a reader cannot accept it.
#[test]
fn falsify_spl_003_inconsistent_mass_is_refused() {
    let mut t = tokens(1, 1_000);
    t[0].topk_mass = f16::from_f32(1.0);
    assert!(encode(&header(DEFAULT_K, 1), &t).is_err());
    let good = tokens(1, 1_000);
    let mut blob = encode(&header(DEFAULT_K, 1), &good).expect("encode");
    // Re-encode the columns with a forged mass, header untouched.
    let hl = u32::from_le_bytes(blob[8..12].try_into().expect("len")) as usize;
    let mut cols = zstd::bulk::decompress(&blob[12 + hl..], 1 << 20).expect("cols");
    let n = cols.len();
    cols[n - 2..].copy_from_slice(&f16::from_f32(1.0).to_bits().to_le_bytes());
    blob.truncate(12 + hl);
    blob.extend(zstd::bulk::compress(&cols, 3).expect("zstd"));
    assert!(decode(&blob)
        .expect_err("forged mass")
        .contains("topk_mass"));
}

/// FALSIFY-SPL-004: the header is closed and the columns are exact-length.
#[test]
fn falsify_spl_004_decoder_is_strict() {
    let t = tokens(2, 500);
    let blob = encode(&header(DEFAULT_K, 2), &t).expect("encode");
    let hl = u32::from_le_bytes(blob[8..12].try_into().expect("len")) as usize;
    let head = std::str::from_utf8(&blob[12..12 + hl]).expect("utf8");
    // Unknown key.
    let extra = head.replacen('{', "{\"extra\":1,", 1);
    let mut b = MAGIC.to_vec();
    b.extend((extra.len() as u32).to_le_bytes());
    b.extend(extra.as_bytes());
    b.extend(&blob[12 + hl..]);
    assert!(decode(&b).is_err(), "unknown header key accepted");
    // n_tokens lies about the column length.
    assert!(encode(&header(DEFAULT_K, 3), &t).is_err());
    let lie = head.replace("\"n_tokens\":2", "\"n_tokens\":1");
    let mut b = MAGIC.to_vec();
    b.extend((lie.len() as u32).to_le_bytes());
    b.extend(lie.as_bytes());
    b.extend(&blob[12 + hl..]);
    assert!(decode(&b).is_err(), "column overrun accepted");
    // Truncation anywhere and a bad magic.
    for cut in [0, 7, 11, 12 + hl - 1, blob.len() - 1] {
        assert!(decode(&blob[..cut]).is_err(), "truncated at {cut} accepted");
    }
    let mut bad = blob.clone();
    bad[0] = b'X';
    assert!(decode(&bad).is_err());
    // A trailing byte after the column frame.
    let mut long = blob.clone();
    long.push(0);
    assert!(decode(&long).is_err(), "trailing byte accepted");
    // Exactly one extra column byte inside a valid zstd frame.
    let mut cols = zstd::bulk::decompress(&blob[12 + hl..], 1 << 20).expect("cols");
    cols.push(0);
    let mut b = blob[..12 + hl].to_vec();
    b.extend(zstd::bulk::compress(&cols, 3).expect("zstd"));
    assert!(decode(&b).is_err(), "one extra column byte accepted");
}

#[test]
fn from_full_logits_rejects_degenerate_rows() {
    assert!(TokenLogits::from_full_logits(&[1.0, 2.0], 0, 0).is_none());
    assert!(TokenLogits::from_full_logits(&[1.0, 2.0], 3, 0).is_none());
    assert!(TokenLogits::from_full_logits(&[1.0, f32::NAN], 1, 0).is_none());
    assert!(TokenLogits::from_full_logits(&[1.0, f32::INFINITY], 1, 0).is_none());
    // k == vocab: all the mass is kept, residual ≈ 0.
    let t = TokenLogits::from_full_logits(&[0.0, 0.0, 0.0], 3, 0).expect("row");
    assert!(t.residual_mass() < 1e-3);
    assert_eq!(t.ids, vec![0, 1, 2], "ties break by ascending id");
}

#[test]
fn header_rejects_non_sha_and_bad_schema() {
    let t = tokens(1, 100);
    let mut h = header(DEFAULT_K, 1);
    h.model_sha256 = "abc".into();
    assert!(encode(&h, &t).is_err());
    let mut h = header(DEFAULT_K, 1);
    h.schema = "sparse-logits-v0".into();
    assert!(encode(&h, &t).is_err());
    let mut h = header(DEFAULT_K, 1);
    h.temperature_of_record = f32::NAN;
    assert!(encode(&h, &t).is_err());
}

/// M6: the RAW figure is exact (130 B/token at k = 20; §2.7 estimated ≈128).
/// The encoded figure here is from a SYNTHETIC, highly regular fixture and is
/// a codec smoke only — it is not M6. M6's encoded bytes/token are measured
/// on real `apr serve` rows once top-k capture exists.
#[test]
fn m6_raw_bytes_per_token_at_k20() {
    assert_eq!(raw_column_bytes(1, 20), 130);
    let n = 256;
    let blob = encode(&header(DEFAULT_K, n), &tokens(n, 151_936)).expect("encode");
    let raw = raw_column_bytes(n as usize, 20);
    // Column frame + header is bounded by raw + zstd frame overhead + header.
    assert!(
        blob.len() < raw + 1024,
        "encoded {} > raw {raw} + 1 KiB",
        blob.len()
    );
}
