//! #3963 (and retroactively #3950): the ORACLE is proven before it judges.
//!
//! A GPU GEMV kernel is admitted when it agrees with `iq_parallel_matvec`, i.e.
//! with aprender-serve's CPU decoder. That decoder is therefore the thing that
//! decides admission, and it must itself be proven against an INDEPENDENT decoder
//! on the real bytes — otherwise a transcription error shared by our grid
//! constant and our decoder would be faithfully reproduced by the kernel and
//! pass every A/B.
//!
//! The independent decoder is llama.cpp's gguf-py (`gguf.quants.dequantize`),
//! which decodes from its OWN hex-encoded grid rather than
//! `quantize::iq_grids`. `scripts/iq_gguf_py_reference.py` writes its output per
//! tensor; this compares ours against it element by element.
//!
//! Every tensor named in the reference MANIFEST must be found in the model, have
//! the MANIFEST's raw-byte sha256 (so the reference is about THESE bytes), and
//! decode to the same values. Nothing is skipped silently: a manifest row whose
//! tensor is missing or whose bytes differ FAILS.
//!
//! Run (no GPU needed):
//!   PYTHONPATH=<llama.cpp>/gguf-py python3 scripts/iq_gguf_py_reference.py \
//!       <model.gguf> IQ3_XXS <refdir>
//!   APR_IQ_AB_MODEL=<model.gguf> APR_GGUFPY_REF_DIR=<refdir> APR_GGUFPY_QTYPE=18 \
//!       cargo test -p aprender-serve --lib every_tensor_decodes_like_gguf_py \
//!       -- --ignored --nocapture

#[test]
#[ignore = "needs APR_IQ_AB_MODEL + APR_GGUFPY_REF_DIR + APR_GGUFPY_QTYPE"]
fn every_tensor_decodes_like_gguf_py() {
    use sha2::{Digest, Sha256};
    let (Ok(model), Ok(refdir), Ok(qtype)) = (
        std::env::var("APR_IQ_AB_MODEL"),
        std::env::var("APR_GGUFPY_REF_DIR"),
        std::env::var("APR_GGUFPY_QTYPE"),
    ) else {
        eprintln!("SKIP: set APR_IQ_AB_MODEL, APR_GGUFPY_REF_DIR and APR_GGUFPY_QTYPE");
        return;
    };
    let qtype: u32 = qtype
        .parse()
        .expect("APR_GGUFPY_QTYPE is a ggml type number");
    let block_bytes = super::iq_dispatch::iq_block_bytes(qtype)
        .unwrap_or_else(|| panic!("ggml type {qtype} is not an IQ type this crate decodes"));
    let block_elems = super::iq_dispatch::iq_block_elems(qtype).expect("elems for an IQ type");

    let manifest = std::fs::read_to_string(format!("{refdir}/MANIFEST.tsv"))
        .expect("the reference MANIFEST.tsv (run scripts/iq_gguf_py_reference.py first)");
    let rows: Vec<Vec<&str>> = manifest.lines().map(|l| l.split('\t').collect()).collect();
    assert!(
        !rows.is_empty(),
        "the MANIFEST names no tensor: nothing would be compared"
    );

    let mapped = crate::gguf::MappedGGUFModel::from_path(&model).expect("map the GGUF");
    let base = mapped.model.tensor_data_start;

    let (mut exact_tensors, mut elems_total, mut elems_exact) = (0usize, 0usize, 0usize);
    let mut worst = (0.0f64, String::new());
    for r in &rows {
        let (name, ne1, ne0, sha) = (
            r[0],
            r[1].parse::<usize>().unwrap(),
            r[2].parse::<usize>().unwrap(),
            r[3],
        );
        let t = mapped
            .model
            .tensors
            .iter()
            .find(|t| t.name == name)
            .unwrap_or_else(|| panic!("MANIFEST names {name}, which the model does not hold"));
        assert_eq!(
            t.qtype, qtype,
            "{name}: model says type {}, reference was built for {qtype}",
            t.qtype
        );
        let nb = ne0.div_ceil(block_elems);
        let start = base + usize::try_from(t.offset).unwrap();
        let raw = &mapped.mmap[start..start + ne1 * nb * block_bytes];
        let got_sha = format!("{:x}", Sha256::digest(raw));
        assert_eq!(
            got_sha, sha,
            "{name}: the reference was computed from different bytes"
        );

        let reference: Vec<f32> = std::fs::read(format!("{refdir}/{name}.f32"))
            .unwrap_or_else(|e| panic!("{name}.f32: {e}"))
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        assert_eq!(reference.len(), ne1 * ne0, "{name}: reference size");

        let mut row_buf = vec![0.0f32; nb * block_elems];
        let mut tensor_exact = true;
        for row in 0..ne1 {
            for b in 0..nb {
                let off = (row * nb + b) * block_bytes;
                super::iq_dispatch::dequantize_iq_block(
                    qtype,
                    &raw[off..off + block_bytes],
                    &mut row_buf[b * block_elems..(b + 1) * block_elems],
                )
                .expect("decode one block");
            }
            for c in 0..ne0 {
                let (a, e) = (row_buf[c], reference[row * ne0 + c]);
                elems_total += 1;
                if a.to_bits() == e.to_bits() {
                    elems_exact += 1;
                } else {
                    tensor_exact = false;
                    let rel = f64::from((a - e).abs()) / f64::from(e.abs()).max(1e-30);
                    // NaN counts as worse, as `!(rel <= worst)` did.
                    if !matches!(
                        rel.partial_cmp(&worst.0),
                        Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
                    ) {
                        worst = (
                            rel,
                            format!("{name}[{row},{c}]: ours {a:e} vs gguf-py {e:e}"),
                        );
                    }
                }
            }
        }
        if tensor_exact {
            exact_tensors += 1;
        }
    }
    eprintln!(
        "CELL ({model}, q={qtype}, CPU decoder vs gguf-py) -> {} tensors, {exact_tensors} \
         bit-exact; {elems_exact}/{elems_total} elements bit-exact; worst relative {:.3e} {}",
        rows.len(),
        worst.0,
        worst.1
    );
    // Two independent f32 transcriptions of `d * (0.5 + s) * k * grid * sign` may
    // differ by rounding order; they may not differ by more than a couple of ulps.
    assert!(
        worst.0 <= 1e-6,
        "our CPU decoder disagrees with gguf-py beyond rounding: {}",
        worst.1
    );
}
