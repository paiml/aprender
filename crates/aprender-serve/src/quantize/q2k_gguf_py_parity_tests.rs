//! #3960: the Q2_K CPU decoder, proven against llama.cpp's gguf-py BEFORE it is allowed
//! to be the oracle for a GPU kernel.
//!
//! Every device A/B trusts its CPU decoder. If the decoder is wrong, the kernel is proven
//! to match a wrong answer, and nothing downstream can tell. This decoder has been wrong
//! before -- its own comment records that a prior scheme "applied the wrong scale to the
//! wrong 2-bit lanes -> corrupt weights -> broken Q2_K inference". So its agreement with
//! an independent implementation is measured first, element by element.

/// ONE-SHOT PROOF over every Q2_K tensor, NOT a regular test (hence `#[ignore]`).
/// Reads `<name>.bin` (raw Q2_K bytes, written by gguf-py's own reader so no offset
/// convention can enter) from `$Q2K_PROBE_DIR`, decodes with `dequantize_q2_k`, and
/// writes `<name>.apr.f32` for an element-wise comparison against gguf-py's
/// `<name>.ref.f32`. It says so when it has nothing to do, rather than passing silently.
#[test]
#[ignore = "one-shot probe: needs Q2K_PROBE_DIR from the gguf-py dump"]
fn probe_dump_q2_k_decodes_for_gguf_py_comparison_3960() {
    let Ok(dir) = std::env::var("Q2K_PROBE_DIR") else {
        panic!("PROBE: Q2K_PROBE_DIR is unset -- this probe did NOT run");
    };
    let mut done = 0usize;
    for entry in std::fs::read_dir(&dir).expect("probe dir") {
        let path = entry.expect("dir entry").path();
        if path.extension().and_then(|e| e.to_str()) != Some("bin") {
            continue;
        }
        let bytes = std::fs::read(&path).expect("read bin");
        let out = super::dequant::dequantize_q2_k(&bytes).expect("dequantize_q2_k");
        let mut buf = Vec::with_capacity(out.len() * 4);
        for v in &out {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        let dst = path.with_extension("apr.f32");
        std::fs::write(&dst, &buf).expect("write apr.f32");
        eprintln!(
            "PROBE {}: {} bytes -> {} values",
            path.display(),
            bytes.len(),
            out.len()
        );
        done += 1;
    }
    assert!(
        done > 0,
        "PROBE: no .bin files in {dir} -- nothing was compared"
    );
}

// ------------------------------------------------------------------------------------
// HERMETIC GOLDEN. The one-shot probe above compared all 11,010,048 values of the three
// Q2_K tensors in Qwen3.5-0.8B-UD-IQ2_XXS against gguf-py (llama.cpp @ df03399) and
// found every one BITWISE identical -- and a planted copy of this decoder's own
// historical bug (the wrong scale on the wrong 2-bit lanes) turned ~49% of them wrong,
// so that comparison can fail. This golden keeps the proof without Python: 12 real
// super-blocks (first, second, middle, last of each tensor) and gguf-py's values for them.
// ------------------------------------------------------------------------------------

const GOLDEN_BLOCKS: &[u8] = include_bytes!("fixtures/q2k_gguf_py_blocks.bin");
const GOLDEN_EXPECTED: &[u8] = include_bytes!("fixtures/q2k_gguf_py_expected.f32");
const Q2K_BLOCK_BYTES: usize = 84;

fn golden_expected() -> Vec<f32> {
    GOLDEN_EXPECTED
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

#[test]
fn the_q2_k_decoder_is_bitwise_identical_to_gguf_py_3960() {
    assert_eq!(
        GOLDEN_BLOCKS.len(),
        12 * Q2K_BLOCK_BYTES,
        "fixture: 12 super-blocks of 84 bytes"
    );
    let want = golden_expected();
    assert_eq!(want.len(), 12 * 256);
    let got = super::dequant::dequantize_q2_k(GOLDEN_BLOCKS).expect("decode golden blocks");
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "value {i} (block {}, lane {}): aprender {g} vs gguf-py {w}",
            i / 256,
            i % 256
        );
    }
    // Not a vacuous golden: real weights, both signs, non-trivial magnitude.
    assert!(want.iter().filter(|v| **v != 0.0).count() > want.len() * 9 / 10);
    assert!(want.iter().any(|v| *v < 0.0) && want.iter().any(|v| *v > 0.0));
}

// ------------------------------------------------------------------------------------
// THE KERNEL'S INDEX MATH, reproduced in Rust and proven against the gguf-py-proven
// decoder BEFORE any PTX is written. The GPU kernel will run one warp per output row;
// lane L (0..32) owns the 8 consecutive outputs 8L..8L+7 of each 256-value super-block.
//
// ggml's order: output o = 128g + 32s + 16h + i  (g 0..2, s 0..4, h 0..2, i 0..16),
//   q byte  = qs[32g + 16h + i], shifted by 2s;  scale byte = scales[8g + 2s + h].
// For o = 8L + j (j 0..8): 8 consecutive outputs never cross a 16-wide half, so
//   g = L/16,  s = (L%16)/4,  h = (L%4)/2,  i = 8(L%2) + j
// -- ONE scale byte and ONE shift per lane per block, and 8 CONTIGUOUS qs bytes, at
//   qs offset 32g + 16h + 8(L%2).
// ------------------------------------------------------------------------------------

/// Decode one super-block exactly as a GPU lane will, lane by lane.
pub(crate) fn q2_k_block_by_lanes(block: &[u8]) -> [f32; 256] {
    let f16 = |lo: u8, hi: u8| half::f16::from_le_bytes([lo, hi]).to_f32();
    let (d, dmin) = (f16(block[80], block[81]), f16(block[82], block[83]));
    let mut out = [0.0f32; 256];
    for lane in 0..32usize {
        let (g, s, h) = (lane / 16, (lane % 16) / 4, (lane % 4) / 2);
        let sc = block[8 * g + 2 * s + h];
        let dl = d * f32::from(sc & 0x0F);
        let ml = dmin * f32::from(sc >> 4);
        let qbase = 16 + 32 * g + 16 * h + 8 * (lane % 2);
        for j in 0..8 {
            let q = (block[qbase + j] >> (2 * s)) & 0x03;
            // Product then difference, as the reference. NOTE this is NOT a place where fma
            // and two roundings can differ: d is f16 (11 significant bits), times a 4-bit
            // nibble and a 2-bit q, is at most 17 bits -- exact in f32's 24 -- so `dl * q`
            // is exact and there is one rounding either way. A mutant replacing this with
            // `mul_add` SURVIVES, and it is an EQUIVALENT mutant, not a gap in the test.
            out[8 * lane + j] = dl * f32::from(q) - ml;
        }
    }
    out
}

#[test]
fn the_kernel_lane_mapping_reproduces_the_decoder_bitwise_3960() {
    let want = golden_expected();
    for (b, blk) in GOLDEN_BLOCKS.chunks_exact(Q2K_BLOCK_BYTES).enumerate() {
        let got = q2_k_block_by_lanes(blk);
        for (lane_j, (g, w)) in got.iter().zip(&want[b * 256..(b + 1) * 256]).enumerate() {
            assert_eq!(
                g.to_bits(),
                w.to_bits(),
                "block {b} output {lane_j} (lane {}, j {}): lane-math {g} vs gguf-py {w}",
                lane_j / 8,
                lane_j % 8
            );
        }
    }
}

/// The geometry the PTX will hardcode. `qs` words are loaded as u32: legal only if every
/// lane's 8-byte run starts 4-aligned. A block is 84 = 4*21 bytes and a row is a whole
/// number of blocks, so block bases are 4-aligned; the run starts at 16 + 8k.
#[test]
fn q2_k_geometry_the_kernel_relies_on_3960() {
    assert_eq!(Q2K_BLOCK_BYTES, 16 + 64 + 2 + 2, "scales[16] qs[64] d dmin");
    assert_eq!(
        Q2K_BLOCK_BYTES % 4,
        0,
        "block bases must stay 4-aligned for u32 qs loads"
    );
    for lane in 0..32usize {
        let (g, h) = (lane / 16, (lane % 4) / 2);
        let off = 16 + 32 * g + 16 * h + 8 * (lane % 2);
        assert_eq!(off % 4, 0, "lane {lane}: qs run at +{off} is not 4-aligned");
        assert!(off + 8 <= 80, "lane {lane}: qs run overlaps d/dmin");
    }
    // Each scale byte is used by exactly 2 lanes (16 outputs); all 16 are used.
    let mut uses = [0u32; 16];
    for lane in 0..32usize {
        uses[8 * (lane / 16) + 2 * ((lane % 16) / 4) + (lane % 4) / 2] += 1;
    }
    assert!(
        uses.iter().all(|u| *u == 2),
        "scale-byte use per lane mapping: {uses:?}"
    );
}
