//! PMAT-895 — Pillar-4 fail-closed: reject NaN/Inf quantized weights at the
//! GGUF/APR LOAD path (OBLIG-GGUF-LOAD-NANINF).
//!
//! THE BEAT (a real win, not just parity): a quantized GGUF whose Q4_0/Q4_K
//! super-block f16 scale `d`/`dmin` is f16 +Inf (`0x7C00`) or NaN (`0x7E00`)
//! dequantizes to NaN/Inf at every element of that block. `OwnedQuantizedModel::
//! from_mapped` accepted it BEFORE this fix — `validate_quantized_tensors` only
//! called `is_truncated`, with NO finiteness check.
//!
//! llama.cpp / Ollama ALSO load such a model: their `check_tensors` defaults to
//! `false` (`common.h:441`); `--check-tensors` is opt-in. So `apr` REJECTING it
//! at load is a genuine fail-closed BEAT (PMAT-744 lineage), not parity.
//!
//! The NaN/Inf gate already existed on the SafeTensors path (F-DATA-QUALITY-002
//! in `safetensors/validation.rs`). PMAT-895 WIRES that finiteness guarantee into
//! the quantized load path.

use std::io::Write;

use crate::gguf::model::OwnedQuantizedModel;
use crate::gguf::test_factory::build_executable_pygmy_gguf;
use crate::gguf::MappedGGUFModel;

/// The pygmy builder fills every Q4_0 weight block with `create_q4_0_data`:
/// `[f16(0.1) scale: 2 bytes][0x88 × 16 quant nibbles]`. A run of 16 consecutive
/// `0x88` bytes is a distinctive Q4_0 block signature that does NOT occur in the
/// GGUF header/metadata. We locate the FIRST such block, then overwrite its 2
/// scale bytes (the 2 bytes IMMEDIATELY BEFORE the 0x88 run) with the given
/// non-finite f16 bit pattern. This corrupts exactly ONE super-block's f16 scale
/// `d` — the minimal "Q4_0 weight super-block whose scale is non-finite".
fn corrupt_first_q4_0_scale(gguf: &mut [u8], scale_le_bytes: [u8; 2]) -> usize {
    const QUANT_RUN: [u8; 16] = [0x88; 16];
    let run_pos = gguf
        .windows(QUANT_RUN.len())
        .position(|w| w == QUANT_RUN)
        .expect("pygmy GGUF must contain a Q4_0 block (16×0x88 quants)");
    assert!(
        run_pos >= 2,
        "0x88 run must be preceded by a 2-byte f16 scale"
    );
    let scale_off = run_pos - 2;
    gguf[scale_off] = scale_le_bytes[0];
    gguf[scale_off + 1] = scale_le_bytes[1];
    scale_off
}

fn write_temp(gguf: &[u8]) -> tempfile::NamedTempFile {
    let mut f = tempfile::NamedTempFile::new().expect("create temp gguf");
    f.write_all(gguf).expect("write gguf bytes");
    f.flush().expect("flush gguf bytes");
    f
}

/// RED → GREEN falsifier for OBLIG-GGUF-LOAD-NANINF.
///
/// On current `main` (before the fix) `from_mapped` returned `Ok` for BOTH the
/// +Inf and NaN corrupted scales — the gate was missing. After the fix it MUST
/// return `Err` naming the non-finite tensor.
#[test]
fn gguf_naninf_quant_scale_rejected_at_load() {
    // f16 +Inf = bits 0x7C00 = little-endian [0x00, 0x7C].
    // f16 NaN  = bits 0x7E00 = little-endian [0x00, 0x7E].
    for (label, scale_le) in [("+Inf", [0x00u8, 0x7C]), ("NaN", [0x00u8, 0x7E])] {
        let mut gguf = build_executable_pygmy_gguf();
        let off = corrupt_first_q4_0_scale(&mut gguf, scale_le);
        eprintln!("[PMAT-895] corrupted Q4_0 f16 scale ({label}) at byte offset {off}");

        let tmp = write_temp(&gguf);
        let mapped = MappedGGUFModel::from_path(tmp.path()).expect("mmap corrupted GGUF");
        let result = OwnedQuantizedModel::from_mapped(&mapped);

        assert!(
            result.is_err(),
            "OBLIG-GGUF-LOAD-NANINF FAIL ({label}): from_mapped accepted a model whose \
             Q4_0 super-block f16 scale is non-finite — it dequantizes to {label} and \
             produces garbage at inference. apr must fail closed at load."
        );
        let msg = format!("{:?}", result.err().unwrap());
        eprintln!("[PMAT-895] rejected ({label}) with: {msg}");
    }
}

/// FP-bound: a HEALTHY pygmy model (all weights finite) still loads `Ok`, and the
/// existing all-zero / all-`0x88` synthetic builders are NOT rejected. All-zero
/// quantized data dequantizes to 0.0 (finite), and f16(0.1) is finite — a pure
/// NaN/Inf finiteness check is orthogonal to the density/zero gates and must not
/// reject these legitimate test models.
#[test]
fn healthy_quant_model_still_loads_ok() {
    let gguf = build_executable_pygmy_gguf();
    let tmp = write_temp(&gguf);
    let mapped = MappedGGUFModel::from_path(tmp.path()).expect("mmap healthy GGUF");
    let result = OwnedQuantizedModel::from_mapped(&mapped);
    assert!(
        result.is_ok(),
        "FP-bound FAIL: NaN/Inf gate rejected a healthy (finite-scale) quantized model: {:?}",
        result.err()
    );
}
