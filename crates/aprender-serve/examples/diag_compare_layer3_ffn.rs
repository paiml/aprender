//! §32.4 layer-3 byte-compare: APR vs GGUF Q4K bytes for ffn_gate, ffn_up,
//! ffn_down at LAYER 3 (the divergence layer per existing trace).
//!
//! If APR bytes ≡ GGUF bytes → bug is downstream of weight loading
//!                              (kernel divergence or trace-mismatch).
//! If APR bytes != GGUF bytes → CONVERTER bug at layer-3 weights.

use realizar::apr_transformer::AprTransformer;
use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

fn byte_compare(label: &str, apr: &[u8], gguf: &[u8]) {
    println!(
        "\n=== {} ===  APR len={} bytes, GGUF len={} bytes",
        label,
        apr.len(),
        gguf.len()
    );
    if apr.len() != gguf.len() {
        println!("  ⚠ LENGTH MISMATCH — formats interpret weight differently");
        return;
    }
    let mut diff_count = 0;
    let mut first_diff_idx = None;
    for (i, (&a, &g)) in apr.iter().zip(gguf.iter()).enumerate() {
        if a != g {
            if first_diff_idx.is_none() {
                first_diff_idx = Some(i);
            }
            diff_count += 1;
        }
    }
    if diff_count == 0 {
        println!("  ✓ APR ≡ GGUF byte-for-byte ({} bytes match)", apr.len());
    } else {
        let pct = (diff_count as f64 / apr.len() as f64) * 100.0;
        println!(
            "  ⚠ {} bytes differ ({:.2}% of total), first at offset {:?}",
            diff_count, pct, first_diff_idx
        );
        if let Some(i) = first_diff_idx {
            let lo = i.saturating_sub(8);
            let hi = (i + 16).min(apr.len());
            println!("  APR[{}..{}]:  {:?}", lo, hi, &apr[lo..hi]);
            println!("  GGUF[{}..{}]: {:?}", lo, hi, &gguf[lo..hi]);
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let apr_path = "/mnt/nvme-raid0/models/ship-two-001/qwen2.5-coder-7b-instruct-q4k.apr";
    let gguf_path = "/mnt/nvme-raid0/models/ship-two-001/qwen2.5-coder-7b-instruct-q4k.gguf";
    println!("Loading APR:  {}", apr_path);
    let apr = AprTransformer::from_apr_file(apr_path)?;
    println!("Loading GGUF: {}", gguf_path);
    let mapped = MappedGGUFModel::from_path(gguf_path)?;
    let gguf = OwnedQuantizedModel::from_mapped(&mapped)?;

    let layer_idx = 3;
    println!("\n>>> Comparing LAYER {} Q4K weight bytes <<<", layer_idx);

    // APR: q4k_layers[3] gives Q4K bytes
    let apr_q4k = apr
        .q4k_layers
        .as_ref()
        .ok_or("q4k_layers None")?
        .get(layer_idx)
        .ok_or("layer 3 missing")?;

    // GGUF: layers()[3] gives owned quantized layer
    let gguf_layer = &gguf.layers()[layer_idx];

    // Compare ffn_gate
    let apr_gate = apr_q4k
        .ffn_gate_weight
        .as_ref()
        .ok_or("APR layer3 ffn_gate_weight None")?;
    let gguf_gate = gguf_layer
        .ffn_gate_weight
        .as_ref()
        .ok_or("GGUF layer3 ffn_gate_weight None")?;
    byte_compare("ffn_gate.weight Q4K", apr_gate, &gguf_gate.data);

    // Compare ffn_up
    let apr_up = apr_q4k
        .ffn_up_weight
        .as_ref()
        .ok_or("APR layer3 ffn_up_weight None")?;
    byte_compare("ffn_up.weight Q4K", apr_up, &gguf_layer.ffn_up_weight.data);

    // Compare ffn_down
    let apr_down = apr_q4k
        .ffn_down_weight
        .as_ref()
        .ok_or("APR layer3 ffn_down_weight None")?;
    byte_compare(
        "ffn_down.weight Q4K",
        apr_down,
        &gguf_layer.ffn_down_weight.data,
    );

    // For comparison, also compare layer 0 (which we know works fine)
    println!("\n\n>>> Sanity: LAYER 0 ffn_gate (should be OK if §30/§31 chain holds) <<<");
    let apr_q4k_l0 = apr
        .q4k_layers
        .as_ref()
        .ok_or("q4k_layers None")?
        .first()
        .ok_or("layer 0 missing")?;
    let gguf_layer_0 = &gguf.layers()[0];
    let apr_gate0 = apr_q4k_l0
        .ffn_gate_weight
        .as_ref()
        .ok_or("APR layer0 ffn_gate_weight None")?;
    byte_compare(
        "layer0 ffn_gate.weight Q4K",
        apr_gate0,
        &gguf_layer_0
            .ffn_gate_weight
            .as_ref()
            .ok_or("GGUF layer0 ffn_gate None")?
            .data,
    );

    println!("\n=== INTERPRETATION ===");
    println!("If layer-3 ffn_gate bytes match: bug is NOT in the converter — it's in the");
    println!("forward path (likely a layer-3-specific code path or a trace-capture site).");
    println!("If layer-3 ffn_gate bytes differ: bug IS in the GGUF→APR converter at layer 3.");
    Ok(())
}
