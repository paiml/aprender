use std::path::Path;

use provable_contracts::roofline::{self, Bottleneck, HardwareProfile};

/// Run the `pv roofline` command: compute performance ceilings from contract.
pub fn run(
    contract_dir: &Path,
    params: u64,
    bits: u32,
    hardware: &str,
    format: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let hw = match hardware {
        "apple-m" => HardwareProfile::apple_m_series(),
        "a100" => HardwareProfile::nvidia_a100(),
        _ => {
            return Err(
                format!("unknown hardware profile '{hardware}'. Use: apple-m, a100").into(),
            );
        }
    };

    if params == 0 {
        return Err("--params must be > 0 (model has no parameters)".into());
    }
    if bits == 0 {
        return Err("--bits must be > 0".into());
    }

    let desc = roofline::load_roofline_contract(contract_dir);
    let r = roofline::compute_roofline(params, bits, &hw);

    match format {
        "json" => print_json(&r, &hw, desc.as_deref()),
        _ => print_text(&r, &hw, hardware, params, bits, desc.as_deref()),
    }

    Ok(())
}

fn print_text(
    r: &roofline::RooflineCeiling,
    hw: &HardwareProfile,
    hw_name: &str,
    params: u64,
    bits: u32,
    desc: Option<&str>,
) {
    println!("Roofline Analysis ({})", r.contract_id);
    if let Some(d) = desc {
        println!("  {d}");
    }
    println!();
    println!("Hardware: {hw_name}");
    println!("  Bandwidth: {:.1} GB/s", hw.bandwidth_gb_s);
    println!("  Compute:   {:.1} GFLOPS", hw.compute_gflops);
    println!();
    println!("Model: {} Q{bits}", format_params(params));
    println!("  Size: {:.2} GB", r.model_bytes / 1e9);
    println!();
    println!("Ceilings:");
    println!("  BW ceiling:      {:.1} tok/s", r.bw_ceiling);
    println!("  Compute ceiling: {:.1} tok/s", r.compute_ceiling);
    println!("  Effective:       {:.1} tok/s", r.throughput_ceiling);
    println!();
    let marker = match r.bottleneck {
        Bottleneck::Bandwidth => "MEMORY-BOUND  (bw_ceiling < compute_ceiling)",
        Bottleneck::Compute => "COMPUTE-BOUND (compute_ceiling < bw_ceiling)",
    };
    println!("Bottleneck: {marker}");
}

fn print_json(r: &roofline::RooflineCeiling, hw: &HardwareProfile, desc: Option<&str>) {
    println!("{{");
    println!("  \"contract_id\": \"{}\",", r.contract_id);
    if let Some(d) = desc {
        println!("  \"description\": \"{d}\",");
    }
    println!("  \"model_bytes\": {:.0},", r.model_bytes);
    println!("  \"model_gb\": {:.4},", r.model_bytes / 1e9);
    println!("  \"bw_ceiling_tok_s\": {:.2},", r.bw_ceiling);
    println!("  \"compute_ceiling_tok_s\": {:.2},", r.compute_ceiling);
    println!(
        "  \"throughput_ceiling_tok_s\": {:.2},",
        r.throughput_ceiling
    );
    println!(
        "  \"bottleneck\": \"{}\",",
        match r.bottleneck {
            Bottleneck::Bandwidth => "bandwidth",
            Bottleneck::Compute => "compute",
        }
    );
    println!("  \"hardware\": {{");
    println!("    \"bandwidth_gb_s\": {:.1},", hw.bandwidth_gb_s);
    println!("    \"compute_gflops\": {:.1},", hw.compute_gflops);
    println!("    \"ops_per_token\": {:.1}", hw.ops_per_token);
    println!("  }}");
    println!("}}");
}

fn format_params(n: u64) -> String {
    #[allow(clippy::cast_precision_loss)]
    let f = n as f64;
    if f >= 1e12 {
        format!("{:.1}T", f / 1e12)
    } else if f >= 1e9 {
        format!("{:.1}B", f / 1e9)
    } else if f >= 1e6 {
        format!("{:.0}M", f / 1e6)
    } else {
        format!("{n}")
    }
}
