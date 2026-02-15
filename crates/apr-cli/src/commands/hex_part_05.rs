
/// Print hex dump of tensor data
fn print_hex_dump(data: &[f32], limit: usize) {
    println!();
    println!(
        "{} (first {} bytes):",
        "Hex dump".bold(),
        (limit * 4).min(data.len() * 4)
    );

    let bytes_to_show = limit.min(data.len());
    for (i, chunk) in data
        .iter()
        .take(bytes_to_show)
        .collect::<Vec<_>>()
        .chunks(4)
        .enumerate()
    {
        print_hex_row(chunk, i * 16);
    }

    if data.len() > bytes_to_show {
        println!(
            "... {} more elements",
            (data.len() - bytes_to_show).to_string().dimmed()
        );
    }
}

// print_tensor_hex removed — v2 path uses inline hex dump in run_apr()

/// Compute basic statistics
fn compute_stats(data: &[f32]) -> (f32, f32, f32, f32) {
    if data.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }

    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0_f64;

    for &x in data {
        min = min.min(x);
        max = max.max(x);
        sum += f64::from(x);
    }

    let mean = (sum / data.len() as f64) as f32;

    let variance: f32 = (data
        .iter()
        .map(|&x| {
            let diff = f64::from(x) - f64::from(mean);
            diff * diff
        })
        .sum::<f64>()
        / data.len() as f64) as f32;

    let std = variance.sqrt();

    (min, max, mean, std)
}

/// Output as JSON (v2 reader)
#[allow(clippy::unnecessary_wraps)]
#[allow(clippy::disallowed_methods)]
fn output_json_v2(
    reader: &AprV2Reader,
    filtered: &[&str],
    limit: usize,
    show_stats: bool,
) -> Result<(), CliError> {
    use serde::Serialize;

    #[derive(Serialize)]
    struct TensorDump {
        name: String,
        shape: Vec<usize>,
        dtype: String,
        offset: u64,
        size_bytes: u64,
        #[serde(skip_serializing_if = "Option::is_none")]
        stats: Option<TensorStatsJson>,
        sample_values: Vec<f32>,
    }

    #[derive(Serialize)]
    struct TensorStatsJson {
        min: f32,
        max: f32,
        mean: f32,
        std: f32,
    }

    let mut results = Vec::new();

    for name in filtered {
        let entry = reader.get_tensor(name);
        let data = reader.get_tensor_as_f32(name);
        let stats = if show_stats {
            data.as_ref().map(|d| {
                let (min, max, mean, std) = compute_stats(d);
                TensorStatsJson {
                    min,
                    max,
                    mean,
                    std,
                }
            })
        } else {
            None
        };

        let sample_values: Vec<f32> = data
            .as_ref()
            .map(|d| d.iter().take(limit).copied().collect())
            .unwrap_or_default();

        if let Some(e) = entry {
            results.push(TensorDump {
                name: e.name.clone(),
                shape: e.shape.clone(),
                dtype: format!("{:?}", e.dtype),
                offset: e.offset,
                size_bytes: e.size,
                stats,
                sample_values,
            });
        }
    }

    if let Ok(json) = serde_json::to_string_pretty(&results) {
        println!("{json}");
    }

    Ok(())
}
