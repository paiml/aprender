
// ============================================================================
// Output Functions
// ============================================================================

fn output_json(report: &DiffReport) {
    let json_result = DiffResultJson::from(report);
    if let Ok(json) = serde_json::to_string_pretty(&json_result) {
        println!("{json}");
    }
}

fn output_text(report: &DiffReport, show_weights: bool) {
    output::header("Model Diff");

    let format_info = if report.same_format() {
        report.format1.clone()
    } else {
        format!("{} vs {}", report.format1, report.format2)
    };

    println!(
        "{}",
        output::kv_table(&[
            ("File A", report.path1.clone()),
            ("File B", report.path2.clone()),
            ("Format", format_info),
        ])
    );
    println!();

    if report.is_identical() {
        println!(
            "  {}",
            output::badge_pass("Models are IDENTICAL in structure and metadata")
        );
    } else {
        let count = report.diff_count();
        println!(
            "  {} {} differences found",
            output::badge_warn("DIFF"),
            count
        );
        println!();

        // Build diff table
        let mut rows: Vec<Vec<String>> = Vec::new();
        for category in [
            DiffCategory::Format,
            DiffCategory::Size,
            DiffCategory::Quantization,
            DiffCategory::Metadata,
            DiffCategory::Tensor,
        ] {
            let diffs = report.differences_by_category(category);
            for diff in diffs {
                rows.push(vec![
                    category.name().to_string(),
                    diff.field.clone(),
                    diff.value1.clone(),
                    diff.value2.clone(),
                ]);
            }
        }
        if !rows.is_empty() {
            println!(
                "{}",
                output::table(&["Category", "Field", "File A", "File B"], &rows)
            );
        }
    }

    if show_weights {
        println!();
        println!(
            "  {} Use --values to compare actual tensor values",
            output::badge_info("TIP")
        );
    }
}
