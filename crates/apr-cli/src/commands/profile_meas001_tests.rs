// MEAS-001 R1 (#4522): the hand-built profile documents stay valid JSON with
// and without a resources{} block, and carry it when measured.
#[test]
fn meas001_resources_member_keeps_the_document_valid() {
    let block = serde_json::json!({"peak_rss_bytes": 1, "vram_peak_bytes": null});
    for resources in [Some(&block), None] {
        let mut doc = String::from("{\n  \"per_layer_us\": [1.00]");
        push_resources_json(&mut doc, resources);
        doc.push_str("}\n");
        let v: serde_json::Value = serde_json::from_str(&doc).expect(&doc);
        assert_eq!(v.get("resources"), resources, "{doc}");
    }
}

#[test]
fn meas001_ci_report_carries_the_measured_block() {
    let block = serde_json::json!({"peak_rss_bytes": 42});
    let results = RealProfileResults {
        resources: Some(block.clone()),
        ..Default::default()
    };
    let report = CiProfileReport::from_results(&results, &CiAssertions::default());
    assert_eq!(report.resources, Some(block));
}
