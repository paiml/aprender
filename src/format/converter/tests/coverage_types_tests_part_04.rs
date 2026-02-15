
#[test]
fn test_merge_report_struct() {
    let report = MergeReport {
        model_count: 3,
        strategy: MergeStrategy::Average,
        tensor_count: 100,
        output_size: 10000,
        weights_used: None,
    };
    assert_eq!(report.model_count, 3);
    assert_eq!(report.tensor_count, 100);
}
