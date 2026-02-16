pub(crate) use super::*;

pub(super) fn sample_df() -> DataFrame {
    let columns = vec![
        ("a".to_string(), Vector::from_slice(&[1.0, 2.0, 3.0])),
        ("b".to_string(), Vector::from_slice(&[4.0, 5.0, 6.0])),
        ("c".to_string(), Vector::from_slice(&[7.0, 8.0, 9.0])),
    ];
    DataFrame::new(columns)
        .expect("sample_df should create valid DataFrame with equal-length columns")
}

#[test]
fn test_new() {
    let df = sample_df();
    assert_eq!(df.shape(), (3, 3));
    assert_eq!(df.n_rows(), 3);
    assert_eq!(df.n_cols(), 3);
}

#[test]
fn test_new_empty_error() {
    let result = DataFrame::new(vec![]);
    assert!(result.is_err());
}

#[test]
fn test_new_mismatched_lengths_error() {
    let columns = vec![
        ("a".to_string(), Vector::from_slice(&[1.0, 2.0, 3.0])),
        ("b".to_string(), Vector::from_slice(&[4.0, 5.0])),
    ];
    let result = DataFrame::new(columns);
    assert!(result.is_err());
}

#[test]
fn test_new_duplicate_names_error() {
    let columns = vec![
        ("a".to_string(), Vector::from_slice(&[1.0, 2.0])),
        ("a".to_string(), Vector::from_slice(&[3.0, 4.0])),
    ];
    let result = DataFrame::new(columns);
    assert!(result.is_err());
}

#[test]
fn test_new_empty_name_error() {
    let columns = vec![(String::new(), Vector::from_slice(&[1.0, 2.0]))];
    let result = DataFrame::new(columns);
    assert!(result.is_err());
}

#[test]
fn test_column_names() {
    let df = sample_df();
    let names = df.column_names();
    assert_eq!(names, vec!["a", "b", "c"]);
}

#[test]
fn test_column() {
    let df = sample_df();
    let col = df
        .column("b")
        .expect("column 'b' should exist in sample_df");
    assert_eq!(col.len(), 3);
    assert!((col[0] - 4.0).abs() < 1e-6);
    assert!((col[1] - 5.0).abs() < 1e-6);
    assert!((col[2] - 6.0).abs() < 1e-6);
}

#[test]
fn test_column_not_found() {
    let df = sample_df();
    let result = df.column("z");
    assert!(result.is_err());
}

#[test]
fn test_select() {
    let df = sample_df();
    let selected = df
        .select(&["a", "c"])
        .expect("select should succeed with existing column names");
    assert_eq!(selected.shape(), (3, 2));
    assert_eq!(selected.column_names(), vec!["a", "c"]);
}

#[test]
fn test_select_empty_error() {
    let df = sample_df();
    let result = df.select(&[]);
    assert!(result.is_err());
}

#[test]
fn test_select_not_found_error() {
    let df = sample_df();
    let result = df.select(&["a", "z"]);
    assert!(result.is_err());
}

#[test]
fn test_row() {
    let df = sample_df();
    let row = df
        .row(1)
        .expect("row index 1 should be valid for 3-row DataFrame");
    assert_eq!(row.len(), 3);
    assert!((row[0] - 2.0).abs() < 1e-6);
    assert!((row[1] - 5.0).abs() < 1e-6);
    assert!((row[2] - 8.0).abs() < 1e-6);
}

#[test]
fn test_row_out_of_bounds() {
    let df = sample_df();
    let result = df.row(10);
    assert!(result.is_err());
}

#[test]
fn test_to_matrix() {
    let df = sample_df();
    let matrix = df.to_matrix();
    assert_eq!(matrix.shape(), (3, 3));

    // Row 0: [1, 4, 7]
    assert!((matrix.get(0, 0) - 1.0).abs() < 1e-6);
    assert!((matrix.get(0, 1) - 4.0).abs() < 1e-6);
    assert!((matrix.get(0, 2) - 7.0).abs() < 1e-6);

    // Row 1: [2, 5, 8]
    assert!((matrix.get(1, 0) - 2.0).abs() < 1e-6);
    assert!((matrix.get(1, 1) - 5.0).abs() < 1e-6);
    assert!((matrix.get(1, 2) - 8.0).abs() < 1e-6);
}

#[test]
fn test_add_column() {
    let mut df = sample_df();
    let new_col = Vector::from_slice(&[10.0, 11.0, 12.0]);
    df.add_column("d".to_string(), new_col)
        .expect("add_column should succeed with matching length");

    assert_eq!(df.n_cols(), 4);
    let col = df
        .column("d")
        .expect("column 'd' should exist after add_column");
    assert!((col[0] - 10.0).abs() < 1e-6);
}

#[test]
fn test_add_column_wrong_length() {
    let mut df = sample_df();
    let new_col = Vector::from_slice(&[10.0, 11.0]);
    let result = df.add_column("d".to_string(), new_col);
    assert!(result.is_err());
}

#[test]
fn test_add_column_duplicate_name() {
    let mut df = sample_df();
    let new_col = Vector::from_slice(&[10.0, 11.0, 12.0]);
    let result = df.add_column("a".to_string(), new_col);
    assert!(result.is_err());
}

#[test]
fn test_add_column_empty_name() {
    let mut df = sample_df();
    let new_col = Vector::from_slice(&[10.0, 11.0, 12.0]);
    let result = df.add_column(String::new(), new_col);
    assert!(result.is_err());
}

#[test]
fn test_drop_column() {
    let mut df = sample_df();
    df.drop_column("b")
        .expect("drop_column should succeed for existing column 'b'");

    assert_eq!(df.n_cols(), 2);
    assert!(df.column("b").is_err());
}

#[test]
fn test_drop_column_not_found() {
    let mut df = sample_df();
    let result = df.drop_column("z");
    assert!(result.is_err());
}

#[test]
fn test_drop_last_column_error() {
    let columns = vec![("a".to_string(), Vector::from_slice(&[1.0, 2.0]))];
    let mut df = DataFrame::new(columns)
        .expect("DataFrame creation should succeed with single valid column");
    let result = df.drop_column("a");
    assert!(result.is_err());
}

#[test]
fn test_describe() {
    let columns = vec![(
        "x".to_string(),
        Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]),
    )];
    let df = DataFrame::new(columns)
        .expect("DataFrame creation should succeed with valid 5-element column");
    let stats = df.describe();

    assert_eq!(stats.len(), 1);
    assert_eq!(stats[0].name, "x");
    assert_eq!(stats[0].count, 5);
    assert!((stats[0].mean - 3.0).abs() < 1e-6);
    assert!((stats[0].min - 1.0).abs() < 1e-6);
    assert!((stats[0].max - 5.0).abs() < 1e-6);
    assert!((stats[0].median - 3.0).abs() < 1e-6);
}

#[test]
fn test_iter_columns() {
    let df = sample_df();
    let cols: Vec<_> = df.iter_columns().collect();
    assert_eq!(cols.len(), 3);
    assert_eq!(cols[0].0, "a");
    assert_eq!(cols[1].0, "b");
    assert_eq!(cols[2].0, "c");
}

#[test]
fn test_select_preserves_property() {
    // Property: select(names).column(name) == original.column(name)
    let df = sample_df();
    let selected = df
        .select(&["a", "c"])
        .expect("select should succeed with existing columns");

    let orig_a = df
        .column("a")
        .expect("column 'a' should exist in original DataFrame");
    let sel_a = selected
        .column("a")
        .expect("column 'a' should exist in selected DataFrame");

    assert_eq!(orig_a.len(), sel_a.len());
    for i in 0..orig_a.len() {
        assert!((orig_a[i] - sel_a[i]).abs() < 1e-6);
    }
}

#[test]
fn test_to_matrix_column_count() {
    // Property: to_matrix().n_cols() == n_selected_columns
    let df = sample_df();
    let selected = df
        .select(&["a", "b"])
        .expect("select should succeed with existing columns 'a' and 'b'");
    let matrix = selected.to_matrix();
    assert_eq!(matrix.n_cols(), 2);
}

#[test]
fn test_describe_median_even_length() {
    // Test median calculation for even-length arrays
    // Median of [1, 2, 3, 4] = (2 + 3) / 2 = 2.5
    let columns = vec![("x".to_string(), Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]))];
    let df = DataFrame::new(columns)
        .expect("DataFrame creation should succeed with valid 4-element column");
    let stats = df.describe();

    // This catches mutations in:
    // - sorted.len() % 2 == 0 (% vs + or /)
    // - sorted[sorted.len() / 2 - 1] (index calculation)
    // - + sorted[sorted.len() / 2] (sum of middle values)
    // - / 2.0 (averaging)
    assert!(
        (stats[0].median - 2.5).abs() < 1e-6,
        "Expected median 2.5, got {}",
        stats[0].median
    );
}

#[test]
fn test_describe_median_odd_length() {
    // Test median calculation for odd-length arrays
    // Median of [1, 2, 3] = 2.0 (middle element)
    let columns = vec![("x".to_string(), Vector::from_slice(&[1.0, 2.0, 3.0]))];
    let df = DataFrame::new(columns)
        .expect("DataFrame creation should succeed with valid 3-element column");
    let stats = df.describe();

    // For odd length, median = sorted[len / 2] = sorted[1] = 2.0
    assert!(
        (stats[0].median - 2.0).abs() < 1e-6,
        "Expected median 2.0, got {}",
        stats[0].median
    );
}

#[test]
fn test_describe_median_two_elements() {
    // Test median with exactly 2 elements
    // Median of [10, 20] = (10 + 20) / 2 = 15
    let columns = vec![("x".to_string(), Vector::from_slice(&[10.0, 20.0]))];
    let df = DataFrame::new(columns)
        .expect("DataFrame creation should succeed with valid 2-element column");
    let stats = df.describe();

    // This catches mutations in median averaging
    assert!(
        (stats[0].median - 15.0).abs() < 1e-6,
        "Expected median 15.0, got {}",
        stats[0].median
    );
}

#[test]
fn test_describe_median_arithmetic_mutations() {
    // Test to catch specific arithmetic mutations
    // Using values where wrong operations give different results
    // [2, 4, 6, 8]: median = (4 + 6) / 2 = 5.0
    let columns = vec![("x".to_string(), Vector::from_slice(&[2.0, 4.0, 6.0, 8.0]))];
    let df = DataFrame::new(columns)
        .expect("DataFrame creation should succeed with valid 4-element column");
    let stats = df.describe();

    // If + becomes - in median sum: (4 - 6) / 2 = -1
    // If / 2.0 becomes * 2.0: (4 + 6) * 2 = 20
    // If / 2 - 1 becomes / 2 + 1: would access wrong index
    assert!(
        (stats[0].median - 5.0).abs() < 1e-6,
        "Expected median 5.0, got {}",
        stats[0].median
    );
    assert!(
        stats[0].median > 0.0,
        "Median should be positive, got {}",
        stats[0].median
    );
    assert!(
        stats[0].median < 10.0,
        "Median should be < 10, got {}",
        stats[0].median
    );
}

#[test]
fn test_describe_median_unsorted_input() {
    // Verify median calculation sorts data correctly
    // Input [5, 1, 3, 2, 4] -> sorted [1, 2, 3, 4, 5] -> median = 3
    let columns = vec![(
        "x".to_string(),
        Vector::from_slice(&[5.0, 1.0, 3.0, 2.0, 4.0]),
    )];
    let df = DataFrame::new(columns)
        .expect("DataFrame creation should succeed with valid 5-element unsorted column");
    let stats = df.describe();

    assert!(
        (stats[0].median - 3.0).abs() < 1e-6,
        "Expected median 3.0, got {}",
        stats[0].median
    );
}

#[test]
fn test_describe_six_elements() {
    // Test with 6 elements to ensure index math is correct
    // [1, 2, 3, 4, 5, 6]: median = (3 + 4) / 2 = 3.5
    // len = 6, len/2 = 3, len/2 - 1 = 2
    // sorted[2] = 3, sorted[3] = 4
    let columns = vec![(
        "x".to_string(),
        Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    )];
    let df = DataFrame::new(columns)
        .expect("DataFrame creation should succeed with valid 6-element column");
    let stats = df.describe();

    assert!(
        (stats[0].median - 3.5).abs() < 1e-6,
        "Expected median 3.5, got {}",
        stats[0].median
    );
}
