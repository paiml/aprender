use super::*;
use std::path::PathBuf;

fn test_path() -> PathBuf {
    PathBuf::from("test.tsp")
}

#[test]
fn test_parse_simple_tsplib() {
    let content = r#"
NAME: test
TYPE: TSP
COMMENT: A simple test
DIMENSION: 3
EDGE_WEIGHT_TYPE: EUC_2D
NODE_COORD_SECTION
1 0.0 0.0
2 3.0 0.0
3 3.0 4.0
EOF
"#;

    let instance = TsplibParser::parse(content, &test_path()).expect("should parse");

    assert_eq!(instance.name, "test");
    assert_eq!(instance.dimension, 3);
    assert_eq!(instance.comment, Some("A simple test".into()));
    assert!(instance.coords.is_some());
    assert_eq!(instance.coords.as_ref().map(|c| c.len()), Some(3));
}

#[test]
fn test_parse_computes_distances() {
    let content = r#"
NAME: triangle
DIMENSION: 3
EDGE_WEIGHT_TYPE: EUC_2D
NODE_COORD_SECTION
1 0.0 0.0
2 3.0 0.0
3 3.0 4.0
EOF
"#;

    let instance = TsplibParser::parse(content, &test_path()).expect("should parse");

    // 3-4-5 triangle
    assert!((instance.distance(0, 1) - 3.0).abs() < 1e-10);
    assert!((instance.distance(1, 2) - 4.0).abs() < 1e-10);
    assert!((instance.distance(0, 2) - 5.0).abs() < 1e-10);
}

#[test]
fn test_parse_ceil_2d() {
    let content = r#"
NAME: ceil_test
DIMENSION: 2
EDGE_WEIGHT_TYPE: CEIL_2D
NODE_COORD_SECTION
1 0.0 0.0
2 1.5 0.0
EOF
"#;

    let instance = TsplibParser::parse(content, &test_path()).expect("should parse");

    // Distance 1.5 should be ceiling'd to 2.0
    assert!((instance.distance(0, 1) - 2.0).abs() < 1e-10);
}

#[test]
fn test_parse_missing_dimension() {
    let content = r#"
NAME: test
NODE_COORD_SECTION
1 0.0 0.0
EOF
"#;

    let result = TsplibParser::parse(content, &test_path());
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("DIMENSION"));
}

#[test]
fn test_parse_no_coords_or_weights() {
    let content = r#"
NAME: test
DIMENSION: 3
EDGE_WEIGHT_TYPE: EUC_2D
EOF
"#;

    let result = TsplibParser::parse(content, &test_path());
    assert!(result.is_err());
}

#[test]
fn test_parse_invalid_coordinate() {
    let content = r#"
NAME: test
DIMENSION: 2
EDGE_WEIGHT_TYPE: EUC_2D
NODE_COORD_SECTION
1 0.0 0.0
2 abc 0.0
EOF
"#;

    let result = TsplibParser::parse(content, &test_path());
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("Invalid x coordinate"));
}

#[test]
fn test_parse_explicit_weights() {
    let content = r#"
NAME: explicit_test
DIMENSION: 3
EDGE_WEIGHT_TYPE: EXPLICIT
EDGE_WEIGHT_SECTION
10
20 30
EOF
"#;

    let instance = TsplibParser::parse(content, &test_path()).expect("should parse");

    // Lower triangular: [1,0]=10, [2,0]=20, [2,1]=30
    assert!((instance.distance(1, 0) - 10.0).abs() < 1e-10);
    assert!((instance.distance(2, 0) - 20.0).abs() < 1e-10);
    assert!((instance.distance(2, 1) - 30.0).abs() < 1e-10);
}

#[test]
fn test_parse_unsupported_edge_type() {
    let content = r#"
NAME: test
DIMENSION: 3
EDGE_WEIGHT_TYPE: UNKNOWN_TYPE
NODE_COORD_SECTION
1 0.0 0.0
EOF
"#;

    let result = TsplibParser::parse(content, &test_path());
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("Unsupported edge weight type"));
}

#[test]
fn test_att_distance() {
    let content = r#"
NAME: att_test
DIMENSION: 2
EDGE_WEIGHT_TYPE: ATT
NODE_COORD_SECTION
1 0.0 0.0
2 100.0 0.0
EOF
"#;

    let instance = TsplibParser::parse(content, &test_path()).expect("should parse");

    // ATT distance should be different from Euclidean
    let dist = instance.distance(0, 1);
    assert!(dist > 0.0);
}

#[test]
fn test_att_distance_matches_tsplib() {
    // Regression test: ATT formula must have division INSIDE sqrt
    // TSPLIB formula: rij = sqrt((xd*xd + yd*yd) / 10.0)
    // Using att48 city 1 (6734, 1453) and city 2 (2233, 10)
    let content = r#"
NAME: att48_subset
DIMENSION: 2
EDGE_WEIGHT_TYPE: ATT
NODE_COORD_SECTION
1 6734 1453
2 2233 10
EOF
"#;

    let instance = TsplibParser::parse(content, &test_path()).expect("should parse");

    // Expected calculation:
    // xd = 6734 - 2233 = 4501
    // yd = 1453 - 10 = 1443
    // r = sqrt((4501^2 + 1443^2) / 10) = sqrt(22341250 / 10) = sqrt(2234125) = 1494.69
    // nint(1494.69) = 1495, and 1495 > 1494.69 so dij = 1495
    let dist = instance.distance(0, 1);
    assert!(
        (dist - 1495.0).abs() < 1.0,
        "ATT distance should be ~1495 (TSPLIB verified), got {}",
        dist
    );
}

#[test]
fn test_name_defaults_to_filename() {
    let content = r#"
DIMENSION: 2
EDGE_WEIGHT_TYPE: EUC_2D
NODE_COORD_SECTION
1 0.0 0.0
2 1.0 0.0
EOF
"#;

    let instance =
        TsplibParser::parse(content, &PathBuf::from("my_instance.tsp")).expect("should parse");
    assert_eq!(instance.name, "my_instance");
}

#[test]
fn test_extract_optimal_from_comment_optimal_tour() {
    let opt = TsplibParser::extract_optimal_from_comment("Optimal tour: 7542");
    assert_eq!(opt, Some(7542.0));
}

#[test]
fn test_extract_optimal_from_comment_best_known() {
    let opt = TsplibParser::extract_optimal_from_comment("Best known: 426");
    assert_eq!(opt, Some(426.0));
}

#[test]
fn test_extract_optimal_from_comment_parentheses() {
    let opt = TsplibParser::extract_optimal_from_comment("52 locations in Berlin (7542)");
    assert_eq!(opt, Some(7542.0));
}

#[test]
fn test_extract_optimal_from_comment_with_thousands() {
    let opt = TsplibParser::extract_optimal_from_comment("Optimal: 10,628");
    assert_eq!(opt, Some(10628.0));
}

#[test]
fn test_extract_optimal_from_comment_no_match() {
    let opt = TsplibParser::extract_optimal_from_comment("Just a plain comment");
    assert_eq!(opt, None);
}

#[test]
fn test_parse_with_optimal_in_comment() {
    let content = r#"
NAME: test_optimal
TYPE: TSP
COMMENT: Optimal tour: 100
DIMENSION: 3
EDGE_WEIGHT_TYPE: EUC_2D
NODE_COORD_SECTION
1 0.0 0.0
2 3.0 0.0
3 3.0 4.0
EOF
"#;

    let instance = TsplibParser::parse(content, &test_path()).expect("should parse");
    assert_eq!(instance.best_known, Some(100.0));
}

#[test]
fn test_parse_with_explicit_best_known_field() {
    let content = r#"
NAME: test_explicit
TYPE: TSP
BEST_KNOWN: 7542
DIMENSION: 3
EDGE_WEIGHT_TYPE: EUC_2D
NODE_COORD_SECTION
1 0.0 0.0
2 3.0 0.0
3 3.0 4.0
EOF
"#;

    let instance = TsplibParser::parse(content, &test_path()).expect("should parse");
    assert_eq!(instance.best_known, Some(7542.0));
}
