//! Tests for the GeoJSON and TopoJSON reader (`geo` feature).
//!
//! Expected rings are written out by hand. The GeoJSON and TopoJSON fixtures encode the same
//! three shapes: `p1` and `p2` share arc 0, `p2` uses it reversed (`-1`), and `p3` is a
//! MultiPolygon whose exterior and hole are each one arc. Every coordinate is a dyadic rational,
//! so the quantized decode is exact and the comparison is plain equality.

use super::reader::{read_geojson, read_topojson, Feature, GeoReadError, Geometry};
use std::collections::BTreeMap;

const GEOJSON: &str = include_str!("testdata/shapes.geojson");
const TOPOJSON: &str = include_str!("testdata/shapes.topojson");
const SQUARE_ARC: &str = "[[[0,0],[1,0],[1,1],[0,0]]]";

fn feature(id: &str, name: &str, geometry: Geometry) -> Feature {
    Feature {
        id: Some(id.to_string()),
        properties: BTreeMap::from([("name".to_string(), name.to_string())]),
        geometry,
    }
}

fn shapes() -> Vec<Feature> {
    let p1 = vec![vec![(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0), (0.0, 0.0)]];
    let p2 = vec![vec![(4.0, 0.0), (8.0, 0.0), (8.0, 4.0), (4.0, 4.0), (4.0, 0.0)]];
    let exterior = vec![(0.0, 6.0), (8.0, 6.0), (8.0, 10.0), (0.0, 10.0), (0.0, 6.0)];
    let hole = vec![(2.0, 7.0), (2.0, 9.0), (6.0, 9.0), (6.0, 7.0), (2.0, 7.0)];
    vec![
        feature("p1", "west", Geometry::Polygon(p1)),
        feature("p2", "east", Geometry::Polygon(p2)),
        feature("p3", "holed", Geometry::MultiPolygon(vec![vec![exterior, hole]])),
    ]
}

fn geo_feature(geometry: &str) -> String {
    format!(r#"{{"type":"Feature","geometry":{geometry}}}"#)
}

fn topo_doc(arcs: &str, geometry: &str) -> String {
    format!(r#"{{"type":"Topology","arcs":{arcs},"objects":{{"s":{geometry}}}}}"#)
}

fn quantized_doc(transform: &str, arcs: &str) -> String {
    format!(
        r#"{{"type":"Topology","transform":{transform},"arcs":{arcs},"objects":{{"s":{{"type":"Polygon","arcs":[[0]]}}}}}}"#
    )
}

#[test]
fn geojson_fixture_decodes_to_the_expected_shapes() {
    let features = read_geojson(GEOJSON).expect("fixture is valid GeoJSON");
    assert_eq!(features, shapes());
}

#[test]
fn topojson_fixture_decodes_to_the_same_rings_as_geojson() {
    let topo = read_topojson(TOPOJSON).expect("fixture is valid TopoJSON");
    assert_eq!(topo, shapes());
    assert_eq!(topo, read_geojson(GEOJSON).expect("fixture is valid GeoJSON"));
}

#[test]
fn topojson_without_transform_reads_absolute_positions() {
    let text = topo_doc(SQUARE_ARC, r#"{"type":"Polygon","arcs":[[0]]}"#);
    let features = read_topojson(&text).expect("absolute positions are valid");
    let square = vec![vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)]];
    assert_eq!(
        features,
        vec![Feature {
            id: None,
            properties: BTreeMap::new(),
            geometry: Geometry::Polygon(square),
        }]
    );
}

#[test]
fn geojson_single_feature_numeric_id_and_scalar_properties() {
    let text = r#"{"type":"Feature","id":7,"properties":{"n":3,"z":null,"s":"x"},"geometry":{"type":"Polygon","coordinates":[[[0,0],[1,0],[1,1],[0,0]]]}}"#;
    let features = read_geojson(text).expect("single feature is valid");
    let properties = BTreeMap::from([
        ("n".to_string(), "3".to_string()),
        ("s".to_string(), "x".to_string()),
        ("z".to_string(), String::new()),
    ]);
    let square = vec![vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 0.0)]];
    assert_eq!(
        features,
        vec![Feature {
            id: Some("7".to_string()),
            properties,
            geometry: Geometry::Polygon(square),
        }]
    );
}

#[test]
fn geojson_bad_json_is_a_json_error() {
    let err = read_geojson("{").expect_err("truncated JSON is refused");
    assert!(matches!(err, GeoReadError::Json(_)));
}

#[test]
fn geojson_top_level_array_has_no_type_member() {
    let err = read_geojson("[]").expect_err("an array is not a feature");
    assert_eq!(err, GeoReadError::Malformed("type"));
}

#[test]
fn geojson_unknown_top_level_type() {
    let err = read_geojson(r#"{"type":"Nope"}"#).expect_err("unknown type is refused");
    assert_eq!(err, GeoReadError::UnknownType("Nope".to_string()));
}

#[test]
fn geojson_feature_collection_without_features() {
    let err = read_geojson(r#"{"type":"FeatureCollection"}"#).expect_err("no features member");
    assert_eq!(err, GeoReadError::Malformed("features"));
}

#[test]
fn geojson_unknown_type_inside_collection() {
    let text = r#"{"type":"FeatureCollection","features":[{"type":"Nope"}]}"#;
    let err = read_geojson(text).expect_err("a non-Feature member is refused");
    assert_eq!(err, GeoReadError::UnknownType("Nope".to_string()));
}

#[test]
fn geojson_point_is_unsupported_not_dropped() {
    let text = geo_feature(r#"{"type":"Point","coordinates":[0,0]}"#);
    let err = read_geojson(&text).expect_err("points are refused, not skipped");
    assert_eq!(err, GeoReadError::UnsupportedGeometry("Point".to_string()));
}

#[test]
fn geojson_unknown_geometry_type() {
    let err = read_geojson(&geo_feature(r#"{"type":"Blob"}"#)).expect_err("unknown geometry");
    assert_eq!(err, GeoReadError::UnknownType("Blob".to_string()));
}

#[test]
fn geojson_feature_without_geometry() {
    let err = read_geojson(r#"{"type":"Feature"}"#).expect_err("geometry is required");
    assert_eq!(err, GeoReadError::Malformed("geometry"));
}

#[test]
fn geojson_non_scalar_id() {
    let text = r#"{"type":"Feature","id":true,"geometry":null}"#;
    let err = read_geojson(text).expect_err("a boolean id is refused");
    assert_eq!(err, GeoReadError::Malformed("id"));
}

#[test]
fn geojson_non_object_properties() {
    let text = r#"{"type":"Feature","properties":5,"geometry":null}"#;
    let err = read_geojson(text).expect_err("properties must be an object or null");
    assert_eq!(err, GeoReadError::Malformed("properties"));
}

#[test]
fn geojson_polygon_without_coordinates() {
    let err = read_geojson(&geo_feature(r#"{"type":"Polygon"}"#)).expect_err("no coordinates");
    assert_eq!(err, GeoReadError::Malformed("coordinates"));
}

#[test]
fn geojson_polygon_coordinates_not_array() {
    let text = geo_feature(r#"{"type":"Polygon","coordinates":5}"#);
    let err = read_geojson(&text).expect_err("coordinates must be an array");
    assert_eq!(err, GeoReadError::Malformed("coordinates"));
}

#[test]
fn geojson_ring_not_array() {
    let text = geo_feature(r#"{"type":"Polygon","coordinates":[5]}"#);
    let err = read_geojson(&text).expect_err("a ring must be an array");
    assert_eq!(err, GeoReadError::Malformed("ring"));
}

#[test]
fn geojson_position_not_pair() {
    let text = geo_feature(r#"{"type":"Polygon","coordinates":[[[0],[1,0],[1,1],[0,0]]]}"#);
    let err = read_geojson(&text).expect_err("a position needs two numbers");
    assert_eq!(err, GeoReadError::Malformed("position"));
}

#[test]
fn geojson_position_not_number() {
    let text = geo_feature(r#"{"type":"Polygon","coordinates":[[["x",0],[1,0],[1,1],[0,0]]]}"#);
    let err = read_geojson(&text).expect_err("a coordinate must be a number");
    assert_eq!(err, GeoReadError::Malformed("position"));
}

#[test]
fn geojson_ring_with_three_positions() {
    let text = geo_feature(r#"{"type":"Polygon","coordinates":[[[0,0],[1,0],[0,0]]]}"#);
    let err = read_geojson(&text).expect_err("a ring needs four positions");
    assert_eq!(err, GeoReadError::RingTooShort { positions: 3 });
}

#[test]
fn geojson_ring_not_closed() {
    let text = geo_feature(r#"{"type":"Polygon","coordinates":[[[0,0],[1,0],[1,1],[0,1]]]}"#);
    let err = read_geojson(&text).expect_err("a ring must close");
    assert_eq!(err, GeoReadError::RingNotClosed);
}

#[test]
fn topojson_root_not_topology() {
    let err = read_topojson(r#"{"type":"Feature"}"#).expect_err("root must be a Topology");
    assert_eq!(err, GeoReadError::UnknownType("Feature".to_string()));
}

#[test]
fn topojson_missing_arcs() {
    let err = read_topojson(r#"{"type":"Topology","objects":{}}"#).expect_err("no arcs member");
    assert_eq!(err, GeoReadError::Malformed("arcs"));
}

#[test]
fn topojson_missing_objects() {
    let err = read_topojson(r#"{"type":"Topology","arcs":[]}"#).expect_err("no objects member");
    assert_eq!(err, GeoReadError::Malformed("objects"));
}

#[test]
fn topojson_arc_not_array() {
    let text = topo_doc("[5]", r#"{"type":"Polygon","arcs":[[0]]}"#);
    let err = read_topojson(&text).expect_err("an arc must be an array");
    assert_eq!(err, GeoReadError::Malformed("arc"));
}

#[test]
fn topojson_arc_index_not_integer() {
    let text = topo_doc(SQUARE_ARC, r#"{"type":"Polygon","arcs":[[0.5]]}"#);
    let err = read_topojson(&text).expect_err("an arc index must be an integer");
    assert_eq!(err, GeoReadError::Malformed("arc index"));
}

#[test]
fn topojson_arc_index_out_of_range_forward_and_reversed() {
    let forward = topo_doc(SQUARE_ARC, r#"{"type":"Polygon","arcs":[[3]]}"#);
    let err = read_topojson(&forward).expect_err("index 3 with one arc");
    assert_eq!(err, GeoReadError::ArcOutOfRange { index: 3, arcs: 1 });

    let reversed = topo_doc(SQUARE_ARC, r#"{"type":"Polygon","arcs":[[-3]]}"#);
    let err = read_topojson(&reversed).expect_err("~(-3) = 2 with one arc");
    assert_eq!(err, GeoReadError::ArcOutOfRange { index: -3, arcs: 1 });
}

#[test]
fn topojson_ring_with_three_positions() {
    let text = topo_doc("[[[0,0],[1,0],[0,0]]]", r#"{"type":"Polygon","arcs":[[0]]}"#);
    let err = read_topojson(&text).expect_err("a ring needs four positions");
    assert_eq!(err, GeoReadError::RingTooShort { positions: 3 });
}

#[test]
fn topojson_unknown_geometry_type() {
    let text = topo_doc(SQUARE_ARC, r#"{"type":"Blob"}"#);
    let err = read_topojson(&text).expect_err("unknown geometry");
    assert_eq!(err, GeoReadError::UnknownType("Blob".to_string()));
}

#[test]
fn topojson_unsupported_geometry() {
    let text = topo_doc(SQUARE_ARC, r#"{"type":"LineString","arcs":[0]}"#);
    let err = read_topojson(&text).expect_err("lines are refused, not skipped");
    assert_eq!(err, GeoReadError::UnsupportedGeometry("LineString".to_string()));
}

#[test]
fn topojson_non_integer_quantized_position() {
    let text = quantized_doc(r#"{"scale":[1,1],"translate":[0,0]}"#, "[[[1.5,0],[1,0]]]");
    let err = read_topojson(&text).expect_err("quantized positions are integers");
    assert_eq!(err, GeoReadError::Malformed("quantized position"));
}

#[test]
fn topojson_quantized_position_overflow() {
    let arcs = "[[[9223372036854775807,0],[1,0]]]";
    let text = quantized_doc(r#"{"scale":[1,1],"translate":[0,0]}"#, arcs);
    let err = read_topojson(&text).expect_err("the running sum overflows i64");
    assert_eq!(err, GeoReadError::Malformed("quantized position overflows"));
}

#[test]
fn topojson_bad_transform() {
    let text = quantized_doc(r#"{"scale":[1],"translate":[0,0]}"#, SQUARE_ARC);
    let err = read_topojson(&text).expect_err("scale needs two numbers");
    assert_eq!(err, GeoReadError::Malformed("transform"));
}

#[test]
fn topojson_scale_overflow_is_non_finite() {
    let text = quantized_doc(r#"{"scale":[1e308,1e308],"translate":[0,0]}"#, "[[[10,10],[0,0]]]");
    let err = read_topojson(&text).expect_err("10 * 1e308 is infinite");
    assert_eq!(err, GeoReadError::NonFinite);
}
