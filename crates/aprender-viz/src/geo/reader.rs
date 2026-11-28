//! GeoJSON and TopoJSON reader (`geo` feature).
//!
//! Both readers return the same [`Feature`] list: geometry as polygons of rings of `(lon, lat)`
//! positions in degrees. Malformed input is a [`GeoReadError`], never a panic. Point and line
//! geometries are refused with [`GeoReadError::UnsupportedGeometry`], not silently dropped.

use serde_json::{Map, Value};
use std::collections::BTreeMap;
use thiserror::Error;

/// A position `(lon, lat)` in degrees.
pub type Position = (f64, f64);

/// A closed ring: its first and last positions are equal, and it has at least four positions.
pub type Ring = Vec<Position>;

/// The geometry of a feature. In each polygon, ring 0 is the exterior and the rest are holes.
#[derive(Debug, Clone, PartialEq)]
pub enum Geometry {
    /// A polygon: a list of rings.
    Polygon(Vec<Ring>),
    /// A multipolygon: a list of polygons, each a list of rings.
    MultiPolygon(Vec<Vec<Ring>>),
}

/// A feature: an optional id, its properties, and its geometry.
#[derive(Debug, Clone, PartialEq)]
pub struct Feature {
    /// The id. A string id is kept; a numeric id is rendered in decimal; absent or `null` is `None`.
    pub id: Option<String>,
    /// The properties. A string is kept as is, `null` becomes `""`, any other value its JSON text.
    pub properties: BTreeMap<String, String>,
    /// The geometry.
    pub geometry: Geometry,
}

/// Why a document was refused. Every malformed input maps to one of these variants.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum GeoReadError {
    /// The text is not JSON.
    #[error("not JSON: {0}")]
    Json(String),
    /// A required member is missing or has the wrong JSON type. Names the member.
    #[error("missing or malformed member: {0}")]
    Malformed(&'static str),
    /// A `type` value this reader does not know.
    #[error("unknown type: {0}")]
    UnknownType(String),
    /// A geometry type this reader does not decode (Point, LineString, and similar).
    #[error("unsupported geometry type: {0}")]
    UnsupportedGeometry(String),
    /// A TopoJSON arc reference past the end of `arcs`. A reversed arc `~i` is `-i - 1`.
    #[error("arc index {index} is out of range for {arcs} arcs")]
    ArcOutOfRange {
        /// The index as written in the document.
        index: i64,
        /// The number of arcs in the topology.
        arcs: usize,
    },
    /// A ring with fewer than four positions.
    #[error("ring has {positions} positions, needs at least 4")]
    RingTooShort {
        /// The number of positions the ring has.
        positions: usize,
    },
    /// A ring whose first and last positions differ.
    #[error("ring is not closed")]
    RingNotClosed,
    /// A coordinate that is NaN or infinite, including after the TopoJSON transform.
    #[error("non-finite coordinate")]
    NonFinite,
    /// The TopoJSON arcs expand to more positions than the document budget allows, summed over
    /// every ring of every object (see `MAX_EXPANDED_POSITIONS`).
    #[error("too many positions: the document expands past {limit}")]
    TooManyPositions {
        /// The budget that was exceeded.
        limit: usize,
    },
}

/// Budget for the positions a TopoJSON document expands to, summed over every ring of every
/// object.
///
/// Arcs are shared, so each arc reference copies its positions into a ring and a small document
/// can expand to a very large one. The budget is checked before each copy, and exceeding it is
/// [`GeoReadError::TooManyPositions`].
const MAX_EXPANDED_POSITIONS: usize = 4_000_000;

/// Reads GeoJSON: a `FeatureCollection`, or a single `Feature`.
///
/// # Errors
///
/// Returns a [`GeoReadError`] for any malformed, unsupported, or non-finite input.
pub fn read_geojson(text: &str) -> Result<Vec<Feature>, GeoReadError> {
    let root = parse(text)?;
    match type_of(&root)? {
        "FeatureCollection" => {
            member_array(&root, "features")?.iter().map(geojson_feature).collect()
        }
        "Feature" => Ok(vec![geojson_feature(&root)?]),
        other => Err(GeoReadError::UnknownType(other.to_string())),
    }
}

/// Reads a TopoJSON `Topology`: every object becomes features, with arcs resolved and
/// (when a `transform` is present) dequantized.
///
/// # Errors
///
/// Returns a [`GeoReadError`] for any malformed, unsupported, or non-finite input.
pub fn read_topojson(text: &str) -> Result<Vec<Feature>, GeoReadError> {
    let root = parse(text)?;
    match type_of(&root)? {
        "Topology" => {}
        other => return Err(GeoReadError::UnknownType(other.to_string())),
    }
    let arcs = decode_arcs(&root)?;
    let mut budget = MAX_EXPANDED_POSITIONS;
    let mut features = Vec::new();
    for object in member_object(&root, "objects")?.values() {
        collect_object(object, &arcs, &mut features, &mut budget)?;
    }
    Ok(features)
}

fn parse(text: &str) -> Result<Value, GeoReadError> {
    serde_json::from_str(text).map_err(|e| GeoReadError::Json(e.to_string()))
}

fn member<'a>(v: &'a Value, key: &'static str) -> Result<&'a Value, GeoReadError> {
    v.get(key).ok_or(GeoReadError::Malformed(key))
}

fn member_array<'a>(v: &'a Value, key: &'static str) -> Result<&'a [Value], GeoReadError> {
    member(v, key)?.as_array().map(Vec::as_slice).ok_or(GeoReadError::Malformed(key))
}

fn member_object<'a>(
    v: &'a Value,
    key: &'static str,
) -> Result<&'a Map<String, Value>, GeoReadError> {
    member(v, key)?.as_object().ok_or(GeoReadError::Malformed(key))
}

fn type_of(v: &Value) -> Result<&str, GeoReadError> {
    member(v, "type")?.as_str().ok_or(GeoReadError::Malformed("type"))
}

fn id_of(v: &Value) -> Result<Option<String>, GeoReadError> {
    match v.get("id") {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => Ok(Some(s.clone())),
        Some(n @ Value::Number(_)) => Ok(Some(n.to_string())),
        Some(_) => Err(GeoReadError::Malformed("id")),
    }
}

fn properties_of(v: &Value) -> Result<BTreeMap<String, String>, GeoReadError> {
    match v.get("properties") {
        None | Some(Value::Null) => Ok(BTreeMap::new()),
        Some(Value::Object(map)) => {
            Ok(map.iter().map(|(k, val)| (k.clone(), property_text(val))).collect())
        }
        Some(_) => Err(GeoReadError::Malformed("properties")),
    }
}

fn property_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

/// Refuses a geometry type this reader does not decode, naming it.
fn refuse<T>(kind: &str) -> Result<T, GeoReadError> {
    match kind {
        "Point" | "MultiPoint" | "LineString" | "MultiLineString" | "GeometryCollection" => {
            Err(GeoReadError::UnsupportedGeometry(kind.to_string()))
        }
        other => Err(GeoReadError::UnknownType(other.to_string())),
    }
}

fn finite(x: f64) -> Result<f64, GeoReadError> {
    if x.is_finite() {
        Ok(x)
    } else {
        Err(GeoReadError::NonFinite)
    }
}

fn finite_number(v: &Value, what: &'static str) -> Result<f64, GeoReadError> {
    v.as_f64().ok_or(GeoReadError::Malformed(what)).and_then(finite)
}

/// A ring is closed (first equals last) and has at least four positions.
fn check_ring(ring: Ring) -> Result<Ring, GeoReadError> {
    if ring.len() < 4 {
        return Err(GeoReadError::RingTooShort { positions: ring.len() });
    }
    if ring.first() != ring.last() {
        return Err(GeoReadError::RingNotClosed);
    }
    Ok(ring)
}

fn geojson_feature(v: &Value) -> Result<Feature, GeoReadError> {
    match type_of(v)? {
        "Feature" => {}
        other => return Err(GeoReadError::UnknownType(other.to_string())),
    }
    Ok(Feature {
        id: id_of(v)?,
        properties: properties_of(v)?,
        geometry: geojson_geometry(member(v, "geometry")?)?,
    })
}

fn geojson_geometry(g: &Value) -> Result<Geometry, GeoReadError> {
    match type_of(g)? {
        "Polygon" => Ok(Geometry::Polygon(polygon_rings(member(g, "coordinates")?)?)),
        "MultiPolygon" => member_array(g, "coordinates")?
            .iter()
            .map(polygon_rings)
            .collect::<Result<Vec<Vec<Ring>>, GeoReadError>>()
            .map(Geometry::MultiPolygon),
        other => refuse(other),
    }
}

fn polygon_rings(v: &Value) -> Result<Vec<Ring>, GeoReadError> {
    let rings = v.as_array().ok_or(GeoReadError::Malformed("coordinates"))?;
    rings.iter().map(ring_from_json).collect()
}

fn ring_from_json(v: &Value) -> Result<Ring, GeoReadError> {
    let positions = v.as_array().ok_or(GeoReadError::Malformed("ring"))?;
    let ring = positions.iter().map(position_from_json).collect::<Result<Ring, GeoReadError>>()?;
    check_ring(ring)
}

fn position_from_json(v: &Value) -> Result<Position, GeoReadError> {
    match v.as_array().map(Vec::as_slice) {
        Some([x, y, ..]) => Ok((finite_number(x, "position")?, finite_number(y, "position")?)),
        _ => Err(GeoReadError::Malformed("position")),
    }
}

/// The TopoJSON `transform`: `x = q * scale + translate`, with `q` delta-encoded per arc.
#[derive(Debug, Clone, Copy)]
struct Transform {
    scale: [f64; 2],
    translate: [f64; 2],
}

fn collect_object(
    object: &Value,
    arcs: &[Vec<Position>],
    out: &mut Vec<Feature>,
    budget: &mut usize,
) -> Result<(), GeoReadError> {
    if type_of(object)? == "GeometryCollection" {
        for g in member_array(object, "geometries")? {
            out.push(topo_feature(g, arcs, budget)?);
        }
    } else {
        out.push(topo_feature(object, arcs, budget)?);
    }
    Ok(())
}

fn topo_feature(
    g: &Value,
    arcs: &[Vec<Position>],
    budget: &mut usize,
) -> Result<Feature, GeoReadError> {
    Ok(Feature {
        id: id_of(g)?,
        properties: properties_of(g)?,
        geometry: topo_geometry(g, arcs, budget)?,
    })
}

fn topo_geometry(
    g: &Value,
    arcs: &[Vec<Position>],
    budget: &mut usize,
) -> Result<Geometry, GeoReadError> {
    match type_of(g)? {
        "Polygon" => Ok(Geometry::Polygon(topo_rings(member(g, "arcs")?, arcs, budget)?)),
        "MultiPolygon" => member_array(g, "arcs")?
            .iter()
            .map(|p| topo_rings(p, arcs, &mut *budget))
            .collect::<Result<Vec<Vec<Ring>>, GeoReadError>>()
            .map(Geometry::MultiPolygon),
        other => refuse(other),
    }
}

fn topo_rings(
    v: &Value,
    arcs: &[Vec<Position>],
    budget: &mut usize,
) -> Result<Vec<Ring>, GeoReadError> {
    let rings = v.as_array().ok_or(GeoReadError::Malformed("arcs"))?;
    rings.iter().map(|r| topo_ring(r, arcs, &mut *budget)).collect()
}

/// Joins a ring's arcs. An arc's first position repeats the previous arc's last position, so it
/// is dropped, except when the ring is still empty (an earlier empty arc has nothing to repeat).
/// Each copy is charged against `budget` before it is made.
fn topo_ring(v: &Value, arcs: &[Vec<Position>], budget: &mut usize) -> Result<Ring, GeoReadError> {
    let refs = v.as_array().ok_or(GeoReadError::Malformed("arcs"))?;
    let mut ring = Ring::new();
    for r in refs {
        let index = r.as_i64().ok_or(GeoReadError::Malformed("arc index"))?;
        let (arc, reversed) = arc_ref(index, arcs)?;
        let skip = usize::from(!ring.is_empty());
        spend(budget, arc.len().saturating_sub(skip))?;
        if reversed {
            ring.extend(arc.iter().rev().skip(skip));
        } else {
            ring.extend(arc.iter().skip(skip));
        }
    }
    check_ring(ring)
}

/// Charges `n` positions against the document budget, refusing before any copy is made.
fn spend(budget: &mut usize, n: usize) -> Result<(), GeoReadError> {
    if n > *budget {
        return Err(GeoReadError::TooManyPositions { limit: MAX_EXPANDED_POSITIONS });
    }
    *budget -= n;
    Ok(())
}

/// The positions of arc `index`, and whether the reference is reversed (`~i`, that is `-i - 1`).
/// Borrows the arc; nothing is copied.
fn arc_ref(index: i64, arcs: &[Vec<Position>]) -> Result<(&[Position], bool), GeoReadError> {
    let (i, reversed) = if index >= 0 { (index, false) } else { (!index, true) };
    let arc = usize::try_from(i)
        .ok()
        .and_then(|i| arcs.get(i))
        .ok_or(GeoReadError::ArcOutOfRange { index, arcs: arcs.len() })?;
    Ok((arc.as_slice(), reversed))
}

fn decode_arcs(root: &Value) -> Result<Vec<Vec<Position>>, GeoReadError> {
    let transform = transform_of(root)?;
    member_array(root, "arcs")?.iter().map(|arc| decode_arc(arc, transform)).collect()
}

fn transform_of(root: &Value) -> Result<Option<Transform>, GeoReadError> {
    match root.get("transform") {
        None | Some(Value::Null) => Ok(None),
        Some(t) => Ok(Some(Transform {
            scale: pair(member(t, "scale")?)?,
            translate: pair(member(t, "translate")?)?,
        })),
    }
}

fn pair(v: &Value) -> Result<[f64; 2], GeoReadError> {
    match v.as_array().map(Vec::as_slice) {
        Some([a, b]) => Ok([finite_number(a, "transform")?, finite_number(b, "transform")?]),
        _ => Err(GeoReadError::Malformed("transform")),
    }
}

fn decode_arc(arc: &Value, transform: Option<Transform>) -> Result<Vec<Position>, GeoReadError> {
    let points = arc.as_array().ok_or(GeoReadError::Malformed("arc"))?;
    match transform {
        None => points.iter().map(position_from_json).collect(),
        Some(t) => dequantize(points, t),
    }
}

/// Undoes the delta encoding and the transform of one arc.
#[allow(clippy::cast_precision_loss)]
fn dequantize(points: &[Value], t: Transform) -> Result<Vec<Position>, GeoReadError> {
    let mut sum = (0_i64, 0_i64);
    let mut out = Vec::with_capacity(points.len());
    for point in points {
        let (dx, dy) = quantized_pair(point)?;
        sum = (checked_sum(sum.0, dx)?, checked_sum(sum.1, dy)?);
        let x = finite(sum.0 as f64 * t.scale[0] + t.translate[0])?;
        let y = finite(sum.1 as f64 * t.scale[1] + t.translate[1])?;
        out.push((x, y));
    }
    Ok(out)
}

fn quantized_pair(v: &Value) -> Result<(i64, i64), GeoReadError> {
    let bad = GeoReadError::Malformed("quantized position");
    match v.as_array().map(Vec::as_slice) {
        Some([x, y, ..]) => Ok((x.as_i64().ok_or(bad.clone())?, y.as_i64().ok_or(bad)?)),
        _ => Err(bad),
    }
}

fn checked_sum(a: i64, b: i64) -> Result<i64, GeoReadError> {
    a.checked_add(b).ok_or(GeoReadError::Malformed("quantized position overflows"))
}
