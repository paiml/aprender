//! Map projections on a sphere of radius `R`, and GeoJSON/TopoJSON reading (`geo` feature).
//!
//! The formulas are the spherical forms in Snyder, *Map Projections: A Working Manual*
//! (USGS Professional Paper 1395). Each projection implements [`Projection`]: geographic
//! degrees in, plane coordinates in units of `R` out (multiply by `R` for the plane).
//!
//! Invalid input is an error, never a number: a non-finite value, a point outside the
//! projection's domain, or a non-finite result is a [`GeoError`].

pub mod fill;
pub mod hex;
pub mod index;
pub mod plot;
pub mod projection;
pub mod reader;

#[cfg(test)]
mod fill_tests;
#[cfg(test)]
mod hex_tests;
#[cfg(test)]
mod index_tests;
#[cfg(test)]
mod plot_tests;
#[cfg(test)]
mod reader_tests;

pub use fill::{fill_polygon_even_odd, svg_path_d, FillError};
pub use reader::{read_geojson, read_topojson, Feature, GeoReadError, Geometry, Position, Ring};
pub mod simplify;

#[cfg(test)]
mod projection_tests;
#[cfg(test)]
mod simplify_tests;

pub use projection::{
    AlbersEqualArea, AlbersUsa, Equirectangular, GeoError, Mercator, Projection,
    MERCATOR_MAX_LAT_DEG,
};
pub use simplify::{douglas_peucker, visvalingam};
