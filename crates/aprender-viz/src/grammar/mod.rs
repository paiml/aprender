//! Grammar of Graphics implementation.
//!
//! Provides declarative visualization specification based on Wilkinson's
//! Grammar of Graphics [Wilkinson 2005].
//!
//! # Components
//!
//! - **Aesthetics**: Mappings from data to visual properties (x, y, color, size, shape)
//! - **Geometries**: Visual representations (point, line, bar, area)
//! - **Statistics**: Data transformations (identity, bin, smooth, density)
//! - **Scales**: Domain-to-range mappings
//! - **Coordinates**: Coordinate systems (cartesian, polar)
//! - **Facets**: Small multiples for conditioning
//!
//! # Example
//!
//! ```rust
//! use trueno_viz::grammar::*;
//!
//! let plot = GGPlot::new()
//!     .data_xy(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0])
//!     .geom(Geom::point())
//!     .aes(Aes::new().color_value(trueno_viz::color::Rgba::BLUE))
//!     .build()
//!     .unwrap();
//! ```
//!
//! # References
//!
//! - Wilkinson, L. (2005). *The Grammar of Graphics*. Springer.
//! - Wickham, H. (2010). "A Layered Grammar of Graphics." Journal of Computational
//!   and Graphical Statistics.

mod aes;
mod build;
mod coord;
mod data;
mod facet;
mod geom;
mod ggplot;
#[cfg(test)]
mod ggplot2_oracle;
#[cfg(test)]
mod ggplot2_oracle_scales;
mod r_format;
mod scale_discrete;
mod scale_spec;
mod scale_trans;
mod stat;
mod theme;

pub use aes::Aes;
pub use build::{
    format_labels, point_defaults, point_diameter_px, point_size_mm_for_diameter_px, BuiltLayer,
    BuiltLayout, PointAes, PositionScale,
};
pub use coord::{apply, apply_limits, Coord};
pub use data::{DataFrame, DataValue};
pub use facet::{panels, Facet, Panel};
pub use geom::Geom;
pub use ggplot::{BuiltGGPlot, GGPlot, Layer};
pub use scale_discrete::{DiscreteSpec, DISCRETE_EXPAND_ADD};
pub use scale_spec::{
    scale_continuous, scale_discrete, scale_log10, ContinuousSpec, Expansion, PositionScaleSpec,
};
pub use stat::Stat;
pub use theme::Theme;
