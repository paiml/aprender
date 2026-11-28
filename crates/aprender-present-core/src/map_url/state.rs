//! Map URL state and its error type (MAP-6 slice 1, contract
//! `aprender-viz-map-url-state-v1`).

use std::collections::BTreeMap;
use std::fmt;

/// Camera: centre `x` and `y`, and `log2_zoom`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Camera {
    pub x: f64,
    pub y: f64,
    pub log2_zoom: f64,
}

impl Camera {
    /// True when all three components are finite. `encode` accepts only finite cameras.
    #[must_use]
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.log2_zoom.is_finite()
    }
}

/// Keys the codec owns. `encode` writes them first, in this order.
pub const FIXED_KEYS: [&str; 6] = ["lens", "city", "state", "at", "advisor", "layers"];

/// Extension keys an extension has registered. `decode` routes these into
/// `MapUrlState::extensions`; every other unrecognised key is kept as opaque.
pub const EXTENSION_KEYS: &[&str] = &["x-tint", "x-legend"];

/// Map URL state. Field order is not the emit order; `encode` fixes that.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MapUrlState {
    pub lens: Option<String>,
    pub city: Option<String>,
    pub state: Option<String>,
    pub camera: Option<Camera>,
    pub advisor: Option<String>,
    pub layers: Vec<String>,
    /// Registered extension keys and their values.
    pub extensions: BTreeMap<String, String>,
    /// Unregistered keys, kept verbatim so a re-save does not drop them.
    pub opaque: BTreeMap<String, String>,
}

impl MapUrlState {
    /// Equality that compares camera floats by bit pattern, so `-0.0` and `0.0` differ.
    #[must_use]
    pub fn bitwise_eq(&self, other: &Self) -> bool {
        let camera_eq = match (self.camera, other.camera) {
            (None, None) => true,
            (Some(a), Some(b)) => {
                a.x.to_bits() == b.x.to_bits()
                    && a.y.to_bits() == b.y.to_bits()
                    && a.log2_zoom.to_bits() == b.log2_zoom.to_bits()
            }
            _ => false,
        };
        camera_eq
            && self.lens == other.lens
            && self.city == other.city
            && self.state == other.state
            && self.advisor == other.advisor
            && self.layers == other.layers
            && self.extensions == other.extensions
            && self.opaque == other.opaque
    }
}

/// Errors from `encode` and `decode`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MapUrlError {
    /// A camera component is NaN or infinite.
    NonFinite,
    /// The hash is not well formed; the message says where.
    Malformed(String),
    /// The same key appears twice in one hash.
    DuplicateKey(String),
    /// A key is empty.
    EmptyKey,
    /// A layer name is empty, so it cannot be written in a comma list.
    EmptyLayer,
    /// An opaque key is a fixed or registered extension key.
    KeyCollision(String),
    /// An extension key is not in `EXTENSION_KEYS`.
    UnregisteredExtension(String),
}

impl fmt::Display for MapUrlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite => write!(f, "camera value is not finite"),
            Self::Malformed(msg) => write!(f, "malformed map URL: {msg}"),
            Self::DuplicateKey(key) => write!(f, "duplicate key {key:?}"),
            Self::EmptyKey => write!(f, "empty key"),
            Self::EmptyLayer => write!(f, "empty layer name"),
            Self::KeyCollision(key) => {
                write!(f, "key {key:?} collides with a fixed or extension key")
            }
            Self::UnregisteredExtension(key) => {
                write!(f, "extension key {key:?} is not registered")
            }
        }
    }
}

impl std::error::Error for MapUrlError {}
