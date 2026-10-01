//! Spectral index computation, single-pass statistics, and color mapping.
//!
//! Provides radiometric calculations for multispectral satellite and drone bands:
//! - NDVI (Normalized Difference Vegetation Index)
//! - NDWI (Normalized Difference Water Index - Gao and McFeeters)
//! - EVI (Enhanced Vegetation Index)
//! - SAVI (Soil-Adjusted Vegetation Index)
//!
//! # Example
//!
//! ```
//! use trueno_image::spectral::{compute_ndvi, compute_stats};
//!
//! let nir = vec![0.8, 0.7, 0.9];
//! let red = vec![0.1, 0.2, 0.1];
//! let mut out = vec![0.0; 3];
//!
//! compute_ndvi(&nir, &red, &mut out).unwrap();
//! assert!(out[0] > 0.7); // High vegetation vigor
//!
//! let stats = compute_stats(&out);
//! assert_eq!(stats.health_category, "Very dense vegetation");
//! ```

use crate::buf::ImageBuf;
use crate::error::ImageError;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Supported spectral indices for vegetation, water, and soil analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum SpectralIndex {
    /// Normalized Difference Vegetation Index: `(NIR - RED) / (NIR + RED)`
    Ndvi,
    /// Normalized Difference Water Index (Gao 1996): `(NIR - SWIR) / (NIR + SWIR)`
    Ndwi,
    /// Normalized Difference Water Index (McFeeters 1996): `(GREEN - NIR) / (GREEN + NIR)`
    NdwiMcFeeters,
    /// Enhanced Vegetation Index: `2.5 * (NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1)`
    Evi,
    /// Soil-Adjusted Vegetation Index: `(1 + L) * (NIR - RED) / (NIR + RED + L)`
    Savi,
}

/// Comprehensive statistics computed over a spectral index raster.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SpectralStats {
    /// Arithmetic mean of valid (non-NaN) pixels.
    pub mean: f32,
    /// Standard deviation of valid pixels.
    pub std_dev: f32,
    /// Minimum valid value.
    pub min: f32,
    /// Maximum valid value.
    pub max: f32,
    /// 25th percentile (first quartile).
    pub p25: f32,
    /// 50th percentile (median).
    pub p50: f32,
    /// 75th percentile (third quartile).
    pub p75: f32,
    /// Number of valid (non-NaN) pixels analyzed.
    pub valid_pixels: usize,
    /// Total pixels in the raster window.
    pub total_pixels: usize,
    /// Human-readable health category (e.g., "Dense vegetation").
    pub health_category: &'static str,
    /// Machine-readable category key (e.g., "dense").
    pub health_category_key: &'static str,
}

/// Color stop for interpolating index values to RGBA colors.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ColorStop {
    /// Threshold value for this stop (e.g., 0.2).
    pub value: f32,
    /// RGBA color `[R, G, B, A]` in `0..=255`.
    pub rgba: [u8; 4],
}

/// Calibrated color map for rendering spectral overlays.
#[derive(Debug, Clone, PartialEq)]
pub struct ColorMap {
    stops: Vec<ColorStop>,
}

impl ColorMap {
    /// Create a custom colormap from an ordered list of color stops.
    pub fn new(mut stops: Vec<ColorStop>) -> Self {
        stops.sort_by(|a, b| {
            a.value
                .partial_cmp(&b.value)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Self { stops }
    }

    /// Standard 6-stop Red-Yellow-Green (RdYlGn) colormap for NDVI.
    ///
    /// - `< -0.2`: Dark red (water/barren)
    /// - `0.0`: Red (bare soil)
    /// - `0.2`: Yellow (sparse/senescent vegetation)
    /// - `0.4`: Light green (moderate vegetation)
    /// - `0.6`: Green (dense crop canopy)
    /// - `0.9`: Dark green (peak vegetative vigor)
    pub fn ndvi_rdylgn() -> Self {
        Self::new(vec![
            ColorStop {
                value: -0.2,
                rgba: [165, 0, 38, 255],
            },
            ColorStop {
                value: 0.0,
                rgba: [215, 48, 39, 255],
            },
            ColorStop {
                value: 0.2,
                rgba: [254, 224, 139, 255],
            },
            ColorStop {
                value: 0.4,
                rgba: [166, 217, 106, 255],
            },
            ColorStop {
                value: 0.6,
                rgba: [26, 152, 80, 255],
            },
            ColorStop {
                value: 0.9,
                rgba: [0, 104, 55, 255],
            },
        ])
    }

    /// Blues colormap for NDWI moisture and open water.
    pub fn ndwi_blues() -> Self {
        Self::new(vec![
            ColorStop {
                value: -0.5,
                rgba: [255, 245, 240, 255],
            },
            ColorStop {
                value: 0.0,
                rgba: [198, 219, 239, 255],
            },
            ColorStop {
                value: 0.2,
                rgba: [107, 174, 214, 255],
            },
            ColorStop {
                value: 0.5,
                rgba: [33, 113, 181, 255],
            },
            ColorStop {
                value: 1.0,
                rgba: [8, 48, 107, 255],
            },
        ])
    }

    /// Map a scalar float value to an interpolated RGBA color.
    ///
    /// NaN or infinite values return fully transparent black: `[0, 0, 0, 0]`.
    pub fn map_value(&self, val: f32) -> [u8; 4] {
        if val.is_nan() || val.is_infinite() || self.stops.is_empty() {
            return [0, 0, 0, 0];
        }

        if val <= self.stops[0].value {
            return self.stops[0].rgba;
        }

        let last_idx = self.stops.len() - 1;
        if val >= self.stops[last_idx].value {
            return self.stops[last_idx].rgba;
        }

        // Find surrounding interval
        for i in 0..last_idx {
            let left = &self.stops[i];
            let right = &self.stops[i + 1];
            if val >= left.value && val <= right.value {
                let range = right.value - left.value;
                let t = if range.abs() > 1e-6 {
                    ((val - left.value) / range).clamp(0.0, 1.0)
                } else {
                    0.0
                };

                return [
                    (left.rgba[0] as f32 + t * (right.rgba[0] as f32 - left.rgba[0] as f32)).round()
                        as u8,
                    (left.rgba[1] as f32 + t * (right.rgba[1] as f32 - left.rgba[1] as f32)).round()
                        as u8,
                    (left.rgba[2] as f32 + t * (right.rgba[2] as f32 - left.rgba[2] as f32)).round()
                        as u8,
                    (left.rgba[3] as f32 + t * (right.rgba[3] as f32 - left.rgba[3] as f32)).round()
                        as u8,
                ];
            }
        }

        self.stops[last_idx].rgba
    }
}

/// Compute NDVI: `(NIR - RED) / (NIR + RED)`.
///
/// Output is clamped to `[-1.0, 1.0]`. If `(NIR + RED) == 0.0` or either input is NaN,
/// `f32::NAN` is written.
pub fn compute_ndvi(nir: &[f32], red: &[f32], out: &mut [f32]) -> Result<(), ImageError> {
    if nir.len() != red.len() || nir.len() != out.len() {
        return Err(ImageError::DimensionMismatch {
            expected: nir.len(),
            got: if red.len() == nir.len() {
                out.len()
            } else {
                red.len()
            },
        });
    }

    for i in 0..nir.len() {
        let n = nir[i];
        let r = red[i];
        let denom = n + r;
        out[i] = if denom.abs() > 1e-6 && !n.is_nan() && !r.is_nan() {
            ((n - r) / denom).clamp(-1.0, 1.0)
        } else {
            f32::NAN
        };
    }

    Ok(())
}

/// Compute NDWI (Gao 1996 for canopy moisture): `(NIR - SWIR) / (NIR + SWIR)`.
pub fn compute_ndwi(nir: &[f32], swir: &[f32], out: &mut [f32]) -> Result<(), ImageError> {
    if nir.len() != swir.len() || nir.len() != out.len() {
        return Err(ImageError::DimensionMismatch {
            expected: nir.len(),
            got: if swir.len() == nir.len() {
                out.len()
            } else {
                swir.len()
            },
        });
    }

    for i in 0..nir.len() {
        let n = nir[i];
        let s = swir[i];
        let denom = n + s;
        out[i] = if denom.abs() > 1e-6 && !n.is_nan() && !s.is_nan() {
            ((n - s) / denom).clamp(-1.0, 1.0)
        } else {
            f32::NAN
        };
    }

    Ok(())
}

/// Compute NDWI (McFeeters 1996 for water bodies): `(GREEN - NIR) / (GREEN + NIR)`.
pub fn compute_ndwi_mcfeeters(
    green: &[f32],
    nir: &[f32],
    out: &mut [f32],
) -> Result<(), ImageError> {
    compute_ndvi(green, nir, out) // Mathematically identical structure
}

/// Compute EVI (Enhanced Vegetation Index): `2.5 * (NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1)`.
pub fn compute_evi(
    nir: &[f32],
    red: &[f32],
    blue: &[f32],
    out: &mut [f32],
) -> Result<(), ImageError> {
    let len = nir.len();
    if red.len() != len || blue.len() != len || out.len() != len {
        return Err(ImageError::DimensionMismatch {
            expected: len,
            got: if red.len() == len {
                if blue.len() == len {
                    out.len()
                } else {
                    blue.len()
                }
            } else {
                red.len()
            },
        });
    }

    for i in 0..len {
        let n = nir[i];
        let r = red[i];
        let b = blue[i];
        let denom = n + 6.0 * r - 7.5 * b + 1.0;
        out[i] = if denom.abs() > 1e-6 && !n.is_nan() && !r.is_nan() && !b.is_nan() {
            (2.5 * (n - r) / denom).clamp(-1.0, 1.0)
        } else {
            f32::NAN
        };
    }

    Ok(())
}

/// Compute SAVI (Soil-Adjusted Vegetation Index): `(1 + L) * (NIR - RED) / (NIR + RED + L)`.
///
/// Default soil adjustment coefficient `l = 0.5`.
pub fn compute_savi(nir: &[f32], red: &[f32], l: f32, out: &mut [f32]) -> Result<(), ImageError> {
    if nir.len() != red.len() || nir.len() != out.len() {
        return Err(ImageError::DimensionMismatch {
            expected: nir.len(),
            got: if red.len() == nir.len() {
                out.len()
            } else {
                red.len()
            },
        });
    }

    let factor = 1.0 + l;
    for i in 0..nir.len() {
        let n = nir[i];
        let r = red[i];
        let denom = n + r + l;
        out[i] = if denom.abs() > 1e-6 && !n.is_nan() && !r.is_nan() {
            (factor * (n - r) / denom).clamp(-1.0, 1.0)
        } else {
            f32::NAN
        };
    }

    Ok(())
}

/// Classify vegetation health based on mean index value.
pub fn classify_health(mean: f32) -> (&'static str, &'static str) {
    if mean < 0.15 {
        ("Bare soil or water", "bare_soil")
    } else if mean < 0.30 {
        ("Sparse vegetation", "sparse")
    } else if mean < 0.50 {
        ("Moderate vegetation", "moderate")
    } else if mean < 0.70 {
        ("Dense vegetation", "dense")
    } else {
        ("Very dense vegetation", "very_dense")
    }
}

/// Compute statistical summary for a flat array of spectral index values.
pub fn compute_stats(values: &[f32]) -> SpectralStats {
    let total_pixels = values.len();
    let mut valid: Vec<f32> = values
        .iter()
        .copied()
        .filter(|v| !v.is_nan() && !v.is_infinite())
        .collect();

    let valid_pixels = valid.len();
    if valid_pixels == 0 {
        let (cat, cat_key) = classify_health(0.0);
        return SpectralStats {
            mean: 0.0,
            std_dev: 0.0,
            min: 0.0,
            max: 0.0,
            p25: 0.0,
            p50: 0.0,
            p75: 0.0,
            valid_pixels: 0,
            total_pixels,
            health_category: cat,
            health_category_key: cat_key,
        };
    }

    let mut sum = 0.0_f64;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;

    for &v in &valid {
        sum += v as f64;
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
    }

    let mean = (sum / valid_pixels as f64) as f32;

    let mut var_sum = 0.0_f64;
    for &v in &valid {
        let diff = v as f64 - mean as f64;
        var_sum += diff * diff;
    }
    let std_dev = (var_sum / valid_pixels as f64).sqrt() as f32;

    // Percentiles via partial sort
    valid.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p25 = percentile_sorted(&valid, 0.25);
    let p50 = percentile_sorted(&valid, 0.50);
    let p75 = percentile_sorted(&valid, 0.75);

    let (health_category, health_category_key) = classify_health(mean);

    SpectralStats {
        mean,
        std_dev,
        min,
        max,
        p25,
        p50,
        p75,
        valid_pixels,
        total_pixels,
        health_category,
        health_category_key,
    }
}

fn percentile_sorted(sorted: &[f32], p: f32) -> f32 {
    debug_assert!(!sorted.is_empty());
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let idx = (n - 1) as f32 * p;
    let low = idx.floor() as usize;
    let high = idx.ceil() as usize;
    let weight = idx - low as f32;

    if low == high {
        sorted[low]
    } else {
        sorted[low] * (1.0 - weight) + sorted[high] * weight
    }
}

/// Render a 4-channel RGBA `ImageBuf` heatmap overlay from spectral index values.
pub fn render_overlay_rgba(
    values: &[f32],
    width: usize,
    height: usize,
    colormap: &ColorMap,
) -> Result<ImageBuf, ImageError> {
    if values.len() != width * height {
        return Err(ImageError::DimensionMismatch {
            expected: width * height,
            got: values.len(),
        });
    }

    let mut rgba_data = Vec::with_capacity(width * height * 4);
    for &val in values {
        let [r, g, b, a] = colormap.map_value(val);
        rgba_data.push(r as f32 / 255.0);
        rgba_data.push(g as f32 / 255.0);
        rgba_data.push(b as f32 / 255.0);
        rgba_data.push(a as f32 / 255.0);
    }

    ImageBuf::new(rgba_data, width, height, 4)
}

/// Extensions to `ImageBuf` for spectral index operations.
impl ImageBuf {
    /// Compute NDVI between this buffer (as NIR) and another buffer (as RED).
    pub fn ndvi(&self, red: &ImageBuf) -> Result<ImageBuf, ImageError> {
        if self.width() != red.width() || self.height() != red.height() {
            return Err(ImageError::DimensionMismatch {
                expected: self.len(),
                got: red.len(),
            });
        }
        let mut out = vec![0.0_f32; self.width() * self.height()];
        compute_ndvi(self.data(), red.data(), &mut out)?;
        ImageBuf::new(out, self.width(), self.height(), 1)
    }

    /// Compute NDWI between this buffer (as NIR) and another buffer (as SWIR).
    pub fn ndwi(&self, swir: &ImageBuf) -> Result<ImageBuf, ImageError> {
        if self.width() != swir.width() || self.height() != swir.height() {
            return Err(ImageError::DimensionMismatch {
                expected: self.len(),
                got: swir.len(),
            });
        }
        let mut out = vec![0.0_f32; self.width() * self.height()];
        compute_ndwi(self.data(), swir.data(), &mut out)?;
        ImageBuf::new(out, self.width(), self.height(), 1)
    }
}
