//! Pixel-Perfect Verification Metrics (PIXEL-001 v2.1 Phase 6)
//!
//! Film-study methodology image comparison metrics:
//! - SSIM (Structural Similarity Index)
//! - PSNR (Peak Signal-to-Noise Ratio)
//! - CIEDE2000 (CIE Delta E 2000 color difference)
//! - PHash (Perceptual Hashing)

use super::heatmap::Rgb;

// ============================================================================
// SSIM - Structural Similarity Index (Wang et al., 2004)
// ============================================================================

/// Structural Similarity Index Measure
/// Range: -1 to 1 (1 = identical, 0 = no similarity)
#[derive(Debug, Clone)]
pub struct SsimMetric {
    /// Window size for local comparison
    pub window_size: u32,
    /// Threshold for "pixel perfect"
    pub perfect_threshold: f32,
    /// Threshold for "acceptable"
    pub accept_threshold: f32,
}

impl Default for SsimMetric {
    fn default() -> Self {
        Self {
            window_size: 11,
            perfect_threshold: 0.99,
            accept_threshold: 0.95,
        }
    }
}

/// Result of SSIM comparison
#[derive(Debug, Clone)]
pub struct SsimResult {
    /// Overall SSIM score (-1.0 to 1.0)
    pub score: f32,
    /// Whether images are pixel-perfect
    pub is_perfect: bool,
    /// Whether images are acceptable
    pub is_acceptable: bool,
    /// Per-channel SSIM (R, G, B)
    pub channel_scores: [f32; 3],
}

impl SsimMetric {
    /// Create new SSIM metric
    #[must_use]
    pub fn new(window_size: u32) -> Self {
        Self {
            window_size,
            ..Default::default()
        }
    }

    /// Set thresholds
    #[must_use]
    pub fn with_thresholds(mut self, perfect: f32, acceptable: f32) -> Self {
        self.perfect_threshold = perfect;
        self.accept_threshold = acceptable;
        self
    }

    /// Compare two images represented as RGB pixel arrays
    /// Images must have same dimensions
    #[must_use]
    pub fn compare(
        &self,
        reference: &[Rgb],
        generated: &[Rgb],
        width: u32,
        height: u32,
    ) -> SsimResult {
        if reference.len() != generated.len() {
            return SsimResult {
                score: 0.0,
                is_perfect: false,
                is_acceptable: false,
                channel_scores: [0.0, 0.0, 0.0],
            };
        }

        // Calculate SSIM for each channel
        let r_ref: Vec<f32> = reference.iter().map(|p| p.r as f32).collect();
        let r_gen: Vec<f32> = generated.iter().map(|p| p.r as f32).collect();
        let g_ref: Vec<f32> = reference.iter().map(|p| p.g as f32).collect();
        let g_gen: Vec<f32> = generated.iter().map(|p| p.g as f32).collect();
        let b_ref: Vec<f32> = reference.iter().map(|p| p.b as f32).collect();
        let b_gen: Vec<f32> = generated.iter().map(|p| p.b as f32).collect();

        let r_ssim = self.calculate_channel_ssim(&r_ref, &r_gen, width, height);
        let g_ssim = self.calculate_channel_ssim(&g_ref, &g_gen, width, height);
        let b_ssim = self.calculate_channel_ssim(&b_ref, &b_gen, width, height);

        // Average across channels (luminance-weighted would be more accurate)
        let score = (r_ssim + g_ssim + b_ssim) / 3.0;

        SsimResult {
            score,
            is_perfect: score >= self.perfect_threshold,
            is_acceptable: score >= self.accept_threshold,
            channel_scores: [r_ssim, g_ssim, b_ssim],
        }
    }

    /// Calculate SSIM for a single channel
    fn calculate_channel_ssim(
        &self,
        reference: &[f32],
        generated: &[f32],
        _width: u32,
        _height: u32,
    ) -> f32 {
        // SSIM constants (for 8-bit images)
        let k1: f32 = 0.01;
        let k2: f32 = 0.03;
        let l: f32 = 255.0; // Dynamic range
        let c1 = (k1 * l).powi(2);
        let c2 = (k2 * l).powi(2);

        // For simplicity, use global statistics (full-image SSIM)
        // A proper implementation would use sliding windows
        let n = reference.len() as f32;

        // Mean
        let mean_ref: f32 = reference.iter().sum::<f32>() / n;
        let mean_gen: f32 = generated.iter().sum::<f32>() / n;

        // Variance and covariance
        let var_ref: f32 = reference
            .iter()
            .map(|&x| (x - mean_ref).powi(2))
            .sum::<f32>()
            / n;
        let var_gen: f32 = generated
            .iter()
            .map(|&x| (x - mean_gen).powi(2))
            .sum::<f32>()
            / n;
        let covar: f32 = reference
            .iter()
            .zip(generated.iter())
            .map(|(&r, &g)| (r - mean_ref) * (g - mean_gen))
            .sum::<f32>()
            / n;

        // SSIM formula
        let numerator = (2.0 * mean_ref * mean_gen + c1) * (2.0 * covar + c2);
        let denominator = (mean_ref.powi(2) + mean_gen.powi(2) + c1) * (var_ref + var_gen + c2);

        if denominator > 0.0 {
            numerator / denominator
        } else {
            1.0 // Identical zero images
        }
    }
}

// ============================================================================
// PSNR - Peak Signal-to-Noise Ratio
// ============================================================================

/// Peak Signal-to-Noise Ratio metric
#[derive(Debug, Clone)]
pub struct PsnrMetric {
    /// Maximum pixel value (255 for 8-bit)
    pub max_value: f32,
    /// Threshold for excellent quality (dB)
    pub excellent_threshold: f32,
    /// Threshold for acceptable quality (dB)
    pub acceptable_threshold: f32,
}

impl Default for PsnrMetric {
    fn default() -> Self {
        Self {
            max_value: 255.0,
            excellent_threshold: 40.0,
            acceptable_threshold: 30.0,
        }
    }
}

/// Result of PSNR comparison
#[derive(Debug, Clone)]
pub struct PsnrResult {
    /// PSNR value in dB (higher = better, infinity = identical)
    pub psnr_db: f32,
    /// Mean Squared Error
    pub mse: f32,
    /// Quality classification
    pub quality: PsnrQuality,
}

/// PSNR quality classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PsnrQuality {
    /// Identical images (PSNR = infinity)
    Identical,
    /// Excellent quality (PSNR >= 40 dB)
    Excellent,
    /// Good quality (PSNR >= 35 dB)
    Good,
    /// Acceptable (PSNR >= 30 dB)
    Acceptable,
    /// Poor quality
    Poor,
}

impl PsnrMetric {
    /// Compare two images
    #[must_use]
    pub fn compare(&self, reference: &[Rgb], generated: &[Rgb]) -> PsnrResult {
        if reference.len() != generated.len() || reference.is_empty() {
            return PsnrResult {
                psnr_db: 0.0,
                mse: f32::MAX,
                quality: PsnrQuality::Poor,
            };
        }

        // Calculate MSE
        let mse: f32 = reference
            .iter()
            .zip(generated.iter())
            .map(|(r, g)| {
                let dr = (r.r as f32 - g.r as f32).powi(2);
                let dg = (r.g as f32 - g.g as f32).powi(2);
                let db = (r.b as f32 - g.b as f32).powi(2);
                (dr + dg + db) / 3.0
            })
            .sum::<f32>()
            / reference.len() as f32;

        let (psnr_db, quality) = if mse < f32::EPSILON {
            (f32::INFINITY, PsnrQuality::Identical)
        } else {
            let psnr = 10.0 * (self.max_value.powi(2) / mse).log10();
            let quality = if psnr >= self.excellent_threshold {
                PsnrQuality::Excellent
            } else if psnr >= 35.0 {
                PsnrQuality::Good
            } else if psnr >= self.acceptable_threshold {
                PsnrQuality::Acceptable
            } else {
                PsnrQuality::Poor
            };
            (psnr, quality)
        };

        PsnrResult {
            psnr_db,
            mse,
            quality,
        }
    }
}

// ============================================================================
// CIEDE2000 - CIE Delta E 2000 Color Difference
// ============================================================================

/// Lab color space representation
#[derive(Debug, Clone, Copy, Default)]
pub struct Lab {
    /// Lightness (0-100)
    pub l: f32,
    /// Green-Red axis (-128 to 127)
    pub a: f32,
    /// Blue-Yellow axis (-128 to 127)
    pub b: f32,
}

impl Lab {
    /// Create a new Lab color
    #[must_use]
    pub fn new(l: f32, a: f32, b: f32) -> Self {
        Self { l, a, b }
    }

    /// Convert RGB to Lab color space
    #[must_use]
    #[allow(clippy::excessive_precision)] // Standard CIE colorimetric constants
    #[allow(clippy::many_single_char_names)] // Standard colorimetric variable names (r,g,b,x,y,z)
    pub fn from_rgb(rgb: &Rgb) -> Self {
        // RGB to XYZ (assuming sRGB)
        let r = Self::srgb_to_linear(rgb.r as f32 / 255.0);
        let g = Self::srgb_to_linear(rgb.g as f32 / 255.0);
        let b = Self::srgb_to_linear(rgb.b as f32 / 255.0);

        // sRGB to XYZ (D65 illuminant)
        let x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375;
        let y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750;
        let z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041;

        // XYZ to Lab (D65 reference white)
        let xn = 0.95047;
        let yn = 1.00000;
        let zn = 1.08883;

        let fx = Self::f_xyz(x / xn);
        let fy = Self::f_xyz(y / yn);
        let fz = Self::f_xyz(z / zn);

        Self {
            l: 116.0 * fy - 16.0,
            a: 500.0 * (fx - fy),
            b: 200.0 * (fy - fz),
        }
    }

    fn srgb_to_linear(c: f32) -> f32 {
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    }

    fn f_xyz(t: f32) -> f32 {
        let delta: f32 = 6.0 / 29.0;
        if t > delta.powi(3) {
            t.cbrt()
        } else {
            t / (3.0 * delta.powi(2)) + 4.0 / 29.0
        }
    }
}

/// CIEDE2000 color difference metric (ISO/CIE 11664-6:2014)
#[derive(Debug, Clone)]
pub struct CieDe2000Metric {
    /// Perceptibility threshold (JND)
    pub jnd_threshold: f32,
    /// Acceptability threshold
    pub accept_threshold: f32,
    /// Weighting factors (kL, kC, kH)
    pub weights: (f32, f32, f32),
}

impl Default for CieDe2000Metric {
    fn default() -> Self {
        Self {
            jnd_threshold: 1.0,
            accept_threshold: 2.0,
            weights: (1.0, 1.0, 1.0), // Unity weights (typical)
        }
    }
}

/// Result of CIEDE2000 comparison
#[derive(Debug, Clone)]
pub struct DeltaEResult {
    /// Mean ΔE₀₀ across all pixels
    pub mean_delta_e: f32,
    /// Maximum ΔE₀₀ found
    pub max_delta_e: f32,
    /// Percentage of pixels below JND
    pub percent_imperceptible: f32,
    /// Percentage of pixels in acceptable range
    pub percent_acceptable: f32,
    /// Perceptibility classification
    pub classification: DeltaEClassification,
}

/// CIEDE2000 perceptibility classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeltaEClassification {
    /// Imperceptible (ΔE₀₀ < 0.8-1.0)
    Imperceptible,
    /// Just noticeable (1.0 < ΔE₀₀ < 1.8)
    JustNoticeable,
    /// Acceptable (1.8 < ΔE₀₀ < 2.8)
    Acceptable,
    /// Noticeable (2.8 < ΔE₀₀ < 3.7)
    Noticeable,
    /// Unacceptable (ΔE₀₀ >= 3.7)
    Unacceptable,
}

impl CieDe2000Metric {
    /// Compare two Lab colors and return ΔE₀₀
    #[must_use]
    pub fn delta_e(&self, lab1: &Lab, lab2: &Lab) -> f32 {
        let (kl, kc, kh) = self.weights;

        // Calculate C'ab and h'ab for both colors
        let c1 = (lab1.a.powi(2) + lab1.b.powi(2)).sqrt();
        let c2 = (lab2.a.powi(2) + lab2.b.powi(2)).sqrt();
        let c_avg = (c1 + c2) / 2.0;

        // G factor
        let c_avg_7 = c_avg.powi(7);
        let g = 0.5 * (1.0 - (c_avg_7 / (c_avg_7 + 25.0_f32.powi(7))).sqrt());

        // a' values
        let a1_prime = lab1.a * (1.0 + g);
        let a2_prime = lab2.a * (1.0 + g);

        // C'ab
        let c1_prime = (a1_prime.powi(2) + lab1.b.powi(2)).sqrt();
        let c2_prime = (a2_prime.powi(2) + lab2.b.powi(2)).sqrt();

        // h'ab (in degrees)
        let h1_prime = if a1_prime.abs() < f32::EPSILON && lab1.b.abs() < f32::EPSILON {
            0.0
        } else {
            lab1.b.atan2(a1_prime).to_degrees().rem_euclid(360.0)
        };
        let h2_prime = if a2_prime.abs() < f32::EPSILON && lab2.b.abs() < f32::EPSILON {
            0.0
        } else {
            lab2.b.atan2(a2_prime).to_degrees().rem_euclid(360.0)
        };

        // Differences
        let delta_l_prime = lab2.l - lab1.l;
        let delta_c_prime = c2_prime - c1_prime;

        let delta_h_prime_deg = if c1_prime * c2_prime < f32::EPSILON {
            0.0
        } else {
            let dh = h2_prime - h1_prime;
            if dh.abs() <= 180.0 {
                dh
            } else if dh > 180.0 {
                dh - 360.0
            } else {
                dh + 360.0
            }
        };

        let delta_h_prime =
            2.0 * (c1_prime * c2_prime).sqrt() * (delta_h_prime_deg.to_radians() / 2.0).sin();

        // Means
        let l_prime_avg = (lab1.l + lab2.l) / 2.0;
        let c_prime_avg = (c1_prime + c2_prime) / 2.0;

        let h_prime_avg = if c1_prime * c2_prime < f32::EPSILON {
            h1_prime + h2_prime
        } else {
            let diff = (h1_prime - h2_prime).abs();
            if diff <= 180.0 {
                (h1_prime + h2_prime) / 2.0
            } else if h1_prime + h2_prime < 360.0 {
                (h1_prime + h2_prime + 360.0) / 2.0
            } else {
                (h1_prime + h2_prime - 360.0) / 2.0
            }
        };

        // Weighting functions
        let t = 1.0 - 0.17 * (h_prime_avg - 30.0).to_radians().cos()
            + 0.24 * (2.0 * h_prime_avg).to_radians().cos()
            + 0.32 * (3.0 * h_prime_avg + 6.0).to_radians().cos()
            - 0.20 * (4.0 * h_prime_avg - 63.0).to_radians().cos();

        let delta_theta = 30.0 * (-((h_prime_avg - 275.0) / 25.0).powi(2)).exp();

        let c_prime_avg_7 = c_prime_avg.powi(7);
        let rc = 2.0 * (c_prime_avg_7 / (c_prime_avg_7 + 25.0_f32.powi(7))).sqrt();

        let l_50_sq = (l_prime_avg - 50.0).powi(2);
        let sl = 1.0 + (0.015 * l_50_sq) / (20.0 + l_50_sq).sqrt();
        let sc = 1.0 + 0.045 * c_prime_avg;
        let sh = 1.0 + 0.015 * c_prime_avg * t;
        let rt = -(2.0 * delta_theta).to_radians().sin() * rc;

        // Final ΔE₀₀
        let dl = delta_l_prime / (kl * sl);
        let dc = delta_c_prime / (kc * sc);
        let dh = delta_h_prime / (kh * sh);

        (dl.powi(2) + dc.powi(2) + dh.powi(2) + rt * dc * dh).sqrt()
    }

    /// Compare two images
    #[must_use]
    pub fn compare(&self, reference: &[Rgb], generated: &[Rgb]) -> DeltaEResult {
        if reference.len() != generated.len() || reference.is_empty() {
            return DeltaEResult {
                mean_delta_e: f32::MAX,
                max_delta_e: f32::MAX,
                percent_imperceptible: 0.0,
                percent_acceptable: 0.0,
                classification: DeltaEClassification::Unacceptable,
            };
        }

        let mut sum_de = 0.0f32;
        let mut max_delta_e = 0.0f32;
        let mut imperceptible_count = 0u32;
        let mut acceptable_count = 0u32;

        for (r, g) in reference.iter().zip(generated.iter()) {
            let lab1 = Lab::from_rgb(r);
            let lab2 = Lab::from_rgb(g);
            let de = self.delta_e(&lab1, &lab2);

            sum_de += de;
            max_delta_e = max_delta_e.max(de);

            if de < self.jnd_threshold {
                imperceptible_count += 1;
            }
            if de < self.accept_threshold {
                acceptable_count += 1;
            }
        }

        let n = reference.len() as f32;
        let mean_delta_e = sum_de / n;

        let classification = if mean_delta_e < 0.8 {
            DeltaEClassification::Imperceptible
        } else if mean_delta_e < 1.8 {
            DeltaEClassification::JustNoticeable
        } else if mean_delta_e < 2.8 {
            DeltaEClassification::Acceptable
        } else if mean_delta_e < 3.7 {
            DeltaEClassification::Noticeable
        } else {
            DeltaEClassification::Unacceptable
        };

        DeltaEResult {
            mean_delta_e,
            max_delta_e,
            percent_imperceptible: imperceptible_count as f32 / n * 100.0,
            percent_acceptable: acceptable_count as f32 / n * 100.0,
            classification,
        }
    }

    /// Is the difference imperceptible?
    #[must_use]
    pub fn is_imperceptible(&self, delta_e: f32) -> bool {
        delta_e < self.jnd_threshold
    }
}

// ============================================================================
// Perceptual Hash (PHash)
// ============================================================================

/// Perceptual hash algorithm selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PhashAlgorithm {
    /// Average hash (fastest, least robust)
    AHash,
    /// Difference hash (good balance)
    #[default]
    DHash,
    /// Perceptual hash using DCT (most robust)
    PHash,
}

/// Perceptual hash for image fingerprinting
#[derive(Debug, Clone)]
pub struct PerceptualHash {
    /// Algorithm to use
    pub algorithm: PhashAlgorithm,
    /// Hash size in bits (must be power of 2, max 64)
    pub hash_bits: u32,
}

impl Default for PerceptualHash {
    fn default() -> Self {
        Self {
            algorithm: PhashAlgorithm::DHash,
            hash_bits: 64,
        }
    }
}

impl PerceptualHash {
    /// Create new hasher with algorithm
    #[must_use]
    pub fn new(algorithm: PhashAlgorithm) -> Self {
        Self {
            algorithm,
            ..Default::default()
        }
    }

    /// Compute hash for image
    #[must_use]
    pub fn compute(&self, image: &[Rgb], width: u32, height: u32) -> u64 {
        match self.algorithm {
            PhashAlgorithm::AHash => self.average_hash(image, width, height),
            PhashAlgorithm::DHash => self.difference_hash(image, width, height),
            PhashAlgorithm::PHash => self.perceptual_hash(image, width, height),
        }
    }

    /// Average hash: compare each pixel to mean
    fn average_hash(&self, image: &[Rgb], width: u32, height: u32) -> u64 {
        // Resize to 8x8 (simple downscale)
        let resized = self.resize_grayscale(image, width, height, 8, 8);

        // Calculate mean
        let mean: f32 = resized.iter().sum::<f32>() / 64.0;

        // Generate hash
        let mut hash: u64 = 0;
        for (i, &pixel) in resized.iter().enumerate() {
            if pixel > mean {
                hash |= 1 << i;
            }
        }
        hash
    }

    /// Difference hash: compare adjacent pixels
    fn difference_hash(&self, image: &[Rgb], width: u32, height: u32) -> u64 {
        // Resize to 9x8 for 8x8 = 64 bit differences
        let resized = self.resize_grayscale(image, width, height, 9, 8);

        let mut hash: u64 = 0;
        let mut bit = 0;
        for row in 0..8 {
            for col in 0..8 {
                let idx = row * 9 + col;
                if resized[idx] < resized[idx + 1] {
                    hash |= 1 << bit;
                }
                bit += 1;
            }
        }
        hash
    }

    /// Perceptual hash using simplified DCT
    fn perceptual_hash(&self, image: &[Rgb], width: u32, height: u32) -> u64 {
        // Resize to 32x32
        let resized = self.resize_grayscale(image, width, height, 32, 32);

        // Simple DCT-like transform (top-left 8x8 low frequencies)
        let mut dct = vec![0.0f32; 64];
        for u in 0..8 {
            for v in 0..8 {
                let mut sum = 0.0f32;
                for x in 0..32 {
                    for y in 0..32 {
                        let cu = std::f32::consts::PI * (2.0 * x as f32 + 1.0) * u as f32 / 64.0;
                        let cv = std::f32::consts::PI * (2.0 * y as f32 + 1.0) * v as f32 / 64.0;
                        sum += resized[(x * 32 + y) as usize] * cu.cos() * cv.cos();
                    }
                }
                dct[(u * 8 + v) as usize] = sum;
            }
        }

        // Skip DC component (index 0), use low frequencies
        let mean: f32 = dct[1..].iter().sum::<f32>() / 63.0;

        let mut hash: u64 = 0;
        for (i, &val) in dct[1..].iter().take(64).enumerate() {
            if val > mean {
                hash |= 1 << i;
            }
        }
        hash
    }

    /// Resize to grayscale
    fn resize_grayscale(
        &self,
        image: &[Rgb],
        width: u32,
        height: u32,
        new_width: u32,
        new_height: u32,
    ) -> Vec<f32> {
        let mut result = vec![0.0f32; (new_width * new_height) as usize];

        let x_ratio = width as f32 / new_width as f32;
        let y_ratio = height as f32 / new_height as f32;

        for y in 0..new_height {
            for x in 0..new_width {
                let src_x = (x as f32 * x_ratio) as u32;
                let src_y = (y as f32 * y_ratio) as u32;
                let src_idx = (src_y * width + src_x) as usize;

                if src_idx < image.len() {
                    let pixel = &image[src_idx];
                    // Luminance conversion
                    let gray =
                        0.299 * pixel.r as f32 + 0.587 * pixel.g as f32 + 0.114 * pixel.b as f32;
                    result[(y * new_width + x) as usize] = gray;
                }
            }
        }

        result
    }

    /// Hamming distance between hashes (0 = identical)
    #[must_use]
    pub fn distance(hash1: u64, hash2: u64) -> u32 {
        (hash1 ^ hash2).count_ones()
    }

    /// Are images perceptually similar? (distance <= threshold)
    #[must_use]
    pub fn is_similar(hash1: u64, hash2: u64, threshold: u32) -> bool {
        Self::distance(hash1, hash2) <= threshold
    }
}

// ============================================================================
// Combined Verification Suite
// ============================================================================

/// Complete pixel-perfect verification suite
#[derive(Debug, Clone, Default)]
pub struct PixelVerificationSuite {
    /// SSIM metric
    pub ssim: SsimMetric,
    /// PSNR metric
    pub psnr: PsnrMetric,
    /// CIEDE2000 metric
    pub delta_e: CieDe2000Metric,
    /// Perceptual hash
    pub phash: PerceptualHash,
}

/// Complete verification result for pixel-perfect comparison
#[derive(Debug, Clone)]
pub struct PixelVerificationResult {
    /// SSIM result
    pub ssim: SsimResult,
    /// PSNR result
    pub psnr: PsnrResult,
    /// Delta E result
    pub delta_e: DeltaEResult,
    /// Perceptual hash distance
    pub phash_distance: u32,
    /// Overall pass/fail
    pub passes: bool,
}

impl PixelVerificationSuite {
    /// Run all verification metrics
    #[must_use]
    pub fn verify(
        &self,
        reference: &[Rgb],
        generated: &[Rgb],
        width: u32,
        height: u32,
    ) -> PixelVerificationResult {
        let ssim = self.ssim.compare(reference, generated, width, height);
        let psnr = self.psnr.compare(reference, generated);
        let delta_e = self.delta_e.compare(reference, generated);

        let ref_hash = self.phash.compute(reference, width, height);
        let gen_hash = self.phash.compute(generated, width, height);
        let phash_distance = PerceptualHash::distance(ref_hash, gen_hash);

        // Overall pass: SSIM acceptable AND Delta E acceptable AND PHash similar
        let passes = ssim.is_acceptable
            && delta_e.classification != DeltaEClassification::Unacceptable
            && phash_distance <= 10;

        PixelVerificationResult {
            ssim,
            psnr,
            delta_e,
            phash_distance,
            passes,
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn test_image_white(size: usize) -> Vec<Rgb> {
        vec![Rgb::new(255, 255, 255); size]
    }

    fn test_image_black(size: usize) -> Vec<Rgb> {
        vec![Rgb::new(0, 0, 0); size]
    }

    fn test_image_gray(size: usize) -> Vec<Rgb> {
        vec![Rgb::new(128, 128, 128); size]
    }

    // =========================================================================
    // SSIM Tests (H0-SSIM-XX)
    // =========================================================================

    #[test]
    fn h0_ssim_01_identical_images() {
        let img = test_image_white(100);
        let ssim = SsimMetric::default();
        let result = ssim.compare(&img, &img, 10, 10);
        assert!(result.score >= 0.99);
        assert!(result.is_perfect);
    }

    #[test]
    fn h0_ssim_02_completely_different() {
        let white = test_image_white(100);
        let black = test_image_black(100);
        let ssim = SsimMetric::default();
        let result = ssim.compare(&white, &black, 10, 10);
        assert!(result.score < 0.5);
        assert!(!result.is_acceptable);
    }

    #[test]
    fn h0_ssim_03_similar_images() {
        let gray1: Vec<Rgb> = (0..100).map(|_| Rgb::new(128, 128, 128)).collect();
        let gray2: Vec<Rgb> = (0..100).map(|_| Rgb::new(130, 130, 130)).collect();
        let ssim = SsimMetric::default();
        let result = ssim.compare(&gray1, &gray2, 10, 10);
        assert!(result.score > 0.95);
        assert!(result.is_acceptable);
    }

    #[test]
    fn h0_ssim_04_mismatched_lengths() {
        let img1 = test_image_white(100);
        let img2 = test_image_white(50);
        let ssim = SsimMetric::default();
        let result = ssim.compare(&img1, &img2, 10, 10);
        assert_eq!(result.score, 0.0);
        assert!(!result.is_perfect);
        assert!(!result.is_acceptable);
        assert_eq!(result.channel_scores, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn h0_ssim_05_new_constructor() {
        let ssim = SsimMetric::new(7);
        assert_eq!(ssim.window_size, 7);
        assert_eq!(ssim.perfect_threshold, 0.99);
        assert_eq!(ssim.accept_threshold, 0.95);
    }

    #[test]
    fn h0_ssim_06_with_thresholds() {
        let ssim = SsimMetric::default().with_thresholds(0.98, 0.90);
        assert_eq!(ssim.perfect_threshold, 0.98);
        assert_eq!(ssim.accept_threshold, 0.90);
    }

    #[test]
    fn h0_ssim_07_zero_denominator_branch() {
        // Test the case where denominator could be zero (identical zero images)
        let zeros = vec![Rgb::new(0, 0, 0); 64];
        let ssim = SsimMetric::default();
        let result = ssim.compare(&zeros, &zeros, 8, 8);
        // Both images are identical black - should have perfect SSIM
        assert!(result.score >= 0.99);
    }

    // =========================================================================
    // PSNR Tests (H0-PSNR-XX)
    // =========================================================================

    #[test]
    fn h0_psnr_01_identical_images() {
        let img = test_image_gray(100);
        let psnr = PsnrMetric::default();
        let result = psnr.compare(&img, &img);
        assert!(result.psnr_db.is_infinite());
        assert_eq!(result.quality, PsnrQuality::Identical);
    }

    #[test]
    fn h0_psnr_02_slight_difference() {
        let gray1: Vec<Rgb> = vec![Rgb::new(128, 128, 128); 100];
        let gray2: Vec<Rgb> = vec![Rgb::new(129, 128, 128); 100];
        let psnr = PsnrMetric::default();
        let result = psnr.compare(&gray1, &gray2);
        assert!(result.psnr_db > 40.0); // Very high quality
    }

    #[test]
    fn h0_psnr_03_major_difference() {
        let white = test_image_white(100);
        let black = test_image_black(100);
        let psnr = PsnrMetric::default();
        let result = psnr.compare(&white, &black);
        assert!(result.psnr_db < 10.0);
        assert_eq!(result.quality, PsnrQuality::Poor);
    }

    #[test]
    fn h0_psnr_04_mismatched_lengths() {
        let img1 = test_image_gray(100);
        let img2 = test_image_gray(50);
        let psnr = PsnrMetric::default();
        let result = psnr.compare(&img1, &img2);
        assert_eq!(result.psnr_db, 0.0);
        assert_eq!(result.mse, f32::MAX);
        assert_eq!(result.quality, PsnrQuality::Poor);
    }

    #[test]
    fn h0_psnr_05_empty_images() {
        let empty: Vec<Rgb> = vec![];
        let psnr = PsnrMetric::default();
        let result = psnr.compare(&empty, &empty);
        assert_eq!(result.psnr_db, 0.0);
        assert_eq!(result.mse, f32::MAX);
        assert_eq!(result.quality, PsnrQuality::Poor);
    }

    #[test]
    fn h0_psnr_06_good_quality() {
        // Create images with a difference that results in PSNR between 35-40 dB
        // PSNR = 10 * log10(255^2 / MSE), MSE = diff^2 for grayscale
        // For PSNR ~37: MSE = 65025 / 10^3.7 ≈ 13, diff ≈ 3.6
        let gray1: Vec<Rgb> = vec![Rgb::new(128, 128, 128); 100];
        let gray2: Vec<Rgb> = vec![Rgb::new(132, 132, 132); 100]; // diff of 4 per channel
        let psnr = PsnrMetric::default();
        let result = psnr.compare(&gray1, &gray2);
        // Should be "Good" quality (35 <= PSNR < 40)
        assert!(result.psnr_db >= 35.0 && result.psnr_db < 40.0);
        assert_eq!(result.quality, PsnrQuality::Good);
    }

    #[test]
    fn h0_psnr_07_acceptable_quality() {
        // Create images with a difference that results in PSNR between 30-35 dB
        // For PSNR ~32: MSE = 65025 / 10^3.2 ≈ 41, diff ≈ 6.4
        let gray1: Vec<Rgb> = vec![Rgb::new(128, 128, 128); 100];
        let gray2: Vec<Rgb> = vec![Rgb::new(135, 135, 135); 100]; // diff of 7 produces ~31 dB
        let psnr = PsnrMetric::default();
        let result = psnr.compare(&gray1, &gray2);
        assert!(result.psnr_db >= 30.0 && result.psnr_db < 35.0);
        assert_eq!(result.quality, PsnrQuality::Acceptable);
    }

    #[test]
    fn h0_psnr_08_excellent_quality() {
        // Create images with a very small difference (PSNR >= 40 dB)
        let gray1: Vec<Rgb> = vec![Rgb::new(128, 128, 128); 100];
        let gray2: Vec<Rgb> = vec![Rgb::new(129, 129, 129); 100]; // tiny diff
        let psnr = PsnrMetric::default();
        let result = psnr.compare(&gray1, &gray2);
        assert!(result.psnr_db >= 40.0);
        assert_eq!(result.quality, PsnrQuality::Excellent);
    }

    // =========================================================================
    // Lab Conversion Tests (H0-LAB-XX)
    // =========================================================================

    #[test]
    fn h0_lab_01_white() {
        let lab = Lab::from_rgb(&Rgb::new(255, 255, 255));
        assert!((lab.l - 100.0).abs() < 1.0); // L should be ~100
    }

    #[test]
    fn h0_lab_02_black() {
        let lab = Lab::from_rgb(&Rgb::new(0, 0, 0));
        assert!(lab.l < 1.0); // L should be ~0
    }

    #[test]
    fn h0_lab_03_gray() {
        let lab = Lab::from_rgb(&Rgb::new(128, 128, 128));
        assert!(lab.l > 40.0 && lab.l < 60.0); // L should be ~53
        assert!(lab.a.abs() < 2.0); // a should be ~0
        assert!(lab.b.abs() < 2.0); // b should be ~0
    }

    #[test]
    fn h0_lab_04_new_constructor() {
        let lab = Lab::new(50.0, 25.0, -30.0);
        assert_eq!(lab.l, 50.0);
        assert_eq!(lab.a, 25.0);
        assert_eq!(lab.b, -30.0);
    }

    #[test]
    fn h0_lab_05_srgb_linear_threshold() {
        // Test the srgb_to_linear branch for values <= 0.04045
        // RGB values <= 10 (10/255 ≈ 0.039) should use the linear branch
        let lab = Lab::from_rgb(&Rgb::new(5, 5, 5));
        assert!(lab.l < 5.0); // Very dark
    }

    #[test]
    fn h0_lab_06_srgb_gamma_branch() {
        // Test the srgb_to_linear branch for values > 0.04045
        // RGB values > 10 should use the gamma branch
        let lab = Lab::from_rgb(&Rgb::new(200, 100, 50));
        assert!(lab.l > 40.0); // Should be reasonably bright
    }

    #[test]
    fn h0_lab_07_f_xyz_linear_branch() {
        // Test f_xyz for very dark colors (t <= delta^3)
        let lab = Lab::from_rgb(&Rgb::new(1, 1, 1));
        assert!(lab.l < 2.0);
    }

    #[test]
    fn h0_lab_08_colored_pixels() {
        // Test various colored pixels for a and b values
        let red = Lab::from_rgb(&Rgb::new(255, 0, 0));
        assert!(red.a > 0.0); // Red has positive a

        let green = Lab::from_rgb(&Rgb::new(0, 255, 0));
        assert!(green.a < 0.0); // Green has negative a

        let blue = Lab::from_rgb(&Rgb::new(0, 0, 255));
        assert!(blue.b < 0.0); // Blue has negative b

        let yellow = Lab::from_rgb(&Rgb::new(255, 255, 0));
        assert!(yellow.b > 0.0); // Yellow has positive b
    }

    // =========================================================================
    // CIEDE2000 Tests (H0-DE-XX)
    // =========================================================================

    #[test]
    fn h0_de_01_identical_colors() {
        let metric = CieDe2000Metric::default();
        let lab = Lab::new(50.0, 0.0, 0.0);
        let de = metric.delta_e(&lab, &lab);
        assert!(de < f32::EPSILON);
    }

    #[test]
    fn h0_de_02_just_noticeable() {
        let metric = CieDe2000Metric::default();
        let lab1 = Lab::new(50.0, 0.0, 0.0);
        let lab2 = Lab::new(51.0, 0.0, 0.0);
        let de = metric.delta_e(&lab1, &lab2);
        assert!(de < 2.0); // Should be small
    }

    #[test]
    fn h0_de_03_image_comparison() {
        let gray1 = test_image_gray(100);
        let gray2: Vec<Rgb> = vec![Rgb::new(135, 135, 135); 100];
        let metric = CieDe2000Metric::default();
        let result = metric.compare(&gray1, &gray2);
        assert!(result.mean_delta_e < 10.0);
    }

    #[test]
    fn h0_de_04_mismatched_lengths() {
        let img1 = test_image_gray(100);
        let img2 = test_image_gray(50);
        let metric = CieDe2000Metric::default();
        let result = metric.compare(&img1, &img2);
        assert_eq!(result.mean_delta_e, f32::MAX);
        assert_eq!(result.max_delta_e, f32::MAX);
        assert_eq!(result.percent_imperceptible, 0.0);
        assert_eq!(result.percent_acceptable, 0.0);
        assert_eq!(result.classification, DeltaEClassification::Unacceptable);
    }

    #[test]
    fn h0_de_05_empty_images() {
        let empty: Vec<Rgb> = vec![];
        let metric = CieDe2000Metric::default();
        let result = metric.compare(&empty, &empty);
        assert_eq!(result.mean_delta_e, f32::MAX);
        assert_eq!(result.classification, DeltaEClassification::Unacceptable);
    }

    #[test]
    fn h0_de_06_is_imperceptible() {
        let metric = CieDe2000Metric::default();
        assert!(metric.is_imperceptible(0.5));
        assert!(!metric.is_imperceptible(1.5));
    }

    #[test]
    fn h0_de_07_classification_imperceptible() {
        let img = test_image_gray(100);
        let metric = CieDe2000Metric::default();
        let result = metric.compare(&img, &img);
        assert_eq!(result.classification, DeltaEClassification::Imperceptible);
    }

    #[test]
    fn h0_de_08_classification_just_noticeable() {
        // Create images with small but noticeable difference (mean ΔE between 0.8 and 1.8)
        // Grayscale differences need larger RGB changes to reach the JustNoticeable threshold
        let gray1: Vec<Rgb> = vec![Rgb::new(128, 128, 128); 100];
        let gray2: Vec<Rgb> = vec![Rgb::new(132, 132, 132); 100]; // diff of 4 produces ΔE ~1.0-1.5
        let metric = CieDe2000Metric::default();
        let result = metric.compare(&gray1, &gray2);
        assert_eq!(result.classification, DeltaEClassification::JustNoticeable);
    }

    #[test]
    fn h0_de_09_classification_acceptable() {
        // Create images with acceptable difference (mean ΔE between 1.8 and 2.8)
        let gray1: Vec<Rgb> = vec![Rgb::new(128, 128, 128); 100];
        let gray2: Vec<Rgb> = vec![Rgb::new(134, 134, 134); 100];
        let metric = CieDe2000Metric::default();
        let result = metric.compare(&gray1, &gray2);
        assert_eq!(result.classification, DeltaEClassification::Acceptable);
    }

    #[test]
    fn h0_de_10_classification_noticeable() {
        // Create images with noticeable difference (mean ΔE between 2.8 and 3.7)
        let gray1: Vec<Rgb> = vec![Rgb::new(128, 128, 128); 100];
        let gray2: Vec<Rgb> = vec![Rgb::new(138, 138, 138); 100];
        let metric = CieDe2000Metric::default();
        let result = metric.compare(&gray1, &gray2);
        assert_eq!(result.classification, DeltaEClassification::Noticeable);
    }

    #[test]
    fn h0_de_11_classification_unacceptable() {
        // Create images with large difference (mean ΔE >= 3.7)
        let gray1: Vec<Rgb> = vec![Rgb::new(128, 128, 128); 100];
        let gray2: Vec<Rgb> = vec![Rgb::new(145, 145, 145); 100];
        let metric = CieDe2000Metric::default();
        let result = metric.compare(&gray1, &gray2);
        assert_eq!(result.classification, DeltaEClassification::Unacceptable);
    }

    #[test]
    fn h0_de_12_hue_angle_branches() {
        // Test various hue angle calculation branches in delta_e
        let metric = CieDe2000Metric::default();

        // Test with achromatic colors (a=0, b=0) - should hit h_prime = 0 branch
        let lab1 = Lab::new(50.0, 0.0, 0.0);
        let lab2 = Lab::new(60.0, 0.0, 0.0);
        let de = metric.delta_e(&lab1, &lab2);
        assert!(de > 0.0);

        // Test with chromatic colors - different hue branches
        let lab3 = Lab::new(50.0, 30.0, 40.0);
        let lab4 = Lab::new(50.0, -30.0, -40.0);
        let de2 = metric.delta_e(&lab3, &lab4);
        assert!(de2 > 0.0);
    }

    #[test]
    fn h0_de_13_delta_h_prime_branches() {
        let metric = CieDe2000Metric::default();

        // Test case where |dh| > 180 (dh > 180)
        let lab1 = Lab::new(50.0, 50.0, 5.0); // ~hue 6°
        let lab2 = Lab::new(50.0, -50.0, -5.0); // ~hue 186°
        let de = metric.delta_e(&lab1, &lab2);
        assert!(de > 0.0);

        // Test case where dh < -180
        let lab3 = Lab::new(50.0, -50.0, 5.0); // ~hue 174°
        let lab4 = Lab::new(50.0, 50.0, -5.0); // ~hue 354°
        let de2 = metric.delta_e(&lab3, &lab4);
        assert!(de2 > 0.0);
    }

    #[test]
    fn h0_de_14_h_prime_avg_branches() {
        let metric = CieDe2000Metric::default();

        // Test h_prime_avg calculation with diff > 180 and sum < 360
        let lab1 = Lab::new(50.0, 40.0, 10.0); // low hue
        let lab2 = Lab::new(50.0, 10.0, -40.0); // high hue
        let de = metric.delta_e(&lab1, &lab2);
        assert!(de > 0.0);

        // Test with sum >= 360
        let lab3 = Lab::new(50.0, -10.0, 40.0);
        let lab4 = Lab::new(50.0, -40.0, -10.0);
        let de2 = metric.delta_e(&lab3, &lab4);
        assert!(de2 > 0.0);
    }

    #[test]
    fn h0_de_15_percent_calculations() {
        let gray1: Vec<Rgb> = vec![Rgb::new(128, 128, 128); 100];
        let gray2: Vec<Rgb> = vec![Rgb::new(129, 129, 129); 100];
        let metric = CieDe2000Metric::default();
        let result = metric.compare(&gray1, &gray2);
        // With very small differences, most pixels should be imperceptible/acceptable
        assert!(result.percent_imperceptible > 0.0);
        assert!(result.percent_acceptable > 0.0);
    }

    // =========================================================================
    // Perceptual Hash Tests (H0-PHASH-XX)
    // =========================================================================

    #[test]
    fn h0_phash_01_identical_images() {
        let img = test_image_gray(64);
        let hasher = PerceptualHash::default();
        let hash1 = hasher.compute(&img, 8, 8);
        let hash2 = hasher.compute(&img, 8, 8);
        assert_eq!(hash1, hash2);
        assert_eq!(PerceptualHash::distance(hash1, hash2), 0);
    }

    #[test]
    fn h0_phash_02_similar_images() {
        let gray1 = test_image_gray(64);
        let gray2: Vec<Rgb> = vec![Rgb::new(130, 130, 130); 64];
        let hasher = PerceptualHash::default();
        let hash1 = hasher.compute(&gray1, 8, 8);
        let hash2 = hasher.compute(&gray2, 8, 8);
        let distance = PerceptualHash::distance(hash1, hash2);
        assert!(distance < 10); // Should be similar
    }

    #[test]
    fn h0_phash_03_different_images() {
        let white = test_image_white(64);
        let black = test_image_black(64);
        let hasher = PerceptualHash::default();
        let hash1 = hasher.compute(&white, 8, 8);
        let hash2 = hasher.compute(&black, 8, 8);
        // Hash of uniform images will differ based on mean comparison
        // Just verify it computes without panic
        let _ = PerceptualHash::distance(hash1, hash2);
    }

    #[test]
    fn h0_phash_04_is_similar() {
        assert!(PerceptualHash::is_similar(0, 1, 5));
        assert!(!PerceptualHash::is_similar(0, u64::MAX, 5));
    }

    #[test]
    fn h0_phash_05_ahash() {
        let img = test_image_gray(64);
        let hasher = PerceptualHash::new(PhashAlgorithm::AHash);
        let hash = hasher.compute(&img, 8, 8);
        // Just verify it runs without panic and produces a valid hash
        let _ = hash; // hash computed successfully
    }

    #[test]
    fn h0_phash_06_phash_algorithm() {
        // Test the PHash (DCT-based) algorithm
        let img = test_image_gray(1024); // 32x32 minimum for PHash
        let hasher = PerceptualHash::new(PhashAlgorithm::PHash);
        let hash = hasher.compute(&img, 32, 32);
        // Verify it produces consistent results
        let hash2 = hasher.compute(&img, 32, 32);
        assert_eq!(hash, hash2);
    }

    #[test]
    fn h0_phash_07_dhash_gradient() {
        // Test DHash with a gradient image
        let gradient: Vec<Rgb> = (0..72)
            .map(|i| {
                let v = (i * 3) as u8;
                Rgb::new(v, v, v)
            })
            .collect();
        let hasher = PerceptualHash::new(PhashAlgorithm::DHash);
        let hash = hasher.compute(&gradient, 9, 8);
        // Gradient should produce a non-trivial hash
        assert_ne!(hash, 0);
    }

    #[test]
    fn h0_phash_08_resize_out_of_bounds() {
        // Test resize_grayscale with a very small image
        let small_img = vec![Rgb::new(128, 128, 128); 4]; // 2x2
        let hasher = PerceptualHash::default();
        // Compute on a 2x2 image but expecting 9x8 resize for dhash
        // This should handle the case where src_idx >= image.len()
        let hash = hasher.compute(&small_img, 2, 2);
        let _ = hash; // Should not panic
    }

    #[test]
    fn h0_phash_09_average_hash_varied() {
        // Test average hash with varied pixel values
        let varied: Vec<Rgb> = (0..64)
            .map(|i| {
                let v = if i % 2 == 0 { 50 } else { 200 };
                Rgb::new(v, v, v)
            })
            .collect();
        let hasher = PerceptualHash::new(PhashAlgorithm::AHash);
        let hash = hasher.compute(&varied, 8, 8);
        // Should produce a pattern based on above/below mean
        assert_ne!(hash, 0);
        assert_ne!(hash, u64::MAX);
    }

    #[test]
    fn h0_phash_10_default_values() {
        let hasher = PerceptualHash::default();
        assert_eq!(hasher.algorithm, PhashAlgorithm::DHash);
        assert_eq!(hasher.hash_bits, 64);
    }

    #[test]
    fn h0_phash_11_algorithm_default() {
        let algo = PhashAlgorithm::default();
        assert_eq!(algo, PhashAlgorithm::DHash);
    }

    // =========================================================================
    // Verification Suite Tests (H0-SUITE-XX)
    // =========================================================================

    #[test]
    fn h0_suite_01_identical_images() {
        let img = test_image_gray(100);
        let suite = PixelVerificationSuite::default();
        let result = suite.verify(&img, &img, 10, 10);
        assert!(result.passes);
        assert!(result.ssim.is_perfect);
        assert_eq!(result.psnr.quality, PsnrQuality::Identical);
    }

    #[test]
    fn h0_suite_02_different_images() {
        let white = test_image_white(100);
        let black = test_image_black(100);
        let suite = PixelVerificationSuite::default();
        let result = suite.verify(&white, &black, 10, 10);
        assert!(!result.passes);
    }

    #[test]
    fn h0_suite_03_similar_but_acceptable() {
        let gray1: Vec<Rgb> = vec![Rgb::new(128, 128, 128); 100];
        let gray2: Vec<Rgb> = vec![Rgb::new(130, 130, 130); 100];
        let suite = PixelVerificationSuite::default();
        let result = suite.verify(&gray1, &gray2, 10, 10);
        // Should pass because SSIM is acceptable, Delta E is small, and PHash distance is low
        assert!(result.passes);
        assert!(result.ssim.is_acceptable);
        assert!(result.phash_distance <= 10);
    }

    #[test]
    fn h0_suite_04_phash_distance_check() {
        let suite = PixelVerificationSuite::default();
        let img = test_image_gray(100);
        let result = suite.verify(&img, &img, 10, 10);
        assert_eq!(result.phash_distance, 0);
    }

    #[test]
    fn h0_suite_05_default_metrics() {
        let suite = PixelVerificationSuite::default();
        // Verify all metrics are properly initialized with defaults
        assert_eq!(suite.ssim.window_size, 11);
        assert_eq!(suite.psnr.max_value, 255.0);
        assert_eq!(suite.delta_e.jnd_threshold, 1.0);
        assert_eq!(suite.phash.algorithm, PhashAlgorithm::DHash);
    }
}

// =============================================================================
// Property-Based Tests (Extreme TDD - L2 Falsification Layer)
// =============================================================================

#[cfg(test)]
mod proptest_tests {
    use super::*;
    use proptest::prelude::*;

    // Strategy for generating random RGB pixels
    fn rgb_strategy() -> impl Strategy<Value = Rgb> {
        (0u8..=255, 0u8..=255, 0u8..=255).prop_map(|(r, g, b)| Rgb::new(r, g, b))
    }

    // Strategy for generating random images
    fn image_strategy(size: usize) -> impl Strategy<Value = Vec<Rgb>> {
        proptest::collection::vec(rgb_strategy(), size)
    }

    // =========================================================================
    // SSIM Property Tests (PROP-SSIM-XX)
    // =========================================================================

    proptest! {
        /// PROP-SSIM-01: SSIM of identical images is always 1.0
        #[test]
        fn prop_ssim_01_identical_is_perfect(img in image_strategy(64)) {
            let ssim = SsimMetric::default();
            let result = ssim.compare(&img, &img, 8, 8);
            prop_assert!(result.score >= 0.99, "SSIM of identical images should be ~1.0, got {}", result.score);
        }

        /// PROP-SSIM-02: SSIM is symmetric
        #[test]
        fn prop_ssim_02_symmetric(
            img1 in image_strategy(64),
            img2 in image_strategy(64)
        ) {
            let ssim = SsimMetric::default();
            let result1 = ssim.compare(&img1, &img2, 8, 8);
            let result2 = ssim.compare(&img2, &img1, 8, 8);
            let diff = (result1.score - result2.score).abs();
            prop_assert!(diff < 0.001, "SSIM should be symmetric, diff={}", diff);
        }

        /// PROP-SSIM-03: SSIM is bounded [-1, 1]
        #[test]
        fn prop_ssim_03_bounded(
            img1 in image_strategy(64),
            img2 in image_strategy(64)
        ) {
            let ssim = SsimMetric::default();
            let result = ssim.compare(&img1, &img2, 8, 8);
            prop_assert!(result.score >= -1.0 && result.score <= 1.0,
                "SSIM must be in [-1, 1], got {}", result.score);
        }
    }

    // =========================================================================
    // PSNR Property Tests (PROP-PSNR-XX)
    // =========================================================================

    proptest! {
        /// PROP-PSNR-01: PSNR of identical images is infinity
        #[test]
        fn prop_psnr_01_identical_is_infinite(img in image_strategy(64)) {
            let psnr = PsnrMetric::default();
            let result = psnr.compare(&img, &img);
            prop_assert!(result.psnr_db.is_infinite() || result.psnr_db > 100.0,
                "PSNR of identical images should be infinite, got {}", result.psnr_db);
        }

        /// PROP-PSNR-02: PSNR is symmetric
        #[test]
        fn prop_psnr_02_symmetric(
            img1 in image_strategy(64),
            img2 in image_strategy(64)
        ) {
            let psnr = PsnrMetric::default();
            let result1 = psnr.compare(&img1, &img2);
            let result2 = psnr.compare(&img2, &img1);
            let diff = if result1.psnr_db.is_infinite() && result2.psnr_db.is_infinite() {
                0.0
            } else {
                (result1.psnr_db - result2.psnr_db).abs()
            };
            prop_assert!(diff < 0.001, "PSNR should be symmetric, diff={}", diff);
        }

        /// PROP-PSNR-03: PSNR is non-negative (or infinite)
        #[test]
        fn prop_psnr_03_non_negative(
            img1 in image_strategy(64),
            img2 in image_strategy(64)
        ) {
            let psnr = PsnrMetric::default();
            let result = psnr.compare(&img1, &img2);
            prop_assert!(result.psnr_db >= 0.0 || result.psnr_db.is_infinite(),
                "PSNR must be non-negative, got {}", result.psnr_db);
        }
    }

    // =========================================================================
    // Lab Conversion Property Tests (PROP-LAB-XX)
    // =========================================================================

    proptest! {
        /// PROP-LAB-01: L (lightness) is bounded [0, 100]
        #[test]
        fn prop_lab_01_lightness_bounded(rgb in rgb_strategy()) {
            let lab = Lab::from_rgb(&rgb);
            prop_assert!(lab.l >= -1.0 && lab.l <= 101.0,
                "Lightness should be ~[0, 100], got {}", lab.l);
        }

        /// PROP-LAB-02: Grayscale pixels have near-zero a,b
        #[test]
        fn prop_lab_02_grayscale_neutral(v in 0u8..=255) {
            let rgb = Rgb::new(v, v, v);
            let lab = Lab::from_rgb(&rgb);
            prop_assert!(lab.a.abs() < 2.0, "Grayscale a should be ~0, got {}", lab.a);
            prop_assert!(lab.b.abs() < 2.0, "Grayscale b should be ~0, got {}", lab.b);
        }
    }

    // =========================================================================
    // Delta E Property Tests (PROP-DE-XX)
    // =========================================================================

    proptest! {
        /// PROP-DE-01: Delta E of identical colors is 0
        #[test]
        fn prop_de_01_identical_is_zero(rgb in rgb_strategy()) {
            let metric = CieDe2000Metric::default();
            let lab = Lab::from_rgb(&rgb);
            let de = metric.delta_e(&lab, &lab);
            prop_assert!(de < 0.001, "ΔE of identical colors should be 0, got {}", de);
        }

        /// PROP-DE-02: Delta E is symmetric
        #[test]
        fn prop_de_02_symmetric(
            rgb1 in rgb_strategy(),
            rgb2 in rgb_strategy()
        ) {
            let metric = CieDe2000Metric::default();
            let lab1 = Lab::from_rgb(&rgb1);
            let lab2 = Lab::from_rgb(&rgb2);
            let de1 = metric.delta_e(&lab1, &lab2);
            let de2 = metric.delta_e(&lab2, &lab1);
            let diff = (de1 - de2).abs();
            prop_assert!(diff < 0.001, "ΔE should be symmetric, diff={}", diff);
        }

        /// PROP-DE-03: Delta E is non-negative
        #[test]
        fn prop_de_03_non_negative(
            rgb1 in rgb_strategy(),
            rgb2 in rgb_strategy()
        ) {
            let metric = CieDe2000Metric::default();
            let lab1 = Lab::from_rgb(&rgb1);
            let lab2 = Lab::from_rgb(&rgb2);
            let de = metric.delta_e(&lab1, &lab2);
            prop_assert!(de >= 0.0, "ΔE must be non-negative, got {}", de);
        }

        /// PROP-DE-04: Delta E is bounded (typical max ~100 for most colors)
        /// Note: CIEDE2000 does NOT satisfy triangle inequality by design
        /// (optimized for perceptual matching, not metric space properties)
        #[test]
        fn prop_de_04_bounded(
            rgb1 in rgb_strategy(),
            rgb2 in rgb_strategy()
        ) {
            let metric = CieDe2000Metric::default();
            let lab1 = Lab::from_rgb(&rgb1);
            let lab2 = Lab::from_rgb(&rgb2);
            let de = metric.delta_e(&lab1, &lab2);
            // ΔE₀₀ is typically bounded by ~100 for 8-bit RGB
            prop_assert!(de <= 150.0, "ΔE should be bounded, got {}", de);
        }
    }

    // =========================================================================
    // Perceptual Hash Property Tests (PROP-PHASH-XX)
    // =========================================================================

    proptest! {
        /// PROP-PHASH-01: Hash of identical images has distance 0
        #[test]
        fn prop_phash_01_identical_distance_zero(img in image_strategy(64)) {
            let hasher = PerceptualHash::default();
            let hash1 = hasher.compute(&img, 8, 8);
            let hash2 = hasher.compute(&img, 8, 8);
            prop_assert_eq!(hash1, hash2, "Hash of identical images should be equal");
            prop_assert_eq!(PerceptualHash::distance(hash1, hash2), 0);
        }

        /// PROP-PHASH-02: Hamming distance is symmetric
        #[test]
        fn prop_phash_02_distance_symmetric(h1: u64, h2: u64) {
            let d1 = PerceptualHash::distance(h1, h2);
            let d2 = PerceptualHash::distance(h2, h1);
            prop_assert_eq!(d1, d2, "Hamming distance should be symmetric");
        }

        /// PROP-PHASH-03: Hamming distance bounded by 64
        #[test]
        fn prop_phash_03_distance_bounded(h1: u64, h2: u64) {
            let d = PerceptualHash::distance(h1, h2);
            prop_assert!(d <= 64, "Hamming distance should be <= 64, got {}", d);
        }

        /// PROP-PHASH-04: Distance to self is 0
        #[test]
        fn prop_phash_04_self_distance_zero(h: u64) {
            prop_assert_eq!(PerceptualHash::distance(h, h), 0);
        }
    }

    // =========================================================================
    // Verification Suite Property Tests (PROP-SUITE-XX)
    // =========================================================================

    proptest! {
        /// PROP-SUITE-01: Identical images always pass verification
        #[test]
        fn prop_suite_01_identical_passes(img in image_strategy(64)) {
            let suite = PixelVerificationSuite::default();
            let result = suite.verify(&img, &img, 8, 8);
            prop_assert!(result.passes, "Identical images should pass verification");
            prop_assert!(result.ssim.is_acceptable);
        }
    }
}
