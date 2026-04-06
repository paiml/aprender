//! Trigonometric and inverse trigonometric functions: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`

use crate::backends::VectorBackend;
use crate::vector::Vector;
use crate::{dispatch_unary_op, Result};

impl Vector<f32> {
    /// Element-wise sine: result\[i\] = sin(x\[i\])
    ///
    /// Computes the sine for each element (input in radians).
    /// Uses Rust's optimized f32::sin() method.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use trueno::Vector;
    /// use std::f32::consts::PI;
    ///
    /// let v = Vector::from_slice(&[0.0, PI / 2.0, PI]);
    /// let result = v.sin()?;
    /// // result ≈ [0.0, 1.0, 0.0]
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Special Cases
    ///
    /// - `sin(0)` returns 0.0
    /// - `sin(π/2)` returns 1.0
    /// - `sin(π)` returns 0.0 (approximately)
    /// - `sin(-x)` returns -sin(x) (odd function)
    /// - Periodic with period 2π: sin(x + 2π) = sin(x)
    ///
    /// # Applications
    ///
    /// - Signal processing: Waveform generation, oscillators, modulation
    /// - Physics: Harmonic motion, wave propagation, pendulums
    /// - Audio: Synthesizers, tone generation, effects processing
    /// - Graphics: Animation, rotation transformations, procedural generation
    /// - Fourier analysis: Frequency decomposition, spectral analysis
    pub fn sin(&self) -> Result<Vector<f32>> {
        // Uninit: backend writes every element before any read.
        let n = self.len();
        let mut result_data: Vec<f32> = Vec::with_capacity(n);
        // SAFETY: Backend writes all elements before any read.
        unsafe {
            result_data.set_len(n);
        }

        if !self.data.is_empty() {
            dispatch_unary_op!(self.backend, sin, &self.data, &mut result_data);
        }

        Ok(Vector { data: result_data, backend: self.backend })
    }

    /// Element-wise cosine: result\[i\] = cos(x\[i\])
    ///
    /// Computes the cosine for each element (input in radians).
    /// Uses Rust's optimized f32::cos() method.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use trueno::Vector;
    /// use std::f32::consts::PI;
    ///
    /// let v = Vector::from_slice(&[0.0, PI / 2.0, PI]);
    /// let result = v.cos()?;
    /// // result ≈ [1.0, 0.0, -1.0]
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Special Cases
    ///
    /// - `cos(0)` returns 1.0
    /// - `cos(π/2)` returns 0.0 (approximately)
    /// - `cos(π)` returns -1.0
    /// - `cos(-x)` returns cos(x) (even function)
    /// - Periodic with period 2π: cos(x + 2π) = cos(x)
    /// - Relation to sine: cos(x) = sin(x + π/2)
    ///
    /// # Applications
    ///
    /// - Signal processing: Phase-shifted waveforms, I/Q modulation, quadrature signals
    /// - Physics: Projectile motion, wave interference, damped oscillations
    /// - Graphics: Rotation matrices, camera transforms, circular motion
    /// - Audio: Stereo panning, spatial audio, frequency synthesis
    /// - Engineering: Control systems, frequency response, AC circuits
    pub fn cos(&self) -> Result<Vector<f32>> {
        // Uninit: backend writes every element before any read.
        let n = self.len();
        let mut result_data: Vec<f32> = Vec::with_capacity(n);
        // SAFETY: Backend writes all elements before any read.
        unsafe {
            result_data.set_len(n);
        }

        if !self.data.is_empty() {
            dispatch_unary_op!(self.backend, cos, &self.data, &mut result_data);
        }

        Ok(Vector { data: result_data, backend: self.backend })
    }

    /// Computes element-wise tangent (tan) of the vector.
    ///
    /// Returns a new vector where each element is the tangent of the corresponding input element.
    /// tan(x) = sin(x) / cos(x)
    ///
    /// # Returns
    /// - `Ok(Vector<f32>)`: New vector with tan(x) for each element
    ///
    /// # Properties
    /// - Odd function: tan(-x) = -tan(x)
    /// - Period: 2π (not π, despite common misconception)
    /// - Undefined at x = π/2 + nπ (where n is any integer)
    /// - tan(x) = sin(x) / cos(x)
    /// - Range: (-∞, +∞)
    ///
    /// # Performance
    /// - Iterator map pattern for cache efficiency
    /// - Leverages Rust's optimized f32::tan()
    /// - Auto-vectorized by LLVM on supporting platforms
    ///
    /// # Examples
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use trueno::Vector;
    /// use std::f32::consts::PI;
    ///
    /// let angles = Vector::from_slice(&[0.0, PI / 4.0, -PI / 4.0]);
    /// let result = angles.tan()?;
    /// // Result: [0.0, 1.0, -1.0] (approximately)
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Use Cases
    /// - Trigonometry: Slope calculations, angle relationships
    /// - Signal processing: Phase analysis, modulation
    /// - Physics: Projectile trajectories, optics (Snell's law angles)
    /// - Graphics: Perspective projection, field of view calculations
    /// - Engineering: Slope gradients, tangent lines to curves
    pub fn tan(&self) -> Result<Vector<f32>> {
        // Uninit: backend writes every element before any read.
        let n = self.len();
        let mut result_data: Vec<f32> = Vec::with_capacity(n);
        // SAFETY: Backend writes all elements before any read.
        unsafe {
            result_data.set_len(n);
        }

        if !self.data.is_empty() {
            dispatch_unary_op!(self.backend, tan, &self.data, &mut result_data);
        }

        Ok(Vector { data: result_data, backend: self.backend })
    }

    /// Computes element-wise arcsine (asin/sin⁻¹) of the vector.
    ///
    /// Returns a new vector where each element is the inverse sine of the corresponding input element.
    /// This is the inverse function of sin: if y = sin(x), then x = asin(y).
    ///
    /// # Returns
    /// - `Ok(Vector<f32>)`: New vector with asin(x) for each element
    ///
    /// # Properties
    /// - Domain: [-1, 1] (inputs outside this range produce NaN)
    /// - Range: [-π/2, π/2]
    /// - Odd function: asin(-x) = -asin(x)
    /// - Inverse relation: asin(sin(x)) = x for x ∈ [-π/2, π/2]
    /// - asin(0) = 0
    /// - asin(1) = π/2
    /// - asin(-1) = -π/2
    ///
    /// # Performance
    /// - Iterator map pattern for cache efficiency
    /// - Leverages Rust's optimized f32::asin()
    /// - Auto-vectorized by LLVM on supporting platforms
    ///
    /// # Examples
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use trueno::Vector;
    /// use std::f32::consts::PI;
    ///
    /// let values = Vector::from_slice(&[0.0, 0.5, 1.0]);
    /// let result = values.asin()?;
    /// // Result: [0.0, π/6, π/2] (approximately)
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Use Cases
    /// - Physics: Calculating angles from sine values in mechanics, optics
    /// - Signal processing: Phase recovery, demodulation
    /// - Graphics: Inverse transformations, angle calculations
    /// - Navigation: GPS calculations, spherical trigonometry
    /// - Control systems: Inverse kinematics, servo positioning
    pub fn asin(&self) -> Result<Vector<f32>> {
        let asin_data: Vec<f32> = self.data.iter().map(|x| x.asin()).collect();
        Ok(Vector { data: asin_data, backend: self.backend })
    }

    /// Computes element-wise arccosine (acos/cos⁻¹) of the vector.
    ///
    /// Returns a new vector where each element is the inverse cosine of the corresponding input element.
    /// This is the inverse function of cos: if y = cos(x), then x = acos(y).
    ///
    /// # Returns
    /// - `Ok(Vector<f32>)`: New vector with acos(x) for each element
    ///
    /// # Properties
    /// - Domain: [-1, 1] (inputs outside this range produce NaN)
    /// - Range: [0, π]
    /// - Symmetry: acos(-x) = π - acos(x)
    /// - Inverse relation: acos(cos(x)) = x for x ∈ [0, π]
    /// - acos(0) = π/2
    /// - acos(1) = 0
    /// - acos(-1) = π
    ///
    /// # Performance
    /// - Iterator map pattern for cache efficiency
    /// - Leverages Rust's optimized f32::acos()
    /// - Auto-vectorized by LLVM on supporting platforms
    ///
    /// # Examples
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use trueno::Vector;
    /// use std::f32::consts::PI;
    ///
    /// let values = Vector::from_slice(&[0.0, 0.5, 1.0]);
    /// let result = values.acos()?;
    /// // Result: [π/2, π/3, 0.0] (approximately)
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Use Cases
    /// - Physics: Angle calculations in mechanics, optics, reflections
    /// - Signal processing: Phase analysis, correlation functions
    /// - Graphics: View angle calculations, lighting models
    /// - Navigation: Bearing calculations, great circle distances
    /// - Robotics: Joint angle solving, orientation calculations
    pub fn acos(&self) -> Result<Vector<f32>> {
        let acos_data: Vec<f32> = self.data.iter().map(|x| x.acos()).collect();
        Ok(Vector { data: acos_data, backend: self.backend })
    }

    /// Computes element-wise arctangent (atan/tan⁻¹) of the vector.
    ///
    /// Returns a new vector where each element is the inverse tangent of the corresponding input element.
    /// This is the inverse function of tan: if y = tan(x), then x = atan(y).
    ///
    /// # Returns
    /// - `Ok(Vector<f32>)`: New vector with atan(x) for each element
    ///
    /// # Properties
    /// - Domain: All real numbers (-∞, +∞)
    /// - Range: (-π/2, π/2)
    /// - Odd function: atan(-x) = -atan(x)
    /// - Inverse relation: atan(tan(x)) = x for x ∈ (-π/2, π/2)
    /// - atan(0) = 0
    /// - atan(1) = π/4
    /// - atan(-1) = -π/4
    /// - lim(x→∞) atan(x) = π/2
    /// - lim(x→-∞) atan(x) = -π/2
    ///
    /// # Performance
    /// - Iterator map pattern for cache efficiency
    /// - Leverages Rust's optimized f32::atan()
    /// - Auto-vectorized by LLVM on supporting platforms
    ///
    /// # Examples
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use trueno::Vector;
    /// use std::f32::consts::PI;
    ///
    /// let values = Vector::from_slice(&[0.0, 1.0, -1.0]);
    /// let result = values.atan()?;
    /// // Result: [0.0, π/4, -π/4] (approximately)
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Use Cases
    /// - Physics: Angle calculations from slopes, velocity components
    /// - Signal processing: Phase unwrapping, FM demodulation
    /// - Graphics: Rotation calculations, camera orientation
    /// - Robotics: Inverse kinematics, steering angles
    /// - Navigation: Heading calculations from coordinates
    pub fn atan(&self) -> Result<Vector<f32>> {
        let atan_data: Vec<f32> = self.data.iter().map(|x| x.atan()).collect();
        Ok(Vector { data: atan_data, backend: self.backend })
    }
}
