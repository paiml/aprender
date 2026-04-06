//! Exponential and logarithmic functions: `exp`, `ln`, `log2`, `log10`

use crate::backends::VectorBackend;
use crate::vector::Vector;
use crate::{dispatch_unary_op, Result};

impl Vector<f32> {
    /// Element-wise exponential: result\[i\] = e^x\[i\]
    ///
    /// Computes the natural exponential (e^x) for each element.
    /// Uses Rust's optimized f32::exp() method.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[0.0, 1.0, 2.0]);
    /// let result = v.exp()?;
    /// // result ≈ [1.0, 2.718, 7.389]
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Special Cases
    ///
    /// - `exp(0.0)` returns 1.0
    /// - `exp(1.0)` returns e ≈ 2.71828
    /// - `exp(-∞)` returns 0.0
    /// - `exp(+∞)` returns +∞
    ///
    /// # Applications
    ///
    /// - Machine learning: Softmax activation, sigmoid, exponential loss
    /// - Statistics: Exponential distribution, log-normal distribution
    /// - Physics: Radioactive decay, population growth models
    /// - Signal processing: Exponential smoothing, envelope detection
    /// - Numerical methods: Solving differential equations
    pub fn exp(&self) -> Result<Vector<f32>> {
        // Uninit: backend writes every element before any read.
        let n = self.len();
        let mut result_data: Vec<f32> = Vec::with_capacity(n);
        // SAFETY: Backend writes all elements before any read.
        unsafe {
            result_data.set_len(n);
        }

        if !self.data.is_empty() {
            // Use parallel processing for large arrays
            #[cfg(feature = "parallel")]
            {
                const PARALLEL_THRESHOLD: usize = 100_000;
                const CHUNK_SIZE: usize = 65536;

                if self.len() >= PARALLEL_THRESHOLD {
                    use rayon::prelude::*;

                    self.data
                        .par_chunks(CHUNK_SIZE)
                        .zip(result_data.par_chunks_mut(CHUNK_SIZE))
                        .for_each(|(chunk_in, chunk_out)| {
                            dispatch_unary_op!(self.backend, exp, chunk_in, chunk_out);
                        });

                    return Ok(Vector { data: result_data, backend: self.backend });
                }
            }

            dispatch_unary_op!(self.backend, exp, &self.data, &mut result_data);
        }

        Ok(Vector { data: result_data, backend: self.backend })
    }

    /// Element-wise natural logarithm: result\[i\] = ln(x\[i\])
    ///
    /// Computes the natural logarithm (base e) for each element.
    /// Uses Rust's optimized f32::ln() method.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[1.0, std::f32::consts::E, std::f32::consts::E.powi(2)]);
    /// let result = v.ln()?;
    /// // result ≈ [0.0, 1.0, 2.0]
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Special Cases
    ///
    /// - `ln(1.0)` returns 0.0
    /// - `ln(e)` returns 1.0
    /// - `ln(x)` for x ≤ 0 returns NaN
    /// - `ln(0.0)` returns -∞
    /// - `ln(+∞)` returns +∞
    ///
    /// # Applications
    ///
    /// - Machine learning: Log loss, log-likelihood, softmax normalization
    /// - Statistics: Log-normal distribution, log transformation for skewed data
    /// - Information theory: Entropy calculation, mutual information
    /// - Economics: Log returns, elasticity calculations
    /// - Signal processing: Decibel conversion, log-frequency analysis
    pub fn ln(&self) -> Result<Vector<f32>> {
        // Uninit: backend writes every element before any read.
        let n = self.len();
        let mut result_data: Vec<f32> = Vec::with_capacity(n);
        // SAFETY: Backend writes all elements before any read.
        unsafe {
            result_data.set_len(n);
        }

        if !self.data.is_empty() {
            dispatch_unary_op!(self.backend, ln, &self.data, &mut result_data);
        }

        Ok(Vector { data: result_data, backend: self.backend })
    }

    /// Element-wise base-2 logarithm: result\[i\] = log₂(x\[i\])
    ///
    /// Computes the base-2 logarithm for each element.
    /// Uses Rust's optimized f32::log2() method.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[1.0, 2.0, 4.0, 8.0]);
    /// let result = v.log2()?;
    /// // result ≈ [0.0, 1.0, 2.0, 3.0]
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Special Cases
    ///
    /// - `log2(1.0)` returns 0.0
    /// - `log2(2.0)` returns 1.0
    /// - `log2(x)` for x ≤ 0 returns NaN
    /// - `log2(0.0)` returns -∞
    /// - `log2(+∞)` returns +∞
    ///
    /// # Applications
    ///
    /// - Information theory: Entropy in bits, mutual information
    /// - Computer science: Bit manipulation, binary search complexity
    /// - Audio: Octave calculations, pitch detection
    /// - Data compression: Huffman coding, arithmetic coding
    pub fn log2(&self) -> Result<Vector<f32>> {
        // Uninit: backend writes every element before any read.
        let n = self.len();
        let mut result_data: Vec<f32> = Vec::with_capacity(n);
        // SAFETY: Backend writes all elements before any read.
        unsafe {
            result_data.set_len(n);
        }

        if !self.data.is_empty() {
            dispatch_unary_op!(self.backend, log2, &self.data, &mut result_data);
        }

        Ok(Vector { data: result_data, backend: self.backend })
    }

    /// Element-wise base-10 logarithm: result\[i\] = log₁₀(x\[i\])
    ///
    /// Computes the base-10 (common) logarithm for each element.
    /// Uses Rust's optimized f32::log10() method.
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[1.0, 10.0, 100.0, 1000.0]);
    /// let result = v.log10()?;
    /// // result ≈ [0.0, 1.0, 2.0, 3.0]
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Special Cases
    ///
    /// - `log10(1.0)` returns 0.0
    /// - `log10(10.0)` returns 1.0
    /// - `log10(x)` for x ≤ 0 returns NaN
    /// - `log10(0.0)` returns -∞
    /// - `log10(+∞)` returns +∞
    ///
    /// # Applications
    ///
    /// - Audio: Decibel calculations (dB = 20 * log10(amplitude))
    /// - Chemistry: pH calculations (-log10(H+ concentration))
    /// - Seismology: Richter scale
    /// - Scientific notation: Order of magnitude calculations
    pub fn log10(&self) -> Result<Vector<f32>> {
        // Uninit: backend writes every element before any read.
        let n = self.len();
        let mut result_data: Vec<f32> = Vec::with_capacity(n);
        // SAFETY: Backend writes all elements before any read.
        unsafe {
            result_data.set_len(n);
        }

        if !self.data.is_empty() {
            dispatch_unary_op!(self.backend, log10, &self.data, &mut result_data);
        }

        Ok(Vector { data: result_data, backend: self.backend })
    }
}
