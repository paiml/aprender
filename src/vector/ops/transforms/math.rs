//! Mathematical transformation operations (sqrt, recip, pow)

#[allow(unused_imports)]
use crate::backends::VectorBackend;
use crate::dispatch_unary_op;
use crate::{Result, Vector};

impl Vector<f32> {
    /// Element-wise square root: result\[i\] = sqrt(self\[i\])
    ///
    /// Computes the square root of each element. For negative values, returns NaN
    /// following IEEE 754 floating-point semantics.
    ///
    /// # Returns
    ///
    /// A new vector where each element is the square root of the corresponding input element
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use trueno::Vector;
    ///
    /// let a = Vector::from_slice(&[4.0, 9.0, 16.0, 25.0]);
    /// let result = a.sqrt()?;
    /// assert_eq!(result.as_slice(), &[2.0, 3.0, 4.0, 5.0]);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// Negative values produce NaN:
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use trueno::Vector;
    ///
    /// let a = Vector::from_slice(&[-1.0, 4.0]);
    /// let result = a.sqrt()?;
    /// assert!(result.as_slice()[0].is_nan());
    /// assert_eq!(result.as_slice()[1], 2.0);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Use Cases
    ///
    /// - Distance calculations: Euclidean distance computation
    /// - Statistics: Standard deviation, RMS (root mean square)
    /// - Machine learning: Normalization, gradient descent with adaptive learning rates
    /// - Signal processing: Amplitude calculations, power spectrum analysis
    /// - Physics simulations: Velocity from kinetic energy, wave propagation
    pub fn sqrt(&self) -> Result<Vector<f32>> {
        // Uninit allocation: dispatch_unary_op!(sqrt) writes every element.
        let n = self.len();
        let mut result_data: Vec<f32> = Vec::with_capacity(n);
        // SAFETY: dispatch_unary_op writes result_data[i] = sqrt(input[i]) for all i.
        unsafe {
            result_data.set_len(n);
        }

        if !self.as_slice().is_empty() {
            // Use parallel processing for large arrays
            #[cfg(feature = "parallel")]
            {
                const PARALLEL_THRESHOLD: usize = 100_000;
                const CHUNK_SIZE: usize = 65536;

                if self.len() >= PARALLEL_THRESHOLD {
                    use rayon::prelude::*;

                    self.as_slice()
                        .par_chunks(CHUNK_SIZE)
                        .zip(result_data.par_chunks_mut(CHUNK_SIZE))
                        .for_each(|(chunk_in, chunk_out)| {
                            dispatch_unary_op!(self.backend(), sqrt, chunk_in, chunk_out);
                        });

                    return Ok(Vector { data: result_data, backend: self.backend() });
                }
            }

            dispatch_unary_op!(self.backend(), sqrt, self.as_slice(), &mut result_data);
        }

        Ok(Vector { data: result_data, backend: self.backend() })
    }

    /// Element-wise reciprocal: result\[i\] = 1 / self\[i\]
    ///
    /// Computes the reciprocal (multiplicative inverse) of each element.
    /// For zero values, returns infinity following IEEE 754 floating-point semantics.
    ///
    /// # Returns
    ///
    /// A new vector where each element is the reciprocal of the corresponding input element
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let a = Vector::from_slice(&[2.0, 4.0, 5.0, 10.0]);
    /// let result = a.recip().unwrap();
    /// assert_eq!(result.as_slice(), &[0.5, 0.25, 0.2, 0.1]);
    /// ```
    ///
    /// Zero values produce infinity:
    /// ```
    /// use trueno::Vector;
    ///
    /// let a = Vector::from_slice(&[0.0, 2.0]);
    /// let result = a.recip().unwrap();
    /// assert!(result.as_slice()[0].is_infinite());
    /// assert_eq!(result.as_slice()[1], 0.5);
    /// ```
    ///
    /// # Use Cases
    ///
    /// - Division optimization: `a / b` -> `a * recip(b)` (multiplication is faster)
    /// - Neural networks: Learning rate schedules, weight normalization
    /// - Statistics: Harmonic mean calculations, inverse transformations
    /// - Physics: Resistance (R = 1/G), optical power (P = 1/f)
    /// - Signal processing: Frequency to period conversion, filter design
    pub fn recip(&self) -> Result<Vector<f32>> {
        // Uninit allocation: dispatch_unary_op!(recip) writes every element.
        let n = self.len();
        let mut result_data: Vec<f32> = Vec::with_capacity(n);
        // SAFETY: dispatch_unary_op writes result_data[i] = 1/input[i] for all i.
        unsafe {
            result_data.set_len(n);
        }

        if !self.as_slice().is_empty() {
            dispatch_unary_op!(self.backend(), recip, self.as_slice(), &mut result_data);
        }

        Ok(Vector { data: result_data, backend: self.backend() })
    }

    /// Element-wise power: result\[i\] = base\[i\]^n
    ///
    /// Raises each element to the given power `n`.
    /// Uses Rust's optimized f32::powf() method.
    ///
    /// # Examples
    ///
    /// ```
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[2.0, 3.0, 4.0]);
    /// let squared = v.pow(2.0).unwrap();
    /// assert_eq!(squared.as_slice(), &[4.0, 9.0, 16.0]);
    ///
    /// let sqrt = v.pow(0.5).unwrap();  // Fractional power = root
    /// ```
    ///
    /// # Special Cases
    ///
    /// - `x.pow(0.0)` returns 1.0 for all x (even x=0)
    /// - `x.pow(1.0)` returns x (identity)
    /// - `x.pow(-1.0)` returns 1/x (reciprocal)
    /// - `x.pow(0.5)` returns sqrt(x) (square root)
    ///
    /// # Applications
    ///
    /// - Statistics: Power transformations (Box-Cox, Yeo-Johnson)
    /// - Machine learning: Polynomial features, activation functions
    /// - Physics: Inverse square law (1/r^2), power laws
    /// - Signal processing: Power spectral density, root mean square
    pub fn pow(&self, n: f32) -> Result<Vector<f32>> {
        let pow_data: Vec<f32> = self.as_slice().iter().map(|x| x.powf(n)).collect();
        Ok(Vector::from_vec(pow_data))
    }
}
