//! Hyperbolic and inverse hyperbolic functions: `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`

#[cfg(target_arch = "x86_64")]
use crate::backends::avx2::Avx2Backend;
#[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
use crate::backends::neon::NeonBackend;
use crate::backends::scalar::ScalarBackend;
#[cfg(target_arch = "x86_64")]
use crate::backends::sse2::Sse2Backend;
#[cfg(target_arch = "wasm32")]
use crate::backends::wasm::WasmBackend;
use crate::backends::VectorBackend;
use crate::vector::Vector;
use crate::{Backend, Result, TruenoError};

impl Vector<f32> {
    /// Computes the hyperbolic sine (sinh) of each element.
    ///
    /// # Mathematical Definition
    ///
    /// sinh(x) = (e^x - e^(-x)) / 2
    ///
    /// # Properties
    ///
    /// - Domain: (-∞, +∞)
    /// - Range: (-∞, +∞)
    /// - Odd function: sinh(-x) = -sinh(x)
    /// - sinh(0) = 0
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[0.0, 1.0, -1.0]);
    /// let result = v.sinh()?;
    /// assert!((result.as_slice()[0] - 0.0).abs() < 1e-5);
    /// # Ok(())
    /// # }
    /// ```
    pub fn sinh(&self) -> Result<Vector<f32>> {
        let sinh_data: Vec<f32> = self.data.iter().map(|x| x.sinh()).collect();
        Ok(Vector { data: sinh_data, backend: self.backend })
    }

    /// Computes the hyperbolic cosine (cosh) of each element.
    ///
    /// # Mathematical Definition
    ///
    /// cosh(x) = (e^x + e^(-x)) / 2
    ///
    /// # Properties
    ///
    /// - Domain: (-∞, +∞)
    /// - Range: [1, +∞)
    /// - Even function: cosh(-x) = cosh(x)
    /// - cosh(0) = 1
    /// - Always positive: cosh(x) ≥ 1 for all x
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[0.0, 1.0, -1.0]);
    /// let result = v.cosh()?;
    /// assert!((result.as_slice()[0] - 1.0).abs() < 1e-5);
    /// # Ok(())
    /// # }
    /// ```
    pub fn cosh(&self) -> Result<Vector<f32>> {
        let cosh_data: Vec<f32> = self.data.iter().map(|x| x.cosh()).collect();
        Ok(Vector { data: cosh_data, backend: self.backend })
    }

    /// Computes the hyperbolic tangent (tanh) of each element.
    ///
    /// # Mathematical Definition
    ///
    /// tanh(x) = sinh(x) / cosh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    ///
    /// # Properties
    ///
    /// - Domain: (-∞, +∞)
    /// - Range: (-1, 1)
    /// - Odd function: tanh(-x) = -tanh(x)
    /// - tanh(0) = 0
    /// - Bounded: -1 < tanh(x) < 1 for all x
    /// - Commonly used as activation function in neural networks
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[0.0, 1.0, -1.0]);
    /// let result = v.tanh()?;
    /// assert!((result.as_slice()[0] - 0.0).abs() < 1e-5);
    /// // All values are in range (-1, 1)
    /// assert!(result.as_slice().iter().all(|&x| x > -1.0 && x < 1.0));
    /// # Ok(())
    /// # }
    /// ```
    pub fn tanh(&self) -> Result<Vector<f32>> {
        if self.data.is_empty() {
            return Err(TruenoError::EmptyVector);
        }

        // OpComplexity::Low - GPU threshold: >100K elements
        #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
        const GPU_THRESHOLD: usize = usize::MAX; // GPU DISABLED - 2-800x slower, see docs/performance-analysis.md

        // Try GPU first for large vectors
        #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
        {
            if self.data.len() >= GPU_THRESHOLD {
                use crate::backends::gpu::GpuDevice;
                if GpuDevice::is_available() {
                    let gpu = GpuDevice::new().map_err(TruenoError::InvalidInput)?;
                    let mut result = vec![0.0; self.data.len()];
                    if gpu.tanh(&self.data, &mut result).is_ok() {
                        return Ok(Vector::from_vec(result));
                    }
                }
            }
        }

        // Uninit: backend writes every element before any read.
        let n = self.len();
        let mut result: Vec<f32> = Vec::with_capacity(n);
        // SAFETY: Backend writes all elements before any read.
        unsafe {
            result.set_len(n);
        }

        // Dispatch to appropriate SIMD backend
        // SAFETY: Unsafe block delegates to backend implementation which maintains safety invariants
        unsafe {
            match self.backend {
                Backend::Scalar => {
                    ScalarBackend::tanh(&self.data, &mut result);
                }
                #[cfg(target_arch = "x86_64")]
                Backend::SSE2 | Backend::AVX => {
                    Sse2Backend::tanh(&self.data, &mut result);
                }
                #[cfg(target_arch = "x86_64")]
                Backend::AVX2 | Backend::AVX512 => {
                    Avx2Backend::tanh(&self.data, &mut result);
                }
                #[cfg(not(target_arch = "x86_64"))]
                Backend::SSE2 | Backend::AVX | Backend::AVX2 | Backend::AVX512 => {
                    ScalarBackend::tanh(&self.data, &mut result);
                }
                #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
                Backend::NEON => {
                    NeonBackend::tanh(&self.data, &mut result);
                }
                #[cfg(not(any(target_arch = "aarch64", target_arch = "arm")))]
                Backend::NEON => {
                    ScalarBackend::tanh(&self.data, &mut result);
                }
                #[cfg(target_arch = "wasm32")]
                Backend::WasmSIMD => {
                    WasmBackend::tanh(&self.data, &mut result);
                }
                #[cfg(not(target_arch = "wasm32"))]
                Backend::WasmSIMD => {
                    ScalarBackend::tanh(&self.data, &mut result);
                }
                Backend::GPU | Backend::Auto => {
                    // Auto should have been resolved at Vector creation
                    // GPU falls back to best available SIMD
                    #[cfg(target_arch = "x86_64")]
                    {
                        if is_x86_feature_detected!("avx2") {
                            Avx2Backend::tanh(&self.data, &mut result);
                        } else {
                            Sse2Backend::tanh(&self.data, &mut result);
                        }
                    }
                    #[cfg(not(target_arch = "x86_64"))]
                    {
                        ScalarBackend::tanh(&self.data, &mut result);
                    }
                }
            }
        }

        Ok(Vector { data: result, backend: self.backend })
    }

    /// Computes the inverse hyperbolic sine (asinh) of each element.
    ///
    /// # Mathematical Definition
    ///
    /// asinh(x) = ln(x + sqrt(x² + 1))
    ///
    /// # Properties
    ///
    /// - Domain: (-∞, +∞)
    /// - Range: (-∞, +∞)
    /// - Odd function: asinh(-x) = -asinh(x)
    /// - asinh(0) = 0
    /// - Inverse of sinh: asinh(sinh(x)) = x
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[0.0, 1.0, -1.0]);
    /// let result = v.asinh()?;
    /// assert!((result.as_slice()[0] - 0.0).abs() < 1e-5);
    /// # Ok(())
    /// # }
    /// ```
    pub fn asinh(&self) -> Result<Vector<f32>> {
        let asinh_data: Vec<f32> = self.data.iter().map(|x| x.asinh()).collect();
        Ok(Vector { data: asinh_data, backend: self.backend })
    }

    /// Computes the inverse hyperbolic cosine (acosh) of each element.
    ///
    /// # Mathematical Definition
    ///
    /// acosh(x) = ln(x + sqrt(x² - 1))
    ///
    /// # Properties
    ///
    /// - Domain: [1, +∞)
    /// - Range: [0, +∞)
    /// - acosh(1) = 0
    /// - Inverse of cosh: acosh(cosh(x)) = x for x >= 0
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
    /// let result = v.acosh()?;
    /// assert!((result.as_slice()[0] - 0.0).abs() < 1e-5);
    /// # Ok(())
    /// # }
    /// ```
    pub fn acosh(&self) -> Result<Vector<f32>> {
        let acosh_data: Vec<f32> = self.data.iter().map(|x| x.acosh()).collect();
        Ok(Vector { data: acosh_data, backend: self.backend })
    }

    /// Computes the inverse hyperbolic tangent (atanh) of each element.
    ///
    /// Domain: (-1, 1)
    /// Range: (-∞, +∞)
    ///
    /// # Examples
    ///
    /// ```
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use trueno::Vector;
    ///
    /// let v = Vector::from_slice(&[0.0, 0.5, -0.5]);
    /// let result = v.atanh()?;
    /// // atanh(0) = 0, atanh(0.5) ≈ 0.549, atanh(-0.5) ≈ -0.549
    /// # Ok(())
    /// # }
    /// ```
    pub fn atanh(&self) -> Result<Vector<f32>> {
        let atanh_data: Vec<f32> = self.data.iter().map(|x| x.atanh()).collect();
        Ok(Vector { data: atanh_data, backend: self.backend })
    }
}
