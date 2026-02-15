
impl HeContext {
    /// Create new HE context with given security level
    pub fn new(security_level: SecurityLevel) -> Result<Self> {
        let mut params = HeParameters::default_128bit();
        params.security_level = security_level;
        params.validate()?;

        Ok(Self { params })
    }

    /// Create with custom parameters
    pub fn with_params(params: HeParameters) -> Result<Self> {
        params.validate()?;
        Ok(Self { params })
    }

    /// Get parameters
    #[must_use]
    pub const fn params(&self) -> &HeParameters {
        &self.params
    }

    /// Generate key pair
    ///
    /// Returns (public_key, secret_key)
    pub fn generate_keys(&self) -> Result<(HePublicKey, HeSecretKey)> {
        // Stub implementation - real impl requires SEAL bindings
        // Size estimates per spec §7.2
        let pk_size = match self.params.security_level {
            SecurityLevel::Bit128 => 1_600_000, // ~1.6 MB
            SecurityLevel::Bit192 => 3_200_000,
            SecurityLevel::Bit256 => 6_400_000,
        };

        let public_key = HePublicKey {
            data: vec![0u8; pk_size],
            params: self.params.clone(),
        };

        let secret_key = HeSecretKey {
            data: vec![0u8; 32], // Much smaller
            params: self.params.clone(),
        };

        Ok((public_key, secret_key))
    }

    /// Generate relinearization keys
    pub fn generate_relin_keys(&self, _secret_key: &HeSecretKey) -> Result<HeRelinKeys> {
        // Stub - ~50MB per spec §7.2
        Ok(HeRelinKeys {
            data: vec![0u8; 1024], // Placeholder
        })
    }

    /// Generate Galois keys for SIMD rotations
    pub fn generate_galois_keys(&self, _secret_key: &HeSecretKey) -> Result<HeGaloisKeys> {
        // Stub - ~200MB per spec §7.2
        Ok(HeGaloisKeys {
            data: vec![0u8; 1024], // Placeholder
        })
    }

    /// Encrypt f64 values using CKKS
    pub fn encrypt_f64(&self, values: &[f64], public_key: &HePublicKey) -> Result<Ciphertext> {
        self.validate_key_params(public_key)?;

        // Stub: encode values as bytes
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        Ok(Ciphertext::new(data, HeScheme::Ckks))
    }

    /// Encrypt u64 values using BFV
    pub fn encrypt_u64(&self, values: &[u64], public_key: &HePublicKey) -> Result<Ciphertext> {
        self.validate_key_params(public_key)?;

        // Stub: encode values as bytes
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        Ok(Ciphertext::new(data, HeScheme::Bfv))
    }

    /// Decrypt to f64 values (CKKS)
    pub fn decrypt_f64(
        &self,
        ciphertext: &Ciphertext,
        secret_key: &HeSecretKey,
    ) -> Result<Vec<f64>> {
        self.validate_key_params_secret(secret_key)?;

        if ciphertext.scheme != HeScheme::Ckks {
            return Err(AprenderError::FormatError {
                message: format!(
                    "Cannot decrypt {:?} ciphertext as f64 (requires CKKS)",
                    ciphertext.scheme
                ),
            });
        }

        // Stub: decode bytes as f64
        let values: Vec<f64> = ciphertext
            .data
            .chunks_exact(8)
            .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap_or([0; 8])))
            .collect();

        Ok(values)
    }

    /// Decrypt to u64 values (BFV)
    pub fn decrypt_u64(
        &self,
        ciphertext: &Ciphertext,
        secret_key: &HeSecretKey,
    ) -> Result<Vec<u64>> {
        self.validate_key_params_secret(secret_key)?;

        if ciphertext.scheme != HeScheme::Bfv {
            return Err(AprenderError::FormatError {
                message: format!(
                    "Cannot decrypt {:?} ciphertext as u64 (requires BFV)",
                    ciphertext.scheme
                ),
            });
        }

        // Stub: decode bytes as u64
        let values: Vec<u64> = ciphertext
            .data
            .chunks_exact(8)
            .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap_or([0; 8])))
            .collect();

        Ok(values)
    }

    /// Add two ciphertexts
    pub fn add(&self, a: &Ciphertext, b: &Ciphertext) -> Result<Ciphertext> {
        if a.scheme != b.scheme {
            return Err(AprenderError::FormatError {
                message: format!(
                    "Cannot add ciphertexts with different schemes: {:?} vs {:?}",
                    a.scheme, b.scheme
                ),
            });
        }

        // Stub: just concatenate for now
        let mut data = a.data.clone();
        data.extend_from_slice(&b.data);

        Ok(Ciphertext {
            data,
            scheme: a.scheme,
            level: a.level.max(b.level),
        })
    }

    /// Multiply two ciphertexts (increases level)
    pub fn multiply(
        &self,
        a: &Ciphertext,
        b: &Ciphertext,
        _relin_keys: &HeRelinKeys,
    ) -> Result<Ciphertext> {
        if a.scheme != b.scheme {
            return Err(AprenderError::FormatError {
                message: format!(
                    "Cannot multiply ciphertexts with different schemes: {:?} vs {:?}",
                    a.scheme, b.scheme
                ),
            });
        }

        let new_level = a.level.saturating_add(1).max(b.level.saturating_add(1));
        let max_level = u8::try_from(self.params.coeff_modulus_bits.len()).unwrap_or(4);

        if new_level >= max_level {
            return Err(AprenderError::FormatError {
                message: format!(
                    "Multiplication depth exceeded: level {new_level} >= max {max_level}"
                ),
            });
        }

        // Stub
        Ok(Ciphertext {
            data: a.data.clone(),
            scheme: a.scheme,
            level: new_level,
        })
    }

    fn validate_key_params(&self, public_key: &HePublicKey) -> Result<()> {
        if public_key.params.security_level != self.params.security_level {
            return Err(AprenderError::FormatError {
                message: "Public key security level doesn't match context".to_string(),
            });
        }
        Ok(())
    }

    fn validate_key_params_secret(&self, secret_key: &HeSecretKey) -> Result<()> {
        if secret_key.params.security_level != self.params.security_level {
            return Err(AprenderError::FormatError {
                message: "Secret key security level doesn't match context".to_string(),
            });
        }
        Ok(())
    }
}
