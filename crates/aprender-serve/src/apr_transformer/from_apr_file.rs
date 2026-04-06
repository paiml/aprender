impl AprTransformer {
    /// Load APR transformer from an APR v2 file
    ///
    /// Parses the APR v2 format (magic "APR2") and extracts transformer weights.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to .apr file
    ///
    /// # Returns
    ///
    /// Loaded transformer ready for inference
    ///
    /// # Errors
    ///
    /// Returns error if file cannot be read or parsed
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let transformer = AprTransformer::from_apr_file("model.apr")?;
    /// let logits = transformer.forward(&[1, 2, 3])?;
    /// ```
    pub fn from_apr_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        use std::io::Read;

        let mut file = File::open(path.as_ref()).map_err(|e| RealizarError::IoError {
            message: format!("Failed to open APR file: {e}"),
        })?;

        let mut data = Vec::new();
        file.read_to_end(&mut data)
            .map_err(|e| RealizarError::IoError {
                message: format!("Failed to read APR file: {e}"),
            })?;

        Self::from_apr_bytes(&data)
    }

    /// Load APR transformer from file with validation (PMAT-235)
    ///
    /// Loads and then validates ALL tensors using `ValidatedAprTransformer::validate()`.
    /// Returns a wrapper that `Deref`s to `AprTransformer` for transparent access.
    ///
    /// # Errors
    ///
    /// Returns error if the file cannot be read, parsed, or if any tensor fails validation.
    pub fn from_apr_file_validated<P: AsRef<Path>>(
        path: P,
    ) -> Result<crate::safetensors::validation::ValidatedAprTransformer> {
        let transformer = Self::from_apr_file(path)?;
        crate::safetensors::validation::ValidatedAprTransformer::validate(transformer)
            .map_err(Into::into)
    }
}
