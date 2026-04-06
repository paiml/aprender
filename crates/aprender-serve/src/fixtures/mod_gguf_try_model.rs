impl ModelFixture {
    // =========================================================================
    // GGUF Fixtures
    // =========================================================================

    /// Create a GGUF model fixture
    ///
    /// # Arguments
    /// * `name` - Model name (used for filename)
    /// * `config` - Model configuration
    ///
    /// # Panics
    /// Panics if fixture creation fails (test should fail fast)
    #[must_use]
    pub fn gguf(name: &str, config: ModelConfig) -> Self {
        Self::try_gguf(name, config).expect("Failed to create GGUF fixture")
    }

    /// Try to create a GGUF model fixture
    pub fn try_gguf(name: &str, config: ModelConfig) -> Result<Self> {
        let temp_dir = TempDir::new().map_err(|e| RealizarError::IoError {
            message: format!("create_temp_dir: {}", e),
        })?;

        let filename = format!("{}.gguf", name);
        let path = temp_dir.path().join(&filename);

        // Generate GGUF data
        let data = Self::generate_gguf_data(&config);

        // Write to file
        fs::write(&path, &data).map_err(|e| RealizarError::IoError {
            message: format!("write_gguf to {}: {}", path.display(), e),
        })?;

        Ok(Self {
            path,
            _temp_dir: temp_dir,
            format: ModelFormat::Gguf,
            config,
        })
    }

    /// Create a GGUF fixture with invalid magic (for error testing)
    #[must_use]
    pub fn gguf_invalid_magic(name: &str) -> Self {
        Self::try_gguf_invalid_magic(name).expect("Failed to create invalid GGUF fixture")
    }

    /// Try to create a GGUF fixture with invalid magic
    pub fn try_gguf_invalid_magic(name: &str) -> Result<Self> {
        let temp_dir = TempDir::new().map_err(|e| RealizarError::IoError {
            message: format!("create_temp_dir: {}", e),
        })?;

        let filename = format!("{}.gguf", name);
        let path = temp_dir.path().join(&filename);

        // Invalid GGUF: wrong magic
        let mut data = Vec::new();
        data.extend_from_slice(&0xDEADBEEFu32.to_le_bytes()); // Invalid magic
        data.extend_from_slice(&3u32.to_le_bytes()); // Version 3
        data.extend_from_slice(&0u64.to_le_bytes()); // 0 tensors
        data.extend_from_slice(&0u64.to_le_bytes()); // 0 metadata
        data.resize(32, 0); // Align

        fs::write(&path, &data).map_err(|e| RealizarError::IoError {
            message: format!("write_gguf to {}: {}", path.display(), e),
        })?;

        Ok(Self {
            path,
            _temp_dir: temp_dir,
            format: ModelFormat::Gguf,
            config: ModelConfig::tiny(),
        })
    }

    /// Create a GGUF fixture with unsupported version
    #[must_use]
    pub fn gguf_invalid_version(name: &str) -> Self {
        Self::try_gguf_invalid_version(name).expect("Failed to create invalid version fixture")
    }

    /// Try to create GGUF with invalid version
    pub fn try_gguf_invalid_version(name: &str) -> Result<Self> {
        let temp_dir = TempDir::new().map_err(|e| RealizarError::IoError {
            message: format!("create_temp_dir: {}", e),
        })?;

        let filename = format!("{}.gguf", name);
        let path = temp_dir.path().join(&filename);

        // Invalid GGUF: unsupported version
        let mut data = Vec::new();
        data.extend_from_slice(&0x46554747u32.to_le_bytes()); // Valid magic "GGUF"
        data.extend_from_slice(&999u32.to_le_bytes()); // Invalid version
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        data.resize(32, 0);

        fs::write(&path, &data).map_err(|e| RealizarError::IoError {
            message: format!("write_gguf to {}: {}", path.display(), e),
        })?;

        Ok(Self {
            path,
            _temp_dir: temp_dir,
            format: ModelFormat::Gguf,
            config: ModelConfig::tiny(),
        })
    }

    // =========================================================================
    // SafeTensors Fixtures
    // =========================================================================

    /// Create a SafeTensors model fixture
    #[must_use]
    pub fn safetensors(name: &str, config: ModelConfig) -> Self {
        Self::try_safetensors(name, config).expect("Failed to create SafeTensors fixture")
    }

    /// Try to create a SafeTensors fixture
    pub fn try_safetensors(name: &str, config: ModelConfig) -> Result<Self> {
        let temp_dir = TempDir::new().map_err(|e| RealizarError::IoError {
            message: format!("create_temp_dir: {}", e),
        })?;

        let filename = format!("{}.safetensors", name);
        let path = temp_dir.path().join(&filename);

        // Generate SafeTensors data
        let data = Self::generate_safetensors_data(&config);

        fs::write(&path, &data).map_err(|e| RealizarError::IoError {
            message: format!("write_safetensors to {}: {}", path.display(), e),
        })?;

        // Generate config.json (required for SafeTensors inference)
        let config_json = format!(
            r#"{{"hidden_size":{},"num_hidden_layers":{},"num_attention_heads":{},"num_key_value_heads":{},"vocab_size":{},"intermediate_size":{},"max_position_embeddings":{},"rope_theta":{},"rms_norm_eps":{},"model_type":"{}"}}"#,
            config.hidden_dim,
            config.num_layers,
            config.num_heads,
            config.num_kv_heads,
            config.vocab_size,
            config.intermediate_dim,
            config.context_length,
            config.rope_theta,
            config.eps,
            config.architecture
        );
        let config_path = temp_dir.path().join("config.json");
        fs::write(&config_path, config_json).map_err(|e| RealizarError::IoError {
            message: format!("write_config to {}: {}", config_path.display(), e),
        })?;

        Ok(Self {
            path,
            _temp_dir: temp_dir,
            format: ModelFormat::SafeTensors,
            config,
        })
    }

    // =========================================================================
    // APR Fixtures
    // =========================================================================

    /// Create an APR model fixture
    #[must_use]
    pub fn apr(name: &str, config: ModelConfig) -> Self {
        Self::try_apr(name, config).expect("Failed to create APR fixture")
    }

    /// Try to create an APR fixture
    pub fn try_apr(name: &str, config: ModelConfig) -> Result<Self> {
        let temp_dir = TempDir::new().map_err(|e| RealizarError::IoError {
            message: format!("create_temp_dir: {}", e),
        })?;

        let filename = format!("{}.apr", name);
        let path = temp_dir.path().join(&filename);

        // Generate APR v2 data
        let data = Self::generate_apr_data(&config);

        fs::write(&path, &data).map_err(|e| RealizarError::IoError {
            message: format!("write_apr to {}: {}", path.display(), e),
        })?;

        Ok(Self {
            path,
            _temp_dir: temp_dir,
            format: ModelFormat::Apr,
            config,
        })
    }

    // =========================================================================
    // Accessors
    // =========================================================================

    /// Get path to the model file
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Get model format
    #[must_use]
    pub fn format(&self) -> ModelFormat {
        self.format
    }

    /// Get model configuration
    #[must_use]
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Read model file bytes
    pub fn read_bytes(&self) -> Result<Vec<u8>> {
        fs::read(&self.path).map_err(|e| RealizarError::IoError {
            message: format!("read_fixture {}: {}", self.path.display(), e),
        })
    }

    // =========================================================================
    // Data Generation (Internal)
    // =========================================================================

    /// Generate valid GGUF data for config
    fn generate_gguf_data(config: &ModelConfig) -> Vec<u8> {
        use crate::gguf::test_factory::GGUFBuilder;

        let arch = &config.architecture;

        // Build minimal GGUF with required tensors
        let builder = GGUFBuilder::new()
            .architecture(arch)
            .hidden_dim(arch, config.hidden_dim as u32)
            .num_layers(arch, config.num_layers as u32)
            .num_heads(arch, config.num_heads as u32)
            .num_kv_heads(arch, config.num_kv_heads as u32)
            .context_length(arch, config.context_length as u32)
            .rope_freq_base(arch, config.rope_theta)
            .rms_epsilon(arch, config.eps)
            .ffn_hidden_dim(arch, config.intermediate_dim as u32);

        // Add token embedding (F32)
        let embed_data: Vec<f32> = (0..config.vocab_size * config.hidden_dim)
            .map(|i| (i as f32) * 0.001)
            .collect();
        let builder = builder.add_f32_tensor(
            "token_embd.weight",
            &[config.hidden_dim as u64, config.vocab_size as u64],
            &embed_data,
        );

        // Add output norm
        let norm_data: Vec<f32> = vec![1.0; config.hidden_dim];
        let builder = builder.add_f32_tensor(
            "output_norm.weight",
            &[config.hidden_dim as u64],
            &norm_data,
        );

        // Add layer weights (Q4_K for compression)
        let head_dim = config.hidden_dim / config.num_heads;
        let kv_dim = config.num_kv_heads * head_dim;
        let qkv_dim = config.hidden_dim + 2 * kv_dim;

        let mut final_builder = builder;
        for layer in 0..config.num_layers {
            // Attention norm
            let norm_data: Vec<f32> = vec![1.0; config.hidden_dim];
            final_builder = final_builder.add_f32_tensor(
                &format!("blk.{}.attn_norm.weight", layer),
                &[config.hidden_dim as u64],
                &norm_data,
            );

            // QKV (Q4_K)
            let qkv_data = Self::create_q4k_data(config.hidden_dim, qkv_dim);
            final_builder = final_builder.add_q4_k_tensor(
                &format!("blk.{}.attn_qkv.weight", layer),
                &[qkv_dim as u64, config.hidden_dim as u64],
                &qkv_data,
            );

            // Attention output
            let attn_out_data = Self::create_q4k_data(config.hidden_dim, config.hidden_dim);
            final_builder = final_builder.add_q4_k_tensor(
                &format!("blk.{}.attn_output.weight", layer),
                &[config.hidden_dim as u64, config.hidden_dim as u64],
                &attn_out_data,
            );

            // FFN up
            let ffn_up_data = Self::create_q4k_data(config.hidden_dim, config.intermediate_dim);
            final_builder = final_builder.add_q4_k_tensor(
                &format!("blk.{}.ffn_up.weight", layer),
                &[config.intermediate_dim as u64, config.hidden_dim as u64],
                &ffn_up_data,
            );

            // FFN down
            let ffn_down_data = Self::create_q4k_data(config.intermediate_dim, config.hidden_dim);
            final_builder = final_builder.add_q4_k_tensor(
                &format!("blk.{}.ffn_down.weight", layer),
                &[config.hidden_dim as u64, config.intermediate_dim as u64],
                &ffn_down_data,
            );

            // FFN gate (for SwiGLU)
            if config.architecture == "llama" || config.architecture == "qwen2" {
                let ffn_gate_data =
                    Self::create_q4k_data(config.hidden_dim, config.intermediate_dim);
                final_builder = final_builder.add_q4_k_tensor(
                    &format!("blk.{}.ffn_gate.weight", layer),
                    &[config.intermediate_dim as u64, config.hidden_dim as u64],
                    &ffn_gate_data,
                );
            }

            // FFN norm
            let ffn_norm_data: Vec<f32> = vec![1.0; config.hidden_dim];
            final_builder = final_builder.add_f32_tensor(
                &format!("blk.{}.ffn_norm.weight", layer),
                &[config.hidden_dim as u64],
                &ffn_norm_data,
            );
        }

        // Output/LM head (tied with embeddings in many models)
        let lm_head_data = Self::create_q4k_data(config.hidden_dim, config.vocab_size);
        final_builder = final_builder.add_q4_k_tensor(
            "output.weight",
            &[config.vocab_size as u64, config.hidden_dim as u64],
            &lm_head_data,
        );

        final_builder.build()
    }

    /// Generate SafeTensors data for config
    fn generate_safetensors_data(config: &ModelConfig) -> Vec<u8> {
        use crate::gguf::format_factory::SafetensorsBuilder;

        let mut builder = SafetensorsBuilder::new();

        // Token embedding
        let embed_data: Vec<f32> = (0..config.vocab_size * config.hidden_dim)
            .map(|i| (i as f32) * 0.001)
            .collect();
        builder = builder.add_f32_tensor(
            "model.embed_tokens.weight",
            &[config.vocab_size, config.hidden_dim],
            &embed_data,
        );

        // Output norm
        let norm_data: Vec<f32> = vec![1.0; config.hidden_dim];
        builder = builder.add_f32_tensor("model.norm.weight", &[config.hidden_dim], &norm_data);

        // Layer weights
        for layer in 0..config.num_layers {
            // Attention norm
            builder = builder.add_f32_tensor(
                &format!("model.layers.{}.input_layernorm.weight", layer),
                &[config.hidden_dim],
                &norm_data,
            );

            // Q projection
            let q_data: Vec<f32> = vec![0.01; config.hidden_dim * config.hidden_dim];
            builder = builder.add_f32_tensor(
                &format!("model.layers.{}.self_attn.q_proj.weight", layer),
                &[config.hidden_dim, config.hidden_dim],
                &q_data,
            );

            // K projection
            let kv_dim = config.num_kv_heads * (config.hidden_dim / config.num_heads);
            let k_data: Vec<f32> = vec![0.01; kv_dim * config.hidden_dim];
            builder = builder.add_f32_tensor(
                &format!("model.layers.{}.self_attn.k_proj.weight", layer),
                &[kv_dim, config.hidden_dim],
                &k_data,
            );

            // V projection
            builder = builder.add_f32_tensor(
                &format!("model.layers.{}.self_attn.v_proj.weight", layer),
                &[kv_dim, config.hidden_dim],
                &k_data,
            );

            // O projection
            builder = builder.add_f32_tensor(
                &format!("model.layers.{}.self_attn.o_proj.weight", layer),
                &[config.hidden_dim, config.hidden_dim],
                &q_data,
            );

            // FFN
            let ffn_data: Vec<f32> = vec![0.01; config.hidden_dim * config.intermediate_dim];
            builder = builder.add_f32_tensor(
                &format!("model.layers.{}.mlp.up_proj.weight", layer),
                &[config.intermediate_dim, config.hidden_dim],
                &ffn_data,
            );
            builder = builder.add_f32_tensor(
                &format!("model.layers.{}.mlp.gate_proj.weight", layer),
                &[config.intermediate_dim, config.hidden_dim],
                &ffn_data,
            );

            let down_data: Vec<f32> = vec![0.01; config.intermediate_dim * config.hidden_dim];
            builder = builder.add_f32_tensor(
                &format!("model.layers.{}.mlp.down_proj.weight", layer),
                &[config.hidden_dim, config.intermediate_dim],
                &down_data,
            );

            // Post attention norm
            builder = builder.add_f32_tensor(
                &format!("model.layers.{}.post_attention_layernorm.weight", layer),
                &[config.hidden_dim],
                &norm_data,
            );
        }

        // LM head
        let lm_head_data: Vec<f32> = vec![0.01; config.vocab_size * config.hidden_dim];
        builder = builder.add_f32_tensor(
            "lm_head.weight",
            &[config.vocab_size, config.hidden_dim],
            &lm_head_data,
        );

        builder.build()
    }
}
