
impl ModelFixture {

    /// Generate APR v2 data for config (PMAT-111: Fixed tensor index serialization)
    ///
    /// APR v2 binary format:
    /// - Header (64 bytes)
    /// - Metadata (JSON, padded to 64 bytes)
    /// - Tensor Index (binary entries)
    /// - Tensor Data (F32 arrays, 64-byte aligned)
    fn generate_apr_data(config: &ModelConfig) -> Vec<u8> {
        // Step 1: Build tensor definitions (name, shape, dtype)
        let mut tensor_defs: Vec<(&str, Vec<usize>, u8)> = Vec::new();
        let head_dim = config.hidden_dim / config.num_heads;
        let kv_dim = config.num_kv_heads * head_dim;

        // Global tensors
        tensor_defs.push((
            "model.embed_tokens.weight",
            vec![config.vocab_size, config.hidden_dim],
            0, // F32
        ));
        tensor_defs.push(("model.norm.weight", vec![config.hidden_dim], 0));
        tensor_defs.push((
            "lm_head.weight",
            vec![config.vocab_size, config.hidden_dim],
            0,
        ));

        // Per-layer tensors
        for i in 0..config.num_layers {
            let prefix = format!("model.layers.{i}");

            // Attention norm
            tensor_defs.push((
                Box::leak(format!("{prefix}.input_layernorm.weight").into_boxed_str()),
                vec![config.hidden_dim],
                0,
            ));

            // Q, K, V, O projections
            tensor_defs.push((
                Box::leak(format!("{prefix}.self_attn.q_proj.weight").into_boxed_str()),
                vec![config.hidden_dim, config.hidden_dim],
                0,
            ));
            tensor_defs.push((
                Box::leak(format!("{prefix}.self_attn.k_proj.weight").into_boxed_str()),
                vec![kv_dim, config.hidden_dim],
                0,
            ));
            tensor_defs.push((
                Box::leak(format!("{prefix}.self_attn.v_proj.weight").into_boxed_str()),
                vec![kv_dim, config.hidden_dim],
                0,
            ));
            tensor_defs.push((
                Box::leak(format!("{prefix}.self_attn.o_proj.weight").into_boxed_str()),
                vec![config.hidden_dim, config.hidden_dim],
                0,
            ));

            // FFN norm
            tensor_defs.push((
                Box::leak(format!("{prefix}.post_attention_layernorm.weight").into_boxed_str()),
                vec![config.hidden_dim],
                0,
            ));

            // FFN projections (gate, up, down)
            tensor_defs.push((
                Box::leak(format!("{prefix}.mlp.gate_proj.weight").into_boxed_str()),
                vec![config.intermediate_dim, config.hidden_dim],
                0,
            ));
            tensor_defs.push((
                Box::leak(format!("{prefix}.mlp.up_proj.weight").into_boxed_str()),
                vec![config.intermediate_dim, config.hidden_dim],
                0,
            ));
            tensor_defs.push((
                Box::leak(format!("{prefix}.mlp.down_proj.weight").into_boxed_str()),
                vec![config.hidden_dim, config.intermediate_dim],
                0,
            ));
        }

        let tensor_count = tensor_defs.len();

        // Step 2: Calculate tensor data sizes and offsets
        let mut tensor_data_offsets: Vec<u64> = Vec::new();
        let mut tensor_sizes: Vec<u64> = Vec::new();
        let mut current_offset: u64 = 0;

        for (_, shape, dtype) in &tensor_defs {
            let element_count: usize = shape.iter().product();
            let bytes_per_element = match dtype {
                0 => 4, // F32
                1 => 2, // F16
                _ => 4, // Default F32
            };
            let size = (element_count * bytes_per_element) as u64;

            tensor_data_offsets.push(current_offset);
            tensor_sizes.push(size);
            current_offset += size;
            // Align to 64 bytes
            current_offset = current_offset.div_ceil(64) * 64;
        }

        let total_tensor_data_size = current_offset;

        // Step 3: Build the APR file
        let mut data = Vec::new();

        // Header (64 bytes)
        // Magic: "APR\0"
        data.extend_from_slice(b"APR\x00");
        // Version: 2.0
        data.push(2); // major
        data.push(0); // minor
                      // Flags: 0
        data.extend_from_slice(&0u16.to_le_bytes());
        // Tensor count
        data.extend_from_slice(&(tensor_count as u32).to_le_bytes());
        // Metadata offset (after header = 64)
        data.extend_from_slice(&64u64.to_le_bytes());

        // Placeholder for metadata size (position 20)
        let metadata_size_offset = data.len();
        data.extend_from_slice(&0u32.to_le_bytes());

        // Placeholder for tensor index offset (position 24)
        let tensor_index_offset_pos = data.len();
        data.extend_from_slice(&0u64.to_le_bytes());

        // Placeholder for data offset (position 32)
        let data_offset_pos = data.len();
        data.extend_from_slice(&0u64.to_le_bytes());

        // Checksum (placeholder)
        data.extend_from_slice(&0u32.to_le_bytes());

        // Reserved (pad to 64 bytes)
        data.resize(64, 0);

        // Metadata (JSON)
        let metadata = format!(
            r#"{{"architecture":"{}","hidden_size":{},"num_layers":{},"num_heads":{},"num_kv_heads":{},"vocab_size":{},"intermediate_size":{},"rope_theta":{},"rms_norm_eps":{}}}"#,
            config.architecture,
            config.hidden_dim,
            config.num_layers,
            config.num_heads,
            config.num_kv_heads,
            config.vocab_size,
            config.intermediate_dim,
            config.rope_theta,
            config.eps
        );
        let metadata_bytes = metadata.as_bytes();

        // Update metadata size
        let metadata_size = metadata_bytes.len() as u32;
        data[metadata_size_offset..metadata_size_offset + 4]
            .copy_from_slice(&metadata_size.to_le_bytes());

        data.extend_from_slice(metadata_bytes);

        // Pad to 64-byte boundary
        let padded_len = data.len().div_ceil(64) * 64;
        data.resize(padded_len, 0);

        // Update tensor index offset
        let tensor_index_offset = data.len() as u64;
        data[tensor_index_offset_pos..tensor_index_offset_pos + 8]
            .copy_from_slice(&tensor_index_offset.to_le_bytes());

        // Step 4: Write tensor index entries in binary format
        // Format per TensorEntry::from_binary:
        //   name_len (u16 LE) + name bytes
        //   dtype (u8)
        //   ndim (u8) + dims (u64 LE each)
        //   offset (u64 LE)
        //   size (u64 LE)
        for (i, (name, shape, dtype)) in tensor_defs.iter().enumerate() {
            // Name
            let name_bytes = name.as_bytes();
            data.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
            data.extend_from_slice(name_bytes);

            // Dtype
            data.push(*dtype);

            // Shape: ndim + dims
            data.push(shape.len() as u8);
            for &dim in shape {
                data.extend_from_slice(&(dim as u64).to_le_bytes());
            }

            // Offset and size
            data.extend_from_slice(&tensor_data_offsets[i].to_le_bytes());
            data.extend_from_slice(&tensor_sizes[i].to_le_bytes());
        }

        // Pad to 64-byte boundary
        let padded_len = data.len().div_ceil(64) * 64;
        data.resize(padded_len, 0);

        // Update data offset
        let data_offset = data.len() as u64;
        data[data_offset_pos..data_offset_pos + 8].copy_from_slice(&data_offset.to_le_bytes());

        // Step 5: Write tensor data (zeros for synthetic fixture)
        // Using zeros will produce garbage output, but the test will RUN (PMAT-111)
        data.resize(data.len() + total_tensor_data_size as usize, 0);

        data
    }

    /// Create Q4_K data for given dimensions
    fn create_q4k_data(in_dim: usize, out_dim: usize) -> Vec<u8> {
        let super_blocks_per_row = in_dim.div_ceil(256);
        let bytes_per_row = super_blocks_per_row * 144;
        let data_size = out_dim * bytes_per_row;
        let mut data = vec![0u8; data_size];

        for row in 0..out_dim {
            for sb in 0..super_blocks_per_row {
                let offset = row * bytes_per_row + sb * 144;
                if offset + 4 <= data.len() {
                    // d=1.0 in f16 format
                    data[offset..offset + 2].copy_from_slice(&0x3C00_u16.to_le_bytes());
                    // dmin=0
                    data[offset + 2..offset + 4].copy_from_slice(&0x0000_u16.to_le_bytes());
                }
            }
        }

        data
    }
}

include!("mod_gguf_try_model.rs");
