
impl TransformerLayer {
    fn new(embed_dim: usize, num_heads: usize, hidden_dim: usize, dropout: f32) -> Self {
        let head_dim = embed_dim / num_heads;

        Self {
            qkv_proj: Linear::new(embed_dim, 3 * embed_dim),
            out_proj: Linear::new(embed_dim, embed_dim),
            ffn1: Linear::new(embed_dim, hidden_dim),
            ffn2: Linear::new(hidden_dim, embed_dim),
            norm1: LayerNorm::new(embed_dim),
            norm2: LayerNorm::new(embed_dim),
            num_heads,
            head_dim,
            dropout_p: dropout,
        }
    }

    fn forward(&self, x: &Tensor, training: bool) -> Tensor {
        // Pre-norm architecture
        let normed = self.norm1.forward(x);

        // Self-attention
        let attn_out = self.self_attention(&normed, training);

        // Residual connection
        let x = x.add(&attn_out);

        // FFN block
        let normed = self.norm2.forward(&x);
        let ffn_out = self.ffn(&normed, training);

        // Residual connection
        x.add(&ffn_out)
    }

    fn self_attention(&self, x: &Tensor, training: bool) -> Tensor {
        let batch_size = x.shape()[0];
        let seq_len = x.shape()[1];

        // Project to Q, K, V
        let qkv = self.qkv_proj.forward(x);

        // Split into Q, K, V and reshape for multi-head attention
        let (q, k, v) = split_qkv(&qkv, self.num_heads, self.head_dim);

        // Scaled dot-product attention
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let k_t = transpose_last_two(&k);
        let scores = batched_matmul(&q, &k_t);
        let scores = scale_tensor(&scores, scale);
        let attn_weights = softmax(&scores, -1);

        // Apply dropout
        let attn_weights = if training && self.dropout_p > 0.0 {
            dropout(&attn_weights, self.dropout_p)
        } else {
            attn_weights
        };

        // Weighted sum
        let attn_out = batched_matmul(&attn_weights, &v);

        // Reshape back
        let attn_out = concat_heads(&attn_out, batch_size, seq_len);

        // Output projection
        self.out_proj.forward(&attn_out)
    }

    fn ffn(&self, x: &Tensor, training: bool) -> Tensor {
        let mut h = self.ffn1.forward(x);
        h = gelu(&h);
        if training && self.dropout_p > 0.0 {
            h = dropout(&h, self.dropout_p);
        }
        self.ffn2.forward(&h)
    }
}

/// Layer normalization.
#[derive(Debug)]
struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    #[allow(dead_code)]
    normalized_shape: usize,
    eps: f32,
}

impl LayerNorm {
    fn new(normalized_shape: usize) -> Self {
        let weight = Tensor::ones(&[normalized_shape]).requires_grad();
        let bias = Tensor::zeros(&[normalized_shape]).requires_grad();

        Self {
            weight,
            bias,
            normalized_shape,
            eps: 1e-5,
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        layer_norm(x, &self.weight, &self.bias, self.eps)
    }
}

/// Vocabulary for tokenization.
#[derive(Debug)]
pub struct Vocabulary {
    /// Token to ID mapping
    token_to_id: HashMap<String, usize>,
    /// Special token IDs
    #[allow(dead_code)]
    pad_id: usize,
    unk_id: usize,
    cls_id: usize,
    sep_id: usize,
    eos_id: usize,
    /// Language token IDs
    lang_tokens: HashMap<String, usize>,
}

impl Vocabulary {
    /// Create a vocabulary for Rust error messages.
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn for_rust_errors() -> Self {
        let mut token_to_id = HashMap::new();
        let mut id = 0;

        // Special tokens
        let pad_id = id;
        token_to_id.insert("[PAD]".to_string(), id);
        id += 1;
        let unk_id = id;
        token_to_id.insert("[UNK]".to_string(), id);
        id += 1;
        let cls_id = id;
        token_to_id.insert("[CLS]".to_string(), id);
        id += 1;
        let sep_id = id;
        token_to_id.insert("[SEP]".to_string(), id);
        id += 1;
        let eos_id = id;
        token_to_id.insert("[EOS]".to_string(), id);
        id += 1;

        // Language tokens
        let mut lang_tokens = HashMap::new();
        for lang in &["python", "rust", "julia", "typescript", "go", "java", "cpp"] {
            token_to_id.insert(format!("[LANG_{lang}]"), id);
            lang_tokens.insert((*lang).to_string(), id);
            id += 1;
        }

        // Common Rust error codes
        for code in &[
            "E0308", "E0382", "E0597", "E0599", "E0433", "E0432", "E0277", "E0425", "E0282",
            "E0412", "E0502", "E0499", "E0596", "E0507", "E0621", "E0106",
        ] {
            token_to_id.insert((*code).to_string(), id);
            id += 1;
        }

        // Common keywords and types
        for word in &[
            "error",
            "expected",
            "found",
            "type",
            "mismatched",
            "types",
            "cannot",
            "borrow",
            "move",
            "lifetime",
            "trait",
            "impl",
            "struct",
            "fn",
            "let",
            "mut",
            "ref",
            "self",
            "String",
            "str",
            "i32",
            "i64",
            "u32",
            "u64",
            "f32",
            "f64",
            "Vec",
            "Option",
            "Result",
            "Box",
            "Rc",
            "Arc",
            "Clone",
            "Copy",
            "Debug",
            "Display",
            "From",
            "Into",
            "as",
            "for",
            "in",
            "if",
            "else",
            "match",
            "return",
            "use",
            "mod",
            "pub",
            "crate",
            "super",
        ] {
            token_to_id.insert((*word).to_string(), id);
            id += 1;
        }

        // Common Python keywords (for transpilation errors)
        for word in &[
            "def",
            "class",
            "import",
            "from",
            "None",
            "True",
            "False",
            "self",
            "list",
            "dict",
            "tuple",
            "set",
            "int",
            "float",
            "bool",
            "numpy",
            "pandas",
            "DataFrame",
            "Series",
            "ndarray",
            "array",
            "shape",
        ] {
            token_to_id.insert((*word).to_string(), id);
            id += 1;
        }

        // Punctuation and operators
        for sym in &[
            "(", ")", "[", "]", "{", "}", "<", ">", ":", ";", ",", ".", "->", "=>", "::", "&", "*",
            "+", "-", "/", "=", "==", "!=", "<=", ">=", "&&", "||", "!", "?", "'", "\"", "`",
        ] {
            token_to_id.insert((*sym).to_string(), id);
            id += 1;
        }

        Self {
            token_to_id,
            pad_id,
            unk_id,
            cls_id,
            sep_id,
            eos_id,
            lang_tokens,
        }
    }

    /// Get the CLS token ID.
    #[must_use]
    pub fn cls_token(&self) -> usize {
        self.cls_id
    }

    /// Get the SEP token ID.
    #[must_use]
    pub fn sep_token(&self) -> usize {
        self.sep_id
    }

    /// Get the EOS token ID.
    #[must_use]
    pub fn eos_token(&self) -> usize {
        self.eos_id
    }

    /// Get the language token ID.
    #[must_use]
    pub fn lang_token(&self, lang: &str) -> usize {
        self.lang_tokens.get(lang).copied().unwrap_or(self.unk_id)
    }

    /// Tokenize a string into token IDs.
    #[must_use]
    pub fn tokenize(&self, text: &str) -> Vec<usize> {
        // Simple whitespace + punctuation tokenization
        let mut tokens = Vec::new();
        let mut current_word = String::new();

        for c in text.chars() {
            if c.is_whitespace() {
                if !current_word.is_empty() {
                    tokens.push(self.get_token_id(&current_word));
                    current_word.clear();
                }
            } else if c.is_ascii_punctuation() {
                if !current_word.is_empty() {
                    tokens.push(self.get_token_id(&current_word));
                    current_word.clear();
                }
                tokens.push(self.get_token_id(&c.to_string()));
            } else {
                current_word.push(c);
            }
        }

        if !current_word.is_empty() {
            tokens.push(self.get_token_id(&current_word));
        }

        tokens
    }

    fn get_token_id(&self, token: &str) -> usize {
        self.token_to_id.get(token).copied().unwrap_or(self.unk_id)
    }

    /// Get vocabulary size.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.token_to_id.len()
    }
}

// ==================== Tensor Operations ====================
// Manual implementations using available Tensor operations

fn embedding_lookup(weight: &Tensor, indices: &Tensor) -> Tensor {
    // Manual embedding lookup: gather rows from weight matrix
    let weight_data = weight.data();
    let indices_data = indices.data();
    let embed_dim = weight.shape()[1];
    let batch_size = indices.shape()[0];
    let seq_len = indices.shape()[1];

    let mut output = Vec::with_capacity(batch_size * seq_len * embed_dim);
    for &idx in indices_data {
        let row_start = (idx as usize) * embed_dim;
        let row_end = row_start + embed_dim;
        if row_end <= weight_data.len() {
            output.extend_from_slice(&weight_data[row_start..row_end]);
        } else {
            // Out of bounds - use zeros
            output.extend(std::iter::repeat(0.0).take(embed_dim));
        }
    }

    Tensor::new(&output, &[batch_size, seq_len, embed_dim])
}

fn mean_pool(x: &Tensor) -> Tensor {
    // Mean over sequence dimension (dim 1)
    let shape = x.shape();
    let batch_size = shape[0];
    let seq_len = shape[1];
    let hidden_dim = shape[2];

    let data = x.data();
    let mut output = Vec::with_capacity(batch_size * hidden_dim);

    for b in 0..batch_size {
        for h in 0..hidden_dim {
            let mut sum = 0.0f32;
            for s in 0..seq_len {
                sum += data[b * seq_len * hidden_dim + s * hidden_dim + h];
            }
            output.push(sum / seq_len as f32);
        }
    }

    Tensor::new(&output, &[batch_size, hidden_dim])
}

fn l2_normalize(x: &Tensor) -> Tensor {
    // L2 normalize along last dimension
    let shape = x.shape();
    let batch_size = shape[0];
    let dim = shape[1];

    let data = x.data();
    let mut output = Vec::with_capacity(batch_size * dim);

    for b in 0..batch_size {
        let start = b * dim;
        let end = start + dim;
        let slice = &data[start..end];

        // Compute L2 norm using trueno SIMD
        let v = Vector::from_slice(slice);
        let norm = v.norm_l2().unwrap_or(1.0);

        // Normalize
        let inv_norm = if norm > 1e-12 { 1.0 / norm } else { 1.0 };
        for &val in slice {
            output.push(val * inv_norm);
        }
    }

    Tensor::new(&output, &[batch_size, dim])
}

fn split_qkv(qkv: &Tensor, num_heads: usize, head_dim: usize) -> (Tensor, Tensor, Tensor) {
    // Split QKV projection into Q, K, V and reshape for multi-head attention
    let shape = qkv.shape();
    let batch_size = shape[0];
    let seq_len = shape[1];
    let embed_dim = num_heads * head_dim;

    let data = qkv.data();
    let mut q = Vec::with_capacity(batch_size * num_heads * seq_len * head_dim);
    let mut k = Vec::with_capacity(batch_size * num_heads * seq_len * head_dim);
    let mut v = Vec::with_capacity(batch_size * num_heads * seq_len * head_dim);

    for b in 0..batch_size {
        for s in 0..seq_len {
            let base = b * seq_len * (3 * embed_dim) + s * (3 * embed_dim);
            // Q, K, V are concatenated in the last dimension
            for h in 0..num_heads {
                for d in 0..head_dim {
                    q.push(data[base + h * head_dim + d]);
                    k.push(data[base + embed_dim + h * head_dim + d]);
                    v.push(data[base + 2 * embed_dim + h * head_dim + d]);
                }
            }
        }
    }

    let q_tensor = Tensor::new(&q, &[batch_size, num_heads, seq_len, head_dim]);
    let k_tensor = Tensor::new(&k, &[batch_size, num_heads, seq_len, head_dim]);
    let v_tensor = Tensor::new(&v, &[batch_size, num_heads, seq_len, head_dim]);

    (q_tensor, k_tensor, v_tensor)
}
