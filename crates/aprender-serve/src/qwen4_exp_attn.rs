//! `qwen4_exp` attention-side ops (#7175): RED stubs returning zeros.

/// Result of an op.
pub type OpResult<T> = Result<T, String>;

/// Indexer weights.
pub struct IndexerParams<'a> {
    /// `index_qk_proj.weight`.
    pub qk_proj: &'a [f32],
    /// `q_layernorm.weight`.
    pub q_norm: &'a [f32],
    /// `k_layernorm.weight`.
    pub k_norm: &'a [f32],
}

/// Indexer hyperparameters.
#[derive(Clone, Copy)]
pub struct IndexerShape {
    /// Hidden size.
    pub hidden: usize,
    /// Indexer heads.
    pub n_heads: usize,
    /// Indexer head dim.
    pub head_dim: usize,
    /// Token budget.
    pub budget: usize,
    /// Compress ratio.
    pub ratio: usize,
    /// RMSNorm eps.
    pub eps: f32,
}

/// Stub.
pub fn indexer_select(
    _p: &IndexerParams,
    _s: &IndexerShape,
    hidden: &[f32],
    _cos: &[f32],
    _sin: &[f32],
    visible: &[bool],
) -> OpResult<Vec<bool>> {
    let _ = hidden;
    Ok(vec![false; visible.len()])
}

/// Attention weights.
pub struct AttentionParams<'a> {
    /// `q_proj.weight`.
    pub q_proj: &'a [f32],
    /// `k_proj.weight`.
    pub k_proj: &'a [f32],
    /// `v_proj.weight`.
    pub v_proj: &'a [f32],
    /// `o_proj.weight`.
    pub o_proj: &'a [f32],
    /// `q_norm.weight`.
    pub q_norm: &'a [f32],
    /// `k_norm.weight`.
    pub k_norm: &'a [f32],
    /// Indexer.
    pub indexer: IndexerParams<'a>,
}

/// Attention hyperparameters.
#[derive(Clone, Copy)]
pub struct AttentionShape {
    /// Query heads.
    pub n_heads: usize,
    /// KV heads.
    pub n_kv: usize,
    /// Head dim.
    pub head_dim: usize,
    /// Indexer.
    pub indexer: IndexerShape,
}

/// Stub.
pub fn indexed_attention(
    _p: &AttentionParams,
    _a: &AttentionShape,
    hidden: &[f32],
    _cos: &[f32],
    _sin: &[f32],
    _visible: &[bool],
) -> OpResult<Vec<f32>> {
    Ok(vec![0.0; hidden.len()])
}

/// hc weights.
pub struct HcParams<'a> {
    /// `hc_norm.weight`.
    pub norm: &'a [f32],
    /// Down projection.
    pub down: &'a [f32],
    /// Up projection.
    pub up: &'a [f32],
    /// Injection projection.
    pub inject: Option<&'a [f32]>,
}

/// hc hyperparameters.
#[derive(Clone, Copy)]
pub struct HcShape {
    /// Hidden.
    pub hidden: usize,
    /// Streams.
    pub count: usize,
    /// Lowrank.
    pub lowrank: usize,
    /// Eps.
    pub eps: f32,
}

/// hc output.
pub struct HcOut {
    /// Mixed.
    pub mixed: Vec<f32>,
    /// Injection weights.
    pub inject: Vec<f32>,
}

/// Stub.
pub fn hc_gated_residual(_p: &HcParams, s: &HcShape, x: &[f32]) -> OpResult<HcOut> {
    let t = x.len() / (s.count * s.hidden);
    Ok(HcOut {
        mixed: vec![0.0; t * s.hidden],
        inject: vec![0.0; t * s.count],
    })
}

/// Stub.
pub fn hc_combine(_s: &HcShape, x: &[f32], _b: &[f32], _i: &[f32]) -> OpResult<Vec<f32>> {
    Ok(vec![0.0; x.len()])
}

/// Stub.
pub fn hc_final_mixer(p: &HcParams, s: &HcShape, x: &[f32]) -> OpResult<Vec<f32>> {
    hc_gated_residual(p, s, x).map(|o| o.mixed)
}

/// Stub.
pub fn hc_stream_init(s: &HcShape, e: &[f32]) -> Vec<f32> {
    vec![0.0; e.len() * s.count]
}
