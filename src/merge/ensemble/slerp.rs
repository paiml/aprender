//! ENT-032: Iterative SLERP merging

use super::{slerp_merge as slerp_merge_impl, MergeError, Model, SlerpConfig};

/// Iterative SLERP: merge models pairwise until one remains
///
/// For N models: ((m1 + m2) + m3) + ... + mN
/// where + is SLERP at parameter t
pub fn iterative_slerp_merge(models: &[Model], t: f32) -> Result<Model, MergeError> {
    let config = SlerpConfig::new(t)?;

    let mut current = models[0].clone();
    for model in models.iter().skip(1) {
        current = slerp_merge_impl(&current, model, &config)?;
    }

    Ok(current)
}
