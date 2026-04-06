//! Progressive Knowledge Distillation
//!
//! Matches student hidden states to teacher hidden states at selected layers.
//! Based on Sun et al. (2019).

use ndarray::Array2;

/// Progressive Knowledge Distillation
///
/// Matches student hidden states to teacher hidden states at selected layers.
/// Based on Sun et al. (2019).
#[derive(Debug, Clone)]
pub struct ProgressiveDistillation {
    /// Layer mapping: (student_layer, teacher_layer)
    pub layer_mapping: Vec<(usize, usize)>,
    /// Loss weight for hidden state matching
    pub hidden_weight: f32,
    /// Projection matrix for dimension alignment (student_dim x teacher_dim)
    /// Used when student hidden size differs from teacher hidden size.
    pub projection: Option<Array2<f32>>,
}

impl Default for ProgressiveDistillation {
    fn default() -> Self {
        Self {
            layer_mapping: vec![(0, 2), (1, 5), (2, 8), (3, 11)],
            hidden_weight: 1.0,
            projection: None,
        }
    }
}

impl ProgressiveDistillation {
    /// Create new progressive distillation config
    #[must_use]
    pub fn new(layer_mapping: Vec<(usize, usize)>) -> Self {
        Self { layer_mapping, hidden_weight: 1.0, projection: None }
    }

    /// Set projection layer for dimension alignment
    ///
    /// Creates a linear projection matrix to align student hidden states
    /// to teacher hidden size. Initialized with Xavier uniform.
    ///
    /// # Arguments
    ///
    /// * `student_dim` - Student model hidden dimension
    /// * `teacher_dim` - Teacher model hidden dimension
    #[must_use]
    pub fn with_projection(mut self, student_dim: usize, teacher_dim: usize) -> Self {
        use rand::Rng;

        // Xavier uniform initialization
        let scale = (6.0 / (student_dim + teacher_dim) as f32).sqrt();
        let mut rng = rand::rng();

        let projection =
            Array2::from_shape_fn((student_dim, teacher_dim), |_| rng.random_range(-scale..scale));

        self.projection = Some(projection);
        self
    }

    /// Set hidden state loss weight
    #[must_use]
    pub fn with_weight(mut self, weight: f32) -> Self {
        self.hidden_weight = weight;
        self
    }

    /// Compute hidden state matching loss
    ///
    /// Uses MSE loss between projected student and teacher hidden states.
    /// If projection layer is set and shapes differ, projects student to teacher dimension.
    pub fn hidden_state_loss(
        &self,
        student_hidden: &[Array2<f32>],
        teacher_hidden: &[Array2<f32>],
    ) -> f32 {
        let mut total_loss = 0.0;
        let mut count = 0;

        for (s_idx, t_idx) in &self.layer_mapping {
            if *s_idx < student_hidden.len() && *t_idx < teacher_hidden.len() {
                let s_h = &student_hidden[*s_idx];
                let t_h = &teacher_hidden[*t_idx];

                // MSE loss - project student if dimensions differ
                if s_h.dim() == t_h.dim() {
                    // Same dimensions: direct MSE
                    let diff = s_h - t_h;
                    let mse = diff.mapv(|x| x * x).mean().unwrap_or(0.0);
                    total_loss += mse;
                    count += 1;
                } else if let Some(ref proj) = self.projection {
                    // Different dimensions: project student to teacher space
                    // s_h: (batch, student_dim), proj: (student_dim, teacher_dim)
                    // result: (batch, teacher_dim)
                    let s_dim = s_h.shape()[1];
                    let t_dim = t_h.shape()[1];

                    // Verify projection dimensions match
                    if proj.shape() == [s_dim, t_dim] {
                        let projected = s_h.dot(proj);
                        let diff = &projected - t_h;
                        let mse = diff.mapv(|x| x * x).mean().unwrap_or(0.0);
                        total_loss += mse;
                        count += 1;
                    }
                }
                // Skip if shapes differ and no projection is set
            }
        }

        if count > 0 {
            self.hidden_weight * total_loss / count as f32
        } else {
            0.0
        }
    }
}
