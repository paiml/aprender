#[allow(clippy::wildcard_imports)]
use super::*;

impl OnlineDistillation {
    /// Create online distillation with specified number of peer networks.
    ///
    /// # Arguments
    ///
    /// * `num_networks` - Number of networks to co-train (typically 2-4)
    /// * `temperature` - Temperature for softening predictions
    /// * `mutual_weight` - Weight for mutual learning loss (vs task loss)
    #[must_use]
    pub fn new(num_networks: usize, temperature: f32, mutual_weight: f32) -> Self {
        assert!(
            num_networks >= 2,
            "Need at least 2 networks for mutual learning"
        );
        assert!(temperature > 0.0, "Temperature must be positive");
        Self {
            num_networks,
            temperature,
            mutual_weight,
        }
    }

    /// Compute mutual learning loss for one network given all peer outputs.
    ///
    /// Each network learns from the average of its peers' predictions.
    #[must_use]
    pub fn mutual_loss(&self, network_idx: usize, all_logits: &[Vec<f32>]) -> f32 {
        assert_eq!(all_logits.len(), self.num_networks);

        let my_logits = &all_logits[network_idx];
        let my_soft = softmax_with_temp(my_logits, self.temperature);

        // Average KL divergence to all other networks
        let mut total_kl = 0.0;
        let mut peer_count = 0;

        for (i, peer_logits) in all_logits.iter().enumerate() {
            if i != network_idx {
                let peer_soft = softmax_with_temp(peer_logits, self.temperature);
                let eps = 1e-10;
                let kl: f32 = peer_soft
                    .iter()
                    .zip(my_soft.iter())
                    .map(|(&p, &s)| p * ((p + eps) / (s + eps)).ln())
                    .sum();
                total_kl += kl * self.temperature * self.temperature;
                peer_count += 1;
            }
        }

        if peer_count > 0 {
            total_kl / peer_count as f32
        } else {
            0.0
        }
    }

    /// Compute combined loss for one network: `task_loss` + `mutual_weight` * `mutual_loss`.
    #[must_use]
    pub fn combined_loss(
        &self,
        network_idx: usize,
        all_logits: &[Vec<f32>],
        task_loss: f32,
    ) -> f32 {
        let mutual = self.mutual_loss(network_idx, all_logits);
        task_loss + self.mutual_weight * mutual
    }

    /// Compute losses for all networks.
    #[must_use]
    pub fn all_losses(&self, all_logits: &[Vec<f32>], task_losses: &[f32]) -> Vec<f32> {
        (0..self.num_networks)
            .map(|i| self.combined_loss(i, all_logits, task_losses[i]))
            .collect()
    }

    #[must_use]
    pub fn num_networks(&self) -> usize {
        self.num_networks
    }

    #[must_use]
    pub fn temperature(&self) -> f32 {
        self.temperature
    }

    #[must_use]
    pub fn mutual_weight(&self) -> f32 {
        self.mutual_weight
    }
}

/// Progressive Distillation (Salimans & Ho, 2022).
///
/// Gradually distills a diffusion model by halving the number of sampling steps.
/// Used to speed up diffusion model inference.
#[derive(Debug, Clone)]
pub struct ProgressiveDistillation {
    /// Current number of steps
    current_steps: usize,
    /// Target number of steps
    target_steps: usize,
    /// Distillation weight
    weight: f32,
}

impl ProgressiveDistillation {
    /// Create progressive distillation from current to target steps.
    #[must_use]
    pub fn new(current_steps: usize, target_steps: usize, weight: f32) -> Self {
        assert!(
            current_steps > target_steps,
            "Current must be > target steps"
        );
        assert!(target_steps > 0, "Target steps must be positive");
        Self {
            current_steps,
            target_steps,
            weight,
        }
    }

    /// Check if we should halve steps (typically after convergence).
    #[must_use]
    pub fn should_halve(&self) -> bool {
        self.current_steps > self.target_steps * 2
    }

    /// Halve the number of steps.
    pub fn halve_steps(&mut self) {
        if self.current_steps > self.target_steps {
            self.current_steps /= 2;
        }
    }

    /// Compute distillation loss between teacher (2N steps) and student (N steps).
    #[must_use]
    pub fn compute_loss(&self, teacher_output: &[f32], student_output: &[f32]) -> f32 {
        assert_eq!(teacher_output.len(), student_output.len());
        let mse: f32 = teacher_output
            .iter()
            .zip(student_output.iter())
            .map(|(&t, &s)| (t - s).powi(2))
            .sum::<f32>()
            / teacher_output.len() as f32;
        self.weight * mse
    }

    #[must_use]
    pub fn current_steps(&self) -> usize {
        self.current_steps
    }

    #[must_use]
    pub fn target_steps(&self) -> usize {
        self.target_steps
    }
}

pub(super) fn softmax_with_temp(logits: &[f32], temp: f32) -> Vec<f32> {
    let scaled: Vec<f32> = logits.iter().map(|&x| x / temp).collect();
    let max = scaled.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp: Vec<f32> = scaled.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    exp.iter().map(|&e| e / sum).collect()
}

/// Prototypical Networks for few-shot learning (Snell et al., 2017).
///
/// Learns a metric space where classification is performed by computing
/// distances to class prototypes (mean embeddings of support examples).
#[derive(Debug, Clone)]
pub struct PrototypicalNetwork {
    /// Distance metric
    distance: DistanceMetric,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DistanceMetric {
    Euclidean,
    Cosine,
}

impl PrototypicalNetwork {
    #[must_use]
    pub fn new(distance: DistanceMetric) -> Self {
        Self { distance }
    }

    /// Compute class prototypes from support set embeddings.
    /// support: Vec of (embedding, `class_label`)
    #[must_use]
    pub fn compute_prototypes(&self, support: &[(Vec<f32>, usize)]) -> Vec<(usize, Vec<f32>)> {
        use std::collections::HashMap;
        let mut class_sums: HashMap<usize, (Vec<f32>, usize)> = HashMap::new();

        for (emb, class) in support {
            let entry = class_sums
                .entry(*class)
                .or_insert_with(|| (vec![0.0; emb.len()], 0));
            for (i, &v) in emb.iter().enumerate() {
                entry.0[i] += v;
            }
            entry.1 += 1;
        }

        class_sums
            .into_iter()
            .map(|(class, (sum, count))| {
                let proto: Vec<f32> = sum.iter().map(|&s| s / count as f32).collect();
                (class, proto)
            })
            .collect()
    }

    /// Classify query embedding against prototypes.
    #[must_use]
    pub fn classify(&self, query: &[f32], prototypes: &[(usize, Vec<f32>)]) -> usize {
        let mut best_class = 0;
        let mut best_dist = f32::INFINITY;

        for (class, proto) in prototypes {
            let dist = self.distance(query, proto);
            if dist < best_dist {
                best_dist = dist;
                best_class = *class;
            }
        }
        best_class
    }

    /// Compute class probabilities (softmax of negative distances).
    pub fn predict_proba(
        &self,
        query: &[f32],
        prototypes: &[(usize, Vec<f32>)],
    ) -> Vec<(usize, f32)> {
        let neg_dists: Vec<(usize, f32)> = prototypes
            .iter()
            .map(|(c, p)| (*c, -self.distance(query, p)))
            .collect();

        let max_d = neg_dists
            .iter()
            .map(|(_, d)| *d)
            .fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = neg_dists.iter().map(|(_, d)| (d - max_d).exp()).sum();

        neg_dists
            .iter()
            .map(|(c, d)| (*c, (d - max_d).exp() / exp_sum))
            .collect()
    }

    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.distance {
            DistanceMetric::Euclidean => a
                .iter()
                .zip(b)
                .map(|(&x, &y)| (x - y).powi(2))
                .sum::<f32>()
                .sqrt(),
            DistanceMetric::Cosine => {
                let dot: f32 = a.iter().zip(b).map(|(&x, &y)| x * y).sum();
                let na: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
                let nb: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
                1.0 - dot / (na * nb + 1e-10)
            }
        }
    }
}

impl Default for PrototypicalNetwork {
    fn default() -> Self {
        Self::new(DistanceMetric::Euclidean)
    }
}

/// Matching Networks for few-shot learning (Vinyals et al., 2016).
#[derive(Debug, Clone)]
pub struct MatchingNetwork {
    temperature: f32,
}

impl MatchingNetwork {
    #[must_use]
    pub fn new(temperature: f32) -> Self {
        Self { temperature }
    }

    /// Predict class by attention-weighted combination over support set.
    #[must_use]
    pub fn predict(&self, query: &[f32], support: &[(Vec<f32>, usize)]) -> usize {
        use std::collections::HashMap;
        let mut class_scores: HashMap<usize, f32> = HashMap::new();

        // Compute attention weights (softmax of cosine similarities)
        let sims: Vec<f32> = support
            .iter()
            .map(|(emb, _)| cosine_similarity(query, emb) / self.temperature)
            .collect();

        let max_sim = sims.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_sum: f32 = sims.iter().map(|&s| (s - max_sim).exp()).sum();
        let weights: Vec<f32> = sims
            .iter()
            .map(|&s| (s - max_sim).exp() / exp_sum)
            .collect();

        for ((_, class), &w) in support.iter().zip(&weights) {
            *class_scores.entry(*class).or_insert(0.0) += w;
        }

        class_scores
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(c, _)| c)
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(&x, &y)| x * y).sum();
    let na: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    dot / (na * nb + 1e-10)
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
