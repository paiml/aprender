//! Training result struct for GAN updates.

/// Training result from a GAN update step
#[derive(Debug, Clone)]
pub struct TrainingResult {
    /// Generator loss
    pub gen_loss: f32,
    /// Discriminator loss
    pub disc_loss: f32,
    /// Discriminator accuracy on real samples
    pub disc_real_acc: f32,
    /// Discriminator accuracy on fake samples
    pub disc_fake_acc: f32,
    /// Gradient penalty value
    pub gradient_penalty: f32,
}
