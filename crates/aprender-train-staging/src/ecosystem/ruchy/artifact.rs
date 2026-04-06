//! Artifact conversion for session data.

use super::error::RuchyBridgeError;
use super::session::EntrenarSession;
use crate::research::{ArtifactType, Author, ContributorRole, License, ResearchArtifact};

/// Convert a Ruchy session to a research artifact.
///
/// Preserves training history and metadata in the artifact.
pub fn session_to_artifact(
    session: &EntrenarSession,
) -> Result<ResearchArtifact, RuchyBridgeError> {
    if !session.has_training_data() && session.code_history.is_empty() {
        return Err(RuchyBridgeError::NoTrainingHistory);
    }

    let mut artifact =
        ResearchArtifact::new(&session.id, &session.name, ArtifactType::Notebook, License::Mit);

    // Add user as author if available
    if let Some(ref user) = session.user {
        let author = Author::new(user)
            .with_role(ContributorRole::Software)
            .with_role(ContributorRole::Investigation);
        artifact = artifact.with_author(author);
    }

    // Add description with metrics summary
    let description = build_session_description(session);
    artifact = artifact.with_description(description);

    // Add keywords from tags
    if session.tags.is_empty() {
        artifact = artifact.with_keywords(["training", "experiment", "entrenar"]);
    } else {
        artifact = artifact.with_keywords(session.tags.iter().map(String::as_str));
    }

    // Set version based on training steps
    let steps = session.metrics.total_steps();
    artifact = artifact.with_version(format!("1.0.0+steps{steps}"));

    Ok(artifact)
}

/// Build a description from session data.
pub(crate) fn build_session_description(session: &EntrenarSession) -> String {
    let mut parts = Vec::new();

    if let Some(ref arch) = session.model_architecture {
        parts.push(format!("Model: {arch}"));
    }

    if let Some(ref dataset) = session.dataset_id {
        parts.push(format!("Dataset: {dataset}"));
    }

    let steps = session.metrics.total_steps();
    if steps > 0 {
        parts.push(format!("Training steps: {steps}"));
    }

    if let Some(loss) = session.metrics.final_loss() {
        parts.push(format!("Final loss: {loss:.4}"));
    }

    if let Some(acc) = session.metrics.final_accuracy() {
        parts.push(format!("Final accuracy: {acc:.2}%"));
    }

    if let Some(duration) = session.duration() {
        let hours = duration.num_hours();
        let minutes = duration.num_minutes() % 60;
        parts.push(format!("Duration: {hours}h {minutes}m"));
    }

    if parts.is_empty() {
        format!("Training session from Ruchy ({})", session.id)
    } else {
        parts.join(". ")
    }
}
