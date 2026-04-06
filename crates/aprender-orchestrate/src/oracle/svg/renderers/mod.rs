//! SVG Renderers
//!
//! Specialized renderers for different diagram types.

pub mod shape_heavy;
pub mod text_heavy;

pub use shape_heavy::ShapeHeavyRenderer;
pub use text_heavy::TextHeavyRenderer;

/// Render mode for SVG generation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RenderMode {
    /// Shape-heavy: architectural diagrams, component diagrams
    #[default]
    ShapeHeavy,
    /// Text-heavy: documentation diagrams, flowcharts with labels
    TextHeavy,
}

impl std::fmt::Display for RenderMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ShapeHeavy => write!(f, "shape-heavy"),
            Self::TextHeavy => write!(f, "text-heavy"),
        }
    }
}

impl std::str::FromStr for RenderMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "shape-heavy" | "shape_heavy" | "shapes" => Ok(Self::ShapeHeavy),
            "text-heavy" | "text_heavy" | "text" => Ok(Self::TextHeavy),
            _ => Err(format!("Unknown render mode: {}", s)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_render_mode_display() {
        assert_eq!(format!("{}", RenderMode::ShapeHeavy), "shape-heavy");
        assert_eq!(format!("{}", RenderMode::TextHeavy), "text-heavy");
    }

    #[test]
    fn test_render_mode_from_str() {
        assert_eq!(
            "shape-heavy".parse::<RenderMode>().expect("parse failed"),
            RenderMode::ShapeHeavy
        );
        assert_eq!(
            "text-heavy".parse::<RenderMode>().expect("parse failed"),
            RenderMode::TextHeavy
        );
        assert!("invalid".parse::<RenderMode>().is_err());
    }
}
