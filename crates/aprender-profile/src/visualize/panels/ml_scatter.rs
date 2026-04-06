//! ML scatter panel - cluster visualization in braille mode

use crate::visualize::app::VisualizeApp;
use crate::visualize::theme::{borders, graph};
use crate::visualize::widgets::braille_scatter;
use ratatui::{
    layout::Rect,
    style::{Modifier, Style},
    text::Span,
    widgets::{Block, Borders},
    Frame,
};

/// Draw the ML scatter panel
pub fn draw(f: &mut Frame, app: &VisualizeApp, area: Rect) {
    // Build header with stats
    let header = format!(
        " ML Clusters {} │ {} outliers │ silhouette: {:.2} ",
        app.cluster_points.iter().map(|(_, _, c)| *c).max().unwrap_or(0),
        app.outlier_count,
        app.silhouette_score
    );

    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(borders::STYLE)
        .border_style(Style::default().fg(borders::ML_SCATTER))
        .title(Span::styled(
            header,
            Style::default().fg(borders::ML_SCATTER).add_modifier(Modifier::BOLD),
        ));

    let inner = block.inner(area);
    f.render_widget(block, area);

    if inner.height < 2 || inner.width < 4 {
        return;
    }

    // Get cluster colors
    let cluster_colors = [graph::CLUSTER_0, graph::CLUSTER_1, graph::CLUSTER_2, graph::CLUSTER_3];

    // Render braille scatter plot
    braille_scatter::render(f, &app.cluster_points, &cluster_colors, inner);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::visualize::VisualizeConfig;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    #[test]
    fn test_draw_ml_scatter() {
        let backend = TestBackend::new(40, 20);
        let mut terminal = Terminal::new(backend).unwrap();
        let mut app = VisualizeApp::new(VisualizeConfig::default());

        // Add some test points
        app.cluster_points = vec![
            (0.2, 0.3, 0),
            (0.3, 0.4, 0),
            (0.7, 0.8, 1),
            (0.8, 0.7, 1),
            (0.5, 0.5, 255), // Outlier
        ];
        app.silhouette_score = 0.72;
        app.outlier_count = 1;

        terminal
            .draw(|f| {
                draw(f, &app, f.area());
            })
            .unwrap();

        let buffer = terminal.backend().buffer();
        let content = buffer_to_string(buffer);
        assert!(content.contains("ML Clusters"));
    }

    fn buffer_to_string(buffer: &ratatui::buffer::Buffer) -> String {
        let mut s = String::new();
        for y in 0..buffer.area.height {
            for x in 0..buffer.area.width {
                s.push(buffer[(x, y)].symbol().chars().next().unwrap_or(' '));
            }
            s.push('\n');
        }
        s
    }
}
