//! Text prompt interface for declarative visualization.
//!
//! Provides a simple DSL for specifying visualizations via text commands.
//!
//! # Syntax
//!
//! ```text
//! <plot_type> <data_spec> [options...]
//!
//! Plot types: scatter, line, histogram, heatmap, boxplot
//!
//! Data specs:
//!   x=[1,2,3] y=[4,5,6]     - Paired x/y data
//!   data=[1,2,3,4,5]        - Single data array
//!   matrix=[[1,2],[3,4]]    - 2D matrix data
//!   groups=[[1,2],[3,4]]    - Multiple groups
//!
//! Options:
//!   width=800 height=600    - Dimensions
//!   color=red|blue|#ff0000  - Colors
//!   title="My Plot"         - Title (quoted)
//!   size=5.0                - Point/line size
//! ```
//!
//! # Example
//!
//! ```rust
//! use trueno_viz::prompt::{parse_prompt, PlotSpec};
//!
//! let spec = parse_prompt("scatter x=[1,2,3] y=[4,5,6] color=blue").unwrap();
//! assert_eq!(spec.plot_type, "scatter");
//! ```

use crate::color::Rgba;
use crate::error::{Error, Result};
use crate::framebuffer::Framebuffer;
use crate::plots::{BoxPlot, Heatmap, Histogram, LineChart, LineSeries, ScatterPlot};
use batuta_common::display::WithDimensions;

/// A parsed plot specification.
#[derive(Debug, Clone)]
pub struct PlotSpec {
    /// Plot type (scatter, line, histogram, heatmap, boxplot)
    pub plot_type: String,
    /// X data (for scatter, line)
    pub x_data: Option<Vec<f32>>,
    /// Y data (for scatter, line)
    pub y_data: Option<Vec<f32>>,
    /// Single data array (for histogram)
    pub data: Option<Vec<f32>>,
    /// Matrix data (for heatmap)
    pub matrix: Option<Vec<Vec<f32>>>,
    /// Groups data (for boxplot)
    pub groups: Option<Vec<Vec<f32>>>,
    /// Width in pixels
    pub width: u32,
    /// Height in pixels
    pub height: u32,
    /// Primary color
    pub color: Rgba,
    /// Title
    pub title: Option<String>,
    /// Point/line size
    pub size: f32,
}

impl Default for PlotSpec {
    fn default() -> Self {
        Self {
            plot_type: String::new(),
            x_data: None,
            y_data: None,
            data: None,
            matrix: None,
            groups: None,
            width: 600,
            height: 400,
            color: Rgba::new(66, 133, 244, 255),
            title: None,
            size: 5.0,
        }
    }
}

impl PlotSpec {
    /// Render this plot specification to a framebuffer.
    ///
    /// # Errors
    ///
    /// Returns an error if the plot cannot be rendered.
    pub fn render(&self) -> Result<Framebuffer> {
        match self.plot_type.as_str() {
            "scatter" => self.render_scatter(),
            "line" => self.render_line(),
            "histogram" => self.render_histogram(),
            "heatmap" => self.render_heatmap(),
            "boxplot" => self.render_boxplot(),
            _ => Err(Error::Rendering(format!("Unknown plot type: {}", self.plot_type))),
        }
    }

    fn render_scatter(&self) -> Result<Framebuffer> {
        let x = self.x_data.as_ref().ok_or(Error::EmptyData)?;
        let y = self.y_data.as_ref().ok_or(Error::EmptyData)?;

        let plot = ScatterPlot::new()
            .x(x)
            .y(y)
            .color(self.color)
            .size(self.size)
            .dimensions(self.width, self.height)
            .build()?;

        plot.to_framebuffer()
    }

    fn render_line(&self) -> Result<Framebuffer> {
        let x = self.x_data.as_ref().ok_or(Error::EmptyData)?;
        let y = self.y_data.as_ref().ok_or(Error::EmptyData)?;

        let plot = LineChart::new()
            .add_series(LineSeries::new("data").data(x, y).color(self.color).thickness(self.size))
            .dimensions(self.width, self.height)
            .build()?;

        plot.to_framebuffer()
    }

    fn render_histogram(&self) -> Result<Framebuffer> {
        let data = self.data.as_ref().ok_or(Error::EmptyData)?;

        let plot = Histogram::new()
            .data(data)
            .color(self.color)
            .dimensions(self.width, self.height)
            .build()?;

        plot.to_framebuffer()
    }

    fn render_heatmap(&self) -> Result<Framebuffer> {
        let matrix = self.matrix.as_ref().ok_or(Error::EmptyData)?;

        let plot = Heatmap::new().data_2d(matrix).dimensions(self.width, self.height).build()?;

        plot.to_framebuffer()
    }

    fn render_boxplot(&self) -> Result<Framebuffer> {
        let groups = self.groups.as_ref().ok_or(Error::EmptyData)?;

        let mut plot = BoxPlot::new().dimensions(self.width, self.height);

        for (i, group) in groups.iter().enumerate() {
            plot = plot.add_group(group, &format!("Group {}", i + 1));
        }

        let built = plot.build()?;
        built.to_framebuffer()
    }
}

/// Parse a text prompt into a plot specification.
///
/// # Errors
///
/// Returns an error if the prompt cannot be parsed.
///
/// # Example
///
/// ```rust
/// use trueno_viz::prompt::parse_prompt;
///
/// let spec = parse_prompt("scatter x=[1,2,3] y=[4,5,6]").unwrap();
/// assert_eq!(spec.plot_type, "scatter");
/// ```
pub fn parse_prompt(prompt: &str) -> Result<PlotSpec> {
    let prompt = prompt.trim();
    if prompt.is_empty() {
        return Err(Error::Rendering("Empty prompt".into()));
    }

    let mut spec = PlotSpec::default();
    let mut parts = prompt.split_whitespace().peekable();

    // First token is plot type
    let plot_type =
        parts.next().ok_or_else(|| Error::Rendering("No plot type specified".into()))?;
    spec.plot_type = plot_type.to_lowercase();

    // Parse remaining tokens
    while let Some(token) = parts.next() {
        if let Some((key, value)) = token.split_once('=') {
            match key.to_lowercase().as_str() {
                "x" => spec.x_data = Some(parse_array(value)?),
                "y" => spec.y_data = Some(parse_array(value)?),
                "data" => spec.data = Some(parse_array(value)?),
                "matrix" => spec.matrix = Some(parse_matrix(value)?),
                "groups" => spec.groups = Some(parse_matrix(value)?),
                "width" => {
                    spec.width =
                        value.parse().map_err(|_| Error::Rendering("Invalid width".into()))?;
                }
                "height" => {
                    spec.height =
                        value.parse().map_err(|_| Error::Rendering("Invalid height".into()))?;
                }
                "size" => {
                    spec.size =
                        value.parse().map_err(|_| Error::Rendering("Invalid size".into()))?;
                }
                "color" => spec.color = parse_color(value)?,
                "title" => {
                    // Handle quoted strings
                    let mut title = value.trim_matches('"').to_string();
                    // Collect continuation if quote wasn't closed
                    if value.starts_with('"') && !value.ends_with('"') {
                        for next in parts.by_ref() {
                            title.push(' ');
                            title.push_str(next.trim_matches('"'));
                            if next.ends_with('"') {
                                break;
                            }
                        }
                    }
                    spec.title = Some(title);
                }
                _ => {} // Ignore unknown options
            }
        }
    }

    // Validate required data
    match spec.plot_type.as_str() {
        "scatter" | "line" => {
            if spec.x_data.is_none() || spec.y_data.is_none() {
                return Err(Error::Rendering("scatter/line requires x=[...] and y=[...]".into()));
            }
        }
        "histogram" => {
            if spec.data.is_none() {
                return Err(Error::Rendering("histogram requires data=[...]".into()));
            }
        }
        "heatmap" => {
            if spec.matrix.is_none() {
                return Err(Error::Rendering("heatmap requires matrix=[[...]]".into()));
            }
        }
        "boxplot" => {
            if spec.groups.is_none() {
                return Err(Error::Rendering("boxplot requires groups=[[...]]".into()));
            }
        }
        _ => {}
    }

    Ok(spec)
}

/// Parse a 1D array like "[1,2,3,4]".
fn parse_array(s: &str) -> Result<Vec<f32>> {
    let s = s.trim().trim_start_matches('[').trim_end_matches(']');
    if s.is_empty() {
        return Ok(Vec::new());
    }

    s.split(',')
        .map(|v| {
            v.trim().parse::<f32>().map_err(|_| Error::Rendering(format!("Invalid number: {v}")))
        })
        .collect()
}

/// Parse a 2D matrix like "[[1,2],[3,4]]".
fn parse_matrix(s: &str) -> Result<Vec<Vec<f32>>> {
    let s = s.trim();
    if !s.starts_with("[[") || !s.ends_with("]]") {
        return Err(Error::Rendering("Matrix must be [[...],[...]]".into()));
    }

    let inner = &s[1..s.len() - 1]; // Remove outer brackets
    let mut result = Vec::new();
    let mut depth = 0;
    let mut current_start = 0;

    for (i, c) in inner.char_indices() {
        match c {
            '[' => {
                if depth == 0 {
                    current_start = i;
                }
                depth += 1;
            }
            ']' => {
                depth -= 1;
                if depth == 0 {
                    let row_str = &inner[current_start..=i];
                    result.push(parse_array(row_str)?);
                }
            }
            _ => {}
        }
    }

    Ok(result)
}

/// Parse a color string.
fn parse_color(s: &str) -> Result<Rgba> {
    let s = s.to_lowercase();
    match s.as_str() {
        "red" => Ok(Rgba::RED),
        "green" => Ok(Rgba::GREEN),
        "blue" => Ok(Rgba::BLUE),
        "black" => Ok(Rgba::BLACK),
        "white" => Ok(Rgba::WHITE),
        "yellow" => Ok(Rgba::new(255, 255, 0, 255)),
        "cyan" => Ok(Rgba::new(0, 255, 255, 255)),
        "magenta" => Ok(Rgba::new(255, 0, 255, 255)),
        "orange" => Ok(Rgba::new(255, 165, 0, 255)),
        "purple" => Ok(Rgba::new(128, 0, 128, 255)),
        "pink" => Ok(Rgba::new(255, 192, 203, 255)),
        "gray" | "grey" => Ok(Rgba::new(128, 128, 128, 255)),
        _ if s.starts_with('#') && s.len() == 7 => {
            // Parse hex color
            let r = u8::from_str_radix(&s[1..3], 16)
                .map_err(|_| Error::Rendering("Invalid hex color".into()))?;
            let g = u8::from_str_radix(&s[3..5], 16)
                .map_err(|_| Error::Rendering("Invalid hex color".into()))?;
            let b = u8::from_str_radix(&s[5..7], 16)
                .map_err(|_| Error::Rendering("Invalid hex color".into()))?;
            Ok(Rgba::new(r, g, b, 255))
        }
        _ => Err(Error::Rendering(format!("Unknown color: {s}"))),
    }
}

/// Create a plot from a text prompt and render to framebuffer.
///
/// # Errors
///
/// Returns an error if parsing or rendering fails.
///
/// # Example
///
/// ```rust,ignore
/// let fb = from_prompt("scatter x=[1,2,3] y=[4,5,6] color=red")?;
/// ```
pub fn from_prompt(prompt: &str) -> Result<Framebuffer> {
    let spec = parse_prompt(prompt)?;
    spec.render()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_array() {
        let arr = parse_array("[1,2,3,4,5]").expect("parsing should succeed");
        assert_eq!(arr.len(), 5);
        assert!((arr[0] - 1.0).abs() < 0.01);
        assert!((arr[4] - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_parse_array_floats() {
        let arr = parse_array("[1.5,2.5,3.5]").expect("parsing should succeed");
        assert_eq!(arr.len(), 3);
        assert!((arr[1] - 2.5).abs() < 0.01);
    }

    #[test]
    fn test_parse_array_empty() {
        let arr = parse_array("[]").expect("parsing should succeed");
        assert!(arr.is_empty());
    }

    #[test]
    fn test_parse_matrix() {
        let mat = parse_matrix("[[1,2],[3,4]]").expect("parsing should succeed");
        assert_eq!(mat.len(), 2);
        assert_eq!(mat[0].len(), 2);
        assert!((mat[0][0] - 1.0).abs() < 0.01);
        assert!((mat[1][1] - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_parse_color_named() {
        assert_eq!(parse_color("red").expect("parsing should succeed"), Rgba::RED);
        assert_eq!(parse_color("BLUE").expect("parsing should succeed"), Rgba::BLUE);
        assert_eq!(parse_color("Green").expect("parsing should succeed"), Rgba::GREEN);
    }

    #[test]
    fn test_parse_color_hex() {
        let color = parse_color("#ff8800").expect("parsing should succeed");
        assert_eq!(color.r, 255);
        assert_eq!(color.g, 136);
        assert_eq!(color.b, 0);
    }

    #[test]
    fn test_parse_prompt_scatter() {
        let spec = parse_prompt("scatter x=[1,2,3] y=[4,5,6]").expect("parsing should succeed");
        assert_eq!(spec.plot_type, "scatter");
        assert_eq!(spec.x_data.as_ref().expect("value should be present").len(), 3);
        assert_eq!(spec.y_data.as_ref().expect("value should be present").len(), 3);
    }

    #[test]
    fn test_parse_prompt_with_options() {
        let spec = parse_prompt("scatter x=[1,2] y=[3,4] width=800 height=600 color=red")
            .expect("parsing should succeed");
        assert_eq!(spec.width, 800);
        assert_eq!(spec.height, 600);
        assert_eq!(spec.color, Rgba::RED);
    }

    #[test]
    fn test_parse_prompt_histogram() {
        let spec =
            parse_prompt("histogram data=[1,2,2,3,3,3,4,4,5]").expect("parsing should succeed");
        assert_eq!(spec.plot_type, "histogram");
        assert_eq!(spec.data.as_ref().expect("value should be present").len(), 9);
    }

    #[test]
    fn test_parse_prompt_heatmap() {
        let spec = parse_prompt("heatmap matrix=[[1,2],[3,4]]").expect("parsing should succeed");
        assert_eq!(spec.plot_type, "heatmap");
        let mat = spec.matrix.as_ref().expect("operation should succeed");
        assert_eq!(mat.len(), 2);
    }

    #[test]
    fn test_parse_prompt_boxplot() {
        let spec =
            parse_prompt("boxplot groups=[[1,2,3],[4,5,6]]").expect("parsing should succeed");
        assert_eq!(spec.plot_type, "boxplot");
        assert_eq!(spec.groups.as_ref().expect("value should be present").len(), 2);
    }

    #[test]
    fn test_parse_prompt_case_insensitive() {
        let spec = parse_prompt("SCATTER X=[1,2] Y=[3,4]").expect("parsing should succeed");
        assert_eq!(spec.plot_type, "scatter");
    }

    #[test]
    fn test_parse_prompt_error_missing_data() {
        let result = parse_prompt("scatter x=[1,2,3]");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_prompt_empty() {
        let result = parse_prompt("");
        assert!(result.is_err());
    }

    #[test]
    fn test_from_prompt_scatter() {
        let fb = from_prompt("scatter x=[1,2,3,4,5] y=[1,4,9,16,25] width=200 height=150")
            .expect("operation should succeed");
        assert_eq!(fb.width(), 200);
        assert_eq!(fb.height(), 150);
    }

    #[test]
    fn test_from_prompt_line() {
        let fb = from_prompt("line x=[0,1,2,3] y=[0,1,0,1] width=200 height=150")
            .expect("operation should succeed");
        assert_eq!(fb.width(), 200);
        assert_eq!(fb.height(), 150);
    }

    #[test]
    fn test_from_prompt_histogram() {
        let fb = from_prompt("histogram data=[1,2,2,3,3,3,4,4,5] width=200 height=150")
            .expect("operation should succeed");
        assert_eq!(fb.width(), 200);
        assert_eq!(fb.height(), 150);
    }

    #[test]
    fn test_from_prompt_heatmap() {
        let fb = from_prompt("heatmap matrix=[[1,2,3],[4,5,6],[7,8,9]] width=200 height=150")
            .expect("operation should succeed");
        assert_eq!(fb.width(), 200);
        assert_eq!(fb.height(), 150);
    }

    #[test]
    fn test_from_prompt_boxplot() {
        let fb = from_prompt(
            "boxplot groups=[[1,2,3,4,5],[2,3,4,5,6],[3,4,5,6,7]] width=200 height=150",
        )
        .expect("operation should succeed");
        assert_eq!(fb.width(), 200);
        assert_eq!(fb.height(), 150);
    }

    #[test]
    fn test_parse_color_all_named() {
        assert_eq!(parse_color("black").expect("parsing should succeed"), Rgba::BLACK);
        assert_eq!(parse_color("white").expect("parsing should succeed"), Rgba::WHITE);
        assert_eq!(
            parse_color("yellow").expect("parsing should succeed"),
            Rgba::new(255, 255, 0, 255)
        );
        assert_eq!(
            parse_color("cyan").expect("parsing should succeed"),
            Rgba::new(0, 255, 255, 255)
        );
        assert_eq!(
            parse_color("magenta").expect("parsing should succeed"),
            Rgba::new(255, 0, 255, 255)
        );
        assert_eq!(
            parse_color("orange").expect("parsing should succeed"),
            Rgba::new(255, 165, 0, 255)
        );
        assert_eq!(
            parse_color("purple").expect("parsing should succeed"),
            Rgba::new(128, 0, 128, 255)
        );
        assert_eq!(
            parse_color("pink").expect("parsing should succeed"),
            Rgba::new(255, 192, 203, 255)
        );
        assert_eq!(
            parse_color("gray").expect("parsing should succeed"),
            Rgba::new(128, 128, 128, 255)
        );
        assert_eq!(
            parse_color("grey").expect("parsing should succeed"),
            Rgba::new(128, 128, 128, 255)
        );
    }

    #[test]
    fn test_parse_color_invalid_hex() {
        let result = parse_color("#gggggg");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_color_unknown() {
        let result = parse_color("unknowncolor");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_array_invalid() {
        let result = parse_array("[1,2,abc,4]");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_matrix_invalid_format() {
        let result = parse_matrix("[1,2,3]");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_prompt_title() {
        let spec = parse_prompt("scatter x=[1,2] y=[3,4] title=\"My Plot\"")
            .expect("parsing should succeed");
        assert_eq!(spec.title, Some("My Plot".to_string()));
    }

    #[test]
    fn test_parse_prompt_title_multiword() {
        let spec = parse_prompt("scatter x=[1,2] y=[3,4] title=\"My Multi Word Title\"")
            .expect("parsing should succeed");
        assert_eq!(spec.title, Some("My Multi Word Title".to_string()));
    }

    #[test]
    fn test_parse_prompt_size() {
        let spec =
            parse_prompt("scatter x=[1,2] y=[3,4] size=10.0").expect("parsing should succeed");
        assert!((spec.size - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_parse_prompt_invalid_width() {
        let result = parse_prompt("scatter x=[1,2] y=[3,4] width=abc");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_prompt_invalid_height() {
        let result = parse_prompt("scatter x=[1,2] y=[3,4] height=abc");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_prompt_invalid_size() {
        let result = parse_prompt("scatter x=[1,2] y=[3,4] size=abc");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_prompt_unknown_option() {
        // Unknown options should be ignored
        let spec =
            parse_prompt("scatter x=[1,2] y=[3,4] unknown=value").expect("parsing should succeed");
        assert_eq!(spec.x_data.as_ref().expect("value should be present").len(), 2);
    }

    #[test]
    fn test_render_unknown_plot_type() {
        let spec = PlotSpec { plot_type: "unknownplot".to_string(), ..PlotSpec::default() };
        let result = spec.render();
        assert!(result.is_err());
    }

    #[test]
    fn test_render_scatter_missing_x() {
        let spec = PlotSpec {
            plot_type: "scatter".to_string(),
            y_data: Some(vec![1.0, 2.0]),
            ..PlotSpec::default()
        };
        let result = spec.render();
        assert!(result.is_err());
    }

    #[test]
    fn test_render_scatter_missing_y() {
        let spec = PlotSpec {
            plot_type: "scatter".to_string(),
            x_data: Some(vec![1.0, 2.0]),
            ..PlotSpec::default()
        };
        let result = spec.render();
        assert!(result.is_err());
    }

    #[test]
    fn test_render_line_missing_data() {
        let spec = PlotSpec { plot_type: "line".to_string(), ..PlotSpec::default() };
        let result = spec.render();
        assert!(result.is_err());
    }

    #[test]
    fn test_render_histogram_missing_data() {
        let spec = PlotSpec { plot_type: "histogram".to_string(), ..PlotSpec::default() };
        let result = spec.render();
        assert!(result.is_err());
    }

    #[test]
    fn test_render_heatmap_missing_matrix() {
        let spec = PlotSpec { plot_type: "heatmap".to_string(), ..PlotSpec::default() };
        let result = spec.render();
        assert!(result.is_err());
    }

    #[test]
    fn test_render_boxplot_missing_groups() {
        let spec = PlotSpec { plot_type: "boxplot".to_string(), ..PlotSpec::default() };
        let result = spec.render();
        assert!(result.is_err());
    }

    #[test]
    fn test_plotspec_default() {
        let spec = PlotSpec::default();
        assert!(spec.plot_type.is_empty());
        assert!(spec.x_data.is_none());
        assert!(spec.y_data.is_none());
        assert!(spec.data.is_none());
        assert!(spec.matrix.is_none());
        assert!(spec.groups.is_none());
        assert_eq!(spec.width, 600);
        assert_eq!(spec.height, 400);
        assert!((spec.size - 5.0).abs() < f32::EPSILON);
        assert!(spec.title.is_none());
    }

    #[test]
    fn test_parse_prompt_histogram_missing_data() {
        let result = parse_prompt("histogram");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_prompt_heatmap_missing_matrix() {
        let result = parse_prompt("heatmap");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_prompt_boxplot_missing_groups() {
        let result = parse_prompt("boxplot");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_prompt_line_missing_y() {
        let result = parse_prompt("line x=[1,2,3]");
        assert!(result.is_err());
    }

    #[test]
    fn test_plotspec_debug_clone() {
        let spec = parse_prompt("scatter x=[1,2] y=[3,4]").expect("parsing should succeed");
        let spec2 = spec.clone();
        let _ = format!("{spec2:?}");
    }

    #[test]
    fn test_parse_matrix_multiple_rows() {
        let mat = parse_matrix("[[1,2,3],[4,5,6],[7,8,9]]").expect("parsing should succeed");
        assert_eq!(mat.len(), 3);
        assert_eq!(mat[0].len(), 3);
        assert_eq!(mat[2].len(), 3);
    }
}
