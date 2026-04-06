//! Canvas2D renderer - renders DrawCommands to HTML5 Canvas.

use presentar_core::draw::{BoxStyle, DrawCommand, FillRule, StrokeStyle};
use presentar_core::{Color, CornerRadius, Point, Rect};
use std::collections::HashMap;
use wasm_bindgen::JsCast;
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement, ImageData};

/// Renderer that draws to an HTML5 Canvas 2D context.
pub struct Canvas2DRenderer {
    canvas: HtmlCanvasElement,
    ctx: CanvasRenderingContext2d,
    /// Cached paths by reference ID.
    path_cache: HashMap<u32, Vec<Point>>,
    /// Cached images by tensor reference ID.
    image_cache: HashMap<u32, ImageData>,
}

impl Canvas2DRenderer {
    /// Create a new renderer for the given canvas element.
    pub fn new(canvas: HtmlCanvasElement) -> Result<Self, String> {
        let ctx = canvas
            .get_context("2d")
            .map_err(|e| format!("Failed to get 2d context: {:?}", e))?
            .ok_or("No 2d context available")?
            .dyn_into::<CanvasRenderingContext2d>()
            .map_err(|_| "Failed to cast to CanvasRenderingContext2d")?;

        Ok(Self {
            canvas,
            ctx,
            path_cache: HashMap::new(),
            image_cache: HashMap::new(),
        })
    }

    /// Register a path for later fill operations.
    pub fn register_path(&mut self, id: u32, points: Vec<Point>) {
        self.path_cache.insert(id, points);
    }

    /// Register an image from RGBA data.
    pub fn register_image(
        &mut self,
        id: u32,
        data: &[u8],
        width: u32,
        height: u32,
    ) -> Result<(), String> {
        let clamped = wasm_bindgen::Clamped(data);
        let image_data = ImageData::new_with_u8_clamped_array_and_sh(clamped, width, height)
            .map_err(|e| format!("Failed to create ImageData: {:?}", e))?;
        self.image_cache.insert(id, image_data);
        Ok(())
    }

    /// Clear path and image caches.
    pub fn clear_caches(&mut self) {
        self.path_cache.clear();
        self.image_cache.clear();
    }

    /// Get canvas width.
    pub fn width(&self) -> u32 {
        self.canvas.width()
    }

    /// Get canvas height.
    pub fn height(&self) -> u32 {
        self.canvas.height()
    }

    /// Clear the canvas.
    pub fn clear(&self) {
        self.ctx.clear_rect(
            0.0,
            0.0,
            f64::from(self.canvas.width()),
            f64::from(self.canvas.height()),
        );
    }

    /// Render a list of draw commands.
    pub fn render(&self, commands: &[DrawCommand]) {
        for cmd in commands {
            self.render_command(cmd);
        }
    }

    fn render_command(&self, cmd: &DrawCommand) {
        match cmd {
            DrawCommand::Rect {
                bounds,
                radius,
                style,
            } => {
                self.draw_rect(bounds, radius, style);
            }
            DrawCommand::Circle {
                center,
                radius,
                style,
            } => {
                self.draw_circle(center, *radius, style);
            }
            DrawCommand::Text {
                content,
                position,
                style,
            } => {
                self.draw_text(content, position, style);
            }
            DrawCommand::Path {
                points,
                closed,
                style,
            } => {
                self.draw_path(points, *closed, style);
            }
            DrawCommand::Group {
                children,
                transform,
            } => {
                self.ctx.save();
                self.ctx
                    .transform(
                        f64::from(transform.matrix[0]),
                        f64::from(transform.matrix[1]),
                        f64::from(transform.matrix[2]),
                        f64::from(transform.matrix[3]),
                        f64::from(transform.matrix[4]),
                        f64::from(transform.matrix[5]),
                    )
                    .ok();
                for child in children {
                    self.render_command(child);
                }
                self.ctx.restore();
            }
            DrawCommand::Clip { bounds, child } => {
                self.ctx.save();
                self.ctx.begin_path();
                self.ctx.rect(
                    f64::from(bounds.x),
                    f64::from(bounds.y),
                    f64::from(bounds.width),
                    f64::from(bounds.height),
                );
                self.ctx.clip();
                self.render_command(child);
                self.ctx.restore();
            }
            DrawCommand::Opacity { alpha, child } => {
                self.ctx.save();
                self.ctx.set_global_alpha(f64::from(*alpha));
                self.render_command(child);
                self.ctx.restore();
            }
            DrawCommand::Arc {
                center,
                radius,
                start_angle,
                end_angle,
                color,
            } => {
                self.draw_arc(center, *radius, *start_angle, *end_angle, color);
            }
            DrawCommand::Fill { path, color, rule } => {
                self.draw_fill(*path, color, rule);
            }
            DrawCommand::Image { tensor, bounds, .. } => {
                self.draw_image(*tensor, bounds);
            }
        }
    }

    fn draw_rect(&self, bounds: &Rect, radius: &CornerRadius, style: &BoxStyle) {
        self.ctx.begin_path();
        if radius.is_zero() {
            self.ctx.rect(
                f64::from(bounds.x),
                f64::from(bounds.y),
                f64::from(bounds.width),
                f64::from(bounds.height),
            );
        } else {
            self.rounded_rect(bounds, radius);
        }

        if let Some(fill) = style.fill {
            self.ctx.set_fill_style_str(&color_to_css(&fill));
            self.ctx.fill();
        }

        if let Some(stroke) = &style.stroke {
            self.ctx.set_stroke_style_str(&color_to_css(&stroke.color));
            self.ctx.set_line_width(f64::from(stroke.width));
            self.ctx.stroke();
        }
    }

    fn rounded_rect(&self, bounds: &Rect, radius: &CornerRadius) {
        let x = f64::from(bounds.x);
        let y = f64::from(bounds.y);
        let w = f64::from(bounds.width);
        let h = f64::from(bounds.height);
        let tl = f64::from(radius.top_left);
        let tr = f64::from(radius.top_right);
        let br = f64::from(radius.bottom_right);
        let bl = f64::from(radius.bottom_left);

        self.ctx.move_to(x + tl, y);
        self.ctx.line_to(x + w - tr, y);
        self.ctx.arc_to(x + w, y, x + w, y + tr, tr).ok();
        self.ctx.line_to(x + w, y + h - br);
        self.ctx.arc_to(x + w, y + h, x + w - br, y + h, br).ok();
        self.ctx.line_to(x + bl, y + h);
        self.ctx.arc_to(x, y + h, x, y + h - bl, bl).ok();
        self.ctx.line_to(x, y + tl);
        self.ctx.arc_to(x, y, x + tl, y, tl).ok();
        self.ctx.close_path();
    }

    fn draw_circle(&self, center: &Point, radius: f32, style: &BoxStyle) {
        self.ctx.begin_path();
        self.ctx
            .arc(
                f64::from(center.x),
                f64::from(center.y),
                f64::from(radius),
                0.0,
                std::f64::consts::TAU,
            )
            .ok();

        if let Some(fill) = style.fill {
            self.ctx.set_fill_style_str(&color_to_css(&fill));
            self.ctx.fill();
        }

        if let Some(stroke) = &style.stroke {
            self.ctx.set_stroke_style_str(&color_to_css(&stroke.color));
            self.ctx.set_line_width(f64::from(stroke.width));
            self.ctx.stroke();
        }
    }

    fn draw_text(
        &self,
        content: &str,
        position: &Point,
        style: &presentar_core::widget::TextStyle,
    ) {
        let weight = match style.weight {
            presentar_core::widget::FontWeight::Bold => "bold",
            presentar_core::widget::FontWeight::Medium => "500",
            presentar_core::widget::FontWeight::Semibold => "600",
            _ => "normal",
        };
        let font = format!("{} {}px sans-serif", weight, style.size);
        self.ctx.set_font(&font);
        self.ctx.set_fill_style_str(&color_to_css(&style.color));
        self.ctx
            .fill_text(
                content,
                f64::from(position.x),
                f64::from(position.y + style.size),
            )
            .ok();
    }

    fn draw_path(&self, points: &[Point], closed: bool, style: &StrokeStyle) {
        if points.is_empty() {
            return;
        }

        self.ctx.begin_path();
        self.ctx
            .move_to(f64::from(points[0].x), f64::from(points[0].y));
        for p in points.iter().skip(1) {
            self.ctx.line_to(f64::from(p.x), f64::from(p.y));
        }
        if closed {
            self.ctx.close_path();
        }

        self.ctx.set_stroke_style_str(&color_to_css(&style.color));
        self.ctx.set_line_width(f64::from(style.width));
        self.ctx.stroke();
    }

    fn draw_arc(
        &self,
        center: &Point,
        radius: f32,
        start_angle: f32,
        end_angle: f32,
        color: &Color,
    ) {
        self.ctx.begin_path();
        self.ctx.move_to(f64::from(center.x), f64::from(center.y));
        self.ctx
            .arc(
                f64::from(center.x),
                f64::from(center.y),
                f64::from(radius),
                f64::from(start_angle),
                f64::from(end_angle),
            )
            .ok();
        self.ctx.close_path();
        self.ctx.set_fill_style_str(&color_to_css(color));
        self.ctx.fill();
    }

    fn draw_fill(&self, path_id: u32, color: &Color, rule: &FillRule) {
        if let Some(points) = self.path_cache.get(&path_id) {
            if points.is_empty() {
                return;
            }

            self.ctx.begin_path();
            self.ctx
                .move_to(f64::from(points[0].x), f64::from(points[0].y));
            for p in points.iter().skip(1) {
                self.ctx.line_to(f64::from(p.x), f64::from(p.y));
            }
            self.ctx.close_path();

            self.ctx.set_fill_style_str(&color_to_css(color));

            // Apply fill rule
            match rule {
                FillRule::NonZero => self.ctx.fill(),
                FillRule::EvenOdd => {
                    self.ctx
                        .fill_with_canvas_winding_rule(web_sys::CanvasWindingRule::Evenodd);
                }
            }
        }
    }

    fn draw_image(&self, tensor_id: u32, bounds: &Rect) {
        if let Some(image_data) = self.image_cache.get(&tensor_id) {
            // Draw at origin first, then scale if needed
            let img_width = image_data.width() as f32;
            let img_height = image_data.height() as f32;

            self.ctx.save();

            // Translate to position
            self.ctx
                .translate(f64::from(bounds.x), f64::from(bounds.y))
                .ok();

            // Scale to fit bounds if different from image size
            if (img_width - bounds.width).abs() > 0.01 || (img_height - bounds.height).abs() > 0.01
            {
                let scale_x = bounds.width / img_width;
                let scale_y = bounds.height / img_height;
                self.ctx.scale(f64::from(scale_x), f64::from(scale_y)).ok();
            }

            // Put the image data
            self.ctx.put_image_data(image_data, 0.0, 0.0).ok();

            self.ctx.restore();
        }
    }
}

fn color_to_css(color: &Color) -> String {
    format!(
        "rgba({},{},{},{})",
        (color.r * 255.0) as u8,
        (color.g * 255.0) as u8,
        (color.b * 255.0) as u8,
        color.a
    )
}
