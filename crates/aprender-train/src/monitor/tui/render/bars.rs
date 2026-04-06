//! Bar rendering, sparklines, and trend indicators.

use super::super::color::{ColorMode, Styled, TrainingPalette};

pub(crate) const BLOCK_FULL: char = '\u{2588}';
pub(crate) const BLOCK_LIGHT: char = '\u{2591}';
const ARROW_UP: &str = "\u{2191}";
const ARROW_DOWN: &str = "\u{2193}";
const ARROW_FLAT: &str = "\u{2192}";
pub(crate) const BRAILLE_BASE: u32 = 0x2800;
pub(crate) const BRAILLE_DOTS: [u32; 8] = [0x01, 0x02, 0x04, 0x40, 0x08, 0x10, 0x20, 0x80];

#[allow(clippy::cast_precision_loss)]
pub fn build_block_bar(percent: f32, width: usize) -> String {
    let pct = percent.clamp(0.0, 100.0);
    let filled_f = ((pct / 100.0) * width as f32).clamp(0.0, width as f32);
    let filled = filled_f as usize;
    let empty = width.saturating_sub(filled);
    format!("{}{}", BLOCK_FULL.to_string().repeat(filled), BLOCK_LIGHT.to_string().repeat(empty))
}

#[allow(clippy::cast_precision_loss)]
pub fn build_colored_block_bar(percent: f32, width: usize, color_mode: ColorMode) -> String {
    let pct = percent.clamp(0.0, 100.0);
    let filled_f = ((pct / 100.0) * width as f32).clamp(0.0, width as f32);
    let filled = filled_f as usize;
    let empty = width.saturating_sub(filled);

    let color = pct_color(pct);
    let filled_str = BLOCK_FULL.to_string().repeat(filled);
    let empty_str = BLOCK_LIGHT.to_string().repeat(empty);

    if color_mode == ColorMode::Mono {
        format!("{filled_str}{empty_str}")
    } else {
        format!(
            "{}{}",
            Styled::new(&filled_str, color_mode).fg(color),
            Styled::new(&empty_str, color_mode).fg((60, 60, 60))
        )
    }
}

/// Safely convert an f32 in [0.0, 255.0] to u8, clamping to valid range.
#[inline]
fn f32_to_u8(v: f32) -> u8 {
    u8::try_from(v.clamp(0.0, 255.0) as u32).unwrap_or(u8::MAX)
}

pub fn pct_color(pct: f32) -> (u8, u8, u8) {
    let p = pct.clamp(0.0, 100.0);
    if p >= 90.0 {
        (255, 64, 64)
    } else if p >= 75.0 {
        let t = (p - 75.0) / 15.0;
        (255, f32_to_u8(180.0 - t * 116.0), 64)
    } else if p >= 50.0 {
        let t = (p - 50.0) / 25.0;
        (255, f32_to_u8(220.0 - t * 40.0), 64)
    } else if p >= 25.0 {
        let t = (p - 25.0) / 25.0;
        (f32_to_u8(100.0 + t * 155.0), 220, f32_to_u8(100.0 - t * 36.0))
    } else {
        let t = p / 25.0;
        (f32_to_u8(64.0 + t * 36.0), f32_to_u8(180.0 + t * 40.0), f32_to_u8(220.0 - t * 120.0))
    }
}

#[allow(clippy::cast_precision_loss)]
pub fn render_sparkline(data: &[f32], width: usize, color_mode: ColorMode) -> String {
    if data.is_empty() {
        return " ".repeat(width);
    }

    let min = data.iter().copied().filter(|v| v.is_finite()).fold(f32::INFINITY, f32::min);
    let max = data.iter().copied().filter(|v| v.is_finite()).fold(f32::NEG_INFINITY, f32::max);
    let range = (max - min).max(0.001);

    let mut result = String::new();
    for i in 0..width {
        let idx = (i * data.len()) / width.max(1);
        let idx2 = ((i * 2 + 1) * data.len()) / (width * 2).max(1);

        let v1 = data.get(idx).copied().unwrap_or(min);
        let v2 = data.get(idx2).copied().unwrap_or(v1);

        let h1 =
            if v1.is_finite() { (((v1 - min) / range) * 3.99).clamp(0.0, 3.0) as usize } else { 0 };
        let h2 =
            if v2.is_finite() { (((v2 - min) / range) * 3.99).clamp(0.0, 3.0) as usize } else { 0 };

        let mut code: u32 = 0;
        for y in 0..=h1.min(3) {
            code |= BRAILLE_DOTS[3 - y];
        }
        for y in 0..=h2.min(3) {
            code |= BRAILLE_DOTS[7 - y];
        }

        result.push(char::from_u32(BRAILLE_BASE + code).unwrap_or('\u{28FF}'));
    }

    let trend_color = if data.len() > 1 {
        let first = data.first().copied().unwrap_or(0.0);
        let last = data.last().copied().unwrap_or(0.0);
        if last < first * 0.95 {
            TrainingPalette::SUCCESS
        } else if last > first * 1.05 {
            TrainingPalette::ERROR
        } else {
            TrainingPalette::INFO
        }
    } else {
        TrainingPalette::INFO
    };

    if color_mode == ColorMode::Mono {
        result
    } else {
        Styled::new(&result, color_mode).fg(trend_color).to_string()
    }
}

pub fn trend_arrow(data: &[f32]) -> &'static str {
    if data.len() < 2 {
        return ARROW_FLAT;
    }
    let recent: Vec<f32> = data.iter().rev().take(5).copied().collect();
    if recent.len() < 2 {
        return ARROW_FLAT;
    }
    // recent.len() is at most 5, so no precision loss converting to f32
    let avg_recent: f32 = recent.iter().sum::<f32>() / (recent.len().max(1) as f32);
    let old_count = data.len().saturating_sub(5).clamp(1, 5);
    // old_count is at most 5, so no precision loss converting to f32
    let avg_old: f32 = data.iter().rev().skip(5).take(5).copied().sum::<f32>() / (old_count as f32);

    if avg_recent < avg_old * 0.95 {
        ARROW_DOWN
    } else if avg_recent > avg_old * 1.05 {
        ARROW_UP
    } else {
        ARROW_FLAT
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_bar() {
        let bar = build_block_bar(50.0, 10);
        assert_eq!(bar.chars().count(), 10);
        assert!(bar.contains(BLOCK_FULL));
        assert!(bar.contains(BLOCK_LIGHT));
    }

    #[test]
    fn test_pct_color_gradient() {
        let mut prev = pct_color(0.0);
        for i in 1..=100 {
            let curr = pct_color(i as f32);
            let dr = (i32::from(curr.0) - i32::from(prev.0)).abs();
            let dg = (i32::from(curr.1) - i32::from(prev.1)).abs();
            let db = (i32::from(curr.2) - i32::from(prev.2)).abs();
            assert!(dr < 50 && dg < 50 && db < 50, "Color jump at {i}%");
            prev = curr;
        }
    }

    #[test]
    fn test_sparkline() {
        let data = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        let spark = render_sparkline(&data, 5, ColorMode::Mono);
        assert!(!spark.is_empty());
    }

    #[test]
    fn test_trend_arrow() {
        let increasing = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        assert_eq!(trend_arrow(&increasing), ARROW_UP);

        let decreasing = vec![10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        assert_eq!(trend_arrow(&decreasing), ARROW_DOWN);
    }
}
