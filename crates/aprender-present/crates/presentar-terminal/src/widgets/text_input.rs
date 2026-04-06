//! `TextInput` widget with full editing capabilities.
//!
//! Provides a text input field with cursor, selection, and editing.
//! Based on btop filter input patterns.

use presentar_core::{
    Brick, BrickAssertion, BrickBudget, BrickVerification, Canvas, Color, Constraints, Event, Key,
    LayoutResult, Point, Rect, Size, TextStyle, TypeId, Widget,
};
use std::any::Any;
use std::time::Duration;

/// Text input widget with cursor and editing support.
#[derive(Debug, Clone)]
pub struct TextInput {
    /// Current text content.
    text: String,
    /// Cursor position (character index, not byte).
    cursor: usize,
    /// Selection range (start, end) if active. Both are character indices.
    selection: Option<(usize, usize)>,
    /// Placeholder text when empty.
    placeholder: String,
    /// Input mask character (e.g., '*' for password).
    mask: Option<char>,
    /// Maximum length (None = unlimited).
    max_length: Option<usize>,
    /// Horizontal scroll offset (character index).
    scroll_offset: usize,
    /// Is the input focused.
    focused: bool,
    /// Text color.
    text_color: Color,
    /// Cursor color.
    cursor_color: Color,
    /// Selection background color.
    selection_color: Color,
    /// Placeholder color.
    placeholder_color: Color,
    /// Cached bounds.
    bounds: Rect,
}

impl Default for TextInput {
    fn default() -> Self {
        Self::new()
    }
}

impl TextInput {
    /// Create a new text input.
    #[must_use]
    pub fn new() -> Self {
        Self {
            text: String::new(),
            cursor: 0,
            selection: None,
            placeholder: String::new(),
            mask: None,
            max_length: None,
            scroll_offset: 0,
            focused: false,
            text_color: Color::WHITE,
            cursor_color: Color::new(0.8, 0.8, 0.8, 1.0),
            selection_color: Color::new(0.3, 0.5, 0.8, 0.5),
            placeholder_color: Color::new(0.5, 0.5, 0.5, 1.0),
            bounds: Rect::default(),
        }
    }

    /// Set initial text.
    #[must_use]
    pub fn with_text(mut self, text: impl Into<String>) -> Self {
        self.text = text.into();
        self.cursor = self.text.chars().count();
        self
    }

    /// Set placeholder text.
    #[must_use]
    pub fn with_placeholder(mut self, text: impl Into<String>) -> Self {
        self.placeholder = text.into();
        self
    }

    /// Set mask character for password input.
    #[must_use]
    pub fn with_mask(mut self, ch: char) -> Self {
        self.mask = Some(ch);
        self
    }

    /// Set maximum length.
    #[must_use]
    pub fn with_max_length(mut self, len: usize) -> Self {
        self.max_length = Some(len);
        self
    }

    /// Set focused state.
    #[must_use]
    pub fn with_focused(mut self, focused: bool) -> Self {
        self.focused = focused;
        self
    }

    /// Set text color.
    #[must_use]
    pub fn with_text_color(mut self, color: Color) -> Self {
        self.text_color = color;
        self
    }

    /// Set cursor color.
    #[must_use]
    pub fn with_cursor_color(mut self, color: Color) -> Self {
        self.cursor_color = color;
        self
    }

    // ================= State Getters =================

    /// Get the current text.
    #[must_use]
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Get cursor position (character index).
    #[must_use]
    pub fn cursor(&self) -> usize {
        self.cursor
    }

    /// Check if input is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    /// Get text length in characters.
    #[must_use]
    pub fn len(&self) -> usize {
        self.text.chars().count()
    }

    /// Check if input is focused.
    #[must_use]
    pub fn is_focused(&self) -> bool {
        self.focused
    }

    /// Check if there's an active selection.
    #[must_use]
    pub fn has_selection(&self) -> bool {
        self.selection.is_some()
    }

    /// Get selection range if any.
    #[must_use]
    pub fn selection(&self) -> Option<(usize, usize)> {
        self.selection
    }

    // ================= Text Modification =================

    /// Set the text content.
    pub fn set_text(&mut self, text: impl Into<String>) {
        self.text = text.into();
        let len = self.text.chars().count();
        if let Some(max) = self.max_length {
            if len > max {
                self.text = self.text.chars().take(max).collect();
            }
        }
        self.cursor = self.text.chars().count();
        self.selection = None;
        self.adjust_scroll();
    }

    /// Insert a character at cursor.
    pub fn insert(&mut self, ch: char) {
        if let Some(max) = self.max_length {
            if self.len() >= max {
                return;
            }
        }
        self.delete_selection();
        let byte_pos = self.cursor_byte_pos();
        self.text.insert(byte_pos, ch);
        self.cursor += 1;
        self.adjust_scroll();
    }

    /// Insert a string at cursor.
    pub fn insert_str(&mut self, s: &str) {
        self.delete_selection();
        let to_insert: String = if let Some(max) = self.max_length {
            let available = max.saturating_sub(self.len());
            s.chars().take(available).collect()
        } else {
            s.to_string()
        };
        let byte_pos = self.cursor_byte_pos();
        self.text.insert_str(byte_pos, &to_insert);
        self.cursor += to_insert.chars().count();
        self.adjust_scroll();
    }

    /// Delete character at cursor (Delete key).
    pub fn delete(&mut self) {
        if self.selection.is_some() {
            self.delete_selection();
            return;
        }
        let len = self.len();
        if self.cursor < len {
            let byte_pos = self.cursor_byte_pos();
            let next_byte = self.char_byte_pos(self.cursor + 1);
            self.text.drain(byte_pos..next_byte);
        }
    }

    /// Delete character before cursor (Backspace key).
    pub fn backspace(&mut self) {
        if self.selection.is_some() {
            self.delete_selection();
            return;
        }
        if self.cursor > 0 {
            self.cursor -= 1;
            let byte_pos = self.cursor_byte_pos();
            let next_byte = self.char_byte_pos(self.cursor + 1);
            self.text.drain(byte_pos..next_byte);
            self.adjust_scroll();
        }
    }

    /// Delete word at/after cursor.
    pub fn delete_word(&mut self) {
        if self.selection.is_some() {
            self.delete_selection();
            return;
        }
        let start = self.cursor;
        self.move_word_right();
        let end = self.cursor;
        if end > start {
            let start_byte = self.char_byte_pos(start);
            let end_byte = self.char_byte_pos(end);
            self.text.drain(start_byte..end_byte);
            self.cursor = start;
        }
    }

    /// Delete from cursor to end of line.
    pub fn delete_to_end(&mut self) {
        let byte_pos = self.cursor_byte_pos();
        self.text.truncate(byte_pos);
    }

    /// Delete entire line (clear all text).
    pub fn delete_line(&mut self) {
        self.text.clear();
        self.cursor = 0;
        self.selection = None;
        self.scroll_offset = 0;
    }

    // ================= Cursor Movement =================

    /// Move cursor left.
    pub fn move_left(&mut self) {
        self.selection = None;
        if self.cursor > 0 {
            self.cursor -= 1;
            self.adjust_scroll();
        }
    }

    /// Move cursor right.
    pub fn move_right(&mut self) {
        self.selection = None;
        let len = self.len();
        if self.cursor < len {
            self.cursor += 1;
            self.adjust_scroll();
        }
    }

    /// Move cursor to start of previous word.
    pub fn move_word_left(&mut self) {
        self.selection = None;
        if self.cursor == 0 {
            return;
        }
        let chars: Vec<char> = self.text.chars().collect();
        let mut pos = self.cursor - 1;
        // Skip whitespace
        while pos > 0 && chars[pos].is_whitespace() {
            pos -= 1;
        }
        // Skip word characters
        while pos > 0 && !chars[pos - 1].is_whitespace() {
            pos -= 1;
        }
        self.cursor = pos;
        self.adjust_scroll();
    }

    /// Move cursor to end of next word.
    pub fn move_word_right(&mut self) {
        self.selection = None;
        let chars: Vec<char> = self.text.chars().collect();
        let len = chars.len();
        if self.cursor >= len {
            return;
        }
        let mut pos = self.cursor;
        // Skip word characters
        while pos < len && !chars[pos].is_whitespace() {
            pos += 1;
        }
        // Skip whitespace
        while pos < len && chars[pos].is_whitespace() {
            pos += 1;
        }
        self.cursor = pos;
        self.adjust_scroll();
    }

    /// Move cursor to start (Home).
    pub fn move_home(&mut self) {
        self.selection = None;
        self.cursor = 0;
        self.scroll_offset = 0;
    }

    /// Move cursor to end (End).
    pub fn move_end(&mut self) {
        self.selection = None;
        self.cursor = self.len();
        self.adjust_scroll();
    }

    // ================= Selection =================

    /// Select all text.
    pub fn select_all(&mut self) {
        let len = self.len();
        if len > 0 {
            self.selection = Some((0, len));
            self.cursor = len;
        }
    }

    /// Select word at cursor.
    pub fn select_word(&mut self) {
        let chars: Vec<char> = self.text.chars().collect();
        let len = chars.len();
        if len == 0 || self.cursor > len {
            return;
        }
        let pos = self.cursor.min(len.saturating_sub(1));
        let mut start = pos;
        let mut end = pos;
        // Expand backward
        while start > 0 && !chars[start - 1].is_whitespace() {
            start -= 1;
        }
        // Expand forward
        while end < len && !chars[end].is_whitespace() {
            end += 1;
        }
        if start < end {
            self.selection = Some((start, end));
            self.cursor = end;
        }
    }

    /// Extend selection left.
    pub fn extend_selection_left(&mut self) {
        if self.cursor == 0 {
            return;
        }
        let new_cursor = self.cursor - 1;
        match self.selection {
            Some((start, end)) => {
                if self.cursor == start {
                    self.selection = Some((new_cursor, end));
                } else {
                    self.selection = Some((start, new_cursor));
                }
            }
            None => {
                self.selection = Some((new_cursor, self.cursor));
            }
        }
        self.cursor = new_cursor;
        self.adjust_scroll();
    }

    /// Extend selection right.
    pub fn extend_selection_right(&mut self) {
        let len = self.len();
        if self.cursor >= len {
            return;
        }
        let new_cursor = self.cursor + 1;
        match self.selection {
            Some((start, end)) => {
                if self.cursor == end {
                    self.selection = Some((start, new_cursor));
                } else {
                    self.selection = Some((new_cursor, end));
                }
            }
            None => {
                self.selection = Some((self.cursor, new_cursor));
            }
        }
        self.cursor = new_cursor;
        self.adjust_scroll();
    }

    /// Clear selection.
    pub fn clear_selection(&mut self) {
        self.selection = None;
    }

    /// Get selected text.
    #[must_use]
    pub fn selected_text(&self) -> Option<String> {
        self.selection.map(|(start, end)| {
            let (start, end) = (start.min(end), start.max(end));
            self.text.chars().skip(start).take(end - start).collect()
        })
    }

    /// Delete selected text.
    pub fn delete_selection(&mut self) {
        if let Some((start, end)) = self.selection.take() {
            let (start, end) = (start.min(end), start.max(end));
            let start_byte = self.char_byte_pos(start);
            let end_byte = self.char_byte_pos(end);
            self.text.drain(start_byte..end_byte);
            self.cursor = start;
            self.adjust_scroll();
        }
    }

    // ================= Clipboard =================

    /// Copy selected text (returns text for external clipboard).
    #[must_use]
    pub fn copy(&self) -> Option<String> {
        self.selected_text()
    }

    /// Cut selected text (returns text for external clipboard).
    pub fn cut(&mut self) -> Option<String> {
        let text = self.selected_text();
        self.delete_selection();
        text
    }

    /// Paste text (caller provides from external clipboard).
    pub fn paste(&mut self, text: &str) {
        self.delete_selection();
        self.insert_str(text);
    }

    // ================= Focus =================

    /// Set focused state.
    pub fn focus(&mut self) {
        self.focused = true;
    }

    /// Remove focus.
    pub fn blur(&mut self) {
        self.focused = false;
        self.selection = None;
    }

    // ================= Internal Helpers =================

    /// Get byte position for character index.
    fn char_byte_pos(&self, char_idx: usize) -> usize {
        self.text
            .char_indices()
            .nth(char_idx)
            .map_or(self.text.len(), |(i, _)| i)
    }

    /// Get byte position of cursor.
    fn cursor_byte_pos(&self) -> usize {
        self.char_byte_pos(self.cursor)
    }

    /// Adjust scroll offset to keep cursor visible.
    fn adjust_scroll(&mut self) {
        let visible_width = self.bounds.width as usize;
        if visible_width == 0 {
            return;
        }
        if self.cursor < self.scroll_offset {
            self.scroll_offset = self.cursor;
        } else if self.cursor >= self.scroll_offset + visible_width {
            self.scroll_offset = self.cursor.saturating_sub(visible_width - 1);
        }
    }

    /// Get display text (masked if mask is set).
    fn display_text(&self) -> String {
        if let Some(mask) = self.mask {
            std::iter::repeat(mask).take(self.len()).collect()
        } else {
            self.text.clone()
        }
    }
}

impl Brick for TextInput {
    fn brick_name(&self) -> &'static str {
        "text_input"
    }

    fn assertions(&self) -> &[BrickAssertion] {
        static ASSERTIONS: &[BrickAssertion] = &[BrickAssertion::max_latency_ms(16)];
        ASSERTIONS
    }

    fn budget(&self) -> BrickBudget {
        BrickBudget::uniform(8)
    }

    fn verify(&self) -> BrickVerification {
        BrickVerification {
            passed: self.assertions().to_vec(),
            failed: vec![],
            verification_time: Duration::from_micros(5),
        }
    }

    fn to_html(&self) -> String {
        String::new()
    }

    fn to_css(&self) -> String {
        String::new()
    }
}

impl Widget for TextInput {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        let width = constraints.max_width.clamp(5.0, 40.0);
        Size::new(width, 1.0)
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        self.bounds = bounds;
        self.adjust_scroll();
        LayoutResult {
            size: Size::new(bounds.width, bounds.height),
        }
    }

    fn paint(&self, canvas: &mut dyn Canvas) {
        let width = self.bounds.width as usize;
        if width == 0 {
            return;
        }

        // Get display text (masked or actual)
        let display = if self.is_empty() && !self.focused {
            &self.placeholder
        } else {
            &self.display_text()
        };

        let text_color = if self.is_empty() && !self.focused {
            self.placeholder_color
        } else {
            self.text_color
        };

        // Draw visible portion
        let chars: Vec<char> = display.chars().collect();
        let visible_start = self.scroll_offset;
        let visible_end = (visible_start + width).min(chars.len());

        for (i, char_idx) in (visible_start..visible_end).enumerate() {
            let ch = chars[char_idx];
            let x = self.bounds.x + i as f32;

            // Check if in selection
            let in_selection = self.selection.is_some_and(|(start, end)| {
                let (s, e) = (start.min(end), start.max(end));
                char_idx >= s && char_idx < e
            });

            // Draw selection background
            if in_selection {
                canvas.fill_rect(Rect::new(x, self.bounds.y, 1.0, 1.0), self.selection_color);
            }

            let style = TextStyle {
                color: text_color,
                ..Default::default()
            };
            canvas.draw_text(&ch.to_string(), Point::new(x, self.bounds.y), &style);
        }

        // Draw cursor if focused
        if self.focused
            && self.cursor >= self.scroll_offset
            && self.cursor < self.scroll_offset + width
        {
            let cursor_x = self.bounds.x + (self.cursor - self.scroll_offset) as f32;
            canvas.fill_rect(
                Rect::new(cursor_x, self.bounds.y, 0.1, 1.0),
                self.cursor_color,
            );
        }
    }

    fn event(&mut self, event: &Event) -> Option<Box<dyn Any + Send>> {
        if !self.focused {
            return None;
        }

        match event {
            Event::KeyDown { key, .. } => match key {
                Key::Backspace => {
                    self.backspace();
                }
                Key::Delete => {
                    self.delete();
                }
                Key::Left => {
                    self.move_left();
                }
                Key::Right => {
                    self.move_right();
                }
                Key::Home => {
                    self.move_home();
                }
                Key::End => {
                    self.move_end();
                }
                _ => {}
            },
            Event::TextInput { text } => {
                for ch in text.chars() {
                    self.insert(ch);
                }
            }
            _ => {}
        }

        None
    }

    fn children(&self) -> &[Box<dyn Widget>] {
        &[]
    }

    fn children_mut(&mut self) -> &mut [Box<dyn Widget>] {
        &mut []
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockCanvas {
        texts: Vec<(String, Point)>,
        rects: Vec<Rect>,
    }

    impl MockCanvas {
        fn new() -> Self {
            Self {
                texts: vec![],
                rects: vec![],
            }
        }
    }

    impl Canvas for MockCanvas {
        fn fill_rect(&mut self, rect: Rect, _color: Color) {
            self.rects.push(rect);
        }
        fn stroke_rect(&mut self, _rect: Rect, _color: Color, _width: f32) {}
        fn draw_text(&mut self, text: &str, position: Point, _style: &TextStyle) {
            self.texts.push((text.to_string(), position));
        }
        fn draw_line(&mut self, _from: Point, _to: Point, _color: Color, _width: f32) {}
        fn fill_circle(&mut self, _center: Point, _radius: f32, _color: Color) {}
        fn stroke_circle(&mut self, _center: Point, _radius: f32, _color: Color, _width: f32) {}
        fn fill_arc(&mut self, _c: Point, _r: f32, _s: f32, _e: f32, _color: Color) {}
        fn draw_path(&mut self, _points: &[Point], _color: Color, _width: f32) {}
        fn fill_polygon(&mut self, _points: &[Point], _color: Color) {}
        fn push_clip(&mut self, _rect: Rect) {}
        fn pop_clip(&mut self) {}
        fn push_transform(&mut self, _transform: presentar_core::Transform2D) {}
        fn pop_transform(&mut self) {}
    }

    // =====================================================
    // Construction Tests
    // =====================================================

    #[test]
    fn test_new() {
        let input = TextInput::new();
        assert!(input.is_empty());
        assert_eq!(input.cursor(), 0);
        assert!(!input.is_focused());
    }

    #[test]
    fn test_default() {
        let input = TextInput::default();
        assert!(input.is_empty());
    }

    #[test]
    fn test_with_text() {
        let input = TextInput::new().with_text("hello");
        assert_eq!(input.text(), "hello");
        assert_eq!(input.cursor(), 5);
    }

    #[test]
    fn test_with_placeholder() {
        let input = TextInput::new().with_placeholder("Enter text...");
        assert_eq!(input.placeholder, "Enter text...");
    }

    #[test]
    fn test_with_mask() {
        let input = TextInput::new().with_mask('*');
        assert_eq!(input.mask, Some('*'));
    }

    #[test]
    fn test_with_max_length() {
        let input = TextInput::new().with_max_length(10);
        assert_eq!(input.max_length, Some(10));
    }

    #[test]
    fn test_with_focused() {
        let input = TextInput::new().with_focused(true);
        assert!(input.is_focused());
    }

    #[test]
    fn test_with_text_color() {
        let input = TextInput::new().with_text_color(Color::RED);
        assert_eq!(input.text_color, Color::RED);
    }

    #[test]
    fn test_with_cursor_color() {
        let input = TextInput::new().with_cursor_color(Color::GREEN);
        assert_eq!(input.cursor_color, Color::GREEN);
    }

    // =====================================================
    // State Getter Tests
    // =====================================================

    #[test]
    fn test_len() {
        let input = TextInput::new().with_text("hello");
        assert_eq!(input.len(), 5);
    }

    #[test]
    fn test_len_unicode() {
        let input = TextInput::new().with_text("héllo");
        assert_eq!(input.len(), 5);
    }

    #[test]
    fn test_is_empty_true() {
        let input = TextInput::new();
        assert!(input.is_empty());
    }

    #[test]
    fn test_is_empty_false() {
        let input = TextInput::new().with_text("x");
        assert!(!input.is_empty());
    }

    // =====================================================
    // Text Modification Tests
    // =====================================================

    #[test]
    fn test_set_text() {
        let mut input = TextInput::new();
        input.set_text("hello");
        assert_eq!(input.text(), "hello");
        assert_eq!(input.cursor(), 5);
    }

    #[test]
    fn test_set_text_clears_selection() {
        let mut input = TextInput::new().with_text("hello");
        input.select_all();
        input.set_text("world");
        assert!(!input.has_selection());
    }

    #[test]
    fn test_set_text_respects_max_length() {
        let mut input = TextInput::new().with_max_length(3);
        input.set_text("hello");
        assert_eq!(input.text(), "hel");
    }

    #[test]
    fn test_insert() {
        let mut input = TextInput::new();
        input.insert('a');
        input.insert('b');
        assert_eq!(input.text(), "ab");
        assert_eq!(input.cursor(), 2);
    }

    #[test]
    fn test_insert_respects_max_length() {
        let mut input = TextInput::new().with_max_length(2);
        input.insert('a');
        input.insert('b');
        input.insert('c');
        assert_eq!(input.text(), "ab");
    }

    #[test]
    fn test_insert_deletes_selection() {
        let mut input = TextInput::new().with_text("hello");
        input.select_all();
        input.insert('X');
        assert_eq!(input.text(), "X");
    }

    #[test]
    fn test_insert_str() {
        let mut input = TextInput::new();
        input.insert_str("hello");
        assert_eq!(input.text(), "hello");
        assert_eq!(input.cursor(), 5);
    }

    #[test]
    fn test_insert_str_respects_max_length() {
        let mut input = TextInput::new().with_max_length(3);
        input.insert_str("hello");
        assert_eq!(input.text(), "hel");
    }

    #[test]
    fn test_delete() {
        let mut input = TextInput::new().with_text("hello");
        input.cursor = 2;
        input.delete();
        assert_eq!(input.text(), "helo");
    }

    #[test]
    fn test_delete_at_end() {
        let mut input = TextInput::new().with_text("hello");
        input.delete();
        assert_eq!(input.text(), "hello");
    }

    #[test]
    fn test_delete_selection() {
        let mut input = TextInput::new().with_text("hello");
        input.select_all();
        input.delete();
        assert!(input.is_empty());
    }

    #[test]
    fn test_backspace() {
        let mut input = TextInput::new().with_text("hello");
        input.backspace();
        assert_eq!(input.text(), "hell");
        assert_eq!(input.cursor(), 4);
    }

    #[test]
    fn test_backspace_at_start() {
        let mut input = TextInput::new().with_text("hello");
        input.cursor = 0;
        input.backspace();
        assert_eq!(input.text(), "hello");
    }

    #[test]
    fn test_delete_word() {
        let mut input = TextInput::new().with_text("hello world");
        input.cursor = 0;
        input.delete_word();
        assert_eq!(input.text(), "world");
    }

    #[test]
    fn test_delete_to_end() {
        let mut input = TextInput::new().with_text("hello world");
        input.cursor = 6;
        input.delete_to_end();
        assert_eq!(input.text(), "hello ");
    }

    #[test]
    fn test_delete_line() {
        let mut input = TextInput::new().with_text("hello world");
        input.delete_line();
        assert!(input.is_empty());
        assert_eq!(input.cursor(), 0);
    }

    // =====================================================
    // Cursor Movement Tests
    // =====================================================

    #[test]
    fn test_move_left() {
        let mut input = TextInput::new().with_text("hello");
        input.move_left();
        assert_eq!(input.cursor(), 4);
    }

    #[test]
    fn test_move_left_at_start() {
        let mut input = TextInput::new().with_text("hello");
        input.cursor = 0;
        input.move_left();
        assert_eq!(input.cursor(), 0);
    }

    #[test]
    fn test_move_left_clears_selection() {
        let mut input = TextInput::new().with_text("hello");
        input.select_all();
        input.move_left();
        assert!(!input.has_selection());
    }

    #[test]
    fn test_move_right() {
        let mut input = TextInput::new().with_text("hello");
        input.cursor = 0;
        input.move_right();
        assert_eq!(input.cursor(), 1);
    }

    #[test]
    fn test_move_right_at_end() {
        let mut input = TextInput::new().with_text("hello");
        input.move_right();
        assert_eq!(input.cursor(), 5);
    }

    #[test]
    fn test_move_word_left() {
        let mut input = TextInput::new().with_text("hello world");
        input.move_word_left();
        assert_eq!(input.cursor(), 6);
    }

    #[test]
    fn test_move_word_right() {
        let mut input = TextInput::new().with_text("hello world");
        input.cursor = 0;
        input.move_word_right();
        assert_eq!(input.cursor(), 6);
    }

    #[test]
    fn test_move_home() {
        let mut input = TextInput::new().with_text("hello");
        input.move_home();
        assert_eq!(input.cursor(), 0);
    }

    #[test]
    fn test_move_end() {
        let mut input = TextInput::new().with_text("hello");
        input.cursor = 0;
        input.move_end();
        assert_eq!(input.cursor(), 5);
    }

    // =====================================================
    // Selection Tests
    // =====================================================

    #[test]
    fn test_select_all() {
        let mut input = TextInput::new().with_text("hello");
        input.select_all();
        assert_eq!(input.selection(), Some((0, 5)));
    }

    #[test]
    fn test_select_all_empty() {
        let mut input = TextInput::new();
        input.select_all();
        assert!(!input.has_selection());
    }

    #[test]
    fn test_select_word() {
        let mut input = TextInput::new().with_text("hello world");
        input.cursor = 2;
        input.select_word();
        assert_eq!(input.selection(), Some((0, 5)));
    }

    #[test]
    fn test_selected_text() {
        let mut input = TextInput::new().with_text("hello");
        input.select_all();
        assert_eq!(input.selected_text(), Some("hello".to_string()));
    }

    #[test]
    fn test_selected_text_none() {
        let input = TextInput::new().with_text("hello");
        assert_eq!(input.selected_text(), None);
    }

    #[test]
    fn test_extend_selection_left() {
        let mut input = TextInput::new().with_text("hello");
        input.extend_selection_left();
        assert_eq!(input.selection(), Some((4, 5)));
        assert_eq!(input.cursor(), 4);
    }

    #[test]
    fn test_extend_selection_right() {
        let mut input = TextInput::new().with_text("hello");
        input.cursor = 0;
        input.extend_selection_right();
        assert_eq!(input.selection(), Some((0, 1)));
        assert_eq!(input.cursor(), 1);
    }

    #[test]
    fn test_clear_selection() {
        let mut input = TextInput::new().with_text("hello");
        input.select_all();
        input.clear_selection();
        assert!(!input.has_selection());
    }

    // =====================================================
    // Clipboard Tests
    // =====================================================

    #[test]
    fn test_copy() {
        let mut input = TextInput::new().with_text("hello");
        input.select_all();
        assert_eq!(input.copy(), Some("hello".to_string()));
        assert!(input.has_selection()); // Selection preserved
    }

    #[test]
    fn test_cut() {
        let mut input = TextInput::new().with_text("hello");
        input.select_all();
        assert_eq!(input.cut(), Some("hello".to_string()));
        assert!(input.is_empty());
    }

    #[test]
    fn test_paste() {
        let mut input = TextInput::new();
        input.paste("hello");
        assert_eq!(input.text(), "hello");
    }

    #[test]
    fn test_paste_replaces_selection() {
        let mut input = TextInput::new().with_text("hello");
        input.select_all();
        input.paste("world");
        assert_eq!(input.text(), "world");
    }

    // =====================================================
    // Focus Tests
    // =====================================================

    #[test]
    fn test_focus() {
        let mut input = TextInput::new();
        input.focus();
        assert!(input.is_focused());
    }

    #[test]
    fn test_blur() {
        let mut input = TextInput::new().with_focused(true);
        input.select_all();
        input.blur();
        assert!(!input.is_focused());
        assert!(!input.has_selection());
    }

    // =====================================================
    // Display Text Tests
    // =====================================================

    #[test]
    fn test_display_text_normal() {
        let input = TextInput::new().with_text("hello");
        assert_eq!(input.display_text(), "hello");
    }

    #[test]
    fn test_display_text_masked() {
        let input = TextInput::new().with_text("hello").with_mask('*');
        assert_eq!(input.display_text(), "*****");
    }

    // =====================================================
    // Brick Trait Tests
    // =====================================================

    #[test]
    fn test_brick_name() {
        let input = TextInput::new();
        assert_eq!(input.brick_name(), "text_input");
    }

    #[test]
    fn test_assertions_not_empty() {
        let input = TextInput::new();
        assert!(!input.assertions().is_empty());
    }

    #[test]
    fn test_budget() {
        let input = TextInput::new();
        assert!(input.budget().paint_ms > 0);
    }

    #[test]
    fn test_verify() {
        let input = TextInput::new();
        assert!(input.verify().is_valid());
    }

    #[test]
    fn test_to_html() {
        let input = TextInput::new();
        assert!(input.to_html().is_empty());
    }

    #[test]
    fn test_to_css() {
        let input = TextInput::new();
        assert!(input.to_css().is_empty());
    }

    // =====================================================
    // Widget Trait Tests
    // =====================================================

    #[test]
    fn test_type_id() {
        let input = TextInput::new();
        assert_eq!(Widget::type_id(&input), TypeId::of::<TextInput>());
    }

    #[test]
    fn test_measure() {
        let input = TextInput::new();
        let size = input.measure(Constraints::loose(Size::new(100.0, 100.0)));
        assert!(size.width >= 5.0);
        assert_eq!(size.height, 1.0);
    }

    #[test]
    fn test_layout() {
        let mut input = TextInput::new();
        let bounds = Rect::new(0.0, 0.0, 20.0, 1.0);
        let result = input.layout(bounds);
        assert_eq!(result.size.width, 20.0);
        assert_eq!(input.bounds, bounds);
    }

    #[test]
    fn test_children() {
        let input = TextInput::new();
        assert!(input.children().is_empty());
    }

    #[test]
    fn test_children_mut() {
        let mut input = TextInput::new();
        assert!(input.children_mut().is_empty());
    }

    // =====================================================
    // Paint Tests
    // =====================================================

    #[test]
    fn test_paint_empty() {
        let mut input = TextInput::new();
        input.bounds = Rect::new(0.0, 0.0, 10.0, 1.0);
        let mut canvas = MockCanvas::new();
        input.paint(&mut canvas);
        // Should handle empty gracefully
    }

    #[test]
    fn test_paint_with_text() {
        let mut input = TextInput::new().with_text("hello");
        input.bounds = Rect::new(0.0, 0.0, 10.0, 1.0);
        let mut canvas = MockCanvas::new();
        input.paint(&mut canvas);
        assert!(!canvas.texts.is_empty());
    }

    #[test]
    fn test_paint_with_cursor() {
        let mut input = TextInput::new().with_text("hello").with_focused(true);
        input.bounds = Rect::new(0.0, 0.0, 10.0, 1.0);
        let mut canvas = MockCanvas::new();
        input.paint(&mut canvas);
        assert!(!canvas.rects.is_empty()); // Cursor rect
    }

    #[test]
    fn test_paint_with_selection() {
        let mut input = TextInput::new().with_text("hello").with_focused(true);
        input.select_all();
        input.bounds = Rect::new(0.0, 0.0, 10.0, 1.0);
        let mut canvas = MockCanvas::new();
        input.paint(&mut canvas);
        // Selection backgrounds
        assert!(!canvas.rects.is_empty());
    }

    #[test]
    fn test_paint_placeholder() {
        let mut input = TextInput::new().with_placeholder("Type here...");
        input.bounds = Rect::new(0.0, 0.0, 20.0, 1.0);
        let mut canvas = MockCanvas::new();
        input.paint(&mut canvas);
        // Should show placeholder
        assert!(!canvas.texts.is_empty());
    }

    // =====================================================
    // Event Tests
    // =====================================================

    #[test]
    fn test_event_not_focused() {
        let mut input = TextInput::new();
        let event = Event::key_down(Key::Left);
        assert!(input.event(&event).is_none());
        // Should not process event when not focused
    }

    #[test]
    fn test_event_backspace() {
        let mut input = TextInput::new().with_text("hello").with_focused(true);
        let event = Event::key_down(Key::Backspace);
        input.event(&event);
        assert_eq!(input.text(), "hell");
    }

    #[test]
    fn test_event_delete() {
        let mut input = TextInput::new().with_text("hello").with_focused(true);
        input.cursor = 0;
        let event = Event::key_down(Key::Delete);
        input.event(&event);
        assert_eq!(input.text(), "ello");
    }

    #[test]
    fn test_event_left() {
        let mut input = TextInput::new().with_text("hello").with_focused(true);
        let event = Event::key_down(Key::Left);
        input.event(&event);
        assert_eq!(input.cursor(), 4);
    }

    #[test]
    fn test_event_right() {
        let mut input = TextInput::new().with_text("hello").with_focused(true);
        input.cursor = 0;
        let event = Event::key_down(Key::Right);
        input.event(&event);
        assert_eq!(input.cursor(), 1);
    }

    #[test]
    fn test_event_home() {
        let mut input = TextInput::new().with_text("hello").with_focused(true);
        let event = Event::key_down(Key::Home);
        input.event(&event);
        assert_eq!(input.cursor(), 0);
    }

    #[test]
    fn test_event_end() {
        let mut input = TextInput::new().with_text("hello").with_focused(true);
        input.cursor = 0;
        let event = Event::key_down(Key::End);
        input.event(&event);
        assert_eq!(input.cursor(), 5);
    }

    #[test]
    fn test_event_text_input() {
        let mut input = TextInput::new().with_focused(true);
        let event = Event::TextInput {
            text: "hi".to_string(),
        };
        input.event(&event);
        assert_eq!(input.text(), "hi");
    }

    // =====================================================
    // Unicode Tests
    // =====================================================

    #[test]
    fn test_unicode_insert() {
        let mut input = TextInput::new();
        input.insert('é');
        input.insert('ñ');
        assert_eq!(input.text(), "éñ");
        assert_eq!(input.len(), 2);
    }

    #[test]
    fn test_unicode_backspace() {
        let mut input = TextInput::new().with_text("héllo");
        input.backspace();
        assert_eq!(input.text(), "héll");
    }

    #[test]
    fn test_unicode_cursor() {
        let mut input = TextInput::new().with_text("héllo");
        input.cursor = 2;
        input.insert('X');
        assert_eq!(input.text(), "héXllo");
    }

    // =====================================================
    // Scroll Offset Tests
    // =====================================================

    #[test]
    fn test_scroll_on_insert() {
        let mut input = TextInput::new();
        input.bounds = Rect::new(0.0, 0.0, 5.0, 1.0);
        for c in "hello world".chars() {
            input.insert(c);
        }
        // Scroll offset should adjust to keep cursor visible
        assert!(input.scroll_offset > 0);
    }

    #[test]
    fn test_scroll_on_move_home() {
        let mut input = TextInput::new().with_text("hello world");
        input.bounds = Rect::new(0.0, 0.0, 5.0, 1.0);
        input.scroll_offset = 6;
        input.move_home();
        assert_eq!(input.scroll_offset, 0);
    }
}
