//! Input event types for testing.

use serde::{Deserialize, Serialize};

/// Touch input action
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TouchAction {
    /// Single tap
    Tap,
    /// Swipe gesture
    Swipe {
        /// End X coordinate
        end_x: f32,
        /// End Y coordinate
        end_y: f32,
        /// Duration in milliseconds
        duration_ms: u32,
    },
    /// Hold/long press
    Hold {
        /// Duration in milliseconds
        duration_ms: u32,
    },
}

/// Touch input for testing
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Touch {
    /// X coordinate
    pub x: f32,
    /// Y coordinate
    pub y: f32,
    /// Touch action type
    pub action: TouchAction,
}

impl Touch {
    /// Create a tap touch event
    #[must_use]
    pub const fn tap(x: f32, y: f32) -> Self {
        Self {
            x,
            y,
            action: TouchAction::Tap,
        }
    }

    /// Create a swipe touch event
    #[must_use]
    pub const fn swipe(
        start_x: f32,
        start_y: f32,
        end_x: f32,
        end_y: f32,
        duration_ms: u32,
    ) -> Self {
        Self {
            x: start_x,
            y: start_y,
            action: TouchAction::Swipe {
                end_x,
                end_y,
                duration_ms,
            },
        }
    }

    /// Create a hold touch event
    #[must_use]
    pub const fn hold(x: f32, y: f32, duration_ms: u32) -> Self {
        Self {
            x,
            y,
            action: TouchAction::Hold { duration_ms },
        }
    }
}

/// Input event types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum InputEvent {
    /// Touch event
    Touch {
        /// X coordinate
        x: f32,
        /// Y coordinate
        y: f32,
    },
    /// Key press event
    KeyPress {
        /// Key code
        key: String,
    },
    /// Key release event
    KeyRelease {
        /// Key code
        key: String,
    },
    /// Mouse click event
    MouseClick {
        /// X coordinate
        x: f32,
        /// Y coordinate
        y: f32,
    },
    /// Mouse move event
    MouseMove {
        /// X coordinate
        x: f32,
        /// Y coordinate
        y: f32,
    },
    /// Gamepad button event
    GamepadButton {
        /// Button index
        button: u8,
        /// Pressed state
        pressed: bool,
    },
}

impl InputEvent {
    /// Create a touch event
    #[must_use]
    pub const fn touch(x: f32, y: f32) -> Self {
        Self::Touch { x, y }
    }

    /// Create a key press event
    #[must_use]
    pub fn key_press(key: impl Into<String>) -> Self {
        Self::KeyPress { key: key.into() }
    }

    /// Create a key release event
    #[must_use]
    pub fn key_release(key: impl Into<String>) -> Self {
        Self::KeyRelease { key: key.into() }
    }

    /// Create a mouse click event
    #[must_use]
    pub const fn mouse_click(x: f32, y: f32) -> Self {
        Self::MouseClick { x, y }
    }

    /// Create a mouse move event
    #[must_use]
    pub const fn mouse_move(x: f32, y: f32) -> Self {
        Self::MouseMove { x, y }
    }

    /// Create a gamepad button event
    #[must_use]
    pub const fn gamepad_button(button: u8, pressed: bool) -> Self {
        Self::GamepadButton { button, pressed }
    }
}
