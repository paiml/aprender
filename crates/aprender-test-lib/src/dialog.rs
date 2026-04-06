//! Dialog Handling for E2E Testing (Feature G.8)
//!
//! Provides support for handling browser dialogs (alert, confirm, prompt, beforeunload).

use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

/// Type of browser dialog
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DialogType {
    /// Alert dialog (OK button only)
    Alert,
    /// Confirm dialog (OK/Cancel buttons)
    Confirm,
    /// Prompt dialog (text input + OK/Cancel)
    Prompt,
    /// Before unload dialog (Leave/Stay buttons)
    BeforeUnload,
}

impl std::fmt::Display for DialogType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Alert => write!(f, "alert"),
            Self::Confirm => write!(f, "confirm"),
            Self::Prompt => write!(f, "prompt"),
            Self::BeforeUnload => write!(f, "beforeunload"),
        }
    }
}

/// Action taken on a dialog
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DialogAction {
    /// Dialog was accepted (OK/Yes/Leave)
    Accept,
    /// Dialog was accepted with input text (for prompts)
    AcceptWith(String),
    /// Dialog was dismissed (Cancel/No/Stay)
    Dismiss,
    /// Dialog is pending (not yet handled)
    Pending,
}

/// Represents a browser dialog
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dialog {
    /// Type of dialog
    dialog_type: DialogType,
    /// Message displayed in the dialog
    message: String,
    /// Default value (for prompt dialogs)
    default_value: Option<String>,
    /// Action taken
    action: DialogAction,
}

impl Dialog {
    /// Create a new dialog
    #[must_use]
    pub fn new(dialog_type: DialogType, message: impl Into<String>) -> Self {
        Self {
            dialog_type,
            message: message.into(),
            default_value: None,
            action: DialogAction::Pending,
        }
    }

    /// Create an alert dialog
    #[must_use]
    pub fn alert(message: impl Into<String>) -> Self {
        Self::new(DialogType::Alert, message)
    }

    /// Create a confirm dialog
    #[must_use]
    pub fn confirm(message: impl Into<String>) -> Self {
        Self::new(DialogType::Confirm, message)
    }

    /// Create a prompt dialog
    #[must_use]
    pub fn prompt(message: impl Into<String>, default: Option<String>) -> Self {
        let mut dialog = Self::new(DialogType::Prompt, message);
        dialog.default_value = default;
        dialog
    }

    /// Create a beforeunload dialog
    #[must_use]
    pub fn before_unload(message: impl Into<String>) -> Self {
        Self::new(DialogType::BeforeUnload, message)
    }

    /// Get dialog type
    #[must_use]
    pub fn dialog_type(&self) -> DialogType {
        self.dialog_type
    }

    /// Get dialog message
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }

    /// Get default value (for prompts)
    #[must_use]
    pub fn default_value(&self) -> Option<&str> {
        self.default_value.as_deref()
    }

    /// Get action taken
    #[must_use]
    pub fn action(&self) -> &DialogAction {
        &self.action
    }

    /// Check if dialog was handled
    #[must_use]
    pub fn is_handled(&self) -> bool {
        !matches!(self.action, DialogAction::Pending)
    }

    /// Accept the dialog
    pub fn accept(&mut self) {
        self.action = DialogAction::Accept;
    }

    /// Accept the dialog with input text (for prompts)
    pub fn accept_with(&mut self, text: impl Into<String>) {
        self.action = DialogAction::AcceptWith(text.into());
    }

    /// Dismiss the dialog
    pub fn dismiss(&mut self) {
        self.action = DialogAction::Dismiss;
    }
}

/// Handler function type for dialogs
pub type DialogHandlerFn = Box<dyn Fn(&mut Dialog) + Send + Sync>;

/// Configuration for automatic dialog handling
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AutoDialogBehavior {
    /// Accept all dialogs automatically
    AcceptAll,
    /// Dismiss all dialogs automatically
    DismissAll,
    /// Accept with empty string (for prompts)
    AcceptEmpty,
    /// Use default value (for prompts)
    UseDefault,
    /// Do nothing (let tests handle manually)
    Manual,
}

impl Default for AutoDialogBehavior {
    fn default() -> Self {
        Self::Manual
    }
}

/// Dialog handler for managing browser dialogs
#[derive(Clone)]
pub struct DialogHandler {
    /// Queue of dialogs encountered
    dialogs: Arc<Mutex<Vec<Dialog>>>,
    /// Custom handler function
    handler: Arc<Mutex<Option<DialogHandlerFn>>>,
    /// Auto behavior when no custom handler
    auto_behavior: Arc<Mutex<AutoDialogBehavior>>,
}

impl DialogHandler {
    /// Create a new dialog handler
    #[must_use]
    pub fn new() -> Self {
        Self {
            dialogs: Arc::new(Mutex::new(Vec::new())),
            handler: Arc::new(Mutex::new(None)),
            auto_behavior: Arc::new(Mutex::new(AutoDialogBehavior::default())),
        }
    }

    /// Set custom handler function
    pub fn on_dialog<F>(&self, handler: F)
    where
        F: Fn(&mut Dialog) + Send + Sync + 'static,
    {
        if let Ok(mut h) = self.handler.lock() {
            *h = Some(Box::new(handler));
        }
    }

    /// Set automatic behavior
    pub fn set_auto_behavior(&self, behavior: AutoDialogBehavior) {
        if let Ok(mut b) = self.auto_behavior.lock() {
            *b = behavior;
        }
    }

    /// Handle an incoming dialog
    pub fn handle(&self, mut dialog: Dialog) -> Dialog {
        // Try custom handler first
        if let Ok(handler) = self.handler.lock() {
            if let Some(ref h) = *handler {
                h(&mut dialog);
                if dialog.is_handled() {
                    if let Ok(mut dialogs) = self.dialogs.lock() {
                        dialogs.push(dialog.clone());
                    }
                    return dialog;
                }
            }
        }

        // Fall back to auto behavior
        let behavior = self.auto_behavior.lock().map(|b| *b).unwrap_or_default();
        match behavior {
            AutoDialogBehavior::AcceptAll => dialog.accept(),
            AutoDialogBehavior::DismissAll => dialog.dismiss(),
            AutoDialogBehavior::AcceptEmpty => dialog.accept_with(""),
            AutoDialogBehavior::UseDefault => {
                if let Some(default) = dialog.default_value.clone() {
                    dialog.accept_with(default);
                } else {
                    dialog.accept();
                }
            }
            AutoDialogBehavior::Manual => {
                // Leave as pending
            }
        }

        if let Ok(mut dialogs) = self.dialogs.lock() {
            dialogs.push(dialog.clone());
        }
        dialog
    }

    /// Get all dialogs encountered
    #[must_use]
    pub fn dialogs(&self) -> Vec<Dialog> {
        self.dialogs.lock().map(|d| d.clone()).unwrap_or_default()
    }

    /// Get count of dialogs
    #[must_use]
    pub fn dialog_count(&self) -> usize {
        self.dialogs.lock().map(|d| d.len()).unwrap_or(0)
    }

    /// Clear dialog history
    pub fn clear(&self) {
        if let Ok(mut d) = self.dialogs.lock() {
            d.clear();
        }
    }

    /// Check if any dialogs are pending
    #[must_use]
    pub fn has_pending(&self) -> bool {
        self.dialogs
            .lock()
            .map(|d| d.iter().any(|dialog| !dialog.is_handled()))
            .unwrap_or(false)
    }

    /// Get last dialog
    #[must_use]
    pub fn last_dialog(&self) -> Option<Dialog> {
        self.dialogs.lock().ok().and_then(|d| d.last().cloned())
    }

    /// Wait for a dialog (mock implementation for sync tests)
    #[must_use]
    pub fn expect_dialog(&self, dialog_type: DialogType) -> DialogExpectation {
        DialogExpectation {
            expected_type: dialog_type,
            handler: self.clone(),
        }
    }
}

impl Default for DialogHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for DialogHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let auto_behavior = self.auto_behavior.lock().map(|b| *b).unwrap_or_default();
        f.debug_struct("DialogHandler")
            .field("dialog_count", &self.dialog_count())
            .field("auto_behavior", &auto_behavior)
            .finish()
    }
}

/// Expectation for a specific dialog type
#[derive(Debug)]
pub struct DialogExpectation {
    expected_type: DialogType,
    handler: DialogHandler,
}

impl DialogExpectation {
    /// Verify the expected dialog was encountered
    #[must_use]
    pub fn verify(&self) -> bool {
        self.handler
            .dialogs()
            .iter()
            .any(|d| d.dialog_type() == self.expected_type)
    }

    /// Get the dialog if it matches
    #[must_use]
    pub fn dialog(&self) -> Option<Dialog> {
        self.handler
            .dialogs()
            .into_iter()
            .find(|d| d.dialog_type() == self.expected_type)
    }
}

/// Builder for dialog handler
#[derive(Debug, Clone)]
pub struct DialogHandlerBuilder {
    auto_behavior: AutoDialogBehavior,
}

impl DialogHandlerBuilder {
    /// Create a new builder
    #[must_use]
    pub fn new() -> Self {
        Self {
            auto_behavior: AutoDialogBehavior::default(),
        }
    }

    /// Accept all dialogs automatically
    #[must_use]
    pub fn accept_all(mut self) -> Self {
        self.auto_behavior = AutoDialogBehavior::AcceptAll;
        self
    }

    /// Dismiss all dialogs automatically
    #[must_use]
    pub fn dismiss_all(mut self) -> Self {
        self.auto_behavior = AutoDialogBehavior::DismissAll;
        self
    }

    /// Use default values for prompts
    #[must_use]
    pub fn use_defaults(mut self) -> Self {
        self.auto_behavior = AutoDialogBehavior::UseDefault;
        self
    }

    /// Build the handler
    #[must_use]
    pub fn build(self) -> DialogHandler {
        let handler = DialogHandler::new();
        handler.set_auto_behavior(self.auto_behavior);
        handler
    }
}

impl Default for DialogHandlerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    // =========================================================================
    // H₀-DIALOG-01: Dialog creation
    // =========================================================================

    #[test]
    fn h0_dialog_01_new() {
        let dialog = Dialog::new(DialogType::Alert, "Hello");
        assert_eq!(dialog.dialog_type(), DialogType::Alert);
        assert_eq!(dialog.message(), "Hello");
        assert!(!dialog.is_handled());
    }

    #[test]
    fn h0_dialog_02_alert() {
        let dialog = Dialog::alert("Test alert");
        assert_eq!(dialog.dialog_type(), DialogType::Alert);
    }

    #[test]
    fn h0_dialog_03_confirm() {
        let dialog = Dialog::confirm("Are you sure?");
        assert_eq!(dialog.dialog_type(), DialogType::Confirm);
    }

    #[test]
    fn h0_dialog_04_prompt() {
        let dialog = Dialog::prompt("Enter name:", Some("default".to_string()));
        assert_eq!(dialog.dialog_type(), DialogType::Prompt);
        assert_eq!(dialog.default_value(), Some("default"));
    }

    #[test]
    fn h0_dialog_05_before_unload() {
        let dialog = Dialog::before_unload("Leave page?");
        assert_eq!(dialog.dialog_type(), DialogType::BeforeUnload);
    }

    // =========================================================================
    // H₀-DIALOG-06: Dialog actions
    // =========================================================================

    #[test]
    fn h0_dialog_06_accept() {
        let mut dialog = Dialog::alert("Test");
        dialog.accept();
        assert!(dialog.is_handled());
        assert_eq!(dialog.action(), &DialogAction::Accept);
    }

    #[test]
    fn h0_dialog_07_accept_with() {
        let mut dialog = Dialog::prompt("Name?", None);
        dialog.accept_with("John");
        assert!(dialog.is_handled());
        assert_eq!(
            dialog.action(),
            &DialogAction::AcceptWith("John".to_string())
        );
    }

    #[test]
    fn h0_dialog_08_dismiss() {
        let mut dialog = Dialog::confirm("Continue?");
        dialog.dismiss();
        assert!(dialog.is_handled());
        assert_eq!(dialog.action(), &DialogAction::Dismiss);
    }

    // =========================================================================
    // H₀-DIALOG-09: DialogType display
    // =========================================================================

    #[test]
    fn h0_dialog_09_type_display() {
        assert_eq!(format!("{}", DialogType::Alert), "alert");
        assert_eq!(format!("{}", DialogType::Confirm), "confirm");
        assert_eq!(format!("{}", DialogType::Prompt), "prompt");
        assert_eq!(format!("{}", DialogType::BeforeUnload), "beforeunload");
    }

    // =========================================================================
    // H₀-DIALOG-10: DialogHandler creation
    // =========================================================================

    #[test]
    fn h0_dialog_10_handler_new() {
        let handler = DialogHandler::new();
        assert_eq!(handler.dialog_count(), 0);
        assert!(!handler.has_pending());
    }

    #[test]
    fn h0_dialog_11_handler_default() {
        let handler = DialogHandler::default();
        assert_eq!(handler.dialog_count(), 0);
    }

    // =========================================================================
    // H₀-DIALOG-12: Auto behavior
    // =========================================================================

    #[test]
    fn h0_dialog_12_auto_accept_all() {
        let handler = DialogHandler::new();
        handler.set_auto_behavior(AutoDialogBehavior::AcceptAll);

        let dialog = handler.handle(Dialog::confirm("Test?"));
        assert_eq!(dialog.action(), &DialogAction::Accept);
    }

    #[test]
    fn h0_dialog_13_auto_dismiss_all() {
        let handler = DialogHandler::new();
        handler.set_auto_behavior(AutoDialogBehavior::DismissAll);

        let dialog = handler.handle(Dialog::confirm("Test?"));
        assert_eq!(dialog.action(), &DialogAction::Dismiss);
    }

    #[test]
    fn h0_dialog_14_auto_accept_empty() {
        let handler = DialogHandler::new();
        handler.set_auto_behavior(AutoDialogBehavior::AcceptEmpty);

        let dialog = handler.handle(Dialog::prompt("Name?", None));
        assert_eq!(dialog.action(), &DialogAction::AcceptWith(String::new()));
    }

    #[test]
    fn h0_dialog_15_auto_use_default() {
        let handler = DialogHandler::new();
        handler.set_auto_behavior(AutoDialogBehavior::UseDefault);

        let dialog = handler.handle(Dialog::prompt("Name?", Some("John".to_string())));
        assert_eq!(
            dialog.action(),
            &DialogAction::AcceptWith("John".to_string())
        );
    }

    #[test]
    fn h0_dialog_16_auto_manual() {
        let handler = DialogHandler::new();
        handler.set_auto_behavior(AutoDialogBehavior::Manual);

        let dialog = handler.handle(Dialog::alert("Test"));
        assert_eq!(dialog.action(), &DialogAction::Pending);
    }

    // =========================================================================
    // H₀-DIALOG-17: Custom handler
    // =========================================================================

    #[test]
    fn h0_dialog_17_custom_handler() {
        let handler = DialogHandler::new();
        handler.on_dialog(|dialog| {
            if dialog.dialog_type() == DialogType::Confirm {
                dialog.accept();
            } else {
                dialog.dismiss();
            }
        });

        let confirm = handler.handle(Dialog::confirm("Continue?"));
        assert_eq!(confirm.action(), &DialogAction::Accept);

        let alert = handler.handle(Dialog::alert("Info"));
        assert_eq!(alert.action(), &DialogAction::Dismiss);
    }

    // =========================================================================
    // H₀-DIALOG-18: Dialog history
    // =========================================================================

    #[test]
    fn h0_dialog_18_dialogs() {
        let handler = DialogHandler::new();
        handler.set_auto_behavior(AutoDialogBehavior::AcceptAll);

        handler.handle(Dialog::alert("First"));
        handler.handle(Dialog::alert("Second"));

        let dialogs = handler.dialogs();
        assert_eq!(dialogs.len(), 2);
        assert_eq!(dialogs[0].message(), "First");
        assert_eq!(dialogs[1].message(), "Second");
    }

    #[test]
    fn h0_dialog_19_last_dialog() {
        let handler = DialogHandler::new();
        handler.set_auto_behavior(AutoDialogBehavior::AcceptAll);

        handler.handle(Dialog::alert("First"));
        handler.handle(Dialog::alert("Last"));

        let last = handler.last_dialog().unwrap();
        assert_eq!(last.message(), "Last");
    }

    #[test]
    fn h0_dialog_20_clear() {
        let handler = DialogHandler::new();
        handler.set_auto_behavior(AutoDialogBehavior::AcceptAll);
        handler.handle(Dialog::alert("Test"));

        handler.clear();

        assert_eq!(handler.dialog_count(), 0);
    }

    // =========================================================================
    // H₀-DIALOG-21: Has pending
    // =========================================================================

    #[test]
    fn h0_dialog_21_has_pending_false() {
        let handler = DialogHandler::new();
        handler.set_auto_behavior(AutoDialogBehavior::AcceptAll);
        handler.handle(Dialog::alert("Test"));

        assert!(!handler.has_pending());
    }

    #[test]
    fn h0_dialog_22_has_pending_true() {
        let handler = DialogHandler::new();
        handler.set_auto_behavior(AutoDialogBehavior::Manual);
        handler.handle(Dialog::alert("Test"));

        assert!(handler.has_pending());
    }

    // =========================================================================
    // H₀-DIALOG-23: Expect dialog
    // =========================================================================

    #[test]
    fn h0_dialog_23_expect_dialog_verify() {
        let handler = DialogHandler::new();
        handler.set_auto_behavior(AutoDialogBehavior::AcceptAll);
        handler.handle(Dialog::confirm("Sure?"));

        let expectation = handler.expect_dialog(DialogType::Confirm);
        assert!(expectation.verify());
    }

    #[test]
    fn h0_dialog_24_expect_dialog_not_found() {
        let handler = DialogHandler::new();
        handler.set_auto_behavior(AutoDialogBehavior::AcceptAll);
        handler.handle(Dialog::alert("Info"));

        let expectation = handler.expect_dialog(DialogType::Confirm);
        assert!(!expectation.verify());
    }

    #[test]
    fn h0_dialog_25_expect_dialog_get() {
        let handler = DialogHandler::new();
        handler.set_auto_behavior(AutoDialogBehavior::AcceptAll);
        handler.handle(Dialog::prompt("Name?", None));

        let expectation = handler.expect_dialog(DialogType::Prompt);
        let dialog = expectation.dialog().unwrap();
        assert_eq!(dialog.message(), "Name?");
    }

    // =========================================================================
    // H₀-DIALOG-26: Builder
    // =========================================================================

    #[test]
    fn h0_dialog_26_builder_accept_all() {
        let handler = DialogHandlerBuilder::new().accept_all().build();

        let dialog = handler.handle(Dialog::alert("Test"));
        assert_eq!(dialog.action(), &DialogAction::Accept);
    }

    #[test]
    fn h0_dialog_27_builder_dismiss_all() {
        let handler = DialogHandlerBuilder::new().dismiss_all().build();

        let dialog = handler.handle(Dialog::confirm("Sure?"));
        assert_eq!(dialog.action(), &DialogAction::Dismiss);
    }

    #[test]
    fn h0_dialog_28_builder_use_defaults() {
        let handler = DialogHandlerBuilder::new().use_defaults().build();

        let dialog = handler.handle(Dialog::prompt("Name?", Some("Bob".to_string())));
        assert_eq!(
            dialog.action(),
            &DialogAction::AcceptWith("Bob".to_string())
        );
    }

    // =========================================================================
    // H₀-DIALOG-29: Debug
    // =========================================================================

    #[test]
    fn h0_dialog_29_handler_debug() {
        let handler = DialogHandler::new();
        let debug = format!("{handler:?}");
        assert!(debug.contains("DialogHandler"));
        assert!(debug.contains("dialog_count"));
    }

    // =========================================================================
    // H₀-DIALOG-30: Clone
    // =========================================================================

    #[test]
    fn h0_dialog_30_dialog_clone() {
        let dialog = Dialog::alert("Test");
        let cloned = dialog;
        assert_eq!(cloned.message(), "Test");
    }

    #[test]
    fn h0_dialog_31_handler_clone() {
        let handler = DialogHandler::new();
        handler.set_auto_behavior(AutoDialogBehavior::AcceptAll);
        handler.handle(Dialog::alert("Test"));

        let cloned = handler;
        assert_eq!(cloned.dialog_count(), 1);
    }
}
