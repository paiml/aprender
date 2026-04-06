#![allow(clippy::missing_const_for_fn)]
#![allow(clippy::match_same_arms)]
#![allow(clippy::unwrap_used)]
#![allow(clippy::disallowed_methods)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::struct_excessive_bools)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::should_implement_trait)]
#![allow(clippy::similar_names)]
#![allow(clippy::derive_partial_eq_without_eq)]
#![allow(clippy::map_unwrap_or)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::manual_let_else)]
#![allow(clippy::if_not_else)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::redundant_closure_for_method_calls)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::cloned_instead_of_copied)]
//! YAML manifest parser for Presentar applications.

mod error;
mod executor;
mod expression;
pub mod formats;
mod manifest;
pub mod pacha;
pub mod scene;

pub use error::ParseError;
pub use executor::{DataContext, ExecutionError, ExpressionExecutor, Value};
pub use expression::{Expression, ExpressionError, ExpressionParser, Transform};
pub use formats::{AldDataset, AprModel, DType, FormatError, ModelLayer, Tensor};
pub use manifest::{DataSource, Manifest, ModelRef, Section, WidgetConfig};
pub use pacha::{
    parse_refresh_interval, ContentType, LoadedResource, PachaError, PachaLoader, PachaUri,
    ResourceType,
};
pub use scene::{
    Binding, BindingAction, ContentItem, DatasetResource, DatasetType, FlexDirection,
    GlobalKeyBinding, GridPosition, HeaderContent, HeaderFooter, KeyBindings, KeySequence,
    LayoutType, ModelResource, ModelType, NavItem, Permissions, ResourceSource, Scene, SceneError,
    SceneLayout, SceneMetadata, SceneTheme, SceneWidget, Threshold,
    WidgetConfig as SceneWidgetConfig, WidgetType,
};
