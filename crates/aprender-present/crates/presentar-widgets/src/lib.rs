//! Widget implementations for Presentar UI framework.

#[macro_use]
#[allow(unused_macros)]
mod generated_contracts;

pub mod button;
pub mod chart;
pub mod checkbox;
pub mod column;
pub mod container;
pub mod data_card;
pub mod data_table;
pub mod formats;
pub mod image;
pub mod list;
pub mod menu;
pub mod modal;
pub mod model_card;
pub mod progress_bar;
pub mod radio_group;
pub mod row;
pub mod select;
pub mod slider;
pub mod stack;
pub mod tabs;
pub mod text;
pub mod text_input;
pub mod toggle;
pub mod tooltip;

pub use button::Button;
pub use chart::{Axis, Chart, ChartType, DataSeries, LegendPosition};
pub use checkbox::{CheckState, Checkbox, CheckboxChanged};
pub use column::Column;
pub use container::Container;
pub use data_card::{DataCard, DataColumn, DataQuality, DataStats};
pub use data_table::{
    CellValue, DataTable, SortDirection, TableColumn, TableRow, TableRowSelected, TableSortChanged,
    TextAlign,
};
pub use image::{Image, ImageFit};
pub use list::{
    List, ListDirection, ListItem, ListItemClicked, ListItemSelected, ListScrolled, SelectionMode,
};
pub use menu::{
    Menu, MenuCheckboxToggled, MenuClosed, MenuItem, MenuItemSelected, MenuToggled, MenuTrigger,
};
pub use modal::{BackdropBehavior, CloseReason, Modal, ModalClosed, ModalOpened, ModalSize};
pub use model_card::{ModelCard, ModelMetric, ModelStatus};
pub use progress_bar::{ProgressBar, ProgressMode};
pub use radio_group::{RadioChanged, RadioGroup, RadioOption, RadioOrientation};
pub use row::Row;
pub use select::{Select, SelectOption, SelectionChanged};
pub use slider::{Slider, SliderChanged};
pub use stack::{Stack, StackAlignment, StackFit};
pub use tabs::{Tab, TabChanged, TabOrientation, Tabs};
pub use text::Text;
pub use text_input::{TextChanged, TextInput, TextSubmitted};
pub use toggle::{Toggle, ToggleChanged};
pub use tooltip::{Tooltip, TooltipPlacement};

// File format conversions
pub use formats::{load_ald_as_card, load_apr_as_card, AldDatasetExt, AprModelExt};
