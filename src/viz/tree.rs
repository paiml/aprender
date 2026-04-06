//! Visualization Frameworks Tree
//!
//! Hierarchical view of Python visualization frameworks and their PAIML replacements.

use serde::{Deserialize, Serialize};

// ============================================================================
// Core Types
// ============================================================================

/// Visualization framework identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Framework {
    Gradio,
    Streamlit,
    Panel,
    Dash,
}

impl Framework {
    /// Get framework display name
    pub fn name(&self) -> &'static str {
        match self {
            Framework::Gradio => "Gradio",
            Framework::Streamlit => "Streamlit",
            Framework::Panel => "Panel",
            Framework::Dash => "Dash",
        }
    }

    /// Get PAIML replacement
    pub fn replacement(&self) -> &'static str {
        match self {
            Framework::Gradio => "Presentar",
            Framework::Streamlit => "Presentar",
            Framework::Panel => "Trueno-Viz",
            Framework::Dash => "Presentar + Trueno-Viz",
        }
    }

    /// Get all frameworks
    pub fn all() -> Vec<Self> {
        vec![Framework::Gradio, Framework::Streamlit, Framework::Panel, Framework::Dash]
    }
}

impl std::fmt::Display for Framework {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Integration type for component mappings
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IntegrationType {
    /// PAIML fully replaces Python component
    Replaces,
    /// Depyler transpiles Python to Rust
    Transpiles,
    /// Can consume output format (PNG, SVG, etc.)
    Compatible,
}

impl IntegrationType {
    /// Get short code for display
    pub fn code(&self) -> &'static str {
        match self {
            IntegrationType::Replaces => "REP",
            IntegrationType::Transpiles => "TRN",
            IntegrationType::Compatible => "CMP",
        }
    }
}

impl std::fmt::Display for IntegrationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.code())
    }
}

// ============================================================================
// Tree Structures
// ============================================================================

/// A component within a framework category
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameworkComponent {
    pub name: String,
    pub description: String,
    pub replacement: String,
    pub sub_components: Vec<String>,
}

impl FrameworkComponent {
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        replacement: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            replacement: replacement.into(),
            sub_components: Vec::new(),
        }
    }

    pub fn with_sub(mut self, sub: impl Into<String>) -> Self {
        self.sub_components.push(sub.into());
        self
    }
}

/// A category within a framework
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameworkCategory {
    pub name: String,
    pub components: Vec<FrameworkComponent>,
}

impl FrameworkCategory {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into(), components: Vec::new() }
    }

    pub fn with_component(mut self, component: FrameworkComponent) -> Self {
        self.components.push(component);
        self
    }
}

/// Complete framework tree
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VizTree {
    pub framework: Framework,
    pub replacement: String,
    pub categories: Vec<FrameworkCategory>,
}

impl VizTree {
    pub fn new(framework: Framework) -> Self {
        Self { replacement: framework.replacement().to_string(), framework, categories: Vec::new() }
    }

    pub fn add_category(mut self, category: FrameworkCategory) -> Self {
        self.categories.push(category);
        self
    }
}

/// Integration mapping between Python and PAIML
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationMapping {
    pub python_component: String,
    pub paiml_component: String,
    pub integration_type: IntegrationType,
    pub category: String,
}

impl IntegrationMapping {
    pub fn new(
        python: impl Into<String>,
        paiml: impl Into<String>,
        int_type: IntegrationType,
        category: impl Into<String>,
    ) -> Self {
        Self {
            python_component: python.into(),
            paiml_component: paiml.into(),
            integration_type: int_type,
            category: category.into(),
        }
    }
}

// ============================================================================
// Tree Builders
// ============================================================================

/// Build Gradio framework tree
pub fn build_gradio_tree() -> VizTree {
    VizTree::new(Framework::Gradio)
        .add_category(
            FrameworkCategory::new("Interface").with_component(
                FrameworkComponent::new("Interface", "Quick demo builder", "Presentar::QuickApp")
                    .with_sub("Inputs")
                    .with_sub("Outputs")
                    .with_sub("Examples"),
            ),
        )
        .add_category(
            FrameworkCategory::new("Blocks").with_component(
                FrameworkComponent::new("Blocks", "Custom layouts", "Presentar::Layout")
                    .with_sub("Layout")
                    .with_sub("Events")
                    .with_sub("State"),
            ),
        )
        .add_category(
            FrameworkCategory::new("Components")
                .with_component(FrameworkComponent::new(
                    "Image",
                    "Image display/upload",
                    "Trueno-Viz::ImageView",
                ))
                .with_component(FrameworkComponent::new(
                    "Audio",
                    "Audio player/recorder",
                    "Presentar::AudioPlayer",
                ))
                .with_component(FrameworkComponent::new(
                    "Video",
                    "Video player",
                    "Presentar::VideoPlayer",
                ))
                .with_component(FrameworkComponent::new(
                    "Chatbot",
                    "Chat interface",
                    "Realizar + Presentar",
                ))
                .with_component(FrameworkComponent::new(
                    "DataFrame",
                    "Data table",
                    "Trueno-Viz::DataGrid",
                ))
                .with_component(FrameworkComponent::new(
                    "Plot",
                    "Chart display",
                    "Trueno-Viz::Chart",
                )),
        )
        .add_category(
            FrameworkCategory::new("Deployment").with_component(
                FrameworkComponent::new("Deployment", "Hosting options", "Batuta deploy")
                    .with_sub("HuggingFace Spaces")
                    .with_sub("Gradio Cloud")
                    .with_sub("Self-hosted"),
            ),
        )
}

/// Build Streamlit framework tree
pub fn build_streamlit_tree() -> VizTree {
    VizTree::new(Framework::Streamlit)
        .add_category(
            FrameworkCategory::new("Widgets")
                .with_component(
                    FrameworkComponent::new("Input", "User input widgets", "Presentar::Widgets")
                        .with_sub("text_input")
                        .with_sub("number_input")
                        .with_sub("slider")
                        .with_sub("selectbox"),
                )
                .with_component(
                    FrameworkComponent::new("Display", "Output widgets", "Presentar + Trueno-Viz")
                        .with_sub("write")
                        .with_sub("dataframe")
                        .with_sub("chart"),
                ),
        )
        .add_category(
            FrameworkCategory::new("Layout").with_component(
                FrameworkComponent::new("Layout", "Page structure", "Presentar::Layout")
                    .with_sub("columns")
                    .with_sub("tabs")
                    .with_sub("sidebar")
                    .with_sub("expander"),
            ),
        )
        .add_category(
            FrameworkCategory::new("Caching")
                .with_component(FrameworkComponent::new(
                    "@st.cache_data",
                    "Data caching",
                    "Trueno::TensorCache",
                ))
                .with_component(FrameworkComponent::new(
                    "@st.cache_resource",
                    "Resource caching",
                    "Presentar::ResourceCache",
                ))
                .with_component(FrameworkComponent::new(
                    "session_state",
                    "Session state",
                    "Presentar::State",
                )),
        )
        .add_category(
            FrameworkCategory::new("Deployment").with_component(
                FrameworkComponent::new("Deployment", "Hosting options", "Batuta deploy")
                    .with_sub("Streamlit Cloud")
                    .with_sub("Community Cloud")
                    .with_sub("Self-hosted"),
            ),
        )
}

/// Build Panel/HoloViz framework tree
pub fn build_panel_tree() -> VizTree {
    VizTree::new(Framework::Panel)
        .add_category(
            FrameworkCategory::new("Panes").with_component(
                FrameworkComponent::new("Panes", "Visualization containers", "Trueno-Viz::Chart")
                    .with_sub("Matplotlib")
                    .with_sub("Plotly")
                    .with_sub("HoloViews")
                    .with_sub("Bokeh"),
            ),
        )
        .add_category(
            FrameworkCategory::new("HoloViz Stack")
                .with_component(FrameworkComponent::new(
                    "HoloViews",
                    "Declarative viz",
                    "Trueno-Viz::ReactiveChart",
                ))
                .with_component(FrameworkComponent::new(
                    "Datashader",
                    "Big data raster",
                    "Trueno-Viz::GPURaster",
                ))
                .with_component(FrameworkComponent::new(
                    "hvPlot",
                    "High-level plotting",
                    "Trueno-Viz::Plot",
                ))
                .with_component(FrameworkComponent::new(
                    "Param",
                    "Parameters",
                    "Presentar::Params",
                )),
        )
        .add_category(
            FrameworkCategory::new("Layout").with_component(
                FrameworkComponent::new("Layout", "Dashboard structure", "Presentar::Layout")
                    .with_sub("Row")
                    .with_sub("Column")
                    .with_sub("Tabs")
                    .with_sub("GridSpec"),
            ),
        )
}

/// Build Dash framework tree
pub fn build_dash_tree() -> VizTree {
    VizTree::new(Framework::Dash)
        .add_category(
            FrameworkCategory::new("Core")
                .with_component(FrameworkComponent::new(
                    "dash.Dash",
                    "App container",
                    "Presentar::App",
                ))
                .with_component(FrameworkComponent::new(
                    "html.*",
                    "HTML components",
                    "Presentar::Html",
                ))
                .with_component(FrameworkComponent::new(
                    "@callback",
                    "Event handlers",
                    "Presentar::on_event",
                ))
                .with_component(FrameworkComponent::new(
                    "State",
                    "State management",
                    "Presentar::State",
                )),
        )
        .add_category(
            FrameworkCategory::new("Components")
                .with_component(FrameworkComponent::new(
                    "dcc.Graph",
                    "Plotly charts",
                    "Trueno-Viz::Chart",
                ))
                .with_component(FrameworkComponent::new(
                    "dcc.Input",
                    "Text input",
                    "Presentar::TextInput",
                ))
                .with_component(FrameworkComponent::new(
                    "dash_table",
                    "Data tables",
                    "Trueno-Viz::DataGrid",
                ))
                .with_component(FrameworkComponent::new(
                    "dcc.Store",
                    "Client storage",
                    "Presentar::Store",
                )),
        )
        .add_category(
            FrameworkCategory::new("Plotly")
                .with_component(
                    FrameworkComponent::new("plotly.express", "Quick charts", "Trueno-Viz::Charts")
                        .with_sub("line")
                        .with_sub("scatter")
                        .with_sub("bar")
                        .with_sub("histogram"),
                )
                .with_component(FrameworkComponent::new(
                    "plotly.graph_objects",
                    "Custom charts",
                    "Trueno-Viz::Figure",
                )),
        )
        .add_category(
            FrameworkCategory::new("Enterprise").with_component(
                FrameworkComponent::new("Enterprise", "Dash Enterprise", "Batuta deploy")
                    .with_sub("Auth")
                    .with_sub("Deployment")
                    .with_sub("Snapshots"),
            ),
        )
}

// ============================================================================
// Integration Mappings
// ============================================================================

/// Build complete integration mappings
pub fn build_integration_mappings() -> Vec<IntegrationMapping> {
    vec![
        // UI Frameworks
        IntegrationMapping::new(
            "gr.Interface",
            "Presentar::QuickApp",
            IntegrationType::Replaces,
            "UI FRAMEWORKS",
        ),
        IntegrationMapping::new(
            "gr.Blocks",
            "Presentar::Layout",
            IntegrationType::Replaces,
            "UI FRAMEWORKS",
        ),
        IntegrationMapping::new(
            "dash.Dash",
            "Presentar::App",
            IntegrationType::Replaces,
            "UI FRAMEWORKS",
        ),
        IntegrationMapping::new(
            "st.columns/sidebar",
            "Presentar::Layout",
            IntegrationType::Replaces,
            "UI FRAMEWORKS",
        ),
        // Visualization
        IntegrationMapping::new(
            "dcc.Graph",
            "Trueno-Viz::Chart",
            IntegrationType::Replaces,
            "VISUALIZATION",
        ),
        IntegrationMapping::new(
            "st.plotly_chart",
            "Trueno-Viz::Chart",
            IntegrationType::Replaces,
            "VISUALIZATION",
        ),
        IntegrationMapping::new(
            "st.dataframe",
            "Trueno-Viz::DataGrid",
            IntegrationType::Replaces,
            "VISUALIZATION",
        ),
        IntegrationMapping::new(
            "dash_table",
            "Trueno-Viz::DataGrid",
            IntegrationType::Replaces,
            "VISUALIZATION",
        ),
        IntegrationMapping::new(
            "datashader",
            "Trueno-Viz::GPURaster",
            IntegrationType::Replaces,
            "VISUALIZATION",
        ),
        IntegrationMapping::new(
            "matplotlib/plotly/bokeh",
            "Trueno-Viz::Plot",
            IntegrationType::Replaces,
            "VISUALIZATION",
        ),
        // Components
        IntegrationMapping::new(
            "st.text_input",
            "Presentar::TextInput",
            IntegrationType::Replaces,
            "COMPONENTS",
        ),
        IntegrationMapping::new(
            "st.slider",
            "Presentar::Slider",
            IntegrationType::Replaces,
            "COMPONENTS",
        ),
        IntegrationMapping::new(
            "st.selectbox",
            "Presentar::Select",
            IntegrationType::Replaces,
            "COMPONENTS",
        ),
        IntegrationMapping::new(
            "st.button",
            "Presentar::Button",
            IntegrationType::Replaces,
            "COMPONENTS",
        ),
        IntegrationMapping::new(
            "gr.Image",
            "Trueno-Viz::ImageView",
            IntegrationType::Replaces,
            "COMPONENTS",
        ),
        // State & Caching
        IntegrationMapping::new(
            "st.session_state",
            "Presentar::State",
            IntegrationType::Replaces,
            "STATE & CACHING",
        ),
        IntegrationMapping::new(
            "@st.cache_data",
            "Trueno::TensorCache",
            IntegrationType::Replaces,
            "STATE & CACHING",
        ),
        IntegrationMapping::new(
            "@callback",
            "Presentar::on_event",
            IntegrationType::Replaces,
            "STATE & CACHING",
        ),
        // Deployment
        IntegrationMapping::new(
            "HuggingFace Spaces",
            "Batuta deploy",
            IntegrationType::Replaces,
            "DEPLOYMENT",
        ),
        IntegrationMapping::new(
            "Streamlit Cloud",
            "Batuta deploy",
            IntegrationType::Replaces,
            "DEPLOYMENT",
        ),
        IntegrationMapping::new(
            "Dash Enterprise",
            "Batuta deploy",
            IntegrationType::Replaces,
            "DEPLOYMENT",
        ),
    ]
}

// ============================================================================
// Formatters
// ============================================================================

/// Format a single framework tree as ASCII
pub fn format_framework_tree(tree: &VizTree) -> String {
    let mut output = String::new();
    output.push_str(&format!(
        "{} (Python) → {} (Rust)\n",
        tree.framework.name().to_uppercase(),
        tree.replacement
    ));

    let cat_count = tree.categories.len();
    for (i, category) in tree.categories.iter().enumerate() {
        let is_last_cat = i == cat_count - 1;
        let cat_prefix = if is_last_cat { "└──" } else { "├──" };
        let cat_cont = if is_last_cat { "    " } else { "│   " };

        output.push_str(&format!("{} {}\n", cat_prefix, category.name));

        let comp_count = category.components.len();
        for (j, component) in category.components.iter().enumerate() {
            let is_last_comp = j == comp_count - 1;
            let comp_prefix = if is_last_comp { "└──" } else { "├──" };
            let comp_cont = if is_last_comp { "    " } else { "│   " };

            output.push_str(&format!(
                "{}{} {} → {}\n",
                cat_cont, comp_prefix, component.name, component.replacement
            ));

            // Sub-components
            let sub_count = component.sub_components.len();
            for (k, sub) in component.sub_components.iter().enumerate() {
                let is_last_sub = k == sub_count - 1;
                let sub_prefix = if is_last_sub { "└──" } else { "├──" };
                output.push_str(&format!("{}{}{} {}\n", cat_cont, comp_cont, sub_prefix, sub));
            }
        }
    }

    output
}

/// Format all frameworks as ASCII tree
pub fn format_all_frameworks() -> String {
    let mut output = String::new();
    output.push_str("VISUALIZATION FRAMEWORKS ECOSYSTEM\n");
    output.push_str("==================================\n\n");

    let trees =
        vec![build_gradio_tree(), build_streamlit_tree(), build_panel_tree(), build_dash_tree()];

    for tree in &trees {
        output.push_str(&format_framework_tree(tree));
        output.push('\n');
    }

    output.push_str(&format!(
        "Summary: {} Python frameworks replaced by 2 Rust libraries (Presentar, Trueno-Viz)\n",
        trees.len()
    ));

    output
}

/// Format integration mappings as ASCII
pub fn format_integration_mappings() -> String {
    let mut output = String::new();
    output.push_str("PAIML REPLACEMENTS FOR PYTHON VIZ\n");
    output.push_str("=================================\n\n");

    let mappings = build_integration_mappings();
    let mut current_category = String::new();

    for mapping in &mappings {
        if mapping.category != current_category {
            if !current_category.is_empty() {
                output.push('\n');
            }
            output.push_str(&format!("{}\n", mapping.category));
            current_category = mapping.category.clone();
        }

        output.push_str(&format!(
            "├── [{}] {} ← {}\n",
            mapping.integration_type.code(),
            mapping.paiml_component,
            mapping.python_component
        ));
    }

    output.push_str("\nLegend: [REP]=Replaces (Python eliminated)\n");
    output.push_str("\nSummary: ");

    let rep_count =
        mappings.iter().filter(|m| m.integration_type == IntegrationType::Replaces).count();

    output.push_str(&format!(
        "{} Python components replaced by sovereign Rust alternatives\n",
        rep_count
    ));
    output.push_str("         Zero Python dependencies in production\n");

    output
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    // ========================================================================
    // VIZ-TREE-001: Framework Tests
    // ========================================================================

    #[test]
    fn test_VIZ_TREE_001_framework_names() {
        assert_eq!(Framework::Gradio.name(), "Gradio");
        assert_eq!(Framework::Streamlit.name(), "Streamlit");
        assert_eq!(Framework::Panel.name(), "Panel");
        assert_eq!(Framework::Dash.name(), "Dash");
    }

    #[test]
    fn test_VIZ_TREE_001_framework_replacements() {
        assert_eq!(Framework::Gradio.replacement(), "Presentar");
        assert_eq!(Framework::Streamlit.replacement(), "Presentar");
        assert_eq!(Framework::Panel.replacement(), "Trueno-Viz");
        assert_eq!(Framework::Dash.replacement(), "Presentar + Trueno-Viz");
    }

    #[test]
    fn test_VIZ_TREE_001_framework_all() {
        let all = Framework::all();
        assert_eq!(all.len(), 4);
    }

    #[test]
    fn test_VIZ_TREE_001_framework_display() {
        assert_eq!(format!("{}", Framework::Gradio), "Gradio");
    }

    // ========================================================================
    // VIZ-TREE-002: Integration Type Tests
    // ========================================================================

    #[test]
    fn test_VIZ_TREE_002_integration_codes() {
        assert_eq!(IntegrationType::Replaces.code(), "REP");
        assert_eq!(IntegrationType::Transpiles.code(), "TRN");
        assert_eq!(IntegrationType::Compatible.code(), "CMP");
    }

    #[test]
    fn test_VIZ_TREE_002_integration_display() {
        assert_eq!(format!("{}", IntegrationType::Replaces), "REP");
    }

    // ========================================================================
    // VIZ-TREE-003: Tree Builder Tests
    // ========================================================================

    #[test]
    fn test_VIZ_TREE_003_gradio_tree() {
        let tree = build_gradio_tree();
        assert_eq!(tree.framework, Framework::Gradio);
        assert_eq!(tree.replacement, "Presentar");
        assert!(!tree.categories.is_empty());
    }

    #[test]
    fn test_VIZ_TREE_003_streamlit_tree() {
        let tree = build_streamlit_tree();
        assert_eq!(tree.framework, Framework::Streamlit);
        assert_eq!(tree.replacement, "Presentar");
        assert!(!tree.categories.is_empty());
    }

    #[test]
    fn test_VIZ_TREE_003_panel_tree() {
        let tree = build_panel_tree();
        assert_eq!(tree.framework, Framework::Panel);
        assert_eq!(tree.replacement, "Trueno-Viz");
        assert!(!tree.categories.is_empty());
    }

    #[test]
    fn test_VIZ_TREE_003_dash_tree() {
        let tree = build_dash_tree();
        assert_eq!(tree.framework, Framework::Dash);
        assert_eq!(tree.replacement, "Presentar + Trueno-Viz");
        assert!(!tree.categories.is_empty());
    }

    // ========================================================================
    // VIZ-TREE-004: Integration Mapping Tests
    // ========================================================================

    #[test]
    fn test_VIZ_TREE_004_mappings_exist() {
        let mappings = build_integration_mappings();
        assert!(!mappings.is_empty());
        assert!(mappings.len() >= 20);
    }

    #[test]
    fn test_VIZ_TREE_004_all_replaces() {
        let mappings = build_integration_mappings();
        // All mappings should be Replaces type (no Python allowed)
        for mapping in &mappings {
            assert_eq!(
                mapping.integration_type,
                IntegrationType::Replaces,
                "Mapping {} should be Replaces",
                mapping.python_component
            );
        }
    }

    #[test]
    fn test_VIZ_TREE_004_mapping_categories() {
        let mappings = build_integration_mappings();
        let categories: std::collections::HashSet<_> =
            mappings.iter().map(|m| m.category.as_str()).collect();
        assert!(categories.contains("UI FRAMEWORKS"));
        assert!(categories.contains("VISUALIZATION"));
        assert!(categories.contains("COMPONENTS"));
        assert!(categories.contains("DEPLOYMENT"));
    }

    // ========================================================================
    // VIZ-TREE-005: Formatter Tests
    // ========================================================================

    #[test]
    fn test_VIZ_TREE_005_format_framework_tree() {
        let tree = build_gradio_tree();
        let output = format_framework_tree(&tree);
        assert!(output.contains("GRADIO"));
        assert!(output.contains("Presentar"));
        assert!(output.contains("Interface"));
    }

    #[test]
    fn test_VIZ_TREE_005_format_all_frameworks() {
        let output = format_all_frameworks();
        assert!(output.contains("VISUALIZATION FRAMEWORKS ECOSYSTEM"));
        assert!(output.contains("GRADIO"));
        assert!(output.contains("STREAMLIT"));
        assert!(output.contains("PANEL"));
        assert!(output.contains("DASH"));
        assert!(output.contains("Summary:"));
    }

    #[test]
    fn test_VIZ_TREE_005_format_integration_mappings() {
        let output = format_integration_mappings();
        assert!(output.contains("PAIML REPLACEMENTS"));
        assert!(output.contains("[REP]"));
        assert!(output.contains("Presentar"));
        assert!(output.contains("Trueno-Viz"));
        assert!(output.contains("Legend:"));
    }

    // ========================================================================
    // VIZ-TREE-006: Component Tests
    // ========================================================================

    #[test]
    fn test_VIZ_TREE_006_framework_component() {
        let component = FrameworkComponent::new("Test", "Description", "Replacement")
            .with_sub("Sub1")
            .with_sub("Sub2");
        assert_eq!(component.name, "Test");
        assert_eq!(component.sub_components.len(), 2);
    }

    #[test]
    fn test_VIZ_TREE_006_framework_category() {
        let category = FrameworkCategory::new("Test Category")
            .with_component(FrameworkComponent::new("Comp1", "Desc1", "Rep1"));
        assert_eq!(category.name, "Test Category");
        assert_eq!(category.components.len(), 1);
    }

    #[test]
    fn test_VIZ_TREE_006_integration_mapping() {
        let mapping =
            IntegrationMapping::new("Python", "Rust", IntegrationType::Replaces, "Category");
        assert_eq!(mapping.python_component, "Python");
        assert_eq!(mapping.paiml_component, "Rust");
    }

    // ========================================================================
    // VIZ-TREE-007: Additional Coverage Tests
    // ========================================================================

    #[test]
    fn test_VIZ_TREE_007_framework_equality() {
        assert_eq!(Framework::Gradio, Framework::Gradio);
        assert_ne!(Framework::Gradio, Framework::Streamlit);
    }

    #[test]
    fn test_VIZ_TREE_007_framework_clone() {
        let f1 = Framework::Panel;
        let f2 = f1;
        assert_eq!(f1, f2);
    }

    #[test]
    fn test_VIZ_TREE_007_integration_type_equality() {
        assert_eq!(IntegrationType::Replaces, IntegrationType::Replaces);
        assert_ne!(IntegrationType::Replaces, IntegrationType::Transpiles);
    }

    #[test]
    fn test_VIZ_TREE_007_integration_type_clone() {
        let t1 = IntegrationType::Compatible;
        let t2 = t1;
        assert_eq!(t1, t2);
    }

    #[test]
    fn test_VIZ_TREE_007_framework_serialization() {
        let framework = Framework::Dash;
        let json = serde_json::to_string(&framework).expect("json serialize failed");
        let parsed: Framework = serde_json::from_str(&json).expect("json deserialize failed");
        assert_eq!(framework, parsed);
    }

    #[test]
    fn test_VIZ_TREE_007_integration_type_serialization() {
        let integration = IntegrationType::Transpiles;
        let json = serde_json::to_string(&integration).expect("json serialize failed");
        let parsed: IntegrationType = serde_json::from_str(&json).expect("json deserialize failed");
        assert_eq!(integration, parsed);
    }

    #[test]
    fn test_VIZ_TREE_007_framework_component_serialization() {
        let component = FrameworkComponent::new("Test", "Desc", "Rep");
        let json = serde_json::to_string(&component).expect("json serialize failed");
        let parsed: FrameworkComponent =
            serde_json::from_str(&json).expect("json deserialize failed");
        assert_eq!(component.name, parsed.name);
    }

    #[test]
    fn test_VIZ_TREE_007_framework_tree_categories() {
        let tree = build_gradio_tree();
        // Gradio should have multiple categories
        assert!(tree.categories.len() > 1);
        // Each category should have components
        for category in &tree.categories {
            assert!(!category.components.is_empty());
        }
    }

    #[test]
    fn test_VIZ_TREE_007_framework_debug() {
        let f = Framework::Gradio;
        let debug = format!("{:?}", f);
        assert!(debug.contains("Gradio"));
    }

    #[test]
    fn test_VIZ_TREE_007_integration_type_debug() {
        let t = IntegrationType::Replaces;
        let debug = format!("{:?}", t);
        assert!(debug.contains("Replaces"));
    }
}
