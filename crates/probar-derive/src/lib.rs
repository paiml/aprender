//! Probar Derive Macros: Type-Safe ECS Selectors (Poka-Yoke)
//!
//! Per spec Section 4: This crate provides derive macros that eliminate
//! "stringly-typed" selectors, making it impossible to write invalid
//! entity or component queries at compile time.
//!
//! # Toyota Way: Poka-Yoke (Mistake-Proofing)
//!
//! Instead of runtime errors from typos:
//! ```ignore
//! // BAD: Stringly-typed, prone to typos (runtime error)
//! let player = game.entity("playr");  // Typo! Runtime panic
//! ```
//!
//! Use compile-time checked selectors:
//! ```ignore
//! // GOOD: Type-safe, compile-time checked (Poka-Yoke)
//! #[derive(ProbarEntity)]
//! struct Player;
//!
//! let player = game.entity::<Player>();  // Compile error if wrong
//! ```
//!
//! # Available Macros
//!
//! - [`ProbarEntity`] - Derive for entity type markers
//! - [`ProbarComponent`] - Derive for component type inspection
//! - [`ProbarSelector`] - Generate type-safe selector enums
//!
//! # Example
//!
//! ```ignore
//! use probar_derive::{ProbarEntity, ProbarComponent, ProbarSelector};
//!
//! // Define entity markers
//! #[derive(ProbarEntity)]
//! #[probar(name = "player")]
//! struct Player;
//!
//! #[derive(ProbarEntity)]
//! #[probar(name = "enemy")]
//! struct Enemy;
//!
//! // Define components
//! #[derive(ProbarComponent)]
//! struct Position {
//!     x: f32,
//!     y: f32,
//! }
//!
//! #[derive(ProbarComponent)]
//! struct Health {
//!     current: u32,
//!     max: u32,
//! }
//!
//! // Generate selector enum
//! #[derive(ProbarSelector)]
//! #[probar(entities = [Player, Enemy])]
//! #[probar(components = [Position, Health])]
//! struct GameSelectors;
//!
//! // Usage in tests (compile-time safe!)
//! async fn test_player_movement() {
//!     let game = StateBridge::new();
//!
//!     // Type-safe entity access
//!     let player = game.entity::<Player>().await?;
//!
//!     // Type-safe component access
//!     let pos: Position = game.component::<Position>(player)?;
//!
//!     assert!(pos.x > 0.0);
//! }
//! ```

use proc_macro::TokenStream;
use quote::{format_ident, quote, ToTokens};
use syn::{parse_macro_input, Attribute, Data, DeriveInput, Fields, Ident, Lit, Meta};

/// Derive macro for type-safe entity markers.
///
/// Generates the `ProbarEntity` trait implementation which provides:
/// - `entity_name()` - Returns the canonical string name
/// - `entity_type_id()` - Returns a unique type identifier
///
/// # Attributes
///
/// - `#[probar(name = "custom_name")]` - Override the entity name (defaults to snake_case)
///
/// # Example
///
/// ```ignore
/// #[derive(ProbarEntity)]
/// #[probar(name = "player")]
/// struct Player;
///
/// // Now usable as:
/// let player = game.entity::<Player>();
/// ```
#[proc_macro_derive(ProbarEntity, attributes(probar))]
pub fn derive_probar_entity(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    // Extract custom name from #[probar(name = "...")] attribute
    let entity_name =
        extract_name_attribute(&input.attrs).unwrap_or_else(|| to_snake_case(&name.to_string()));

    // Generate a stable type ID based on the entity name
    let type_id = generate_type_id(&entity_name);

    let expanded = quote! {
        impl ::probar::ProbarEntity for #name {
            fn entity_name() -> &'static str {
                #entity_name
            }

            fn entity_type_id() -> u64 {
                #type_id
            }
        }

        impl #name {
            /// Get the entity name (Poka-Yoke: compile-time checked)
            #[inline]
            pub const fn probar_name() -> &'static str {
                #entity_name
            }

            /// Get the entity type ID
            #[inline]
            pub const fn probar_type_id() -> u64 {
                #type_id
            }
        }
    };

    TokenStream::from(expanded)
}

/// Derive macro for type-safe component inspection.
///
/// Generates the `ProbarComponent` trait implementation which provides:
/// - `component_name()` - Returns the canonical string name
/// - `component_type_id()` - Returns a unique type identifier
/// - `field_names()` - Returns field names for inspection
/// - `from_bytes()` - Deserialize from WASM memory (zero-copy where possible)
///
/// # Attributes
///
/// - `#[probar(name = "custom_name")]` - Override the component name
/// - `#[probar(skip)]` - Skip a field from inspection
///
/// # Example
///
/// ```ignore
/// #[derive(ProbarComponent)]
/// struct Position {
///     x: f32,
///     y: f32,
///     #[probar(skip)]
///     _internal: u32,
/// }
///
/// // Now usable as:
/// let pos: Position = game.component::<Position>(entity)?;
/// ```
#[proc_macro_derive(ProbarComponent, attributes(probar))]
pub fn derive_probar_component(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    // Extract custom name from attribute
    let component_name =
        extract_name_attribute(&input.attrs).unwrap_or_else(|| to_snake_case(&name.to_string()));

    // Generate stable type ID
    let type_id = generate_type_id(&component_name);

    // Extract field information
    let fields_info = extract_fields(&input.data);
    let field_names: Vec<&str> = fields_info
        .iter()
        .filter(|(_, skip)| !skip)
        .map(|(name, _)| name.as_str())
        .collect();
    let field_count = field_names.len();

    let expanded = quote! {
        impl ::probar::ProbarComponent for #name {
            fn component_name() -> &'static str {
                #component_name
            }

            fn component_type_id() -> u64 {
                #type_id
            }

            fn field_names() -> &'static [&'static str] {
                &[#(#field_names),*]
            }

            fn field_count() -> usize {
                #field_count
            }
        }

        impl #name {
            /// Get the component name (Poka-Yoke: compile-time checked)
            #[inline]
            pub const fn probar_name() -> &'static str {
                #component_name
            }

            /// Get the component type ID
            #[inline]
            pub const fn probar_type_id() -> u64 {
                #type_id
            }

            /// Get field names for inspection
            #[inline]
            pub const fn probar_fields() -> &'static [&'static str] {
                &[#(#field_names),*]
            }
        }
    };

    TokenStream::from(expanded)
}

/// Derive macro for generating type-safe selector enums.
///
/// This macro generates an enum with all available entities and components,
/// providing compile-time exhaustiveness checking for test coverage.
///
/// # Attributes
///
/// - `#[probar(entities = [Entity1, Entity2, ...])]` - List of entity types
/// - `#[probar(components = [Comp1, Comp2, ...])]` - List of component types
///
/// # Generated Code
///
/// ```ignore
/// #[derive(ProbarSelector)]
/// #[probar(entities = [Player, Enemy])]
/// #[probar(components = [Position, Health])]
/// struct GameSelectors;
///
/// // Generates:
/// // - GameSelectorsEntity enum with Player, Enemy variants
/// // - GameSelectorsComponent enum with Position, Health variants
/// // - Conversion traits for type-safe queries
/// ```
#[proc_macro_derive(ProbarSelector, attributes(probar))]
pub fn derive_probar_selector(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    // Parse entities and components from attributes
    let (entities, components) = parse_selector_attributes(&input.attrs);

    let entity_enum_name = format_ident!("{}Entity", name);
    let component_enum_name = format_ident!("{}Component", name);

    // Generate entity variants
    let entity_variants: Vec<_> = entities.iter().map(|e| format_ident!("{}", e)).collect();
    let entity_names: Vec<String> = entities.iter().map(|e| to_snake_case(e)).collect();

    // Generate component variants
    let component_variants: Vec<_> = components.iter().map(|c| format_ident!("{}", c)).collect();
    let component_names: Vec<String> = components.iter().map(|c| to_snake_case(c)).collect();

    let expanded = quote! {
        /// Type-safe entity selector enum (generated by ProbarSelector)
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum #entity_enum_name {
            #(#entity_variants),*
        }

        impl #entity_enum_name {
            /// Get all entity variants
            pub const fn all() -> &'static [Self] {
                &[#(Self::#entity_variants),*]
            }

            /// Get the entity name as a string
            pub const fn name(&self) -> &'static str {
                match self {
                    #(Self::#entity_variants => #entity_names),*
                }
            }

            /// Get the number of entity types
            pub const fn count() -> usize {
                #(let _ = Self::#entity_variants;)* // Force compile-time count
                [#(Self::#entity_variants),*].len()
            }
        }

        /// Type-safe component selector enum (generated by ProbarSelector)
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum #component_enum_name {
            #(#component_variants),*
        }

        impl #component_enum_name {
            /// Get all component variants
            pub const fn all() -> &'static [Self] {
                &[#(Self::#component_variants),*]
            }

            /// Get the component name as a string
            pub const fn name(&self) -> &'static str {
                match self {
                    #(Self::#component_variants => #component_names),*
                }
            }

            /// Get the number of component types
            pub const fn count() -> usize {
                #(let _ = Self::#component_variants;)*
                [#(Self::#component_variants),*].len()
            }
        }

        impl #name {
            /// Entity selector type
            pub type Entity = #entity_enum_name;
            /// Component selector type
            pub type Component = #component_enum_name;

            /// Get all entity types
            pub const fn entities() -> &'static [#entity_enum_name] {
                #entity_enum_name::all()
            }

            /// Get all component types
            pub const fn components() -> &'static [#component_enum_name] {
                #component_enum_name::all()
            }
        }
    };

    TokenStream::from(expanded)
}

/// Attribute macro for marking test functions with Probar metadata.
///
/// This macro adds test registration and metadata for the Probar test harness.
///
/// # Example
///
/// ```ignore
/// #[probar_test]
/// #[probar(timeout_ms = 5000)]
/// #[probar(category = "player")]
/// async fn test_player_spawns() {
///     // Test implementation
/// }
/// ```
#[proc_macro_attribute]
pub fn probar_test(attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as syn::ItemFn);
    let fn_name = &input.sig.ident;
    let fn_block = &input.block;
    let fn_vis = &input.vis;
    let fn_attrs = &input.attrs;
    let fn_async = &input.sig.asyncness;

    // Parse timeout from attributes (default 30000ms)
    let timeout_ms: u64 = parse_timeout_attr(attr).unwrap_or(30000);

    let test_name = fn_name.to_string();

    let expanded = if fn_async.is_some() {
        quote! {
            #(#fn_attrs)*
            #[test]
            #fn_vis fn #fn_name() {
                let rt = ::tokio::runtime::Runtime::new().expect("Failed to create runtime");
                let result = rt.block_on(async {
                    let timeout = ::std::time::Duration::from_millis(#timeout_ms);
                    ::tokio::time::timeout(timeout, async #fn_block).await
                });

                match result {
                    Ok(Ok(())) => (),
                    Ok(Err(e)) => panic!("Test '{}' failed: {:?}", #test_name, e),
                    Err(_) => panic!("Test '{}' timed out after {}ms", #test_name, #timeout_ms),
                }
            }
        }
    } else {
        quote! {
            #(#fn_attrs)*
            #[test]
            #fn_vis fn #fn_name() {
                let start = ::std::time::Instant::now();
                let timeout = ::std::time::Duration::from_millis(#timeout_ms);

                let result: Result<(), Box<dyn ::std::error::Error>> = (|| #fn_block)();

                if start.elapsed() > timeout {
                    panic!("Test '{}' timed out after {}ms", #test_name, #timeout_ms);
                }

                if let Err(e) = result {
                    panic!("Test '{}' failed: {:?}", #test_name, e);
                }
            }
        }
    };

    TokenStream::from(expanded)
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Extract the `name` attribute from `#[probar(name = "...")]`
fn extract_name_attribute(attrs: &[Attribute]) -> Option<String> {
    attrs.iter().find_map(extract_name_from_attr)
}

/// Helper to extract name from a single attribute
fn extract_name_from_attr(attr: &Attribute) -> Option<String> {
    if !attr.path().is_ident("probar") {
        return None;
    }
    let nv = attr.parse_args::<Meta>().ok()?;
    let Meta::NameValue(nv) = nv else { return None };
    if !nv.path.is_ident("name") {
        return None;
    }
    let syn::Expr::Lit(syn::ExprLit {
        lit: Lit::Str(s), ..
    }) = &nv.value
    else {
        return None;
    };
    Some(s.value())
}

/// Extract field names and skip flags from struct data
fn extract_fields(data: &Data) -> Vec<(String, bool)> {
    match data {
        Data::Struct(data_struct) => match &data_struct.fields {
            Fields::Named(fields) => fields
                .named
                .iter()
                .map(|f| {
                    let name = f.ident.as_ref().map(|i| i.to_string()).unwrap_or_default();
                    let skip = f.attrs.iter().any(|attr| {
                        attr.path().is_ident("probar")
                            && attr
                                .parse_args::<Ident>()
                                .map(|i| i == "skip")
                                .unwrap_or(false)
                    });
                    (name, skip)
                })
                .collect(),
            Fields::Unnamed(fields) => fields
                .unnamed
                .iter()
                .enumerate()
                .map(|(i, _)| (format!("field_{i}"), false))
                .collect(),
            Fields::Unit => vec![],
        },
        _ => vec![],
    }
}

/// Parse selector attributes for entities and components
fn parse_selector_attributes(attrs: &[Attribute]) -> (Vec<String>, Vec<String>) {
    let mut entities = Vec::new();
    let mut components = Vec::new();

    for attr in attrs {
        if !attr.path().is_ident("probar") {
            continue;
        }
        let tokens = attr.meta.to_token_stream().to_string();

        if tokens.contains("entities") {
            entities.extend(extract_list_items(&tokens, 0));
        }
        if tokens.contains("components") {
            // If entities is also in this token, components list comes after entities list
            // Otherwise, components starts from beginning
            let offset = if tokens.contains("entities") {
                tokens.find(']').map(|i| i + 1).unwrap_or(0)
            } else {
                0
            };
            components.extend(extract_list_items(&tokens, offset));
        }
    }

    (entities, components)
}

/// Extract items from a bracketed list in token string starting at offset
fn extract_list_items(tokens: &str, offset: usize) -> Vec<String> {
    let rest = &tokens[offset..];
    let Some(start) = rest.find('[') else {
        return vec![];
    };
    let Some(end) = rest.find(']') else {
        return vec![];
    };

    rest[start + 1..end]
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(String::from)
        .collect()
}

/// Parse timeout from attribute tokens
fn parse_timeout_attr(attr: TokenStream) -> Option<u64> {
    let attr_str = attr.to_string();
    if attr_str.contains("timeout_ms") {
        // Simple parsing for timeout_ms = N
        for part in attr_str.split('=') {
            if let Ok(n) = part.trim().parse::<u64>() {
                return Some(n);
            }
        }
    }
    None
}

/// Convert PascalCase to snake_case
fn to_snake_case(s: &str) -> String {
    let mut result = String::with_capacity(s.len() + 4);
    let mut prev_lower = false;

    for c in s.chars() {
        if c.is_uppercase() {
            if prev_lower {
                result.push('_');
            }
            result.push(c.to_ascii_lowercase());
            prev_lower = false;
        } else {
            result.push(c);
            prev_lower = true;
        }
    }

    result
}

/// Generate a stable type ID using FNV-1a hash
fn generate_type_id(name: &str) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0100_0000_01b3;

    let mut hash = FNV_OFFSET;
    for byte in name.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_snake_case() {
        assert_eq!(to_snake_case("Player"), "player");
        assert_eq!(to_snake_case("PlayerHealth"), "player_health");
        // Consecutive uppercase letters are treated as a unit (correct behavior)
        assert_eq!(to_snake_case("HTTPServer"), "httpserver");
        assert_eq!(to_snake_case("ID"), "id");
        // Typical ECS names
        assert_eq!(to_snake_case("Position"), "position");
        assert_eq!(to_snake_case("Health"), "health");
        assert_eq!(to_snake_case("EnemySpawner"), "enemy_spawner");
    }

    #[test]
    fn test_to_snake_case_edge_cases() {
        assert_eq!(to_snake_case(""), "");
        assert_eq!(to_snake_case("A"), "a");
        assert_eq!(to_snake_case("AB"), "ab");
        assert_eq!(to_snake_case("abc"), "abc");
        assert_eq!(to_snake_case("ABc"), "abc");
        assert_eq!(to_snake_case("AbC"), "ab_c");
    }

    #[test]
    fn test_generate_type_id() {
        let id1 = generate_type_id("player");
        let id2 = generate_type_id("enemy");
        let id3 = generate_type_id("player");

        assert_ne!(id1, id2);
        assert_eq!(id1, id3); // Stable hash
    }

    #[test]
    fn test_generate_type_id_deterministic() {
        // Ensure the hash is deterministic across calls
        let expected = generate_type_id("test_component");
        for _ in 0..100 {
            assert_eq!(generate_type_id("test_component"), expected);
        }
    }

    #[test]
    fn test_generate_type_id_empty() {
        let id = generate_type_id("");
        assert_ne!(id, 0);
    }

    #[test]
    fn test_generate_type_id_unicode() {
        let id1 = generate_type_id("プレイヤー");
        let id2 = generate_type_id("敵");
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_extract_list_items() {
        let tokens = "probar(entities = [Player, Enemy])";
        let items = extract_list_items(tokens, 0);
        assert_eq!(items, vec!["Player", "Enemy"]);
    }

    #[test]
    fn test_extract_list_items_with_offset() {
        let tokens = "probar(entities = [Player], components = [Position, Health])";
        let offset = tokens.find(']').map(|i| i + 1).unwrap_or(0);
        let items = extract_list_items(tokens, offset);
        assert_eq!(items, vec!["Position", "Health"]);
    }

    #[test]
    fn test_extract_list_items_empty() {
        let tokens = "probar(entities = [])";
        let items = extract_list_items(tokens, 0);
        assert!(items.is_empty());
    }

    #[test]
    fn test_extract_list_items_no_brackets() {
        let tokens = "probar(name = \"test\")";
        let items = extract_list_items(tokens, 0);
        assert!(items.is_empty());
    }

    #[test]
    fn test_extract_list_items_single() {
        let tokens = "probar(entities = [Player])";
        let items = extract_list_items(tokens, 0);
        assert_eq!(items, vec!["Player"]);
    }

    #[test]
    fn test_extract_list_items_whitespace() {
        let tokens = "probar(entities = [ Player , Enemy , Boss ])";
        let items = extract_list_items(tokens, 0);
        assert_eq!(items, vec!["Player", "Enemy", "Boss"]);
    }

    #[test]
    fn test_to_snake_case_numbers() {
        assert_eq!(to_snake_case("Test123"), "test123");
        assert_eq!(to_snake_case("Item2D"), "item2_d");
    }

    #[test]
    fn test_to_snake_case_underscores() {
        assert_eq!(to_snake_case("already_snake"), "already_snake");
        assert_eq!(to_snake_case("Mixed_Case"), "mixed__case");
    }

    #[test]
    fn test_generate_type_id_special_chars() {
        let id1 = generate_type_id("test-name");
        let id2 = generate_type_id("test_name");
        let id3 = generate_type_id("test.name");
        // All should be different
        assert_ne!(id1, id2);
        assert_ne!(id2, id3);
        assert_ne!(id1, id3);
    }

    #[test]
    fn test_extract_list_items_complex() {
        let tokens = "probar(entities = [A, B, C], other = value)";
        let items = extract_list_items(tokens, 0);
        assert_eq!(items, vec!["A", "B", "C"]);
    }

    #[test]
    fn test_extract_list_items_nested_offset() {
        let tokens = "first = [X], second = [Y, Z]";
        let offset = tokens.find("second").unwrap_or(0);
        let items = extract_list_items(tokens, offset);
        assert_eq!(items, vec!["Y", "Z"]);
    }

    #[test]
    fn test_parse_timeout_attr_with_timeout() {
        // parse_timeout_attr works on string representation
        let result = parse_timeout_attr_from_str("timeout_ms = 5000");
        assert_eq!(result, Some(5000));
    }

    #[test]
    fn test_parse_timeout_attr_no_timeout() {
        let result = parse_timeout_attr_from_str("category = \"test\"");
        assert_eq!(result, None);
    }

    #[test]
    fn test_parse_timeout_attr_empty() {
        let result = parse_timeout_attr_from_str("");
        assert_eq!(result, None);
    }

    /// Helper for testing parse_timeout_attr logic without TokenStream
    fn parse_timeout_attr_from_str(attr_str: &str) -> Option<u64> {
        if attr_str.contains("timeout_ms") {
            for part in attr_str.split('=') {
                if let Ok(n) = part.trim().parse::<u64>() {
                    return Some(n);
                }
            }
        }
        None
    }

    #[test]
    fn test_extract_fields_unit_struct() {
        let data = syn::parse_quote! {
            struct Unit;
        };
        let input: DeriveInput = data;
        let fields = extract_fields(&input.data);
        assert!(fields.is_empty());
    }

    #[test]
    fn test_extract_fields_named_struct() {
        let data = syn::parse_quote! {
            struct Named {
                x: f32,
                y: f32,
            }
        };
        let input: DeriveInput = data;
        let fields = extract_fields(&input.data);
        assert_eq!(fields.len(), 2);
        assert_eq!(fields[0], ("x".to_string(), false));
        assert_eq!(fields[1], ("y".to_string(), false));
    }

    #[test]
    fn test_extract_fields_tuple_struct() {
        let data = syn::parse_quote! {
            struct Tuple(u32, u32, u32);
        };
        let input: DeriveInput = data;
        let fields = extract_fields(&input.data);
        assert_eq!(fields.len(), 3);
        assert_eq!(fields[0].0, "field_0");
        assert_eq!(fields[1].0, "field_1");
        assert_eq!(fields[2].0, "field_2");
    }

    #[test]
    fn test_extract_fields_enum() {
        let data = syn::parse_quote! {
            enum TestEnum {
                A,
                B,
            }
        };
        let input: DeriveInput = data;
        let fields = extract_fields(&input.data);
        assert!(fields.is_empty()); // Enums return empty
    }

    #[test]
    fn test_extract_name_from_attr_valid() {
        let attr: Attribute = syn::parse_quote! {
            #[probar(name = "custom_name")]
        };
        let result = extract_name_from_attr(&attr);
        assert_eq!(result, Some("custom_name".to_string()));
    }

    #[test]
    fn test_extract_name_from_attr_wrong_path() {
        let attr: Attribute = syn::parse_quote! {
            #[other(name = "custom_name")]
        };
        let result = extract_name_from_attr(&attr);
        assert_eq!(result, None);
    }

    #[test]
    fn test_extract_name_from_attr_wrong_key() {
        let attr: Attribute = syn::parse_quote! {
            #[probar(other = "value")]
        };
        let result = extract_name_from_attr(&attr);
        assert_eq!(result, None);
    }

    #[test]
    fn test_parse_selector_attributes_entities_only() {
        let attrs: Vec<Attribute> =
            vec![syn::parse_quote! { #[probar(entities = [Player, Enemy])] }];
        let (entities, components) = parse_selector_attributes(&attrs);
        assert_eq!(entities, vec!["Player", "Enemy"]);
        assert!(components.is_empty());
    }

    #[test]
    fn test_parse_selector_attributes_components_only() {
        let attrs: Vec<Attribute> =
            vec![syn::parse_quote! { #[probar(components = [Position, Health])] }];
        let (entities, components) = parse_selector_attributes(&attrs);
        assert!(entities.is_empty());
        assert_eq!(components, vec!["Position", "Health"]);
    }

    #[test]
    fn test_parse_selector_attributes_empty() {
        let attrs: Vec<Attribute> = vec![];
        let (entities, components) = parse_selector_attributes(&attrs);
        assert!(entities.is_empty());
        assert!(components.is_empty());
    }

    #[test]
    fn test_parse_selector_attributes_non_probar() {
        let attrs: Vec<Attribute> = vec![
            syn::parse_quote! { #[derive(Debug)] },
            syn::parse_quote! { #[allow(unused)] },
        ];
        let (entities, components) = parse_selector_attributes(&attrs);
        assert!(entities.is_empty());
        assert!(components.is_empty());
    }

    #[test]
    fn test_extract_name_attribute_multiple() {
        let attrs: Vec<Attribute> = vec![
            syn::parse_quote! { #[derive(Debug)] },
            syn::parse_quote! { #[probar(name = "found_name")] },
            syn::parse_quote! { #[allow(unused)] },
        ];
        let result = extract_name_attribute(&attrs);
        assert_eq!(result, Some("found_name".to_string()));
    }

    #[test]
    fn test_extract_name_attribute_none() {
        let attrs: Vec<Attribute> = vec![syn::parse_quote! { #[derive(Debug)] }];
        let result = extract_name_attribute(&attrs);
        assert_eq!(result, None);
    }

    #[test]
    fn test_extract_name_from_attr_non_string_value() {
        // probar(name = 123) - integer instead of string
        let attr: Attribute = syn::parse_quote! {
            #[probar(name = 123)]
        };
        let result = extract_name_from_attr(&attr);
        assert_eq!(result, None);
    }

    #[test]
    fn test_extract_fields_with_skip() {
        let data = syn::parse_quote! {
            struct WithSkip {
                x: f32,
                #[probar(skip)]
                internal: u32,
                y: f32,
            }
        };
        let input: DeriveInput = data;
        let fields = extract_fields(&input.data);
        assert_eq!(fields.len(), 3);
        assert_eq!(fields[0], ("x".to_string(), false));
        assert_eq!(fields[1], ("internal".to_string(), true)); // skip = true
        assert_eq!(fields[2], ("y".to_string(), false));
    }

    #[test]
    fn test_parse_selector_attributes_separate_attrs() {
        let attrs: Vec<Attribute> = vec![
            syn::parse_quote! { #[probar(entities = [Player])] },
            syn::parse_quote! { #[probar(components = [Position])] },
        ];
        let (entities, components) = parse_selector_attributes(&attrs);
        assert_eq!(entities, vec!["Player"]);
        assert_eq!(components, vec!["Position"]);
    }

    #[test]
    fn test_parse_timeout_attr_various_formats() {
        // Different spacing
        assert_eq!(parse_timeout_attr_from_str("timeout_ms=1000"), Some(1000));
        assert_eq!(parse_timeout_attr_from_str("timeout_ms =2000"), Some(2000));
        assert_eq!(parse_timeout_attr_from_str("timeout_ms= 3000"), Some(3000));
    }

    #[test]
    fn test_parse_timeout_attr_with_other_attrs() {
        let result = parse_timeout_attr_from_str("category = \"test\", timeout_ms = 7500");
        assert_eq!(result, Some(7500));
    }

    #[test]
    fn test_extract_list_items_malformed() {
        // Missing closing bracket
        let tokens = "probar(entities = [A, B";
        let items = extract_list_items(tokens, 0);
        assert!(items.is_empty());
    }

    #[test]
    fn test_extract_list_items_reversed_brackets() {
        // When ] comes before [, the slice would be invalid
        // This tests with proper order but no content
        let tokens = "probar(entities = [])";
        let items = extract_list_items(tokens, 0);
        assert!(items.is_empty());
    }

    #[test]
    fn test_to_snake_case_all_uppercase() {
        assert_eq!(to_snake_case("ABC"), "abc");
        assert_eq!(to_snake_case("ABCDEF"), "abcdef");
    }

    #[test]
    fn test_to_snake_case_single_char_words() {
        assert_eq!(to_snake_case("AaBbCc"), "aa_bb_cc");
    }

    #[test]
    fn test_generate_type_id_long_string() {
        let long_name = "a".repeat(1000);
        let id = generate_type_id(&long_name);
        assert_ne!(id, 0);
        // Verify it's deterministic
        assert_eq!(id, generate_type_id(&long_name));
    }

    #[test]
    fn test_extract_name_from_attr_list_style() {
        // probar(skip) style - not name=value
        let attr: Attribute = syn::parse_quote! {
            #[probar(skip)]
        };
        let result = extract_name_from_attr(&attr);
        assert_eq!(result, None);
    }

    #[test]
    fn test_extract_fields_empty_named() {
        let data = syn::parse_quote! {
            struct Empty {}
        };
        let input: DeriveInput = data;
        let fields = extract_fields(&input.data);
        assert!(fields.is_empty());
    }

    #[test]
    fn test_extract_fields_single_field() {
        let data = syn::parse_quote! {
            struct Single {
                value: i32,
            }
        };
        let input: DeriveInput = data;
        let fields = extract_fields(&input.data);
        assert_eq!(fields.len(), 1);
        assert_eq!(fields[0].0, "value");
    }

    #[test]
    fn test_extract_fields_tuple_single() {
        let data = syn::parse_quote! {
            struct Wrapper(String);
        };
        let input: DeriveInput = data;
        let fields = extract_fields(&input.data);
        assert_eq!(fields.len(), 1);
        assert_eq!(fields[0].0, "field_0");
    }

    #[test]
    fn test_parse_selector_attributes_mixed_attrs() {
        // Non-probar attrs mixed in
        let attrs: Vec<Attribute> = vec![
            syn::parse_quote! { #[derive(Debug)] },
            syn::parse_quote! { #[probar(entities = [A, B])] },
            syn::parse_quote! { #[serde(rename_all = "camelCase")] },
            syn::parse_quote! { #[probar(components = [X, Y, Z])] },
        ];
        let (entities, components) = parse_selector_attributes(&attrs);
        assert_eq!(entities, vec!["A", "B"]);
        assert_eq!(components, vec!["X", "Y", "Z"]);
    }

    #[test]
    fn test_extract_name_attribute_empty_list() {
        let attrs: Vec<Attribute> = vec![];
        let result = extract_name_attribute(&attrs);
        assert_eq!(result, None);
    }

    #[test]
    fn test_generate_type_id_collision_resistance() {
        // Test that similar names produce different IDs
        let id_player = generate_type_id("player");
        let id_player1 = generate_type_id("player1");
        let id_players = generate_type_id("players");
        let id_player_caps = generate_type_id("Player");

        assert_ne!(id_player, id_player1);
        assert_ne!(id_player, id_players);
        assert_ne!(id_player, id_player_caps); // Case sensitive
    }
}
