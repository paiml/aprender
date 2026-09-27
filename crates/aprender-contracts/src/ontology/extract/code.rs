//! ONT-001 §3.7, §5 ONT-4b2 — `extract:code`: the Rust symbols the bindings name become `ont:Symbol` nodes.
//!
//! The focus objects are the **bound** symbols — every `module_path` + `function` a `binding.yaml` under the
//! contract dir declares — not every function in the workspace (3.9 M lines; a graph of every `fn` would be a
//! second copy of the tree, and no contract constrains a symbol nothing binds). Each is resolved by a `syn`
//! **module-tree walk**: the crate root from the workspace's `Cargo.toml`s (`[package] name` and `[lib] name`,
//! both spellings), then one `mod` per path segment — inline, `mod x;` as `x.rs` / `x/mod.rs` / `#[path]`, or a
//! `pub use` re-export followed once — and finally a free `fn` or an `impl` method of that name. A last segment
//! that is no module may name the self type of an impl in the module reached (`Type::method`). Only the files
//! on the walk are parsed, so the cost is bindings × depth, not the workspace.
//!
//! **Fail-closed, said in the graph.** A symbol the walk cannot find is still a node — `sym:resolved false`
//! with `sym:unresolvedReason` naming the file and the segment — so a shape can require `sym:resolved true`
//! and a ghost binding (ONT-3a's thesis) is a violation naming the symbol, not a silently smaller graph.
//! Nothing is inferred: the module path is the binding's own text, normalized only by dropping a trailing
//! segment equal to the function (the registries write both `a::b::f` + `f` and `a::b` + `f`).
//!
//! IRI: `https://ont.paiml.dev/v1alpha1/symbol/<crate>::<module>::<function>`. Vocabulary `sym:*`. Attributes
//! are recorded by path (`inline`, `kernel`, …) so ONT-4c3 can type a `#[kernel]` symbol `ont:Kernel` from the
//! same walk. Deterministic: registries and bindings are visited in byte order and the graph is a set (R-15).

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::binding::{parse_binding, BindingRegistry, ImplStatus};
use crate::ontology::rdf::{iri, ont, Graph, Term, PROV_ENTITY, RDF_TYPE};

/// A `sym:*` vocabulary term.
#[must_use]
pub fn sym(name: &str) -> String {
    ont(&format!("sym/{name}"))
}

/// What the walk found for one bound symbol.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Resolved {
    /// Repo-relative file the item is defined in.
    pub file: String,
    /// `pub`, `pub(crate)`, `pub(super)`, `pub(in …)` or `private`.
    pub visibility: String,
    /// `fn` or `method`.
    pub kind: String,
    /// Attribute paths on the item, in source order (`inline`, `kernel`, `cfg`, …).
    pub attributes: Vec<String>,
    /// ONT-4c4: no `unsafe fn` signature and no `unsafe { … }` block in the body.
    pub unsafe_free: bool,
    /// ONT-4c4: no `get_unchecked` / `get_unchecked_mut` call in the body — every index is bounds-checked.
    pub bounds_checked: bool,
}

/// Why the walk stopped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Unresolved {
    pub reason: String,
}

/// Counts the extractor reports beside the graph.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CodeStats {
    pub registries: usize,
    pub symbols: usize,
    pub resolved: usize,
    pub unresolved: usize,
    /// Unresolved AND named by `binding-allowlist.yaml` — the debt ledger, reported apart from `resolved` and
    /// `unresolved`: `resolved + unresolved + allowlisted == symbols`. Its goal is 0.
    pub allowlisted: usize,
    pub files_parsed: usize,
}

/// One bound symbol as a registry states it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bound {
    pub contract: String,
    pub equation: String,
    pub module_path: String,
    pub function: String,
    pub status: ImplStatus,
}

/// Every `binding.yaml` under `contract_dir` (any depth, byte order), parsed. Walked here, not through
/// `collect_yaml_files`, which excludes registries by name because they are not contracts. A file that does not
/// parse as a registry is skipped: registries are validated by `pv validate` (BINDING-001..006), not here.
#[must_use]
pub fn registries(contract_dir: &Path) -> Vec<(PathBuf, BindingRegistry)> {
    let mut files = Vec::new();
    let mut stack = vec![contract_dir.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for path in entries.flatten().map(|e| e.path()) {
            if path.is_dir() {
                stack.push(path);
            } else if path.file_name().is_some_and(|n| n == "binding.yaml") {
                files.push(path);
            }
        }
    }
    files.sort();
    files
        .into_iter()
        .filter_map(|f| parse_binding(&f).ok().map(|r| (f, r)))
        .collect()
}

/// The bound symbols of one registry, with the module path normalized (a trailing segment equal to the function
/// is dropped). Bindings without a `function` bind nothing and are skipped.
#[must_use]
pub fn bound_of(registry: &BindingRegistry) -> Vec<Bound> {
    registry
        .bindings
        .iter()
        .filter_map(|b| {
            let function = b.function.clone()?;
            let mut module_path = b
                .module_path
                .clone()
                .unwrap_or_else(|| registry.target_crate.clone());
            // `function: tokenize::run` under `module_path: apr_cli::commands::tokenize` — the registries
            // also write a qualified function; the qualifier joins the module path (once, if it is not
            // already its tail) and the function is the last segment.
            let function = match function.rsplit_once("::") {
                Some((qualifier, name)) => {
                    if !module_path.ends_with(&format!("::{qualifier}")) && module_path != qualifier
                    {
                        module_path = format!("{module_path}::{qualifier}");
                    }
                    name.to_string()
                }
                None => function,
            };
            let module_path = module_path
                .strip_suffix(&format!("::{function}"))
                .unwrap_or(&module_path)
                .to_string();
            Some(Bound {
                contract: contract_stem(&b.contract).to_string(),
                equation: b.equation.clone(),
                module_path,
                function,
                status: b.status,
            })
        })
        .collect()
}

/// The contract IRI a binding row names: a contract's IRI is its file stem, so a
/// row spelled `../gguf-format-safety-v1.yaml` (a registry in a subdirectory
/// pointing up) names `contract/gguf-format-safety-v1`, not `contract/..%2F…`.
fn contract_stem(contract: &str) -> &str {
    let id = crate::binding::normalize_contract_id(contract);
    id.rsplit('/').next().unwrap_or(id)
}

/// The crates of a workspace: every `Cargo.toml` under `root` (skipping build and vcs dirs), keyed by both the
/// package name and the `[lib] name`, `-` spelled `_`, mapped to the crate root files that claim the name —
/// more than one when a facade package and the library it re-exports share it (aprender's root `aprender`
/// is `pub use aprender_ml::*` over `crates/aprender-core`, whose `[lib] name` is also `aprender`). The walk
/// tries each in byte order of the manifest path and keeps the first that resolves.
#[derive(Debug, Default)]
pub struct Workspace {
    pub root: PathBuf,
    pub crates: BTreeMap<String, Vec<PathBuf>>,
}

impl Workspace {
    /// Scan `root` for manifests. A manifest without a `[package]` name (a virtual workspace root) contributes
    /// nothing.
    #[must_use]
    pub fn scan(root: &Path) -> Self {
        let mut ws = Self {
            root: root.to_path_buf(),
            crates: BTreeMap::new(),
        };
        let mut stack = vec![root.to_path_buf()];
        while let Some(dir) = stack.pop() {
            let Ok(entries) = std::fs::read_dir(&dir) else {
                continue;
            };
            let mut children: Vec<PathBuf> = entries.flatten().map(|e| e.path()).collect();
            children.sort();
            for path in children {
                if path.is_dir() {
                    let skip = matches!(
                        path.file_name().and_then(|n| n.to_str()),
                        Some("target" | ".git" | ".lake" | "node_modules")
                    );
                    if !skip {
                        stack.push(path);
                    }
                } else if path.file_name().is_some_and(|n| n == "Cargo.toml") {
                    ws.index_manifest(&path);
                }
            }
        }
        ws
    }

    fn index_manifest(&mut self, manifest: &Path) {
        let Ok(text) = std::fs::read_to_string(manifest) else {
            return;
        };
        let m = manifest_names(&text);
        let dir = manifest.parent().unwrap_or(manifest);
        let lib_path = m
            .lib_path
            .map_or_else(|| dir.join("src/lib.rs"), |p| dir.join(p));
        for name in [m.package, m.lib].into_iter().flatten() {
            let roots = self.crates.entry(name.replace('-', "_")).or_default();
            if !roots.contains(&lib_path) {
                roots.push(lib_path.clone());
            }
        }
    }
}

#[derive(Debug, Default)]
struct ManifestNames {
    package: Option<String>,
    lib: Option<String>,
    lib_path: Option<String>,
}

/// `[package] name`, `[lib] name` and `[lib] path` by a section-aware line scan — the two keys this walk needs,
/// read without a TOML crate.
fn manifest_names(text: &str) -> ManifestNames {
    let mut out = ManifestNames::default();
    let mut section = String::new();
    for line in text.lines() {
        let t = line.trim();
        if t.starts_with('[') && t.ends_with(']') {
            section = t.trim_matches(['[', ']']).trim().to_string();
            continue;
        }
        let Some((k, v)) = t.split_once('=') else {
            continue;
        };
        let k = k.trim();
        let v = v.trim().trim_matches('"').to_string();
        match (section.as_str(), k) {
            ("package", "name") => out.package = Some(v),
            ("lib", "name") => out.lib = Some(v),
            ("lib", "path") => out.lib_path = Some(v),
            _ => {}
        }
    }
    out
}

/// The `syn` walker with a per-file parse cache.
pub struct Resolver<'a> {
    ws: &'a Workspace,
    cache: BTreeMap<PathBuf, Option<std::rc::Rc<syn::File>>>,
}

/// One module on the walk: its items, the file they came from, and the directory its child files live in.
struct Module {
    items: Vec<syn::Item>,
    file: PathBuf,
    child_dir: PathBuf,
}

impl<'a> Resolver<'a> {
    #[must_use]
    pub fn new(ws: &'a Workspace) -> Self {
        Self {
            ws,
            cache: BTreeMap::new(),
        }
    }

    /// Files parsed so far (each once).
    #[must_use]
    pub fn files_parsed(&self) -> usize {
        self.cache.values().filter(|f| f.is_some()).count()
    }

    fn parse(&mut self, file: &Path) -> Option<std::rc::Rc<syn::File>> {
        if let Some(hit) = self.cache.get(file) {
            return hit.clone();
        }
        let parsed = std::fs::read_to_string(file)
            .ok()
            .and_then(|src| syn::parse_file(&src).ok())
            .map(std::rc::Rc::new);
        self.cache.insert(file.to_path_buf(), parsed.clone());
        parsed
    }

    /// The crate root files that claim `krate`, in manifest order.
    fn crate_roots(&self, krate: &str) -> Result<Vec<PathBuf>, Unresolved> {
        let roots: Vec<PathBuf> = self
            .ws
            .crates
            .get(&krate.replace('-', "_"))
            .cloned()
            .unwrap_or_default();
        if roots.is_empty() {
            return Err(Unresolved {
                reason: format!("crate `{krate}` is not a workspace member"),
            });
        }
        Ok(roots)
    }

    fn file_module(&mut self, file: &Path, child_dir: PathBuf) -> Result<Module, Unresolved> {
        let Some(ast) = self.parse(file) else {
            return Err(Unresolved {
                reason: format!("`{}` is missing or does not parse", self.rel(file)),
            });
        };
        let dir = file.parent().unwrap_or(file).to_path_buf();
        Ok(Module {
            items: self.expand_includes(&ast.items, &dir, 0),
            file: file.to_path_buf(),
            child_dir,
        })
    }

    /// `items` with every item-position `include!("x.rs")` replaced by the items of `x.rs`, relative to `dir`
    /// (the including file's directory, as rustc resolves it). The tree splits large modules this way, and a
    /// walk that stopped at the macro would call every fn in the included file a ghost. An include that is
    /// missing or does not parse is kept as the macro, so it resolves nothing.
    fn expand_includes(&mut self, items: &[syn::Item], dir: &Path, depth: usize) -> Vec<syn::Item> {
        let mut out = Vec::with_capacity(items.len());
        for item in items {
            let target = match item {
                syn::Item::Macro(m) if depth < 8 && m.mac.path.is_ident("include") => m
                    .mac
                    .parse_body::<syn::LitStr>()
                    .ok()
                    .map(|l| dir.join(l.value())),
                _ => None,
            };
            match target.and_then(|t| self.parse(&t).map(|ast| (t, ast))) {
                Some((t, ast)) => {
                    let sub = t.parent().unwrap_or(dir).to_path_buf();
                    out.extend(self.expand_includes(&ast.items, &sub, depth + 1));
                }
                None => out.push(item.clone()),
            }
        }
        out
    }

    /// Resolve `module_path::function` to its definition. `depth` bounds re-export chasing.
    pub fn resolve(&mut self, module_path: &str, function: &str) -> Result<Resolved, Unresolved> {
        self.resolve_depth(module_path, function, 0)
    }

    fn resolve_depth(
        &mut self,
        module_path: &str,
        function: &str,
        depth: usize,
    ) -> Result<Resolved, Unresolved> {
        if depth > 8 {
            return Err(Unresolved {
                reason: format!("`{module_path}::{function}`: re-export chain deeper than 8"),
            });
        }
        let mut segs = module_path.split("::").filter(|s| !s.is_empty());
        let Some(krate) = segs.next() else {
            return Err(Unresolved {
                reason: "empty module path".to_string(),
            });
        };
        let segs: Vec<&str> = segs.collect();
        let mut last = Unresolved {
            reason: String::new(),
        };
        for root in self.crate_roots(krate)? {
            match self.resolve_in_root(&root, &segs, function, depth) {
                Ok(r) => return Ok(r),
                Err(e) => last = e,
            }
        }
        Err(last)
    }

    /// The walk from one crate root file.
    fn resolve_in_root(
        &mut self,
        root: &Path,
        segs: &[&str],
        function: &str,
        depth: usize,
    ) -> Result<Resolved, Unresolved> {
        let mut module = self.file_module(root, root.parent().unwrap_or(root).to_path_buf())?;
        for (i, seg) in segs.iter().enumerate() {
            // `Type::method`: the last segment may name an impl's self type, not a module. The impl here wins
            // over a `use super::Type` here — an impl can sit in any module, and following the `use` would
            // walk back to where the type is defined and miss this impl.
            if i + 1 == segs.len() {
                if let Some(mut r) = find_impl_method(&module.items, seg, function) {
                    r.file = self.rel(&module.file);
                    return Ok(r);
                }
            }
            let step = self.step(&module, seg)?;
            match step {
                Step::Module(next) => module = next,
                Step::ReExport(target) => {
                    let rest = segs[i + 1..].join("::");
                    let path = if rest.is_empty() {
                        target
                    } else {
                        format!("{target}::{rest}")
                    };
                    return self.resolve_depth(&path, function, depth + 1);
                }
            }
        }
        match find_item(&module.items, function) {
            Some(mut r) => {
                r.file = self.rel(&module.file);
                Ok(r)
            }
            None => match use_target(&module.items, function) {
                Some(target) => {
                    let target = absolute_use(&target, &module, self.ws);
                    self.resolve_by_use(&target, depth)
                }
                None => Err(Unresolved {
                    reason: format!(
                        "no `fn {function}` (free or in an impl) in `{}`",
                        self.rel(&module.file)
                    ),
                }),
            },
        }
    }

    fn resolve_by_use(&mut self, target: &str, depth: usize) -> Result<Resolved, Unresolved> {
        let (path, name) = target.rsplit_once("::").ok_or_else(|| Unresolved {
            reason: format!("re-export `{target}` has no module path"),
        })?;
        self.resolve_depth(path, name, depth + 1)
    }

    fn step(&mut self, module: &Module, seg: &str) -> Result<Step, Unresolved> {
        for item in &module.items {
            if let syn::Item::Mod(m) = item {
                if m.ident != seg {
                    continue;
                }
                if let Some((_, content)) = &m.content {
                    let dir = module.file.parent().unwrap_or(&module.file).to_path_buf();
                    return Ok(Step::Module(Module {
                        items: self.expand_includes(content, &dir, 0),
                        file: module.file.clone(),
                        child_dir: module.child_dir.join(seg),
                    }));
                }
                let (file, child_dir) = child_file(&module.child_dir, seg, &m.attrs)?;
                return self.file_module(&file, child_dir).map(Step::Module);
            }
        }
        if let Some(target) = use_target(&module.items, seg) {
            let target = absolute_use(&target, module, self.ws);
            return Ok(Step::ReExport(target));
        }
        Err(Unresolved {
            reason: format!(
                "no `mod {seg}` or `use … {seg}` in `{}`",
                self.rel(&module.file)
            ),
        })
    }

    fn rel(&self, file: &Path) -> String {
        file.strip_prefix(&self.ws.root)
            .unwrap_or(file)
            .to_string_lossy()
            .replace('\\', "/")
    }
}

enum Step {
    Module(Module),
    ReExport(String),
}

/// `mod seg;` → `#[path = "…"]` relative to the child dir, else `seg.rs`, else `seg/mod.rs`. The child dir of
/// the new module is `<dir>/seg/` either way.
fn child_file(
    child_dir: &Path,
    seg: &str,
    attrs: &[syn::Attribute],
) -> Result<(PathBuf, PathBuf), Unresolved> {
    for a in attrs {
        if a.path().is_ident("path") {
            if let syn::Meta::NameValue(nv) = &a.meta {
                if let syn::Expr::Lit(syn::ExprLit {
                    lit: syn::Lit::Str(s),
                    ..
                }) = &nv.value
                {
                    let f = child_dir.join(s.value());
                    return Ok((f.clone(), f.parent().unwrap_or(&f).to_path_buf()));
                }
            }
        }
    }
    let flat = child_dir.join(format!("{seg}.rs"));
    if flat.is_file() {
        return Ok((flat, child_dir.join(seg)));
    }
    let nested = child_dir.join(seg).join("mod.rs");
    if nested.is_file() {
        return Ok((nested, child_dir.join(seg)));
    }
    Err(Unresolved {
        reason: format!(
            "`mod {seg};` declared but neither `{}` nor `{}` exists",
            flat.display(),
            nested.display()
        ),
    })
}

/// A free `fn name` or an `impl … { fn name }` among `items`.
fn find_item(items: &[syn::Item], name: &str) -> Option<Resolved> {
    items.iter().find_map(|item| match item {
        syn::Item::Fn(f) if f.sig.ident == name => {
            Some(found("fn", &f.vis, &f.attrs, &f.sig, &f.block))
        }
        syn::Item::Impl(im) => find_method(&im.items, name),
        _ => None,
    })
}

/// The `fn name` of an `impl Type` or `impl Trait for Type` block among `items`.
fn find_impl_method(items: &[syn::Item], ty: &str, name: &str) -> Option<Resolved> {
    items.iter().find_map(|item| match item {
        syn::Item::Impl(im)
            if matches!(&*im.self_ty, syn::Type::Path(p)
                if p.path.segments.last().is_some_and(|s| s.ident == ty)) =>
        {
            find_method(&im.items, name)
        }
        _ => None,
    })
}

/// The `fn name` of one `impl` block.
fn find_method(items: &[syn::ImplItem], name: &str) -> Option<Resolved> {
    items.iter().find_map(|ii| match ii {
        syn::ImplItem::Fn(f) if f.sig.ident == name => {
            Some(found("method", &f.vis, &f.attrs, &f.sig, &f.block))
        }
        _ => None,
    })
}

/// A resolution with the file filled in by the caller that knows it.
fn found(
    kind: &str,
    vis: &syn::Visibility,
    attrs: &[syn::Attribute],
    sig: &syn::Signature,
    block: &syn::Block,
) -> Resolved {
    let mut body = BodySafety::default();
    syn::visit::Visit::visit_block(&mut body, block);
    Resolved {
        file: String::new(),
        visibility: visibility_of(vis),
        kind: kind.to_string(),
        attributes: attr_paths(attrs),
        unsafe_free: sig.unsafety.is_none() && body.unsafe_blocks == 0,
        bounds_checked: body.unchecked_calls == 0,
    }
}

/// What ONT-4c4's `kernel-safety` reads off a function body: `unsafe` blocks and unchecked indexing.
#[derive(Default)]
struct BodySafety {
    unsafe_blocks: usize,
    unchecked_calls: usize,
}

impl<'ast> syn::visit::Visit<'ast> for BodySafety {
    fn visit_expr_unsafe(&mut self, e: &'ast syn::ExprUnsafe) {
        self.unsafe_blocks += 1;
        syn::visit::visit_expr_unsafe(self, e);
    }
    fn visit_expr_method_call(&mut self, e: &'ast syn::ExprMethodCall) {
        if is_unchecked(&e.method.to_string()) {
            self.unchecked_calls += 1;
        }
        syn::visit::visit_expr_method_call(self, e);
    }
    fn visit_expr_path(&mut self, e: &'ast syn::ExprPath) {
        if e.path
            .segments
            .last()
            .is_some_and(|s| is_unchecked(&s.ident.to_string()))
        {
            self.unchecked_calls += 1;
        }
        syn::visit::visit_expr_path(self, e);
    }
}

fn is_unchecked(name: &str) -> bool {
    matches!(name, "get_unchecked" | "get_unchecked_mut")
}

/// The path a `use` item binds `name` to, as written (`crate::a::b`, `self::x`, `super::y`, `other::z`), when
/// one does; renames (`as`) are honoured.
fn use_target(items: &[syn::Item], name: &str) -> Option<String> {
    for item in items {
        if let syn::Item::Use(u) = item {
            if let Some(t) = use_tree_target(&u.tree, name, "") {
                return Some(t);
            }
        }
    }
    None
}

fn use_tree_target(tree: &syn::UseTree, name: &str, prefix: &str) -> Option<String> {
    let join = |p: &str, s: &str| {
        if p.is_empty() {
            s.to_string()
        } else {
            format!("{p}::{s}")
        }
    };
    match tree {
        syn::UseTree::Path(p) => {
            use_tree_target(&p.tree, name, &join(prefix, &p.ident.to_string()))
        }
        syn::UseTree::Name(n) if n.ident == name => Some(join(prefix, &n.ident.to_string())),
        syn::UseTree::Rename(r) if r.rename == name => Some(join(prefix, &r.ident.to_string())),
        syn::UseTree::Group(g) => g
            .items
            .iter()
            .find_map(|t| use_tree_target(t, name, prefix)),
        _ => None,
    }
}

/// Make a `use` path absolute from the crate root: `crate::` → the crate's name; `self::` → this module;
/// `super::` → its parent; a bare leading segment is first tried as a workspace crate, else as a sibling module.
fn absolute_use(target: &str, module: &Module, ws: &Workspace) -> String {
    let here = module_path_of(module, ws);
    let krate = here.split("::").next().unwrap_or_default().to_string();
    let (head, rest) = target.split_once("::").unwrap_or((target, ""));
    let base = match head {
        "crate" => krate,
        "self" => here.clone(),
        "super" => here
            .rsplit_once("::")
            .map_or(here.clone(), |(p, _)| p.to_string()),
        other if ws.crates.contains_key(&other.replace('-', "_")) => other.to_string(),
        other => format!("{here}::{other}"),
    };
    if rest.is_empty() {
        base
    } else {
        format!("{base}::{rest}")
    }
}

/// The module path of a module on the walk, from its child dir relative to the crate's `src/`:
/// `<crate>` for the root, `<crate>::a::b` below it.
fn module_path_of(module: &Module, ws: &Workspace) -> String {
    // The crate whose `src/` is the LONGEST prefix of the module's dir owns it (a workspace root package's
    // `src/` is a prefix of nothing under `crates/`, but nested crates exist).
    let mut best: Option<(usize, String)> = None;
    for (name, roots) in &ws.crates {
        for root in roots {
            let Some(src) = root.parent() else { continue };
            let Ok(rel) = module.child_dir.strip_prefix(src) else {
                continue;
            };
            let depth = src.components().count();
            if best.as_ref().is_some_and(|(d, _)| *d >= depth) {
                continue;
            }
            let tail: Vec<String> = rel
                .components()
                .map(|c| c.as_os_str().to_string_lossy().to_string())
                .collect();
            let path = if tail.is_empty() {
                name.clone()
            } else {
                format!("{name}::{}", tail.join("::"))
            };
            best = Some((depth, path));
        }
    }
    best.map(|(_, p)| p).unwrap_or_default()
}

fn visibility_of(v: &syn::Visibility) -> String {
    match v {
        syn::Visibility::Public(_) => "pub".to_string(),
        syn::Visibility::Restricted(r) => {
            let p = r
                .path
                .segments
                .iter()
                .map(|s| s.ident.to_string())
                .collect::<Vec<_>>()
                .join("::");
            format!("pub({p})")
        }
        syn::Visibility::Inherited => "private".to_string(),
    }
}

fn attr_paths(attrs: &[syn::Attribute]) -> Vec<String> {
    attrs
        .iter()
        .map(|a| {
            a.path()
                .segments
                .iter()
                .map(|s| s.ident.to_string())
                .collect::<Vec<_>>()
                .join("::")
        })
        .collect()
}

/// The symbol IRI of a bound symbol.
#[must_use]
pub fn symbol_iri(b: &Bound) -> String {
    iri("symbol", &format!("{}::{}", b.module_path, b.function))
}

/// The allowlist's file name under the contract dir (ONT-001 §4 artifact table).
pub const ALLOWLIST_FILE: &str = "binding-allowlist.yaml";
/// The allowlist's schema id; a file carrying any other is read as empty.
pub const ALLOWLIST_SCHEMA: &str = "ont.paiml.dev/binding-allowlist/v1alpha1";

/// One `binding-allowlist.yaml` entry: a bound symbol known not to resolve, why, and the ticket that retires it.
/// `symbol` is `<module_path>::<function>` after [`bound_of`]'s normalization — the symbol IRI's tail.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AllowEntry {
    pub symbol: String,
    pub reason: String,
    pub ticket: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AllowFile {
    schema: String,
    #[serde(default)]
    entries: Vec<AllowEntry>,
}

/// `binding-allowlist.yaml` under `contract_dir`, keyed by symbol. Absent, unparsable or of another schema, it is
/// EMPTY: the allowlist only ever excuses, so a broken one excuses nothing and every ghost stays a violation
/// (fail-closed). A duplicate symbol keeps its first entry.
#[must_use]
pub fn allowlist(contract_dir: &Path) -> BTreeMap<String, AllowEntry> {
    let Ok(text) = std::fs::read_to_string(contract_dir.join(ALLOWLIST_FILE)) else {
        return BTreeMap::new();
    };
    parse_allowlist(&text)
}

fn parse_allowlist(text: &str) -> BTreeMap<String, AllowEntry> {
    let mut out = BTreeMap::new();
    let Ok(file) = serde_yaml::from_str::<AllowFile>(text) else {
        return out;
    };
    if file.schema != ALLOWLIST_SCHEMA {
        return out;
    }
    for e in file.entries {
        out.entry(e.symbol.clone()).or_insert(e);
    }
    out
}

/// The allowlist facts on `s`: typed `ont:AllowlistedSymbol`, with the entry's reason and ticket. The
/// `allowlisted-symbols-ticketed` shape reads them: an entry for a symbol that resolves, that no binding names, or
/// that carries no reason or no ticket number is a violation — the ledger cannot hold a stale line.
fn emit_allow(g: &mut Graph, s: &str, e: &AllowEntry) {
    g.insert(s.to_string(), RDF_TYPE, Term::iri(ont("AllowlistedSymbol")));
    g.insert(
        s.to_string(),
        sym("allowlistReason"),
        Term::string(&e.reason),
    );
    g.insert(
        s.to_string(),
        sym("allowlistTicket"),
        Term::string(&e.ticket),
    );
}

/// One bound symbol into `g`, resolved or not.
pub fn emit(g: &mut Graph, b: &Bound, found: &Result<Resolved, Unresolved>) {
    emit_as(g, b, found, None);
}

/// [`emit`], with the symbol's allowlist entry if it has one. An allowlisted symbol that does NOT resolve is typed
/// `ont:AllowlistedSymbol` INSTEAD of `ont:Symbol`, so `bound-symbols-resolve` does not target it; one that does
/// resolve keeps `ont:Symbol` and gains `ont:AllowlistedSymbol`, so its stale entry is a violation.
fn emit_as(
    g: &mut Graph,
    b: &Bound,
    found: &Result<Resolved, Unresolved>,
    allow: Option<&AllowEntry>,
) {
    let s = symbol_iri(b);
    match (allow, found) {
        (Some(e), Err(_)) => emit_allow(g, &s, e),
        (Some(e), Ok(_)) => {
            emit_allow(g, &s, e);
            g.insert(s.clone(), RDF_TYPE, Term::iri(ont("Symbol")));
        }
        (None, _) => g.insert(s.clone(), RDF_TYPE, Term::iri(ont("Symbol"))),
    }
    g.insert(s.clone(), RDF_TYPE, Term::iri(PROV_ENTITY));
    let krate = b.module_path.split("::").next().unwrap_or_default();
    g.insert(s.clone(), sym("crate"), Term::string(krate));
    g.insert(s.clone(), sym("module"), Term::string(&b.module_path));
    g.insert(s.clone(), sym("name"), Term::string(&b.function));
    g.insert(
        s.clone(),
        sym("implements"),
        Term::iri(iri("contract", &b.contract)),
    );
    g.insert(s.clone(), sym("equation"), Term::string(&b.equation));
    g.insert(
        s.clone(),
        sym("bindingStatus"),
        Term::string(format!("{:?}", b.status).to_lowercase()),
    );
    match found {
        Ok(r) => {
            g.insert(s.clone(), sym("resolved"), Term::boolean(true));
            g.insert(s.clone(), sym("file"), Term::string(&r.file));
            g.insert(s.clone(), sym("visibility"), Term::string(&r.visibility));
            g.insert(s.clone(), sym("kind"), Term::string(&r.kind));
            for a in &r.attributes {
                g.insert(s.clone(), sym("attribute"), Term::string(a));
            }
            // ONT-4c4: only a `#[kernel]` carries the body facts — `kernel-safety` reads them; no other shape does.
            if r.attributes.iter().any(|a| a == "kernel") {
                g.insert(s.clone(), sym("unsafeFree"), Term::boolean(r.unsafe_free));
                g.insert(
                    s.clone(),
                    sym("boundsChecked"),
                    Term::boolean(r.bounds_checked),
                );
            }
        }
        Err(u) => {
            g.insert(s.clone(), sym("resolved"), Term::boolean(false));
            g.insert(s, sym("unresolvedReason"), Term::string(&u.reason));
        }
    }
}

/// Every bound symbol of every registry under `contract_dir`, resolved against the workspace at the contract
/// dir's parent, into `g`.
pub fn extract(contract_dir: &Path, g: &mut Graph) -> CodeStats {
    let root_buf = super::repo_root(contract_dir);
    let root = root_buf.as_path();
    let ws = Workspace::scan(root);
    let mut resolver = Resolver::new(&ws);
    let mut stats = CodeStats::default();
    let allow = allowlist(contract_dir);
    let mut named = BTreeSet::new();
    for (_file, registry) in registries(contract_dir) {
        stats.registries += 1;
        for b in bound_of(&registry) {
            let found = resolver.resolve(&b.module_path, &b.function);
            let key = format!("{}::{}", b.module_path, b.function);
            let entry = allow.get(&key);
            match (&found, entry) {
                (Ok(_), _) => stats.resolved += 1,
                (Err(_), Some(_)) => stats.allowlisted += 1,
                (Err(_), None) => stats.unresolved += 1,
            }
            stats.symbols += 1;
            if entry.is_some() {
                named.insert(key);
            }
            emit_as(g, &b, &found, entry);
        }
    }
    // An entry no binding names: a node with no `sym:implements`, which the shape refuses.
    for (key, e) in allow.iter().filter(|(k, _)| !named.contains(*k)) {
        emit_allow(g, &iri("symbol", key), e);
    }
    stats.files_parsed = resolver.files_parsed();
    stats
}

/// Positive control (`pc_extract.code`, run in memory every gate run): the resolver must find a function that
/// is there and must refuse one that is not — a walker that resolved everything would make every ghost binding
/// a Pass.
#[must_use]
pub fn positive_control() -> bool {
    let src = "pub mod m { pub fn present() {} }\npub use m::present as alias;";
    let Ok(ast) = syn::parse_file(src) else {
        return false;
    };
    let present = find_item(&ast.items, "present").is_none()
        && match ast.items.first() {
            Some(syn::Item::Mod(m)) => m
                .content
                .as_ref()
                .is_some_and(|(_, items)| find_item(items, "present").is_some()),
            _ => false,
        };
    let ghost =
        find_item(&ast.items, "absent").is_none() && use_target(&ast.items, "absent").is_none();
    let aliased = use_target(&ast.items, "alias").as_deref() == Some("m::present");
    present && ghost && aliased
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/ont/code-bound")
    }

    #[test]
    fn the_manifest_scan_indexes_package_and_lib_names() {
        let ws = Workspace::scan(&fixture());
        assert!(ws.crates.contains_key("kern"), "{:?}", ws.crates);
        assert!(ws.crates.contains_key("kern_crate"), "{:?}", ws.crates);
        assert!(ws.crates["kern"][0].ends_with("crates/kern/src/lib.rs"));
    }

    #[test]
    fn the_walk_resolves_files_inline_modules_methods_and_re_exports_and_refuses_a_ghost() {
        let ws = Workspace::scan(&fixture());
        let mut r = Resolver::new(&ws);
        let softmax = r
            .resolve("kern::nn::functional", "softmax")
            .expect("softmax");
        assert_eq!(softmax.file, "crates/kern/src/nn/functional.rs");
        assert_eq!(softmax.visibility, "pub");
        assert_eq!(softmax.kind, "fn");
        assert_eq!(softmax.attributes, vec!["inline".to_string()]);
        let relu = r.resolve("kern::nn::functional", "relu").expect("relu");
        assert_eq!(relu.visibility, "pub(crate)");
        let forward = r.resolve("kern::nn", "forward").expect("forward");
        assert_eq!(forward.kind, "method");
        assert_eq!(forward.file, "crates/kern/src/nn.rs");
        let helper = r
            .resolve("kern", "helper")
            .expect("helper via pub use … as");
        assert_eq!(helper.file, "crates/kern/src/lib.rs");
        assert_eq!(helper.visibility, "pub");
        let kernel = r
            .resolve("kern::nn::functional", "gated_rmsnorm")
            .expect("kernel");
        assert_eq!(kernel.attributes, vec!["kernel".to_string()]);
        let ghost = r
            .resolve("kern::nn::functional", "no_such_function")
            .unwrap_err();
        assert!(
            ghost.reason.contains("no `fn no_such_function`"),
            "{}",
            ghost.reason
        );
        assert!(
            ghost.reason.contains("crates/kern/src/nn/functional.rs"),
            "{}",
            ghost.reason
        );
        let no_crate = r.resolve("nowhere::x", "f").unwrap_err();
        assert!(no_crate.reason.contains("not a workspace member"));
        let no_mod = r.resolve("kern::nope", "f").unwrap_err();
        assert!(no_mod.reason.contains("no `mod nope`"), "{}", no_mod.reason);
        assert_eq!(r.files_parsed(), 3);
    }

    #[test]
    fn the_walk_follows_include_relative_to_the_including_file() {
        let root =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/ont/code-include");
        let ws = Workspace::scan(&root);
        let mut r = Resolver::new(&ws);
        r.resolve("inc", "from_a").expect("fn in an included file");
        // split_a.rs includes nested/split_b.rs: the path is relative to split_a.rs, not to lib.rs.
        let step = r
            .resolve("inc::Engine", "step")
            .expect("method in a nested include");
        assert_eq!(step.kind, "method");
        r.resolve("inc::inner", "in_inner")
            .expect("include inside an inline mod");
        let ghost = r.resolve("inc", "not_there").unwrap_err();
        assert!(
            ghost.reason.contains("no `fn not_there`"),
            "{}",
            ghost.reason
        );
    }

    #[test]
    fn a_method_resolves_in_the_impl_beside_a_use_of_its_type() {
        let root =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/ont/code-include");
        let ws = Workspace::scan(&root);
        let mut r = Resolver::new(&ws);
        let exec = r
            .resolve("inc::batch::execute::Batch", "execute")
            .expect("Batch::execute in the child module");
        assert_eq!(exec.file, "crates/inc/src/batch/execute.rs");
        assert_eq!(exec.kind, "method");
        // A method in neither impl is still a ghost, and the defining module's impl still answers for itself.
        assert!(r.resolve("inc::batch::execute::Batch", "absent").is_err());
        assert!(r.resolve("inc::batch::Batch", "new").is_ok());
        assert!(r.resolve("inc::batch::Batch", "execute").is_err());
    }

    #[test]
    fn a_type_qualified_method_resolves_in_its_impl_and_only_there() {
        let ws = Workspace::scan(&fixture());
        let mut r = Resolver::new(&ws);
        let forward = r
            .resolve("kern::nn::Layer", "forward")
            .expect("Layer::forward");
        assert_eq!(forward.kind, "method");
        assert_eq!(forward.file, "crates/kern/src/nn.rs");
        let wrong_type = r.resolve("kern::nn::Other", "forward").unwrap_err();
        assert!(
            wrong_type.reason.contains("no `mod Other`"),
            "{}",
            wrong_type.reason
        );
        assert!(r.resolve("kern::nn::Layer", "backward").is_err());
        // Only the last segment may be a type: `Layer::x::forward` is still a ghost.
        assert!(r.resolve("kern::nn::Layer::x", "forward").is_err());
    }

    #[test]
    fn the_graph_carries_every_bound_symbol_resolved_or_not_and_is_deterministic() {
        let dir = fixture().join("contracts");
        let mut g = Graph::new();
        let stats = extract(&dir, &mut g);
        assert_eq!(stats.registries, 1);
        assert_eq!(stats.symbols, 6);
        assert_eq!(stats.resolved, 5);
        assert_eq!(stats.unresolved, 1);
        let nt = g.to_ntriples();
        assert!(nt.contains("/symbol/kern::nn::functional::softmax> <https://ont.paiml.dev/v1alpha1/sym/resolved> \"true\""), "{nt}");
        assert!(nt.contains("/symbol/kern::nn::functional::no_such_function> <https://ont.paiml.dev/v1alpha1/sym/resolved> \"false\""), "{nt}");
        assert!(
            nt.contains("sym/unresolvedReason> \"no `fn no_such_function`"),
            "{nt}"
        );
        assert!(
            nt.contains(
                "sym/implements> <https://ont.paiml.dev/v1alpha1/contract/softmax-kernel-v1>"
            ),
            "{nt}"
        );
        assert!(nt.contains("sym/attribute> \"kernel\""), "{nt}");
        assert!(!nt.contains("_:"));
        let mut g2 = Graph::new();
        extract(&dir, &mut g2);
        assert_eq!(nt, g2.to_ntriples());
    }

    #[test]
    fn a_relative_contract_path_names_the_contract_by_its_stem() {
        assert_eq!(
            contract_stem("../gguf-format-safety-v1.yaml"),
            "gguf-format-safety-v1"
        );
        assert_eq!(contract_stem("../../a/b-v1"), "b-v1");
        assert_eq!(contract_stem("softmax-kernel-v1.yaml"), "softmax-kernel-v1");
        assert_eq!(contract_stem("softmax-kernel-v1"), "softmax-kernel-v1");
    }

    #[test]
    fn the_allowlist_reads_only_its_own_schema_and_a_broken_one_excuses_nothing() {
        let ok = "schema: ont.paiml.dev/binding-allowlist/v1alpha1\nentries:\n  - {symbol: a::f, reason: r, ticket: '#1'}\n  - {symbol: a::f, reason: second, ticket: '#2'}\n";
        let got = parse_allowlist(ok);
        assert_eq!(got.len(), 1);
        assert_eq!(got["a::f"].reason, "r", "a duplicate keeps its first entry");
        let other = ok.replace("v1alpha1", "v9");
        assert!(parse_allowlist(&other).is_empty(), "another schema");
        let extra = ok.replace("ticket: '#1'", "ticket: '#1', owner: x");
        assert!(parse_allowlist(&extra).is_empty(), "an unknown key");
        assert!(parse_allowlist("entries: [").is_empty(), "unparsable");
        assert!(allowlist(Path::new("/nonexistent/contracts")).is_empty());
    }

    #[test]
    fn an_allowlisted_ghost_is_retyped_and_counted_apart() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/ont/code-allow-stale/contracts");
        let mut g = Graph::new();
        let stats = extract(&dir, &mut g);
        assert_eq!(
            (
                stats.symbols,
                stats.resolved,
                stats.allowlisted,
                stats.unresolved
            ),
            (2, 1, 1, 0)
        );
        let ghost = iri("symbol", "kern::nn::functional::no_such_function");
        let stale = iri("symbol", "kern::nn::functional::softmax");
        let orphan = iri("symbol", "kern::gone::vanished");
        let symbols = g.instances_of(&ont("Symbol"));
        let allowed = g.instances_of(&ont("AllowlistedSymbol"));
        assert!(
            !symbols.iter().any(|s| *s == ghost),
            "an excused ghost leaves ont:Symbol"
        );
        assert!(allowed.iter().any(|s| *s == ghost));
        assert!(symbols.iter().any(|s| *s == stale) && allowed.iter().any(|s| *s == stale));
        assert!(allowed.iter().any(|s| *s == orphan));
        assert!(g.objects(&orphan, &sym("implements")).is_empty());
    }

    #[test]
    fn a_trailing_segment_equal_to_the_function_is_dropped_from_the_module_path() {
        let reg = parse_binding(&fixture().join("contracts/binding.yaml")).expect("registry");
        let bound = bound_of(&reg);
        let softmax = bound
            .iter()
            .find(|b| b.function == "softmax")
            .expect("softmax");
        assert_eq!(softmax.module_path, "kern::nn::functional");
        assert_eq!(
            symbol_iri(softmax),
            "https://ont.paiml.dev/v1alpha1/symbol/kern::nn::functional::softmax"
        );
    }

    #[test]
    fn the_positive_control_fires() {
        assert!(positive_control());
    }

    #[test]
    fn manifest_names_reads_package_and_lib_sections_only() {
        let m = manifest_names("[package]\nname = \"a-b\"\n[dependencies]\nname = \"x\"\n[lib]\nname = \"ab\"\npath = \"src/x.rs\"\n");
        assert_eq!(m.package.as_deref(), Some("a-b"));
        assert_eq!(m.lib.as_deref(), Some("ab"));
        assert_eq!(m.lib_path.as_deref(), Some("src/x.rs"));
    }

    #[test]
    fn the_body_walk_reads_unsafe_and_unchecked_indexing() {
        let src = r"
            pub fn clean(x: &[f32]) -> f32 { x[0] }
            pub fn blocky(x: &[f32]) -> f32 { unsafe { *x.as_ptr() } }
            pub unsafe fn signed(x: &[f32]) -> f32 { x[0] }
            pub fn method(x: &[f32]) -> f32 { let v = || x.len(); if v() > 0 { *x.get_unchecked(0) } else { 0.0 } }
            pub fn path(x: &[f32]) -> f32 { *<[f32]>::get_unchecked_mut(&mut [0.0][..], 0) + x[0] }
        ";
        let ast = syn::parse_file(src).expect("parses");
        let r = |n: &str| find_item(&ast.items, n).expect(n);
        assert!(r("clean").unsafe_free && r("clean").bounds_checked);
        assert!(!r("blocky").unsafe_free && r("blocky").bounds_checked);
        assert!(!r("signed").unsafe_free);
        assert!(r("method").unsafe_free && !r("method").bounds_checked);
        assert!(!r("path").bounds_checked);
    }
}
