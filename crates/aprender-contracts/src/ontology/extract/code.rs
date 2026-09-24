//! ONT-001 §3.7, §5 ONT-4b2 — `extract:code`: the Rust symbols the bindings name become `ont:Symbol` nodes.
//!
//! The focus objects are the **bound** symbols — every `module_path` + `function` a `binding.yaml` under the
//! contract dir declares — not every function in the workspace (3.9 M lines; a graph of every `fn` would be a
//! second copy of the tree, and no contract constrains a symbol nothing binds). Each is resolved by a `syn`
//! **module-tree walk**: the crate root from the workspace's `Cargo.toml`s (`[package] name` and `[lib] name`,
//! both spellings), then one `mod` per path segment — inline, `mod x;` as `x.rs` / `x/mod.rs` / `#[path]`, or a
//! `pub use` re-export followed once — and finally a free `fn` or an `impl` method of that name. Only the files
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

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

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
    pub files_parsed: usize,
    /// Distinct crate roots a walk entered (ONT-3a `crates_scanned`: a resolver that only ever opened one crate
    /// has not checked a workspace).
    pub crates_scanned: usize,
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
                contract: crate::binding::normalize_contract_id(&b.contract).to_string(),
                equation: b.equation.clone(),
                module_path,
                function,
                status: b.status,
            })
        })
        .collect()
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
    /// `root/Cargo.toml` has a `[workspace]` table. When it does, only its `members` (globs honoured) minus its
    /// `exclude` are crates — a stray manifest under the root (a test fixture, an excluded canary) is not a
    /// workspace member and a binding into it does not resolve (ONT-3a).
    pub is_workspace_root: bool,
}

impl Workspace {
    /// Scan `root` for manifests. A manifest without a `[package]` name (a virtual workspace root) contributes
    /// nothing.
    #[must_use]
    pub fn scan(root: &Path) -> Self {
        let membership = std::fs::read_to_string(root.join("Cargo.toml"))
            .ok()
            .and_then(|t| workspace_membership(&t));
        let mut ws = Self {
            root: root.to_path_buf(),
            crates: BTreeMap::new(),
            is_workspace_root: membership.is_some(),
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
                } else if path.file_name().is_some_and(|n| n == "Cargo.toml")
                    && membership
                        .as_ref()
                        .is_none_or(|m| m.admits(root, path.parent().unwrap_or(root)))
                {
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

/// `[workspace] members` / `exclude` of the root manifest, plus whether the root is itself a package.
#[derive(Debug, Default, PartialEq, Eq)]
struct Membership {
    members: Vec<String>,
    exclude: Vec<String>,
    root_package: bool,
}

impl Membership {
    /// Is the crate in `dir` a member? Paths compare relative to the root, `.` for the root itself.
    fn admits(&self, root: &Path, dir: &Path) -> bool {
        let rel = dir.strip_prefix(root).unwrap_or(dir).to_string_lossy().replace('\\', "/");
        if rel.is_empty() {
            return self.root_package || self.members.iter().any(|m| m == ".");
        }
        self.members.iter().any(|m| glob_match(m, &rel))
            && !self.exclude.iter().any(|e| glob_match(e, &rel))
    }
}

/// The `[workspace]` table's `members` and `exclude` arrays (single- or multi-line, either quote), or `None` when
/// the manifest has no `[workspace]` table.
fn workspace_membership(text: &str) -> Option<Membership> {
    let mut out = Membership::default();
    let mut section = String::new();
    let mut seen = false;
    let mut open: Option<bool> = None; // Some(true) = collecting members, Some(false) = exclude
    for line in text.lines() {
        let t = line.split('#').next().unwrap_or_default().trim();
        if open.is_none() && t.starts_with('[') && t.ends_with(']') && !t.contains('=') {
            section = t.trim_matches(['[', ']']).trim().to_string();
            seen |= section == "workspace";
            out.root_package |= section == "package";
            continue;
        }
        let body = match open {
            Some(_) => t,
            None if section == "workspace" => match t.split_once('=') {
                Some((k, v)) if matches!(k.trim(), "members" | "exclude") => {
                    open = Some(k.trim() == "members");
                    v
                }
                _ => continue,
            },
            None => continue,
        };
        let list = if open == Some(true) { &mut out.members } else { &mut out.exclude };
        list.extend(quoted(body));
        if body.contains(']') {
            open = None;
        }
    }
    seen.then_some(out)
}

/// The quoted strings of one TOML line, `"…"` or `'…'`.
fn quoted(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = line;
    while let Some(start) = rest.find(['"', '\'']) {
        let q = rest[start..].chars().next().unwrap_or('"');
        let tail = &rest[start + 1..];
        let Some(end) = tail.find(q) else { break };
        out.push(tail[..end].trim_end_matches('/').to_string());
        rest = &tail[end + 1..];
    }
    out
}

/// Cargo's member globs, component-wise: `*` matches within one component (`crates/*`, `crates/apr-*`).
fn glob_match(pattern: &str, rel: &str) -> bool {
    let p: Vec<&str> = pattern.trim_start_matches("./").split('/').collect();
    let r: Vec<&str> = rel.split('/').collect();
    p.len() == r.len()
        && p.iter().zip(&r).all(|(p, r)| match p.split_once('*') {
            None => p == r,
            Some((pre, suf)) => r.len() >= pre.len() + suf.len() && r.starts_with(pre) && r.ends_with(suf),
        })
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
    /// Crate root files a walk entered.
    crates_walked: std::collections::BTreeSet<PathBuf>,
    /// `(module_path, name)` pairs already tried by the current `resolve` — glob re-exports can cycle
    /// (`pub use a::*` in `b`, `pub use b::*` in `a`), and a pair tried twice cannot succeed the second time.
    seen: std::collections::BTreeSet<(String, String)>,
}

/// One module on the walk: its items with `include!()` spliced in, the file each item came from, the module's
/// own file, and the directory its child files live in (an included file's `mod x;` is still relative to the
/// including module, as in rustc).
struct Module {
    items: Vec<syn::Item>,
    origins: Vec<PathBuf>,
    file: PathBuf,
    child_dir: PathBuf,
}

/// `include!` nesting bound — the macro recurses, a cycle must not.
const INCLUDE_DEPTH: usize = 8;

impl<'a> Resolver<'a> {
    #[must_use]
    pub fn new(ws: &'a Workspace) -> Self {
        Self {
            ws,
            cache: BTreeMap::new(),
            crates_walked: std::collections::BTreeSet::new(),
            seen: std::collections::BTreeSet::new(),
        }
    }

    /// Distinct crate roots entered so far.
    #[must_use]
    pub fn crates_walked(&self) -> usize {
        self.crates_walked.len()
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
        Ok(self.module_of(&ast.items, file, child_dir))
    }

    /// A module from `items` written in `file`, `include!()`s spliced.
    fn module_of(&mut self, items: &[syn::Item], file: &Path, child_dir: PathBuf) -> Module {
        let mut module = Module {
            items: Vec::new(),
            origins: Vec::new(),
            file: file.to_path_buf(),
            child_dir,
        };
        self.splice(items, file, 0, &mut module);
        module
    }

    /// Append `items` to `module`, replacing each `include!("p")` item by the items of `p` — resolved against the
    /// directory of the file the macro is written in, recursively. An include that is missing or does not parse
    /// contributes nothing, so what it would have defined stays unresolved (fail-closed).
    fn splice(&mut self, items: &[syn::Item], origin: &Path, depth: usize, module: &mut Module) {
        for item in items {
            match include_path(item) {
                Some(rel) if depth < INCLUDE_DEPTH => {
                    let file = origin.parent().unwrap_or(origin).join(rel);
                    if let Some(ast) = self.parse(&file) {
                        self.splice(&ast.items, &file, depth + 1, module);
                    }
                }
                _ => {
                    module.items.push(item.clone());
                    module.origins.push(origin.to_path_buf());
                }
            }
        }
    }

    /// Resolve `module_path::function` to its definition. `depth` bounds re-export chasing.
    pub fn resolve(&mut self, module_path: &str, function: &str) -> Result<Resolved, Unresolved> {
        self.seen.clear();
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
        if !self
            .seen
            .insert((module_path.to_string(), function.to_string()))
        {
            return Err(Unresolved {
                reason: format!("`{module_path}::{function}`: re-export cycle"),
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
        self.crates_walked.insert(root.to_path_buf());
        for (i, seg) in segs.iter().enumerate() {
            let rest = &segs[i + 1..];
            match self.step(&module, seg)? {
                Step::Module(next) => module = next,
                Step::ReExport(target) => {
                    return self.resolve_depth(&join_path(&target, rest), function, depth + 1);
                }
                Step::Globs(targets) => {
                    return self.first_of(&targets, seg, rest, function, depth, &module);
                }
                Step::Type => return self.type_member(&module, seg, rest, function),
            }
        }
        if let Some((i, mut r)) = find_indexed(&module.items, function) {
            r.file = self.rel(&module.origins[i]);
            return Ok(r);
        }
        if let Some(target) = use_target(&module.items, function) {
            let target = absolute_use(&target, &module, self.ws);
            return self.resolve_by_use(&target, depth);
        }
        let globs = pub_globs(&module.items, &module, self.ws);
        if globs.is_empty() {
            return Err(self.not_found(function, &module));
        }
        let mut last = self.not_found(function, &module);
        for g in globs {
            match self.resolve_depth(&g, function, depth + 1) {
                Ok(r) => return Ok(r),
                Err(e) => last = e,
            }
        }
        Err(last)
    }

    fn not_found(&self, function: &str, module: &Module) -> Unresolved {
        Unresolved {
            reason: format!(
                "no `fn {function}` (free or in an impl) or item `{function}` in `{}`",
                self.rel(&module.file)
            ),
        }
    }

    /// Continue through the first `pub use <glob>::*` under which `seg::rest` resolves.
    fn first_of(
        &mut self,
        globs: &[String],
        seg: &str,
        rest: &[&str],
        function: &str,
        depth: usize,
        module: &Module,
    ) -> Result<Resolved, Unresolved> {
        let mut last = Unresolved {
            reason: format!(
                "no `mod {seg}` or `use … {seg}` in `{}` (nor under its glob re-exports)",
                self.rel(&module.file)
            ),
        };
        for g in globs {
            let path = join_path(&format!("{g}::{seg}"), rest);
            match self.resolve_depth(&path, function, depth + 1) {
                Ok(r) => return Ok(r),
                Err(e) if e.reason.contains("re-export cycle") => {}
                Err(e) => last = e,
            }
        }
        Err(last)
    }

    /// `Type::function` (or `Type::<Arg>::function`): a method of an `impl` whose self type is `ty` in this
    /// module, or a method of `trait ty`. Only the module that defines the type is searched — an `impl` in
    /// another module is not found, and says so.
    fn type_member(
        &self,
        module: &Module,
        ty: &str,
        rest: &[&str],
        function: &str,
    ) -> Result<Resolved, Unresolved> {
        let generic = match rest {
            [] => None,
            [g] if g.starts_with('<') => Some(g.trim_matches(['<', '>']).trim()),
            _ => {
                return Err(Unresolved {
                    reason: format!(
                        "`{ty}` in `{}` is a type, not a module",
                        self.rel(&module.file)
                    ),
                })
            }
        };
        for (i, item) in module.items.iter().enumerate() {
            let hit = match item {
                syn::Item::Impl(im) if impl_is_for(im, ty, generic) => {
                    find_method(&im.items, function)
                }
                syn::Item::Trait(t) if t.ident == ty && generic.is_none() => {
                    find_trait_fn(&t.items, function)
                }
                _ => None,
            };
            if let Some(mut r) = hit {
                r.file = self.rel(&module.origins[i]);
                return Ok(r);
            }
        }
        Err(Unresolved {
            reason: format!(
                "no `fn {function}` in an `impl {ty}` or `trait {ty}` in `{}`",
                self.rel(&module.file)
            ),
        })
    }

    fn resolve_by_use(&mut self, target: &str, depth: usize) -> Result<Resolved, Unresolved> {
        let (path, name) = target.rsplit_once("::").ok_or_else(|| Unresolved {
            reason: format!("re-export `{target}` has no module path"),
        })?;
        self.resolve_depth(path, name, depth + 1)
    }

    fn step(&mut self, module: &Module, seg: &str) -> Result<Step, Unresolved> {
        for (i, item) in module.items.iter().enumerate() {
            if let syn::Item::Mod(m) = item {
                if m.ident != seg {
                    continue;
                }
                if let Some((_, content)) = &m.content {
                    let origin = module.origins[i].clone();
                    let mut inner = self.module_of(content, &origin, module.child_dir.join(seg));
                    inner.file = module.file.clone();
                    return Ok(Step::Module(inner));
                }
                let (file, child_dir) = child_file(&module.child_dir, seg, &m.attrs)?;
                return self.file_module(&file, child_dir).map(Step::Module);
            }
        }
        if let Some(target) = use_target(&module.items, seg) {
            let target = absolute_use(&target, module, self.ws);
            return Ok(Step::ReExport(target));
        }
        if defines_type(&module.items, seg) {
            return Ok(Step::Type);
        }
        let globs = pub_globs(&module.items, module, self.ws);
        if !globs.is_empty() {
            return Ok(Step::Globs(globs));
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
    /// The segment names a type (or a trait) defined in this module.
    Type,
    /// No such segment here, but these `pub use …::*` globs may bring it in.
    Globs(Vec<String>),
}

/// `target` with the remaining segments appended.
fn join_path(target: &str, rest: &[&str]) -> String {
    if rest.is_empty() {
        target.to_string()
    } else {
        format!("{target}::{}", rest.join("::"))
    }
}

/// The path of an `include!("…")` item (a string literal only — `concat!(env!("OUT_DIR"), …)` is a build
/// artifact the tree does not hold).
fn include_path(item: &syn::Item) -> Option<String> {
    let syn::Item::Macro(m) = item else {
        return None;
    };
    if !m.mac.path.is_ident("include") {
        return None;
    }
    syn::parse2::<syn::LitStr>(m.mac.tokens.clone())
        .ok()
        .map(|l| l.value())
}

/// A struct, enum, union, trait or type alias named `name`, or an `impl` for it, among `items`.
fn defines_type(items: &[syn::Item], name: &str) -> bool {
    items.iter().any(|item| match item {
        syn::Item::Struct(s) => s.ident == name,
        syn::Item::Enum(e) => e.ident == name,
        syn::Item::Union(u) => u.ident == name,
        syn::Item::Trait(t) => t.ident == name,
        syn::Item::Type(t) => t.ident == name,
        syn::Item::Impl(im) => impl_is_for(im, name, None),
        _ => false,
    })
}

/// Is `im` an `impl` for the type `ty` — and, when `generic` is given, for `ty<generic>` or a blanket
/// `impl<T> ty<T>`?
fn impl_is_for(im: &syn::ItemImpl, ty: &str, generic: Option<&str>) -> bool {
    let syn::Type::Path(tp) = im.self_ty.as_ref() else {
        return false;
    };
    let Some(last) = tp.path.segments.last() else {
        return false;
    };
    if last.ident != ty {
        return false;
    }
    let Some(generic) = generic else {
        return true;
    };
    let syn::PathArguments::AngleBracketed(args) = &last.arguments else {
        return false;
    };
    let params: Vec<String> = im
        .generics
        .type_params()
        .map(|p| p.ident.to_string())
        .collect();
    args.args.iter().any(|a| match a {
        syn::GenericArgument::Type(syn::Type::Path(p)) => p
            .path
            .segments
            .last()
            .is_some_and(|s| s.ident == generic || params.contains(&s.ident.to_string())),
        _ => false,
    })
}

/// The `fn name` of a trait (declared or defaulted).
fn find_trait_fn(items: &[syn::TraitItem], name: &str) -> Option<Resolved> {
    items.iter().find_map(|ti| match ti {
        syn::TraitItem::Fn(f) if f.sig.ident == name => Some(found(
            "trait-method",
            &syn::Visibility::Inherited,
            &f.attrs,
        )),
        _ => None,
    })
}

/// The absolute paths of every `pub use <path>::*` among `items`.
fn pub_globs(items: &[syn::Item], module: &Module, ws: &Workspace) -> Vec<String> {
    let mut out = Vec::new();
    for item in items {
        if let syn::Item::Use(u) = item {
            if !matches!(u.vis, syn::Visibility::Inherited) {
                glob_paths(&u.tree, "", &mut out);
            }
        }
    }
    out.into_iter()
        .map(|g| absolute_use(&g, module, ws))
        .collect()
}

fn glob_paths(tree: &syn::UseTree, prefix: &str, out: &mut Vec<String>) {
    let join = |s: &str| {
        if prefix.is_empty() {
            s.to_string()
        } else {
            format!("{prefix}::{s}")
        }
    };
    match tree {
        syn::UseTree::Path(p) => glob_paths(&p.tree, &join(&p.ident.to_string()), out),
        syn::UseTree::Glob(_) if !prefix.is_empty() => out.push(prefix.to_string()),
        syn::UseTree::Group(g) => g.items.iter().for_each(|t| glob_paths(t, prefix, out)),
        _ => {}
    }
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

/// A free `fn name`, an `impl … { fn name }`, or a named item (`struct`, `enum`, `trait`, …) among `items`.
fn find_item(items: &[syn::Item], name: &str) -> Option<Resolved> {
    find_indexed(items, name).map(|(_, r)| r)
}

/// [`find_item`] with the index of the item it came from (so the caller can name its origin file).
fn find_indexed(items: &[syn::Item], name: &str) -> Option<(usize, Resolved)> {
    items
        .iter()
        .enumerate()
        .find_map(|(i, item)| item_named(item, name).map(|r| (i, r)))
}

fn item_named(item: &syn::Item, name: &str) -> Option<Resolved> {
    let named = |ident: &syn::Ident, kind: &str, vis: &syn::Visibility, attrs: &[syn::Attribute]| {
        (ident == name).then(|| found(kind, vis, attrs))
    };
    match item {
        syn::Item::Fn(f) => named(&f.sig.ident, "fn", &f.vis, &f.attrs),
        syn::Item::Impl(im) => find_method(&im.items, name),
        syn::Item::Struct(s) => named(&s.ident, "struct", &s.vis, &s.attrs),
        syn::Item::Enum(e) => named(&e.ident, "enum", &e.vis, &e.attrs),
        syn::Item::Union(u) => named(&u.ident, "union", &u.vis, &u.attrs),
        syn::Item::Trait(t) => named(&t.ident, "trait", &t.vis, &t.attrs),
        syn::Item::Type(t) => named(&t.ident, "type", &t.vis, &t.attrs),
        syn::Item::Const(c) => named(&c.ident, "const", &c.vis, &c.attrs),
        syn::Item::Static(s) => named(&s.ident, "static", &s.vis, &s.attrs),
        _ => None,
    }
}

/// The `fn name` of one `impl` block.
fn find_method(items: &[syn::ImplItem], name: &str) -> Option<Resolved> {
    items.iter().find_map(|ii| match ii {
        syn::ImplItem::Fn(f) if f.sig.ident == name => Some(found("method", &f.vis, &f.attrs)),
        _ => None,
    })
}

/// A resolution with the file filled in by the caller that knows it.
fn found(kind: &str, vis: &syn::Visibility, attrs: &[syn::Attribute]) -> Resolved {
    Resolved {
        file: String::new(),
        visibility: visibility_of(vis),
        kind: kind.to_string(),
        attributes: attr_paths(attrs),
    }
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

/// One bound symbol into `g`, resolved or not.
pub fn emit(g: &mut Graph, b: &Bound, found: &Result<Resolved, Unresolved>) {
    let s = symbol_iri(b);
    g.insert(s.clone(), RDF_TYPE, Term::iri(ont("Symbol")));
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
        }
        Err(u) => {
            g.insert(s.clone(), sym("resolved"), Term::boolean(false));
            g.insert(s, sym("unresolvedReason"), Term::string(&u.reason));
        }
    }
}

/// Every bound symbol of every registry under `contract_dir`, each with its resolution — the walk shared by
/// [`extract`] (graph) and the `bindings` lint gate (ONT-3a, verdict).
#[derive(Debug, Default)]
pub struct Resolution {
    pub stats: CodeStats,
    pub symbols: Vec<(Bound, Result<Resolved, Unresolved>)>,
    /// The repo root has a `[workspace]` manifest.
    pub at_workspace_root: bool,
}

/// Resolve every bound symbol under `contract_dir` against the workspace at the contract dir's parent.
#[must_use]
pub fn resolve_all(contract_dir: &Path) -> Resolution {
    let root_buf = super::repo_root(contract_dir);
    let ws = Workspace::scan(root_buf.as_path());
    let mut resolver = Resolver::new(&ws);
    let mut out = Resolution {
        at_workspace_root: ws.is_workspace_root,
        ..Resolution::default()
    };
    for (_file, registry) in registries(contract_dir) {
        out.stats.registries += 1;
        for b in bound_of(&registry) {
            let found = resolver.resolve(&b.module_path, &b.function);
            if found.is_ok() {
                out.stats.resolved += 1;
            } else {
                out.stats.unresolved += 1;
            }
            out.stats.symbols += 1;
            out.symbols.push((b, found));
        }
    }
    out.stats.files_parsed = resolver.files_parsed();
    out.stats.crates_scanned = resolver.crates_walked();
    out
}

/// Every bound symbol of every registry under `contract_dir`, resolved against the workspace at the contract
/// dir's parent, into `g`.
pub fn extract(contract_dir: &Path, g: &mut Graph) -> CodeStats {
    let r = resolve_all(contract_dir);
    for (b, found) in &r.symbols {
        emit(g, b, found);
    }
    r.stats
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
}
