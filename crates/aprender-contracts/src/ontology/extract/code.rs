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
    /// Non-member manifests that declare their OWN `[workspace]` — a standalone cargo workspace under the root
    /// (`experiments/cuda-oxide/gated-rmsnorm`: its own toolchain and `Cargo.lock`). Indexed, NOT admitted: only
    /// a `kernel: true` binding naming one admits it ([`Workspace::admit_standalone`], ONT-4c4).
    pub standalone: BTreeMap<String, Vec<PathBuf>>,
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
            standalone: BTreeMap::new(),
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
                    let dir = path.parent().unwrap_or(root);
                    if membership.as_ref().is_none_or(|m| m.admits(root, dir)) {
                        ws.index_manifest(&path);
                    } else if dir != root {
                        ws.index_standalone(&path);
                    }
                }
            }
        }
        ws
    }

    fn index_manifest(&mut self, manifest: &Path) {
        if let Ok(text) = std::fs::read_to_string(manifest) {
            index_names(&mut self.crates, manifest, &text);
        }
    }

    /// A non-member manifest is recorded only when it is its own workspace root; a stray manifest (a test
    /// fixture, an excluded canary) stays invisible, as ONT-3a requires.
    fn index_standalone(&mut self, manifest: &Path) {
        if let Ok(text) = std::fs::read_to_string(manifest) {
            if workspace_membership(&text).is_some() {
                index_names(&mut self.standalone, manifest, &text);
            }
        }
    }

    /// Admit the standalone crate `krate` (a `kernel: true` binding names it) unless a member already owns the
    /// name. Returns whether anything was admitted.
    pub fn admit_standalone(&mut self, krate: &str) -> bool {
        if self.crates.contains_key(krate) {
            return false;
        }
        match self.standalone.get(krate) {
            Some(roots) => {
                self.crates.insert(krate.to_string(), roots.clone());
                true
            }
            None => false,
        }
    }
}

/// Key `manifest`'s package and `[lib]` names (`-` spelled `_`) to its crate root in `map`.
fn index_names(map: &mut BTreeMap<String, Vec<PathBuf>>, manifest: &Path, text: &str) {
    let m = manifest_names(text);
    let dir = manifest.parent().unwrap_or(manifest);
    let lib_path = m
        .lib_path
        .map_or_else(|| dir.join("src/lib.rs"), |p| dir.join(p));
    for name in [m.package, m.lib].into_iter().flatten() {
        let roots = map.entry(name.replace('-', "_")).or_default();
        if !roots.contains(&lib_path) {
            roots.push(lib_path.clone());
        }
    }
}

/// `[workspace] members` / `exclude` of the root manifest, plus whether the root is itself a package.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct Membership {
    members: Vec<String>,
    exclude: Vec<String>,
    root_package: bool,
}

impl Membership {
    /// Is the crate in `dir` a member? Paths compare relative to the root, `.` for the root itself.
    pub(crate) fn admits(&self, root: &Path, dir: &Path) -> bool {
        let rel = dir
            .strip_prefix(root)
            .unwrap_or(dir)
            .to_string_lossy()
            .replace('\\', "/");
        if rel.is_empty() {
            return self.root_package || self.members.iter().any(|m| m == ".");
        }
        self.members.iter().any(|m| glob_match(m, &rel))
            && !self.exclude.iter().any(|e| glob_match(e, &rel))
    }
}

/// The `[workspace]` table's `members` and `exclude` arrays (single- or multi-line, either quote), or `None` when
/// the manifest has no `[workspace]` table.
pub(crate) fn workspace_membership(text: &str) -> Option<Membership> {
    let mut out = Membership::default();
    let mut section = String::new();
    let mut seen = false;
    let mut open: Option<bool> = None; // Some(true) = collecting members, Some(false) = exclude
    for line in text.lines() {
        let t = line.split('#').next().unwrap_or_default().trim();
        if let Some(name) = open.is_none().then(|| section_name(t)).flatten() {
            seen |= name == "workspace";
            out.root_package |= name == "package";
            section = name;
            continue;
        }
        let Some(body) = list_body(t, &mut open, section == "workspace") else {
            continue;
        };
        let list = if open == Some(true) {
            &mut out.members
        } else {
            &mut out.exclude
        };
        list.extend(quoted(body));
        if body.contains(']') {
            open = None;
        }
    }
    seen.then_some(out)
}

/// The table name of a `[section]` header line, or `None` when `t` is not one.
fn section_name(t: &str) -> Option<String> {
    (t.starts_with('[') && t.ends_with(']') && !t.contains('='))
        .then(|| t.trim_matches(['[', ']']).trim().to_string())
}

/// The part of `t` that holds array items: all of it inside an open array, or the value of a `members` /
/// `exclude` key in `[workspace]`, which opens one (`open`: `Some(true)` = members, `Some(false)` = exclude).
fn list_body<'t>(t: &'t str, open: &mut Option<bool>, in_workspace: bool) -> Option<&'t str> {
    if open.is_some() {
        return Some(t);
    }
    if !in_workspace {
        return None;
    }
    let (k, v) = t.split_once('=')?;
    let k = k.trim();
    if !matches!(k, "members" | "exclude") {
        return None;
    }
    *open = Some(k == "members");
    Some(v)
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
            Some((pre, suf)) => {
                r.len() >= pre.len() + suf.len() && r.starts_with(pre) && r.ends_with(suf)
            }
        })
}

#[derive(Debug, Default)]
pub(crate) struct ManifestNames {
    pub(crate) package: Option<String>,
    lib: Option<String>,
    lib_path: Option<String>,
}

/// `[package] name`, `[lib] name` and `[lib] path` by a section-aware line scan — the two keys this walk needs,
/// read without a TOML crate.
pub(crate) fn manifest_names(text: &str) -> ManifestNames {
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
    /// Every file `include!()`d into this module (at any nesting depth), keyed by its stem: `ops::activation::matmul`
    /// names `matmul` in the `activation.rs` that `ops/mod.rs` includes. Rust has no such path, but the binding rows
    /// name the file the item is written in, and the item does exist (infra-83 ruling on #4502).
    includes: Vec<(String, PathBuf)>,
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
            includes: Vec::new(),
        };
        self.splice(items, file, 0, &mut module);
        module
    }

    /// Append `items` to `module`, replacing each `include!` item naming a path `p` by the items of `p` — resolved against the
    /// directory of the file the macro is written in, recursively. An include that is missing or does not parse
    /// contributes nothing, so what it would have defined stays unresolved (fail-closed).
    fn splice(&mut self, items: &[syn::Item], origin: &Path, depth: usize, module: &mut Module) {
        for item in items {
            match include_path(item) {
                Some(rel) if depth < INCLUDE_DEPTH => {
                    let file = origin.parent().unwrap_or(origin).join(rel);
                    if let Some(ast) = self.parse(&file) {
                        if let Some(stem) = file.file_stem().and_then(|s| s.to_str()) {
                            module.includes.push((stem.to_string(), file.clone()));
                        }
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
                    return self.re_export(&module, seg, rest, &target, function, depth);
                }
                Step::Globs(targets) => {
                    return self.first_of(&targets, seg, rest, function, depth, &module);
                }
                Step::Type => return self.type_member(&module, seg, rest, function),
            }
        }
        self.resolve_leaf(&module, function, depth)
    }

    /// `seg` is re-exported here from `target`. `use crate::a::T;` then `impl T { fn f() }` in THIS module: the
    /// binding names this module, so its own impls are searched before following the `use` to T's definition.
    fn re_export(
        &mut self,
        module: &Module,
        seg: &str,
        rest: &[&str],
        target: &str,
        function: &str,
        depth: usize,
    ) -> Result<Resolved, Unresolved> {
        if let Ok(r) = self.type_member(module, seg, rest, function) {
            return Ok(r);
        }
        self.resolve_depth(&join_path(target, rest), function, depth + 1)
    }

    /// `function` in the module the walk ended in: defined here, `use`d here, or under a `pub use …::*`.
    fn resolve_leaf(
        &mut self,
        module: &Module,
        function: &str,
        depth: usize,
    ) -> Result<Resolved, Unresolved> {
        if let Some((i, mut r)) = find_indexed(&module.items, function) {
            r.file = self.rel(&module.origins[i]);
            return Ok(r);
        }
        if let Some(target) = use_target(&module.items, function) {
            let target = absolute_use(&target, module, self.ws);
            return self.resolve_by_use(&target, depth);
        }
        let globs = pub_globs(&module.items, module, self.ws);
        if globs.is_empty() {
            return Err(self.not_found(function, module));
        }
        let mut last = self.not_found(function, module);
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
    /// module, or a method of `trait ty`. Searched: the module that defines the type, and a module the binding
    /// names that imports the type (`use`) and implements it there. An `impl` anywhere else is not found.
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
        if let Some(step) = self.step_into_mod(module, seg) {
            return step;
        }
        if let Some(target) = use_target(&module.items, seg) {
            let target = absolute_use(&target, module, self.ws);
            return Ok(Step::ReExport(target));
        }
        if defines_type(&module.items, seg) {
            return Ok(Step::Type);
        }
        if let Some(step) = self.step_into_include(module, seg) {
            return step;
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

    /// `mod seg` in this module (inline, or its file), or `None` when the module declares no such `mod`.
    fn step_into_mod(&mut self, module: &Module, seg: &str) -> Option<Result<Step, Unresolved>> {
        let (i, m) = module
            .items
            .iter()
            .enumerate()
            .find_map(|(i, item)| match item {
                syn::Item::Mod(m) if m.ident == seg => Some((i, m)),
                _ => None,
            })?;
        if let Some((_, content)) = &m.content {
            let origin = module.origins[i].clone();
            let mut inner = self.module_of(content, &origin, module.child_dir.join(seg));
            inner.file = module.file.clone();
            return Some(Ok(Step::Module(inner)));
        }
        Some(
            child_file(&module.child_dir, seg, &m.attrs)
                .and_then(|(file, child_dir)| self.file_module(&file, child_dir).map(Step::Module)),
        )
    }

    /// `seg` is the stem of a file `include!()`d into this module: step into THAT file's items alone (its own
    /// `include!`s spliced), so a fn written in a sibling included file is still refused. A real `mod`, `use` or
    /// type of the same name is tried first.
    ///
    /// The file may also be included by a CHILD module's file in the same directory: `metaheuristics::
    /// cmaes_include_01::optimize`, where `metaheuristics/cmaes.rs` (`mod cmaes;`) `include!`s `cmaes_include_01.rs`.
    fn step_into_include(
        &mut self,
        module: &Module,
        seg: &str,
    ) -> Option<Result<Step, Unresolved>> {
        let file = match module.includes.iter().find(|(stem, _)| stem == seg) {
            Some((_, file)) => file.clone(),
            None => self.included_by_child(module, seg)?,
        };
        Some(
            self.file_module(&file, module.child_dir.clone())
                .map(Step::Module),
        )
    }

    /// `child_dir/seg.rs`, when a file-backed `mod x;` of this module lives in `child_dir` and includes it.
    fn included_by_child(&mut self, module: &Module, seg: &str) -> Option<PathBuf> {
        let target = module.child_dir.join(format!("{seg}.rs"));
        if !target.is_file() {
            return None;
        }
        let children: Vec<PathBuf> = module
            .items
            .iter()
            .filter_map(|item| match item {
                syn::Item::Mod(m) if m.content.is_none() => {
                    child_file(&module.child_dir, &m.ident.to_string(), &m.attrs)
                        .ok()
                        .map(|(file, _)| file)
                }
                _ => None,
            })
            .filter(|file| file.parent() == Some(module.child_dir.as_path()))
            .collect();
        for child in children {
            let Some(ast) = self.parse(&child) else {
                continue;
            };
            let dir = child.parent().unwrap_or(&child).to_path_buf();
            if ast
                .items
                .iter()
                .filter_map(include_path)
                .any(|rel| dir.join(rel) == target)
            {
                return Some(target);
            }
        }
        None
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

/// The path an `include!` item names (a string literal only — `concat!(env!("OUT_DIR"), …)` is a build
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
        syn::TraitItem::Fn(f) if f.sig.ident == name => Some(found_bodiless(
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
    let named =
        |ident: &syn::Ident, kind: &str, vis: &syn::Visibility, attrs: &[syn::Attribute]| {
            (ident == name).then(|| found_bodiless(kind, vis, attrs))
        };
    match item {
        syn::Item::Fn(f) => {
            (f.sig.ident == name).then(|| found("fn", &f.vis, &f.attrs, &f.sig, &f.block))
        }
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

/// [`found`] for an item with no function body to read (a type, a const, a trait method's declaration). ONT-4c4's
/// two safety facts are read only for a bound `#[kernel]` fn, which always resolves through [`found`]; nothing
/// here is ever graded on them, so they are recorded as holding rather than left unset.
fn found_bodiless(kind: &str, vis: &syn::Visibility, attrs: &[syn::Attribute]) -> Resolved {
    Resolved {
        file: String::new(),
        visibility: visibility_of(vis),
        kind: kind.to_string(),
        attributes: attr_paths(attrs),
        unsafe_free: true,
        bounds_checked: true,
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
    let mut ws = Workspace::scan(root_buf.as_path());
    let regs = registries(contract_dir);
    // ONT-4c4: a cuda-oxide `#[kernel]` lives in its own cargo workspace (pinned nightly), never a member; the
    // `kernel: true` binding that names its crate is what admits it — nothing else does.
    for (_file, registry) in &regs {
        for b in registry.bindings.iter().filter(|b| b.kernel) {
            let path = b.module_path.as_deref().unwrap_or(&registry.target_crate);
            if let Some(krate) = path.split("::").next() {
                ws.admit_standalone(krate);
            }
        }
    }
    let mut resolver = Resolver::new(&ws);
    let mut out = Resolution {
        at_workspace_root: ws.is_workspace_root,
        ..Resolution::default()
    };
    for (_file, registry) in regs {
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

    /// `use crate::a::T;` then `impl T<P> { fn f() }` in another module: the binding naming that module resolves
    /// to the impl there (aprender-contrastive-data's `attestation::PreparedDataset::<Canonical>::…`), and a
    /// method no impl has is still refused.
    #[test]
    fn a_method_implemented_in_the_module_that_imports_the_type_resolves() {
        let tmp = tempfile::tempdir().unwrap();
        let w = tmp.path();
        std::fs::write(w.join("Cargo.toml"), "[workspace]\nmembers = [\"k\"]\n").unwrap();
        std::fs::create_dir_all(w.join("k/src")).unwrap();
        std::fs::write(w.join("k/Cargo.toml"), "[package]\nname = \"k\"\n").unwrap();
        std::fs::write(w.join("k/src/lib.rs"), "pub mod a;\npub mod b;\n").unwrap();
        std::fs::write(w.join("k/src/a.rs"), "pub struct T<P>(P);\npub struct C;\n").unwrap();
        std::fs::write(
            w.join("k/src/b.rs"),
            "use crate::a::{C, T};\nimpl T<C> { pub fn f() {} }\n",
        )
        .unwrap();
        let ws = Workspace::scan(w);
        let mut r = Resolver::new(&ws);
        let f = r
            .resolve("k::b::T::<C>", "f")
            .expect("impl in the importing module");
        assert_eq!(f.file, "k/src/b.rs");
        assert_eq!(f.kind, "method");
        assert!(r.resolve("k::b::T::<C>", "g").is_err(), "a ghost method");
        assert!(
            r.resolve("k::a::T::<C>", "f").is_err(),
            "a does not implement f"
        );
    }

    /// ONT-4c4 (C196): a cuda-oxide `#[kernel]` lives in its own cargo workspace, never a root member. A
    /// `kernel: true` binding naming that crate admits it and the kernel resolves, attribute and all; the same
    /// row without `kernel: true` stays unresolved (ONT-3a), and a stray non-member manifest with no
    /// `[workspace]` of its own is never indexed.
    #[test]
    fn a_kernel_binding_admits_its_standalone_workspace_and_nothing_else_does() {
        let tmp = tempfile::tempdir().unwrap();
        let w = tmp.path();
        std::fs::write(w.join("Cargo.toml"), "[workspace]\nmembers = [\"k\"]\n").unwrap();
        std::fs::create_dir_all(w.join("k/src")).unwrap();
        std::fs::write(w.join("k/Cargo.toml"), "[package]\nname = \"k\"\n").unwrap();
        std::fs::write(w.join("k/src/lib.rs"), "pub fn r() {}\n").unwrap();
        std::fs::create_dir_all(w.join("exp/ox/src")).unwrap();
        std::fs::write(
            w.join("exp/ox/Cargo.toml"),
            "[package]\nname = \"ox_kern\"\n\n[workspace]\n",
        )
        .unwrap();
        std::fs::write(
            w.join("exp/ox/src/lib.rs"),
            "pub mod kernels {\n    #[kernel]\n    pub fn g() {}\n}\n",
        )
        .unwrap();
        std::fs::create_dir_all(w.join("fx/src")).unwrap();
        std::fs::write(w.join("fx/Cargo.toml"), "[package]\nname = \"stray\"\n").unwrap();
        std::fs::write(w.join("fx/src/lib.rs"), "pub fn s() {}\n").unwrap();

        let mut ws = Workspace::scan(w);
        assert!(!ws.crates.contains_key("ox_kern"), "{:?}", ws.crates);
        assert!(ws.standalone.contains_key("ox_kern"), "{:?}", ws.standalone);
        assert!(!ws.standalone.contains_key("stray"), "{:?}", ws.standalone);
        assert!(!ws.admit_standalone("k"), "a member is never re-admitted");
        assert!(!ws.admit_standalone("stray"));
        assert!(ws.admit_standalone("ox_kern"));
        assert!(!ws.admit_standalone("ox_kern"), "admitted once");
        let g = Resolver::new(&ws)
            .resolve("ox_kern::kernels", "g")
            .expect("admitted kernel resolves");
        assert_eq!(g.attributes, vec!["kernel".to_string()]);

        let row = |kernel: &str| {
            format!(
                "version: 1.0.0\ntarget_crate: k\nbindings:\n- contract: c-v1.yaml\n  equation: g\n  \
                 module_path: ox_kern::kernels\n  function: g\n  status: implemented\n{kernel}"
            )
        };
        std::fs::create_dir_all(w.join("contracts")).unwrap();
        std::fs::write(w.join("contracts/binding.yaml"), row("  kernel: true\n")).unwrap();
        let r = resolve_all(&w.join("contracts"));
        assert_eq!(
            (r.stats.resolved, r.stats.unresolved),
            (1, 0),
            "{:?}",
            r.symbols
        );
        std::fs::write(w.join("contracts/binding.yaml"), row("")).unwrap();
        let r = resolve_all(&w.join("contracts"));
        assert_eq!(
            (r.stats.resolved, r.stats.unresolved),
            (0, 1),
            "{:?}",
            r.symbols
        );
    }

    /// #4502: a binding row names the file an `include!()`d item is written in (`ops::activation::matmul`, with
    /// `ops/mod.rs` `include!`-ing `activation.rs`; `cmaes::cmaes_include_01::optimize` for a method in a file
    /// the sibling `cmaes.rs` includes). The item exists, so the row resolves; an item of a DIFFERENT included file,
    /// or a stem nothing includes, is still refused.
    #[test]
    fn an_item_in_an_included_file_resolves_under_the_file_stem() {
        let tmp = tempfile::tempdir().unwrap();
        let w = tmp.path();
        std::fs::write(w.join("Cargo.toml"), "[workspace]\nmembers = [\"k\"]\n").unwrap();
        std::fs::create_dir_all(w.join("k/src/ops")).unwrap();
        std::fs::write(w.join("k/Cargo.toml"), "[package]\nname = \"k\"\n").unwrap();
        std::fs::write(w.join("k/src/lib.rs"), "pub mod ops;\npub mod cmaes;\n").unwrap();
        std::fs::write(
            w.join("k/src/ops/mod.rs"),
            "include!(\"activation.rs\");\ninclude!(\"masking.rs\");\n",
        )
        .unwrap();
        std::fs::write(w.join("k/src/ops/activation.rs"), "pub fn matmul() {}\n").unwrap();
        std::fs::write(w.join("k/src/ops/masking.rs"), "pub fn mask() {}\n").unwrap();
        std::fs::write(
            w.join("k/src/cmaes.rs"),
            "pub struct Cma;\ninclude!(\"cmaes_include_01.rs\");\n",
        )
        .unwrap();
        std::fs::write(
            w.join("k/src/cmaes_include_01.rs"),
            "impl Cma { pub fn optimize(&self) {} }\n",
        )
        .unwrap();
        let ws = Workspace::scan(w);
        let mut r = Resolver::new(&ws);
        let m = r
            .resolve("k::ops::activation", "matmul")
            .expect("the included file's stem is a segment");
        assert_eq!(m.file, "k/src/ops/activation.rs");
        r.resolve("k::ops", "matmul")
            .expect("the Rust path still resolves");
        r.resolve("k::cmaes_include_01", "optimize")
            .expect("a method in a file the sibling module includes, named under the parent");
        r.resolve("k::cmaes::cmaes_include_01", "optimize")
            .expect("the same file under the including module");
        std::fs::write(w.join("k/src/stray.rs"), "pub fn optimize() {}\n").unwrap();
        assert!(
            Resolver::new(&ws).resolve("k::stray", "optimize").is_err(),
            "a file on disk that nothing declares or includes"
        );
        let wrong = r
            .resolve("k::ops::activation", "mask")
            .expect_err("mask is in masking.rs, not activation.rs");
        assert!(wrong.reason.contains("no `fn mask`"), "{}", wrong.reason);
        assert!(
            r.resolve("k::ops::pooling", "matmul").is_err(),
            "a stem nothing includes"
        );
    }

    /// Two modules that glob-re-export each other (`a: pub use crate::b::*`, `b: pub use crate::a::*`) cycle when
    /// a segment neither defines is chased. The cycle is a dead end, not the answer: the refusal names the last
    /// real miss (`no mod x … in b.rs`), never `re-export cycle` (mutant `code.rs:628` guard → false, #4588).
    #[test]
    fn a_glob_re_export_cycle_reports_the_real_miss_not_the_cycle() {
        let tmp = tempfile::tempdir().unwrap();
        let w = tmp.path();
        std::fs::write(w.join("Cargo.toml"), "[workspace]\nmembers = [\"k\"]\n").unwrap();
        std::fs::create_dir_all(w.join("k/src")).unwrap();
        std::fs::write(w.join("k/Cargo.toml"), "[package]\nname = \"k\"\n").unwrap();
        std::fs::write(w.join("k/src/lib.rs"), "pub mod a;\npub mod b;\n").unwrap();
        std::fs::write(w.join("k/src/a.rs"), "pub use crate::b::*;\n").unwrap();
        std::fs::write(w.join("k/src/b.rs"), "pub use crate::a::*;\n").unwrap();
        let ws = Workspace::scan(w);
        let mut r = Resolver::new(&ws);
        let e = r.resolve("k::a::x", "f").expect_err("x exists nowhere");
        assert!(!e.reason.contains("re-export cycle"), "{}", e.reason);
        assert!(
            e.reason.contains("no `mod x`") && e.reason.contains("k/src/b.rs"),
            "{}",
            e.reason
        );
    }

    /// A one-crate workspace `k` with `lib` as `k/src/lib.rs` and `files` as `(path under k/src, text)`.
    fn crate_k(lib: &str, files: &[(&str, &str)]) -> tempfile::TempDir {
        let tmp = tempfile::tempdir().unwrap();
        let w = tmp.path();
        std::fs::write(w.join("Cargo.toml"), "[workspace]\nmembers = [\"k\"]\n").unwrap();
        std::fs::create_dir_all(w.join("k/src")).unwrap();
        std::fs::write(w.join("k/Cargo.toml"), "[package]\nname = \"k\"\n").unwrap();
        std::fs::write(w.join("k/src/lib.rs"), lib).unwrap();
        for (path, text) in files {
            std::fs::write(w.join("k/src").join(path), text).unwrap();
        }
        tmp
    }

    /// A struct, enum, union or type alias with no impl in its module is still a TYPE segment: the refusal
    /// names the missing method of that type, never a missing `mod` (mutants `code.rs:835-839`, each arm
    /// deleted, #4588).
    #[test]
    fn every_kind_of_type_definition_is_a_type_segment() {
        let tmp = crate_k(
            "pub struct S;\npub enum E { A }\npub union U { a: u32 }\npub type T = u8;\n",
            &[],
        );
        let ws = Workspace::scan(tmp.path());
        for ty in ["S", "E", "U", "T"] {
            let e = Resolver::new(&ws)
                .resolve(&format!("k::{ty}"), "f")
                .expect_err("no impl anywhere");
            assert!(
                e.reason.contains(&format!("in an `impl {ty}`")),
                "{ty}: {}",
                e.reason
            );
        }
    }

    /// `Type::function` finds a method only in an impl FOR that type or a trait OF that name, and an extra
    /// segment that is not `<Arg>` is refused as "a type, not a module" (mutants `code.rs:647` guard → true,
    /// `:659` guard → true, `:662` guard → true and `&&` → `||`, #4588).
    #[test]
    fn a_type_member_is_searched_only_in_its_own_impls_and_trait() {
        let tmp = crate_k(
            "pub struct T;\npub struct U;\nimpl U { pub fn f() {} }\n\
             pub trait Other { fn g(); }\npub trait Tr { fn h(); }\n",
            &[],
        );
        let ws = Workspace::scan(tmp.path());
        let mut r = Resolver::new(&ws);
        r.resolve("k::U", "f").expect("U's own impl");
        r.resolve("k::Tr", "h").expect("the trait's own fn");
        assert!(r.resolve("k::T", "f").is_err(), "f is U's, not T's");
        assert!(
            r.resolve("k::T", "g").is_err(),
            "g is trait Other's, not T's"
        );
        assert!(
            r.resolve("k::Tr::<X>", "h").is_err(),
            "a trait takes no <Arg> here"
        );
        let e = r.resolve("k::U::x", "f").expect_err("x is not a module");
        assert!(e.reason.contains("is a type, not a module"), "{}", e.reason);
    }

    /// Only a FILE-backed child `mod x;` can include a sibling file: an inline `mod c { }` that happens to share
    /// its name with a file `c.rs` including `inc.rs` does not make `k::inc` resolve (mutant `code.rs:766` guard
    /// → true, #4588).
    #[test]
    fn an_inline_module_does_not_include_a_sibling_file() {
        let tmp = crate_k(
            "pub mod c {}\n",
            &[
                ("c.rs", "include!(\"inc.rs\");\n"),
                ("inc.rs", "pub fn f() {}\n"),
            ],
        );
        let ws = Workspace::scan(tmp.path());
        assert!(Resolver::new(&ws).resolve("k::inc", "f").is_err());
        let file_backed = crate_k(
            "pub mod c;\n",
            &[
                ("c.rs", "include!(\"inc.rs\");\n"),
                ("inc.rs", "pub fn f() {}\n"),
            ],
        );
        let ws = Workspace::scan(file_backed.path());
        Resolver::new(&ws)
            .resolve("k::inc", "f")
            .expect("the same layout with a file-backed mod resolves");
    }

    /// An inline `mod foo { }` is not a file-backed child: a `foo.rs` on disk that happens to include `inc.rs`
    /// must not make `inc` resolvable from the parent.
    #[test]
    fn an_inline_mod_is_not_a_file_backed_child_that_includes_a_sibling() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let w = tmp.path();
        std::fs::write(w.join("Cargo.toml"), "[workspace]\nmembers = [\"k\"]\n").expect("write");
        std::fs::create_dir_all(w.join("k/src")).expect("mkdir");
        std::fs::write(w.join("k/Cargo.toml"), "[package]\nname = \"k\"\n").expect("write");
        std::fs::write(w.join("k/src/lib.rs"), "pub mod foo {}\n").expect("write");
        std::fs::write(w.join("k/src/foo.rs"), "include!(\"inc.rs\");\n").expect("write");
        std::fs::write(w.join("k/src/inc.rs"), "pub fn f() {}\n").expect("write");
        let ws = Workspace::scan(w);
        assert!(Resolver::new(&ws).resolve("k::inc", "f").is_err());
    }

    #[test]
    fn join_path_is_exact() {
        assert_eq!(join_path("a::b", &[]), "a::b");
        assert_eq!(join_path("a", &["b", "c"]), "a::b::c");
    }

    #[test]
    fn defines_type_sees_each_type_item_only_by_its_own_name() {
        for (src, name) in [
            ("struct S;", "S"),
            ("enum E { A }", "E"),
            ("union U { a: u8 }", "U"),
            ("trait T {}", "T"),
            ("type A = u8;", "A"),
        ] {
            let ast = syn::parse_file(src).expect("parses");
            assert!(defines_type(&ast.items, name), "{src}");
            assert!(!defines_type(&ast.items, "Nope"), "{src}");
        }
        let f = syn::parse_file("fn S() {}").expect("parses");
        assert!(!defines_type(&f.items, "S"));
    }

    #[test]
    fn find_trait_fn_matches_the_named_fn_only() {
        let one: syn::ItemTrait = syn::parse_str("trait T { fn b(); }").expect("parses");
        assert!(find_trait_fn(&one.items, "b").is_some());
        assert!(find_trait_fn(&one.items, "c").is_none());
        let two: syn::ItemTrait = syn::parse_str("trait T { fn a(); fn b() {} }").expect("parses");
        assert!(find_trait_fn(&two.items, "a").is_some());
        assert!(find_trait_fn(&two.items, "b").is_some());
        assert!(find_trait_fn(&two.items, "c").is_none());
    }

    #[test]
    fn glob_paths_reads_groups_and_refuses_a_bare_glob() {
        let globs = |src: &str| {
            let u: syn::ItemUse = syn::parse_str(src).expect("parses");
            let mut out = Vec::new();
            glob_paths(&u.tree, "", &mut out);
            out
        };
        assert_eq!(globs("use a::b::*;"), ["a::b"]);
        assert_eq!(globs("use a::{b::*, c::d::*, e};"), ["a::b", "a::c::d"]);
        assert!(globs("use *;").is_empty());
    }

    #[test]
    fn item_named_reads_union_and_static_items() {
        let ast = syn::parse_file("pub union U { a: u8 }\nstatic S: u8 = 0;").expect("parses");
        let u = item_named(&ast.items[0], "U").expect("union");
        assert_eq!(u.kind, "union");
        assert_eq!(u.visibility, "pub");
        assert!(item_named(&ast.items[0], "Nope").is_none());
        let s = item_named(&ast.items[1], "S").expect("static");
        assert_eq!(s.kind, "static");
        assert!(item_named(&ast.items[1], "Nope").is_none());
    }

    #[test]
    fn the_positive_control_fires() {
        assert!(positive_control());
    }

    /// `[workspace]` membership: multi-line arrays, either quote, trailing `/`, comments, `exclude`, keys outside
    /// `[workspace]` ignored, a root `[package]` recorded, and no `[workspace]` table at all is `None`.
    #[test]
    fn workspace_membership_reads_members_and_exclude_in_every_shape() {
        let text = "[package]\nname = \"root\"\nmembers = [\"not-a-workspace-key\"]\n\n\
                    [workspace]\nresolver = \"2\"\nmembers = [\n  \"crates/*\", # every crate\n  'tools/x/',\n]\n\
                    exclude = [\"crates/skip\"]\n\n[workspace.dependencies]\nmembers = [\"nope\"]\n";
        let m = workspace_membership(text).expect("a [workspace] table");
        assert_eq!(m.members, ["crates/*", "tools/x"]);
        assert_eq!(m.exclude, ["crates/skip"]);
        assert!(m.root_package);
        let root = Path::new("/r");
        assert!(m.admits(root, Path::new("/r/crates/a")));
        assert!(!m.admits(root, Path::new("/r/crates/skip")));
        assert!(m.admits(root, Path::new("/r")));
        assert!(workspace_membership("[package]\nname = \"x\"\n").is_none());
    }

    #[test]
    fn manifest_names_reads_package_and_lib_sections_only() {
        let m = manifest_names("[package]\nname = \"a-b\"\n[dependencies]\nname = \"x\"\n[lib]\nname = \"ab\"\npath = \"src/x.rs\"\n");
        assert_eq!(m.package.as_deref(), Some("a-b"));
        assert_eq!(m.lib.as_deref(), Some("ab"));
        assert_eq!(m.lib_path.as_deref(), Some("src/x.rs"));
    }

    /// ONT-4c4: only a `#[kernel]` item carries `unsafeFree` / `boundsChecked`; any other attribute carries neither.
    #[test]
    fn emit_writes_the_body_facts_only_for_a_kernel_attribute() {
        let b = Bound {
            contract: "c-v1".into(),
            equation: "e".into(),
            module_path: "kern::m".into(),
            function: "f".into(),
            status: ImplStatus::Implemented,
        };
        let nt_of = |attrs: &[&str]| {
            let found: Result<Resolved, Unresolved> = Ok(Resolved {
                file: "src/m.rs".into(),
                visibility: "pub".into(),
                kind: "fn".into(),
                attributes: attrs.iter().map(|a| (*a).to_string()).collect(),
                unsafe_free: true,
                bounds_checked: false,
            });
            let mut g = Graph::new();
            emit(&mut g, &b, &found);
            g.to_ntriples()
        };
        let kernel = nt_of(&["kernel"]);
        assert!(kernel.contains("/sym/unsafeFree> \"true\""), "{kernel}");
        assert!(kernel.contains("/sym/boundsChecked> \"false\""), "{kernel}");
        let plain = nt_of(&["inline"]);
        assert!(!plain.contains("unsafeFree"), "{plain}");
        assert!(!plain.contains("boundsChecked"), "{plain}");
        assert!(plain.contains("/sym/attribute> \"inline\""), "{plain}");
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

    /// A one-crate workspace `k` under a tempdir: `files` are `(path under k/src, text)`, `lib` is `lib.rs`.
    fn one_crate(lib: &str, files: &[(&str, &str)]) -> tempfile::TempDir {
        let tmp = tempfile::tempdir().expect("tempdir");
        let w = tmp.path();
        std::fs::write(w.join("Cargo.toml"), "[workspace]\nmembers = [\"k\"]\n").expect("write");
        std::fs::create_dir_all(w.join("k/src")).expect("mkdir");
        std::fs::write(w.join("k/Cargo.toml"), "[package]\nname = \"k\"\n").expect("write");
        std::fs::write(w.join("k/src/lib.rs"), lib).expect("write");
        for (p, t) in files {
            std::fs::write(w.join("k/src").join(p), t).expect("write");
        }
        tmp
    }

    /// #4587: only a member's own `Cargo.toml` is indexed — an excluded crate's manifest and a non-manifest
    /// `.toml` file in a member directory are not.
    #[test]
    fn scan_indexes_member_manifests_only() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let w = tmp.path();
        std::fs::write(
            w.join("Cargo.toml"),
            "[workspace]\nmembers = [\"crates/*\"]\nexclude = [\"crates/skip\"]\n",
        )
        .expect("write");
        for c in ["keep", "skip"] {
            std::fs::create_dir_all(w.join("crates").join(c)).expect("mkdir");
            std::fs::write(
                w.join("crates").join(c).join("Cargo.toml"),
                format!("[package]\nname = \"{c}\"\n"),
            )
            .expect("write");
        }
        std::fs::write(
            w.join("crates/keep/other.toml"),
            "[package]\nname = \"sneak\"\n",
        )
        .expect("write");
        let ws = Workspace::scan(w);
        assert!(ws.is_workspace_root);
        assert!(ws.crates.contains_key("keep"), "{:?}", ws.crates);
        assert!(!ws.crates.contains_key("skip"), "{:?}", ws.crates);
        assert!(!ws.crates.contains_key("sneak"), "{:?}", ws.crates);
    }

    /// The root directory is a member only as a root `[package]` or as an explicit `"."` member.
    #[test]
    fn admits_the_root_only_as_a_package_or_a_dot_member() {
        let root = Path::new("/r");
        let ws_only = workspace_membership("[workspace]\nmembers = [\"crates/a\"]\n").expect("ws");
        assert!(!ws_only.root_package, "no [package] table");
        assert!(!ws_only.admits(root, root), "no `.` member, no package");
        let dot = workspace_membership("[workspace]\nmembers = [\".\"]\n").expect("ws");
        assert!(dot.admits(root, root));
        assert!(!dot.admits(root, Path::new("/r/crates/a")));
    }

    #[test]
    fn section_name_reads_a_bracketed_header_line_only() {
        assert_eq!(section_name("[workspace]").as_deref(), Some("workspace"));
        assert_eq!(section_name("[ a.b ]").as_deref(), Some("a.b"));
        assert_eq!(section_name("[open"), None, "no closing bracket");
        assert_eq!(section_name("close]"), None, "no opening bracket");
        assert_eq!(section_name("plain"), None);
        assert_eq!(section_name("[a = b]"), None, "a key line");
        assert_eq!(section_name("x = [\"a\"]"), None);
    }

    #[test]
    fn quoted_reads_every_string_on_the_line() {
        assert_eq!(quoted(r#""a", "b""#), ["a", "b"]);
        assert_eq!(quoted(r#"'a/' , "b" 'c'"#), ["a", "b", "c"]);
        assert!(quoted("none here").is_empty());
    }

    #[test]
    fn glob_match_needs_room_for_both_prefix_and_suffix() {
        assert!(glob_match("crates/apr-*", "crates/apr-cli"));
        assert!(glob_match("ab*bc", "abxbc"));
        assert!(!glob_match("ab*bc", "abc"), "prefix and suffix overlap");
        assert!(!glob_match("ab*b", "ab"), "too short for ab + b");
        assert!(!glob_match("a*c", "xc"), "suffix only");
        assert!(!glob_match("a*c", "ax"), "prefix only");
        assert!(glob_match("a*", "a"));
        assert!(!glob_match("crates/*", "crates/a/b"));
    }

    /// `include!` recurses to a bound: a file written at depth 8 is still spliced, one included from it is not,
    /// and a file that includes itself terminates.
    #[test]
    fn include_nesting_stops_at_the_bound_and_a_cycle_terminates() {
        let mut files: Vec<(String, String)> = Vec::new();
        for i in 1..=9 {
            let next = if i < 9 {
                format!("include!(\"f{}.rs\");\n", i + 1)
            } else {
                String::new()
            };
            files.push((format!("f{i}.rs"), format!("pub fn at_{i}() {{}}\n{next}")));
        }
        files.push((
            "cyc.rs".into(),
            "include!(\"cyc.rs\");\npub fn cyc_fn() {}\n".into(),
        ));
        let refs: Vec<(&str, &str)> = files
            .iter()
            .map(|(a, b)| (a.as_str(), b.as_str()))
            .collect();
        let tmp = one_crate("include!(\"f1.rs\");\ninclude!(\"cyc.rs\");\n", &refs);
        let ws = Workspace::scan(tmp.path());
        let mut r = Resolver::new(&ws);
        r.resolve("k", "at_1").expect("depth 1");
        r.resolve("k", "at_8").expect("a file spliced at depth 8");
        assert!(
            r.resolve("k", "at_9").is_err(),
            "an include written at depth 8 is refused"
        );
        r.resolve("k", "cyc_fn")
            .expect("a self-including file still terminates");
    }

    /// A re-export chain longer than the bound is refused; a short one resolves.
    #[test]
    fn a_long_re_export_chain_is_refused() {
        let mut lib = String::from("pub mod real { pub fn f() {} }\n");
        for i in 0..10 {
            let to = if i == 9 {
                "real".to_string()
            } else {
                format!("n{}", i + 1)
            };
            lib.push_str(&format!("pub use crate::{to} as n{i};\n"));
        }
        let tmp = one_crate(&lib, &[]);
        let ws = Workspace::scan(tmp.path());
        let mut r = Resolver::new(&ws);
        r.resolve("k::n8", "f").expect("two hops resolve");
        let e = r
            .resolve("k::n0", "f")
            .expect_err("ten hops exceed the bound");
        assert!(e.reason.contains("deeper than 8"), "{}", e.reason);
    }

    /// A chain of `pub use …::*` globs longer than the bound is refused; a short one resolves.
    #[test]
    fn a_long_glob_chain_is_refused() {
        let mut lib = String::from("pub use crate::g1::*;\n");
        for i in 1..=10 {
            if i == 10 {
                lib.push_str("pub mod g10 { pub fn f() {} }\n");
            } else {
                lib.push_str(&format!(
                    "pub mod g{i} {{ pub use crate::g{}::*; }}\n",
                    i + 1
                ));
            }
        }
        let tmp = one_crate(&lib, &[]);
        let ws = Workspace::scan(tmp.path());
        let mut r = Resolver::new(&ws);
        r.resolve("k::g7", "f").expect("three glob hops resolve");
        let e = r
            .resolve("k", "f")
            .expect_err("ten glob hops exceed the bound");
        assert!(e.reason.contains("deeper than 8"), "{}", e.reason);
    }

    /// Under several globs the error reported is the last NON-cycle one: a cycle on a later glob neither
    /// replaces it nor is it dropped in favour of the generic message.
    #[test]
    fn first_of_reports_the_last_non_cycle_error() {
        let tmp = one_crate(
            "pub mod a;\npub mod b;\npub use crate::a::*;\npub use crate::b::*;\n",
            &[
                ("a.rs", "pub use crate::b::*;\n"),
                ("b.rs", "pub fn other() {}\n"),
            ],
        );
        let ws = Workspace::scan(tmp.path());
        let e = Resolver::new(&ws)
            .resolve("k::seg", "f")
            .expect_err("no such module");
        assert!(e.reason.contains("k/src/b.rs"), "{}", e.reason);
        assert!(!e.reason.contains("re-export cycle"), "{}", e.reason);
        assert!(!e.reason.contains("nor under"), "{}", e.reason);
    }

    /// `Type::seg::f` — a non-generic segment after a type is not a generic argument.
    #[test]
    fn a_plain_segment_after_a_type_is_not_a_module() {
        let tmp = one_crate("pub struct T;\nimpl T { pub fn f() {} }\n", &[]);
        let ws = Workspace::scan(tmp.path());
        let e = Resolver::new(&ws)
            .resolve("k::T::x", "f")
            .expect_err("T is a type");
        assert!(e.reason.contains("is a type, not a module"), "{}", e.reason);
    }

    /// #4587: `Type::f` is found only in an `impl` for `Type` — another type's `impl` with an `f` is no hit.
    #[test]
    fn type_member_ignores_an_impl_for_another_type() {
        let tmp = one_crate(
            "pub struct T;\npub struct U;\nimpl U { pub fn f() {} }\n",
            &[],
        );
        let ws = Workspace::scan(tmp.path());
        let e = Resolver::new(&ws)
            .resolve("k::T", "f")
            .expect_err("T has no f; only U does");
        assert!(
            e.reason.contains("no `fn f` in an `impl T`"),
            "{}",
            e.reason
        );
    }

    /// #4587: `Type::f` is found in `trait Type` only — another trait's `f`, or a generic-qualified path
    /// to a trait, is no hit.
    #[test]
    fn type_member_matches_the_named_non_generic_trait_only() {
        let tmp = one_crate(
            "pub struct T;\npub trait Other { fn f(); }\npub trait G { fn g(); }\n",
            &[],
        );
        let ws = Workspace::scan(tmp.path());
        let e = Resolver::new(&ws)
            .resolve("k::T", "f")
            .expect_err("Other::f is not T::f");
        assert!(e.reason.contains("no `fn f`"), "{}", e.reason);
        assert!(
            Resolver::new(&ws).resolve("k::G", "g").is_ok(),
            "trait G::g resolves"
        );
        assert!(
            Resolver::new(&ws).resolve("k::G::<u8>", "g").is_err(),
            "a trait is not generic-qualified"
        );
    }
}
