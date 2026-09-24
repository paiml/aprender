"""List every include!() target in a crate, resolved against the including file.
Separate file rather than an inline heredoc: bashrs parses an embedded heredoc
as shell, so python assignments read as SC1007 "space after =" -- eight phantom
errors. Same reason assertions_exclude.awk and workflow_path_filters.py live
here.
argv: <crate-dir> [--escapes]
stdout: "<crate-relative target>\t<crate-relative including file>" per line

--escapes (PMAT-958): instead of include!() targets, list every include_str!/
include_bytes! target in NON-TEST, HOST-COMPILED code whose path escapes the
crate directory. Such a file can never be in the package tarball, so `cargo
publish` fails its verification build (aprender-test-lib 0.65.1:
`../../../../scripts/perf-matrix.yaml`, cascade stuck at 67/74). Skipped, and
why: test code (a `#[cfg(test)]` module is not compiled by the verification
build; the crate-local *_tests.rs files legitimately read
`../../../../contracts/*.yaml`) and wasm32-only files (`use wasm_bindgen`),
which the host verification build never compiles either -- those are printed
on stderr as SKIPPED so the residual is visible, never silent.

A whole FILE is test code too when its parent declares it out of line under a
test-only cfg -- `#[cfg(test)] mod guard;` -- and so is every module that file
declares in turn (#4048: aprender-serve's fusion_call_site_guard_3985.rs, whose
own text carries no cfg at all). Those are SKIPPED on stderr, by name. A module
declared under `#[cfg(any(test, ...))]`, or with a test cfg on only SOME of its
items, is compiled by the host build and still judged.
"""
import os
import re
import sys

PAT_INCLUDE = re.compile(r'include!\s*\(\s*"([^"]+)"\s*\)')
PAT_DATA = re.compile(r'include_(?:str|bytes)!\s*\(\s*"([^"]+)"\s*\)')
# include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../x")) is the same escape
# spelled through concat!; OUT_DIR-based concat! is a build-script product and fine.
PAT_CONCAT = re.compile(
    r'include_(?:str|bytes)!\s*\(\s*concat!\s*\(\s*env!\s*\(\s*"CARGO_MANIFEST_DIR"\s*\)\s*,\s*"([^"]+)"'
)
LINE_COMMENT = re.compile(r"//[^\n]*")
BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.S)
# `#[cfg(test)]` and `#[cfg(all(test, …))]` are test-only; `any(test, …)` is not.
CFG_TEST = re.compile(r"#\[cfg\((?:test|all\(\s*test\b[^)]*\))\)\]")
WASM_USE = re.compile(r"^\s*use\s+wasm_bindgen", re.M)
# An out-of-line module declaration, possibly behind more attributes (`#[path = "…"]`,
# `#[allow(…)]`) after the cfg: `#[cfg(test)] #[path = "g.rs"] pub(crate) mod guard;`
ATTR = re.compile(r"\s*#\[[^\]]*\]")
MOD_DECL = re.compile(r"\s*(?:pub(?:\s*\([^)]*\))?\s+)?mod\s+([A-Za-z_][A-Za-z0-9_]*)\s*;")
PATH_ATTR = re.compile(r'#\[\s*path\s*=\s*"([^"]+)"\s*\]')


def is_test_file(rel):
    base = os.path.basename(rel)
    return base.endswith("_tests.rs") or base == "tests.rs" or "/tests/" in rel or "/tests_" in rel


def rust_files(src):
    for root, _dirs, files in os.walk(src):
        for fn in files:
            if fn.endswith(".rs"):
                yield root, os.path.join(root, fn)


def read(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return None


def strip_comments(text):
    return LINE_COMMENT.sub("", BLOCK_COMMENT.sub("", text))


def item_end(text, start):
    """Index just past the item that starts at `start`: a brace-matched body, or the next `;`."""
    brace = text.find("{", start)
    semi = text.find(";", start)
    if brace < 0 or (0 <= semi < brace):
        return len(text) if semi < 0 else semi + 1
    depth, i = 0, brace
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return len(text)


def host_body(text):
    """The part of a file the host verification build compiles: comments gone, every
    `#[cfg(test)]` item removed (a `mod tests { … }` block or a single item), and
    production code AFTER a test module kept — truncating at the first `#[cfg(test)]`
    would hide an escape that follows it."""
    text = strip_comments(text)
    out, pos = [], 0
    for m in CFG_TEST.finditer(text):
        if m.start() < pos:
            continue
        out.append(text[pos : m.start()])
        pos = item_end(text, m.end())
    out.append(text[pos:])
    return "".join(out)


def child_module_files(path, text):
    """[(child_file, test_only)] for every out-of-line `mod x;` in `path`, resolved the way
    rustc resolves it (`x.rs` / `x/mod.rs` beside a lib.rs/main.rs/mod.rs, under `<stem>/`
    otherwise, or `#[path]`)."""
    text = strip_comments(text)
    d = os.path.dirname(path)
    base = os.path.basename(path)
    sub = d if base in ("lib.rs", "main.rs", "mod.rs") else os.path.join(d, base[:-3])
    out = []
    for m in MOD_DECL.finditer(text):
        # the attributes directly above this declaration
        attrs, j = [], m.start()
        head = text[:j]
        while True:
            a = re.search(r"#\[[^\]]*\]\s*$", head)
            if not a:
                break
            attrs.insert(0, a.group(0).strip())
            head = head[: a.start()]
        test_only = any(CFG_TEST.fullmatch(a) for a in attrs)
        name = m.group(1)
        pa = next((PATH_ATTR.fullmatch(a) for a in attrs if PATH_ATTR.fullmatch(a)), None)
        if pa:
            cands = [os.path.normpath(os.path.join(d, pa.group(1)))]
        else:
            cands = [os.path.join(sub, name + ".rs"), os.path.join(sub, name, "mod.rs")]
        f = next((c for c in cands if os.path.isfile(c)), None)
        if f:
            out.append((os.path.normpath(f), test_only))
    return out


def test_only_module_files(src):
    """Every file compiled only under a test cfg: declared `#[cfg(test)] mod x;`, or declared
    (any way) by such a file."""
    kids = {}
    for _root, path in rust_files(src):
        text = read(path)
        if text is not None:
            kids[os.path.normpath(path)] = child_module_files(path, text)
    test_only, frontier = set(), [c for cs in kids.values() for c, t in cs if t]
    while frontier:
        f = frontier.pop()
        if f in test_only:
            continue
        test_only.add(f)
        frontier.extend(c for c, _t in kids.get(f, []))
    return test_only


TEST_ONLY = set()


def escapes_in(crate, root, rel_path, text):
    if is_test_file(rel_path):
        return []
    if os.path.normpath(os.path.join(crate, rel_path)) in TEST_ONLY:
        print(f"SKIPPED (cfg(test)-only module, not in the host verification build): {rel_path}", file=sys.stderr)
        return []
    body = host_body(text)
    if WASM_USE.search(body):  # a real `use wasm_bindgen` line, not one in a comment
        print(f"SKIPPED (wasm32-only, not in the host verification build): {rel_path}", file=sys.stderr)
        return []
    found = []
    for m in PAT_DATA.finditer(body):
        rel = os.path.relpath(os.path.normpath(os.path.join(root, m.group(1))), crate)
        if rel.startswith(".."):
            found.append((rel, rel_path))
    for m in PAT_CONCAT.finditer(body):
        rel = os.path.relpath(os.path.normpath(crate + "/" + m.group(1)), crate)
        if rel.startswith(".."):
            found.append((rel, rel_path))
    return found


def includes_in(crate, root, rel_path, text):
    return [
        (os.path.relpath(os.path.normpath(os.path.join(root, m.group(1))), crate), rel_path)
        for m in PAT_INCLUDE.finditer(text)
    ]


def main():
    crate = sys.argv[1]
    escape_mode = "--escapes" in sys.argv[2:]
    judge = escapes_in if escape_mode else includes_in
    if escape_mode:
        TEST_ONLY.update(test_only_module_files(os.path.join(crate, "src")))
    for root, path in rust_files(os.path.join(crate, "src")):
        text = read(path)
        if text is None:
            continue
        for target, from_file in judge(crate, root, os.path.relpath(path, crate), text):
            print(f"{target}\t{from_file}")


main()
