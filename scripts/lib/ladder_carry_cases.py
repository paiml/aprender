"""ladder_carry_cases.py -- hermetic case table + mutants for ladder_carry.py (#4037).

    python3 scripts/lib/ladder_carry_cases.py            # every row must hold
    python3 scripts/lib/ladder_carry_cases.py --mutants  # every mutant must break its NAMED row

The fixture is a synthetic workspace (metadata built here, files written to a temp git repo), so no
row depends on the real tree. MUST-RED: `inference-src` -- a diff touching the inference path must NOT
carry forward.
"""

import importlib.util
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)   # ladder_carry imports ladder_equiv (the Cargo.lock rule) from beside it

TOML_A = """[workspace.package]
version = "0.69.1"
[workspace.dependencies]
aprender-core = { path = "crates/aprender-core", version = "0.69.1" }
serde = { version = "1.0", features = ["derive"] }
[package]
name = "aprender"
version = "0.69.1"
"""
TOML_BUMP = TOML_A.replace('"0.69.1"', '"0.70.0"')
TOML_DEP_ADDED = TOML_BUMP + '[dependencies]\nrand = "0.9"\n'
TOML_SERDE_MOVED = TOML_BUMP.replace('version = "1.0"', 'version = "1.1"')
TOML_FEATURE = TOML_BUMP.replace('features = ["derive"]', 'features = ["derive", "rc"]')
LOCK_A = """version = 4
[[package]]
name = "aprender-core"
version = "0.69.1"
dependencies = ["serde"]
[[package]]
name = "serde"
version = "1.0.200"
source = "registry+https://github.com/rust-lang/crates.io-index"
checksum = "aa"
"""
LOCK_BUMP = LOCK_A.replace('"0.69.1"', '"0.70.0"')
LOCK_SERDE = LOCK_BUMP.replace('"1.0.200"', '"1.0.201"').replace('"aa"', '"bb"')


def pkg(name, d, deps=(), apr=False):
    targets = [{"kind": ["lib"], "name": name.replace("-", "_")}]
    if apr:
        targets.append({"kind": ["bin"], "name": "apr"})
    return {"name": name, "manifest_path": "/ws/%sCargo.toml" % (d + "/" if d else ""), "targets": targets,
            "dependencies": [dict(name=n, path="/ws/x", kind=k, optional=o) for n, k, o in deps]}


def meta(joiner=False):
    serve_deps = [("buildc", "build", False), ("optc", None, True)] + ([("joiner", None, False)] if joiner else [])
    return {"workspace_root": "/ws", "packages": [
        pkg("aprender", "", [("apr-cli", None, False)], apr=True),
        pkg("apr-cli", "crates/apr-cli", [("serve", None, False), ("devonly", "dev", False)], apr=True),
        pkg("serve", "crates/serve", serve_deps),
        pkg("buildc", "crates/buildc"), pkg("optc", "crates/optc"), pkg("devonly", "crates/devonly"),
        pkg("other", "crates/other"), pkg("joiner", "crates/joiner"),
    ]}


FILES = {
    "crates/apr-cli/src/lib.rs": 'const C: &str = include_str!("../../../contracts/c.yaml");\n',
    "crates/serve/src/lib.rs": 'const F: &[u8] = include_bytes!("../tests/golden.bin");\n',
    "crates/serve/build.rs": 'fn main() { let _ = "configs"; println!("cargo:rerun-if-changed=../../data/model.bin"); }\n',
    "contracts/c.yaml": "x: 1\n", "contracts/other.yaml": "x: 2\n", "configs/a.yaml": "a: 1\n", "data/model.bin": "m",
    "scripts/prod.sh": 'echo \'{"schema": "crux-inference-receipt/v1"}\'; python3 scripts/lib/h.py\n',
    "scripts/lib/h.py": "print(1)\n",
    "scripts/drive.sh": "bash scripts/prod.sh\n",
    "scripts/other.sh": "echo hi\n",
    "scripts/check_prod.sh": 'grep -q \'"schema": "crux-inference-receipt/v1"\' r.json\n',
}


def repo():
    d = tempfile.mkdtemp(prefix="ladder-carry-")
    for p, t in FILES.items():
        os.makedirs(os.path.join(d, os.path.dirname(p)), exist_ok=True)
        open(os.path.join(d, p), "w").write(t)
    subprocess.run(["git", "init", "-q", d], check=True)
    subprocess.run(["git", "-C", d, "add", "-A"], check=True)
    return d


# (row, paths, carries?, meta_a joiner, meta_b joiner, use roots)
ROWS = [
    ("inference-src", ["crates/serve/src/infer.rs"], False, 0, 0, 1),        # MUST-RED
    ("mixed-diff", ["docs/x.md", "crates/serve/src/infer.rs"], False, 0, 0, 1),
    ("docs-only", ["docs/x.md", "evidence/y.json"], True, 0, 0, 1),
    ("non-closure-crate", ["crates/other/src/lib.rs"], True, 0, 0, 1),
    ("dev-dep-crate", ["crates/devonly/src/lib.rs"], True, 0, 0, 1),
    ("build-dep-crate", ["crates/buildc/src/lib.rs"], False, 0, 0, 1),
    ("optional-dep-crate", ["crates/optc/src/lib.rs"], False, 0, 0, 1),
    ("test-only", ["crates/serve/tests/t.rs", "crates/serve/benches/b.rs", "crates/apr-cli/examples/e.rs"], True, 0, 0, 1),
    ("embedded-test-fixture", ["crates/serve/tests/golden.bin"], False, 0, 0, 1),
    ("closure-manifest", ["crates/serve/Cargo.toml"], False, 0, 0, 1),
    ("root-facade-src", ["src/bin/apr.rs"], False, 0, 0, 1),
    ("cargo-lock", ["Cargo.lock"], False, 0, 0, 1),
    ("root-manifest", ["Cargo.toml"], False, 0, 0, 1),
    ("toolchain", ["rust-toolchain.toml"], False, 0, 0, 1),
    ("cargo-config", [".cargo/config.toml"], False, 0, 0, 1),
    ("joins-closure-at-b", ["crates/joiner/src/lib.rs"], False, 0, 1, 1),
    ("leaves-closure-at-b", ["crates/joiner/src/lib.rs"], False, 1, 0, 1),
    ("embedded-contract", ["contracts/c.yaml"], False, 0, 0, 1),
    ("unembedded-contract", ["contracts/other.yaml"], True, 0, 0, 1),
    ("build-rs-dir", ["configs/a.yaml"], False, 0, 0, 1),
    ("build-rs-relative", ["data/model.bin"], False, 0, 0, 1),
    ("receipt-producer", ["scripts/prod.sh"], False, 0, 0, 1),
    ("producer-helper", ["scripts/lib/h.py"], False, 0, 0, 1),
    ("producer-driver", ["scripts/drive.sh"], False, 0, 0, 1),
    ("unrelated-script", ["scripts/other.sh"], True, 0, 0, 1),
    ("judge-script", ["scripts/check_prod.sh"], True, 0, 0, 1),
    ("unknown-sets-impact", ["scripts/other.sh"], False, 0, 0, 0),
]


def run(mod):
    root = repo()
    res = {}
    for row, paths, want, ja, jb, use in ROWS:
        got, proof = mod.carry(paths, meta(ja), meta(jb), roots=[root] if use else None)
        res[row] = (got == want, "want carry=%s got %s -- %s" % (want, got, proof[:160]))
    nob = {"workspace_root": "/ws", "packages": [pkg("other", "crates/other")]}
    got, proof = mod.carry(["docs/x.md"], nob, nob)
    res["no-apr-binary"] = (got is False and "no package" in proof, proof[:120])
    vo = mod.version_only
    res["bump-toml-version-only"] = (vo("Cargo.toml", TOML_A, TOML_BUMP) is True, "workspace + path-dep versions")
    res["toml-dep-added-not-bump"] = (vo("Cargo.toml", TOML_A, TOML_DEP_ADDED) is False, "a new dependency")
    res["toml-external-version-moved"] = (vo("Cargo.toml", TOML_A, TOML_SERDE_MOVED) is False, "serde 1.0 -> 1.1")
    res["toml-feature-changed"] = (vo("Cargo.toml", TOML_A, TOML_FEATURE) is False, "a feature added")
    res["lock-bump-version-only"] = (vo("Cargo.lock", LOCK_A, LOCK_BUMP) is True, "workspace package versions")
    res["lock-external-dep-moved"] = (vo("Cargo.lock", LOCK_A, LOCK_SERDE) is False, "serde 1.0.200 -> 1.0.201")
    try:
        absent = vo("Cargo.toml", None, TOML_BUMP)
    except Exception as exc:   # a crash is not an answer: this row is RED
        absent = "raised %r" % exc
    res["one-side-absent-not-bump"] = (absent is False, absent)
    got, _ = mod.carry(["Cargo.lock", "Cargo.toml"], meta(), meta(), roots=[root], bumped={"Cargo.lock", "Cargo.toml"})
    got2, _ = mod.carry(["Cargo.lock", "Cargo.toml"], meta(), meta(), roots=[root])
    res["carry-bumped"] = (got is True and got2 is False, (got, got2))
    got, proof = mod.carry(["crates/serve/src/infer.rs"], meta(), meta(), roots=[root])
    res["refusal-names-path"] = ("crates/serve/src/infer.rs" in proof and "serve" in proof, proof[:120])
    got, proof = mod.carry(["docs/x.md", "crates/other/a.rs"], meta(), meta(), roots=[root])
    res["proof-names-cleared"] = ("docs/x.md" in proof and "crates/other/a.rs" in proof, proof[:120])
    return res


# (mutant, old, new, row that must go RED)
MUTANTS = [
    ("dev-filter-flipped", 'if d.get("kind") == "dev":', 'if d.get("kind") == "build":', "build-dep-crate"),
    ("optional-skipped", 'if d.get("kind") == "dev":', 'if d.get("kind") == "dev" or d.get("optional"):', "optional-dep-crate"),
    ("union-drops-a", "clos = dict(ca)", "clos = {}", "leaves-closure-at-b"),
    ("union-drops-b", "clos.update(cb)", "pass", "joins-closure-at-b"),
    ("src-test-only", 'TEST_ONLY_DIRS = ("tests", "benches", "examples")', 'TEST_ONLY_DIRS = ("tests", "benches", "examples", "src")', "inference-src"),
    ("lock-ignored", 'ROOT_IMPACT = ("Cargo.lock", "Cargo.toml")', 'ROOT_IMPACT = ("Cargo.toml",)', "cargo-lock"),
    ("toolchain-ignored", 'CARGO_BUILD_INPUTS = ("rust-toolchain", "rust-toolchain.toml", ".cargo/")', 'CARGO_BUILD_INPUTS = (".cargo/",)', "toolchain"),
    ("embedded-ignored", 'return True, "embedded or read at build time', 'return False, "embedded or read at build time', "embedded-contract"),
    ("build-rs-ignored", "if lit in top:", "if False:", "build-rs-dir"),
    ("build-rs-relative-ignored", 'if "/" in lit and ".." in lit:', "if False:", "build-rs-relative"),
    ("producers-ignored", 'return True, "produces the ladder/CRUX receipts', 'return False, "produces the ladder/CRUX receipts', "receipt-producer"),
    ("no-drivers", "seen = writers | {f for f in files if any(n in text(f) for n in wnames)}", "seen = set(writers)", "producer-driver"),
    ("no-forward-refs", "seen.add(m.group(0)); todo.append(m.group(0))", "pass", "producer-helper"),
    ("judges-produce", 'files = [f for f in files if not posixpath.basename(f).startswith("check_") and "_cases" not in f]', "files = files", "judge-script"),
    ("unknown-clears", 'return True, "belongs to no crate, and the build/measurement', 'return False, "belongs to no crate, and the build/measurement', "unknown-sets-impact"),
    ("no-crate-all-impact", 'return False, "belongs to no crate, is not embedded', 'return True, "belongs to no crate, is not embedded', "unrelated-script"),
    ("no-root-package", 'if best is None and path.split("/", 1)[0] in ROOT_PACKAGE_PATHS:', "if False:", "root-facade-src"),
    ("fixture-not-embedded", "if hit and rest.split", "if False and rest.split", "embedded-test-fixture"),
    ("path-agnostic-strip", '            if "path" in x:\n                x.pop("version", None)', '            x.pop("version", None)', "toml-external-version-moved"),
    ("lock-always-bump", "        return ladder_equiv.lock_dep_change(before, after) is None", "        return True", "lock-external-dep-moved"),
    ("bumped-ignored", "    if path in bumped:", "    if False:", "carry-bumped"),
    ("absent-is-bump", "    if before is None or after is None:\n        return False", "    if False:\n        return False", "one-side-absent-not-bump"),
    ("no-bin-carries", 'return False, "no package declares an `apr` binary', 'return True, "no package declares an `apr` binary', "no-apr-binary"),
]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    src = os.path.join(HERE, "ladder_carry.py")
    res = run(load(src, "ladder_carry"))
    bad = [r for r, (ok, _) in res.items() if not ok]
    for r, (ok, msg) in res.items():
        print("%s %-24s %s" % ("PASS" if ok else "FAIL", r, "" if ok else msg))
    print("rows: %d pass, %d fail" % (len(res) - len(bad), len(bad)))
    if bad:
        return 1
    if "--mutants" not in sys.argv:
        return 0
    text, survived = open(src).read(), 0
    tmp = tempfile.mkdtemp(prefix="ladder-carry-mut-")
    for name, old, new, row in MUTANTS:
        if text.count(old) != 1:
            print("MUTANT %-22s ANCHOR %d != 1" % (name, text.count(old)))
            survived += 1
            continue
        p = os.path.join(tmp, name + ".py")
        open(p, "w").write(text.replace(old, new))
        r = run(load(p, "m_" + name.replace("-", "_")))
        killed = not r[row][0]
        print("MUTANT %-22s %s by %s" % (name, "KILLED" if killed else "SURVIVED", row))
        survived += not killed
    print("mutants: %d/%d killed by their named row" % (len(MUTANTS) - survived, len(MUTANTS)))
    return 1 if survived else 0


if __name__ == "__main__":
    sys.exit(main())
