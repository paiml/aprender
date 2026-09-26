#!/usr/bin/env python3
"""Run CI job definitions as parallel SECTIONS inside one GitHub Actions job (#4433).

Operator ruling 2026-09-25 16:58 (Madrid): "CI = 5 fat jobs (x86 main, gx10,
yoga, mac-refusal, determinism). Each runs all parts to completion on the
biggest runner in its class ... Same test set as today (Σ executed unchanged)."

Actions runs the steps of one job serially. Today's clean-room fan-out was 15
jobs whose serial sum is ~9300 s against a 2733 s critical path, so a fat job
that ran them one after another would be 3x slower. This driver keeps every
job definition VERBATIM (ci/sections.yml, ci/vendor/sovereign-ci.yml) and runs
each one as a section: its own clone of the checkout, its own env, its own
step outcomes, all sections concurrently, `needs:` honoured between them.

What it emulates, and nothing more (anything else is refused, never guessed):
  * `${{ }}` in run/env/with/if: the expression subset the section files use.
  * GITHUB_OUTPUT / GITHUB_ENV / GITHUB_PATH / GITHUB_STEP_SUMMARY per step.
  * step `if`, `id`, `env`, `shell`, `working-directory`, `continue-on-error`,
    `timeout-minutes`; job `if`, `env`, `needs`, `outputs`, `timeout-minutes`,
    `continue-on-error`, `container` (docker exec into a long-lived container).
  * `uses:` handlers in USES below. An unknown action is a hard error.
  * `timeout-minutes` is SCALED by FAT_TIMEOUT_SCALE (default 2): see
    timeout_seconds().

Modes:
  run --sections a,b --results FILE [--background-until SECTION]
  wait --results FILE       (join a run started with --background-until)
  list                      (print the section/step inventory; the parity input)
"""
from __future__ import annotations

import argparse
import fnmatch
import glob
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
import zipfile
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - the runner image has PyYAML
    sys.exit("fat_driver: PyYAML is required")

ROOT = Path(__file__).resolve().parents[2]
SECTIONS_FILE = ROOT / "ci" / "sections.yml"
SOV_FILE = ROOT / "ci" / "vendor" / "sovereign-ci.yml"
CI_FILE = ROOT / ".github" / "workflows" / "ci.yml"


# --------------------------------------------------------------------------
# expressions
# --------------------------------------------------------------------------
class AnyShard:
    """matrix.shard in a one-shard run: every `matrix.shard == k` is true.

    aprender-84 (#4424): with SHARD=1 SHARDS=1 every fragment runs, and the
    once-only steps gated `matrix.shard == 1` AND `== 3` must BOTH run.
    """

    def __init__(self, text: str):
        self.text = text

    def __eq__(self, other):  # noqa: D105
        return True

    def __hash__(self):
        return 0

    def __str__(self):
        return self.text


class ExprError(Exception):
    pass


TOKEN_RE = re.compile(
    r"\s*(?:(?P<num>-?\d+(?:\.\d+)?)|(?P<str>'(?:[^']|'')*')|"
    r"(?P<op>==|!=|<=|>=|&&|\|\||[!<>()\[\],.*])|(?P<id>[A-Za-z_][A-Za-z0-9_-]*))"
)


def tokenize(src: str):
    pos, out = 0, []
    while pos < len(src):
        if src[pos:].strip() == "":
            break
        m = TOKEN_RE.match(src, pos)
        if not m or m.end() == pos:
            raise ExprError(f"cannot tokenize at {src[pos:]!r}")
        pos = m.end()
        kind = m.lastgroup
        out.append((kind, m.group(kind)))
    out.append(("end", None))
    return out


def truthy(v) -> bool:
    if isinstance(v, AnyShard):
        return True
    if v is None or v is False:
        return False
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return v != 0
    if isinstance(v, str):
        return v != ""
    return True


def to_str(v) -> str:
    if v is None:
        return ""
    if v is True:
        return "true"
    if v is False:
        return "false"
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    if isinstance(v, (dict, list)):
        return json.dumps(v)
    return str(v)


def loose_eq(a, b) -> bool:
    if isinstance(a, AnyShard) or isinstance(b, AnyShard):
        return True
    if type(a) is type(b) and not isinstance(a, str):
        return a == b
    if isinstance(a, str) and isinstance(b, str):
        return a.lower() == b.lower()

    def num(x):
        if x is None:
            return 0
        if isinstance(x, bool):
            return int(x)
        if isinstance(x, (int, float)):
            return x
        if isinstance(x, str):
            try:
                return float(x) if x.strip() else 0
            except ValueError:
                return float("nan")
        return float("nan")

    return num(a) == num(b)


class Evaluator:
    def __init__(self, contexts: dict, funcs: dict):
        self.ctx = contexts
        self.funcs = funcs

    def eval(self, src: str):
        self.toks = tokenize(src)
        self.i = 0
        v = self.or_()
        if self.toks[self.i][0] != "end":
            raise ExprError(f"trailing tokens in {src!r}")
        return v

    def peek(self):
        return self.toks[self.i]

    def take(self, val=None):
        t = self.toks[self.i]
        if val is not None and t[1] != val:
            raise ExprError(f"expected {val!r}, got {t[1]!r}")
        self.i += 1
        return t

    def or_(self):
        v = self.and_()
        while self.peek()[1] == "||":
            self.take()
            r = self.and_()
            v = v if truthy(v) else r
        return v

    def and_(self):
        v = self.cmp()
        while self.peek()[1] == "&&":
            self.take()
            r = self.cmp()
            v = r if truthy(v) else v
        return v

    def cmp(self):
        v = self.unary()
        while self.peek()[1] in ("==", "!=", "<", ">", "<=", ">="):
            op = self.take()[1]
            r = self.unary()
            if op == "==":
                v = loose_eq(v, r)
            elif op == "!=":
                v = not loose_eq(v, r)
            else:
                a, b = float(to_str(v) or 0), float(to_str(r) or 0)
                v = {"<": a < b, ">": a > b, "<=": a <= b, ">=": a >= b}[op]
        return v

    def unary(self):
        if self.peek()[1] == "!":
            self.take()
            return not truthy(self.unary())
        return self.postfix()

    def postfix(self):
        kind, val = self.peek()
        if kind == "num":
            self.take()
            v = float(val) if "." in val else int(val)
        elif kind == "str":
            self.take()
            v = val[1:-1].replace("''", "'")
        elif val == "(":
            self.take()
            v = self.or_()
            self.take(")")
        elif kind == "id":
            self.take()
            if val in ("true", "false"):
                return val == "true"
            if val == "null":
                return None
            if self.peek()[1] == "(":
                self.take()
                args = []
                if self.peek()[1] != ")":
                    args.append(self.or_())
                    while self.peek()[1] == ",":
                        self.take()
                        args.append(self.or_())
                self.take(")")
                fn = self.funcs.get(val.lower())
                if fn is None:
                    raise ExprError(f"unsupported function {val}()")
                return fn(*args)
            if val not in self.ctx:
                raise ExprError(f"unsupported context {val!r}")
            v = self.ctx[val]
        else:
            raise ExprError(f"unexpected token {val!r}")
        while self.peek()[1] in (".", "["):
            if self.take()[1] == ".":
                t = self.take()
                if t[1] == "*":
                    v = list(v.values()) if isinstance(v, dict) else (v or [])
                    # `a.*.b` projects over the list
                    if self.peek()[1] == ".":
                        self.take()
                        k = self.take()[1]
                        v = [x.get(k) if isinstance(x, dict) else None for x in v]
                    continue
                v = index(v, t[1])
            else:
                k = self.or_()
                self.take("]")
                v = index(v, k)
        return v


def index(v, k):
    if isinstance(v, dict):
        if k in v:
            return v[k]
        if isinstance(k, str):
            for kk in v:
                if isinstance(kk, str) and kk.lower() == k.lower():
                    return v[kk]
        return None
    if isinstance(v, list) and isinstance(k, (int, float)):
        k = int(k)
        return v[k] if 0 <= k < len(v) else None
    return None


EXPR_RE = re.compile(r"\$\{\{(.*?)\}\}", re.S)


def interpolate(text: str, ev: Evaluator) -> str:
    return EXPR_RE.sub(lambda m: to_str(ev.eval(m.group(1))), text)


def make_funcs(failed, workspace) -> dict:
    """The expression functions; `failed`/`workspace` are thunks on the section."""
    return {
        "always": lambda: True,
        "success": lambda: not failed() and not CANCELLED.is_set(),
        "failure": lambda: failed(),
        "cancelled": lambda: CANCELLED.is_set(),
        "startswith": lambda a, b: to_str(a).lower().startswith(to_str(b).lower()),
        "endswith": lambda a, b: to_str(a).lower().endswith(to_str(b).lower()),
        "contains": lambda a, b: (any(loose_eq(x, b) for x in a) if isinstance(a, list)
                                  else to_str(b).lower() in to_str(a).lower()),
        "format": lambda f, *a: re.sub(r"\{(\d+)\}", lambda m: to_str(a[int(m.group(1))]), f),
        "join": lambda a, sep=",": sep.join(to_str(x) for x in (a or [])),
        "tojson": lambda v: json.dumps(v, indent=2),
        "fromjson": lambda s: json.loads(s),
        "hashfiles": lambda *pats: hash_files(workspace(), pats),
    }


def resolve_input(v, ev):
    if not isinstance(v, str):
        return v
    m = re.fullmatch(r"\s*\$\{\{(.*?)\}\}\s*", v, re.S)
    if m and "${{" not in m.group(1):
        return ev.eval(m.group(1))
    return interpolate(v, ev)


def eval_if(cond, ev: Evaluator, default_status: bool) -> bool:
    """An `if:` with no status function is implicitly `success() && (...)`."""
    if cond is None:
        return ev.funcs["success"]() if default_status else True
    if isinstance(cond, bool):
        return cond and (ev.funcs["success"]() if default_status else True)
    s = str(cond).strip()
    m = re.fullmatch(r"\$\{\{(.*)\}\}", s, re.S)
    if m:
        s = m.group(1)
    has_status = re.search(r"\b(always|success|failure|cancelled)\s*\(", s)
    v = truthy(ev.eval(s))
    if not has_status and default_status:
        v = v and ev.funcs["success"]()
    return v


# --------------------------------------------------------------------------
# section model
# --------------------------------------------------------------------------
def load_yaml(p: Path) -> dict:
    with open(p) as f:
        return yaml.safe_load(f)


SOV_TOKEN = "@SCCACHE_HOST_DIR@"


def sccache_host_dir(sec: dict) -> str:
    """The fleet's sccache mount, spelled ONCE (ci/sections.yml, the shard job's env):
    the machine-specific-path ratchet counts literals, and the vendored copy would
    otherwise add four more."""
    return sec["jobs"]["workspace-test-shard"]["env"]["SCCACHE_HOST_DIR"]


def resolve_vendored(text: str, host_dir: str) -> str:
    """The vendored sovereign-ci.yml with its one declared substitution undone.
    Refuses (SystemExit) unless the body -- everything after the header's first
    blank line -- hashes to the upstream sha256 the header records: a hand edit, a
    second substitution or a stale re-vendor never runs."""
    head, sep, body = text.partition("\n\n")
    m = re.search(r"^# sha256 of the upstream file: ([0-9a-f]{64})$", head, re.M)
    if not sep or not m:
        raise SystemExit(f"fat_driver: {SOV_FILE.relative_to(ROOT)} has no header sha256 line")
    body = body.replace(SOV_TOKEN, host_dir)
    got = hashlib.sha256(body.encode()).hexdigest()
    if got != m.group(1):
        raise SystemExit(f"fat_driver: {SOV_FILE.relative_to(ROOT)} body sha256 {got} != upstream "
                         f"{m.group(1)} -- re-vendor from the pin; do not edit it by hand")
    return body


def section_catalogue() -> dict:
    """name -> {job, source, inputs, matrix, needs_map}. The ONE plan."""
    ci = load_yaml(CI_FILE)
    sec = load_yaml(SECTIONS_FILE)
    sov = yaml.safe_load(resolve_vendored(SOV_FILE.read_text(), sccache_host_dir(sec)))
    wf_env = sov.get("env") or {}
    sov_call = sec["sovereign-ci"]
    sov_inputs = {}
    for k, spec in sov[True]["workflow_call"]["inputs"].items():
        sov_inputs[k] = spec.get("default", "" if spec.get("type") == "string" else False)
    sov_inputs.update(sov_call.get("with") or {})
    cat = {}
    for name, job in sov["jobs"].items():
        needs = job.get("needs") or []
        needs = [needs] if isinstance(needs, str) else needs
        cat[f"sov.{name}"] = dict(
            job=job, source="ci/vendor/sovereign-ci.yml", inputs=sov_inputs,
            wf_env=wf_env, matrix={}, needs={n: f"sov.{n}" for n in needs},
        )
    for name, job in sec["jobs"].items():
        needs = job.get("needs") or []
        needs = [needs] if isinstance(needs, str) else needs
        # `ci` is the old reusable-workflow call; its result is sov.gate's.
        nmap = {n: ("sov.gate" if n == "ci" else n) for n in needs}
        matrix = (job.get("strategy") or {}).get("matrix") or {}
        if matrix:
            for combo in expand_matrix(matrix, sec["matrix-pins"].get(name)):
                label = ",".join(str(v) for v in combo.values())
                cat[f"{name}[{label}]"] = dict(
                    job=job, source="ci/sections.yml", inputs={}, wf_env={},
                    matrix=combo, needs=nmap,
                )
        else:
            cat[name] = dict(job=job, source="ci/sections.yml", inputs={},
                             wf_env={}, matrix={}, needs=nmap)
    del ci
    return cat


def expand_matrix(matrix: dict, pin) -> list:
    """A pin replaces the matrix: shard -> AnyShard, arch -> one value."""
    if pin is None:
        raise SystemExit(f"fat_driver: matrix {matrix} has no matrix-pins entry")
    out = []
    for combo in pin:
        c = {}
        for k, v in combo.items():
            c[k] = AnyShard(str(v["any"])) if isinstance(v, dict) and "any" in v else v
        out.append(c)
    return out


def resolve_section_names(cat: dict, spec: str) -> list:
    names = []
    for part in [p.strip() for p in spec.split(",") if p.strip()]:
        hits = [n for n in cat if n == part or fnmatch.fnmatch(n, part)]
        if not hits:
            raise SystemExit(f"fat_driver: no section matches {part!r}")
        names += [h for h in hits if h not in names]
    return names


# --------------------------------------------------------------------------
# running
# --------------------------------------------------------------------------
CANCELLED = threading.Event()
PROCS: set = set()
PROCS_LOCK = threading.Lock()
PRINT_LOCK = threading.Lock()


def say(msg: str):
    with PRINT_LOCK:
        print(msg, flush=True)


def parse_file_command(path: Path) -> dict:
    out = {}
    if not path.exists():
        return out
    lines = path.read_text(errors="replace").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"^([^=<]+)<<(.+)$", line)
        if m:
            key, delim = m.group(1), m.group(2)
            buf = []
            i += 1
            while i < len(lines) and lines[i] != delim:
                buf.append(lines[i])
                i += 1
            out[key] = "\n".join(buf)
        elif "=" in line:
            k, v = line.split("=", 1)
            out[k] = v
        i += 1
    return out


class Section:
    def __init__(self, name: str, spec: dict, ctx: "RunCtx"):
        self.name = name
        self.spec = spec
        self.job = spec["job"]
        self.ctx = ctx
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
        # The Actions layout, one level down: <S>/_temp is RUNNER_TEMP and
        # <S>/ws is RUNNER_WORKSPACE, so a step that derives the _work root as
        # the common parent of both (sovereign-ci's ownership restore) lands on
        # <S> and can never reach a sibling section or the real _work.
        self.dir = ctx.base / safe
        self.temp = self.dir / "_temp"
        self.log_path = ctx.base / f"{safe}.log"
        self.result = None  # success|failure|skipped|cancelled
        self.outputs = {}
        self.steps_ctx = {}
        self.step_rows = []
        self.failed = False
        self.env = {}
        self.path_prepend = []
        self.container = None
        self.started = self.ended = None
        self.log = None

    # -- expression plumbing -------------------------------------------------
    def evaluator(self, step_env=None) -> Evaluator:
        c = self.ctx
        env = dict(self.env)
        if step_env:
            env.update(step_env)
        contexts = {
            "github": c.github_ctx(self.workspace),
            "env": env,
            "steps": self.steps_ctx,
            "inputs": self.inputs_ctx(),
            "matrix": self.spec["matrix"],
            "needs": {k: {"result": c.need_result(v), "outputs": c.need_outputs(v)}
                      for k, v in self.spec["needs"].items()},
            "runner": {"os": "Linux", "arch": c.runner_arch, "temp": str(self.temp),
                       "tool_cache": os.environ.get("RUNNER_TOOL_CACHE", ""),
                       "name": os.environ.get("RUNNER_NAME", "")},
            "secrets": {"GITHUB_TOKEN": c.token, **c.secrets},
            "vars": c.vars,
            "job": {"status": "failure" if self.failed else "success"},
            "strategy": {},
        }
        funcs = make_funcs(lambda: self.failed, lambda: self.workspace)
        return Evaluator(contexts, funcs)

    def inputs_ctx(self):
        """A reusable workflow's `inputs` are the caller's `with:` values EVALUATED in
        the caller's contexts (github, vars, secrets, matrix, needs) -- not the raw
        text. Run 36157991647 handed sov.test the literal
        `${{ github.event.repository.name }}` as `inputs.repo`; dash rejected it
        ("Bad substitution") and the container mounted a target dir by that name.
        A value that is one whole expression keeps its type (a boolean stays one)."""
        if getattr(self, "_inputs", None) is None:
            c = self.ctx
            caller = Evaluator({
                "github": c.github_ctx(self.workspace), "vars": c.vars,
                "secrets": {"GITHUB_TOKEN": c.token, **c.secrets},
                "matrix": self.spec["matrix"], "inputs": {}, "strategy": {},
                "needs": {k: {"result": c.need_result(v), "outputs": c.need_outputs(v)}
                          for k, v in self.spec["needs"].items()},
            }, make_funcs(lambda: False, lambda: self.workspace))
            self._inputs = {k: resolve_input(v, caller) for k, v in self.spec["inputs"].items()}
        return self._inputs

    @property
    def workspace(self) -> Path:
        return self.dir / "ws" / self.ctx.repo_name

    def interp(self, v, ev):
        if isinstance(v, str):
            return interpolate(v, ev)
        if isinstance(v, dict):
            return {k: self.interp(x, ev) for k, x in v.items()}
        if isinstance(v, list):
            return [self.interp(x, ev) for x in v]
        return v

    # -- lifecycle -----------------------------------------------------------
    def run(self):
        self.started = time.time()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = open(self.log_path, "w", buffering=1)
        try:
            self._run()
        except Exception:  # noqa: BLE001 - a driver bug is a RED section, loudly
            self.failed = True
            self.result = "failure"
            self.log.write("::error::fat_driver internal error\n" + traceback.format_exc())
        finally:
            self.stop_container()
            self.ended = time.time()
            self.log.close()

    def _run(self):
        job = self.job
        # job-level if (implicit success() over needs)
        needs_ok = all(self.ctx.need_result(v) == "success"
                       for v in self.spec["needs"].values())
        ev = self.evaluator()
        cond = job.get("if")
        if cond is None:
            run_it = needs_ok
        else:
            s = str(cond)
            status = re.search(r"\b(always|success|failure|cancelled)\s*\(", s)
            ev.funcs["success"] = lambda: needs_ok and not CANCELLED.is_set()
            ev.funcs["failure"] = lambda: any(
                self.ctx.need_result(v) == "failure" for v in self.spec["needs"].values())
            run_it = eval_if(cond, ev, default_status=False) and (needs_ok or bool(status))
        if not run_it:
            self.result = "skipped"
            self.log.write(f"section skipped by job if: {cond!r} (needs_ok={needs_ok})\n")
            return
        self.setup_workspace()
        base_env = {}
        for k, v in (self.spec["wf_env"] or {}).items():
            base_env[k] = to_str(interpolate(to_str(v), self.evaluator()))
        self.env = dict(base_env)
        for k, v in (job.get("env") or {}).items():
            self.env[k] = to_str(interpolate(to_str(v), self.evaluator()))
        if "container" in job:
            self.start_container(job["container"])
        timeout = timeout_seconds(job.get("timeout-minutes"), 360)
        deadline = self.started + timeout
        for idx, step in enumerate(job.get("steps") or []):
            self.run_step(idx, step, deadline)
        # outputs
        ev = self.evaluator()
        for k, v in (job.get("outputs") or {}).items():
            self.outputs[k] = to_str(interpolate(to_str(v), ev))
        if CANCELLED.is_set():
            self.result = "cancelled"
        else:
            self.result = "failure" if self.failed else "success"

    def setup_workspace(self):
        if self.dir.exists():
            shutil.rmtree(self.dir, ignore_errors=True)
        self.temp.mkdir(parents=True)
        (self.dir / "home").mkdir()
        self.workspace.parent.mkdir(parents=True)
        # A local clone HARDLINKS the checkout's object files: self-contained, and
        # nearly free on one filesystem. Not --shared: its alternates point at the
        # runner checkout, which a section's own `docker run -v $GITHUB_WORKSPACE:...`
        # cannot see -- run 36157991647's shard died in `cargo publish --dry-run` on
        # "An object with id ... could not be found". Refs are this clone's own, so
        # concurrent `git fetch` in two sections never contend on a ref lock.
        src = self.ctx.checkout
        subprocess.run(["git", "clone", "-q", "--local", "--no-checkout", str(src),
                        str(self.workspace)], check=True)
        g = ["git", "-C", str(self.workspace)]
        subprocess.run(g + ["remote", "set-url", "origin", self.ctx.origin_url], check=True)
        hdr = subprocess.run(["git", "-C", str(src), "config", "--get-regexp",
                              r"^http\..*\.extraheader$"], capture_output=True, text=True)
        for line in hdr.stdout.splitlines():
            k, _, v = line.partition(" ")
            subprocess.run(g + ["config", k, v], check=True)
        subprocess.run(g + ["checkout", "-q", "--detach", self.ctx.head], check=True)

    def start_container(self, spec):
        if isinstance(spec, str):
            spec = {"image": spec}
        ev = self.evaluator()
        spec = self.interp(spec, ev)
        name = f"fat-{self.ctx.run_id}-{re.sub(r'[^a-z0-9]+', '-', self.name.lower())}"
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        cmd = ["docker", "run", "-d", "--name", name, "--entrypoint", "tail",
               "-v", f"{self.dir}:{self.dir}", "-w", str(self.workspace),
               "-e", f"HOME={self.dir / 'home'}"]
        evp = os.environ.get("GITHUB_EVENT_PATH")
        if evp and os.path.exists(evp):
            cmd += ["-v", f"{evp}:{evp}:ro"]
        for vol in spec.get("volumes") or []:
            cmd += ["-v", vol]
        opts = spec.get("options") or ""
        cmd += opts.split() if opts else []
        for k, v in (spec.get("env") or {}).items():
            cmd += ["-e", f"{k}={to_str(v)}"]
        cmd += [spec["image"], "-f", "/dev/null"]
        self.log.write(f"container: {' '.join(cmd)}\n")
        subprocess.run(cmd, check=True, stdout=self.log, stderr=subprocess.STDOUT)
        self.container = name

    def stop_container(self):
        if self.container:
            # The container ran as root: hand the section tree back to the runner
            # user before the container goes, or the runner's own _temp cleanup
            # meets root-owned files (the EACCES the sov restore steps exist for).
            subprocess.run(["docker", "exec", self.container, "chown", "-R",
                            f"{os.getuid()}:{os.getgid()}", str(self.dir)], capture_output=True)
            subprocess.run(["docker", "rm", "-f", self.container], capture_output=True)
            self.container = None

    # -- steps ---------------------------------------------------------------
    def run_step(self, idx: int, step: dict, deadline: float):
        sid = step.get("id") or f"__step{idx}"
        label = step.get("name") or step.get("uses") or (step.get("run") or "").split("\n")[0]
        ev = self.evaluator(step.get("env"))
        row = {"index": idx, "name": label, "id": step.get("id")}
        try:
            run_it = eval_if(step.get("if"), ev, default_status=True)
        except ExprError as e:
            self.log.write(f"::error::step {label!r}: if: {e}\n")
            run_it, self.failed = False, True
        if not run_it:
            row.update(outcome="skipped", conclusion="skipped", seconds=0)
            self.steps_ctx[sid] = {"outputs": {}, "outcome": "skipped", "conclusion": "skipped"}
            self.step_rows.append(row)
            return
        t0 = time.time()
        self.log.write(f"\n##[step {idx}] {label}\n")
        outputs = {}
        try:
            if "uses" in step:
                ok, outputs = self.run_uses(step, ev)
            else:
                ok, outputs = self.run_script(step, ev, deadline)
        except ExprError as e:
            self.log.write(f"::error::step {label!r}: {e}\n")
            ok = False
        outcome = "success" if ok else "failure"
        coe = step.get("continue-on-error")
        if isinstance(coe, str):
            coe = truthy(ev.eval(EXPR_RE.sub(lambda m: m.group(1), coe)))
        conclusion = "success" if (ok or coe) else "failure"
        if conclusion == "failure":
            self.failed = True
        self.steps_ctx[sid] = {"outputs": outputs, "outcome": outcome, "conclusion": conclusion}
        row.update(outcome=outcome, conclusion=conclusion, seconds=round(time.time() - t0, 1))
        self.step_rows.append(row)
        self.log.write(f"##[step {idx}] {outcome} ({row['seconds']}s)\n")

    def step_env(self, step, ev) -> dict:
        # The driver's own inputs (FAT_*, and the token it uses for the API) are not
        # part of any section's environment: a real job step sees GITHUB_TOKEN only
        # where the job or step env names it, which env.update() below restores.
        env = {k: v for k, v in os.environ.items()
               if not k.startswith("FAT_") and k != "GITHUB_TOKEN"}
        env.update(self.env)
        for k, v in (step.get("env") or {}).items():
            env[k] = to_str(interpolate(to_str(v), ev))
        env["GITHUB_WORKSPACE"] = str(self.workspace)
        env["RUNNER_WORKSPACE"] = str(self.workspace.parent)
        env["RUNNER_TEMP"] = str(self.temp)
        env["HOME"] = str(self.dir / "home") if self.container else env.get("HOME", "")
        env["GITHUB_JOB"] = self.name
        env["CI"] = "true"
        env["GITHUB_ACTIONS"] = "true"
        # The section's own log, for a step that audits an earlier step's output
        # (the Σ-executed step reads the compute and integration steps from it).
        env["FAT_SECTION_LOG"] = str(self.log_path)
        if self.path_prepend:
            env["PATH"] = ":".join(reversed(self.path_prepend)) + ":" + env.get("PATH", "")
        return env

    def run_script(self, step, ev, deadline):
        script = interpolate(step["run"], ev)
        env = self.step_env(step, ev)
        files = {}
        for k in ("GITHUB_OUTPUT", "GITHUB_ENV", "GITHUB_PATH", "GITHUB_STEP_SUMMARY", "GITHUB_STATE"):
            p = self.temp / f"_{k.lower()}_{len(self.step_rows)}"
            p.write_text("")
            env[k] = str(p)
            files[k] = p
        spath = self.temp / f"_step_{len(self.step_rows)}.sh"
        spath.write_text(script)
        shell = step.get("shell")
        if shell in (None, "") and self.container:
            argv = ["sh", "-e", str(spath)]  # the Actions default inside a job container
        elif shell in (None, ""):
            argv = ["bash", "-e", str(spath)]
        elif shell == "bash":
            argv = ["bash", "--noprofile", "--norc", "-eo", "pipefail", str(spath)]
        elif shell == "sh":
            argv = ["sh", "-e", str(spath)]
        elif shell == "python":
            argv = ["python3", str(spath)]
        elif "{0}" in shell:
            argv = shell.replace("{0}", str(spath)).split()
        else:
            raise ExprError(f"unsupported shell {shell!r}")
        wd = step.get("working-directory")
        cwd = str((self.workspace / interpolate(wd, ev)) if wd else self.workspace)
        tmo = timeout_seconds(step.get("timeout-minutes"), 0) or None
        remain = deadline - time.time()
        tmo = min(t for t in (tmo, remain) if t is not None)
        if self.container:
            execv = ["docker", "exec", "-i", "-w", cwd]
            for k, v in env.items():
                # Actions hands a container step the job/step env and the runner's
                # GITHUB_*/RUNNER_* set -- never the host's PATH, HOME or shell env.
                keep = (k in self.env or k in (step.get("env") or {})
                        or k.startswith(("GITHUB_", "RUNNER_", "ACTIONS_")) or k in ("CI", "HOME"))
                if keep and k != "PATH":
                    execv += ["-e", f"{k}={v}"]
            argv = execv + [self.container] + argv
            cwd = None
        rc = self.spawn(argv, env if not self.container else None, cwd, tmo)
        # file commands
        outputs = parse_file_command(files["GITHUB_OUTPUT"])
        for k, v in parse_file_command(files["GITHUB_ENV"]).items():
            self.env[k] = v
        for line in files["GITHUB_PATH"].read_text().splitlines():
            if line.strip():
                self.path_prepend.append(line.strip())
        summ = files["GITHUB_STEP_SUMMARY"].read_text()
        if summ.strip():
            with open(self.ctx.base / "step-summary.md", "a") as f:
                f.write(f"\n### {self.name}\n{summ}\n")
        return rc == 0, outputs

    def spawn(self, argv, env, cwd, timeout) -> int:
        if CANCELLED.is_set() and timeout and timeout > 120:
            timeout = 120  # always()/cancelled() cleanup steps only, bounded
        p = subprocess.Popen(argv, env=env, cwd=cwd, stdout=self.log, stderr=subprocess.STDOUT,
                             start_new_session=True)
        with PROCS_LOCK:
            PROCS.add(p)
        try:
            return p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.log.write(f"::error::step exceeded its timeout ({timeout:.0f}s); killing group {p.pid}\n")
            kill_group(p)
            return 124
        finally:
            with PROCS_LOCK:
                PROCS.discard(p)

    # -- uses: ----------------------------------------------------------------
    def run_uses(self, step, ev):
        uses = step["uses"]
        action = uses.split("@")[0]
        w = self.interp(step.get("with") or {}, ev)
        h = USES.get(action)
        if h is None:
            raise ExprError(f"unsupported action {uses}")
        return h(self, w, ev)


def kill_group(p):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(p.pid, sig)
        except ProcessLookupError:
            return
        try:
            p.wait(timeout=15)
            return
        except subprocess.TimeoutExpired:
            continue


def hash_files(ws: Path, pats) -> str:
    h = hashlib.sha256()
    files = set()
    for pat in pats:
        files.update(glob.glob(str(ws / pat), recursive=True))
    for f in sorted(files):
        if os.path.isfile(f):
            h.update(hashlib.sha256(Path(f).read_bytes()).digest())
    return h.hexdigest() if files else ""


# uses: handlers --------------------------------------------------------------
def uses_checkout(sec: Section, w, ev):
    ref = w.get("ref")
    if ref:
        g = ["git", "-C", str(sec.workspace)]
        rc = sec.spawn(g + ["fetch", "-q", "origin", f"+refs/heads/{ref}:refs/remotes/origin/{ref}"],
                       None, None, 600)
        if rc != 0:
            return False, {}
        rc = sec.spawn(g + ["checkout", "-q", "-B", ref, f"origin/{ref}"], None, None, 120)
        return rc == 0, {}
    # The fat job checks out with full history so every section can have what its
    # own checkout asked for; a section that asked for less must not see more. A
    # depth-1 checkout has no parents and no tags, and guards depend on that: the
    # guard-tree ratchets fall back to the origin/main TIP when the merge-base is
    # unreachable. `.git/shallow` naming HEAD is exactly git's depth-1 state.
    depth = int(to_str(w.get("fetch-depth", 1)) or 1)
    g = ["git", "-C", str(sec.workspace)]
    if depth == 1:
        tags = subprocess.run(g + ["tag", "-l"], capture_output=True, text=True, check=True).stdout.split()
        if tags:
            subprocess.run(g + ["tag", "-d", *tags], capture_output=True, check=True)
        gitdir = subprocess.run(g + ["rev-parse", "--absolute-git-dir"], capture_output=True,
                                text=True, check=True).stdout.strip()
        Path(gitdir, "shallow").write_text(sec.ctx.head + "\n")
    elif depth != 0:
        sec.log.write(f"::error::checkout: fetch-depth {depth} is not emulated (only 0 and 1)\n")
        return False, {}
    sec.log.write(f"checkout: section clone at {sec.ctx.head} (fetch-depth {depth})\n")
    return True, {}


def uses_cache(sec: Section, w, ev):
    sec.log.write(f"cache: not restored inside a fat job (key {w.get('key')!r}); "
                  "the host-persistent target/registry dirs carry the warmth\n")
    return True, {"cache-hit": "false"}


def upload_plan(ws: Path, path_spec: str) -> tuple:
    """(root, [files]) the way actions/upload-artifact lays out an artifact.

    Its root is the least common ancestor of the search paths: a directory
    pattern contributes itself, a file its parent. So `target/x/a.json` and
    `target/x/b.json` land flat at the artifact root, which is what the
    determinism compare's `merge-multiple` download relies on.
    """
    roots, files = [], []
    for pat in str(path_spec).splitlines():
        pat = pat.strip()
        if not pat or pat.startswith("!"):
            continue
        full = pat if os.path.isabs(pat) else str(ws / pat)
        for hit in sorted(glob.glob(full, recursive=True)):
            hp = Path(hit)
            if hp.is_dir():
                roots.append(hp)
                files += [f for f in sorted(hp.rglob("*")) if f.is_file()]
            elif hp.is_file():
                roots.append(hp.parent)
                files.append(hp)
    if not roots:
        return None, []
    root = Path(os.path.commonpath([str(r) for r in roots]))
    return root, sorted(set(files))


def uses_upload(sec: Section, w, ev):
    name = w["name"]
    dest = sec.ctx.stage / name
    root, files = upload_plan(sec.workspace, w.get("path", ""))
    for f in files:
        out = dest / f.relative_to(root)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, out)
    sec.log.write(f"upload-artifact {name!r}: staged {len(files)} file(s) from {root} at {dest}\n")
    with open(sec.ctx.base / "artifacts.jsonl", "a") as fh:
        fh.write(json.dumps({"section": sec.name, "name": name, "count": len(files),
                             "if_no_files_found": w.get("if-no-files-found", "warn")}) + "\n")
    if not files and w.get("if-no-files-found") == "error":
        sec.log.write(f"::error::upload-artifact {name!r}: no files found\n")
        return False, {}
    return True, {}


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def fetch_artifact_zip(url: str, token: str, out: Path, timeout: int = 300):
    """The artifact API answers 302 to a pre-signed blob URL. urllib's own redirect
    carries the Authorization header along, and the blob store rejects a request that
    has one (401, measured on run 36157991647) -- so follow the hop ourselves, bare."""
    opener = urllib.request.build_opener(_NoRedirect)
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        r = opener.open(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        loc = e.headers.get("Location") if e.code in (301, 302, 303, 307, 308) else None
        if not loc:
            raise
        r = urllib.request.urlopen(urllib.request.Request(loc), timeout=timeout)
    with r, open(out, "wb") as f:
        shutil.copyfileobj(r, f)


def uses_download(sec: Section, w, ev):
    """Poll this run's artifacts (the X64 half is uploaded by another job)."""
    pattern = w.get("pattern") or w.get("name")
    dest = sec.workspace / (w.get("path") or ".")
    dest.mkdir(parents=True, exist_ok=True)
    want = [p.strip() for p in os.environ.get("FAT_EXPECT_ARTIFACTS", "").split(",") if p.strip()]
    deadline = time.time() + float(os.environ.get("FAT_ARTIFACT_WAIT_S", "3600"))
    api = f"{sec.ctx.api}/repos/{sec.ctx.repo}/actions/runs/{sec.ctx.run_id}/artifacts?per_page=100"
    while True:
        arts = gh_json(api, sec.ctx.token).get("artifacts", [])
        hits = [a for a in arts if fnmatch.fnmatch(a["name"], pattern)]
        local = sorted(p.name for p in sec.ctx.stage.glob(pattern)) if sec.ctx.stage.exists() else []
        names = {a["name"] for a in hits} | set(local)
        if not want or all(x in names for x in want):
            break
        if CANCELLED.is_set() or time.time() > deadline:
            sec.log.write(f"::error::download-artifact: waited for {want}, have {sorted(names)}\n")
            return False, {}
        time.sleep(20)
    for a in hits:
        z = sec.temp / f"{a['name']}.zip"
        fetch_artifact_zip(a["archive_download_url"], sec.ctx.token, z)
        out = dest if w.get("merge-multiple") else dest / a["name"]
        with zipfile.ZipFile(z) as zf:
            zf.extractall(out)
    for name in local:
        if name in {a["name"] for a in hits}:
            continue
        out = dest if w.get("merge-multiple") else dest / name
        shutil.copytree(sec.ctx.stage / name, out, dirs_exist_ok=True)
    sec.log.write(f"download-artifact {pattern!r}: {sorted(names)} -> {dest}\n")
    return True, {}


def uses_toolchain(sec: Section, w, ev):
    tc = w.get("toolchain", "stable")
    rc = sec.spawn(["rustup", "toolchain", "install", tc, "--profile", "minimal", "--no-self-update"],
                   sec.step_env({}, ev), str(sec.workspace), 900)
    # `rustup default` in the action is runner-global; a section pins itself only.
    sec.env["RUSTUP_TOOLCHAIN"] = tc
    return rc == 0, {}


def uses_deferred(kind):
    """Actions that need the runner's action runtime: staged, run as real steps
    of the fat job after the driver (their inputs are written for that step)."""
    def h(sec: Section, w, ev):
        with open(sec.ctx.base / "deferred.jsonl", "a") as f:
            f.write(json.dumps({"section": sec.name, "action": kind, "with": w,
                                "workspace": str(sec.workspace)}) + "\n")
        sec.log.write(f"{kind}: deferred to the fat job's own step (needs the action runtime)\n")
        return True, {}
    return h


USES = {
    "actions/checkout": uses_checkout,
    "actions/cache": uses_cache,
    "actions/upload-artifact": uses_upload,
    "actions/download-artifact": uses_download,
    "dtolnay/rust-toolchain": uses_toolchain,
    "codecov/codecov-action": uses_deferred("codecov"),
    "actions/attest-build-provenance": uses_deferred("attest"),
}


def gh_json(url: str, token: str) -> dict:
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}",
                                               "Accept": "application/vnd.github+json"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception:  # noqa: BLE001 - transient API errors are retried, then raised
            if attempt == 4:
                raise
            time.sleep(5 * (attempt + 1))
    return {}


# --------------------------------------------------------------------------
class RunCtx:
    def __init__(self, base: Path):
        self.base = base
        self.stage = base / "artifacts"
        self.checkout = Path(os.environ.get("GITHUB_WORKSPACE", os.getcwd()))
        self.repo = os.environ.get("GITHUB_REPOSITORY", "paiml/aprender")
        self.repo_name = self.repo.split("/")[-1]
        self.run_id = os.environ.get("GITHUB_RUN_ID", "local")
        self.token = os.environ.get("GITHUB_TOKEN", "")
        self.api = os.environ.get("GITHUB_API_URL", "https://api.github.com")
        server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
        self.origin_url = f"{server}/{self.repo}"
        self.head = subprocess.run(["git", "-C", str(self.checkout), "rev-parse", "HEAD"],
                                   capture_output=True, text=True, check=True).stdout.strip()
        self.runner_arch = os.environ.get("RUNNER_ARCH", "X64")
        self.vars = json.loads(os.environ.get("FAT_VARS_JSON") or "{}")
        # FAT_SECRET_<NAME> -> secrets.<NAME>: the fat job passes each secret a
        # section references explicitly; nothing else is visible.
        self.secrets = {k[len("FAT_SECRET_"):]: v for k, v in os.environ.items()
                        if k.startswith("FAT_SECRET_")}
        ev_path = os.environ.get("GITHUB_EVENT_PATH")
        self.event = json.load(open(ev_path)) if ev_path and os.path.exists(ev_path) else {}
        self.sections: dict = {}

    def members(self, need: str) -> list:
        """A need names a job; a matrix job is every expansion present HERE.

        determinism-compare needs `determinism`, whose X64 half runs in the
        x86-main job: only the ARM64 half is in this run, and the X64 half
        reaches the compare as an artifact (a missing one fails the download).
        """
        if need in self.sections:
            return [need]
        return [n for n in self.sections if n.startswith(need + "[")]

    def need_result(self, need: str):
        rs = [self.sections[n].result for n in self.members(need)]
        if not rs or any(r is None for r in rs):
            return None
        for r in ("failure", "cancelled"):
            if r in rs:
                return r
        return "success" if "success" in rs else "skipped"

    def need_outputs(self, need: str) -> dict:
        out = {}
        for n in self.members(need):
            out.update(self.sections[n].outputs)
        return out

    def github_ctx(self, ws: Path) -> dict:
        e = os.environ
        return {
            "event": self.event, "event_name": e.get("GITHUB_EVENT_NAME", ""),
            "sha": e.get("GITHUB_SHA", self.head), "ref": e.get("GITHUB_REF", ""),
            "ref_name": e.get("GITHUB_REF_NAME", ""), "head_ref": e.get("GITHUB_HEAD_REF", ""),
            "base_ref": e.get("GITHUB_BASE_REF", ""), "run_id": self.run_id,
            "run_number": e.get("GITHUB_RUN_NUMBER", ""), "run_attempt": e.get("GITHUB_RUN_ATTEMPT", ""),
            "repository": self.repo, "repository_owner": self.repo.split("/")[0],
            "workspace": str(ws), "token": self.token, "actor": e.get("GITHUB_ACTOR", ""),
            "server_url": e.get("GITHUB_SERVER_URL", "https://github.com"),
            "api_url": self.api, "workflow": e.get("GITHUB_WORKFLOW", ""),
            "job": e.get("GITHUB_JOB", ""),
        }


DEFAULT_TIMEOUT_SCALE = 2.0


def timeout_seconds(minutes, default_minutes=0) -> float:
    """A job/step `timeout-minutes`, in seconds, scaled for a shared runner.

    Each value in ci/sections.yml was sized for a DEDICATED runner. Here every
    section shares one box with the nextest shards, so the same work takes
    longer: guard-tree finished in 1501 s of its 1800 s on 45aa0f4d5 and was
    killed at 1800 s on 1e760c005, same intel host, same body, with its
    cargo-free step at ~20 min on main's own runner (#4433). The budget is a
    hang detector, not a speed gate; the fat job's own `timeout-minutes` still
    bounds the whole run. FAT_TIMEOUT_SCALE overrides; below 1 is refused, as
    that would kill a section sooner than its dedicated job would have been.
    """
    raw = os.environ.get("FAT_TIMEOUT_SCALE", "")
    scale = float(raw) if raw.strip() else DEFAULT_TIMEOUT_SCALE
    if scale < 1:
        raise SystemExit(f"fat_driver: FAT_TIMEOUT_SCALE={raw!r} is below 1")
    return float(minutes if minutes is not None else default_minutes) * 60 * scale


def schedule(ctx: RunCtx, names: list, early: str | None, early_done: threading.Event):
    pending = list(names)
    running: dict = {}
    while pending or running:
        for n in list(pending):
            deps = ctx.sections[n].spec["needs"].values()
            missing = [d for d in deps if not ctx.members(d)]
            if missing:
                raise SystemExit(f"fat_driver: section {n} needs {missing}, not in this run")
            if all(ctx.need_result(d) is not None for d in deps):
                pending.remove(n)
                t = threading.Thread(target=ctx.sections[n].run, name=n, daemon=True)
                t.start()
                running[n] = t
                say(f"::notice::section {n} started")
        for n, t in list(running.items()):
            if not t.is_alive():
                del running[n]
                s = ctx.sections[n]
                say(f"section {n}: {s.result} in {s.ended - s.started:.0f}s")
                if n == early:
                    early_done.set()
        time.sleep(1)
    early_done.set()


def write_results(ctx: RunCtx, path: Path):
    res = {}
    for n, s in ctx.sections.items():
        job = s.job
        res[n] = {
            "result": s.result, "source": s.spec["source"],
            "continue_on_error": bool(job.get("continue-on-error")),
            "seconds": round((s.ended or time.time()) - (s.started or time.time()), 1),
            "outputs": s.outputs, "steps": s.step_rows,
        }
    path.write_text(json.dumps(res, indent=1))
    return res


def print_logs(ctx: RunCtx):
    for n, s in ctx.sections.items():
        say(f"::group::{n} — {s.result}")
        try:
            with open(s.log_path, errors="replace") as f:
                for line in f:
                    sys.stdout.write(line)
        except FileNotFoundError:
            pass
        sys.stdout.flush()
        say("::endgroup::")


def verdict(res: dict) -> int:
    bad = [n for n, r in res.items()
           if r["result"] not in ("success", "skipped") and not r["continue_on_error"]]
    for n in bad:
        say(f"::error::section {n}: {res[n]['result']}")
    return 1 if bad else 0


def emit_results_output(res: dict) -> None:
    """`results={section: result}` on the fat job's step output, for the gate job."""
    out = os.environ.get("GITHUB_OUTPUT")
    if not out:
        return
    compact = {n: {"result": r["result"], "continue_on_error": r.get("continue_on_error", False)}
               for n, r in res.items()}
    # A section's job-level outputs ride along when it has any: gate reads
    # `changes.outputs.docs_only` for the #3668 docs-only rule.
    for n, r in res.items():
        if r.get("outputs"):
            compact[n]["outputs"] = r["outputs"]
    # The actions a section staged for the fat job's own steps (codecov, attest).
    kinds = set()
    dpath = Path(os.environ.get("RUNNER_TEMP", "/tmp")) / "fat" / "deferred.jsonl"
    if dpath.exists():
        kinds = {json.loads(l)["action"] for l in dpath.read_text().splitlines() if l.strip()}
    with open(out, "a") as f:
        f.write(f"results={json.dumps(compact, separators=(',', ':'))}\n")
        f.write(f"deferred={','.join(sorted(kinds))}\n")


def on_signal(signum, frame):
    say(f"fat_driver: signal {signum}; cancelling every section")
    CANCELLED.set()
    with PROCS_LOCK:
        procs = list(PROCS)
    for p in procs:
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def cmd_run(a):
    cat = section_catalogue()
    names = resolve_section_names(cat, a.sections)
    base = Path(a.base or Path(os.environ.get("RUNNER_TEMP", "/tmp")) / "fat")
    base.mkdir(parents=True, exist_ok=True)
    ctx = RunCtx(base)
    for n in names:
        ctx.sections[n] = Section(n, cat[n], ctx)
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    results = Path(a.results)
    early_done = threading.Event()
    if a.background_until:
        # Fork: the child runs every section; the parent returns once the named
        # section is done, so the job's next step (an upload) runs mid-flight.
        marker = base / "early.json"
        pid = os.fork()
        if pid:
            (base / "driver.pid").write_text(str(pid))
            while not marker.exists():
                try:
                    wpid, _ = os.waitpid(pid, os.WNOHANG)
                except ChildProcessError:
                    wpid = pid
                if wpid == pid:
                    break
                time.sleep(2)
            say(f"fat_driver: {a.background_until} finished; the rest continue in pid {pid}")
            if marker.exists():
                r = json.loads(marker.read_text())
                return 0 if r.get("result") == "success" else 1
            return 1
        os.setsid()
        sys.stdout = open(base / "driver.out", "w", buffering=1)
        sys.stderr = sys.stdout
    t = threading.Thread(target=schedule, args=(ctx, names, a.background_until, early_done), daemon=True)
    t.start()
    if a.background_until:
        early_done.wait()
        s = ctx.sections.get(a.background_until)
        (base / "early.json").write_text(json.dumps({"result": s.result if s else None}))
    t.join()
    res = write_results(ctx, results)
    if not a.background_until:
        print_logs(ctx)
        emit_results_output(res)
    rc = verdict(res)
    (base / "driver.rc").write_text(str(rc))
    return rc


def cmd_wait(a):
    base = Path(a.base or Path(os.environ.get("RUNNER_TEMP", "/tmp")) / "fat")
    pid = int((base / "driver.pid").read_text())
    signal.signal(signal.SIGTERM, lambda *_: os.kill(pid, signal.SIGTERM))
    signal.signal(signal.SIGINT, lambda *_: os.kill(pid, signal.SIGTERM))
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(5)
    sys.stdout.write((base / "driver.out").read_text(errors="replace"))
    res = json.loads(Path(a.results).read_text())
    emit_results_output(res)
    for n in res:
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", n)
        say(f"::group::{n} — {res[n]['result']}")
        p = base / f"{safe}.log"
        if p.exists():
            sys.stdout.write(p.read_text(errors="replace"))
        say("::endgroup::")
    rc_file = base / "driver.rc"
    return int(rc_file.read_text()) if rc_file.exists() else 1


def cmd_list(a):
    cat = section_catalogue()
    for n, spec in cat.items():
        for i, st in enumerate(spec["job"].get("steps") or []):
            label = st.get("name") or st.get("uses") or (st.get("run") or "").split("\n")[0]
            print(f"{n}\t{i}\t{label}")


def cmd_self_test(a):
    """Case table: each row names what it would read if the rule it guards were
    deleted (AnyShard, implicit success(), the depth-1 cut, the verdict rule)."""
    import tempfile
    from types import SimpleNamespace
    rows = []

    def row(label, got, want):
        rows.append((label, got == want, got, want))

    def ev(failed=False, **ctx):
        base = {"github": {"event_name": "pull_request"}, "matrix": {"shard": AnyShard("1"), "shards": 1},
                "steps": {"tier": {"outputs": {"tier": "full"}}}, "env": {}}
        base.update(ctx)
        return Evaluator(base, make_funcs(lambda: failed, lambda: Path(".")))

    real = SOV_FILE.read_text()
    hd = sccache_host_dir(load_yaml(SECTIONS_FILE))

    def vend_rc(text):
        try:
            resolve_vendored(text, hd)
            return 0
        except SystemExit:
            return 1
    row("vendored sov: token restored, body == upstream sha256", vend_rc(real), 0)
    row("vendored sov: resolved volumes name the one sccache dir",
        resolve_vendored(real, hd).count(f"- {hd}:/sccache"), 4)
    row("vendored sov: MUTANT hand edit of the body is refused",
        vend_rc(real.replace("runs-on:", "runs-on: ", 1)), 1)
    row("vendored sov: MUTANT unresolved token (wrong host dir) is refused",
        vend_rc(real.replace(f"- {SOV_TOKEN}:", "- /elsewhere:", 1)), 1)
    row("vendored sov: MUTANT header without sha256 is refused",
        vend_rc(real.replace("# sha256 of the upstream file:", "# sha:", 1)), 1)

    e = ev()
    row("event == 'pull_request'", truthy(e.eval("github.event_name == 'pull_request'")), True)
    row("string == is case-insensitive", truthy(e.eval("github.event_name == 'PULL_REQUEST'")), True)
    row("AnyShard: shard == 1", truthy(e.eval("steps.tier.outputs.tier == 'full' && matrix.shard == 1")), True)
    row("AnyShard: shard == 3 too (once-only steps of every shard run)", truthy(e.eval("matrix.shard == 3")), True)
    row("tier quick turns the full-tier step off",
        truthy(ev(steps={"tier": {"outputs": {"tier": "quick"}}}).eval("steps.tier.outputs.tier == 'full'")), False)
    row("contains(list)", truthy(e.eval("contains(fromJSON('[\"a\",\"b\"]'), 'b')")), True)
    row("startsWith is case-insensitive", truthy(e.eval("startsWith('Refs/Tags/v1', 'refs/tags/')")), True)
    row("format", to_str(e.eval("format('{0}-{1}', 'a', 2)")), "a-2")
    row("|| yields a value", to_str(e.eval("github.missing || 'fallback'")), "fallback")
    row("!cancelled()", truthy(e.eval("!cancelled()")), True)
    row("implicit success(): a failed section skips a plain if",
        eval_if("github.event_name == 'pull_request'", ev(failed=True), True), False)
    row("always() overrides the failure", eval_if("always() && github.event_name == 'pull_request'",
                                                   ev(failed=True), True), True)
    row("no if: runs only on success", eval_if(None, ev(failed=True), True), False)
    row("with: whole expression evaluates in the caller",
        resolve_input("${{ github.event_name }}", e), "pull_request")
    row("with: whole expression keeps its type", resolve_input("${{ 1 == 1 }}", e), True)
    row("with: embedded expression interpolates", resolve_input("r-${{ github.event_name }}", e), "r-pull_request")
    row("with: plain value unchanged", resolve_input(False, e), False)
    row("verdict: failure + continue-on-error passes",
        verdict({"a": {"result": "failure", "continue_on_error": True}}), 0)
    row("verdict: failure fails", verdict({"a": {"result": "failure", "continue_on_error": False}}), 1)
    row("verdict: skipped passes", verdict({"a": {"result": "skipped", "continue_on_error": False}}), 0)
    row("verdict: cancelled fails", verdict({"a": {"result": "cancelled", "continue_on_error": False}}), 1)
    cat = {"determinism[X64]": 0, "determinism[ARM64]": 0, "determinism-compare": 0, "sov.test": 0, "sov.gate": 0}
    row("a bracketed name resolves to itself only", resolve_section_names(cat, "determinism[X64]"),
        ["determinism[X64]"])
    row("sov.* globs", resolve_section_names(cat, "sov.*"), ["sov.test", "sov.gate"])
    saved = os.environ.pop("FAT_TIMEOUT_SCALE", None)
    try:
        row("timeout: a section's 30 min is scaled x2 on the shared runner", timeout_seconds(30), 3600.0)
        row("timeout: an absent job timeout keeps Actions' 360 min, scaled", timeout_seconds(None, 360), 43200.0)
        row("timeout: an absent step timeout stays unbounded", timeout_seconds(None, 0) or None, None)
        os.environ["FAT_TIMEOUT_SCALE"] = "1"
        row("timeout: FAT_TIMEOUT_SCALE=1 is the dedicated-runner budget", timeout_seconds(30), 1800.0)
        os.environ["FAT_TIMEOUT_SCALE"] = "0.5"
        try:
            timeout_seconds(30)
            row("timeout: MUTANT scale below 1 is refused", "accepted", "refused")
        except SystemExit:
            row("timeout: MUTANT scale below 1 is refused", "refused", "refused")
    finally:
        os.environ.pop("FAT_TIMEOUT_SCALE", None)
        if saved is not None:
            os.environ["FAT_TIMEOUT_SCALE"] = saved
    import http.server
    import threading

    class Blob(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/api"):
                self.send_response(302)
                self.send_header("Location", f"http://127.0.0.1:{self.server.server_port}/blob")
                self.end_headers()
            elif self.headers.get("Authorization"):
                self.send_response(401)
                self.end_headers()
            else:
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"zipbytes")

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), Blob)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    with tempfile.TemporaryDirectory() as td:
        z = Path(td) / "a.zip"
        try:
            fetch_artifact_zip(f"http://127.0.0.1:{srv.server_port}/api/x/zip", "tok", z)
            got = z.read_bytes()
        except Exception as e:  # noqa: BLE001 -- the row reports it
            got = repr(e)
        row("artifact download drops Authorization on the blob redirect", got, b"zipbytes")
    srv.shutdown()
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "out"
        f.write_text("a=1\nb<<EOF\nx\ny=z\nEOF\nc=3\n")
        row("file command heredoc", parse_file_command(f), {"a": "1", "b": "x\ny=z", "c": "3"})
        src = Path(td) / "src"
        g = ["git", "-C", str(src)]
        subprocess.run(["git", "init", "-q", str(src)], check=True)
        for i in range(3):
            (src / "f").write_text(str(i))
            subprocess.run(g + ["add", "f"], check=True)
            subprocess.run(g + ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", str(i)], check=True)
        subprocess.run(g + ["tag", "v0"], check=True)
        head = subprocess.run(g + ["rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()

        def clone(name):
            ws = Path(td) / name
            subprocess.run(["git", "clone", "-q", "--local", str(src), str(ws)], check=True)
            return SimpleNamespace(workspace=ws, ctx=SimpleNamespace(head=head),
                                   log=open(os.devnull, "w"))

        sec = clone("d1")
        ok, _ = uses_checkout(sec, {}, None)
        cnt = subprocess.run(["git", "-C", str(sec.workspace), "rev-list", "--count", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
        tags = subprocess.run(["git", "-C", str(sec.workspace), "tag", "-l"], capture_output=True,
                              text=True).stdout.split()
        row("checkout default depth 1: one commit, no tags", (ok, cnt, tags), (True, "1", []))
        alt = sec.workspace / ".git" / "objects" / "info" / "alternates"
        row("a section clone has no alternates (a container sees every object)", alt.exists(), False)
        sec = clone("d0")
        ok, _ = uses_checkout(sec, {"fetch-depth": 0}, None)
        cnt = subprocess.run(["git", "-C", str(sec.workspace), "rev-list", "--count", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
        row("checkout fetch-depth 0: full history", (ok, cnt), (True, "3"))
        ok, _ = uses_checkout(clone("d2"), {"fetch-depth": 2}, None)
        row("checkout fetch-depth 2 refuses (not emulated)", ok, False)
    bad = 0
    for label, good, got, want in rows:
        bad += not good
        print(f"{'ok  ' if good else 'BAD '} {label}" + ("" if good else f"  got={got!r} want={want!r}"))
    print(f"fat_driver self-test: {len(rows) - bad}/{len(rows)} rows as expected")
    return 1 if bad else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--sections", required=True)
    r.add_argument("--results", required=True)
    r.add_argument("--base")
    r.add_argument("--background-until")
    w = sub.add_parser("wait")
    w.add_argument("--results", required=True)
    w.add_argument("--base")
    sub.add_parser("list")
    sub.add_parser("self-test")
    a = ap.parse_args(argv)
    return {"run": cmd_run, "wait": cmd_wait, "list": cmd_list, "self-test": cmd_self_test}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
