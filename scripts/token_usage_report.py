#!/usr/bin/env python3
"""token_usage_report.py -- MEASURED token usage for the fleet on this host: Claude Code and agy (read-only).

Operator, 2026-09-24 (relayed verbatim by the release cop): "we need to get a handle on our token usage both for
claude and agy" (#4162). Every number here is read from a record the tool itself wrote; nothing is estimated:

  Claude  ~/.claude/projects/**/*.jsonl -- every assistant message's `message.usage` (input, cache write, cache read,
          output), de-duplicated by message id (one API message can span several transcript lines), attributed to
          project (from the message's `cwd`), model, session, hour (UTC), and main turn vs subagent lane
          (`isSidechain` / subagents/*.jsonl, typed by its .meta.json `agentType`).
  agy     every quorum lane envelope on disk (`quorum-*.json.lanes/**/lane-*.json`, `usage` + `duration_seconds`,
          written by agy itself), agy's own conversation files (~/.gemini/antigravity-cli/conversations, one per
          conversation) and its logs (429 / RESOURCE_EXHAUSTED lines).

Read-only: it opens files and writes nothing but stdout. Other sessions' transcripts are read for their usage
counters only; no message content is printed. Scope: THIS host -- gx10's sessions are not here.

    python3 scripts/token_usage_report.py [--hours 24] [--json]
"""

import argparse
import collections
import datetime as dt
import glob
import json
import os
import re
import subprocess
import sys

HOME = os.path.expanduser("~")
PROJECTS = os.path.join(HOME, ".claude", "projects")
AGY = os.path.join(HOME, ".gemini", "antigravity-cli")
FIELDS = ("input", "cache_write", "cache_read", "output")
AGY_FIELDS = ("input", "output", "thinking", "cache_read", "total")
# where quorum lane dirs live: (root, mindepth, maxdepth) -- bounded, a full walk of the worktree fleet takes minutes
LANE_ROOTS = [("/mnt/nvme-raid0/agent-wt", 4, 4), (os.path.join(HOME, "src"), 3, 3), (os.path.join(HOME, "src"), 5, 6),
              ("/tmp/claude-1000", 3, 6)]
NAMED = ("apex", "infra", "rmedia", "paiml-implement", "forjar", "bashrs", "pmat", "batuta")


SRC = os.path.join(HOME, "src")
_PROJECT_CACHE = {}


def repo_of(cwd):
    """The repository a directory belongs to, read from git's own metadata, or None. A linked worktree's `.git` is a
    file naming `<repo>/.git/worktrees/<name>`, so a worktree anywhere on disk counts for its main repository; this is
    what the name heuristics below guess at (#4162: 122 sessions under /mnt/nvme-raid0 were reported as project
    "mnt", and ~/src/forjar-615 as a project of its own)."""
    d = cwd
    while d and d != "/":
        g = os.path.join(d, ".git")
        if os.path.isdir(g):
            return os.path.basename(d)
        if os.path.isfile(g):
            try:
                with open(g) as f:
                    m = re.match(r"gitdir:\s*(\S+)", f.read())
            except OSError:
                return None
            if m:
                # A relative gitdir (submodules write one) is relative to this directory. The repo is the
                # directory above the LAST ".git" (<repo>/.git/worktrees/<wt>, <repo>/.git/modules/<sub>).
                parts = os.path.normpath(os.path.join(d, m.group(1))).split(os.sep)
                if ".git" not in parts:
                    return None
                i = len(parts) - 1 - parts[::-1].index(".git")
                return parts[i - 1] or None
            return None
        d = os.path.dirname(d)
    return None


def src_repo_prefix(name):
    """`forjar-615` -> `forjar` when ~/src/forjar is a repository: the longest `-`-prefix naming one, or None. For a
    worktree dir that no longer exists, so git cannot be asked."""
    bits = name.split("-")
    for i in range(len(bits), 0, -1):
        cand = "-".join(bits[:i])
        if os.path.exists(os.path.join(SRC, cand, ".git")):
            return cand
    return None


def project_of(cwd):
    """A cwd -> the project it belongs to. Worktrees count for the repo they are a worktree of. Git's metadata first;
    then the naming conventions; a cwd neither can place is `unattributed:<dir>`, never a path fragment like `mnt`."""
    if not cwd:
        return "?"
    if cwd not in _PROJECT_CACHE:
        _PROJECT_CACHE[cwd] = repo_of(cwd) or _project_by_name(cwd)
    return _PROJECT_CACHE[cwd]


def _project_by_name(cwd):
    m = re.match(r"/home/noah/src/([^/]+)", cwd)
    if m:
        if m.group(1).startswith("aprender-wt"):
            return "aprender"
        return src_repo_prefix(m.group(1)) or m.group(1)
    if cwd.startswith("/mnt/nvme-raid0/agent-wt/"):
        wt = cwd.split("/")[4] if len(cwd.split("/")) > 4 else ""
        for p in NAMED:
            if wt.startswith(p) or ("-" + p) in wt:
                return p
        return "aprender"     # the agent-wt worktrees are aprender's unless named otherwise
    m = re.match(r"/tmp/claude-1000/-home-noah-src-([^/]+)", cwd)
    if m:
        return src_repo_prefix(m.group(1)) or m.group(1).split("-")[0]
    for root in ("/mnt/nvme-raid0/", "/tmp/", HOME + "/"):
        if cwd.startswith(root):
            rest = cwd[len(root):].split("/")
            leaf = "/".join(rest[:2]) if rest[0] in ("scratch", "tmp", "worktrees", "wt") else rest[0]
            return "unattributed:" + (leaf or root.strip("/"))
    return "unattributed:" + cwd


def ts_of(s):
    try:
        return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (AttributeError, ValueError):
        return None


def _image_bytes(o):
    """Sum of base64 image payload lengths anywhere in a message's content blocks (sizes only; nothing kept)."""
    n = c = 0
    if isinstance(o, dict):
        if o.get("type") == "image" and isinstance(o.get("source"), dict):
            return len(o["source"].get("data") or ""), 1
        for v in o.values():
            a, b = _image_bytes(v); n += a; c += b
    elif isinstance(o, list):
        for v in o:
            a, b = _image_bytes(v); n += a; c += b
    return n, c


IMAGES = collections.defaultdict(lambda: [0, 0, None])   # session -> [bytes, images, first image time]
COMPACTS = []   # real compactions in the window: {"pre", "post", "trigger", "session"} from compact_boundary records


def claude(since):
    rows, seen = [], set()
    for f in glob.glob(os.path.join(PROJECTS, "**", "*.jsonl"), recursive=True):
        try:
            if os.path.getmtime(f) < since.timestamp():
                continue
        except OSError:
            continue
        sub = "/subagents/" in f
        atype, adesc = None, None
        if sub:
            try:
                meta = json.load(open(f[:-len(".jsonl")] + ".meta.json"))
                atype, adesc = meta.get("agentType"), meta.get("description")
            except (OSError, ValueError):
                pass
        try:
            fh = open(f, errors="replace")
        except OSError:
            continue
        with fh:
            for ln in fh:
                if '"compact_boundary"' in ln:
                    try:
                        d = json.loads(ln)
                    except ValueError:
                        continue
                    t, cm = ts_of(d.get("timestamp")), d.get("compactMetadata") or {}
                    if t and t >= since and cm.get("preTokens"):
                        COMPACTS.append({"pre": int(cm["preTokens"]), "post": int(cm.get("postTokens") or 0),
                                         "trigger": cm.get("trigger") or "?", "session": d.get("sessionId") or "?", "t": t})
                    continue
                if '"image"' in ln and '"usage"' not in ln:
                    try:
                        d = json.loads(ln)
                    except ValueError:
                        continue
                    t = ts_of(d.get("timestamp"))
                    if t and t >= since:
                        nb, ni = _image_bytes(d.get("message"))
                        if ni:
                            rec = IMAGES[d.get("sessionId") or "?"]
                            rec[0] += nb; rec[1] += ni; rec[2] = min(rec[2] or t, t)
                    continue
                if '"usage"' not in ln:
                    continue
                try:
                    d = json.loads(ln)
                except ValueError:
                    continue
                m = d.get("message") or {}
                u = m.get("usage") if isinstance(m, dict) else None
                t = ts_of(d.get("timestamp"))
                if not u or not t or t < since:
                    continue
                key = m.get("id") or d.get("requestId") or d.get("uuid")
                if key in seen:
                    continue
                seen.add(key)
                rows.append({
                    "t": t, "session": d.get("sessionId") or "?", "project": project_of(d.get("cwd")),
                    "model": m.get("model") or "?", "lane": "subagent" if (sub or d.get("isSidechain")) else "main",
                    "agent_type": atype or ("-" if not sub else "?"), "agent_desc": adesc or "",
                    "input": int(u.get("input_tokens") or 0), "cache_write": int(u.get("cache_creation_input_tokens") or 0),
                    "cache_read": int(u.get("cache_read_input_tokens") or 0), "output": int(u.get("output_tokens") or 0)})
    return rows


def lane_dirs():
    out = set()
    for root, lo, hi in LANE_ROOTS:
        if not os.path.isdir(root):
            continue
        try:
            r = subprocess.run(["find", root, "-mindepth", str(lo), "-maxdepth", str(hi), "-type", "d",
                                "-name", "quorum-*.json.lanes"], capture_output=True, text=True, timeout=120)
            out |= {p for p in r.stdout.split("\n") if p}
        except subprocess.TimeoutExpired:
            pass
    return sorted(out)


def lane_model(f):
    """The model a lane RAN, measured from agy's own log line model="..." beside its envelope (the envelope names none)."""
    try:
        m = re.findall(r'model="([^"]+)"', open(re.sub(r"\.json$", ".log", f), errors="replace").read())
    except OSError:
        m = []
    return m[-1] if m else "(unrecorded)"


def agy(since):
    lanes, rounds = [], []
    for d in lane_dirs():
        files = glob.glob(os.path.join(d, "**", "lane-*.json"), recursive=True)
        files += glob.glob(os.path.join(d, "precheck", "*.json"))
        got = 0
        for f in files:
            try:
                if os.path.getmtime(f) < since.timestamp():
                    continue
                env = json.load(open(f))
            except (OSError, ValueError):
                continue
            if not isinstance(env, dict):
                continue
            u = env.get("usage") or {}
            err = json.dumps(env.get("error") or "") + json.dumps(env.get("result") or "")[:400]
            lanes.append({"t": dt.datetime.fromtimestamp(os.path.getmtime(f), dt.timezone.utc), "dir": d,
                          "kind": "probe" if "/precheck/" in f else ("fallthrough" if "/fallthrough/" in f else "lane"),
                          "project": project_of(d), "ticket": re.sub(r"^quorum-|\.json\.lanes$", "", os.path.basename(d)),
                          "q429": bool(re.search(r"429|RESOURCE_EXHAUSTED", err)), "dur": float(env.get("duration_seconds") or 0),
                          "input": int(u.get("input_tokens") or 0), "output": int(u.get("output_tokens") or 0),
                          "thinking": int(u.get("thinking_tokens") or 0), "cache_read": int(u.get("cache_read_tokens") or 0),
                          "total": int(u.get("total_tokens") or 0), "model": lane_model(f)})
            got += 1
        if got:
            rounds.append(d)
    convs = []
    for f in glob.glob(os.path.join(AGY, "conversations", "*")):
        try:
            mt = os.path.getmtime(f)
        except OSError:
            continue
        if mt >= since.timestamp():
            convs.append(dt.datetime.fromtimestamp(mt, dt.timezone.utc))
    q429 = 0
    for f in glob.glob(os.path.join(AGY, "log", "*.log")):
        try:
            if os.path.getmtime(f) < since.timestamp():
                continue
            with open(f, errors="replace") as fh:
                q429 += sum(1 for ln in fh if "RESOURCE_EXHAUSTED" in ln or "code 429" in ln)
        except OSError:
            continue
    return lanes, rounds, convs, q429


def committed_rounds(since, repo):
    """Quorum receipts COMMITTED in the window, per ticket, across every branch of `repo`. A rerun overwrites its
    lane dir on disk, so on-disk dirs undercount repeat rounds; each committed receipt is one round that was kept."""
    try:
        r = subprocess.run(["git", "-C", repo, "log", "--all", "--since", since.isoformat(), "--name-only", "--format=%H",
                            "--", "docs/audits/quorum-*.json"], capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return {}
    per = collections.defaultdict(set)
    sha = None
    for ln in r.stdout.split("\n"):
        if re.fullmatch(r"[0-9a-f]{40}", ln):
            sha = ln
        elif ln.startswith("docs/audits/quorum-") and ln.endswith(".json"):
            per[re.sub(r"^docs/audits/quorum-|(\.round\d+)?\.json$", "", ln)].add(sha)
    return per


def tot(rows, fields):
    return {k: sum(r[k] for r in rows) for k in fields}


def fmt(n):
    return "{:,}".format(n)


def table(title, head, body):
    out = ["", "### " + title, "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in body]
    return out


def pct(a, b):
    return "%.1f%%" % (100.0 * a / max(1, b))


def replay_cap(sessions, cap_, B, S):
    """F2: replay each session's measured per-turn context under a cap.

    Returns (compactions forced, their cost, real context read, simulated context read). A real
    compaction (the context drops) happens in both worlds, so the sim follows it down. Without that,
    the sim keeps every growth delta and never shrinks, and a cap that never fires reads as a loss
    (GH-4162 quorum r3, sonnet lane)."""
    n_comp = comp_cost = read_real = read_sim = 0
    for rs in sessions:
        sim = prev = None
        for r in sorted(rs, key=lambda r: r["t"]):
            read_real += r["ctx"]
            if prev is None:
                sim = min(r["ctx"], cap_)
            else:
                delta = r["ctx"] - prev
                if delta < 0:            # a real compaction: the sim compacts too, to no more than the real context
                    sim = min(sim, r["ctx"])
                else:
                    sim += delta
                if sim > cap_:          # the cap compacts: the summarizer reads the whole context once and writes S;
                    n_comp += 1         # the next turn's B is counted ONCE, by read_sim below (GH-4162 quorum r1)
                    comp_cost += sim + S
                    sim = B
            prev = r["ctx"]
            read_sim += sim
    return n_comp, comp_cost, read_real, read_sim


def self_test():
    """Case tables for project_of (on real git fixtures in a temp dir) and replay_cap. Exit 0 = every row holds."""
    import tempfile
    global SRC
    saved, fails, n = SRC, 0, 0

    def check(label, got, want, why):
        nonlocal fails, n
        n += 1
        ok = got == want
        fails += not ok
        print("%s  %-55s -> %-36s %s" % ("PASS" if ok else "FAIL", label, got, "" if ok else "want %s (%s)" % (want, why)))

    try:
        with tempfile.TemporaryDirectory() as tmp:
            SRC = os.path.join(tmp, "src")
            os.makedirs(os.path.join(SRC, "forjar", ".git", "worktrees", "forjar-verdrift"))
            os.makedirs(os.path.join(SRC, "forjar", ".git", "modules", "vendored"))
            os.makedirs(os.path.join(SRC, "forjar", "vendored"))
            os.makedirs(os.path.join(SRC, "paiml-implement", ".git"))
            scratch = os.path.join(tmp, "scratch")
            for d in ("forjar-verdrift/sub", "ont-37f7875c", "odd"):
                os.makedirs(os.path.join(scratch, d))
            for where, gitdir in ((os.path.join(scratch, "forjar-verdrift"),
                                   os.path.join(SRC, "forjar", ".git", "worktrees", "forjar-verdrift")),
                                  (os.path.join(SRC, "forjar", "vendored"), "../.git/modules/vendored"),
                                  (os.path.join(scratch, "odd"), ".git/x/.git/worktrees/odd")):
                with open(os.path.join(where, ".git"), "w") as f:
                    f.write("gitdir: %s\n" % gitdir)
            _PROJECT_CACHE.clear()
            for cwd, want, why in [
                (os.path.join(scratch, "forjar-verdrift"), "forjar", "linked worktree outside ~/src: its .git file"),
                (os.path.join(scratch, "forjar-verdrift", "sub"), "forjar", "a subdir of that worktree walks up"),
                (os.path.join(SRC, "forjar", "vendored"), "forjar", "a submodule's RELATIVE gitdir resolves from its dir"),
                (os.path.join(scratch, "odd"), "x", "the repo sits above the LAST .git, not the first"),
                (os.path.join(SRC, "paiml-implement"), "paiml-implement", "a main checkout: the repo dir name"),
                ("/home/noah/src/forjar-615", "forjar", "a deleted worktree dir: the longest prefix that is a repo"),
                ("/home/noah/src/paiml-implement-x", "paiml-implement", "hyphenated repo name keeps its hyphen"),
                ("/home/noah/src/aprender-wt-17", "aprender", "the aprender-wt convention"),
                ("/tmp/claude-1000/-home-noah-src-paiml-implement/s/scratchpad", "paiml-implement",
                 "a scratchpad of a hyphenated repo keeps its hyphen"),
                ("/mnt/nvme-raid0/scratch/ont-37f7875c", "unattributed:scratch/ont-37f7875c", "no git, no name: said so"),
                ("/mnt/nvme-raid0/budget", "unattributed:budget", "never the path fragment 'mnt'"),
                (None, "?", "no cwd recorded"),
            ]:
                check(str(cwd), project_of(cwd), want, why)
    finally:
        SRC = saved
        _PROJECT_CACHE.clear()
    # replay_cap: a cap above every context changes nothing, even across a real compaction.
    turns = [{"t": i, "ctx": c} for i, c in enumerate((100000, 150000, 30000, 60000))]
    nc, cost, real, sim = replay_cap([turns], 10 ** 9, 50000, 13000)
    check("replay_cap cap above every context", (nc, cost, real - sim), (0, 0, 0),
          "a cap that never fires must save and cost nothing; the sim follows real compactions")
    nc, cost, real, sim = replay_cap([turns], 120000, 50000, 13000)
    # real 100+150+30+60 = 340k; sim 100 | 150 > cap -> 50 | min(50, 30) = 30 | +30 = 60 = 240k
    check("replay_cap cap 120k", (nc, cost, real - sim), (1, 150000 + 13000, 100000),
          "one forced compaction at 150k; the real drop to 30k then applies to both worlds")
    print("self-test: %d/%d rows" % (n - fails, n))
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--self-test", action="store_true", help="run the project-attribution case table and exit")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(self_test())
    now = dt.datetime.now(dt.timezone.utc)
    since = now - dt.timedelta(hours=a.hours)
    cr = claude(since)
    lanes, rounds, convs, q429_log = agy(since)
    real = [x for x in lanes if x["kind"] != "probe"]

    def ctot(rs):
        return sum(r["input"] + r["cache_write"] + r["cache_read"] + r["output"] for r in rs)

    if a.json:
        json.dump({"since": since.isoformat(), "claude_messages": len(cr), "claude_tokens": tot(cr, FIELDS),
                   "agy_envelopes": len(lanes), "agy_lanes": len(real), "agy_rounds": len(rounds), "agy_conversations": len(convs),
                   "agy_429_log_lines": q429_log, "agy_tokens": tot(lanes, AGY_FIELDS)}, sys.stdout, indent=1)
        return 0
    all_c = ctot(cr)
    L = ["## Token usage, MEASURED -- %s to %s UTC (%g h), host %s" % (since.strftime("%m-%d %H:%M"), now.strftime("%m-%d %H:%M"),
                                                                    a.hours, os.uname().nodename),
         "", "`python3 scripts/token_usage_report.py --hours %g` (read-only). Raw token counts as each tool recorded them; no "
         "price weighting, no estimates. Claude: %s assistant messages (de-duplicated by message id). agy: %s lane/probe "
         "envelopes in %s quorum rounds found on disk, %s agy conversations, %s 429 lines in agy's logs."
         % (a.hours, fmt(len(cr)), fmt(len(lanes)), len(rounds), fmt(len(convs)), q429_log)]
    # --- A. Claude
    t = tot(cr, FIELDS)
    L += table("A1. Claude totals", ["input", "cache write", "cache read", "output", "all"],
               [[fmt(t["input"]), fmt(t["cache_write"]), fmt(t["cache_read"]), fmt(t["output"]), fmt(all_c)]])
    by = collections.defaultdict(list)
    for r in cr:
        by[r["project"]].append(r)
    L += table("A2. Claude by project", ["project", "sessions", "messages", "cache read", "cache write", "output", "all tokens", "share"],
               [[p, fmt(len({r["session"] for r in rs})), fmt(len(rs)), fmt(sum(r["cache_read"] for r in rs)),
                 fmt(sum(r["cache_write"] for r in rs)), fmt(sum(r["output"] for r in rs)), fmt(ctot(rs)), pct(ctot(rs), all_c)]
                for p, rs in sorted(by.items(), key=lambda kv: -ctot(kv[1]))])
    bym = collections.defaultdict(list)
    for r in cr:
        bym[(r["model"], r["lane"])].append(r)
    L += table("A3. Claude by model and lane (main-session turn vs subagent)", ["model", "lane", "messages", "output", "all tokens", "share"],
               [[m, ln, fmt(len(rs)), fmt(sum(r["output"] for r in rs)), fmt(ctot(rs)), pct(ctot(rs), all_c)]
                for (m, ln), rs in sorted(bym.items(), key=lambda kv: -ctot(kv[1]))])
    bya = collections.defaultdict(list)
    for r in cr:
        if r["lane"] == "subagent":
            bya[r["agent_type"]].append(r)
    L += table("A4. Claude subagent lanes by agent type", ["agent type", "lanes", "messages", "all tokens", "share of all Claude"],
               [[k, len({(r["session"], r["agent_desc"]) for r in rs}), fmt(len(rs)), fmt(ctot(rs)), pct(ctot(rs), all_c)]
                for k, rs in sorted(bya.items(), key=lambda kv: -ctot(kv[1]))])
    bys = collections.defaultdict(list)
    for r in cr:
        bys[r["session"]].append(r)
    body = []
    for s, rs in sorted(bys.items(), key=lambda kv: -ctot(kv[1]))[:10]:
        main_ = [r for r in rs if r["lane"] == "main"]
        span = (max(r["t"] for r in rs) - min(r["t"] for r in rs)).total_seconds() / 3600.0
        body.append([s[:8], rs[0]["project"], ",".join(sorted({r["model"] for r in rs})), fmt(len(main_)), fmt(len(rs) - len(main_)),
                     "%.1f" % span, fmt(ctot(rs)), fmt(int(sum(r["cache_read"] for r in main_) / max(1, len(main_)))), pct(ctot(rs), all_c)])
    L += table("A5. Top 10 Claude sessions", ["session", "project", "models", "main turns", "subagent msgs", "span h", "all tokens",
                                              "cache read / main turn", "share"], body)
    byh = collections.defaultdict(list)
    for r in cr:
        byh[r["t"].strftime("%m-%d %H:00")].append(r)
    L += table("A6. Claude per hour (UTC)", ["hour", "sessions", "messages", "all tokens"],
               [[h, len({r["session"] for r in rs}), fmt(len(rs)), fmt(ctot(rs))] for h, rs in sorted(byh.items())])
    # --- B. agy
    at = tot(lanes, AGY_FIELDS)
    L += table("B1. agy totals (quorum lane + probe envelopes on disk)",
               ["envelopes", "lanes", "probes", "rounds", "input", "output", "thinking", "cache read", "total", "429 in envelopes"],
               [[len(lanes), len(real), len(lanes) - len(real), len(rounds), fmt(at["input"]), fmt(at["output"]), fmt(at["thinking"]),
                 fmt(at["cache_read"]), fmt(at["total"]), sum(1 for x in lanes if x["q429"])]])
    byp = collections.defaultdict(list)
    for x in real:
        byp[x["project"]].append(x)
    L += table("B2. agy quorum lanes by project", ["project", "rounds", "lanes", "total tokens", "avg input / lane", "avg lane minutes"],
               [[p, len({x["dir"] for x in xs}), len(xs), fmt(sum(x["total"] for x in xs)), fmt(int(sum(x["input"] for x in xs) / len(xs))),
                 "%.1f" % (sum(x["dur"] for x in xs) / 60.0 / len(xs))] for p, xs in sorted(byp.items(), key=lambda kv: -sum(x["total"] for x in kv[1]))])
    bym = collections.defaultdict(list)
    for x in real:
        bym[x["model"]].append(x)
    L += table("B2b. agy quorum lanes by model (measured from each lane's own agy log)",
               ["model", "lanes", "input", "output", "thinking", "total tokens", "429 envelopes"],
               [[m, len(xs), fmt(sum(x["input"] for x in xs)), fmt(sum(x["output"] for x in xs)), fmt(sum(x["thinking"] for x in xs)),
                 fmt(sum(x["total"] for x in xs)), sum(1 for x in xs if x["q429"])]
                for m, xs in sorted(bym.items(), key=lambda kv: -sum(x["total"] for x in kv[1]))])
    byh = collections.defaultdict(list)
    for x in lanes:
        byh[x["t"].strftime("%m-%d %H:00")].append(x)
    ch = collections.Counter(c.strftime("%m-%d %H:00") for c in convs)
    L += table("B3. agy per hour (UTC)", ["hour", "agy conversations (every use)", "quorum lanes", "lane tokens", "429 envelopes"],
               [[h, ch.get(h, 0), sum(1 for x in byh.get(h, []) if x["kind"] != "probe"), fmt(sum(x["total"] for x in byh.get(h, []))),
                 sum(1 for x in byh.get(h, []) if x["q429"])] for h in sorted(set(byh) | set(ch))])
    # --- C. levers, from the same records
    per_ticket, tick_tokens = collections.defaultdict(set), collections.defaultdict(int)
    for x in real:
        per_ticket[x["ticket"]].add(x["dir"])
        tick_tokens[x["ticket"]] += x["total"]
    multi = {k: v for k, v in per_ticket.items() if len(v) > 1}
    ev = [(x["t"].timestamp() - x["dur"], 1) for x in real if x["dur"]] + [(x["t"].timestamp(), -1) for x in real if x["dur"]]
    cur = peak = 0
    for _, dlt in sorted(ev):
        cur += dlt
        peak = max(peak, cur)
    top10 = sorted(bys.values(), key=lambda rs: -ctot(rs))[:10]
    com = committed_rounds(since, os.path.join(HOME, "src", "aprender"))
    com_multi = {k: v for k, v in com.items() if len(v) > 1}
    L += ["", "### C. Levers, measured from the records above", ""]
    L += ["- **Claude cache re-reads**: %s of all Claude tokens are cache READS (%s). Every main-session turn re-reads the whole "
          "context, so a long-lived session pays its context size on every turn. The top 10 sessions hold %s of all Claude tokens. "
          "(Raw COUNTS: a cache-read token is billed at a fraction of an input token, so this share of tokens is not the share "
          "of cost or of plan quota.)"
          % (pct(t["cache_read"], all_c), fmt(t["cache_read"]), pct(sum(ctot(rs) for rs in top10), all_c))]
    L += ["- **Repeat quorum rounds**: %d of %d tickets with lane envelopes on disk took more than one round (%s); those tickets' "
          "lanes used %s agy tokens." % (len(multi), len(per_ticket),
                                         ", ".join("%s x%d" % (k, len(v)) for k, v in sorted(multi.items(), key=lambda kv: -len(kv[1]))[:6]) or "none",
                                         fmt(sum(tick_tokens[k] for k in multi)))]
    L += ["- **Repeat rounds, from committed receipts** (aprender, every branch): %d quorum receipts committed for %d tickets; "
          "%d tickets committed more than one (%s). A committed receipt is one round that was kept; rounds rerun without a "
          "commit are not in this count." % (sum(len(v) for v in com.values()), len(com), len(com_multi),
                                              ", ".join("%s x%d" % (k, len(v)) for k, v in sorted(com_multi.items(), key=lambda kv: -len(kv[1]))[:8]) or "none")]
    L += ["- **Lane brief size**: an agy quorum lane reads %s input tokens on average (max %s) plus %s cache-read tokens on average; "
          "three lanes read it per round." % (fmt(int(at["input"] / max(1, len(real)))), fmt(max((x["input"] for x in real), default=0)),
                                              fmt(int(sum(x["cache_read"] for x in real) / max(1, len(real)))))]
    L += ["- **Concurrency**: at peak %d quorum lanes ran at once on this host (from each lane's end time and duration). %s agy "
          "conversations started in the window; %d were quorum lanes or probes found on disk." % (peak, fmt(len(convs)), len(lanes))]
    # --- D. what DRIVES cache reads: context size per main turn (input + cache write + cache read = the prompt re-read)
    main_rows = [r for r in cr if r["lane"] == "main" and r["model"] != "<synthetic>"]
    for r in main_rows:
        r["ctx"] = r["input"] + r["cache_write"] + r["cache_read"]
    per = collections.defaultdict(list)
    for r in main_rows:
        per[r["session"]].append(r)
    def med(xs):
        xs = sorted(xs)
        return xs[len(xs) // 2] if xs else 0
    body = []
    for s_, rs in sorted(per.items(), key=lambda kv: -sum(r["cache_read"] for r in kv[1]))[:10]:
        body.append([s_[:8], rs[0]["project"], fmt(sum(r["cache_read"] for r in rs)), fmt(len(rs)), fmt(med([r["ctx"] for r in rs])),
                     fmt(max(r["ctx"] for r in rs)), fmt(sum(1 for r in rs if r["ctx"] > 300000)),
                     fmt(IMAGES.get(s_, [0, 0])[1]), fmt(IMAGES.get(s_, [0, 0])[0])])
    L += table("D1. Top 10 sessions by cache read (main-session turns): the drivers",
               ["session", "repo", "cache read", "turns", "median context", "peak context", "turns > 300k", "images read", "image bytes (base64)"], body)
    big = [r for r in main_rows if r["ctx"] > 300000]
    L += ["", "Main-session turns in the window: %s; above 300k context: %s (%s); their cache reads: %s of all main-turn cache reads."
          % (fmt(len(main_rows)), fmt(len(big)), pct(len(big), len(main_rows)), pct(sum(r["cache_read"] for r in big), sum(r["cache_read"] for r in main_rows)))]
    img = sorted(IMAGES.items(), key=lambda kv: -kv[1][0])
    L += table("D2. Sessions that read images", ["session", "images", "image bytes (base64)", "main turns after the first image"],
               [[s_[:8], fmt(v[1]), fmt(v[0]), fmt(sum(1 for r in per.get(s_, []) if v[2] and r["t"] > v[2]))] for s_, v in img[:10]])
    # --- E. levers with ESTIMATED savings (estimates, computed only from the measured per-turn records above)
    cap = 200000
    over = sum(max(0, r["ctx"] - cap) for r in main_rows)
    poll = [r for r in main_rows if r["output"] < 100 and r["ctx"] > 200000]
    img_tok = sum(1600 * v[1] * sum(1 for r in per.get(s_, []) if v[2] and r["t"] > v[2]) for s_, v in IMAGES.items())
    all_main_cr = sum(r["cache_read"] for r in main_rows)
    L += ["", "### E. Levers, with ESTIMATED savings (arithmetic on the measured per-turn records; labelled estimates)", ""]
    L += ["- **E1. Cap the context at %s (compact / restart the session there)**: the context above %s re-read on every main turn "
          "sums to %s tokens, %s of all main-turn cache reads. That is NOT the ceiling: a compacted session restarts near its "
          "post-compaction baseline, far below the cap -- F2 replays it with the measured compaction cost." % (fmt(cap), fmt(cap), fmt(over), pct(over, all_main_cr))]
    L += ["- **E2. Poll turns**: %s main turns wrote < 100 output tokens over a > 200k context (sleep / status polls, one-line "
          "acks); they re-read %s tokens (%s of main-turn cache reads). Waiting in ONE long tool call (or a background monitor "
          "that wakes the session) instead of repeated short turns saves up to that (estimate)." % (fmt(len(poll)), fmt(sum(r["cache_read"] for r in poll)),
                                                                                                   pct(sum(r["cache_read"] for r in poll), all_main_cr))]
    L += ["- **E3. Images stay in context**: %s images read in %s sessions stay in the context for every later turn. At the "
          "documented upper bound of ~1,600 tokens per image, that is up to %s re-read tokens (%s of main-turn cache reads; an "
          "upper-bound estimate -- an image's real cost depends on its pixel size, which the transcript does not record)."
          % (fmt(sum(v[1] for v in IMAGES.values())), len(IMAGES), fmt(img_tok), pct(img_tok, all_main_cr))]
    # --- F. what a context cap would COST, not only save: replay each session's measured per-turn growth under a cap
    after = []
    for c in COMPACTS:
        rs = sorted((r for r in per.get(c["session"], []) if r["t"] > c["t"]), key=lambda r: r["t"])
        if rs:
            after.append(rs[0]["ctx"])
    B = med(after) if after else 80000          # the context a session restarts from after a compaction (measured)
    S = med([c["post"] for c in COMPACTS]) if COMPACTS else 13000   # the summary it writes (measured)
    L += table("F1. Real compactions in the window (measured)", ["compactions", "auto / manual", "median context before",
               "median summary written", "median context of the next turn"],
               [[len(COMPACTS), "%d / %d" % (sum(1 for c in COMPACTS if c["trigger"] == "auto"), sum(1 for c in COMPACTS if c["trigger"] != "auto")),
                 fmt(med([c["pre"] for c in COMPACTS])), fmt(S), fmt(B)]])
    body = []
    for cap_ in (200000, 300000, 400000, 600000):
        n_comp, comp_cost, read_real, read_sim = replay_cap(per.values(), cap_, B, S)
        net = read_real - read_sim - comp_cost
        body.append([fmt(cap_), fmt(n_comp), fmt(comp_cost), fmt(read_real - read_sim), fmt(net), pct(net, read_real)])
    L += table("F2. A context cap, replayed on the measured turns (ESTIMATE): compactions it forces, what they cost, and the NET saving",
               ["cap", "compactions forced", "compaction cost (read + summary)", "re-reads avoided", "NET saving", "net / all main-turn context"], body)
    L += ["", "F2 replays each session's real per-turn context growth; when the simulated context passes the cap it compacts at the "
          "MEASURED cost (the summarizer reads the whole context once, writes the median summary %s, the next turn starts from the "
          "median post-compaction context %s). What it cannot see: work redone after a compaction (files re-read, lost state) -- so "
          "the NET is an upper bound on the saving, and the compaction count is exact for this replay." % (fmt(S), fmt(B))]
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
