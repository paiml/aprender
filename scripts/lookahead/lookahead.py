#!/usr/bin/env python3
"""lookahead.py -- the look-ahead slot machinery of APR-LOOKAHEAD-001.

One tool, one verb per spec row, all reading the same state file
(cop-state/lookahead.json; interim /mnt/nvme-raid0/cop-inbox/lookahead.json).
The cop is the single writer of that file; workers write heartbeat files.

  validate STATE                      LA-00  schema + I1 (coverage) + I2 (uniqueness)
  heartbeat STATE --slot L --worker W LA-02  a worker's own heartbeat file (never the state)
  tick STATE                          LA-02  I1-I4 report + the actions due this tick
  respawn STATE --slot L --worker NEW LA-02  refill a dead slot; refuse a live one
  prompt --slot L --train X.Y         LA-03  the §7 standing prompt, checked against the spec
  pr-budget STATE --prs SNAP          LA-04  §2 open-PR budgets, repo-cap yield order, queue order
  readiness STATE                     LA-05  §6 Definition of Ready, judged against the tree
  latency --tag V --tag-at T ...      LA-05  train-start latency receipt (tag N -> first PR of N+1)
  budget STATE --window-pct X         LA-06  §5 pause ladder: which slots run this window
  rotate STATE --published X.Y ...    LA-07  atomic rotation; N+4 committed + launched before announce
  rotation-audit LOG                  LA-07  a rotation log is legal only in that order

Exit codes, shared by every verb:
  0  the property holds (or the action was done)
  1  a violation / a refusal -- printed one per line as `VIOLATION <id>: ...`
  2  cannot judge (file missing, not JSON, schema missing) -- never a silent pass
"""
import datetime
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCHEMA = os.environ.get(
    "LOOKAHEAD_SCHEMA", os.path.join(ROOT, "docs", "lookahead", "lookahead-state.schema.json")
)
SLOTS = ("L1", "L2", "L3")


class CannotJudge(Exception):
    """Unknown is not a pass: every path that cannot decide exits 2."""


def load_json(path, dup_sink=None):
    """Parse JSON, recording duplicate keys: `{"L1": .., "L1": ..}` is two workers
    in one slot, and a plain json.load would silently keep the last one."""

    def hook(pairs):
        seen = {}
        for k, v in pairs:
            if k in seen and dup_sink is not None:
                dup_sink.append(k)
            seen[k] = v
        return seen

    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh, object_pairs_hook=hook)
    except FileNotFoundError as exc:
        raise CannotJudge(f"{path}: not found") from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise CannotJudge(f"{path}: not JSON ({exc})") from exc


# ---- a subset of JSON Schema, enough for the schema file above ------------------
def _resolve(schema, node):
    ref = node.get("$ref")
    if ref is None:
        return node
    if not ref.startswith("#/"):
        raise CannotJudge(f"schema: unsupported $ref {ref}")
    target = schema
    for part in ref[2:].split("/"):
        if not isinstance(target, dict) or part not in target:
            raise CannotJudge(f"schema: $ref {ref} does not resolve")
        target = target[part]
    return target


def schema_errors(schema, node, value, path="$"):
    node = _resolve(schema, node)
    errs = []
    want = node.get("type")
    kinds = {"object": dict, "array": list, "string": str}
    if want == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            return [f"{path}: expected integer"]
    elif want in kinds and not isinstance(value, kinds[want]):
        return [f"{path}: expected {want}"]
    if "pattern" in node and isinstance(value, str) and not re.search(node["pattern"], value):
        errs.append(f"{path}: {value!r} does not match {node['pattern']}")
    if isinstance(value, dict):
        props = node.get("properties", {})
        for key in node.get("required", []):
            if key not in value:
                errs.append(f"{path}: missing required {key!r}")
        for key, sub in value.items():
            if key in props:
                errs.extend(schema_errors(schema, props[key], sub, f"{path}.{key}"))
            elif node.get("additionalProperties") is False:
                errs.append(f"{path}: unexpected key {key!r}")
    if isinstance(value, list) and "items" in node:
        for i, item in enumerate(value):
            errs.extend(schema_errors(schema, node["items"], item, f"{path}[{i}]"))
    return errs


# ---- trains ------------------------------------------------------------------
def train_add(train, k):
    major, minor = train.split(".")
    return f"{major}.{int(minor) + k}"


def state_violations(state, dups):
    """Every reason `state` is not a legal look-ahead state, as (id, message)."""
    out = [("I2", f"duplicate key {k!r}: two entries for one slot") for k in dups]
    schema = load_json(SCHEMA)
    out += [("SCHEMA", e) for e in schema_errors(schema, schema, state)]
    if out:
        return out
    cur = state["current_train"]
    for k, slot in enumerate(SLOTS, start=1):
        want = train_add(cur, k)
        got = state["slots"][slot]["train"]
        if got != want:
            out.append(("I1", f"{slot} holds train {got}, want {want} (current {cur} + {k})"))
        want_handoff = f"docs/lookahead/{got}.md"
        if state["slots"][slot]["handoff"] != want_handoff:
            out.append(("I1", f"{slot} handoff {state['slots'][slot]['handoff']} is not {want_handoff}"))
    owner = {}
    for slot in SLOTS:
        w = state["slots"][slot]["worker"]
        if w in owner:
            out.append(("I2", f"worker {w!r} holds both {owner[w]} and {slot}"))
        owner.setdefault(w, slot)
    return out


def cmd_validate(args):
    if len(args) != 1:
        raise CannotJudge("usage: validate STATE")
    dups = []
    state = load_json(args[0], dups)
    bad = state_violations(state, dups)
    for vid, msg in bad:
        print(f"VIOLATION {vid}: {msg}")
    if bad:
        return 1
    print(f"OK {args[0]}: current {state['current_train']}, slots "
          + ", ".join(f"{s}={state['slots'][s]['train']}:{state['slots'][s]['worker']}" for s in SLOTS))
    return 0


# ---- time, atomic writes, handoff files -------------------------------------------
TS_FMT = "%Y-%m-%dT%H:%M:%SZ"
HEARTBEAT_MAX_MIN = 120  # I3: every slot's heartbeat is less than 2 h old
HANDOFF_MAX_H = 24  # I4: every handoff file updated within 24 h


def parse_ts(text, what):
    try:
        return datetime.datetime.strptime(text, TS_FMT).replace(tzinfo=datetime.timezone.utc)
    except (TypeError, ValueError) as exc:
        raise CannotJudge(f"{what}: {text!r} is not {TS_FMT}") from exc


def fmt_ts(when):
    return when.strftime(TS_FMT)


def now_from(opts):
    """--now pins the clock, so every verb is deterministic under test."""
    if opts.get("now"):
        return parse_ts(opts["now"], "--now")
    return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0)


def write_atomic(path, obj):
    """One rename = one state transition: a reader sees the old state or the new one."""
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)
        fh.write("\n")
    os.replace(tmp, path)


def parse_opts(args, flags, bools=()):
    """--key value pairs (and bare --flag booleans) after positionals."""
    pos, opts, i = [], {}, 0
    while i < len(args):
        a = args[i]
        if a.startswith("--"):
            key = a[2:].replace("-", "_")
            if key in bools:
                opts[key] = True
                i += 1
                continue
            if key not in flags or i + 1 >= len(args):
                raise CannotJudge(f"unknown or valueless option {a}")
            opts[key] = args[i + 1]
            i += 2
        else:
            pos.append(a)
            i += 1
    return pos, opts


HANDOFF_FENCE = re.compile(r"```json lookahead-handoff\n(.*?)\n```", re.S)


def load_handoff(root, rel):
    path = os.path.join(root, rel)
    try:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    except FileNotFoundError as exc:
        raise CannotJudge(f"handoff {rel}: not found") from exc
    m = HANDOFF_FENCE.search(text)
    if not m:
        raise CannotJudge(f"handoff {rel}: no ```json lookahead-handoff block")
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError as exc:
        raise CannotJudge(f"handoff {rel}: block is not JSON ({exc})") from exc


def load_legal_state(path):
    dups = []
    state = load_json(path, dups)
    bad = state_violations(state, dups)
    return state, bad


def hb_dir_for(state_path, opts):
    return opts.get("hb_dir") or os.path.join(os.path.dirname(os.path.abspath(state_path)), "lookahead-hb")


def effective_heartbeat(state, slot, hb_dir):
    """The newest heartbeat for the slot's CURRENT worker: the cop's record, or the
    worker's own file. A file written by a different worker (a replaced one) is
    ignored -- a dead worker's last word must not keep its successor's slot 'live'."""
    rec = state["slots"][slot]
    best = parse_ts(rec["heartbeat"], f"{slot}.heartbeat")
    path = os.path.join(hb_dir, f"{slot}.json")
    if os.path.exists(path):
        hb = load_json(path)
        if isinstance(hb, dict) and hb.get("worker") == rec["worker"]:
            best = max(best, parse_ts(hb.get("at"), f"{path}.at"))
    return best


def launch_prompt(slot, train):
    return f"Run docs/prompts/lookahead-worker.md with SLOT={slot} TRAIN={train} autonomously."


# ---- LA-02: heartbeat / tick / respawn ----------------------------------------------
def cmd_heartbeat(args):
    """A worker writes its own heartbeat file; it never writes the state (single writer)."""
    pos, opts = parse_opts(args, {"slot", "worker", "now", "hb_dir", "item"})
    if len(pos) != 1 or not opts.get("slot") or not opts.get("worker"):
        raise CannotJudge("usage: heartbeat STATE --slot L? --worker NAME [--item TEXT] [--now T] [--hb-dir D]")
    state, bad = load_legal_state(pos[0])
    if bad:
        raise CannotJudge(f"{pos[0]} is not a legal state ({bad[0][0]}: {bad[0][1]})")
    slot = opts["slot"]
    if slot not in SLOTS:
        raise CannotJudge(f"no slot {slot}")
    owner = state["slots"][slot]["worker"]
    if opts["worker"] != owner:
        print(f"REFUSED: {opts['worker']} does not hold {slot} (held by {owner}); a second worker on one slot breaks I2")
        return 1
    hb_dir = hb_dir_for(pos[0], opts)
    os.makedirs(hb_dir, exist_ok=True)
    rec = {"slot": slot, "worker": owner, "train": state["slots"][slot]["train"],
           "at": fmt_ts(now_from(opts)), "item": opts.get("item", "")}
    write_atomic(os.path.join(hb_dir, f"{slot}.json"), rec)
    print(f"HEARTBEAT {slot} {owner} {rec['at']}")
    return 0


def tick_report(state_path, opts):
    state, bad = load_legal_state(state_path)
    now = now_from(opts)
    hb_dir = hb_dir_for(state_path, opts)
    root = opts.get("root") or ROOT
    inv = {"I1": "ok", "I2": "ok", "I3": "ok", "I4": "ok"}
    actions = []
    for vid, msg in bad:
        inv["I1" if vid == "I1" else "I2"] = "fail"
        actions.append({"action": "REPAIR", "invariant": vid, "detail": msg})
    if bad:  # an illegal state has no trustworthy slots to judge liveness on
        return {"invariants": inv, "actions": actions, "slots": []}
    slots = []
    for slot in SLOTS:
        rec = state["slots"][slot]
        age = int((now - effective_heartbeat(state, slot, hb_dir)).total_seconds() // 60)
        row = {"slot": slot, "train": rec["train"], "worker": rec["worker"], "heartbeat_age_min": age}
        if age >= HEARTBEAT_MAX_MIN:
            inv["I3"] = "fail"
            actions.append({"action": "RESPAWN", "slot": slot, "train": rec["train"],
                            "dead_worker": rec["worker"], "heartbeat_age_min": age,
                            "handoff": rec["handoff"], "prompt": launch_prompt(slot, rec["train"])})
        try:
            ho = load_handoff(root, rec["handoff"])
            ho_age_h = (now - parse_ts(ho.get("updated"), f"{rec['handoff']}.updated")).total_seconds() / 3600
            row["handoff_age_h"] = round(ho_age_h, 1)
            stale = ho_age_h >= HANDOFF_MAX_H or ho.get("train") != rec["train"]
        except CannotJudge as exc:
            row["handoff_age_h"] = None
            stale, ho_age_h = True, str(exc)
        if stale:
            inv["I4"] = "fail"
            actions.append({"action": "HANDOFF-STALE", "slot": slot, "worker": rec["worker"],
                            "handoff": rec["handoff"], "detail": ho_age_h})
        slots.append(row)
    return {"current_train": state["current_train"], "invariants": inv, "actions": actions, "slots": slots}


def cmd_tick(args):
    """LA-02: the cop's hourly tick. Prints a JSON report (§10 `invariants`, `slots`)
    plus the actions the cop must take this tick; exit 1 iff any action is due."""
    pos, opts = parse_opts(args, {"now", "hb_dir", "root"})
    if len(pos) != 1:
        raise CannotJudge("usage: tick STATE [--now T] [--hb-dir D] [--root REPO]")
    rep = tick_report(pos[0], opts)
    print(json.dumps(rep, indent=2))
    for a in rep["actions"]:
        print(f"ACTION {a['action']} {a.get('slot', a.get('invariant', ''))}")
    return 1 if rep["actions"] else 0


def cmd_respawn(args):
    """Refill a slot whose worker is dead. Refuses when the slot's worker is live
    (a planted duplicate) or when the new worker already holds a slot (I2)."""
    pos, opts = parse_opts(args, {"slot", "worker", "now", "hb_dir"})
    if len(pos) != 1 or opts.get("slot") not in SLOTS or not opts.get("worker"):
        raise CannotJudge("usage: respawn STATE --slot L? --worker NEW [--now T] [--hb-dir D]")
    state, bad = load_legal_state(pos[0])
    if bad:
        raise CannotJudge(f"{pos[0]} is not a legal state ({bad[0][0]}: {bad[0][1]}); repair it first")
    slot, new, now = opts["slot"], opts["worker"], now_from(opts)
    age = int((now - effective_heartbeat(state, slot, hb_dir_for(pos[0], opts))).total_seconds() // 60)
    if age < HEARTBEAT_MAX_MIN:
        print(f"REFUSED: {slot} is held by live worker {state['slots'][slot]['worker']} "
              f"(heartbeat {age} min old < {HEARTBEAT_MAX_MIN}); a second worker breaks I2")
        return 1
    for other in SLOTS:
        if state["slots"][other]["worker"] == new:
            print(f"REFUSED: {new} already holds {other}; no worker holds two slots (I2)")
            return 1
    rec = state["slots"][slot]
    dead = rec["worker"]
    rec.update({"worker": new, "tmux": new, "since": fmt_ts(now), "heartbeat": fmt_ts(now)})
    after = state_violations(json.loads(json.dumps(state)), [])
    if after:
        raise CannotJudge(f"respawn would write an illegal state: {after[0]}")
    write_atomic(pos[0], state)
    print(f"RESPAWNED {slot} train {rec['train']}: {dead} -> {new} (dead {age} min)")
    print(f"LAUNCH tmux={new} handoff={rec['handoff']} prompt={launch_prompt(slot, rec['train'])!r}")
    return 0


# ---- LA-03: the standing prompt ---------------------------------------------------
PROMPT_DOC = "docs/prompts/lookahead-worker.md"
SPEC_DOC = "docs/specifications/APR-LOOKAHEAD-001-rolling-epic-workers.md"
FENCE = re.compile(r"^```\n(You are the dedicated look-ahead worker.*?)^```$", re.S | re.M)


def standing_block(root, rel):
    try:
        with open(os.path.join(root, rel), encoding="utf-8") as fh:
            text = fh.read()
    except FileNotFoundError as exc:
        raise CannotJudge(f"{rel}: not found") from exc
    blocks = FENCE.findall(text)
    if len(blocks) != 1:
        raise CannotJudge(f"{rel}: {len(blocks)} standing-prompt blocks, want exactly 1")
    return blocks[0], text


def cmd_prompt(args):
    """Render the standing prompt for one slot. Refuses (1) when the prompt file's
    block has drifted from spec §7, or when it lacks the launch line tick prints."""
    pos, opts = parse_opts(args, {"slot", "train", "root"})
    slot, train, root = opts.get("slot"), opts.get("train"), opts.get("root") or ROOT
    if pos or slot not in SLOTS or not re.fullmatch(r"[0-9]+\.[0-9]+", train or ""):
        raise CannotJudge("usage: prompt --slot L? --train X.Y [--root REPO]")
    block, text = standing_block(root, PROMPT_DOC)
    spec_block, _ = standing_block(root, SPEC_DOC)
    if block != spec_block:
        print(f"VIOLATION PROMPT: {PROMPT_DOC} standing block differs from {SPEC_DOC} §7")
        return 1
    if launch_prompt("L1", "0.71") not in text:
        print(f"VIOLATION PROMPT: {PROMPT_DOC} lacks the launch line tick prints: {launch_prompt('L1', '0.71')!r}")
        return 1
    print(block.replace("${SLOT}", slot).replace("${TRAIN}", train), end="")
    return 0


# ---- LA-04: open-PR budgets and queue order ----------------------------------------
PR_BUDGET = {"L1": 2, "L2": 1, "L3": 1}  # §2
YIELD_ORDER = ["L3", "L2", "L1"]  # at the repo cap L3 yields first, then L2; L1 only to train N
DOC_PREFIXES = ("docs/", "contracts/")  # what L2 may land undrafted and L3 may land at all


def pr_train(pr):
    m = re.fullmatch(r"v?([0-9]+\.[0-9]+)(\.[0-9]+)?", str(pr.get("milestone") or ""))
    return m.group(1) if m else None


def normalize_pr(raw):
    """Accept `gh pr list --json number,milestone,isDraft,createdAt,files` rows as well
    as the plain snapshot shape, so the cop can pipe gh output straight in."""
    if not isinstance(raw, dict):
        raise CannotJudge(f"PR entry is not an object: {raw!r}")
    ms = raw.get("milestone")
    files = raw.get("files")
    if isinstance(files, list):
        files = [f.get("path") if isinstance(f, dict) else f for f in files]
    return {"number": raw.get("number"),
            "milestone": ms.get("title") if isinstance(ms, dict) else ms,
            "draft": raw.get("draft", raw.get("isDraft", False)),
            "created": raw.get("created", raw.get("createdAt", "")),
            "files": files}


def budget_actions(state, snap):
    """The actions due on a PR snapshot {cap, prs:[{number, milestone, draft, created,
    files}], queue:[numbers in merge-queue order]} under the §2 rules."""
    by_train = {state["slots"][s]["train"]: s for s in SLOTS}
    cur = state["current_train"]
    prs = snap.get("prs")
    cap = snap.get("cap")
    if not isinstance(prs, list) or not isinstance(cap, int):
        raise CannotJudge("snapshot needs an integer `cap` and a `prs` list")
    slot_of, actions = {}, []
    for pr in prs:
        if not isinstance(pr, dict) or not isinstance(pr.get("number"), int):
            raise CannotJudge(f"PR entry without an integer number: {pr!r}")
        s = by_train.get(pr_train(pr))
        if s:
            slot_of[pr["number"]] = s
    blocked = set()
    for s in SLOTS:  # per-slot budget: the NEWEST PRs over the budget are blocked
        mine = sorted((p for p in prs if slot_of.get(p["number"]) == s),
                      key=lambda p: (str(p.get("created", "")), p["number"]))
        for p in mine[PR_BUDGET[s]:]:
            blocked.add(p["number"])
            actions.append(("BLOCK", p["number"], f"{s} holds {len(mine)} open PRs, budget {PR_BUDGET[s]}"))
    for p in prs:  # what each slot may land
        s = slot_of.get(p["number"])
        files = p.get("files")
        if s in ("L2", "L3") and files is None:
            raise CannotJudge(f"#{p['number']} ({s}) has no `files`; cannot judge its content rule")
        code = [f for f in (files or []) if not f.startswith(DOC_PREFIXES)]
        if s == "L3" and code:
            actions.append(("BLOCK", p["number"], f"L3 lands specs and docs only; touches {code[0]}"))
        if s == "L2" and code and not p.get("draft"):
            actions.append(("DRAFT", p["number"], f"L2 code must be a draft PR; touches {code[0]}"))
    live = [p for p in prs if p["number"] not in blocked]
    over = len(live) - cap
    for s in YIELD_ORDER:  # repo cap: look-ahead yields, never train N or unslotted work
        if over <= 0:
            break
        for p in sorted((p for p in live if slot_of.get(p["number"]) == s),
                        key=lambda p: (str(p.get("created", "")), p["number"]), reverse=True):
            if over <= 0:
                break
            actions.append(("YIELD", p["number"], f"repo at {len(live)} open PRs, cap {cap}: {s} yields"))
            over -= 1
    queue = snap.get("queue", [])
    known = {p["number"]: p for p in prs}
    train_pos = [i for i, n in enumerate(queue) if pr_train(known.get(n, {})) == cur]
    last_train = max(train_pos) if train_pos else -1
    seen_train = set()
    for i, n in enumerate(queue):
        s = slot_of.get(n)
        if not s:
            continue
        if i < last_train:
            actions.append(("REQUEUE", n, f"{s} PR queued ahead of a train-{cur} PR"))
        tr = pr_train(known[n])
        if tr in seen_train:
            actions.append(("REQUEUE", n, f"a second train-{tr} look-ahead PR in the queue"))
        seen_train.add(tr)
    return actions


def cmd_pr_budget(args):
    pos, opts = parse_opts(args, {"prs", "cap", "queue"})
    if len(pos) != 1 or not opts.get("prs"):
        raise CannotJudge("usage: pr-budget STATE --prs SNAPSHOT.json | GH_PR_LIST.json [--cap N] [--queue n,n,...]")
    state, bad = load_legal_state(pos[0])
    if bad:
        raise CannotJudge(f"{pos[0]} is not a legal state ({bad[0][0]}: {bad[0][1]})")
    snap = load_json(opts["prs"])
    if isinstance(snap, list):  # raw gh output
        snap = {"prs": snap}
    if not isinstance(snap, dict):
        raise CannotJudge(f"{opts['prs']}: not a snapshot object or a PR list")
    try:
        if "cap" in opts:
            snap["cap"] = int(opts["cap"])
        if "queue" in opts:
            snap["queue"] = [int(n) for n in opts["queue"].split(",") if n]
    except ValueError as exc:
        raise CannotJudge(f"--cap/--queue: {exc}") from exc
    snap["prs"] = [normalize_pr(p) for p in snap.get("prs") or []] if isinstance(snap.get("prs"), list) else None
    actions = budget_actions(state, snap)
    for verb, n, why in actions:
        print(f"{verb} #{n}: {why}")
    if not actions:
        print("OK: every look-ahead PR is within its slot budget and queued behind the current train")
    return 1 if actions else 0


# ---- LA-05: readiness (§6) and train-start latency ---------------------------------
SCHEDULE_DOC = "docs/specifications/06x-release-schedule.md"
CRITERION_STATES = ("open", "met", "moved")
LATENCY_TARGET_MIN = 120  # §6: tag of N -> first merged PR of N+1


def tree_file(root, rel):
    """A handoff path is judged against the tree: a repo-relative file that exists
    (an `#anchor` suffix is allowed on spec paths)."""
    if not isinstance(rel, str) or not rel or rel.startswith("/") or ".." in rel.split("/"):
        return False
    return os.path.isfile(os.path.join(root, rel.split("#", 1)[0]))


def row_problems(root, row):
    """Why one handoff row is not Ready (§6: spec, contract, planted falsifier, baseline, owner)."""
    if not isinstance(row, dict):
        return ["row is not an object"]
    out = []
    if not tree_file(root, row.get("spec")):
        out.append(f"spec {row.get('spec')!r} not in the tree")
    contract = row.get("contract")
    if not (isinstance(contract, str) and contract.startswith("contracts/") and tree_file(root, contract)):
        out.append(f"contract {contract!r} not a file under contracts/")
    else:
        fid = row.get("falsifier")
        with open(os.path.join(root, contract), encoding="utf-8") as fh:
            if not (isinstance(fid, str) and fid and re.search(rf"\b{re.escape(fid)}\b", fh.read())):
                out.append(f"falsifier {fid!r} not in {contract}")
    if not tree_file(root, row.get("baseline")):
        out.append(f"baseline {row.get('baseline')!r} not in the tree")
    if not (isinstance(row.get("owner"), str) and row["owner"].strip()):
        out.append("no owner proposed")
    return out


def ranked(rows):
    """Rows by EV, highest first; a row without a numeric `ev` is unranked and sorts last."""
    def key(r):
        ev = r.get("ev") if isinstance(r, dict) else None
        return -ev if isinstance(ev, (int, float)) and not isinstance(ev, bool) else float("inf")
    return sorted(rows, key=key)


def criteria_view(root, crit, problems, train):
    """`met` counts only with a receipt file in the tree; anything else stays open."""
    view = {}
    for cid, val in sorted((crit or {}).items()):
        status = val.get("status") if isinstance(val, dict) else val
        if status not in CRITERION_STATES:
            problems.append(f"{train} {cid}: status {status!r} not one of {'|'.join(CRITERION_STATES)}")
            status = "open"
        elif status == "met" and not (isinstance(val, dict) and tree_file(root, val.get("receipt"))):
            problems.append(f"{train} {cid}: met without a receipt file in the tree")
            status = "open"
        view[cid] = status
    return view


def theme_ruled(root, train, theme):
    try:
        with open(os.path.join(root, SCHEDULE_DOC), encoding="utf-8") as fh:
            text = fh.read()
    except FileNotFoundError:
        return False
    return bool(theme) and re.search(rf"^\| {re.escape(train)} \| \*\*{re.escape(theme)}\*\* \|", text, re.M) is not None


def readiness_report(state, root, epics):
    slots = state["slots"]
    ho = {s: load_handoff(root, slots[s]["handoff"]) for s in SLOTS}
    problems = {s: [] for s in SLOTS}
    for s in SLOTS:
        if ho[s].get("train") != slots[s]["train"]:
            problems[s].append(f"handoff names train {ho[s].get('train')!r}, slot holds {slots[s]['train']}")
    exit_criteria = {slots[s]["train"]: criteria_view(root, ho[s].get("exit_criteria"), problems[s], slots[s]["train"])
                     for s in SLOTS}
    # N+1: the Definition of Ready at the cut of N
    h1, t1 = ho["L1"], slots["L1"]["train"]
    top10 = ranked(h1.get("rows") or [])[:10]
    ready10 = 0
    for r in top10:
        why = row_problems(root, r)
        if why:
            problems["L1"].append(f"row {r.get('id', '?') if isinstance(r, dict) else '?'}: {'; '.join(why)}")
        else:
            ready10 += 1
    if len(top10) < 10:
        problems["L1"].append(f"only {len(top10)} rows ranked; the top 10 must each be Ready")
    first5 = h1.get("first5_prs") or []
    first5_ok = len(first5) >= 5 and all(isinstance(p, dict) and str(p.get("owner", "")).strip() for p in first5[:5])
    if not first5_ok:
        problems["L1"].append("first 5 PRs not identified with owners")
    pending = [q for q in h1.get("rulings_pending") or [] if not (isinstance(q, dict) and q.get("request"))]
    if pending:
        problems["L1"].append(f"{len(pending)} operator question(s) without a filed ruling request")
    groomed = None
    ep = (epics or {}).get(str(h1.get("epic")))
    if isinstance(ep, dict) and isinstance(ep.get("tickets"), int) and ep["tickets"] > 0:
        groomed = round(100 * ep.get("groomed", 0) / ep["tickets"])
    if groomed != 100:
        problems["L1"].append("epic grooming unverified" if groomed is None else f"epic {groomed}% groomed")
    # N+2: theme ruled, top 20 ranked, top 5 specified
    h2, t2 = ho["L2"], slots["L2"]["train"]
    rows2 = h2.get("rows") or []
    ranked2 = [r for r in rows2 if isinstance(r, dict) and isinstance(r.get("ev"), (int, float))]
    spec5 = sum(1 for r in ranked(rows2)[:5] if isinstance(r, dict) and tree_file(root, r.get("spec"))
                and tree_file(root, r.get("contract")) and str(r.get("contract")).startswith("contracts/"))
    ruled2 = theme_ruled(root, t2, h2.get("theme"))
    # N+3: theme proposed, risk register, measurement plan (files, not flags)
    h3 = ho["L3"]
    report = {
        "exit_criteria": exit_criteria,
        "n_plus_1": {"train": t1, "epic_groomed_pct": groomed, "top10_ready": f"{ready10}/10",
                     "first5_prs_identified": first5_ok, "rulings_pending": len(pending),
                     "ready": not problems["L1"]},
        "n_plus_2": {"train": t2, "theme_ruled": ruled2, "top20_ranked": len(ranked2) >= 20,
                     "top5_specified": f"{spec5}/5"},
        "n_plus_3": {"train": slots["L3"]["train"], "theme_proposed": bool(str(h3.get("theme") or "").strip()),
                     "risk_register": tree_file(root, h3.get("risk_register")),
                     "measurement_plan": tree_file(root, h3.get("measurement_plan"))},
        "not_ready": {s: problems[s] for s in SLOTS if problems[s]},
    }
    return report


def cmd_readiness(args):
    """LA-05: the §10 `readiness` section, every field derived from the handoff blocks
    judged against the tree. Exit 1 while train N+1 is NOT READY."""
    pos, opts = parse_opts(args, {"root", "epics"})
    if len(pos) != 1:
        raise CannotJudge("usage: readiness STATE [--root REPO] [--epics EPICS.json]")
    state, bad = load_legal_state(pos[0])
    if bad:
        raise CannotJudge(f"{pos[0]} is not a legal state ({bad[0][0]}: {bad[0][1]})")
    epics = load_json(opts["epics"]) if opts.get("epics") else None
    rep = readiness_report(state, opts.get("root") or ROOT, epics)
    print(json.dumps({"readiness": rep}, indent=2))
    t1 = rep["n_plus_1"]["train"]
    if rep["n_plus_1"]["ready"]:
        print(f"READY {t1}")
        return 0
    for why in rep["not_ready"].get("L1", []):
        print(f"NOT READY {t1}: {why}")
    return 1


def cmd_latency(args):
    """LA-05: train-start latency receipt -- minutes from the tag of N to the first
    merged PR of N+1 (`gh pr list --state merged --json number,milestone,mergedAt`)."""
    pos, opts = parse_opts(args, {"tag", "tag_at", "train", "merged"})
    if pos or not all(opts.get(k) for k in ("tag", "tag_at", "train", "merged")):
        raise CannotJudge("usage: latency --tag vX.Y.Z --tag-at T --train X.Y --merged GH_MERGED.json")
    tag_at = parse_ts(opts["tag_at"], "--tag-at")
    merged = load_json(opts["merged"])
    if not isinstance(merged, list):
        raise CannotJudge(f"{opts['merged']}: not a PR list")
    first = None
    for raw in merged:
        pr = normalize_pr(raw)
        at = raw.get("mergedAt") if isinstance(raw, dict) else None
        if pr_train(pr) != opts["train"] or not at:
            continue
        when = parse_ts(at, f"#{pr['number']}.mergedAt")
        if when >= tag_at and (first is None or when < first[1]):
            first = (pr["number"], when)
    rec = {"tag": opts["tag"], "tag_at": fmt_ts(tag_at), "train": opts["train"],
           "first_pr": first[0] if first else None,
           "latency_min": int((first[1] - tag_at).total_seconds() // 60) if first else None,
           "target_min": LATENCY_TARGET_MIN}
    rec["met"] = rec["latency_min"] is not None and rec["latency_min"] <= LATENCY_TARGET_MIN
    print(json.dumps(rec))
    return 0 if rec["met"] else 1


# ---- LA-06: the §5 budget ladder ---------------------------------------------------
L3_PAUSE_PCT = 70  # 5-hour window at or above this: L3 pauses
L2_PAUSE_PCT = 85  # above this: L2 pauses too
SHARE_CAP_PCT = 15  # S-7: look-ahead share above this for 2 consecutive windows


def pct(opts, key, required):
    raw = opts.get(key)
    if raw is None:
        if required:
            raise CannotJudge(f"--{key.replace('_', '-')} is required")
        return None
    try:
        val = float(raw)
    except ValueError as exc:
        raise CannotJudge(f"--{key.replace('_', '-')}: {raw!r} is not a number") from exc
    if not 0 <= val <= 100:
        raise CannotJudge(f"--{key.replace('_', '-')}: {val} is outside 0..100")
    return val


def ladder(window, account, train_needs, s1, share, prev_share):
    """Slot modes under §5 and §8: active | paused (heartbeat only) | no-arm (L1 under S-1)."""
    mode = {s: "active" for s in SLOTS}
    pauses, stops = [], []
    if window >= L3_PAUSE_PCT:
        mode["L3"] = "paused"
        pauses.append(f"L3: window {window:g}% >= {L3_PAUSE_PCT}%")
    if window > L2_PAUSE_PCT:
        mode["L2"] = "paused"
        pauses.append(f"L2: window {window:g}% > {L2_PAUSE_PCT}%")
    if account is not None and account >= 100 and train_needs:
        mode["L1"] = "paused"
        pauses.append("L1: account at 100% and the current train needs the budget")
    if s1 and mode["L1"] == "active":
        mode["L1"] = "no-arm"
        pauses.append("L1: S-1 release cut in progress, no arming")
    if share is not None and prev_share is not None and share > SHARE_CAP_PCT and prev_share > SHARE_CAP_PCT:
        stops.append(f"S-7: look-ahead share {prev_share:g}% then {share:g}% > {SHARE_CAP_PCT}% for 2 windows")
        if mode["L3"] == "active":
            mode["L3"] = "paused"
            pauses.append("L3: S-7")
    return mode, pauses, stops


def cmd_budget(args):
    """LA-06: which slots run this window. Prints the §10 `budget` section; exit 1 when
    an S-7 stop is due (the cop reports it), else 0."""
    pos, opts = parse_opts(args, {"window_pct", "account_pct", "share_pct", "prev_share_pct"},
                           bools=("train_needs_budget", "s1"))
    if len(pos) != 1:
        raise CannotJudge("usage: budget STATE --window-pct X [--account-pct Y --train-needs-budget] [--s1] [--share-pct S --prev-share-pct P]")
    state, bad = load_legal_state(pos[0])
    if bad:
        raise CannotJudge(f"{pos[0]} is not a legal state ({bad[0][0]}: {bad[0][1]})")
    window = pct(opts, "window_pct", True)
    share = pct(opts, "share_pct", False)
    mode, pauses, stops = ladder(window, pct(opts, "account_pct", False), opts.get("train_needs_budget", False),
                                 opts.get("s1", False), share, pct(opts, "prev_share_pct", False))
    print(json.dumps({"budget": {"window_pct": window, "lookahead_share_pct": share, "pauses": pauses,
                                 "slots": {s: {"train": state["slots"][s]["train"], "mode": mode[s]} for s in SLOTS},
                                 "stops": stops}}, indent=2))
    for s in SLOTS:
        print(f"SLOT {s} {mode[s]}")
    for stop in stops:
        print(f"STOP {stop}")
    return 1 if stops else 0


# ---- LA-07: rotation (§3.3, Proposition 1) ----------------------------------------
# The order is the proof: the N+4 slot exists in the committed state, and its worker
# is launched, BEFORE the release is announced. A reader of the state never sees
# {N+2, N+3} alone (I1), and a tick cannot race the rotation into two L3s (I2).
ROTATION_ORDER = ["HANDOFF", "STATE", "LAUNCH", "ANNOUNCE"]


def rotated(state, new_worker, now):
    old = json.loads(json.dumps(state))
    n1 = old["slots"]["L1"]
    n4 = train_add(old["slots"]["L3"]["train"], 1)
    new = {"current_train": n1["train"],
           "slots": {"L1": old["slots"]["L2"], "L2": old["slots"]["L3"],
                     "L3": {"train": n4, "worker": new_worker, "tmux": new_worker, "since": fmt_ts(now),
                            "heartbeat": fmt_ts(now), "handoff": f"docs/lookahead/{n4}.md"}},
           "pool": list(old.get("pool") or []) + [{"worker": n1["worker"], "train": n1["train"],
                                                   "role": "epic lead", "since": fmt_ts(now)}]}
    return new, n1


HANDOFF_SKELETON = """# Look-ahead handoff — train {train} (L3)

Opened by the rotation of {when} (APR-LOOKAHEAD-001 §3.3). The L3 worker proposes
the theme (§2a) and lands this file on `main`.

```json lookahead-handoff
{{
  "train": "{train}",
  "slot": "L3",
  "theme": "",
  "updated": "{when}",
  "exit_criteria": {{}},
  "rows": [],
  "first5_prs": [],
  "rulings_pending": [],
  "risk_register": null,
  "measurement_plan": null
}}
```

## Done

## Next

## Risks

## Rulings needed

## Baselines
"""


def log_step(log, step, detail, now):
    print(f"STEP {step}: {detail}")
    if log:
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"step": step, "at": fmt_ts(now), "detail": detail}) + "\n")


def cmd_rotate(args):
    """LA-07: train N published to crates.io -> one atomic state write that relabels
    L2->L1, L3->L2, opens L3 for N+4 and moves the old L1 worker to the pool."""
    pos, opts = parse_opts(args, {"published", "new_l3_worker", "now", "root", "log"}, bools=("dry_run",))
    if len(pos) != 1 or not opts.get("published") or not opts.get("new_l3_worker"):
        raise CannotJudge("usage: rotate STATE --published X.Y --new-l3-worker NAME [--now T] [--root REPO] [--log F] [--dry-run]")
    state, bad = load_legal_state(pos[0])
    if bad:
        raise CannotJudge(f"{pos[0]} is not a legal state ({bad[0][0]}: {bad[0][1]}); repair before rotating")
    if opts["published"] != state["current_train"]:
        print(f"REFUSED: published {opts['published']} is not the current train {state['current_train']}; a slip rotates nothing")
        return 1
    new_worker, now = opts["new_l3_worker"], now_from(opts)
    new, n1 = rotated(state, new_worker, now)
    after = state_violations(json.loads(json.dumps(new)), [])
    if after:
        print(f"REFUSED: the rotated state would be illegal ({after[0][0]}: {after[0][1]})")
        return 1
    if opts.get("dry_run"):
        print(json.dumps(new, indent=2))
        print("DRY-RUN slots " + " ".join(f"{s}={new['slots'][s]['train']}" for s in SLOTS))
        return 0
    root, log, n4 = opts.get("root") or ROOT, opts.get("log"), new["slots"]["L3"]["train"]
    for step in ROTATION_ORDER:
        if step == "HANDOFF":
            rel = new["slots"]["L3"]["handoff"]
            path = os.path.join(root, rel)
            if not os.path.exists(path):
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(HANDOFF_SKELETON.format(train=n4, when=fmt_ts(now)))
            log_step(log, step, f"{rel} present (the new L3 worker lands it on main)", now)
        elif step == "STATE":
            write_atomic(pos[0], new)
            log_step(log, step, "slots " + " ".join(f"{s}={new['slots'][s]['train']}" for s in SLOTS)
                     + f"; {n1['worker']} -> pool as {n1['train']} epic lead", now)
        elif step == "LAUNCH":
            log_step(log, step, f"tmux={new_worker} prompt={launch_prompt('L3', n4)!r}", now)
        elif step == "ANNOUNCE":
            log_step(log, step, f"{opts['published']} published; current train is now {new['current_train']}", now)
    return 0


def cmd_rotation_audit(args):
    """LA-07: a rotation log is legal only in ROTATION_ORDER, each step once."""
    if len(args) != 1:
        raise CannotJudge("usage: rotation-audit LOG")
    try:
        with open(args[0], encoding="utf-8") as fh:
            steps = [json.loads(line)["step"] for line in fh if line.strip()]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise CannotJudge(f"{args[0]}: {exc}") from exc
    if steps != ROTATION_ORDER:
        first = next((i for i, (a, b) in enumerate(zip(steps, ROTATION_ORDER)) if a != b), min(len(steps), len(ROTATION_ORDER)))
        print(f"VIOLATION ROTATION: steps {steps}, want {ROTATION_ORDER} (first difference at step {first + 1}); "
              "announcing before the N+4 slot is committed and launched breaks I1/I2")
        return 1
    print("OK rotation " + " -> ".join(steps))
    return 0


VERBS = {"validate": cmd_validate, "heartbeat": cmd_heartbeat, "tick": cmd_tick,
         "respawn": cmd_respawn, "prompt": cmd_prompt, "pr-budget": cmd_pr_budget,
         "readiness": cmd_readiness, "latency": cmd_latency,
         "budget": cmd_budget, "rotate": cmd_rotate, "rotation-audit": cmd_rotation_audit}


def main(argv):
    if len(argv) == 2 and argv[1] in ("-h", "--help"):
        print(__doc__)
        return 0
    if len(argv) < 2 or argv[1] not in VERBS:
        print("usage: lookahead.py {" + "|".join(VERBS) + "} ...", file=sys.stderr)
        return 2
    try:
        return VERBS[argv[1]](argv[2:])
    except CannotJudge as exc:
        print(f"CANNOT-JUDGE: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
