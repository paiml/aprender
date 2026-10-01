"""Task and row validation, text-level overlap refusal and the seeded calibration split (D-05, D-06).

No torch import (module level or anywhere): `python data.py --selftest` runs with numpy + pyyaml only.

    load_task(path)            task.json, the D-05 / decide-apr-v1 `task_json_schema`: exactly
                               {type: "choice", instructions: string, criteria: object}; criteria
                               DOCUMENT ORDER is the label index (parsed with object_pairs_hook=list
                               so both order and duplicates are observable); >= 2 unique names; each
                               description a string or null.
    load_rows(path, task, role)
                               train.jsonl / eval.jsonl / shift.jsonl (role train | eval | shift),
                               decide-apr-v1 `train_row_schema`: exactly {text: string, label:
                               <criterion NAME>}; unknown names and keys refused.
    normalized_sha256(text)    sha256 of `nfc-trim-ws-v1` (aprender-contrastive-data hash.rs): NFC, trim,
                               collapse every Unicode White_Space run to one U+0020 -- the Rust
                               `split_whitespace` set, NOT Python's str.split() (which also splits on
                               U+001C..U+001F).
    refuse_overlap(train, ev, role="eval")
                               an eval (or shift) text whose normalized hash is a train hash is refused.
    in_distribution_heldout(validation, train_pool, shot_ids, excluded_ids, group_member_ids, shots)
                               laya-finetune-gate-v1 `eval_set.demo_rule` (A2, 1.4.0) as a pure function.
    group_train(rows)          train rows grouped by normalized hash; a group whose copies carry
                               different labels is refused (row-disjoint is not text-disjoint).
    calibration_split(rows, fraction, min_per_class, seed, labels)
                               draws whole GROUPS, stratified by label, seeded: per class at least
                               max(min_per_class, ceil(fraction * n_class)) rows, so every copy of a
                               text lands on one side. Returns (fit_ids, calib_ids, slice_ids,
                               slice_ids_sha256) with slice_ids the SORTED 0-based train.jsonl row
                               indices and its sha256 over the compact JSON array bytes.

Every refusal is a DataError whose message starts `REFUSED <rule>:` so the rule is named.
"""
import json
import math
import sys
import unicodedata
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import sha256_bytes, sha256_file  # noqa: E402,F401  (re-exported: data.sha256_file)

# Unicode White_Space (the property Rust's char::is_whitespace / str::split_whitespace use).
_WHITE_SPACE = frozenset(
    [chr(c) for c in range(0x09, 0x0E)] + [" ", "\u0085", " ", " "]
    + [chr(c) for c in range(0x2000, 0x200B)] + [" ", " ", " ", " ", "　"])
TASK_KEYS = ("type", "instructions", "criteria")
ROW_KEYS = ("text", "label")
ROW_ROLES = ("train", "eval", "shift")


class DataError(ValueError):
    """A refused input; `rule` names the violated rule."""

    def __init__(self, rule, detail):
        super().__init__("REFUSED %s: %s" % (rule, detail))
        self.rule = rule


def normalize(text):
    composed = unicodedata.normalize("NFC", text)
    words, cur = [], []
    for ch in composed:
        if ch in _WHITE_SPACE:
            if cur:
                words.append("".join(cur))
                cur = []
        else:
            cur.append(ch)
    if cur:
        words.append("".join(cur))
    return " ".join(words)


def normalized_sha256(text):
    return sha256_bytes(normalize(text).encode("utf-8"))


def exact_sha256(text):
    return sha256_bytes(text.encode("utf-8"))


# ------------------------------------------------------------------------------------------ task.json

def _pairs(obj, where):
    """object_pairs_hook=list gives objects as lists of (key, value) pairs; refuse anything else."""
    if not (isinstance(obj, list) and all(isinstance(p, tuple) and len(p) == 2 for p in obj)):
        raise DataError("task-schema", "%s must be a JSON object" % where)
    return obj


def load_task(path):
    path = Path(path)
    if not path.is_file():
        raise DataError("task-missing", "%s does not exist" % path)
    try:
        raw = json.loads(path.read_bytes().decode("utf-8"), object_pairs_hook=list)
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise DataError("task-schema", "%s is not UTF-8 JSON (%s)" % (path, e))
    top = _pairs(raw, "task.json")
    keys = [k for k, _ in top]
    if len(set(keys)) != len(keys):
        raise DataError("task-schema", "task.json repeats a key: %s" % keys)
    unknown = sorted(set(keys) - set(TASK_KEYS))
    if unknown:
        raise DataError("task-unknown-key", "task.json has unknown key(s) %s; allowed %s" % (unknown, list(TASK_KEYS)))
    missing = [k for k in TASK_KEYS if k not in keys]
    if missing:
        raise DataError("task-schema", "task.json is missing %s" % missing)
    d = dict(top)
    if d["type"] != "choice":
        raise DataError("task-type", "type must be \"choice\", got %r" % (d["type"],))
    if not isinstance(d["instructions"], str) or not d["instructions"].strip():
        raise DataError("task-schema", "instructions must be a non-empty string")
    crit = _pairs(d["criteria"], "criteria")
    names = [k for k, _ in crit]
    if len(set(names)) != len(names):
        dup = sorted({n for n in names if names.count(n) > 1})
        raise DataError("task-duplicate-criterion", "criterion name(s) %s appear more than once" % dup)
    if len(names) < 2:
        raise DataError("task-too-few-criteria", "a choice task needs at least 2 criteria, got %d" % len(names))
    for name, desc in crit:
        if not name.strip():
            raise DataError("task-schema", "a criterion name is empty")
        if desc is not None and not isinstance(desc, str):
            raise DataError("task-schema", "criterion %r description must be a string or null" % name)
    return {"type": "choice", "instructions": d["instructions"], "criteria": dict(crit), "labels": names}


def laya_question(task):
    """The Laya question dict for a validated task (criteria in document order)."""
    return {"type": task["type"], "instructions": task["instructions"], "criteria": dict(task["criteria"])}


# ------------------------------------------------------------------------------------------ rows

def jsonl_lines(text):
    """The lines of `text` exactly as Rust's `str::lines` (verify.rs parse_rows) yields them: split on
    '\n' ONLY, one '\r' immediately before a '\n' stripped, a final '\n' ending no extra line. Python's
    str.splitlines also breaks on U+0085, U+2028, U+2029, \x0b, \x0c and \x1c-\x1e, so a tweet holding
    one would be one row in Rust and two in Python (WR-05)."""
    parts = text.split("\n")
    tail = parts.pop()                              # after the last '\n': "" or an unterminated line
    out = [p[:-1] if p.endswith("\r") else p for p in parts]
    if tail:
        out.append(tail)                            # an unterminated last line keeps its '\r' (as Rust)
    return out


def decode_jsonl(raw, role, name):
    """UTF-8 bytes -> jsonl_lines; an invalid byte is REFUSED `<role>-row-encoding` (Rust's from_utf8
    refuses the same file), never a UnicodeDecodeError traceback."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        raise DataError("%s-row-encoding" % role, "%s is not valid UTF-8 (byte %d: %s)" % (name, e.start, e.reason))
    return jsonl_lines(text)


def refuse_unencodable(value, role, name, n, field):
    """A JSON string holding a lone surrogate escape (\\ud800) decodes in Python but not in Rust
    (serde_json refuses it), and cannot be written as UTF-8: REFUSED `<role>-row-encoding`."""
    if isinstance(value, str):
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as e:
            raise DataError("%s-row-encoding" % role, "%s line %d %s holds a lone surrogate (U+%04X at %d); it is "
                            "not UTF-8 and Rust refuses the row" % (name, n, field, ord(value[e.start]), e.start))


def load_rows(path, task, role):
    """[(text, label_index)] from a jsonl file; `role` (train | eval | shift) names the file in refusals."""
    if role not in ROW_ROLES:
        raise ValueError("load_rows role %r is not one of %s" % (role, ROW_ROLES))
    path = Path(path)
    if not path.is_file():
        raise DataError("%s-missing" % role, "%s does not exist (%s.jsonl is required)" % (path, role))
    labels = task["labels"]
    rows = []
    for n, line in enumerate(decode_jsonl(path.read_bytes(), role, path.name), 1):
        if not line.strip():
            raise DataError("%s-row-schema" % role, "%s line %d is blank" % (path.name, n))
        try:
            obj = json.loads(line, object_pairs_hook=list)
        except json.JSONDecodeError as e:
            raise DataError("%s-row-schema" % role, "%s line %d is not JSON (%s)" % (path.name, n, e))
        if not (isinstance(obj, list) and all(isinstance(p, tuple) for p in obj)):
            raise DataError("%s-row-schema" % role, "%s line %d is not a JSON object" % (path.name, n))
        keys = [k for k, _ in obj]
        extra = sorted(set(keys) - set(ROW_KEYS))
        if extra:
            raise DataError("%s-row-unknown-key" % role, "%s line %d has unknown key(s) %s" % (path.name, n, extra))
        if sorted(keys) != sorted(ROW_KEYS):
            raise DataError("%s-row-schema" % role, "%s line %d must hold exactly %s" % (path.name, n, list(ROW_KEYS)))
        d = dict(obj)
        refuse_unencodable(d["text"], role, path.name, n, "text")
        refuse_unencodable(d["label"], role, path.name, n, "label")
        if not isinstance(d["text"], str) or not d["text"].strip():
            raise DataError("%s-row-schema" % role, "%s line %d text must be a non-empty string" % (path.name, n))
        if not isinstance(d["label"], str):
            raise DataError("%s-row-label" % role, "%s line %d label must be a criterion NAME, got %r"
                            % (path.name, n, d["label"]))
        if d["label"] not in labels:
            raise DataError("%s-row-label" % role, "%s line %d label %r is not a criterion (%s)"
                            % (path.name, n, d["label"], labels))
        rows.append((d["text"], labels.index(d["label"])))
    if not rows:
        raise DataError("%s-empty" % role, "%s has no rows" % path)
    if role == "eval":
        # The gate's macro-F1 averages over the labels present in y | pred, so a criterion with
        # no eval row lets the margin be decided by which model happens to predict it.
        present = {y for _, y in rows}
        missing = [lab for i, lab in enumerate(labels) if i not in present]
        if missing:
            raise DataError("eval-class-coverage", "%s holds no row of criterion %s: every criterion needs at "
                            "least one eval row for the gate's macro-F1 to compare the same labels"
                            % (path.name, missing))
    return rows


def refuse_overlap(train, ev, role="eval"):
    """An `role` row (eval, or the shift probe) whose normalized text equals a train text is refused."""
    train_hashes = {normalized_sha256(t) for t, _ in train}
    hits = [i for i, (t, _) in enumerate(ev) if normalized_sha256(t) in train_hashes]
    if hits:
        raise DataError("%s-train-overlap" % role, "%d %s row(s) equal a train text after NFC/trim/whitespace "
                        "collapse (%s.jsonl rows %s)" % (len(hits), role, role, hits[:10]))


def in_distribution_heldout(validation, train_pool, shot_ids, excluded_ids, group_member_ids, shots):
    """laya-finetune-gate-v1 `eval_set.demo_rule` (A2, 1.4.0), torch-free and I/O-free.

    `validation` and `train_pool` are [(id, text, label)] in source file order; `shot_ids` the selection's
    ordered_examples ids; `excluded_ids` its exclusions.excluded_train_ids; `group_member_ids` every
    exclusions.groups[*].members entry as a (split, id) pair; `shots` the [(text, label)] shots.
    Validation rows first, then train rows, each in file order; a row is dropped when its id is a shot,
    in excluded_ids, or (split, id) is a group member. A remaining row whose nfc-trim-ws-v1 text equals a
    shot's is REFUSED (DataError heldout-shot-overlap), never silently dropped. Duplicates by normalized
    text are dropped keeping the FIRST occurrence. Returns [(text, label)]."""
    shot_ids, excluded_ids = set(shot_ids), set(excluded_ids)
    members = {(str(sp), str(i)) for sp, i in group_member_ids}
    kept = []
    for split, rows in (("validation", validation), ("train", train_pool)):
        for rid, text, label in rows:
            if rid in shot_ids or rid in excluded_ids or (split, str(rid)) in members:
                continue
            kept.append((split, rid, text, label))
    shot_norm = {normalized_sha256(t) for t, _ in shots}
    hits = [(sp, rid) for sp, rid, t, _ in kept if normalized_sha256(t) in shot_norm]
    if hits:
        raise DataError("heldout-shot-overlap", "%d held-out row(s) equal a shot after NFC/trim/whitespace collapse "
                        "(%s)" % (len(hits), hits[:10]))
    seen, out = set(), []
    for _, _, text, label in kept:
        h = normalized_sha256(text)
        if h not in seen:
            seen.add(h)
            out.append((text, label))
    return out


def group_train(rows):
    """{normalized hash: [row indices]} in first-seen order; conflicting labels refused."""
    groups = {}
    for i, (t, _) in enumerate(rows):
        groups.setdefault(normalized_sha256(t), []).append(i)
    for h, ids in groups.items():
        labs = sorted({rows[i][1] for i in ids})
        if len(labs) > 1:
            raise DataError("train-conflicting-labels", "train.jsonl rows %s carry the same normalized text "
                            "under different labels %s" % (ids, labs))
    return groups


def class_counts(rows, k):
    counts = [0] * k
    for _, y in rows:
        counts[y] += 1
    return counts


def calibration_split(rows, fraction, min_per_class, seed, k):
    """Group-disjoint, stratified, seeded calibration slice of the train rows."""
    groups = group_train(rows)
    counts = class_counts(rows, k)
    rng = np.random.RandomState(int(seed))
    calib = []
    for c in range(k):
        need = max(int(min_per_class), int(math.ceil(float(fraction) * counts[c])))
        cls_groups = [ids for ids in groups.values() if rows[ids[0]][1] == c]
        order = rng.permutation(len(cls_groups))
        took = 0
        for g in order:
            if took >= need:
                break
            calib.extend(cls_groups[g])
            took += len(cls_groups[g])
        if took < need or took >= counts[c]:
            raise DataError("train-class-too-small", "class %d has %d train row(s) in %d text group(s): a "
                            "calibration slice of >= %d row(s) must leave at least one fit row"
                            % (c, counts[c], len(cls_groups), need))
    slice_ids = sorted(calib)
    in_slice = set(slice_ids)
    fit_ids = [i for i in range(len(rows)) if i not in in_slice]
    return fit_ids, slice_ids, slice_ids, slice_ids_sha256(slice_ids)


def slice_ids_sha256(slice_ids):
    return sha256_bytes(json.dumps(list(slice_ids), separators=(",", ":")).encode())


# ------------------------------------------------------------------------------------------ self-test

# The D-19 demo's train.jsonl label layout (16 none, 16 against, 16 favor, 48 distinct texts) and the
# calibration slice both failing demo runs recorded (gate-report calibration.slice_ids / sha256). The
# split reads labels, text GROUPS and the seed only -- never the text itself -- so 48 distinct synthetic
# texts in the same label layout must reproduce the recorded slice exactly. No tweet text is needed.
DEMO_LABEL_LAYOUT = [0] * 16 + [1] * 16 + [2] * 16
DEMO_RECORDED_SLICE = [1, 8, 11, 12, 23, 24, 25, 26, 41, 43, 45, 47]
DEMO_RECORDED_SLICE_SHA256 = "0640137d67666af226d719823648bbfdc2fc6310476ae0fa93d7c6aac37802e4"


def _demo_rule_on_local_data(case):
    """The demo_rule on the REAL pinned splits (FALSIFY-LAYA-GATE-013): prepare_stance's s64 cell must rebuild
    exactly demo_s64.eval_rows rows with eval_class_counts. Local evidence only (the dataset is gitignored):
    absent -> an explicit SKIP line, never a pass. Counts only; no text is printed."""
    import contract
    repo = contract.REPO
    src = repo / "data" / "tweet-eval-stance"
    need = [src / n for n in ("train.jsonl", "validation.jsonl", "test.jsonl")]
    missing = [str(q.relative_to(repo)) for q in need if not q.is_file()]
    if missing:
        print("  SKIP demo_rule on the pinned splits: %s not present (gitignored local dataset)" % ", ".join(missing))
        return
    import prepare_stance                          # torch-free
    decl = contract.gate_contract()["demo_s64"]
    name = "demo_rule on the pinned splits: %s rows, class counts %s" % (decl["eval_rows"], list(decl["eval_class_counts"]))
    try:
        files, summary = prepare_stance.build("s64")
    except SystemExit:                             # prepare_stance.fail() printed PREPARE FAILED: <why> above
        case(name, False, "prepare_stance refused the s64 cell (PREPARE FAILED above)")
        return
    except DataError as e:
        case(name, False, str(e)[:110])
        return
    rows = [json.loads(ln) for ln in decode_jsonl(files["eval.jsonl"], "eval", "eval.jsonl")]
    order = list(decl["criteria_order"])
    got = [sum(1 for r in rows if r["label"] == lab) for lab in order]
    case(name, len(rows) == int(decl["eval_rows"]) and got == [int(x) for x in decl["eval_class_counts"]],
         "%d rows %s, shift %s, eval sha256 %s..." % (len(rows), got, summary["shift"],
                                                     sha256_bytes(files["eval.jsonl"])[:16]))


# The run of record's recipe_id (plan 08-16, models/decide/laya-stance-64, deployed as 24a44d7e...): the
# typed reader must leave every recipe.json byte where it was.
RUN_OF_RECORD_RECIPE_ID = "6a5489afa3462f56"


def _with_contract(path, mutate, fn):
    """fn() with the parsed contract at `path` replaced by a mutated deep copy; always restored."""
    import copy

    import contract
    key = str(path)
    orig = contract.load_yaml(path)
    patched = copy.deepcopy(orig)
    mutate(patched)
    contract._CACHE[key] = patched
    try:
        return fn()
    finally:
        contract._CACHE[key] = orig


def _contract_value_cases(case):
    """contract.number, the ONE typed reader (plan 08-29, WR-09 / V13-d): no contract number is coerced."""
    import contract
    import yaml

    print("contract-value: the typed contract-number reader (contract.number):")

    def accepts(name, fn, want):
        try:
            got = fn()
            case("contract-value " + name, type(got) is type(want) and got == want, "%r" % (got,))
        except contract.ContractValueError as e:
            case("contract-value " + name, False, "refused: %s" % str(e)[:90])

    def refuses(name, fn, key):
        try:
            got = fn()
            case("contract-value " + name, False, "accepted %r" % (got,))
        except contract.ContractValueError as e:
            case("contract-value " + name, e.rule == "contract-value" and str(e).startswith("REFUSED contract-value: ")
                 and key in str(e), str(e)[:100])

    parity = contract.PARITY_CONTRACT

    def set_k(v):
        return lambda p: p["constants"].__setitem__("pack_rescore_noise_k", v)
    accepts("k = 4 (the committed integer) is read as the float 4.0", lambda: contract.noise_policy()[0], 4.0)
    accepts("k = 2.5 stays 2.5 (never int(2.5) = 2)",
            lambda: _with_contract(parity, set_k(2.5), lambda: contract.noise_policy()[0]), 2.5)
    one_e_6 = yaml.safe_load("v: 1e-6")["v"]
    case("contract-value PyYAML reads `1e-6` (no dot) as a string", one_e_6 == "1e-6", "%r" % (one_e_6,))
    refuses('k = "1e-6" (a YAML string) is refused naming the key',
            lambda: _with_contract(parity, set_k(one_e_6), contract.noise_policy),
            "laya-parity-v1 constants.pack_rescore_noise_k")
    refuses("k = true is refused naming the key", lambda: _with_contract(parity, set_k(True), contract.noise_policy),
            "laya-parity-v1 constants.pack_rescore_noise_k")
    gate_c = contract.GATE_CONTRACT
    refuses("ece_bins = 15.0 is refused (an integer is declared; never coerced)",
            lambda: _with_contract(gate_c, lambda c: c["constants"].__setitem__("ece_bins", 15.0), contract.thresholds),
            "constants.ece_bins")
    refuses("seed_policy.rank_scale = 1e4 (a float) is refused",
            lambda: _with_contract(gate_c, lambda c: c["seed_policy"].__setitem__("rank_scale", 1e4),
                                   contract.seed_selection_decl), "seed_policy.rank_scale")
    refuses("recipe.epochs_above_16_max = true is refused",
            lambda: _with_contract(gate_c, lambda c: c["recipe"].__setitem__("epochs_above_16_max", True),
                                   lambda: contract.resolve_epochs("production", 64, 8)), "recipe.epochs_above_16_max")
    refuses("early_stopping.min_delta = \"1e-3\" (a YAML string) is refused",
            lambda: _with_contract(gate_c, lambda c: c["early_stopping"].__setitem__("min_delta", "1e-3"),
                                   contract.early_stopping_decl), "early_stopping.min_delta")
    refuses("check_numbers (train.py's refusal block) refuses a string encoder_lr",
            lambda: _with_contract(gate_c, lambda c: c["recipe"].__setitem__("encoder_lr", "2.5e-5"),
                                   contract.check_numbers), "recipe.encoder_lr")
    try:
        contract.check_numbers()
        case("contract-value the committed contracts pass check_numbers", True)
    except contract.ContractValueError as e:
        case("contract-value the committed contracts pass check_numbers", False, str(e)[:100])
    recipe = contract.recipe_json("production", 64, 12, contract.declared_seed(), contract.production_base_block(),
                                  "early_stopping", contract.seed_selection_decl())
    rid = sha256_bytes(json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    case("contract-value recipe.json of the run of record rebuilds to recipe_id %s..." % RUN_OF_RECORD_RECIPE_ID,
         rid.startswith(RUN_OF_RECORD_RECIPE_ID), rid[:16])


# The back-office sources whose refusal ids the python_refusals table must cover (lifecycle.py is the harness).
REFUSAL_SOURCES = ("contract.py", "data.py", "gate.py", "prepare_stance.py", "train.py")


def _python_refusals_sweep(case):
    """laya-finetune-gate-v1 `python_refusals` (plan 08-29): run every in_process row's hostile input and require
    the named refusal; check every lifecycle row's case exists in lifecycle.py and is called; refuse a rule id in
    the sources with no row, a row with no case, and a case with no row; and run prepare_stance.py as a
    subprocess on an invalid UTF-8 source split (exit 2, REFUSED, no Traceback)."""
    import re
    import subprocess
    import tempfile

    import contract
    import gate
    import prepare_stance

    here = Path(__file__).resolve().parent
    table = contract.gate_contract()["python_refusals"]["rows"]
    print("python_refusals sweep (laya-finetune-gate-v1 python_refusals, %d rows):" % len(table))
    with tempfile.TemporaryDirectory(prefix="laya-refusals-") as tmp:
        tmp = Path(tmp)

        def write(name, body):
            q = tmp / name
            q.write_bytes(body if isinstance(body, bytes) else body.encode("utf-8"))
            return q
        good = {"type": "choice", "instructions": "Which?", "criteria": {"a": None, "b": None}}
        t = load_task(write("task.json", json.dumps(good)))
        rows = [{"text": "sweep row %d" % i, "label": lab} for i, lab in enumerate("abab")]
        body = "".join(json.dumps(r) + "\n" for r in rows)
        bad_utf8 = body.encode("utf-8") + b'{"text": "caf\xe9", "label": "a"}\n'
        surrogate = body + '{"text": "x \\ud800", "label": "a"}\n'
        scale = contract.seed_selection_decl()["rank_scale"]

        def median_report_shipping(shipped):
            per = [{"seed": 13, "macro_f1": 0.6, "ece_post": 0.15, "pass": False},
                   {"seed": 17, "macro_f1": 0.6, "ece_post": 0.12, "pass": False},
                   {"seed": 23, "macro_f1": 0.62, "ece_post": 0.05, "pass": True}]
            top = [r for r in per if r["seed"] == shipped][0]
            rep = gate._fabricated(0.5, top["macro_f1"], top["ece_post"], passed=top["pass"])
            rep["seeds"] = {"policy": contract.seed_selection_decl()["policy"], "shipped": shipped, "per_seed": per}
            return rep

        def nan_stopper():
            st = gate.EarlyStopper(contract.early_stopping_decl(), 3)
            for e in (1, 2, 3):
                st.update(e, float("nan"))
            return st.record(3)

        def write_once_differing():
            out = tmp / "prep-out"
            prepare_stance.write_once(out, {"train.jsonl": b"a\n"})
            return prepare_stance.write_once(out, {"train.jsonl": b"b\n"})
        th = dict(contract.thresholds(), max_ece=0.2)
        cases = {       # (rule, entry) -> fn(role): the hostile input, which must be refused naming the rule
            ("task-missing", "data.load_task"): lambda r: load_task(tmp / "absent.json"),
            ("task-schema", "data.load_task"): lambda r: load_task(write("ts.json", "{not json")),
            ("task-unknown-key", "data.load_task"): lambda r: load_task(write("tu.json", json.dumps(dict(good, labels=[])))),
            ("task-type", "data.load_task"): lambda r: load_task(write("tt.json", json.dumps(dict(good, type="score")))),
            ("task-duplicate-criterion", "data.load_task"):
                lambda r: load_task(write("td.json", '{"type":"choice","instructions":"W","criteria":{"a":null,"a":null}}')),
            ("task-too-few-criteria", "data.load_task"):
                lambda r: load_task(write("tf.json", json.dumps(dict(good, criteria={"a": None})))),
            ("<role>-missing", "data.load_rows"): lambda r: load_rows(tmp / ("absent-%s.jsonl" % r), t, r),
            ("<role>-row-encoding", "data.load_rows"):
                lambda r: load_rows(write("enc-%s.jsonl" % r, surrogate if r == "shift" else bad_utf8), t, r),
            ("<role>-row-schema", "data.load_rows"): lambda r: load_rows(write("sch-%s.jsonl" % r, body + "\n" + body), t, r),
            ("<role>-row-unknown-key", "data.load_rows"):
                lambda r: load_rows(write("uk-%s.jsonl" % r, body + '{"text": "x", "label": "a", "id": 1}\n'), t, r),
            ("<role>-row-label", "data.load_rows"):
                lambda r: load_rows(write("lab-%s.jsonl" % r, body + '{"text": "x", "label": "zzz"}\n'), t, r),
            ("<role>-empty", "data.load_rows"): lambda r: load_rows(write("empty-%s.jsonl" % r, b""), t, r),
            ("eval-class-coverage", "data.load_rows"):
                lambda r: load_rows(write("cov.jsonl", '{"text": "only a", "label": "a"}\n'), t, "eval"),
            ("<role>-train-overlap", "data.refuse_overlap"):
                lambda r: refuse_overlap([("sweep row 0", 0)], [(" sweep  row 0", 0)], r),
            ("heldout-shot-overlap", "data.in_distribution_heldout"):
                lambda r: in_distribution_heldout([("validation:0", "  shot   x ", "a")], [], [], [], [], [("shot x", "a")]),
            ("train-conflicting-labels", "data.group_train"): lambda r: group_train([("same", 0), (" same ", 1)]),
            ("train-class-too-small", "data.calibration_split"):
                lambda r: calibration_split([("a%d" % i, 0) for i in range(8)] + [("b0", 1), ("b1", 1)], 0.25, 2, 13, 2),
            ("contract-value", "contract.number"): lambda r: contract.number("1e-6", "sweep.k", "float"),
            ("epochs", "contract.resolve_epochs"): lambda r: contract.resolve_epochs("production", 16, 12),
            ("variant", "contract.resolve_epochs"): lambda r: contract.resolve_epochs("staging", 16, None),
            ("stopping", "contract.resolve_stopping"): lambda r: contract.resolve_stopping("never"),
            ("seeds", "contract.resolve_seeds"): lambda r: contract.resolve_seeds(1, "production"),
            ("seeds", "gate.select_median_seed"):
                lambda r: gate.select_median_seed([{"seed": 13, "ece_post": 0.1}, {"seed": 17, "ece_post": 0.2}], scale),
            ("seeds", "gate.verify_report"): lambda r: gate.verify_report(median_report_shipping(23)),
            ("thresholds", "gate.verify_report"): lambda r: gate.verify_report(gate._fabricated(0.5, 0.9, 0.01, th)),
            ("pass", "gate.verify_report"): lambda r: gate.verify_report(gate._fabricated(0.5, 0.9, 0.5, passed=True)),
            ("early-stopping", "gate.EarlyStopper.record"): lambda r: nan_stopper(),
            ("source-row-encoding", "prepare_stance.read_jsonl"): lambda r: prepare_stance.read_jsonl(write("src.jsonl", bad_utf8)),
            ("source-row-schema", "prepare_stance.read_jsonl"): lambda r: prepare_stance.read_jsonl(write("srcs.jsonl", "[1, 2]\n")),
            ("prepare-out-dir", "prepare_stance.write_once"): lambda r: write_once_differing(),
        }
        life_src = (here / "lifecycle.py").read_text(encoding="utf-8")
        swept, life, used = 0, 0, set()
        for row in table:
            key = (row["rule"], row["entry"])
            for role in (row.get("roles") or [None]):
                rid = row["rule"].replace("<role>", role) if role else row["rule"]
                name = "python_refusals %s via %s" % (rid, row["entry"])
                if row.get("sweep") == "lifecycle":
                    fn = str(row.get("case", ""))
                    ok = bool(fn) and ("def %s(" % fn) in life_src and life_src.count("%s(" % fn) >= 2
                    case(name + " [lifecycle %s]" % fn, ok, "" if ok else "lifecycle.py does not define and call %r" % fn)
                    life += 1
                    continue
                if row.get("sweep") != "in_process":
                    case(name, False, "unknown sweep %r" % (row.get("sweep"),))
                    continue
                if key not in cases:
                    case(name, False, "an in_process row with no sweep case")
                    continue
                used.add(key)
                try:
                    cases[key](role)
                    case(name, False, "accepted: %s" % row["hostile_input"][:60])
                except SystemExit as e:
                    case(name, False, "exited %s instead of raising REFUSED %s" % (e.code, rid))
                except Exception as e:
                    msg = str(e)
                    ok = msg.startswith("REFUSED %s:" % rid) and getattr(e, "rule", rid) == rid
                    case(name, ok, msg[:90] if ok else "raised %s: %s" % (type(e).__name__, msg[:80]))
                swept += 1
        for key in sorted(set(cases) - used):
            case("python_refusals sweep case %s via %s has a table row" % key, False, "a case with no row (unknown rule)")
        in_table = {row["rule"] for row in table}
        found = set()
        for src in REFUSAL_SOURCES:
            text = (here / src).read_text(encoding="utf-8")
            found |= {m.replace("%s", "<role>") for m in re.findall(r'DataError\(\s*"([^"]+)"', text)}
            found |= set(re.findall(r'"REFUSED ([a-z][a-z-]*):', text))
        missing = sorted(found - in_table)
        case("python_refusals lists every rule id the back-office sources raise (%d found)" % len(found), not missing,
             "no row for %s" % missing if missing else "")

        src = tmp / "cli-src"
        src.mkdir()
        (src / "train.jsonl").write_bytes(b'{"id": "train:0", "input": "caf\xe9", "label": 0}\n')
        (src / "test.jsonl").write_bytes(b'{"id": "test:0", "input": "fine", "label": 0}\n')
        out = tmp / "cli-out"
        proc = subprocess.run([sys.executable, str(here / "prepare_stance.py"), "--cell", "s16", "--src", str(src),
                               "--out", str(out)], capture_output=True, text=True)
        both = proc.stdout + proc.stderr
        case("python_refusals CLI boundary: prepare_stance.py on an invalid UTF-8 split -> exit 2, REFUSED, no traceback",
             proc.returncode == 2 and "REFUSED source-row-encoding:" in proc.stderr and "Traceback" not in both
             and not out.exists(), "exit %d: %s" % (proc.returncode, proc.stderr.strip()[:80]))
    print("PYTHON REFUSALS swept=%d lifecycle=%d" % (swept, life))


def selftest():
    """Every data refusal over temporary files, plus split determinism / stratification / text-disjointness
    and the recipe epoch rule. numpy + pyyaml only (no torch)."""
    import tempfile

    import contract

    failures = []

    def case(name, ok, detail=""):
        print("  %-4s %-68s %s" % ("ok" if ok else "FAIL", name, detail))
        if not ok:
            failures.append(name)

    def expect(name, rule, fn):
        try:
            fn()
            case(name, False, "accepted")
        except DataError as e:
            case(name, e.rule == rule, str(e)[:110])
        except Exception as e:                        # a traceback-class escape is a FAIL, never a crash
            case(name, False, "raised %s, not a DataError %s: %s" % (type(e).__name__, rule, str(e)[:70]))

    frac = contract.constant("calibration_slice_fraction", "float")
    min_pc = contract.constant("calibration_slice_min_per_class", "int")
    good_task = {"type": "choice", "instructions": "Which?", "criteria": {"a": "first", "b": None, "c": "third"}}

    with tempfile.TemporaryDirectory(prefix="laya-data-selftest-") as tmp:
        tmp = Path(tmp)

        def write(name, text):
            path = tmp / name
            path.write_text(text, encoding="utf-8")
            return path

        def rows_text(rows):
            return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)

        print("task.json refusals:")
        expect('type "score"', "task-type",
               lambda: load_task(write("t1.json", json.dumps(dict(good_task, type="score")))))
        expect("one criterion", "task-too-few-criteria",
               lambda: load_task(write("t2.json", json.dumps(dict(good_task, criteria={"a": None})))))
        expect("duplicate criterion", "task-duplicate-criterion",
               lambda: load_task(write("t3.json", '{"type":"choice","instructions":"Which?",'
                                                  '"criteria":{"a":null,"b":null,"a":"again"}}')))
        expect("unknown task key", "task-unknown-key",
               lambda: load_task(write("t4.json", json.dumps(dict(good_task, labels=["a"])))))
        expect("task.json missing", "task-missing", lambda: load_task(tmp / "absent.json"))
        t = load_task(write("task.json", json.dumps(good_task)))
        case("criteria document order is the label index", t["labels"] == ["a", "b", "c"])
        t_rev = load_task(write("task_rev.json", '{"type":"choice","instructions":"Which?",'
                                                  '"criteria":{"c":null,"a":null,"b":null}}'))
        case("document order kept (not sorted)", t_rev["labels"] == ["c", "a", "b"])

        print("row refusals:")
        base_rows = [{"text": "row %d of class %s" % (i, lab), "label": lab} for lab in "abc" for i in range(4)]
        expect("train label not a criterion", "train-row-label",
               lambda: load_rows(write("r1.jsonl", rows_text(base_rows + [{"text": "x", "label": "zzz"}])), t, "train"))
        expect("train label given as an index", "train-row-label",
               lambda: load_rows(write("r2.jsonl", rows_text(base_rows + [{"text": "x", "label": 0}])), t, "train"))
        expect("train row with an extra key", "train-row-unknown-key",
               lambda: load_rows(write("r3.jsonl", rows_text(base_rows + [{"text": "x", "label": "a", "id": 1}])),
                                 t, "train"))
        expect("eval.jsonl missing", "eval-missing", lambda: load_rows(tmp / "eval.jsonl", t, "eval"))
        expect("eval.jsonl with no row of one criterion", "eval-class-coverage",
               lambda: load_rows(write("e1.jsonl", rows_text([r for r in base_rows if r["label"] != "c"])), t, "eval"))
        case("a shift file may miss a criterion (only eval is gated)",
             len(load_rows(write("s1.jsonl", rows_text([r for r in base_rows if r["label"] != "c"])), t, "shift")) == 8)
        train = load_rows(write("train.jsonl", rows_text(base_rows)), t, "train")
        case("valid train.jsonl loads as (text, label index)", len(train) == 12 and train[4][1] == 1)
        expect("eval text equal to a train text after NFC/trim/whitespace collapse", "eval-train-overlap",
               lambda: refuse_overlap(train, [("a fresh eval row", 1), ("row 0 of\tclass  a ", 0)]))
        refuse_overlap(train, [("  Row 0 OF class a", 0)])    # case differs: no casefolding, not an overlap
        case("case-different eval text is not an overlap", True)
        expect("two train rows, same normalized text, different labels", "train-conflicting-labels",
               lambda: group_train(train + [(" row 0 of  class a", 1)]))
        too_small = [r for r in train if r[1] != 2] + [("only c %d" % i, 2) for i in range(min_pc)]
        expect("a class with only calibration_slice_min_per_class shots", "train-class-too-small",
               lambda: calibration_split(too_small, frac, min_pc, 13, 3))
        just_enough = [r for r in train if r[1] != 2] + [("enough c %d" % i, 2) for i in range(min_pc + 1)]
        fit, sl, _, _ = calibration_split(just_enough, frac, min_pc, 13, 3)
        case("a class with calibration_slice_min_per_class + 1 shots is accepted",
             class_counts([just_enough[i] for i in fit], 3)[2] >= 1)

        print("row splitting (Rust str::lines: '\\n' only, one '\\r' before it stripped) and row-encoding refusals:")

        def loads(name, fn, check):
            try:
                got = fn()
                case(name, check(got), "%d rows" % len(got))
            except Exception as e:
                case(name, False, "raised %s: %s" % (type(e).__name__, str(e)[:80]))
        for cp in ("\u0085", "\u2028", "\u2029"):
            one = {"text": "one tweet%swith U+%04X inside" % (cp, ord(cp)), "label": "a"}
            loads("row-split: a text holding U+%04X (ensure_ascii=False) is ONE row, as in Rust" % ord(cp),
                  lambda one=one, cp=cp: load_rows(write("sep-%04x.jsonl" % ord(cp), rows_text(base_rows + [one])), t, "train"),
                  lambda got, one=one: len(got) == 13 and got[-1] == (one["text"], 0))
        crlf = tmp / "crlf.jsonl"
        crlf.write_bytes(rows_text(base_rows).replace("\n", "\r\n").encode("utf-8"))
        loads("row-split: a CRLF file loads with the '\\r' stripped (the LF rows exactly)", lambda: load_rows(crlf, t, "train"),
              lambda got: got == load_rows(write("lf.jsonl", rows_text(base_rows)), t, "train"))
        rust_lines = (("a\nb", ["a", "b"]), ("a\r\nb\r\n", ["a", "b"]), ("a\n\n", ["a", ""]), ("a\r", ["a\r"]),
                      ("a\rb\n", ["a\rb"]), ("", []), ("\n", [""]), ("x\u2028y\u0085z\n", ["x\u2028y\u0085z"]),
                      ("a\n\r\n", ["a", ""]))
        split = globals().get("jsonl_lines")
        bad = [src for src, want in rust_lines if split is None or split(src) != want]
        case("row-split: jsonl_lines == Rust str::lines on the rustc 1.98 case table (%d cases)" % len(rust_lines),
             not bad, "differs on %r" % (bad,) if bad else "")
        bad_utf8 = tmp / "bad-utf8.jsonl"
        bad_utf8.write_bytes(rows_text(base_rows[:3]).encode("utf-8") + b'{"text": "caf\xe9", "label": "a"}\n')
        expect("train.jsonl with an invalid UTF-8 byte -> train-row-encoding (not UnicodeDecodeError)",
               "train-row-encoding", lambda: load_rows(bad_utf8, t, "train"))
        expect("eval.jsonl with an invalid UTF-8 byte -> eval-row-encoding", "eval-row-encoding",
               lambda: load_rows(bad_utf8, t, "eval"))
        expect("a text holding a lone surrogate escape (\\ud800) -> train-row-encoding", "train-row-encoding",
               lambda: load_rows(write("sur1.jsonl", rows_text(base_rows) + '{"text": "bad \\ud800 text", "label": "a"}\n'),
                                 t, "train"))
        expect("a label holding a lone surrogate escape (\\udc00) -> shift-row-encoding", "shift-row-encoding",
               lambda: load_rows(write("sur2.jsonl", rows_text(base_rows) + '{"text": "fine", "label": "\\udc00"}\n'),
                                 t, "shift"))

        print("prepare_stance.write_once (a re-run into an existing out dir):")
        import prepare_stance                      # torch-free
        files = {"task.json": b"{}\n", "train.jsonl": b"row\n"}
        prep = tmp / "prep"
        prepare_stance.write_once(prep, files)
        (prep / ".DS_Store").write_bytes(b"\x00\x01")

        def rerun(name, fn, want_rule):
            try:
                status = fn()
                case(name, want_rule is None and str(status).startswith("unchanged"), "%s" % status)
            except DataError as e:
                case(name, e.rule == want_rule, str(e)[:100])
            except SystemExit as e:
                case(name, False, "exited %s instead of a %s" % (e.code, "re-run" if want_rule is None else "REFUSED " + want_rule))
            except Exception as e:
                case(name, False, "raised %s: %s" % (type(e).__name__, str(e)[:70]))
        rerun("a byte-identical re-run beside a .DS_Store succeeds (dotfiles are not compared)",
              lambda: prepare_stance.write_once(prep, files), None)
        rerun("a written file that differs is still refused (prepare-out-dir)",
              lambda: prepare_stance.write_once(prep, dict(files, **{"train.jsonl": b"other\n"})), "prepare-out-dir")
        (prep / "shift.jsonl").write_bytes(b"stale\n")
        rerun("a foreign non-dot file (a stale shift.jsonl train.py would read) is still refused",
              lambda: prepare_stance.write_once(prep, files), "prepare-out-dir")

        print("NFC / whitespace normalization (nfc-trim-ws-v1, the Rust split_whitespace set):")
        case("NFC composes e + U+0301", normalize("é") == "é")
        case("Unicode White_Space collapses (U+3000, U+0085, runs)", normalize("　a  b\u0085") == "a b")
        case("U+001F is NOT whitespace (unlike str.split)", normalize("a\u001fb") == "a\u001fb")

    print("calibration_split:")
    rows = [("shot %d" % i, y) for i, y in enumerate(DEMO_LABEL_LAYOUT)]
    a, b = calibration_split(rows, frac, min_pc, 13, 3), calibration_split(rows, frac, min_pc, 13, 3)
    case("deterministic for a seed (slice_ids and sha)", a == b, "sha=%s..." % a[3][:16])
    counts = class_counts([rows[i] for i in a[2]], 3)
    case("stratified: ceil(0.25 * 16) = 4 per class", counts == [4, 4, 4], str(counts))
    case("slice_ids sorted, unique, disjoint from fit, together cover train",
         a[2] == sorted(set(a[2])) and not set(a[0]) & set(a[2]) and sorted(a[0] + a[2]) == list(range(48)))
    case("slice_ids_sha256 is sha256 of the compact JSON array",
         a[3] == sha256_bytes(json.dumps(a[2], separators=(",", ":")).encode()))
    case("reproduces BOTH failing demo runs' recorded slice (seed 13, 16/16/16 layout)",
         a[2] == DEMO_RECORDED_SLICE and a[3] == DEMO_RECORDED_SLICE_SHA256, str(a[2]))
    case("a different seed changes the slice", calibration_split(rows, frac, min_pc, 17, 3)[2] != a[2])
    dup = rows + [("  shot   0 ", 0), ("shot 0", 0)]          # "shot 0" three times after normalization
    group = [i for i, (txt, _) in enumerate(dup) if normalized_sha256(txt) == normalized_sha256("shot 0")]
    sides, in_slice = set(), 0
    for seed in range(50):
        fit, _, sl, _ = calibration_split(dup, frac, min_pc, seed, 3)
        side = {i in set(sl) for i in group}
        sides.add(len(side))
        in_slice += int(True in side)
    case("a text duplicated 3x lands wholly on one side, over 50 seeds",
         len(group) == 3 and sides == {1} and 0 < in_slice < 50, "group in slice for %d/50 seeds" % in_slice)

    print("in-distribution held-out rule (eval_set.demo_rule, A2, FALSIFY-LAYA-GATE-013) on a synthetic manifest:")
    shots = [("shot a", "none"), ("shot b", "against")]
    shot_ids = ["train:0", "train:1"]
    excluded = ["train:3"]
    members = [("train", "train:3"), ("validation", "validation:3")]      # the real manifest's group shape
    val = [("validation:0", "val zero", "none"), ("validation:1", "val one", "against"),
           ("validation:2", "dup  text", "favor"), ("validation:3", "group partner", "against")]
    pool = [("train:0", "shot a", "none"), ("train:1", "shot b", "against"), ("train:2", " dup text ", "none"),
            ("train:3", "excluded row", "favor"), ("train:4", "train four", "favor"), ("train:5", "train five", "none")]
    held = in_distribution_heldout(val, pool, shot_ids, excluded, members, shots)
    texts = [normalize(t) for t, _ in held]
    case("a shot's row is dropped (by id), never re-used as eval", "shot a" not in texts and "shot b" not in texts)
    case("an excluded_train_ids row is dropped", "excluded row" not in texts)
    case("a validation row that is an exclusion-group member is dropped", "group partner" not in texts)
    case("a duplicate by normalized text is kept once, the validation copy (label) winning",
         texts.count("dup text") == 1 and dict((normalize(t), lab) for t, lab in held)["dup text"] == "favor")
    case("order: validation rows first, then train rows, each in file order",
         texts == ["val zero", "val one", "dup text", "train four", "train five"], str(texts))
    labs = ["none", "against", "favor"]
    case("the resulting class counts", [sum(1 for _, x in held if x == lab) for lab in labs] == [2, 1, 2])
    expect("a held-out row whose normalized text equals a shot is REFUSED, not dropped", "heldout-shot-overlap",
           lambda: in_distribution_heldout(val + [("validation:9", "  Shot  b", "none"), ("validation:8", " shot\tb ", "none")],
                                           pool, shot_ids, excluded, members, shots))
    _demo_rule_on_local_data(case)

    _contract_value_cases(case)
    _python_refusals_sweep(case)

    print("recipe epoch rule (contract.resolve_epochs):")

    def epochs(name, variant, spc, arg, want):
        try:
            got = contract.resolve_epochs(variant, spc, arg)
            case(name, got == want, "epochs=%s" % got)
        except contract.RecipeError as e:
            case(name, want is None, str(e)[:110])
    epochs("16/class, no --epochs -> 12", "production", 16, None, 12)
    epochs("16/class, --epochs 12 refused (fixed, not ignored)", "production", 16, 12, None)
    epochs("64/class, --epochs 8 -> 8", "production", 64, 8, 8)
    epochs("64/class, --epochs 3 refused (< 4)", "production", 64, 3, None)
    epochs("64/class, --epochs 13 refused (> 12)", "production", 64, 13, None)
    epochs("64/class, no --epochs refused", "production", 64, None, None)
    epochs("synthetic-fixture, --epochs 1 -> 1", "synthetic-fixture", 4, 1, 1)

    if failures:
        print("DATA SELFTEST FAILED: %s" % ", ".join(failures))
        return 1
    print("DATA SELFTEST OK")
    return 0


if __name__ == "__main__":
    # One DataError class: prepare_stance / gate `import data`, which must be THIS module, not a second copy
    # whose DataError the self-test's `except DataError` would not catch.
    sys.modules.setdefault("data", sys.modules[__name__])
    if sys.argv[1:] == ["--selftest"]:
        sys.exit(selftest())
    print("usage: python data.py --selftest", file=sys.stderr)
    sys.exit(2)
