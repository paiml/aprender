"""The tracer's thin slice: a REAL train -> F16 save -> complete dir -> reload -> calibrate -> gate lifecycle
on the committed tiny Laya checkpoint, in seconds on CPU (just laya-train-lifecycle).

    uv run --project scripts/laya_train --frozen python scripts/laya_train/lifecycle.py

Runs train.py as a subprocess (the exact CLI a user runs) with --variant synthetic-fixture on
crates/aprender-decide/tests/fixtures/laya_tiny/{checkpoint,data} into a fresh temp dir (deleted
afterwards), then asserts:

  * the out dir holds every laya-finetune-gate-v1 run_dir_layout file;
  * CHECKPOINT COMPLETE was logged before SCORING START (rl_agent_config.json existed before the
    first reload), and RECIPE WRITTEN before both;
  * the reload left every checkpoint sha256 unchanged (train.py refuses otherwise; re-checked here by
    reloading once more and hashing);
  * recipe.json is variant synthetic-fixture with the tiny checkpoint's recorded sha256, and its
    sha256 is the gate report's recipe_id; the gate report parses in the contract schema;
  * the digest MAPPING is Laya's real API: the tiny base loads with {"model.safetensors": <its sha>}
    and a wrong sha is refused by Laya's own ValueError;
  * forcing --device cpu is recorded as device_used "cpu" / device_is_cpu true, and eval-probs.json equals
    a SEPARATE F16 reload's re-score of every eval row within laya-parity-v1 pack_rescore_probs_abs;
  * BOTH declared stopping rules run: early_stopping (the contract default, max 3 epochs here) logs
    EPOCH lines and a STOP line before CHECKPOINT COMPLETE, its recipe.json carries the contract's
    `early_stopping` object and its rl_agent_config.json `training.stopping` record names a best epoch
    in [first_candidate_epoch, epochs_run]; fixed_epochs (1 epoch) logs no STOP and its recipe.json has
    no `early_stopping` key (the 1.0.0 bytes); in both, mtime recipe.json <= checkpoint/model.safetensors
    <= eval-probs.json (no eval probability before the checkpoint is fixed);
  * the seed policy (FALSIFY-LAYA-GATE-006 / -011): the single-seed runs keep the LEGACY rule (no
    `seed_selection`, label "single seed", seed 13 ships); a `--seeds 3` run (fixed_epochs, 1 epoch)
    trains 13, 17, 23 in seeds/seed-<s>/, ships gate.select_median_seed of its own per_seed rows, keeps
    exactly ONE model.safetensors (at checkpoint/, hashing to the shipped row), binds every
    seeds/seed-<s>/eval-probs.json, judges the gate on the shipped seed, carries the contract's
    `seed_selection` in recipe.json, trains seed 13 bit-identically to the single-seed run, and -- when
    13 is the median -- ships that run's checkpoint tree (modulo the recipe_id provenance field);
    `--seeds 0 / 2 / 4` (synthetic) and `--variant production --seeds 1` are refused (exit 2) before
    anything is written;
  * the A1 float64 noise record on EVERY run: schema, k and floor equal laya-parity-v1, control 0.0,
    one row per eval row in order, max_abs recomputed from the written files bit for bit, bound ==
    max(floor, k x max_abs), argmax agreement recomputed, sha256 equal to the gate report's.

  * the A2 shift probe: the same three-seed run with a shift.jsonl added to its data dir decides an
    IDENTICAL gate (pass, margin, zero_shot, fine_tuned, calibration, seeds), carries `shift_probe`
    (gate_clause false) and the two hash-bound shift files; a shift row equal to a train row is refused.

With LAYA_LIFECYCLE_KEEP=<dir>, the three-seed SHIFT run dir (every 1.4.0 file, shift probe included) is
copied to <dir>/run and its data dir to <dir>/data (plan 08-15's cross-language reader test parses
exactly what this writer wrote).

The gate outcome itself is not asserted (a 1-3 epoch tiny model is not expected to pass): exit 0 or 3.
Prints LIFECYCLE OK.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import contract  # noqa: E402
from common import sha256_bytes, sha256_file, tree_sha256  # noqa: E402
from data import jsonl_lines  # noqa: E402  (rows split as Rust's str::lines)

REPO = contract.REPO
TINY = REPO / "crates" / "aprender-decide" / "tests" / "fixtures" / "laya_tiny"


def fail(msg):
    print("LIFECYCLE FAILED: " + msg, file=sys.stderr, flush=True)
    sys.exit(1)


def main():
    tiny_sha = json.loads((TINY / "recipe.json").read_text())["base"]["sha256"]
    if sha256_file(TINY / "checkpoint" / "model.safetensors") != tiny_sha:
        fail("the tiny checkpoint's model.safetensors does not match its recorded sha256")

    # The digest mapping against the REAL API (T-08-08-01): right sha loads, wrong sha is refused.
    from laya import Agent
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        Agent(str(TINY / "checkpoint"), device="cpu", expected_sha256={"model.safetensors": tiny_sha})
        wrong = ("0" if tiny_sha[0] != "0" else "1") + tiny_sha[1:]
        try:
            Agent(str(TINY / "checkpoint"), device="cpu", expected_sha256={"model.safetensors": wrong})
            fail("Laya loaded the tiny base under a WRONG sha256")
        except ValueError as e:
            if "SHA-256 mismatch" not in str(e):
                fail("wrong-sha load raised an unexpected ValueError: %s" % e)
    print("digest mapping: right sha loads, wrong sha refused by Laya (ValueError: SHA-256 mismatch)")
    check_early_stopping_refusal()

    tmp = Path(tempfile.mkdtemp(prefix="laya-lifecycle-"))
    try:
        for stopping, epochs in (("early_stopping", 3), ("fixed_epochs", 1)):
            run_one(tmp / stopping, tiny_sha, stopping, epochs, Agent)
        check_seed_refusals(tmp, tiny_sha)
        check_train_cli_refusals(tmp, tiny_sha)
        check_data_changed_refusal(tmp)
        check_noise_control_refusal(tmp, tmp / "fixed_epochs", tiny_sha)
        single = tmp / "fixed_epochs"
        multi = tmp / "fixed_epochs-seeds3"
        run_one(multi, tiny_sha, "fixed_epochs", 1, Agent, seeds=3)
        check_median_run(single, multi)
        shift_run, shift_data = check_shift_probe(tmp, tiny_sha, multi, Agent)
        keep_run(shift_run, shift_data)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("LIFECYCLE OK")


def _refused(proc, rule, what):
    """A train.py subprocess refused `rule`: exit 2, a `REFUSED <rule>:` line on stderr, no traceback."""
    if proc.returncode != 2 or ("REFUSED %s:" % rule) not in proc.stderr or "Traceback" in proc.stderr:
        fail("%s was not REFUSED %s with exit 2 and no traceback (exit %d): %s"
             % (what, rule, proc.returncode, proc.stderr[-300:]))


def check_train_cli_refusals(tmp, tiny_sha):
    """python_refusals rows `base`, `out-dir` and the train.py CLI boundary of `<role>-row-encoding`
    (laya-finetune-gate-v1 python_refusals, plan 08-29): each refused with exit 2, one REFUSED line, no
    traceback, before any model loads and without writing into the out dir."""
    run = lambda cmd: subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ))  # noqa: E731
    out = tmp / "refused-production-base"
    proc = run([sys.executable, str(HERE / "train.py"), "--data", str(TINY / "data"), "--out", str(out),
                "--variant", "production", "--base", str(TINY / "checkpoint"), "--base-sha256", tiny_sha,
                "--stopping", "fixed_epochs", "--device", "cpu"])
    _refused(proc, "base", "--variant production with --base")
    out = tmp / "refused-synthetic-no-base"
    proc = run([sys.executable, str(HERE / "train.py"), "--data", str(TINY / "data"), "--out", str(out),
                "--variant", "synthetic-fixture", "--epochs", "1", "--stopping", "fixed_epochs", "--device", "cpu"])
    _refused(proc, "base", "--variant synthetic-fixture without --base")
    out = tmp / "refused-out-dir"
    out.mkdir()
    (out / "already-here.txt").write_text("x")
    _refused(run(train_cmd(out, tiny_sha, "fixed_epochs", 1)), "out-dir", "a non-empty --out")
    if sorted(q.name for q in out.iterdir()) != ["already-here.txt"]:
        fail("REFUSED out-dir wrote into the out dir: %s" % sorted(q.name for q in out.iterdir()))
    bad = tmp / "data-bad-utf8"
    shutil.copytree(TINY / "data", bad)
    (bad / "train.jsonl").write_bytes((bad / "train.jsonl").read_bytes() + b'{"text": "caf\xe9", "label": "a"}\n')
    out = tmp / "refused-row-encoding"
    _refused(run(train_cmd(out, tiny_sha, "fixed_epochs", 1, data_dir=bad)), "train-row-encoding",
             "a train.jsonl with an invalid UTF-8 byte")
    if out.exists():
        fail("REFUSED train-row-encoding created the out dir")
    print("train.py CLI refusals: base (production --base, synthetic without --base), out-dir (nothing written), "
          "train-row-encoding (invalid UTF-8) -- each exit 2, one REFUSED line, no traceback")


def check_data_changed_refusal(tmp):
    """python_refusals row `data-changed`: train.refuse_if_data_changed passes an untouched data dir and refuses
    (exit 2, REFUSED data-changed) one whose eval.jsonl changed, and one that gained a shift.jsonl."""
    import contextlib
    import io

    import train
    d = tmp / "data-changed"
    shutil.copytree(TINY / "data", d)
    bound = {n: sha256_file(d / n) for n in ("task.json", "train.jsonl", "eval.jsonl")}
    train.refuse_if_data_changed(d, bound)                 # unchanged: returns
    for what, mutate in (("eval.jsonl edited", lambda: (d / "eval.jsonl").write_bytes((d / "eval.jsonl").read_bytes() + b"\n")),
                         ("a shift.jsonl added", lambda: (d / "shift.jsonl").write_bytes(b"x\n"))):
        shutil.rmtree(d)
        shutil.copytree(TINY / "data", d)
        mutate()
        err = io.StringIO()
        try:
            with contextlib.redirect_stderr(err):
                train.refuse_if_data_changed(d, bound)
            fail("data-changed: %s was accepted" % what)
        except SystemExit as e:
            if e.code != 2 or not err.getvalue().startswith("REFUSED data-changed: "):
                fail("data-changed: %s -> exit %r, stderr %r" % (what, e.code, err.getvalue()[:200]))
    print("data-changed refusal: an edited eval.jsonl and an added shift.jsonl -> REFUSED data-changed, exit 2")


def check_noise_control_refusal(tmp, run_dir, tiny_sha):
    """python_refusals row `noise-control`: when the manual fp32 forward does not reproduce the Scorer's logits
    (rescore_noise_set returns no record), train.write_noise_record refuses (exit 2, REFUSED noise-control) and
    writes no rescore-noise.json. The mismatch is injected by replacing rescore_noise_set for this one call."""
    import argparse
    import contextlib
    import io

    import data
    import train
    out = tmp / "noise-control"
    shutil.copytree(run_dir, out)
    (out / "rescore-noise.json").unlink()
    task = data.load_task(out / "task.json")
    eval_rows = data.load_rows(TINY / "data" / "eval.jsonl", task, "eval")
    base = train.Base(argparse.Namespace(variant="synthetic-fixture", base=str(TINY / "checkpoint"),
                                         base_sha256=tiny_sha))
    real = train.rescore_noise_set
    train.rescore_noise_set = lambda *a, **k: (None, 0.5, [0])
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            train.write_noise_record(out, base, eval_rows, data.laya_question(task), task["labels"])
        fail("noise-control: a non-zero control was accepted and a record written")
    except SystemExit as e:
        if e.code != 2 or not err.getvalue().startswith("REFUSED noise-control: "):
            fail("noise-control: exit %r, stderr %r" % (e.code, err.getvalue()[:200]))
    finally:
        train.rescore_noise_set = real
    if (out / "rescore-noise.json").exists():
        fail("noise-control: rescore-noise.json was written despite the refusal")
    print("noise-control refusal: a control mismatch -> REFUSED noise-control, exit 2, no rescore-noise.json")


def check_early_stopping_refusal():
    """train.py's boundary for EarlyStopper's typed refusal (plan 08-29, IN-08): an early-stopping run whose
    calibration monitor was never finite ends as ONE `REFUSED early-stopping: ...` line on stderr and exit 2
    (train.stopping_record_or_refuse), never a ValueError traceback hours into a run."""
    import contextlib
    import io

    import gate
    import train
    st = gate.EarlyStopper(contract.early_stopping_decl(), 3)
    for e in (1, 2, 3):
        st.update(e, float("nan"))
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            train.stopping_record_or_refuse(st, 3)
    except SystemExit as e:
        if e.code != 2 or not err.getvalue().startswith("REFUSED early-stopping: ") or "Traceback" in err.getvalue():
            fail("early-stopping with no finite monitor: exit %r, stderr %r" % (e.code, err.getvalue()[:200]))
    except Exception as e:  # noqa: BLE001 -- the defect this check exists for
        fail("early-stopping with no finite monitor raised %s instead of REFUSED early-stopping, exit 2: %s"
             % (type(e).__name__, e))
    else:
        fail("early-stopping with no finite monitor was accepted (a record with nothing to restore)")
    print("early-stopping refusal: no finite calibration monitor -> REFUSED early-stopping, exit 2 (train.py boundary)")


def keep_run(run_dir, data_dir):
    """LAYA_LIFECYCLE_KEEP=<dir>: copy a three-seed run dir to <dir>/run and its data dir to <dir>/data."""
    keep = os.environ.get("LAYA_LIFECYCLE_KEEP")
    if not keep:
        return
    keep = Path(keep)
    keep.mkdir(parents=True, exist_ok=True)
    for src, name in ((run_dir, "run"), (data_dir, "data")):
        if (keep / name).exists():
            fail("LAYA_LIFECYCLE_KEEP %s already holds %s; a kept run is written once" % (keep, name))
        shutil.copytree(src, keep / name)
    print("kept: %s -> %s/run, %s -> %s/data" % (run_dir.name, keep, data_dir.name, keep))


def train_cmd(out, tiny_sha, stopping, epochs, extra=(), data_dir=None):
    data_dir = TINY / "data" if data_dir is None else data_dir
    return [sys.executable, str(HERE / "train.py"), "--data", str(data_dir), "--out", str(out),
            "--variant", "synthetic-fixture", "--base", str(TINY / "checkpoint"), "--base-sha256", tiny_sha,
            "--epochs", str(epochs), "--stopping", stopping, "--device", "cpu"] + list(extra)


def check_seed_refusals(tmp, tiny_sha):
    """Seed counts outside the rule are refused (exit 2) before any model loads or any file is written:
    synthetic-fixture --seeds 0, 2 (the median needs an odd N) and len(variance_seeds) + 1; production
    --seeds 1 (production trains exactly production_seeds_required seeds, A3) -- refused in the
    data-validation block, before the production base is touched."""
    n_max = len(contract.seed_policy()["variance_seeds"])
    for n in (0, 2, n_max + 1):
        out = tmp / ("refused-seeds-%d" % n)
        proc = subprocess.run(train_cmd(out, tiny_sha, "fixed_epochs", 1, ["--seeds", str(n)]),
                              capture_output=True, text=True, env=dict(os.environ))
        if proc.returncode != 2 or "REFUSED seeds" not in proc.stderr or (out.exists() and any(out.iterdir())):
            fail("--seeds %d was not refused before training (exit %d): %s" % (n, proc.returncode, proc.stderr[-300:]))
    out = tmp / "refused-production-seeds-1"
    proc = subprocess.run([sys.executable, str(HERE / "train.py"), "--data", str(TINY / "data"), "--out", str(out),
                           "--variant", "production", "--seeds", "1", "--device", "cpu"],
                          capture_output=True, text=True, env=dict(os.environ))
    if proc.returncode != 2 or "REFUSED seeds" not in proc.stderr or out.exists():
        fail("--variant production --seeds 1 was not refused before anything was written (exit %d): %s"
             % (proc.returncode, proc.stderr[-300:]))
    print("seed refusals: synthetic --seeds 0 / 2 / %d and production --seeds 1 refused (exit 2, nothing written)"
          % (n_max + 1))


def check_shift_probe(tmp, tiny_sha, without, Agent):
    """A2 on the tiny fixture: a data dir with shift.jsonl (the tiny eval rows, text-disjoint from train)
    runs the SAME three-seed recipe as `without` (the tiny data dir, byte-identical otherwise); the gate is
    identical -- pass, margin, zero_shot, fine_tuned, calibration, seeds -- and only the shift run carries
    shift_probe (gate_clause false) and the two shift files. A shift row equal to a train row is refused
    (exit 2) before anything is written. Returns (the shift run dir, its data dir)."""
    data_dir = tmp / "data-shift"
    shutil.copytree(TINY / "data", data_dir)
    (data_dir / "shift.jsonl").write_bytes((TINY / "data" / "eval.jsonl").read_bytes())
    out = tmp / "fixed_epochs-seeds3-shift"
    with_rep = run_one(out, tiny_sha, "fixed_epochs", 1, Agent, seeds=3, data_dir=data_dir)
    without_rep = json.loads((without / "gate-report.json").read_text())
    for key in ("pass", "margin", "zero_shot", "fine_tuned", "calibration", "seeds", "recipe_id", "eval_probs_sha256",
                "zero_shot_probs_sha256", "rescore_noise_sha256"):
        if with_rep[key] != without_rep[key]:
            fail("the shift probe changed the gate: %s differs with and without shift.jsonl" % key)
    probe = with_rep["shift_probe"]
    if probe.get("gate_clause") is not False or "shift_probe" in without_rep:
        fail("shift_probe must carry gate_clause false and appear only with shift.jsonl")
    for name, key in (("shift-probs.json", "probs_sha256"), ("shift-zero-shot-probs.json", "zero_shot_probs_sha256")):
        if sha256_file(out / name) != probe[key] or (without / name).exists():
            fail("%s is not bound by shift_probe.%s, or the run without shift.jsonl wrote it" % (name, key))
    if with_rep["inputs_sha256"]["shift_jsonl"] != sha256_file(data_dir / "shift.jsonl"):
        fail("inputs_sha256.shift_jsonl is not the sha256 of shift.jsonl")
    if probe["n"] != len(jsonl_lines((data_dir / "shift.jsonl").read_text(encoding="utf-8"))):
        fail("shift_probe.n %s is not the shift.jsonl row count" % probe["n"])
    bad = tmp / "data-shift-overlap"
    shutil.copytree(data_dir, bad)
    first_train = jsonl_lines((TINY / "data" / "train.jsonl").read_text(encoding="utf-8"))[0]
    (bad / "shift.jsonl").write_text((data_dir / "shift.jsonl").read_text(encoding="utf-8") + first_train + "\n",
                                     encoding="utf-8")
    refused = tmp / "refused-shift-overlap"
    proc = subprocess.run(train_cmd(refused, tiny_sha, "fixed_epochs", 1, ["--seeds", "3"], bad),
                          capture_output=True, text=True, env=dict(os.environ))
    if proc.returncode != 2 or "REFUSED shift-train-overlap" not in proc.stderr or refused.exists():
        fail("a shift.jsonl row equal to a train row was not refused before anything was written (exit %d): %s"
             % (proc.returncode, proc.stderr[-300:]))
    print("shift probe: gate identical with and without shift.jsonl (pass=%s, margin %.4f); shift_probe n %d "
          "ft macro_f1 %.4f ece_post %.4f, gate_clause false; a shift row equal to a train row refused (exit 2)"
          % (with_rep["pass"], with_rep["margin"], probe["n"], probe["fine_tuned"]["macro_f1"],
             probe["fine_tuned"]["ece_post"]))
    return out, data_dir


def _ck_tree_modulo_recipe_id(ck):
    """tree_sha256 of a checkpoint dir with rl_agent_config.json compared WITHOUT training.recipe_id: a
    three-seed recipe.json carries seed_selection, so its recipe_id (provenance only) differs."""
    t = tree_sha256(ck)
    cfg = json.loads((ck / "rl_agent_config.json").read_text())
    cfg.get("training", {}).pop("recipe_id", None)
    t["rl_agent_config.json"] = json.dumps(cfg, sort_keys=True)
    return t


def check_median_run(single, multi):
    """FALSIFY-LAYA-GATE-006 / -011 on the tiny fixture: the median-ECE seed ships (A3)."""
    import gate                                   # torch-free
    decl = contract.seed_selection_decl()
    seeds = decl["seeds"]
    rep = json.loads((multi / "gate-report.json").read_text())
    sb = rep["seeds"]
    per = sb.get("per_seed") or []
    if [r["seed"] for r in per] != seeds or sb.get("n") != len(seeds) or sb.get("policy") != decl["policy"]:
        fail("gate report seeds block: per_seed seeds %s, n %s, policy %s" % ([r.get("seed") for r in per],
                                                                              sb.get("n"), sb.get("policy")))
    if sb.get("label") != contract.seeds_label(len(seeds), decl["policy"]) or sb.get("declared") != seeds[0]:
        fail("gate report seeds label / declared %s" % {k: sb.get(k) for k in ("label", "declared")})
    for r in per:
        if r["rank_key"] != gate.rank_key(r["ece_post"], decl["rank_scale"]):
            fail("per_seed seed %d rank_key %s != floor(ece_post x rank_scale)" % (r["seed"], r["rank_key"]))
    shipped = gate.select_median_seed(per, decl["rank_scale"])
    if sb.get("shipped") != shipped:
        fail("seeds.shipped %s is not the median of its own per_seed rows (%d)" % (sb.get("shipped"), shipped))
    row = [r for r in per if r["seed"] == shipped][0]
    models = sorted(str(q.relative_to(multi)) for q in multi.rglob("model.safetensors"))
    if models != ["checkpoint/model.safetensors"]:
        fail("a median run must keep exactly one checkpoint (the shipped seed's), found %s" % models)
    if sha256_file(multi / "checkpoint" / "model.safetensors") != row["model_safetensors_sha256"]:
        fail("checkpoint/model.safetensors is not the shipped seed's recorded model_safetensors_sha256")
    if (multi / "eval-probs.json").read_bytes() != (multi / "seeds" / ("seed-%d" % shipped) / "eval-probs.json").read_bytes():
        fail("eval-probs.json is not byte-identical to seeds/seed-%d/eval-probs.json" % shipped)
    for r in per:
        if sha256_file(multi / "seeds" / ("seed-%d" % r["seed"]) / "eval-probs.json") != r["eval_probs_sha256"]:
            fail("seeds/seed-%d/eval-probs.json does not hash to its per_seed row" % r["seed"])
    ft = rep["fine_tuned"]
    if (ft["macro_f1"], ft["f_avg"], ft["ece_post"], rep["margin"], rep["pass"], rep["calibration"]["t_applied"]) != (
            row["macro_f1"], row["f_avg"], row["ece_post"], row["margin"], row["pass"], row["t_applied"]):
        fail("the top-level gate is not the shipped seed's (the pass rule reads only the median)")
    training = json.loads((multi / "checkpoint" / "rl_agent_config.json").read_text())["training"]
    if training.get("seed") != shipped:
        fail("checkpoint/rl_agent_config.json training.seed %s != shipped %d" % (training.get("seed"), shipped))
    recipe = json.loads((multi / "recipe.json").read_text())
    if recipe.get("seed_selection") != decl:
        fail("recipe.json seed_selection %s != the contract's %s" % (recipe.get("seed_selection"), decl))
    single_recipe = json.loads((single / "recipe.json").read_text())
    if "seed_selection" in single_recipe or {k: v for k, v in recipe.items() if k != "seed_selection"} != single_recipe:
        fail("the three-seed recipe.json differs from the single-seed one by more than seed_selection")
    row13 = [r for r in per if r["seed"] == seeds[0]][0]
    if row13["model_safetensors_sha256"] != sha256_file(single / "checkpoint" / "model.safetensors"):
        fail("seed %d trained differently in the three-seed run than in the single-seed run" % seeds[0])
    same_tree = None
    if shipped == seeds[0]:
        same_tree = _ck_tree_modulo_recipe_id(multi / "checkpoint") == _ck_tree_modulo_recipe_id(single / "checkpoint")
        if not same_tree:
            fail("seed %d is the median but the shipped checkpoint differs from the single-seed run's" % seeds[0])
    vr = json.loads((multi / "variance-report.json").read_text())
    if [r["seed"] for r in vr["per_seed"]] != seeds or vr.get("n") != len(seeds):
        fail("variance-report.json per_seed / n")
    print("median[seeds %s]: rank_keys %s -> shipped %d; one checkpoint kept (sha = its per_seed row); seed %d "
          "model bit-identical to the single-seed run's%s; label %r"
          % ("/".join(map(str, seeds)), [r["rank_key"] for r in per], shipped, seeds[0],
             "" if same_tree is None else ", checkpoint tree identical (13 is the median)", sb["label"]))


def check_noise_record(out, report, rows, labels):
    """laya-parity-v1 A1 / rescore_noise_schema on a run dir: every field the verifier recomputes, recomputed."""
    k_mult, floor, _ = contract.noise_policy()
    raw = (out / "rescore-noise.json").read_bytes()
    if sha256_bytes(raw) != report["rescore_noise_sha256"]:
        fail("rescore-noise.json sha256 differs from the gate report's rescore_noise_sha256")
    rec = json.loads(raw)
    n = len(rows)
    want = sorted(contract.gate_contract()["rescore_noise_schema"])
    if sorted(rec) != want:
        fail("rescore-noise.json keys %s != rescore_noise_schema %s" % (sorted(rec), want))
    if (rec["schema"], rec["reference"]) != ("laya-rescore-noise-v1", "float64"):
        fail("rescore-noise.json schema / reference %s / %s" % (rec["schema"], rec["reference"]))
    if not (isinstance(rec["k"], float) and rec["k"] == k_mult and rec["floor_abs"] == floor):
        fail("rescore-noise.json k / floor_abs %r / %r != laya-parity-v1 %r / %r" % (rec["k"], rec["floor_abs"], k_mult, floor))
    if rec["control_max_abs"] != 0.0 or rec["control_rows"] != list(range(min(5, n))):
        fail("rescore-noise.json control %r on rows %s" % (rec["control_max_abs"], rec["control_rows"]))
    if [st["which"] for st in rec["sets"]] != ["fine_tuned", "zero_shot"]:
        fail("rescore-noise.json sets %s" % [st.get("which") for st in rec["sets"]])
    summary = []
    for st, name in zip(rec["sets"], ("eval-probs.json", "zero-shot-probs.json")):
        p32 = [r["probabilities"] for r in json.loads((out / name).read_text())["rows"]]
        if st["scored"] != "eval" or st["n"] != n or [r["row"] for r in st["rows"]] != list(range(n)):
            fail("%s set does not cover every eval row once, in order" % st["which"])
        p64 = [r["probabilities_f64"] for r in st["rows"]]
        if any(len(a) != len(labels) for a in p64):
            fail("%s set has a row that is not K = %d probabilities" % (st["which"], len(labels)))
        mx = max(abs(float(a) - float(b)) for r64, r32 in zip(p64, p32) for a, b in zip(r64, r32))
        if mx != st["max_abs"]:
            fail("%s max_abs %r != %r recomputed from the files" % (st["which"], st["max_abs"], mx))
        if st["bound"] != max(floor, k_mult * mx):
            fail("%s bound %r != max(floor, k x max_abs) = %r" % (st["which"], st["bound"], max(floor, k_mult * mx)))
        am = lambda v: max(range(len(v)), key=lambda j: v[j])  # noqa: E731
        agree = sum(am(a) == am(b) for a, b in zip(p64, p32))
        if agree != st["argmax_agree"]:
            fail("%s argmax_agree %s != %d recomputed" % (st["which"], st["argmax_agree"], agree))
        summary.append("%s max_abs %.3g bound %.3g argmax %d/%d" % (st["which"], mx, st["bound"], agree, n))
    if rec["sets"][0]["t_applied"] != report["calibration"]["t_applied"]:
        fail("fine_tuned t_applied %r != calibration t_applied %r" % (rec["sets"][0]["t_applied"],
                                                                     report["calibration"]["t_applied"]))
    return "; ".join(summary)


def run_one(out, tiny_sha, stopping, epochs, Agent, seeds=1, data_dir=None):
    """One train.py run in `out`, then every assertion of the module docstring. Returns the gate report."""
    data_dir = TINY / "data" if data_dir is None else Path(data_dir)
    has_shift = (data_dir / "shift.jsonl").is_file()
    seed_list = contract.resolve_seeds(seeds, "synthetic-fixture")
    cmd = train_cmd(out, tiny_sha, stopping, epochs, ["--seeds", str(seeds)] if seeds != 1 else [], data_dir)
    proc = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ))
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    if proc.returncode not in (0, 3):
        fail("train.py exited %d" % proc.returncode)
    lines = proc.stdout.splitlines()

    def first(marker):
        idx = [i for i, ln in enumerate(lines) if ln.startswith(marker)]
        if not idx:
            fail("train.py never logged %r" % marker)
        return idx[0]
    if not first("RECIPE WRITTEN") < first("CHECKPOINT COMPLETE") < first("SCORING START"):
        fail("log order is not RECIPE WRITTEN < CHECKPOINT COMPLETE < SCORING START")
    has_stop = any(ln.startswith("STOP ") for ln in lines)
    if stopping == "early_stopping":
        if not first("RECIPE WRITTEN") < first("EPOCH ") < first("STOP ") < first("CHECKPOINT COMPLETE"):
            fail("early_stopping log order is not RECIPE WRITTEN < EPOCH < STOP < CHECKPOINT COMPLETE")
    elif has_stop:
        fail("fixed_epochs logged a STOP line")
    for rel in contract.run_dir_files(seed_list, has_shift):
        if not (out / rel).is_file():
            fail("run dir is missing %s" % rel)
    recipe_bytes = (out / "recipe.json").read_bytes()
    recipe = json.loads(recipe_bytes)
    if recipe["variant"] != "synthetic-fixture" or recipe["base"]["sha256"] != tiny_sha:
        fail("recipe.json is not the synthetic-fixture variant on the tiny base")
    want_keys = contract.expected_keys("recipe_json_schema", stopping, seeds, has_shift)
    if sorted(recipe) != want_keys:
        fail("recipe.json keys %s differ from recipe_json_schema for %s, %d seed(s): %s"
             % (sorted(recipe), stopping, seeds, want_keys))
    if stopping == "early_stopping" and recipe["early_stopping"] != contract.early_stopping_decl():
        fail("recipe.json early_stopping differs from the contract block")
    if recipe["epochs"] != epochs:
        fail("recipe.json epochs %s != %s" % (recipe["epochs"], epochs))
    training = json.loads((out / "checkpoint" / "rl_agent_config.json").read_text())["training"]
    if stopping == "early_stopping":
        st = training.get("stopping") or {}
        decl = contract.early_stopping_decl()
        if not (decl["first_candidate_epoch"] <= st.get("best_epoch", -1) <= st.get("epochs_run", -2) <= epochs):
            fail("training.stopping record %s is not a best epoch within the run" % st)
        if [e["epoch"] for e in st["per_epoch"]] != list(range(1, st["epochs_run"] + 1)):
            fail("training.stopping per_epoch does not cover every run epoch")
    elif "stopping" in training:
        fail("fixed_epochs wrote a training.stopping record")
    t_recipe, t_ck, t_eval = ((out / rel).stat().st_mtime_ns for rel in
                              ("recipe.json", "checkpoint/model.safetensors", "eval-probs.json"))
    if not (t_recipe <= t_ck <= t_eval):
        fail("mtime order recipe.json <= model.safetensors <= eval-probs.json violated")
    report = json.loads((out / "gate-report.json").read_text())
    want_report = contract.expected_keys("gate_report_schema", stopping, seeds, has_shift)
    if sorted(report) != want_report:
        fail("gate-report.json keys %s differ from gate_report_schema for this run: %s" % (sorted(report), want_report))
    if ("shift_jsonl" in report["inputs_sha256"]) is not has_shift:
        fail("inputs_sha256.shift_jsonl must be present exactly when the data dir carries shift.jsonl")
    if report["recipe_id"] != sha256_bytes(recipe_bytes):
        fail("gate report recipe_id is not the sha256 of recipe.json")
    if report["thresholds"] != contract.thresholds():
        fail("gate report thresholds differ from the contract")
    if report["device_used"] != "cpu" or report["device_is_cpu"] is not True:
        fail("forced --device cpu was not recorded as device_used cpu / device_is_cpu true")
    if seeds == 1:                                # the LEGACY rule: no seed_selection, seed 13 ships
        if report["seeds"] != {"declared": 13, "n": 1, "label": "single seed"}:
            fail("gate report seeds block %s" % report["seeds"])
        if (out / "variance-report.json").exists() or (out / "seeds").exists():
            fail("a single-seed run wrote variance-report.json or seeds/")
        if training.get("seed") != 13:
            fail("a single-seed run shipped seed %s, not the declared 13" % training.get("seed"))
    s = report["calibration"]["slice_ids"]
    if s != sorted(set(s)) or len(s) != report["calibration"]["slice_size"]:
        fail("calibration.slice_ids is not a sorted, duplicate-free list of slice_size")
    for name, key in (("eval-probs.json", "eval_probs_sha256"), ("zero-shot-probs.json", "zero_shot_probs_sha256"),
                      ("probes.json", "probes_sha256")):
        if sha256_file(out / name) != report[key]:
            fail("%s sha256 differs from the report's %s" % (name, key))
    before = tree_sha256(out / "checkpoint")
    import data                                   # torch-free
    import train                                  # imports torch + laya (already loaded by main)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        agent = train.load_for_scoring(out / "checkpoint", before["model.safetensors"])
    if tree_sha256(out / "checkpoint") != before:
        fail("a further reload changed a checkpoint file")
    # FALSIFY-LAYA-GATE-007 (second half): the written eval probabilities are what a SEPARATE F16 reload of
    # the shipped checkpoint computes through Laya's own predict path, within pack_rescore_probs_abs.
    task = data.load_task(out / "task.json")
    rows = data.load_rows(data_dir / "eval.jsonl", task, "eval")
    noise = check_noise_record(out, report, rows, task["labels"])
    P_again, _ = train.score_rows(agent, rows, data.laya_question(task))
    written = json.loads((out / "eval-probs.json").read_text())["rows"]
    tol = float(contract.load_yaml(contract.REPO / "contracts" / "laya-parity-v1.yaml")
                ["equations"]["pack_rescore_probs_abs"]["float_tolerance"])
    dp = max(abs(float(a) - float(b)) for r, pr in zip(written, P_again) for a, b in zip(r["probabilities"], pr))
    same_argmax = all(max(range(len(pr)), key=lambda j: pr[j]) == max(range(len(r["probabilities"])),
                      key=lambda j: r["probabilities"][j]) for r, pr in zip(written, P_again))
    if len(written) != len(rows) or dp > tol or not same_argmax:
        fail("eval-probs.json differs from a separate F16 reload: max |dp| %.3g (tol %g), argmax equal %s"
             % (dp, tol, same_argmax))
    del agent
    print("lifecycle[%s, seeds %d%s]: exit %d, %d checkpoint files unchanged across reload, separate-reload "
          "re-score max |dp| %.3g over %d rows, gate pass=%s; noise record ok (%s)"
          % (stopping, seeds, ", shift" if has_shift else "", proc.returncode, len(before), dp, len(rows),
             report["pass"], noise))
    return report


if __name__ == "__main__":
    main()
