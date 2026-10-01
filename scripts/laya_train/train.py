"""Fine-tune Laya on a user's shots, calibrate, and gate -- the local back office (D-01..D-08).

    uv run --project scripts/laya_train --frozen python scripts/laya_train/train.py \
        --data DIR --out DIR [--epochs E] [--stopping early_stopping|fixed_epochs] [--seeds N]
        [--device mps|cuda|cpu]
    (or: just laya-train <data dir> <run dir> [args])

    --variant synthetic-fixture --base CKPT_DIR --base-sha256 HEX --epochs E
        the lifecycle self-test on a tiny synthetic checkpoint (just laya-train-lifecycle). The
        production variant ALWAYS uses the contract's pinned base; --base is refused without it.

The run, in order (every value from contracts/laya-finetune-gate-v1.yaml via contract.py):

  1. validate the data dir (data.py): task.json, train.jsonl, a REQUIRED eval.jsonl and an OPTIONAL
     shift.jsonl; eval or shift text overlapping train (NFC/trim/whitespace-normalized) is refused;
     conflicting train duplicates refused.
  2. resolve the recipe, write <out>/recipe.json (sort_keys, compact) and print RECIPE WRITTEN <sha256>
     BEFORE any model scores anything (recipe_before_scores, D-04).
  3. request the device mps -> cuda -> cpu, load Laya's own Agent on the pinned base with
     expected_sha256={"model.safetensors": <sha>} (the mapping laya.revisions.verify_digests requires),
     and record the device READ BACK from the parameters (Laya silently falls back to CPU, D-03).
  4. split fit / calibration by normalized-text GROUP (seeded, stratified), build rows with Laya's own
     Agent._encode_state, and run the spike-024 ft_laya.py loop. Under the contract's default
     `early_stopping` rule (1.1.0) `epochs` is the maximum: after every epoch the CALIBRATION slice's
     NLL at the bounded fitted T is the monitor, the best epoch's weights are restored and STOP is
     logged. eval.jsonl rows never reach this step (the training function is never given them).
  5. write the COMPLETE checkpoint dir (F16 weights with an F32 temperature, rl_agent_config.json,
     encoder/, tokenizer/ with tokenizer_config.json already in Laya's fixed form) and print
     CHECKPOINT COMPLETE -- before anything reloads it.
  6. RELOAD it through Agent(<ckpt>, device="cpu") in fp32 (sha256s asserted unchanged), print
     SCORING START, fit T by NLL on the calibration slice within [0.5, 5.0], write T into
     rl_agent_config.json, and reload AGAIN so every eval probability comes from Laya's own predict
     path with the saved temperature (f16_reload_scoring, Pitfall 5).
  7. zero-shot: the declared base on eval.jsonl -> zero-shot-probs.json; fine-tuned -> eval-probs.json;
     probes on the decide-apr-v1 probe task -> probes.json.
  8. gate.evaluate_gate; under the median rule, gate.select_median_seed picks the shipped seed.
  9. the float64 re-score noise record (A1, laya-parity-v1 rescore_noise_reference) of the shipped
     checkpoint and the base -> rescore-noise.json; a manual fp32 forward that does not reproduce the
     Scorer's logits exactly refuses the run (exit 2) with no record and no gate report.
 10. with shift.jsonl, the shift probe (A2) -- scored only now, with a fresh reload of the shipped
     checkpoint and the base -> shift-probs.json / shift-zero-shot-probs.json, gate-report `shift_probe`
     with gate_clause false. It never touches `pass`, the per-seed rows or the exit code.
 11. gate-report.json (rescore_noise_sha256 binds the record); GATE PASS (exit 0) or GATE FAIL (exit 3).

Seeds (D-08 as amended by A3, laya-finetune-gate-v1 1.4.0 seed_policy). PRODUCTION trains exactly the
three gate seeds (13, 17, 23; `--seeds` other than 3 is refused before anything is written) on the SAME
data, split and recipe, each in `<out>/seeds/seed-<s>/` (steps 4-7). Only after every seed's checkpoint
is fixed are they ranked by (floor(ece_post x rank_scale), seed) and the MEDIAN seed ships: its
checkpoint becomes `<out>/checkpoint`, the other two checkpoints are deleted, every seed's
`seeds/seed-<s>/eval-probs.json` is kept and hash-bound in gate-report `seeds.per_seed`, and the gate
passes only if the shipped seed passes BOTH clauses. recipe.json carries `seed_selection`. The median
is selected WITH eval labels (seed_policy.honesty); variance-report.json (mean +- sample sd) is
information. The seed varies the training RNG (torch / random / numpy: head init, batch shuffle), not
the data or the calibration slice. The synthetic-fixture variant also accepts `--seeds 1` (its default):
the LEGACY single-seed rule, no `seed_selection`, the declared seed ships, label `single seed`.

Ordering (asserted, not assumed): RECIPE WRITTEN < STOP < CHECKPOINT COMPLETE < SCORING START, and
recipe.json <= checkpoint/model.safetensors <= eval-probs.json by mtime -- no eval probability exists
before the stopping decision and the shipped weights are fixed.

Logs carry counts, hashes and timings only -- never input text.
"""
import argparse
import gc
import json
import math
import os
import random
import shutil
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import laya.agent as laya_agent  # noqa: E402
from laya import Agent  # noqa: E402
from laya.common import QTYPES, proper_reward, temp_bucket  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import contract  # noqa: E402
from common import f32_hex_list, f32_list, save_f16, sha256_bytes, tree_sha256, write_json  # noqa: E402
import data  # noqa: E402
import gate  # noqa: E402
import metrics  # noqa: E402

EXIT_PASS, EXIT_REFUSED, EXIT_GATE_FAIL = 0, 2, 3
BASE_FILES = ("rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*")


def log(msg):
    print(msg, flush=True)


def refuse(msg):
    print(msg if msg.startswith("REFUSED") else "REFUSED " + msg, file=sys.stderr, flush=True)
    sys.exit(EXIT_REFUSED)


# ------------------------------------------------------------------------------------------ Laya glue

class _ConfidenceSpy:
    """Captures the unrounded `p` Laya's Agent._decode_answers computes (at its answer_confidence call).

    `predict` publishes round(p, 4); the gate needs the unrounded probabilities of that same path, so the
    spy records them and the caller asserts the public answer equals round(p, 4)."""

    def __init__(self):
        self.captured = []
        self._orig = None

    def __enter__(self):
        self._orig = laya_agent.answer_confidence

        def spy(p, k):
            self.captured.append(np.array(p, dtype=p.dtype, copy=True))
            return self._orig(p, k)
        laya_agent.answer_confidence = spy
        return self

    def __exit__(self, *exc):
        laya_agent.answer_confidence = self._orig
        return False


class Scorer:
    """Runs Laya's OWN `Agent.predict` for one (text, question) and returns (p, logits[:K], n_tokens)."""

    def __init__(self, agent):
        self.agent = agent
        self.logits = []
        orig_forward = agent._forward

        def forward(b):
            out = orig_forward(b)
            self.logits.append(np.array(out[0], copy=True))
            return out
        agent._forward = forward

    def score(self, text, question):
        agent = self.agent
        internal = {"q": Agent._to_internal(question)}
        it = agent._encode_state(text, ["q"], internal)[0]
        k = len(it["markers"])
        self.logits.clear()
        with _ConfidenceSpy() as spy:
            res = agent.predict(text, {"q": question})
        if len(spy.captured) != 1 or len(self.logits) != 1:
            raise RuntimeError("expected one decoded question and one forward, got %d / %d"
                               % (len(spy.captured), len(self.logits)))
        p = spy.captured[0]
        z = self.logits[0][0, :k]
        public = list(res["answers"]["q"]["probabilities"].values())
        if public != [round(float(v), 4) for v in p]:
            raise RuntimeError("predict's public answer is not round(p, 4): the spy captured another path")
        t = agent.temperature_by_options.get(temp_bucket(it["qtype"], k), agent.temperature[it["qtype"]])
        ref = gate.softmax(z[None, :], t)[0]
        if np.abs(ref - p.astype(np.float64)).max() > 1e-6:
            raise RuntimeError("captured p is not softmax(logits / T) at the applied temperature")
        return p, z, len(it["ids"])


def load_agent(src, device, digest, revision=None):
    """Laya's own loader with the digest MAPPING; returns the agent and the device READ BACK."""
    kw = {"device": device, "expected_sha256": {"model.safetensors": digest}}
    if revision is not None:
        kw["revision"] = revision
    agent = Agent(str(src), **kw)
    return agent, str(next(agent.model.parameters()).device)


def load_for_scoring(src, digest, revision=None):
    agent, used = load_agent(src, "cpu", digest, revision)
    agent.model.float().eval()
    if used != "cpu":
        raise RuntimeError("scoring agent landed on %s, expected cpu" % used)
    return agent


def request_device(forced):
    if forced:
        return forced
    for d in contract.device_order():
        if d == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        if d == "cuda" and torch.cuda.is_available():
            return "cuda"
        if d == "cpu":
            return "cpu"
    return "cpu"


def fixed_tokenizer_config_bytes(src_bytes):
    """tokenizer_config.json in the form laya.agent._fix_tokenizer_config produces, so Laya's loader
    never rewrites it in place (a hashed checkpoint file changing after the fact, T-08-08-06)."""
    tcfg = json.loads(src_bytes)
    changed = False
    if tcfg.get("tokenizer_class") in (None, "TokenizersBackend"):
        tcfg["tokenizer_class"] = "PreTrainedTokenizerFast"
        tcfg.pop("backend", None)
        tcfg.pop("is_local", None)
        changed = True
    extra = tcfg.get("extra_special_tokens")
    if isinstance(extra, list):
        tcfg["extra_special_tokens"] = {"extra_%d" % i: t for i, t in enumerate(extra)}
        changed = True
    return json.dumps(tcfg, indent=2).encode("utf-8") if changed else src_bytes


def collate(items, pad):
    """spike-024 ft_laya.py collate."""
    n, L, km = len(items), max(len(i["ids"]) for i in items), max(len(i["markers"]) for i in items)
    ids = torch.full((n, L), pad, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, km), dtype=torch.long)
    mm = torch.zeros((n, km), dtype=torch.bool)
    for j, it in enumerate(items):
        ids[j, :len(it["ids"])] = torch.tensor(it["ids"])
        att[j, :len(it["ids"])] = 1
        mpos[j, :len(it["markers"])] = torch.tensor(it["markers"])
        mm[j, :len(it["markers"])] = True
    return ids, att, mpos, mm, torch.tensor([it["qtype"] for it in items])


# ------------------------------------------------------------------------------------------ stages

class Base:
    """Where the base checkpoint comes from and how it is pinned."""

    def __init__(self, args):
        if args.variant == "production":
            if args.base or args.base_sha256:
                refuse("REFUSED base: --base is accepted only with --variant synthetic-fixture; production "
                       "always uses the contract base")
            b = contract.base()
            self.src, self.revision, self.digest = b["repo"], b["revision"], b["model_safetensors_sha256"]
            self.block = contract.production_base_block()
            self._dir = None
        else:
            if not args.base or not args.base_sha256:
                refuse("REFUSED base: --variant synthetic-fixture needs --base CKPT_DIR and --base-sha256 HEX")
            self.src, self.revision, self.digest = str(Path(args.base).resolve()), None, args.base_sha256
            self.block = {"family": "laya", "repo": "synthetic", "revision": "local", "checkpoint": "tiny-synthetic",
                          "sha256": self.digest}
            self._dir = Path(self.src)

    def load(self, device):
        return load_agent(self.src, device, self.digest, self.revision)

    def directory(self):
        """The on-disk base snapshot (resolved after Agent has fetched and verified it)."""
        if self._dir is None:
            from huggingface_hub import snapshot_download
            self._dir = Path(snapshot_download(self.src, revision=self.revision, allow_patterns=list(BASE_FILES),
                                               local_files_only=True))
            got = data.sha256_file(self._dir / "model.safetensors")
            if got != self.digest:
                raise RuntimeError("base snapshot model.safetensors sha256 %s != contract %s" % (got, self.digest))
        return self._dir


def calibration_logits(model, items, k, pad, dev, bs):
    """fp32 logits[:, :k] on the calibration rows, eval mode, no grad, the training collate (the
    early-stopping SELECTION signal; every reported score still comes from the F16 reload)."""
    was_training = model.training
    model.eval()
    out = []
    with torch.no_grad():
        for b in range(0, len(items), bs):
            ids_t, att, mpos, mm, qt = (x.to(dev) for x in collate(items[b:b + bs], pad))
            z, _ = model(ids_t, att, mpos, mm, qt)
            out.append(z[:, :k].float().cpu().numpy())
    if was_training:
        model.train()
    return np.concatenate(out).astype(np.float64)


def refuse_if_data_changed(data_dir, input_sha):
    """The data dir must still hold exactly what was trained on: a file edited (or a shift.jsonl added / removed)
    during the run is REFUSED data-changed (exit 2) with no report, never bound to this model."""
    now = {name: data.sha256_file(data_dir / name) for name in ("task.json", "train.jsonl", "eval.jsonl", "shift.jsonl")
           if (data_dir / name).is_file()}
    if now != input_sha:
        changed = sorted(n for n in set(now) | set(input_sha) if now.get(n) != input_sha.get(n))
        refuse("REFUSED data-changed: %s in %s changed during the run; no gate report is written" % (changed, data_dir))


def stopping_record_or_refuse(stopper, epochs_run):
    """gate.EarlyStopper.record, whose typed refusal (no finite calibration monitor, nothing to restore)
    ends the run as `REFUSED early-stopping: ...`, exit 2 -- never a traceback."""
    try:
        return stopper.record(epochs_run)
    except data.DataError as e:
        refuse(str(e))


def train_seed(seed, base, requested, question, fit_rows, k, epochs, stopping="fixed_epochs", calib_rows=None):
    """The spike-024 ft_laya.py loop on Laya's own rows; returns (agent, info).

    `calib_rows` (train-side, the calibration slice) is read ONLY by the early-stopping monitor. This
    function is never given eval rows, so eval cannot influence the stopping epoch or the weights."""
    rn = contract.recipe_number
    t_min, t_max = contract.constant("calibration_temp_min", "float"), contract.constant("calibration_temp_max", "float")
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    agent, device_used = base.load(requested)
    log("DEVICE seed=%d requested=%s used=%s torch=%s" % (seed, requested, device_used, torch.__version__))
    if device_used == "cpu":
        log("WARNING: training on CPU (device_is_cpu true)%s"
            % ("" if requested == "cpu" else " -- %s was requested and Laya fell back" % requested))
    Agent._check_question("q", question)
    internal = {"q": Agent._to_internal(question)}
    items = [agent._encode_state(t, ["q"], internal)[0] for t, _ in fit_rows]
    if any(len(it["markers"]) != k for it in items):
        raise RuntimeError("a training row does not carry one marker per criterion")
    ys = torch.tensor([y for _, y in fit_rows])
    bucket = temp_bucket(QTYPES["choice"], k)
    t_loss = agent.temperature_by_options.get(bucket, agent.temperature[QTYPES["choice"]])
    model = agent.model
    model.train()
    enc = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    rest = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    opt = torch.optim.AdamW([{"params": enc, "lr": rn("encoder_lr", "float")},
                             {"params": rest, "lr": rn("head_lr", "float")}], weight_decay=rn("weight_decay", "float"))
    bs = rn("batch_size", "int")
    steps = epochs * math.ceil(len(items) / bs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps), eta_min=rn("eta_min", "float"))
    pad, dev = agent.tok.pad_token_id, agent.device
    losses = []
    stopper, best_state, calib_items, yc = None, None, None, None
    if stopping == "early_stopping":
        if not calib_rows:
            raise RuntimeError("early_stopping needs the calibration slice rows")
        stopper = gate.EarlyStopper(contract.early_stopping_decl(), epochs)
        calib_items = [agent._encode_state(t, ["q"], internal)[0] for t, _ in calib_rows]
        yc = np.array([y for _, y in calib_rows], dtype=np.int64)
    steps_run, epochs_run = 0, 0
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        order = list(range(len(items)))
        random.shuffle(order)
        for b in range(0, len(order), bs):
            sel = order[b:b + bs]
            ids_t, att, mpos, mm, qt = (x.to(dev) for x in collate([items[i] for i in sel], pad))
            z, act = model(ids_t, att, mpos, mm, qt)
            z = z[:, :k]
            yb = ys[sel].to(dev)
            ce = F.cross_entropy(z, yb)
            q = torch.softmax(z / t_loss, -1)
            onehot = F.one_hot(yb, k).float()
            rl = -proper_reward(q, onehot, qt, mm[:, :k].float(),
                                w_sph=rn("proper_reward_w_sph", "float"), w_rps=rn("proper_reward_w_rps", "float")).mean()
            loss = ce + rl + 0.0 * act.sum()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), rn("grad_clip", "float"))
            opt.step()
            sched.step()
            losses.append(float(ce.item()))
            steps_run += 1
        epochs_run = epoch
        if stopper is not None and stopper.evaluates(epoch):
            zc = calibration_logits(model, calib_items, k, pad, dev, bs)
            m, t_star = gate.calibration_monitor(zc, yc, t_min, t_max)
            improved, stop = stopper.update(epoch, m, t_star)
            if improved:        # an exact copy of this epoch's weights (restore: best)
                best_state = {n: v.detach().to("cpu", copy=True) for n, v in model.state_dict().items()}
            log("EPOCH seed=%d epoch=%d ce_last=%.4f calib_nll_at_t*=%.6f t*=%s best_epoch=%s%s"
                % (seed, epoch, losses[-1], m, "%.4f" % t_star if t_star is not None else "nan",
                   stopper.best_epoch, " improved" if improved else ""))
            if stop:
                break
    if dev.type == "mps":
        torch.mps.synchronize()
    train_s = time.time() - t0
    stop_record = None
    if stopper is not None:
        stop_record = stopping_record_or_refuse(stopper, epochs_run)
        model.load_state_dict(best_state)
        log("STOP seed=%d rule=early_stopping best_epoch=%d best_calib_nll=%.6f epochs_run=%d reason=%s"
            % (seed, stop_record["best_epoch"], stop_record["best_monitor"], epochs_run, stop_record["reason"]))
        del best_state
    model.eval()
    info = {"seed": seed, "device_requested": requested, "device_used": device_used, "epochs": epochs,
            "stopping": stopping, "steps": steps_run, "steps_planned": steps, "epochs_run": epochs_run,
            "fit_rows": len(items), "train_seconds": round(train_s, 1), "loss_temperature": t_loss,
            "ce_first": round(losses[0], 4) if losses else None, "ce_last": round(losses[-1], 4) if losses else None,
            "stopping_record": stop_record}
    log("TRAIN seed=%d fit_rows=%d epochs=%d epochs_run=%d steps=%d/%d seconds=%.1f ce_first=%s ce_last=%s"
        % (seed, len(items), epochs, epochs_run, steps_run, steps, train_s, info["ce_first"], info["ce_last"]))
    return agent, info


def write_checkpoint(agent, base_dir, ck, provenance):
    """The COMPLETE checkpoint dir, before anything reloads it; returns {relpath: sha256}."""
    ck.mkdir(parents=True)
    save_f16(agent.model.state_dict(), ck / "model.safetensors")
    cfg = json.loads((base_dir / "rl_agent_config.json").read_bytes())
    cfg["training"] = provenance            # Laya reads head/len/temperature keys; this is provenance only
    write_json(ck / "rl_agent_config.json", cfg)
    for sub in ("encoder", "tokenizer"):
        for src in sorted((base_dir / sub).rglob("*")):
            if src.is_file():
                dst = ck / sub / src.relative_to(base_dir / sub)
                dst.parent.mkdir(parents=True, exist_ok=True)
                raw = src.read_bytes()
                if dst.name == "tokenizer_config.json":
                    raw = fixed_tokenizer_config_bytes(raw)
                dst.write_bytes(raw)
    for need in ("encoder/config.json", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"):
        if not (ck / need).is_file():
            raise RuntimeError("base snapshot has no %s" % need)
    return tree_sha256(ck)


def reload_checked(ck, before):
    """Agent(<ckpt>, device="cpu") in fp32; the reload must not change a single checkpoint file."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # the base ships choice:11+ = 0.10 (clamped)
        agent = load_for_scoring(ck, before["model.safetensors"])
    after = tree_sha256(ck)
    if after != before:
        raise RuntimeError("Laya's loader changed checkpoint file(s): %s"
                           % sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k)))
    return agent


def score_rows(agent, rows, question):
    sc = Scorer(agent)
    P, Z = [], []
    for t, _ in rows:
        p, z, _ = sc.score(t, question)
        P.append(p)
        Z.append(z)
    return np.array(P, dtype=np.float32), np.array(Z, dtype=np.float32)


def calibrate_and_score(ck, sha_before, calib_rows, eval_rows, question, k, stopping):
    """F16 reload -> calibration fit -> T written -> reload again -> eval probabilities."""
    assert_checkpoint_fixed(ck, stopping)
    agent = reload_checked(ck, sha_before)
    bucket = temp_bucket(QTYPES["choice"], k)
    t_pre = agent.temperature_by_options.get(bucket, agent.temperature[QTYPES["choice"]])
    _, zc = score_rows(agent, calib_rows, question)
    yc = np.array([y for _, y in calib_rows])
    t_fit, t_applied, clamp_hit = gate.fit_temperature(zc, yc, contract.constant("calibration_temp_min", "float"),
                                                       contract.constant("calibration_temp_max", "float"))
    del agent
    cfg_path = ck / "rl_agent_config.json"
    cfg = json.loads(cfg_path.read_bytes())
    cfg.setdefault("temperature_by_options", {})[bucket] = t_applied
    write_json(cfg_path, cfg)
    sha_cal = {**sha_before, "rl_agent_config.json": data.sha256_file(cfg_path)}   # the only file rewritten
    agent = reload_checked(ck, sha_cal)
    if agent.temperature_by_options.get(bucket) != t_applied:
        raise RuntimeError("the reloaded checkpoint does not apply T %r to %s" % (t_applied, bucket))
    assert_checkpoint_fixed(ck, stopping)
    P, Z = score_rows(agent, eval_rows, question)
    pre = gate.softmax(Z, t_pre)
    calib = {"bucket": bucket, "t_pre": t_pre, "t_fitted": t_fit, "t_applied": t_applied, "clamp_hit": clamp_hit}
    return agent, P, pre, calib, sha_cal


def assert_checkpoint_fixed(ck, stopping):
    """Before ANY eval probability: the shipped weights and (early_stopping) the stopping record are on
    disk. The eval set can then not have influenced the epoch, the weights or T."""
    if not (ck / "model.safetensors").is_file():
        raise RuntimeError("eval scoring requested before checkpoint/model.safetensors exists")
    training = json.loads((ck / "rl_agent_config.json").read_bytes()).get("training", {})
    if stopping == "early_stopping" and not (training.get("stopping") or {}).get("best_epoch"):
        raise RuntimeError("eval scoring requested before the early-stopping record is in the checkpoint")


def assert_eval_after_checkpoint(recipe_path, ck, eval_path):
    """mtime order recipe.json <= checkpoint/model.safetensors <= eval-probs.json (recipe_before_scores,
    early_stopping_train_side) -- for the shipped run dir and for every seeds/seed-<s>/ of a median run."""
    t_recipe, t_ck, t_eval = (Path(p).stat().st_mtime_ns for p in (recipe_path, ck / "model.safetensors", eval_path))
    if not (t_recipe <= t_ck <= t_eval):
        raise RuntimeError("ordering violated: mtime recipe.json %d, %s %d, %s %d"
                           % (t_recipe, ck / "model.safetensors", t_ck, eval_path, t_eval))


def eval_probs_obj(labels, eval_rows, P):
    return {"labels": list(labels), "rows": [
        {"row": i, "text_sha256": data.exact_sha256(t), "probabilities": f32_list(P[i])}
        for i, (t, _) in enumerate(eval_rows)]}


def probes_obj(agent):
    task, inputs, max_tokens = contract.probe_policy()
    labels = list(task["criteria"])
    sc = Scorer(agent)
    probes = []
    for i, text in enumerate(inputs):
        p, _, n = sc.score(text, task)
        if n > max_tokens:
            raise RuntimeError("probe %d built a %d-token row, over decide-apr-v1 probe_max_row_tokens %d"
                               % (i, n, max_tokens))
        probes.append({"input_index": i, "tokens": n, "label": labels[int(np.argmax(p))],
                       "probabilities_f32_hex": f32_hex_list(p)})
    return {"probes": probes}


# ------------------------------------------------------------------------------------------ A1 noise record

def manual_logits(model, item):
    """One built row through the model by hand -- encoder -> type embedding -> head layers -> marker
    gather -> scorer, batch 1, no padding (spike 028 torch_triad `run`) -- in the model's own dtype.
    In fp32 it must reproduce the Scorer's logits EXACTLY (the control); cast to float64 it is the
    reference of laya-parity-v1 rescore_noise_reference."""
    n = len(item["ids"])
    dev = next(model.parameters()).device
    with torch.no_grad():
        ids = torch.tensor([item["ids"]], dtype=torch.long, device=dev)
        h = model.encoder(input_ids=ids,
                          attention_mask=torch.ones(1, n, dtype=torch.long, device=dev)).last_hidden_state
        h = h + model.type_emb(torch.tensor([item["qtype"]], device=dev))[:, None, :]
        if model.head is not None:
            pad = torch.zeros(1, n, dtype=torch.bool, device=dev)
            for layer in model.head.layers:
                h = layer(h, src_key_padding_mask=pad)
        idx = torch.tensor([item["markers"]], device=dev)[:, :, None].expand(-1, -1, h.size(-1))
        z = model.scorer(torch.gather(h, 1, idx)).squeeze(-1)[0]
    return z.cpu().numpy()


def rescore_noise_set(agent, which, rows, question, probs_path, labels, k_mult, floor):
    """(set record | None, control max |dz|, control rows) for one (checkpoint, eval set) pair.

    p32 are EXACTLY the probabilities written to `probs_path`, widened to float64. The manual fp32
    forward must reproduce the Scorer's logits bit for bit on the first min(5, n) rows; otherwise None
    is returned (the caller refuses) and the model is left in fp32. Then the SAME model is cast to
    float64, every row forwarded from Laya's own ids / markers, p_f64 = softmax(z_f64 / T) in float64
    at the temperature the Scorer applied. max_abs = max |p32 - p_f64| over rows and components.
    CONSUMES the agent: its model is float64 afterwards."""
    written = json.loads(Path(probs_path).read_text())
    if written["labels"] != list(labels) or [r["row"] for r in written["rows"]] != list(range(len(rows))):
        raise RuntimeError("%s does not hold one row per eval row in order under %s" % (probs_path, labels))
    p32 = np.array([r["probabilities"] for r in written["rows"]], dtype=np.float64)
    k = len(labels)
    internal = {"q": Agent._to_internal(question)}
    items = [agent._encode_state(t, ["q"], internal)[0] for t, _ in rows]
    if any(len(it["markers"]) != k for it in items) or len({it["qtype"] for it in items}) != 1:
        raise RuntimeError("an eval row does not carry one marker per criterion under one question type")
    qt = items[0]["qtype"]
    t_applied = float(agent.temperature_by_options.get(temp_bucket(qt, k), agent.temperature[qt]))
    sc = Scorer(agent)
    ctrl_rows = list(range(min(5, len(rows))))
    dz = []
    for i in ctrl_rows:
        _, z_sc, _ = sc.score(rows[i][0], question)
        z_man = manual_logits(agent.model, items[i])[:k]
        dz.append(float(np.abs(z_man.astype(np.float64) - np.asarray(z_sc, dtype=np.float64)).max()))
    ctrl = float(np.max(dz))                       # NaN propagates: a NaN control is not 0.0
    if not ctrl == 0.0:
        return None, ctrl, ctrl_rows
    m64 = agent.model.to(torch.float64).eval()
    P64 = np.stack([gate.softmax(manual_logits(m64, it)[None, :k].astype(np.float64), t_applied)[0]
                    for it in items])
    max_abs = float(np.abs(p32 - P64).max())
    rec = {"which": which, "scored": "eval", "t_applied": t_applied, "n": len(rows),
           "argmax_agree": int((P64.argmax(1) == p32.argmax(1)).sum()), "max_abs": max_abs,
           "bound": max(floor, k_mult * max_abs),
           "rows": [{"row": i, "probabilities_f64": [float(v) for v in P64[i]]} for i in range(len(rows))]}
    return rec, ctrl, ctrl_rows


def write_noise_record(out, base, eval_rows, question, labels):
    """rescore-noise.json (laya-finetune-gate-v1 rescore_noise_schema, laya-parity-v1 A1) for the shipped
    checkpoint (a fresh reload of <out>/checkpoint) and the declared base, each over every eval row; k and
    the floor READ from laya-parity-v1. A non-zero control refuses the run (exit 2) with no record and
    no gate report. The trainer only records: the Rust verifier derives and enforces the bound."""
    k_mult, floor, ceiling = contract.noise_policy()
    ck = out / "checkpoint"

    def base_fp32():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return load_for_scoring(base.src, base.digest, base.revision)
    sets, ctrl_rows = [], None
    for which, load, name in (("fine_tuned", lambda: reload_checked(ck, tree_sha256(ck)), "eval-probs.json"),
                              ("zero_shot", base_fp32, "zero-shot-probs.json")):
        agent = load()
        rec, ctrl, ctrl_rows = rescore_noise_set(agent, which, eval_rows, question, out / name, labels, k_mult, floor)
        del agent
        gc.collect()
        if rec is None:
            refuse("REFUSED noise-control: the manual fp32 forward differs from the Scorer's logits for %s "
                   "(max |dz| %r on rows %s); a float64 record would describe another model, so none is written"
                   % (which, ctrl, ctrl_rows))
        log("NOISE which=%s max_abs=%.6e bound=%.6e argmax=%d/%d t_applied=%.6f"
            % (which, rec["max_abs"], rec["bound"], rec["argmax_agree"], rec["n"], rec["t_applied"]))
        if rec["bound"] > ceiling or rec["argmax_agree"] < rec["n"]:
            log("WARN pack will refuse: %s bound %.6e (ceiling %g), argmax agree %d/%d -- recorded, not decided here"
                % (which, rec["bound"], ceiling, rec["argmax_agree"], rec["n"]))
        sets.append(rec)
    write_json(out / "rescore-noise.json", {
        "schema": "laya-rescore-noise-v1", "reference": "float64", "k": k_mult, "floor_abs": floor,
        "control_max_abs": 0.0, "control_rows": ctrl_rows, "sets": sets})
    return data.sha256_file(out / "rescore-noise.json")


def score_shift_probe(out, base, shift_rows, question, labels, f_avg_labels):
    """laya-finetune-gate-v1 eval_set.shift_probe_rule (A2): the shipped checkpoint (a fresh reload, at its
    applied T) and the declared base on shift.jsonl -> shift-probs.json / shift-zero-shot-probs.json
    (eval_probs_schema), macro-F1, F_avg and house ECE through metrics.py on exactly the written
    probabilities. Returns the gate-report `shift_probe` block, `gate_clause` false."""
    ck = out / "checkpoint"
    bins = int(contract.thresholds()["ece_bins"])
    y = np.array([lab for _, lab in shift_rows])
    as64 = lambda P: np.array([f32_list(r) for r in P], dtype=np.float64)  # noqa: E731
    agent = reload_checked(ck, tree_sha256(ck))
    P_ft, _ = score_rows(agent, shift_rows, question)
    del agent
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        zs_agent = load_for_scoring(base.src, base.digest, base.revision)
    P_zs, _ = score_rows(zs_agent, shift_rows, question)
    del zs_agent
    gc.collect()
    write_json(out / "shift-probs.json", eval_probs_obj(labels, shift_rows, P_ft))
    write_json(out / "shift-zero-shot-probs.json", eval_probs_obj(labels, shift_rows, P_zs))
    ft_P, zs_P = as64(P_ft), as64(P_zs)
    fav = lambda P: None if f_avg_labels is None else metrics.f_avg(P, y, list(f_avg_labels))  # noqa: E731
    zs = {"macro_f1": metrics.macro_f1(zs_P, y), "f_avg": fav(zs_P), "ece": metrics.ece_top_label(zs_P, y, bins)}
    ft = {"macro_f1": metrics.macro_f1(ft_P, y), "f_avg": fav(ft_P), "ece_post": metrics.ece_top_label(ft_P, y, bins)}
    probe = {"gate_clause": False, "n": len(shift_rows), "zero_shot": zs, "fine_tuned": ft,
             "margin": ft["macro_f1"] - zs["macro_f1"],
             "probs_sha256": data.sha256_file(out / "shift-probs.json"),
             "zero_shot_probs_sha256": data.sha256_file(out / "shift-zero-shot-probs.json")}
    fmt = lambda v: "null" if v is None else "%.4f" % v  # noqa: E731
    log("SHIFT PROBE n=%d ft macro_f1=%s f_avg=%s ece_post=%s | zs macro_f1=%s ece=%s | margin=%s "
        "(reported, not a gate clause)" % (len(shift_rows), fmt(ft["macro_f1"]), fmt(ft["f_avg"]), fmt(ft["ece_post"]),
                                          fmt(zs["macro_f1"]), fmt(zs["ece"]), fmt(probe["margin"])))
    return probe


class SeedRun:
    """Steps 4-6 for one seed on fixed data, split, recipe and base: train -> COMPLETE checkpoint dir ->
    reload -> calibrate -> reload -> eval probabilities. Used for the declared seed and every variance seed."""

    def __init__(self, base, requested, question, fit_rows, calib_rows, eval_rows, k, epochs, stopping, recipe_id):
        self.base, self.requested, self.question = base, requested, question
        self.fit_rows, self.calib_rows, self.eval_rows = fit_rows, calib_rows, eval_rows
        self.k, self.epochs, self.stopping, self.recipe_id = k, epochs, stopping, recipe_id

    def train_and_score(self, seed, ck):
        agent, info = train_seed(seed, self.base, self.requested, self.question, self.fit_rows, self.k,
                                 self.epochs, self.stopping, self.calib_rows)
        provenance = {"fine_tuned_from": self.base.block, "recipe_id": self.recipe_id, "seed": seed,
                      "epochs": self.epochs, "steps": info["steps"], "fit_rows": info["fit_rows"],
                      "device_used": info["device_used"]}
        if info["stopping_record"] is not None:
            provenance["stopping"] = info["stopping_record"]
        sha_before = write_checkpoint(agent, self.base.directory(), ck, provenance)
        del agent
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        log("CHECKPOINT COMPLETE seed=%d %d files (model.safetensors %s)"
            % (seed, len(sha_before), sha_before["model.safetensors"]))
        log("SCORING START seed=%d" % seed)
        agent, P_ft, P_pre, calib, _ = calibrate_and_score(ck, sha_before, self.calib_rows, self.eval_rows,
                                                           self.question, self.k, self.stopping)
        log("CALIBRATION seed=%d bucket=%s t_pre=%.6f t_fitted=%.6f t_applied=%.6f clamp_hit=%s slice=%d"
            % (seed, calib["bucket"], calib["t_pre"], calib["t_fitted"], calib["t_applied"], calib["clamp_hit"],
               len(self.calib_rows)))
        return agent, P_ft, P_pre, calib, info


def variance_row(seed, g, calib, info):
    ft = g["fine_tuned"]
    st = info["stopping_record"] or {}
    return {"seed": seed, "macro_f1": ft["macro_f1"], "f_avg": ft["f_avg"], "ece_pre": ft["ece_pre"],
            "ece_post": ft["ece_post"], "nll": ft["nll"], "margin": g["margin"], "pass": g["pass"],
            "t_fitted": calib["t_fitted"], "t_applied": calib["t_applied"], "clamp_hit": calib["clamp_hit"],
            "epochs_run": info["epochs_run"], "best_epoch": st.get("best_epoch"), "device_used": info["device_used"],
            "train_seconds": info["train_seconds"]}


def variance_report(declared, seeds, rows, recipe_id, zero_shot, shipped):
    """variance-report.json: per-seed rows and mean / sample sd (ddof 1). INFORMATION: the shipped seed was
    chosen by the median rule (seed_policy.selection) before this report exists; nothing here selects."""
    def stat(key, fn):
        vals = [r[key] for r in rows]
        if any(v is None or not math.isfinite(v) for v in vals):
            return None
        return float(fn(np.array(vals, dtype=np.float64)))
    keys = ("macro_f1", "f_avg", "ece_post", "margin")
    return {"schema": "laya-variance-report-v1", "declared_seed": declared, "seeds": list(seeds), "n": len(seeds),
            "label": contract.seeds_label(len(seeds)), "recipe_id": recipe_id,
            "note": "information only: seed %d ships as the median-ECE seed of %s (seed_policy.selection %s, A3); "
                    "the gate is judged on that seed alone, and mean / sd never select anything"
                    % (shipped, list(seeds), contract.seed_selection_decl()["policy"]),
            "sd_ddof": 1, "zero_shot": zero_shot, "per_seed": rows,
            "mean": {k: stat(k, np.mean) for k in keys},
            "sd": {k: stat(k, lambda a: a.std(ddof=1)) for k in keys}}


# ------------------------------------------------------------------------------------------ main

def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True, help="data dir: task.json, train.jsonl, eval.jsonl")
    ap.add_argument("--out", required=True, help="run dir to create (must not exist or be empty)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--stopping", choices=("early_stopping", "fixed_epochs"), default=None,
                    help="stopping rule (default: the contract's recipe.stopping_default)")
    ap.add_argument("--seeds", type=int, default=None,
                    help="production: exactly the contract's production_seeds_required (default); the median-ECE "
                         "seed ships (A3). synthetic-fixture: 1 (default, legacy rule) or that number")
    ap.add_argument("--device", choices=("mps", "cuda", "cpu"), default=None,
                    help="force a device (default: the contract's device_order, first available)")
    ap.add_argument("--variant", choices=("production", "synthetic-fixture"), default="production")
    ap.add_argument("--base", default=None, help="synthetic-fixture only: a local Laya checkpoint dir")
    ap.add_argument("--base-sha256", default=None, help="synthetic-fixture only: its model.safetensors sha256")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    t_start = time.time()
    data_dir, out = Path(args.data), Path(args.out)

    # 1. data (and every recipe and contract-value refusal, before any model loads or any file is written)
    try:
        contract.check_numbers()
        seed = contract.declared_seed()
        seeds = contract.resolve_seeds(args.seeds, args.variant)
        task = data.load_task(data_dir / "task.json")
        train_rows = data.load_rows(data_dir / "train.jsonl", task, "train")
        eval_rows = data.load_rows(data_dir / "eval.jsonl", task, "eval")
        data.refuse_overlap(train_rows, eval_rows)
        shift_rows = None
        if (data_dir / "shift.jsonl").is_file():      # the OPTIONAL shift probe (A2): reported, never a gate clause
            shift_rows = data.load_rows(data_dir / "shift.jsonl", task, "shift")
            data.refuse_overlap(train_rows, shift_rows, "shift")
        # The inputs this run TRAINS on, bound now: the report's inputs_sha256 and the run dir's task.json copy
        # are taken from these, never re-read from the data dir hours later (a mid-run edit would otherwise be
        # bound to a model that never saw it). The data dir is re-checked against them before the report.
        task_bytes = (data_dir / "task.json").read_bytes()
        input_sha = {"task.json": sha256_bytes(task_bytes)}
        for name in ("train.jsonl", "eval.jsonl") + (("shift.jsonl",) if shift_rows is not None else ()):
            input_sha[name] = data.sha256_file(data_dir / name)
        fit_ids, calib_ids, slice_ids, slice_sha = data.calibration_split(
            train_rows, contract.constant("calibration_slice_fraction", "float"),
            contract.constant("calibration_slice_min_per_class", "int"), seed,
            len(task["labels"]))
        epochs = None
        shots_per_class = max(data.class_counts(train_rows, len(task["labels"])))
        epochs = contract.resolve_epochs(args.variant, shots_per_class, args.epochs)
        stopping = contract.resolve_stopping(args.stopping)
    except (data.DataError, contract.RecipeError, contract.ContractValueError) as e:
        refuse(str(e))
    base = Base(args)
    if out.exists() and any(out.iterdir()):
        refuse("REFUSED out-dir: %s exists and is not empty; a run dir is written once" % out)
    labels, k = task["labels"], len(task["labels"])
    question = data.laya_question(task)
    log("DATA task=%s K=%d train=%d eval=%d shift=%s shots_per_class=%d fit=%d calibration=%d seeds=%s"
        % (data_dir / "task.json", k, len(train_rows), len(eval_rows), "none" if shift_rows is None else len(shift_rows),
           shots_per_class, len(fit_ids), len(calib_ids), ",".join(str(s) for s in seeds)))

    # 2. recipe first (seed_selection whenever three seeds run -- every production run, A3)
    selection = contract.seed_selection_decl() if len(seeds) > 1 else None
    out.mkdir(parents=True, exist_ok=True)
    recipe = contract.recipe_json(args.variant, shots_per_class, epochs, seed, base.block, stopping, selection)
    recipe_bytes = json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode("utf-8")
    (out / "recipe.json").write_bytes(recipe_bytes)
    recipe_id = sha256_bytes(recipe_bytes)
    log("RECIPE WRITTEN %s stopping=%s epochs%s=%d" % (recipe_id, stopping, "_max" if stopping == "early_stopping"
                                                        else "", epochs))

    requested = request_device(args.device)
    fit_rows = [train_rows[i] for i in fit_ids]
    calib_rows = [train_rows[i] for i in calib_ids]
    ck = out / "checkpoint"
    run = SeedRun(base, requested, question, fit_rows, calib_rows, eval_rows, k, epochs, stopping, recipe_id)
    y = np.array([lab for _, lab in eval_rows])
    as64 = lambda P: np.array([f32_list(r) for r in P], dtype=np.float64)  # noqa: E731
    demo = contract.demo()
    f_avg_labels = None
    if labels == list(demo["criteria_order"]):
        f_avg_labels = [labels.index("against"), labels.index("favor")]

    def score_zero_shot():
        """7. zero-shot: the declared base on eval.jsonl, loaded the same way; task.json copied."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            zs_agent = load_for_scoring(base.src, base.digest, base.revision)
        P, _ = score_rows(zs_agent, eval_rows, question)
        del zs_agent
        write_json(out / "zero-shot-probs.json", eval_probs_obj(labels, eval_rows, P))
        (out / "task.json").write_bytes(task_bytes)
        return P

    if len(seeds) == 1:
        # 3-8, LEGACY single-seed rule (synthetic-fixture only since 1.4.0): the declared seed ships.
        agent, P_ft, P_pre, calib, info = run.train_and_score(seed, ck)
        write_json(out / "eval-probs.json", eval_probs_obj(labels, eval_rows, P_ft))
        assert_eval_after_checkpoint(out / "recipe.json", ck, out / "eval-probs.json")
        write_json(out / "probes.json", probes_obj(agent))
        del agent
        P_zs = score_zero_shot()
        g = gate.evaluate_gate(as64(P_zs), as64(P_ft), P_pre, y, f_avg_labels)
        seeds_block = {"declared": seed, "n": 1, "label": contract.seeds_label(1)}
    else:
        # 3-8, MEDIAN rule (A3): every seed trains in seeds/seed-<s>/ on the same data, split and recipe; each
        # seed's checkpoint and stopping record exist before its eval probabilities (asserted per seed).
        runs = {}
        for s in seeds:
            sdir = out / "seeds" / ("seed-%d" % s)
            ck_s = sdir / "checkpoint"
            agent, P_s, P_s_pre, cal_s, info_s = run.train_and_score(s, ck_s)
            del agent
            gc.collect()
            write_json(sdir / "eval-probs.json", eval_probs_obj(labels, eval_rows, P_s))
            assert_eval_after_checkpoint(out / "recipe.json", ck_s, sdir / "eval-probs.json")
            runs[s] = {"P": P_s, "P_pre": P_s_pre, "calib": cal_s, "info": info_s,
                       "model_sha": data.sha256_file(ck_s / "model.safetensors"),
                       "probs_sha": data.sha256_file(sdir / "eval-probs.json")}
        P_zs = score_zero_shot()
        # Selection reads eval only now, after every seed's checkpoint is fixed (A3).
        rank_scale = selection["rank_scale"]
        per_seed, var_rows = [], []
        for s in seeds:
            r = runs[s]
            r["g"] = g_s = gate.evaluate_gate(as64(P_zs), as64(r["P"]), r["P_pre"], y, f_avg_labels)
            ft = g_s["fine_tuned"]
            try:
                rk = gate.rank_key(ft["ece_post"], rank_scale)
            except gate.GateError as e:
                refuse("%s (seed %d)" % (e, s))
            per_seed.append({"seed": s, "macro_f1": ft["macro_f1"], "f_avg": ft["f_avg"], "ece_post": ft["ece_post"],
                             "margin": g_s["margin"], "pass": g_s["pass"], "t_applied": r["calib"]["t_applied"],
                             "rank_key": rk, "eval_probs_sha256": r["probs_sha"],
                             "model_safetensors_sha256": r["model_sha"]})
            var_rows.append(variance_row(s, g_s, r["calib"], r["info"]))
            log("SEED seed=%d macro_f1=%.4f ece_post=%.6f margin=%.4f pass=%s rank_key=%d"
                % (s, ft["macro_f1"], ft["ece_post"], g_s["margin"], g_s["pass"], rk))
        shipped = gate.select_median_seed(per_seed, rank_scale)
        log("MEDIAN seeds=%s rank_keys=%s shipped=%d (policy %s, tie_break %s)"
            % (",".join(str(r["seed"]) for r in per_seed), ",".join(str(r["rank_key"]) for r in per_seed), shipped,
               selection["policy"], selection["tie_break"]))
        os.replace(out / "seeds" / ("seed-%d" % shipped) / "checkpoint", ck)
        for s in seeds:
            if s != shipped:
                shutil.rmtree(out / "seeds" / ("seed-%d" % s) / "checkpoint")
                log("DELETED seed=%d checkpoint" % s)
        shutil.copyfile(out / "seeds" / ("seed-%d" % shipped) / "eval-probs.json", out / "eval-probs.json")
        assert_eval_after_checkpoint(out / "recipe.json", ck, out / "eval-probs.json")
        if data.sha256_file(ck / "model.safetensors") != runs[shipped]["model_sha"]:
            raise RuntimeError("the shipped checkpoint's model.safetensors changed when it moved to checkpoint/")
        agent = reload_checked(ck, tree_sha256(ck))
        write_json(out / "probes.json", probes_obj(agent))
        del agent
        g, calib, info = runs[shipped]["g"], runs[shipped]["calib"], runs[shipped]["info"]
        vr = variance_report(seed, seeds, var_rows, recipe_id, g["zero_shot"], shipped)
        write_json(out / "variance-report.json", vr)
        fmt_ms = lambda key: "n/a" if vr["mean"][key] is None else "%.4f ± %.4f" % (vr["mean"][key], vr["sd"][key])  # noqa: E731
        log("VARIANCE %s: macro_f1 %s | f_avg %s | ece_post %s | margin %s -- information; seed %d ships (median)"
            % (vr["label"], fmt_ms("macro_f1"), fmt_ms("f_avg"), fmt_ms("ece_post"), fmt_ms("margin"), shipped))
        seeds_block = {"declared": seed, "n": len(seeds),
                       "label": contract.seeds_label(len(seeds), selection["policy"]),
                       "policy": selection["policy"], "shipped": shipped, "per_seed": per_seed}

    # A1: the float64 re-score noise record of the shipped checkpoint and the base (after the gate is decided).
    noise_sha = write_noise_record(out, base, eval_rows, question, labels)

    # A2: the shift probe, scored only now -- after the gate, the median and the noise record -- and never read
    # by `pass`, the per-seed rows or the exit code.
    shift_probe = None
    if shift_rows is not None:
        shift_probe = score_shift_probe(out, base, shift_rows, question, labels, f_avg_labels)

    report = {
        "schema": "laya-gate-report-v1",
        "pass": g["pass"],
        "thresholds": g["thresholds"],
        "zero_shot": g["zero_shot"],
        "fine_tuned": g["fine_tuned"],
        "margin": g["margin"],
        "calibration": {"bucket": calib["bucket"], "t_fitted": calib["t_fitted"], "t_applied": calib["t_applied"],
                        "clamp_hit": calib["clamp_hit"], "slice_size": len(slice_ids), "slice_ids": slice_ids,
                        "slice_ids_sha256": slice_sha},
        "seeds": seeds_block,
        "device_used": info["device_used"],
        "device_is_cpu": info["device_used"] == "cpu",
        "torch_version": torch.__version__,
        "recipe_id": recipe_id,
        "inputs_sha256": {"task_json": input_sha["task.json"],
                          "train_jsonl": input_sha["train.jsonl"],
                          "eval_jsonl": input_sha["eval.jsonl"],
                          "base_model": base.digest,
                          "tokenizer_json": data.sha256_file(ck / "tokenizer" / "tokenizer.json")},
        "eval_probs_sha256": data.sha256_file(out / "eval-probs.json"),
        "zero_shot_probs_sha256": data.sha256_file(out / "zero-shot-probs.json"),
        "probes_sha256": data.sha256_file(out / "probes.json"),
        "rescore_noise_sha256": noise_sha,
    }
    if shift_probe is not None:
        report["inputs_sha256"]["shift_jsonl"] = input_sha["shift.jsonl"]
        report["shift_probe"] = shift_probe
    # The data dir must still hold exactly what was trained on: a file edited (or a shift.jsonl added / removed)
    # during the run is refused with no report, never bound to this model.
    refuse_if_data_changed(data_dir, input_sha)
    write_json(out / "gate-report.json", report)

    zs, ft = g["zero_shot"], g["fine_tuned"]
    fmt = lambda v: "null" if v is None else "%.4f" % v  # noqa: E731
    log("RESULT zero_shot macro_f1=%s f_avg=%s ece=%s | fine_tuned macro_f1=%s f_avg=%s ece_pre=%s ece_post=%s "
        "nll=%s | margin=%s (need >= %s) ece_post (need <= %s)"
        % (fmt(zs["macro_f1"]), fmt(zs["f_avg"]), fmt(zs["ece"]), fmt(ft["macro_f1"]), fmt(ft["f_avg"]),
           fmt(ft["ece_pre"]), fmt(ft["ece_post"]), fmt(ft["nll"]), fmt(g["margin"]),
           g["thresholds"]["min_macro_f1_margin"], g["thresholds"]["max_ece"]))
    if f_avg_labels is not None and args.variant == "production":
        log("INFO spike 024 baseline F_avg @%d: %s" % (demo["shots_per_class"], demo["spike_baseline_f_avg"]))
    log("SEEDS %s | shipped=%s device_used=%s torch=%s train_seconds=%s total_seconds=%.1f"
        % (report["seeds"]["label"], report["seeds"].get("shipped", seed), info["device_used"], torch.__version__,
           info["train_seconds"], time.time() - t_start))
    if report["pass"]:
        log("GATE PASS")
        return EXIT_PASS
    log("GATE FAIL")
    return EXIT_GATE_FAIL


if __name__ == "__main__":
    sys.exit(main())
