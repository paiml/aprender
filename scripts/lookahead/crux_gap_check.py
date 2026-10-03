#!/usr/bin/env python3
"""EXT-24b draft rule: a `crux_comparison:` block on a CRUX contract (la-0.71).

Reuses two vocabularies already on main rather than inventing new ones:
  gap_effect  closes | widens | none        PR-REVIEW-SKILL-002 §3.C
  comparator  {command, version, env_sha256, artifact_sha256, log_path}  §3.C.1

Rules (EXT-001 R-15..R-17, S-12..S-14):
  G1  gap_effect is in the §3.C vocabulary
  G2  gap_effect closes|widens needs a full comparator on every arm it cites
  G3  an unrunnable-hermetically arm is report-only: it cannot carry closes|widens
  G4  a conceded arm (R-16) cannot be measured on a speed metric
  G5  an arm whose quant differs from the subject's must be labelled separate
  G6  gap_effect agrees with the recorded gaps and metric direction (R-15 delta)
  G7  a runnable arm pins its artifact sha256
  G8  every arm's competitor is in the CRUX_COMPETITORS registry

Exit 0 means GREEN, exit 2 means RED with one line per violation.
--self-test runs fixtures/crux_gap/.
"""
import json
import sys
from pathlib import Path

import yaml

EFFECTS = {"closes", "widens", "none"}
COMPARATOR = ("command", "version", "env_sha256", "artifact_sha256", "log_path")
SPEED = {"ttft_ms", "itl_ms", "e2e_ms", "load_ms", "tok_s", "train_step_s"}
UNRUNNABLE = "unrunnable-hermetically"


def arm_errors(arm, subject_quant, registry):
    name = arm.get("competitor", "?")
    errs = []
    if name not in registry:
        errs.append(f"G8 {name}: competitor not registered")
    comp = arm.get("comparator")
    if comp != UNRUNNABLE and not (arm.get("pin") or {}).get("sha256"):
        errs.append(f"G7 {name}: runnable arm without pin.sha256")
    quant = arm.get("quant")
    if quant and subject_quant and quant != subject_quant and not arm.get("separate_arm"):
        errs.append(f"G5 {name}: quant {quant} vs subject {subject_quant}, not labelled separate_arm")
    return errs


def comparator_full(comp):
    return isinstance(comp, dict) and all(comp.get(k) for k in COMPARATOR)


def effect_errors(block, arms):
    effect = block.get("gap_effect")
    if effect not in EFFECTS:
        return [f"G1 gap_effect {effect!r} not in {sorted(EFFECTS)}"]
    if effect == "none":
        return []
    errs = []
    for arm in arms:
        name, comp = arm.get("competitor", "?"), arm.get("comparator")
        if comp == UNRUNNABLE:
            errs.append(f"G3 {name}: unrunnable-hermetically arm carries gap_effect {effect}")
        elif not comparator_full(comp):
            errs.append(f"G2 {name}: gap_effect {effect} without a full comparator {list(COMPARATOR)}")
    return errs


def speed_errors(metric, arms):
    if metric.get("name") not in SPEED:
        return []
    return [f"G4 {a.get('competitor', '?')}: conceded arm on speed metric {metric['name']}"
            for a in arms if a.get("conceded")]


def expected_effect(gap_prev, gap_now, direction):
    if gap_prev is None or gap_now == gap_prev:
        return "none"
    worse = gap_now > gap_prev if direction == "lower" else gap_now < gap_prev
    return "widens" if worse else "closes"


def delta_errors(block, metric):
    gaps = block.get("gap") or {}
    if block.get("gap_effect") not in EFFECTS or "now" not in gaps:
        return []
    want = expected_effect(gaps.get("previous"), gaps["now"], metric.get("direction"))
    if want != block["gap_effect"]:
        return [f"G6 gap_effect {block['gap_effect']} but gap {gaps.get('previous')}->{gaps['now']} "
                f"({metric.get('direction')}-is-better) is {want}"]
    return []


def evaluate(doc, registry):
    block = (doc or {}).get("crux_comparison")
    if not isinstance(block, dict):
        return ["G0 no crux_comparison block"]
    arms = block.get("arms") or []
    if not arms:
        return ["G0 crux_comparison has no arms (vacuous)"]
    metric = block.get("metric") or {}
    errs = []
    for arm in arms:
        errs += arm_errors(arm, block.get("subject_quant"), registry)
    errs += effect_errors(block, arms)
    errs += speed_errors(metric, arms)
    errs += delta_errors(block, metric)
    return errs


def self_test(fix):
    registry = set(json.loads((fix / "registry.json").read_text()))
    cases = json.loads((fix / "cases.json").read_text())
    bad = 0
    for c in cases:
        errs = evaluate(yaml.safe_load((fix / c["fixture"]).read_text()), registry)
        rule_ok = not c.get("rule") or any(e.startswith(c["rule"]) for e in errs)
        if (2 if errs else 0) != c["expect"] or not rule_ok:
            bad += 1
            print(f"FAIL {c['id']}: expect {c['expect']} {c.get('rule', '')} got {errs}")
    print(f"self-test: {len(cases) - bad}/{len(cases)} ok")
    return 0 if bad == 0 else 1


def main(argv):
    root = Path(__file__).resolve().parents[2]
    if argv[1:2] == ["--self-test"]:
        return self_test(root / "fixtures/crux_gap")
    sys.path.insert(0, str(root / "scripts/lookahead"))
    from crux_bind_check import registry as parse_registry
    reg = parse_registry((root / "crates/aprender-contracts/src/schema/validator.rs").read_text())
    errs = evaluate(yaml.safe_load(Path(argv[1]).read_text()), reg or set())
    for e in errs:
        print(e)
    print("crux-gap:", "RED" if errs else "GREEN")
    return 2 if errs else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
