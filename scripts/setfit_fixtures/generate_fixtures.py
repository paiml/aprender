#!/usr/bin/env python3
"""Freeze the entire ENC-01..06 fixture corpus in ONE deterministic pass (D-15).

Run:  uv run python slice_model.py && uv run python generate_fixtures.py
Developer workflow only -- never wired into CI (D-12).

A SECOND, INDEPENDENT MODE lives at the bottom of this file (Phase 2, plan 02-04):

    uv run python generate_fixtures.py --pairs [--rebaseline]

It emits the SetFit pair-count reference fixtures into
crates/aprender-contrastive-data/tests/setfit_reference/ and touches NOTHING under
crates/aprender-core/tests/fixtures/setfit/. It needs no torch model and no network. See
the section banner above ``PAIR_FIXTURE_DIR`` for why there are two fixture families.

WHY ONE PASS
------------
Every slice-driven fixture must come from the SAME sliced torch model that produced the
committed slice_model.apr, otherwise "Rust disagrees with Python" becomes ambiguous
between a real defect and two different reference models. So this script loads
build/slice_model.safetensors -- the exact bytes slice_model.py converted to APR -- and
derives every slice fixture from that one object.

DROPOUT IS INERT (D-16)
-----------------------
Both dropout probabilities are set to 0.0 AND the model is put in eval() mode. Python and
Rust RNG streams are never compared bit-for-bit; Rust-side dropout placement,
determinism and statistics are tested separately.

ASSUMPTION A1, RESOLVED FROM THE LOCKED SOURCE (not from memory)
-----------------------------------------------------------------
Read out of sentence_transformers 5.7.0 in this very environment:

  Pooling._forward_padded (sentence_transformer/modules/pooling.py):
      mean_sum  = (token_embeddings * mask).sum(dim=1)
      mean_mask = torch.clamp(mask.sum(dim=1), min=1e-9)      <-- clamp constant 1e-9
      mean      = mean_sum / mean_mask

  Normalize.forward (sentence_transformer/modules/normalize.py):
      F.normalize(sentence_embedding, p=2, dim=1)             <-- torch default eps 1e-12

So the clamp constant is 1e-9 and the normalize eps is 1e-12, both CONFIRMED rather than
assumed. ``full_model_reference`` additionally cross-checks this pipeline against a real
SentenceTransformer forward, so a wrong constant would fail here rather than in wave 6.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from tokenizers import Tokenizer
from transformers import BertConfig, BertModel

import corpus
import jsonfmt
from slice_model import (
    BUILD_DIR,
    FIXTURE_DIR,
    REPO_ID,
    REPO_ROOT,
    REVISION,
    fetch_pinned,
    sha256_file,
)

SEED = 0

# ST pooling/normalize constants, verified from the locked 5.7.0 source (see module docstring).
ST_POOLING_CLAMP_MIN = 1e-9
ST_NORMALIZE_EPS = 1e-12

# AdamW hyperparameters for the single controlled step (ENC-04).
ADAMW = {"lr": 2e-5, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.01}

# D55 — steps in the multi-step trajectory obligation. At step 1 bias correction makes
# the update beta-INDEPENDENT (m_hat = g, v_hat = g^2 for every beta1/beta2), so no
# single-step fixture at any tolerance can constrain the betas. The moments only start
# to carry history from step 2 onward. 20 is where the measured beta separation is
# ~5 orders of magnitude above f32 noise while the trajectory is still cheap to replay
# in Rust; `assert_separation` re-measures it on every regeneration rather than trusting
# this comment.
MULTISTEP_N = 20

# --- tolerance floors ---------------------------------------------------------------
# WHY A FLOOR IS MANDATORY: `10 x observed f32/f64 delta` alone can produce ZERO (the two
# paths coincide on a small case) or a value far tighter than legitimate Rust/PyTorch
# REDUCTION-ORDER differences -- which an f32/f64 round trip does not measure at all,
# because both paths sum in the same order. A zero tolerance silently disables the
# comparison it is supposed to gate (T-1-07).
#
# DERIVATION: differing summation orders over W terms behave like a random walk, so the
# error grows as sqrt(W) * eps_f32 * |x|. With K = 8 as a safety factor on |x| and on the
# walk constant:
#
#     floor(W) = K * sqrt(W) * EPS_F32
#
# W is the characteristic reduction width of the family: 1 for a pointwise activation,
# the sequence length (<=64) for a pooled mean, the projection width for a forward pass
# (256 = the slice FFN intermediate), and a whole-batch backward accumulation for
# gradients / optimizer state. full_model_reference uses the FULL model's 1536-wide FFN.
#
# D55 — `optimizer_step` is W = 1, NOT 1024, and that is not a typo. A post-step
# PARAMETER is not a 1024-wide reduction of anything. At step 1 bias correction gives
# m_hat = g and v_hat = g^2, so the update is lr*g/(|g|+eps) -- it SATURATES to
# lr*sign(g), and gradient reduction-order noise therefore does not propagate into it at
# all. Inheriting the gradient family's W made the floor 3.05e-05 while the entire
# displacement being gated is ~lr = 2.01e-05: a floor 1.5x the signal, i.e. a gate that
# cannot fail. The residual really is pointwise f32 representation noise at the
# parameter magnitude, which is why this family also passes an explicit `scale`.
EPS_F32 = 1.1920928955078125e-07
FLOOR_K = 8.0
FAMILY_REDUCTION_WIDTH = {
    "activation": 1,
    "forward_per_layer": 256,
    "pooling_normalize": 64,
    "loss_pair": 64,
    "gradients": 1024,
    "optimizer_step": 1,
    # A trajectory of pair losses: each entry is one loss_pair reduction.
    "optimizer_multistep": 64,
    "batch_invariance": 64,
    "full_model_reference": 1536,
}


def family_floor(family: str, scale: float = 1.0) -> float:
    return FLOOR_K * math.sqrt(FAMILY_REDUCTION_WIDTH[family]) * EPS_F32 * scale


def record_tolerance(
    tolerances: dict, family: str, delta: float, scale: float = 1.0
) -> None:
    """Record one family's measured delta, floor and recommended tolerance.

    The family name is spelled ONCE per call site. The previous form repeated it
    three times per block (dict key plus two `family_floor` arguments) across eight
    near-identical blocks, so a mismatch between the key and the floor lookup would
    silently record the wrong reduction width -- and `recommended_tolerance` is
    exactly what the Rust conformance gates load.

    `scale` is the MAGNITUDE the family's comparison actually lives at. It is 1.0 for
    every family that compares normalized or O(1) quantities, and is passed explicitly
    by the optimizer families, which compare raw parameter values. Leaving it implicit
    is what produced the vacuous D55 tolerance.
    """
    floor = family_floor(family, scale)
    tolerances[family] = {
        "max_abs_f32_f64_delta": delta,
        "floor": floor,
        "recommended_tolerance": max(10 * delta, floor),
    }


def assert_separation(tolerances: dict, family: str, signal: float, what: str) -> None:
    """Fail generation unless `family`'s tolerance sits 10x BELOW the signal it gates.

    A tolerance at or above the effect it is supposed to resolve is not a loose gate,
    it is an absent one -- the D55 defect in one line. The `activation` family has had
    this guard since 01-04; every family whose tolerance must SEPARATE two behaviours
    (rather than merely absorb round-off) needs it, and the optimizer families are
    exactly that case.
    """
    tol = tolerances[family]["recommended_tolerance"]
    if tol * 10 > signal:
        sys.exit(
            f"FATAL: {family} tolerance {tol:.6e} does not sit 10x below {what} "
            f"({signal:.6e}); margin is {signal / tol:.2f}x. A gate at this tolerance "
            "cannot distinguish the behaviour it claims to gate."
        )
    print(f"  {family}: tol {tol:.6e} separates {what} ({signal:.6e}) by {signal / tol:.1f}x")


def flat(t: torch.Tensor) -> list[float]:
    """Row-major flattened f32 values."""
    return [float(v) for v in t.detach().to(torch.float32).reshape(-1).tolist()]


def max_abs_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach().to(torch.float64) - b.detach().to(torch.float64)).abs().max())


# ------------------------------------------------------------------ tokenizer ------
def load_tokenizer() -> Tokenizer:
    return Tokenizer.from_file(str(FIXTURE_DIR / "tokenizer.json"))


def encode_case(tok: Tokenizer, texts: list[str], max_length: int) -> dict:
    """Tokenize a batch, padding to LONGEST-IN-BATCH (never padding='max_length').

    max_length is a TRUNCATION bound only. 01-05 Task 2 asserts exact integer equality
    against these arrays with no tolerance, so a max_length-padded corpus would fail at
    wave 3 and force a wave-2 fixture + manifest + tolerance regeneration.
    """
    tok.no_truncation()
    tok.no_padding()
    raw = tok.encode_batch(list(texts))
    original_counts = [len(e.ids) for e in raw]

    tok.enable_truncation(max_length=max_length)
    tok.enable_padding(pad_id=0, pad_token="[PAD]", pad_type_id=0)  # length=None => longest
    enc = tok.encode_batch(list(texts))
    tok.no_truncation()
    tok.no_padding()

    return {
        "texts": list(texts),
        "max_length": max_length,
        "input_ids": [list(e.ids) for e in enc],
        "token_type_ids": [list(e.type_ids) for e in enc],
        "attention_mask": [list(e.attention_mask) for e in enc],
        "truncated": [c > max_length for c in original_counts],
        "original_token_counts": original_counts,
    }


# ------------------------------------------------------------------ slice model ----
def slice_config() -> dict:
    return json.loads((FIXTURE_DIR / "slice_config.json").read_text())


def vocab_remap() -> dict[str, int]:
    return json.loads((FIXTURE_DIR / "vocab_remap.json").read_text())["orig_to_slice"]


def build_slice_model(dtype: torch.dtype = torch.float32) -> BertModel:
    """Fresh slice model from the EXACT bytes slice_model.py converted to APR."""
    cfg_json = slice_config()
    cfg = BertConfig(
        vocab_size=cfg_json["vocab"],
        hidden_size=cfg_json["hidden"],
        num_hidden_layers=cfg_json["num_layers"],
        num_attention_heads=cfg_json["heads"],
        intermediate_size=cfg_json["intermediate"],
        max_position_embeddings=cfg_json["positions"],
        type_vocab_size=cfg_json["type_vocab_size"],
        layer_norm_eps=cfg_json["layer_norm_eps"],
        hidden_act=cfg_json["hidden_act"],
        pad_token_id=cfg_json["pad_token_id"],
        position_embedding_type="absolute",
        # D-16: dropout inert. eval() below would suffice; 0.0 removes all doubt.
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    torch.manual_seed(SEED)
    model = BertModel(cfg, add_pooling_layer=False)
    sd = load_file(str(BUILD_DIR / "slice_model.safetensors"))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    real_missing = [k for k in missing if not k.endswith("position_ids")]
    if real_missing or unexpected:
        sys.exit(f"FATAL: slice state_dict mismatch\n  missing={real_missing}\n  unexpected={unexpected}")
    model.eval()
    return model.to(dtype)


def remap_ids(ids: list[list[int]], remap: dict[str, int]) -> list[list[int]]:
    out = []
    for row in ids:
        remapped = []
        for i in row:
            key = str(i)
            if key not in remap:
                sys.exit(
                    f"FATAL: canonical id {i} is outside the slice vocabulary closure. "
                    "The corpus changed without re-running slice_model.py."
                )
            remapped.append(remap[key])
        out.append(remapped)
    return out


def run_slice(model: BertModel, case: dict, remap: dict[str, int], dtype: torch.dtype):
    """Forward the slice model on a case; returns (hidden_states tuple, mask tensor)."""
    ids_slice = remap_ids(case["input_ids"], remap)
    input_ids = torch.tensor(ids_slice, dtype=torch.long)
    mask = torch.tensor(case["attention_mask"], dtype=torch.long)
    type_ids = torch.tensor(case["token_type_ids"], dtype=torch.long)
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=mask,
            token_type_ids=type_ids,
            output_hidden_states=True,
        )
    return out.hidden_states, mask.to(dtype)


def masked_mean(token_emb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).expand(token_emb.size()).to(token_emb.dtype)
    summed = (token_emb * m).sum(dim=1)
    denom = torch.clamp(m.sum(dim=1), min=ST_POOLING_CLAMP_MIN)
    return summed / denom


def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, p=2, dim=1, eps=ST_NORMALIZE_EPS)


def encode_slice(model: BertModel, case: dict, remap: dict[str, int], dtype: torch.dtype):
    """Graph-connected encode used by the loss/gradient fixtures (no no_grad)."""
    ids_slice = remap_ids(case["input_ids"], remap)
    out = model(
        input_ids=torch.tensor(ids_slice, dtype=torch.long),
        attention_mask=torch.tensor(case["attention_mask"], dtype=torch.long),
        token_type_ids=torch.tensor(case["token_type_ids"], dtype=torch.long),
    )
    mask = torch.tensor(case["attention_mask"], dtype=torch.long).to(dtype)
    return l2_normalize(masked_mean(out.last_hidden_state, mask))


def pair_loss(model: BertModel, ca: dict, cb: dict, remap, dtype) -> torch.Tensor:
    za = encode_slice(model, ca, remap, dtype)
    zb = encode_slice(model, cb, remap, dtype)
    cos = F.cosine_similarity(za, zb, dim=1)
    labels = torch.tensor(corpus.LOSS_PAIR_LABELS, dtype=dtype)
    return F.mse_loss(cos, labels)


def adamw_trajectory(steps, ca, cb, remap, dtype, **overrides):
    """Run `steps` AdamW steps from a FRESH slice model on the recorded pair batch.

    Returns `(post_step, losses)` where `losses[i]` is the loss measured BEFORE step i
    and `losses[steps]` is the loss after the last step, so a trajectory of length
    `steps + 1` brackets every update.

    `overrides` replaces individual ADAMW hyperparameters. The generator uses that to
    run the SAME mutations the conformance gate claims to detect (decay deleted, wrong
    betas) and measure how far each one moves the trajectory -- so the separation this
    fixture can prove is measured, never asserted from a comment.
    """
    hp = {**ADAMW, **overrides}
    model = build_slice_model(dtype)
    params = [p for n, p in model.named_parameters() if not n.startswith("pooler.")]
    opt = torch.optim.AdamW(
        params,
        lr=hp["lr"],
        betas=tuple(hp["betas"]),
        eps=hp["eps"],
        weight_decay=hp["weight_decay"],
    )
    losses = []
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = pair_loss(model, ca, cb, remap, dtype)
        losses.append(float(loss.detach().to(torch.float64)))
        loss.backward()
        opt.step()
    with torch.no_grad():
        losses.append(float(pair_loss(model, ca, cb, remap, dtype).detach().to(torch.float64)))
    post = {n: p for n, p in model.named_parameters() if not n.startswith("pooler.")}
    return post, losses


# ------------------------------------------------------------------ main -----------
def main() -> None:
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)

    cfg_json = slice_config()
    if cfg_json["source_revision"] != REVISION:
        sys.exit("FATAL: slice_config.json pins a different revision than slice_model.py")

    verify_upstream_digests()

    tok = load_tokenizer()
    remap = vocab_remap()
    tolerances: dict[str, dict] = {}

    # ---------------------------------------------------------- corpus of record ---
    cases = {cid: encode_case(tok, texts, corpus.MAX_LENGTH) for cid, texts in corpus.CASES.items()}

    tokenizer_cases = {
        "revision": REVISION,
        "tokenizer_sha256": cfg_json["tokenizer_sha256"],
        "cases": [{"id": cid, **cases[cid]} for cid in corpus.CASES],
    }

    # The slice keeps only `positions` position embeddings; anything longer cannot be
    # driven through it. Asserted, not left to the corpus comment.
    for cid in corpus.SLICE_DRIVEN_CASES:
        seq = len(cases[cid]["input_ids"][0])
        if seq > cfg_json["positions"]:
            sys.exit(
                f"FATAL: case '{cid}' is {seq} tokens but the slice has only "
                f"{cfg_json['positions']} positions."
            )
    if not any(any(c["truncated"]) for c in cases.values()):
        sys.exit("FATAL: no case exercises truncation; the >256-token case is missing.")

    def joined(cid: str) -> dict:
        """Fixture-side view of a case: case_id + verbatim texts + canonical ids (B6)."""
        c = cases[cid]
        return {
            "case_id": cid,
            "texts": list(c["texts"]),
            "input_ids_canonical": [list(r) for r in c["input_ids"]],
            "attention_mask": [list(r) for r in c["attention_mask"]],
        }

    # ---------------------------------------------------------- activation --------
    grid = torch.linspace(-6.0, 6.0, 2401, dtype=torch.float32)
    y_exact = F.gelu(grid, approximate="none")
    y_tanh = F.gelu(grid, approximate="tanh")
    tanh_delta = float((y_exact - y_tanh).abs().max())
    y64 = F.gelu(grid.to(torch.float64), approximate="none")
    act_delta = max_abs_delta(y_exact, y64)

    # The gate must SEPARATE the exact erf form from the tanh approximation, so the
    # activation tolerance has to sit far below the gap between them. MEASURED here at
    # ~4.7e-4 near x = -2.7 -- note this is BELOW the 1e-3 the plan predicted, so the
    # measured value is recorded and the assertion is set from it rather than the other
    # way round (CLAUDE.md: never label a run by intent).
    if tanh_delta <= 1e-4:
        sys.exit(f"FATAL: exact-vs-tanh GELU gap {tanh_delta:.3e} is too small to gate on")

    record_tolerance(tolerances, "activation", act_delta)
    if tolerances["activation"]["recommended_tolerance"] >= tanh_delta / 10:
        sys.exit(
            "FATAL: activation tolerance is not far enough below the exact-vs-tanh gap; "
            "a tanh implementation could pass the gate."
        )

    jsonfmt.write(
        FIXTURE_DIR / "activation_reference.json",
        {
            "op": "gelu_exact",
            "note": (
                "torch.nn.functional.gelu(x, approximate='none') -- the EXACT erf form "
                "0.5*x*(1+erf(x/sqrt(2))), matching the pinned config's hidden_act='gelu'. "
                f"MEASURED max|exact - tanh_approx| over this grid is {tanh_delta:.6e} at the "
                "grid point nearest x=-2.699. The tanh approximation is a DIFFERENT function, "
                "not an acceptable implementation: this gap is ~3 orders of magnitude above "
                "f32 round-trip noise, so the gate separates them instead of absorbing the "
                "difference into tolerance."
            ),
            "approximate": "none",
            "tanh_vs_exact_max_delta": tanh_delta,
            "x": flat(grid),
            "y": flat(y_exact),
            "max_abs_f32_f64_delta": act_delta,
        },
    )

    # ---------------------------------------------------------- forward per layer --
    model32 = build_slice_model(torch.float32)
    model64 = build_slice_model(torch.float64)

    fwd_cases, fwd_delta = [], 0.0
    for cid in ("single_short", "mixed_length_pair"):
        c = cases[cid]
        hs32, _ = run_slice(model32, c, remap, torch.float32)
        hs64, _ = run_slice(model64, c, remap, torch.float64)
        fwd_delta = max(fwd_delta, max_abs_delta(hs32[-1], hs64[-1]))
        b, s, h = hs32[-1].shape
        if len(hs32) != cfg_json["num_layers"] + 1:
            sys.exit("FATAL: expected num_layers+1 hidden states")
        fwd_cases.append(
            {
                **joined(cid),
                "input_ids_slice": remap_ids(c["input_ids"], remap),
                "shape": {"batch": b, "seq": s, "hidden": h},
                "embeddings_out": flat(hs32[0]),
                "layer_outputs": [flat(hs32[i + 1]) for i in range(cfg_json["num_layers"])],
                # Duplicates layer_outputs[-1] on purpose: the duplication is what lets a
                # Rust mismatch name embedding-vs-layer-N-vs-final instead of "the encoder".
                "final_tokens": flat(hs32[-1]),
            }
        )
    record_tolerance(tolerances, "forward_per_layer", fwd_delta)
    jsonfmt.write(FIXTURE_DIR / "forward_per_layer.json", {"cases": fwd_cases})

    # ---------------------------------------------------------- pooling/normalize --
    pool_cases, pool_delta = [], 0.0
    for cid in ("mixed_length_pair", "pooling_batch"):
        c = cases[cid]
        hs32, m32 = run_slice(model32, c, remap, torch.float32)
        hs64, m64 = run_slice(model64, c, remap, torch.float64)
        pooled32 = masked_mean(hs32[-1], m32)
        norm32 = l2_normalize(pooled32)
        pooled64 = masked_mean(hs64[-1], m64)
        pool_delta = max(pool_delta, max_abs_delta(norm32, l2_normalize(pooled64)))
        pool_cases.append(
            {
                **joined(cid),
                "shape": {"batch": pooled32.shape[0], "hidden": pooled32.shape[1]},
                "pooled": flat(pooled32),
                "normalized": flat(norm32),
            }
        )
    record_tolerance(tolerances, "pooling_normalize", pool_delta)
    jsonfmt.write(FIXTURE_DIR / "pooling_normalize.json", {"cases": pool_cases})

    # ---------------------------------------------------------- pair loss ---------
    ca, cb = cases["loss_pair_a"], cases["loss_pair_b"]
    za = encode_slice(model32, ca, remap, torch.float32)
    zb = encode_slice(model32, cb, remap, torch.float32)
    cos32 = F.cosine_similarity(za, zb, dim=1)
    labels = torch.tensor(corpus.LOSS_PAIR_LABELS, dtype=torch.float32)
    mse32 = F.mse_loss(cos32, labels)
    with torch.no_grad():
        za64 = encode_slice(model64, ca, remap, torch.float64)
        zb64 = encode_slice(model64, cb, remap, torch.float64)
        cos64 = F.cosine_similarity(za64, zb64, dim=1)
    loss_delta = max(
        max_abs_delta(cos32, cos64),
        max_abs_delta(mse32, F.mse_loss(cos64, labels.to(torch.float64))),
    )
    record_tolerance(tolerances, "loss_pair", loss_delta)
    jsonfmt.write(
        FIXTURE_DIR / "loss_pair.json",
        {
            "pair": {
                "a_case_id": "loss_pair_a",
                "a_texts": list(ca["texts"]),
                "a_ids_canonical": [list(r) for r in ca["input_ids"]],
                "a_ids_slice": remap_ids(ca["input_ids"], remap),
                "a_mask": [list(r) for r in ca["attention_mask"]],
                "b_case_id": "loss_pair_b",
                "b_texts": list(cb["texts"]),
                "b_ids_canonical": [list(r) for r in cb["input_ids"]],
                "b_ids_slice": remap_ids(cb["input_ids"], remap),
                "b_mask": [list(r) for r in cb["attention_mask"]],
                "labels": list(corpus.LOSS_PAIR_LABELS),
            },
            "cosine": flat(cos32),
            "mse": float(mse32.detach()),
        },
    )

    # ---------------------------------------------------------- gradients ---------
    gmodel = build_slice_model(torch.float32)
    gmodel.zero_grad(set_to_none=True)
    loss = pair_loss(gmodel, ca, cb, remap, torch.float32)
    loss.backward()

    named = [(n, p) for n, p in gmodel.named_parameters() if not n.startswith("pooler.")]
    parameter_order = [n for n, _ in named]
    grads = {}
    max_abs = {}
    for n, p in named:
        if p.grad is None:
            sys.exit(f"FATAL: parameter {n} received no gradient at all")
        g = p.grad.detach()
        if not torch.isfinite(g).all():
            sys.exit(f"FATAL: non-finite gradient on {n}")
        grads[n] = {"shape": list(p.shape), "grad": flat(g)}
        max_abs[n] = float(g.abs().max())

    # zero_grad_floor derivation: the reference gradients separate into a cluster that is
    # numerically zero (|g| ~ 1e-10 and below, i.e. analytically zero perturbed by f32
    # rounding) and a cluster of genuinely non-zero components many orders of magnitude
    # larger. The floor is placed one order of magnitude BELOW the smallest genuinely
    # non-zero max|grad|, so it cannot swallow a real gradient, and the split is read off
    # the observed distribution rather than assumed.
    ordered = sorted(max_abs.values())
    gap_idx = max(
        range(1, len(ordered)),
        key=lambda i: (math.log10(ordered[i] + 1e-300) - math.log10(ordered[i - 1] + 1e-300)),
    )
    smallest_nonzero = ordered[gap_idx]
    zero_grad_floor = smallest_nonzero / 10.0

    analytically_zero = []
    for n in parameter_order:
        if max_abs[n] <= zero_grad_floor:
            analytically_zero.append(
                {
                    "name": n,
                    "max_abs_grad": max_abs[n],
                    "justification": (
                        "the key bias adds the same constant to every key, so for a fixed "
                        "query q_i the term q_i . b_k is identical across all keys j; softmax "
                        "is invariant under a constant shift of all logits in a row, therefore "
                        "dL/db_k = 0 in exact arithmetic"
                        if n.endswith("attention.self.key.bias")
                        else "measured analytically-zero gradient under this loss and batch"
                    ),
                }
            )
    if not analytically_zero:
        sys.exit("FATAL: analytically_zero is empty; the ENC-04 exemption gate would be vacuous")
    if not any(e["name"].endswith("attention.self.key.bias") for e in analytically_zero):
        sys.exit(
            "FATAL: no attention.self.key.bias in analytically_zero. Either the slice or the "
            "loss is degenerate, or the reference gradient is wrong. Do NOT proceed by "
            "deleting the expectation."
        )

    g64 = build_slice_model(torch.float64)
    g64.zero_grad(set_to_none=True)
    pair_loss(g64, ca, cb, remap, torch.float64).backward()
    grad_delta = 0.0
    for (n, p), (_, p64) in zip(named, [(n, p) for n, p in g64.named_parameters() if not n.startswith("pooler.")]):
        grad_delta = max(grad_delta, max_abs_delta(p.grad, p64.grad))
    record_tolerance(tolerances, "gradients", grad_delta)

    source_block = {"fixture": "loss_pair.json", "a_case_id": "loss_pair_a", "b_case_id": "loss_pair_b"}
    jsonfmt.write(
        FIXTURE_DIR / "gradients.json",
        {
            "source": source_block,
            "note": (
                "Gradients of the cosine-similarity/MSE pair loss w.r.t. every slice "
                "parameter, pooler.* excluded. `source` names the exact batch these were "
                "recorded on so the Rust gate rebuilds the SAME batch by tokenizing those "
                "texts. `parameter_order` is torch's named_parameters() iteration order -- "
                "compare against IT, not against JSON object key order, which carries no "
                "ordering guarantee."
            ),
            "parameter_order": parameter_order,
            "zero_grad_floor": zero_grad_floor,
            "analytically_zero": analytically_zero,
            "grads": grads,
        },
    )

    # ---------------------------------------------------------- optimizer step ----
    omodel = build_slice_model(torch.float32)
    oparams = [p for n, p in omodel.named_parameters() if not n.startswith("pooler.")]
    before_step = {
        n: p.detach().clone()
        for n, p in omodel.named_parameters()
        if not n.startswith("pooler.")
    }
    opt = torch.optim.AdamW(
        oparams,
        lr=ADAMW["lr"],
        betas=tuple(ADAMW["betas"]),
        eps=ADAMW["eps"],
        weight_decay=ADAMW["weight_decay"],
    )
    opt.zero_grad(set_to_none=True)
    loss_before = pair_loss(omodel, ca, cb, remap, torch.float32)
    loss_before.backward()
    opt.step()
    with torch.no_grad():
        loss_after = pair_loss(omodel, ca, cb, remap, torch.float32)

    post_step = {
        n: flat(p) for n, p in omodel.named_parameters() if not n.startswith("pooler.")
    }

    # D55 (CLOSED) — the optimizer family measures its OWN f32/f64 delta.
    # It previously reused `grad_delta`, so no f64 optimizer step was ever run and the
    # recorded tolerance (3.052e-05) exceeded the entire step-1 displacement (~lr =
    # 2.01e-05) it was supposed to resolve. The `scale` is the parameter magnitude the
    # comparison actually lives at; see the FAMILY_REDUCTION_WIDTH note on why W = 1.
    post64, _ = adamw_trajectory(1, ca, cb, remap, torch.float64)
    step_delta = 0.0
    for n, p in omodel.named_parameters():
        if n.startswith("pooler."):
            continue
        step_delta = max(step_delta, max_abs_delta(p, post64[n]))
    max_abs_param = max(
        float(p.detach().abs().max())
        for n, p in omodel.named_parameters()
        if not n.startswith("pooler.")
    )
    record_tolerance(tolerances, "optimizer_step", step_delta, scale=max_abs_param)

    # The displacement this gate must resolve. Measured against the pre-step snapshot,
    # not predicted from lr: a step that silently did nothing would otherwise be gated
    # by a tolerance derived from the step it failed to take.
    max_displacement = max(
        float((p.detach() - before_step[n]).abs().max())
        for n, p in omodel.named_parameters()
        if not n.startswith("pooler.")
    )
    assert_separation(
        tolerances, "optimizer_step", max_displacement, "the step-1 displacement"
    )
    jsonfmt.write(
        FIXTURE_DIR / "optimizer_step.json",
        {
            "source": source_block,
            "note": (
                "ALL parameters are trainable in this fixture -- no frozen groups, matching "
                "SetFit's full-body fine-tuning default (D-20). It therefore CANNOT validate "
                "a Rust model configured with frozen groups; 01-08 uses a second, clean model "
                "for the frozen-byte-identity proof. loss_before/loss_after let the "
                "loss-decrease assertion compare against a recorded reference instead of an "
                "unpinned expectation."
            ),
            "adamw": ADAMW,
            "all_trainable": True,
            "loss_before": float(loss_before.detach()),
            "loss_after": float(loss_after.detach()),
            "post_step": post_step,
        },
    )

    # ------------------------------------------------------ optimizer multi-step --
    # D55 — the obligation that makes the BETAS falsifiable.
    #
    # Why a loss TRAJECTORY and not a second post-step parameter dump: the discriminating
    # power of a max-abs parameter comparison is set by its noisiest single element, and
    # both mutations below stay inside that noise at every step count measured (the
    # weight-decay term peaks at 2.6x the f32/f64 delta at N = 50). The loss contracts
    # every parameter into one number and the trajectory accumulates the divergence
    # coherently, which buys ~5 orders of magnitude on the betas -- and it costs 21
    # floats instead of 1.6 MB.
    _, traj32 = adamw_trajectory(MULTISTEP_N, ca, cb, remap, torch.float32)
    _, traj64 = adamw_trajectory(MULTISTEP_N, ca, cb, remap, torch.float64)
    traj_delta = max(abs(a - b) for a, b in zip(traj32, traj64))
    record_tolerance(tolerances, "optimizer_multistep", traj_delta)

    # Run the mutations THIS obligation claims to detect and measure the separation.
    # `assert_separation` then fails generation if the recorded tolerance could not
    # actually tell them apart -- the check that D55 was missing.
    _, traj_betas = adamw_trajectory(
        MULTISTEP_N, ca, cb, remap, torch.float32, betas=[0.5, 0.5]
    )
    betas_signal = max(abs(a - b) for a, b in zip(traj32, traj_betas))
    assert_separation(
        tolerances, "optimizer_multistep", betas_signal, "a betas (0.5, 0.5) trajectory"
    )

    # The decay control is measured and RECORDED but deliberately not asserted here.
    # At this lr/weight_decay the decay term is ~3 f32 ulp of the parameters it acts on,
    # so no tolerance over this fixture can separate it. Deleting decoupled decay is
    # caught instead by `falsify_aw_001_decoupled_weight_decay` (adamw-kernel-v1), which
    # compares AdamW against Adam algebraically rather than against a f32 reference.
    _, traj_nodecay = adamw_trajectory(
        MULTISTEP_N, ca, cb, remap, torch.float32, weight_decay=0.0
    )
    decay_signal = max(abs(a - b) for a, b in zip(traj32, traj_nodecay))

    jsonfmt.write(
        FIXTURE_DIR / "optimizer_multistep.json",
        {
            "source": source_block,
            "note": (
                f"{MULTISTEP_N} consecutive AdamW steps on the SAME recorded pair batch, "
                "from a fresh all-trainable slice model. `losses[i]` is the loss measured "
                "BEFORE step i, and the final entry is the loss after the last step, so the "
                "trajectory brackets every update. This is the only obligation in the phase "
                "that constrains beta1/beta2: at step 1 bias correction makes the update "
                "beta-independent (m_hat = g, v_hat = g^2), so a single-step fixture cannot "
                "constrain them at ANY tolerance. `separation` records what the mutations "
                "actually move, measured during generation."
            ),
            "adamw": ADAMW,
            "all_trainable": True,
            "steps": MULTISTEP_N,
            "losses": traj32,
            "separation": {
                "f32_f64_noise": traj_delta,
                "betas_0.5_0.5": betas_signal,
                "weight_decay_0": decay_signal,
                "decay_not_gated_here": (
                    "the decay term is ~3 f32 ulp of the parameters it acts on at this "
                    "lr/weight_decay, so no tolerance over this fixture separates it; "
                    "adamw-kernel-v1's falsify_aw_001 owns that defect"
                ),
            },
        },
    )

    # ---------------------------------------------------------- batch invariance --
    cs, cbatch = cases["invariance_single"], cases["invariance_batch3"]
    hs_s, m_s = run_slice(model32, cs, remap, torch.float32)
    hs_b, m_b = run_slice(model32, cbatch, remap, torch.float32)
    emb_single = l2_normalize(masked_mean(hs_s[-1], m_s))
    emb_batch = l2_normalize(masked_mean(hs_b[-1], m_b))
    row = corpus.INVARIANCE_TARGET_ROW
    inv_delta = float((emb_single[0].to(torch.float64) - emb_batch[row].to(torch.float64)).abs().max())
    record_tolerance(tolerances, "batch_invariance", inv_delta)
    jsonfmt.write(
        FIXTURE_DIR / "batch_invariance.json",
        {
            "note": (
                "The same sentence encoded at batch 1 and inside a padded batch of 3. "
                "`case_id` lives on EACH of `single` and `padded_batch` (mirroring loss_pair's "
                "a_case_id/b_case_id) because the two batches hold different text sets and "
                "cannot resolve to one tokenizer case. `target_row` says which padded row is "
                "the sentence that also appears in `single`."
            ),
            "single": {**joined("invariance_single"), "embedding": flat(emb_single)},
            "padded_batch": {
                **joined("invariance_batch3"),
                "embeddings": flat(emb_batch),
                "target_row": row,
            },
            "observed_max_abs_delta": inv_delta,
        },
    )

    # ---------------------------------------------------------- full model --------
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(REPO_ID, revision=REVISION, device="cpu")
    st.eval()
    (full_case_id,) = corpus.FULL_MODEL_CASES
    trio = corpus.CASES[full_case_id]
    with torch.no_grad():
        st_emb = torch.tensor(st.encode(trio, convert_to_numpy=True, normalize_embeddings=True))

    # Cross-check: our manual masked-mean + L2 pipeline must reproduce the real
    # SentenceTransformer forward. This is what turns A1 from an assumption into a
    # verified fact -- a wrong clamp constant or eps fails HERE, not in wave 6.
    full_case = cases[full_case_id]
    bert = st[0].auto_model.eval()
    with torch.no_grad():
        out = bert(
            input_ids=torch.tensor(full_case["input_ids"], dtype=torch.long),
            attention_mask=torch.tensor(full_case["attention_mask"], dtype=torch.long),
            token_type_ids=torch.tensor(full_case["token_type_ids"], dtype=torch.long),
        )
        manual = l2_normalize(
            masked_mean(out.last_hidden_state, torch.tensor(full_case["attention_mask"], dtype=torch.float32))
        )
    st_vs_manual = max_abs_delta(st_emb, manual)
    if st_vs_manual > 1e-5:
        sys.exit(
            f"FATAL: manual masked-mean/L2 pipeline diverges from SentenceTransformer by "
            f"{st_vs_manual:.3e}. The ST pooling clamp / normalize eps read from source do not "
            "describe what the library actually does."
        )
    record_tolerance(tolerances, "full_model_reference", st_vs_manual)
    jsonfmt.write(
        FIXTURE_DIR / "full_model_reference.json",
        {
            "case_id": full_case_id,
            "texts": list(trio),
            "shape": {"batch": st_emb.shape[0], "hidden": st_emb.shape[1]},
            "embeddings": flat(st_emb),
            "st_vs_manual_pipeline_max_delta": st_vs_manual,
        },
    )

    # ---------------------------------------------------------- tokenizer + tol ---
    tolerances["tokenizer"] = {
        "max_abs_f32_f64_delta": 0.0,
        "floor": 0.0,
        "recommended_tolerance": 0.0,
        "exact": True,
        "note": "integer equality -- token ids/type ids/masks admit NO tolerance",
    }
    jsonfmt.write(FIXTURE_DIR / "tokenizer_cases.json", tokenizer_cases)
    jsonfmt.write(FIXTURE_DIR / "tolerances_measured.json", tolerances)

    for fam, t in tolerances.items():
        if fam == "tokenizer":
            continue
        if t["recommended_tolerance"] <= 0.0 or t["floor"] <= 0.0:
            sys.exit(f"FATAL: family '{fam}' produced a zero tolerance/floor")

    # ---------------------------------------------------------- join integrity ----
    verify_joins()

    # ---------------------------------------------------------- manifest ----------
    write_manifest()
    print("\nfixture corpus complete.")


def verify_upstream_digests() -> None:
    """Re-verify the pinned upstream digests, fail closed (T-1-06).

    slice_model.py checks these too, but this generator can be re-run on its own, and a
    re-pointed upstream artifact must not be able to reach a fixture through THAT door
    either. A guard that does not cover the surface where the decision is made is theater.
    """
    manifest_path = FIXTURE_DIR / "upstream_manifest.json"
    if not manifest_path.exists():
        sys.exit("FATAL: upstream_manifest.json missing; run slice_model.py first")
    recorded = json.loads(manifest_path.read_text())
    if recorded.get("revision") != REVISION:
        sys.exit(
            f"FATAL: upstream_manifest.json pins {recorded.get('revision')} but this "
            f"generator pins {REVISION}"
        )
    for name, path in fetch_pinned().items():
        got = sha256_file(path)
        want = recorded["files"].get(name)
        if got != want:
            sys.exit(
                f"FATAL: upstream digest mismatch for {name}\n  recorded: {want}\n  fetched : {got}"
            )
    print(f"upstream digests re-verified ({len(recorded['files'])} files) @ {REVISION[:12]}")


def verify_joins() -> None:
    """Assert B6 over the COMMITTED files, not over in-memory state."""
    tc = json.loads((FIXTURE_DIR / "tokenizer_cases.json").read_text())
    by_id = {c["id"]: c for c in tc["cases"]}

    def check(cid, texts, ids, where):
        if cid not in by_id:
            sys.exit(f"FATAL: {where} case_id '{cid}' does not resolve in tokenizer_cases.json")
        if by_id[cid]["texts"] != texts:
            sys.exit(f"FATAL: {where} texts differ from tokenizer_cases.json case '{cid}'")
        if ids is not None and by_id[cid]["input_ids"] != ids:
            sys.exit(f"FATAL: {where} input_ids_canonical != tokenizer_cases.json case '{cid}'")

    for c in json.loads((FIXTURE_DIR / "forward_per_layer.json").read_text())["cases"]:
        check(c["case_id"], c["texts"], c["input_ids_canonical"], "forward_per_layer")
    for c in json.loads((FIXTURE_DIR / "pooling_normalize.json").read_text())["cases"]:
        check(c["case_id"], c["texts"], c["input_ids_canonical"], "pooling_normalize")
    lp = json.loads((FIXTURE_DIR / "loss_pair.json").read_text())["pair"]
    check(lp["a_case_id"], lp["a_texts"], lp["a_ids_canonical"], "loss_pair.a")
    check(lp["b_case_id"], lp["b_texts"], lp["b_ids_canonical"], "loss_pair.b")
    bi = json.loads((FIXTURE_DIR / "batch_invariance.json").read_text())
    check(bi["single"]["case_id"], bi["single"]["texts"], bi["single"]["input_ids_canonical"], "batch_invariance.single")
    check(
        bi["padded_batch"]["case_id"],
        bi["padded_batch"]["texts"],
        bi["padded_batch"]["input_ids_canonical"],
        "batch_invariance.padded_batch",
    )
    fm = json.loads((FIXTURE_DIR / "full_model_reference.json").read_text())
    check(fm["case_id"], fm["texts"], None, "full_model_reference")
    print("join integrity: every fixture case_id resolves with matching texts and ids")


def write_manifest(directory: Path = FIXTURE_DIR) -> None:
    """Write `manifest.sha256` over every file in `directory` except the manifest itself.

    Paths are recorded as BARE FILENAMES, i.e. relative to the manifest file's own
    directory. That is what makes a consumer's resolution rule well-defined: resolve each
    entry against the directory the manifest was read from, and the check is independent
    of the caller's working directory. `shasum -a 256 -c` only agrees with that rule when
    it is run FROM this directory, which is why the self-verification below passes
    `cwd=directory` and why every documented convenience invocation states the `cd`.

    `directory` defaults to the Phase 1 fixture tree so the Phase 1 call site is unchanged.
    """
    files = sorted(p for p in directory.iterdir() if p.is_file() and p.name != "manifest.sha256")
    lines = []
    for p in files:
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        lines.append(f"{h}  {p.name}")
    (directory / "manifest.sha256").write_text("\n".join(lines) + "\n")

    proc = subprocess.run(
        ["shasum", "-a", "256", "-c", "manifest.sha256"],
        cwd=directory,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        sys.exit(f"FATAL: manifest self-verification failed\n{proc.stdout}\n{proc.stderr}")
    print(f"{directory.name}/manifest.sha256 covers {len(files)} files; shasum -c passed")


# ====================================================================================
# PHASE 2 / plan 02-04 -- SetFit PAIR-COUNT reference fixtures
# ====================================================================================
#
#   uv run python generate_fixtures.py --pairs              # dry run: temp dir + diff
#   uv run python generate_fixtures.py --pairs --rebaseline # deliberate replacement
#
# This section is INDEPENDENT of the Phase 1 corpus above. It writes into its own
# output directory (crates/aprender-contrastive-data/tests/setfit_reference/) and never
# touches crates/aprender-core/tests/fixtures/setfit/. Running `--pairs` therefore cannot
# re-baseline a Phase 1 fixture even by accident.
#
# WHY TWO FAMILIES
# ----------------
# `setfit_measured_*.json` records what the pinned setfit 1.1.3 sampler DOES.
# `aprender_contracted_*.json` records what Aprender's contract SAYS.
# They disagree, on purpose, and the disagreement is the artifact. The pinned
# implementation includes the diagonal (`shuffle_combinations(..., replacement=True)` ->
# `np.triu_indices(n, 0)`), contradicting SetFit's own published documentation. A fixture
# hand-written from those docs would be green and would falsely attest reference parity;
# a fixture asserting "no self-pairs" against the reference would be red on day one.
# So: measure one family, compute the other, and keep them in separate files.
#
# WHY EVERY RECORDED COUNT IS RNG-INDEPENDENT
# -------------------------------------------
# `shuffle_combinations` permutes with a HARDCODED `np.random.RandomState(seed=42)`, so
# the trainer seed never reaches pair identity. The permutation changes only the ORDER in
# which the triangle is walked -- and every number recorded here is a cardinality of that
# triangle, not a function of its order. The one exception is `self_pair_count` under a
# `max_pairs` cap, where WHICH 50 positives were kept does depend on the permutation;
# that fixture lists the field in `rng_dependent_fields` rather than pretending otherwise.
#
# THREE-WAY AGREEMENT, OR FATAL
# -----------------------------
# Each measured number must equal (a) a closed form derived from reading sampler.py, and
# (b) where the plan/contract states one, a hardcoded literal. Measurement alone would
# turn a broken venv into a new baseline; the literal alone would let a wrong closed form
# through. Disagreement aborts, printing all three.
#
# INTEGRITY
# ---------
# manifest.sha256 covers every emitted file, with paths relative to the manifest's own
# directory. The CANONICAL check is the Rust verifier
# (crates/aprender-contrastive-data/tests/reference_fixtures.rs), which resolves each
# entry against the manifest file's directory and so is working-directory-independent.
# The shell convenience is, and must include, the `cd`:
#     cd crates/aprender-contrastive-data/tests/setfit_reference && \
#         shasum -a 256 -c manifest.sha256

PAIR_FIXTURE_DIR = (
    REPO_ROOT / "crates" / "aprender-contrastive-data" / "tests" / "setfit_reference"
)
UV_LOCK_PATH = Path(__file__).resolve().parent / "uv.lock"

SETFIT_PIN = "1.1.3"

# contracts/contrastive-pair-protocol-v1.yaml, equation `default_epoch_budget`.
DEFAULT_HARD_CAP = 1048576

# Copied VERBATIM in substance from OBLIG-CPP-DEVIATION-DECLARED in
# contracts/contrastive-pair-protocol-v1.yaml so the contract and the fixtures cannot
# drift into two different stories. Attribution is "aprender" in every clause: PF-008
# forbids attributing our exclusion to SetFit, whose pinned code does the opposite.
DEVIATION_ATTRIBUTION = "aprender"
DEVIATION_CLAUSES = [
    {
        "clause_id": "sampled_identities",
        "statement": (
            "Pair IDENTITIES are SAMPLED from the pair space, not enumerated-then-shuffled, "
            "so identities cannot match the reference's Python RNG and only counts are "
            "comparable."
        ),
    },
    {
        "clause_id": "capped_count",
        "statement": "The per-epoch count is CAPPED above N by a configurable hard cap.",
    },
    {
        "clause_id": "self_pairs_excluded",
        "statement": (
            "SELF-PAIRS ARE EXCLUDED, whereas the pinned setfit 1.1.3 implementation "
            "INCLUDES the diagonal (`shuffle_combinations` defaults to replacement=True, "
            "i.e. `np.triu_indices(n, k=0)`), which contradicts SetFit's own published "
            "documentation; the exclusion is therefore ours and matches the docs, not the "
            "pinned code."
        ),
    },
]

REFERENCE_NOTES = [
    "shuffle_combinations permutes with a HARDCODED np.random.RandomState(seed=42); the "
    "trainer seed does not reach pair identity at all (setfit/sampler.py:29).",
    "Enumeration materializes the FULL O(N^2) index triangle via np.triu_indices(n, 0) "
    "BEFORE any max_pairs cap can apply (setfit/sampler.py:28), so the cap bounds what is "
    "STORED, never what is allocated.",
    "replacement defaults to True, so k=0 and the diagonal is included: every example "
    "yields a positive pair with itself.",
    "oversampling sets len_pos = len_neg = max(len(pos_pairs), len(neg_pairs)); the "
    "shorter list is cycled, so the epoch length is 2 * that maximum.",
]


# --- closed forms -------------------------------------------------------------------
# Both families are derived here from FIRST PRINCIPLES -- the Aprender contract for one,
# a reading of setfit/sampler.py for the other. Neither is derived from Aprender's Rust
# implementation, which does not exist yet (plan 02-07 builds it against these files). A
# fixture produced by recording Rust's output would make every downstream conformance
# claim circular.


def contracted_positive_capacity(sizes: list[int]) -> int:
    """Sum of C(n_k, 2) -- self-pairs EXCLUDED (Aprender policy, deviation clause 3)."""
    return sum(n * (n - 1) // 2 for n in sizes)


def contracted_negative_capacity(sizes: list[int]) -> int:
    """Sum over j<k of n_j * n_k, computed in O(K) from S and the sum of squares."""
    total = sum(sizes)
    return (total * total - sum(n * n for n in sizes)) // 2


def predicted_reference_counts(sizes: list[int], max_pairs: int) -> dict:
    """Predict the pinned sampler's cardinalities by READING setfit/sampler.py.

    Enumeration is over `np.triu_indices(n, 0)`, i.e. every (i, j) with i <= j exactly
    once. A pair is positive iff the two labels agree, so:

        stored_pos = sum_k [ C(n_k, 2) + n_k ]   (the +n_k is the diagonal)
        stored_neg = sum_{j<k} n_j * n_k

    With `max_pairs != -1` each list is capped at `max_pairs // 2` and the walk stops only
    once BOTH are full, so each list reaches min(cap, its full cardinality) regardless of
    the permutation. Only i <= j is ever produced, so both orientations of one unordered
    pair can never both appear: the orientation-duplicate count is identically 0.
    """
    n_examples = sum(sizes)
    full_pos = contracted_positive_capacity(sizes) + n_examples
    full_neg = contracted_negative_capacity(sizes)
    cap = -1 if max_pairs == -1 else max_pairs // 2
    stored_pos = full_pos if cap == -1 else min(cap, full_pos)
    stored_neg = full_neg if cap == -1 else min(cap, full_neg)
    balanced = max(stored_pos, stored_neg)
    return {
        "stored_pos": stored_pos,
        "stored_neg": stored_neg,
        "self_pair_count": n_examples if cap == -1 else None,  # None => RNG-dependent
        "orientation_duplicate_count": 0,
        "len_pos": balanced,
        "len_neg": balanced,
        "total": 2 * balanced,
    }


# --- the layouts ---------------------------------------------------------------------
# `literals` are the numbers stated by plan 02-04 and by
# contracts/contrastive-pair-protocol-v1.yaml (FALSIFY-CPP-016 / -017 / the K~N
# prediction). They are checked against BOTH the closed form and the measurement.
PAIR_LAYOUTS = [
    {
        "fixture_id": "8_4_8",
        "layout": [8, 4, 8],
        "max_pairs": -1,
        "why": (
            "SetFit's own documented worked example. The docs claim 62 positives / 128 "
            "negatives / 256 total; the total is right and the composition is not."
        ),
        "literals": {"stored_pos": 82, "stored_neg": 128, "total": 256, "self_pair_count": 20},
        "contracted_literals": {
            "positive_capacity": 62,
            "negative_capacity": 128,
            "default_epoch_budget": 256,
        },
    },
    {
        "fixture_id": "4_1",
        "layout": [4, 1],
        "max_pairs": -1,
        "why": (
            "The Pitfall 2 divergence case: a singleton class. The reference gives the "
            "singleton a positive SELF-pair; Aprender gives it none, so a singleton class "
            "contributes zero positive capacity -- while the four-member class still "
            "contributes its six positives."
        ),
        "literals": {"total": 22, "stored_pos": 11, "self_pair_count": 5, "stored_neg": 4},
        "contracted_literals": {
            "positive_capacity": 6,
            "negative_capacity": 4,
            "default_epoch_budget": 12,
        },
    },
    {
        "fixture_id": "8_8_8",
        "layout": [8, 8, 8],
        "max_pairs": -1,
        "why": "The 8-shot 3-class layout of this milestone. D-14's worked value: 384.",
        "literals": {"total": 384},
        "contracted_literals": {"default_epoch_budget": 384},
    },
    {
        "fixture_id": "64_64_64",
        "layout": [64, 64, 64],
        "max_pairs": -1,
        "why": "The 64-shot 3-class layout. D-14's worked value: 24,576.",
        "literals": {"total": 24576},
        "contracted_literals": {"default_epoch_budget": 24576},
    },
    {
        "fixture_id": "8_4_8_maxpairs100",
        "layout": [8, 4, 8],
        "max_pairs": 100,
        "why": (
            "The cap's semantics: max_pairs // 2 PER LIST, not max_pairs in total, and it "
            "bounds what is stored rather than what is enumerated."
        ),
        "literals": {"stored_pos": 50, "stored_neg": 50, "total": 100},
        "contracted_literals": {
            "positive_capacity": 62,
            "negative_capacity": 128,
            "default_epoch_budget": 256,
        },
    },
    {
        "fixture_id": "singletons_32",
        "layout": [1] * 32,
        "max_pairs": -1,
        "why": (
            "The K = N adversarial layout (32 classes, 32 examples). This is the row plan "
            "02-08 measures an O(K^2) sampler against; a three-class fixture set could "
            "never expose one. Aprender's positive capacity is 0 here, so the stream is "
            "negatives-only -- and note the two families agree on the TOTAL (992) while "
            "disagreeing on every pair in it."
        ),
        "literals": {},
        "contracted_literals": {
            "positive_capacity": 0,
            "negative_capacity": 496,
            "default_epoch_budget": 992,
        },
    },
]


def _uv_version() -> str:
    """A version string alone does not identify an environment; record the resolver too."""
    proc = subprocess.run(["uv", "--version"], capture_output=True, text=True)
    if proc.returncode != 0:
        sys.exit(f"FATAL: `uv --version` failed (rc={proc.returncode})\n{proc.stderr}")
    return proc.stdout.strip()


def _check(fixture_id: str, field: str, measured, predicted, literal) -> None:
    """Three-way agreement or FATAL, printing every number that disagreed."""
    if predicted is not None and measured != predicted:
        sys.exit(
            f"FATAL: {fixture_id}.{field} -- the pinned venv does not match the closed form "
            f"read out of setfit/sampler.py.\n  measured : {measured}\n  closed form: {predicted}\n"
            "This means the installed sampler is not the pinned setfit 1.1.3, or its "
            "semantics changed. Do NOT re-baseline; fix the environment."
        )
    if literal is not None and measured != literal:
        sys.exit(
            f"FATAL: {fixture_id}.{field} -- the measurement contradicts the number stated "
            f"by plan 02-04 / contracts/contrastive-pair-protocol-v1.yaml.\n"
            f"  measured: {measured}\n  contract: {literal}\n"
            "Surface the discrepancy; do NOT edit the fixture to match code."
        )


def measure_reference(sizes: list[int], max_pairs: int) -> dict:
    """Run the PINNED sampler and record its cardinalities."""
    from setfit.sampler import ContrastiveDataset

    labels = [k for k, n in enumerate(sizes) for _ in range(n)]
    sentences = [f"s{i}" for i in range(len(labels))]
    ds = ContrastiveDataset(
        sentences,
        labels,
        multilabel=False,
        sampling_strategy="oversampling",
        max_pairs=max_pairs,
    )
    self_pairs = sum(1 for p in ds.pos_pairs if p["sentence_1"] == p["sentence_2"])

    seen: set[tuple[str, str]] = set()
    orientation_duplicates = 0
    for p in list(ds.pos_pairs) + list(ds.neg_pairs):
        a, b = p["sentence_1"], p["sentence_2"]
        if a != b and (b, a) in seen:
            orientation_duplicates += 1
        seen.add((a, b))

    return {
        "stored_pos": len(ds.pos_pairs),
        "stored_neg": len(ds.neg_pairs),
        "self_pair_count": self_pairs,
        "orientation_duplicate_count": orientation_duplicates,
        "len_pos": ds.len_pos_pairs,
        "len_neg": ds.len_neg_pairs,
        "total": len(ds),
    }


def build_pair_fixtures() -> dict[str, dict]:
    """Build every pair-count fixture payload in memory. Nothing is written here."""
    import setfit

    if setfit.__version__ != SETFIT_PIN:
        sys.exit(
            f"FATAL: setfit {setfit.__version__} is installed but the reference pin is "
            f"{SETFIT_PIN}. Every number below is an artifact of the pinned version."
        )
    if not UV_LOCK_PATH.exists():
        sys.exit(f"FATAL: {UV_LOCK_PATH} is missing; the environment cannot be attested")

    attestation = {
        "setfit_version": setfit.__version__,
        "uv_lock_sha256": sha256_file(UV_LOCK_PATH),
        "uv_version": _uv_version(),
    }
    print(
        f"environment attestation: setfit {attestation['setfit_version']}, "
        f"uv.lock {attestation['uv_lock_sha256'][:12]}, {attestation['uv_version']}"
    )

    payloads: dict[str, dict] = {}
    for spec in PAIR_LAYOUTS:
        fid = spec["fixture_id"]
        sizes = list(spec["layout"])
        max_pairs = spec["max_pairs"]

        measured = measure_reference(sizes, max_pairs)
        predicted = predicted_reference_counts(sizes, max_pairs)
        for field, value in measured.items():
            _check(fid, field, value, predicted.get(field), spec["literals"].get(field))

        rng_dependent = [] if max_pairs == -1 else ["self_pair_count"]
        payloads[f"setfit_measured_{fid}.json"] = {
            "fixture_family": "setfit_measured",
            "fixture_id": fid,
            "layout": sizes,
            "n_examples": sum(sizes),
            "n_classes": len(sizes),
            "sampling_strategy": "oversampling",
            "multilabel": False,
            "max_pairs": max_pairs,
            "stored_pos": measured["stored_pos"],
            "stored_neg": measured["stored_neg"],
            "self_pair_count": measured["self_pair_count"],
            "orientation_duplicate_count": measured["orientation_duplicate_count"],
            "len_pos": measured["len_pos"],
            "len_neg": measured["len_neg"],
            "total": measured["total"],
            "rng_dependent_fields": rng_dependent,
            "why_this_layout": spec["why"],
            "derivation": (
                "MEASURED by executing setfit.sampler.ContrastiveDataset in the hash-locked "
                "venv, then cross-checked against a closed form read out of "
                "setfit/sampler.py: stored_pos = sum_k [C(n_k,2) + n_k] (the +n_k is the "
                "included diagonal), stored_neg = sum_{j<k} n_j*n_k, each capped at "
                "max_pairs//2 when max_pairs != -1, and total = 2*max(stored_pos, "
                "stored_neg) under the oversampling strategy. The two agree, or this file "
                "is not written."
            ),
            "reference_notes": list(REFERENCE_NOTES),
            **attestation,
        }

        pos_cap = contracted_positive_capacity(sizes)
        neg_cap = contracted_negative_capacity(sizes)
        closed_form = 2 * max(pos_cap, neg_cap)
        default_budget = min(closed_form, DEFAULT_HARD_CAP)
        explicit_budget = None if max_pairs == -1 else max_pairs
        resolved = default_budget if explicit_budget is None else explicit_budget

        if pos_cap == 0 and neg_cap == 0:
            degenerate, res_pos, res_neg = "no_capacity", 0, 0
        elif pos_cap == 0:
            degenerate, res_pos, res_neg = "negatives_only", 0, resolved
        elif neg_cap == 0:
            degenerate, res_pos, res_neg = "positives_only", resolved, 0
        else:
            degenerate = None
            res_pos, res_neg = (resolved + 1) // 2, resolved // 2

        for field, value in (
            ("positive_capacity", pos_cap),
            ("negative_capacity", neg_cap),
            ("default_epoch_budget", default_budget),
        ):
            literal = spec["contracted_literals"].get(field)
            if literal is not None and value != literal:
                sys.exit(
                    f"FATAL: {fid}.{field} closed form {value} contradicts the contracted "
                    f"value {literal}. Surface the discrepancy; do not edit the fixture."
                )

        measured_total = measured["total"]
        if resolved != measured_total:
            divergence = (
                f"DIVERGES: Aprender's resolved budget is {resolved} while the pinned "
                f"reference's epoch length is {measured_total}. Deliberate, and covered by "
                "deviation clauses 2 and 3."
            )
        elif res_pos != measured["len_pos"] or res_neg != measured["len_neg"]:
            divergence = (
                f"Totals coincide at {resolved}, composition does not: Aprender emits "
                f"{res_pos} positives / {res_neg} negatives, the reference emits "
                f"{measured['len_pos']} / {measured['len_neg']} -- and every one of its "
                f"'positives' here is a self-pair. Agreement on a total is not agreement."
            )
        else:
            divergence = (
                f"Agrees with the reference epoch length ({measured_total}) because "
                "negatives dominate this layout, so the excluded diagonal never reaches "
                "the max()."
            )

        payloads[f"aprender_contracted_{fid}.json"] = {
            "fixture_family": "aprender_contracted",
            "fixture_id": fid,
            "layout": sizes,
            "n_examples": sum(sizes),
            "n_classes": len(sizes),
            "positive_capacity": pos_cap,
            "negative_capacity": neg_cap,
            "closed_form_budget": closed_form,
            "hard_cap": DEFAULT_HARD_CAP,
            "clamp_engaged": closed_form > DEFAULT_HARD_CAP,
            "explicit_budget": explicit_budget,
            "default_epoch_budget": default_budget,
            "resolved_budget": resolved,
            "resolved_pos_count": res_pos,
            "resolved_neg_count": res_neg,
            "degenerate_case": degenerate,
            "self_pairs_excluded": True,
            "measured_counterpart": f"setfit_measured_{fid}.json",
            "measured_total": measured_total,
            "divergence_note": divergence,
            "deviation_attribution": DEVIATION_ATTRIBUTION,
            "deviation_clauses": [dict(c) for c in DEVIATION_CLAUSES],
            "why_this_layout": spec["why"],
            "derivation": (
                "COMPUTED from the closed forms in "
                "contracts/contrastive-pair-protocol-v1.yaml: positive_capacity = "
                "sum_k C(n_k,2) with self-pairs EXCLUDED, negative_capacity = "
                "sum_{j<k} n_j*n_k, closed_form_budget = 2*max(pos, neg), "
                "default_epoch_budget = min(closed_form, hard_cap). No Aprender Rust code "
                "is executed to produce these numbers -- they are the reference the Rust "
                "sampler is measured against, so deriving them from it would be circular."
            ),
            **attestation,
        }

    return payloads


def _diff_against_committed(staged: Path, committed: Path) -> list[str]:
    """Return a human-readable diff of `staged` vs `committed`. Empty list == identical."""
    import difflib

    staged_files = {p.name for p in staged.iterdir() if p.is_file()}
    committed_files = (
        {p.name for p in committed.iterdir() if p.is_file()} if committed.exists() else set()
    )
    report: list[str] = []
    for name in sorted(staged_files | committed_files):
        new = staged / name
        old = committed / name
        if name not in committed_files:
            report.append(f"+ ADDED    {name}")
            continue
        if name not in staged_files:
            report.append(f"- REMOVED  {name}  (no longer emitted by the generator)")
            continue
        if new.read_bytes() == old.read_bytes():
            continue
        report.append(f"~ CHANGED  {name}")
        report.extend(
            line.rstrip("\n")
            for line in difflib.unified_diff(
                old.read_text().splitlines(keepends=True),
                new.read_text().splitlines(keepends=True),
                fromfile=f"committed/{name}",
                tofile=f"regenerated/{name}",
                n=1,
            )
        )
    return report


def main_pairs(rebaseline: bool) -> None:
    """Emit into a temp dir, diff against the committed tree, replace only on request.

    Re-baselining a reference fixture is how a wrong implementation becomes the new
    truth, so it is a separate, deliberate act with a reviewable diff in front of it.
    Dry-run exit status: 0 when the committed tree already matches, 1 when it does not.
    """
    import shutil
    import tempfile

    payloads = build_pair_fixtures()

    staging = Path(tempfile.mkdtemp(prefix="apr-pair-fixtures-"))
    try:
        for name, payload in payloads.items():
            jsonfmt.write(staging / name, payload)
        write_manifest(staging)

        report = _diff_against_committed(staging, PAIR_FIXTURE_DIR)
        rel = PAIR_FIXTURE_DIR.relative_to(REPO_ROOT)
        if not report:
            print(f"\n{rel}: IDENTICAL -- {len(payloads)} fixtures + manifest already committed")
            return

        print(f"\ndiff vs committed {rel}:")
        for line in report:
            print(f"  {line}")

        if not rebaseline:
            print(
                "\nDRY RUN -- nothing was written. Re-run with --rebaseline to replace the "
                "committed fixtures, and review the diff above as part of that change."
            )
            sys.exit(1)

        PAIR_FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
        for stale in PAIR_FIXTURE_DIR.iterdir():
            if stale.is_file() and stale.name not in payloads and stale.name != "manifest.sha256":
                stale.unlink()
        for name in list(payloads) + ["manifest.sha256"]:
            shutil.copyfile(staging / name, PAIR_FIXTURE_DIR / name)

        proc = subprocess.run(
            ["shasum", "-a", "256", "-c", "manifest.sha256"],
            cwd=PAIR_FIXTURE_DIR,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            sys.exit(
                f"FATAL: manifest does not verify in the committed tree\n{proc.stdout}\n{proc.stderr}"
            )
        print(f"\nREBASELINED {rel}: {len(payloads)} fixtures + manifest, shasum -c passed")
    finally:
        shutil.rmtree(staging, ignore_errors=True)


USAGE = """usage:
  uv run python generate_fixtures.py                        Phase 1 ENC-01..06 corpus
  uv run python generate_fixtures.py --pairs                pair-count fixtures (dry run)
  uv run python generate_fixtures.py --pairs --rebaseline   pair-count fixtures (replace)
"""


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        main()
    elif args[0] == "--pairs" and set(args[1:]) <= {"--rebaseline"}:
        main_pairs(rebaseline="--rebaseline" in args)
    else:
        sys.exit(USAGE)
