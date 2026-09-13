#!/usr/bin/env python3
"""Cluster the aprender surface audit by feature similarity.

Representation, and why
-----------------------
Three signals concatenated, because feature NAME alone under-separates:

  1. the feature string, word-tokenized  ("apr train plan --task pretrain")
  2. char 3-5 grams of the same, so `ptx` / `ptx-map` / `ptx-debug` land together
     even though word tokenization splits them apart
  3. the evidence MODULE stem (`train_commands`, `banco/router`), because two
     features implemented in the same module are related by construction — this
     is the only signal that is not a restatement of the name

Transport prefix (`mcp:`, `GET `, `POST `) is kept: a route and a subcommand that
share a noun are genuinely different features with different failure modes.

WHAT THIS SCRIPT DOES AND DOES NOT PRODUCE -- read before trusting it
---------------------------------------------------------------------
It produces the K-SELECTION EVIDENCE: the inertia/silhouette sweep over k in
[2,45], the knee, and docs/audits/surface_audit_elbow.png. That is all. It does
NOT write the `cluster_id` / `cluster_label` columns of
docs/audits/surface_audit.csv, and re-running it will not regenerate them.

That is deliberate, and it is the same rule as T1 in the coverage gate:

  * `cluster_id` PERMUTES across runs whenever the input moves. It is provenance,
    never an identity. Nothing may key on it -- enforced by
    scripts/check_no_cluster_id_keys.sh.
  * `cluster_label` is HUMAN-OWNED after the first assignment. A label is a name
    a person gave a group of features; regenerating it from k-means would silently
    re-point every gate obligation that cites it.

So: run this when the surface has moved enough that k itself is in question, read
the plot, and then decide by hand which rows change label. Do not wire it into a
pipeline that rewrites the ledger.

CLUSTERING IS A PRIOR, NEVER EVIDENCE (T3). It says where to look. It cannot
assert that a feature works, and `quality_1_10` must never be derived from it.
"""
import csv, re, sys, json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Paths are repo-root-relative so the script runs from the repo root, the way
# every other scripts/dogfood_*.py does. The draft hardcoded a bare
# "surface_audit.csv" and could only run from inside docs/audits/ -- a script
# that cannot be run as documented is the shipped-but-unreachable failure class.
SRC = sys.argv[1] if len(sys.argv) > 1 else "docs/audits/surface_audit.csv"
OUT_PNG = "docs/audits/surface_audit_elbow.png"
rows = list(csv.DictReader(open(SRC, newline="", encoding="utf-8")))
print(f"rows: {len(rows)}")


def module_stem(path):
    """crates/apr-cli/src/train_commands.rs:30 -> apr-cli train_commands"""
    p = path.split(" ")[0].split(":")[0]
    m = re.match(r"crates/([^/]+)/src/(.+)\.rs$", p)
    if not m:
        return re.sub(r"[^a-z0-9]+", " ", p.lower())
    crate, rest = m.group(1), m.group(2)
    return f"{crate} " + rest.replace("/", " ").replace("_", " ")


def transport(feat):
    if feat.startswith("mcp:"):
        return "mcp"
    if re.match(r"^(GET|POST|PUT|DELETE|PATCH|HEAD) ", feat):
        return "http"
    return "cli"


texts, feats = [], []
for r in rows:
    f = r["feature"]
    feats.append(f)
    # strip the binary name prefix so the binary does not dominate similarity;
    # binary identity is already its own column
    body = f
    if body.startswith(r["binary"] + " "):
        body = body[len(r["binary"]) + 1:]
    texts.append(f"{transport(f)} {body} {body.replace('-', ' ')} {module_stem(r['evidence_path'])}")

union = FeatureUnion([
    ("word", TfidfVectorizer(analyzer="word", token_pattern=r"[a-zA-Z0-9_./-]+",
                             ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
    ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                             min_df=3, sublinear_tf=True)),
])
X = union.fit_transform(texts)
print(f"tf-idf matrix: {X.shape}")

# ── k sweep ────────────────────────────────────────────────────────────
KS = list(range(2, 46))
inertia, sil = [], []
for k in KS:
    km = KMeans(n_clusters=k, random_state=0, n_init=10)
    lab = km.fit_predict(X)
    inertia.append(km.inertia_)
    sil.append(silhouette_score(X, lab, metric="cosine", random_state=0))
    print(f"k={k:3}  inertia={km.inertia_:9.3f}  silhouette={sil[-1]:.4f}")

# knee via max distance to the chord from first to last point
xs = np.array(KS, float)
ys = np.array(inertia, float)
xn = (xs - xs.min()) / (xs.max() - xs.min())
yn = (ys - ys.min()) / (ys.max() - ys.min())
p0, p1 = np.array([xn[0], yn[0]]), np.array([xn[-1], yn[-1]])
d = p1 - p0
dist = np.abs(d[0] * (p0[1] - yn) - (p0[0] - xn) * d[1]) / np.linalg.norm(d)
k_elbow = int(xs[int(np.argmax(dist))])
k_sil = int(xs[int(np.argmax(sil))])
print(f"\nelbow k={k_elbow}   best-silhouette k={k_sil}")

fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
ax[0].plot(KS, inertia, marker="o", ms=3.5, lw=1.4, color="#1f4e79")
ax[0].axvline(k_elbow, color="#c0392b", ls="--", lw=1.2,
              label=f"knee: k={k_elbow}")
ax[0].set_xlabel("k"); ax[0].set_ylabel("inertia (within-cluster SSE)")
ax[0].set_title("Elbow — aprender surface, 830 features")
ax[0].legend(); ax[0].grid(alpha=.25)

ax[1].plot(KS, sil, marker="o", ms=3.5, lw=1.4, color="#1f4e79")
ax[1].axvline(k_sil, color="#c0392b", ls="--", lw=1.2, label=f"max: k={k_sil}")
ax[1].axvline(k_elbow, color="#7f8c8d", ls=":", lw=1.2, label=f"knee: k={k_elbow}")
ax[1].set_xlabel("k"); ax[1].set_ylabel("silhouette (cosine)")
ax[1].set_title("Silhouette")
ax[1].legend(); ax[1].grid(alpha=.25)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=170)
print(f"wrote {OUT_PNG}")

json.dump({"ks": KS, "inertia": inertia, "silhouette": sil,
           "k_elbow": k_elbow, "k_sil": k_sil},
          open("docs/audits/surface_audit_sweep.json", "w"))
