"""Exports the charts used on the GitHub Pages site into docs/assets/.
Runs the notebook's code first so every chart uses the exact same data and models."""
import os, contextlib, io
import nbformat
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
OUT = os.path.join(HERE, "docs", "assets")
os.makedirs(OUT, exist_ok=True)

# --- run the notebook pipeline -------------------------------------------------
plt.show = lambda *a, **k: plt.close("all")
g = {}
nb = nbformat.read("valorant_role_classifier.ipynb", 4)
with contextlib.redirect_stdout(io.StringIO()):
    for c in nb.cells:
        if c.cell_type == "code":
            exec(c.source, g)

import numpy as np, pandas as pd, seaborn as sns
from sklearn.metrics import confusion_matrix

raw, data, ROLE_ORDER = g["raw"], g["data"], g["ROLE_ORDER"]

# --- shared style ----------------------------------------------------------------
ROLE_COLORS = {"duelist": "#2a78d6", "initiator": "#eb6834",
               "controller": "#1baf7a", "sentinel": "#eda100"}
INK, INK2, GRID, SURF = "#1d1d1b", "#5f5e58", "#e6e4dd", "#ffffff"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titleweight": "bold", "axes.titlesize": 12, "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
})
def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, name), dpi=180, bbox_inches="tight")
    plt.close("all")
def bars(ax, labels, values, fmt="{:.0f}"):
    ax.bar([l.title() for l in labels], values, color=[ROLE_COLORS[r] for r in labels], width=0.6, edgecolor=SURF, linewidth=2)
    ax.grid(axis="x", visible=False)
    for i, v in enumerate(values):
        ax.text(i, v, fmt.format(v), ha="center", va="bottom", color=INK, fontsize=10, fontweight="bold")
    ax.margins(y=0.15)

# 1. class balance
counts = data["Role"].value_counts().reindex(ROLE_ORDER)
fig, ax = plt.subplots(figsize=(6, 3.4))
bars(ax, ROLE_ORDER, counts.values)
ax.set_ylabel("Rows"); ax.set_title("Rows per role (after cleanup)", loc="left")
save("01_role_balance.png")

# 2. outlier example: K:D before/after the 3×IQR fence
kd_raw = raw[(raw["Match Type"] != "All Match Types") & (~raw["Agents"].str.contains(","))]["Kills:Deaths"]
q1, q3 = kd_raw.quantile([.25, .75]); upper = q3 + 3 * (q3 - q1)
fig, ax = plt.subplots(figsize=(7, 3.2))
ax.hist(kd_raw.clip(upper=6), bins=60, color="#86b6ef", edgecolor=SURF, linewidth=0.5)
ax.set_yscale("log")
ax.axvline(upper, color=INK, lw=1.5, ls="--")
ax.text(upper + 0.08, 300, f"3×IQR fence = {upper:.2f}\nrows to the right are removed",
        color=INK, fontsize=9.5, va="top")
ax.set_xlabel("Kills : Deaths (values above 6 shown at 6)"); ax.set_ylabel("Rows (log scale)")
ax.set_title("Kills:Deaths has a long tail of small-sample flukes", loc="left")
save("02_outliers_kd.png")

# 3. correlation heatmap (before dropping)
num = g["NUMERIC"] + ["Average Combat Score", "Average Damage Per Round", "Rating"]
num = [c for c in dict.fromkeys(num) if c in data.columns]
short = {"Average Combat Score": "ACS", "Average Damage Per Round": "ADR", "Kills:Deaths": "K:D",
         "Kills Per Round": "Kills/rd", "Assists Per Round": "Assists/rd", "First Kills Per Round": "First kills/rd",
         "First Deaths Per Round": "First deaths/rd", "Headshot %": "HS %", "Clutches Played Per Round": "Clutches/rd",
         "Opening Duels Per Round": "Opening duels/rd", "KAST %": "KAST %", "Rating": "Rating"}
corr = data[num].rename(columns=short).corr()
fig, ax = plt.subplots(figsize=(8, 6.4))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, center=0, square=True,
            linewidths=1.5, linecolor=SURF, cbar_kws={"shrink": .7}, annot_kws={"size": 8}, ax=ax)
ax.set_title("Correlation between stats — ACS, ADR and Rating were dropped", loc="left")
ax.grid(False)
save("03_correlation.png")

# 4. role fingerprint: two small panels, one measure each
means = data.groupby("Role")[["Opening Duels Per Round", "Assists Per Round"]].mean().reindex(ROLE_ORDER)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4))
bars(a1, ROLE_ORDER, means["Opening Duels Per Round"].values, "{:.2f}")
a1.set_title("Opening duels per round", loc="left")
bars(a2, ROLE_ORDER, means["Assists Per Round"].values, "{:.2f}")
a2.set_title("Assists per round", loc="left")
save("04_role_fingerprint.png")

# 5. clusters vs roles in PCA space
pcs, clusters = g["pcs"], data["Cluster"]
cl_colors = ["#5f5e58", "#c3c2b7", "#184f95", "#86b6ef"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True, sharey=True)
for k in sorted(clusters.unique()):
    m = clusters == k
    a1.scatter(pcs[m, 0], pcs[m, 1], s=6, alpha=.45, color=cl_colors[k], label=f"Cluster {k}")
for r in ROLE_ORDER:
    m = data["Role"] == r
    a2.scatter(pcs[m, 0], pcs[m, 1], s=6, alpha=.45, color=ROLE_COLORS[r], label=r.title())
a1.set_title("K-Means clusters (no labels used)", loc="left"); a2.set_title("Same players, colored by real role", loc="left")
for a in (a1, a2):
    a.set_xlabel("PCA component 1"); a.legend(markerscale=3, frameon=False, fontsize=9, loc="upper right")
a1.set_ylabel("PCA component 2")
save("05_clusters_pca.png")

# 6. role share inside each cluster
ct = pd.crosstab(clusters, data["Role"], normalize="index").reindex(columns=ROLE_ORDER)
ct.columns = [c.title() for c in ct.columns]; ct.index = [f"Cluster {i}" for i in ct.index]
fig, ax = plt.subplots(figsize=(6.5, 3.2))
sns.heatmap(ct, annot=True, fmt=".0%", cmap="Blues", vmin=0, vmax=.8, cbar=False,
            linewidths=2, linecolor=SURF, ax=ax)
ax.grid(False); ax.set_title("Which roles fall into each cluster", loc="left"); ax.set_ylabel("")
save("06_cluster_roles.png")

# 7. confusion matrix, row-normalized
cm = confusion_matrix(g["y_test"], g["y_pred"], labels=ROLE_ORDER, normalize="true")
labels = [r.title() for r in ROLE_ORDER]
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt=".0%", cmap="Blues", vmin=0, vmax=1, cbar=False, square=True,
            linewidths=2, linecolor=SURF, xticklabels=labels, yticklabels=labels, annot_kws={"size": 12}, ax=ax)
ax.grid(False); ax.set_xlabel("Predicted role"); ax.set_ylabel("Actual role")
ax.set_title("Confusion matrix — test set (unseen players)", loc="left")
save("07_confusion_matrix.png")

print(g["results"])
print(sorted(os.listdir(OUT)))
