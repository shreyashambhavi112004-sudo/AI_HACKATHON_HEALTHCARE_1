"""Draws the proof plot straight from fingerprint_results.csv, so the chart can never disagree with the table.
Run from the project folder:  python3 plot_from_csv.py
"""
import csv, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CSV = sys.argv[1] if len(sys.argv) > 1 else "fingerprint_results.csv"
OUT = "proof_plot_final.png"
NAMES = {
    "held-out real": "Real\n(held-out)",
    "set A": "Synthetic A\n(similar density)",
    "set B": "Synthetic B\n(denser)",
    "set C": "Synthetic C\n(fewer, larger)",
    "old synthetic": "Old synthetic\n(sparse)",
}
rows = list(csv.DictReader(open(CSV)))
labels = [NAMES.get(r["set"], r["set"]) for r in rows]
mean = [float(r["mean_score"]) for r in rows]
std = [float(r["std_score"]) for r in rows]
g = np.array([float(r["green_share"]) for r in rows])
a = np.array([float(r["amber_share"]) for r in rows])
rd = np.array([float(r["red_share"]) for r in rows])

G, A, R = "#2e9e4f", "#f2a31b", "#d23f3f"
# bar colour = the verdict most groups got
cols = [[G, A, R][int(np.argmax([g[i], a[i], rd[i]]))] for i in range(len(rows))]
x = np.arange(len(rows))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), gridspec_kw={"height_ratios": [3, 2]})
ax1.bar(x, mean, yerr=std, capsize=6, color=cols, edgecolor="black", linewidth=0.6)
for i, m in enumerate(mean):
    ax1.text(i, m + std[i] + 2, f"{m:.1f}", ha="center", fontsize=11)
ax1.set_xticks(x); ax1.set_xticklabels(labels)
ax1.set_ylim(0, 100)
ax1.set_ylabel("Mean match score (0-100)")
ax1.set_title("How closely each batch matches the real reference", fontsize=14)

ax2.bar(x, g, color=G, label="Green (consistent)")
ax2.bar(x, a, bottom=g, color=A, label="Amber (inconclusive)")
ax2.bar(x, rd, bottom=g + a, color=R, label="Red (deviates)")
ax2.set_xticks(x); ax2.set_xticklabels(labels)
ax2.set_ylim(0, 1)
ax2.set_ylabel("Share of 50 sample groups")
ax2.set_title("Verdict for each set (50 random groups of 10 images)", fontsize=12)
ax2.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)
plt.tight_layout()
plt.savefig(OUT, dpi=200)
print("Saved", OUT)
for r in rows:
    print(r["set"], "green", r["green_share"], "amber", r["amber_share"], "red", r["red_share"])
