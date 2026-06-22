"""Generate PRIM notebooks for SRQ2 restricted to old distance ranges (same_bn 10-20, inter_bn 15-35)."""
import json, pathlib

POLICIES = [
    {"name": "fac", "label": "Facility pooling", "fac_elig": 1, "amb_elig": 0},
    {"name": "amb", "label": "Ambulance pooling", "fac_elig": 0, "amb_elig": 1},
    {"name": "both", "label": "Both pooling", "fac_elig": 1, "amb_elig": 1},
]

def code(cells, src):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in src.strip().split("\n")]
    })

def md(cells, src):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in src.strip().split("\n")]
    })


def build_notebook(pol):
    cells = []
    name = pol["name"]
    label = pol["label"]
    fac_e = pol["fac_elig"]
    amb_e = pol["amb_elig"]

    md(cells, f"# {label} vs Stovepipe — PRIM on delta (2 amb, OLD distance ranges)")

    code(cells, f"""import sys
from pathlib import Path
_root = Path(".").resolve()
while not (_root / "paths.py").exists():
    _root = _root.parent
sys.path.insert(0, str(_root))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ema_workbench.analysis import prim

RESULTS_PATH = _root / "data_cluster/resource_grid/results.csv"
df = pd.read_csv(RESULTS_PATH)
df = df[df["n_amb_r1_to_ccp"] == 2]

# Restrict to old distance ranges
df = df[(df["ccp_r1_same_bn_min"] >= 10) & (df["ccp_r1_same_bn_min"] <= 20)]
df = df[(df["ccp_r1_inter_bn_min"] >= 15) & (df["ccp_r1_inter_bn_min"] <= 35)]
df = df.reset_index(drop=True)
print(f"Loaded {{len(df)}} runs (2 amb, same_bn 10-20, inter_bn 15-35)")

UNCERTAINTIES = ["ccp_r1_same_bn_min", "ccp_r1_inter_bn_min", "bcas_rate"]
LEVERS = UNCERTAINTIES
MATCH_COLS = ["seed", "ccp_r1_same_bn_min", "ccp_r1_inter_bn_min", "bcas_rate"]

def get_policy(df_sub, fac, amb):
    return df_sub[(df_sub["facility_eligibility"] == fac) & (df_sub["ambulance_eligibility"] == amb)]

OUTCOME_COLS = [c for c in df.columns if c.startswith(("death_rate", "mortality_", "median_wait", "frac_", "n_patients"))]
policy_raw = get_policy(df, {fac_e}, {amb_e})
stove_raw  = get_policy(df, 0, 0)

policy = policy_raw.groupby(LEVERS)[OUTCOME_COLS].mean().reset_index()
stove  = stove_raw.groupby(LEVERS)[OUTCOME_COLS].mean().reset_index()
policy = policy.sort_values(LEVERS).reset_index(drop=True)
stove  = stove.sort_values(LEVERS).reset_index(drop=True)

assert len(policy) == len(stove), f"Mismatch: {{len(policy)}} vs {{len(stove)}}"
assert (policy[LEVERS].values == stove[LEVERS].values).all()

delta = policy["death_rate_T1_BCAS"].values - stove["death_rate_T1_BCAS"].values
print(f"{{len(policy)}} scenarios")
print(f"Delta: min={{delta.min():.4f}}, median={{np.median(delta):.4f}}, mean={{delta.mean():.4f}}, max={{delta.max():.4f}}")
print(f"Helps: {{(delta < 0).sum()}}, Indifferent: {{(delta == 0).sum()}}, Hurts: {{(delta > 0).sum()}}")

x = policy[UNCERTAINTIES].astype(float)

def show_tradeoff_highlighted(box, step=None):
    fig = box.show_tradeoff()
    if step is not None:
        traj = box.peeling_trajectory
        if step in traj.index:
            row = traj.loc[step]
            ax = fig.axes[0]
            ax.plot(row["coverage"], row["density"], "o", color="red", markersize=14, zorder=5,
                    markerfacecolor="none", markeredgecolor="red", markeredgewidth=2.5)
    plt.show()


""")

    md(cells, "## Delta distribution")

    code(cells, f"""sorted_d = np.sort(delta)
cdf = np.arange(1, len(sorted_d) + 1) / len(sorted_d)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(sorted_d, cdf, color="steelblue", linewidth=1.5)
axes[0].axvline(0, color="black", linestyle="--", alpha=0.5)
axes[0].set_xlabel("Delta mortality ({name} pooling - stovepipe)")
axes[0].set_ylabel("Cumulative fraction")
axes[0].set_title("CDF — {label} delta (old ranges)")
axes[0].grid(True, alpha=0.3)

for p, ls in [(0.25, "--"), (0.50, "-"), (0.75, "--"), (0.90, ":"), (0.95, ":")]:
    val = np.percentile(delta, p * 100)
    axes[0].axhline(p, color="grey", linestyle=ls, alpha=0.5, linewidth=0.8)
    axes[0].axvline(val, color="grey", linestyle=ls, alpha=0.5, linewidth=0.8)
    axes[0].text(val, p + 0.02, f"p{{int(p*100)}}={{val:.3f}}", fontsize=8, ha="center")

axes[1].plot(sorted_d, 1 - cdf, color="#e74c3c", linewidth=1.5)
axes[1].axvline(0, color="black", linestyle="--", alpha=0.5)
axes[1].set_xlabel("Delta mortality ({name} pooling - stovepipe)")
axes[1].set_ylabel("Fraction exceeding threshold")
axes[1].set_title("CCDF — Fraction above threshold")
axes[1].grid(True, alpha=0.3)

for t in [-0.10, -0.05, 0.0, 0.05, 0.10]:
    frac = (delta > t).mean()
    axes[1].plot(t, frac, "o", color="black", markersize=4)
    axes[1].annotate(f"{{t:.2f}} -> {{frac:.1%}}", (t, frac), textcoords="offset points",
                     xytext=(8, 5), fontsize=8)

plt.tight_layout()
plt.show()""")

    sections = [
        ("Pooling helps", "delta < 0", "box1", "p_box1", "STEP_1", 0.2, "1"),
        ("Indifferent", "delta == 0.00", "box2", "p_box2", "STEP_2", 0.5, "2"),
        ("Pooling hurts", "delta > 0", "box3", "p_box3", "STEP_3", 0.5, "3"),
    ]

    for section_title, condition, box_var, prim_var, step_var, threshold, num in sections:
        md(cells, f"## PRIM — {section_title}")

        code(cells, f"""# === {section_title}: {condition} ===
y = {condition}

print(f"Target scenarios: {{y.sum()}} / {{len(y)}} ({{y.mean():.1%}})")
{prim_var} = prim.Prim(x, y, threshold={threshold})
{box_var} = {prim_var}.find_box()
{box_var}.show_tradeoff()
plt.show()""")

        code(cells, f"""{step_var} = None   # e.g. 10, 15, 20

if {step_var} is not None:
    {box_var}.select({step_var})
show_tradeoff_highlighted({box_var}, step={step_var})

{box_var}.inspect(style="graph")
plt.show()""")

        code(cells, f"""{box_var}.inspect()""")

        code(cells, f"""from scipy.stats import bootstrap as _bootstrap
_in = delta[{box_var}.yi]
_out = np.delete(delta, {box_var}.yi)

fig, ax = plt.subplots(figsize=(8, 4))
bp = ax.boxplot([_in, _out], labels=["Inside box", "Outside box"], patch_artist=True, showfliers=False,
                medianprops=dict(color="black", linewidth=2))
bp["boxes"][0].set_facecolor("steelblue"); bp["boxes"][0].set_alpha(0.7)
bp["boxes"][1].set_facecolor("#999999"); bp["boxes"][1].set_alpha(0.7)
ax.axhline(0, color="black", linestyle="--", alpha=0.5)
ax.set_ylabel("Delta mortality")
ax.grid(True, alpha=0.3, axis="y")
plt.title(f"Delta inside vs outside PRIM box (n={{len(_in)}} vs {{len(_out)}})")
plt.tight_layout()
plt.show()

_ci_in = _bootstrap((_in,), np.median, confidence_level=0.95, n_resamples=9999, method='percentile').confidence_interval
print(f"Inside:  n={{len(_in)}}, helps={{(_in<0).mean():.1%}}, hurts={{(_in>0).mean():.1%}}, median={{np.median(_in):+.4f}}, 95% CI [{{_ci_in.low:+.4f}}, {{_ci_in.high:+.4f}}]")
print(f"Outside: n={{len(_out)}}, helps={{(_out<0).mean():.1%}}, hurts={{(_out>0).mean():.1%}}, median={{np.median(_out):+.4f}}")""")

        code(cells, f"""_idx = {box_var}.yi
_l1, _l2 = "{label}", "Stovepipe"
_df1, _df2 = policy, stove

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
_c1, _c2 = "#e74c3c", "#2ecc71"
_w = 0.35

for _ai, (_col, _title) in enumerate([
    ("mortality_ccp_transport_T1", "CCP transport mortality"),
    ("mortality_r1_care_T1", "R1 care mortality"),
    ("death_rate_T1_BCAS", "Overall mortality"),
]):
    _d1 = _df1[_col].values[_idx]
    _d2 = _df2[_col].values[_idx]
    _bp1 = axes[_ai].boxplot([_d1], positions=[-_w/2], widths=_w*0.8, patch_artist=True,
                              showfliers=False, medianprops=dict(color="black", linewidth=2))
    _bp2 = axes[_ai].boxplot([_d2], positions=[_w/2], widths=_w*0.8, patch_artist=True,
                              showfliers=False, medianprops=dict(color="black", linewidth=2))
    _bp1["boxes"][0].set_facecolor(_c1); _bp1["boxes"][0].set_alpha(0.7)
    _bp2["boxes"][0].set_facecolor(_c2); _bp2["boxes"][0].set_alpha(0.7)
    axes[_ai].set_xticks([0])
    axes[_ai].set_xticklabels([f"Inside box\\nn={{len(_idx)}}"])
    axes[_ai].set_title(_title)
    axes[_ai].set_ylabel("Mortality rate")
    axes[_ai].grid(True, alpha=0.3, axis="y")
    for _bp, _xp in [(_bp1, -_w/2), (_bp2, _w/2)]:
        _med = _bp["medians"][0].get_ydata()[0]
        axes[_ai].text(_xp, _med, f"  {{_med:.4f}}", ha="left", fontsize=8, fontweight="bold", va="bottom")

axes[0].legend([_bp1["boxes"][0], _bp2["boxes"][0]], [_l1, _l2], loc="upper right")
plt.suptitle(f"Mortality inside PRIM box (n={{len(_idx)}})", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()

for _col in ["mortality_ccp_transport_T1", "mortality_r1_care_T1"]:
    _f1 = (_df1[_col].values[_idx] > 0).mean()
    _f2 = (_df2[_col].values[_idx] > 0).mean()
    print(f"  {{_col}}: {{_l1}}={{_f1:.1%}}, {{_l2}}={{_f2:.1%}}")

from scipy.stats import bootstrap as _bootstrap
print(f"\\nMedian mortality with 95% CI inside PRIM box (n={{len(_idx)}}):")
for _col in ["mortality_ccp_transport_T1", "mortality_r1_care_T1", "death_rate_T1_BCAS"]:
    _vals = _df1[_col].values[_idx]
    if len(set(_vals)) > 1:
        _ci = _bootstrap((_vals,), np.median, confidence_level=0.95, n_resamples=9999, method='percentile').confidence_interval
        print(f"  {{_col}}: {{_l1}} median={{np.median(_vals):.4f}} CI=[{{_ci.low:.4f}}, {{_ci.high:.4f}}]")
    else:
        print(f"  {{_col}}: {{_l1}} median={{np.median(_vals):.4f}} (constant)")
    _vals2 = _df2[_col].values[_idx]
    if len(set(_vals2)) > 1:
        _ci2 = _bootstrap((_vals2,), np.median, confidence_level=0.95, n_resamples=9999, method='percentile').confidence_interval
        print(f"  {{_col}}: {{_l2}} median={{np.median(_vals2):.4f}} CI=[{{_ci2.low:.4f}}, {{_ci2.high:.4f}}]")
    else:
        print(f"  {{_col}}: {{_l2}} median={{np.median(_vals2):.4f}} (constant)")""")


    md(cells, "## Summary: delta across PRIM boxes (helps / indifferent / hurts)")

    code(cells, f"""from scipy.stats import bootstrap as _bootstrap

_d1 = delta[box1.yi]
_d2 = delta[box2.yi]
_d3 = delta[box3.yi]

fig, ax = plt.subplots(figsize=(10, 5))
bp = ax.boxplot([_d1, _d2, _d3], labels=["Helps (box1)", "Indifferent (box2)", "Hurts (box3)"],
                patch_artist=True, showfliers=False, medianprops=dict(color="black", linewidth=2))
colors = ["#2ecc71", "#f39c12", "#e74c3c"]
for b, c in zip(bp["boxes"], colors):
    b.set_facecolor(c); b.set_alpha(0.7)
ax.axhline(0, color="black", linestyle="--", alpha=0.5)
ax.set_ylabel("Delta mortality")
ax.grid(True, alpha=0.3, axis="y")
plt.title("Delta distribution across PRIM boxes")
plt.tight_layout()
plt.show()

for bname, _d in [("Helps", _d1), ("Indifferent", _d2), ("Hurts", _d3)]:
    if len(set(_d)) > 1:
        _ci = _bootstrap((_d,), np.median, confidence_level=0.95, n_resamples=9999, method='percentile').confidence_interval
        print(f"{{bname:<12}} n={{len(_d):>5}}, helps={{(_d<0).mean():>5.1%}}, hurts={{(_d>0).mean():>5.1%}}, median={{np.median(_d):+.4f}}, 95% CI [{{_ci.low:+.4f}}, {{_ci.high:+.4f}}]")
    else:
        print(f"{{bname:<12}} n={{len(_d):>5}}, helps={{(_d<0).mean():>5.1%}}, hurts={{(_d>0).mean():>5.1%}}, median={{np.median(_d):+.4f}}")""")

    return cells


for pol in POLICIES:
    cells = build_notebook(pol)
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "cells": cells,
    }

    out = pathlib.Path(__file__).parent / f"{pol['name']}_vs_stove_prim_old_ranges.ipynb"
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"Wrote {out.name}")
