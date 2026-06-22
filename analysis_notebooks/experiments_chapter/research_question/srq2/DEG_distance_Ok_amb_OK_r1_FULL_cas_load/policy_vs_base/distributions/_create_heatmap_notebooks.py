"""Generate delta heatmap notebooks for each policy vs stovepipe (2 ambulances)."""
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

    md(cells, f"# {label} vs Stovepipe — Delta heatmaps across distance space (2 ambulances)")

    code(cells, f"""import sys
from pathlib import Path
_root = Path(".").resolve()
while not (_root / "paths.py").exists():
    _root = _root.parent
sys.path.insert(0, str(_root))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(_root / "data_cluster/resource_grid/results.csv")
df = df[df["n_amb_r1_to_ccp"] == 2].reset_index(drop=True)
print(f"Loaded {{len(df)}} runs (2 ambulances)")

MATCH_COLS = ["seed", "ccp_r1_same_bn_min", "ccp_r1_inter_bn_min"]

stove = df[(df["facility_eligibility"] == 0) & (df["ambulance_eligibility"] == 0)]
policy = df[(df["facility_eligibility"] == {fac_e}) & (df["ambulance_eligibility"] == {amb_e})]
print(f"Stovepipe: {{len(stove)}}, {label}: {{len(policy)}}")
print(f"bcas rates: {{sorted(df['bcas_rate'].unique())}}")""")

    md(cells, f"## Fraction of seeds where {name} pooling hurts (delta > 0) per distance cell")

    code(cells, f"""for bcas in sorted(df["bcas_rate"].unique()):
    s = stove[stove["bcas_rate"] == bcas].sort_values(MATCH_COLS).reset_index(drop=True)
    p = policy[policy["bcas_rate"] == bcas].sort_values(MATCH_COLS).reset_index(drop=True)
    assert (s[MATCH_COLS].values == p[MATCH_COLS].values).all()

    merged = s[MATCH_COLS].copy()
    merged["delta"] = p["death_rate_T1_BCAS"].values - s["death_rate_T1_BCAS"].values
    merged["hurts"] = merged["delta"] > 0

    grid = merged.groupby(["ccp_r1_same_bn_min", "ccp_r1_inter_bn_min"]).agg(
        frac_hurts=("hurts", "mean"),
        n_hurts=("hurts", "sum"),
        n_total=("hurts", "size"),
    ).reset_index()

    pivot_frac = grid.pivot(index="ccp_r1_inter_bn_min", columns="ccp_r1_same_bn_min", values="frac_hurts")
    pivot_n = grid.pivot(index="ccp_r1_inter_bn_min", columns="ccp_r1_same_bn_min", values="n_hurts")

    fig, ax = plt.subplots(figsize=(16, 10))
    im = ax.imshow(pivot_frac.values, cmap="YlOrRd", origin="lower", aspect="auto",
                   interpolation="nearest", vmin=0, vmax=1)

    ax.set_xticks(range(len(pivot_frac.columns)))
    ax.set_xticklabels(pivot_frac.columns)
    ax.set_yticks(range(len(pivot_frac.index)))
    ax.set_yticklabels(pivot_frac.index)
    ax.set_xlabel("Own R1 distance (min)", fontsize=13)
    ax.set_ylabel("Neighbor R1 distance (min)", fontsize=13)
    ax.set_title(f"{label} hurts (delta > 0) — bcas={{bcas}} (n=100 seeds/cell)", fontsize=14)

    for i in range(len(pivot_frac.index)):
        for j in range(len(pivot_frac.columns)):
            frac = pivot_frac.values[i, j]
            n = int(pivot_n.values[i, j])
            color = "white" if frac > 0.5 else "black"
            ax.text(j, i, f"{{n}}", ha="center", va="center", fontsize=7, fontweight="bold", color=color)

    cb = plt.colorbar(im, ax=ax)
    cb.set_label("Fraction of seeds where {name} pooling hurts", fontsize=11)
    plt.tight_layout()
    plt.show()""")

    md(cells, f"## Fraction of seeds where {name} pooling helps (delta < 0) per distance cell")

    code(cells, f"""for bcas in sorted(df["bcas_rate"].unique()):
    s = stove[stove["bcas_rate"] == bcas].sort_values(MATCH_COLS).reset_index(drop=True)
    p = policy[policy["bcas_rate"] == bcas].sort_values(MATCH_COLS).reset_index(drop=True)

    merged = s[MATCH_COLS].copy()
    merged["delta"] = p["death_rate_T1_BCAS"].values - s["death_rate_T1_BCAS"].values
    merged["helps"] = merged["delta"] < 0

    grid = merged.groupby(["ccp_r1_same_bn_min", "ccp_r1_inter_bn_min"]).agg(
        frac_helps=("helps", "mean"),
        n_helps=("helps", "sum"),
    ).reset_index()

    pivot_frac = grid.pivot(index="ccp_r1_inter_bn_min", columns="ccp_r1_same_bn_min", values="frac_helps")
    pivot_n = grid.pivot(index="ccp_r1_inter_bn_min", columns="ccp_r1_same_bn_min", values="n_helps")

    fig, ax = plt.subplots(figsize=(16, 10))
    im = ax.imshow(pivot_frac.values, cmap="YlGn", origin="lower", aspect="auto",
                   interpolation="nearest", vmin=0, vmax=1)

    ax.set_xticks(range(len(pivot_frac.columns)))
    ax.set_xticklabels(pivot_frac.columns)
    ax.set_yticks(range(len(pivot_frac.index)))
    ax.set_yticklabels(pivot_frac.index)
    ax.set_xlabel("Own R1 distance (min)", fontsize=13)
    ax.set_ylabel("Neighbor R1 distance (min)", fontsize=13)
    ax.set_title(f"{label} helps (delta < 0) — bcas={{bcas}} (n=100 seeds/cell)", fontsize=14)

    for i in range(len(pivot_frac.index)):
        for j in range(len(pivot_frac.columns)):
            frac = pivot_frac.values[i, j]
            n = int(pivot_n.values[i, j])
            color = "white" if frac > 0.5 else "black"
            ax.text(j, i, f"{{n}}", ha="center", va="center", fontsize=7, fontweight="bold", color=color)

    cb = plt.colorbar(im, ax=ax)
    cb.set_label("Fraction of seeds where {name} pooling helps", fontsize=11)
    plt.tight_layout()
    plt.show()""")

    md(cells, f"## Median delta ({name} - stovepipe) per distance cell")

    code(cells, f"""for bcas in sorted(df["bcas_rate"].unique()):
    s = stove[stove["bcas_rate"] == bcas].sort_values(MATCH_COLS).reset_index(drop=True)
    p = policy[policy["bcas_rate"] == bcas].sort_values(MATCH_COLS).reset_index(drop=True)

    merged = s[MATCH_COLS].copy()
    merged["delta"] = p["death_rate_T1_BCAS"].values - s["death_rate_T1_BCAS"].values

    grid = merged.groupby(["ccp_r1_same_bn_min", "ccp_r1_inter_bn_min"])["delta"].median().reset_index()
    pivot = grid.pivot(index="ccp_r1_inter_bn_min", columns="ccp_r1_same_bn_min", values="delta")

    vmax = max(abs(pivot.values.min()), abs(pivot.values.max()), 0.01)

    fig, ax = plt.subplots(figsize=(16, 10))
    im = ax.imshow(pivot.values, cmap="RdYlGn_r", origin="lower", aspect="auto",
                   interpolation="nearest", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Own R1 distance (min)", fontsize=13)
    ax.set_ylabel("Neighbor R1 distance (min)", fontsize=13)
    ax.set_title(f"{label} vs Stovepipe — Median delta (bcas={{bcas}}, n=100 seeds/cell)", fontsize=14)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            color = "white" if abs(val) > vmax * 0.5 else "black"
            ax.text(j, i, f"{{val:+.2f}}", ha="center", va="center", fontsize=7, fontweight="bold", color=color)

    cb = plt.colorbar(im, ax=ax)
    cb.set_label("Median delta (red = {name} hurts, green = {name} helps)", fontsize=11)
    plt.tight_layout()
    plt.show()""")

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

    out = pathlib.Path(__file__).parent / f"{pol['name']}_vs_stove_heatmap.ipynb"
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"Wrote {out.name}")
