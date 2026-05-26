"""Rebuild tree_pooling_regret notebooks with all 5 binary trees per combo."""
import json, copy, pathlib

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.split("\n"),
            "id": None}

def code(source):
    return {"cell_type": "code", "metadata": {}, "source": source.split("\n"),
            "outputs": [], "execution_count": None, "id": None}

HEADER = md("""# Decision Tree — When Does Pooling Help or Hurt Survival?

Uses `sklearn.tree.DecisionTreeClassifier` to identify which uncertainty
dimensions predict whether a pooling policy improves or damages `safe_rate`
relative to the strict baseline (both levers = 0).

Runs separately for each of the 3 lever combos:
- **Facility only (1,0)**
- **Ambulance only (0,1)**
- **Both (1,1)**

For each combo, five binary trees are trained:
1. **Helps** — Δ safe_rate > 0
2. **Helps a lot** — Δ safe_rate > threshold
3. **Harms** — Δ safe_rate < 0
4. **Harms a lot** — Δ safe_rate < −threshold
5. **4-class** — harms a lot / harms a bit / helps a bit / helps a lot""")

SETUP = code("""import sys
from pathlib import Path
_root = Path(".").resolve()
while not (_root / "paths.py").exists():
    _root = _root.parent
sys.path.insert(0, str(_root))
import paths""")

def config_cell(folder_name):
    return code(f"""FOLDER_NAME = "{folder_name}"
LEVER_COLS  = ["facility_eligibility", "ambulance_eligibility"]
OUTCOME     = "safe_rate"
THRESHOLD   = 0.05  # 5 pp boundary between "a bit" and "a lot" """)

LOAD = code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

run_folder = paths.MULTI_RUN_OUTPUT / FOLDER_NAME

df = pd.read_csv(run_folder / "results.csv")
print(f"Loaded {len(df)} runs  |  {df['scenario'].nunique()} scenarios  |  "
      f"{df['policy'].nunique()} distinct policies")""")

PREP = code("""meta_cols      = ["scenario", "policy", "model"]
meta_start     = min(df.columns.get_loc(c) for c in meta_cols if c in df.columns)
input_cols     = df.columns[:meta_start].tolist()
uncertainty_cols = [c for c in input_cols if c not in LEVER_COLS]

feature_cols = [c for c in uncertainty_cols if c != "seed"]

baseline_mask = df[LEVER_COLS].sum(axis=1) == 0

baseline_outcome = (
    df.loc[baseline_mask]
    .groupby("scenario")[OUTCOME]
    .mean()
    .rename("strict")
)

COMBOS = {
    "Facility only (1,0)": {"facility_eligibility": 1, "ambulance_eligibility": 0},
    "Ambulance only (0,1)": {"facility_eligibility": 0, "ambulance_eligibility": 1},
    "Both (1,1)":           {"facility_eligibility": 1, "ambulance_eligibility": 1},
}

CLASS_NAMES_4 = ["Harms a lot", "Harms a bit", "Helps a bit", "Helps a lot"]

def make_4class(delta, threshold):
    labels = np.full(len(delta), 1, dtype=int)
    labels[delta < -threshold] = 0
    labels[(delta >= 0) & (delta < threshold)] = 2
    labels[delta >= threshold] = 3
    return labels

x_base = (
    df.loc[baseline_mask]
    .drop_duplicates(subset="scenario")
    .set_index("scenario")
)

combo_data = {}
for label, lvls in COMBOS.items():
    mask = pd.Series(True, index=df.index)
    for col, val in lvls.items():
        mask &= df[col] == val
    combo_sr = df.loc[mask].groupby("scenario")[OUTCOME].mean().rename("combo")
    comp = pd.concat([baseline_outcome, combo_sr], axis=1).dropna()
    comp["delta"] = comp["combo"] - comp["strict"]

    X = x_base.loc[comp.index, feature_cols].reset_index(drop=True).astype(float)
    y4 = make_4class(comp["delta"].values, THRESHOLD)
    combo_data[label] = {"comp": comp, "X": X, "y4": y4}

    counts = {CLASS_NAMES_4[i]: (y4 == i).sum() for i in range(4)}
    print(f"{label}:  n={len(comp)}  mean \\u0394={comp['delta'].mean():.4f}  "
          + "  ".join(f"{k}: {v}" for k, v in counts.items()))

print(f"\\nFeatures ({len(feature_cols)}): {feature_cols}")""")

RUN_TREE = code("""def run_tree(X, y, label, question, class_names, max_depth=4, test_size=0.3):
    unique, counts_arr = np.unique(y, return_counts=True)
    present = [c for c, n in zip(unique, counts_arr) if n >= 3]

    sep = "=" * 70
    print(sep)
    print(f"  {label} — {question}")
    for c, n in zip(unique, counts_arr):
        cname = class_names[c] if isinstance(class_names, (list, dict)) else str(c)
        print(f"    {cname}: {n}")
    print(sep)

    if len(present) < 2:
        print("  Fewer than 2 classes with >=3 samples — skipping.\\n")
        return None, None, None

    present_names = [class_names[c] for c in present]
    mask = np.isin(y, present)
    X_f, y_f = X[mask].reset_index(drop=True), y[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X_f, y_f, test_size=test_size, random_state=42, stratify=y_f,
    )

    tree = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=max(10, int(0.02 * len(y_train))),
        random_state=42,
    )
    tree.fit(X_train, y_train)

    y_pred_test = tree.predict(X_test)
    print(f"  Train acc: {accuracy_score(y_train, tree.predict(X_train)):.3f}  "
          f"Test acc: {accuracy_score(y_test, y_pred_test):.3f}  "
          f"(train={len(X_train)}, test={len(X_test)})")
    print()
    print(classification_report(y_test, y_pred_test,
                                labels=present, target_names=present_names, zero_division=0))

    importances = pd.Series(tree.feature_importances_, index=X_f.columns)
    importances = importances[importances > 0.01].sort_values(ascending=False)
    if not importances.empty:
        print("  Feature importances (>1%):")
        for feat, imp in importances.items():
            print(f"    {feat:30s}  {imp:.3f}")
    print()

    # Confusion matrix only
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred_test, labels=present,
        display_labels=present_names, ax=ax,
        cmap="Blues", colorbar=False,
    )
    ax.set_title(f"{label} — {question}\\nConfusion matrix (test set)")
    plt.tight_layout()
    plt.show()

    return tree, X_test, y_test""")

_display_rules_src = (
    'def display_rules(tree, X_test, y_test, feature_names, class_names, target_class=1, label=""):\n'
    '    """Extract and display every path leading to target_class with per-rule metrics."""\n'
    '    if tree is None:\n'
    '        return\n'
    '\n'
    '    tree_ = tree.tree_\n'
    '    if isinstance(class_names, dict):\n'
    '        idx_to_name = class_names\n'
    '    else:\n'
    '        idx_to_name = {i: n for i, n in enumerate(class_names)}\n'
    '\n'
    '    def fmt_thresh(val):\n'
    '        """Format threshold with enough precision to avoid ambiguity."""\n'
    '        if abs(val) < 1:\n'
    '            return f"{val:.4g}"\n'
    '        return f"{val:.1f}"\n'
    '\n'
    '    def recurse(node, path):\n'
    '        if tree_.children_left[node] == tree_.children_right[node]:\n'
    '            predicted = tree_.value[node].argmax()\n'
    '            original_class = tree.classes_[predicted]\n'
    '            return [(path[:], original_class, tree_.value[node].flatten())]\n'
    '        feat = feature_names[tree_.feature[node]]\n'
    '        thresh = tree_.threshold[node]\n'
    '        t = fmt_thresh(thresh)\n'
    '        results = []\n'
    '        results += recurse(tree_.children_left[node],\n'
    '                           path + [f"{feat} ≤ {t}"])\n'
    '        results += recurse(tree_.children_right[node],\n'
    '                           path + [f"{feat} > {t}"])\n'
    '        return results\n'
    '\n'
    '    all_paths = recurse(0, [])\n'
    '    target_paths = [(p, v) for p, cls, v in all_paths if cls == target_class]\n'
    '\n'
    '    if not target_paths:\n'
    '        target_n = idx_to_name.get(target_class, target_class)\n'
    '        print(f"  No paths lead to class {target_n}.\\n")\n'
    '        return\n'
    '\n'
    '    target_name = idx_to_name.get(target_class, str(target_class))\n'
    '    sep = "=" * 70\n'
    '    print(f"\\n{sep}")\n'
    '    print(f"  {label} — Rules predicting: {target_name}")\n'
    '    print(f"{sep}\\n")\n'
    '\n'
    '    y_pred_all = tree.predict(X_test)\n'
    '    leaf_ids = tree.apply(X_test)\n'
    '\n'
    '    def get_leaf_id(path):\n'
    '        node = 0\n'
    '        for condition in path:\n'
    '            if "≤" in condition:\n'
    '                node = tree_.children_left[node]\n'
    '            else:\n'
    '                node = tree_.children_right[node]\n'
    '        return node\n'
    '\n'
    '    for i, (path, train_values) in enumerate(target_paths, 1):\n'
    '        leaf_node = get_leaf_id(path)\n'
    '        in_leaf = leaf_ids == leaf_node\n'
    '        n_test = in_leaf.sum()\n'
    '\n'
    '        if n_test == 0:\n'
    '            continue\n'
    '\n'
    '        correct = ((y_pred_all == target_class) & (y_test == target_class) & in_leaf).sum()\n'
    '        predicted_pos = (y_pred_all[in_leaf] == target_class).sum()\n'
    '        actual_pos = (y_test[in_leaf] == target_class).sum()\n'
    '        total_actual_pos = (y_test == target_class).sum()\n'
    '\n'
    '        precision = correct / predicted_pos if predicted_pos > 0 else 0\n'
    '        recall = correct / total_actual_pos if total_actual_pos > 0 else 0\n'
    '\n'
    '        print(f"  Rule {i}:")\n'
    '        for j, cond in enumerate(path):\n'
    '            prefix = "  IF  " if j == 0 else "  AND "\n'
    '            print(f"    {prefix}{cond}")\n'
    '        print(f"    → {target_name}")\n'
    '        print(f"    Test: {n_test} samples | precision={precision:.0%} | "\n'
    '              f"recall={recall:.0%} (covers {correct}/{total_actual_pos} positives)")\n'
    '        print()'
)
DISPLAY_RULES = {"cell_type": "code", "metadata": {}, "source": _display_rules_src.split("\n"),
                 "outputs": [], "execution_count": None, "id": None}


def combo_cells(tag, short):
    """Generate all cells for one lever combo."""
    # tag = "Facility only (1,0)", short = "fac"
    cells = []
    cells.append(md(f"## {tag}"))

    # --- setup ---
    cells.append(code(f"""cd = combo_data["{tag}"]
X, comp = cd["X"], cd["comp"]"""))

    # --- 1. Helps ---
    cells.append(code(f"""tree_{short}_helps, X_test_{short}_h, y_test_{short}_h = run_tree(
    X, (comp["delta"] > 0).astype(int).values,
    "{tag}", "Does pooling help? (\\u0394 > 0)",
    class_names={{0: "No", 1: "Yes"}},
)"""))
    cells.append(code(f"""display_rules(tree_{short}_helps, X_test_{short}_h, y_test_{short}_h,
              feature_cols, {{0: "No", 1: "Yes"}}, target_class=1,
              label="{tag} — helps")"""))

    # --- 2. Helps a lot ---
    cells.append(code(f"""tree_{short}_lot, X_test_{short}_l, y_test_{short}_l = run_tree(
    X, (comp["delta"] > THRESHOLD).astype(int).values,
    "{tag}", f"Does pooling help a lot? (\\u0394 > {{THRESHOLD:.0%}})",
    class_names={{0: "No", 1: "Yes"}},
)"""))
    cells.append(code(f"""display_rules(tree_{short}_lot, X_test_{short}_l, y_test_{short}_l,
              feature_cols, {{0: "No", 1: "Yes"}}, target_class=1,
              label="{tag} — helps a lot")"""))

    # --- 3. Harms (NEW) ---
    cells.append(code(f"""tree_{short}_harms, X_test_{short}_hr, y_test_{short}_hr = run_tree(
    X, (comp["delta"] < 0).astype(int).values,
    "{tag}", "Does pooling harm? (\\u0394 < 0)",
    class_names={{0: "No", 1: "Yes"}},
)"""))
    cells.append(code(f"""display_rules(tree_{short}_harms, X_test_{short}_hr, y_test_{short}_hr,
              feature_cols, {{0: "No", 1: "Yes"}}, target_class=1,
              label="{tag} — harms")"""))

    # --- 4. Harms a lot (NEW) ---
    cells.append(code(f"""tree_{short}_hlot, X_test_{short}_hl, y_test_{short}_hl = run_tree(
    X, (comp["delta"] < -THRESHOLD).astype(int).values,
    "{tag}", f"Does pooling harm a lot? (\\u0394 < -{{THRESHOLD:.0%}})",
    class_names={{0: "No", 1: "Yes"}},
)"""))
    cells.append(code(f"""display_rules(tree_{short}_hlot, X_test_{short}_hl, y_test_{short}_hl,
              feature_cols, {{0: "No", 1: "Yes"}}, target_class=1,
              label="{tag} — harms a lot")"""))

    # --- 5. 4-class ---
    cells.append(code(f"""tree_{short}_4c, X_test_{short}_4, y_test_{short}_4 = run_tree(
    cd["X"], cd["y4"],
    "{tag}", "4-class",
    class_names=CLASS_NAMES_4,
)"""))
    cells.append(code(f"""for tc in range(4):
    display_rules(tree_{short}_4c, X_test_{short}_4, y_test_{short}_4,
                  feature_cols, CLASS_NAMES_4, target_class=tc,
                  label="{tag} — 4-class")"""))

    return cells


SUMMARY_MD = md("## Summary — Feature Importance Across All Trees")

SUMMARY_CODE = code("""all_trees = {
    ("Facility (1,0)", "helps"): tree_fac_helps,
    ("Facility (1,0)", "helps a lot"): tree_fac_lot,
    ("Facility (1,0)", "harms"): tree_fac_harms,
    ("Facility (1,0)", "harms a lot"): tree_fac_hlot,
    ("Facility (1,0)", "4-class"): tree_fac_4c,
    ("Ambulance (0,1)", "helps"): tree_amb_helps,
    ("Ambulance (0,1)", "helps a lot"): tree_amb_lot,
    ("Ambulance (0,1)", "harms"): tree_amb_harms,
    ("Ambulance (0,1)", "harms a lot"): tree_amb_hlot,
    ("Ambulance (0,1)", "4-class"): tree_amb_4c,
    ("Both (1,1)", "helps"): tree_both_helps,
    ("Both (1,1)", "helps a lot"): tree_both_lot,
    ("Both (1,1)", "harms"): tree_both_harms,
    ("Both (1,1)", "harms a lot"): tree_both_hlot,
    ("Both (1,1)", "4-class"): tree_both_4c,
}

rows = []
for (combo, question), tree in all_trees.items():
    if tree is None:
        continue
    for feat, imp in zip(feature_cols, tree.feature_importances_):
        if imp > 0.01:
            rows.append({"combo": combo, "question": question,
                         "feature": feat, "importance": imp})

if rows:
    imp_df = pd.DataFrame(rows)
    pivot = imp_df.pivot_table(index="feature", columns=["combo", "question"],
                               values="importance", fill_value=0)
    pivot["total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("total", ascending=False).drop(columns="total")

    fig, ax = plt.subplots(figsize=(14, max(4, len(pivot) * 0.5)))
    pivot.plot.barh(ax=ax, stacked=True, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Feature importance (sum across trees)")
    ax.set_title(f"Feature importance summary — {FOLDER_NAME}")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    plt.tight_layout()
    plt.show()

    display(pivot.round(3))
else:
    print("No trees produced meaningful splits.")""")


def build_nb(folder_name):
    cells = [HEADER, SETUP, config_cell(folder_name), LOAD, PREP, RUN_TREE, DISPLAY_RULES]
    cells += combo_cells("Facility only (1,0)", "fac")
    cells += combo_cells("Ambulance only (0,1)", "amb")
    cells += combo_cells("Both (1,1)", "both")
    cells += [SUMMARY_MD, SUMMARY_CODE]

    # Deep copy and assign unique IDs
    final_cells = []
    for i, c in enumerate(cells):
        c2 = copy.deepcopy(c)
        c2["id"] = f"cell-{i}"
        # Ensure source is list of lines
        if isinstance(c2["source"], str):
            c2["source"] = c2["source"].split("\n")
        # Fix: ensure source lines end with \n except last
        lines = []
        src = c2["source"]
        for j, line in enumerate(src):
            if j < len(src) - 1 and not line.endswith("\n"):
                lines.append(line + "\n")
            else:
                lines.append(line)
        c2["source"] = lines
        final_cells.append(c2)

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {"name": "python", "version": "3.11.0"}
        },
        "cells": final_cells
    }
    return nb


BASE = pathlib.Path(r"C:\Users\User\Documents\thesis\medevac_model\analysis_notebooks\multi_run_analysis\tree_pooling_regret")

# Template
nb = build_nb("FOLDER_NAME")
# Fix template: FOLDER_NAME = FOLDER_NAME (not quoted)
for c in nb["cells"]:
    if c["cell_type"] == "code":
        new_src = []
        for line in c["source"]:
            new_src.append(line.replace('"FOLDER_NAME"', 'FOLDER_NAME'))
        c["source"] = new_src
with open(BASE / "template" / "tree_pooling_regret_template.ipynb", "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Wrote template")

# Big run
nb = build_nb("20260515_201832_scenario_elia_midterm")
with open(BASE / "tree_pooling_regret_elia_midterm_big_run.ipynb", "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Wrote big_run")

# Big big run
nb = build_nb("20260516_170952_scenario_elia_midterm")
with open(BASE / "tree_pooling_regret_elia_midterm_big_big_run.ipynb", "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Wrote big_big_run")

print("Done — all 3 notebooks rebuilt.")
