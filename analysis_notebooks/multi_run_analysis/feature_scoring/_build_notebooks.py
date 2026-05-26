"""
Build feature-scoring analysis notebooks.

Generates:
  - template/feature_scoring_template.ipynb   (FOLDER_NAME placeholder)
  - feature_scoring_<name>.ipynb              (one per entry in INSTANCES)
"""
import json
from pathlib import Path

HERE = Path(__file__).parent

# ── Instances to generate ───────────────────────────────────────────────────
INSTANCES = {
    "elia_midterm_16k": "20260516_170952_scenario_elia_midterm",
}

# ── Notebook skeleton ───────────────────────────────────────────────────────

_cell_counter = 0

def _cell(source, cell_type="code"):
    global _cell_counter
    _cell_counter += 1
    cell = {
        "cell_type": cell_type,
        "id": f"cell-{_cell_counter:04d}",
        "metadata": {},
        "source": source.split("\n") if isinstance(source, str) else source,
    }
    if cell_type == "code":
        cell["outputs"] = []
        cell["execution_count"] = None
    return cell


def _md(source):
    return _cell(source, cell_type="markdown")


def build_cells(folder_name_expr: str):
    """Return list of notebook cells. folder_name_expr is either a quoted
    string (instance) or bare FOLDER_NAME (template placeholder)."""
    cells = []

    # ── 0. Title ────────────────────────────────────────────────────────
    cells.append(_md(
        "# Feature Scoring — Input × Outcome Importance Matrix\n"
        "\n"
        "Uses `ema_workbench.analysis.feature_scoring` (Extra Trees) to\n"
        "compute how much each uncertainty / lever drives each outcome.\n"
        "Produces a heatmap: inputs on one axis, outcomes on the other,\n"
        "cells coloured by importance score."
    ))

    # ── 1. Bootstrap paths ──────────────────────────────────────────────
    cells.append(_cell(
        "import sys\n"
        "from pathlib import Path\n"
        "_root = Path(\".\").resolve()\n"
        "while not (_root / \"paths.py\").exists():\n"
        "    _root = _root.parent\n"
        "sys.path.insert(0, str(_root))\n"
        "import paths"
    ))

    # ── 2. Config ───────────────────────────────────────────────────────
    cells.append(_cell(
        f"FOLDER_NAME = {folder_name_expr}\n"
        "\n"
        "# Outcomes to include in the heatmap (set to None for all scalar outcomes)\n"
        "SELECTED_OUTCOMES = [\n"
        "    \"death_rate\",\n"
        "    \"safe_rate\",\n"
        "    \"death_rate_T1_BCAS\",\n"
        "    \"death_rate_T2_BCAS\",\n"
        "    \"death_rate_T3_BCAS\",\n"
        "    \"mean_wait_care_T1_BCAS\",\n"
        "    \"mean_wait_care_T2_BCAS\",\n"
        "    \"mean_wait_care_T3_BCAS\",\n"
        "    \"mean_wait_transport_T1_BCAS\",\n"
        "    \"mean_wait_transport_T2_BCAS\",\n"
        "    \"mean_wait_transport_T3_BCAS\",\n"
        "    \"total_transport_time\",\n"
        "]\n"
    ))

    # ── 3. Load data ───────────────────────────────────────────────────
    cells.append(_cell(
        "import pandas as pd\n"
        "import numpy as np\n"
        "\n"
        "run_folder = paths.MULTI_RUN_OUTPUT / FOLDER_NAME\n"
        "df = pd.read_csv(run_folder / \"results.csv\")\n"
        "print(f\"Loaded {len(df)} runs  |  {df['scenario'].nunique()} scenarios  |  \"\n"
        "      f\"{df['policy'].nunique()} distinct policies\")"
    ))

    # ── 4. Identify columns ────────────────────────────────────────────
    cells.append(_cell(
        "# Columns before 'scenario' are inputs; after 'model' are outcomes.\n"
        "meta_cols  = [\"scenario\", \"policy\", \"model\"]\n"
        "meta_start = min(df.columns.get_loc(c) for c in meta_cols if c in df.columns)\n"
        "input_cols = df.columns[:meta_start].tolist()\n"
        "model_idx  = df.columns.get_loc(\"model\")\n"
        "outcome_cols = df.columns[model_idx + 1:].tolist()\n"
        "\n"
        "print(f\"Inputs  ({len(input_cols)}): {input_cols}\")\n"
        "print(f\"Outcomes ({len(outcome_cols)}): {len(outcome_cols)} total\")\n"
        "\n"
        "if SELECTED_OUTCOMES is not None:\n"
        "    outcome_cols = [c for c in SELECTED_OUTCOMES if c in outcome_cols]\n"
        "    print(f\"Selected {len(outcome_cols)} outcomes for scoring\")"
    ))

    # ── 5. Prepare x and y ─────────────────────────────────────────────
    cells.append(_cell(
        "# x = inputs (uncertainties + levers), y = dict of outcome arrays\n"
        "# Drop rows with NaN in any selected outcome\n"
        "keep_cols = input_cols + outcome_cols\n"
        "df_clean = df[keep_cols].dropna()\n"
        "print(f\"Kept {len(df_clean)}/{len(df)} rows after dropping NaN\")\n"
        "\n"
        "x = df_clean[input_cols].astype(float)\n"
        "y = {col: df_clean[col].values for col in outcome_cols}"
    ))

    # ── 6. Feature scoring ─────────────────────────────────────────────
    cells.append(_cell(
        "from ema_workbench.analysis import feature_scoring\n"
        "\n"
        "scores = feature_scoring.get_feature_scores_all(x, y)\n"
        "print(f\"Score matrix shape: {scores.shape}\")\n"
        "print(f\"Inputs: {list(scores.index)}\")\n"
        "print(f\"Outcomes: {list(scores.columns)}\")\n"
        "scores"
    ))

    # ── 7. Heatmap ─────────────────────────────────────────────────────
    cells.append(_cell(
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(max(12, len(outcome_cols) * 1.2),\n"
        "                                max(6, len(input_cols) * 0.5)))\n"
        "sns.heatmap(\n"
        "    scores,\n"
        "    annot=True,\n"
        "    fmt=\".2f\",\n"
        "    cmap=\"YlOrRd\",\n"
        "    linewidths=0.5,\n"
        "    ax=ax,\n"
        "    vmin=0,\n"
        "    vmax=1,\n"
        ")\n"
        "ax.set_title(\"Feature Scoring — Input Importance per Outcome\", fontsize=14)\n"
        "ax.set_xlabel(\"Outcome\")\n"
        "ax.set_ylabel(\"Input (uncertainty / lever)\")\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))

    # ── 8. Top-N per outcome ───────────────────────────────────────────
    cells.append(_md(
        "## Top inputs per outcome\n"
        "\n"
        "For each outcome, list the inputs ranked by importance score."
    ))

    cells.append(_cell(
        "TOP_N = 5\n"
        "\n"
        "for outcome in scores.columns:\n"
        "    ranked = scores[outcome].sort_values(ascending=False)\n"
        "    print(f\"\\n{outcome}:\")\n"
        "    for i, (inp, sc) in enumerate(ranked.head(TOP_N).items()):\n"
        "        bar = '█' * int(sc * 40)\n"
        "        print(f\"  {i+1}. {inp:35s} {sc:.3f}  {bar}\")"
    ))

    # ── 9. Per-policy feature scoring ──────────────────────────────────
    cells.append(_md(
        "## Per-Policy Feature Scoring\n"
        "\n"
        "Run feature scoring separately for each policy (lever combination)\n"
        "to see whether input importance shifts across policies."
    ))

    cells.append(_cell(
        "lever_cols = [c for c in input_cols if c in [\"facility_eligibility\", \"ambulance_eligibility\"]]\n"
        "uncertainty_cols = [c for c in input_cols if c not in lever_cols]\n"
        "\n"
        "policies = df[lever_cols].drop_duplicates().sort_values(lever_cols).reset_index(drop=True)\n"
        "print(f\"{len(policies)} distinct policies:\")\n"
        "display(policies)\n"
        "\n"
        "for idx, policy_row in policies.iterrows():\n"
        "    mask = pd.Series(True, index=df.index)\n"
        "    for col in lever_cols:\n"
        "        mask &= df[col] == policy_row[col]\n"
        "    df_pol = df.loc[mask, uncertainty_cols + outcome_cols].dropna()\n"
        "    \n"
        "    x_pol = df_pol[uncertainty_cols].astype(float)\n"
        "    y_pol = {col: df_pol[col].values for col in outcome_cols}\n"
        "    \n"
        "    scores_pol = feature_scoring.get_feature_scores_all(x_pol, y_pol)\n"
        "    \n"
        "    label = \", \".join(f\"{c}={int(policy_row[c])}\" for c in lever_cols)\n"
        "    fig, ax = plt.subplots(figsize=(max(12, len(outcome_cols) * 1.2),\n"
        "                                    max(6, len(uncertainty_cols) * 0.5)))\n"
        "    sns.heatmap(\n"
        "        scores_pol,\n"
        "        annot=True,\n"
        "        fmt=\".2f\",\n"
        "        cmap=\"YlOrRd\",\n"
        "        linewidths=0.5,\n"
        "        ax=ax,\n"
        "        vmin=0,\n"
        "        vmax=1,\n"
        "    )\n"
        "    ax.set_title(f\"Feature Scoring — {label}\", fontsize=14)\n"
        "    ax.set_xlabel(\"Outcome\")\n"
        "    ax.set_ylabel(\"Uncertainty\")\n"
        "    plt.tight_layout()\n"
        "    plt.show()"
    ))

    return cells


def make_notebook(cells):
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "cells": cells,
    }


def write_notebook(path, cells):
    nb = make_notebook(cells)
    # Fix: split source lines properly for JSON
    for cell in nb["cells"]:
        if isinstance(cell["source"], str):
            cell["source"] = cell["source"]
        else:
            src = "\n".join(cell["source"])
            cell["source"] = src
    path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"  Written: {path}")


if __name__ == "__main__":
    # Template
    template_cells = build_cells("FOLDER_NAME")
    write_notebook(HERE / "template" / "feature_scoring_template.ipynb", template_cells)

    # Instances
    for name, folder in INSTANCES.items():
        instance_cells = build_cells(f'"{folder}"')
        write_notebook(HERE / f"feature_scoring_{name}.ipynb", instance_cells)

    print("Done.")
