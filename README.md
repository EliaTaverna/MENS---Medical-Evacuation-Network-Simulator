# MEDEVAC Simulation Model

Discrete-event simulation model for military medical evacuation (MEDEVAC), built on `salabim`.

The model represents the chain from casualty generation to definitive care through CCPs, Role 1 facilities, Role 2 facilities, treatment units, ambulances, AXP nodes, and C2 dispatchers. It supports single-run debugging with detailed logs and multi-run experiments with EMA Workbench.

This README is split into five working sections:

1. Datasets
2. Casualty Dataset Builder
3. Dataset Builder
4. Single Run
5. Multi Run

## 1. Datasets

All simulation inputs live under:

```text
data/single_run/input/
```

Scenario YAML files live under:

```text
data/single_run/input/scenarios/
```

The paths inside a scenario YAML are relative to `data/single_run/input/`, not to the repository root.

### Scenario YAML

A scenario is the single-run entry point. It points to all component datasets, policy matrices, casualty settings, runtime options, and output settings.

Current shape:

```yaml
name: scenario_elia_testing
seed: 42
description: Auto-generated scenario

components:
  distance_matrix: components/distances/distance_matrix_elia_testing.xlsx
  coordinates: components/distances/coordinates_elia_testing.xlsx
  axp: components/AXPs/axp_elia_testing.xlsx
  facilities: components/facilities/facilities_elia_testing.xlsx
  units: components/units/units_elia_testing.xlsx
  ambulances: components/ambulances/amb_elia_testing.xlsx
  c2: components/c2/c2_elia_testing.xlsx

policy_matrixes:
  facility_eligibility:
    0: policy_matrixes/facility_eligibility/facility_eligibility_elia_testing_strict.xlsx
    1: policy_matrixes/facility_eligibility/facility_eligibility_elia_testing_loose.xlsx
  ambulance_eligibility:
    0: policy_matrixes/ambulance_eligibility/ambulance_eligibility_elia_testing_strict.xlsx
    1: policy_matrixes/ambulance_eligibility/ambulance_eligibility_elia_testing_loose.xlsx

casualty_generation:
  config: casualty_generation_settings/sampling_settings/config_elia_testing.yaml

levers:
  dispatch_policy: greedy
  facility_eligibility: 0
  ambulance_eligibility: 0

simulation:
  till: 4320
  trace: false
  data_collector_interval: 10.0

video:
  enabled: false
  fps: 24
  time_scale: 5.0
  dpi: 100
  state_log_interval: 5.0
  start_time: 0
  duration: null

logs:
  enabled: true
  path: simulation_logs

outputs:
  patient_log: true
  safe_patients: true
  transport_events: true
  wait_times: true
  facility_states: false
```

The scenario seed is the master seed for a single run. In multi-run experiments, the `seed` column usually comes from the experiment YAML instead.

### Facility ID Naming

All facility IDs produced by the dataset builder follow a hierarchical `role_br{N}_bt{M}_{i}` scheme, where the brigade and (when applicable) battalion are embedded directly in the ID. Battalion-level facilities (CCP, Role 1) carry both a brigade and a battalion slot. Brigade-level facilities (Role 2 basic, Role 2 enhanced, AXP) carry only a brigade slot.

| Type            | Template                     | Parent    | Example IDs                                   |
|-----------------|------------------------------|-----------|-----------------------------------------------|
| CCP             | `ccp_br{N}_bt{M}_{i}`        | battalion | `ccp_br1_bt1_1`, `ccp_br1_bt2_1`, ...         |
| Role 1          | `r1_br{N}_bt{M}_{i}`         | battalion | `r1_br1_bt1_1`, `r1_br2_bt1_1`, ...           |
| Role 2 basic    | `r2b_br{N}_{i}`              | brigade   | `r2b_br1_1`, `r2b_br2_1`, ...                 |
| Role 2 enhanced | `r2e_br{N}_{i}`              | brigade   | `r2e_br1_1`, `r2e_br2_1`, ...                 |
| AXP             | `axp_br{N}_{i}`              | brigade   | `axp_br1_1`, `axp_br2_1`, ...                 |

Battalion numbering **resets per brigade**: every brigade has its own `bt1..bt{N_BN_PER_BRIGADE}`. So `r1_br1_bt1_1` and `r1_br2_bt1_1` are distinct facilities in two different brigades. The parent brigade/battalion is readable from the ID itself — no external mapping needed.

How to read an ID:

```
r1_br2_bt1_1
│   │   │  └─ 1st Role 1 inside this battalion
│   │   └──── battalion 1 (within brigade 2)
│   └──────── brigade 2
└──────────── Role 1

r2b_br3_1
│   │   └─ 1st Role-2-basic inside brigade 3
│   └───── brigade 3
└───────── Role 2 basic
```

The same IDs are used everywhere downstream: distance matrix row/column labels, `facilities.xlsx`, `units.xlsx`, ambulance `start_loc_id` and eligibility columns, C2 rosters, and multi-run lever names. When writing a per-pair distance lever such as `distance_r1_br1_bt1_1__r2b_br1_1`, the double underscore separates the two IDs (each ID itself contains single underscores).

### Component Datasets

Component files are Excel files in `data/single_run/input/components/`.

`distance_matrix.xlsx`

Square matrix of travel distances in km. Row and column labels are facility IDs following the scheme above (`ccp_br1_bt1_1`, `r1_br1_bt1_1`, `r2b_br1_1`, `r2e_br1_1`, `axp_br1_1`, …). The matrix must include every location that an ambulance can depart from or arrive at. Unreachable pairs are written as empty cells (NaN); at run time the C2 dispatcher raises if it ever assigns an ambulance along such a leg.

`coordinates.xlsx`

Optional visualization reference. Used for `network.png` and videos, not for simulation logic. Expected columns:

```text
facility_id, facility_type, x, y
```

`facilities.xlsx`

One row per CCP, Role 1, and Role 2 facility. Expected columns:

```text
facility_id, facility_type, c2_up_id, unload_policy, load_policy
```

`c2_up_id` is blank for Role 2 because Role 2 is the end of the evacuation chain.

`units.xlsx`

One row per care unit. Expected columns include:

```text
unit_id, facility_id, service_type, capacity, service_time_min, service_time_max
```

The current model uses `DCR`, `PEC`, and `DCS` service types.

`axp.xlsx`

One row per Ambulance Exchange Point. Expected columns:

```text
axp_id, assigned_facility_ids
```

`ambulances.xlsx`

One row per ambulance. Expected columns include:

```text
amb_id, start_loc_id, capacity, speed_kmh
```

`c2.xlsx`

One row per dispatcher node. Expected columns:

```text
c2_id, facility_ids, ambulance_ids
```

`facility_ids` are the facilities whose outbound transport requests are controlled by this C2. `ambulance_ids` are the ambulances this C2 may dispatch.

### Policy Matrices

Policy matrix files live under:

```text
data/single_run/input/policy_matrixes/
```

`facility_eligibility/*.xlsx`

Binary matrix. Row index is origin facility, columns are candidate destination facilities. `1` means that route is allowed.

`ambulance_eligibility/*.xlsx`

Binary matrix. Row index is ambulance ID, columns are facilities. `1` means that ambulance can serve that facility.

The active matrix is selected at runtime by scenario levers:

```yaml
levers:
  facility_eligibility: 0
  ambulance_eligibility: 0
```

### Casualty Generation Datasets

Casualty-generation inputs live under:

```text
data/single_run/input/casualty_generation_settings/
```

`sampling_settings/config_<suffix>.yaml`

This is the casualty-generation control file. Current shape:

```yaml
n_days: 1
waves_per_day: 3
wave_duration_min: 30

bcas_rate: 0.05
dnbi_rate: 0

diagnosis_mix: casualty_generation_settings/diagnosis/diagnosis_mix_elia_testing.xlsx
prescriptions: casualty_generation_settings/prescriptions/prescriptions_elia_testing.xlsx
force_structure: casualty_generation_settings/military_force_structure/force_structure_elia_testing.xlsx
```

`n_days`, `waves_per_day`, and `wave_duration_min` control when casualty waves occur. `bcas_rate` and `dnbi_rate` control expected casualties per personnel-at-risk per day.

`force_structure.xlsx`

Current columns:

```text
division, brigade, battalion, par, involved, bcas_spawning_points, dnbi_spawning_points
```

`par` is personnel at risk. Only rows with `involved == True` generate baseline BCAS casualties. `bcas_spawning_points` is the list of CCP IDs where battle casualties can spawn. `dnbi_spawning_points` is the list of Role 1 or other facility IDs where DNBI casualties can spawn when `dnbi_rate > 0`.

When `n_battalions_in_combat` is used in a multi-run experiment, the model randomly selects that many battalions and sums their `par` values into a single total. This total PAR is then used uniformly across all spawning points of the selected battalions to compute expected casualties. Because of this, the casualty calculation is most accurate when all battalions have equal `par` values. If battalions have unequal `par`, the model still applies the same summed total to every spawning point, which can over- or under-represent casualties at individual CCPs.

The older `spawning_points` column is no longer the current schema.

`diagnosis_mix.xlsx`

One row per triage label. Expected columns:

```text
triage, frequency, lifetime_distribution, lifetime_low, lifetime_mode, lifetime_high
```

The set of triage labels is **hardcoded** in the engine and cannot be extended or renamed from this file:

```text
T1_BCAS, T2_BCAS, T3_BCAS, KIA_BCAS, MIA_BCAS, T1_DNBI, T2_DNBI, T3_DNBI
```

Only the per-label `frequency` and lifetime parameters are user-editable — i.e. you can reshape the mix, but not introduce new diagnosis types. KIA/MIA rows may be generated but are filtered out before the simulation run.

`prescriptions.xlsx`

One row per possible care pathway. Expected columns:

```text
label, frequency, prescription
```

Example prescriptions:

```text
['DCR', 'DCS']
['DCR']
['PEC', 'DCS']
['PEC']
```

### Optional Pre-Generated Casualty Dataset

The preferred mode is dynamic casualty generation through `casualty_generation.config`. A scenario can also load a pre-generated casualty spreadsheet:

```yaml
casualty_generation:
  from_dataset: true
  casualties_dataframe: casualty_dataframes/casualties.xlsx
```

That file must include at least:

```text
t_create, spawn_facility_id, triage, prescription,
lifetime_dist_type, lifetime_low, lifetime_high, lifetime_mode
```

## 2. Casualty Dataset Builder

Build or update casualty settings before building the rest of a scenario.

The casualty settings are:

```text
data/single_run/input/casualty_generation_settings/diagnosis/diagnosis_mix_<suffix>.xlsx
data/single_run/input/casualty_generation_settings/prescriptions/prescriptions_<suffix>.xlsx
data/single_run/input/casualty_generation_settings/military_force_structure/force_structure_<suffix>.xlsx
data/single_run/input/casualty_generation_settings/sampling_settings/config_<suffix>.yaml
```

For a new scenario suffix:

1. Copy an existing diagnosis mix file and rename it to `diagnosis_mix_<suffix>.xlsx`.
2. Copy an existing prescriptions file and rename it to `prescriptions_<suffix>.xlsx`.
3. Copy an existing force-structure file and rename it to `force_structure_<suffix>.xlsx`.
4. Edit the force-structure file using the current columns:

```text
division, brigade, battalion, par, involved, bcas_spawning_points, dnbi_spawning_points
```

5. Copy an existing config YAML and rename it to `config_<suffix>.yaml`.
6. Edit the config YAML so it points to the new diagnosis, prescription, and force-structure files.
7. Set `n_days`, `waves_per_day`, `wave_duration_min`, `bcas_rate`, and `dnbi_rate`.

The single-run path can generate casualties dynamically from this config. If you want a fixed pre-generated casualty Excel file instead, use the `from_dataset` scenario mode described above.

## 3. Dataset Builder

The dataset-builder notebook builds the remaining network datasets and the scenario YAML.

Notebooks live under `dataset_generation_notebooks/`:

```text
dataset_generation_notebooks/
├── datasets_builder_elia_testing.ipynb   ← working copy (edit freely)
└── template/
    └── datasets_builder.ipynb            ← canonical template (copy before editing)
```

The `OUT_BASE` path in each notebook is already set relative to its own location, so the notebooks are runnable in place — no `cd` needed. If you copy the template into a different directory, adjust `OUT_BASE` to point back at `data/single_run/input/`.

The notebook is driven by its first configuration cell. The important fields are:

```python
N_CCP = 3
N_R1 = 1
N_AXP = 1
N_R2 = 1

# For dataset_generation_notebooks/datasets_builder_elia_testing.ipynb (one level deep):
OUT_BASE = pathlib.Path("../data/single_run/input")
# For dataset_generation_notebooks/template/datasets_builder.ipynb (two levels deep):
# OUT_BASE = pathlib.Path("../../data/single_run/input")
SUFFIX = "elia_testing"

CASUALTY_CONFIG = f"config_{SUFFIX}"
SCENARIO_NAME = f"scenario_{SUFFIX}"
SEED = 42
SIM_TILL_MIN = 60 * 24 * 3
DISPATCH_POLICY = "greedy"
DATA_COLLECTOR_INTERVAL = 10.0
```

What the notebook does:

1. Builds facility coordinates for CCPs, Role 1s, AXPs, and Role 2s.
2. Computes the Euclidean distance matrix.
3. Visualizes the layout for sanity checking.
4. Builds facility, AXP, ambulance, and C2 tables.
5. Builds care units using the service-time and capacity parameters in the config cell.
6. Builds strict and loose facility eligibility matrices.
7. Builds strict and loose ambulance eligibility matrices.
8. Verifies that `casualty_generation_settings/sampling_settings/{CASUALTY_CONFIG}.yaml` exists.
9. Writes the scenario YAML to `data/single_run/input/scenarios/{SCENARIO_NAME}.yaml`.

Outputs written by the notebook:

```text
components/distances/distance_matrix_<suffix>.xlsx
components/distances/coordinates_<suffix>.xlsx
components/facilities/facilities_<suffix>.xlsx
components/units/units_<suffix>.xlsx
components/AXPs/axp_<suffix>.xlsx
components/ambulances/amb_<suffix>.xlsx
components/c2/c2_<suffix>.xlsx
policy_matrixes/facility_eligibility/facility_eligibility_<suffix>_strict.xlsx
policy_matrixes/facility_eligibility/facility_eligibility_<suffix>_loose.xlsx
policy_matrixes/ambulance_eligibility/ambulance_eligibility_<suffix>_strict.xlsx
policy_matrixes/ambulance_eligibility/ambulance_eligibility_<suffix>_loose.xlsx
scenarios/scenario_<suffix>.yaml
```

Recommended workflow:

1. Build or edit the casualty settings first.
2. Copy `dataset_generation_notebooks/template/datasets_builder.ipynb` into `dataset_generation_notebooks/` and rename it (e.g. `datasets_builder_<suffix>.ipynb`). Update `OUT_BASE` from `../../data/single_run/input` to `../data/single_run/input` to account for the shallower depth.
3. Set `SUFFIX` and `CASUALTY_CONFIG` to match the casualty config.
4. Set facility counts, geometry, ambulance parameters, unit capacities, and simulation duration.
5. Run the notebook top to bottom.
6. Check that the final scenario YAML points to the new files.

## 4. Single Run

Single runs are for inspecting one scenario with detailed logs, plots, and optionally a video.

Entry point:

```text
src/launch/runner_single.py
```

Current path inside the script:

```python
scenario_path="data/single_run/input/scenarios/scenario_elia_testing.yaml"
```

To run a different scenario, edit that path.

### Single-Run Input

A single run needs:

1. Casualty-generation settings:

```text
data/single_run/input/casualty_generation_settings/...
```

2. Component datasets:

```text
data/single_run/input/components/...
```

3. Policy matrices:

```text
data/single_run/input/policy_matrixes/...
```

4. Scenario YAML:

```text
data/single_run/input/scenarios/scenario_<suffix>.yaml
```

### Single-Run Steps

1. Build the casualty settings:

```text
diagnosis_mix_<suffix>.xlsx
prescriptions_<suffix>.xlsx
force_structure_<suffix>.xlsx
config_<suffix>.yaml
```

2. Copy `dataset_generation_notebooks/template/datasets_builder.ipynb` into `dataset_generation_notebooks/` and rename it; update `OUT_BASE` to `../data/single_run/input`.
3. Set `SUFFIX`, `CASUALTY_CONFIG`, and the network parameters.
4. Run the notebook to create the remaining datasets and scenario YAML.
5. Open `src/launch/runner_single.py`.
6. Set `scenario_path` to the new scenario YAML.
7. Run:

```powershell
python -m src.launch.runner_single
```

### Single-Run Output

Outputs are written to:

```text
data/single_run/output/<timestamp>_<scenario_name>/
```

Common files:

```text
patient_log.xlsx
safe_patients.xlsx
transport_events.xlsx
wait_times.xlsx
C2_*.xlsx
network.png
gantt_waves.html
simulation.mp4
```

`simulation.mp4` is only created when `video.enabled: true`.

`patient_log.xlsx` is the most detailed patient trace. `transport_events.xlsx` contains ambulance departures and arrivals. `wait_times.xlsx` contains per-patient wait episodes for care and transport. `safe_patients.xlsx` contains patients who completed all required care. `C2_*.xlsx` files contain dispatch-decision logs.

## 5. Multi Run

Multi runs are EMA Workbench experiments over uncertainty and lever spaces.

Entry point:

```text
src/launch/runner_multiple.py
```

Current path inside the script:

```python
EXPERIMENT_PATH = "data/multi_run/input/experiment_elia_testing.yaml"
```

To run a different experiment, edit that path.

### Multi-Run Experiment YAML

Experiment files live under:

```text
data/multi_run/input/
```

A full experiment template listing every available uncertainty, lever, and outcome is at:

```text
data/multi_run/input/template/experiment_template.yaml
```

To create a new experiment, copy the template, rename it, and remove or comment out the parameters you do not need.

Minimal structure:

```yaml
scenario: data/single_run/input/scenarios/scenario_elia_testing.yaml
output_folder: data/multi_run/output

uncertainties:
- name: seed
  type: integer
  lower: 1
  upper: 10000

levers:
- name: bcas_rate
  type: categorical
  values: [0.005, 0.05]
- name: wave_duration_min
  type: categorical
  values: [10, 20, 40, 50]

outcomes:
- death_rate
- safe_rate
- total_transport_time

array_outcomes:
- ts_transport_r1_1_T1
- ts_transport_r1_1_T2
- ts_transport_r1_1_T3

run_settings:
  n_scenarios: 1000
  n_policies: 1
  uncertainty_sampler: lhs
  lever_sampler: fullfactorial
  batch_size: 50
  max_parallel: 4
```

### Uncertainties, Levers, and Seeds

Uncertainties are sampled into EMA scenarios. Levers are sampled into EMA policies.

If `seed` is listed under `uncertainties`, then `n_scenarios` controls how many different seeds are sampled.

For example:

```yaml
uncertainties:
- name: seed
  type: integer
  lower: 1
  upper: 10000

run_settings:
  n_scenarios: 1
```

This samples one seed. If `lever_sampler: fullfactorial` expands to eight policies, the results will show the same seed on eight rows because those eight policies are all crossed with the same single scenario.

To get many seeds for every lever combination, increase `n_scenarios`.

With this lever space:

```yaml
bcas_rate: [0.005, 0.05]
wave_duration_min: [10, 20, 40, 50]
```

full factorial creates `2 * 4 = 8` policy combinations. With `n_scenarios: 1000`, that gives about `1000 * 8 = 8000` runs.

### Multi-Run Steps

1. Make sure the target single-run scenario works first.
2. Copy an existing experiment YAML in `data/multi_run/input/`.
3. Rename the copy, for example:

```text
experiment_my_scenario.yaml
```

4. Edit the `scenario` path to point to your scenario YAML.
5. Edit `uncertainties`.
6. Edit `levers`.
7. Edit `outcomes` and `array_outcomes`.
8. Edit `run_settings`.
9. Open `src/launch/runner_multiple.py`.
10. Set `EXPERIMENT_PATH` to the new experiment YAML.
11. Run:

```powershell
python -m src.launch.runner_multiple
```

If Windows multiprocessing fails with an access or process-startup error in a restricted shell, set:

```yaml
max_parallel: 1
```

or run from a normal terminal.

### Multi-Run Output

Outputs are written to:

```text
data/multi_run/output/<timestamp>_<scenario_name>/
```

Common files:

```text
results.csv
timeseries.npz
```

`results.csv` contains one row per run. The columns are:

```text
uncertainties, levers, EMA metadata, scalar outcomes
```

Typical columns:

```text
seed, bcas_rate, wave_duration_min, scenario, policy, model,
death_rate, safe_rate, total_transport_time
```

`timeseries.npz` contains array outcomes. Each key is an array outcome name such as:

```text
ts_transport_r1_1_T1
ts_dcs_r2_1_T2
```

Each array has shape:

```text
n_runs x n_timesteps
```

where `n_timesteps` depends on `simulation.till` and `data_collector_interval`.
