# MEDEVAC Simulation Model

A discrete-event simulation (DES) model of military medical evacuation (MEDEVAC) operations, built on [salabim](https://www.salabim.org/). The model simulates the full casualty chain — from injury to definitive care — through a network of medical facilities, treatment units, and ambulances operating under configurable policy rules.

Supports both **single-run** analysis (with video output and detailed logs) and **multi-run** policy exploration via [EMA Workbench](https://ema-workbench.readthedocs.io/).

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Single Run](#single-run)
5. [Multi Run (EMA Workbench)](#multi-run-ema-workbench)
6. [Model Architecture](#model-architecture)
7. [Components](#components)
8. [Policy System](#policy-system)
9. [Casualty Generation](#casualty-generation)
10. [Observability](#observability)
11. [Output](#output)

---

## Project Structure

```
medevac_model/
│
├── src/
│   ├── components/             # Simulation entities (salabim Components)
│   │   ├── ambulance.py        # Ambulance movement and transport logic
│   │   ├── axp.py              # Ambulance Exchange Point
│   │   ├── builder.py          # ModelBuilder / ModelConnector
│   │   ├── c2.py               # Command & Control dispatcher
│   │   ├── facility.py         # Medical facility (queues, load/unload)
│   │   ├── patient.py          # Patient with death clock
│   │   ├── requests.py         # TransportRequest and PlanStep data classes
│   │   ├── spawner.py          # CasualtySpawner
│   │   └── unit.py             # Treatment unit
│   │
│   ├── helpers/                # Pure functions — no salabim state
│   │   ├── runner.py           # run_simulation() — the core simulation function
│   │   ├── helpers_run.py      # Single-run helpers: prepare, save, video, print
│   │   ├── helpers_multi.py    # Multi-run helpers: EMA model, outcomes, saving
│   │   ├── observer.py         # SimulationObserver context manager
│   │   ├── casualty_generator.py  # Wave-based casualty generation
│   │   ├── scenario.py         # Scenario YAML loading and data parsing
│   │   └── visualization.py    # Network plot, video generation, state logger
│   │
│   ├── launch/                 # Entry points — minimal, no business logic
│   │   ├── runner_single.py    # Run a single scenario
│   │   └── runner_multiple.py  # Run an EMA Workbench experiment
│   │
│   ├── policies/
│   │   └── policies.py         # All policy functions and registries
│   │
│   └── __init__.py             # Package-level exports
│
├── data/
│   ├── single_run/
│   │   ├── input/
│   │   │   ├── components/         # facilities, units, ambulances, AXPs, c2, distances
│   │   │   ├── policy_matrixes/    # corridors matrix, ambulance eligibility matrix
│   │   │   ├── casualty_generation_settings/  # diagnosis mix, prescriptions, config
│   │   │   ├── casualty_dataframes/           # pre-generated casualty datasets
│   │   │   └── scenarios/          # scenario YAML files
│   │   └── output/                 # timestamped run folders (auto-generated)
│   │
│   └── multi_run/
│       ├── input_settings/         # experiment YAML files
│       └── outputs/                # timestamped experiment folders (auto-generated)
│
└── tests/
    ├── test_casualty_generator.py
    └── test_kpis.py
```

---

## Installation

```bash
pip install salabim ema_workbench pandas openpyxl matplotlib numpy pyyaml
```

Python 3.10+ recommended.

---

## Quick Start

**Single run:**
```bash
python src/launch/runner_single.py
```

**Multi run (EMA Workbench):**
```bash
python src/launch/runner_multiple.py
```

---

## Single Run

### How it works

`runner_single.py` executes in four steps:

```python
data, settings, run_folder = prepare_run(scenario_path="data/single_run/input/scenarios/scenario_2.yaml")
results = run_simulation(data, settings, run_folder)
if settings["save_logs"]: save_logs(results)
if settings["make_video"]: make_video(data, results, settings)
print_summary(results)
```

1. **`prepare_run`** — loads the scenario YAML, resolves settings, creates the output folder, and loads/generates casualties
2. **`run_simulation`** — builds the salabim environment, runs the simulation, returns results
3. **`save_logs`** — writes Excel files to the run folder
4. **`make_video`** — renders an MP4 animation of transport events
5. **`print_summary`** — prints statistics to the console

### Scenarios

Scenarios live in `data/single_run/input/scenarios/`. Each scenario YAML defines:

```yaml
name: "scenario_2"

components:           # paths to Excel files (relative to input/)
  distance_matrix: "components/distances/distance_matrix.xlsx"
  axp:              "components/AXPs/axp.xlsx"
  facilities:       "components/facilities/facilities.xlsx"
  units:            "components/units/units.xlsx"
  ambulances:       "components/ambulances/ambulances.xlsx"
  c2:               "components/c2/c2.xlsx"

policy_matrixes:
  corridors:               "policy_matrixes/corridors/facility_eligibility_1.xlsx"
  ambulances_eligibility:  "policy_matrixes/ambulances_eligibility/ambulance_eligibility_1.xlsx"

casualty_generation:
  from_dataset: true                              # load pre-generated casualties
  casualties_dataframe: "casualty_dataframes/one_casualty_1.xlsx"
  # --- OR: generate dynamically ---
  # from_dataset: false
  # config:        "casualty_generation_settings/sampling_settings/config.yaml"
  # diagnosis_mix: "casualty_generation_settings/diagnosis/diagnosis_mix.xlsx"
  # prescriptions: "casualty_generation_settings/prescriptions/prescriptions_meisner.xlsx"

levers:                      # policy indices (see Policy System)
  facility_eligibility: 0
  facility_assignment:  0
  ambulance_eligibility: 0
  ambulance_assignment: 0

simulation:
  till: 500          # simulation end time (minutes)
  trace: false

video:
  enabled: true
  fps: 24
  time_scale: 5.0
  dpi: 100
  state_log_interval: 5.0

logs:
  enabled: true
```

### Output folder

Each run creates a timestamped folder under `data/single_run/output/`:

```
20260311_170758_scenario_2/
├── scenario_2.yaml         # copy of the scenario used
├── patient_log.xlsx        # full event log per patient
├── transport_events.xlsx   # all ambulance departure/arrival events
├── facility_states.xlsx    # periodic snapshots of facility state
├── safe_patients.xlsx      # patients who reached definitive care
└── simulation.mp4          # animation (if video enabled)
```

---

## Multi Run (EMA Workbench)

### How it works

`runner_multiple.py` runs an EMA Workbench experiment over a configurable space of uncertainties and levers:

```python
experiment = load_experiment("data/multi_run/input_settings/experiment.yaml")
model = build_ema_model(experiment)         # pre-loads structural data once
experiments_df, outcomes = run_ema(model, experiment)
save_experiment_results(experiments_df, outcomes, experiment)
```

**Key design decision:** structural data (facilities, ambulances, distances, policies) is loaded **once** before the experiment loop. Each individual run only re-generates casualties and re-runs the simulation engine. This avoids redundant I/O on every iteration.

### Experiment YAML

Experiments live in `data/multi_run/input_settings/`. Each experiment YAML is fully self-contained — it references its own component and casualty config files directly, independently of any single-run scenario.

```yaml
name: "medevac_experiment_1"

data_root: "data/single_run/input"   # base path for all component paths below

components:
  distance_matrix: "components/distances/distance_matrix.xlsx"
  axp:              "components/AXPs/axp.xlsx"
  facilities:       "components/facilities/facilities.xlsx"
  units:            "components/units/units.xlsx"
  ambulances:       "components/ambulances/ambulances.xlsx"
  c2:               "components/c2/c2.xlsx"

policy_matrixes:
  corridors:              "policy_matrixes/corridors/facility_eligibility_1.xlsx"
  ambulances_eligibility: "policy_matrixes/ambulances_eligibility/ambulance_eligibility_1.xlsx"

casualty_config:
  diagnosis_mix:  "casualty_generation_settings/diagnosis/diagnosis_mix.xlsx"
  prescriptions:  "casualty_generation_settings/prescriptions/prescriptions_meisner.xlsx"
  facility_ids:
    - "ccp1"
    - "ccp2"

simulation:
  till: 500.0

uncertainties:
  - { name: casualty_seed,     type: integer, lower: 1,     upper: 10000 }
  - { name: par_total,         type: integer, lower: 5000,  upper: 30000 }
  - { name: rate_per_day,      type: real,    lower: 0.005, upper: 0.030 }
  - { name: waves_per_day,     type: integer, lower: 1,     upper: 6     }
  - { name: wave_duration_min, type: integer, lower: 15,    upper: 120   }
  - { name: n_days,            type: integer, lower: 1,     upper: 7     }

levers:
  - { name: facility_eligibility,  type: integer, lower: 0, upper: 0 }
  - { name: facility_assignment,   type: integer, lower: 0, upper: 0 }
  - { name: ambulance_eligibility, type: integer, lower: 0, upper: 0 }
  - { name: ambulance_assignment,  type: integer, lower: 0, upper: 0 }

outcomes:
  - name: death_rate
  - name: death_rate_T1
  - name: death_rate_T2
  - name: death_rate_T3
  - name: safe_rate
  - name: mean_wait_care_T1
  - name: mean_wait_care_T2
  - name: mean_wait_care_T3
  - name: mean_wait_transport_T1
  - name: mean_wait_transport_T2
  - name: mean_wait_transport_T3
  - name: total_transport_time

run_settings:
  n_scenarios: 100
  n_policies:  1
  evaluator:   sequential
```

**Uncertainties** are parameters sampled by EMA (things you don't control — how many casualties, when they arrive, how severe). **Levers** are policy choices (which policy function index to use for each decision point). **Outcomes** are the scalar KPIs returned by each run.

### Output folder

Each experiment creates a timestamped folder under `data/multi_run/outputs/`:

```
20260313_154802_medevac_experiment_1/
└── results.csv     # experiments + outcomes merged into a single flat table
                    # one row per run, columns = uncertainties + levers + KPIs
```

---

## Model Architecture

The simulation models the military MEDEVAC chain as a network of nodes and transport links:

```
[Injury site]
     │  spawned at CCP (Combat Collection Point)
     ▼
[CCP / Role 1]  ──► treatment units (e.g. AIDSTA)
     │  transport request → C2
     ▼
[Role 2]        ──► treatment units (e.g. FST, ICU)
     │  transport request → C2
     ▼
[Role 3]        ──► treatment units (e.g. SUR, ICU)
     │
     ▼
  [SAFE]
```

Each patient has a **death clock**: a survival time drawn from a triage-specific distribution. The clock runs while the patient is waiting (for care or transport) and is **paused** when the patient is actively being treated or transported. If the clock expires during a wait, the patient dies.

### Component interactions

```
CasualtySpawner
    │ creates Patient at spawn facility
    ▼
Facility
    │ puts patient in entry_line queue
    ▼
Unit (treatment)
    │ serves patient, then patient waits for transport
    ├─► Facility generates TransportRequest
    │       │
    │       ▼
    │   TransportC2 (C2)
    │       │ applies policy chain (eligibility → assignment → plan → dispatch)
    │       ▼
    │   Ambulance
    │       │ travels to pickup facility, loads patient, travels to destination
    │       ▼
    │   next Facility (or safe zone)
    │
    └─► [Patient marked SAFE when reaching final destination]
```

---

## Components

### `Ambulance`
Moves between facilities and AXPs following a **plan** — an ordered list of `PlanStep` objects, each specifying a location and which patients to load or unload there. When idle, an ambulance waits at its assigned AXP.

### `AXP` (Ambulance Exchange Point)
A waypoint where ambulances idle between missions. Each AXP is associated with a set of downstream facilities it serves.

### `TransportC2`
The dispatcher. When a facility has patients ready for transport, it raises a `TransportRequest`. The C2 collects pending requests, applies a configurable four-step policy chain (see [Policy System](#policy-system)), crafts ambulance plans, and dispatches ambulances.

### `Facility`
A medical node in the evacuation chain. Each facility has:
- **Entry lines** — queues of patients waiting for specific treatment units
- **Exit queue** — patients waiting for transport to the next echelon
- **Units** — one or more treatment units (e.g. AIDSTA, FST, SUR, ICU)
- A reference to its upstream C2

Facility types: `CCP`, `Role1`, `Role2`, `Role3`.

### `Unit`
A treatment unit inside a facility with a fixed capacity. Patients are served in **triage-priority order** (T1 first). Service time is sampled from a configurable distribution. When service ends, the unit calls a completion callback that transitions the patient to the transport queue.

### `Patient`
A casualty with:
- **Triage level**: T1 (immediate), T2 (delayed), T3 (minimal)
- **Prescription**: an ordered list of unit types the patient must visit (e.g. `["AIDSTA", "FST", "ICU"]`)
- **Death clock**: counts down during waiting; paused during treatment and transport
- **State**: `waiting_care` → `in_care` → `waiting_transport` → `in_transport` → `safe` / `dead`

### `CasualtySpawner`
Reads a pre-built `casualties_df` (one row per patient with scheduled creation time and spawn facility) and injects patients into the simulation at the correct simulation times.

---

## Policy System

Policies govern how C2 makes decisions. Each decision point has a **registry** mapping integer keys to policy functions. The lever values in a scenario or experiment select which function is used at runtime.

| Lever | Decision | Registry |
|---|---|---|
| `facility_eligibility` | Which destination facilities are reachable? | `FACILITY_ELIGIBILITY_POLICIES` |
| `facility_assignment` | Which facility to assign to this patient? | `FACILITY_ASSIGNMENT_POLICIES` |
| `ambulance_eligibility` | Which ambulances are available? | `AMBULANCE_ELIGIBILITY_POLICIES` |
| `ambulance_assignment` | Which ambulance to assign? | `AMBULANCE_ASSIGNMENT_POLICIES` |

**Currently implemented (index 0 = base):**

- **Facility eligibility 0** — consults the corridors matrix to find reachable facilities of the required type
- **Facility assignment 0** — random selection from eligible facilities
- **Ambulance eligibility 0** — idle ambulances whose service coverage matches the required facility type
- **Ambulance assignment 0** — random selection from eligible ambulances

### Adding a new policy

1. Write a new function in `src/policies/policies.py` matching the signature of existing policies
2. Add it to the relevant registry dict with the next integer key
3. Set the corresponding lever to that key in the scenario YAML or experiment YAML

---

## Casualty Generation

Casualties are generated using a **wave model** that mimics realistic battle rhythms.

### Parameters

| Parameter | Description |
|---|---|
| `par_total` | Total Personnel At Risk |
| `rate_per_day` | Casualties per soldier per day |
| `n_days` | Simulation duration in days |
| `waves_per_day` | Number of engagement waves per day |
| `wave_duration_min` | Duration of each wave (minutes) |
| `seed` | Random seed for reproducibility |
| `facility_ids` | Locations where casualties spawn |

### Generation steps

1. **Place waves** — non-overlapping time windows are placed randomly across the simulation horizon, `waves_per_day` per day
2. **Allocate casualties** — the expected total (`par_total × rate_per_day × n_days`) is distributed across facilities using a Poisson draw; each facility gets casualties proportional to its weight
3. **Fill waves** — each facility's casualties are distributed randomly across its waves; within each wave, creation times are drawn uniformly
4. **Assign attributes** — triage level and prescription are sampled from the diagnosis mix; survival time is sampled from triage-specific distributions (Uniform, Triangular, Exponential, or Normal)

### Two sources

| Mode | When to use |
|---|---|
| `from_dataset: true` | Reproducible single runs with a fixed casualty file |
| `from_dataset: false` | Dynamic generation with configurable parameters |

In multi-run mode, casualties are **always generated dynamically** — the EMA sampler controls `par_total`, `rate_per_day`, etc. as uncertainties.

---

## Observability

The `SimulationObserver` context manager collects runtime data without modifying any component source code. It monkey-patches five methods before the simulation starts and restores them on exit:

| Patched method | Data collected |
|---|---|
| `Ambulance._travel_to` | Departure and arrival events with timestamps, locations, onboard patients |
| `Facility.patient_handed_over` | Transport wait start time per patient |
| `Facility.process` | Transport wait duration (time from handover to ambulance loading) |
| `Unit.process` | Care wait duration (time in queue before treatment starts) |
| `Unit.service_done` | Unit busy periods (start/end of each service slot) |
| `Patient.setup` | Registration of all patients in `env.all_patients` |

Usage:
```python
with SimulationObserver(env) as observer:
    # build and run simulation
    env.run(till=till)

data = observer.results()
# data keys: transport_events, wait_times, unit_busy_periods,
#            all_patients, facility_states, patient_log_rows
```

---

## Output

### Single run logs

| File | Contents |
|---|---|
| `patient_log.xlsx` | Full event log: one row per patient action (spawned, entered queue, started treatment, etc.) |
| `transport_events.xlsx` | Ambulance departure and arrival events |
| `facility_states.xlsx` | Periodic snapshots: patient counts by triage and status at each facility |
| `safe_patients.xlsx` | Patients who completed evacuation: ID, triage, time safe, spawn location |
| `simulation.mp4` | Animated video showing ambulance movements and facility status over time |

### Console summary (single run)

```
SIMULATION RESULTS
======================================================================
--- PATIENT OUTCOMES ---
  T1:  23 patients
    Deaths:        8  ( 34.8%)
    Safe:         12  ( 52.2%)
    In progress:   3  ( 13.0%)
  ...

--- WAITING TIME STATISTICS (minutes) ---
  Waiting for CARE:
    At CCP:
      T1:    0.0 /    0.0 /    2.1  (n=12)
  ...

--- CUMULATIVE TRANSPORT TIME ---
  Total transport time (all ambulances): 214.3 minutes
    amb_1: 87.0 minutes
    amb_2: 63.1 minutes

--- UNIT UTILIZATION ---
  ccp1_AIDSTA: 18.4% (92.0 min busy / 500.0 min total)
  ...
```

### Multi-run results

`results.csv` — one row per experiment run, columns are all uncertainties, all levers, and all scalar outcomes. Ready for direct import into EMA Workbench analysis notebooks.

```
casualty_seed | par_total | rate_per_day | ... | death_rate | safe_rate | mean_wait_care_T1 | ...
         4821 |     12340 |        0.014 | ... |       0.23 |      0.61 |               4.2 | ...
```
