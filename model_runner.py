from __future__ import annotations

import uuid
import math
import random
import pandas as pd
import salabim as sim
from typing import Callable, Dict, List, Optional


# ============================================================
# POLICY PIPELINE (lever functions + reservation-aware planner)
# ============================================================


# ============================================================
# MODEL OBJECTS
# ============================================================

class TransportRequest:
    def __init__(
            self,
            req_id: str,
            patient: Patient,
            issuing_facility: Facility,
            t_issued: float,
            injury_time: float,
            triage: str,
            needed_unit: str | None,
    ):
        self.req_id = req_id
        self.patient = patient
        self.casualty_id = patient.patient_id
        self.triage = triage
        self.issuing_facility = issuing_facility
        self.injury_time = float(injury_time)
        self.t_issued = float(t_issued)
        self.needed_unit = needed_unit
        self.status = "pending"


class PlanStep:
    def __init__(self, facility, unload_patients, load_patients):
        self.facility = facility
        self.unload_patients = list(unload_patients)
        self.load_patients = list(load_patients)


def unload_planned(amb, facility):
    step = amb.current_step
    return step.unload_patients if (step and step.facility is facility) else []


def load_planned(amb, facility):
    step = amb.current_step
    return step.load_patients if (step and step.facility is facility) else []


class Patient(sim.Component):
    def setup(self, patient_id: str, triage: str, prescription: list, life_time_dist, spawn_facility, injury_time):
        self.patient_id = patient_id
        self.triage = triage
        self.prescription = list(prescription)
        self.life_time_dist = life_time_dist
        self.spawn_facility = spawn_facility
        self.current_facility = spawn_facility
        self.rx_index = 0
        self.unit_needed = self.prescription[0] if self.prescription else None
        self.is_safe = False
        self.is_dead = False
        self.t_safe = None
        self.patient_state = "in_care"
        self.injury_time = float(injury_time)

        self._last_life_budget = None

        self._log(action="SPAWNED", life_budget=None)

    def _loc_id(self):
        loc = getattr(self, "current_facility", None)
        return getattr(loc, "loc_id", None) or str(loc)

    def _log(self, action: str, life_budget):
        if not hasattr(self.env, "patient_log_rows"):
            self.env.patient_log_rows = []

        self.env.patient_log_rows.append({
            "time": float(self.env.now()),
            "patient_id": self.patient_id,
            "action": str(action),
            "life_budget": None if life_budget is None else float(life_budget),
            "needed_unit": self.unit_needed,
            "current_location": self._loc_id(),
            "is_dead": bool(self.is_dead),
            "is_safe": bool(self.is_safe),
        })

    def process(self):
        while True:
            budget = float(self.life_time_dist.sample())
            self._last_life_budget = budget
            self._log(action="LIFE_BUDGET_SAMPLED", life_budget=budget)

            t0 = self.env.now()
            self.hold(budget)

            # if this hold fully elapsed, they died
            if (self.env.now() - t0) >= budget - 1e-9:
                self.is_dead = True
                self.patient_state = "dead"
                self._log(action="DIED", life_budget=self._last_life_budget)
                self.cancel()
                return

    def care_completed(self, unit):
        # If patient died during service, do nothing
        if self.is_dead:
            return

        # unit is a Unit instance; name() is the unit_id
        unit_id = unit.name() if hasattr(unit, "name") else str(unit)

        self._log(action=f"CARE_COMPLETED:{unit_id}", life_budget=self._last_life_budget)

        self.rx_index += 1
        if self.rx_index >= len(self.prescription):
            self.is_safe = True
            self.unit_needed = None
            self._log(action="BECAME_SAFE", life_budget=self._last_life_budget)
            self.activate(process="out_of_danger")
            return

        self.unit_needed = self.prescription[self.rx_index]
        self._log(action="NEXT_UNIT_SET", life_budget=self._last_life_budget)
        self.activate()

    def out_of_danger(self):
        self.is_safe = True
        self.t_safe = self.env.now()
        self.patient_state = "safe"
        self.env.safe.append(self)

        self._log(action="LOGGED_SAFE_AND_EXIT", life_budget=self._last_life_budget)
        self.cancel()


class Unit(sim.Component):
    def setup(self, facility, capacity: int, service_time_dist, service_type: str):
        self.facility = facility
        self.capacity = int(capacity)
        self.service_time_dist = service_time_dist
        self.service_type = service_type
        self.in_service = 0

        # realized queue waiting times (when service starts)
        self.wait_monitor = sim.Monitor(name=f"{self.name()}_wait_realized")

        # expected waiting times at queue-join moments (your request)
        self.ewait_monitor = sim.Monitor(name=f"{self.name()}_wait_expected_at_join")

    def _uniform_ab(self, d):
        # Try common salabim Uniform parameter names
        for a_name, b_name in [("_lowerbound", "_upperbound"), ("a", "b"), ("low", "high"), ("min", "max"), ("lower", "upper")]:
            if hasattr(d, a_name) and hasattr(d, b_name):
                a = float(getattr(d, a_name))
                b = float(getattr(d, b_name))
                return a, b
        raise AttributeError("Could not read (a,b) from Uniform distribution; add attribute mapping here.")

    def expected_wait_if_join_now(self) -> float:
        q = self.facility.entry_lines[self.service_type]
        n = len(q)  # people ahead
        M = self.capacity

        if M == 0:
            return float("inf")

        # Uniform(a,b)
        a, b = self._uniform_ab(self.service_time_dist)

        ES = (a + b) / 2.0

        return (n * ES) / M

    def process(self):
        while True:
            q = self.facility.entry_lines[self.service_type]

            while self.in_service < self.capacity and len(q) > 0:
                p = q.pop(0)

                # realized waiting time
                t_enter = getattr(p, "_unit_queue_entry_time", None)
                if t_enter is not None:
                    self.wait_monitor.tally(self.env.now() - t_enter)

                self.in_service += 1
                UnitService(name=f"{self.name()}_svc", unit=self, patient=p)

            self.passivate()

    def service_done(self, p):
        self.in_service -= 1
        p.care_completed(self)

        if not p.is_safe and not p.is_dead:
            self.facility.patient_handed_over(p)

        self.activate()


class UnitService(sim.Component):
    def setup(self, unit: Unit, patient):
        self.unit = unit
        self.patient = patient

    def process(self):
        self.hold(self.unit.service_time_dist.sample())
        self.unit.service_done(self.patient)
        self.cancel()


class Facility(sim.Component):
    def setup(self, facility_id: str, facility_type: str, c2_up_id: str | None, unload_key: str, load_key: str,
              service_types: list[str]):
        self.loc_id = facility_id
        self.facility_type = facility_type
        self.c2_up_id = c2_up_id
        self.c2_up = None
        self.service_types = list(service_types)
        self.unload_key = unload_key
        self.load_key = load_key
        self.units = {st: None for st in self.service_types}
        self.entry_lines = {st: [] for st in self.service_types}
        self.exit_queue: list[TransportRequest] = []
        self._amb = None

    def ambulance_arrives(self, amb):
        self._amb = amb
        self.activate()

    def patient_handed_over(self, p: Patient):
        p.current_facility = self
        if hasattr(p, "_log_action"):
            p._log_action(life_budget=getattr(p, "_last_life_budget", None))

        needed = p.unit_needed

        if needed in self.entry_lines:
            # stamp join time (for realized wait)
            p._unit_queue_entry_time = self.env.now()
            unit = self.units[needed]
            p._expected_wait_at_join = unit.expected_wait_if_join_now()
            unit.ewait_monitor.tally(p._expected_wait_at_join)

            self.entry_lines[needed].append(p)
            unit.activate()
            return

        req = TransportRequest(
            req_id=uuid.uuid4().hex,
            patient=p,
            issuing_facility=self,
            t_issued=self.env.now(),
            injury_time=p.injury_time,
            triage=p.triage,
            needed_unit=p.unit_needed,
        )

        p.patient_state = "waiting_transport"
        self.exit_queue.append(req)
        self.c2_up.receive_request(req)

    def process(self):
        while True:
            self.passivate()
            amb = self._amb
            self._amb = None

            unload_fn = self.env.unload_policy[self.unload_key]
            for p in unload_fn(amb, self):
                amb.onboard.remove(p)
                self.patient_handed_over(p)

            load_fn = self.env.load_policy[self.load_key]
            for p in load_fn(amb, self):
                # remove the assigned request from exit_queue
                for req in list(self.exit_queue):
                    if req.patient is p and req.status == "assigned":
                        self.exit_queue.remove(req)
                        break
                amb.onboard.append(p)
                p.patient_state = "in_transport"

            amb.activate()


class AXP(sim.Component):
    def setup(self, axp_id: str, assigned_facility_ids: list[str]):
        self.loc_id = axp_id
        self.assigned_facility_ids = list(assigned_facility_ids)
        self.ambulances_here = []

    def arrive(self, amb):
        if amb not in self.ambulances_here:
            self.ambulances_here.append(amb)

    def depart(self, amb):
        if amb in self.ambulances_here:
            self.ambulances_here.remove(amb)


class Ambulance(sim.Component):
    def setup(
            self,
            amb_id: str,
            capacity: int,
            speed_kmh: float,
            start_loc_id: str,
            c2_id: str,
    ):
        self.amb_id = amb_id
        self.capacity = int(capacity)
        self.speed_kmh = float(speed_kmh)
        self.start_loc_id = start_loc_id
        self.c2_id = c2_id
        self.c2 = None
        self.location = None  # AXP or Facility object, set by connector
        self.onboard: list[Patient] = []
        self.plan: list[PlanStep] | None = None
        self.current_step: PlanStep | None = None

    def _loc_id(self, x):
        return getattr(x, "loc_id", x)

    def _announce_idle_here(self):
        """
        Optional bookkeeping: if you're at an AXP, register in its list.
        If you're at a Facility, you might not need any list, just tell C2 you're idle.
        """
        if isinstance(self.location, AXP):
            self.location.arrive(self)
        # If at Facility, you can skip facility bookkeeping (unless you want a 'parking lot' list).
        self.c2.ambulance_idle(self)

    def _leave_current_spot(self):
        if isinstance(self.location, AXP):
            self.location.depart(self)

    def process(self):
        while True:
            # 1) I'm idle wherever I am right now
            self._announce_idle_here()
            self.passivate()  # C2 wakes me when it assigns plan

            # 2) I've been dispatched
            self._leave_current_spot()

            # 3) Execute plan to completion
            for step in self.plan or []:
                self.current_step = step

                self._travel_to(step.facility)

                self.location = step.facility
                step.facility.ambulance_arrives(self)
                self.passivate()  # facility wakes me after load/unload

            # 4) Plan complete: stay where you are (last location)
            self.current_step = None
            self.plan = None
            # loop back: will announce idle at this same location next iteration

    def _travel_to(self, destination):
        from_id = self._loc_id(self.location)
        to_id = self._loc_id(destination)
        if from_id == to_id:
            return
        dist_km = self.env.distance_matrix.loc[from_id, to_id]
        minutes = (dist_km / self.speed_kmh) * 60.0
        self.hold(minutes)


# ============================================================
# POLICY REGISTRIES (ADD NEW LEVERS HERE, NO CHANGES IN C2 CODE)
# ============================================================

# ---- type aliases for policy functions ----
FacilityEligibilityFn = Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]
FacilityAssignmentFn = Callable[[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]], pd.DataFrame]
AmbEligibilityFn = Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]
AmbAssignmentFn = Callable[[pd.DataFrame, List["Ambulance"], Optional[pd.DataFrame]], pd.DataFrame]


# ---- (templates) policy functions you will implement ----
def fac_eligibility_base(requests_df: pd.DataFrame, corridors_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Add column 'eligible_facilities' (list[str]) to requests_df.

    corridors_matrix: DataFrame with facilities as both index and columns.
    Value 1 if row facility can send to column facility, 0 otherwise.
    """
    df = requests_df.copy()
    df["eligible_facilities"] = [[] for _ in range(len(df))]

    if df.empty or corridors_matrix is None or corridors_matrix.empty:
        return df

    # For each request, find eligible destinations based on issuing facility
    def get_eligible(issuing_fac_id):
        if issuing_fac_id not in corridors_matrix.index:
            return []
        row = corridors_matrix.loc[issuing_fac_id]
        return [col for col in row.index if row[col] == 1]

    df["eligible_facilities"] = df["issuing_facility_id"].apply(get_eligible)
    return df


def fac_assign_template(
        eligible_facilities_df: pd.DataFrame,
        facilities_df: pd.DataFrame,
        distance_matrix: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """
    Choose randomly one facility among eligible_facilities.

    Adds column 'assigned_facility_id' and returns the enhanced df.
    """
    df = eligible_facilities_df.copy()

    if "eligible_facilities" not in df.columns:
        df["eligible_facilities"] = [[] for _ in range(len(df))]

    def pick_one(xs):
        return random.choice(xs) if isinstance(xs, list) and len(xs) > 0 else None

    df["assigned_facility_id"] = df["eligible_facilities"].apply(pick_one)
    return df


def amb_eligibility_base(
        assigned_facilities_df: pd.DataFrame,
        ambulances_eligibility: pd.DataFrame,
        idle_ambulances: List["Ambulance"],
) -> pd.DataFrame:
    """
    Add column 'eligible_ambulances' (list[str]) to assigned_facilities_df.

    ambulances_eligibility: DataFrame with ambulances as index, facilities as columns.
    Value 1 if ambulance serves that facility, 0 otherwise.

    Eligible ambulance = idle now AND serves both issuing_facility_id and assigned_facility_id.
    """
    df = assigned_facilities_df.copy()
    df["eligible_ambulances"] = [[] for _ in range(len(df))]

    if df.empty:
        return df

    required_req = {"issuing_facility_id", "assigned_facility_id"}
    if not required_req.issubset(df.columns):
        return df

    idle_ids = [a.amb_id for a in (idle_ambulances or [])]

    # If no constraint table, every idle ambulance is eligible
    if ambulances_eligibility is None or ambulances_eligibility.empty:
        df["eligible_ambulances"] = [idle_ids[:] for _ in range(len(df))]
        return df

    ae = ambulances_eligibility

    def get_eligible_ambs(row):
        issuing = row["issuing_facility_id"]
        assigned = row["assigned_facility_id"]
        eligible = []
        for amb_id in idle_ids:
            if amb_id not in ae.index:
                continue
            # Ambulance must serve both pickup and dropoff facilities
            serves_issuing = issuing in ae.columns and ae.loc[amb_id, issuing] == 1
            serves_assigned = assigned in ae.columns and ae.loc[amb_id, assigned] == 1
            if serves_issuing and serves_assigned:
                eligible.append(amb_id)
        return eligible

    df["eligible_ambulances"] = df.apply(get_eligible_ambs, axis=1)
    return df


def amb_assign_template(
        eligible_pairs_df: pd.DataFrame,
        distance_matrix: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """
    Choose randomly one ambulance among eligible_ambulances.

    Adds column 'assigned_amb_id' and returns the enhanced df.
    """
    df = eligible_pairs_df.copy()

    if "eligible_ambulances" not in df.columns:
        df["eligible_ambulances"] = [[] for _ in range(len(df))]

    def pick_one(xs):
        return random.choice(xs) if isinstance(xs, list) and len(xs) > 0 else None

    df["assigned_amb_id"] = df["eligible_ambulances"].apply(pick_one)
    return df


def craft_plans(
        c2: "TransportC2",
        assigned_ambulances_df: pd.DataFrame,
) -> dict[str, list["PlanStep"]]:
    """
    PURE planner: returns plans, does NOT dispatch/activate.

    Input must contain:
      - req_id
      - issuing_facility_id
      - assigned_facility_id
      - assigned_amb_id

    Output:
      { amb_id: [PlanStep(pickup_facility), PlanStep(dropoff_facility)] }

    Logic (your spec made concrete):
      1) group by assigned_amb_id
      2) for each ambulance:
         - choose the issuing_facility_id with most assigned rows ("most human cargo")
         - from that pickup-facility group, take up to ambulance.capacity requests (random subset)
         - choose dropoff as the most common assigned_facility_id among those selected
         - build a 2-step plan: pickup(load those patients) -> dropoff(unload those patients)

    Notes:
      - This function assumes patients are still present in pickup_facility.exit_queue as TransportRequest objects.
      - It does not modify request statuses; you can do that in C2 after you accept the plan.
    """
    plans: dict[str, list["PlanStep"]] = {}

    if assigned_ambulances_df is None or assigned_ambulances_df.empty:
        return plans

    df = assigned_ambulances_df.copy()

    required = {"req_id", "issuing_facility_id", "assigned_facility_id", "assigned_amb_id"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"send_plans missing columns: {missing}")

    env = c2.env

    # helper: for a facility, map req_id -> TransportRequest
    def req_map_for_facility(fac):
        return {getattr(r, "req_id", None): r for r in getattr(fac, "exit_queue", [])}

    for amb_id, g in df.groupby("assigned_amb_id", sort=False):
        if amb_id is None or (isinstance(amb_id, float) and pd.isna(amb_id)):
            continue
        amb_id = str(amb_id)

        amb = env.amb_by_id.get(amb_id)
        if amb is None:
            continue

        # pickup facility = the facility with most cargo for this ambulance
        pickup_facility_id = g["issuing_facility_id"].value_counts().index[0]
        gg = g[g["issuing_facility_id"] == pickup_facility_id].copy()
        if gg.empty:
            continue

        # choose up to capacity reqs from this pickup facility
        gg = gg.sample(frac=1).reset_index(drop=True)
        gg = gg.iloc[: min(int(amb.capacity), len(gg))]

        # dropoff facility = most common among the selected ones
        dropoff_facility_id = gg["assigned_facility_id"].value_counts().index[0]

        pickup_fac = env.facility_by_id.get(pickup_facility_id)
        dropoff_fac = env.facility_by_id.get(dropoff_facility_id)
        if pickup_fac is None or dropoff_fac is None:
            continue

        # resolve Patient objects from pickup_fac.exit_queue using req_ids
        req_ids = gg["req_id"].astype(str).tolist()
        rmap = req_map_for_facility(pickup_fac)
        req_objs = [rmap.get(rid) for rid in req_ids]
        req_objs = [r for r in req_objs if r is not None and getattr(r, "patient", None) is not None]
        patients = [r.patient for r in req_objs]

        if not patients:
            continue

        pickup_step = PlanStep(facility=pickup_fac, unload_patients=[], load_patients=patients)
        dropoff_step = PlanStep(facility=dropoff_fac, unload_patients=patients, load_patients=[])

        plans[amb_id] = [pickup_step, dropoff_step]

    return plans


def dispatch_ambulances(c2: "TransportC2", plans_craft: dict[str, list["PlanStep"]]):
    """
    Equip ambulances with a plan and dispatch them (activate).

    Contract:
      - plans_craft: {amb_id: [PlanStep, ...]} from craft_and_send_plans()
      - Each PlanStep uses Facility objects already (not IDs).
      - Ambulance objects live in c2.env.amb_by_id and are tracked as idle in c2.idle_ambulances.

    What this does:
      1) for each amb_id in plans_craft:
         - skip if ambulance missing or not idle
         - set amb.plan
         - remove amb from idle list
         - mark involved requests in c2.pending_transport_requests_df as 'assigned'
           (based on loaded patients' ids)
         - activate ambulance
    """
    env = c2.env

    for amb_id, plan in (plans_craft or {}).items():
        amb = env.amb_by_id.get(amb_id)
        if amb is None:
            continue

        # avoid double dispatch
        if amb not in c2.idle_ambulances:
            continue

        # attach plan
        amb.plan = plan

        # mark ambulance busy
        c2.idle_ambulances.remove(amb)

        # mark requests assigned (recommended; otherwise they stay "pending" forever)
        loaded_patient_ids = {p.patient_id for step in plan for p in step.load_patients}
        if loaded_patient_ids:
            mask = (
                    (c2.pending_transport_requests_df["status"] == "pending")
                    & (c2.pending_transport_requests_df["casualty_id"].astype(str).isin(
                {str(x) for x in loaded_patient_ids}))
            )
            c2.pending_transport_requests_df.loc[mask, "status"] = "assigned"

        # dispatch
        amb.activate()


# ---- registries: lever_value -> function ----
FACILITY_ELIGIBILITY_POLICIES: Dict[int, FacilityEligibilityFn] = {
    0: fac_eligibility_base,
}

FACILITY_ASSIGNMENT_POLICIES: Dict[int, FacilityAssignmentFn] = {
    0: fac_assign_template,
}

AMBULANCE_ELIGIBILITY_POLICIES: Dict[int, AmbEligibilityFn] = {
    0: amb_eligibility_base,
}

AMBULANCE_ASSIGNMENT_POLICIES: Dict[int, AmbAssignmentFn] = {
    0: amb_assign_template,
}


# ============================================================
# SNAPSHOT + DISPATCH HELPERS (TEMPLATES)
# ============================================================


# ============================================================
# TRANSPORT C2 (CORE CLASS)
# ============================================================

class TransportC2(sim.Component):
    """
    C2 dispatcher for one evacuation leg.

    Dataflow (fixed in code):
      pending requests -> facility snapshot -> request->facility eligibility
      -> request->facility assignment -> request/facility->ambulance eligibility
      -> ambulance assignment -> craft plans -> dispatch

    Levers are implemented by dict registries:
      FACILITY_ELIGIBILITY_POLICIES
      FACILITY_ASSIGNMENT_POLICIES
      AMBULANCE_ELIGIBILITY_POLICIES
      AMBULANCE_ASSIGNMENT_POLICIES

    Add new lever:
      1) implement new function with the correct signature
      2) register it in the appropriate dict with a new integer key
      No changes required in process().
    """

    def setup(
            self,
            c2_id: str,
            facility_ids: list[str],
            ambulance_ids: list[str],
            corridors_matrix: pd.DataFrame,
            ambulances_eligibility: pd.DataFrame,
            lever_value_facility_eligibility: int = 0,
            lever_value_facility_assignment: int = 0,
            lever_value_ambulance_eligibility: int = 0,
            lever_value_ambulance_assignment: int = 0,
    ):
        self.c2_id = c2_id
        self.facility_ids = list(facility_ids)
        self.ambulance_ids = list(ambulance_ids)

        # connected later
        self.facilities: list["Facility"] = []
        self.ambulances: list["Ambulance"] = []
        self.idle_ambulances: list["Ambulance"] = []

        # core tables
        self.pending_transport_requests_df = self._init_pending_requests_df()
        self.corridors_matrix = corridors_matrix.copy()
        self.ambulances_eligibility = ambulances_eligibility.copy()

        # lever values
        self.lever_value_facility_eligibility = int(lever_value_facility_eligibility)
        self.lever_value_facility_assignment = int(lever_value_facility_assignment)
        self.lever_value_ambulance_eligibility = int(lever_value_ambulance_eligibility)
        self.lever_value_ambulance_assignment = int(lever_value_ambulance_assignment)

    # --------------------------------------------------------
    # inbound events
    # --------------------------------------------------------

    def receive_request(self, req: "TransportRequest"):
        """
        Convert received TransportRequest -> row in pending_transport_requests_df and activate C2.

        The TransportRequest is assumed minimal (per your spec):
          casualty_id, triage, issuing facility, injury time,
          time of issuing, needed unit
        """
        issuing_facility_id = getattr(getattr(req, "issuing_facility", None), "loc_id", None) \
                              or getattr(req, "issuing_facility_id", None)

        row = {
            "req_id": getattr(req, "req_id", None),
            "casualty_id": getattr(req, "casualty_id", None) or getattr(req.patient, "patient_id", None),
            "triage": getattr(req, "triage", None),
            "issuing_facility_id": issuing_facility_id,
            "injury_time": getattr(req, "injury_time", None),
            "t_issued": getattr(req, "t_issued", None),
            "needed_unit": getattr(req, "needed_unit", None),
            "status": getattr(req, "status", "pending"),
        }

        self.pending_transport_requests_df = pd.concat(
            [self.pending_transport_requests_df, pd.DataFrame([row])],
            ignore_index=True
        )

        self.activate()

    def ambulance_idle(self, amb: "Ambulance"):
        """
        Mark ambulance idle, then activate C2 to try dispatching.
        """
        if amb in self.ambulances and amb not in self.idle_ambulances:
            self.idle_ambulances.append(amb)
            self.activate()

    # --------------------------------------------------------
    # main loop
    # --------------------------------------------------------

    def process(self):
        while True:
            self.passivate()

            requests_df = self._get_pending_requests()
            if requests_df.empty:
                continue

            if len(self.idle_ambulances) == 0:
                continue

            facilities_df = snapshot_facilities(self.facilities)

            # ---- requests -> eligible facilities (policy) ----
            fac_elig_fn = FACILITY_ELIGIBILITY_POLICIES.get(self.lever_value_facility_eligibility)

            eligible_facilities_df = fac_elig_fn(
                requests_df=requests_df,
                corridors_matrix=self.corridors_matrix,
            )

            # ---- eligible facilities -> assigned facility (policy) ----
            fac_assign_fn = FACILITY_ASSIGNMENT_POLICIES.get(self.lever_value_facility_assignment)
            assigned_facilities_df = fac_assign_fn(
                eligible_facilities_df=eligible_facilities_df,
                facilities_df=facilities_df,
                distance_matrix=self.env.distance_matrix,
            )

            # ---- assigned facility -> eligible ambulances (policy) ----
            amb_elig_fn = AMBULANCE_ELIGIBILITY_POLICIES.get(self.lever_value_ambulance_eligibility)
            eligible_pairs_df = amb_elig_fn(
                assigned_facilities_df=assigned_facilities_df,
                ambulances_eligibility=self.ambulances_eligibility,
                idle_ambulances=self.idle_ambulances,
            )

            # ---- eligible pairs -> assigned ambulances (policy) ----
            amb_assign_fn = AMBULANCE_ASSIGNMENT_POLICIES.get(self.lever_value_ambulance_assignment)
            assigned_ambulances_df = amb_assign_fn(
                eligible_pairs_df=eligible_pairs_df,
                distance_matrix=self.env.distance_matrix,
            )

            plans_craft = craft_plans(c2=self, assigned_ambulances_df=assigned_ambulances_df)
            dispatch_ambulances(c2=self, plans_craft=plans_craft)

    # --------------------------------------------------------
    # internal df helpers
    # --------------------------------------------------------

    def _init_pending_requests_df(self) -> pd.DataFrame:
        """
        Canonical schema for pending transport requests.
        Keep stable so policies can rely on column names.
        """
        return pd.DataFrame(columns=[
            "req_id",
            "casualty_id",
            "triage",
            "issuing_facility_id",
            "injury_time",
            "t_issued",
            "needed_unit",
            "status",  # pending/assigned/cancelled/...
        ])

    def _get_pending_requests(self) -> pd.DataFrame:
        """
        Return only rows still pending.
        """
        df = self.pending_transport_requests_df
        if df.empty:
            return df
        return df[df["status"] == "pending"].copy()


def snapshot_facilities(facilities: List["Facility"]) -> pd.DataFrame:
    """
    Snapshot current facility state into a DataFrame.

    Spec:
      - One row per facility
      - One column per service type (global set across all facilities)
      - Cell = expected waiting time *if someone joins that service queue now*
      - Cell = False if that service is not present in the facility

    Requirements:
      - Each Facility has:
          - loc_id
          - units: dict[service_type -> Unit or None]
      - Each Unit has:
          - expected_wait_if_join_now() -> float
    """
    if not facilities:
        return pd.DataFrame()

    # 1) Collect the global set of service types across all facilities
    all_service_types = set()
    for fac in facilities:
        all_service_types.update(fac.units.keys())
    all_service_types = sorted(all_service_types)

    # 2) Build rows
    rows = []
    for fac in facilities:
        row = {"facility_id": fac.loc_id}

        for service_type in all_service_types:
            u = fac.units.get(service_type, None)

            # "False if the service is not in the facility"
            if u is None:
                row[service_type] = False
            else:
                # expected wait time right now
                row[service_type] = float(u.expected_wait_if_join_now())

        rows.append(row)

    df = pd.DataFrame(rows)

    return df


class CasualtySpawner(sim.Component):
    def setup(self, casualties_df):
        self.df = casualties_df.sort_values("t_create").reset_index(drop=True)

    def process(self):
        for _, row in self.df.iterrows():
            dt = float(row.t_create) - self.env.now()
            if dt > 0:
                self.hold(dt)
            spawn = self.env.facility_by_id[row.spawn_facility_id]
            injury_time = float(getattr(row, "injury_time", row.t_create))
            p = Patient(
                patient_id=uuid.uuid4().hex,
                triage=row.triage,
                prescription=row.prescription,
                life_time_dist=row.lifetime_dist,
                spawn_facility=spawn,
                injury_time=injury_time,
            )
            spawn.patient_handed_over(p)


# ============================================================
# BUILDER + CONNECTOR
# ============================================================

class ModelBuilder:
    def __init__(self, env):
        self.env = env

    def build_roads(self, distance_matrix: pd.DataFrame):
        """
        distance_matrix: DataFrame with locations (facilities/AXPs) as both index and columns.
        Values are distances in km. Use inf or NaN for no direct connection.
        """
        self.env.distance_matrix = distance_matrix.copy()

    def build_axps(self, axp_df: pd.DataFrame):
        self.env.axp_by_id = {}
        for r in axp_df.itertuples(index=False):
            self.env.axp_by_id[r.axp_id] = AXP(name=r.axp_id, axp_id=r.axp_id,
                                               assigned_facility_ids=r.assigned_facility_ids)

    def build_c2s(self, c2_df: pd.DataFrame):
        self.env.c2_by_id = {}
        levers = getattr(self.env, "levers", {})
        for r in c2_df.itertuples(index=False):
            corridors_matrix = getattr(r, "corridors_matrix", pd.DataFrame())
            ambulances_eligibility = getattr(r, "ambulances_eligibility", pd.DataFrame())
            self.env.c2_by_id[r.c2_id] = TransportC2(
                name=r.c2_id,
                c2_id=r.c2_id,
                facility_ids=r.facility_ids,
                ambulance_ids=r.ambulance_ids,
                corridors_matrix=corridors_matrix,
                ambulances_eligibility=ambulances_eligibility,
                lever_value_facility_eligibility=levers.get("facility_eligibility", 0),
                lever_value_facility_assignment=levers.get("facility_assignment", 0),
                lever_value_ambulance_eligibility=levers.get("ambulance_eligibility", 0),
                lever_value_ambulance_assignment=levers.get("ambulance_assignment", 0),
            )

    def build_facilities(self, facility_df: pd.DataFrame, units_df: pd.DataFrame):
        fac_to_service_types = {}
        for r in units_df.itertuples(index=False):
            fac_to_service_types.setdefault(r.facility_id, []).append(r.service_type)

        self.env.facility_by_id = {}
        for r in facility_df.itertuples(index=False):
            service_types = fac_to_service_types.get(r.facility_id, [])
            self.env.facility_by_id[r.facility_id] = Facility(
                name=r.facility_id,
                facility_id=r.facility_id,
                facility_type=r.facility_type,
                c2_up_id=r.c2_up_id,
                unload_key=r.unload_policy,
                load_key=r.load_policy,
                service_types=service_types,
            )

    def build_units(self, units_df: pd.DataFrame):
        self.env.unit_by_id = {}
        for r in units_df.itertuples(index=False):
            unit_id = str(r.unit_id)
            u = Unit(
                name=unit_id,
                facility=None,
                capacity=int(r.capacity),
                service_time_dist=r.service_time_dist,
                service_type=r.service_type,
            )
            u._facility_id = r.facility_id
            self.env.unit_by_id[unit_id] = u

    def build_ambulances(self, ambulance_df: pd.DataFrame):
        self.env.amb_by_id = {}
        for r in ambulance_df.itertuples(index=False):
            amb = Ambulance(
                name=r.amb_id,
                amb_id=r.amb_id,
                capacity=int(r.capacity),
                speed_kmh=float(r.speed_kmh),
                start_loc_id=r.start_loc_id,
                c2_id=r.c2_id,
            )
            self.env.amb_by_id[r.amb_id] = amb

    def build_all(self, distance_matrix, axp_df, facility_df, units_df, ambulance_df, c2_df):
        self.build_roads(distance_matrix)
        self.build_axps(axp_df)
        self.build_c2s(c2_df)
        self.build_facilities(facility_df, units_df)
        self.build_units(units_df)
        self.build_ambulances(ambulance_df)


class ModelConnector:
    def __init__(self, env):
        self.env = env

    def connect_units(self):
        for unit_id, u in self.env.unit_by_id.items():
            fac = self.env.facility_by_id[u._facility_id]
            u.facility = fac

    def connect_facilities(self):
        for fac in self.env.facility_by_id.values():
            fac.c2_up = self.env.c2_by_id.get(fac.c2_up_id) if fac.c2_up_id else None
            # Connect units by service_type: find units belonging to this facility
            for unit in self.env.unit_by_id.values():
                if unit._facility_id == fac.loc_id:
                    fac.units[unit.service_type] = unit

    def connect_axps(self):
        self.env.axp_of_facility = {}
        for axp in self.env.axp_by_id.values():
            axp.assigned_facilities = [
                self.env.facility_by_id[fid]
                for fid in axp.assigned_facility_ids
                if fid in self.env.facility_by_id
            ]
            for fid in axp.assigned_facility_ids:
                self.env.axp_of_facility[fid] = axp

    def connect_ambulances(self):
        for amb in self.env.amb_by_id.values():
            amb.c2 = self.env.c2_by_id[amb.c2_id]

            # resolve start location
            if amb.start_loc_id in self.env.axp_by_id:
                amb.location = self.env.axp_by_id[amb.start_loc_id]
            else:
                amb.location = self.env.facility_by_id[amb.start_loc_id]

    def connect_c2s(self):
        for c2 in self.env.c2_by_id.values():
            c2.facilities = [self.env.facility_by_id[fid] for fid in c2.facility_ids]
            c2.ambulances = [self.env.amb_by_id[aid] for aid in c2.ambulance_ids]

    def connect_all(self):
        self.connect_units()
        self.connect_facilities()
        self.connect_axps()
        self.connect_ambulances()
        self.connect_c2s()


# ============================================================
# RUN MODEL FUNCTION (inputs: dfs + lever values)
# ============================================================

def run_model(
        distance_matrix: pd.DataFrame,
        axp_df: pd.DataFrame,
        facility_df: pd.DataFrame,
        units_df: pd.DataFrame,
        ambulance_df: pd.DataFrame,
        c2_df: pd.DataFrame,
        casualties_df: pd.DataFrame,
        levers: dict,
        till: float = 300.0,
        trace: bool = False,
):
    """
    Build + run the simulation using dataframe inputs and lever values.

    levers example: {
        "facility_eligibility": 0,
        "facility_assignment": 0,
        "ambulance_eligibility": 0,
        "ambulance_assignment": 0,
    }
    """
    env = sim.Environment(trace=trace)
    env.patient_log_rows = []
    env.patient_log_columns = ["time", "patient_id", "action", "life_budget", "needed_unit", "current_location",
                               "is_dead", "is_safe"]
    env.safe = []

    # set levers on env for C2 construction
    env.levers = {
        "facility_eligibility": int(levers.get("facility_eligibility", 0)),
        "facility_assignment": int(levers.get("facility_assignment", 0)),
        "ambulance_eligibility": int(levers.get("ambulance_eligibility", 0)),
        "ambulance_assignment": int(levers.get("ambulance_assignment", 0)),
    }

    env.unload_policy = {"unload_planned": unload_planned}
    env.load_policy = {"load_planned": load_planned}

    builder = ModelBuilder(env)
    builder.build_all(distance_matrix, axp_df, facility_df, units_df, ambulance_df, c2_df)

    connector = ModelConnector(env)
    connector.connect_all()

    spawner = CasualtySpawner(casualties_df=casualties_df)
    env.run(till=till)
    env.patient_log_df = pd.DataFrame(env.patient_log_rows, columns=env.patient_log_columns)

    # return env + some results
    safe = [(p.patient_id, p.triage, p.t_safe, p.spawn_facility.loc_id) for p in env.safe]
    return env, pd.DataFrame(safe, columns=["patient_id", "triage", "t_safe", "spawn_facility_id"])


# ============================================================
# DATA LOADING
# ============================================================

def load_data_from_csv(data_folder: str = "data", load_casualties: bool = True):
    """
    Load all simulation data from CSV files in the specified folder.

    Expected structure:
        data_folder/
            simulation_initiation/
                distance_matrix.csv
                axp.csv
                facilities.csv
                units.csv
                ambulances.csv
                corridors_matrix.csv
                ambulances_eligibility.csv
                c2.csv
                casualties.csv  (optional, generated by casualty_generator.py)

    Parameters:
        data_folder: path to data folder
        load_casualties: if True, load casualties.csv; if False, skip it

    Returns a dict with all DataFrames ready for run_model().
    """
    import os

    sim_init_folder = os.path.join(data_folder, "simulation_initiation")

    def path(filename):
        return os.path.join(sim_init_folder, filename)

    # Distance matrix
    distance_matrix = pd.read_csv(path("distance_matrix.csv"), index_col=0)

    # AXP data - convert comma-separated string to list
    axp_df = pd.read_csv(path("axp.csv"))
    axp_df["assigned_facility_ids"] = axp_df["assigned_facility_ids"].apply(
        lambda x: [s.strip() for s in str(x).split(",")] if pd.notna(x) else []
    )

    # Facilities
    facility_df = pd.read_csv(path("facilities.csv"))
    facility_df["c2_up_id"] = facility_df["c2_up_id"].replace({pd.NA: None, "": None}).where(
        facility_df["c2_up_id"].notna(), None
    )

    # Units - convert min/max to Uniform distribution
    # unit_id is unique identifier, service_type is the type of service (DCR, PEC, DCS, etc.)
    units_df = pd.read_csv(path("units.csv"), dtype={"unit_id": str})
    units_df["service_time_dist"] = units_df.apply(
        lambda r: sim.Uniform(r["service_time_min"], r["service_time_max"]), axis=1
    )

    # Ambulances
    ambulance_df = pd.read_csv(path("ambulances.csv"))

    # Corridors matrix
    corridors_matrix = pd.read_csv(path("corridors_matrix.csv"), index_col=0)

    # Ambulances eligibility
    ambulances_eligibility = pd.read_csv(path("ambulances_eligibility.csv"), index_col=0)

    # C2 data - convert comma-separated strings to lists and attach matrices
    c2_df = pd.read_csv(path("c2.csv"))
    c2_df["facility_ids"] = c2_df["facility_ids"].apply(
        lambda x: [s.strip() for s in str(x).split(",")] if pd.notna(x) else []
    )
    c2_df["ambulance_ids"] = c2_df["ambulance_ids"].apply(
        lambda x: [s.strip() for s in str(x).split(",")] if pd.notna(x) else []
    )
    c2_df["corridors_matrix"] = [corridors_matrix] * len(c2_df)
    c2_df["ambulances_eligibility"] = [ambulances_eligibility] * len(c2_df)

    result = {
        "distance_matrix": distance_matrix,
        "axp_df": axp_df,
        "facility_df": facility_df,
        "units_df": units_df,
        "ambulance_df": ambulance_df,
        "c2_df": c2_df,
    }

    if load_casualties:
        # Casualties (generated by casualty_generator.py)
        # Columns: t_create, spawn_facility_id, triage, prescription, lifetime_low, lifetime_high
        casualties_df = pd.read_csv(path("casualties.csv"))
        casualties_df["prescription"] = casualties_df["prescription"].apply(eval)
        casualties_df["lifetime_dist"] = casualties_df.apply(
            lambda r: sim.Uniform(r["lifetime_low"], r["lifetime_high"]), axis=1
        )
        result["casualties_df"] = casualties_df

    return result


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    data = load_data_from_csv("data")

    env, safe_df = run_model(
        distance_matrix=data["distance_matrix"],
        axp_df=data["axp_df"],
        facility_df=data["facility_df"],
        units_df=data["units_df"],
        ambulance_df=data["ambulance_df"],
        c2_df=data["c2_df"],
        casualties_df=data["casualties_df"],
        levers={
            "facility_eligibility": 0,
            "facility_assignment": 0,
            "ambulance_eligibility": 0,
            "ambulance_assignment": 0,
        },
        till=15000,
    )

    print(env.patient_log_df)