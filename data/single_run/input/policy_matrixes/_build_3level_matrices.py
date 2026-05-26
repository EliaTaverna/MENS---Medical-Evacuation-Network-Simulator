"""
Build 3-level eligibility matrices for the elia_midterm scenario.

For each dimension (facility, ambulance), create the middle level:
  0 = strict:   own battalion only (already exists)
  1 = bn_pool:  any battalion within same brigade (NEW)
  2 = bde_pool: across brigades too (= current loose, just copied)
"""
import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent


def get_brigade(facility_id: str) -> str:
    """Extract brigade from ID like ccp_br1_bt2_3 or r2e_br1_1."""
    for part in facility_id.lower().split("_"):
        if part.startswith("br"):
            return part
    return "?"


# ─── FACILITY ELIGIBILITY ───────────────────────────────────────────
fac_strict = pd.read_excel(
    BASE / "facility_eligibility" / "facility_eligibility_elia_midterm_strict.xlsx",
    index_col=0,
)
fac_loose = pd.read_excel(
    BASE / "facility_eligibility" / "facility_eligibility_elia_midterm_loose.xlsx",
    index_col=0,
)

# bn_pool = strict + intra-brigade edges from loose
fac_bn = fac_strict.copy()
diff = fac_loose - fac_strict
for origin in diff.index:
    for dest in diff.columns:
        if diff.loc[origin, dest] == 1:
            if get_brigade(origin) == get_brigade(dest):
                fac_bn.loc[origin, dest] = 1

# bde_pool = full loose
fac_bde = fac_loose.copy()

print("=== FACILITY ELIGIBILITY ===")
s, b, d = (fac_strict == 1).sum().sum(), (fac_bn == 1).sum().sum(), (fac_bde == 1).sum().sum()
print(f"  strict:   {s} edges")
print(f"  bn_pool:  {b} edges  (+{b - s} intra-brigade)")
print(f"  bde_pool: {d} edges  (+{d - b} cross-brigade)")

fac_bn.to_excel(BASE / "facility_eligibility" / "facility_eligibility_elia_midterm_bn_pool.xlsx")
fac_bde.to_excel(BASE / "facility_eligibility" / "facility_eligibility_elia_midterm_bde_pool.xlsx")
print("  Saved bn_pool and bde_pool.\n")

# ─── AMBULANCE ELIGIBILITY ─────────────────────────────────────────
amb_strict = pd.read_excel(
    BASE / "ambulance_eligibility" / "ambulance_eligibility_elia_midterm_strict.xlsx",
    index_col=0,
)
amb_loose = pd.read_excel(
    BASE / "ambulance_eligibility" / "ambulance_eligibility_elia_midterm_loose.xlsx",
    index_col=0,
)

# Determine each ambulance's brigade from its strict assignments
amb_brigades = {}
for amb in amb_strict.index:
    served = amb_strict.columns[amb_strict.loc[amb] == 1]
    bdes = {get_brigade(f) for f in served if get_brigade(f) != "?"}
    amb_brigades[amb] = bdes

# bn_pool = strict + intra-brigade edges from loose
amb_bn = amb_strict.copy()
diff_a = amb_loose - amb_strict
for amb in diff_a.index:
    a_bdes = amb_brigades[amb]
    for fac in diff_a.columns:
        if diff_a.loc[amb, fac] == 1:
            if get_brigade(fac) in a_bdes:
                amb_bn.loc[amb, fac] = 1

# bde_pool = full loose
amb_bde = amb_loose.copy()

print("=== AMBULANCE ELIGIBILITY ===")
s, b, d = (amb_strict == 1).sum().sum(), (amb_bn == 1).sum().sum(), (amb_bde == 1).sum().sum()
print(f"  strict:   {s} edges")
print(f"  bn_pool:  {b} edges  (+{b - s} intra-brigade)")
print(f"  bde_pool: {d} edges  (+{d - b} cross-brigade)")

amb_bn.to_excel(BASE / "ambulance_eligibility" / "ambulance_eligibility_elia_midterm_bn_pool.xlsx")
amb_bde.to_excel(BASE / "ambulance_eligibility" / "ambulance_eligibility_elia_midterm_bde_pool.xlsx")
print("  Saved bn_pool and bde_pool.\n")

print("Done — 4 new matrices created.")
