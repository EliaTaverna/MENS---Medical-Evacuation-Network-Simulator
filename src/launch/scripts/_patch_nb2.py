"""Patch build_distance_matrix_meisner.ipynb — 1 CCP per R1 + config export."""
import json
from pathlib import Path

nb_path = Path("build_distance_matrix_meisner.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))

def set_cell(idx, src):
    nb["cells"][idx]["source"] = src

# ── Cell 3: config ─────────────────────────────────────────────────────────────
set_cell(3, """\
import pathlib   # needed for OUT_BASE

# ── Battlefield geometry (km) ─────────────────────────────────────────────────
BN_STRIP_DEPTH =  8.0
BDE_AREA_DEPTH = 42.0

DEPTH = {
    "CCP":   0.500 * BN_STRIP_DEPTH,   #  4.0 km
    "Role1": 0.875 * BN_STRIP_DEPTH,   #  7.0 km
    "AXP":   BN_STRIP_DEPTH + 4.0,     # 12.0 km
    "Role2": BN_STRIP_DEPTH + 22.0,    # 30.0 km
}

# n_r1: one CCP is created PER R1 — the only way to spread load across R1s
BRIGADES = [
    {
        "id": "bde1", "par": 3000,
        "x_start":  0.0, "width": 30.0,
        "n_r1": 3, "n_r2": 3,
        "battalions": [
            {"id": "bn11", "par":  800},
            {"id": "bn12", "par":  800},
            {"id": "bn13", "par": 1000},
        ],
    },
    {
        "id": "bde2", "par": 4000,
        "x_start": 30.0, "width": 30.0,
        "n_r1": 5, "n_r2": 4,   # central brigade: 5 R1s per paper
        "battalions": [
            {"id": "bn21", "par": 1200},
            {"id": "bn22", "par": 1200},
            {"id": "bn23", "par": 1200},
        ],
    },
    {
        "id": "bde3", "par": 3000,
        "x_start": 60.0, "width": 30.0,
        "n_r1": 3, "n_r2": 3,
        "battalions": [
            {"id": "bn31", "par": 1000},
            {"id": "bn32", "par":  800},
            {"id": "bn33", "par":  800},
        ],
    },
]

# ── Unit service-time parameters ──────────────────────────────────────────────
DCS_CAPACITY, DCS_MIN, DCS_MAX = 2,  94, 183
DCR_CAPACITY, DCR_MIN, DCR_MAX = 2,  27,  28
PEC_CAPACITY, PEC_MIN, PEC_MAX = 2,  27,  28

# ── Ambulance parameters ──────────────────────────────────────────────────────
CCP_AMBS_PER_R1   = 3
R1R2_AMBS_PER_BDE = 4
AMB_CAPACITY      = 5
AMB_SPEED_KMH     = 30

# ── Casualty rates (per soldier per day) ──────────────────────────────────────
CAS_BCAS_RATE = 0.023
CAS_DNBI_RATE = 0.0013

# ── Output ────────────────────────────────────────────────────────────────────
OUT_BASE = pathlib.Path("data/single_run/input")
SUFFIX   = "meisner"
""")

# ── Cell 5: 1 CCP per R1 ──────────────────────────────────────────────────────
set_cell(5, """\
R2_SUFFIXES = ["", "_b", "_c", "_d", "_e", "_f", "_g", "_h", "_i"]

all_nodes    = []   # every map node (CCP, R1, AXP, R2)
ccp_r1_pairs = []   # [(ccp_id, r1_id)] — one pair per R1

for bde in BRIGADES:
    bns           = bde["battalions"]
    n_bn          = len(bns)
    n_r1          = bde.get("n_r1", n_bn)
    n_r2          = bde["n_r2"]
    bde_id        = bde["id"]
    bde_cx        = bde["x_start"] + bde["width"] / 2
    bde_total_par = bde["par"] + sum(bn["par"] for bn in bns)

    # ONE CCP per R1 at the same x — every R1 gets its own casualty stream
    for r in range(1, n_r1 + 1):
        r1_x = bde["x_start"] + (r - 0.5) * bde["width"] / n_r1
        if n_r1 == n_bn:
            r1_id  = f"r1_{bns[r-1]['id']}"
            ccp_id = f"ccp_{bns[r-1]['id']}"
            par    = bns[r-1]["par"]
        else:
            r1_id  = f"r1_{bde_id}_{r}"
            ccp_id = f"ccp_{bde_id}_{r}"
            par    = bde_total_par / n_r1   # evenly split

        all_nodes.append({"facility_id": ccp_id, "facility_type": "CCP",
                           "x": r1_x, "y": DEPTH["CCP"],
                           "par": par, "brigade": bde_id,
                           "battalion": bns[r-1]["id"] if n_r1 == n_bn else None})
        all_nodes.append({"facility_id": r1_id, "facility_type": "Role1",
                           "x": r1_x, "y": DEPTH["Role1"],
                           "par": par, "brigade": bde_id,
                           "battalion": bns[r-1]["id"] if n_r1 == n_bn else None})
        ccp_r1_pairs.append((ccp_id, r1_id))

    # AXP: one per brigade
    all_nodes.append({"facility_id": f"axp_{bde_id}", "facility_type": "AXP",
                       "x": bde_cx, "y": DEPTH["AXP"],
                       "par": bde_total_par, "brigade": bde_id, "battalion": None})

    # R2s: evenly spaced across brigade width
    for r in range(n_r2):
        r2_x  = bde["x_start"] + (r + 0.5) * bde["width"] / n_r2
        r2_id = f"r2_{bde_id}{R2_SUFFIXES[r]}"
        all_nodes.append({"facility_id": r2_id, "facility_type": "Role2",
                           "x": r2_x, "y": DEPTH["Role2"],
                           "par": bde_total_par, "brigade": bde_id, "battalion": None})

fac_df = pd.DataFrame(all_nodes)
print(f"Map nodes: {len(fac_df)}  "
      f"(CCP={sum(fac_df.facility_type=='CCP')}, "
      f"R1={sum(fac_df.facility_type=='Role1')}, "
      f"AXP={sum(fac_df.facility_type=='AXP')}, "
      f"R2={sum(fac_df.facility_type=='Role2')})")
print(f"CCP-R1 pairs ({len(ccp_r1_pairs)}):")
for ccp, r1 in ccp_r1_pairs:
    print(f"  {ccp:22s} -> {r1}")
""")

# ── Cell 17: strict eligibility via ccp_to_r1 ────────────────────────────────
set_cell(17, """\
def brigade_of(fid):
    for bde in BRIGADES:
        bid  = bde["id"]
        bnum = bid.replace("bde", "")
        if f"_{bid}" in fid or f"_bn{bnum}" in fid:
            return bid
    return None

ccp_to_r1 = dict(ccp_r1_pairs)   # built in Cell 2

def make_fac_eligibility(loose=False):
    dm = pd.DataFrame(0, index=sim_fac_ids, columns=sim_fac_ids)
    for src in sim_fac_ids:
        src_type = fac_type_map[src]
        src_bde  = brigade_of(src)
        if src_type == "CCP":
            for dst in sim_fac_ids:
                if fac_type_map[dst] != "Role1":
                    continue
                if loose:
                    dm.loc[src, dst] = 1
                elif ccp_to_r1.get(src) == dst:
                    dm.loc[src, dst] = 1
        elif src_type == "Role1":
            for dst in sim_fac_ids:
                if fac_type_map[dst] != "Role2":
                    continue
                if loose or brigade_of(dst) == src_bde:
                    dm.loc[src, dst] = 1
    return dm

def make_amb_eligibility(loose=False):
    dm = pd.DataFrame(0, index=amb_df["amb_id"].tolist(), columns=sim_fac_ids)
    for _, arow in amb_df.iterrows():
        aid, c2id, start = arow["amb_id"], arow["c2_id"], arow["start_loc_id"]
        bde = brigade_of(start)
        if c2id == "C2_CCP_R1":
            if loose:
                for fid in sim_fac_ids:
                    if fac_type_map[fid] in ("CCP", "Role1"):
                        dm.loc[aid, fid] = 1
            else:
                # R1-stationed amb: its own R1 + paired CCP
                ccp_pair = next((c for c, r in ccp_r1_pairs if r == start), None)
                for fid in [start, ccp_pair]:
                    if fid and fid in sim_fac_ids:
                        dm.loc[aid, fid] = 1
        elif c2id == "C2_R1_R2":
            for fid in sim_fac_ids:
                if fac_type_map[fid] in ("Role1", "Role2"):
                    if loose or brigade_of(fid) == bde:
                        dm.loc[aid, fid] = 1
    return dm

fac_strict = make_fac_eligibility(loose=False)
fac_loose  = make_fac_eligibility(loose=True)
amb_strict = make_amb_eligibility(loose=False)
amb_loose  = make_amb_eligibility(loose=True)

print("Facility eligibility (strict) — non-zero rows:")
for fid in sim_fac_ids:
    s = fac_strict.loc[fid].sum()
    if s > 0:
        eligible = [c for c in sim_fac_ids if fac_strict.loc[fid, c]]
        print(f"  {fid:25s} -> {eligible}")
print(f"\\nAmbulance eligibility shapes: strict={amb_strict.shape}, loose={amb_loose.shape}")
""")

# ── Cell 18: exports + config_meisner.txt ─────────────────────────────────────
set_cell(18, """\
import yaml

def _out(subpath):
    p = OUT_BASE / subpath
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

# ── Build PAR-weighted spawn_map for config file ──────────────────────────────
par_total = sum(bde["par"] + sum(bn["par"] for bn in bde["battalions"])
                for bde in BRIGADES)

spawn_map = {}
for ccp_id, r1_id in ccp_r1_pairs:
    ccp_par = float(fac_df.loc[fac_df.facility_id == ccp_id, "par"].iloc[0])
    spawn_map[ccp_id] = {
        "labels": ["T1_BCAS", "T2_BCAS", "T3_BCAS", "KIA_BCAS", "MIA_BCAS"],
        "rate":   round(ccp_par / par_total * CAS_BCAS_RATE, 8),
    }
    r1_par = float(fac_df.loc[fac_df.facility_id == r1_id, "par"].iloc[0])
    spawn_map[r1_id] = {
        "labels": ["T1_DNBI", "T2_DNBI", "T3_DNBI"],
        "rate":   round(r1_par / par_total * CAS_DNBI_RATE, 8),
    }

config_dict = {
    "par_total":         par_total,
    "n_days":            7,
    "waves_per_day":     3,
    "wave_duration_min": 30,
    "seed":              42,
    "spawn_map":         spawn_map,
}

# ── xlsx exports ──────────────────────────────────────────────────────────────
exports = [
    (_out(f"components/distances/distance_matrix_{SUFFIX}.xlsx"),
     lambda p: dist_df.round(4).to_excel(p)),
    (_out(f"components/distances/coordinates_{SUFFIX}.xlsx"),
     lambda p: fac_df[["facility_id","facility_type","brigade","battalion","x","y","par"]]
               .to_excel(p, index=False)),
    (_out(f"components/facilities/facilities_{SUFFIX}.xlsx"),
     lambda p: sim_fac_df[["facility_id","facility_type","c2_up_id",
                            "unload_policy","load_policy"]].to_excel(p, index=False)),
    (_out(f"components/units/units_{SUFFIX}.xlsx"),
     lambda p: units_df.to_excel(p, index=False)),
    (_out(f"components/AXPs/axp_{SUFFIX}.xlsx"),
     lambda p: axp_df.to_excel(p, index=False)),
    (_out(f"components/ambulances/amb_{SUFFIX}.xlsx"),
     lambda p: amb_df.to_excel(p, index=False)),
    (_out(f"components/c2/c2_{SUFFIX}.xlsx"),
     lambda p: c2_df.to_excel(p, index=False)),
    (_out(f"policy_matrixes/facility_eligibility/facility_eligibility_{SUFFIX}_strict.xlsx"),
     lambda p: fac_strict.to_excel(p)),
    (_out(f"policy_matrixes/facility_eligibility/facility_eligibility_{SUFFIX}_loose.xlsx"),
     lambda p: fac_loose.to_excel(p)),
    (_out(f"policy_matrixes/ambulance_eligibility/ambulance_eligibility_{SUFFIX}_strict.xlsx"),
     lambda p: amb_strict.to_excel(p)),
    (_out(f"policy_matrixes/ambulance_eligibility/ambulance_eligibility_{SUFFIX}_loose.xlsx"),
     lambda p: amb_loose.to_excel(p)),
]

print(f"Writing {len(exports)} xlsx files to {OUT_BASE} ...")
for path, writer in exports:
    writer(path)
    print(f"  ✓  {path.relative_to(OUT_BASE)}")

cfg_path = _out(f"casualty_generation_settings/sampling_settings/config_{SUFFIX}.txt")
with open(cfg_path, "w", encoding="utf-8") as f:
    n_ccp = sum(fac_df.facility_type == "CCP")
    n_r1v = sum(fac_df.facility_type == "Role1")
    f.write(f"# Casualty Generation Config — {SUFFIX}\\n")
    f.write(f"# {n_ccp} CCPs (1 per R1), {n_r1v} R1s — PAR-weighted rates\\n")
    f.write(f"# Total PAR: {par_total}  BCAS: {CAS_BCAS_RATE}/soldier/day  DNBI: {CAS_DNBI_RATE}/soldier/day\\n\\n")
    yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
print(f"  ✓  {cfg_path.relative_to(OUT_BASE)}")
print("Done.")
""")

nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Notebook patched successfully.")
