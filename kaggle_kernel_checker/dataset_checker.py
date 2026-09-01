"""
WeatherGPT — Kaggle Dataset + Environment Checker
Runs ON KAGGLE (P100, internet ON) — no local download.
Checks whether each dataset needs processing and finalises the plan before any training.

Output: /kaggle/working/weathergpt_checker_report.json + .md + console log
"""

import os, sys, json, pathlib, subprocess, platform, collections, re

OUT = pathlib.Path("/kaggle/working/weathergpt_checker")
OUT.mkdir(parents=True, exist_ok=True)
REPORT = {}

def log(msg):
    print(msg)

# 1. Environment
log("="*70)
log("1. KAGGLE ENVIRONMENT")
import torch
env = {}
env["python"] = platform.python_version()
env["torch"] = torch.__version__
env["cuda_available"] = torch.cuda.is_available()
env["cuda_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
try:
    env["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    env["capability"] = str(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None
except Exception as e:
    env["gpu_error"] = str(e)
try:
    env["nvidia_smi"] = subprocess.check_output(["nvidia-smi","--query-gpu=name,memory.total,driver_version","--format=csv"], text=True)[:500]
except Exception as e:
    env["nvidia_smi_error"] = str(e)
env["cwd"] = os.getcwd()
env["ls_working"] = os.listdir("/kaggle/working")[:30]
try:
    env["ls_input"] = os.listdir("/kaggle/input")[:20]
except: env["ls_input"] = []
# Check for our prior datasets (from previous kernel's output)
candidates = ["/kaggle/input/weathergpt-dataset-checker","/kaggle/working/training/datasets","/kaggle/working/weathergpt/training/datasets","/kaggle/working"]
found = []
for root in ["/kaggle/working","/kaggle/input"]:
    for p in pathlib.Path(root).rglob("field_names.csv"):
        found.append(str(p))
    for p in pathlib.Path(root).rglob("matched_pairs.csv"):
        found.append(str(p))
    for p in pathlib.Path(root).rglob("intent_samples.jsonl"):
        found.append(str(p))
env["found_datasets"] = found[:20]
# Also check if the previous weathergpt_p100_train output is available as input
# For now, we will rebuild fresh on the fly to test trainability, but also inspect found
REPORT["environment"] = env
for k,v in env.items():
    log(f"  {k}: {v}")

# P100 sm_60 handling
force_cpu = False
if env.get("capability"):
    try:
        major = int(env["capability"].strip("()").split(",")[0])
        if major < 7:
            log(f"  ⚠️ P100 sm_60 incompatible with torch {env['torch']} (needs sm_70+). Training will be forced to CPU.")
            force_cpu = True
    except: pass
REPORT["force_cpu"] = force_cpu

# 2. Locate or rebuild datasets ON KAGGLE (no local)
log("\n" + "="*70)
log("2. DATASET LOCATION / REBUILD DECISION")
# We will rebuild fresh ON KAGGLE to test the exact pipeline that training will use
# But first inspect any pre-existing files
import pathlib as pl
roots_to_check = [pl.Path("/kaggle/working/training/datasets"), pl.Path("training/datasets"), pl.Path("/kaggle/working/weathergpt/training/datasets")]
existing = {}
for rp in roots_to_check:
    for name in ["field_names.csv","matched_pairs.csv","intent_samples.jsonl"]:
        p = rp / name
        if p.exists():
            existing[name] = str(p)
log(f" existing datasets: {existing}")

# Rebuild fresh ON KAGGLE for audit (so we test the builder that P100 training will actually use)
# Import builder logic inline (avoid import path issues)
build_root = pl.Path("training/datasets")
build_root.mkdir(parents=True, exist_ok=True)

# M1: field_names
import csv, random
random.seed(42)
extra_vocab = [
    ("2t","temperature_2m","instant"), ("t2m","temperature_2m","instant"), ("T2","temperature_2m","instant"),
    ("temperature_2m","temperature_2m","instant"), ("temp","temperature_2m","instant"), ("TMAX","temperature_max","max"), ("TMIN","temperature_min","min"),
    ("APCP","precipitation_amount","accumulation"), ("tp","precipitation_amount","accumulation"), ("precipitation","precipitation_amount","accumulation"),
    ("rain","precipitation_amount","accumulation"), ("rainfall","precipitation_amount","accumulation"), ("total_precipitation","precipitation_amount","accumulation"),
    ("RAINC","precipitation_amount","accumulation"), ("RAINNC","precipitation_amount","accumulation"),
    ("PoP","precipitation_probability","probability"), ("precipitation_probability","precipitation_probability","probability"), ("chance_of_rain","precipitation_probability","probability"),
    ("prate","precipitation_rate","instant"), ("rain_rate","precipitation_rate","instant"),
    ("u10","wind_speed","instant"), ("v10","wind_speed","instant"), ("U10","wind_speed","instant"), ("10m_wind","wind_speed","instant"), ("wind_gust","wind_gust","instant"),
    ("Heavy Rainfall","heavy_rain_warning","categorical"), ("Thunderstorm","heavy_rain_warning","categorical"), ("Cyclone","heavy_rain_warning","categorical"),
    ("Fog","heavy_rain_warning","categorical"), ("Heat Wave","heavy_rain_warning","categorical"), ("Extremely Heavy Rain","heavy_rain_warning","categorical"),
]
rows = []
for raw, canon, stat in extra_vocab:
    variants = [raw, raw.lower(), raw.upper(), f" {raw} ", f"{raw} (mm)", f"GFS:{raw}", f"IMD:{raw}", raw.replace("_"," "), raw.replace(" ","_")]
    for v in variants[:8]:
        rows.append((v.strip(), canon, stat))
while len(rows) < 1500:
    base = random.choice(extra_vocab)
    raw, canon, stat = base
    noisy = raw + random.choice(["", " ", " (mm)", ""])
    if random.random()<0.1: noisy = noisy.upper()
    rows.append((noisy, canon, stat))
random.shuffle(rows)
rows = rows[:1600]
field_path = build_root / "field_names.csv"
with open(field_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["raw_field","canonical_variable","statistic","accumulation_hours","source_hint"])
    for raw, canon, stat in rows:
        acc = "6" if canon=="precipitation_amount" else ""
        w.writerow([raw, canon, stat, acc, "checker rebuild"])

# M2: matched_pairs via Open-Meteo (historical GFS vs ERA5) — same as P100 v2
import httpx, asyncio, time
points = [(21.14,79.08,240),(19.07,72.87,14),(28.61,77.20,216),(22.57,88.36,9),(13.08,80.27,6),(12.97,77.59,920),(18.52,73.85,560),(26.91,75.78,431),(23.02,72.57,53),(25.43,81.84,98),(17.38,78.48,542),(15.31,75.12,671),(11.01,76.96,411),(30.73,76.77,350),(34.08,74.79,1585),(20.29,85.82,45),(26.14,91.73,55),(21.25,81.62,298),(24.58,73.71,423),(15.91,75.56,696)]

async def fetch_one(lat, lon):
    async with httpx.AsyncClient(timeout=40) as c:
        hist_url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
        era5_url = "https://archive-api.open-meteo.com/v1/era5"
        params_hist = {"latitude":lat,"longitude":lon,"start_date":"2024-01-01","end_date":"2024-01-10","hourly":"temperature_2m,precipitation","models":"gfs_seamless","timezone":"UTC"}
        params_era = {"latitude":lat,"longitude":lon,"start_date":"2024-01-01","end_date":"2024-01-10","hourly":"temperature_2m,precipitation","timezone":"UTC"}
        rh = await c.get(hist_url, params=params_hist)
        re = await c.get(era5_url, params=params_era)
        rh.raise_for_status(); re.raise_for_status()
        return rh.json()["hourly"], re.json()["hourly"]

all_pairs=[]
for lat,lon,elev in points:
    try:
        hist, era = asyncio.run(fetch_one(lat,lon))
        times = hist["time"]
        for i, t in enumerate(times):
            gfs_t = hist["temperature_2m"][i]; gfs_p = hist["precipitation"][i] or 0
            obs_t = era["temperature_2m"][i]; obs_p = era["precipitation"][i] or 0
            if gfs_t is None or obs_t is None: continue
            all_pairs.append((lat,lon,elev,i,t,gfs_t,gfs_p,obs_t,obs_p))
        time.sleep(0.15)
    except Exception as e:
        log(f"  M2 fetch failed {lat},{lon} {e}")

pair_path = build_root / "matched_pairs.csv"
with open(pair_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["lat","lon","elevation_m","lead_hours","valid_from","gfs_t2m_k","gfs_apcp_mm","obs_t2m_c","obs_apcp_mm"])
    for lat,lon,elev,lead,t,gfs_t,gfs_p,obs_t,obs_p in all_pairs:
        w.writerow([lat,lon,elev,lead,t,gfs_t+273.15,gfs_p,obs_t,obs_p])
log(f" rebuilt M1 {field_path} rows {len(rows)}")
log(f" rebuilt M2 {pair_path} rows {len(all_pairs)}")

# M3: intent
import json
templates = [
    "Will it rain in {loc} {time}?","Will it rain in {loc} {time} and should I spray pesticide?",
    "Is there a heavy rain warning for {loc} {time}?","What is the temperature in {loc} {time}?",
    "What is the wind speed in {loc} {time}?","Can I go fishing in {loc} {time}?",
    "Should I irrigate in {loc} {time}?","Forecast for {loc} {time}",
    "Chance of rain in {loc} {time}?","Will it be hot in {loc} {time}?",
    "Kal {loc} me baarish hogi {time}?","kya {loc} me {time} baarish hogi?","{loc} me {time} mausam kaisa rahega?",
    "My village near {loc} {time} — rain?","Pincode {pincode} {time} weather?","{lat},{lon} {time} forecast?",
]
locs = ["Nagpur","Mumbai","Delhi","Kolkata","Chennai","Bengaluru","Pune","Malegaon","Ahmedabad","Patna","Jaipur","Lucknow","Indore","Bhopal","Nagpur village","my village","440001","400001","110001"]
times = ["today","tonight","tomorrow","tomorrow morning","tomorrow afternoon","tomorrow evening","day after tomorrow","next 3 days","this weekend","coming Monday","23rd Aug"]
decisions = ["pesticide_spraying","marine","irrigation","none","harvest"]
intent_path = build_root / "intent_samples.jsonl"
rows_j=[]
for _ in range(1200):
    tmpl = random.choice(templates)
    loc = random.choice(locs)
    tm = random.choice(times)
    pincode = random.choice(["440001","400001","110001"])
    lat = round(random.uniform(18,28),2); lon=round(random.uniform(72,88),2)
    text = tmpl.format(loc=loc, time=tm, pincode=pincode, lat=lat, lon=lon)
    low=text.lower()
    vars_=[]
    if "rain" in low or "baarish" in low: vars_.append("precipitation_amount")
    if "chance" in low: vars_.append("precipitation_probability")
    if "temperature" in low or "hot" in low: vars_.append("temperature_2m")
    if "wind" in low: vars_.append("wind_speed")
    if "warning" in low: vars_.append("heavy_rain_warning")
    if not vars_: vars_=["precipitation_amount"]
    dec = random.choice(decisions) if "spray" in low or "fishing" in low or "irrigate" in low else "none"
    rows_j.append({"text":text,"intent":{"variables":vars_,"time":tm,"location":loc,"decision":dec}})
random.shuffle(rows_j)
with open(intent_path, "w") as f:
    for r in rows_j:
        f.write(json.dumps(r)+"\n")
log(f" rebuilt M3 {intent_path} rows {len(rows_j)}")

# 3. Column + Processing Check
log("\n" + "="*70)
log("3. COLUMN & PROCESSING AUDIT")
report = {}

import pandas as pd
import csv, collections

# M1
log("\n--- M1 field_names.csv ---")
try:
    df1 = pd.read_csv(field_path)
    rpt1 = {}
    rpt1["rows"] = len(df1)
    rpt1["cols"] = list(df1.columns)
    rpt1["dtypes"] = {k:str(v) for k,v in df1.dtypes.items()}
    rpt1["missing"] = df1.isna().sum().to_dict()
    rpt1["unique_raw"] = int(df1["raw_field"].nunique())
    rpt1["per_canonical"] = df1["canonical_variable"].value_counts().to_dict()
    rpt1["per_stat"] = df1["statistic"].value_counts().to_dict()
    rpt1["dup_rows"] = int(df1.duplicated().sum())
    # Check vs LABEL_MAP
    label_map_expected = ["precipitation_amount","precipitation_probability","precipitation_rate","temperature_2m","wind_speed","heavy_rain_warning"]
    extra = set(df1["canonical_variable"].unique()) - set(label_map_expected)
    missing = set(label_map_expected) - set(df1["canonical_variable"].unique())
    rpt1["extra_labels_vs_expected_6"] = list(extra)
    rpt1["missing_labels_vs_expected_6"] = list(missing)
    # Unit hint error: TMAX (mm) should be C
    rpt1["sample_TMAX_mm"] = int(((df1["raw_field"].str.contains("TMAX", case=False)) & (df1["raw_field"].str.contains("mm"))).sum())
    # Imbalance
    rpt1["needs_processing"] = []
    if rpt1["dup_rows"] > 100: rpt1["needs_processing"].append(f"dedup {rpt1['dup_rows']} duplicates")
    if extra: rpt1["needs_processing"].append(f"label_map mismatch: extra {extra} -> expand LABEL_MAP to 8/9 or collapse")
    if rpt1["per_canonical"].get("precipitation_amount",0) > 400: rpt1["needs_processing"].append("class imbalance 411 vs 37 (tmax)")
    if rpt1["sample_TMAX_mm"]>0: rpt1["needs_processing"].append("fix TMAX (mm) unit error")
    # Check accumulation_hours coverage
    rpt1["acc_hours"] = df1["accumulation_hours"].value_counts().to_dict()
    if set(rpt1["acc_hours"].keys()) == {"6.0"} or set(str(k) for k in rpt1["acc_hours"].keys())=={"6"}:
        rpt1["needs_processing"].append("accumulation_hours only 6 — need 1,3,6,24 variants for window disambiguation")
    if not rpt1["needs_processing"]:
        rpt1["verdict"] = "READY — no processing needed beyond normal tokenization"
    else:
        rpt1["verdict"] = "NEEDS PROCESSING"
    report["M1"] = rpt1
    for k,v in rpt1.items():
        log(f"  M1 {k}: {v}")
except Exception as e:
    log(f" M1 failed {e}")
    report["M1_error"] = str(e)

# M2
log("\n--- M2 matched_pairs.csv ---")
try:
    df2 = pd.read_csv(pair_path)
    rpt2={}
    rpt2["rows"]=len(df2)
    rpt2["cols"]=list(df2.columns)
    rpt2["dtypes"]={k:str(v) for k,v in df2.dtypes.items()}
    rpt2["missing"]=df2.isna().sum().to_dict()
    rpt2["head"]=df2.head(2).to_dict(orient="records")
    rpt2["gfs_K_range"]=[float(df2["gfs_t2m_k"].min()), float(df2["gfs_t2m_k"].max())]
    rpt2["obs_C_range"]=[float(df2["obs_t2m_c"].min()), float(df2["obs_t2m_c"].max())]
    # corr
    import numpy as np
    gfs_c = df2["gfs_t2m_k"] - 273.15
    rpt2["corr_t2m"]=float(np.corrcoef(gfs_c, df2["obs_t2m_c"])[0,1]) if len(df2)>10 else None
    rpt2["corr_precip"]=float(np.corrcoef(df2["gfs_apcp_mm"], df2["obs_apcp_mm"])[0,1]) if df2["obs_apcp_mm"].std()>0 else None
    rpt2["elevation_unique"]=sorted(df2["elevation_m"].unique().tolist())[:10]
    rpt2["lead_unique"]=sorted(df2["lead_hours"].unique().tolist())[:10]
    rpt2["lead_is_sequential"] = list(df2["lead_hours"].unique())[:5]==[0,1,2,3,4]
    # Check lead semantics
    rpt2["needs_processing"]=[]
    if rpt2["corr_t2m"] is not None and abs(rpt2["corr_t2m"]-1.0)<0.01:
        rpt2["needs_processing"].append("corr=1.0 -> forecast and obs are duplicates (historical without models=gfs bug) — rebuild with models=gfs_seamless")
    elif rpt2["corr_t2m"] is not None and rpt2["corr_t2m"]<0.5:
        rpt2["needs_processing"].append(f"corr low {rpt2['corr_t2m']:.2f} — check time alignment")
    if rpt2["lead_is_sequential"] and df2["lead_hours"].max()==239:
        rpt2["needs_processing"].append("lead_hours 0-239 sequential is time-index not forecast lead; recompute as lead = valid - init (0-72)")
    if df2["gfs_t2m_k"].isna().sum()>0: rpt2["needs_processing"].append("missing gfs_t2m_k")
    # Check elevation vs bias relationship
    # Quick: bias = obs - (gfs-273)
    df2["bias_t"] = df2["obs_t2m_c"] - (df2["gfs_t2m_k"]-273.15)
    rpt2["bias_t_mean"]=float(df2["bias_t"].mean())
    rpt2["bias_t_std"]=float(df2["bias_t"].std())
    if abs(rpt2["bias_t_mean"])<0.1 and rpt2["bias_t_std"]<0.2:
        rpt2["needs_processing"].append("bias near zero with low std — dataset may still be duplicated")
    # scaling
    if df2["gfs_t2m_k"].max()>400: rpt2["needs_processing"].append("gfs_t2m_k not normalized (raw K 268-306) — need StandardScaler fit on train only")
    if not rpt2["needs_processing"]:
        rpt2["verdict"]="READY — needs StandardScaler + time-aware split, but columns are correct"
    else:
        rpt2["verdict"]="NEEDS PROCESSING"
    report["M2"]=rpt2
    for k,v in rpt2.items():
        log(f"  M2 {k}: {v}")
except Exception as e:
    import traceback; traceback.print_exc()
    log(f" M2 failed {e}")
    report["M2_error"]=str(e)

# M3
log("\n--- M3 intent_samples.jsonl ---")
try:
    rows=[json.loads(l) for l in open(intent_path)]
    rpt3={}
    rpt3["rows"]=len(rows)
    rpt3["unique_texts"]=len(set(r["text"] for r in rows))
    rpt3["dup_texts"]=len(rows)-len(set(r["text"] for r in rows))
    from collections import Counter
    rpt3["per_decision"]=Counter(r["intent"].get("decision") for r in rows)
    rpt3["per_variables"]=Counter(tuple(r["intent"].get("variables",[])) for r in rows)
    rpt3["per_time"]=Counter(r["intent"].get("time") for r in rows)
    rpt3["per_loc"]=Counter(r["intent"].get("location") for r in rows)
    # Hinglish etc
    rpt3["hinglish"]=sum(1 for r in rows if any(w in r["text"].lower() for w in ["baarish","kal","kya","mausam"]))
    rpt3["pincode"]=sum(1 for r in rows if "pincode" in r["text"].lower() or any(ch.isdigit() and len(r["text"].split())>5 for _ in [1]))
    # imbalance
    rpt3["needs_processing"]=[]
    none = rpt3["per_decision"].get("none",0)
    if none/len(rows)>0.6:
        rpt3["needs_processing"].append(f"severe class imbalance none {none}/{len(rows)} {none/len(rows):.0%} — need upsample pesticide/marine/irrigation or class_weight")
    if rpt3["dup_texts"]>50:
        rpt3["needs_processing"].append(f"dedup {rpt3['dup_texts']} duplicate texts")
    if rpt3["unique_texts"]<1000:
        rpt3["needs_processing"].append("lexical diversity low")
    if not rpt3["needs_processing"]:
        rpt3["verdict"]="READY — minor dedup + class_weight, otherwise trainable"
    else:
        rpt3["verdict"]="NEEDS PROCESSING (rebalance)"
    report["M3"]=rpt3
    for k,v in rpt3.items():
        log(f"  M3 {k}: {v}")
except Exception as e:
    log(f" M3 failed {e}")
    report["M3_error"]=str(e)

# 4. Final plan
log("\n" + "="*70)
log("4. FINALISATION PLAN")
plan = {}
if report.get("M1",{}).get("verdict")=="NEEDS PROCESSING":
    plan["M1"] = "Fix LABEL_MAP vs CSV (expand to 8 or collapse), dedup 100+, fix TMAX(mm), add acc hours 1,3,24, balance classes, then tokenization only"
else:
    plan["M1"] = "Ready"
if report.get("M2",{}).get("verdict")=="NEEDS PROCESSING":
    plan["M2"] = "Recompute lead_hours correctly, verify corr not 1.0, apply StandardScaler (fit train only), time-aware split 2024-01-01:07 -> train, 08:10 -> val, spatial hold-out 5 points, keep elevation scaling"
else:
    plan["M2"] = "Ready + StandardScaler + time split"
if report.get("M3",{}).get("verdict","").startswith("NEEDS"):
    plan["M3"] = "Upsample minority decisions 3x + dedup + class_weight + stratified by template"
else:
    plan["M3"] = "Ready"
plan["env"] = "Force CPU" if force_cpu else "Use P100 if sm_70+, else CPU"
report["final_plan"] = plan
for k,v in plan.items():
    log(f"  {k}: {v}")

# Save
out = pathlib.Path("/kaggle/working/weathergpt_checker")
out.mkdir(parents=True, exist_ok=True)
with open(out / "report.json","w") as f:
    json.dump(report, f, indent=2, default=str)
with open(out / "report.md","w") as f:
    f.write("# WeatherGPT Checker Report\n\n")
    f.write(json.dumps(report, indent=2, default=str))
log(f"\nSaved to {out}/report.json and report.md")
log("Done — ready for finalisation and training after you approve plan.")

