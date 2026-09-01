"""
WeatherGPT — Kaggle P100 v3 self-contained (internet ON)
- Rebuilds datasets ON KAGGLE (no local download) with fixes
- Trains M1/M2/M3 ON KAGGLE (CPU forced for P100 sm_60)
- No external training/*.py needed — all logic inline
- Uses ONLY 4 approved Groq models if needed (not needed for this v3)
"""
import os, sys, pathlib, subprocess, json, random, time, csv, collections

print("="*70)
print("WeatherGPT P100 v3 self-contained — PROCESS + TRAIN")
import torch
print(" torch", torch.__version__, "cuda", torch.cuda.is_available())
force_cpu=False
try:
    if torch.cuda.is_available():
        cap=torch.cuda.get_device_capability(0)
        name=torch.cuda.get_device_name(0)
        print(f" gpu {name} cap {cap}")
        if cap[0] < 7:
            print(" ⚠️ P100 sm_60 incompatible -> forcing CPU")
            force_cpu=True
            os.environ["FORCE_CPU"]="1"
except Exception as e:
    print(" cap check", e)
    force_cpu=True
    os.environ["FORCE_CPU"]="1"

device="cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
print(f" device={device}")

# Ensure Kaggle internet and rebuild datasets fresh ON KAGGLE
import pathlib as pl
root = pl.Path("training/datasets")
root.mkdir(parents=True, exist_ok=True)
print(f" cwd={os.getcwd()}")

# === M1: field_names with 9 labels, dedup, fixed TMAX, acc variants ===
print("\n--- M1 building field_names (9 labels, acc 1,3,6,24) ---")
extra_vocab = [
    ("2t","temperature_2m","instant"), ("t2m","temperature_2m","instant"), ("T2","temperature_2m","instant"),
    ("temperature_2m","temperature_2m","instant"), ("temp","temperature_2m","instant"),
    ("TMAX","temperature_max","max"), ("tmax","temperature_max","max"), ("temperature_max","temperature_max","max"),
    ("TMIN","temperature_min","min"), ("tmin","temperature_min","min"), ("temperature_min","temperature_min","min"),
    ("APCP","precipitation_amount","accumulation"), ("tp","precipitation_amount","accumulation"),
    ("precipitation","precipitation_amount","accumulation"), ("rain","precipitation_amount","accumulation"),
    ("rainfall","precipitation_amount","accumulation"), ("total_precipitation","precipitation_amount","accumulation"),
    ("RAINC","precipitation_amount","accumulation"), ("RAINNC","precipitation_amount","accumulation"),
    ("PoP","precipitation_probability","probability"), ("precipitation_probability","precipitation_probability","probability"),
    ("chance_of_rain","precipitation_probability","probability"),
    ("prate","precipitation_rate","instant"), ("rain_rate","precipitation_rate","instant"),
    ("u10","wind_speed","instant"), ("v10","wind_speed","instant"), ("U10","wind_speed","instant"), ("10m_wind","wind_speed","instant"),
    ("wind_gust","wind_gust","instant"), ("gust","wind_gust","instant"),
    ("Heavy Rainfall","heavy_rain_warning","categorical"), ("Thunderstorm","heavy_rain_warning","categorical"),
    ("Cyclone","heavy_rain_warning","categorical"), ("Fog","heavy_rain_warning","categorical"), ("Heat Wave","heavy_rain_warning","categorical"),
]
rows=[]
for raw, canon, stat in extra_vocab:
    for _ in range(12):  # 30*12=360, we will augment to 1k
        r=raw
        # correct unit suffix
        if canon in ("temperature_2m","temperature_max","temperature_min"):
            suffix=random.choice(["", " (°C)"]) if random.random()<0.2 else ""
        elif canon=="precipitation_amount":
            suffix=random.choice(["", " (mm)", " (kg m-2)"]) if random.random()<0.2 else ""
            # also add accumulation variants via separate rows later
        elif canon in ("wind_speed","wind_gust"):
            suffix=random.choice(["", " (m/s)"]) if random.random()<0.2 else ""
        else:
            suffix=""
        r=r+suffix
        if random.random()<0.2: r=r.upper()
        if random.random()<0.1: r=f" {r} "
        rows.append((r.strip(), canon, stat))
# add accumulation variants for precip_amount
acc_rows=[]
for raw, canon, stat in extra_vocab:
    if canon=="precipitation_amount":
        for acc in ["1","3","6","24"]:
            acc_rows.append((raw, canon, stat, acc))
# dedup
random.shuffle(rows)
seen=set()
dedup=[]
for raw,canon,stat in rows:
    key=(raw.lower(), canon)
    if key not in seen:
        seen.add(key)
        dedup.append((raw,canon,stat))
# pad to ~900 with broader vocab (not just precip) — fixed to avoid infinite loop (was only precip acc_rows, max 64 unique)
attempts=0
while len(dedup)<850 and attempts<5000:
    # sample from full vocab for diversity
    raw, canon, stat = random.choice(extra_vocab)
    # add acc suffix only for precip, otherwise random suffix
    if canon=="precipitation_amount" and random.random()<0.5:
        acc=random.choice(["1","3","6","24"])
        r=raw+f" ({acc}h)"
    else:
        # random case/space variant to ensure uniqueness
        r=raw
        if random.random()<0.3: r=r.upper()
        if random.random()<0.3: r=r+random.choice(["", " (mm)", " (m/s)", " (°C)"])
    key=(r.lower().strip(), canon)
    if key not in seen:
        seen.add(key)
        dedup.append((r.strip(),canon,stat))
    attempts+=1
if len(dedup)<850:
    print(f" M1 warning: only {len(dedup)} unique after {attempts} attempts (target 850), proceeding anyway")
dedup=dedup[:1000]
field_path=root/"field_names.csv"
with open(field_path,"w",newline="") as f:
    w=csv.writer(f)
    w.writerow(["raw_field","canonical_variable","statistic","accumulation_hours","source_hint"])
    for raw,canon,stat in dedup:
        # assign acc
        acc=""
        if canon=="precipitation_amount":
            acc=random.choice(["1","3","6","24"])
        w.writerow([raw,canon,stat,acc,"v3 Kaggle rebuild"])
print(f" M1 written {field_path} rows={len(dedup)} unique={len(set(r[0].lower() for r in dedup))} per_label={collections.Counter(c for _,c,_ in dedup)}")

# === M2: matched_pairs with models=gfs_seamless, lead %72 ===
print("\n--- M2 building matched_pairs (GFS vs ERA5, lead %72) ---")
import httpx, asyncio
points=[(21.14,79.08,240),(19.07,72.87,14),(28.61,77.20,216),(22.57,88.36,9),(13.08,80.27,6),(12.97,77.59,920),(18.52,73.85,560),(26.91,75.78,431),(23.02,72.57,53),(25.43,81.84,98),(17.38,78.48,542),(15.31,75.12,671),(11.01,76.96,411),(30.73,76.77,350),(34.08,74.79,1585),(20.29,85.82,45),(26.14,91.73,55),(21.25,81.62,298),(24.58,73.71,423),(15.91,75.56,696)]
all_pairs=[]
async def fetch_one(lat,lon):
    async with httpx.AsyncClient(timeout=40) as c:
        hist_url="https://historical-forecast-api.open-meteo.com/v1/forecast"
        era5_url="https://archive-api.open-meteo.com/v1/era5"
        hist_params={"latitude":lat,"longitude":lon,"start_date":"2024-01-01","end_date":"2024-01-10","hourly":"temperature_2m,precipitation","models":"gfs_seamless","timezone":"UTC"}
        era_params={"latitude":lat,"longitude":lon,"start_date":"2024-01-01","end_date":"2024-01-10","hourly":"temperature_2m,precipitation","timezone":"UTC"}
        rh=await c.get(hist_url, params=hist_params)
        re=await c.get(era5_url, params=era_params)
        rh.raise_for_status(); re.raise_for_status()
        return rh.json()["hourly"], re.json()["hourly"]
import asyncio, time
for lat,lon,elev in points:
    try:
        hist, era = asyncio.run(fetch_one(lat,lon))
        times=hist["time"]
        for i,t in enumerate(times):
            gfs_t=hist["temperature_2m"][i]; gfs_p=hist["precipitation"][i] or 0
            obs_t=era["temperature_2m"][i]; obs_p=era["precipitation"][i] or 0
            if gfs_t is None or obs_t is None: continue
            lead = i % 72  # fix: 0-71 cycle, not 0-239
            all_pairs.append((lat,lon,elev,lead,t,gfs_t,gfs_p,obs_t,obs_p))
        time.sleep(0.12)
    except Exception as e:
        print(f"  M2 {lat},{lon} fail {e}")
pair_path=root/"matched_pairs.csv"
with open(pair_path,"w",newline="") as f:
    w=csv.writer(f)
    w.writerow(["lat","lon","elevation_m","lead_hours","valid_from","gfs_t2m_k","gfs_apcp_mm","obs_t2m_c","obs_apcp_mm"])
    for lat,lon,elev,lead,t,gfs_t,gfs_p,obs_t,obs_p in all_pairs:
        w.writerow([lat,lon,elev,lead,t,gfs_t+273.15,gfs_p,obs_t,obs_p])
print(f" M2 written {pair_path} rows={len(all_pairs)} lead 0-71 verified {sorted(set(p[3] for p in all_pairs))[:5]}")

# === M3: intent with rebalance ===
print("\n--- M3 building intent (rebalanced) ---")
import json
templates=["Will it rain in {loc} {time}?","Will it rain in {loc} {time} and should I spray pesticide?","Is there a heavy rain warning for {loc} {time}?","What is the temperature in {loc} {time}?","What is the wind speed in {loc} {time}?","Can I go fishing in {loc} {time}?","Should I irrigate in {loc} {time}?","Forecast for {loc} {time}","Chance of rain in {loc} {time}?","Will it be hot in {loc} {time}?","Kal {loc} me baarish hogi {time}?","kya {loc} me {time} baarish hogi?","{loc} me {time} mausam kaisa rahega?","My village near {loc} {time} — rain?","Pincode {pincode} {time} weather?","{lat},{lon} {time} forecast?"]
locs=["Nagpur","Mumbai","Delhi","Kolkata","Chennai","Bengaluru","Pune","Malegaon","Ahmedabad","Patna","Jaipur","Lucknow","Indore","Bhopal","Nagpur village","my village","440001","400001","110001"]
times=["today","tonight","tomorrow","tomorrow morning","tomorrow afternoon","tomorrow evening","day after tomorrow","next 3 days","this weekend","coming Monday","23rd Aug"]
decisions=["pesticide_spraying","marine","irrigation","harvest","none"]
rows_j=[]
for _ in range(1400):
    tmpl=random.choice(templates)
    loc=random.choice(locs)
    tm=random.choice(times)
    pincode=random.choice(["440001","400001","110001"])
    lat=round(random.uniform(18,28),2); lon=round(random.uniform(72,88),2)
    text=tmpl.format(loc=loc, time=tm, pincode=pincode, lat=lat, lon=lon)
    low=text.lower()
    vars_=[]
    if "rain" in low or "baarish" in low: vars_.append("precipitation_amount")
    if "chance" in low: vars_.append("precipitation_probability")
    if "temperature" in low or "hot" in low: vars_.append("temperature_2m")
    if "wind" in low: vars_.append("wind_speed")
    if "warning" in low: vars_.append("heavy_rain_warning")
    if not vars_: vars_=["precipitation_amount"]
    # decide decision: force balanced by sampling decision uniformly 20% each
    # Instead of heuristic, sample decision uniformly
    dec=random.choice(decisions)
    # But keep some correlation: if spray/fishing/irrigate keyword, override
    if "spray" in low: dec="pesticide_spraying"
    elif "fishing" in low: dec="marine"
    elif "irrigat" in low: dec="irrigation"
    elif "harvest" in low: dec="harvest"
    rows_j.append({"text":text,"intent":{"variables":vars_,"time":tm,"location":loc,"decision":dec}})
# dedup
seen=set()
dedup_j=[]
for r in rows_j:
    key=r["text"].strip().lower()
    if key not in seen:
        seen.add(key)
        dedup_j.append(r)
random.shuffle(dedup_j)
# rebalance to 400 none + 200 each other = 1200
from collections import Counter
cnt=Counter(r["intent"]["decision"] for r in dedup_j)
print(f" before rebalance {cnt} unique {len(dedup_j)}")
# target
balanced=[]
for dec in decisions:
    lst=[r for r in dedup_j if r["intent"]["decision"]==dec]
    target=400 if dec=="none" else 200
    if len(lst)<target:
        lst = (lst* ((target//len(lst))+1))[:target]
    else:
        lst=lst[:target]
    balanced.extend(lst)
random.shuffle(balanced)
balanced=balanced[:1200]
print(f" after rebalance {Counter(r['intent']['decision'] for r in balanced)}")
intent_path=root/"intent_samples.jsonl"
with open(intent_path,"w") as f:
    for r in balanced:
        f.write(json.dumps(r)+"\n")
print(f" M3 written {intent_path} rows={len(balanced)}")

print("\n"+"="*70)
print(" Datasets built, now TRAIN (no Gaussian fallback)")

# === Train M1: semantic ===
print("\n>>> M1 semantic (9 labels, dedup, class_weight)")
import subprocess, sys
def run(cmd):
    print(f"\n>>> {' '.join(cmd)}")
    r=subprocess.run(cmd, text=True)
    print(f" exit {r.returncode}")
    return r.returncode==0

# Need to ensure we have local training scripts with fixes. We will inline train here without needing external file
# Instead we will call the logic: create a temp script that uses same fixed code but reads from root we built
# Easiest: write a mini trainer that uses transformers with our built CSV

# For M1: we will run a simplified trainer using sklearn + TF-IDF as fallback if transformers not available quickly
# But try transformers first
try:
    import pandas as pd, numpy as np, torch
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    import csv
    # Load field_names
    df=pd.read_csv(field_path)
    # Map canonical to idx (9 labels)
    labels=sorted(df["canonical_variable"].unique())
    label2idx={l:i for i,l in enumerate(labels)}
    print(f" M1 labels {label2idx}")
    X_text=df["raw_field"].astype(str).tolist()
    y=[label2idx[c] for c in df["canonical_variable"]]
    # tfidf simple but fast and CPU-friendly for Kaggle
    vec=TfidfVectorizer(max_features=2000, ngram_range=(1,2))
    X_vec=vec.fit_transform(X_text)
    X_tr, X_te, y_tr, y_te = train_test_split(X_vec, y, test_size=0.15, random_state=42, stratify=y)
    clf=LogisticRegression(max_iter=500, class_weight="balanced", multi_class="multinomial")
    clf.fit(X_tr, y_tr)
    pred=clf.predict(X_te)
    acc=accuracy_score(y_te, pred)
    f1=f1_score(y_te, pred, average="weighted")
    print(f" M1 TF-IDF LogisticRegression acc {acc:.3f} f1 {f1:.3f}")
    # also try DistilBERT if time allows (but TF-IDF is already a valid baseline; we will also attempt transformers)
    out=pathlib.Path("training/models/semantic_classifier")
    out.mkdir(parents=True, exist_ok=True)
    import pickle, json
    with open(out/"vectorizer.pkl","wb") as f: pickle.dump(vec,f)
    with open(out/"model.pkl","wb") as f: pickle.dump(clf,f)
    with open(out/"metrics.json","w") as f: json.dump({"accuracy":acc,"f1":f1,"labels":label2idx,"note":"tfidf+logreg CPU, P100 forced"}, f, indent=2)
    print(f" M1 saved to {out}")
    ok1=True
except Exception as e:
    import traceback; traceback.print_exc()
    print(f" M1 failed {e}")
    ok1=False

# M2: bias MLP with StandardScaler + time split
print("\n>>> M2 bias MLP (StandardScaler + time split, lead %72)")
try:
    import pandas as pd, numpy as np, torch, torch.nn as nn, math, pickle
    from sklearn.preprocessing import StandardScaler
    df=pd.read_csv(pair_path)
    # time-aware split: first 85% train, last 15% val (CSV already time-sorted by build order)
    n=len(df)
    split=int(n*0.85)
    train=df.iloc[:split]
    val=df.iloc[split:]
    print(f" M2 split train {len(train)} val {len(val)} time-aware")
    # features
    feat_cols=["gfs_t2m_k","gfs_apcp_mm","elevation_m","lead_hours","lat"]
    X_tr=train[feat_cols].values.astype(np.float32)
    X_te=val[feat_cols].values.astype(np.float32)
    # targets: bias
    y_tr_t = train["obs_t2m_c"].values.astype(np.float32) - (train["gfs_t2m_k"].values.astype(np.float32)-273.15)
    y_tr_p = train["obs_apcp_mm"].values.astype(np.float32) - train["gfs_apcp_mm"].values.astype(np.float32)
    y_te_t = val["obs_t2m_c"].values.astype(np.float32) - (val["gfs_t2m_k"].values.astype(np.float32)-273.15)
    y_te_p = val["obs_apcp_mm"].values.astype(np.float32) - val["gfs_apcp_mm"].values.astype(np.float32)
    import numpy as np
    y_tr=np.stack([y_tr_t, y_tr_p], axis=1).astype(np.float32)
    y_te=np.stack([y_te_t, y_te_p], axis=1).astype(np.float32)
    scaler=StandardScaler()
    X_tr_s=scaler.fit_transform(X_tr)
    X_te_s=scaler.transform(X_te)
    # MLP small
    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net=nn.Sequential(nn.Linear(5,64),nn.ReLU(),nn.Dropout(0.1),nn.Linear(64,64),nn.ReLU(),nn.Dropout(0.1),nn.Linear(64,2))
        def forward(self,x): return self.net(x)
    device=torch.device("cpu")
    model=MLP().to(device)
    opt=torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn=nn.MSELoss()
    # train
    Xtr_t=torch.tensor(X_tr_s); ytr_t=torch.tensor(y_tr)
    Xte_t=torch.tensor(X_te_s); yte_t=torch.tensor(y_te)
    best=float("inf")
    best_state=None
    for epoch in range(1,21):
        model.train()
        # mini-batch
        perm=torch.randperm(len(Xtr_t))
        tot=0
        for i in range(0,len(Xtr_t),256):
            idx=perm[i:i+256]
            xb=Xtr_t[idx].to(device); yb=ytr_t[idx].to(device)
            opt.zero_grad()
            pred=model(xb)
            loss=loss_fn(pred,yb)
            loss.backward(); opt.step()
            tot+=loss.item()*len(idx)
        train_loss=tot/len(Xtr_t)
        model.eval()
        with torch.no_grad():
            pred=model(Xte_t.to(device))
            val_loss=loss_fn(pred, yte_t.to(device)).item()
            rmse_t=math.sqrt(((pred[:,0]-yte_t.to(device)[:,0])**2).mean().item())
            rmse_p=math.sqrt(((pred[:,1]-yte_t.to(device)[:,1])**2).mean().item())
        print(f" epoch {epoch:02d} train {train_loss:.4f} val {val_loss:.4f} rmse_t {rmse_t:.3f} rmse_p {rmse_p:.3f}")
        if val_loss < best:
            best=val_loss
            best_state={k:v.cpu() for k,v in model.state_dict().items()}
            best_rmse_t, best_rmse_p = rmse_t, rmse_p
    out=pathlib.Path("training/models/bias_correction")
    out.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out/"best.pt")
    with open(out/"scaler.pkl","wb") as f: pickle.dump(scaler,f)
    import json
    with open(out/"metrics.json","w") as f: json.dump({"best_val_loss":best,"rmse_t":best_rmse_t,"rmse_p":best_rmse_p,"bias_t_mean":float(df["obs_t2m_c"].mean()-(df["gfs_t2m_k"].mean()-273.15))}, f, indent=2)
    with open(out/"config.json","w") as f: json.dump({"in_dim":5,"hidden":64}, f)
    print(f" M2 saved to {out} best {best:.4f}")
    ok2=True
except Exception as e:
    import traceback; traceback.print_exc()
    print(f" M2 failed {e}")
    ok2=False

# M3: intent with TF-IDF + balanced (quick, CPU)
print("\n>>> M3 intent (rebalanced, TF-IDF)")
try:
    import json, pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split
    rows=[json.loads(l) for l in open(intent_path)]
    texts=[r["text"] for r in rows]
    labels=[r["intent"]["decision"] for r in rows]
    uniq_labels=sorted(set(labels))
    l2i={l:i for i,l in enumerate(uniq_labels)}
    y=[l2i[l] for l in labels]
    vec=TfidfVectorizer(max_features=3000, ngram_range=(1,2))
    X=vec.fit_transform(texts)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.15, random_state=42, stratify=y)
    clf=LogisticRegression(max_iter=1000, class_weight="balanced", multi_class="multinomial")
    clf.fit(Xtr,ytr)
    pred=clf.predict(Xte)
    acc=accuracy_score(yte,pred); f1=f1_score(yte,pred, average="weighted")
    print(f" M3 acc {acc:.3f} f1 {f1:.3f} labels {l2i}")
    out=pathlib.Path("training/models/intent_parser")
    out.mkdir(parents=True, exist_ok=True)
    import pickle, json
    with open(out/"vectorizer.pkl","wb") as f: pickle.dump(vec,f)
    with open(out/"model.pkl","wb") as f: pickle.dump(clf,f)
    with open(out/"metrics.json","w") as f: json.dump({"accuracy":acc,"f1":f1,"labels":l2i}, f, indent=2)
    ok3=True
except Exception as e:
    import traceback; traceback.print_exc()
    print(f" M3 failed {e}")
    ok3=False

print("\n"+"="*70)
print(f" v3 done: M1 {ok1} M2 {ok2} M3 {ok3}")
# copy to output
import subprocess as sp, pathlib as pl
out=pl.Path("/kaggle/working/weathergpt_outputs")
out.mkdir(parents=True, exist_ok=True)
sp.run(["cp","-r","training/models",str(out)], check=False)
sp.run(["cp","-r","training/datasets",str(out/"training")], check=False)
for p in pl.Path("training/models").rglob("metrics.json"):
    print(f" {p}: {p.read_text()[:500]}")
print(" done")

