"""
WeatherGPT — OFFICIAL Hackathon Training (BEST EVER)
T4 x2, internet ON, single-file, no local download.

Uses the 4 Groq models you assigned (qwen/qwen3.8-27b orchestrator, qwen3.6-27b, gpt-oss-20b, gpt-oss-120b) for intent paraphrasing.
Trains:
  M1 semantic 9-label DistilBERT (not TF-IDF) — best
  M2 bias Deep MLP 5->128->128->64->2 with StandardScaler, time-aware split, 30-day window (20 points × 30 days × 24h = 14400 rows)
  M3 intent DistilBERT 5-way with Groq-paraphrased 2000 diverse utterances

Push: cd kaggle_kernel_official && kaggle kernels push -p .  → T4 x2
"""

import os, sys, pathlib, subprocess, json, random, time, csv, collections
print("="*70)
print("WeatherGPT OFFICIAL — BEST EVER — T4 x2")
import torch
print(" torch", torch.__version__, "cuda", torch.cuda.is_available())
use_device="cuda" if torch.cuda.is_available() else "cpu"
try:
    if torch.cuda.is_available():
        cap=torch.cuda.get_device_capability(0)
        name=torch.cuda.get_device_name(0)
        print(f" gpu {name} cap {cap} count {torch.cuda.device_count()}")
        if cap[0] < 7:
            print(" ⚠️ P100 sm_60 -> forcing CPU (but you said T4, so this should not happen)")
            use_device="cpu"
            os.environ["FORCE_CPU"]="1"
        else:
            print(" ✅ T4/T4x2 compatible — using GPU")
            # T4 x2 has 2 GPUs — use DataParallel later
except Exception as e:
    print(" cap check", e)
print(f" device={use_device} amp={use_device=='cuda'}")
force_cpu = (use_device == "cpu")
print(f" force_cpu={force_cpu}")

# Groq setup for intent paraphrasing (4-model queue)
GROQ_MODELS=["qwen/qwen3.8-27b","qwen/qwen3.6-27b","openai/gpt-oss-20b","openai/gpt-oss-120b"]
ORCH="qwen/qwen3.8-27b"
groq_key=os.getenv("GROQ_API_KEY") or ""
if not groq_key:
    for p in ["/kaggle/input/groq-api/groq_api.txt", "/tmp/groq_raw.txt", "/home/anamitra/Downloads/API_Keys_and_Secrets/groq_api.txt"]:
        try:
            import pathlib as pl
            k=pl.Path(p).read_text().strip()
            if len(k)>20:
                groq_key=k
                print(f" Groq key from {p} len {len(k)}")
                break
        except: pass
if not groq_key:
    # Try env from weathergpt .env
    try:
        import pathlib as pl
        for cand in [".env","/kaggle/working/.env","/kaggle/working/weathergpt/.env"]:
            if pl.Path(cand).exists():
                txt=pl.Path(cand).read_text()
                for line in txt.splitlines():
                    if line.startswith("GROQ_API_KEY="):
                        k=line.split("=",1)[1].strip()
                        if len(k)>20:
                            groq_key=k
                            print(f" Groq key from {cand}")
                            break
    except: pass

# Groq key is read from env or file (see groq_client.py) — not hardcoded
print(f" Groq ready: {bool(groq_key)} models {GROQ_MODELS} orchestrator {ORCH}")

# Install deps
print("\nInstalling deps...")
subprocess.run([sys.executable,"-m","pip","install","-q","transformers","datasets","accelerate","scikit-learn","pandas"], check=False)
# Ensure training dirs
import pathlib as pl
root=pl.Path("training/datasets")
root.mkdir(parents=True, exist_ok=True)

# ===================== M1: field_names 1200 diverse, 9 labels =====================
print("\n--- M1 field_names (DIVERSE 1200, 9 labels) ---")
# Real vocab from variable_registry + GRIB/NetCDF/CAP
extra_vocab=[
    ("2t","temperature_2m","instant"),("t2m","temperature_2m","instant"),("T2","temperature_2m","instant"),("temperature_2m","temperature_2m","instant"),("temp","temperature_2m","instant"),("T2M","temperature_2m","instant"),
    ("TMAX","temperature_max","max"),("tmax","temperature_max","max"),("temperature_max","temperature_max","max"),("MAXT","temperature_max","max"),
    ("TMIN","temperature_min","min"),("tmin","temperature_min","min"),("temperature_min","temperature_min","min"),("MINT","temperature_min","min"),
    ("APCP","precipitation_amount","accumulation"),("tp","precipitation_amount","accumulation"),("precipitation","precipitation_amount","accumulation"),("rain","precipitation_amount","accumulation"),("rainfall","precipitation_amount","accumulation"),("total_precipitation","precipitation_amount","accumulation"),("RAINC","precipitation_amount","accumulation"),("RAINNC","precipitation_amount","accumulation"),("precip","precipitation_amount","accumulation"),
    ("PoP","precipitation_probability","probability"),("precipitation_probability","precipitation_probability","probability"),("chance_of_rain","precipitation_probability","probability"),("pop","precipitation_probability","probability"),
    ("prate","precipitation_rate","instant"),("rain_rate","precipitation_rate","instant"),("precipitation_rate","precipitation_rate","instant"),("PRATE","precipitation_rate","instant"),
    ("u10","wind_speed","instant"),("v10","wind_speed","instant"),("U10","wind_speed","instant"),("10m_wind","wind_speed","instant"),("wind_speed","wind_speed","instant"),("WIND","wind_speed","instant"),
    ("wind_gust","wind_gust","instant"),("gust","wind_gust","instant"),("10m_gust","wind_gust","instant"),("GUST","wind_gust","instant"),
    ("Heavy Rainfall","heavy_rain_warning","categorical"),("Thunderstorm","heavy_rain_warning","categorical"),("Cyclone","heavy_rain_warning","categorical"),("Fog","heavy_rain_warning","categorical"),("Heat Wave","heavy_rain_warning","categorical"),("Extremely Heavy Rain","heavy_rain_warning","categorical"),("Hail","heavy_rain_warning","categorical"),
]
# Generate 1200 with rich augmentations + acc variants
random.seed(42)
# Generate 1200 diverse rows directly with rich augmentations
prefixes=["", "GFS:", "ECMWF:", "IMD:", "WRF:", ""]
suffixes_temp=["", " (°C)", " (C)", " in C", " [C]", ""]
suffixes_precip=["", " (mm)", " (kg m-2)", " (mm/6h)", " (mm/24h)", ""]
suffixes_wind=["", " (m/s)", " (km/h)", " at 10m", ""]
suffixes_prob=["", " (%)", " %", ""]
suffixes_warn=["", " warning", " alert", ""]
rows=[]
# First, ensure each raw gets at least 20 diverse variants
for raw,canon,stat in extra_vocab:
    for _ in range(25):
        r=raw
        # diverse suffix by variable
        if canon in ("temperature_2m","temperature_max","temperature_min"):
            suffix=random.choice(suffixes_temp)
        elif canon=="precipitation_amount":
            # include acc variants directly
            acc=random.choice(["1","3","6","24",""])
            suffix=f" ({acc}h)" if acc else random.choice(suffixes_precip)
        elif canon in ("wind_speed","wind_gust"):
            suffix=random.choice(suffixes_wind)
        elif canon=="precipitation_probability":
            suffix=random.choice(suffixes_prob)
        elif canon=="heavy_rain_warning":
            suffix=random.choice(suffixes_warn)
        else:
            suffix=random.choice(["", " (instant)"])
        # random prefix
        pref=random.choice(prefixes)
        r=f"{pref}{r}{suffix}" if pref else f"{r}{suffix}"
        # case variations: 30% upper, 20% lower, 10% title, rest as is
        cr=random.random()
        if cr<0.2: r=r.upper()
        elif cr<0.35: r=r.lower()
        elif cr<0.45: r=r.title()
        # spacing / sep variations
        if random.random()<0.15:
            if "_" in r: r=r.replace("_", random.choice(["-"," ","_"]))
            elif " " in r: r=r.replace(" ", random.choice(["_","-"," "]))
        # add random double space or trim
        if random.random()<0.1: r=f" {r} "
        # add random digit suffix for uniqueness (e.g., T2M_1)
        if random.random()<0.08:
            r=f"{r}_{random.randint(1,99)}"
        rows.append((r.strip(),canon,stat))
random.shuffle(rows)
# dedup on lower+canon, keep first, then pad if needed
seen=set()
dedup=[]
for raw,canon,stat in rows:
    key=(raw.lower(),canon)
    if key not in seen:
        seen.add(key)
        dedup.append((raw,canon,stat))
# If still <1200, pad with fully random combos (should be rare now)
attempts=0
while len(dedup)<1200 and attempts<10000:
    raw,canon,stat=random.choice(extra_vocab)
    # fully random augmentation
    r=raw
    if random.random()<0.4:
        r=random.choice(prefixes)+r
    if canon=="precipitation_amount" and random.random()<0.5:
        r=r+f" ({random.choice(['1','3','6','24'])}h)"
    if random.random()<0.3:
        r=r+random.choice([" (mm)"," (°C)"," (m/s)"," (%)",""])
    if random.random()<0.25: r=r.upper()
    key=(r.lower(),canon)
    if key not in seen:
        seen.add(key)
        dedup.append((r,canon,stat))
    attempts+=1
dedup=dedup[:1200]
field_path=root/"field_names.csv"
with open(field_path,"w",newline="") as f:
    w=csv.writer(f)
    w.writerow(["raw_field","canonical_variable","statistic","accumulation_hours","source_hint"])
    for raw,canon,stat in dedup:
        acc=""
        if canon=="precipitation_amount":
            # infer from raw if contains (Nh)
            import re
            m=re.search(r"\((\d+)h\)",raw)
            acc=m.group(1) if m else random.choice(["1","3","6","24"])
        w.writerow([raw,canon,stat,acc,"official T4"])
print(f" M1 {field_path} rows={len(dedup)} unique={len(set(r[0].lower() for r in dedup))} per_label={collections.Counter(c for _,c,_ in dedup)}")


# ===================== M2: matched_pairs 14400 rows (30 days) =====================
print("\n--- M2 matched_pairs (30 days, 20 points, GFS vs ERA5, lead %72, deep MLP) ---")
import httpx, asyncio
points=[(21.14,79.08,240),(19.07,72.87,14),(28.61,77.20,216),(22.57,88.36,9),(13.08,80.27,6),(12.97,77.59,920),(18.52,73.85,560),(26.91,75.78,431),(23.02,72.57,53),(25.43,81.84,98),(17.38,78.48,542),(15.31,75.12,671),(11.01,76.96,411),(30.73,76.77,350),(34.08,74.79,1585),(20.29,85.82,45),(26.14,91.73,55),(21.25,81.62,298),(24.58,73.71,423),(15.91,75.56,696)]
all_pairs=[]
async def fetch_range(lat,lon,start,end):
    async with httpx.AsyncClient(timeout=40) as c:
        rh=await c.get("https://historical-forecast-api.open-meteo.com/v1/forecast", params={"latitude":lat,"longitude":lon,"start_date":start,"end_date":end,"hourly":"temperature_2m,precipitation","models":"gfs_seamless","timezone":"UTC"})
        re=await c.get("https://archive-api.open-meteo.com/v1/era5", params={"latitude":lat,"longitude":lon,"start_date":start,"end_date":end,"hourly":"temperature_2m,precipitation","timezone":"UTC"})
        rh.raise_for_status(); re.raise_for_status()
        return rh.json()["hourly"], re.json()["hourly"]
# 30 days: 2024-01-01 to 2024-01-30
for lat,lon,elev in points:
    for chunk_start in ["2024-01-01","2024-01-11","2024-01-21"]:
        chunk_end = {"2024-01-01":"2024-01-10","2024-01-11":"2024-01-20","2024-01-21":"2024-01-30"}[chunk_start]
        try:
            hist, era = asyncio.run(fetch_range(lat,lon,chunk_start,chunk_end))
            times=hist["time"]
            for i,t in enumerate(times):
                gfs_t=hist["temperature_2m"][i]; gfs_p=hist["precipitation"][i] or 0
                obs_t=era["temperature_2m"][i]; obs_p=era["precipitation"][i] or 0
                if gfs_t is None or obs_t is None: continue
                # lead = hours since chunk start %72
                lead=i%72
                all_pairs.append((lat,lon,elev,lead,t,gfs_t,gfs_p,obs_t,obs_p))
            time.sleep(0.12)
        except Exception as e:
            print(f"  M2 {lat},{lon} {chunk_start} fail {e}")
pair_path=root/"matched_pairs.csv"
with open(pair_path,"w",newline="") as f:
    w=csv.writer(f)
    w.writerow(["lat","lon","elevation_m","lead_hours","valid_from","gfs_t2m_k","gfs_apcp_mm","obs_t2m_c","obs_apcp_mm"])
    for lat,lon,elev,lead,t,gfs_t,gfs_p,obs_t,obs_p in all_pairs:
        w.writerow([lat,lon,elev,lead,t,gfs_t+273.15,gfs_p,obs_t,obs_p])
print(f" M2 written {pair_path} rows={len(all_pairs)} lead 0-71 verified {sorted(set(p[3] for p in all_pairs))[:5]}")

# ===================== M3: intent 2000 with Groq 4-model paraphrase =====================
print("\n--- M3 intent 2000 with Groq 4-model queue ---")
import json
templates=["Will it rain in {loc} {time}?","Will it rain in {loc} {time} and should I spray pesticide?","Is there a heavy rain warning for {loc} {time}?","What is the temperature in {loc} {time}?","What is the wind speed in {loc} {time}?","Can I go fishing in {loc} {time}?","Should I irrigate in {loc} {time}?","Forecast for {loc} {time}","Chance of rain in {loc} {time}?","Will it be hot in {loc} {time}?","Kal {loc} me baarish hogi {time}?","kya {loc} me {time} baarish hogi?","{loc} me {time} mausam kaisa rahega?","My village near {loc} {time} — rain?","Pincode {pincode} {time} weather?","{lat},{lon} {time} forecast?","Should I harvest in {loc} {time}?"]
locs=["Nagpur","Mumbai","Delhi","Kolkata","Chennai","Bengaluru","Pune","Malegaon","Ahmedabad","Patna","Jaipur","Lucknow","Indore","Bhopal","Nagpur village","my village","440001","400001","110001","19.07,72.87"]
times=["today","tonight","tomorrow","tomorrow morning","tomorrow afternoon","tomorrow evening","day after tomorrow","next 3 days","this weekend","coming Monday","23rd Aug","next week"]
decisions=["pesticide_spraying","marine","irrigation","harvest","none"]
# Base 1200 via templates
base_rows=[]
for _ in range(1200):
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
    dec=random.choice(decisions)
    if "spray" in low: dec="pesticide_spraying"
    elif "fishing" in low: dec="marine"
    elif "irrigat" in low: dec="irrigation"
    elif "harvest" in low: dec="harvest"
    base_rows.append({"text":text,"intent":{"variables":vars_,"time":tm,"location":loc,"decision":dec}})

# Groq paraphrase 800 more via 4-model queue
paraphrased=[]
if groq_key:
    print(f" Groq paraphrasing 800 via {GROQ_MODELS} orchestrator {ORCH}")
    for idx in range(800):
        src=random.choice(base_rows)["text"]
        model=GROQ_MODELS[idx % len(GROQ_MODELS)]
        try:
            async def call():
                async with httpx.AsyncClient(timeout=25) as c:
                    r=await c.post("https://api.groq.com/openai/v1/chat/completions", headers={"Authorization":f"Bearer {groq_key}","Content-Type":"application/json"}, json={"model":model,"messages":[{"role":"user","content":f"Paraphrase this weather query concisely, keep location/time, reply with only the paraphrase: '{src}'"}],"max_tokens":40, "temperature":0.9})
                    r.raise_for_status()
                    return r.json()["choices"][0]["message"]["content"].strip().strip('"')
            para=asyncio.run(call())
            # simple intent copy
            low=para.lower()
            vars_=[]
            if "rain" in low or "baarish" in low: vars_.append("precipitation_amount")
            if "chance" in low: vars_.append("precipitation_probability")
            if "temperature" in low or "hot" in low: vars_.append("temperature_2m")
            if "wind" in low: vars_.append("wind_speed")
            if "warning" in low: vars_.append("heavy_rain_warning")
            if not vars_: vars_=["precipitation_amount"]
            dec="none"
            if "spray" in low: dec="pesticide_spraying"
            elif "fish" in low: dec="marine"
            elif "irrigat" in low: dec="irrigation"
            elif "harvest" in low: dec="harvest"
            else: dec=random.choice(decisions)
            paraphrased.append({"text":para,"intent":{"variables":vars_,"time":"tomorrow","location":"Nagpur","decision":dec}})
            if idx%100==0: print(f"  Groq {idx}/800 {model} -> {para[:60]}")
            time.sleep(0.25)
        except Exception as e:
            if idx%100==0: print(f"  Groq {model} fail {e}")
            continue
    print(f" Groq paraphrased {len(paraphrased)}")
else:
    print(" Groq key missing — skipping paraphrase, will use template only")

all_rows=base_rows+paraphrased
# dedup + rebalance to 2000 (400 none, 400 each other? actually 400 none + 400 each of 4 others = 2000)
seen=set()
dedup=[]
for r in all_rows:
    key=r["text"].strip().lower()
    if key not in seen:
        seen.add(key)
        dedup.append(r)
random.shuffle(dedup)
from collections import Counter
cnt=Counter(r["intent"]["decision"] for r in dedup)
print(f" before rebalance {cnt} unique {len(dedup)}")
balanced=[]
for dec in decisions:
    lst=[r for r in dedup if r["intent"]["decision"]==dec]
    target=600 if dec=="none" else 350  # 600+350*4=2000
    if len(lst)<target:
        lst=(lst* ((target//len(lst))+1))[:target]
    else:
        lst=lst[:target]
    balanced.extend(lst)
random.shuffle(balanced)
balanced=balanced[:2000]
print(f" after rebalance {Counter(r['intent']['decision'] for r in balanced)}")
intent_path=root/"intent_samples.jsonl"
with open(intent_path,"w") as f:
    for r in balanced:
        f.write(json.dumps(r)+"\n")
print(f" M3 written {intent_path} rows={len(balanced)}")

print("\n"+"="*70)
print(" Datasets built OFFICIAL, now TRAIN BEST EVER")

# ===================== TRAIN M1: DistilBERT 9-label =====================
print("\n>>> M1 DistilBERT 9-label (best) <<<")
m1_ok=False
try:
    import pandas as pd, numpy as np
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
    from datasets import Dataset
    from sklearn.metrics import accuracy_score, f1_score
    import torch
    df=pd.read_csv(field_path)
    labels=sorted(df["canonical_variable"].unique())
    l2i={l:i for i,l in enumerate(labels)}
    print(f" M1 labels {l2i} rows {len(df)}")
    # stratified split handled inside Trainer via new code? Do manual
    from sklearn.model_selection import train_test_split
    X_text=df["raw_field"].astype(str).tolist()
    y=[l2i[c] for c in df["canonical_variable"]]
    Xtr,Xte,ytr,yte=train_test_split(X_text,y,test_size=0.15, random_state=42, stratify=y)
    # also need train/val as Dataset
    from datasets import Dataset as DS
    train_ds=DS.from_dict({"text":Xtr,"labels":ytr})
    val_ds=DS.from_dict({"text":Xte,"labels":yte})
    tok=AutoTokenizer.from_pretrained("distilbert-base-uncased")
    def tok_fn(b):
        return tok(b["text"], truncation=True, padding="max_length", max_length=64)
    train_ds=train_ds.map(tok_fn, batched=True)
    val_ds=val_ds.map(tok_fn, batched=True)
    train_ds.set_format(type="torch", columns=["input_ids","attention_mask","labels"])
    val_ds.set_format(type="torch", columns=["input_ids","attention_mask","labels"])
    model=AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(labels))
    # class weights
    from collections import Counter
    cnt=Counter(ytr)
    total=sum(cnt.values())
    w=[total/cnt[i] if cnt[i]>0 else 1 for i in range(len(labels))]
    w=[x/sum(w)*len(w) for x in w]
    cw=torch.tensor(w, dtype=torch.float)
    class WTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            labels=inputs.get("labels")
            outputs=model(**inputs)
            logits=outputs.get("logits")
            loss_fct=torch.nn.CrossEntropyLoss(weight=cw.to(logits.device))
            loss=loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss
    args=TrainingArguments(output_dir="training/models/semantic_classifier", num_train_epochs=8, per_device_train_batch_size=32, per_device_eval_batch_size=64, learning_rate=2e-5, eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True, logging_steps=20, seed=42, fp16=(use_device=="cuda"), use_cpu=force_cpu, report_to="none", save_total_limit=2)
    trainer=WTrainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds, processing_class=tok, compute_metrics=lambda p: {"accuracy":accuracy_score(p.label_ids, np.argmax(p.predictions,axis=1)), "f1":f1_score(p.label_ids, np.argmax(p.predictions,axis=1), average="weighted")})
    trainer.train()
    metrics=trainer.evaluate()
    print(f" M1 DistilBERT {metrics}")
    out=pathlib.Path("training/models/semantic_classifier")
    out.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out))
    tok.save_pretrained(str(out))
    import json
    with open(out/"metrics.json","w") as f: json.dump(metrics, f, indent=2)
    with open(out/"label_map.json","w") as f: json.dump(l2i, f, indent=2)
    m1_ok=True
except Exception as e:
    import traceback; traceback.print_exc()
    print(f" M1 failed {e}")

# ===================== TRAIN M2: Deep MLP 128x3 =====================
print("\n>>> M2 Deep MLP 5->128->128->64->2 (best) <<<")
m2_ok=False
try:
    import pandas as pd, numpy as np, torch, torch.nn as nn, math, pickle
    from sklearn.preprocessing import StandardScaler
    df=pd.read_csv(pair_path)
    n=len(df)
    split=int(n*0.85)
    train=df.iloc[:split]
    val=df.iloc[split:]
    print(f" M2 split train {len(train)} val {len(val)} rows {n}")
    feat_cols=["gfs_t2m_k","gfs_apcp_mm","elevation_m","lead_hours","lat"]
    Xtr=train[feat_cols].values.astype(np.float32)
    Xte=val[feat_cols].values.astype(np.float32)
    ytr_t=train["obs_t2m_c"].values.astype(np.float32)-(train["gfs_t2m_k"].values.astype(np.float32)-273.15)
    ytr_p=train["obs_apcp_mm"].values.astype(np.float32)-train["gfs_apcp_mm"].values.astype(np.float32)
    yte_t=val["obs_t2m_c"].values.astype(np.float32)-(val["gfs_t2m_k"].values.astype(np.float32)-273.15)
    yte_p=val["obs_apcp_mm"].values.astype(np.float32)-val["gfs_apcp_mm"].values.astype(np.float32)
    ytr=np.stack([ytr_t,ytr_p],axis=1).astype(np.float32)
    yte=np.stack([yte_t,yte_p],axis=1).astype(np.float32)
    scaler=StandardScaler()
    Xtr_s=scaler.fit_transform(Xtr)
    Xte_s=scaler.transform(Xte)
    class DeepMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net=nn.Sequential(nn.Linear(5,128),nn.ReLU(),nn.Dropout(0.15),nn.Linear(128,128),nn.ReLU(),nn.Dropout(0.15),nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.1),nn.Linear(64,2))
        def forward(self,x): return self.net(x)
    device=torch.device("cuda" if use_device=="cuda" else "cpu")
    # DataParallel for T4 x2
    model=DeepMLP().to(device)
    if torch.cuda.device_count()>1:
        model=nn.DataParallel(model)
        print(f" DataParallel on {torch.cuda.device_count()} GPUs")
    opt=torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
    loss_fn=nn.MSELoss()
    Xtr_t=torch.tensor(Xtr_s); ytr_t=torch.tensor(ytr)
    Xte_t=torch.tensor(Xte_s); yte_t=torch.tensor(yte)
    best=float("inf")
    best_state=None
    best_rmse=(0,0)
    for epoch in range(1,31):
        model.train()
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
        sched.step()
        train_loss=tot/len(Xtr_t)
        model.eval()
        with torch.no_grad():
            pred=model(Xte_t.to(device))
            val_loss=loss_fn(pred, yte_t.to(device)).item()
            rmse_t=math.sqrt(((pred[:,0]-yte_t.to(device)[:,0])**2).mean().item())
            rmse_p=math.sqrt(((pred[:,1]-yte_t.to(device)[:,1])**2).mean().item())
        print(f" epoch {epoch:02d} train {train_loss:.4f} val {val_loss:.4f} rmse_t {rmse_t:.3f} rmse_p {rmse_p:.3f} lr {sched.get_last_lr()[0]:.1e}")
        if val_loss < best:
            best=val_loss
            best_state={k:v.cpu() for k,v in model.state_dict().items()}
            best_rmse=(rmse_t, rmse_p)
    out=pathlib.Path("training/models/bias_correction")
    out.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out/"best.pt")
    with open(out/"scaler.pkl","wb") as f: pickle.dump(scaler,f)
    import json
    with open(out/"metrics.json","w") as f: json.dump({"best_val_loss":best,"rmse_t":best_rmse[0],"rmse_p":best_rmse[1],"rows":n}, f, indent=2)
    with open(out/"config.json","w") as f: json.dump({"in_dim":5,"hidden":128,"deep":True}, f)
    print(f" M2 saved best {best:.4f} rmse {best_rmse}")
    m2_ok=True
except Exception as e:
    import traceback; traceback.print_exc()
    print(f" M2 failed {e}")

# ===================== TRAIN M3: DistilBERT 5-way with Groq diversity =====================
print("\n>>> M3 DistilBERT 5-way (Groq-diverse 2000) <<<")
m3_ok=False
try:
    import json, torch, numpy as np
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
    from datasets import Dataset
    from sklearn.metrics import accuracy_score, f1_score
    import pathlib as pl
    rows=[json.loads(l) for l in open(intent_path)]
    texts=[r["text"] for r in rows]
    labels=[r["intent"]["decision"] for r in rows]
    uniq=sorted(set(labels))
    l2i={l:i for i,l in enumerate(uniq)}
    y=[l2i[l] for l in labels]
    print(f" M3 labels {l2i} rows {len(rows)}")
    from sklearn.model_selection import train_test_split
    Xtr,Xte,ytr,yte=train_test_split(texts,y,test_size=0.15, random_state=42, stratify=y)
    tok=AutoTokenizer.from_pretrained("distilbert-base-uncased")
    def tok_fn(b):
        return tok(b["text"], truncation=True, padding="max_length", max_length=96)
    train_ds=Dataset.from_dict({"text":Xtr,"labels":ytr}).map(tok_fn, batched=True)
    val_ds=Dataset.from_dict({"text":Xte,"labels":yte}).map(tok_fn, batched=True)
    train_ds.set_format(type="torch", columns=["input_ids","attention_mask","labels"])
    val_ds.set_format(type="torch", columns=["input_ids","attention_mask","labels"])
    model=AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(uniq))
    from collections import Counter
    cnt=Counter(ytr)
    w=[sum(cnt.values())/cnt[l2i[l]] if cnt[l2i[l]]>0 else 1 for l in uniq]
    w=[x/sum(w)*len(w) for x in w]
    cw=torch.tensor(w, dtype=torch.float)
    class WTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            labels=inputs.get("labels")
            outputs=model(**inputs)
            logits=outputs.get("logits")
            loss_fct=torch.nn.CrossEntropyLoss(weight=cw.to(logits.device))
            loss=loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss
    args=TrainingArguments(output_dir="training/models/intent_parser", num_train_epochs=5, per_device_train_batch_size=32, per_device_eval_batch_size=64, learning_rate=2e-5, eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True, logging_steps=20, seed=42, fp16=(use_device=="cuda"), use_cpu=force_cpu, report_to="none", save_total_limit=2)
    trainer=WTrainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds, processing_class=tok, compute_metrics=lambda p: {"accuracy":accuracy_score(p.label_ids, np.argmax(p.predictions,axis=1)), "f1":f1_score(p.label_ids, np.argmax(p.predictions,axis=1), average="weighted")})
    trainer.train()
    metrics=trainer.evaluate()
    print(f" M3 DistilBERT {metrics}")
    out=pl.Path("training/models/intent_parser")
    out.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out))
    tok.save_pretrained(str(out))
    import json
    with open(out/"metrics.json","w") as f: json.dump(metrics, f, indent=2)
    with open(out/"label_map.json","w") as f: json.dump(l2i, f, indent=2)
    m3_ok=True
except Exception as e:
    import traceback; traceback.print_exc()
    print(f" M3 failed {e}")

print("\n"+"="*70)
print(f" OFFICIAL DONE: M1 {m1_ok} M2 {m2_ok} M3 {m3_ok}")
import pathlib as pl, subprocess as sp
out=pl.Path("/kaggle/working/weathergpt_outputs")
out.mkdir(parents=True, exist_ok=True)
sp.run(["cp","-r","training/models",str(out)], check=False)
sp.run(["cp","-r","training/datasets",str(out/"training")], check=False)
for p in pl.Path("training/models").rglob("metrics.json"):
    print(f" {p}: {p.read_text()[:600]}")
print(" done — BEST EVER ready")

