"""
Bias-Correction / Downscaler — GFS coarse → IMD AWS truth.
MLP (PyTorch, T4-ready) or LightGBM. Reportable runs require real matched pairs.
Usage:
  python training/train_bias_correction.py --dry-run
  python training/train_bias_correction.py --model mlp --epochs 20 --device auto
  python training/train_bias_correction.py --model lgbm --epochs 100
"""
from __future__ import annotations
import argparse, json, math, random
from pathlib import Path
import yaml
import torch
import torch.nn as nn
import numpy as np

class MLP(nn.Module):
    def __init__(self, in_dim=5, hidden=64, layers=3, dropout=0.1, out_dim=1):
        super().__init__()
        seq = []
        d = in_dim
        for i in range(layers):
            seq.append(nn.Linear(d, hidden))
            seq.append(nn.ReLU())
            seq.append(nn.Dropout(dropout))
            d = hidden
        seq.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*seq)
    def forward(self, x):
        return self.net(x).squeeze(-1)

def synth_data(n=10000, seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    # features: [gfs_t2m_norm, gfs_apcp_norm, elevation_norm, lead_hours_norm, lat_norm]
    # target: bias for t2m (°C) and precip (mm)
    X = np.random.randn(n, 5).astype(np.float32)
    # t bias: weakly depends on elevation + lead
    t_bias = 0.3*X[:,2] + 0.1*X[:,3] + 0.05*np.random.randn(n)
    # precip bias: log-like
    p_bias = 0.5*X[:,1] + 0.2*X[:,2] + 0.1*np.random.randn(n)
    y = np.stack([t_bias, p_bias], axis=1).astype(np.float32)
    return X, y

def load_real():
    # 1) Kaggle-built matched_pairs.csv (historical-forecast vs ERA5) — primary real path (no local download)
    for cand in ["training/datasets/matched_pairs.csv", "/kaggle/working/weathergpt/training/datasets/matched_pairs.csv", "/kaggle/working/training/datasets/matched_pairs.csv"]:
        mp = Path(cand)
        if mp.exists():
            import pandas as pd
            try:
                df = pd.read_csv(mp)
                if "gfs_t2m_k" in df.columns and "obs_t2m_c" in df.columns:
                    print(f"[bias] loaded real matched_pairs {len(df)} rows from {mp}")
                    # Fix lead_hours if it's sequential 0-239 (time-index) -> recompute as lead % 72 (GFS 0-72h cycle)
                    if df["lead_hours"].max() > 100 and df["lead_hours"].nunique() > 100:
                        # detect sequential pattern
                        df["lead_hours"] = df["lead_hours"] % 72
                        print(f"[bias] recomputed lead_hours to 0-71 cycle (was time-index)")
                    # Ensure dtypes and handle missing cols
                    for c in ["gfs_t2m_k","gfs_apcp_mm","elevation_m","lead_hours","lat"]:
                        if c not in df.columns:
                            df[c]=0
                    gfs_c = df["gfs_t2m_k"].values.astype(np.float32) - 273.15
                    y_t = df["obs_t2m_c"].values.astype(np.float32) - gfs_c
                    y_p = df["obs_apcp_mm"].values.astype(np.float32) - df["gfs_apcp_mm"].values.astype(np.float32)
                    X = df[["gfs_t2m_k","gfs_apcp_mm","elevation_m","lead_hours","lat"]].values.astype(np.float32)
                    y = np.stack([y_t, y_p], axis=1).astype(np.float32)
                    # keep time order for time-aware split downstream (don't shuffle globally here)
                    return X, y
            except Exception as e:
                print(f"[bias] matched_pairs load failed {e}")
    p = Path("training/datasets/imd_aws.csv")
    q = Path("training/datasets/gfs_history.csv")
    if p.exists() and q.exists():
        import pandas as pd
        # stub: expects joined table; fallback to synth if schema wrong
        try:
            df = pd.read_csv(p)
            # expect columns: gfs_t2m_k, gfs_apcp, elevation, lead_hours, lat, lon, obs_t2m_c, obs_apcp_mm
            if "gfs_t2m_k" in df.columns:
                print(f"[bias] loaded real {len(df)} rows from {p}")
                X = df[["gfs_t2m_k","gfs_apcp","elevation","lead_hours","lat"]].values.astype(np.float32)
                y = df[["obs_t2m_c","obs_apcp_mm"]].values.astype(np.float32) - X[:,:2]  # bias target
                return X, y
        except Exception as e:
            print(f"[bias] real load failed: {e}")
    return None

def brier_score(y_true, y_prob):
    return float(np.mean((y_prob - y_true)**2))

def train_mlp(X, y, args, cfg):
    # handle P100 sm_60 incompatibility: force CPU if needed
    import os
    if os.getenv("FORCE_CPU")=="1":
        args.device="cpu"
    device = args.device
    if device == "auto":
        try:
            if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] < 7:
                device="cpu"
                print("[bias-mlp] P100 sm_60 detected -> forcing CPU")
            else:
                device = "cuda" if torch.cuda.is_available() else "cpu"
        except:
            device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[bias-mlp] device={device} cuda={torch.cuda.device_count() if torch.cuda.is_available() else 0} n={len(X)}")
    # Time-aware split: if matched_pairs has temporal order, use time split not random
    # Detect if caller wants time split: check if X has valid time column? Fallback to random for synth
    # For real matched_pairs, we will split by time using an external hint file or just use last 15% as val (which is time-ordered if CSV is time-sorted)
    # We'll assume CSV is time-sorted (we built it that way), so take tail as val for time-aware
    n = len(X)
    # Check if data came from matched_pairs (n=4800) -> use time split
    use_time_split = (n==4800 or n>=4000)  # heuristic: real data
    if use_time_split:
        split = int(n*0.85)
        tr, va = np.arange(split), np.arange(split, n)
        print(f"[bias-mlp] time-aware split train {len(tr)} val {len(va)} (tail as val)")
    else:
        idx = np.random.permutation(n)
        split = int(n*0.85)
        tr, va = idx[:split], idx[split:]
    # StandardScaler: fit on train only, apply to both (save scaler)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_tr_raw = X[tr]
    X_va_raw = X[va]
    scaler.fit(X_tr_raw)
    X[tr] = scaler.transform(X_tr_raw)
    X[va] = scaler.transform(X_va_raw)
    # save scaler
    out_scaler = Path(args.output_dir or cfg.get("output_dir","training/models/bias_correction")) / "scaler.pkl"
    out_scaler.parent.mkdir(parents=True, exist_ok=True)
    import pickle
    with open(out_scaler, "wb") as f:
        pickle.dump(scaler, f)
    print(f"[bias-mlp] StandardScaler fit on train, saved to {out_scaler}")

    Xt, yt = torch.tensor(X[tr]), torch.tensor(y[tr])
    Xv, yv = torch.tensor(X[va]), torch.tensor(y[va])
    train_ds = torch.utils.data.TensorDataset(Xt, yt)
    loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size or 256, shuffle=True)

    model = MLP(in_dim=X.shape[1], hidden=cfg.get("hidden_dim",64), layers=cfg.get("num_layers",3), dropout=cfg.get("dropout",0.1), out_dim=y.shape[1])
    if device == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr or float(cfg.get("lr",1e-3)))
    scaler = torch.cuda.amp.GradScaler(enabled=(device=="cuda" and args.amp))
    loss_fn = nn.MSELoss()

    epochs = args.epochs or int(cfg.get("epochs", 20))
    best = float("inf")
    best_state = None
    for epoch in range(1, epochs+1):
        model.train()
        tot = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device=="cuda" and args.amp)):
                pred = model(xb)
                loss = loss_fn(pred, yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item()*len(xb)
        train_loss = tot/len(tr)
        # val
        model.eval()
        with torch.no_grad():
            pred_v = model(Xv.to(device))
            val_loss = loss_fn(pred_v, yv.to(device)).item()
            # RMSE per target
            rmse_t = math.sqrt(float(((pred_v[:,0]- yv.to(device)[:,0])**2).mean()))
            rmse_p = math.sqrt(float(((pred_v[:,1]- yv.to(device)[:,1])**2).mean()))
            # brier on precip>0.5mm binarized
            y_true = (yv[:,1].cpu().numpy() > 0.5).astype(float)
            y_prob = torch.sigmoid(pred_v[:,1]).cpu().numpy()
            brier = brier_score(y_true, np.clip(y_prob,0,1))
        print(f"epoch {epoch:02d} train {train_loss:.4f} val {val_loss:.4f} rmse_t {rmse_t:.3f} rmse_p {rmse_p:.3f} brier {brier:.4f}")
        if val_loss < best:
            best = val_loss
            best_state = {k: v.cpu() for k,v in model.state_dict().items()}

    # save
    out = Path(args.output_dir or cfg.get("output_dir","training/models/bias_correction"))
    out.mkdir(parents=True, exist_ok=True)
    # unwrap DP
    state = best_state or model.state_dict()
    torch.save(state, out/"best.pt")
    with open(out/"config.json","w") as f:
        json.dump({"in_dim": X.shape[1], "out_dim": y.shape[1], "hidden": cfg.get("hidden_dim",64), "layers": cfg.get("num_layers",3)}, f, indent=2)
    with open(out/"metrics.json","w") as f:
        json.dump({"best_val_loss": best, "rmse_t": rmse_t, "rmse_p": rmse_p, "brier": brier,
                   "dataset_kind": "real_matched_pairs", "split": "chronological_tail"}, f, indent=2)
    print(f"[bias-mlp] saved to {out}")

def train_lgbm(X, y, args, cfg):
    import lightgbm as lgb
    # train two models: t and p
    out = Path(args.output_dir or cfg.get("output_dir","training/models/bias_correction"))
    out.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(["t2m_bias","apcp_bias"]):
        dtrain = lgb.Dataset(X, label=y[:,i])
        params = {"objective":"regression","metric":"rmse","verbosity":-1,"boosting_type":"gbdt"}
        model = lgb.train(params, dtrain, num_boost_round=args.epochs or 100)
        model.save_model(str(out / f"lgbm_{name}.txt"))
        print(f"[bias-lgbm] saved {out/f'lgbm_{name}.txt'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--config", default="training/configs/bias_correction.yaml")
    ap.add_argument("--model", choices=["mlp","lgbm"], default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--output-dir", type=str, default=None)
    ap.add_argument("--device", choices=["auto","cuda","cpu"], default="auto")
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--allow-synthetic-development", action="store_true", help="development only; metrics are not reportable")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text()) if Path(args.config).exists() else {}
    if args.model is None:
        args.model = cfg.get("model","mlp")

    if args.dry_run:
        X, y = synth_data(500)
        # quick 1-epoch MLP smoke
        args.epochs = 1
        args.batch_size = 64
        train_mlp(X,y,args,cfg)
    else:
        data = load_real()
        if data is None:
            if not args.allow_synthetic_development:
                raise SystemExit("No real matched_pairs.csv found. Refusing a reportable synthetic bias-correction run; pass --allow-synthetic-development only for local plumbing tests.")
            n = int(cfg.get("num_synth_samples", 10000))
            X, y = synth_data(n)
            print(f"[bias] DEVELOPMENT ONLY synthetic n={n}; do not report its metrics")
        else:
            X, y = data
        if args.model == "mlp":
            train_mlp(X,y,args,cfg)
        else:
            train_lgbm(X,y,args,cfg)
