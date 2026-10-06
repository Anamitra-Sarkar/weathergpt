"""Markdown results tables straight from the training kernels' log summaries (no number is typed by hand).

usage: python3 backup/event_models/results_from_logs.py LOG [LOG ...] [--base BASE_LOG ...]
Each LOG is a file holding a kernel log (see kaggle_log.py) with a TRAIN_SUMMARY_BEGIN ... TRAIN_SUMMARY_END JSON block.
Logs after `--base` are the USE_EXT=0 runs; targets present in both get an ext-vs-base comparison (NOT row-paired).
"""
import json
import re
import sys


def load(path: str) -> dict:
    text = open(path).read()
    block = re.search(r"TRAIN_SUMMARY_BEGIN\s*(\{.*\})\s*TRAIN_SUMMARY_END", text, re.S)
    if not block:
        raise SystemExit(f"{path}: no TRAIN_SUMMARY block (kernel not finished?)")
    data = json.loads(block.group(1))
    data.pop("provenance", None)
    return data


def f(x, nd=3):
    return "n/a" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def tables(models: dict) -> str:
    binary, quant, curve = [], [], []
    for name, m in models.items():
        sp = m.get("metrics", {}).get("test_space")
        if m.get("skipped") or not sp:
            continue
        if m["kind"] in ("binary", "xent"):
            binary.append(f"| {name} | {sp['n']} | {f(sp['base_rate'], 4)} | {f(sp['auc'])} | {f(sp.get('auc_rule_feature'))} | "
                          f"{f(sp.get('bss_vs_zone_month'))} | {f(sp.get('bss_vs_global'))} | {f(sp.get('ece'), 4)} |")
        elif m["kind"] == "quantile":
            quant.append(f"| {name} | {sp['n']} | {f(sp['median_mae'])} | {f(sp['gfs_raw_mae'])} | {f(sp['coverage_conformal'])} | "
                         f"{f(sp['mean_width_conformal'], 2)} | {f(sp['static_interval_coverage'])} / {f(sp['static_interval_width'], 2)} |")
        elif m["kind"] == "curve":
            key = list((sp.get("detail_key_thresholds") or {}) or sp["thresholds"])
            for t, e in sp["thresholds"].items():
                curve.append(f"| {name} | {t} | {f(e['base_rate'], 4)} | {f(e.get('auc'))} | {f(e.get('auc_raw_gfs'))} | "
                             f"{f(e.get('bss_vs_zone_month'))} | {f(e.get('bss_vs_gfs_calibrated'))} |")
    out = []
    if binary:
        out += ["| target | n | base rate | AUC | AUC raw-forecast feature | BSS vs zone x month | BSS vs global | ECE |", "|---|---|---|---|---|---|---|---|"] + binary + [""]
    if quant:
        out += ["| target | n | median MAE | raw GFS MAE | conformal coverage | mean width | static interval coverage / width |", "|---|---|---|---|---|---|---|"] + quant + [""]
    if curve:
        out += ["| curve | threshold mm | base rate | AUC | AUC raw GFS | BSS vs zone x month | BSS vs calibrated GFS |", "|---|---|---|---|---|---|---|"] + curve + [""]
    return "\n".join(out)


def ablation(ext: dict, base: dict) -> str:
    rows = ["| target | ext AUC / BSS | base AUC / BSS | ext n | base n |", "|---|---|---|---|---|"]
    for name in ext:
        if name in base and ext[name]["kind"] in ("binary", "xent"):
            a, b = ext[name]["metrics"]["test_space"], base[name]["metrics"]["test_space"]
            rows.append(f"| {name} | {f(a['auc'])} / {f(a.get('bss_vs_zone_month'))} | {f(b['auc'])} / {f(b.get('bss_vs_zone_month'))} | {a['n']} | {b['n']} |")
    return "\n".join(rows)


if __name__ == "__main__":
    args = sys.argv[1:]
    split = args.index("--base") if "--base" in args else len(args)
    ext, base = {}, {}
    for p in args[:split]:
        ext.update(load(p))
    for p in args[split + 1:]:
        base.update(load(p))
    print("### test_space (places never seen, future dates)\n")
    print(tables(ext))
    if base:
        print("\n### Extension features vs base features (rows differ: the extension run keeps only rows that have all inputs)\n")
        print(ablation(ext, base))
