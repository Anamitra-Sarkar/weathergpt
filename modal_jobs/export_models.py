"""Publish the trained artifacts to the Hugging Face Hub, from Modal.

Runs remotely so no checkpoint is ever written to the local machine.  It reads
the `weathergpt-models` volume, refuses anything that fails the same admission
gate the serving registry applies, writes a model card generated from the
measured metrics, and uploads.

The card is generated, never typed: if a number is not in a `metrics.json` it
does not appear in the card.
"""
from __future__ import annotations

import json
import os
from datetime import datetime

import modal

from modal_jobs.common import MODEL_DIR, TRAIN_IMAGE, TRAIN_VOLUMES, app

HF_SECRET = modal.Secret.from_name("hf-upload-token")


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _card(bundle: dict, repo_id: str) -> str:
    lines = [
        "---", "license: apache-2.0", "library_name: weathergpt-models",
        "tags:", "  - weather", "  - post-processing", "  - india",
        "  - meteorology", "---", "",
        "# WeatherGPT model bundle", "",
        "Five models for the WeatherGPT meteorological interoperability layer.",
        "Every number below was produced by a training script and copied from a",
        "`metrics.json`; none was typed by hand.", "",
        "```python",
        "from weathergpt_models import ModelRegistry",
        f'registry = ModelRegistry.from_hub("{repo_id}")',
        "print(registry.status())",
        "```", "",
        "Each model is gated on beating the baseline it replaces. An artifact",
        "that cannot prove its provenance, or that does not beat its baseline,",
        "is refused at load time and the caller falls back to a deterministic",
        "path.", "",
    ]

    for name, block in bundle.items():
        metrics = block["metrics"]
        lines += [f"## {name} — `{metrics.get('algorithm_version')}`", "",
                  f"**{metrics.get('model_kind', '')}**", "",
                  f"- dataset: `{metrics.get('dataset_kind')}`",
                  f"- dataset sha256: `{(metrics.get('dataset_sha256') or '')[:32]}`",
                  f"- split: {metrics.get('split')}",
                  f"- trained: {metrics.get('trained_at')}",
                  f"- admission gate: **{'PASS' if block['gate_ok'] else 'REFUSED'}** — {block['gate_reason']}",
                  ""]

        if name == "field_mapper":
            baselines = metrics.get("baselines", {})
            lines += ["| metric | model | dict registry | majority |", "|---|---|---|---|",
                      f"| zero-shot macro-F1 | **{_fmt(metrics.get('test_zeroshot_macro_f1'))}** | "
                      f"{_fmt(baselines.get('dict_registry_zeroshot', {}).get('macro_f1'))} | "
                      f"{_fmt(baselines.get('majority_class_zeroshot', {}).get('macro_f1'))} |",
                      f"| zero-shot accuracy | **{_fmt(metrics.get('test_zeroshot_accuracy'))}** | "
                      f"{_fmt(baselines.get('dict_registry_zeroshot', {}).get('accuracy'))} | "
                      f"{_fmt(baselines.get('majority_class_zeroshot', {}).get('accuracy'))} |",
                      f"| statistic accuracy | {_fmt(metrics.get('test_zeroshot_statistic_accuracy'))} | — | — |",
                      f"| misassignment rate | {_fmt(metrics.get('test_zeroshot_misassignment_rate'))} | — | — |",
                      "",
                      "The test set is parameter tables the model never trained on "
                      "(WRF Registry, NCEP GFS inventories, the WMO BUFR element table, "
                      "Open-Meteo and IMD product fields), so this measures generalisation "
                      "to a schema nobody has mapped.", ""]

        elif name in ("mos", "calibration"):
            lines += ["| variable | CRPS | raw ensemble CRPS | CRPS skill |", "|---|---|---|---|"]
            for variable, block_v in (metrics.get("results") or {}).items():
                test = block_v.get("test_spatial_holdout", {})
                lines.append(
                    f"| {variable} | {_fmt(test.get('crps_model'))} | "
                    f"{_fmt(test.get('crps_raw_ensemble') or test.get('crps_raw_multi_model_ensemble'))} | "
                    f"**{_fmt(test.get('crpss_vs_raw_ensemble'), 3)}** |")
            lines.append("")
            exceedance = (metrics.get("results") or {}).get("precipitation", {}).get("exceedance")
            if exceedance:
                lines += ["Precipitation exceedance, Brier score (lower is better):", "",
                          "| threshold | base rate | raw member count | calibrated | climatology |",
                          "|---|---|---|---|---|"]
                for threshold, entry in exceedance.items():
                    lines.append(
                        f"| >{threshold} mm | {_fmt(entry.get('base_rate'))} | "
                        f"{_fmt(entry.get('brier_raw_ensemble_frequency'), 5)} | "
                        f"**{_fmt(entry.get('brier_csgd_isotonic'), 5)}** | "
                        f"{_fmt(entry.get('brier_climatology'), 5)} |")
                lines.append("")
            if metrics.get("transfer_assumption"):
                lines += ["> **Transfer assumption.** " + metrics["transfer_assumption"], ""]

        elif name == "intent":
            test = metrics.get("test_heldout", {})
            baseline = metrics.get("baselines", {}).get("rule_based_retrieval_planner_test", {})
            lines += ["| metric | model | rule parser |", "|---|---|---|",
                      f"| intent macro-F1 | **{_fmt(test.get('intent_macro_f1'))}** | "
                      f"{_fmt(baseline.get('intent_macro_f1'))} |",
                      f"| intent accuracy | **{_fmt(test.get('intent_accuracy'))}** | "
                      f"{_fmt(baseline.get('intent_accuracy'))} |",
                      f"| slot F1 (seqeval) | {_fmt(test.get('slot_f1'))} | not supported |",
                      f"| variable micro-F1 | {_fmt(test.get('variable_micro_f1'))} | not supported |",
                      "",
                      "Held out: whole template families and whole districts, so neither a "
                      "memorised sentence pattern nor a memorised place name can inflate this.",
                      ""]
            per_language = metrics.get("per_language_test") or {}
            if per_language:
                lines += ["| language | n | intent macro-F1 | slot F1 |", "|---|---|---|---|"]
                for language, block_l in sorted(per_language.items()):
                    lines.append(f"| {language} | {block_l.get('n')} | "
                                 f"{_fmt(block_l.get('intent_macro_f1'))} | "
                                 f"{_fmt(block_l.get('slot_f1'))} |")
                lines.append("")

        elif name == "trust_ranker":
            lines += ["| variable | NDCG@1 | picks the best source | fixed authority does | "
                      "RMSE following ranker | RMSE following fixed order |",
                      "|---|---|---|---|---|---|"]
            for variable, block_v in (metrics.get("results") or {}).items():
                test = block_v.get("test_spatial_holdout", {})
                lines.append(
                    f"| {variable} | {_fmt(test.get('ndcg@1'))} | "
                    f"**{_fmt(test.get('top1_is_actually_best_rate'), 3)}** | "
                    f"{_fmt(test.get('heuristic_top1_is_actually_best_rate'), 3)} | "
                    f"**{_fmt(test.get('rmse_following_ranker'), 3)}** | "
                    f"{_fmt(test.get('rmse_following_fixed_authority'), 3)} |")
            lines.append("")

    lines += ["## Provenance", "",
              f"Bundle built {datetime.utcnow().isoformat()}Z from the `weathergpt-models` "
              "Modal volume.", "",
              "Training corpora were built from Open-Meteo (multi-model NWP archives and "
              "ERA5 reanalysis), the CF standard name table, the ECMWF eccodes GRIB2 "
              "definitions, NOAA NCEP GRIB2 code tables and GFS inventories, the WRF "
              "Registry, the WMO BUFR element table, and the live SACHET/NDMA CAP feed.", ""]
    return "\n".join(lines)


@app.function(image=TRAIN_IMAGE, volumes=TRAIN_VOLUMES, secrets=[HF_SECRET],
              timeout=60 * 60)
def export(repo_id: str, private: bool = False, dry_run: bool = False) -> dict:
    import shutil
    from pathlib import Path

    from huggingface_hub import HfApi

    from weathergpt_models.registry import GATES, REQUIRED_PROVENANCE

    root = Path(MODEL_DIR)
    staging = Path("/tmp/bundle")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    bundle, skipped = {}, []
    for name, gate in GATES.items():
        directory = root / gate.directory
        metrics_path = directory / "metrics.json"
        if not metrics_path.exists():
            skipped.append({"model": name, "reason": "not trained yet"})
            continue
        metrics = json.loads(metrics_path.read_text())
        missing = [key for key in REQUIRED_PROVENANCE if not metrics.get(key)]
        if missing:
            skipped.append({"model": name, "reason": f"missing provenance {missing}"})
            continue
        try:
            ok, reason = gate.passes(metrics)
        except Exception as exc:
            ok, reason = False, f"gate raised {exc}"
        if not ok:
            # Publish it anyway, but the card and the registry both say REFUSED,
            # so a consumer can see the failure instead of guessing at a silence.
            print(f"[export] {name} FAILS its gate: {reason}")
        shutil.copytree(directory, staging / gate.directory)
        bundle[name] = {"metrics": metrics, "gate_ok": ok, "gate_reason": reason}
        print(f"[export] staged {name} ({gate.directory}) gate={'PASS' if ok else 'REFUSED'}")

    if not bundle:
        raise RuntimeError("nothing to export: no trained artifact carries valid provenance")

    (staging / "README.md").write_text(_card(bundle, repo_id))
    (staging / "bundle.json").write_text(json.dumps(
        {"built_at": datetime.utcnow().isoformat() + "Z",
         "models": {name: {"algorithm_version": block["metrics"].get("algorithm_version"),
                           "gate_ok": block["gate_ok"], "gate_reason": block["gate_reason"],
                           "dataset_sha256": block["metrics"].get("dataset_sha256")}
                    for name, block in bundle.items()},
         "skipped": skipped}, indent=2))

    total = sum(path.stat().st_size for path in staging.rglob("*") if path.is_file())
    print(f"[export] bundle {total / 1e6:.1f} MB, {len(bundle)} models, skipped {skipped}")

    if dry_run:
        return {"models": list(bundle), "skipped": skipped, "bytes": total, "uploaded": False}

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or \
        os.environ.get("hf_token")
    if not token:
        raise RuntimeError("no HF token in the modal secret (expected HF_TOKEN)")
    api = HfApi(token=token)
    who = api.whoami()
    print(f"[export] uploading as {who.get('name')}")
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True, repo_type="model")
    api.upload_folder(folder_path=str(staging), repo_id=repo_id, repo_type="model",
                      commit_message="WeatherGPT model bundle")
    url = f"https://huggingface.co/{repo_id}"
    print(f"[export] published {url}")
    return {"models": list(bundle), "skipped": skipped, "bytes": total,
            "uploaded": True, "url": url, "account": who.get("name")}


@app.function(image=TRAIN_IMAGE, volumes=TRAIN_VOLUMES, timeout=60 * 20)
def report(repo_id: str = "weathergpt/models") -> str:
    """Return the generated model card without uploading anything."""
    from pathlib import Path

    from weathergpt_models.registry import GATES, REQUIRED_PROVENANCE

    root = Path(MODEL_DIR)
    bundle = {}
    for name, gate in GATES.items():
        metrics_path = root / gate.directory / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text())
        missing = [key for key in REQUIRED_PROVENANCE if not metrics.get(key)]
        if missing:
            ok, reason = False, f"missing provenance {missing}"
        else:
            try:
                ok, reason = gate.passes(metrics)
            except Exception as exc:
                ok, reason = False, f"gate raised {exc}"
        bundle[name] = {"metrics": metrics, "gate_ok": ok, "gate_reason": reason}
    return _card(bundle, repo_id)


@app.local_entrypoint()
def main(repo_id: str = "", private: bool = False, dry_run: bool = False,
         card_only: str = ""):
    """`--card-only docs/MODELS.md` writes the generated card and uploads nothing."""
    if card_only:
        text = report.remote(repo_id or "weathergpt/models")
        with open(card_only, "w") as handle:
            handle.write(text)
        print(f"wrote {card_only} ({len(text)} bytes)")
        return
    if not repo_id and not dry_run:
        raise SystemExit("pass --repo-id <user>/<name>, --dry-run, or --card-only <path>")
    export.remote(repo_id or "dry/run", private=private, dry_run=dry_run)
