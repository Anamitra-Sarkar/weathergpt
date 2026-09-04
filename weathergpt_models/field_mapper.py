"""M1 inference — what does this native field name actually mean?"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from weathergpt_models.taxonomy import (ALLOWED_UNIT_FAMILIES, CANONICAL_VARIABLES,
                                        EVIDENCE_CLASSES, STATISTICS, VERTICAL_LEVELS,
                                        classify_native_field, unit_family)
from weathergpt_models.types import FieldMapping


class FieldMapper:
    """Maps an arbitrary provider field name onto the canonical vocabulary.

    Two paths, deliberately in this order:

    1. The **deterministic taxonomy** first.  When a field's own metadata
       identifies it unambiguously — `APCP` with unit `kg m-2` and a
       `0-3 hour acc fcst` time range — there is nothing for a model to add, and
       a rule that can be read is better than a rule that must be trusted.
    2. The **trained mapper** when the taxonomy abstains, which is the case the
       model exists for: a schema nobody has written rules for.

    The model abstains too, below a threshold calibrated on validation.  A
    refusal is a valid answer; a confident wrong mapping is the failure this
    project is built to prevent, and the trained model's rate of that on schemas
    it had never seen is recorded in its metrics as `misassignment_rate`.
    """

    def __init__(self, directory: str | Path, *, device: str = "cpu"):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.directory = Path(directory)
        self.device = device
        config = json.loads((self.directory / "config.json").read_text())
        metrics = json.loads((self.directory / "metrics.json").read_text())
        self.algorithm_version = metrics.get("algorithm_version", "m1")
        self.threshold = float(config.get("abstention_threshold", 0.5))
        self.max_len = int(config.get("max_len", 64))
        self.variables = list(config.get("canonical_variables", CANONICAL_VARIABLES))
        self.statistics = list(config.get("statistics", STATISTICS))
        self.levels = list(config.get("levels", VERTICAL_LEVELS))
        self.evidence_classes = list(config.get("evidence_classes", EVIDENCE_CLASSES))

        payload = torch.load(self.directory / "model.pt", map_location="cpu", weights_only=False)
        base_model = payload.get("base_model", config["base_model"])
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.directory))
        backbone = AutoModel.from_pretrained(base_model)

        hidden = backbone.config.hidden_size
        self._torch = torch
        self.backbone = backbone
        self.stat_head = torch.nn.Linear(hidden, len(self.statistics))
        self.level_head = torch.nn.Linear(hidden, len(self.levels))
        self.class_head = torch.nn.Linear(hidden, len(self.evidence_classes))

        state = payload["state_dict"]
        self.backbone.load_state_dict(
            {k[len("backbone."):]: v for k, v in state.items() if k.startswith("backbone.")})
        for prefix, module in (("stat_head.", self.stat_head), ("level_head.", self.level_head),
                               ("class_head.", self.class_head)):
            module.load_state_dict(
                {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)})

        for module in (self.backbone, self.stat_head, self.level_head, self.class_head):
            module.to(device).eval()

        label_path = self.directory / "label_embeddings.npy"
        self.label_embeddings = np.load(label_path) if label_path.exists() else None

    # --- encoding -----------------------------------------------------------
    def _embed(self, texts: list) -> np.ndarray:
        torch = self._torch
        batch = self.tokenizer(texts, padding=True, truncation=True,
                               max_length=self.max_len, return_tensors="pt")
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with torch.no_grad():
            output = self.backbone(input_ids=batch["input_ids"],
                                   attention_mask=batch["attention_mask"])
            mask = batch["attention_mask"].unsqueeze(-1).float()
            pooled = (output.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            self._last_pooled = pooled
        return pooled.cpu().numpy()

    @staticmethod
    def _render(name: str, unit, description: str, level_text: str, time_range_text: str) -> str:
        parts = [f"field: {name}"]
        if unit:
            parts.append(f"unit: {unit}")
        if description:
            parts.append(f"means: {description}")
        if level_text:
            parts.append(f"level: {level_text}")
        if time_range_text:
            parts.append(f"period: {time_range_text}")
        return "query: " + " | ".join(parts)

    # --- public API ---------------------------------------------------------
    def map_field(self, name: str, *, unit: str | None = None, description: str = "",
                  level_text: str = "", time_range_text: str = "",
                  grib_statistical_processing: int | None = None,
                  evidence_class_hint: str | None = None,
                  prefer_rules: bool = True) -> FieldMapping:
        if prefer_rules:
            rule = classify_native_field(
                name, description=description, unit=unit, level_text=level_text,
                time_range_text=time_range_text,
                grib_statistical_processing=grib_statistical_processing,
                evidence_class_hint=evidence_class_hint)
            if rule is not None:
                return FieldMapping(
                    canonical_variable=rule.canonical_variable, statistic=rule.statistic,
                    accumulation_hours=rule.accumulation_hours,
                    vertical_level=rule.vertical_level, evidence_class=rule.evidence_class,
                    confidence=rule.confidence, abstained=False, source="rule",
                    algorithm_version=self.algorithm_version)

        text = self._render(name, unit, description, level_text, time_range_text)
        vector = self._embed([text])[0]
        vector = vector / (np.linalg.norm(vector) + 1e-9)
        similarity = self.label_embeddings @ vector
        order = np.argsort(-similarity)

        # Hard veto: no similarity score, however confident, may place a field
        # into a variable whose declared unit contradicts it.  The trained model
        # abstains on plenty of real unit contradictions, but it is a statistical
        # judgement and not a guarantee -- verified directly: TMAX + unit "mm"
        # scored 0.75 similarity to temperature_max, comfortably above the
        # calibrated abstention threshold, because the model's confidence was
        # never asked to certify a hard physical constraint.  This walks the
        # ranked candidates and takes the first whose unit family is compatible,
        # falling through to `other` (abstain) if none is.
        declared_family = unit_family(unit) if unit is not None else None

        def compatible(index: int) -> bool:
            allowed = ALLOWED_UNIT_FAMILIES.get(self.variables[index])
            return (declared_family is None or declared_family == "categorical"
                    or allowed is None or declared_family in allowed)

        best = next((int(candidate) for candidate in order if compatible(int(candidate))), None)
        no_compatible_candidate = best is None
        if no_compatible_candidate:
            # every candidate's declared unit family contradicts it -- there is
            # nothing safe to return, so fall back to the top score purely to
            # have a runner-up/margin to report, and force an abstain below
            best = int(order[0])
        picked_past_top = best != int(order[0])
        second = int(order[1]) if int(order[1]) != best else int(order[2])
        score = float(similarity[best])

        torch = self._torch
        with torch.no_grad():
            pooled = self._last_pooled
            statistic = self.statistics[int(self.stat_head(pooled).argmax(-1))]
            level = self.levels[int(self.level_head(pooled).argmax(-1))]
            evidence = self.evidence_classes[int(self.class_head(pooled).argmax(-1))]

        abstained = (score < self.threshold or self.variables[best] == "other"
                    or no_compatible_candidate)
        from weathergpt_models.taxonomy import parse_accumulation_hours

        accumulation = (parse_accumulation_hours(f"{time_range_text} {name} {description}")
                        if statistic == "accumulation" else None)
        return FieldMapping(
            canonical_variable="other" if abstained else self.variables[best],
            statistic=statistic, accumulation_hours=accumulation, vertical_level=level,
            evidence_class=evidence_class_hint or evidence,
            confidence=score, abstained=abstained,
            runner_up=self.variables[second], margin=float(similarity[best] - similarity[second]),
            source=("abstain" if abstained else
                    ("model_unit_filtered" if picked_past_top else "model")),
            algorithm_version=self.algorithm_version)

    def map_many(self, fields: list) -> list:
        """`fields` is a list of dicts accepted by `map_field`."""
        return [self.map_field(**item) for item in fields]
