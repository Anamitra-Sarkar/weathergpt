"""M3 inference — natural-language question to structured intent."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from weathergpt_models.types import ParsedQuery, Slot


# Tags are assigned to whitespace tokens, so the last token of a span carries
# whatever punctuation followed it — a TIME slot comes back as "tomorrow
# afternoon?" and the downstream time parser then has to cope with the question
# mark.  Trimmed here rather than in the corpus, because a user's own question
# mark is real input.
_EDGE_PUNCTUATION = re.compile(r"^[\s\u2018\u2019\u201c\u201d'\"(\[{,;:।]+|"
                               r"[\s\u2018\u2019\u201c\u201d'\"?!.,;:)\]}।]+$")


def _trim(text: str) -> str:
    trimmed = _EDGE_PUNCTUATION.sub("", text).strip()
    return trimmed or text.strip()


class IntentParser:
    """Joint intent classification, slot tagging and variable selection.

    Returns the *substrings* for location, time and crop rather than a guess at
    their meaning.  Resolving "Bhandara district" to coordinates and "kal dopahar"
    to a UTC window is the framework's job; this model's job is to say which
    part of the sentence is which, in whichever of thirteen Indian languages and
    scripts it was typed.
    """

    def __init__(self, directory: str | Path, *, device: str = "cpu"):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.directory = Path(directory)
        self.device = device
        self._torch = torch
        config = json.loads((self.directory / "config.json").read_text())
        metrics = json.loads((self.directory / "metrics.json").read_text())
        self.algorithm_version = metrics.get("algorithm_version", "m3")
        self.intents = list(config["intents"])
        self.bio_labels = list(config["bio_labels"])
        self.variables = list(config["canonical_variables"])
        self.max_len = int(config.get("max_len", 64))

        payload = torch.load(self.directory / "model.pt", map_location="cpu", weights_only=False)
        base_model = payload.get("base_model", config["base_model"])
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.directory))
        self.backbone = AutoModel.from_pretrained(base_model)
        hidden = self.backbone.config.hidden_size
        self.intent_head = torch.nn.Linear(hidden, len(self.intents))
        self.slot_head = torch.nn.Linear(hidden, len(self.bio_labels))
        self.variable_head = torch.nn.Linear(hidden, len(self.variables))

        state = payload["state_dict"]
        self.backbone.load_state_dict(
            {k[len("backbone."):]: v for k, v in state.items() if k.startswith("backbone.")})
        for prefix, module in (("intent_head.", self.intent_head),
                               ("slot_head.", self.slot_head),
                               ("variable_head.", self.variable_head)):
            module.load_state_dict(
                {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)})
        for module in (self.backbone, self.intent_head, self.slot_head, self.variable_head):
            module.to(device).eval()

    def parse(self, text: str, *, variable_threshold: float = 0.5) -> ParsedQuery:
        torch = self._torch
        words = text.split()
        if not words:
            return ParsedQuery(intent="none", intent_confidence=0.0, variables=[], slots=[],
                               algorithm_version=self.algorithm_version)

        batch = self.tokenizer([words], is_split_into_words=True, padding="max_length",
                               truncation=True, max_length=self.max_len, return_tensors="pt")
        word_ids = batch.word_ids(0)
        inputs = {k: v.to(self.device) for k, v in batch.items()
                  if k in ("input_ids", "attention_mask", "token_type_ids")}

        with torch.no_grad():
            output = self.backbone(**inputs)
            sequence = output.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (sequence * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            intent_logits = self.intent_head(pooled)[0]
            slot_logits = self.slot_head(sequence)[0]
            variable_probabilities = torch.sigmoid(self.variable_head(pooled))[0].cpu().numpy()

        probabilities = torch.softmax(intent_logits, dim=-1).cpu().numpy()
        best = int(probabilities.argmax())

        # first subword of each word carries the tag, matching how it was trained
        tags, previous = {}, None
        best_tags = slot_logits.argmax(-1).cpu().numpy()
        for position, word_id in enumerate(word_ids):
            if word_id is None or word_id == previous:
                continue
            previous = word_id
            if word_id < len(words):
                tags[word_id] = self.bio_labels[int(best_tags[position])]

        slots, current = [], None
        for index in range(len(words)):
            tag = tags.get(index, "O")
            if tag.startswith("B-"):
                if current:
                    slots.append(current)
                current = {"kind": tag[2:], "start": index, "end": index}
            elif tag.startswith("I-") and current and current["kind"] == tag[2:]:
                current["end"] = index
            else:
                if current:
                    slots.append(current)
                current = None
        if current:
            slots.append(current)

        return ParsedQuery(
            intent=self.intents[best],
            intent_confidence=float(probabilities[best]),
            variables=[self.variables[i] for i in np.flatnonzero(
                variable_probabilities > variable_threshold)],
            slots=[Slot(kind=item["kind"],
                        text=_trim(" ".join(words[item["start"]:item["end"] + 1])),
                        start_token=item["start"], end_token=item["end"])
                   for item in slots],
            algorithm_version=self.algorithm_version)
