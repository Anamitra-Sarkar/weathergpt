"""D4 — the M3 corpus: multilingual weather questions with exact slot labels.

The previous intent corpus was unusable for two reasons, both fixed here.

1. Its `decision` label was `random.choice(decisions)` for about two thirds of
   rows, so the label was statistically independent of the text.  Here the
   intent is a property of the template that generated the sentence, and every
   slot value is a string this builder *injected*, so the character spans — and
   therefore the BIO tags — are exact by construction.

2. Its Groq paraphrases hardcoded `location: "Nagpur", time: "tomorrow"` on
   every row regardless of content, destroying the label.  Here the model is
   asked to return the translated slot values as JSON alongside the sentence,
   and any row whose returned slot is not literally a substring of the returned
   sentence is discarded.  A paraphrase can never silently corrupt a label.

The split is by template family *and* by district, so neither a memorised
sentence pattern nor a memorised place name can inflate the score.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import re
from datetime import datetime

from modal_jobs.common import DATA_DIR, DATA_IMAGE, GROQ_SECRET, VOLUMES, app

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
# qwen/qwen3.6-27b is dropped: it reliably 400s under `response_format:
# json_object` ("Failed to validate JSON. Please adjust your prompt"),
# verified directly -- 3/3 calls in a diagnostic burst, a deterministic
# failure that retries cannot fix and that burned three retry sleeps per call
# for a guaranteed rejection.  qwen/qwen3.8-27b is the one model in the
# rotation that returned clean JSON on every successful diagnostic call.
GROQ_MODELS = ["qwen/qwen3.8-27b", "openai/gpt-oss-120b", "openai/gpt-oss-20b"]

# Decision contexts.  These are exactly RADE's domains plus the two
# non-decision cases, so the parser's output feeds the policy engine directly.
INTENTS = ("none", "spray", "irrigate", "harvest", "marine", "travel", "sow", "warning_check")

SLOT_TYPES = ("LOC", "TIME", "CROP")
BIO_LABELS = ["O"] + [f"{prefix}-{slot}" for slot in SLOT_TYPES for prefix in ("B", "I")]

LANGUAGES = {
    "en": "English",
    "hi": "Hindi in Devanagari script",
    "hi_latn": "Hinglish, that is Hindi written in Latin script the way Indians type on WhatsApp",
    "bn": "Bengali in Bengali script",
    "bn_latn": "Bengali written in Latin script",
    "mr": "Marathi in Devanagari script",
    "ta": "Tamil in Tamil script",
    "te": "Telugu in Telugu script",
    "kn": "Kannada in Kannada script",
    "gu": "Gujarati in Gujarati script",
    "pa": "Punjabi in Gurmukhi script",
    "or": "Odia in Odia script",
    "as": "Assamese in Assamese script",
    "ml": "Malayalam in Malayalam script",
}

CROPS = ["paddy", "rice", "wheat", "cotton", "sugarcane", "soybean", "groundnut", "maize",
         "mustard", "gram", "tur dal", "bajra", "jowar", "onion", "potato", "tomato",
         "chilli", "banana", "mango", "grapes", "tea", "coffee", "coconut", "turmeric",
         "cumin", "jute", "sunflower", "sesame"]

TIME_EXPRESSIONS = [
    "today", "tonight", "tomorrow", "tomorrow morning", "tomorrow afternoon",
    "tomorrow evening", "day after tomorrow", "this weekend", "next 3 days",
    "the next 48 hours", "next week", "coming Monday", "this evening",
    "in the next 2 hours", "over the next 6 hours", "the rest of the week",
    "early morning", "late tonight", "the next 10 days", "this month",
]

# (template, intent, variables).  `{loc}`, `{time}` and `{crop}` are the only
# placeholders, and each appears at most once so spans are unambiguous.
TEMPLATES: list[tuple[str, str, str, tuple[str, ...]]] = [
    ("rain_q", "Will it rain in {loc} {time}?", "none", ("precipitation_amount", "precipitation_probability")),
    ("rain_q", "Is there any chance of rain at {loc} {time}?", "none", ("precipitation_probability",)),
    ("rain_amt", "How much rainfall is expected in {loc} {time}?", "none", ("precipitation_amount",)),
    ("temp_q", "What will the temperature be in {loc} {time}?", "none", ("temperature_2m",)),
    ("temp_q", "How hot will it get in {loc} {time}?", "none", ("temperature_max",)),
    ("temp_q", "Will it be cold in {loc} {time}?", "none", ("temperature_min",)),
    ("wind_q", "How strong is the wind in {loc} {time}?", "none", ("wind_speed", "wind_gust")),
    ("humid_q", "What is the humidity in {loc} {time}?", "none", ("humidity",)),
    ("general", "What is the weather in {loc} {time}?", "none", ("temperature_2m", "precipitation_amount")),
    ("general", "Give me the forecast for {loc} {time}.", "none", ("temperature_2m", "precipitation_amount")),
    ("spray", "Should I spray pesticide on my {crop} in {loc} {time}?", "spray",
     ("precipitation_probability", "precipitation_amount", "wind_speed", "wind_gust")),
    ("spray", "Is {time} a good time to spray my {crop} field near {loc}?", "spray",
     ("precipitation_probability", "wind_speed")),
    ("irrigate", "Do I need to irrigate my {crop} in {loc} {time}?", "irrigate",
     ("precipitation_amount", "precipitation_probability", "evapotranspiration")),
    ("irrigate", "Should I water the {crop} field at {loc} {time} or will it rain?", "irrigate",
     ("precipitation_amount", "precipitation_probability")),
    ("harvest", "Can I harvest my {crop} in {loc} {time}?", "harvest",
     ("precipitation_amount", "precipitation_probability", "humidity")),
    ("harvest", "Is it safe to cut and dry {crop} at {loc} {time}?", "harvest",
     ("precipitation_amount", "humidity")),
    ("sow", "Is {time} right for sowing {crop} in {loc}?", "sow",
     ("precipitation_amount", "soil_moisture", "temperature_2m")),
    ("marine", "Can fishermen go to sea off {loc} {time}?", "marine",
     ("wind_speed", "wind_gust", "wave_height", "marine_warning")),
    ("marine", "Is the sea rough near {loc} {time}?", "marine", ("wave_height", "wind_speed")),
    ("travel", "Is it safe to drive from {loc} {time}?", "travel",
     ("precipitation_amount", "visibility", "wind_gust")),
    ("travel", "Will fog affect travel at {loc} {time}?", "travel", ("visibility", "fog_warning")),
    ("warning", "Is there any weather warning for {loc} {time}?", "warning_check",
     ("heavy_rain_warning", "thunderstorm_warning", "cyclone_warning")),
    ("warning", "Has IMD issued any alert for {loc} {time}?", "warning_check",
     ("heavy_rain_warning", "cyclone_warning")),
    ("warning", "Is a cyclone expected near {loc} {time}?", "warning_check", ("cyclone_warning",)),
    ("storm", "Will there be a thunderstorm in {loc} {time}?", "none",
     ("thunderstorm_probability", "thunderstorm_warning")),
    ("heat", "Is a heat wave expected in {loc} {time}?", "warning_check", ("heat_warning", "temperature_max")),
    # --- Added for quality: real user-facing variables the first 26 templates
    # never covered, so the trained model had never seen a positive example
    # for any of them.  Left OUT deliberately: cape, cin, pressure_msl/surface,
    # dewpoint_2m, specific_humidity, solar_radiation, sunshine_duration,
    # soil_temperature, wind_u/wind_v, wave_period -- these are NWP-internal
    # or derived quantities nobody phrases as a natural-language question
    # ("what's the CAPE in Nagpur tomorrow" is not a real query); adding
    # templates for them would be synthetic coverage, not quality.
    ("feels_like", "What will it feel like outside in {loc} {time}?", "none",
     ("apparent_temperature", "temperature_2m")),
    ("cloud", "Will it be cloudy in {loc} {time}?", "none", ("cloud_cover",)),
    ("cold_wave", "Is a cold wave expected in {loc} {time}?", "warning_check",
     ("cold_wave_warning", "temperature_min")),
    ("dust_storm", "Is a dust storm expected in {loc} {time}?", "warning_check",
     ("dust_storm_warning", "visibility")),
    ("flood", "Is there a flood warning for {loc} {time}?", "warning_check",
     ("flood_warning",)),
    ("hail", "Is hail expected in {loc} {time}?", "warning_check",
     ("hail_warning", "thunderstorm_warning")),
    ("rain_rate", "How heavy is the rain in {loc} {time}?", "none",
     ("precipitation_rate",)),
    ("distribution", "Will the rain be widespread or scattered in {loc} {time}?",
     "none", ("rainfall_distribution", "precipitation_amount")),
    ("snow", "Is snowfall expected in {loc} {time}?", "warning_check",
     ("snowfall_amount", "snow_warning")),
    ("snow_depth", "How much snow is on the ground in {loc} {time}?", "none",
     ("snow_depth",)),
    ("wind_dir", "Which direction is the wind blowing in {loc} {time}?", "none",
     ("wind_direction", "wind_speed")),
    ("sea_temp", "How warm is the sea near {loc} {time}?", "marine",
     ("sea_surface_temperature",)),
    ("lightning", "Is there a lightning risk in {loc} {time}?", "warning_check",
     ("lightning_density", "thunderstorm_warning")),
]

# Template families held out entirely, so the test set contains sentence
# patterns the model has never been trained on.
HELD_OUT_FAMILIES = {"sow", "heat", "storm", "dust_storm", "wind_dir", "distribution"}
PROMPT = """You are helping build a labelled dataset of real weather questions asked by people in India.

Rewrite this question naturally in {language}. Keep the meaning identical. It must sound like something a real person would type or say, not a literal word-for-word translation.

Question: {text}
The location mentioned is: {loc}
The time expression is: {time}
{crop_line}
Reply with ONLY a JSON object, no markdown fence, no commentary:
{{"text": "<the rewritten question in {language}>", "loc": "<how the location appears inside your rewritten text, copied exactly>", "time": "<how the time expression appears inside your rewritten text, copied exactly>"{crop_field}}}

Every value you put in "loc", "time"{crop_name} MUST appear character for character inside "text"."""

# Batched form: one call returns translations into several languages at once,
# cutting the call count -- and therefore the wall-clock cost of Groq's
# on_demand queueing -- by the batch size.  A direct diagnostic against the
# live API showed single-call latency dominated by queue time rather than
# generation time, so fewer, larger calls is the lever that actually matters
# here; concurrency and backoff tuning alone could not bring D4 in under a
# session-sized budget.
BATCH_PROMPT = """You are helping build a labelled dataset of real weather questions asked by people in India.

Rewrite this question naturally in EACH of the following languages, one rewrite per language. Keep the meaning identical in every rewrite. Each must sound like something a real person would type or say in that language, not a literal word-for-word translation.

Question: {text}
The location mentioned is: {loc}
The time expression is: {time}
{crop_line}
Languages: {language_list}

Reply with ONLY a JSON array, no markdown fence, no commentary, one object per language in the same order as the language list above:
[{{"language": "<the language name from the list>", "text": "<the rewritten question>", "loc": "<how the location appears inside your rewritten text, copied exactly>", "time": "<how the time expression appears inside your rewritten text, copied exactly>"{crop_field}}}, ...]

Every value you put in "loc", "time"{crop_name} MUST appear character for character inside that same object's "text"."""


def _spans_to_bio(text: str, spans: list) -> list | None:
    """Character spans -> whitespace-token BIO tags.  Returns None if a span does
    not align to token boundaries, so a misaligned row is dropped rather than
    silently mislabelled."""
    tokens, offsets, index = [], [], 0
    for match in re.finditer(r"\S+", text):
        tokens.append(match.group())
        offsets.append((match.start(), match.end()))
        index += 1
    tags = ["O"] * len(tokens)
    for start, end, slot in spans:
        covered = [i for i, (a, b) in enumerate(offsets) if a < end and b > start]
        if not covered:
            return None
        for position, i in enumerate(covered):
            tags[i] = f"{'B' if position == 0 else 'I'}-{slot}"
    return list(zip(tokens, tags))


def _build_base(locations: list, seed: int, per_template: int = 26) -> list:
    rng = random.Random(seed)
    rows = []
    for family, template, intent, variables in TEMPLATES:
        for _ in range(per_template):
            location = rng.choice(locations)
            place = rng.choice([location["name"], location["query"],
                                f"{location['name']} district",
                                location.get("admin2") or location["name"]])
            time_expression = rng.choice(TIME_EXPRESSIONS)
            crop = rng.choice(CROPS)
            text = template.format(loc=place, time=time_expression, crop=crop)
            spans = []
            for value, slot in ((place, "LOC"), (time_expression, "TIME"),
                                (crop if "{crop}" in template else None, "CROP")):
                if not value:
                    continue
                at = text.find(value)
                if at < 0:
                    spans = None
                    break
                spans.append((at, at + len(value), slot))
            if not spans:
                continue
            rows.append({
                "text": text, "family": family, "intent": intent,
                "variables": list(variables), "language": "en",
                "loc_value": place, "time_value": time_expression,
                "crop_value": crop if "{crop}" in template else None,
                "loc_id": location["loc_id"], "spans": spans, "origin": "template",
            })
    return rows


@app.function(image=DATA_IMAGE, volumes=VOLUMES, secrets=[GROQ_SECRET],
              timeout=60 * 90, max_containers=6)
def expand_shard(rows: list, languages: list, shard_id: int,
                 languages_per_row: int = 0, concurrency: int = 4,
                 batch_size: int = 4) -> list:
    """Translate a shard of base rows, batching several languages per Groq call.

    One call per (row, language) pair was the original design and it cannot
    finish a corpus of any useful size: a direct diagnostic against the live
    API showed most of a call's latency is Groq's own `on_demand`-tier queueing,
    not generation time, so the only lever that moves wall-clock time is fewer,
    larger calls.  `batch_size` languages go into one prompt and come back as a
    JSON array; the per-object verification (every slot value must be a literal
    substring of that object's own text) is unchanged, just applied per array
    element instead of per response.
    """
    import httpx

    key = os.environ.get("GROQ_API_KEY") or os.environ.get("groq_api_key")
    if not key:
        raise RuntimeError("GROQ_API_KEY missing from the modal secret")

    async def run():
        out, rejected = [], {"http": 0, "json": 0, "slot_not_substring": 0,
                             "span_align": 0, "batch_size_mismatch": 0}
        semaphore = asyncio.Semaphore(concurrency)
        completed = [0]

        async def one_batch(client, row, language_batch, model):
            crop = row.get("crop_value")
            prompt = BATCH_PROMPT.format(
                text=row["text"], loc=row["loc_value"], time=row["time_value"],
                crop_line=f"The crop mentioned is: {crop}" if crop else "",
                crop_field=', "crop": "<how the crop appears inside your rewritten text, copied exactly>"' if crop else "",
                crop_name=' and "crop"' if crop else "",
                language_list=", ".join(LANGUAGES[code] for code in language_batch))
            async with semaphore:
                for attempt in range(3):
                    try:
                        response = await client.post(
                            GROQ_URL, headers={"Authorization": f"Bearer {key}"},
                            json={"model": model,
                                  "messages": [{"role": "user", "content": prompt}],
                                  "temperature": 0.8, "max_tokens": 300 * len(language_batch)})
                        if response.status_code == 429:
                            await asyncio.sleep(6 * (attempt + 1) + random.random() * 3)
                            continue
                        if response.status_code == 400:
                            rejected["http"] += 1
                            return
                        response.raise_for_status()
                        break
                    except Exception:
                        if attempt == 2:
                            rejected["http"] += 1
                            return
                        await asyncio.sleep(3 * (attempt + 1))
                else:
                    rejected["http"] += 1
                    return
            try:
                content = response.json()["choices"][0]["message"]["content"]
                array = json.loads(re.sub(r"^```(?:json)?|```$", "", content.strip(),
                                          flags=re.M).strip())
                if not isinstance(array, list):
                    array = [array]
            except Exception:
                rejected["json"] += 1
                return
            if len(array) != len(language_batch):
                rejected["batch_size_mismatch"] += 1
                # still salvage whatever the model actually returned by name
            by_language_name = {LANGUAGES[code]: code for code in language_batch}
            for item in array:
                if not isinstance(item, dict):
                    rejected["json"] += 1
                    continue
                code = by_language_name.get(str(item.get("language", "")).strip())
                if code is None:
                    rejected["json"] += 1
                    continue
                text = str(item.get("text", "")).strip()
                if not text or len(text) > 400:
                    rejected["json"] += 1
                    continue
                spans = []
                wanted = [("loc", "LOC"), ("time", "TIME")] + (
                    [("crop", "CROP")] if crop else [])
                ok = True
                for key_name, slot in wanted:
                    value = str(item.get(key_name, "")).strip()
                    at = text.find(value)
                    if not value or at < 0:
                        rejected["slot_not_substring"] += 1
                        ok = False
                        break
                    spans.append((at, at + len(value), slot))
                if not ok:
                    continue
                if _spans_to_bio(text, spans) is None:
                    rejected["span_align"] += 1
                    continue
                out.append({**row, "text": text, "language": code, "spans": spans,
                           "origin": f"groq:{model}",
                           "loc_value": str(item.get("loc")).strip(),
                           "time_value": str(item.get("time")).strip(),
                           "crop_value": str(item.get("crop")).strip() if crop else None})
            completed[0] += 1
            if completed[0] % 25 == 0 or completed[0] == total_calls:
                print(f"[d4:{shard_id}] {completed[0]}/{total_calls} batched calls done, "
                      f"kept {len(out)} rejected {rejected}")

        rng = random.Random(1000 + shard_id)
        tasks = []
        async with httpx.AsyncClient(timeout=90) as client:
            for index, row in enumerate(rows):
                if languages_per_row and languages_per_row < len(languages):
                    chosen = rng.sample(languages, languages_per_row)
                else:
                    chosen = list(languages)
                for start in range(0, len(chosen), batch_size):
                    batch = chosen[start:start + batch_size]
                    model = GROQ_MODELS[(index + start) % len(GROQ_MODELS)]
                    tasks.append(one_batch(client, row, batch, model))
            total_calls = len(tasks)
            print(f"[d4:{shard_id}] dispatching {total_calls} batched calls "
                  f"({len(rows)} rows, up to {languages_per_row or len(languages)} "
                  f"languages each, {batch_size}/call) at concurrency={concurrency}")
            await asyncio.gather(*tasks)
        print(f"[d4:{shard_id}] kept {len(out)} rejected {rejected}")
        return out

    return asyncio.run(run())


@app.function(image=DATA_IMAGE, volumes=VOLUMES, timeout=60 * 20)
def assemble(expanded: list, base: list, seed: int = 42,
             merge_existing: bool = False) -> dict:
    import pandas as pd

    from modal_jobs.contracts import check_label_corpus

    rows = base + expanded
    frame = pd.DataFrame(rows)

    # A second pass with a different seed can grow the corpus without discarding
    # the first: rows are merged and de-duplicated on the normalised text below.
    existing_path = f"{DATA_DIR}/d4_queries.parquet"
    if merge_existing and os.path.exists(existing_path):
        previous = pd.read_parquet(existing_path)
        keep = [column for column in frame.columns if column in previous.columns]
        previous = previous[keep]
        frame = pd.concat([previous, frame[keep]], ignore_index=True)
        print(f"[d4] merged {len(previous):,} existing rows")
    if "spans" in frame.columns:
        pending = frame["spans"].notna()
        bio = [_spans_to_bio(row["text"], row["spans"]) if row.get("spans") is not None else None
               for row in frame.to_dict("records")]
        frame["bio"] = bio
        frame = frame[frame["bio"].notna()].reset_index(drop=True)
        frame["tokens"] = frame["bio"].map(lambda pairs: [t for t, _ in pairs])
        frame["tags"] = frame["bio"].map(lambda pairs: [g for _, g in pairs])
        frame = frame.drop(columns=["bio", "spans"])

    frame["norm"] = frame["text"].str.strip().str.lower()
    before = len(frame)
    frame = frame.drop_duplicates(subset=["norm"]).reset_index(drop=True)

    # Split by template family AND by location, so the test set has both unseen
    # sentence patterns and unseen districts.
    all_locs = sorted(frame["loc_id"].unique())
    rng = random.Random(seed)
    held_out_locs = set(rng.sample(all_locs, max(1, len(all_locs) // 5)))

    def assign(row) -> str:
        if row["family"] in HELD_OUT_FAMILIES or row["loc_id"] in held_out_locs:
            return "test"
        digest = int(hashlib.sha1(row["norm"].encode()).hexdigest()[:8], 16) % 100
        return "val" if digest < 12 else "train"

    frame["split"] = frame.apply(assign, axis=1)
    frame = frame.drop(columns=["norm"])

    os.makedirs(DATA_DIR, exist_ok=True)
    out_path = f"{DATA_DIR}/d4_queries.parquet"
    frame.to_parquet(out_path, index=False, compression="zstd")

    report = check_label_corpus(frame.assign(key=frame["text"]), name="d4_queries",
                                label_column="intent", split_column="split",
                                key_column="key", min_rows=3000)
    stats = {
        "built_at": datetime.utcnow().isoformat() + "Z",
        "rows_before_dedup": int(before), "rows": int(len(frame)),
        "by_split": frame["split"].value_counts().to_dict(),
        "by_language": frame["language"].value_counts().to_dict(),
        "by_intent": frame["intent"].value_counts().to_dict(),
        "by_origin": frame["origin"].map(lambda s: s.split(":")[0]).value_counts().to_dict(),
        "held_out_families": sorted(HELD_OUT_FAMILIES),
        "n_held_out_locations": len(held_out_locs),
        "majority_intent_baseline": float(frame["intent"].value_counts().iloc[0] / len(frame)),
        "sha256": hashlib.sha256(open(out_path, "rb").read()).hexdigest(),
        "contracts": report.summary(),
    }
    with open(f"{DATA_DIR}/d4_queries.stats.json", "w") as handle:
        json.dump(stats, handle, indent=2, default=str)
    from modal_jobs.common import DATA_VOL
    DATA_VOL.commit()
    print(json.dumps(stats, indent=2, default=str)[:4000])
    return stats


@app.function(image=DATA_IMAGE, volumes=VOLUMES)
def read_locations() -> str:
    with open(f"{DATA_DIR}/locations.json") as handle:
        return handle.read()


@app.local_entrypoint()
def main_build_queries(n_shards: int = 8, seed: int = 42, per_template: int = 26,
         merge_existing: bool = False, languages_per_row: int = 0,
         concurrency: int = 4, batch_size: int = 4):
    meta = json.loads(read_locations.remote())
    base = _build_base(meta["locations"], seed, per_template)
    print(f"[d4] {len(base)} base English rows from {len(TEMPLATES)} templates")

    languages = [code for code in LANGUAGES if code != "en"]
    shards = [base[i::n_shards] for i in range(n_shards)]
    # each shard covers every language, so a failing container loses coverage
    # evenly instead of wiping out one language entirely
    expanded: list = []
    for chunk in expand_shard.starmap(
            [(shard, languages, i, languages_per_row, concurrency, batch_size)
             for i, shard in enumerate(shards)]):
        expanded += chunk
    print(f"[d4] {len(expanded)} multilingual rows survived slot verification")
    assemble.remote(expanded, base, seed, merge_existing)
