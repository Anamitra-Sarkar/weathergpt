"""Inline the `event_models.*` (and `weathergpt_events.*`) modules an entry script imports into ONE file.

A Kaggle script kernel is a single file and we do not want to depend on the
GitHub repo being current, so this produces a self-contained bundle:

    python event_models/bundle.py event_models/collect_truth.py OUT.py

Each `event_models/<name>.py` that the entry (or another bundled module)
imports via `from event_models import <name>` is embedded as a string and
registered in `sys.modules` before the entry code runs.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PACKAGES = {"event_models": ROOT, "weathergpt_events": ROOT.parent / "weathergpt_events"}     # package -> source folder
IMPORTS = [(pkg, rx) for pkg in PACKAGES for rx in (re.compile(rf"^from {pkg} import (\w+)(?: as \w+)?\s*$", re.M),
                                                    re.compile(rf"^from {pkg}\.(\w+) import ", re.M))]


def modules_needed(source: str, seen: dict) -> None:
    for pkg, rx in IMPORTS:
        for name in rx.findall(source):
            if (pkg, name) in seen:
                continue
            text = (PACKAGES[pkg] / f"{name}.py").read_text()
            modules_needed(text, seen)  # dependencies first
            seen[(pkg, name)] = text


def bundle(entry: Path, env: dict | None = None) -> str:
    source = entry.read_text()
    mods: dict = {}
    modules_needed(source, mods)
    header = ["import os as _os"] + [f"_os.environ[{k!r}] = {v!r}" for k, v in (env or {}).items()]
    header += ["import sys as _sys, types as _types",
               "def _load(pkg, name, src):",
               "    if pkg not in _sys.modules:",
               "        _p = _types.ModuleType(pkg); _p.__path__ = []; _sys.modules[pkg] = _p",
               "    m = _types.ModuleType(pkg + '.' + name); _sys.modules[pkg + '.' + name] = m",
               "    setattr(_sys.modules[pkg], name, m); exec(compile(src, pkg + '/' + name + '.py', 'exec'), m.__dict__)"]
    for (pkg, name), text in mods.items():
        header.append(f"_load({pkg!r}, {name!r}, {text!r})")
    if "_bundled_sources" in source:     # entry scripts that republish their own code (publish_hf) read this {relative path: text}
        header.append(f"_sys._bundled_sources = {({f'{pkg}/{name}.py': text for (pkg, name), text in mods.items()})!r}")
    # `from __future__` must be the first statement, so hoist it above the header
    future = "from __future__ import annotations\n"
    body = source.replace(future, "", 1)
    return future + "\n".join(header) + "\n" + body


def parse_env(argv: list) -> dict:
    """Accepts `--env KEY=VALUE` and `--env=KEY=VALUE`; anything else is an error (a silent skip once baked a bogus
    variable and would have made every training kernel train every target)."""
    env, i = {}, 0
    while i < len(argv):
        token = argv[i]
        if token == "--env" and i + 1 < len(argv):
            item, i = argv[i + 1], i + 2
        elif token.startswith("--env="):
            item, i = token[len("--env="):], i + 1
        else:
            raise SystemExit(f"unrecognised argument {token!r}; expected --env KEY=VALUE")
        key, sep, value = item.partition("=")
        if not sep or not key or key.startswith("-"):
            raise SystemExit(f"bad --env value {item!r}; expected KEY=VALUE")
        env[key] = value
    return env


if __name__ == "__main__":
    # usage: bundle.py ENTRY OUT [--env KEY=VALUE ...]   (env is baked into the bundle)
    entry, out = Path(sys.argv[1]), Path(sys.argv[2])
    env = parse_env(sys.argv[3:])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(bundle(entry, env))
    print(f"wrote {out} ({out.stat().st_size} bytes)")
