"""Global timeline selection and portable JSON helpers."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)

def layout(total=121, first=5, block=6):
    if total < first:
        raise ValueError("Video is shorter than its first block")
    result = [(0, first)]
    for start in range(first, total, block):
        result.append((start, min(start + block, total)))
    return result

def event_plan(row, total=121, radius=4.0, fps=16.0):
    """Same historical ±4s selection; GU globals remain available untimed."""
    starts, ends, texts, roles = row["prompt"]
    result = []
    for lo, hi in layout(total):
        begin = 0.0 if lo == 0 else (4 * lo - 3) / fps
        end = (4 * (hi - 1) + 1) / fps
        selected = [i for i, (a, b, role) in enumerate(zip(starts, ends, roles))
                    if role == 0 or (b >= begin - radius - 1e-9
                                     and a <= end + radius + 1e-9)]
        if not any(roles[i] > 0 for i in selected):
            raise ValueError("No action condition for a generation block")
        result.append({"start": lo, "stop": hi, "indices": selected,
                       "prompt": [[column[i] for i in selected]
                                  for column in (starts, ends, texts, roles)]})
    return result
