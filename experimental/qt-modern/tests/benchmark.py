#!/usr/bin/env python3
"""Reproducible CPU-only comparisons; medians of independent process runs.
Run: python3 tests/benchmark.py build/xaos-bench docs/benchmark-results.json
"""
import argparse
import platform
import csv
import io
import json
import statistics
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("binary", nargs="?", default="build/xaos-bench")
parser.add_argument("output", nargs="?", type=Path, default=Path("benchmark-results.json"))
parser.add_argument("--repeats", type=int, default=7)
options = parser.parse_args()
if options.repeats < 1:
    parser.error("--repeats must be positive")
binary, output = options.binary, options.output
base = ["--width", "640", "--height", "400", "--center-re", "-0.7435",
        "--center-im", "0.1314", "--span", "0.005", "--iterations", "1024"]
cases = {
    "scalar_1_worker": base + ["--threads", "1", "--counts", "--scalar"],
    "avx2_1_worker": base + ["--threads", "1", "--counts"],
    "avx2_4_workers": base + ["--threads", "4", "--counts"],
    "adaptive_zoom": base + ["--threads", "4", "--counts", "--frames", "20", "--zoom", "0.98"],
    "uniform_zoom": base + ["--threads", "4", "--counts", "--frames", "20", "--zoom", "0.98", "--uniform"],
    "raise_counts": base + ["--threads", "4", "--counts", "--limits", "256,512,1024,2048"],
    "raise_state": base + ["--threads", "4", "--state", "--limits", "256,512,1024,2048"],
    "gmp_counts": ["--width", "128", "--height", "80", "--precision", "256", "--threads", "4",
                   "--counts", "--formula", "julia", "--limits", "128,256,512"],
    "gmp_state": ["--width", "128", "--height", "80", "--precision", "256", "--threads", "4",
                  "--state", "--formula", "julia", "--limits", "128,256,512"],
}
result = {"repeats": options.repeats, "environment": {"platform": platform.system() + " " + platform.release()}, "cases": {}}
for name, args in cases.items():
    runs = []
    for _ in range(result["repeats"]):
        completed = subprocess.run([binary, *args], check=True, text=True, capture_output=True, timeout=90)
        rows = list(csv.DictReader(io.StringIO(completed.stdout)))
        if not rows or any(int(row["pending"]) for row in rows):
            raise RuntimeError(f"Incomplete benchmark frame: {name}")
        runs.append(rows)
    frames = []
    for i in range(len(runs[0])):
        rows = [run[i] for run in runs]
        frames.append({"frame": i, "median_ms": statistics.median(float(row["ms"]) for row in rows),
                       "min_ms": min(float(row["ms"]) for row in rows),
                       "max_ms": max(float(row["ms"]) for row in rows),
                       **{key: rows[0][key] for key in ["backend", "bits", "limit", "reused", "started", "resumed", "steps", "simd"]}})
    result["cases"][name] = {"command": [binary, *args], "frames": frames,
                              "median_total_ms": statistics.median(sum(float(row["ms"]) for row in run) for run in runs)}
    print(name, result["cases"][name]["median_total_ms"], flush=True)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(result, indent=2) + "\n")
