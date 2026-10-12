#!/usr/bin/env python3
"""Run every test_*.py in this directory (no pytest dependency required)."""
import glob, os, subprocess, sys
here = os.path.dirname(__file__)
files = sorted(glob.glob(os.path.join(here, "test_*.py")))
fail = False
for f in files:
    print(f"\n== {os.path.basename(f)} ==")
    r = subprocess.run([sys.executable, f])
    fail = fail or (r.returncode != 0)
sys.exit(1 if fail else 0)
