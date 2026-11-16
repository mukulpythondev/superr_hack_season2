"""Paths. Set SUPERR_DATA to use a data pack stored somewhere else."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("SUPERR_DATA", os.path.join(ROOT, "data", "raw", "Data Pack"))

OUTPUTS    = os.path.join(ROOT, "outputs")
INTERIM    = os.path.join(OUTPUTS, "interim")       # large ball-by-ball tables, not in git
DIAG       = os.path.join(OUTPUTS, "diagnostics")   # reliability, skill spread, sensitivity
OPEN_POOL  = os.path.join(OUTPUTS, "open_pool")     # rankings over every eligible player in the data

for _d in (INTERIM, DIAG, OPEN_POOL):
    os.makedirs(_d, exist_ok=True)
