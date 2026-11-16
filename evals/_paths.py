"""Import path and output folder for the eval scripts."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
EOUT = os.path.join(ROOT, "evals", "out")
os.makedirs(EOUT, exist_ok=True)
