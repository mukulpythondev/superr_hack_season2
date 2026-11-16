"""Run all steps, or some:  python -m superr_selector.run_pipeline [--from N | --only N]"""
import argparse
import importlib
import time

STEPS = [
    (1, "clean_state", "clean the data and rebuild the match situation before every ball"),
    (2, "pressure",    "win-probability models and the model-based pressure label"),
    (3, "baselines",   "expected runs and wickets, value above expectation, three pieces"),
    (4, "situations",  "cricket pressure situations A-H and the knockout weight"),
    (5, "reliability", "split-half test: how much of the spread between players is real"),
    (6, "ratings",     "open-pool ratings: direct top 10, role-mix top 10, model cross-check"),
    (7, "checks",      "plain-stat sanity check, sensitivity and agreement of the open-pool lists"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="start", type=int, default=1, help="first step to run (1-7)")
    parser.add_argument("--only", type=int, help="run a single step")
    args = parser.parse_args()
    for number, module, what in STEPS:
        if (args.only and number != args.only) or (not args.only and number < args.start):
            continue
        print(f"\n[{number}] {module}: {what}")
        started = time.time()
        importlib.import_module(f"superr_selector.{module}").main()
        print(f"    done in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
