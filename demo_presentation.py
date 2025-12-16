"""Quick demo runner for presentations.

Runs the trained TAL (TD3) policy on a couple of maps, then runs the Pure
Pursuit baseline on the same maps. Keep it short by default (2 laps per map)
so it finishes quickly; override laps with DEMO_LAPS env var.
"""

import os
from pathlib import Path

import TrajectoryAidedLearning.TestSimulation as ts
from TrajectoryAidedLearning.TestSimulation import TestSimulation

# Maps to showcase; tweak as you like.
DEMO_MAPS = ["f1_aut", "f1_gbr", "f1_mco"]
# How many laps per map to run (short on purpose for a quick demo).
DEMO_LAPS = int(os.environ.get("DEMO_LAPS", 2))

# Config files to reuse for trained TAL and baseline PP.
TAL_RUN_FILE = "TAL_allmaps_shared"
PP_RUN_FILE = "PP_maps8"


def _filter_runs(sim: TestSimulation, maps):
    """Limit runs to selected maps and shorten lap counts."""
    sim.run_data = [r for r in sim.run_data if r.map_name in maps]
    for run in sim.run_data:
        run.n_test_laps = DEMO_LAPS
    return sim


def _verify_tal_checkpoints(run_data):
    """Ensure the trained TAL checkpoints exist before running."""
    missing = []
    for run in run_data:
        actor_path = Path("Data") / "Vehicles" / run.path / run.run_name / f"{run.run_name}_actor.pth"
        if not actor_path.exists():
            missing.append((run.map_name, actor_path))
    if missing:
        print("Missing TAL checkpoints for:")
        for map_name, actor_path in missing:
            print(f"  {map_name}: {actor_path}")
        raise SystemExit("Train TAL first (TrainAgents.py) before running the demo.")


def run_tal():
    print("\n=== TAL demo (trained policy) ===")
    tal = _filter_runs(TestSimulation(TAL_RUN_FILE), DEMO_MAPS)
    _verify_tal_checkpoints(tal.run_data)
    tal.run_testing_evaluation()


def run_pp():
    print("\n=== Pure Pursuit baseline ===")
    pp = _filter_runs(TestSimulation(PP_RUN_FILE), DEMO_MAPS)
    pp.run_testing_evaluation()


def main():
    # Ensure rendering is visible in the VNC session
    ts.SHOW_TEST = True
    print(f"Running {DEMO_LAPS} lap(s) per map on: {', '.join(DEMO_MAPS)}")
    run_tal()
    run_pp()


if __name__ == "__main__":
    main()
