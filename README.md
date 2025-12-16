# Trajectory-Aided Learning (re-implementation)

Personal re-implementation and experiments based on the paper **High-Speed Autonomous Racing Using Trajectory-Aided Deep Reinforcement Learning** (Evans et al., RA-L 2023). This repo contains:
- A TAL (trajectory-aided) reward and controller
- Baseline Pure Pursuit comparison
- Training/testing scripts for multiple maps and speeds
- Containerized demo with noVNC viewing

Original paper/code credit: Benjamin David Evans et al. ([paper](https://ieeexplore.ieee.org/document/10182327), [original repo](https://github.com/BDEvan5/TrajectoryAidedLearning)). Please cite the paper when using this work.

## What’s here
- `TrajectoryAidedLearning/`: training + testing code and planners
- `config/`: experiment configs for TAL and PP runs
- `Data/`: saved trajectories/models/results used in my runs (kept small)
- `demo_*.py`, `launch_menu.py`: quick demos/presentations
- `Dockerfile`, `docker-entrypoint.sh`: container with noVNC viewer

## Quickstart
Prereqs: Python 3.8+, pip, and (optionally) Docker. From repo root:
```bash
pip install -r requirements.txt
```

### Run the noVNC demo (container)
```bash
cd TrajectoryAidedLearning
docker build -t tal-novnc .
docker rm -f tal-demo 2>$null
docker run -d --name tal-demo -p 6080:6080 -p 5900:5900 -e DEMO_LAPS=2 tal-novnc
# Open http://localhost:6080 to watch the TAL vs PP runs
```

### Train TAL on multiple maps
```bash
cd TrajectoryAidedLearning
python - <<'PY'
from TrajectoryAidedLearning.TrainAgents import TrainSimulation
sim = TrainSimulation("TAL_allmaps")  # edit config/TAL_allmaps.yaml as needed
sim.run_training_evaluation()
PY
```
Checkpoints/results land under `Data/Vehicles/TAL_allmaps/<run_name>/`.

### Test a trained TAL policy (or PP baseline)
```bash
cd TrajectoryAidedLearning
python - <<'PY'
from TrajectoryAidedLearning.TestSimulation import TestSimulation
sim = TestSimulation("TAL_maps2")  # or "TAL_allmaps", "PP_maps8", etc.
sim.run_testing_evaluation()
PY
```
To change speed or maps, edit the corresponding config (`max_speed`, `runs:` list).

### Presentation demo (short showcase)
```bash
cd TrajectoryAidedLearning
python demo_presentation.py
```
Adjust laps via `DEMO_LAPS` env var or edit `DEMO_MAPS` inside the script.

## Notes
- Data directory is kept small; heavy logs or future rollouts should be added to `.gitignore` if they grow.
- Line endings: repo may normalize to CRLF on Windows when editing.

## Citation (original paper)
If you use this work, please cite:
```
@ARTICLE{10182327,
    author={Evans, Benjamin David and Engelbrecht, Herman Arnold and Jordaan, Hendrik Willem},
    journal={IEEE Robotics and Automation Letters},
    title={High-Speed Autonomous Racing Using Trajectory-Aided Deep Reinforcement Learning},
    year={2023},
    volume={8},
    number={9},
    pages={5353-5359},
    doi={10.1109/LRA.2023.3295252}
}
```
