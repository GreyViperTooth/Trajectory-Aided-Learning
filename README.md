# TrajectoryAidedLearning

This repo contains the source code for the paper entitled, "[High-speed Autonomous Racing using Trajectory-aided Deep Reinforcement Learning](https://ieeexplore.ieee.org/document/10182327)"

We present a reward signal that incorporates an optimal trajectory to train deep reinforcement learning agents for high-speed autonomous racing.

![](Data/tal_calculation.png)

Training agents with our reward signal results in significatly improved training performance.
The most noteable performance difference is at high-speeds where previous rewards failed.

![](Data/TAL_vs_baseline_reward.png)

The improved training results in higher average progrresses at high speeds.

![](Data/tal_progress.png)

# Result Generation

The results in the paper are generated through a two step process of:
1. Train and test the agents
2. Process and plot the data

For every test:
- Run calculate_statistics
- Run calculate_averages

## Tests:

### Maximum Speed Investigation

- Aim: Understand how performance changes with different speeds.
- Config files: CthSpeeds, TAL_speeds 
- Results: 
    - Training graph: Cth_TAL_speeds_TrainingGraph
    - Lap times and % success: Cth_TAL_speeds_Barplot

### 6 m/s Performance Comparision 

- Aim: Compare the baseline and TAL on different maps with a maximum speed of 6 m/s.
- Config file: Cth_maps, TAL_maps
- Results:
    - Training graphs: TAL_Cth_maps_TrainingGraph
    - Lap times and success bar plot: TAL_Cth_maps_Barplot

### Speed Profile Analysis 

- Aim: Study the speed profiles
- Requires the pure pursuit (PP_speeds) results
- Results:
    - Trajectories: GenerateVelocityProfiles, set the folder to TAL_speeds
    - Speed profile pp TAL: TAL_speed_profiles
    - Speed profile x3: TAL_speed_profiles 
    - Slip profile: TAL_speed_profiles

### Comparison with Literatures

- Aim: Compare our method with the literature
- Results:
    - Bar plot: LiteratureComparison
- Note that the results from the literature are hard coded.

![](Data/animation.gif)


## Citation

If you find this work useful, please consider citing:
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

## Quick how-to: train and test TAL vs Pure Pursuit

- **Train TAL on multiple maps (map-independent policy)**  
  - Edit `config/TAL_allmaps.yaml` to change maps, seeds (`n`/`set_n`), speed (`max_speed`), or steps (`n_train_steps`).  
  - Run training:  
    ```bash
    cd TrajectoryAidedLearning
    python - <<'PY'
    from TrajectoryAidedLearning.TrainAgents import TrainSimulation
    sim = TrainSimulation("TAL_allmaps")
    sim.run_training_evaluation()
    PY
    ```
  - Checkpoints land under `Data/Vehicles/TAL_allmaps/<run_name>/`.

- **Test the trained TAL controller on any map/speed**  
  - For a quick single-map test edit `config/TAL_maps2.yaml` (`runs:` list and `max_speed`) or reuse `TAL_allmaps.yaml` for the full set.  
  - Run testing locally:  
    ```bash
    cd TrajectoryAidedLearning
    python - <<'PY'
    from TrajectoryAidedLearning.TestSimulation import TestSimulation
    sim = TestSimulation("TAL_maps2")  # or "TAL_allmaps"
    sim.run_testing_evaluation()
    PY
    ```
  - Or via container/noVNC:  
    ```bash
    docker run -d --name tal-demo -p 6080:6080 -p 5900:5900 -e RUN_FILE=TAL_maps2 tal-novnc
    # open http://localhost:6080/ to watch
    ```
- To change test speed: set `max_speed` in the chosen config file. To change map: edit the `runs:` list.

- **Compare to Pure Pursuit baseline**  
  - Use `config/PP_maps8.yaml` (or `PP_speeds.yaml` to sweep speeds).  
  - Run testing:  
    ```bash
    cd TrajectoryAidedLearning
    python - <<'PY'
    from TrajectoryAidedLearning.TestSimulation import TestSimulation
    sim = TestSimulation("PP_maps8")
    sim.run_testing_evaluation()
    PY
    ```
  - For live viewing swap the run file in the container: `-e RUN_FILE=PP_maps8`.

## Quick demo (TAL vs PP on a few maps)

For a short presentation-ready showcase (2 laps each on three maps) run:
```bash
cd TrajectoryAidedLearning
python demo_presentation.py
```
Set `DEMO_LAPS=5` (or similar) to change lap counts, and edit `DEMO_MAPS` inside `demo_presentation.py` if you want different tracks. The script first uses the trained TAL policy from `TAL_allmaps_shared`, then runs the Pure Pursuit baseline (`PP_maps8`).
