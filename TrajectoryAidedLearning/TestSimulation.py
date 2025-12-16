from TrajectoryAidedLearning.f110_gym.f110_env import F110Env
from TrajectoryAidedLearning.Utils.utils import *
from TrajectoryAidedLearning.Utils.HistoryStructs import VehicleStateHistory

from TrajectoryAidedLearning.Planners.PurePursuit import PurePursuit
from TrajectoryAidedLearning.Planners.AgentPlanners import AgentTester



import torch
import numpy as np
import time
from pathlib import Path

# settings
SHOW_TRAIN = False
SHOW_TEST = False
# SHOW_TEST = True
VERBOSE = True
LOGGING = True


class TestSimulation():
    def __init__(self, run_file: str):
        self.run_data = setup_run_list(run_file)
        self.conf = load_conf("config_file")

        self.env = None
        self.planner = None
        
        self.n_test_laps = None
        self.lap_times = None
        self.completed_laps = None
        self.prev_obs = None
        self.prev_action = None

        self.std_track = None
        self.map_name = None
        self.reward = None
        self.noise_rng = None
        self.vehicle_radius = 0.30  # approximate footprint radius for safety margin

        # falsification / perturbation hooks (defaults are no-op)
        self.noise_std = 0.0
        self.scan_noise_std = 0.0
        self.scan_dropout_prob = 0.0
        self.lidar_rng = None
        self.pose_jitter = None
        self.lidar_mitigation = False
        self.lidar_floor = 0.30
        self.safety_times = []
        self.safety_margins = []

        # flags 
        self.vehicle_state_history = None

    def run_testing_evaluation(self):
        for run in self.run_data:
            print(run)
            print("_________________________________________________________")
            print(run.run_name)
            print("_________________________________________________________")
            seed = run.random_seed + 10*run.n
            np.random.seed(seed) # repetition seed
            torch.use_deterministic_algorithms(True)
            torch.manual_seed(seed)

            if run.noise_std > 0:
                self.noise_std = run.noise_std
                self.noise_rng = np.random.default_rng(seed=seed)

            self.env = F110Env(map=run.map_name)
            self.map_name = run.map_name

            if run.architecture == "PP": 
                planner = PurePursuit(self.conf, run)
            elif run.architecture == "fast": 
                planner = AgentTester(run, self.conf)
            else: raise AssertionError(f"Planner {run.planner} not found")

            if run.test_mode == "Std": self.planner = planner
            else: raise AssertionError(f"Test mode {run.test_mode} not found")

            self.vehicle_state_history = VehicleStateHistory(run, "Testing/")

            self.n_test_laps = run.n_test_laps
        self.lap_times = []
        self.completed_laps = 0
        self.safety_times = []
        self.safety_margins = []

        eval_dict = self.run_testing()
        run_dict = vars(run)
        run_dict.update(eval_dict)

        save_conf_dict(run_dict)

        self.env.close_rendering()

    def run_testing(self):
        assert self.env != None, "No environment created"
        start_time = time.time()

        for i in range(self.n_test_laps):
            observation = self.reset_simulation()
            if self.safety_times is None:
                self.safety_times = []
                self.safety_margins = []

            while not observation['colision_done'] and not observation['lap_done']:
                action = self.planner.plan(observation)
                observation = self.run_step(action)
                if SHOW_TEST: self.env.render('human_fast')

            self.planner.lap_complete()
            if observation['lap_done']:
                if VERBOSE: print(f"Lap {i} Complete in time: {observation['current_laptime']}")
                self.lap_times.append(observation['current_laptime'])
                self.completed_laps += 1

            if observation['colision_done']:
                if VERBOSE: print(f"Lap {i} Crashed in time: {observation['current_laptime']}")
                    

            if self.vehicle_state_history: self.vehicle_state_history.save_history(i, test_map=self.map_name)

        print(f"Tests are finished in: {time.time() - start_time}")

        success_rate = (self.completed_laps / (self.n_test_laps) * 100)
        if len(self.lap_times) > 0:
            avg_times, std_dev = np.mean(self.lap_times), np.std(self.lap_times)
        else:
            avg_times, std_dev = 0, 0

        print(f"Crashes: {self.n_test_laps - self.completed_laps} VS Completes {self.completed_laps} --> {success_rate:.2f} %")
        print(f"Lap times Avg: {avg_times} --> Std: {std_dev}")

        eval_dict = {}
        eval_dict['success_rate'] = float(success_rate)
        eval_dict['avg_times'] = float(avg_times)
        eval_dict['std_dev'] = float(std_dev)
        eval_dict['completed_laps'] = int(self.completed_laps)
        eval_dict['n_test_laps'] = int(self.n_test_laps)
        eval_dict['min_safety_margin'] = float(min(self.safety_margins)) if len(self.safety_margins) > 0 else 0.0

        return eval_dict

    # this is an overide
    def run_step(self, action):
        sim_steps = self.conf.sim_steps
        if self.vehicle_state_history: 
            self.vehicle_state_history.add_action(action)
        self.prev_action = action

        sim_steps, done = sim_steps, False
        while sim_steps > 0 and not done:
            obs, step_reward, done, _ = self.env.step(action[None, :])
            sim_steps -= 1
        
        observation = self.build_observation(obs, done)
        
        return observation

    def build_observation(self, obs, done):
        """Build observation

        Returns 
            state:
                [0]: x
                [1]: y
                [2]: yaw
                [3]: v
                [4]: steering
            scan:
                Lidar scan beams 
            
        """
        observation = {}
        observation['current_laptime'] = obs['lap_times'][0]
        scan = obs['scans'][0]

        # optional lidar perturbations (noise/dropout) for falsification
        if (self.scan_noise_std > 0 or self.scan_dropout_prob > 0) and self.lidar_rng is None:
            # tie the lidar rng to the main noise rng if available for reproducibility
            self.lidar_rng = self.noise_rng if self.noise_rng is not None else np.random.default_rng()
        if self.scan_dropout_prob > 0 and self.lidar_rng is not None:
            mask = self.lidar_rng.random(scan.shape) < self.scan_dropout_prob
            scan = np.where(mask, 0.0, scan)
        if self.scan_noise_std > 0 and self.lidar_rng is not None:
            scan = scan + self.lidar_rng.normal(scale=self.scan_noise_std, size=scan.shape)
            scan = np.clip(scan, 0.0, None)
        # simple mitigation for dropout: fill zeros with median and floor to avoid spurious zeros
        if self.lidar_mitigation:
            if np.any(scan <= 1e-6):
                nz = scan[scan > 1e-6]
                fill = np.median(nz) if nz.size > 0 else 0.0
                scan = np.where(scan <= 1e-6, fill, scan)
            if self.lidar_floor is not None:
                scan = np.clip(scan, self.lidar_floor, None)

        observation['scan'] = scan #TODO: introduce slicing here
        
        if self.noise_rng:
            noise = self.noise_rng.normal(scale=self.noise_std, size=2)
        else: noise = np.zeros(2)
        pose_x = obs['poses_x'][0] + noise[0]
        pose_y = obs['poses_y'][0] + noise[1]
        theta = obs['poses_theta'][0]
        linear_velocity = obs['linear_vels_x'][0]
        steering_angle = obs['steering_deltas'][0]
        state = np.array([pose_x, pose_y, theta, linear_velocity, steering_angle])

        observation['state'] = state
        observation['lap_done'] = False
        observation['colision_done'] = False

        observation['reward'] = 0.0
        # safety margin proxy: closest obstacle distance minus vehicle radius
        safety_margin = float(np.min(scan) - self.vehicle_radius)
        observation['safety_margin'] = safety_margin
        if self.safety_times is not None and self.safety_margins is not None:
            # use simulated time as time axis for monitoring; each plan step advances sim_steps * timestep
            current_time = len(self.safety_times) * (self.env.timestep * self.conf.sim_steps)
            self.safety_times.append(current_time)
            self.safety_margins.append(safety_margin)

        if done and obs['lap_counts'][0] == 0: 
            observation['colision_done'] = True
        if self.std_track is not None:
            if self.std_track.check_done(observation) and obs['lap_counts'][0] == 0:
                observation['colision_done'] = True

            if self.prev_obs is None: observation['progress'] = 0
            elif self.prev_obs['lap_done'] == True: observation['progress'] = 0
            else: observation['progress'] = max(self.std_track.calculate_progress_percent(state[0:2]), self.prev_obs['progress'])
            # self.racing_race_track.plot_vehicle(state[0:2], state[2])
            # taking the max progress
            

        if obs['lap_counts'][0] == 1:
            observation['lap_done'] = True

        if self.reward:
            observation['reward'] = self.reward(observation, self.prev_obs, self.prev_action)

        if self.vehicle_state_history:
            self.vehicle_state_history.add_state(obs['full_states'][0])

        return observation

    def reset_simulation(self):
        reset_pose = np.zeros(3)
        if self.pose_jitter is not None:
            dx, dy, dyaw = self.pose_jitter
            reset_pose = reset_pose + np.array([dx, dy, dyaw])
        reset_pose = reset_pose[None, :]

        obs, step_reward, done, _ = self.env.reset(reset_pose)

        if SHOW_TRAIN: self.env.render('human_fast')

        self.prev_obs = None
        self.safety_times = []
        self.safety_margins = []
        observation = self.build_observation(obs, done)
        # self.prev_obs = observation
        if self.std_track is not None:
            self.std_track.max_distance = 0.0

        return observation


def main():
    # run_file = "PP_speeds"
    run_file = "PP_maps8"
    # run_file = "Eval_RewardsSlow"
    
    
    sim = TestSimulation(run_file)
    sim.run_testing_evaluation()


if __name__ == '__main__':
    main()


