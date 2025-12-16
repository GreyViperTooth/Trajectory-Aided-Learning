"""
Simulation-based falsification harness for the TAL controller.

It perturbs the initial pose, lidar, and friction to search for crash
counterexamples. Results (summary + state/action traces) are stored
under each vehicle folder: Data/Vehicles/<run.path><run_name>/Falsification/.
"""

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import cma

from TrajectoryAidedLearning.TestSimulation import TestSimulation
from TrajectoryAidedLearning.Planners.AgentPlanners import AgentTester
from TrajectoryAidedLearning.Utils.utils import load_conf, setup_run_list
from TrajectoryAidedLearning.Utils.HistoryStructs import VehicleStateHistory
from TrajectoryAidedLearning.f110_gym.f110_env import F110Env

# Default vehicle parameters from F110Env; we copy them so we can scale mu safely
DEFAULT_PARAMS = {
    "mu": 1.0489,
    "C_Sf": 4.718,
    "C_Sr": 5.4562,
    "lf": 0.15875,
    "lr": 0.17145,
    "h": 0.074,
    "m": 3.74,
    "I": 0.04712,
    "s_min": -0.4189,
    "s_max": 0.4189,
    "sv_min": -3.2,
    "sv_max": 3.2,
    "v_switch": 7.319,
    "a_max": 9.51,
    "v_min": -5.0,
    "v_max": 20.0,
    "width": 0.31,
    "length": 0.58,
}


@dataclass
class Scenario:
    name: str
    map_name: str
    mu_scale: float = 1.0
    pose_jitter: Optional[tuple] = None  # (dx, dy, dyaw radians)
    scan_noise_std: float = 0.0
    scan_dropout_prob: float = 0.0
    laps: int = 1
    seed: int = 0
    note: str = ""


class FalsificationRunner(TestSimulation):
    def __init__(self, run_file: str):
        super().__init__(run_file)
        self.conf = load_conf("config_file")
        self.run_lookup: Dict[str, object] = {r.map_name: r for r in self.run_data}
        self.last_rho = None

    def _pick_run(self, map_name: str):
        if map_name not in self.run_lookup:
            raise KeyError(f"Map {map_name} not found in run file")
        return self.run_lookup[map_name]

    def run_scenario(self, scenario: Scenario) -> Dict:
        run = self._pick_run(scenario.map_name)
        # seeds
        seed = scenario.seed
        np.random.seed(seed)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(seed)

        self.noise_rng = np.random.default_rng(seed=seed)
        self.lidar_rng = np.random.default_rng(seed=seed + 1)

        self.scan_noise_std = scenario.scan_noise_std
        self.scan_dropout_prob = scenario.scan_dropout_prob
        self.pose_jitter = scenario.pose_jitter

        env_params = dict(DEFAULT_PARAMS)
        env_params["mu"] = DEFAULT_PARAMS["mu"] * scenario.mu_scale

        self.env = F110Env(map=run.map_name, params=env_params)
        self.map_name = run.map_name
        self.planner = AgentTester(run, self.conf)

        self.n_test_laps = scenario.laps
        self.lap_times = []
        self.completed_laps = 0

        folder = f"Falsification/{scenario.name}/seed_{seed}/"
        self.vehicle_state_history = VehicleStateHistory(run, folder)

        eval_dict = self.run_testing()
        self.env.close_rendering()

        rho = self._compute_robustness()
        self.last_rho = rho

        result = {
            "scenario": asdict(scenario),
            "run_name": run.run_name,
            "map": run.map_name,
            "mu_used": env_params["mu"],
            "success_rate": eval_dict["success_rate"],
            "completed_laps": eval_dict["completed_laps"],
            "n_test_laps": eval_dict["n_test_laps"],
            "avg_times": eval_dict["avg_times"],
            "lap_times": self.lap_times,
            "crashes": scenario.laps - eval_dict["completed_laps"],
            "robustness": rho,
        }
        self._write_summary(run, folder, result)
        return result

    def _write_summary(self, run, folder: str, result: Dict):
        path = Path("Data") / "Vehicles" / run.path / run.run_name / folder
        os.makedirs(path, exist_ok=True)
        with open(path / "summary.json", "w") as f:
            json.dump(result, f, indent=2)

    def _compute_robustness(self) -> float:
        if not self.safety_times or not self.safety_margins:
            return 0.0
        try:
            import rtamt  # local import to tolerate env issues
            spec = rtamt.STLDenseTimeSpecification()
            spec.declare_var("s", "float")
            spec.spec = "alw (s > 0)"
            spec.parse()
            trace = {"s": list(zip(self.safety_times, self.safety_margins))}
            rho_trace = spec.evaluate(trace)
            # robustness at time 0
            rho = float(rho_trace[0][1])
        except Exception as exc:  # noqa: BLE001
            print(f"rtamt robustness failed, using min margin fallback: {exc}")
            rho = float(min(self.safety_margins))
        return rho

    def cma_search(self, map_name: str = "f1_esp", budget: int = 30):
        """
        Run a CMA-ES search over (noise_std, dropout, yaw, mu_scale) to minimize robustness.
        Bounds:
          noise_std in [0, 0.3], dropout in [0, 0.15], yaw in [-12deg, 12deg], mu in [0.6, 1.2]
        """
        deg12 = 12 * math.pi / 180.0
        bounds = [[0.0, 0.0, -deg12, 0.6], [0.3, 0.15, deg12, 1.2]]
        x0 = [0.1, 0.05, 0.0, 1.0]
        sigma = 0.1

        best = {"rho": float("inf"), "params": None, "result": None}

        def evaluate(x):
            # clip to bounds
            noise = float(np.clip(x[0], bounds[0][0], bounds[1][0]))
            drop = float(np.clip(x[1], bounds[0][1], bounds[1][1]))
            yaw = float(np.clip(x[2], bounds[0][2], bounds[1][2]))
            mu = float(np.clip(x[3], bounds[0][3], bounds[1][3]))

            sc = Scenario(
                name=f"cma_noise{noise:.3f}_drop{drop:.3f}_yaw{yaw:.3f}_mu{mu:.3f}",
                map_name=map_name,
                mu_scale=mu,
                pose_jitter=(0.0, 0.0, yaw),
                scan_noise_std=noise,
                scan_dropout_prob=drop,
                laps=1,
                seed=750,  # fixed for determinism during search
                note="CMA-ES search candidate",
            )
            res = self.run_scenario(sc)
            rho = res.get("robustness", 0.0)
            if rho < best["rho"]:
                best["rho"] = rho
                best["params"] = (noise, drop, yaw, mu)
                best["result"] = res
            return rho

        cma.fmin(
            evaluate,
            x0,
            sigma,
            {
                "bounds": bounds,
                "maxfevals": budget,
                "verb_disp": 0,
            },
        )
        return best


def default_scenarios() -> List[Scenario]:
    """
    Hand-picked stress cases for falsification. First entries target f1_esp as requested.
    """
    deg = math.pi / 180.0
    return [
        Scenario(
            name="esp_pose_yaw_noise",
            map_name="f1_esp",
            mu_scale=0.85,
            pose_jitter=(0.5, -0.3, 10 * deg),
            scan_noise_std=0.35,
            scan_dropout_prob=0.1,
            laps=1,
            seed=101,
            note="Moderate friction drop + pose offset + lidar noise/dropout",
        ),
        Scenario(
            name="esp_low_mu_high_dropout",
            map_name="f1_esp",
            mu_scale=0.6,
            pose_jitter=(0.3, 0.2, -8 * deg),
            scan_noise_std=0.2,
            scan_dropout_prob=0.25,
            laps=1,
            seed=102,
            note="Aggressive friction reduction with lidar dropout",
        ),
        Scenario(
            name="esp_baseline_noise",
            map_name="f1_esp",
            mu_scale=1.0,
            pose_jitter=None,
            scan_noise_std=0.15,
            scan_dropout_prob=0.05,
            laps=1,
            seed=103,
            note="Nominal friction, mild sensor noise",
        ),
        Scenario(
            name="gbr_low_mu_yaw",
            map_name="f1_gbr",
            mu_scale=0.65,
            pose_jitter=(0.4, -0.4, 12 * deg),
            scan_noise_std=0.25,
            scan_dropout_prob=0.15,
            laps=1,
            seed=201,
            note="Second map probe with yaw/pose perturbation and low friction",
        ),
    ]


def sweep_esp_thresholds() -> List[Scenario]:
    """
    Lighter perturbation sweep on ESP to find crash thresholds across seeds.
    Gradually increase lidar noise/dropout; no pose jitter; nominal friction.
    """
    scenarios: List[Scenario] = []
    noise_levels = [0.0, 0.1, 0.2, 0.3]
    dropout_levels = [0.0, 0.05, 0.1]
    seeds = [301, 302, 303]

    for noise in noise_levels:
        for dropout in dropout_levels:
            for seed in seeds:
                name = f"esp_sweep_noise{noise:.2f}_drop{dropout:.2f}"
                scenarios.append(
                    Scenario(
                        name=name,
                        map_name="f1_esp",
                        mu_scale=1.0,
                        pose_jitter=None,
                        scan_noise_std=noise,
                        scan_dropout_prob=dropout,
                        laps=1,
                        seed=seed,
                        note="Threshold sweep: nominal friction, varying lidar noise/dropout, no pose jitter",
                    )
                )
    return scenarios


def sweep_esp_pose_jitter() -> List[Scenario]:
    """
    Probe robustness in the robust pocket (noise-only) by adding small pose/yaw jitter.
    Uses noise=0.1, drop=0.0, varied pose offsets/yaw, multiple seeds.
    """
    scenarios: List[Scenario] = []
    # meters, meters, radians
    jitters = [
        (0.0, 0.0, 0.0),
        (0.3, 0.0, 5 * math.pi / 180),
        (-0.3, 0.0, -5 * math.pi / 180),
        (0.3, 0.3, 8 * math.pi / 180),
        (-0.3, -0.3, -8 * math.pi / 180),
        (0.5, -0.2, 10 * math.pi / 180),
    ]
    seeds = [401, 402, 403]
    for jitter in jitters:
        for seed in seeds:
            name = f"esp_pose_jitter_noise0.10_dx{jitter[0]:.2f}_dy{jitter[1]:.2f}_yaw{jitter[2]*180/math.pi:.1f}"
            scenarios.append(
                Scenario(
                    name=name,
                    map_name="f1_esp",
                    mu_scale=1.0,
                    pose_jitter=jitter,
                    scan_noise_std=0.1,
                    scan_dropout_prob=0.0,
                    laps=1,
                    seed=seed,
                    note="Pose/yaw jitter sweep in noise-only robust pocket",
                )
            )
    return scenarios


def sweep_gbr_thresholds() -> List[Scenario]:
    """
    Extend noise/dropout sweep to f1_gbr for comparison.
    """
    scenarios: List[Scenario] = []
    noise_levels = [0.0, 0.1, 0.2]
    dropout_levels = [0.0, 0.05, 0.1]
    seeds = [501, 502, 503]
    for noise in noise_levels:
        for dropout in dropout_levels:
            for seed in seeds:
                name = f"gbr_sweep_noise{noise:.2f}_drop{dropout:.2f}"
                scenarios.append(
                    Scenario(
                        name=name,
                        map_name="f1_gbr",
                        mu_scale=1.0,
                        pose_jitter=None,
                        scan_noise_std=noise,
                        scan_dropout_prob=dropout,
                        laps=1,
                        seed=seed,
                        note="GBR sweep: nominal friction, varying lidar noise/dropout",
                    )
                )
    return scenarios


def sweep_mu_variation(map_name: str, base_noise: float, dropout: float, mu_scales: List[float], seeds: List[int]) -> List[Scenario]:
    """
    Friction robustness sweep at fixed noise/dropout.
    """
    scenarios: List[Scenario] = []
    for mu_scale in mu_scales:
        for seed in seeds:
            name = f"{map_name}_mu{mu_scale:.2f}_noise{base_noise:.2f}_drop{dropout:.2f}"
            scenarios.append(
                Scenario(
                    name=name,
                    map_name=map_name,
                    mu_scale=mu_scale,
                    pose_jitter=None,
                    scan_noise_std=base_noise,
                    scan_dropout_prob=dropout,
                    laps=1,
                    seed=seed,
                    note="Friction sweep at fixed noise/dropout",
                )
            )
    return scenarios


def main():
    runner = FalsificationRunner("TAL_allmaps_shared")
    # First, targeted stress cases
    scenarios = default_scenarios()
    # Then, add a finer sweep on ESP to find failure thresholds
    scenarios.extend(sweep_esp_thresholds())
    # Probe pose jitter around the noise-only robust pocket on ESP
    scenarios.extend(sweep_esp_pose_jitter())
    # Extend sweeps to another map for comparison (f1_gbr)
    scenarios.extend(sweep_gbr_thresholds())
    # Friction (mu) sensitivity sweeps at the stable noise-only setting and a brittle dropout setting
    mu_scales = [0.6, 0.8, 1.0, 1.2]
    scenarios.extend(sweep_mu_variation("f1_esp", base_noise=0.10, dropout=0.0, mu_scales=mu_scales, seeds=[601, 602, 603]))
    scenarios.extend(sweep_mu_variation("f1_esp", base_noise=0.10, dropout=0.05, mu_scales=mu_scales, seeds=[611, 612]))
    scenarios.extend(sweep_mu_variation("f1_gbr", base_noise=0.10, dropout=0.0, mu_scales=mu_scales, seeds=[621, 622, 623]))
    scenarios.extend(sweep_mu_variation("f1_gbr", base_noise=0.10, dropout=0.05, mu_scales=mu_scales, seeds=[631, 632]))

    results = []
    for sc in scenarios:
        print(f"\n--- Running scenario: {sc.name} on {sc.map_name} (seed {sc.seed}) ---")
        res = runner.run_scenario(sc)
        print(
            f"  crashes: {res['crashes']}/{res['n_test_laps']} | "
            f"success_rate: {res['success_rate']:.1f}% | "
            f"avg_lap_time: {res['avg_times']:.2f} | "
            f"rho: {res.get('robustness', 0.0):.3f}"
        )
        results.append(res)

    print("\n=== CMA-ES search on ESP (noise, dropout, yaw, mu) ===")
    best = runner.cma_search(map_name="f1_esp", budget=30)
    if best["result"]:
        res = best["result"]
        print(
            f"Best CMA rho: {best['rho']:.3f} | "
            f"params (noise,drop,yaw,mu): {best['params']} | "
            f"crashes {res['crashes']}/{res['n_test_laps']} | "
            f"seed {res['scenario']['seed']}"
        )
        results.append(res)

    print("\nSummary:")
    for r in results:
        status = "CRASH" if r["crashes"] > 0 else "OK"
        print(
            f"{status} | {r['scenario']['name']} | map {r['map']} | "
            f"mu {r['mu_used']:.3f} | seed {r['scenario']['seed']} | "
            f"crashes {r['crashes']}/{r['n_test_laps']} | "
            f"rho {r.get('robustness', 0.0):.3f}"
        )


if __name__ == "__main__":
    main()
