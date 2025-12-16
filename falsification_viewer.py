"""
Minimal GUI launcher to replay crash-prone falsification scenarios with rendering.

Run:
    cd TrajectoryAidedLearning
    python falsification_viewer.py

Select a scenario and click "Run scenario"; a Pyglet window will open and
show the lap until crash/complete. This focuses on a few high-crash cases,
not every hook.
"""

import threading
import tkinter as tk
from tkinter import ttk

import TrajectoryAidedLearning.TestSimulation as ts
from falsify_tal import FalsificationRunner, Scenario


# Focused crash-prone scenarios
SCENARIOS = [
    Scenario(
        name="esp_drop10_seed301",
        map_name="f1_esp",
        mu_scale=1.0,
        pose_jitter=None,
        scan_noise_std=0.0,
        scan_dropout_prob=0.10,
        laps=1,
        seed=301,
        note="ESP, 10% dropout (likely crash)",
    ),
    Scenario(
        name="esp_noise0.10_drop0.05_seed303",
        map_name="f1_esp",
        mu_scale=1.0,
        pose_jitter=None,
        scan_noise_std=0.10,
        scan_dropout_prob=0.05,
        laps=1,
        seed=303,
        note="ESP, 10% noise + 5% dropout",
    ),
    Scenario(
        name="gbr_drop10_seed501",
        map_name="f1_gbr",
        mu_scale=1.0,
        pose_jitter=None,
        scan_noise_std=0.0,
        scan_dropout_prob=0.10,
        laps=1,
        seed=501,
        note="GBR, 10% dropout",
    ),
]


class ViewerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Falsification Viewer")

        ttk.Label(root, text="Select a crash-prone scenario to replay:").pack(
            padx=10, pady=8
        )
        self.listbox = tk.Listbox(root, width=60, height=len(SCENARIOS))
        for idx, sc in enumerate(SCENARIOS):
            self.listbox.insert(
                tk.END, f"{idx+1}. {sc.name} | map={sc.map_name} | note={sc.note}"
            )
        self.listbox.selection_set(0)
        self.listbox.pack(padx=10, pady=4)

        self.run_button = ttk.Button(root, text="Run scenario", command=self.run_selected)
        self.run_button.pack(padx=10, pady=6)

        self.run_default_button = ttk.Button(
            root,
            text="Run default crash (ESP 10% dropout)",
            command=self.run_default,
        )
        self.run_default_button.pack(padx=10, pady=4)

        self.run_gbr_button = ttk.Button(
            root,
            text="Run GBR crash (10% dropout)",
            command=self.run_gbr_default,
        )
        self.run_gbr_button.pack(padx=10, pady=4)

        self.status = ttk.Label(root, text="Idle")
        self.status.pack(padx=10, pady=6)

        # Prepare runner
        ts.SHOW_TEST = True  # enable rendering
        self.runner = FalsificationRunner("TAL_allmaps_shared")
        self.runner.lidar_mitigation = False  # keep raw behavior for replay

    def run_selected(self):
        sel = self.listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        scenario = SCENARIOS[idx]
        self.status.config(text=f"Running {scenario.name} ...")
        threading.Thread(target=self._run_scenario, args=(scenario,), daemon=True).start()

    def run_default(self):
        """Quick-launch the primary crash scenario without selecting."""
        scenario = SCENARIOS[0]  # ESP 10% dropout
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(0)
        self.status.config(text=f"Running {scenario.name} ...")
        threading.Thread(target=self._run_scenario, args=(scenario,), daemon=True).start()

    def run_gbr_default(self):
        """Quick-launch the GBR 10% dropout crash scenario."""
        scenario = SCENARIOS[2]  # GBR 10% dropout
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(2)
        self.status.config(text=f"Running {scenario.name} ...")
        threading.Thread(target=self._run_scenario, args=(scenario,), daemon=True).start()

    def _run_scenario(self, scenario: Scenario):
        res = self.runner.run_scenario(scenario)
        status = (
            f"Done: crashes {res['crashes']}/{res['n_test_laps']}, "
            f"lap_time {res['avg_times']:.2f}, rho {res.get('robustness', 0.0):.3f}"
        )
        self.status.config(text=status)


def main():
    root = tk.Tk()
    ViewerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
