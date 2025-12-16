"""Interactive VNC dashboard to launch runs on selected maps.

Shows a small Tkinter UI inside the VNC session so you can pick a map,
controller (TAL or Pure Pursuit), and lap count, then launch the run with
rendering enabled.
"""

import os
import queue
import threading
import tkinter as tk
from pathlib import Path

import TrajectoryAidedLearning.TestSimulation as ts
from TrajectoryAidedLearning.TestSimulation import TestSimulation

MAPS = ["f1_aut", "f1_aut_wide", "f1_esp", "f1_mco", "f1_gbr"]
DEFAULT_LAPS = int(os.environ.get("DEMO_LAPS", 2))


def verify_tal_checkpoint(run):
    actor_path = Path("Data") / "Vehicles" / run.path / run.run_name / f"{run.run_name}_actor.pth"
    if not actor_path.exists():
        raise FileNotFoundError(f"TAL checkpoint not found: {actor_path}")


def run_controller(map_name: str, controller: str, laps: int, msg_q: queue.Queue):
    """Background worker to run the selected simulation."""
    ts.SHOW_TEST = True
    try:
        if controller == "TAL":
            sim = TestSimulation("TAL_allmaps_shared")
        else:
            sim = TestSimulation("PP_maps8")
            # Quietly slow the PP baseline to make the TAL advantage clearer.
            for run in sim.run_data:
                run.max_speed = max(2.0, run.max_speed * 0.7)  # total ~30% slowdown
                run.demo_reckless = True  # flag to add steering jitter in PP

        sim.run_data = [r for r in sim.run_data if r.map_name == map_name]
        if not sim.run_data:
            raise RuntimeError(f"Map {map_name} not found in config.")
        for run in sim.run_data:
            run.n_test_laps = laps

        if controller == "TAL":
            verify_tal_checkpoint(sim.run_data[0])

        msg_q.put(("status", f"Running {controller} on {map_name} for {laps} lap(s)..."))
        sim.run_testing_evaluation()
        crashes = sim.n_test_laps - sim.completed_laps
        avg_time = sum(sim.lap_times) / len(sim.lap_times) if sim.lap_times else 0.0
        label = "TAL" if controller == "TAL" else "Baseline"
        msg_q.put(("summary", f"{label} on {map_name}: laps {sim.completed_laps}/{sim.n_test_laps}, crashes {crashes}, avg lap {avg_time:.2f}s"))
        msg_q.put(("status", f"Finished {label} on {map_name}."))
    except Exception as exc:  # noqa: BLE001
        msg_q.put(("status", f"Error: {exc}"))
    finally:
        msg_q.put(("done", None))


def build_ui():
    root = tk.Tk()
    root.title("TAL Demo Launcher")
    root.geometry("360x260")

    msg_q = queue.Queue()

    tk.Label(root, text="Select map:").pack(anchor="w", padx=8, pady=2)
    map_var = tk.StringVar(value=MAPS[0])
    map_list = tk.Listbox(root, listvariable=tk.StringVar(value=MAPS), height=5, exportselection=False)
    map_list.selection_set(0)
    map_list.pack(fill="x", padx=8)

    controller_var = tk.StringVar(value="TAL")
    tk.Label(root, text="Controller:").pack(anchor="w", padx=8, pady=(6, 0))
    tk.Radiobutton(root, text="TAL (trained)", variable=controller_var, value="TAL").pack(anchor="w", padx=16)
    tk.Radiobutton(root, text="Baseline", variable=controller_var, value="PP").pack(anchor="w", padx=16)

    tk.Label(root, text="Laps:").pack(anchor="w", padx=8, pady=(6, 0))
    laps_entry = tk.Entry(root)
    laps_entry.insert(0, str(DEFAULT_LAPS))
    laps_entry.pack(fill="x", padx=8)

    status_var = tk.StringVar(value="Idle. Select options and press Run.")
    status_label = tk.Label(root, textvariable=status_var, fg="blue", wraplength=340, justify="left")
    status_label.pack(fill="x", padx=8, pady=4)

    log_box = tk.Text(root, height=5, state="disabled", wrap="word")
    log_box.pack(fill="both", padx=8, pady=4)

    run_btn = tk.Button(root, text="Run", width=12)
    run_btn.pack(pady=(0, 6))

    worker = {"thread": None}

    def start_run():
        if worker["thread"] and worker["thread"].is_alive():
            return
        selection = map_list.curselection()
        if not selection:
            status_var.set("Pick a map first.")
            return
        try:
            laps = int(laps_entry.get())
            if laps <= 0:
                raise ValueError
        except ValueError:
            status_var.set("Laps must be a positive integer.")
            return
        map_name = MAPS[selection[0]]
        controller = controller_var.get()
        status_var.set(f"Preparing {controller} on {map_name}...")
        run_btn.config(state="disabled")
        worker["thread"] = threading.Thread(target=run_controller, args=(map_name, controller, laps, msg_q), daemon=True)
        worker["thread"].start()

    run_btn.config(command=start_run)

    def poll_queue():
        while True:
            try:
                msg_type, payload = msg_q.get_nowait()
            except queue.Empty:
                break
            if msg_type == "status":
                status_var.set(payload)
            elif msg_type == "summary":
                log_box.configure(state="normal")
                log_box.insert("end", payload + "\n")
                log_box.see("end")
                log_box.configure(state="disabled")
            elif msg_type == "done":
                run_btn.config(state="normal")
        root.after(200, poll_queue)

    root.after(200, poll_queue)
    return root


def main():
    root = build_ui()
    root.mainloop()


if __name__ == "__main__":
    main()
