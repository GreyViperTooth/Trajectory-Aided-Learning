"""
Simple launcher GUI to access the controller viewers (dashboard/demo) and the falsification viewer.

Run inside the container (DISPLAY set): python3 launch_menu.py
Buttons spawn the corresponding tool in separate subprocesses.
"""

import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk


class Launcher:
    def __init__(self, root):
        self.root = root
        self.root.title("TAL-X Launch Menu")

        ttk.Label(root, text="Select a viewer to launch:").pack(padx=10, pady=10)

        ttk.Button(root, text="Dashboard (TAL vs PP)", command=self.run_dashboard).pack(
            padx=10, pady=6, fill="x"
        )
        ttk.Button(root, text="Demo presentation", command=self.run_demo).pack(
            padx=10, pady=6, fill="x"
        )
        ttk.Button(root, text="Falsification viewer", command=self.run_falsification).pack(
            padx=10, pady=6, fill="x"
        )

        self.status = ttk.Label(root, text="Idle")
        self.status.pack(padx=10, pady=10)

    def _spawn(self, args):
        # Run subprocess detached; rely on DISPLAY from environment
        subprocess.Popen(args)

    def run_dashboard(self):
        self.status.config(text="Launching dashboard...")
        threading.Thread(target=self._spawn, args=([sys.executable, "demo_dashboard.py"],), daemon=True).start()

    def run_demo(self):
        self.status.config(text="Launching demo presentation...")
        threading.Thread(target=self._spawn, args=([sys.executable, "demo_presentation.py"],), daemon=True).start()

    def run_falsification(self):
        self.status.config(text="Launching falsification viewer...")
        threading.Thread(target=self._spawn, args=([sys.executable, "falsification_viewer.py"],), daemon=True).start()


def main():
    root = tk.Tk()
    Launcher(root)
    root.mainloop()


if __name__ == "__main__":
    main()
