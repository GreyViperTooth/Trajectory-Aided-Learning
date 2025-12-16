#!/bin/bash
set -e

export DISPLAY=:0
export RUN_FILE=${RUN_FILE:-menu}  # default to launcher menu
export DEMO_LAPS=${DEMO_LAPS:-2}
XVFB_WHD=${XVFB_WHD:-"1280x720x24"}

# Start virtual display
Xvfb :0 -screen 0 $XVFB_WHD -ac +extension GLX +render -noreset &
sleep 2

# Lightweight window manager for window placement
fluxbox > /tmp/fluxbox.log 2>&1 &

# VNC server bridged to the virtual display
x11vnc -display :0 -nopw -forever -shared -rfbport 5900 -quiet > /tmp/x11vnc.log 2>&1 &

# Websockify/noVNC endpoint on :6080
websockify --web=/usr/share/novnc/ 6080 localhost:5900 > /tmp/websockify.log 2>&1 &

# Run the TAL simulation with rendering enabled
cd /app/TrajectoryAidedLearning
mkdir -p Data/Vehicles

# pre-create run folders so utils can write logs
python3 - <<'PY'
import os
from TrajectoryAidedLearning.Utils.utils import setup_run_list

run_file = os.environ.get("RUN_FILE", "dashboard")
targets = []
if run_file in ["demo", "dashboard", "viewer", "menu"]:
    targets = ["TAL_allmaps_shared", "PP_maps8"]
else:
    targets = [run_file]

os.makedirs("Data/Vehicles", exist_ok=True)
for rf in targets:
    runs = setup_run_list(rf)
    for run in runs:
        base = os.path.join("Data", "Vehicles", run.path)
        os.makedirs(base, exist_ok=True)
        os.makedirs(os.path.join(base, run.run_name, "Testing"), exist_ok=True)
PY

python3 - <<'PY'
import os
import TrajectoryAidedLearning.TestSimulation as ts

run_file = os.environ.get("RUN_FILE", "dashboard")
ts.SHOW_TEST = True
if run_file == "demo":
    from demo_presentation import main
    main()
elif run_file == "dashboard":
    from demo_dashboard import main
    main()
elif run_file == "viewer":
    from falsification_viewer import main
    main()
elif run_file == "menu":
    from launch_menu import main
    main()
else:
    sim = ts.TestSimulation(run_file)
    sim.run_testing_evaluation()
PY

echo "Simulation finished; keeping container alive for viewing."
tail -f /dev/null
