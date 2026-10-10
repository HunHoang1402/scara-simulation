#!/usr/bin/env python3
"""
SCARA Robot Control Panel v5.0
 - Changed: minimal UI (standard widgets, neutral buttons; red only for STOP)
 - Added: light / dark theme toggle (remembered in panel_settings.json)
 - Fixed: Z no longer sags 0.14 mm under gravity (gravcomp in scara_real.xml);
          no segfault/hang on close (viewer thread is joined before exit)
 - Added: joint-limit-aware IK, elbow selector, speed override, Linear (XYZ) paths
 - Added: waypoint edit dialog, 3-field status bar, measured vs commanded plots
 - Added: tooltips, path progress bar + active row highlight, soft-limit colors
 - Fixed: Path ">> Go" moves smoothly; Z jog works at Home; Stop/Run race
 - Added: global STOP/Home bar, keyboard jog, click-to-move on top view
 - Added: ghost arm (commanded pose) + tracking error readout
 - Fixed: Absolute XYZ boxes no longer overwritten while typing
 - Fixed: Joint tab entries follow the robot; Linux fonts (DejaVu)
 - Fixed: top view redraws only moving items (no flicker)
 - Fixed: emoji -> plain text (Linux compat)
 - Fixed: trail filtering (no zigzag on fast drag)
 - Fixed: window resizable for trajectory viewing
"""

import os
# Native-Wayland GLFW can block forever in eglSwapBuffers, which made the viewer
# thread hang/segfault on exit. Use the X11 build (runs under XWayland) unless the
# user explicitly chose a variant.
os.environ.setdefault("PYGLFW_LIBRARY_VARIANT", "x11")

import mujoco
import mujoco.viewer
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import math
import time
import collections
import queue
import numpy as np
import json
import re

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# ─── Constants ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(BASE_DIR, "mujoco_sim", "scara_real.xml")
WAYPOINTS_PATH = os.path.join(BASE_DIR, "waypoints.json")
SETTINGS_PATH  = os.path.join(BASE_DIR, "panel_settings.json")

TRAIL_MIN_DIST = 0.0005   # 0.5mm minimum distance between trail points

UI_FONT   = "DejaVu Sans"
MONO_FONT = "DejaVu Sans Mono"

from scara_kinematics import (L1, L2, JOINT_LIMITS, DEAD_ZONE, within_limits,  # noqa: F401 (re-exported)
                              wrap_angle, ik_nearest, ik_locked)
import scara_kinematics as kin

def fk(j1, j2, j3):
    """Tool position (x, y, z) in the base frame {0} (see scara_kinematics)."""
    return kin.fk_position(j1, j2, j3)

def tool_phi(q):
    """Tool angle phi = theta1 + theta2 + theta4 (wrapped)."""
    return wrap_angle(q[0] + q[1] + q[3])

# ─── Motion Profiles (scara_trajectory) ──────────────────────────────────────
from scara_trajectory import (PROFILES, profile_linear, profile_trapezoidal,  # noqa: F401
                              profile_scurve)
import scara_trajectory as traj_mod
import scara_dynamics as dyn
import scara_control as ctl
import shapes

# ─── Shared state ─────────────────────────────────────────────────────────────
class RobotState:
    def __init__(self):
        self.lock = threading.Lock()
        self.ctrl = [0.0]*4          # desired joints q_d (written by GUI/motion threads)
        self.ctrl_vel = [0.0]*4      # desired joint velocity qd_d
        self.ctrl_acc = [0.0]*4      # desired joint acceleration qdd_d
        self.ctrl_time = 0.0         # time.monotonic() when the setpoint was written
        self.qpos = [0.0]*4          # measured joints (from the sim)
        self.qvel = [0.0]*4
        self.tau = [0.0]*4           # actuator torque/force: PID output, or the torque the
                                     # reference needs (inverse dynamics) in kinematic mode
        self.saturated = [False]*4
        self.mode = "control"        # "control" (PID on the torque plant) | "kinematic"
        self.pid = None              # scara_control.PID4, built by sim_thread / PID tab
        self.params = None           # scara_dynamics.RobotParams
        self.show_frames = True      # draw frames {0} and {tool} in the 3D viewer
        self.show_trail3d = True     # draw the tool-tip trail in the 3D viewer
        self.clear_trail3d = False   # request from the GUI to erase the 3D trail
        self.running = True

    def set_setpoint(self, q, qd=None, qdd=None):
        """Caller holds the lock. Velocity/acceleration default to zero (static target)."""
        self.ctrl[:] = list(q)
        self.ctrl_vel[:] = list(qd) if qd is not None else [0.0]*4
        self.ctrl_acc[:] = list(qdd) if qdd is not None else [0.0]*4
        self.ctrl_time = time.monotonic()
state = RobotState()

# ─── Trajectory recorder ─────────────────────────────────────────────────────
class TrajectoryRecorder:
    """Records commanded (ctrl) and measured (qpos) joints and the actuator torques.
    Velocities/accelerations use the analytic values when given (desired: from the
    trajectory; measured velocity: from the sim), otherwise finite differences."""
    SERIES = ("cmd", "meas")
    MIN_DT = 0.002   # skip samples closer than this (they make derivative spikes)

    def __init__(self):
        self.lock = threading.Lock()
        self.clear()
    def clear(self):
        with self.lock:
            self.time_data = []
            self.data = {k: {"q": [[] for _ in range(4)], "qd": [[] for _ in range(4)],
                             "qdd": [[] for _ in range(4)]} for k in self.SERIES}
            self.tau = [[] for _ in range(4)]
            self.recording = False
            self._t0 = 0; self._prev_t = None
    def start(self):
        self.clear()
        with self.lock: self.recording = True; self._t0 = time.time()
    def stop(self):
        with self.lock: self.recording = False
    @staticmethod
    def _append(d, q_list, dt, qd_list=None, qdd_list=None):
        for i in range(4):
            q, qd_hist = d["q"][i], d["qd"][i]
            prev_q = q[-1] if q else None
            q.append(q_list[i])
            if qd_list is not None:
                qd = qd_list[i]
            else:
                qd = (q_list[i] - prev_q) / dt if prev_q is not None and dt > 1e-6 else 0.0
            if qdd_list is not None:
                qdd = qdd_list[i]
            else:
                qdd = (qd - qd_hist[-1]) / dt if qd_hist and dt > 1e-6 else 0.0
            qd_hist.append(qd); d["qdd"][i].append(qdd)
    def record(self, cmd, meas, meas_vel=None, cmd_vel=None, cmd_acc=None, tau=None):
        """meas_vel: measured joint velocity from the sim; cmd_vel/cmd_acc: desired
        derivatives from the trajectory (preferred over differencing)."""
        with self.lock:
            if not self.recording: return
            t = time.time() - self._t0
            dt = t - self._prev_t if self._prev_t is not None else 0.0
            if self._prev_t is not None and dt < self.MIN_DT: return
            self.time_data.append(t)
            self._append(self.data["cmd"], cmd, dt, cmd_vel, cmd_acc)
            self._append(self.data["meas"], meas, dt, meas_vel)
            for i in range(4): self.tau[i].append(tau[i] if tau is not None else 0.0)
            self._prev_t = t
    def get_data(self):
        """Returns (t, {"cmd"|"meas": {"q"|"qd"|"qdd": [4 lists]}}).  Torques: get_tau()."""
        with self.lock:
            return (list(self.time_data),
                    {k: {n: [list(v) for v in d[n]] for n in d} for k, d in self.data.items()})
    def get_tau(self):
        with self.lock:
            return [list(v) for v in self.tau]
recorder = TrajectoryRecorder()

_params_cache = []

def get_params():
    """Dynamic parameters of the robot (read once from the MuJoCo model)."""
    if not _params_cache:
        _params_cache.append(dyn.params_from_model(mujoco.MjModel.from_xml_path(XML_PATH)))
    return _params_cache[0]

# ─── MuJoCo sim thread ───────────────────────────────────────────────────────
viewer_threads = []   # MuJoCo viewer render thread(s), filled by sim_thread
FF_EVERY = 5          # recompute feed-forward every 5 steps (setpoints change at 100 Hz)
TRAIL3D_LEN = 300

def _world(model, data, x, y, z):
    """Point of the base frame {0} -> MuJoCo world ({0} is world rotated by pi about z)."""
    o = data.xanchor[model.joint("joint1").id]
    return np.array([o[0] - x, o[1] - y, o[2] + z])

def _tool_tip_offset(model, data):
    """Height of the bottom of the tool shaft above {0}'s origin at d3 = 0 (from the mesh)."""
    b = model.body("link3_and_4").id
    g = next(i for i in range(model.ngeom) if model.geom_bodyid[i] == b)
    c, h = model.geom_aabb[g][:3], model.geom_aabb[g][3:]
    R = data.geom_xmat[g].reshape(3, 3)
    zs = [(data.geom_xpos[g] + R @ (c + h * np.array([sx, sy, sz])))[2]
          for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    return min(zs) - data.xanchor[model.joint("joint1").id][2] - data.qpos[2]

def _draw_overlays(viewer, model, data, trail, tip_dz):
    """Frames {0} and {tool} (x red, y green, z blue) and the tool-tip trail."""
    scn = viewer.user_scn
    scn.ngeom = 0
    def add(gtype, a, b, rgba, width):
        if scn.ngeom >= scn.maxgeom: return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, gtype, np.zeros(3), np.zeros(3), np.zeros(9), np.array(rgba, np.float32))
        mujoco.mjv_connector(g, gtype, width, np.asarray(a, float), np.asarray(b, float))
        scn.ngeom += 1
    if state.show_frames:
        q = data.qpos[:4]
        x, y, z, phi = kin.fk(q)
        for origin, ang in ((( 0.0, 0.0, 0.0), 0.0), ((x, y, z + tip_dz), phi)):
            o = _world(model, data, *origin)
            for k, rgba in enumerate(([1, .2, .2, 1], [.2, .8, .2, 1], [.2, .4, 1, 1])):
                if k == 2: d = np.array([0, 0, 0.05])
                else:      # axis direction in {0}, mapped to world (rotate by pi about z)
                    a = ang + (0 if k == 0 else math.pi / 2)
                    d = np.array([-math.cos(a), -math.sin(a), 0]) * 0.05
                add(mujoco.mjtGeom.mjGEOM_ARROW, o, o + d, rgba, 0.004)
    if state.show_trail3d:
        for a, b in zip(trail, list(trail)[1:]):
            add(mujoco.mjtGeom.mjGEOM_LINE, a, b, [0.9, 0.3, 0.1, 1], 2)

def sim_thread():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    params = get_params()
    ctl.make_torque_plant(model)                 # PID drives torques (XML file unchanged)
    data  = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    tip_dz = _tool_tip_offset(model, data)
    with state.lock:
        state.params = params
        if state.pid is None:
            state.pid = ctl.PID4(ctl.tuned_gains(params), dt=model.opt.timestep,
                                 feedforward="gravity", params=params)
    trail = collections.deque(maxlen=TRAIL3D_LEN)
    before = set(threading.enumerate())
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # launch_passive renders in its own daemon thread; remember it so we can
        # wait for it below (close() only *requests* exit)
        viewer_threads[:] = set(threading.enumerate()) - before
        viewer.cam.distance = 0.85; viewer.cam.azimuth = 140; viewer.cam.elevation = -22
        step = 0; ff = np.zeros(4); last_pid = None; tau = np.zeros(4); sat = [False]*4
        dt = model.opt.timestep
        wall0, sim0, last_sync = time.monotonic(), data.time, 0.0
        while viewer.is_running() and state.running:
            # Real time: run as many physics steps as needed to catch up with the wall
            # clock (the reference trajectory is played in wall-clock time).
            behind = (time.monotonic() - wall0) - (data.time - sim0)
            if behind > 0.1:                       # too slow (e.g. window dragged): resync
                wall0 = time.monotonic() - (data.time - sim0); behind = 0.0
            for _ in range(min(int(behind / dt), 25)):
                with state.lock:
                    q_d, qd_d, qdd_d = (np.array(state.ctrl), np.array(state.ctrl_vel),
                                        np.array(state.ctrl_acc))
                    t_set = state.ctrl_time
                    mode, pid = state.mode, state.pid
                # Setpoints arrive at ~100 Hz; between updates extrapolate with the desired
                # velocity/acceleration so the 500 Hz controller sees a smooth reference.
                # (h < 0: this step catches up on a moment before the setpoint was written)
                h = min(max(wall0 + (data.time - sim0) - t_set, -0.05), 0.02)
                if h != 0 and qd_d.any():
                    q_d = q_d + qd_d * h + 0.5 * qdd_d * h * h
                    qd_d = qd_d + qdd_d * h
                if pid is not last_pid:            # new gains from the PID tab
                    pid.dt = dt; pid.reset(); last_pid = pid
                if mode == "kinematic":
                    # follow the reference exactly; show the torque it would need
                    data.qpos[:4] = q_d; data.qvel[:4] = qd_d
                    mujoco.mj_forward(model, data)
                    data.time += dt
                    if step % FF_EVERY == 0:
                        tau = dyn.inverse_dynamics(q_d, qd_d, qdd_d, params)
                    pid.reset()
                    sat = [False]*4
                else:
                    q, qd = data.qpos[:4].copy(), data.qvel[:4].copy()
                    if step % FF_EVERY == 0:
                        ff = pid.feedforward_torque(q, q_d, qd_d, qdd_d)
                    tau = pid.update(q_d, qd_d, qdd_d, q, qd, ff=ff)
                    sat = list(pid.saturated)
                    data.ctrl[:4] = tau
                    mujoco.mj_step(model, data)
                step += 1
                with state.lock:
                    for i in range(4):
                        state.qpos[i] = data.qpos[i]; state.qvel[i] = data.qvel[i]
                    state.tau[:] = [float(v) for v in tau]
                    state.saturated[:] = sat
                    if step % 5 == 0:          # setpoint and response of the SAME instant
                        recorder.record(list(q_d), list(data.qpos[:4]), list(data.qvel[:4]),
                                        list(qd_d), list(qdd_d), state.tau)
            if state.clear_trail3d:
                trail.clear(); state.clear_trail3d = False
            now = time.monotonic()
            if now - last_sync >= 1 / 60:          # viewer at ~60 Hz
                last_sync = now
                x, y, z = kin.fk_position(*data.qpos[:3])
                p = _world(model, data, x, y, z + tip_dz)
                if not trail or np.linalg.norm(p - trail[-1]) > 5e-4:
                    trail.append(p)
                with viewer.lock():
                    _draw_overlays(viewer, model, data, trail, tip_dz)
                viewer.sync()
            time.sleep(0.001)
    # Wait until the viewer's render loop has really stopped, so glfw/EGL isn't
    # torn down at interpreter exit while it is still swapping buffers
    for t in viewer_threads:
        t.join(timeout=5)
    state.running = False

# ─── Palette ──────────────────────────────────────────────────────────────────
# Two minimal palettes: one neutral surface, input fields, colour only for state.
# Every colour the UI uses must come from here (keys are identical in both).
THEMES = {
    "light": dict(bg="#f3f3f3", panel="#f3f3f3", field="#ffffff", trough="#d9d9d9",
                  border="#c8c8c8", accent="#1f5fa8",
                  danger="#c62828", success="#2e7d32", warn="#a35f00",
                  text="#1f1f1f", muted="#6f6f6f", green="#1f1f1f", bright="#111111",
                  canvas_bg="#ffffff", grid="#eeeeee", trail="#7aa6d6",
                  tool_dot="#c62828", joint_dot="#4a4a4a",
                  tip_bg="#ffffe1", tab="#e4e4e4", heading="#e8e8e8",
                  btn="#e6e6e6", btn_hi="#f6f6f6", btn_lo="#d4d4d4", btn_active="#d9d9d9",
                  btn_disabled="#eeeeee", text_disabled="#a0a0a0",
                  stop_lo="#9e1f1f", stop_active="#a72222",
                  axis="#cfcfcf", ring="#b5b5b5", ghost="#c4c4c4", arm1="#555555", arm2="#7a7a7a",
                  row_sel="#cfe0f3", row_active="#e1efe3", row_bad="#fbe0e0", plot_grid="#e2e2e2"),
    "dark":  dict(bg="#1e1e1e", panel="#1e1e1e", field="#2a2a2a", trough="#3a3a3a",
                  border="#444444", accent="#5b9be0",
                  danger="#e05252", success="#5cb85c", warn="#e0a43a",
                  text="#e2e2e2", muted="#9a9a9a", green="#e2e2e2", bright="#f5f5f5",
                  canvas_bg="#252525", grid="#2e2e2e", trail="#5b8fd0",
                  tool_dot="#e05252", joint_dot="#c8c8c8",
                  tip_bg="#3a3a2e", tab="#2b2b2b", heading="#2b2b2b",
                  btn="#333333", btn_hi="#3c3c3c", btn_lo="#262626", btn_active="#404040",
                  btn_disabled="#2a2a2a", text_disabled="#6a6a6a",
                  stop_lo="#9e1f1f", stop_active="#a72222",
                  axis="#3d3d3d", ring="#555555", ghost="#555555", arm1="#c8c8c8", arm2="#9a9a9a",
                  row_sel="#2d4a6b", row_active="#1f3b26", row_bad="#4a1f1f", plot_grid="#333333"),
}
C = dict(THEMES["light"])   # active palette, mutated in place by set_theme()

def set_theme(name):
    C.clear(); C.update(THEMES[name])

def load_settings():
    try:
        with open(SETTINGS_PATH) as f: return json.load(f)
    except (OSError, ValueError):
        return {}

def save_settings(d):
    try:
        with open(SETTINGS_PATH, "w") as f: json.dump(d, f)
    except OSError as e:
        print("Error saving settings:", e)

JOINT_CFG = [("J1 - Shoulder [deg]", -140, 140, "deg", 0),
             ("J2 - Elbow    [deg]", -144, 144, "deg", 1),
             ("J3 - Z-axis    [mm]", -130,   0,  "mm", 2),
             ("J4 - Tool-R   [deg]", -360, 360, "deg", 3)]

JOINT_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]   # plots only
JOINT_NAMES  = ["J1", "J2", "J3", "J4"]
JOINT_UNITS  = ["deg", "deg", "mm", "deg"]
VEL_UNITS    = ["deg/s", "deg/s", "mm/s", "deg/s"]
ACC_UNITS    = ["deg/s^2", "deg/s^2", "mm/s^2", "deg/s^2"]

# ─── Tooltip ──────────────────────────────────────────────────────────────────
class Tooltip:
    """Small hover tooltip, shown after 500 ms."""
    def __init__(self, widget, text):
        self.w = widget; self.text = text; self.tip = None; self.job = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")
    def _schedule(self, e=None):
        self._hide(); self.job = self.w.after(500, self._show)
    def _show(self):
        self.job = None
        x = self.w.winfo_rootx() + 10
        y = self.w.winfo_rooty() + self.w.winfo_height() + 4
        self.tip = tw = tk.Toplevel(self.w)
        tw.wm_overrideredirect(True); tw.wm_geometry(f"+{x}+{y}")
        tk.Label(tw, text=self.text, bg=C["tip_bg"], fg=C["text"], font=(UI_FONT,8),
                 padx=6, pady=3, relief="solid", bd=1, justify="left").pack()
    def _hide(self, e=None):
        if self.job: self.w.after_cancel(self.job); self.job = None
        if self.tip: self.tip.destroy(); self.tip = None

# ─── GUI ──────────────────────────────────────────────────────────────────────
class ControlPanel:
    TRAIL_LEN = 800

    def __init__(self, root):
        self.root = root
        root.title("Yamaha YK-XG  |  SCARA Control Panel")
        root.configure(bg=C["bg"])
        root.minsize(900, 600)
        # Allow resize
        root.resizable(True, True)

        self.path_running = False
        self.motion_id    = 0      # bumped to cancel PTP/Home/path motion threads
        self._motion_busy = False
        self._stopped     = False
        self.override     = 1.0    # speed override 0.1-1.0 (plain float: read by worker threads)
        self.path_index   = 0; self.path_total = 0
        self._limit_state = ()
        self._jog_after   = None
        self.trail        = collections.deque(maxlen=self.TRAIL_LEN)
        self.plan_xy      = []       # previewed path (XY, m) drawn dashed on the top view
        self._plan_dirty  = True
        self.path_status  = "Path"
        # Worker threads must never call Tk (a cross-thread Tk call blocks until the main
        # loop serves it, which stalled the motion thread for up to 0.5 s). They queue
        # (function, args) here and _tick runs them on the main thread.
        self.ui_queue     = queue.SimpleQueue()
        
        self.waypoints = []
        self.theme = load_settings().get("theme", "light")
        if self.theme not in THEMES: self.theme = "light"
        set_theme(self.theme)

        self._build_ui()
        self._load_waypoints()
        self._bind_keys()

        root.after(60, self._tick)
        root.protocol("WM_DELETE_WINDOW", self._quit)

    def _build_ui(self):
        """Create every widget from the active palette C. Called again on theme switch."""
        root = self.root
        root.configure(bg=C["bg"])
        self._styles()
        self._toolbar()

        # Main layout using grid for proper resizing
        main = tk.Frame(root, bg=C["bg"])
        main.pack(fill="both", expand=True, padx=6, pady=2)
        main.columnconfigure(0, weight=3)  # left tabs get more space
        main.columnconfigure(1, weight=1, minsize=400)  # right panel
        main.rowconfigure(0, weight=1)

        left = tk.Frame(main, bg=C["bg"])
        left.grid(row=0, column=0, sticky="nsew", padx=(0,4))

        right = tk.Frame(main, bg=C["bg"])
        right.grid(row=0, column=1, sticky="nsew")

        self._notebook(left)
        self._build_right_panel(right)
        self._statusbar()

    # Tk variables carried across a theme switch (everything else is rebuilt from state)
    UI_VARS = ("profile_var", "move_var", "loop_var", "speed_var", "dwell_var", "step_var",
               "elbow_var", "plot_series", "show_workspace", "show_links", "override_var",
               "jog_msg", "path_progress", "path_type", "plot_view")

    def _toggle_theme(self):
        snap = {n: getattr(self, n).get() for n in self.UI_VARS}
        snap["abs"] = {}
        for a, v in self.abs_vars.items():
            try: snap["abs"][a] = v.get()
            except tk.TclError: pass        # half-typed box: keep default
        snap["plot_joints"] = [v.get() for v in self.plot_joints]
        snap["tab"] = self.nb.index("current")
        snap["msg"] = (self.sb_msg.cget("text"), self.sb_msg.cget("fg"))

        self.theme = "dark" if self.theme == "light" else "light"
        set_theme(self.theme)
        save_settings({"theme": self.theme})
        for w in self.root.winfo_children(): w.destroy()
        self._build_ui()

        for n in self.UI_VARS: getattr(self, n).set(snap[n])
        for a, v in snap["abs"].items(): self.abs_vars[a].set(v)
        for var, v in zip(self.plot_joints, snap["plot_joints"]): var.set(v)
        self._set_override(snap["override_var"])
        self.nb.select(snap["tab"])
        self._show_path_type()
        self._refresh_wp()
        if self.path_running:
            self.run_btn.state(["disabled"]); self.stop_btn.state(["!disabled"])
            self.path_pb.config(maximum=self.path_total, value=max(0, self.path_index - 1))
            self._highlight_wp(self.path_index - 1)
        self._update_trajectory_plot()
        self._notify(f"{self.theme.capitalize()} theme")

    def _styles(self):
        self.root.option_add("*Radiobutton.highlightThickness", 0)
        self.root.option_add("*Checkbutton.highlightThickness", 0)
        s = ttk.Style(self.root); s.theme_use("clam")
        s.configure(".", background=C["bg"], foreground=C["text"], font=(UI_FONT,9))
        s.configure("TNotebook", background=C["bg"], borderwidth=0,
                    bordercolor=C["border"], lightcolor=C["bg"], darkcolor=C["bg"])
        s.configure("TNotebook.Tab", background=C["tab"], foreground=C["text"],
                    bordercolor=C["border"], lightcolor=C["tab"],
                    padding=[14,4], font=(UI_FONT,9))
        s.map("TNotebook.Tab", background=[("selected", C["bg"])])
        s.configure("TFrame", background=C["bg"])
        # All action buttons look the same; STOP is the only coloured one
        for nm in ("TButton", "A.TButton", "D.TButton", "G.TButton", "W.TButton",
                   "JP.TButton", "JN.TButton"):
            s.configure(nm, background=C["btn"], foreground=C["text"],
                        bordercolor=C["border"], lightcolor=C["btn_hi"], darkcolor=C["btn_lo"],
                        font=(UI_FONT,9), padding=[8,3])
            s.map(nm, background=[("disabled", C["btn_disabled"]), ("active", C["btn_active"])],
                  foreground=[("disabled", C["text_disabled"])])
        s.configure("Stop.TButton", background=C["danger"], foreground="white",
                    bordercolor=C["stop_lo"], lightcolor=C["danger"], darkcolor=C["stop_lo"],
                    font=(UI_FONT,10,"bold"), padding=[14,4])
        s.map("Stop.TButton", background=[("active", C["stop_active"])])
        s.configure("G.Horizontal.TProgressbar", troughcolor=C["tab"],
                    background=C["accent"], bordercolor=C["border"],
                    lightcolor=C["accent"], darkcolor=C["accent"])

    def _tip(self, widget, text):
        Tooltip(widget, text); return widget

    def _toolbar(self):
        """Top bar, visible on every tab: title, speed override, Home, STOP."""
        bar = tk.Frame(self.root, bg=C["bg"]); bar.pack(fill="x", padx=8, pady=(6,2))
        tk.Label(bar, text="SCARA YK-XG", bg=C["bg"], fg=C["text"],
                 font=(UI_FONT,12,"bold")).pack(side="left")
        self.sim_dot = tk.Label(bar, text="\u25cf sim", bg=C["bg"], fg=C["success"],
                                font=(UI_FONT,9))
        self.sim_dot.pack(side="left", padx=10)
        tk.Label(bar, text="Mode", bg=C["bg"], fg=C["muted"],
                 font=(UI_FONT,9)).pack(side="left", padx=(14,4))
        self.mode_var = tk.StringVar(value=state.mode)
        for val, txt, tip in [
                ("kinematic", "Kinematic", "Kinematic simulation: the robot follows the reference\n"
                                           "trajectory exactly; torque shown = torque it would need"),
                ("control", "PID control", "Dynamic simulation: the PID drives joint torques and the\n"
                                           "robot responds with its real dynamics (tracking error)")]:
            self._tip(tk.Radiobutton(bar, text=txt, variable=self.mode_var, value=val,
                                     command=self._set_mode, bg=C["bg"], fg=C["text"],
                                     selectcolor=C["field"], activebackground=C["bg"],
                                     font=(UI_FONT,9)), tip).pack(side="left")

        b = ttk.Button(bar, text="STOP  (Esc)", style="Stop.TButton", command=self._stop_all)
        b.pack(side="right", padx=(6,0))
        self._tip(b, "Stop all motion and hold the current position (Esc)")
        b = ttk.Button(bar, text="Home", command=self._go_home)
        b.pack(side="right", padx=6)
        self._tip(b, "Move all joints to zero with a smooth joint-space move")
        other = "dark" if self.theme == "light" else "light"
        b = ttk.Button(bar, text=f"{other.capitalize()} mode", command=self._toggle_theme)
        b.pack(side="right", padx=(0,6))
        self._tip(b, f"Switch to the {other} theme (remembered next time)")

        self.override_lbl = tk.Label(bar, text="100 %", bg=C["bg"], fg=C["text"],
                                     font=(MONO_FONT,9), width=6)
        self.override_lbl.pack(side="right", padx=(0,10))
        self.override_var = tk.IntVar(value=100)
        sc = tk.Scale(bar, from_=10, to=100, resolution=10, orient="horizontal",
                      variable=self.override_var, showvalue=False, length=120,
                      bg=C["bg"], fg=C["text"], troughcolor=C["trough"],
                      activebackground=C["accent"], highlightthickness=0, bd=0,
                      command=self._set_override)
        sc.pack(side="right")
        tk.Label(bar, text="Speed", bg=C["bg"], fg=C["muted"],
                 font=(UI_FONT,9)).pack(side="right", padx=(0,4))
        self._tip(sc, "Speed override: scales Go, Home and path motion time")
        ttk.Separator(self.root).pack(fill="x", padx=8, pady=(2,0))

    def _set_mode(self):
        mode = self.mode_var.get()
        with state.lock:
            state.mode = mode
            if mode == "control":       # start PID from where the reference is
                state.set_setpoint(state.ctrl)
        self._notify("Kinematic mode: robot follows the reference exactly" if mode == "kinematic"
                     else "PID control mode: torques from the controller")

    def _set_override(self, v):
        self.override = max(0.1, min(1.0, float(v) / 100.0))
        self.override_lbl.config(text=f"{int(float(v))} %")

    def _new_motion(self):
        """Cancel any running path/PTP/Home motion; return the new motion id."""
        self.path_running = False
        self._motion_busy = False
        self._stopped = False
        self.motion_id += 1
        return self.motion_id

    def _manual(self):
        """Manual control (jog / joint slider) takes over from any running motion."""
        if self.path_running or self._motion_busy:
            self._new_motion()
        self._stopped = False
        with state.lock:
            state.ctrl_vel[:] = [0.0]*4; state.ctrl_acc[:] = [0.0]*4

    def _stop_all(self):
        self._new_motion()
        self._jog_stop()
        with state.lock:
            state.set_setpoint(state.qpos)     # hold where the robot actually is
        self._stopped = True
        self._notify("STOPPED - holding position", "error")

    def _bind_keys(self):
        self.root.bind("<Key>", self._on_key)
        self.root.bind("<Escape>", lambda e: self._stop_all())

    JOG_KEYS = {"Left": ("x",-1), "Right": ("x",+1), "Up": ("y",+1), "Down": ("y",-1),
                "Prior": ("z",+1), "Next": ("z",-1),
                "q": ("r",-1), "Q": ("r",-1), "e": ("r",+1), "E": ("r",+1)}

    def _on_key(self, e):
        # Leave keys alone in widgets that use them for editing/selection
        if isinstance(e.widget, (tk.Entry, ttk.Entry, ttk.Treeview, tk.Scale)):
            return
        jog = self.JOG_KEYS.get(e.keysym)
        if jog:
            self._jog_once(*jog)
            return "break"

    def _statusbar(self):
        ttk.Separator(self.root).pack(fill="x")
        sb = tk.Frame(self.root, bg=C["panel"]); sb.pack(fill="x")
        tk.Label(sb, text="Arrows X/Y   PgUp/PgDn Z   Q/E R   click top view to move",
                 bg=C["panel"], fg=C["muted"], font=(UI_FONT,8), padx=8
                 ).pack(side="right")
        self.sb_sim = tk.Label(sb, text="\u25cf Sim running", bg=C["panel"], fg=C["success"],
                               font=(UI_FONT,8,"bold"), padx=8, pady=2, width=16, anchor="w")
        self.sb_sim.pack(side="left")
        tk.Frame(sb, bg=C["border"], width=1).pack(side="left", fill="y", pady=2)
        self.sb_motion = tk.Label(sb, text="Idle", bg=C["panel"], fg=C["text"],
                                  font=(UI_FONT,8,"bold"), padx=8, pady=2, width=14, anchor="w")
        self.sb_motion.pack(side="left")
        tk.Frame(sb, bg=C["border"], width=1).pack(side="left", fill="y", pady=2)
        self.sb_msg = tk.Label(sb, text="Ready", bg=C["panel"], fg=C["accent"],
                               font=(UI_FONT,8), padx=8, pady=2, anchor="w")
        self.sb_msg.pack(side="left", fill="x", expand=True)

    NOTIFY_COLORS = {"info": "accent", "ok": "success", "warn": "warn", "error": "danger"}

    def _notify(self, msg, level="info"):
        """Show a message in the status bar. Main thread only (workers use self.ui_queue)."""
        self.sb_msg.config(text=msg, fg=C[self.NOTIFY_COLORS.get(level, "accent")])

    def _jog_warn(self, msg):
        """Jog-tab message that is also mirrored to the status bar."""
        self.jog_msg.set(msg)
        if msg:
            self._notify(msg, "warn" if msg.startswith("!!") else "info")

    def _notebook(self, parent):
        nb = self.nb = ttk.Notebook(parent)
        nb.pack(fill="both", expand=True, pady=2)
        # Arrow keys jog the robot instead of switching tabs
        for key in ("<Left>", "<Right>"): nb.bind(key, self._on_key)
        t1 = ttk.Frame(nb); nb.add(t1, text="  Joint  ")
        t2 = ttk.Frame(nb); nb.add(t2, text="  Jog  ")
        t3 = ttk.Frame(nb); nb.add(t3, text="  Path  ")
        t4 = ttk.Frame(nb); nb.add(t4, text="  Trajectory  ")
        t5 = ttk.Frame(nb); nb.add(t5, text="  PID  ")
        self._tab_joint(t1)
        self._tab_jog(t2)
        self._tab_path(t3)
        self._tab_trajectory(t4)
        self._tab_pid(t5)

    def _build_right_panel(self, parent):
        parent.rowconfigure(0, weight=3)
        parent.rowconfigure(1, weight=0)
        parent.rowconfigure(2, weight=0)
        parent.columnconfigure(0, weight=1)

        # Canvas
        lf = tk.LabelFrame(parent, text="  Top View (XY)  ",
                           bg=C["panel"], fg=C["text"],
                           font=(UI_FONT,8,"bold"), padx=3, pady=3)
        lf.grid(row=0, column=0, sticky="nsew", pady=(2,1))
        self.canvas = tk.Canvas(lf, bg=C["canvas_bg"], highlightthickness=0,
                                cursor="crosshair")
        self.canvas.pack(fill="both", expand=True)
        self._geo = None          # (cx, cy, ppm) once the canvas has a size
        self._wp_drawn = None
        self._trail_dirty = True
        self.canvas.bind("<Configure>", lambda e: self._canvas_setup())
        self.canvas.bind("<Button-1>", self._canvas_click)

        bf = tk.Frame(lf, bg=C["panel"]); bf.pack(fill="x", pady=1)
        self._tip(ttk.Button(bf, text="Clear trail", style="D.TButton",
                             command=self._clear_trail),
                  "Erase the tool trail (top view and 3D) and the previewed path").pack(side="left", padx=2)
        self.show_workspace = tk.BooleanVar(value=True)
        self.show_links = tk.BooleanVar(value=True)
        self.show_frames3d = tk.BooleanVar(value=state.show_frames)
        self.show_trail3d = tk.BooleanVar(value=state.show_trail3d)
        for text, var, tip in [("3D frames", self.show_frames3d, "Show frames {0} and {tool} in the 3D viewer"),
                               ("3D trail", self.show_trail3d, "Show the tool-tip trail in the 3D viewer")]:
            self._tip(tk.Checkbutton(bf, text=text, variable=var, command=self._apply_3d_options,
                                     bg=C["panel"], fg=C["text"], selectcolor=C["field"],
                                     font=(UI_FONT,7)), tip).pack(side="right")
        tk.Checkbutton(bf, text="Workspace", variable=self.show_workspace,
                       command=self._apply_visibility,
                       bg=C["panel"], fg=C["text"], selectcolor=C["field"],
                       font=(UI_FONT,7)).pack(side="left")
        tk.Checkbutton(bf, text="Links", variable=self.show_links,
                       command=self._apply_visibility,
                       bg=C["panel"], fg=C["text"], selectcolor=C["field"],
                       font=(UI_FONT,7)).pack(side="left")

        # Coordinates
        cf = tk.LabelFrame(parent, text="  Position  ",
                           bg=C["panel"], fg=C["text"],
                           font=(UI_FONT,8,"bold"), padx=6, pady=4)
        cf.grid(row=1, column=0, sticky="ew", pady=(2,1))
        self.coord_labels = {}
        for name, color in [(a, C["muted"]) for a in "XYZR"]:
            row = tk.Frame(cf, bg=C["panel"]); row.pack(fill="x")
            tk.Label(row, text="\u03c6:" if name == "R" else f"{name}:", bg=C["panel"], fg=color,
                     font=(MONO_FONT,12,"bold"), width=3).pack(side="left")
            lbl = tk.Label(row, text="  0.00 mm", bg=C["panel"], fg=C["bright"],
                           font=(MONO_FONT,12,"bold"), anchor="e", width=12)
            lbl.pack(side="right")
            self.coord_labels[name] = lbl
        row = tk.Frame(cf, bg=C["panel"]); row.pack(fill="x")
        self._tip(tk.Label(row, text="Err:", bg=C["panel"], fg=C["muted"],
                           font=(MONO_FONT,12,"bold"), width=4),
                  "Tracking error: distance between commanded (ghost)\n"
                  "and measured tool position").pack(side="left")
        self.err_label = tk.Label(row, text="  0.00 mm", bg=C["panel"], fg=C["success"],
                                  font=(MONO_FONT,12,"bold"), anchor="e", width=12)
        self.err_label.pack(side="right")

        # Joints
        jf = tk.LabelFrame(parent, text="  Joints  ",
                           bg=C["panel"], fg=C["text"],
                           font=(UI_FONT,8,"bold"), padx=6, pady=4)
        jf.grid(row=2, column=0, sticky="ew", pady=(2,1))
        self.jval_labels = {}
        for name in ["J1","J2","J3","J4"]:
            color = C["muted"]
            row = tk.Frame(jf, bg=C["panel"]); row.pack(fill="x")
            tk.Label(row, text=f"{name}:", bg=C["panel"], fg=color,
                     font=(MONO_FONT,9,"bold"), width=4).pack(side="left")
            lbl = tk.Label(row, text="  0.0 deg", bg=C["panel"], fg=C["bright"],
                           font=(MONO_FONT,9), anchor="e", width=10)
            lbl.pack(side="right")
            self.jval_labels[name] = lbl

    def _clear_trail(self):
        self.trail.clear(); self._trail_dirty = True
        self.plan_xy = []; self._plan_dirty = True
        state.clear_trail3d = True

    def _apply_3d_options(self):
        state.show_frames = self.show_frames3d.get()
        state.show_trail3d = self.show_trail3d.get()

    def _w2c(self, wx, wy):
        cx, cy, ppm = self._geo
        return cx+wx*ppm, cy-wy*ppm

    def _canvas_setup(self):
        """(Re)build all canvas items. Runs on resize only; _update_canvas moves them."""
        cv = self.canvas; cv.delete("all")
        w = cv.winfo_width(); h = cv.winfo_height()
        if w < 10 or h < 10: self._geo = None; return
        cx, cy = w//2, h//2
        self._geo = (cx, cy, min(w, h) / (2*(L1+L2)*1.15))
        ppm = self._geo[2]; w2c = self._w2c

        # Static: grid, axes, workspace rings, labels, scale bar
        step_m = 0.05; r_max = L1+L2+0.02; g = -r_max
        while g <= r_max:
            cv.create_line(*w2c(g,-r_max), *w2c(g,r_max), fill=C["grid"])
            cv.create_line(*w2c(-r_max,g), *w2c(r_max,g), fill=C["grid"])
            g += step_m
        cv.create_line(cx,0,cx,h, fill=C["axis"], width=1)
        cv.create_line(0,cy,w,cy, fill=C["axis"], width=1)
        ro=(L1+L2)*ppm; ri=abs(L1-L2)*ppm
        cv.create_oval(cx-ro,cy-ro,cx+ro,cy+ro, outline=C["ring"], dash=(3,3), tags="ws")
        cv.create_oval(cx-ri,cy-ri,cx+ri,cy+ri, outline=C["ring"], dash=(3,3), tags="ws")
        cv.create_text(w-16,cy-10, text="X+", fill=C["muted"], font=(UI_FONT,8))
        cv.create_text(cx+12,12, text="Y+", fill=C["muted"], font=(UI_FONT,8))
        bar_px = 0.05 * ppm
        cv.create_line(8,h-8,8+bar_px,h-8, fill=C["text"], width=2)
        cv.create_text(8+bar_px/2,h-16, text="50mm", fill=C["muted"], font=(UI_FONT,7))

        # Dynamic items (coords updated every tick)
        self.plan_item = cv.create_line(0,0,0,0, fill=C["accent"], width=1, dash=(4,3), state="hidden")
        self.trail_item = cv.create_line(0,0,0,0, fill=C["trail"], width=2, state="hidden")
        self.ghost_items = [
            cv.create_line(0,0,0,0, fill=C["ghost"], width=4, capstyle="round", dash=(4,3), tags="ghost"),
            cv.create_line(0,0,0,0, fill=C["ghost"], width=3, capstyle="round", dash=(4,3), tags="ghost"),
            cv.create_oval(0,0,0,0, outline=C["tool_dot"], dash=(2,2), tags="ghost")]
        self.arm_items = [
            cv.create_line(0,0,0,0, fill=C["arm1"], width=4, capstyle="round", tags="arm"),
            cv.create_line(0,0,0,0, fill=C["arm2"], width=3, capstyle="round", tags="arm"),
            cv.create_oval(0,0,0,0, fill=C["joint_dot"], outline=C["canvas_bg"], tags="arm"),
            cv.create_oval(0,0,0,0, fill=C["joint_dot"], outline=C["canvas_bg"], tags="arm"),
            cv.create_oval(0,0,0,0, fill=C["tool_dot"], outline=C["canvas_bg"], width=2, tags="arm")]
        self._wp_drawn = None; self._trail_dirty = True; self._plan_dirty = True
        self._apply_visibility()

    def _apply_visibility(self):
        cv = self.canvas
        cv.itemconfigure("ws", state="normal" if self.show_workspace.get() else "hidden")
        links = "normal" if self.show_links.get() else "hidden"
        cv.itemconfigure("arm", state=links); cv.itemconfigure("ghost", state=links)

    def _arm_points(self, j1, j2, j3):
        o = self._w2c(0, 0)
        e = self._w2c(L1*math.cos(j1), L1*math.sin(j1))
        x, y, _ = fk(j1, j2, j3)
        return o, e, self._w2c(x, y)

    def _update_canvas(self, act, tgt):
        if self._geo is None: return
        cv = self.canvas

        # Waypoint markers: redraw only when the list changes
        wp_key = [tuple(wp[:3]) for wp in self.waypoints]
        if wp_key != self._wp_drawn:
            cv.delete("wp")
            for j1, j2, j3 in wp_key:
                px, py = self._w2c(*fk(j1, j2, j3)[:2])
                cv.create_rectangle(px-3,py-3,px+3,py+3, fill=C["accent"],
                                    outline=C["accent"], tags="wp")
            cv.tag_lower("wp", self.trail_item)
            self._wp_drawn = wp_key

        if self._plan_dirty:
            if len(self.plan_xy) > 1:
                cv.coords(self.plan_item, *[c for p in self.plan_xy for c in self._w2c(*p)])
                cv.itemconfigure(self.plan_item, state="normal")
            else:
                cv.itemconfigure(self.plan_item, state="hidden")
            self._plan_dirty = False

        if self._trail_dirty:
            if len(self.trail) > 1:
                pts = [c for p in self.trail for c in self._w2c(*p)]
                cv.coords(self.trail_item, *pts)
                cv.itemconfigure(self.trail_item, state="normal")
            else:
                cv.itemconfigure(self.trail_item, state="hidden")
            self._trail_dirty = False

        for items, q in ((self.ghost_items, tgt), (self.arm_items, act)):
            (ox,oy), (ex,ey), (tx,ty) = self._arm_points(*q[:3])
            cv.coords(items[0], ox,oy,ex,ey)
            cv.coords(items[1], ex,ey,tx,ty)
            if len(items) == 3:
                cv.coords(items[2], tx-7,ty-7,tx+7,ty+7)
            else:
                cv.coords(items[2], ox-4,oy-4,ox+4,oy+4)
                cv.coords(items[3], ex-3,ey-3,ex+3,ey+3)
                cv.coords(items[4], tx-5,ty-5,tx+5,ty+5)

    def _canvas_click(self, e):
        """Move the tool to the clicked XY point (same path as the Go button)."""
        if self._geo is None: return
        cx, cy, ppm = self._geo
        self.abs_vars["X"].set(round((e.x-cx)/ppm*1000, 2))
        self.abs_vars["Y"].set(round((cy-e.y)/ppm*1000, 2))
        with state.lock: q = list(state.ctrl)
        self.abs_vars["Z"].set(round(q[2]*1000, 2))
        self.abs_vars["PHI"].set(round(math.degrees(tool_phi(q)), 2))   # keep the tool angle
        self._go_abs_xyz()

    # ══════════════════════════════════════════════════════════════════════
    # TAB 1 - JOINT
    # ══════════════════════════════════════════════════════════════════════
    def _tab_joint(self, p):
        self.jvars = []; self.jentries = []
        for lbl, lo, hi, unit, idx in JOINT_CFG:
            row = tk.Frame(p, bg=C["bg"]); row.pack(fill="x", padx=8, pady=4)
            tk.Label(row, text=lbl, bg=C["bg"], fg=C["muted"],
                     font=(UI_FONT,9), width=20, anchor="w").pack(side="left")
            var = tk.DoubleVar(value=0.0); self.jvars.append(var)
            tk.Scale(row, from_=lo, to=hi, resolution=0.5,
                     orient="horizontal", variable=var,
                     bg=C["panel"], fg=C["text"],
                     troughcolor=C["trough"], activebackground=C["accent"],
                     highlightthickness=0, bd=0, length=200,
                     command=lambda v,i=idx,u=unit: self._apply_joint(i,u)
                     ).pack(side="left", padx=4)
            ent = tk.Entry(row, width=7, bg=C["field"], fg=C["text"],
                           insertbackground=C["text"], font=(MONO_FONT,9), bd=0,
                           highlightthickness=1, highlightcolor=C["accent"], highlightbackground=C["border"], relief="flat")
            ent.insert(0,"0.0"); ent.pack(side="left"); self.jentries.append(ent)
            ent.bind("<Return>",
                     lambda e,i=idx,u=unit,lo=lo,hi=hi,v=var,ent=ent:
                     self._entry_joint(i,u,lo,hi,v,ent))
        ttk.Separator(p).pack(fill="x", padx=8, pady=4)
        self._tip(ttk.Button(p, text="Home", style="D.TButton",
                             command=self._go_home),
                  "Move all joints to zero").pack(pady=4)

    def _apply_joint(self, idx, unit):
        self._manual()
        v = self.jvars[idx].get()
        val = math.radians(v) if unit=="deg" else v/1000.0
        with state.lock: state.ctrl[idx] = val
    def _entry_joint(self, idx, unit, lo, hi, var, ent):
        try:
            v = max(lo, min(hi, float(ent.get())))
            var.set(v); self._apply_joint(idx, unit)
        except ValueError: pass

    # ══════════════════════════════════════════════════════════════════════
    # TAB 2 - JOG
    # ══════════════════════════════════════════════════════════════════════
    def _tab_jog(self, p):
        sf = tk.Frame(p, bg=C["bg"]); sf.pack(fill="x", padx=8, pady=(6,2))
        tk.Label(sf, text="Step size:", bg=C["bg"], fg=C["muted"],
                 font=(UI_FONT,9)).pack(side="left")
        self.step_var = tk.DoubleVar(value=5.0)
        for v in [0.5, 1, 5, 10, 20]:
            tk.Radiobutton(sf, text=f"{v} mm", variable=self.step_var, value=v,
                           bg=C["bg"], fg=C["text"], selectcolor=C["field"],
                           font=(UI_FONT,8)).pack(side="left", padx=3)

        jf = tk.LabelFrame(p, text="  Jog Axes  ", bg=C["panel"], fg=C["text"],
                           font=(UI_FONT,9), pady=6, padx=8)
        jf.pack(fill="x", padx=8, pady=4)
        for lbl,neg,pos,key in [("X","-X","+X","x"),("Y","-Y","+Y","y"),
                                ("Z","-Z","+Z","z"),("R","-R","+R","r")]:
            row = tk.Frame(jf, bg=C["panel"]); row.pack(fill="x", pady=3)
            tk.Label(row, text=lbl, bg=C["panel"], fg=C["text"],
                     font=(UI_FONT,11,"bold"), width=3).pack(side="left")
            self._jog_btn(row, neg, "JN.TButton", key, -1)
            self._jog_btn(row, pos, "JP.TButton", key, +1)

        abs_f = tk.LabelFrame(p, text="  Move to pose X, Y, Z, phi  ", bg=C["panel"],
                              fg=C["text"], font=(UI_FONT,9), pady=6, padx=8)
        abs_f.pack(fill="x", padx=8, pady=4)
        max_r = (L1+L2)*1000
        self.abs_vars = {}   # user input only; never overwritten by _tick
        init_vals = {"X": 258.0, "Y": 0.0, "Z": 0.0, "PHI": 0.0}
        for axis, lo, hi in [("X",-max_r,max_r),("Y",-max_r,max_r),("Z",-130,0),("PHI",-360,360)]:
            row2 = tk.Frame(abs_f, bg=C["panel"]); row2.pack(fill="x", pady=2)
            text = "phi [deg]:" if axis == "PHI" else f"{axis} [mm]:"
            tk.Label(row2, text=text, bg=C["panel"], fg=C["muted"],
                     font=(UI_FONT,9), width=8).pack(side="left")
            var = tk.DoubleVar(value=init_vals[axis]); self.abs_vars[axis] = var
            tk.Entry(row2, width=9, textvariable=var, bg=C["field"], fg=C["text"],
                     insertbackground=C["text"], font=(MONO_FONT,9), bd=0,
                     highlightthickness=1, highlightcolor=C["accent"], highlightbackground=C["border"], relief="flat"
                     ).pack(side="left", padx=6)
        ef = tk.Frame(abs_f, bg=C["panel"]); ef.pack(fill="x", pady=2)
        tk.Label(ef, text="Elbow:", bg=C["panel"], fg=C["muted"],
                 font=(UI_FONT,9), width=8).pack(side="left")
        self.elbow_var = tk.StringVar(value="auto")
        for val, txt, tip in [("auto", "Auto", "Use whichever elbow solution is closest\n"
                                               "to the current pose (within joint limits)"),
                              ("left", "Left (+)", "Force elbow-left solution (J2 > 0)"),
                              ("right", "Right (-)", "Force elbow-right solution (J2 < 0)")]:
            self._tip(tk.Radiobutton(ef, text=txt, variable=self.elbow_var, value=val,
                                     bg=C["panel"], fg=C["text"], selectcolor=C["field"],
                                     activebackground=C["panel"], font=(UI_FONT,8)),
                      tip).pack(side="left", padx=3)
        bf = tk.Frame(abs_f, bg=C["panel"]); bf.pack(pady=3)
        for txt, sty, cmd, tip in [
                ("Go", "A.TButton", self._go_abs_xyz,
                 "Move the tool to X, Y, Z and tool angle phi\n(also: click the top view)"),
                ("Use Current", "W.TButton", self._abs_from_current,
                 "Fill the boxes with the current tool pose"),
                ("Home", "D.TButton", self._go_home, "Move all joints to zero")]:
            self._tip(ttk.Button(bf, text=txt, style=sty, command=cmd),
                      tip).pack(side="left", padx=4)
        self.jog_msg = tk.StringVar()
        tk.Label(p, textvariable=self.jog_msg, bg=C["bg"], fg=C["warn"],
                 font=(UI_FONT,8,"italic")).pack(anchor="w", padx=8)

    def _jog_btn(self, parent, label, style, axis, d):
        btn = ttk.Button(parent, text=label, style=style, width=8)
        btn.pack(side="left", padx=4)
        self._tip(btn, f"Hold to jog {axis.upper()} {'+' if d > 0 else '-'} by the step size")
        btn.bind("<ButtonPress-1>",   lambda e: self._jog_start(axis,d))
        btn.bind("<ButtonRelease-1>", lambda e: self._jog_stop())
    def _jog_start(self, a, d): self._jog_once(a,d); self._jog_repeat(a,d)
    def _jog_repeat(self, a, d):
        self._jog_once(a,d)
        self._jog_after = self.root.after(80, lambda: self._jog_repeat(a,d))
    def _jog_stop(self):
        if self._jog_after: self.root.after_cancel(self._jog_after); self._jog_after=None

    def _abs_from_current(self):
        with state.lock: q = list(state.ctrl)
        for axis, v in zip("XYZ", fk(*q[:3])):
            self.abs_vars[axis].set(round(v*1000, 2))
        self.abs_vars["PHI"].set(round(math.degrees(tool_phi(q)), 2))

    def _jog_once(self, axis, direction):
        self._manual()   # jogging takes over from any running motion
        step = self.step_var.get() * direction
        with state.lock: j1, j2, j3, j4 = state.ctrl
        sx, sy, sz = fk(j1, j2, j3)
        j2_sign = 1 if j2 >= 0 else -1

        if axis == "r":
            with state.lock:
                state.ctrl[3] = max(-6.28, min(6.28, j4 + math.radians(step)))
            return

        if axis == "z":
            # Z is independent of XY: move J3 directly (no IK, works at full reach)
            lo, hi = JOINT_LIMITS[2]
            tz = sz + step / 1000
            if not lo <= tz <= hi:
                tz = max(lo, min(hi, tz))
                self._jog_warn("!! Z limit reached")
            else:
                self._jog_warn("")
            with state.lock: state.ctrl[2] = tz
            return

        tx = sx + (step / 1000 if axis == "x" else 0.0)
        ty = sy + (step / 1000 if axis == "y" else 0.0)
        tz = sz

        d_target = math.sqrt(tx**2 + ty**2)
        if d_target < DEAD_ZONE:
            self._jog_warn("!! Singular point (dead zone)!")
            return

        reach = L1 + L2
        res = ik_locked(tx, ty, tz, j2_sign)
        if res is None or d_target > reach - 0.002:
            dx, dy = tx - sx, ty - sy
            lo, hi = 0.0, 1.0
            for _ in range(18):
                mid = (lo + hi) / 2.0
                px, py = sx + dx * mid, sy + dy * mid
                dm = math.sqrt(px**2 + py**2)
                if ik_locked(px, py, tz, j2_sign) is not None and dm < reach - 0.002:
                    lo = mid
                else:
                    hi = mid
            if lo < 0.02:
                self._jog_warn("!! Workspace / joint limit")
                return
            tx = sx + dx * lo * 0.99
            ty = sy + dy * lo * 0.99
            self._jog_warn("At workspace boundary")
        else:
            self._jog_warn("")

        dist_mm = math.sqrt((tx - sx)**2 + (ty - sy)**2) * 1000
        N = max(1, min(int(dist_mm), 25))
        last_res = None
        for k in range(1, N + 1):
            s = k / N
            cx = sx + (tx - sx) * s
            cy = sy + (ty - sy) * s
            r = ik_locked(cx, cy, tz, j2_sign)
            if r is None: break
            last_res = r
        if last_res:
            with state.lock:
                state.ctrl[0], state.ctrl[1], state.ctrl[2] = last_res

    def _go_abs_xyz(self):
        try:
            tx, ty, tz = (self.abs_vars[a].get() / 1000 for a in "XYZ")
            phi = math.radians(self.abs_vars["PHI"].get())
        except tk.TclError:
            self._jog_warn("!! Invalid number"); return
        d = math.sqrt(tx**2 + ty**2)
        if d < DEAD_ZONE:
            self._jog_warn("!! Singular point (dead zone)!"); return
        if d > L1 + L2 - 1e-5:
            self._jog_warn("!! Out of reach!"); return
        mode = self.elbow_var.get()
        if mode == "auto":
            with state.lock: cj1, cj2 = state.ctrl[0], state.ctrl[1]
            res = ik_nearest(tx, ty, tz, cj1, cj2)
        else:
            sign = 1 if mode == "left" else -1
            res = ik_locked(tx, ty, tz, sign)
            if res is None and ik_locked(tx, ty, tz, -sign) is not None:
                self._jog_warn("!! Only reachable with the other elbow"); return
        if res is None:
            self._jog_warn("!! Joint limit - target not reachable"); return
        with state.lock: cur_t4 = state.ctrl[3]
        t4 = kin.theta4_for_phi(phi, res[0], res[1], cur_t4)
        if t4 is None:
            self._jog_warn("!! phi needs J4 outside its limits"); return
        self._jog_warn("")
        self._notify(f"Moving to X {tx*1000:.1f}  Y {ty*1000:.1f}  Z {tz*1000:.1f} mm, "
                     f"phi {math.degrees(phi):.1f} deg")
        mid = self._new_motion()
        threading.Thread(target=self._joint_move, args=([res[0], res[1], res[2], t4], mid),
                         daemon=True).start()

    def _joint_move(self, target, mid):
        """Joint-space S-curve move, speed-limited (60 deg/s J1/J2/J4, 50 mm/s J3)
        and scaled by the speed override.
        Stops early if another motion starts (motion_id changes)."""
        self._motion_busy = True
        try:
            with state.lock:
                start = list(state.ctrl)
            speeds = [math.radians(60), math.radians(60), 0.05, math.radians(60)]
            dists  = [abs(target[i] - start[i]) for i in range(4)]
            T = max(max(dists[i] / speeds[i] for i in range(4)), 0.5) / self.override

            # Wall-clock time so move duration doesn't depend on thread scheduling.
            # q_d, qd_d, qdd_d from the S-curve so the controller gets the derivatives.
            d = [target[i] - start[i] for i in range(4)]
            t0 = time.monotonic(); t = 0.0
            while t <= T and state.running and self.motion_id == mid:
                s, ds, dds = profile_scurve(min(t, T), T)
                with state.lock:
                    state.set_setpoint([start[i] + d[i] * s for i in range(4)],
                                       [d[i] * ds for i in range(4)], [d[i] * dds for i in range(4)])
                time.sleep(0.02)
                t = time.monotonic() - t0

            if self.motion_id != mid: return False
            with state.lock:
                state.set_setpoint(target)
            return True
        finally:
            if self.motion_id == mid: self._motion_busy = False

    # ══════════════════════════════════════════════════════════════════════
    # TAB 3 - PATH
    # ══════════════════════════════════════════════════════════════════════
    # Parameters of the task-space path types: key -> (label, default, tooltip).
    # Units in the GUI: mm, deg, s, mm/s (converted to SI when the path is built).
    PATH_TYPES = ("Waypoints", "Line", "Circle", "Shape")
    PATH_PARAMS = {
        "Line": [("lx0", "Start X [mm]", 200.0, "Start point (filled by 'Start = current')"),
                 ("ly0", "Start Y [mm]", -60.0, ""), ("lz0", "Start Z [mm]", -20.0, ""),
                 ("lp0", "Start phi [deg]", 0.0, "Tool angle phi = theta1 + theta2 + theta4"),
                 ("lx1", "End X [mm]", 200.0, ""), ("ly1", "End Y [mm]", 80.0, ""),
                 ("lz1", "End Z [mm]", -50.0, ""), ("lp1", "End phi [deg]", 45.0, ""),
                 ("lT", "Time [s]", 2.0, "Duration of the straight line")],
        "Circle": [("cx", "Center X [mm]", 170.0, ""), ("cy", "Center Y [mm]", 0.0, ""),
                   ("cr", "Radius [mm]", 50.0, ""), ("cz", "Z [mm]", -30.0, "Height of the circle plane"),
                   ("cp", "phi [deg]", 0.0, "Constant tool angle"),
                   ("ca", "Start angle [deg]", 0.0, "Where the circle starts, measured from +X"),
                   ("cs", "Sweep [deg]", 360.0, "360 = full circle, negative = clockwise"),
                   ("cT", "Time [s]", 4.0, "Duration of the circle")],
        "Shape": [("sx", "Center X [mm]", 170.0, ""), ("sy", "Center Y [mm]", 0.0, ""),
                  ("ss", "Size [mm]", 30.0, "Text height, or shape size"),
                  ("szu", "Pen up Z [mm]", -20.0, "Height for moving between strokes"),
                  ("szd", "Pen down Z [mm]", -40.0, "Drawing height"),
                  ("sp", "phi [deg]", 0.0, "Constant tool angle"),
                  ("sv", "Speed [mm/s]", 50.0, "Average drawing speed")],
    }

    def _tab_path(self, p):
        lbl = dict(bg=C["panel"], fg=C["muted"], font=(UI_FONT,9), anchor="w")
        rb  = dict(bg=C["panel"], fg=C["text"], selectcolor=C["field"],
                   activebackground=C["panel"], font=(UI_FONT,9))
        sc  = dict(orient="horizontal", bg=C["panel"], fg=C["text"], troughcolor=C["trough"],
                   activebackground=C["accent"], highlightthickness=0, bd=0)

        # Path type + motion profile (shared by every type)
        mf = tk.LabelFrame(p, text="  Trajectory  ", bg=C["panel"],
                           fg=C["text"], font=(UI_FONT,9), padx=6, pady=4)
        mf.pack(fill="x", padx=8, pady=(4,3))
        tk.Label(mf, text="Path:", **lbl).grid(row=0, column=0, sticky="w")
        self.path_type = tk.StringVar(value="Waypoints")
        for c, (val, tip) in enumerate([
                ("Waypoints", "Taught points, joint or straight moves (3.3)"),
                ("Line", "Straight line between two poses (3.1)"),
                ("Circle", "Circle / arc in a horizontal plane (3.2)"),
                ("Shape", "Draw a shape or text with pen up/down (3.3)")], 1):
            self._tip(tk.Radiobutton(mf, text=val, variable=self.path_type, value=val,
                                     command=self._show_path_type, **rb),
                      tip).grid(row=0, column=c, sticky="w", padx=4)
        tk.Label(mf, text="Profile:", **lbl).grid(row=1, column=0, sticky="w")
        self.profile_var = tk.StringVar(value="S-Curve")
        for c, (pr, tip) in enumerate([
                ("Linear", "Constant velocity (jerky start/stop)"),
                ("Trapezoidal", "Accelerate, cruise, decelerate"),
                ("S-Curve", "Quintic: zero velocity and acceleration at both ends (default)")], 1):
            self._tip(tk.Radiobutton(mf, text=pr, variable=self.profile_var, value=pr, **rb),
                      tip).grid(row=1, column=c, sticky="w", padx=4)

        body = tk.Frame(p, bg=C["bg"]); body.pack(fill="both", expand=True)
        self.path_frames = {t: tk.Frame(body, bg=C["bg"]) for t in self.PATH_TYPES}

        # ── Waypoints ──
        wf = self.path_frames["Waypoints"]
        tf = tk.Frame(wf, bg=C["bg"]); tf.pack(fill="x", padx=8, pady=(2,2))
        self._tip(ttk.Button(tf, text="Add waypoint", style="A.TButton",
                             command=self._add_waypoint),
                  "Save the current pose with the time/dwell below").pack(side="left", padx=2)
        self._tip(ttk.Button(tf, text="Clear all", style="D.TButton",
                             command=self._clear_waypoints),
                  "Delete all waypoints (asks first)").pack(side="left", padx=2)
        of = tk.Frame(wf, bg=C["panel"]); of.pack(fill="x", padx=8, pady=2)
        tk.Label(of, text="Move:", **lbl).grid(row=0, column=0, sticky="w")
        self.move_var = tk.StringVar(value="joint")
        for c, (val, txt, tip) in enumerate([
                ("joint", "Joint (PTP)", "Interpolate joint angles (curved tool path)"),
                ("linear", "Linear (XYZ)", "Straight tool line between waypoints;\n"
                                           "checked for reach and elbow flips before running")], 1):
            self._tip(tk.Radiobutton(of, text=txt, variable=self.move_var, value=val, **rb),
                      tip).grid(row=0, column=c, sticky="w", padx=4)
        tk.Label(of, text="Time [s]:", **lbl).grid(row=1, column=0, sticky="sw", pady=(0,2))
        self.speed_var = tk.DoubleVar(value=1.0)
        self._tip(tk.Scale(of, from_=0.2, to=5.0, resolution=0.1, variable=self.speed_var,
                           length=110, **sc),
                  "Seconds per segment for new waypoints").grid(row=1, column=1, sticky="w")
        tk.Label(of, text="Dwell [s]:", **lbl).grid(row=1, column=2, sticky="se",
                                                     padx=(8,0), pady=(0,2))
        self.dwell_var = tk.DoubleVar(value=0.3)
        self._tip(tk.Scale(of, from_=0.0, to=5.0, resolution=0.1, variable=self.dwell_var,
                           length=90, **sc),
                  "Pause at each new waypoint").grid(row=1, column=3, sticky="w")
        self.loop_var = tk.BooleanVar(value=False)
        self._tip(tk.Checkbutton(of, text="Loop", variable=self.loop_var, **rb),
                  "Repeat the path until stopped").grid(row=1, column=4, sticky="sw",
                                                       padx=6, pady=(0,2))

        lf = tk.LabelFrame(wf, text="  Waypoints  (double-click to edit)  ", bg=C["panel"],
                           fg=C["text"], font=(UI_FONT,9), pady=4, padx=6)
        lf.pack(fill="both", expand=True, padx=8, pady=3)
        cols = ("No.","X mm","Y mm","Z mm","phi deg", "Spd", "Dwl")
        self.wp_tree = ttk.Treeview(lf, columns=cols, show="headings",
                                     height=5, selectmode="browse")
        st = ttk.Style()
        st.configure("Treeview", background=C["field"], foreground=C["text"],
                     fieldbackground=C["field"], font=(MONO_FONT,9), rowheight=20,
                     bordercolor=C["border"])
        st.map("Treeview", background=[("selected", C["row_sel"])],
               foreground=[("selected", C["text"])])
        st.configure("Treeview.Heading", background=C["heading"],
                     bordercolor=C["border"], lightcolor=C["heading"], darkcolor=C["heading"],
                     foreground=C["text"], font=(UI_FONT,8,"bold"))
        for col in cols:
            self.wp_tree.heading(col, text=col)
            w = 55 if col != "No." else 35
            self.wp_tree.column(col, width=w, anchor="center")
        self.wp_tree.tag_configure("active", background=C["row_active"], foreground=C["text"])
        self.wp_tree.tag_configure("bad", background=C["row_bad"], foreground=C["danger"])
        self.wp_tree.bind("<Double-1>", self._edit_waypoint)
        vsb = ttk.Scrollbar(lf, orient="vertical", command=self.wp_tree.yview)
        self.wp_tree.configure(yscrollcommand=vsb.set)
        self.wp_tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        wbf = tk.Frame(wf, bg=C["bg"]); wbf.pack(fill="x", padx=8, pady=1)
        for txt,sty,cmd,tip in [("Up","W.TButton",lambda:self._move_wp(-1), "Move selected up"),
                                ("Down","W.TButton",lambda:self._move_wp(+1), "Move selected down"),
                                ("Edit","W.TButton",self._edit_waypoint, "Edit speed / dwell / pose"),
                                ("Go to","A.TButton",self._goto_selected, "Move smoothly to the selected waypoint"),
                                ("Delete","D.TButton",self._delete_waypoint, "Delete selected waypoint")]:
            self._tip(ttk.Button(wbf, text=txt, style=sty, command=cmd),
                      tip).pack(side="left", padx=2)

        # ── Line / Circle / Shape parameters ──
        if not hasattr(self, "traj_vars"):    # survives a theme rebuild
            self.traj_vars = {k: tk.DoubleVar(value=d) for specs in self.PATH_PARAMS.values()
                              for k, _, d, _ in specs}
            self.shape_kind = tk.StringVar(value="Text")
            self.shape_text = tk.StringVar(value="HUST")
            self.corner_var = tk.StringVar(value="stop")
        else:                                   # re-bind the old values to new Tk vars
            self.traj_vars = {k: tk.DoubleVar(value=v.get()) for k, v in self.traj_vars.items()}
            self.shape_kind = tk.StringVar(value=self.shape_kind.get())
            self.shape_text = tk.StringVar(value=self.shape_text.get())
            self.corner_var = tk.StringVar(value=self.corner_var.get())
        for ptype, specs in self.PATH_PARAMS.items():
            fr = tk.LabelFrame(self.path_frames[ptype], text=f"  {ptype} parameters  ",
                               bg=C["panel"], fg=C["text"], font=(UI_FONT,9), padx=6, pady=4)
            fr.pack(fill="x", padx=8, pady=3)
            row = 0
            if ptype == "Shape":
                tk.Label(fr, text="Draw:", **lbl).grid(row=0, column=0, sticky="w", pady=2)
                ttk.Combobox(fr, textvariable=self.shape_kind, state="readonly", width=10,
                             values=["Text"] + list(shapes.SHAPES)).grid(row=0, column=1, sticky="w")
                tk.Label(fr, text="Text:", **lbl).grid(row=0, column=2, sticky="w", padx=(10,0))
                self._tip(tk.Entry(fr, textvariable=self.shape_text, width=12, bg=C["field"],
                                   fg=C["text"], insertbackground=C["text"], font=(MONO_FONT,9),
                                   relief="flat", highlightthickness=1,
                                   highlightbackground=C["border"], highlightcolor=C["accent"]),
                          "Letters A-Z, digits 0-9 and spaces").grid(row=0, column=3, sticky="w")
                row = 1
            for n, (key, text, _, tip) in enumerate(specs):
                r, c = row + n // 2, (n % 2) * 2
                tk.Label(fr, text=text, **lbl).grid(row=r, column=c, sticky="w", pady=2,
                                                    padx=(0 if c == 0 else 10, 4))
                e = tk.Entry(fr, textvariable=self.traj_vars[key], width=9, bg=C["field"],
                             fg=C["text"], insertbackground=C["text"], font=(MONO_FONT,9),
                             relief="flat", highlightthickness=1,
                             highlightbackground=C["border"], highlightcolor=C["accent"])
                e.grid(row=r, column=c + 1, sticky="w")
                if tip: self._tip(e, tip)
            last = row + (len(specs) + 1) // 2
            if ptype == "Line":
                self._tip(ttk.Button(fr, text="Start = current", command=self._line_from_current),
                          "Fill the start pose with the current tool pose"
                          ).grid(row=last, column=0, columnspan=2, sticky="w", pady=(4,0))
            if ptype == "Shape":
                tk.Label(fr, text="Corners:", **lbl).grid(row=last, column=0, sticky="w", pady=2)
                for c, (val, txt, tip) in enumerate([
                        ("stop", "Stop", "Stop at every corner of a stroke"),
                        ("smooth", "Smooth", "One time law along each stroke (no stop at corners)")], 1):
                    self._tip(tk.Radiobutton(fr, text=txt, variable=self.corner_var, value=val, **rb),
                              tip).grid(row=last, column=c, sticky="w")
            tk.Label(self.path_frames[ptype], bg=C["bg"], fg=C["muted"], font=(UI_FONT,8),
                     justify="left", text="The robot first moves (joint move) to the start of the path.\n"
                                          "Elbow choice: Jog tab.  Preview draws the path on the top view."
                     ).pack(anchor="w", padx=10)

        # ── Run / preview / progress (shared) ──
        rf = tk.Frame(p, bg=C["bg"]); rf.pack(fill="x", padx=8, pady=4)
        self.run_btn = self._tip(ttk.Button(rf, text="Run path", style="G.TButton",
                                            command=self._run_path),
                                 "Build, check and run the selected path")
        self.run_btn.pack(side="left", padx=4)
        self.stop_btn = self._tip(ttk.Button(rf, text="Stop path", style="D.TButton",
                                             command=self._stop_path, state="disabled"),
                                  "Stop the path and hold position")
        self.stop_btn.pack(side="left", padx=4)
        self._tip(ttk.Button(rf, text="Preview", command=self._preview_path),
                  "Check the path and draw it on the top view without moving").pack(side="left", padx=4)
        self.path_pb = ttk.Progressbar(rf, mode="determinate", maximum=1,
                                       style="G.Horizontal.TProgressbar")
        self.path_pb.pack(side="left", fill="x", expand=True, padx=6)
        self.path_progress = tk.StringVar()
        tk.Label(p, textvariable=self.path_progress, bg=C["bg"], fg=C["muted"],
                 font=(UI_FONT,9)).pack(anchor="w", padx=8)
        self._show_path_type()

    def _show_path_type(self):
        for t, fr in self.path_frames.items():
            fr.pack_forget()
        self.path_frames[self.path_type.get()].pack(fill="both", expand=True)

    def _line_from_current(self):
        with state.lock: q = list(state.ctrl)
        x, y, z, phi = kin.fk(q)
        for k, v in (("lx0", x * 1000), ("ly0", y * 1000), ("lz0", z * 1000), ("lp0", math.degrees(phi))):
            self.traj_vars[k].set(round(v, 2))

    def _load_waypoints(self):
        if not os.path.exists(WAYPOINTS_PATH): return
        try:
            with open(WAYPOINTS_PATH, "r") as f:
                text = f.read().strip()
            self.waypoints = json.loads(text) if text else []
        except (OSError, ValueError) as e:
            print("Error loading waypoints:", e)
            self.waypoints = []
        self._refresh_wp()

    def _save_waypoints(self):
        try:
            with open(WAYPOINTS_PATH, "w") as f:
                json.dump(self.waypoints, f)
        except Exception as e:
            print("Error saving waypoints:", e)

    def _add_waypoint(self):
        with state.lock: j1,j2,j3,j4=list(state.ctrl)
        spd = self.speed_var.get(); dwl = self.dwell_var.get()
        self.waypoints.append([j1,j2,j3,j4, spd, dwl])
        self._refresh_wp()
        self._notify(f"Waypoint {len(self.waypoints)} added", "ok")
        self._save_waypoints()

    def _clear_waypoints(self):
        if not self.waypoints: return
        if not messagebox.askyesno("Clear waypoints",
                                   f"Delete all {len(self.waypoints)} waypoints?",
                                   parent=self.root):
            return
        self.waypoints.clear(); self._refresh_wp()
        self._save_waypoints()
        self._notify("All waypoints cleared")

    def _delete_waypoint(self):
        sel=self.wp_tree.selection()
        if not sel: return
        self.waypoints.pop(int(sel[0])); self._refresh_wp()
        self._save_waypoints()

    def _move_wp(self, d):
        sel=self.wp_tree.selection()
        if not sel: return
        idx=int(sel[0]); ni=idx+d
        if ni<0 or ni>=len(self.waypoints): return
        self.waypoints[idx],self.waypoints[ni]=self.waypoints[ni],self.waypoints[idx]
        self._refresh_wp(); self.wp_tree.selection_set(str(ni))
        self._save_waypoints()

    def _refresh_wp(self):
        for i in self.wp_tree.get_children(): self.wp_tree.delete(i)
        for i,wp in enumerate(self.waypoints):
            j1,j2,j3,j4 = wp[:4]
            spd = wp[4] if len(wp)>4 else 1.0
            dwl = wp[5] if len(wp)>5 else 0.3
            x,y,z=fk(j1,j2,j3)
            self.wp_tree.insert("","end",iid=str(i),
                values=(i+1,f"{x*1000:.1f}",f"{y*1000:.1f}",f"{z*1000:.1f}",f"{math.degrees(tool_phi(wp)):.1f}", f"{spd:.1f}", f"{dwl:.1f}"))

    def _highlight_wp(self, idx, tag="active"):
        for iid in self.wp_tree.get_children(): self.wp_tree.item(iid, tags=())
        if idx is not None and self.wp_tree.exists(str(idx)):
            self.wp_tree.item(str(idx), tags=(tag,)); self.wp_tree.see(str(idx))

    def _edit_waypoint(self, event=None):
        """Dialog to edit speed / dwell, optionally replacing the pose."""
        sel = self.wp_tree.selection()
        if not sel or self.path_running: return
        i = int(sel[0]); wp = self.waypoints[i]
        dlg = tk.Toplevel(self.root); dlg.title(f"Edit waypoint {i+1}")
        dlg.configure(bg=C["panel"], padx=10, pady=8)
        dlg.transient(self.root); dlg.resizable(False, False)
        spd = tk.DoubleVar(value=wp[4] if len(wp)>4 else 1.0)
        dwl = tk.DoubleVar(value=wp[5] if len(wp)>5 else 0.3)
        repose = tk.BooleanVar(value=False)
        for r, (label, var, lo, hi) in enumerate([("Speed [sec/point]", spd, 0.2, 5.0),
                                                   ("Dwell [sec]", dwl, 0.0, 5.0)]):
            tk.Label(dlg, text=label, bg=C["panel"], fg=C["text"],
                     font=(UI_FONT,9)).grid(row=r, column=0, sticky="w", pady=4)
            tk.Spinbox(dlg, from_=lo, to=hi, increment=0.1, format="%.1f", width=7,
                       textvariable=var, bg=C["field"], fg=C["text"], font=(MONO_FONT,10),
                       buttonbackground=C["panel"], insertbackground=C["text"]
                       ).grid(row=r, column=1, sticky="w", padx=8)
        tk.Checkbutton(dlg, text="Replace pose with current robot position", variable=repose,
                       bg=C["panel"], fg=C["text"], selectcolor=C["field"],
                       activebackground=C["panel"], font=(UI_FONT,9)
                       ).grid(row=2, column=0, columnspan=2, sticky="w", pady=4)

        def save():
            try:
                sv = min(max(spd.get(), 0.2), 5.0); dv = min(max(dwl.get(), 0.0), 5.0)
            except tk.TclError:
                self._notify("!! Invalid speed/dwell", "warn"); return
            if repose.get():
                with state.lock: pose = list(state.ctrl)
            else:
                pose = list(wp[:4])
            self.waypoints[i] = pose + [sv, dv]
            self._refresh_wp(); self.wp_tree.selection_set(str(i))
            self._save_waypoints()
            self._notify(f"Waypoint {i+1} updated", "ok")
            dlg.destroy()

        bf = tk.Frame(dlg, bg=C["panel"]); bf.grid(row=3, column=0, columnspan=2, pady=(6,0))
        ttk.Button(bf, text="Save", style="G.TButton", command=save).pack(side="left", padx=4)
        ttk.Button(bf, text="Cancel", style="D.TButton", command=dlg.destroy).pack(side="left", padx=4)
        dlg.bind("<Return>", lambda e: save())
        dlg.bind("<Escape>", lambda e: dlg.destroy())
        dlg.vars = (spd, dwl, repose); dlg.save = save   # handles for tests/automation
        self._edit_dlg = dlg
        return dlg

    def _goto_selected(self):
        sel=self.wp_tree.selection()
        if not sel: return
        wp = self.waypoints[int(sel[0])]
        if len(wp) > 4:
            self.speed_var.set(wp[4]); self.dwell_var.set(wp[5])
        mid = self._new_motion()
        self._notify(f"Moving to waypoint {int(sel[0])+1}")
        threading.Thread(target=self._joint_move, args=(list(wp[:4]), mid), daemon=True).start()

    @staticmethod
    def _elbow(j2):
        """Elbow sign of a pose: +1 / -1, or 0 when the arm is (nearly) straight."""
        return 0 if abs(j2) < 1e-3 else (1 if j2 > 0 else -1)

    # ── building trajectories (main thread: reads Tk variables) ──
    def _path_builder(self):
        """Snapshot the selected path's settings and return build(q_now) -> Trajectory.
        Raises TrajectoryError / ValueError for bad input."""
        ptype, profile = self.path_type.get(), self.profile_var.get()
        elbow = self.elbow_var.get()
        if ptype == "Waypoints":
            if not self.waypoints:
                raise traj_mod.TrajectoryError("no waypoints - add some first")
            cfg = dict(linear=self.move_var.get() == "linear", speed=self.speed_var.get(),
                       dwell=self.dwell_var.get(), profile=profile,
                       wps=[list(w) for w in self.waypoints])
            return lambda q: self._waypoint_traj(q, cfg), self.loop_var.get()
        v = {k: var.get() for k, var in self.traj_vars.items()}   # TclError on bad numbers
        mm, deg = 0.001, math.radians
        if ptype == "Line":
            X0 = [v["lx0"] * mm, v["ly0"] * mm, v["lz0"] * mm, deg(v["lp0"])]
            X1 = [v["lx1"] * mm, v["ly1"] * mm, v["lz1"] * mm, deg(v["lp1"])]
            build = lambda q: traj_mod.line(X0, X1, v["lT"], q, profile, elbow)
        elif ptype == "Circle":
            build = lambda q: traj_mod.circle((v["cx"] * mm, v["cy"] * mm), v["cr"] * mm, v["cz"] * mm,
                                              deg(v["cp"]), v["cT"], q, deg(v["ca"]), deg(v["cs"]),
                                              profile, elbow)
        else:
            kind, size = self.shape_kind.get(), v["ss"] * mm
            if kind == "Text":
                lines = shapes.centered_text(self.shape_text.get().strip(), v["sx"] * mm, v["sy"] * mm, size)
            else:
                lines = shapes.SHAPES[kind](v["sx"] * mm, v["sy"] * mm, size)
            corner = self.corner_var.get()
            build = lambda q: traj_mod.strokes(lines, v["szu"] * mm, v["szd"] * mm, deg(v["sp"]),
                                               v["sv"] * mm, q, corner, profile, elbow)
        return (lambda q: traj_mod.approach(build(q), q, profile)), False

    def _waypoint_traj(self, q_now, cfg):
        """Waypoint path as one trajectory: a joint or straight move to each waypoint
        (time = its Spd), then its dwell. Segment labels 'point i' drive the progress."""
        wps, parts, q_prev = cfg["wps"], [], np.asarray(q_now, float)
        for i, wp in enumerate(wps):
            target = np.asarray(wp[:4], float)
            T = wp[4] if len(wp) > 4 else cfg["speed"]
            dwell = wp[5] if len(wp) > 5 else cfg["dwell"]
            name = f"point {i+1}"
            if cfg["linear"] and not kin.is_singular(q_prev):
                ea, eb = self._elbow(q_prev[1]), self._elbow(target[1])
                if ea and eb and ea != eb:
                    raise traj_mod.TrajectoryError(
                        f"Segment to waypoint {i+1} needs an elbow flip - use Joint mode", )
                side = "left" if (eb or ea or 1) > 0 else "right"
                try:
                    seg = traj_mod.line(kin.fk(q_prev), kin.fk(target), T, q_prev, cfg["profile"], side)
                except traj_mod.TrajectoryError:
                    raise traj_mod.TrajectoryError(
                        f"Straight line to waypoint {i+1} leaves the workspace or joint limits "
                        f"- use Joint mode") from None
            else:   # joint move (also used to leave a singular pose such as Home)
                seg = traj_mod.joint_move(q_prev, target, T, cfg["profile"])
            seg.segments = [(0.0, name)]
            parts.append(seg); q_prev = seg.q[-1]
            if dwell > 0:
                h = traj_mod.hold(q_prev, dwell); h.segments = [(0.0, name)]
                parts.append(h)
        tr = traj_mod.Trajectory.concat(parts, "waypoints")
        tr.waypoint_count = len(wps)
        return tr

    def _check_path(self):
        """Build and validate the selected path from the current setpoint.
        Returns (trajectory, rebuild-or-None) or None after reporting the problem."""
        try:
            build, loop = self._path_builder()
            with state.lock: q_now = list(state.ctrl)
            tr = build(q_now)
            issues = traj_mod.validate(tr)
            warn = None
            if self.profile_var.get() == "Linear":     # chosen on purpose: jerky starts allowed
                jumps = [i for i in issues if "velocity jumps" in i]
                issues = [i for i in issues if i not in jumps]
                warn = jumps[0] if jumps else None
            if issues: raise traj_mod.TrajectoryError(issues[0])
        except (traj_mod.TrajectoryError, ValueError, tk.TclError) as e:
            msg = f"!! {e}" if not isinstance(e, tk.TclError) else "!! Invalid number in the parameters"
            m = re.search(r"waypoint (\d+)", msg)
            if m and self.path_type.get() == "Waypoints":
                self._highlight_wp(int(m.group(1)) - 1, "bad")
            self._notify(msg, "error"); self.path_progress.set(msg)
            return None
        self.plan_xy = [(float(x), float(y)) for x, y in tr.X[:, :2]]
        self._plan_dirty = True
        if warn:
            self._notify(f"Linear profile: {warn}", "warn")
        return tr, (build if loop else None)

    def _preview_path(self):
        if self.path_running: return
        res = self._check_path()
        if res:
            tr = res[0]
            msg = f"Path OK: {tr.duration:.1f} s, {len(tr.segments)} segments (shown on the top view)"
            self._notify(msg, "ok"); self.path_progress.set(msg)

    def _run_path(self):
        if self.path_running: return
        res = self._check_path()
        if not res: return
        tr, rebuild = res
        mid = self._new_motion()
        self.path_running = True
        self.path_index, self.path_total = 0, getattr(tr, "waypoint_count", 0)
        self.path_status = "Path"
        self.run_btn.state(["disabled"]); self.stop_btn.state(["!disabled"])
        self.wp_tree.selection_remove(*self.wp_tree.selection())   # so the active row shows
        self.path_pb.config(maximum=tr.duration, value=0)
        self._notify(f"Running {self.path_type.get().lower()} path ({tr.duration:.1f} s, "
                     f"{self.profile_var.get()}, {state.mode})")
        threading.Thread(target=self._exec_traj, args=(mid, tr, rebuild), daemon=True).start()

    def _stop_path(self): self._stop_all()

    def _traj_step(self, name, t, duration):
        """Main-thread UI update when the trajectory enters a new segment."""
        if name.startswith("point ") and self.path_total:
            i = int(name.split()[1])
            self.path_index = i
            self.path_status = f"Path {i}/{self.path_total}"
            self.path_progress.set(f"Moving to point {i}/{self.path_total}")
            self._highlight_wp(i - 1)
        else:
            self.path_status = f"Path: {name}"
            self.path_progress.set(f"{name}  ({t:.1f} / {duration:.1f} s)")

    def _path_done(self, mid, finished):
        if self.motion_id == mid: self.path_running = False
        self.run_btn.state(["!disabled"]); self.stop_btn.state(["disabled"])
        if finished: self.path_pb.config(value=float(self.path_pb.cget("maximum")))
        self.path_progress.set("Path finished" if finished else "Path stopped")
        self._notify("Path finished" if finished else "Path stopped",
                     "ok" if finished else "warn")
        self._highlight_wp(None); self.path_index = 0
        self._update_trajectory_plot()

    def _exec_traj(self, mid, tr, rebuild):
        """Play a trajectory in real time (scaled by the speed override). Runs in a worker
        thread: never touches Tk directly, only through self.ui_queue."""
        alive = lambda: self.path_running and self.motion_id == mid and state.running
        recorder.start()
        finished = False
        while alive():
            starts = [ts for ts, _ in tr.segments]
            t, last, seg, n = 0.0, time.monotonic(), -1, 0
            while alive() and t <= tr.duration:
                q, qd, qdd = tr.sample(t)
                ov = self.override
                with state.lock:
                    state.set_setpoint(q, qd * ov, qdd * ov * ov)
                k = max(0, int(np.searchsorted(starts, t, side="right")) - 1)
                if k != seg:
                    seg = k
                    self.ui_queue.put((self._traj_step, (tr.segments[k][1], t, tr.duration)))
                if n % 10 == 0:
                    self.ui_queue.put((lambda v: self.path_pb.config(value=v), (t,)))
                n += 1
                time.sleep(0.01)
                now = time.monotonic(); t += (now - last) * ov; last = now
            if not alive(): break            # stopped: hold where it is, don't snap
            with state.lock:
                state.set_setpoint(tr.q[-1])
            if rebuild is None:
                finished = True; break
            with state.lock: q_now = list(state.ctrl)
            try:
                tr = rebuild(q_now)           # next lap of a looping waypoint path
            except traj_mod.TrajectoryError:
                break
        time.sleep(0.05)                     # let the sim thread record the last samples
        recorder.stop()
        self.ui_queue.put((self._path_done, (mid, finished)))

    # ══════════════════════════════════════════════════════════════════════
    # TAB 4 - TRAJECTORY PLOT
    # ══════════════════════════════════════════════════════════════════════
    PLOT_VIEWS = [("joints", "Joints", "Joint position, velocity and acceleration"),
                  ("error", "Error", "Tracking error: joints, tool position and tool angle"),
                  ("torque", "Torque", "Actuator torque / force with its limits"),
                  ("tool", "Tool", "Tool pose x, y, z, phi over time"),
                  ("xy", "XY path", "Tool path seen from above: reference vs response")]

    def _tab_trajectory(self, p):
        cf = tk.Frame(p, bg=C["bg"]); cf.pack(fill="x", padx=8, pady=(4,0))
        ttk.Button(cf, text="Refresh plot", style="A.TButton",
                   command=self._update_trajectory_plot).pack(side="left", padx=3)
        ttk.Button(cf, text="Clear data", style="D.TButton",
                   command=self._clear_trajectory).pack(side="left", padx=3)
        tk.Label(cf, text="View:", bg=C["bg"], fg=C["muted"],
                 font=(UI_FONT,9)).pack(side="left", padx=(10,2))
        self.plot_view = tk.StringVar(value="joints")
        for val, txt, tip in self.PLOT_VIEWS:
            self._tip(tk.Radiobutton(cf, text=txt, variable=self.plot_view, value=val,
                                     command=self._update_trajectory_plot,
                                     bg=C["bg"], fg=C["text"], selectcolor=C["field"],
                                     activebackground=C["bg"], font=(UI_FONT,9)),
                      tip).pack(side="left", padx=1)

        cf2 = tk.Frame(p, bg=C["bg"]); cf2.pack(fill="x", padx=8, pady=(0,2))
        tk.Label(cf2, text="Show:", bg=C["bg"], fg=C["muted"],
                 font=(UI_FONT,9)).pack(side="left", padx=(3,2))
        self.plot_series = tk.StringVar(value="both")
        for val, txt, tip in [("cmd", "Reference", "The desired trajectory (set point)"),
                              ("meas", "Response", "What the simulated robot actually did"),
                              ("both", "Both", "Response solid, reference dashed")]:
            self._tip(tk.Radiobutton(cf2, text=txt, variable=self.plot_series, value=val,
                                     command=self._update_trajectory_plot,
                                     bg=C["bg"], fg=C["text"], selectcolor=C["field"],
                                     activebackground=C["bg"], font=(UI_FONT,9)),
                      tip).pack(side="left", padx=2)
        self.plot_joints = []
        for i, (name, color) in enumerate(zip(["J1","J2","J3","J4"], JOINT_COLORS)):
            var = tk.BooleanVar(value=True); self.plot_joints.append(var)
            tk.Checkbutton(cf2, text=name, variable=var, command=self._update_trajectory_plot,
                           bg=C["bg"], fg=color, selectcolor=C["field"],
                           activebackground=C["bg"],
                           font=(UI_FONT,9,"bold")).pack(side="left", padx=3)
        self.plot_stats = tk.StringVar()
        tk.Label(p, textvariable=self.plot_stats, bg=C["bg"], fg=C["muted"],
                 font=(MONO_FONT,8), anchor="w", justify="left").pack(fill="x", padx=10)

        self.traj_fig = Figure(figsize=(6, 5), dpi=100, facecolor=C["bg"])  # matches window
        self.traj_canvas = FigureCanvasTkAgg(self.traj_fig, master=p)
        # Small requested size: the widget grows to fill, but doesn't squeeze the right panel
        self.traj_canvas.get_tk_widget().config(width=300, height=300, bg=C["bg"],
                                                highlightthickness=0)
        self.traj_canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=2)
        self._update_trajectory_plot()

    def _clear_trajectory(self):
        recorder.clear(); self._update_trajectory_plot()

    def _style_ax(self, ax, ylabel, xlabel=None):
        ax.set_facecolor(C["field"])
        ax.grid(True, color=C["plot_grid"], linewidth=0.6)
        ax.tick_params(colors=C["muted"], labelsize=7)
        for spine in ax.spines.values(): spine.set_color(C["border"])
        ax.set_ylabel(ylabel, color=C["text"], fontsize=8)
        if xlabel: ax.set_xlabel(xlabel, color=C["text"], fontsize=8)

    def _legend(self, ax, ncol=1):
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper right", fontsize=7, ncol=ncol,
                      facecolor=C["field"], edgecolor=C["border"], labelcolor=C["text"])

    @staticmethod
    def _joint_units(i, arr):
        return arr * 1000 if i == 2 else np.degrees(arr)

    def _update_trajectory_plot(self):
        t_data, data = recorder.get_data()
        tau = np.array(recorder.get_tau()).T if t_data else None
        fig, view = self.traj_fig, self.plot_view.get()
        fig.clf()
        fig.subplots_adjust(hspace=0.5, wspace=0.3, left=0.11, right=0.97, top=0.95, bottom=0.09)
        self.plot_stats.set("")
        if not t_data:
            ax = self.ax_q = fig.add_subplot(111)
            self._style_ax(ax, "")
            ax.text(0.5, 0.5, "No data - run a path first", transform=ax.transAxes,
                    ha="center", va="center", color=C["muted"], fontsize=11)
            self.traj_canvas.draw(); return
        t = np.array(t_data)
        cmd = {n: np.array(data["cmd"][n]).T for n in ("q", "qd", "qdd")}     # (N, 4)
        meas = {n: np.array(data["meas"][n]).T for n in ("q", "qd", "qdd")}
        joints = [i for i in range(4) if self.plot_joints[i].get()]
        mode = self.plot_series.get()
        series = [("meas", "-", ""), ("cmd", "--", " ref")] if mode == "both" else [(mode, "-", "")]
        src = {"cmd": cmd, "meas": meas}

        if view == "joints":
            axes = [fig.add_subplot(3, 1, k) for k in (1, 2, 3)]
            self.ax_q = axes[0]
            for ax, key, unit in zip(axes, ("q", "qd", "qdd"),
                                     ("Position", "Velocity", "Acceleration")):
                self._style_ax(ax, f"{unit} (deg, mm)")
                for i in joints:
                    for name, ls, suffix in series:
                        units = (JOINT_UNITS, VEL_UNITS, ACC_UNITS)[("q", "qd", "qdd").index(key)][i]
                        ax.plot(t, self._joint_units(i, src[name][key][:, i]), color=JOINT_COLORS[i],
                                linestyle=ls, linewidth=1.5 if ls == "-" else 1.0,
                                label=f"{JOINT_NAMES[i]}{suffix} [{units}]")
                self._legend(ax, 2 if mode == "both" else 1)
            axes[-1].set_xlabel("Time (s)", color=C["text"], fontsize=8)

        elif view == "error":
            e = cmd["q"] - meas["q"]
            Xr = np.array([kin.fk(q) for q in cmd["q"]]); Xm = np.array([kin.fk(q) for q in meas["q"]])
            ep = np.linalg.norm(Xr[:, :3] - Xm[:, :3], axis=1) * 1000
            ephi = np.degrees([wrap_angle(a - b) for a, b in zip(Xr[:, 3], Xm[:, 3])])
            a1, a2, a3 = (fig.add_subplot(3, 1, k) for k in (1, 2, 3))
            self.ax_q = a1
            self._style_ax(a1, "Joint error (deg, mm)")
            for i in joints:
                a1.plot(t, self._joint_units(i, e[:, i]), color=JOINT_COLORS[i],
                        label=f"e{JOINT_NAMES[i]} [{JOINT_UNITS[i]}]")
            self._legend(a1, 2)
            self._style_ax(a2, "Tool position error (mm)")
            a2.plot(t, ep, color=C["accent"], label="|p_ref - p|")
            self._legend(a2)
            self._style_ax(a3, "Tool angle error (deg)", "Time (s)")
            a3.plot(t, ephi, color=C["warn"], label="phi_ref - phi")
            self._legend(a3)
            rms = np.sqrt((e ** 2).mean(axis=0))
            self.plot_stats.set(
                "RMS joint error: " + "  ".join(f"{JOINT_NAMES[i]} {self._joint_units(i, rms[i]):.3f}"
                                                for i in range(4))
                + f"   |  tool: max {ep.max():.3f} mm, RMS {np.sqrt((ep**2).mean()):.3f} mm,"
                  f" angle max {np.abs(ephi).max():.3f} deg")

        elif view == "torque":
            axes = [fig.add_subplot(2, 2, k) for k in (1, 2, 3, 4)]
            self.ax_q = axes[0]
            for i, ax in enumerate(axes):
                unit = "N" if i == 2 else "N*m"
                self._style_ax(ax, f"{JOINT_NAMES[i]} [{unit}]", "Time (s)" if i >= 2 else None)
                ax.plot(t, tau[:, i], color=JOINT_COLORS[i], label="tau")
                lim = float(ctl.TAU_LIMITS[i])
                for v in (lim, -lim):
                    ax.axhline(v, color=C["danger"], linestyle="--", linewidth=0.8)
            peak = np.abs(tau).max(axis=0)
            self.plot_stats.set("Peak |tau|: " + "  ".join(
                f"{JOINT_NAMES[i]} {peak[i]:.2f} {'N' if i == 2 else 'N*m'}" for i in range(4))
                + f"   (mode: {state.mode}; kinematic mode shows the torque the reference needs)")

        elif view == "tool":
            Xr = np.array([kin.fk(q) for q in cmd["q"]]); Xm = np.array([kin.fk(q) for q in meas["q"]])
            axes = [fig.add_subplot(2, 2, k) for k in (1, 2, 3, 4)]
            self.ax_q = axes[0]
            for k, (ax, name) in enumerate(zip(axes, ("x (mm)", "y (mm)", "z (mm)", "phi (deg)"))):
                self._style_ax(ax, name, "Time (s)" if k >= 2 else None)
                conv = (lambda v: np.degrees(np.unwrap(v))) if k == 3 else (lambda v: v * 1000)
                for which, X, ls in (("meas", Xm, "-"), ("cmd", Xr, "--")):
                    if mode in (which, "both"):
                        ax.plot(t, conv(X[:, k]), color=C["accent"] if which == "meas" else C["muted"],
                                linestyle=ls, label="response" if which == "meas" else "reference")
                self._legend(ax)

        else:   # XY path
            Xr = np.array([kin.fk(q) for q in cmd["q"]]); Xm = np.array([kin.fk(q) for q in meas["q"]])
            ax = self.ax_q = fig.add_subplot(111)
            self._style_ax(ax, "y (mm)", "x (mm)")
            if mode in ("cmd", "both"):
                ax.plot(Xr[:, 0] * 1000, Xr[:, 1] * 1000, "--", color=C["muted"], label="reference")
            if mode in ("meas", "both"):
                ax.plot(Xm[:, 0] * 1000, Xm[:, 1] * 1000, color=C["accent"], label="response")
            ax.set_aspect("equal", adjustable="datalim")
            self._legend(ax)
        self.traj_canvas.draw()

    # ══════════════════════════════════════════════════════════════════════
    # TAB 5 - PID
    # ══════════════════════════════════════════════════════════════════════
    def _tab_pid(self, p):
        info = ("u = Kp e + Ki \u222be dt + Kd (qd_ref - qd) + feed-forward,  tau = clip(u, \u00b1tau_max)\n"
                "Auto-tune: triple pole at -wn on J s\u00b2 + B s  ->  Kp = 3J wn\u00b2, Ki = J wn\u00b3, "
                "Kd = 3J wn - B.  Applies in 'PID control' mode.")
        tk.Label(p, text=info, bg=C["bg"], fg=C["muted"], font=(UI_FONT,8), justify="left"
                 ).pack(anchor="w", padx=10, pady=(6,2))
        P = get_params()
        with state.lock: pid = state.pid
        if not hasattr(self, "pid_vars"):
            g = pid.gains if pid else ctl.tuned_gains(P)
            tl = pid.tau_limits if pid else ctl.TAU_LIMITS
            vals = {"wn": ctl.DEFAULT_WN, "kp": g.kp, "ki": g.ki, "kd": g.kd, "tau": tl}
            self.pid_vars = {(k, i): tk.DoubleVar(value=round(float(vals[k][i]), 4))
                             for k in vals for i in range(4)}
            self.ff_var = tk.StringVar(value=pid.feedforward if pid else "gravity")
            self.aw_var = tk.BooleanVar(value=pid.anti_windup if pid else True)
        else:   # theme rebuild: keep the values
            self.pid_vars = {k: tk.DoubleVar(value=v.get()) for k, v in self.pid_vars.items()}
            self.ff_var = tk.StringVar(value=self.ff_var.get())
            self.aw_var = tk.BooleanVar(value=self.aw_var.get())

        gf = tk.LabelFrame(p, text="  Gains per axis  ", bg=C["panel"], fg=C["text"],
                           font=(UI_FONT,9), padx=6, pady=4)
        gf.pack(fill="x", padx=8, pady=3)
        heads = [("", None), ("wn [rad/s]", "wn"), ("Kp", "kp"), ("Ki", "ki"), ("Kd", "kd"),
                 ("tau max", "tau")]
        for c, (h, _) in enumerate(heads):
            tk.Label(gf, text=h, bg=C["panel"], fg=C["muted"], font=(UI_FONT,8,"bold")
                     ).grid(row=0, column=c, padx=4, sticky="w")
        for i in range(4):
            unit = "N" if i == 2 else "N*m"
            tk.Label(gf, text=f"{JOINT_NAMES[i]} ({unit})", bg=C["panel"], fg=C["text"],
                     font=(UI_FONT,9)).grid(row=i + 1, column=0, sticky="w")
            for c, (_, key) in enumerate(heads[1:], 1):
                tk.Entry(gf, textvariable=self.pid_vars[(key, i)], width=10, bg=C["field"],
                         fg=C["text"], insertbackground=C["text"], font=(MONO_FONT,9),
                         relief="flat", highlightthickness=1, highlightbackground=C["border"],
                         highlightcolor=C["accent"]).grid(row=i + 1, column=c, padx=3, pady=1)

        of = tk.Frame(p, bg=C["bg"]); of.pack(fill="x", padx=8, pady=2)
        tk.Label(of, text="Feed-forward:", bg=C["bg"], fg=C["muted"], font=(UI_FONT,9)).pack(side="left")
        for val, txt, tip in [("none", "None", "Pure PID"),
                              ("gravity", "Gravity G(q)", "Add the gravity torque of the dynamic model"),
                              ("full", "Full dynamics", "Add M q''_ref + C q'_ref + G + B q'_ref (computed torque)")]:
            self._tip(tk.Radiobutton(of, text=txt, variable=self.ff_var, value=val, bg=C["bg"],
                                     fg=C["text"], selectcolor=C["field"], activebackground=C["bg"],
                                     font=(UI_FONT,9)), tip).pack(side="left", padx=2)
        self._tip(tk.Checkbutton(of, text="Anti-windup", variable=self.aw_var, bg=C["bg"],
                                 fg=C["text"], selectcolor=C["field"], activebackground=C["bg"],
                                 font=(UI_FONT,9)),
                  "Freeze the integral while the output is saturated (clamping)").pack(side="left", padx=10)

        bf = tk.Frame(p, bg=C["bg"]); bf.pack(fill="x", padx=8, pady=4)
        for txt, cmd, tip in [("Apply", self._apply_pid, "Use these gains now"),
                              ("Auto-tune from wn", self._autotune_pid, "Compute Kp, Ki, Kd from wn (pole placement)"),
                              ("Defaults", self._default_pid, "Reset wn, gains and limits to the defaults")]:
            self._tip(ttk.Button(bf, text=txt, command=cmd), tip).pack(side="left", padx=3)

        lf = tk.LabelFrame(p, text="  Live  ", bg=C["panel"], fg=C["text"],
                           font=(UI_FONT,9), padx=6, pady=4)
        lf.pack(fill="x", padx=8, pady=3)
        self.pid_live = {}
        for c, h in enumerate(("", "error", "torque", "saturated")):
            tk.Label(lf, text=h, bg=C["panel"], fg=C["muted"], font=(UI_FONT,8,"bold")
                     ).grid(row=0, column=c, padx=6, sticky="w")
        for i in range(4):
            tk.Label(lf, text=JOINT_NAMES[i], bg=C["panel"], fg=C["text"],
                     font=(UI_FONT,9)).grid(row=i + 1, column=0, sticky="w")
            for c, key in enumerate(("e", "tau", "sat"), 1):
                lbl = tk.Label(lf, text="-", bg=C["panel"], fg=C["bright"], font=(MONO_FONT,9),
                               width=14, anchor="e")
                lbl.grid(row=i + 1, column=c, padx=6)
                self.pid_live[(key, i)] = lbl

    def _pid_values(self, key):
        return np.array([self.pid_vars[(key, i)].get() for i in range(4)], float)

    def _autotune_pid(self):
        try:
            g = ctl.tuned_gains(get_params(), self._pid_values("wn"))
        except tk.TclError:
            self._notify("!! Invalid number in the PID table", "error"); return
        for i in range(4):
            for key, arr in (("kp", g.kp), ("ki", g.ki), ("kd", g.kd)):
                self.pid_vars[(key, i)].set(round(float(arr[i]), 4))
        self._notify("Gains computed from wn - press Apply to use them")

    def _default_pid(self):
        for i in range(4):
            self.pid_vars[("wn", i)].set(float(ctl.DEFAULT_WN[i]))
            self.pid_vars[("tau", i)].set(float(ctl.TAU_LIMITS[i]))
        self._autotune_pid()

    def _apply_pid(self):
        try:
            kp, ki, kd, tl = (self._pid_values(k) for k in ("kp", "ki", "kd", "tau"))
        except tk.TclError:
            self._notify("!! Invalid number in the PID table", "error"); return
        if np.any(np.array([kp, ki, kd]) < 0) or np.any(tl <= 0):
            self._notify("!! Gains must be >= 0 and tau max > 0", "error"); return
        if np.any(tl > ctl.TAU_LIMITS + 1e-9):
            self._notify(f"!! tau max above the actuator limit {list(ctl.TAU_LIMITS)}", "error"); return
        pid = ctl.PID4(ctl.Gains(kp, ki, kd), tl, feedforward=self.ff_var.get(),
                       params=get_params(), anti_windup=self.aw_var.get())
        with state.lock: state.pid = pid
        self._notify("PID gains applied", "ok")

    def _update_pid_live(self, e, tau, sat):
        for i in range(4):
            unit = "mm" if i == 2 else "deg"
            self.pid_live[("e", i)].config(text=f"{self._joint_units(i, e[i]):+9.3f} {unit}")
            self.pid_live[("tau", i)].config(text=f"{tau[i]:+9.3f} {'N' if i == 2 else 'N*m'}")
            self.pid_live[("sat", i)].config(text="YES" if sat[i] else "no",
                                             fg=C["danger"] if sat[i] else C["bright"])

    # ══════════════════════════════════════════════════════════════════════
    # HOME / TICK / QUIT
    # ══════════════════════════════════════════════════════════════════════
    def _go_home(self):
        """Move safely to Home: cancel any motion, then joint-space S-curve to zero."""
        mid = self._new_motion()
        self._notify("Returning to Home...")
        threading.Thread(target=self._home_move, args=(mid,), daemon=True).start()

    def _home_move(self, mid):
        if self._joint_move([0.0, 0.0, 0.0, 0.0], mid):
            self.ui_queue.put((self._notify, ("At Home", "ok")))

    def _drain_ui_queue(self):
        while True:
            try: fn, args = self.ui_queue.get_nowait()
            except queue.Empty: return
            fn(*args)

    def _tick(self):
        if not state.running: self.root.destroy(); return
        self._drain_ui_queue()
        try:
            with state.lock:
                act_j1, act_j2, act_j3, act_j4 = list(state.qpos)
                tgt_j1, tgt_j2, tgt_j3, tgt_j4 = list(state.ctrl)
            
            x, y, z = fk(act_j1, act_j2, act_j3)
            tx, ty, tz = fk(tgt_j1, tgt_j2, tgt_j3)

            # Trail: only add if moved enough (prevents zigzag)
            if len(self.trail) == 0:
                self.trail.append((x,y))
            else:
                lx, ly = self.trail[-1]
                dist = math.sqrt((x-lx)**2 + (y-ly)**2)
                if dist >= TRAIL_MIN_DIST:
                    self.trail.append((x,y))
                    self._trail_dirty = True

            self._update_canvas((act_j1, act_j2, act_j3), (tgt_j1, tgt_j2, tgt_j3))

            err = math.sqrt((x-tx)**2 + (y-ty)**2 + (z-tz)**2) * 1000
            err_color = (C["success"] if err < 1 else C["warn"] if err < 5 else C["danger"])
            self.err_label.config(text=f"{err:8.2f} mm", fg=err_color)

            self.coord_labels["X"].config(text=f"{x*1000:+8.2f} mm")
            self.coord_labels["Y"].config(text=f"{y*1000:+8.2f} mm")
            self.coord_labels["Z"].config(text=f"{z*1000:+8.2f} mm")
            self.coord_labels["R"].config(
                text=f"{math.degrees(tool_phi((act_j1, act_j2, act_j3, act_j4))):+8.2f} deg")
            self.jval_labels["J1"].config(text=f"{math.degrees(act_j1):+7.1f} deg")
            self.jval_labels["J2"].config(text=f"{math.degrees(act_j2):+7.1f} deg")
            self.jval_labels["J3"].config(text=f"{act_j3*1000:+7.1f} mm")
            self.jval_labels["J4"].config(text=f"{math.degrees(act_j4):+7.1f} deg")
            self._update_limit_colors((act_j1, act_j2, act_j3, act_j4))
            self._update_status()
            with state.lock: tau, sat = list(state.tau), list(state.saturated)
            e = [t - a for t, a in zip((tgt_j1, tgt_j2, tgt_j3, tgt_j4), (act_j1, act_j2, act_j3, act_j4))]
            self._update_pid_live(e, tau, sat)

            disp=[math.degrees(tgt_j1),math.degrees(tgt_j2),tgt_j3*1000,math.degrees(tgt_j4)]
            focus = self.root.focus_get()
            for i,v in enumerate(disp):
                self.jvars[i].set(round(v,1))
                ent = self.jentries[i]
                txt = f"{v:.1f}"
                if ent is not focus and ent.get() != txt:   # don't fight the user's typing
                    ent.delete(0, "end"); ent.insert(0, txt)
        except Exception as e:
            # Print each distinct error once instead of every tick
            if str(e) != getattr(self, "_last_tick_err", None):
                self._last_tick_err = str(e)
                print("UI update error:", repr(e))
        self.root.after(60, self._tick)

    # Joints checked for soft limits. J3 is excluded: its retracted Home (0 mm) IS the limit.
    LIMIT_WARN_JOINTS = (0, 1, 3)

    def _update_limit_colors(self, q):
        near = []
        for i in self.LIMIT_WARN_JOINTS:
            lo, hi = JOINT_LIMITS[i]
            margin = min(q[i] - lo, hi - q[i]) / (hi - lo)
            color = C["danger"] if margin <= 0.002 else C["warn"] if margin < 0.05 else C["bright"]
            self.jval_labels[JOINT_NAMES[i]].config(fg=color)
            if color != C["bright"]: near.append(JOINT_NAMES[i])
        if tuple(near) != self._limit_state:   # notify only when the set changes
            self._limit_state = tuple(near)
            if near: self._notify(f"!! {', '.join(near)} near joint limit", "warn")

    def _update_status(self):
        running = state.running
        self.sb_sim.config(text="\u25cf Sim running" if running else "\u25cf Sim stopped",
                           fg=C["success"] if running else C["danger"])
        self.sim_dot.config(fg=C["success"] if running else C["danger"])
        if self.path_running:
            txt, fg = self.path_status, C["accent"]
        elif self._motion_busy:
            txt, fg = "Moving", C["accent"]
        elif self._stopped:
            txt, fg = "STOPPED", C["danger"]
        else:
            txt, fg = "Idle", C["text"]
        self.sb_motion.config(text=txt, fg=fg)

    def _quit(self):
        state.running = False; self.root.destroy()


if __name__ == "__main__":
    sim = threading.Thread(target=sim_thread, daemon=True)
    sim.start(); time.sleep(0.5)
    root = tk.Tk()
    app  = ControlPanel(root)
    root.mainloop()
    # Let the viewer close itself before the interpreter exits. Otherwise glfw's
    # atexit terminate() tears down EGL while the sim thread is still in
    # glfwSwapBuffers -> segfault / hang on close (seen on Wayland).
    state.running = False
    sim.join(timeout=5)
    # Last resort: if the viewer is still rendering, skip interpreter teardown
    # (glfw's atexit terminate would crash under it). Nothing is lost: waypoints
    # and settings are saved as soon as they change.
    if sim.is_alive() or any(t.is_alive() for t in viewer_threads):
        os._exit(0)
