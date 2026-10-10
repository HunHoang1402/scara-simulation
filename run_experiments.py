#!/usr/bin/env python3
"""Simulation results for the report (BTL chapter 7), headless.

    python3 run_experiments.py            # all cases -> results/
    python3 run_experiments.py 7.4 7.5    # only some sections

Every case runs the trajectory on the torque-driven MuJoCo plant with the 4-axis PID
(scara_control), records reference, response and torques, and saves:
    results/<section>/*.png   plots
    results/<section>/*.csv   raw data (SI units)
    results/summary.md        tables of the numbers quoted in the report
"""
import csv
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

import scara_control as ctl
import scara_dynamics as dyn
import scara_kinematics as kin
import scara_trajectory as st
import shapes

BASE = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(BASE, "mujoco_sim", "scara_real.xml")
OUT = os.path.join(BASE, "results")
NAMES, UNITS = ["J1", "J2", "J3", "J4"], ["deg", "deg", "mm", "deg"]
Q_DRAW = [0.3, 1.2, 0.0, 0.0]          # elbow-left pose near the drawing area
summary = []


# ─── helpers ─────────────────────────────────────────────────────────────────
def params():
    return dyn.params_from_model(mujoco.MjModel.from_xml_path(XML))


def run(traj, feedforward="gravity", anti_windup=True, gains=None, t_extra=0.5):
    """Simulate traj on a fresh torque plant; returns the record dict plus tool poses."""
    P = params()
    plant = ctl.make_torque_plant(mujoco.MjModel.from_xml_path(XML))
    pid = ctl.PID4(gains or ctl.tuned_gains(P), params=P, feedforward=feedforward,
                   anti_windup=anti_windup)
    r = ctl.simulate(plant, traj, pid, t_extra=t_extra)
    r["X_d"] = np.array([kin.fk(q) for q in r["q_d"]])
    r["X"] = np.array([kin.fk(q) for q in r["q"]])
    return r


def conv(i, a):
    return np.asarray(a) * 1000 if i == 2 else np.degrees(a)


def metrics(r):
    e = r["e"]
    ep = np.linalg.norm(r["X_d"][:, :3] - r["X"][:, :3], axis=1) * 1000
    ephi = np.degrees([kin.wrap_angle(a - b) for a, b in zip(r["X_d"][:, 3], r["X"][:, 3])])
    return dict(rms_joint=[conv(i, np.sqrt((e[:, i] ** 2).mean())) for i in range(4)],
                max_joint=[conv(i, np.abs(e[:, i]).max()) for i in range(4)],
                pos_max=ep.max(), pos_rms=np.sqrt((ep ** 2).mean()), phi_max=np.abs(ephi).max(),
                tau_peak=list(np.abs(r["tau"]).max(axis=0)),
                saturated=bool(np.any(np.abs(r["tau"]) >= ctl.TAU_LIMITS - 1e-9)))


def section(name):
    d = os.path.join(OUT, name.replace(" ", "_"))
    os.makedirs(d, exist_ok=True)
    return d


def save_csv(path, r):
    cols = ["t"] + [f"{k}{i+1}" for k in ("q_d", "qd_d", "qdd_d", "q", "qd", "tau", "e") for i in range(4)]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for k in range(len(r["t"])):
            w.writerow([f"{r['t'][k]:.4f}"] + [f"{r[key][k, i]:.6g}" for key in
                                              ("q_d", "qd_d", "qdd_d", "q", "qd", "tau", "e") for i in range(4)])


def plot_joints(r, path, title):
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for ax, (ref, meas, lab) in zip(axes, (("q_d", "q", "position"), ("qd_d", "qd", "velocity"),
                                           ("qdd_d", None, "acceleration (reference)"))):
        for i in range(4):
            ax.plot(r["t"], conv(i, r[ref][:, i]), "--", color=f"C{i}", label=f"{NAMES[i]} ref")
            if meas:
                ax.plot(r["t"], conv(i, r[meas][:, i]), color=f"C{i}", label=f"{NAMES[i]}")
        ax.set_ylabel(f"{lab}\n(deg, mm)/s^n"); ax.grid(alpha=0.3)
    axes[0].legend(ncol=4, fontsize=8); axes[-1].set_xlabel("time (s)")
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_tracking(r, path, title):
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for i in range(4):
        axes[0].plot(r["t"], conv(i, r["e"][:, i]), color=f"C{i}", label=f"e{NAMES[i]} [{UNITS[i]}]")
    axes[0].set_ylabel("joint error"); axes[0].legend(ncol=4, fontsize=8)
    ep = np.linalg.norm(r["X_d"][:, :3] - r["X"][:, :3], axis=1) * 1000
    axes[1].plot(r["t"], ep, color="C0"); axes[1].set_ylabel("tool position error (mm)")
    for i in range(4):
        axes[2].plot(r["t"], r["tau"][:, i], color=f"C{i}", label=f"{NAMES[i]} [{'N' if i == 2 else 'N m'}]")
    axes[2].set_ylabel("torque / force"); axes[2].legend(ncol=4, fontsize=8)
    for ax in axes: ax.grid(alpha=0.3)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_xy(r, path, title, mask=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    Xd, X = r["X_d"] * 1000, r["X"] * 1000
    if mask is not None:   # pen-down samples only (drawing)
        Xd, X = np.where(mask[:, None], Xd, np.nan), np.where(mask[:, None], X, np.nan)
    ax.plot(Xd[:, 0], Xd[:, 1], "--", color="gray", label="reference")
    ax.plot(X[:, 0], X[:, 1], color="C0", label="response")
    ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend()
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def table(rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def fmt4(v, nd=3):
    return ", ".join(f"{x:.{nd}f}" for x in v)


def report_case(name, r):
    m = metrics(r)
    return [name, fmt4(m["rms_joint"]), fmt4(m["max_joint"]), f"{m['pos_max']:.3f}",
            f"{m['pos_rms']:.3f}", f"{m['phi_max']:.3f}", fmt4(m["tau_peak"], 2),
            "yes" if m["saturated"] else "no"]


CASE_HEADER = ["case", "RMS joint err (deg,deg,mm,deg)", "max joint err", "max tool err (mm)",
               "RMS tool err (mm)", "max phi err (deg)", "peak tau (N m, N m, N, N m)", "saturated"]


# ─── 7.1 conditions ──────────────────────────────────────────────────────────
# ─── 2.1 geometry figure and workspace ───────────────────────────────────────
def _frame(ax, T, name, size=0.03, plane=(0, 1)):
    """Draw the two in-plane axes of homogeneous transform T and label the origin."""
    o = T[:3, 3][list(plane)]
    for k, c in ((plane[0], "tab:red"), (plane[1], "tab:green")):
        d = T[:3, k][list(plane)] * size
        if np.hypot(*d) > 1e-9:
            ax.annotate("", o + d, o, arrowprops=dict(arrowstyle="->", color=c, lw=1.5))
    ax.text(*(o + np.array([0.006, -0.014])), name, fontsize=9)


def s21():
    d = section("2.1")
    q = [math.radians(40), math.radians(60), -0.06, math.radians(-30)]
    T1, T2, T3, T4 = kin.transforms(q)
    p1, p4 = T1[:3, 3], T4[:3, 3]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 5.6), gridspec_kw=dict(width_ratios=[1.25, 1]))
    # top view (x0, y0)
    ax.add_patch(plt.Circle((0, 0), 0.03, color="0.85", zorder=0))
    ax.plot([0, p1[0], p4[0]], [0, p1[1], p4[1]], "-", color="0.25", lw=7, solid_capstyle="round", alpha=0.35)
    ax.plot([0, p1[0], p4[0]], [0, p1[1], p4[1]], "ko", ms=6)
    ax.plot([0, 0.09], [0, 0], "k--", lw=0.8)
    ax.plot([p1[0], p1[0] + 0.07 * math.cos(q[0])], [p1[1], p1[1] + 0.07 * math.sin(q[0])], "k--", lw=0.8)
    a = np.linspace(0, q[0], 30)
    ax.plot(0.06 * np.cos(a), 0.06 * np.sin(a), "b-"); ax.text(0.064, 0.02, r"$\theta_1$", color="b", fontsize=12)
    a = np.linspace(q[0], q[0] + q[1], 30)
    ax.plot(p1[0] + 0.045 * np.cos(a), p1[1] + 0.045 * np.sin(a), "b-")
    ax.text(p1[0] + 0.05 * math.cos(q[0] + q[1] / 2), p1[1] + 0.05 * math.sin(q[0] + q[1] / 2),
            r"$\theta_2$", color="b", fontsize=12)
    phi = q[0] + q[1] + q[3]
    ax.annotate("", p4[:2] + 0.06 * np.array([math.cos(phi), math.sin(phi)]), p4[:2],
                arrowprops=dict(arrowstyle="->", color="tab:purple", lw=2, alpha=0.6))
    ax.text(p4[0] + 0.012, p4[1] + 0.055,
            r"$\varphi=\theta_1+\theta_2+\theta_4$", color="tab:purple", fontsize=10)
    ax.text(*(0.5 * p1[:2] + np.array([-0.03, 0.012])), r"$L_1$ = 142 mm", fontsize=10)
    ax.text(*(0.5 * (p1[:2] + p4[:2]) + np.array([0.008, 0.0])), r"$L_2$ = 116 mm", fontsize=10)
    _frame(ax, np.eye(4), "{0}"); _frame(ax, T1, "{1}"); _frame(ax, T4, "{2},{3},{4}")
    ax.set(title="Top view (base frame {0})",
           xlabel="x0 [m]", ylabel="y0 [m]", aspect="equal", xlim=(-0.05, 0.22), ylim=(-0.04, 0.29))
    ax.grid(alpha=0.3)
    # side view (x0, z0), arm stretched along x0 so all links are visible
    x1, x2 = kin.L1, kin.L1 + kin.L2
    bx.add_patch(plt.Rectangle((-0.03, -0.25), 0.06, 0.25, color="0.85"))
    bx.plot([0, x1, x2], [0, 0, 0], "-", color="0.25", lw=7, alpha=0.35, solid_capstyle="round")
    bx.plot([x2, x2], [0.04, q[2]], "-", color="0.25", lw=4)
    bx.plot([0, x1, x2], [0, 0, 0], "ko", ms=6); bx.plot(x2, q[2], "ks", ms=7)
    for x, lab in ((0, r"$\theta_1$"), (x1, r"$\theta_2$"), (x2, r"$\theta_4$")):
        bx.annotate("", (x, 0.07), (x, 0.0), arrowprops=dict(arrowstyle="->", color="b"))
        bx.text(x + 0.004, 0.06, lab + " (axis z)", color="b", fontsize=9)
    bx.annotate("", (x2 + 0.02, q[2]), (x2 + 0.02, 0), arrowprops=dict(arrowstyle="<->", color="tab:orange"))
    bx.text(x2 + 0.025, q[2] / 2, r"$d_3 \in [-130, 0]$ mm", color="tab:orange", fontsize=9)
    bx.plot([x2 - 0.015, x2 + 0.015], [-0.13, -0.13], "k:", lw=1); bx.text(x2 - 0.06, -0.14, "lowest tool point", fontsize=8)
    bx.text(x2 + 0.006, q[2] - 0.016, "tool point = origin {4}", fontsize=9)
    _frame(bx, np.eye(4), "{0}", plane=(0, 2))
    bx.set(title="Side view (x0-z0), theta1 = theta2 = 0", xlabel="x0 [m]", ylabel="z0 [m]",
           aspect="equal", xlim=(-0.05, 0.38), ylim=(-0.2, 0.1))
    bx.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(d, "scara_geometry.png"), dpi=150); plt.close(fig)

    # workspace: tool points over the joint limits, both elbow branches
    t1 = np.linspace(*kin.JOINT_LIMITS[0], 241)
    t2 = np.linspace(*kin.JOINT_LIMITS[1], 241)
    A, B = np.meshgrid(t1, t2)
    X = kin.L1 * np.cos(A) + kin.L2 * np.cos(A + B)
    Y = kin.L1 * np.sin(A) + kin.L2 * np.sin(A + B)
    r_in = math.hypot(kin.L1 + kin.L2 * math.cos(kin.JOINT_LIMITS[1][1]), kin.L2 * math.sin(kin.JOINT_LIMITS[1][1]))
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot((X * 1000)[B > 0], (Y * 1000)[B > 0], ".", ms=1, color="tab:blue", alpha=0.25, label="elbow left (theta2 > 0)")
    ax.plot((X * 1000)[B < 0], (Y * 1000)[B < 0], ".", ms=1, color="tab:orange", alpha=0.25, label="elbow right (theta2 < 0)")
    a = np.linspace(0, 2 * math.pi, 200)
    for r, lab in ((kin.L1 + kin.L2, f"outer reach {1000 * (kin.L1 + kin.L2):.0f} mm"),
                   (r_in, f"inner reach {1000 * r_in:.0f} mm (J2 limit)")):
        ax.plot(r * 1000 * np.cos(a), r * 1000 * np.sin(a), "k--", lw=0.8)
        ax.text(r * 1000 * 0.72, r * 1000 * 0.72, lab, fontsize=8)
    for lim in kin.JOINT_LIMITS[0]:
        ax.plot([0, 280 * math.cos(lim)], [0, 280 * math.sin(lim)], "r:", lw=1)
    ax.plot(0, 0, "k+", ms=12)
    ax.set(title="Workspace in the XY plane (z from 0 to -130 mm)", xlabel="x0 [mm]", ylabel="y0 [mm]", aspect="equal")
    ax.legend(loc="lower left", markerscale=8, fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(d, "workspace.png"), dpi=150); plt.close(fig)

    rows = [[str(i + 1), f"{a_ * 1000:.0f}", f"{al:.0f}", dd, th]
            for i, ((a_, al, _, _), dd, th) in enumerate(zip(kin.dh_params([0, 0, 0, 0]),
                                                             ["0", "0", "d3", "0"],
                                                             ["theta1", "theta2", "0", "theta4"]))]
    summary.append("## 2.1 Geometry and DH table\n")
    summary.append("Figures: `results/2.1/scara_geometry.png` (joint variables, frames, tool point), "
                   "`results/2.1/workspace.png` (reachable XY region).\n")
    summary.append(table(rows, ["i", "a_i (mm)", "alpha_i (rad)", "d_i", "theta_i"]) + "\n")
    summary.append(f"Tool point = origin of {{4}}; phi = theta1 + theta2 + theta4. Workspace: annulus "
                   f"{1000 * r_in:.0f}-{1000 * (kin.L1 + kin.L2):.0f} mm, J1 within "
                   f"+-{math.degrees(kin.JOINT_LIMITS[0][1]):.1f} deg, z from 0 to -130 mm.\n")


def s71():
    P = params(); g = ctl.tuned_gains(P)
    m = mujoco.MjModel.from_xml_path(XML)
    summary.append("## 7.1 Simulation conditions\n")
    summary.append(f"- Physics: MuJoCo {mujoco.__version__}, time step {m.opt.timestep*1000:.1f} ms "
                   f"(controller rate {1/m.opt.timestep:.0f} Hz), gravity {dyn.GRAVITY} m/s^2.")
    summary.append("- Robot: Yamaha YK-XG geometry, L1 = 142 mm, L2 = 116 mm, Z stroke 130 mm; "
                   "J1 +-139.8 deg, J2 +-143.8 deg, J4 +-360 deg.")
    summary.append(f"- Actuator limits (assumed): [{fmt4(ctl.TAU_LIMITS, 0)}] (N m, N m, N, N m); "
                   f"viscous friction B = [{fmt4(dyn.B_PLANT, 2)}] (N m s/rad, N m s/rad, N s/m, N m s/rad).")
    summary.append(f"- Trajectory sample period {st.DT*1000:.0f} ms; S-curve (quintic) time law unless stated.\n")
    summary.append("Dynamic parameters (from the CAD model, in the DH link frames):\n")
    summary.append(dyn.params_table(P) + "\n")
    summary.append("PID gains (triple pole at -wn):\n")
    summary.append(table([[NAMES[i], f"{ctl.DEFAULT_WN[i]:g}", f"{g.kp[i]:.1f}", f"{g.ki[i]:.1f}",
                           f"{g.kd[i]:.2f}"] for i in range(4)], ["axis", "wn (rad/s)", "Kp", "Ki", "Kd"]) + "\n")


# ─── 7.2 per-joint motion and kinematic verification ─────────────────────────
def s72():
    d = section("7.2")
    rng = np.random.default_rng(0)
    pos_err = phi_err = q_err = jac_err = 0.0
    for _ in range(2000):
        sign = rng.choice([1, -1])
        q = np.array([rng.uniform(-2.4, 2.4), sign * rng.uniform(0.05, 2.5), rng.uniform(-0.13, 0),
                      rng.uniform(-6, 6)])
        X = kin.fk(q)
        q2 = kin.ik(*X, q_prev=q, elbow="left" if sign > 0 else "right")
        X2 = kin.fk(q2)
        pos_err = max(pos_err, np.linalg.norm(np.subtract(X2[:3], X[:3])))
        phi_err = max(phi_err, abs(kin.wrap_angle(X2[3] - X[3])))
        q_err = max(q_err, np.abs(np.subtract(q2, q)).max())
        jac_err = max(jac_err, np.abs(kin.jacobian(q) - kin.numeric_jacobian(q)).max())
    unreachable = {"behind base (-200, 0)": kin.ik(-0.2, 0, 0, 0) is None,
                   "beyond reach (300, 0)": kin.ik(0.3, 0, 0, 0) is None,
                   "inside inner reach (50, 0)": kin.ik(0.05, 0, 0, 0) is None}
    summary.append("## 7.2 Joint motion and kinematic verification\n")
    summary.append(table([["FK -> IK -> FK position error (2000 random poses)", f"{pos_err:.2e} m"],
                          ["FK -> IK -> FK orientation error", f"{phi_err:.2e} rad"],
                          ["IK joint error", f"{q_err:.2e}"],
                          ["analytic vs numeric Jacobian (max entry)", f"{jac_err:.2e}"]]
                         + [[f"unreachable point {k} rejected", "yes" if v else "NO"] for k, v in unreachable.items()],
                         ["check", "result"]) + "\n")
    t2 = np.linspace(-math.pi, math.pi, 721)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(np.degrees(t2), [kin.det_jacobian([0, a, 0, 0]) * 1e6 for a in t2])
    for a in (-180, 0, 180): ax.axvline(a, color="red", ls="--", lw=0.8)
    ax.set_xlabel("theta2 (deg)"); ax.set_ylabel("det J (mm^2)"); ax.grid(alpha=0.3)
    ax.set_title("det J = L1 L2 sin(theta2): singular when stretched (0) or folded (+-180)")
    fig.tight_layout(); fig.savefig(os.path.join(d, "det_jacobian.png"), dpi=120); plt.close(fig)
    rows = []
    for i, amp in enumerate([1.0, 1.0, -0.08, 2.0]):
        q1 = np.zeros(4); q1[i] = amp
        tr = st.joint_move(np.zeros(4), q1, T=1.5)
        r = run(tr)
        plot_joints(r, os.path.join(d, f"joint{i+1}_motion.png"), f"7.2 {NAMES[i]} alone: 0 -> {conv(i, amp):.0f} {UNITS[i]}")
        save_csv(os.path.join(d, f"joint{i+1}.csv"), r)
        rows.append(report_case(f"{NAMES[i]} alone", r))
    summary.append("Single-joint moves (1.5 s, S-curve, PID + gravity feed-forward):\n")
    summary.append(table(rows, CASE_HEADER) + "\n")


# ─── 7.3 moves between given poses ───────────────────────────────────────────
def s73():
    d = section("7.3")
    poses = [[0, 0, 0, 0], [0.8, 1.2, -0.05, 0.5], [-0.6, -1.4, -0.1, -1.0], [0.3, 0.9, -0.02, 2.0], [0, 0, 0, 0]]
    tr = st.Trajectory.concat([st.joint_move(a, b, profile="S-Curve") for a, b in zip(poses, poses[1:])])
    r = run(tr)
    plot_joints(r, os.path.join(d, "poses_joints.png"), "7.3 Moves between given poses (joint space)")
    plot_tracking(r, os.path.join(d, "poses_tracking.png"), "7.3 Tracking error and torques")
    save_csv(os.path.join(d, "poses.csv"), r)
    summary.append("## 7.3 Moves between given poses\n")
    summary.append("Poses (theta1, theta2, d3, theta4): " + " -> ".join(str(p) for p in poses) + "\n")
    summary.append(table([report_case("pose sequence", r)], CASE_HEADER) + "\n")


# ─── 7.4 line, 7.5 circle, 7.6 drawing ───────────────────────────────────────
def cartesian_case(sec, name, traj, mask_fn=None):
    d = section(sec)
    full = st.approach(traj, np.zeros(4))
    issues = st.validate(full)
    r = run(full)
    mask = mask_fn(r) if mask_fn else None
    plot_xy(r, os.path.join(d, f"{name}_xy.png"), f"{sec} {name}: XY path", mask)
    plot_tracking(r, os.path.join(d, f"{name}_tracking.png"), f"{sec} {name}: tracking error and torques")
    plot_joints(r, os.path.join(d, f"{name}_joints.png"), f"{sec} {name}: joint trajectories")
    save_csv(os.path.join(d, f"{name}.csv"), r)
    return r, issues


def s74():
    X0, X1 = [0.20, -0.08, -0.02, 0.0], [0.20, 0.08, -0.05, math.radians(45)]
    r, issues = cartesian_case("7.4", "line", st.line(X0, X1, 2.0, Q_DRAW))
    on = r["t"] >= (r["t"][-1] - 0.5 - 2.0)          # the line itself (after the approach)
    dev = np.abs(r["X"][on, 0] - 0.20).max() * 1000
    summary.append("## 7.4 Straight line\n")
    summary.append(f"Line (200, -80, -20) mm, phi 0 -> (200, 80, -50) mm, phi 45 deg in 2 s. "
                   f"Max deviation of the response from the line x = 200 mm: {dev:.3f} mm. "
                   f"Trajectory check: {issues or 'OK'}\n")
    summary.append(table([report_case("line (incl. approach)", r)], CASE_HEADER) + "\n")


def s75():
    rows = []
    for T in (4.0, 2.0):
        r, issues = cartesian_case("7.5", f"circle_{T:.0f}s",
                                   st.circle((0.17, 0.0), 0.05, -0.03, 0.0, T, Q_DRAW))
        on = np.abs(np.hypot(r["X_d"][:, 0] - 0.17, r["X_d"][:, 1]) - 0.05) < 1e-4
        rad = np.abs(np.hypot(r["X"][on, 0] - 0.17, r["X"][on, 1]) - 0.05).max() * 1000
        row = report_case(f"circle r=50 mm in {T:.0f} s", r) + [f"{rad:.3f}"]
        rows.append(row)
    summary.append("## 7.5 Circle\n")
    summary.append("Circle centre (170, 0) mm, radius 50 mm, z = -30 mm, phi = 0.\n")
    summary.append(table(rows, CASE_HEADER + ["max radius error (mm)"]) + "\n")


def s76():
    z_up, z_down = -0.02, -0.04
    lines = shapes.centered_text("HUST", 0.17, 0.0, 0.03)
    tr = st.strokes(lines, z_up, z_down, 0.0, 0.05, Q_DRAW, corner="stop")
    r, issues = cartesian_case("7.6", "text_HUST", tr,
                               mask_fn=lambda r: np.abs(r["X_d"][:, 2] - z_down) < 1e-6)
    tr2 = st.strokes(shapes.star(0.17, 0.0, 0.05), z_up, z_down, 0.0, 0.05, Q_DRAW, corner="smooth")
    r2, _ = cartesian_case("7.6", "star_smooth", tr2,
                           mask_fn=lambda r: np.abs(r["X_d"][:, 2] - z_down) < 1e-6)
    down = np.abs(r["X_d"][:, 2] - z_down) < 1e-6
    ep = np.linalg.norm(r["X_d"][down, :3] - r["X"][down, :3], axis=1) * 1000
    summary.append("## 7.6 Drawing a shape / text\n")
    summary.append(f"Text 'HUST', height 30 mm, centred at (170, 0) mm, pen up {z_up*1000:.0f} mm / "
                   f"down {z_down*1000:.0f} mm, 50 mm/s, stop at corners: {len(lines)} strokes, "
                   f"{tr.duration:.1f} s. While drawing (pen down): max error {ep.max():.3f} mm, "
                   f"RMS {np.sqrt((ep**2).mean()):.3f} mm.\n")
    summary.append(table([report_case("text HUST (stop)", r), report_case("star (smooth corners)", r2)],
                         CASE_HEADER) + "\n")


# ─── 7.7 PID evaluation ──────────────────────────────────────────────────────
def s77():
    d = section("7.7")
    circle = st.approach(st.circle((0.17, 0.0), 0.05, -0.03, 0.0, 2.0, Q_DRAW), np.zeros(4))
    rows, runs = [], {}
    for ff in ctl.FEEDFORWARD_MODES:
        runs[ff] = run(circle, feedforward=ff)
        rows.append(report_case(f"circle 2 s, feed-forward = {ff}", runs[ff]))
    fig, ax = plt.subplots(figsize=(9, 4))
    for ff, r in runs.items():
        ax.plot(r["t"], np.linalg.norm(r["X_d"][:, :3] - r["X"][:, :3], axis=1) * 1000, label=f"ff = {ff}")
    ax.set_yscale("log"); ax.set_xlabel("time (s)"); ax.set_ylabel("tool position error (mm)")
    ax.grid(alpha=0.3, which="both"); ax.legend(); ax.set_title("7.7 Effect of feed-forward on tracking")
    fig.tight_layout(); fig.savefig(os.path.join(d, "feedforward_comparison.png"), dpi=120); plt.close(fig)
    # saturation: large step with and without anti-windup
    q1 = np.array([1.5, 1.5, -0.1, 4.0])
    step = st.Trajectory.concat([st.hold(np.zeros(4), 0.02), st.hold(q1, 3.0)])
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    step_rows = []
    for aw, ls in ((True, "-"), (False, "--")):
        r = run(step, anti_windup=aw, t_extra=0.0)
        over = [conv(i, max(0.0, (r["q"][:, i] - q1[i]).max() if q1[i] > 0 else (q1[i] - r["q"][:, i]).max()))
                for i in range(4)]
        settle = []
        for i in range(4):
            bad = np.nonzero(np.abs(r["e"][:, i]) > 0.02 * abs(q1[i]))[0]
            settle.append(r["t"][bad[-1]] if len(bad) else 0.0)
        step_rows.append([f"anti-windup {'on' if aw else 'off'}", fmt4(over, 2), fmt4(settle, 2),
                          fmt4([conv(i, abs(r["e"][-1, i])) for i in range(4)], 4)])
        for i in (0, 1, 3):
            axes[0].plot(r["t"], np.degrees(r["q"][:, i]), ls, color=f"C{i}",
                         label=f"{NAMES[i]} AW {'on' if aw else 'off'}")
            axes[1].plot(r["t"], r["tau"][:, i], ls, color=f"C{i}")
    for i in (0, 1, 3): axes[0].axhline(math.degrees(q1[i]), color=f"C{i}", lw=0.6, alpha=0.5)
    axes[0].set_ylabel("joint (deg)"); axes[0].legend(ncol=3, fontsize=7)
    axes[1].set_ylabel("torque (N m)"); axes[1].set_xlabel("time (s)")
    for ax in axes: ax.grid(alpha=0.3)
    fig.suptitle("7.7 Large step: saturation with / without anti-windup")
    fig.tight_layout(); fig.savefig(os.path.join(d, "anti_windup.png"), dpi=120); plt.close(fig)
    summary.append("## 7.7 PID evaluation\n")
    summary.append("Circle (r = 50 mm, 2 s) with each feed-forward mode (gravity only affects J3, "
                   "the vertical axis):\n")
    summary.append(table(rows, CASE_HEADER) + "\n")
    summary.append(f"Large step 0 -> ({fmt4(q1, 2)}) (rad, rad, m, rad); torques saturate:\n")
    summary.append(table(step_rows, ["case", "overshoot (deg,deg,mm,deg)", "2% settling time (s)",
                                     "final error at 3 s"]) + "\n")
    summary.append("- Joint and tool errors, velocities, accelerations and torques for every case "
                   "are in the CSV files and tracking plots of 7.2-7.6.\n")


SECTIONS = {"2.1": s21, "7.1": s71, "7.2": s72, "7.3": s73, "7.4": s74, "7.5": s75, "7.6": s76, "7.7": s77}


def main(argv):
    wanted = argv or list(SECTIONS)
    os.makedirs(OUT, exist_ok=True)
    summary.append("# Simulation results (generated by run_experiments.py)\n")
    for key in wanted:
        print(f"running {key} ...", flush=True)
        SECTIONS[key]()
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write("\n".join(summary) + "\n")
    print(f"done: {os.path.join(OUT, 'summary.md')}")


if __name__ == "__main__":
    main(sys.argv[1:])
