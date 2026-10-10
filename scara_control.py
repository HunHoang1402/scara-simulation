"""Four-axis joint PID controller and torque-driven plant (BTL chapter 5).

Control loop (per joint, runs at the physics rate 500 Hz):

    q_d, qd_d, qdd_d  ──►( e = q_d - q )──► PID ──►(+ feed-forward)──► saturate ──► tau ──► robot
                ▲                                                                      │
                └─────────────────────────── q, q_dot (measured) ◄─────────────────────┘

    u   = Kp e + Ki ∫e dt + Kd (qd_d - q_dot) + tau_ff
    tau = clip(u, -tau_max, +tau_max)
    anti-windup (clamping): the integral is frozen while the output is saturated
    and the error would push it further into saturation.

Inputs: desired joint trajectory (q_d, qd_d, qdd_d) and measured (q, q_dot).
Output: joint torques / Z force tau [N*m, N*m, N, N*m].

Tuning (pole placement): per axis the plant is J s^2 + B s with J the effective inertia
(diagonal of M at Home, where it is largest). With PID the closed loop is
J s^3 + (B + Kd) s^2 + Kp s + Ki; placing a triple pole at -wn gives
    Kd = 3 J wn - B,   Kp = 3 J wn^2,   Ki = J wn^3.
"""
from dataclasses import dataclass

import numpy as np

import scara_dynamics as dyn

# Assumed actuator limits [N*m, N*m, N, N*m] (motor torque x gear ratio)
TAU_LIMITS = np.array([40.0, 20.0, 50.0, 2.0])
# Closed-loop bandwidth per axis [rad/s] used by tuned_gains()
DEFAULT_WN = np.array([25.0, 25.0, 30.0, 25.0])
FEEDFORWARD_MODES = ("none", "gravity", "full")


@dataclass
class Gains:
    kp: np.ndarray
    ki: np.ndarray
    kd: np.ndarray

    def as_rows(self):
        return [(float(self.kp[i]), float(self.ki[i]), float(self.kd[i])) for i in range(4)]


def tuned_gains(P, wn=DEFAULT_WN):
    """Triple-pole placement on the decoupled axis model J s^2 + B s."""
    wn = np.asarray(wn, float)
    J = np.diag(dyn.mass_matrix([0.0, 0.0, 0.0, 0.0], P))
    kd = np.maximum(3 * J * wn - P.damping, 0.0)
    return Gains(kp=3 * J * wn ** 2, ki=J * wn ** 3, kd=kd)


class PID4:
    def __init__(self, gains, tau_limits=TAU_LIMITS, dt=0.002, feedforward="gravity",
                 params=None, anti_windup=True):
        self.gains, self.dt = gains, dt
        self.tau_limits = np.asarray(tau_limits, float)
        self.feedforward, self.params, self.anti_windup = feedforward, params, anti_windup
        if feedforward != "none" and params is None:
            raise ValueError("feed-forward needs the dynamic parameters")
        self.reset()

    def reset(self):
        self.integral = np.zeros(4)
        self.e = np.zeros(4)
        self.u = np.zeros(4)          # before saturation
        self.tau = np.zeros(4)        # after saturation
        self.ff = np.zeros(4)
        self.saturated = np.zeros(4, bool)

    def feedforward_torque(self, q, q_d, qd_d, qdd_d):
        if self.feedforward == "gravity":
            return dyn.gravity_vector(q, self.params)
        if self.feedforward == "full":      # computed-torque feed-forward on the reference
            return dyn.inverse_dynamics(q_d, qd_d, qdd_d, self.params)
        return np.zeros(4)

    def update(self, q_d, qd_d, qdd_d, q, qd, ff=None):
        """One control step; returns the saturated torque. `ff` overrides the feed-forward
        term (lets a caller compute it less often than the control rate)."""
        g = self.gains
        q_d, qd_d, q, qd = (np.asarray(v, float) for v in (q_d, qd_d, q, qd))
        e = q_d - q
        self.ff = self.feedforward_torque(q, q_d, qd_d, qdd_d) if ff is None else np.asarray(ff, float)
        u = g.kp * e + g.ki * self.integral + g.kd * (qd_d - qd) + self.ff
        tau = np.clip(u, -self.tau_limits, self.tau_limits)
        sat = np.abs(u) > self.tau_limits
        # clamping anti-windup: don't integrate in the direction that deepens saturation
        integrate = ~(sat & (np.sign(e) == np.sign(u))) if self.anti_windup else np.ones(4, bool)
        self.integral[integrate] += e[integrate] * self.dt
        self.e, self.u, self.tau, self.saturated = e, u, tau, sat
        return tau


# ─── Torque-driven plant ─────────────────────────────────────────────────────
def make_torque_plant(model, tau_limits=TAU_LIMITS, damping=dyn.B_PLANT):
    """Turn the XML's position (PD) actuators into plain motors: actuator force = ctrl,
    limited to +-tau_limits. Uses realistic viscous friction and no gravity
    compensation, so the PID controller has to do all the work. Modifies the model in
    place (the XML file is not changed; the ROS 2 bridge keeps the PD actuators)."""
    import mujoco
    for i in range(4):
        model.actuator_gaintype[i] = mujoco.mjtGain.mjGAIN_FIXED
        model.actuator_gainprm[i, :] = 0.0
        model.actuator_gainprm[i, 0] = 1.0
        model.actuator_biastype[i] = mujoco.mjtBias.mjBIAS_NONE
        model.actuator_biasprm[i, :] = 0.0
        model.actuator_ctrllimited[i] = 1
        model.actuator_ctrlrange[i] = (-tau_limits[i], tau_limits[i])
    model.dof_damping[:4] = damping
    model.body_gravcomp[:] = 0.0
    return model


def simulate(model, traj, controller, t_extra=0.5, data=None):
    """Run a trajectory on the torque plant headless. Returns a dict of arrays sampled
    every control step: t, q_d, qd_d, qdd_d, q, qd, tau, e."""
    import mujoco
    if data is None:
        data = mujoco.MjData(model)
        data.qpos[:4] = traj.q[0]
        mujoco.mj_forward(model, data)
    dt = model.opt.timestep
    controller.dt = dt
    controller.reset()
    n = int(round((traj.duration + t_extra) / dt)) + 1
    rec = {k: np.zeros((n, 4)) for k in ("q_d", "qd_d", "qdd_d", "q", "qd", "tau", "e")}
    rec["t"] = np.arange(n) * dt
    for k in range(n):
        q_d, qd_d, qdd_d = traj.sample(k * dt)
        q, qd = data.qpos[:4].copy(), data.qvel[:4].copy()
        tau = controller.update(q_d, qd_d, qdd_d, q, qd)
        data.ctrl[:4] = tau
        for key, val in (("q_d", q_d), ("qd_d", qd_d), ("qdd_d", qdd_d), ("q", q), ("qd", qd),
                         ("tau", tau), ("e", controller.e)):
            rec[key][k] = val
        mujoco.mj_step(model, data)
    return rec
