"""Trajectory design (BTL chapter 3).

Every generator returns a Trajectory sampled every DT seconds with the desired
joint position, velocity and acceleration (q_d, qd_d, qdd_d) and the task pose X_d.

* Time law s(t) in [0, 1] from a motion profile (Linear / Trapezoidal / S-Curve),
  which also gives s_dot and s_ddot. S-Curve (quintic) has zero velocity AND
  acceleration at both ends (boundary conditions).
* Joint-space moves interpolate q directly.
* Task-space moves (line, circle, strokes) interpolate X = [x, y, z, phi]; joints come
  from inverse kinematics with ONE elbow branch for the whole path, and rates from
  the Jacobian:  q_dot = J^-1 X_dot,  q_ddot = J^-1 (X_ddot - J_dot q_dot).
* validate() checks reach, joint limits, singularities and velocity/acceleration limits.
"""
import math
from dataclasses import dataclass, field

import numpy as np

import scara_kinematics as kin

DT = 0.01   # s, sample period of generated trajectories

# Assumed actuator capability used to check designed trajectories
# [theta1 rad/s, theta2 rad/s, d3 m/s, theta4 rad/s] and the same per s^2.
VEL_LIMITS = np.array([6.0, 6.0, 1.0, 10.0])
ACC_LIMITS = np.array([40.0, 40.0, 10.0, 80.0])

# Speeds used to pick a duration for joint moves (as in the GUI's Go/Home)
JOINT_MOVE_SPEED = np.array([math.radians(60), math.radians(60), 0.05, math.radians(60)])


# ─── Motion profiles  s(t), s_dot(t), s_ddot(t) ──────────────────────────────
def profile_linear(t, T):
    s = t / T
    ds = 1.0 / T
    dds = 0.0
    return s, ds, dds

def profile_trapezoidal(t, T, accel_frac=0.25):
    ta = T * accel_frac
    td = T * accel_frac
    tc = T - ta - td
    if tc < 0:
        ta = td = T / 2; tc = 0
    v_max = 1.0 / (ta/2 + tc + td/2) if (ta/2+tc+td/2) > 0 else 1.0/T
    a_acc = v_max / ta if ta > 0 else 0
    if t <= ta:
        s = 0.5 * a_acc * t**2; ds = a_acc * t; dds = a_acc
    elif t <= ta + tc:
        dt = t - ta
        s = 0.5*a_acc*ta**2 + v_max*dt; ds = v_max; dds = 0.0
    else:
        dt = t - ta - tc
        s = 0.5*a_acc*ta**2 + v_max*tc + v_max*dt - 0.5*a_acc*dt**2
        ds = v_max - a_acc*dt; dds = -a_acc
    return max(0, min(1, s)), ds, dds

def profile_scurve(t, T):
    tau = t / T
    s   = 10*tau**3 - 15*tau**4 + 6*tau**5
    ds  = (30*tau**2 - 60*tau**3 + 30*tau**4) / T
    dds = (60*tau - 180*tau**2 + 120*tau**3) / (T**2)
    return s, ds, dds

PROFILES = {"Linear": profile_linear, "Trapezoidal": profile_trapezoidal,
            "S-Curve": profile_scurve}


# ─── Trajectory container ────────────────────────────────────────────────────
class TrajectoryError(ValueError):
    """A designed trajectory cannot be executed (unreachable point, limit, ...)."""


@dataclass
class Trajectory:
    t: np.ndarray            # (N,)   s
    q: np.ndarray            # (N, 4) desired joints
    qd: np.ndarray           # (N, 4) desired joint velocities
    qdd: np.ndarray          # (N, 4) desired joint accelerations
    X: np.ndarray            # (N, 4) desired task pose [x, y, z, phi]
    label: str = ""
    segments: list = field(default_factory=list)   # [(t_start, name), ...]

    @property
    def duration(self):
        return float(self.t[-1]) if len(self.t) else 0.0

    def sample(self, t):
        """(q_d, qd_d, qdd_d) at time t (linear interpolation, held at the ends)."""
        if t <= 0: k, a = 0, 0.0
        elif t >= self.duration: k, a = len(self.t) - 1, 0.0
        else:   # grid is not uniform at piece boundaries: search, don't divide by DT
            k = min(int(np.searchsorted(self.t, t, side="right")) - 1, len(self.t) - 2)
            a = (t - self.t[k]) / (self.t[k + 1] - self.t[k])
        if a == 0.0:
            return self.q[k].copy(), self.qd[k].copy(), self.qdd[k].copy()
        lerp = lambda A: A[k] + (A[k + 1] - A[k]) * a
        return lerp(self.q), lerp(self.qd), lerp(self.qdd)

    @staticmethod
    def concat(parts, label=""):
        """Join trajectories end to start (the duplicate joint sample is dropped)."""
        parts = [p for p in parts if p is not None and len(p.t)]
        t, arrays, segs, t0 = [], {k: [] for k in ("q", "qd", "qdd", "X")}, [], 0.0
        for i, p in enumerate(parts):
            sl = slice(0 if i == 0 else 1, None)
            t.append(p.t[sl] + t0)
            for k in arrays: arrays[k].append(getattr(p, k)[sl])
            segs += [(t0 + ts, name) for ts, name in (p.segments or [(0.0, p.label)])]
            t0 += p.duration
        return Trajectory(np.concatenate(t), *(np.concatenate(arrays[k]) for k in ("q", "qd", "qdd", "X")),
                          label=label, segments=segs)


def _time_grid(T):
    """0, DT, 2*DT, ... and exactly T at the end (last step may be shorter)."""
    return np.append(np.arange(0.0, T - 1e-9, DT), T)


def _profile_samples(T, profile):
    fn = PROFILES[profile]
    t = _time_grid(T)
    s = np.array([fn(min(tk, T), T) for tk in t])
    return t, s[:, 0], s[:, 1], s[:, 2]


# ─── Joint-space moves ───────────────────────────────────────────────────────
def joint_move_duration(q0, q1, speed_scale=1.0, t_min=0.5):
    d = np.abs(np.asarray(q1, float) - np.asarray(q0, float))
    return max(float(np.max(d / (JOINT_MOVE_SPEED * speed_scale))), t_min)


def joint_move(q0, q1, T=None, profile="S-Curve", label="joint move"):
    """Point-to-point move interpolated in joint space."""
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    T = joint_move_duration(q0, q1) if T is None else T
    t, s, ds, dds = _profile_samples(T, profile)
    dq = q1 - q0
    q = q0 + np.outer(s, dq)
    X = np.array([kin.fk(qk) for qk in q])
    return Trajectory(t, q, np.outer(ds, dq), np.outer(dds, dq), X, label, [(0.0, label)])


def hold(q, T, label="dwell"):
    """Stand still at q for T seconds."""
    t = _time_grid(T)
    q = np.tile(np.asarray(q, float), (len(t), 1))
    z = np.zeros_like(q)
    return Trajectory(t, q, z, z.copy(), np.array([kin.fk(qk) for qk in q]), label, [(0.0, label)])


# ─── Task-space pieces ───────────────────────────────────────────────────────
@dataclass
class TaskPiece:
    """One task-space segment: pose(s, ds, dds) -> (X, X_dot, X_ddot), duration T."""
    T: float
    pose: callable
    label: str
    profile: str = "S-Curve"


def line_piece(X0, X1, T, profile="S-Curve", label="line"):
    X0, X1 = np.asarray(X0, float), np.asarray(X1, float)
    dX = X1 - X0
    dX[3] = kin.wrap_angle(dX[3])
    return TaskPiece(T, lambda s, ds, dds: (X0 + s * dX, ds * dX, dds * dX), label, profile)


def circle_piece(center, radius, start_angle, sweep, z, phi, T, profile="S-Curve", label="circle"):
    """Arc in the XY plane at height z: angle = start_angle + sweep * s(t)."""
    cx, cy = center

    def pose(s, ds, dds):
        a = start_angle + sweep * s
        ad, add = sweep * ds, sweep * dds
        ca, sa = math.cos(a), math.sin(a)
        X = np.array([cx + radius * ca, cy + radius * sa, z, phi])
        Xd = np.array([-radius * sa * ad, radius * ca * ad, 0.0, 0.0])
        Xdd = np.array([-radius * (ca * ad * ad + sa * add), radius * (-sa * ad * ad + ca * add), 0.0, 0.0])
        return X, Xd, Xdd
    return TaskPiece(T, pose, label, profile)


def _fillet_path(points, radius):
    """Polyline -> list of primitives with every corner rounded by an arc, so the
    direction (velocity) is continuous. ("line", p0, p1) / ("arc", c, r, a0, sweep).
    The fillet radius shrinks where the neighbouring segments are short."""
    P = [np.asarray(p, float) for p in points]
    P = [p for k, p in enumerate(P) if k == 0 or np.linalg.norm(p - P[k - 1]) > 1e-12]
    if len(P) < 2:
        return []
    prims, start = [], P[0]
    for k in range(1, len(P) - 1):
        a, b, c = P[k - 1], P[k], P[k + 1]
        u, v = (b - a) / np.linalg.norm(b - a), (c - b) / np.linalg.norm(c - b)
        turn = math.atan2(u[0] * v[1] - u[1] * v[0], float(np.dot(u, v)))   # signed corner angle
        if abs(turn) < 1e-6 or radius <= 0:
            continue
        if abs(abs(turn) - math.pi) < 1e-6:     # reversal: cannot round, keep a stop-free cusp
            continue
        cut = radius * math.tan(abs(turn) / 2)
        cut_max = 0.45 * min(np.linalg.norm(b - a), np.linalg.norm(c - b))
        if cut > cut_max:
            cut = cut_max
        r = cut / math.tan(abs(turn) / 2)
        p_in, p_out = b - u * cut, b + v * cut
        n = np.array([-u[1], u[0]]) * (1 if turn > 0 else -1)              # towards the centre
        centre = p_in + n * r
        a0 = math.atan2(p_in[1] - centre[1], p_in[0] - centre[0])
        prims.append(("line", start, p_in))
        prims.append(("arc", centre, r, a0, turn))
        start = p_out
    prims.append(("line", start, P[-1]))
    return [pr for pr in prims if pr[0] == "arc" or np.linalg.norm(pr[2] - pr[1]) > 1e-12]


def _prim_length(pr):
    return np.linalg.norm(pr[2] - pr[1]) if pr[0] == "line" else abs(pr[4]) * pr[2]


def polyline_piece(points, z, phi, T, profile="S-Curve", label="stroke", blend_radius=0.004):
    """Whole polyline with one time law along its arc length. Corners are rounded with
    arcs of `blend_radius` (m) so velocity stays continuous; the centripetal term
    (tool speed)^2 / r is included in the acceleration."""
    prims = _fillet_path([p[:2] for p in points], blend_radius)
    lengths = np.array([_prim_length(pr) for pr in prims])
    cum = np.concatenate([[0.0], np.cumsum(lengths)])
    L = cum[-1]

    def pose(s, ds, dds):
        u = min(max(s, 0.0), 1.0) * L
        i = min(int(np.searchsorted(cum, u, side="right") - 1), len(prims) - 1)
        pr, w = prims[i], u - cum[i]
        v, a_t = L * ds, L * dds                      # speed and tangential acceleration
        if pr[0] == "line":
            tan = (pr[2] - pr[1]) / lengths[i]
            xy, acc = pr[1] + tan * w, tan * a_t
        else:
            _, c, r, a0, sweep = pr
            sgn = 1.0 if sweep > 0 else -1.0
            ang = a0 + sgn * w / r
            radial = np.array([math.cos(ang), math.sin(ang)])
            tan = sgn * np.array([-radial[1], radial[0]])
            xy = c + r * radial
            acc = tan * a_t - radial * (v * v / r)    # tangential + centripetal
        return (np.array([xy[0], xy[1], z, phi]),
                np.array([tan[0] * v, tan[1] * v, 0.0, 0.0]),
                np.array([acc[0], acc[1], 0.0, 0.0]))
    return TaskPiece(T, pose, label, profile)


def _elbow_of(q):
    return 1 if q[1] >= 0 else -1


def task_trajectory(pieces, q_ref, elbow="auto", label=""):
    """Sample task pieces and convert to joint space with one elbow branch.
    q_ref: current joints (chooses the branch when elbow == "auto" and theta4's turn)."""
    if not pieces:
        raise TrajectoryError("empty path")
    X0, _, _ = pieces[0].pose(0.0, 0.0, 0.0)
    q0 = kin.ik(*X0, q_prev=q_ref, elbow=elbow)
    if q0 is None:
        raise TrajectoryError(f"{pieces[0].label}: start point ({X0[0]*1000:.1f}, {X0[1]*1000:.1f}) mm "
                              f"is out of reach or outside the joint limits")
    side = "left" if _elbow_of(q0) > 0 else "right"
    parts, q_prev = [], q0
    for piece in pieces:
        t, s, ds, dds = _profile_samples(piece.T, piece.profile)
        n = len(t)
        q, qd, qdd, X = (np.zeros((n, 4)) for _ in range(4))
        for k in range(n):
            Xk, Xdk, Xddk = piece.pose(s[k], ds[k], dds[k])
            qk = kin.ik(*Xk, q_prev=q_prev, elbow=side)
            if qk is None:
                raise TrajectoryError(f"{piece.label}: point at t={t[k]:.2f} s "
                                      f"({Xk[0]*1000:.1f}, {Xk[1]*1000:.1f}, {Xk[2]*1000:.1f}) mm "
                                      f"is out of reach or outside the joint limits")
            if kin.is_singular(qk):
                raise TrajectoryError(f"{piece.label}: passes a singular configuration "
                                      f"(arm stretched/folded) at t={t[k]:.2f} s")
            q[k], X[k] = qk, Xk
            qd[k], qdd[k] = kin.joint_rates(qk, Xdk, Xddk)
            q_prev = qk
        parts.append(Trajectory(t, q, qd, qdd, X, piece.label, [(0.0, piece.label)]))
    return Trajectory.concat(parts, label)


def approach(traj, q_now, profile="S-Curve"):
    """Prepend a joint move from the current pose to the start of traj."""
    if np.allclose(q_now, traj.q[0], atol=1e-6):
        return traj
    return Trajectory.concat([joint_move(q_now, traj.q[0], profile=profile, label="approach"), traj],
                             traj.label)


# ─── Ready-made paths ────────────────────────────────────────────────────────
def line(X0, X1, T, q_ref, profile="S-Curve", elbow="auto"):
    """Straight line from X0 to X1 (x, y, z, phi)."""
    return task_trajectory([line_piece(X0, X1, T, profile)], q_ref, elbow, "line")


def circle(center, radius, z, phi, T, q_ref, start_angle=0.0, sweep=2 * math.pi,
           profile="S-Curve", elbow="auto"):
    """Circle (or arc) of given center/radius in the plane z = const."""
    if radius <= 0:
        raise TrajectoryError("radius must be positive")
    return task_trajectory([circle_piece(center, radius, start_angle, sweep, z, phi, T, profile)],
                           q_ref, elbow, "circle")


def strokes(polylines, z_up, z_down, phi, speed, q_ref, corner="stop",
            profile="S-Curve", elbow="auto", t_min=0.25):
    """Multi-stroke drawing: for each stroke travel at z_up, lower to z_down, draw,
    lift. corner="stop" stops at every vertex; "smooth" rounds every corner with a small
    arc and runs the stroke with one time law along its arc length (no stop)."""
    polylines = [list(p) for p in polylines if len(p) >= 1]
    if not polylines:
        raise TrajectoryError("nothing to draw")
    if speed <= 0:
        raise TrajectoryError("speed must be positive")
    dur = lambda d: max(d / speed, t_min)
    pieces, cur = [], None
    for n, pl in enumerate(polylines, 1):
        start = np.array([pl[0][0], pl[0][1], z_up, phi])
        if cur is not None and np.linalg.norm(start[:2] - cur[:2]) > 1e-9:
            pieces.append(line_piece(cur, start, dur(np.linalg.norm(start[:2] - cur[:2])), profile, "travel"))
        down = start.copy(); down[2] = z_down
        pieces.append(line_piece(start, down, dur(abs(z_up - z_down)), profile, f"pen down {n}"))
        if len(pl) > 1:
            if corner == "smooth":
                length = sum(_prim_length(pr) for pr in _fillet_path(pl, 0.004))
                pieces.append(polyline_piece(pl, z_down, phi, dur(length), profile, f"stroke {n}"))
            else:
                for a, b in zip(pl, pl[1:]):
                    if math.dist(a, b) > 1e-12:
                        pieces.append(line_piece([a[0], a[1], z_down, phi], [b[0], b[1], z_down, phi],
                                                 dur(math.dist(a, b)), profile, f"stroke {n}"))
        end = np.array([pl[-1][0], pl[-1][1], z_down, phi])
        up = end.copy(); up[2] = z_up
        pieces.append(line_piece(end, up, dur(abs(z_up - z_down)), profile, f"pen up {n}"))
        cur = up
    return task_trajectory(pieces, q_ref, elbow, "strokes")


# ─── Checks ──────────────────────────────────────────────────────────────────
def validate(traj, vel_limits=VEL_LIMITS, acc_limits=ACC_LIMITS):
    """Problems that would prevent tracking this trajectory (empty list = OK)."""
    issues = []
    names = ("J1", "J2", "J3", "J4")
    bad = [k for k in range(len(traj.t)) if not kin.within_limits(*traj.q[k])]
    if bad:
        issues.append(f"joint limit exceeded at t={traj.t[bad[0]]:.2f} s")
    for name, arr, lim in (("velocity", traj.qd, vel_limits), ("acceleration", traj.qdd, acc_limits)):
        ratio = np.abs(arr) / lim
        if ratio.max() > 1.0:
            k, j = np.unravel_index(np.argmax(ratio), ratio.shape)
            issues.append(f"{names[j]} {name} {abs(arr[k, j]):.2f} exceeds limit {lim[j]:.2f} "
                          f"at t={traj.t[k]:.2f} s (x{ratio.max():.1f}) - increase the time")
    if len(traj.t) > 2:
        dtk = np.diff(traj.t)
        ok = dtk > 1e-9
        fd = np.abs(np.diff(traj.qd, axis=0)[ok] / dtk[ok, None]) / acc_limits
        if fd.max() > 1.0:
            k, j = np.unravel_index(np.argmax(fd), fd.shape)
            issues.append(f"{names[j]} velocity jumps at t={traj.t[1:][ok][k]:.2f} s "
                          f"(x{fd.max():.1f} the acceleration limit) - corner not smoothed")
    jumps = np.abs(np.diff(traj.q, axis=0)).max(axis=1) if len(traj.t) > 1 else np.array([])
    if len(jumps) and jumps.max() > 0.2:
        issues.append(f"joint jump of {jumps.max():.2f} at t={traj.t[int(np.argmax(jumps))]:.2f} s")
    return issues
