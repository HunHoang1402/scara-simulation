"""SCARA kinematics: DH model, forward/inverse kinematics, Jacobian, singularities.

Robot: Yamaha YK-XG geometry as modelled in mujoco_sim/scara_real.xml.
Joint vector  q = [theta1, theta2, d3, theta4]  (rad, rad, m, rad)
Task vector   X = [x, y, z, phi]                (m, m, m, rad)

Frames (standard Denavit-Hartenberg)
  {0}  base frame on the joint-1 axis; z0 up; x0 along the stretched arm at Home
       (q = 0). The MuJoCo world frame is {0} rotated by pi about z (and lifted),
       i.e.  x_world = -x0,  y_world = -y0.
  DH table (i: a_i, alpha_i, d_i, theta_i):
       1:  L1, 0, 0,  theta1
       2:  L2, 0, 0,  theta2
       3:  0,  0, d3, 0          (prismatic, d3 in [-0.13, 0]; negative = down)
       4:  0,  0, 0,  theta4
  End-effector point = origin of {4} on the tool axis;  phi = theta1 + theta2 + theta4.
"""
import math

import numpy as np

L1 = 0.142   # m, joint 1 -> joint 2
L2 = 0.116   # m, joint 2 -> tool axis

# Joint limits [theta1 rad, theta2 rad, d3 m, theta4 rad]. Must match the actuator
# ctrlrange / joint range in mujoco_sim/scara_real.xml (checked by tests).
JOINT_LIMITS = [(-2.44, 2.44), (-2.51, 2.51), (-0.13, 0.0), (-6.28, 6.28)]
DEAD_ZONE = abs(L1 - L2) + 0.008   # keep away from the inner workspace boundary
SINGULAR_TOL = 0.02   # |sin(theta2)| below this counts as (near) singular


def within_limits(*q):
    return all(lo - 1e-9 <= v <= hi + 1e-9 for v, (lo, hi) in zip(q, JOINT_LIMITS))


def wrap_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


# ─── Forward kinematics ──────────────────────────────────────────────────────
def dh(a, alpha, d, theta):
    """Standard DH homogeneous transform  {i-1} -> {i}."""
    ct, st, ca, sa = math.cos(theta), math.sin(theta), math.cos(alpha), math.sin(alpha)
    return np.array([[ct, -st * ca,  st * sa, a * ct],
                     [st,  ct * ca, -ct * sa, a * st],
                     [0.0,      sa,       ca,      d],
                     [0.0,     0.0,      0.0,    1.0]])


def dh_params(q):
    """Numeric DH rows (a, alpha, d, theta) for joint vector q."""
    t1, t2, d3, t4 = q
    return [(L1, 0.0, 0.0, t1), (L2, 0.0, 0.0, t2), (0.0, 0.0, d3, 0.0), (0.0, 0.0, 0.0, t4)]


def transforms(q):
    """[0T1, 0T2, 0T3, 0T4]: each frame expressed in the base frame {0}."""
    T = np.eye(4)
    out = []
    for row in dh_params(q):
        T = T @ dh(*row)
        out.append(T)
    return out


def fk(q):
    """Task pose X = (x, y, z, phi) of the end-effector, from 0T4."""
    T = transforms(q)[-1]
    return T[0, 3], T[1, 3], T[2, 3], math.atan2(T[1, 0], T[0, 0])


def fk_position(t1, t2, d3):
    """Closed-form position (x, y, z); same result as fk() without the matrices."""
    return (L1 * math.cos(t1) + L2 * math.cos(t1 + t2),
            L1 * math.sin(t1) + L2 * math.sin(t1 + t2),
            d3)


# ─── Inverse kinematics ──────────────────────────────────────────────────────
def _elbow_solutions(x, y):
    """Both (theta1, theta2) solutions for a planar target, or [] if unreachable.
    Existence: |L1 - L2| <= d <= L1 + L2 (law of cosines)."""
    d = math.hypot(x, y)
    if d > L1 + L2 - 1e-5 or d < abs(L1 - L2) + 1e-5:
        return []
    c2 = max(-1.0, min(1.0, (d * d - L1 * L1 - L2 * L2) / (2 * L1 * L2)))
    sols = []
    for t2 in (math.acos(c2), -math.acos(c2)):        # elbow left (+) / right (-)
        t1 = wrap_angle(math.atan2(y, x) - math.atan2(L2 * math.sin(t2), L1 + L2 * math.cos(t2)))
        sols.append((t1, t2))
    return sols


def _clamp_z(z):
    lo, hi = JOINT_LIMITS[2]
    return max(lo, min(hi, z))


def ik_nearest(x, y, z, cur_t1, cur_t2):
    """(theta1, theta2, d3) within joint limits closest to the current pose, else None."""
    d3 = _clamp_z(z)
    best, best_err = None, float("inf")
    for t1, t2 in _elbow_solutions(x, y):
        if not within_limits(t1, t2, d3):
            continue
        err = (t1 - cur_t1) ** 2 + (t2 - cur_t2) ** 2
        if err < best_err:
            best, best_err = (t1, t2, d3), err
    return best


def ik_locked(x, y, z, sign):
    """(theta1, theta2, d3) with a fixed elbow sign (+1 left, -1 right), else None.
    Used for jogging / straight lines so the arm never flips configuration."""
    d3 = _clamp_z(z)
    for t1, t2 in _elbow_solutions(x, y):
        if (t2 >= 0) == (sign > 0) and within_limits(t1, t2, d3):
            return t1, t2, d3
    return None


def theta4_for_phi(phi, t1, t2, t4_prev=0.0):
    """theta4 giving tool angle phi; among 2*pi-equivalent values pick the one nearest
    t4_prev that is within the J4 limits (None if none is)."""
    lo, hi = JOINT_LIMITS[3]
    base = phi - t1 - t2
    k0 = round((t4_prev - base) / (2 * math.pi))
    cands = [base + 2 * math.pi * k for k in (k0 - 1, k0, k0 + 1)]
    cands = [c for c in cands if lo - 1e-9 <= c <= hi + 1e-9]
    return min(cands, key=lambda c: abs(c - t4_prev)) if cands else None


def ik(x, y, z, phi, q_prev=(0.0, 0.0, 0.0, 0.0), elbow="auto"):
    """Full IK: task pose (x, y, z, phi) -> q = (theta1, theta2, d3, theta4), or None.
    elbow: "auto" (in-limit solution nearest q_prev), "left" (+), "right" (-)."""
    if elbow == "auto":
        r = ik_nearest(x, y, z, q_prev[0], q_prev[1])
    else:
        r = ik_locked(x, y, z, 1 if elbow == "left" else -1)
    if r is None:
        return None
    t4 = theta4_for_phi(phi, r[0], r[1], q_prev[3])
    return None if t4 is None else (r[0], r[1], r[2], t4)


# ─── Jacobian and singularities ──────────────────────────────────────────────
def jacobian(q):
    """Analytic Jacobian J_X(q):  X_dot = J_X(q) q_dot,  X = [x, y, z, phi]."""
    t1, t2 = q[0], q[1]
    s1, c1 = math.sin(t1), math.cos(t1)
    s12, c12 = math.sin(t1 + t2), math.cos(t1 + t2)
    return np.array([[-L1 * s1 - L2 * s12, -L2 * s12, 0.0, 0.0],
                     [ L1 * c1 + L2 * c12,  L2 * c12, 0.0, 0.0],
                     [0.0,                  0.0,      1.0, 0.0],
                     [1.0,                  1.0,      0.0, 1.0]])


def jacobian_dot(q, qd):
    """Time derivative of J_X along the motion (needed for joint accelerations)."""
    t1, t2 = q[0], q[1]
    w1, w12 = qd[0], qd[0] + qd[1]
    s1, c1 = math.sin(t1), math.cos(t1)
    s12, c12 = math.sin(t1 + t2), math.cos(t1 + t2)
    Jd = np.zeros((4, 4))
    Jd[0, 0] = -L1 * c1 * w1 - L2 * c12 * w12
    Jd[0, 1] = -L2 * c12 * w12
    Jd[1, 0] = -L1 * s1 * w1 - L2 * s12 * w12
    Jd[1, 1] = -L2 * s12 * w12
    return Jd


def det_jacobian(q):
    """det J_X = L1 * L2 * sin(theta2): zero when the arm is stretched or folded."""
    return L1 * L2 * math.sin(q[1])


def is_singular(q, tol=SINGULAR_TOL):
    """True near theta2 = 0 (stretched, outer boundary) or +-pi (folded, inner boundary)."""
    return abs(math.sin(q[1])) < tol


def numeric_jacobian(q, h=1e-6):
    """Central-difference Jacobian of fk(), used to verify jacobian()."""
    J = np.zeros((4, 4))
    for j in range(4):
        qp, qm = list(q), list(q)
        qp[j] += h; qm[j] -= h
        Xp, Xm = np.array(fk(qp)), np.array(fk(qm))
        dX = Xp - Xm
        dX[3] = wrap_angle(dX[3])
        J[:, j] = dX / (2 * h)
    return J


def joint_rates(q, Xd, Xdd=None):
    """Joint velocity (and acceleration) for a task-space velocity (acceleration):
       q_dot = J^-1 X_dot,   q_ddot = J^-1 (X_ddot - J_dot q_dot)."""
    J = jacobian(q)
    qd = np.linalg.solve(J, np.asarray(Xd, float))
    if Xdd is None:
        return qd, None
    qdd = np.linalg.solve(J, np.asarray(Xdd, float) - jacobian_dot(q, qd) @ qd)
    return qd, qdd
