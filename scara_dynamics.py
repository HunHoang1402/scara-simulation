"""Rigid-body dynamics of the SCARA by the Jacobian (Lagrange) method (BTL chapter 4).

    M(q) q_ddot + C(q, q_dot) q_dot + G(q) + B q_dot = tau

* Link parameters (mass, centre of mass, inertia tensor) come from the MuJoCo model,
  which computes them from the CAD meshes (density 2700 kg/m^3, aluminium). They are
  re-expressed in the DH link frames of scara_kinematics at q = 0.
* Link 3 (the d3 slide) is massless: the Z shaft and the R axis are one rigid body in
  the model, carried by frame {4} (moves with d3 and theta4).
* A payload is a point mass at the end-effector point (origin of {4}).
* armature = rotor inertia reflected through the gearbox (added to the diagonal of M);
  B = viscous friction.
* Gravity acts along -z0, so only the vertical axis (d3) has a gravity term.
"""
from dataclasses import dataclass

import numpy as np

import scara_kinematics as kin

GRAVITY = 9.81   # m/s^2

# Assumed viscous friction for the torque-controlled plant [N*m*s/rad, ..., N*s/m, ...].
# The XML uses much larger damping (50/50/100/20) to keep its built-in PD actuators calm;
# those values would hide the inertial dynamics, so the PID plant uses these instead.
B_PLANT = np.array([0.5, 0.3, 5.0, 0.02])

# MuJoCo bodies -> DH link index carrying them (link 3 is massless)
BODY_TO_LINK = {"link1": 1, "link2": 2, "link3_and_4": 4}


@dataclass
class LinkParams:
    name: str
    link: int              # DH frame index i (1, 2 or 4)
    mass: float            # kg
    com: np.ndarray        # (3,) centre of mass in frame {i}, m
    inertia: np.ndarray    # (3, 3) inertia about the COM, axes of frame {i}, kg*m^2


@dataclass
class RobotParams:
    links: list
    armature: np.ndarray   # (4,)
    damping: np.ndarray    # (4,) viscous friction B
    payload: float = 0.0   # kg, already included in `links` when > 0


def _rot_z_pi():
    return np.diag([-1.0, -1.0, 1.0])


def params_from_model(model, payload=0.0, damping=B_PLANT):
    """Read link parameters from a MuJoCo model (at q = 0) in the DH link frames."""
    import mujoco
    data = mujoco.MjData(model)
    data.qpos[:] = 0.0
    mujoco.mj_forward(model, data)
    Rz = _rot_z_pi()                                  # world axes -> {0} axes
    o0 = data.xanchor[model.joint("joint1").id]       # origin of {0} in world
    origins = [T[:3, 3] for T in kin.transforms([0.0, 0.0, 0.0, 0.0])]   # frames 1..4 at q=0
    links = []
    for body, i in BODY_TO_LINK.items():
        b = model.body(body).id
        p0 = Rz @ (data.xipos[b] - o0)                   # COM in {0}
        R = data.ximat[b].reshape(3, 3)
        I_world = R @ np.diag(model.body_inertia[b]) @ R.T
        links.append(LinkParams(body, i, float(model.body_mass[b]),
                                p0 - origins[i - 1], Rz @ I_world @ Rz.T))
    if payload > 0:
        links.append(LinkParams("payload", 4, float(payload), np.zeros(3), np.zeros((3, 3))))
    return RobotParams(links, np.array(model.dof_armature[:4], float), np.asarray(damping, float), payload)


# ─── Jacobian method ─────────────────────────────────────────────────────────
JOINT_TYPES = ("R", "R", "P", "R")


def com_jacobians(q, P):
    """For each link: (LinkParams, COM position p_c in {0}, rotation R_i, Jv (3x4), Jw (3x4)).
    Column j uses the axis z_{j-1} and origin o_{j-1} of frame {j-1}:
      revolute:  Jv_j = z_{j-1} x (p_c - o_{j-1}),  Jw_j = z_{j-1}
      prismatic: Jv_j = z_{j-1},                    Jw_j = 0
    and is zero for joints after link i."""
    Ts = [np.eye(4)] + kin.transforms(q)
    out = []
    for L in P.links:
        Ti = Ts[L.link]
        Ri = Ti[:3, :3]
        pc = Ti[:3, 3] + Ri @ L.com
        Jv, Jw = np.zeros((3, 4)), np.zeros((3, 4))
        for j in range(L.link):                       # joints 1..i
            z, o = Ts[j][:3, 2], Ts[j][:3, 3]
            if JOINT_TYPES[j] == "R":
                Jv[:, j] = np.cross(z, pc - o)
                Jw[:, j] = z
            else:
                Jv[:, j] = z
        out.append((L, pc, Ri, Jv, Jw))
    return out


def mass_matrix(q, P):
    """M(q) = sum_i ( m_i Jv_i^T Jv_i + Jw_i^T R_i I_i R_i^T Jw_i ) + diag(armature)."""
    M = np.diag(P.armature).astype(float)
    for L, _, R, Jv, Jw in com_jacobians(q, P):
        M += L.mass * Jv.T @ Jv + Jw.T @ (R @ L.inertia @ R.T) @ Jw
    return M


def _dM_dq(q, P, h=1e-6):
    q = np.asarray(q, float)
    dM = np.zeros((4, 4, 4))                          # dM[k] = dM/dq_k
    for k in range(4):
        e = np.zeros(4); e[k] = h
        dM[k] = (mass_matrix(q + e, P) - mass_matrix(q - e, P)) / (2 * h)
    return dM


def coriolis_matrix(q, qd, P):
    """C(q, q_dot) from the Christoffel symbols of M:
       c_ijk = 1/2 (dM_kj/dq_i + dM_ki/dq_j - dM_ij/dq_k),  C_kj = sum_i c_ijk q_dot_i."""
    dM = _dM_dq(q, P)
    qd = np.asarray(qd, float)
    C = np.zeros((4, 4))
    for k in range(4):
        for j in range(4):
            C[k, j] = sum(0.5 * (dM[i][k, j] + dM[j][k, i] - dM[k][i, j]) * qd[i] for i in range(4))
    return C


def potential_energy(q, P):
    """V(q) = sum_i m_i g z_ci  (z measured from the origin of {0})."""
    return sum(L.mass * GRAVITY * pc[2] for L, pc, _, _, _ in com_jacobians(q, P))


def kinetic_energy(q, qd, P):
    qd = np.asarray(qd, float)
    return 0.5 * qd @ mass_matrix(q, P) @ qd


def gravity_vector(q, P):
    """G(q) = dV/dq = sum_i Jv_i^T m_i g z0  (only the d3 component is non-zero)."""
    G = np.zeros(4)
    for L, _, _, Jv, _ in com_jacobians(q, P):
        G += Jv.T @ np.array([0.0, 0.0, L.mass * GRAVITY])
    return G


def inverse_dynamics(q, qd, qdd, P, friction=True):
    """tau = M q_ddot + C q_dot + G (+ B q_dot)."""
    qd, qdd = np.asarray(qd, float), np.asarray(qdd, float)
    tau = mass_matrix(q, P) @ qdd + coriolis_matrix(q, qd, P) @ qd + gravity_vector(q, P)
    return tau + P.damping * qd if friction else tau


def forward_dynamics(q, qd, tau, P, friction=True):
    """q_ddot = M^-1 (tau - C q_dot - G - B q_dot)."""
    qd = np.asarray(qd, float)
    rhs = np.asarray(tau, float) - coriolis_matrix(q, qd, P) @ qd - gravity_vector(q, P)
    if friction:
        rhs -= P.damping * qd
    return np.linalg.solve(mass_matrix(q, P), rhs)


def params_table(P):
    """Markdown table of the dynamic parameters (BTL 4.1)."""
    rows = ["| Link | Frame | Mass [kg] | COM in frame [mm] | Ixx, Iyy, Izz [kg m^2] |",
            "|---|---|---|---|---|"]
    for L in P.links:
        com = ", ".join(f"{v * 1000:.1f}" for v in L.com)
        I = ", ".join(f"{v:.3e}" for v in np.diag(L.inertia))
        rows.append(f"| {L.name} | {{{L.link}}} | {L.mass:.3f} | ({com}) | {I} |")
    fmt = lambda v: "[" + ", ".join(f"{x:g}" for x in v) + "]"
    rows.append(f"\nArmature (reflected rotor inertia): {fmt(P.armature)}  "
                f"Viscous friction B: {fmt(P.damping)}  g = {GRAVITY} m/s^2  payload = {P.payload:g} kg")
    return "\n".join(rows)
