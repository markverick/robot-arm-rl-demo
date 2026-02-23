import numpy as np


def skew(u: np.ndarray) -> np.ndarray:
    ux, uy, uz = u
    return np.array([[0.0, -uz,  uy],
                     [uz,  0.0, -ux],
                     [-uy, ux,  0.0]], dtype=np.float64)


def rot_axis_angle(u: np.ndarray, theta: float) -> np.ndarray:
    u = np.asarray(u, dtype=np.float64)
    u = u / (np.linalg.norm(u) + 1e-12)
    K = skew(u)
    I = np.eye(3)
    return I + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


class Star5Reference:
    """
    Generates:
      p_d(t), R_d(t), v_d(t), w_d(t)
    where:
      - p_d traces a smooth 5-point star loop using a polar radius modulation
      - phi_dot(t) has sinusoidal speed modulation (velocity profile)
      - orientation oscillates while traversing the loop
    """

    def __init__(
        self,
        a: float,
        b: float,
        pc_world: np.ndarray,
        R_plane_world: np.ndarray,
        f_hz: float,
        alpha: float,
        R0_world: np.ndarray,
        u_world: np.ndarray,
        theta_max_rad: float,
        seed: int = 0,
    ):
        self.a = float(a)
        self.b = float(b)
        self.pc = np.asarray(pc_world, dtype=np.float64).reshape(3)
        self.Rp = np.asarray(R_plane_world, dtype=np.float64).reshape(3, 3)
        self.f = float(f_hz)
        self.alpha = float(alpha)
        self.R0 = np.asarray(R0_world, dtype=np.float64).reshape(3, 3)
        self.u = np.asarray(u_world, dtype=np.float64).reshape(3)
        self.u = self.u / (np.linalg.norm(self.u) + 1e-12)
        self.theta_max = float(theta_max_rad)

        self.rng = np.random.default_rng(seed)
        self.t = 0.0
        self.phi = 0.0

    def reset(self, t0: float = 0.0, random_phase: bool = True):
        self.t = float(t0)
        self.phi = float(self.rng.uniform(0.0, 2.0 * np.pi)) if random_phase else 0.0

    def _phi_dot(self, t: float) -> float:
        return 2.0 * np.pi * self.f * (1.0 + self.alpha * np.sin(2.0 * np.pi * self.f * t))

    def step(self, dt: float):
        phi_dot = self._phi_dot(self.t)
        self.phi += phi_dot * dt
        self.t += dt
        return self.eval()

    def eval(self):
        t = self.t
        phi = self.phi
        phi_dot = self._phi_dot(t)

        # Smooth 5-point star in polar form: r = a - b*cos(5*phi)
        r = self.a - self.b * np.cos(5.0 * phi)
        dr_dphi = 5.0 * self.b * np.sin(5.0 * phi)

        x = r * np.cos(phi)
        y = r * np.sin(phi)
        p_local = np.array([x, y, 0.0], dtype=np.float64)

        p_d = self.pc + self.Rp @ p_local

        dx_dphi = dr_dphi * np.cos(phi) - r * np.sin(phi)
        dy_dphi = dr_dphi * np.sin(phi) + r * np.cos(phi)
        v_local = np.array([dx_dphi, dy_dphi, 0.0], dtype=np.float64) * phi_dot
        v_d = self.Rp @ v_local

        theta = self.theta_max * np.sin(2.0 * np.pi * self.f * t)
        R_osc = rot_axis_angle(self.u, theta)
        R_d = self.R0 @ R_osc

        theta_dot = self.theta_max * (2.0 * np.pi * self.f) * np.cos(2.0 * np.pi * self.f * t)
        w_d = self.u * theta_dot

        phase = np.array([np.sin(phi), np.cos(phi)], dtype=np.float64)
        return p_d, R_d, v_d, w_d, phase
