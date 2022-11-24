import math
import numpy as np
from copy import deepcopy
from envs.quadrotor import QuadRotorAsset


class PIDController(QuadRotorAsset):
    def __init__(self,
                 args,
                 sample_time=0.01,
                 ):
        super(PIDController, self).__init__(sample_time, args)

        self.position_dim = 3
        self.velocity_dim = 3
        self.rotation_dim = 6
        self.angular_velocity_dim = 3
        self.state_dim = self.position_dim + self.velocity_dim + self.rotation_dim + self.angular_velocity_dim
        self.action_dim = 4

        # For PID controller
        self.error_sum_xy, self.error_sum_z, self.error_sum_rp, self.error_sum_y = np.zeros(2), np.zeros(1), np.zeros(2), np.zeros(1)
        self.ud_xy_tmp = np.zeros_like(self.error_sum_xy)
        self.ud_z_tmp = np.zeros_like(self.error_sum_z)
        self.ud_rp_tmp = np.zeros_like(self.error_sum_rp)
        self.ud_y_tmp = np.zeros_like(self.error_sum_y)

        self.Kpid_acc_xy = np.array([5., 2., 0.2])
        self.Kpid_acc_z = np.array([5., 5., 0.])
        self.Kpid_omega_rp = np.array([20., 100., 0.4])
        self.Kpid_omega_y = np.array([20., 100., 0.4])

    def get_force(self, state, obs):

        self._get_obs(state, obs)
        acc_ref = self._get_ref_acc()
        eta_ref, p2dot_ref = self._acc_to_attitude(acc_ref)
        omega1dot, eta = self._get_ref_omega1dot(eta_ref)
        f = self._acc_to_force(omega1dot, p2dot_ref, eta)
        f = np.clip(f, 0.1, 2.0)

        # print("*******************************************************")
        # print("error_distance", obs['position_error_obs'][:self.position_dim])
        # print("acc_ref", acc_ref)
        # print("eta_ref", eta_ref, "p2dot_ref", p2dot_ref)
        # print("omega1dot", omega1dot, "eta", eta)
        # print("cal_f", f)
        return f

    def _get_obs(self, state, obs):
        self.state = state
        self.obs = obs

    def _pid(self, error, prev_error_sum, ud_tmp, gain, clip_range):

        error_sum = prev_error_sum + self.sample_time * error

        ud = (gain[2] * error) * 15 - ud_tmp

        u = gain[0] * error + \
                   np.clip(gain[1] * error_sum, -clip_range, clip_range) + ud

        return u, error_sum, ud_tmp + self.sample_time * ud

    def _get_ref_acc(self):
        Ep = self.obs["position_error_obs"][:self.position_dim]

        # product the time coefficient
        E_xy_now = np.clip(1.5 * Ep[:2], -3., 3.) - self.state[6:8]
        E_z_now = np.clip(1.5 * Ep[-1], -0.5, 0.5) - self.state[8]

        u_acc_xy, self.error_sum_xy, self.ud_xy_tmp = self._pid(E_xy_now, self.error_sum_xy, self.ud_xy_tmp, self.Kpid_acc_xy, 1.)
        u_acc_z, self.error_sum_z, self.ud_z_tmp = self._pid(E_z_now, self.error_sum_z, self.ud_z_tmp, self.Kpid_acc_z, 3.)

        return np.hstack((u_acc_xy, u_acc_z))

    def _acc_to_attitude(self, acc_ref):

        R = self.obs["rotation_obs"][:self.rotation_dim].reshape([3, 2])
        np.sin_psi = R[2, 0]
        np.cos_psi = R[2, 1]

        acc_ref_z = acc_ref[2] + self.gravity

        phi_des = math.asin((acc_ref[0] * np.sin_psi - acc_ref[1] * np.cos_psi) /
                         (math.sqrt(acc_ref[0]**2 + acc_ref[1]**2 + acc_ref_z**2)))
        theta_des = math.atan2((acc_ref[0] * np.cos_psi + acc_ref[1] * np.sin_psi), acc_ref_z)

        return np.array([phi_des, theta_des, 0]), acc_ref_z

    def _get_ref_omega1dot(self, eta_des):

        eta_now = self.state[3:6]
        omega_now = self.state[9:12]

        E_phi_theta_now = 5*(np.clip(eta_des[:2], -math.pi/6, math.pi/6) - eta_now[:2])
        E_psi_now = 3*self._psi_transform(eta_des[2] - eta_now[2])
        eta1dot = np.hstack((E_phi_theta_now, E_psi_now))
        omega_ref = np.array([eta1dot[0] - np.sin(eta_now[1]) * eta1dot[2],
                          np.cos(eta_now[0]) * eta1dot[1] + np.sin(eta_now[0]) * np.cos(eta_now[1]) * eta1dot[2],
                          -np.sin(eta_now[0]) * eta1dot[1] + np.cos(eta_now[0]) * np.cos(eta_now[1]) * eta1dot[2]])

        E_omega = omega_ref - omega_now
        u_omega_rp, self.error_sum_rp, self.ud_rp_tmp = self._pid(E_omega[:2], self.error_sum_rp, self.ud_rp_tmp, self.Kpid_omega_rp, 30.)
        u_omega_y, self.error_sum_y, self.ud_y_tmp = self._pid(E_omega[-1], self.error_sum_y, self.ud_y_tmp, self.Kpid_omega_y, 10.)

        return np.hstack((u_omega_rp, u_omega_y)), eta_now

    def _acc_to_force(self, omega1dot, p2dot, eta):

        cosPhi = np.cos(eta[0])
        cosTheta = np.cos(eta[1])

        d_force = p2dot / (cosTheta * cosPhi)
        f_altitude = (self.mass / 4) * d_force * np.ones_like(4)
        f_roll = (self.inertia[0][0] / (2 * self.length)) * omega1dot[0] * np.array([0, 0, 1, -1])
        f_pitch = (self.inertia[1][1] / (2 * self.length)) * omega1dot[1] * np.array([-1, 1, 0, 0])
        t_yaw = (self.inertia[2][2] / 4) * omega1dot[2] * np.array([-1, -1, 1, 1])

        f = f_altitude + f_roll + f_pitch + 78.62778730703259 * t_yaw

        return f

    def _psi_transform(self, psi):

        if psi >= 0:
            psi = psi % (2*math.pi)
        else:
            psi = psi % (2*math.pi) - 2*math.pi

        if psi > math.pi:
            psi -= 2*math.pi
        if psi < -math.pi:
            psi += 2*math.pi
        return psi



