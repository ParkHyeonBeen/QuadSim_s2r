import numpy as np
import math

class QuadRotorAsset():
    def __init__(self, sample_time=0.01, args=None):

        """
        state = x y z phi tht psi x_dot y_dot z_dot p q r --> 12 dim
        action = Thrust or rpm or pwm or turque of each motor --> 4dim

        """
        self.step_size = args.step_size          # unit: second
        self.sample_time = sample_time
        self.init_lag_ratio = args.lag_ratio
        assert self.sample_time % self.step_size == 0
        self.n_frame = int(self.sample_time/self.step_size)
        self.mode = args.quad_ver

        if self.mode == "v1":  # 3d printed micro drone
            self.init_mass = 0.407
            self.init_length = 0.1125
            self.init_inertia = \
                np.array([
                    [0.001318214634861, 0, 0],
                    [0, 0.001443503665657, 0],
                    [0, 0, 0.002477708981071]])
            self.init_kt = 9.168e-9
            self.init_max_rpm = 14617.16784233
            self.init_kq = 1.166e-10

        elif self.mode == "v2":  # carbon mini drone
            self.init_mass = 0.640
            self.init_length = 0.14
            self.init_inertia = \
                np.array([
                    [0.00728, 0, 0],
                    [0, 0.00728, 0],
                    [0, 0, 0.00993]])
            self.init_kt = 4.612e-8
            self.init_max_rpm = 14945.4
            self.init_kq = 6.411e-10

        self.gravity = args.gravity         # gravity coefficient
        self.mass = self.init_mass          # mass
        self.length = self.init_length      # arm length
        self.inertia = self.init_inertia    # inertia matrix

        self.tc = args.tc                   # time constant between motor and propeller
        self.kt = self.init_kt              # RPM to Thrust coefficient
        self.max_rpm = self.init_max_rpm    # Maximum RPM
        self.kq = self.init_kq              # RPM to Torque coefficient
        self.lag_ratio = self.init_lag_ratio

        # related to disturbance
        self.alpha = args.alpha             # Thrust to Torque coefficient
        self.delta = args.delta
        self.sigma = args.sigma

    def do_simulation(self, thrust):
        # RK4
        for _ in range(self.n_frame):
            xd1 = self._get_state_dot(self.state, thrust)
            xd2 = self._get_state_dot(self.state + (self.step_size/2)*xd1, thrust)
            xd3 = self._get_state_dot(self.state + (self.step_size/2)*xd2, thrust)
            xd4 = self._get_state_dot(self.state + self.step_size*xd3, thrust)
            xd = (xd1 + 2*xd2 + 2*xd3 + xd4)/6
            self.state += self.step_size*xd

    def set_state(self, qpos, qvel):
        self.state = np.concatenate((qpos, qvel))

    def _get_state_dot(self, state, thrust):

        """
        thrust info
        f1 : x-axis, positive, ccw
        f2 : x-axis, negative, ccw
        f3 : y-axis, positive, cw
        f4 : y-axis, negative, cw
        """
        # thrust = np.clip(thrust, 0.1*np.ones_like(thrust), 2.0*np.ones_like(thrust))
        # print(thrust)
        f1, f2, f3, f4 = thrust
        _, _, _, phi, tht, psi, x_dot, y_dot, z_dot, p, q, r = state.copy()

        R = self._get_rotation_matrix([phi, tht, psi])
        C_inv = self._get_C_inv([phi, tht, psi])

        x_2dot = -R[0][2]*(f1 + f2 + f3 + f4)/self.mass
        y_2dot = -R[1][2]*(f1 + f2 + f3 + f4)/self.mass
        z_2dot = -R[2][2]*(f1 + f2 + f3 + f4)/self.mass + self.gravity

        phi_dot = C_inv[0][0]*p + C_inv[0][1]*q + C_inv[0][2]*r
        tht_dot = C_inv[1][0]*p + C_inv[1][1]*q + C_inv[1][2]*r
        psi_dot = C_inv[2][0]*p + C_inv[2][1]*q + C_inv[2][2]*r

        p_dot = ((self.inertia[1][1] - self.inertia[2][2]) / self.inertia[0][0]) * q * r + self.length * (f2 - f1) / self.inertia[0][0]
        q_dot = ((self.inertia[2][2] - self.inertia[0][0]) / self.inertia[1][1]) * p * r + self.length * (f3 - f4) / self.inertia[1][1]
        r_dot = ((self.inertia[0][0] - self.inertia[1][1]) / self.inertia[2][2]) * p * q + (self.kq / self.kt) * (f1 + f2 - f3 - f4) / self.inertia[2][2]

        return np.array([x_dot, y_dot, z_dot,
                         phi_dot, tht_dot, psi_dot,
                         x_2dot, y_2dot, z_2dot,
                         p_dot, q_dot, r_dot])

    def _get_rotation_matrix(self, inp):
        if len(inp) != 0:
            phi, theta, psi = inp
        else:
            phi, theta, psi = self.state[3:6]

        return np.array([
            [math.cos(theta)*math.cos(psi), math.sin(phi)*math.sin(theta)*math.cos(psi) - math.cos(phi)*math.sin(psi), math.cos(phi)*math.sin(theta)*math.cos(psi) + math.sin(phi)*math.sin(psi)],
            [math.cos(theta)*math.sin(psi), math.sin(phi)*math.sin(theta)*math.sin(psi) + math.cos(phi)*math.cos(psi), math.cos(phi)*math.sin(theta)*math.sin(psi) - math.sin(phi)*math.cos(psi)],
            [-math.sin(theta), math.sin(phi)*math.cos(theta), math.cos(phi)*math.cos(theta)]])

    def _get_euler(self, inp):
        if len(inp) != 0:
            phi, theta, psi = inp
        else:
            phi, theta, psi = self.state[3:6]

        return np.array([
            [math.sin(phi), math.cos(phi)],
            [math.sin(theta), math.cos(theta)],
            [math.sin(psi), math.cos(psi)]])

    def _get_C_inv(self, inp):     # inverse matrix of Euler rate to Body angular rate matrix
        if len(inp) != 0:
            phi, theta, psi = inp
        else:
            phi, theta, psi = self.state[3:6]

        return np.array([
            [1, (math.sin(phi)*math.sin(theta))/math.cos(theta), (math.cos(phi)*math.sin(theta))/math.cos(theta)],
            [0, math.cos(phi), -math.sin(phi)],
            [0, math.sin(phi)/math.cos(theta), math.cos(phi)/math.cos(theta)]])

    def _dshot_to_thrust(self, dshot):
        """
        Dshot range = 0 ~ 1000
        thrust = 6.888e-6*dshot**2 + 0.00393*dshot -0.5164 , range = -0.5164 ~ 10.3016
        """
        return 6.888e-6*dshot**2 + 0.00393*dshot -0.5164

    def _thrust_to_dshot(self, thrust):   #
        """
        linear model
        only drone : -93.27*thrust**4 + 450.9*thrust**3 - 759.7*thrust**2 + 931.9*thrust + 36.57, range = 122.604573 ~ 976.45
        with guard : -0.2664*thrust**4 + 5.203*thrust**3 - 38.2*thrust**2 + 204.7*thrust + 95.96, range = 116.05317636 ~ 389.92159999999996
        """
        return -93.27*thrust**4 + 450.9*thrust**3 - 759.7*thrust**2 + 931.9*thrust + 36.57

    def _pwm_to_rpm(self, pwm):     # rpm range = 3266.50557168 ~ 14617.16784233
        return -0.006748*pwm**2 + 20.71*pwm + 828.8

    def _rpm_to_thrust(self, rpm):  # Thrust range (N) : 0.1 ~ 2.0
        """
        linear model
        only drone : 9.168e-9*rpm**2, range = 0.1 ~ 2.0
        with guard : 5.149e-8*rpm**2 + -6.711e-5*rpm + 0.1102, range = 116.05317636 ~ 389.92159999999996
        """
        return 9.168e-9*rpm**2

    def _rpm_to_torque(self, rpm): # torque range (N*m): 0.0012441288385679795 ~ 0.02491294206221667
        return 1.166e-10*rpm**2

if __name__ == "__main__":
    env = QuadRotorAsset()
    ctrl = np.array([2, 2, 0.1, 2])
    for i in range(100):
        print("i th:", i)
        env.do_simulation(ctrl)
        print("state : ", env.state)
