import gym
from gym import spaces
from gym.utils import seeding
import numpy as np


class TTOPEnv(gym.Env):

    def __init__(self):
        self.mass = 300 * (10**3)
        self.gravity_acceleration = 9.8
        self.radious = 0.8
        self.time_step = 1
        self.planning_time = 90
        self.distance_between_stations = 2000
        self.F_lower = 864 * (10**3)
        self.init_F_upper = 576 * (10**3)
        self.F_upper = self.init_F_upper
        self.threshold_velocity1 = 40
        self.threshold_velocity2 = 70
        self.velocity_limit = 110
        self.allowable_error_of_position = 1.0
        self.norm_vector = np.array([self.planning_time, self.distance_between_stations, self.velocity_limit/3.6], dtype=np.float32)

        self.n = 10
        self.alpha = 1.32
        self.beta = 0.0164
        self.gamma = (0.0280 + 0.0078*(self.n-1)) / ((self.mass/(10**3))*self.gravity_acceleration)

        self.low = np.array([0, 0, 0], dtype=np.float32)
        self.high = np.array([
            self.planning_time,
            self.distance_between_stations + self.velocity_limit*self.time_step,
            self.velocity_limit,
        ], dtype=np.float32)

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

        self.s0 = (0, 0, 0)
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def maximum_force(self, velocity):
        if velocity*3.6 <= self.threshold_velocity1:
            max_torque = 576
        elif self.threshold_velocity1 < velocity*3.6 <= self.threshold_velocity2:
            max_torque = 34560 / (velocity*3.6 + 20)
        else:
            max_torque = 448331 / ((velocity*3.6 - 29.2638)**2) + 113.842
        max_force = (max_torque*(10**3)) / self.radious
        return max_force

    def runnning_resistance(self, velocity):
        Rr_weight = self.alpha + self.beta*(velocity*3.6) + self.gamma*((velocity*3.6)**2)
        Rr = Rr_weight * (self.mass / (10**3)) * self.gravity_acceleration
        return Rr

    def reward_function(self, next_state, energy):
        reward = -0.001 * energy
        time, position, velocity = next_state

        done = bool(time >= self.planning_time)

        if done and abs(position - self.distance_between_stations) <= self.allowable_error_of_position and velocity == 0:
            reward += 1
        return reward, done

    def accel(self, action, velocity):
        running_resistance = self.runnning_resistance(velocity)
        if action == 0:
            force = -self.F_lower
        elif action == 1:
            if velocity == 0:
                running_resistance = 0
            force = 0
        elif action == 2:
            if velocity == 0:
                running_resistance = 0
            force = running_resistance
        else:
            force = self.maximum_force(velocity)
        acceleration = (1/self.mass) * (force - running_resistance)
        return acceleration, force

    def _move(self, action):
        time, position, velocity = self.state
        acceleration, force = self.accel(action, velocity)

        previous_velocity = velocity
        time += self.time_step
        velocity += acceleration * self.time_step

        if velocity < 0:
            velocity = 0
            position += 0.5 * (previous_velocity**2) / (-acceleration)
            energy = 0
        else:
            position += previous_velocity * self.time_step + 0.5 * acceleration * (self.time_step**2)
            if force > 0:
                energy = 0.5 * (force * previous_velocity + force * velocity) * self.time_step
            else:
                energy = 0

        energy = energy / (10**6)
        return (time, position, velocity), energy, acceleration

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = action.item()
        self.state, energy, acceleration = self._move(action)
        reward, done = self.reward_function(self.state, energy)
        norm_state = self.normalize_state()
        return norm_state, reward, done, {"energy": energy, "action": action, "accel": acceleration}

    def normalize_state(self):
        state = np.array(self.state, dtype=np.float32)
        return state / self.norm_vector

    def inverse_norm_state(self, norm_state):
        state = norm_state * self.norm_vector
        state[0] = int(state[0])
        return state

    def reset(self):
        self.state = self.s0
        return self.normalize_state()

    def close(self):
        pass
