

import math

# import sys, os
# sys.path.append("/opt/conda/lib/python3.7/site-packages/")

# print(sys.path)

import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt
import random


class TTOPEnv(gym.Env):
    """
    説明：
        ポールは非駆動ジョイントでカートに取り付けられ、摩擦のないトラック上を移動します。
        ペンデュラムは直立状態で始まり、カートの速度を増減させることで倒れるのを防ぐことが目標です。

    ソース：
        [Wang21]を参考にしました。

    観測：
        タイプ：Box(3)
        番号    観測                     最小値                     最大値
        0       経過時間               0               計画時間 "T"
        1       列車の速度             0             速度制限 "V_lim"
        2       列車の位置             0          駅間距離 "L"

    行動：
        タイプ：Discrete(4)
        番号   行動
        0     最大力行
        1     速度維持
        2     惰性走行
        3     最大制動

        注意：速度が減少または増加する量は固定されていません。これは、ポールの重心がカートの下で移動するために必要なエネルギー量を増加させるためです。

    報酬：
        エージェントがゴール状態に到達すると報酬は1です。

    開始状態：
        (0, 0, 0)

    エピソード終了：
        経過時間が "T" 秒に等しい場合。
        エピソードの長さが200を超える場合。
        解決要件：
        100回の連続した試行で平均リターンが195.0以上になった場合に解決とみなされます。
    """

    # "render.modes"キーは、環境の描画モード  "video.frames_per_second"キーは、ビデオのフレームレート
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50} 

    def __init__(self):
        """
        self.mass[kg]: 列車の総重量
        self.radious[m]: 車輪の半径
        self.time_step[sec]: アクションの離散化された時間
        self.delta_time[sec]: パワーを計算するための離散化された時間
        self.planning_time[sec]: 駅間の走行計画時間
        self.distance_between_stations[m]: 駅間の距離
        self.F_lower[N]: 最大制動の下限値
        self.init_F_upper[N]: "v=0"のときの牽引力の上限値
        self.F_upper[N]: 各速度における牽引力の上限値
        self.threshold_velocity1[km/h]: 定トルク領域と定電力領域の閾値
        self.threshold_velocity2[km/h]: 定電力領域と特性領域の閾値
        self.velocity_limit[km/h]: 駅間の速度制限
        self.alpha: Davis方程式のパラメータ
        self.beta: Davis方程式のパラメータ
        self.gamma: Davis方程式のパラメータ
        """
        
        # 初期値
        self.mass = 300 * (10**3)
        self.gravity_acceleration = 9.8
        self.radious = 0.8
        self.time_step = 1
        self.delta_time = 1e-2
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
        # self.alpha = 5.8584
        # self.beta = 0.0206
        # self.gamma = 0.001
        self.num_envs = 1
        self.s0_distrib = []
        
        # JR (denki-tetsudo, pp.90-91)
        self.n = 10
        self.alpha = 1.32
        self.beta = 0.0164
        self.gamma = (0.0280 + 0.0078*(self.n-1)) / ((self.mass/(10**3))*self.gravity_acceleration)
        
        # 状態空間の定義
        self.low = np.array(
            [
                0,
                0,
                0,
            ],
            dtype=np.float32,
        )
        self.high = np.array(
            [
                self.planning_time,
                self.distance_between_stations + self.velocity_limit*self.time_step,
                self.velocity_limit,
            ],
            dtype=np.float32,
        )
        
        # 行動空間の定義
        Num_of_Action = 4
        self.action_space = spaces.Discrete(Num_of_Action)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

        self.previous_velocity = self.dictionary()
        
        self.can_be_position = self.can_reach_position()
        
        #　初期状態
        self.s0 = (0, 0, 0)
        self.reset()

    #　シード値を設定
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    #　与えられた速度に基づいて最大力行を計算
    def maximum_force(self, velocity):
        """
        <input>
        velocity[km/h]
        <output>
        max_force[N]: 最大力行
        """
        if velocity*3.6 <= self.threshold_velocity1:
            max_torque = 576
        elif self.threshold_velocity1 < velocity*3.6 <= self.threshold_velocity2:
            max_torque = 34560 / (velocity*3.6 + 20)
        else:
            max_torque = 448331 / ((velocity*3.6 - 29.2638)**2) + 113.842
        max_force = (max_torque*(10**3)) / self.radious
        return max_force

    def runnning_resistance(self, velocity):
        """
        <input>
        velocity[m/sec]
        <output>
        R_r[N]: 走行抵抗
        """
        Rr_weight = self.alpha + self.beta*(velocity*3.6) + self.gamma*((velocity*3.6)**2)
        Rr = Rr_weight * (self.mass / (10**3)) * self.gravity_acceleration
        return Rr
    
    def can_reach_position(self):
        self.s0 = (0, 0, 0)
        self.reset()
        can_be_position = {0: 0}
        for i in range(self.planning_time):
            if self.state[2] < (self.velocity_limit/3.6):
                action = 3
            elif self.state[2] == (self.velocity_limit/3.6):
                action = 2
            state = self.state
            next_state = self.step(action)[0].tolist()
            if next_state[2] > (self.velocity_limit/3.6):
                next_state = list(state)
                acceleration = self.accel(action, state[0])[0]
                next_state[2] = (self.velocity_limit/3.6)
                time = (next_state[2] - state[2]) / acceleration
                middle_position = state[1] + state[2]*time + 0.5*acceleration*(time**2)
                next_state[1] = middle_position + next_state[2]*(self.time_step - time)
                next_state[0] += 1
                self.state = tuple(next_state)
            if self.state[1] > self.distance_between_stations:
                can_be_position[self.state[0]] = self.distance_between_stations
            else:
                can_be_position[self.state[0]] = self.state[1]
        return can_be_position




    def can_action(self):
        velocity = self.state[2]
        feasible_A = []
        for action in range(self.action_space.n):
                feasible_A.append(action)
        if velocity > 0:
            return feasible_A
        else:
            feasible_A.remove(0)
            return feasible_A


    def reward_function(self, next_state, energy):
        reward = - 0.001 * energy
        time, position, velocity = next_state

        done = bool(
            time >= self.planning_time
        )

        if done:
            if abs(position - self.distance_between_stations) <= self.allowable_error_of_position and velocity == 0:
                reward += 1      
        return reward, done

    def accel(self, action, velocity):
        """
        force[N]
        acceleration[m/sec^2]
        """
        running_resistance = self.runnning_resistance(velocity)
        if action==0:
            force = -self.F_lower
        elif action==1:
            if velocity==0:
                running_resistance = 0
            force = 0
        elif action==2:
            if velocity==0:
                running_resistance = 0
            force = running_resistance
        else:
            force = self.maximum_force(velocity)
        acceleration = (1/self.mass) * (force - running_resistance)
        return acceleration, force, running_resistance
    
    def max_velocity(self, time):
        max_v1 = (self.init_F_upper / self.mass) * time
        max_v2 = 1.2 * (self.F_lower / self.mass) * (time - self.planning_time)
        return min(max_v1, max_v2, self.velocity_limit/3.6)
    
    def dictionary(self):
        v = 0
        dic = {}
        for vv in range(int(self.velocity_limit / 0.01)):
            v = vv * 0.01 / 3.6
            for action in range(self.action_space.n):
                accel = self.accel(action, v)[0]
                v_dash = v + accel * self.time_step
                v_dash = round(v_dash, 1)
                if 0 <= v_dash < self.velocity_limit:
                    if v_dash not in dic:
                        dic[v_dash] = {}
                        dic[v_dash][action] = v
                    else:
                        dic[v_dash][action] = v
        return dic

    def _move(self, action):
        time, position, velocity = self.state
        acceleration, force, running_resistance = self.accel(action, velocity)

        previous_velocity = velocity
        
        time += self.time_step
        velocity += acceleration*self.time_step
        if velocity < 0:
            velocity = 0
            position += -(previous_velocity**2)/acceleration + 0.5*(previous_velocity**2)/acceleration
            energy = 0
        else:
            position += previous_velocity*self.time_step + 0.5*acceleration*(self.time_step**2)
            if force > 0:
                energy = 0.5 * (force*previous_velocity + force*velocity) * self.time_step
            else:
                energy = 0

        next_state = (time, position, velocity)

        energy = energy / (10**6) # [MJ]

        return next_state, energy, acceleration

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg
        # state trainsition
        self.state, energy, acceleration = self._move(action)
        # calculate reward
        reward, done = self.reward_function(self.state, energy)
        # normalize state vector
        norm_state = self.normalize_state()
        # output result
        return norm_state, reward, done, {"energy": energy, "action": action, "accel": acceleration}

    def random_sampling_state(self):
        # retention state
        state = self.state
        # lower_and_higher bound
        lower_time = self.low[0]
        lower_position = self.low[1]
        lower_speed = self.low[2]
        higher_time = self.high[0]
        higher_position = self.high[1]
        higher_speed = self.high[2]
        # random value of each dimension
        time = random.randint(lower_time, higher_time)
        position = np.random.uniform(low=lower_position, high=higher_position)
        speed = np.random.uniform(low=lower_speed, high=higher_speed)
        self.state = (time, position, speed)
        norm_state = self.normalize_state()
        self.state = state
        # output result
        return norm_state
    
    def partially_sampling_state(self, t_width, x_width, v_width):
        state = self.state
        time, position, speed = state
        # lower_and_higher bound
        lower_time = self.low[0]
        lower_position = self.low[1]
        lower_speed = self.low[2]
        higher_time = self.high[0]
        higher_position = self.high[1]
        higher_speed = self.high[2]
        # replace bound
        if time - t_width >= lower_time:
            lower_time = time - t_width
        if position - x_width >= lower_position:
            lower_position = position - x_width
        if speed - v_width >= lower_speed:
            lower_speed = speed - v_width
        if time + t_width <= higher_time:
            higher_time = time + t_width
        if position + x_width <= higher_position:
            higher_position = position + x_width
        if speed + v_width <= higher_speed:
            higher_speed = speed + v_width
        # random value of each dimension
        time = random.randint(lower_time, higher_time)
        position = np.random.uniform(low=lower_position, high=higher_position)
        speed = np.random.uniform(low=lower_speed, high=higher_speed)
        self.state = (time, position, speed)
        norm_state = self.normalize_state()
        self.state = state
        # output result
        return norm_state

    def normalize_state(self):
        state = np.array(self.state, dtype=np.float32)
        norm_state = state / self.norm_vector
        return norm_state
    
    def inverse_norm_state(self, norm_state):
        state = norm_state * self.norm_vector
        state[0] = int(state[0])
        return state

    def set_s0(self, s0):   # check
        self.s0 = s0

    def reset(self):
        self.state = self.s0
        norm_state = self.normalize_state()
        return norm_state

    def close(self):
        pass


def draw_speed_sequence(speed_seq):
    index_seq = range(len(speed_seq))
    speed_seq_array = np.array(speed_seq)
    plt.plot(index_seq, speed_seq_array, color="red")
    plt.savefig("./speed_trajectory.png")



def test():
    env = TTOPEnv()
    print(env.state)  # (時間、位置、速度)
    energy_sum = 0
    vel_seq = [0, ]  # 速度の時系列
    for i in range(18):
        _, _, _, info = env.step(3)
        energy_sum += info["energy"]
        print(env.state)
        vel_seq.append(env.state[2])
    for i in range(18):
        _, _, _, info = env.step(2)
        energy_sum += info["energy"]
        print(env.state)
        vel_seq.append(env.state[2])
    for i in range(36):
        _, _, _, info = env.step(1)
        energy_sum += info["energy"]
        print(env.state)
        vel_seq.append(env.state[2])
    for i in range(10):
        _, _, _, info = env.step(0)
        energy_sum += info["energy"]
        print(env.state)
        vel_seq.append(env.state[2])
    
    print("energy sum:", energy_sum, "[MJ]")
    print("\nVelocity_sequence:\n", vel_seq)

    draw_speed_sequence(vel_seq)

if __name__=='__main__':
    test()
