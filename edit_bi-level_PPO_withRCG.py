# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 11:37:39 2021

@author: tori1
"""

import sys
sys.path.append('/home/user/Downloads/train/pfrl/experiments')

import random
from collections import defaultdict

import pfrl.experiments.train_agent_batch
from pfrl.experiments.train_agent_batch import train_agent_batch_with_evaluation

import argparse
import os
import functools

import time
import collections

import gym
import gym.spaces
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.optim as optim

import pfrl
from pfrl import experiments, utils, explorers
from pfrl.agents import PPO
from pfrl import nn as pnn
from pfrl import q_functions, replay_buffers
from pfrl.agents.dqn import DQN

from gym.envs.registration import (
    register,
)

register(
    id="ttop-v1",
    entry_point="Env_TrainOperation_energy:TTOPEnv",
)

register(
    id="ttop-v2",
    entry_point="Env_TrainOperation_energy:TTOPEnv",
)



class ReverseCurriculum():
    
    def __init__(
        self,
        env,
        agent,
        steps=1000,
        eval_n_steps=1000,
        eval_n_episodes=10,
        eval_interval=1000,
        log_interval=100,
        outdir=None,
        eval_env=None,
        timestep_limit=100,
        sample=100,
        N_new=1000,
        N_old = 10,
        iteration=20,
        T_rev=7,
        Rmin=0.1,
        Rmax=0.6,
        ):
        self.env = env
        self.agent = agent
        self.steps = steps
        self.eval_n_steps = eval_n_steps
        self.eval_n_runs = eval_n_episodes
        self.eval_interval = eval_interval
        self.log_interval = log_interval
        self.outdir = outdir
        self.eval_env=eval_env,
        self.train_max_episode_len=timestep_limit,
        self.X = self.env.distance_between_stations
        self.T = self.env.planning_time
        self.sg = (self.T, self.X, 0)
        self.S_int = []
        self.F_lower = self.env.F_lower
        self.F_upper = self.env.init_F_upper
        self.a_max = self.env.init_F_upper / self.env.mass
        self.v_limit = self.env.velocity_limit
        self.allowable_error = self.env.allowable_error_of_position
        self.action_number = self.env.action_space.n
        self.sample_s0 = sample
        self.action_list = []
        for a in range(self.action_number):
            self.action_list.append(a)
        self.dt = self.env.time_step
        # self.N = 3   # number of generate from each state
        self.N_new = N_new   # generate state in each iteration
        self.N_old = N_old   # number of state sampling from start_s_old
        self.goal_state_num = self.N_old
        self.M = self.N_new * 3
        self.iteration = iteration
        self.T_rev = T_rev   # time-step for reverse
        self.Rmin = Rmin
        self.Rmax = Rmax
        self.last_count = 10
        self.steps = steps
        self.delete_dict = {}
        self.delete_dict = defaultdict(lambda: 0)
        self.s0_dict = {}
        self.s0_dict = defaultdict(lambda: 0)
    
    def reverse_move(self, state, action):
        b_state = list(state)
        b_state[0] = state[0] - self.dt
        b_state[2] = self.env.previous_velocity[round(state[2], 1)][action]
        accel = self.env.accel(action, b_state[2])[0]
        b_state[1] = state[1] - b_state[2]*self.dt - 0.5*accel*(self.dt**2)
        return tuple(b_state)
    
    def _create_state(self, state):
        S = []
        org_state = state
        state = org_state

        for t in range(self.T_rev):
            a_list = self.action_list.copy()
            for a in self.action_list:
                if a not in self.env.previous_velocity[round(state[2], 1)]:
                    a_list.remove(a)
            if a_list == []:
                break
            else:
                action = random.choice(a_list)
                state = self.reverse_move(state, action)
            if state[2] < 0:
                continue
            elif state[2] > (self.env.init_F_upper / self.env.mass) * state[0] or state[2] <= 0 or state[1] < 0 or state[0] < 0:
                continue
            else:
                S.append(state)
        return S
    
    def sample_nearby(self, start_s):
        start_s_new = []
        S_0 = start_s.copy()
        count = 0
        while len(S_0) < self.M:
            sc = random.choice(start_s)
            S = self._create_state(sc)
            for s in S:
                if s not in start_s and s[1] <= self.env.can_be_position[s[0]]:
                    # start_s.append(s)
                    S_0.append(s)

            if len(S_0)>self.N_new and count>(self.M):
                break
        start_s_new = random.sample(S_0, self.N_new)
        return start_s_new

    def train_policy(self, start_s):
        _start_s = []
        rews = []
        for i in range(self.sample_s0):
            while True:
                s0 = random.choice(start_s)
                if type(s0[0]) == type((2, 3, 4)):
                    print(s0)
                if s0[0] < self.env.planning_time:
                    break
            _start_s.append(s0)
            print("\n sample", i+1, "s0:", s0)
            self.env.s0_distrib = [s0]
            self.agent, _, R, _ = experiments.train_agent_batch_with_evaluation(
                agent=self.agent,
                env=self.env,
                eval_env=self.env,
                outdir=self.outdir,
                steps=self.steps,
                eval_n_steps=None,
                eval_n_episodes=self.eval_n_runs,
                eval_interval=self.eval_interval,
                log_interval=self.log_interval,
                max_episode_len=self.train_max_episode_len,
                save_best_so_far_agent=False,
                s0=s0
            )
            rews.append(R)
        return _start_s, rews

    def train_policy2(self, s0_distrib, step):
        self.env.s0_distrib = s0_distrib
        s0 = (0, 0, 0)
        self.agent, _, R, r_list = experiments.train_agent_batch_with_evaluation(
            agent=self.agent,
            env=self.env,
            eval_env=self.env,
            outdir=self.outdir,
            steps=step,
            eval_n_steps=None,
            eval_n_episodes=self.eval_n_runs,
            eval_interval=self.eval_interval,
            log_interval=self.log_interval,
            max_episode_len=self.train_max_episode_len,
            save_best_so_far_agent=False,
            s0=s0
        )
        return self.agent, r_list
    
    def generate_goal_state(self, n):
        goal_state_set = []
        rews = []
        for i in range(n):
            position_error = random.uniform(- self.allowable_error, self.allowable_error)
            goal_position = self.X + position_error
            goal_state = (self.T, goal_position, 0)
            goal_state_set.append(goal_state)
            rews.append(1)
        return goal_state_set, rews
    
    def select(self, _start_s, rews):
        start_s = []
        for s0 in _start_s:
            if self.Rmin < rews[_start_s.index(s0)] < self.Rmax:
                start_s.append(s0)
        return start_s
    
    def reverse_curriculum(self):
        start_s_old, rews = self.generate_goal_state(self.goal_state_num)
        start_s = start_s_old.copy()
        after_start_s_list = []
        for iter in range(self.iteration):
            print("\n \n \n \n \n \n \n [iteration", iter+1, "]\n")
            if start_s == []:
                start_s += random.sample(start_s_old, self.N_old)
            print("previous_start_s: \n", start_s)
            start_s = self.sample_nearby(start_s)
            print("\n near_s0: \n", start_s)
            start_s += random.sample(start_s_old, self.N_old)
            print("\n after_start_s: \n ", start_s)
            after_start_s_list.append(start_s)
            # start_s = self.sample_nearby(start_s)
            _start_s, rews = self.train_policy(start_s)
            print("\n _start_s: \n", _start_s)
            print("\n rews: \n", rews)
            start_s = self.select(_start_s, rews)
            start_s_old += start_s
        with open('after_start_s.txt', 'w') as f:
            print(after_start_s_list, file=f)
        return self.agent


def main():
    import logging

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0, help="GPU to use, set to -1 if no GPU.")
    parser.add_argument("--env", type=str, default="ttop-v1", help="OpenAI Gym MuJoCo env to perform algorithm on.",)
    parser.add_argument("--num-envs", type=int, default=1, help="Number of envs run in parallel.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed [0, 2 ** 32)")
    parser.add_argument("--is-random-seed", action="store_true", default=False)
    parser.add_argument("--outdir1", type=str, default="result_phase1", help=("Directory path to save output files." " If it does not exist, it will be created."),)
    parser.add_argument("--outdir2", type=str, default="result_phase2", help=("Directory path to save output files." " If it does not exist, it will be created."),)
    parser.add_argument("--steps", type=int, default=1000, help="Total number of timesteps to train the agent.",)
    parser.add_argument("--eval-interval", type=int, default=9000, help="Interval in timesteps between evaluations.",)
    parser.add_argument("--eval-n-runs", type=int, default=10, help="Number of episodes run for each evaluation.",)
    parser.add_argument("--render", action="store_true", help="Render env states in a GUI window.")
    parser.add_argument("--demo", action="store_true", help="Just run evaluation, not training.")
    parser.add_argument("--load-pretrained", action="store_true", default=False)
    parser.add_argument("--entropy-coefficient", type=float, default=0.2)
    parser.add_argument("--entropy-coefficient2", type=float, default=0.01)
    parser.add_argument("--entropy-coefficient3", type=float, default=0.001)
    parser.add_argument("--load", type=str, default="", help="Directory to load agent from.")
    parser.add_argument("--log-level", type=int, default=logging.INFO, help="Level of the root logger.")
    parser.add_argument("--monitor", action="store_true", help="Wrap env with gym.wrappers.Monitor.")
    parser.add_argument("--log-interval", type=int, default=1000, help="Interval in timesteps between outputting log messages during training",)
    parser.add_argument("--update-interval", type=int, default=2048, help="Interval in timesteps between model updates.",)
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to update model for per PPO iteration.",)
    parser.add_argument("--batch-size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--sample", type=int, default=100, help="Number of samples to train policy in each iteration.",)
    parser.add_argument("--N-new", type=int, default=100, help="Number of states generated by reverse move.",)
    parser.add_argument("--N-old", type=int, default=10, help="Number of states sampled by buffer to adress forgetting.",)
    parser.add_argument("--iter", type=int, default=20, help="Number of iteration for reverse curriculum.",)
    parser.add_argument("--T-rev", type=int, default=7, help="Number of reverse to generate states near start_s.",)
    parser.add_argument("--R-min", type=float, default=0.1, help="Lower threshold to decide whether to leave state.",)
    parser.add_argument("--R-max", type=float, default=0.7, help="Upper threshold to decide whether to leave state.",)
    parser.add_argument("--steps2", type=int, default=10 ** 6, help="Number of setps to learn agent after reverse curriculum.",)
    parser.add_argument("--checks", type=int, default=50, help="Number of trajectories to check entropy of policy.",)
    parser.add_argument("--experiment", type=int, default=0, help="The number of experiments.",)
    parser.add_argument("--learning-rate1", type=float, default=3e-4,)
    parser.add_argument("--learning-rate2", type=float, default=3e-4,)
    # parameter of DQN
    parser.add_argument("--env2", type=str, default="ttop-v2", help="OpenAI Gym MuJoCo env to perform algorithm on.",)
    parser.add_argument("--final-exploration-steps", type=int, default=10 ** 4)
    parser.add_argument("--noisy-net-sigma", type=float, default=None)
    parser.add_argument("--steps3", type=int, default=10 ** 6)
    parser.add_argument("--prioritized-replay", action="store_true")
    parser.add_argument("--replay-start-size", type=int, default=1000)
    parser.add_argument("--eval-n-runs2", type=int, default=10, help="Number of episodes run for each evaluation.",)
    parser.add_argument("--target-update-interval", type=int, default=10 ** 2)
    parser.add_argument("--target-update-method", type=str, default="hard")
    parser.add_argument("--soft-update-tau", type=float, default=1e-2)
    parser.add_argument("--update-interval2", type=int, default=1)
    parser.add_argument("--n-hidden-channels", type=int, default=100)
    parser.add_argument("--n-hidden-layers", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--minibatch-size", type=int, default=None)
    parser.add_argument("--render-train", action="store_true")
    parser.add_argument("--render-eval", action="store_true")
    parser.add_argument("--reward-scale-factor", type=float, default=1e-3)
    parser.add_argument(
        "--actor-learner",
        action="store_true",
        help="Enable asynchronous sampling with asynchronous actor(s)",
    )  # NOQA
    args = parser.parse_args()

    vv_env=gym.make(args.env)
    logging.basicConfig(level=args.log_level)

    # Set a random seed used in PFRL
    if args.is_random_seed:
        random.seed()
        args.seed = random.randint(0, 1000000)
        # print("args_seed:", args.seed)
    utils.set_random_seed(args.seed)

    # Set different random seeds for different subprocesses.
    process_seeds = np.arange(args.num_envs) + args.seed * args.num_envs
    assert process_seeds.max() < 2 ** 32
    args.outdir1 = experiments.prepare_output_dir(args, args.outdir1)

    def make_env(process_idx=0, env_id='ttop-v1', test=False):
        env = gym.make(env_id)
        # Use different random seeds for train and test envs
        # process_seed = int(process_seeds[process_idx])
        # env_seed = 2 ** 32 - 1 - process_seed if test else process_seed
        # env.seed(env_seed)
        # # Cast observations to float32 because our model uses float32
        # env = pfrl.wrappers.CastObservationToFloat32(env)
        # if args.monitor:
        #     env = pfrl.wrappers.Monitor(env, args.outdir)
        # if args.render:
        #     env = pfrl.wrappers.Render(env)
        return env

    def make_batch_env(test):
        return pfrl.envs.MultiprocessVectorEnv(
            [
                functools.partial(make_env, idx, test)
                for idx, env in enumerate(range(args.num_envs))
            ]
        )

    # Only for getting timesteps, and obs-action spaces
    sample_env = gym.make(args.env)
    timestep_limit = sample_env.spec.max_episode_steps
    obs_space = sample_env.observation_space
    action_space = sample_env.action_space
    print("Observation space:", obs_space)
    print("Action space:", action_space)

    # Normalize observations based on their empirical mean and variance
    obs_normalizer = pfrl.nn.EmpiricalNormalization(
        obs_space.low.size, clip_threshold=5
    )

    obs_size = obs_space.low.size
    action_num = action_space.n
    policy = torch.nn.Sequential(
        nn.Linear(obs_size, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, action_num),
        pfrl.policies.SoftmaxCategoricalHead(),
    )

    vf = torch.nn.Sequential(
        nn.Linear(obs_size, 64),
        nn.Tanh(),
        nn.Linear(64,64),
        nn.Tanh(),
        nn.Linear(64,1),
    )
    
    # While the original paper initialized weights by normal distribution,
    # we use orthogonal initialization as the latest openai/baselines does.
    def ortho_init(layer, gain):
        nn.init.orthogonal_(layer.weight, gain=gain)
        nn.init.zeros_(layer.bias)

    ortho_init(policy[0], gain=1)
    ortho_init(policy[2], gain=1)
    ortho_init(policy[4], gain=1e-2)
    ortho_init(vf[0], gain=1)
    ortho_init(vf[2], gain=1)
    ortho_init(vf[4], gain=1)

    # Combine a policy and a value function into a single model
    model = pfrl.nn.Branched(policy, vf)

    # opt = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-5)
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate1, eps=1e-5)

    agent = PPO(
        model,
        opt,
        obs_normalizer=obs_normalizer,
        gpu=args.gpu,
        update_interval=args.update_interval,
        minibatch_size=args.batch_size,
        epochs=args.epochs,
        clip_eps_vf=None,
        entropy_coef=args.entropy_coefficient,
        standardize_advantages=True,
        gamma=0.995,
        lambd=0.97,
    )

    env = make_env(env_id=args.env, test=False)
    eval_env = make_env(env_id=args.env, test=True)

    if args.load or args.load_pretrained:
        # either load or load_pretrained must be false
        assert not args.load or not args.load_pretrained
        if args.load:
            agent.load(args.load)
        else:
            agent.load(utils.download_model("PPO", args.env, model_type="final")[0])

    RCG = ReverseCurriculum(
        env=env,
        agent=agent,
        steps=args.steps,
        eval_n_steps=None,
        eval_n_episodes=args.eval_n_runs,
        eval_interval=args.eval_interval,
        log_interval=args.log_interval,
        outdir=args.outdir1,
        eval_env=eval_env,
        timestep_limit=timestep_limit,
        sample=args.sample,
        N_new=args.N_new,
        N_old=args.N_old,
        iteration=args.iter,
        T_rev=args.T_rev,
        Rmin=float(args.R_min),
        Rmax=float(args.R_max),
    )

    agent1 = RCG.reverse_curriculum()
    model = agent1.model

    print("RCG training is end")

    RCG.agent = PPO(
        model,
        opt,
        obs_normalizer=obs_normalizer,
        gpu=args.gpu,
        minibatch_size=args.batch_size,
        epochs=args.epochs,
        clip_eps_vf=None,
        entropy_coef=args.entropy_coefficient2,
        standardize_advantages=True,
        gamma=0.995,
        lambd=0.97,
    )
    s0_distrib = []
    for t in range(11):
        s0_distrib.append((t, 0, 0))
    agent2, r_list_1 = RCG.train_policy2(s0_distrib, args.steps2)

    model = agent2.model

    env = make_env(env_id=args.env2, test=False)
    eval_env = make_env(env_id=args.env2, test=True)
    args.outdir2 = experiments.prepare_output_dir(args, args.outdir2)

    RCG2 = ReverseCurriculum(
        env=env,
        agent=agent2,
        steps=args.steps,
        eval_n_steps=None,
        eval_n_episodes=args.eval_n_runs,
        eval_interval=args.eval_interval,
        log_interval=args.log_interval,
        outdir=args.outdir2,
        eval_env=eval_env,
        timestep_limit=timestep_limit,
        sample=args.sample,
        N_new=args.N_new,
        N_old=args.N_old,
        iteration=args.iter,
        T_rev=args.T_rev,
        Rmin=float(args.R_min),
        Rmax=float(args.R_max),
    )

    RCG2.agent = PPO(
        model,
        opt,
        obs_normalizer=obs_normalizer,
        gpu=args.gpu,
        minibatch_size=args.batch_size,
        epochs=args.epochs,
        clip_eps_vf=None,
        entropy_coef=args.entropy_coefficient3,
        standardize_advantages=True,
        gamma=0.995,
        lambd=0.97,
    )

    test_env = gym.make(args.env2)

    s0_distrib = []
    for t in range(11):
        s0_distrib.append((t, 0, 0))
    agent3, r_list_2 = RCG2.train_policy2(s0_distrib, args.steps3)

    # dirname saved in
    suffix = ""
    dirname1 = os.path.join(args.outdir1, "{}{}".format(t, suffix))
    dirname2 = os.path.join(args.outdir2, "{}{}".format(t, suffix))
    print("directory where model of phase1 is saved in:", dirname1)
    print("directory where model of phase2 is saved in:", dirname2)

    return agent3, test_env, args, r_list_1, r_list_2


def evaluation_1(agent, env, checks, experiment):
    """
    エージェントの評価を行う関数です。
    Parameters:
        agent (object): エージェントオブジェクト
        env (object): 環境オブジェクト
        checks (int): 評価回数
        experiment (str): 実験名
    Returns:
        各評価のトラジェクトリ（軌跡）のエネルギー合計
        ユニークなトラジェクトリの数
        ゴール状態に到達するユニークなトラジェクトリの数
    処理:
        'unique_trajectory'という名前のファイルに、ユニークなトラジェクトリのリストを書き込み
        'value_of_unique_trajectory'という名前のファイルに、ユニークなトラジェクトリのエネルギー、アクション、および加速度の値を書き込み
        'final_result'という名前のファイルに、ゴール状態に到達するユニークなトラジェクトリの数を書き込み
    """
    traject_memory = []
    energy_list = []
    action_list = []
    acceleration_list = []
    for i in range(checks):
        traject = []
        energy_sum = 0
        env.set_s0((8, 0, 0))
        state = env.reset()
        traject.append(tuple(state))
        done = False
        a_list = []
        accel_list = []
        while not done:
            state = [state]
            action = agent.batch_act(state)[0]
            state, r, done, infos = env.step(action)
            energy_sum += infos["energy"]
            a_list.append(infos["action"])
            accel_list.append(infos["accel"])
            traject.append(state)
        print("trajectory", i+1)
        print("\n energy_sum:", energy_sum, "[MJ]")
        trajectory = []
        for norm_s in traject:
            state = tuple(env.inverse_norm_state(norm_s))
            trajectory.append(state)
        energy_list.append(energy_sum)
        traject_memory.append(tuple(trajectory))
        action_list.append(tuple(a_list))
        acceleration_list.append(tuple(accel_list))
    c = collections.Counter(traject_memory)
    print("numbers of unique trajectory:", len(c))
    count = 0
    dic = {}
    output_tra = []
    output_ene = []
    output_action = []
    output_accel = []
    for key in c.keys():
        if key[len(key)-1][2]==0 and abs(key[len(key)-1][1]-env.distance_between_stations)<=env.allowable_error_of_position:
            count += c[key]
            dic[key] = c[key]
            output_ene.append(energy_list[traject_memory.index(key)])
            output_action.append(action_list[traject_memory.index(key)])
            output_accel.append(acceleration_list[traject_memory.index(key)])
            output_tra.append(list(key))
    with open('unique_trajectory%s' % (experiment), 'w') as f:
        print(output_tra, file=f)
    with open('value_of_unique_trajectory%s' % (experiment), 'w') as f2:
        print("energy:\n", output_ene, file=f2)
        print("\n\n\n action:\n", output_action, file=f2)
        print("\n\n\n accel:\n", output_accel, file=f2)
    print("numbers of trajectory reaching goal states:", count)
    count = 0
    for key in dic:
        count += 1
    with open('final_result%s_1' % (experiment), 'w') as f3:
        print("number of unique trajectory reaching goal state:", count, file=f3)


def evaluation_2(agent, env, experiment, reward_log_1, reward_log_2):
    """
    エージェントの評価を行う関数です。
    Parameters:
        agent (object): エージェントオブジェクト
        env (object): 環境オブジェクト
        experiment (str): 実験名
        reward_log_1 (list): エージェント1の報酬のリスト
        reward_log_2 (list): エージェント2の報酬のリスト
    Returns:
        None
    処理:
        'best_information'という名前のファイルに、最もエネルギー消費量が少ないトラジェクトリを書き込み
        'some_initial_state'という名前のファイルに、いくつかの初期状態からのトラジェクトリを書き込み
        'final_result'という名前のファイルに、エージェント2の報酬のリストを書き込み
    """
    agent.training = False
    agent.act_deterministically = True
    traject = []
    energy_sum = 0
    state = env.reset()
    traject.append(tuple(env.inverse_norm_state(state)))
    done = False
    action_sequence = []
    accel_sequence = []
    action_distrib_list = []
    while not done:
        print("\n state:", env.inverse_norm_state(state))
        state = [state]
        b_state = agent.batch_states(state, agent.device, agent.phi)
        distrib = agent.model(b_state)[0]
        action_prob_distrib = distrib.probs[0].tolist()
        action_distrib_list.append(action_prob_distrib)
        print("action_distribution:", action_prob_distrib)
        action = agent.batch_act(state)
        state, r, done, infos = env.step(action)
        energy_sum += infos["energy"]
        action_sequence.append(infos["action"])
        accel_sequence.append(infos["accel"])
        traject.append(state)
    trajectory = []
    for norm_s in traject:
        state = tuple(env.inverse_norm_state(norm_s))
        trajectory.append(state)
    with open('best_information%s.txt' % (experiment), 'w') as f1:
        print("energy consumption:\n", energy_sum, "[MJ]", file=f1)
        print("\n trajectory:\n", trajectory, file=f1)
        print("\n action probavility:\n", action_distrib_list, file=f1)
        print("\n action sequence:\n", action_sequence, file=f1)
        print("\n accel_sequence:\n", accel_sequence, file=f1)
    
    trajectory_list = []
    action_sequence_list = []
    accel_sequence_list = []
    distrib_sequence_list = []
    energy_list = []
    traject = []
    energy_sum = 0
    # time variation
    S0 = [(1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0), (5, 0, 0), (6, 0, 0), (7, 0, 0), (8, 0, 0), (9, 0, 0), (10, 0, 0)]
    # position variation
    # S0 = [(0, -1, 0), (0, -0.75, 0), (0, -0.5, 0), (0, -0.25, 0), (0, 0, 0), (0, 0.25, 0), (0, 0.5, 0), (0, 0.75, 0), (0, 1, 0)]
    # both time and position
    # S0 = [(1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0), (5, 0, 0), (6, 0, 0), (7, 0, 0), (8, 0, 0), (9, 0, 0), (10, 0, 0)]
    # S0 += [(0, -1, 0), (0, -0.75, 0), (0, -0.5, 0), (0, -0.25, 0), (0, 0, 0), (0, 0.25, 0), (0, 0.5, 0), (0, 0.75, 0), (0, 1, 0)]
    for s0 in S0:
        traject = []
        energy_sum = 0
        env.set_s0(s0)
        state = env.reset2()
        traject.append(state)
        done = False
        action_sequence = []
        accel_sequence = []
        distrib_sequence = []
        while not done:
            print("\n state:", env.inverse_norm_state(state))
            state = [state]
            b_state = agent.batch_states(state, agent.device, agent.phi)
            distrib = agent.model(b_state)[0]
            action_prob_distrib = distrib.probs[0].tolist()
            distrib_sequence.append(action_prob_distrib)
            print("action_distribution:", action_prob_distrib)
            # action = action_prob_distrib.index(max(action_prob_distrib))
            action = agent.batch_act(state)
            state, r, done, infos = env.step(action)
            energy_sum += infos["energy"]
            action_sequence.append(infos["action"])
            accel_sequence.append(infos["accel"])
            traject.append(state)
        trajectory = []
        for norm_s in traject:
            state = tuple(env.inverse_norm_state(norm_s))
            trajectory.append(state)
        distrib_sequence_list.append(distrib_sequence)
        trajectory_list.append(trajectory)
        action_sequence_list.append(action_sequence)
        accel_sequence_list.append(accel_sequence)
        energy_list.append(energy_sum)
        with open('some_initial_state%s.txt' % (experiment), 'w') as f2:
            for i in range(len(trajectory_list)):
                print("\n result%s:\n" % (i+1), file=f2)
                print("energy consumption:\n", energy_list[i], "[MJ]", file=f2)
                print("\n trajectory:\n", trajectory_list[i], file=f2)
                print("\n action probavility:\n", distrib_sequence_list[i], file=f2)
                print("\n action sequence:\n", action_sequence_list[i], file=f2)
                print("\n accel_sequence:\n", accel_sequence_list[i], file=f2)
    
    agent.act_deterministically = False
    trajectory_list = []
    action_sequence_list = []
    accel_sequence_list = []
    distrib_sequence_list = []
    energy_list = []
    for i in range(1):
        traject = []
        energy_sum = 0
        state = env.reset()
        traject.append(tuple(env.inverse_norm_state(state)))
        done = False
        action_sequence = []
        accel_sequence = []
        distrib_sequence = []
        while not done:
            state = [state]
            b_state = agent.batch_states(state, agent.device, agent.phi)
            distrib = agent.model(b_state)[0]
            action_prob_distrib = distrib.probs[0].tolist()
            action = agent.batch_act(state)[0]
            state, r, done, infos = env.step(action)
            energy_sum += infos["energy"]
            distrib_sequence.append(action_prob_distrib)
            action_sequence.append(infos["action"])
            accel_sequence.append(infos["accel"])
            traject.append(state)
        trajectory = []
        for norm_s in traject:
            state = tuple(env.inverse_norm_state(norm_s))
            trajectory.append(state)
        distrib_sequence_list.append(distrib_sequence)
        trajectory_list.append(trajectory)
        action_sequence_list.append(action_sequence)
        accel_sequence_list.append(accel_sequence)
        energy_list.append(energy_sum)
    with open('final_result%s.txt' % (experiment), 'w') as f3:
        for i in range(len(trajectory_list)):
            print("\n result%s:\n" % (i+1), file=f3)
            print("energy consumption:\n", energy_list[i], "[MJ]", file=f3)
            print("trajectory:\n", trajectory_list[i], file=f3)
            print("action probavility:\n", distrib_sequence_list[i], file=f3)
            print("action sequence:\n", action_sequence_list[i], file=f3)
            print("accel_sequence:\n", accel_sequence_list[i], file=f3)
        print("\n reward_log_1:\n", reward_log_1, file=f3)
        print("\n reward_log_2:\n", reward_log_2, file=f3)


def draw_trajectory(trajectory, k, experiment):
    t_list = []
    x_list = []
    v_list = []
    for i in range(len(trajectory)):
        t_list.append(trajectory[i][0])
        x_list.append(trajectory[i][1])
        v_list.append(trajectory[i][2])
    fig = plt.figure()
    # Create graph2
    ax1 = fig.add_subplot(111)
    ln1 = ax1.plot(t_list, x_list, 'C0', label=r'$Position$')
    # Create graph1
    ax2 = ax1.twinx()
    ln2 = ax2.plot(t_list, v_list, 'C1', label=r'$Velocity$')
    # graph1 + graph2
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc='lower right')
    # Draw graph
    ax1.set_xlabel('Time[sec]')
    ax1.set_ylabel(r'$Position[m]$')
    ax1.grid(True)
    ax2.set_ylabel(r'$Velocity[m/sec]$')
    plt.savefig("result%s_%s.jpg" % (experiment, k))
    if k==1:
        print("\n \n \n experiment%s \n \n \n " % (experiment))


def show_reward(R_list_1, R_list_2, experiment):
    y_1 = []
    y_2 = []
    x_1 = []
    x_2 = []
    split = 100
    for i in range(int(len(R_list_1)/split)):
        x_1.append(i)
        r = []
        for j in range(split):
            r.append(R_list_1[split*i+j])
        y_1.append(np.mean(r))
    for i in range(int(len(R_list_2)/split)):
        x_2.append(i)
        r = []
        for j in range(split):
            r.append(R_list_2[split*i+j])
        y_2.append(np.mean(r))
    x_1 = np.array(x_1)
    x_2 = np.array(x_2)
    y_1 = np.array(y_1)
    y_2 = np.array(y_2)
    # draw figure
    plt.plot(x_1, y_1)
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.savefig("reward_log%s_1" % (experiment))
    plt.clf()
    plt.plot(x_2, y_2)
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.savefig("reward_log%s_2" % (experiment))


if __name__=='__main__':
    # 実験の開始時間
    begin = time.time()
    # 二段階のDRL
    agent, env2, args, r_list_1, r_list_2 = main()
    # パラメータ
    # checks = args.checks
    experiment = args.experiment
    # 評価1
    # evaluation_1(agent, env2, checks, experiment)
    # 報酬曲線を描画
    show_reward(r_list_1, r_list_2, experiment)
    # 評価2
    # evaluation_2(agent, env2, experiment, r_list_1, r_list_2)
    # 実行時間を出力
    print("\n running_time:", int(time.time() - begin), "[sec] \n")
    print("\nargs_seed:", args.seed)
