from collections import deque
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from event_env import TTOPEnv


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def __len__(self):
        return len(self.buffer)

    def get_batch(self):
        data = random.sample(self.buffer, self.batch_size)
        state = torch.tensor(np.stack([x[0] for x in data]))
        action = torch.tensor(np.array([x[1] for x in data]), dtype=torch.long)
        reward = torch.tensor(np.array([x[2] for x in data]).astype(np.float32))
        next_state = torch.tensor(np.stack([x[3] for x in data]))
        done = torch.tensor(np.array([x[4] for x in data]).astype(np.int32))
        return state, action, reward, next_state, done


class QNet(nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.l1 = nn.Linear(3, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)


class DQNAgent:
    def __init__(self):
        self.gamma = 0.99
        self.lr = 0.0005
        self.epsilon = 0.1
        self.buffer_size = 1000
        self.batch_size = 32
        self.action_size = 80
        self.time_step = 2

        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.qnet = QNet(self.action_size)
        self.qnet_target = QNet(self.action_size)
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=self.lr)
        self.env = TTOPEnv()

    def get_action(self, state, train=True):
        if np.random.rand() < self.epsilon and train:
            return np.random.choice(self.action_size)
        else:
            state = torch.tensor(state[np.newaxis, :])
            qs = self.qnet(state)
            return qs.argmax().item()

    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.batch_size:
            return None

        state, action, reward, next_state, done = self.replay_buffer.get_batch()
        qs = self.qnet(state)
        q = qs[np.arange(len(action)), action]

        next_qs = self.qnet_target(next_state)
        next_q = next_qs.max(1)[0]

        _, a = self.env.convert_action(action)
        next_q.detach()
        target = reward + (1 - done) * (self.gamma ** (a / self.time_step)) * next_q

        loss = nn.MSELoss()(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def sync_qnet(self):
        self.qnet_target.load_state_dict(self.qnet.state_dict())


def train(episodes=10000, sync_interval=20):
    env = TTOPEnv()
    agent = DQNAgent()

    for episode in range(episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.get_action(state)
            action1, action2 = env.convert_action(action)
            current_time = env.inverse_norm_state(state)[0]
            reward = 0

            while env.inverse_norm_state(state)[0] < current_time + action2:
                next_state, r, done, _ = env.step(current_time, action1, action2)
                reward += r
                total_reward += r

                if env.inverse_norm_state(next_state)[0] >= current_time + action2:
                    agent.update(state, action, reward, next_state, done)

                state = next_state

        if episode % sync_interval == 0:
            agent.sync_qnet()

        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward:.4f}")

    return agent


if __name__ == "__main__":
    train()
