import numpy as np
from time_env import TTOPEnv


class DPAgent:

    def __init__(self, env):
        self.env = env
        self.gamma = 0.98

        self.time_steps = env.planning_time + 1
        self.position_resolution = 20.0
        self.velocity_resolution = 1.0

        self.n_positions = int(env.distance_between_stations / self.position_resolution) + 1
        self.n_velocities = int(env.velocity_limit / self.velocity_resolution) + 1

        self.V = np.full((self.time_steps, self.n_positions, self.n_velocities), -np.inf)
        self.policy = np.zeros((self.time_steps, self.n_positions, self.n_velocities), dtype=int)

    def discretize_state(self, state):
        time, position, velocity = state
        t_idx = max(0, min(int(round(time)), self.time_steps - 1))
        x_idx = max(0, min(int(round(position / self.position_resolution)), self.n_positions - 1))
        v_idx = max(0, min(int(round((velocity * 3.6) / self.velocity_resolution)), self.n_velocities - 1))
        return t_idx, x_idx, v_idx

    def continuous_state(self, t_idx, x_idx, v_idx):
        return (float(t_idx), float(x_idx * self.position_resolution), float(v_idx * self.velocity_resolution / 3.6))

    def is_goal_state(self, state):
        time, position, velocity = state
        if time >= self.env.planning_time:
            position_ok = abs(position - self.env.distance_between_stations) <= self.env.allowable_error_of_position
            velocity_ok = abs(velocity) < 0.1
            return position_ok and velocity_ok
        return False

    def solve(self):
        T = self.time_steps - 1

        for x_idx in range(self.n_positions):
            for v_idx in range(self.n_velocities):
                state = self.continuous_state(T, x_idx, v_idx)
                self.V[T][x_idx][v_idx] = 1.0 if self.is_goal_state(state) else -10.0

        for t in range(T - 1, -1, -1):
            for x_idx in range(self.n_positions):
                for v_idx in range(self.n_velocities):
                    state = self.continuous_state(t, x_idx, v_idx)
                    best_value = -np.inf
                    best_action = 0

                    for action in range(self.env.action_space.n):
                        if v_idx == 0 and action == 0:
                            continue

                        self.env.state = state
                        try:
                            _, reward, done, info = self.env.step(action)
                            next_state = tuple(self.env.state)
                            next_t, next_x, next_v = self.discretize_state(next_state)

                            if next_t >= self.time_steps:
                                continue

                            energy_penalty = -0.001 * info["energy"]

                            if done and next_t >= T:
                                value = reward if self.is_goal_state(next_state) else energy_penalty - 10.0
                            else:
                                next_value = self.V[next_t][next_x][next_v]
                                if next_value == -np.inf:
                                    continue
                                value = energy_penalty + self.gamma * next_value

                            if value > best_value:
                                best_value = value
                                best_action = action
                        except:
                            continue

                    if best_value > -np.inf:
                        self.V[t][x_idx][v_idx] = best_value
                        self.policy[t][x_idx][v_idx] = best_action

    def get_action(self, state):
        t_idx, x_idx, v_idx = self.discretize_state(state)
        if t_idx >= self.time_steps or t_idx < 0:
            return 1
        action = self.policy[t_idx][x_idx][v_idx]
        if state[2] == 0 and action == 0:
            action = 1
        return action


def train():
    env = TTOPEnv()
    agent = DPAgent(env)
    agent.solve()
    return agent


if __name__ == "__main__":
    train()
