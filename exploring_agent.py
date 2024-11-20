import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch
from torch import tensor
from scipy.spatial import distance
import matplotlib.pyplot as plt

# Environment definition
class DynamicRoutePlanningEnv(gym.Env):
    def __init__(self, slices=10, grid_size=20, max_steps=100, device="cuda"):
        super(DynamicRoutePlanningEnv, self).__init__()
        self.slices = slices
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.device = device
        self.agent_pos = torch.tensor([0, 10, 0], device=device)
        self.goal_pos = torch.tensor([slices - 1, grid_size - 1, grid_size - 1], device=device)
        self.observation_space = gym.spaces.MultiBinary((slices, grid_size, grid_size))
        self.action_space = gym.spaces.Discrete(6)
        self.visited_positions = set()

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        total_elements = self.slices * self.grid_size * self.grid_size
        max_ones = int(total_elements * 0.03)
        state = torch.zeros((self.slices, self.grid_size, self.grid_size), dtype=torch.int, device=self.device)
        one_indices = np.random.choice(total_elements, max_ones, replace=False)
        state.view(-1)[torch.tensor(one_indices, device=self.device)] = 1
        state[0, 10, 0] = 0
        self.state = state
        self.agent_pos = torch.tensor([0, 10, 0], device=self.device)
        self.step_count = 0
        self.visited_positions = set([tuple(self.agent_pos.tolist())])
        return self.state.cpu().numpy(), {}

    def step(self, action):
        slice_index, x, y = self.agent_pos.tolist()

        # Move based on action
        if action == 0:
            slice_index = max(0, slice_index - 1)
        elif action == 1:
            slice_index = min(self.slices - 1, slice_index + 1)
        elif action == 2:
            x = max(0, x - 1)
        elif action == 3:
            x = min(self.grid_size - 1, x + 1)
        elif action == 4:
            y = max(0, y - 1)
        elif action == 5:
            y = min(self.grid_size - 1, y + 1)

        new_pos = torch.tensor([slice_index, x, y], device=self.device)
        reward = self.calculate_reward(new_pos)
        self.agent_pos = new_pos
        done = self.check_done_condition()
        self.step_count += 1
        truncated = self.step_count >= self.max_steps
        self.visited_positions.add(tuple(self.agent_pos.tolist()))

        return self.state.cpu().numpy(), reward, done, truncated, {}

    def calculate_reward(self, new_pos):
        reward = 0  # Remove general step penalty

        # Goal-related rewards
        goal_distance = distance.euclidean(new_pos.cpu().numpy(), self.goal_pos.cpu().numpy())
        prev_distance = distance.euclidean(self.agent_pos.cpu().numpy(), self.goal_pos.cpu().numpy())
        
        if self.state[new_pos[0], new_pos[1], new_pos[2]] == 1:
            # Penalty for colliding with an obstacle
            reward -= 200
        elif torch.equal(new_pos, self.goal_pos):
            # Reward for reaching the goal
            reward += 100
        else:
            # Proximity reward - increases as agent gets closer to goal
            proximity_reward = max(0, 100 - goal_distance)
            reward += proximity_reward / 10

            # Direction reward: positive if closer to goal, negative if further
            reward += 5 if goal_distance < prev_distance else -1

            # Exploration bonus: reward for moving to unvisited cells
            if tuple(new_pos.tolist()) not in self.visited_positions:
                reward += 0.5
            else:
                reward -= 2  # Penalty for backtracking

        return reward

    def check_done_condition(self):
        if torch.equal(self.agent_pos, self.goal_pos):
            return True
        if self.state[self.agent_pos[0], self.agent_pos[1], self.agent_pos[2]] == 1:
            return True
        return False

def make_env():
    return DynamicRoutePlanningEnv(slices=10, grid_size=60, max_steps=300, device="cuda")

# Train the agent
def train_agent(max_iterations=1200, n_envs=12):
    envs = SubprocVecEnv([make_env for _ in range(n_envs)])  # Create parallel environments
    model = PPO("MlpPolicy", envs, verbose=1, device="cuda", n_steps=256, batch_size=64 * n_envs, gamma=0.95)
    
    model_save_path = "trained_model_60"
    try:
        model.learn(total_timesteps=max_iterations * 512)
        model.save(model_save_path)
        print(f"Model saved to {model_save_path}.")
    except KeyboardInterrupt:
        print("\nTraining interrupted! Saving model...")
        model.save(model_save_path)
        print(f"Model saved to {model_save_path}.")

# Visualization function
def visualize_path(agent_positions, goal_position, grid_size):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([0, grid_size])
    ax.set_ylim([0, grid_size])
    ax.set_zlim([0, len(agent_positions)])

    # Plot agent path
    agent_positions = np.array(agent_positions)
    ax.plot(agent_positions[:, 1], agent_positions[:, 2], agent_positions[:, 0], label="Agent Path", color="blue", marker="o")

    # Plot goal position
    ax.scatter(*goal_position, color="red", s=100, label="Goal")

    # Labels and legend
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Slice (Altitude)")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    train_agent(max_iterations=3600, n_envs=24)
