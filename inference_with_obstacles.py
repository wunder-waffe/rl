import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import torch
from minimized_steps import DynamicRoutePlanningEnv

# Function to visualize agent's path, goal, and obstacles in 3D
def visualize_path(agent_positions, goal_position, obstacle_positions, grid_size, slices):
    print(agent_positions[-1])
    print(goal_position)
    if (agent_positions[-1]==goal_position).all():
        print("successful nav")
    else:
        print("crash")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([0, grid_size])
    ax.set_ylim([0, grid_size])
    ax.set_zlim([0, slices])

    # Plot agent path
    agent_positions = np.array(agent_positions)
    ax.plot(agent_positions[:, 1], agent_positions[:, 2], agent_positions[:, 0], 
            label="Agent Path", color="blue", marker="o")

    # Plot goal position, unpacking coordinates correctly
    ax.scatter(goal_position[1], goal_position[2], goal_position[0], 
               color="red", s=100, label="Goal")

    # Plot obstacles
    obstacle_positions = np.array(obstacle_positions)
    if obstacle_positions.size > 0:  # Check if there are obstacles
        ax.scatter(obstacle_positions[:, 1], obstacle_positions[:, 2], obstacle_positions[:, 0], 
                   color="black", s=20, label="Obstacles")

    # Labels and legend
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Slice (Altitude)")
    ax.legend()
    plt.show()

# Load the environment and model
env = DynamicRoutePlanningEnv(slices=3, grid_size=20, max_steps=100, device="cuda")
model = PPO.load("trained_model")

# Identify obstacle positions in the environment
obstacle_positions = []
state, _ = env.reset()
for z in range(env.slices):
    for x in range(env.grid_size):
        for y in range(env.grid_size):
            if env.state[z, x, y] == 1:
                obstacle_positions.append((z, x, y))

# Run inference
agent_positions = []
obs, _ = env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, truncated, _ = env.step(action)
    agent_positions.append(env.agent_pos.cpu().numpy())
    if truncated:
        break

# Visualize the path taken by the agent along with obstacles and goal
visualize_path(agent_positions, env.goal_pos.cpu().numpy(), obstacle_positions, env.grid_size, env.slices)
