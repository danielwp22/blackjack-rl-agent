import gymnasium as gym
from collections import Counter
from gymnasium.envs.registration import register
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import torch
from torch import nn
import torch.nn.functional as F
import math


register(
    id="CustomBlackjack-v0",
    entry_point="blackjack_deepq_env:DeepQBlackjackEnv",
)
env = gym.make("CustomBlackjack-v0")

class DQN(nn.Module):
    def __init__(self, in_states, h1_nodes, h2_nodes, out_actions):
        super().__init__()
        self.fc1 = nn.Linear(in_states, h1_nodes)
        self.fc2 = nn.Linear(h1_nodes, h2_nodes)
        self.out = nn.Linear(h2_nodes, out_actions)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return(x)

episodes = 10000
q_state = Counter()
discount = 0.5
learning_rate = 0.001
episode_numbers = []
episode_rewards = []
policy_iterations = 5
epsilon_start, epsilon_end = 1.0, 0.1
epsilon_decay = 10000   # decay over first 10 000 steps
test_win_list = []
test_draw_list = []
test_loss_list = []

#Initialize deep q networks
policy_dqn = DQN(4,10,10,2)
target_dqn = DQN(4,10,10,2)
target_dqn.load_state_dict(policy_dqn.state_dict())

episode_numbers_plot = []
winrates = []

#Test model
def test_model():
    test_episodes = 1000
    test_wins = 0
    test_draws = 0
    test_losses = 0
    for episode in range(test_episodes):
        observation, info = env.reset()
        print(f"Starting observation {observation}")
        episode_over = False
        total_reward = 0
        next_action = 0
        while not episode_over:
            my_hand = observation["my_hand"]
            dealer_hand = observation["dealer_hand"]
            my_hand_has_ace = observation["my_hand_has_ace"]
            dealer_hand_has_ace = observation["dealer_hand_has_ace"]

            my_hand_total = 0
            for card in my_hand:
                my_hand_total += card
            dealer_hand_total = 0
            for card in dealer_hand:
                dealer_hand_total += card

            input_state = [my_hand_total, dealer_hand_total, int(my_hand_has_ace), int(dealer_hand_has_ace)]
            input_tensor = torch.tensor(input_state, dtype=torch.float32).unsqueeze(0) 

            #Next action to take
            q_values = policy_dqn(input_tensor)
            action_values = q_values.squeeze()
            if action_values[0] == action_values[1]:
                action = np.random.randint(2)
            else:
                action = torch.argmax(action_values)
                print(f"taking action {action}")
            
            #Take a step and update values
            observation, reward, terminated, truncated, info = env.step(action)

            #Update reward + finish if needed
            total_reward += reward
            if terminated:
                if total_reward == -1:
                    test_losses +=1
                elif total_reward == 0:
                    test_draws +=1
                else:
                    test_wins +=1
            episode_over = terminated or truncated

        print(f"Episode finished! Total reward: {total_reward}")

    print(f"Done training, steps: {episodes}")

    print(f"Test wins: {test_wins}")
    print(f"Test draws: {test_draws}")
    print(f"Test losses: {test_losses}")

    winrate = test_wins/test_episodes*100
    print(f"test winrate: {winrate}")
    return winrate, test_wins, test_draws, test_losses

optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=learning_rate)

#Train the model
for episode in range(episodes):
    observation, info = env.reset()
    print(f"Starting observation {observation}")
    episode_over = False
    total_reward = 0
    next_action = 0
    total_policy_iterations = 0
    if len(episode_numbers) % 1000 == 0:
        winrate, test_wins, test_draws, test_losses = test_model()
        winrates.append(winrate)
        test_win_list.append(test_wins)
        test_draw_list.append(test_draws)
        test_loss_list.append(test_losses)
        episode_numbers_plot.append(episode)
    while not episode_over:
        my_hand = observation["my_hand"]
        dealer_hand = observation["dealer_hand"]
        my_hand_has_ace = observation["my_hand_has_ace"]
        dealer_hand_has_ace = observation["dealer_hand_has_ace"]

        my_hand_total = 0
        for card in my_hand:
            my_hand_total += card
        dealer_hand_total = 0
        for card in dealer_hand:
            dealer_hand_total += card

        input_state = [my_hand_total, dealer_hand_total, int(my_hand_has_ace), int(dealer_hand_has_ace)]
        input_tensor = torch.tensor(input_state, dtype=torch.float32).unsqueeze(0) 

        #Next action to take
        q_values = policy_dqn(input_tensor)
        action_values = q_values.squeeze()
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-episode / epsilon_decay)
        random_action = np.random.choice([0, 1], p=[1-epsilon, epsilon])

        if random_action:
            action = np.random.randint(2)
        elif action_values[0] == action_values[1]:
            action = np.random.randint(2)
        else:
            action = torch.argmax(action_values)
            print(f"taking action {action}")
        
        #Take a step and update values
        observation, reward, terminated, truncated, info = env.step(action)
        my_hand_new = observation["my_hand"]
        dealer_hand_new = observation["dealer_hand"]
        my_hand_has_ace_new = observation["my_hand_has_ace"]
        dealer_hand_has_ace_new = observation["dealer_hand_has_ace"]  
        
        my_hand_total_new = 0
        for card in my_hand_new:
            my_hand_total_new += card
        dealer_hand_total_new = 0
        for card in dealer_hand_new:
            dealer_hand_total_new += card

        #Target calculation
        if terminated:
            target = reward
        else:
            new_input_state = [my_hand_total_new, dealer_hand_total_new, int(my_hand_has_ace_new), int(dealer_hand_has_ace_new)]
            new_input_tensor = torch.tensor(new_input_state, dtype=torch.float32).unsqueeze(0) 
            target_q_values = target_dqn(new_input_tensor)
            action_values = target_q_values.squeeze().tolist()
            max_action_value = max(action_values)
            target = reward + discount * max_action_value

        target_tensor = torch.tensor([target], dtype=torch.float32)

        #Calculate loss + gradient descent for action
        q_value_for_action = q_values[0,action]
        loss = F.mse_loss(q_value_for_action.unsqueeze(0), target_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #Check if we need to update target_dqn
        total_policy_iterations += 1

        if total_policy_iterations % policy_iterations == 0:
            target_dqn.load_state_dict(policy_dqn.state_dict())

        #Update reward + finish if needed
        total_reward += reward
        episode_over = terminated or truncated

    print(f"Episode finished! Total reward: {total_reward}")
    episode_numbers.append(episode)
    episode_rewards.append(total_reward)

print(f"state dict: {policy_dqn.state_dict()}" )
print(f"test win list: {test_win_list}")
print(f"test loss list: {test_loss_list}")
print(f"test draw list: {test_draw_list}")
print(f"winrates: {winrates}")
print(f"episodes:{episode_numbers_plot}")

plt.plot(episode_numbers_plot, winrates)
plt.xlabel("episode number")
plt.ylabel("winrate")
plt.show()


