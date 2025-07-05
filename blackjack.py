import gymnasium as gym
from collections import Counter
from gymnasium.envs.registration import register
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate


register(
    id="CustomBlackjack-v0",
    entry_point="blackjack_env:BlackjackEnv",
)
env = gym.make("CustomBlackjack-v0")




episodes = 100
q_state = Counter()
alpha = 0.5
learning_rate = 1.0
episode_numbers = []
episode_rewards = []


for episode in range(episodes):
    observation, info = env.reset()
    print(f"Starting observation {observation}")
    episode_over = False
    total_reward = 0
    while not episode_over:
        my_hand = observation["my_hand"]
        dealer_hand = observation["dealer_hand"]

        my_hand_total = 0
        for card in my_hand:
            my_hand_total += card
        dealer_hand_total = 0
        for card in dealer_hand:
            dealer_hand_total += card

        if  q_state[(my_hand_total, 0)] > q_state[(my_hand_total, 1)]:
            action = 0
        elif q_state[(my_hand_total, 0)] < q_state[(my_hand_total, 1)]:
            action = 1
        else: 
            action = np.random.randint(0,1)

        observation, reward, terminated, truncated, info = env.step(action)

        new_hand = observation["my_hand"]
        new_hand_total = 0
        for card in new_hand:
            new_hand_total += card
        
        if  q_state[(new_hand_total, 0)] > q_state[(new_hand_total, 1)]:
            new_action = 0
        elif q_state[(new_hand_total, 0)] < q_state[(new_hand_total, 1)]:
            new_action = 1
        else: 
            new_action = np.random.randint(0,1)

        sample = reward + learning_rate*q_state[(new_hand_total, new_action)]

        q_state[(my_hand_total, action)] = (1-alpha)*q_state[(my_hand_total, action)] + alpha*sample    

        total_reward += reward
        episode_over = terminated or truncated
    print(f"Episode finished! Total reward: {total_reward}")
    episode_numbers.append(episode)
    episode_rewards.append(total_reward)


print(f"Done training, steps: {episodes}")
print(f"heres the q state:{q_state}")

data = []

for key,value in q_state.items():
    state = int(key[0])
    if key[1] == 0:
        action = "stand"
    else:
        action = "hit"
    reward = value
    data.append([state,action,reward])
data.sort()
data.insert(0,["State", "Action", "Reward"])

print(tabulate(data, headers="firstrow", tablefmt="grid"))

    

print(f"episode numbers: {episode_numbers}, episode rewards{episode_rewards}")
plt.plot(episode_numbers, episode_rewards)
plt.xlabel("Episode number")
plt.ylabel("Episode reward")
plt.title("Reward over episodes")
plt.grid(True)
plt.show()
