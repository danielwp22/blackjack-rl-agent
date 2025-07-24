## Blackjack Solver using Deep Q-Learning

This project implements both **Q-learning** and **Deep Q-Learning (DQN)** algorithms to train agents to play Blackjack using environments from **Gymnasium** (OpenAI's Gym fork). The goal is to develop an agent that can beat the house and consistently win games using reinforcement learning.

## 🧠 Summary

- The **basic Q-learning agent** achieved around **40% win rate**, significantly outperforming a random player (~20%).
- The **Deep Q-Learning agent**, implemented with PyTorch, was able to **win more than it lost by about 1%**, theoretically enabling it to **profit** over time.

![Learning Curve](https://i.imgur.com/v0Q7fVO.png)

## 🛠️ Technologies

- **Python**
- **PyTorch**
- **Gymnasium (Blackjack-v1, Custom Envs)**
- **Matplotlib** (for visualization)

## 🚀 Key Features

- Tabular Q-learning implementation with epsilon-greedy exploration
- Deep Q-Learning (DQN) with fully connected neural network
- Custom Gymnasium environments for advanced training scenarios
- Learning curve visualizations for performance tracking

## 🧪 Planned Improvements

- 🔄 Add batch training for more stable gradient descent updates  
- 🃏 Remove dealer ace as a neural network input (to reduce state space)  
- 🧮 Enable card counting by giving the agent access to the deck state  

