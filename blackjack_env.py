from typing import Optional
import gymnasium as gym
import numpy as np
class BlackjackEnv(gym.Env):

    def __init__(self):
        self.max_hand_size = 11
        self.action_space = gym.spaces.Discrete(2)
        self.my_hand_ = [np.random.randint(1, 12), np.random.randint(1,12)]
        self.dealer_hand_ = [np.random.randint(1, 12), np.random.randint(1,12)]

        self.observation_space = gym.spaces.Dict(
            {
                "my_hand": gym.spaces.Box(0, 11, shape=(self.max_hand_size,), dtype=int),
                "dealer_hand": gym.spaces.Box(0, 11, shape=(self.max_hand_size,), dtype=int)
            }
        )

    def _get_obs(self):
        def pad_hand(hand):
            padded = np.zeros(self.max_hand_size, dtype=int)
            padded[:len(hand)] = hand
            return padded

        return {
            "my_hand": pad_hand(self.my_hand_),
            "dealer_hand": pad_hand(self.dealer_hand_),
        }
    
    
    def _get_info(self):
        total = 0
        for card in self.my_hand_:
            total += card

        return {
            "distance_from_21": 21 - total
        }
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.my_hand_ = [np.random.randint(1, 12), np.random.randint(1,12)]
        self.dealer_hand_ = [np.random.randint(1, 12), np.random.randint(1,12)]
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        if action: #Its 1, aka hit
            new_card = np.random.randint(1,11)
            self.my_hand_.append(new_card)
            hand_total = 0
            for card in self.my_hand_:
                hand_total += card
            if hand_total > 21:
                reward = -1
                terminated = True
            else:
                reward = 0
                terminated = False

        else:
            dealer_hand_total = 0
            for card in self.dealer_hand_:
                dealer_hand_total += card
            while dealer_hand_total <= 16:
                new_card = np.random.randint(1,11)
                self.dealer_hand_.append(new_card)
                dealer_hand_total += new_card
            
            hand_total = 0
            for card in self.my_hand_:
                hand_total += card

            if dealer_hand_total > hand_total and dealer_hand_total <= 21:
                reward = -1
            elif dealer_hand_total == hand_total:
                reward = 0
            else:
                reward = 1
            terminated = True
        truncated = False
        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, truncated, info
    