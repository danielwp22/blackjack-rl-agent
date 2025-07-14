from typing import Optional
import gymnasium as gym
import numpy as np
import random

class DeepQBlackjackEnv(gym.Env):

    def __init__(self):
        self.max_hand_size = 11
        self.action_space = gym.spaces.Discrete(2)
        self.my_hand_has_ace_ = False
        self.dealer_hand_has_ace_ = False
        self.deck_ = self.make_deck()
        self.my_hand_ = self.deal_hand("me")
        self.dealer_hand_ = self.deal_hand("dealer")
        self.observation_space = gym.spaces.Dict(
            {
                "my_hand": gym.spaces.Box(0, 10, shape=(self.max_hand_size,), dtype=int),
                "dealer_hand": gym.spaces.Box(0, 10, shape=(self.max_hand_size,), dtype=int),
                "my_hand_has_ace": gym.spaces.Discrete(2),
                "dealer_hand_has_ace": gym.spaces.Discrete(2)
            }
        )

    def make_deck(self):
        deck = []
        for i in range(9):
            for j in range(4):
                deck.append(i+2)
        for i in range(12):
            deck.append(10)
        for i in range(4):
            deck.append("ace")
        random.shuffle(deck)
        return deck

    def deal_hand(self, name):
        hand = []
        for i in range(2):
            card = self.deck_.pop()
            if card == "ace":
                if name == "me":
                    self.my_hand_has_ace_ = True
                else:
                    self.dealer_hand_has_ace_ = True
                hand.append(1)
            else:
                hand.append(card)
        return hand

    def deal_card(self, name):
        card = self.deck_.pop()
        if card == "ace":
            if name == "me":
                self.my_hand_has_ace_ = True
            else:
                self.dealer_hand_has_ace_ = True
            return(1)
        else:
            return(card)


    def _get_obs(self):
        def pad_hand(hand):
            padded = np.zeros(self.max_hand_size, dtype=int)
            padded[:len(hand)] = hand
            return padded

        return {
            "my_hand": pad_hand(self.my_hand_),
            "dealer_hand": pad_hand(self.dealer_hand_),
            "my_hand_has_ace": self.my_hand_has_ace_,
            "dealer_hand_has_ace": self.dealer_hand_has_ace_
        }
    
    
    def _get_info(self):
        total = 0
        for card in self.my_hand_:
            total += card

        return {
            "distance_from_21": 21 - total
        }
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        self.my_hand_has_ace_ = False
        self.dealer_hand_has_ace_ = False
        self.deck_ = self.make_deck()
        self.my_hand_ = self.deal_hand("me")
        self.dealer_hand_ = self.deal_hand("dealer")
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        if action: #Its 1, aka hit
            print(f"Hitting on hand: {self.my_hand_}")
            new_card = self.deal_card("me")
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
            print(f"Standing on hand: {self.my_hand_}")
            dealer_hand_total = 0
            for card in self.dealer_hand_:
                dealer_hand_total += card
            while dealer_hand_total <= 16:
                new_card = self.deal_card("dealer")
                self.dealer_hand_.append(new_card)
                dealer_hand_total += new_card
            
            hand_total = 0
            for card in self.my_hand_:
                hand_total += card
            if self.my_hand_has_ace_:
                if (hand_total + 10) <= 21:
                    hand_total += 10 

            if self.dealer_hand_has_ace_:
                if (dealer_hand_total + 10) <= 21:
                    dealer_hand_total += 10 

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
    