# Blackjack solver using Q-Learning

## Using Gymnasium from OpenAI's Gym. 

A blackjack environment was created in which a player and the dealer are dealt cards from 1 to 11.
The player has two actions, hit or stand. If the player hits, they get an additional card. If they go above 21 they lose, with a reward of -1.
If they stand, if the dealer's hand value is less than or equal to 16 he must hit. If he goes above 21 the player wins, with a reward of +1, if his hand is higher, the player loses with -1, or if the player's hand is higher the player wins with a reward of +1.

A basic Q-Learning algorithm was implemented. 

Initial iteration
Out of 100 games, the Q-learning solver wins 40% of the time.
Compared to a random solver, which wins about 20% of the time.
The q values of different hands was displayed below, and it can be seen the total reward patterns follow what logically makes sense; standing at 21 has the highest reward, and hitting at lower values has high rewards as well. 
| State | Action | Reward   |
|-------|--------|----------|
| 2     | hit    | 201.902  |
| 2     | stand  | -0.5     |
| 3     | hit    | 354.892  |
| 3     | stand  | -0.5     |
| 4     | hit    | 263.927  |
| 4     | stand  | -0.5     |
| 5     | hit    | 134.433  |
| 5     | stand  | -0.5     |
| 6     | hit    | 168.865  |
| 6     | stand  | -0.5     |
| 7     | hit    | 373.471  |
| 7     | stand  | -0.5     |
| 8     | hit    | 264.194  |
| 8     | stand  | -0.5     |
| 9     | hit    | 385.843  |
| 9     | stand  | -0.5     |
| 10    | hit    | 307.628  |
| 10    | stand  | -0.5     |
| 11    | hit    | 318.515  |
| 11    | stand  | -0.5     |
| 12    | hit    | 259.765  |
| 12    | stand  | -0.5     |
| 13    | hit    | 233.654  |
| 13    | stand  | -1       |
| 14    | hit    | 60.5561  |
| 14    | stand  | -1       |
| 15    | hit    | 313.707  |
| 15    | stand  | -1       |
| 16    | hit    | 130.238  |
| 16    | stand  | -1       |
| 17    | hit    | 31.7813  |
| 17    | stand  | -1       |
| 18    | hit    | 114.796  |
| 18    | stand  | -1       |
| 19    | stand  | 375      |
| 20    | stand  | 541.5    |
| 21    | stand  | 651      |
| 22    | stand  | 42       |


The next objective is to add the player's ability to see the dealer's hand. After that more parameters and deep Q learning will be implemented. 
