# Blackjack solver using Deep Q-Learning

### Using Gymnasium from OpenAI's Gym. 

In this project, both Q learning and Deep Q learning algorithms were implemented on different blackjack gymnasium environments. 
The basic Q-learning algorithm was able to achieve 40% winrates on test games, which is much better than a completely random player who achieves aronud a 20% winrate.

With the deep Q learning algorithm, models were achieved such that the algorithm won more than it lost by about 1%, allowing the player to win money. 

Future updates:
- Adding batch sizes for cleaner gradient descent step
- Removing dealer ace as input to neural network (not necessary)
- Allowing the bot to see the deck, allowing it to count cards and achieve a higher winrate.
