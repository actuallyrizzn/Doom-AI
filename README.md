# Doom-AI
Build and train an AI to play the game of Doom

## Game Environment
* The input states are the frames of the game
* Output actions:
  * Attack
  * Move right
  * Move left
  * Move forward
  * Move backward
  * Turn right
  * Turn left
 * Rewards:
   * Plus distance for getting closer to the vest
   * Minus distance for getting further from the vest
   * Minus 100 points for getting killed
   
 ## Machine learning framework
 PyTorch: See [Documentation](https://pytorch.org/docs/master/)
