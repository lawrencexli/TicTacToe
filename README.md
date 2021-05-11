# CSCI 357 Final Project

Bucknell University

Spring 2021

## Group members
* Christina Yu
* Ian Herdt
* Lawrence Li
* Steven Iovine

## Group name: Tic Tac Toe Tac Tic

## Project description

This project seeks to predict human behavior in the game of tic tac toe. 
More specifically, this project is interested in creating a neural network that is able to predict user decision-making based on their previous behavior. 
To train the network, a large number of training games must be played between AI and human player, which can predict human behaviors in the context of the game.

The project chooses the long short-term memory neural network, which can learn order dependence in sequence prediction problems, 
and connect past action sequences to predict the next action. 
One strength of LSTM networks is that they can remember information for long periods of time, 
which is suitable for the game like tic tac toe where the network needs to remember lots of opponent’s actions to predict the next move, 
and hence predicting the human behavior. 

## Running the Project

### Required library

Install the following libraries using ```pip install```

```numpy keras tensorflow```

### Instructions to run the program

1. ```cd src``` and run ```python main.py```
2. Play the tic tac toe game by entering the number from 1 to 9. 
   Each number represents a slot in the tic tac toe
   You are playing against a computer player
3. When the game is finished, you can choose to train the agent by entering ```y``` or ```n```
4. You can also choose to play the game again by entering ```y``` or ```n```. Program will exit if you enter ```n```.

