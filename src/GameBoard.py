import random
import sys
from time import sleep

import numpy as np

from NeuralAgent import NeuralAgent

"""
GameBoard is a representation of the tic tac toe game session 
and the neural network prediction platform

Code specialized from https://github.com/geoffreyyip/numpy-tictactoe/blob/master/tictactoe.py
Modified and adapted for neural network training and human behavioral prediction
Rewritten in object oriented format.
"""


class GameBoard:

    def __init__(self):
        self.board_arr = None
        self.tutorial_arr = None
        self.record_arr = None
        self.user_num = None
        self.comp_num = None
        self.record_choice_arr = None
        self.record_board_arr = None
        self.agent = NeuralAgent()
        print()
        print("<=================================>")
        print("Welcome to the game of tic tac toe!")
        print("<=================================>")
        print()
        sleep(1)

    """
    Creates a blank 3x3 numpy array for game board representation. Randomizes who goes first.
    """

    def new_game(self):
        # 0 act as "O"
        # 1 act as "X"
        # 3 act as empty slots
        self.board_arr = np.array([[3, 3, 3],
                                   [3, 3, 3],
                                   [3, 3, 3]])
        self.tutorial_arr = np.array([[1, 2, 3],
                                      [4, 5, 6],
                                      [7, 8, 9]])
        self.record_board_arr = []
        self.record_choice_arr = []
        self.agent.load_model()
        self.user_num = 0  # Make a user number for identification
        self.comp_num = 1  # Make a computer number for identification

        # Flip a coin to determine which player goes first
        coin_flip = np.random.randint(0, 2)

        if coin_flip == 0:
            print("Computer goes first. Your letter is O.")
            self.comp_turn()
        else:
            print("You go first. Your letter is O.")
            self.user_turn()

    """
    Converts internal numpy array into a visual ASCII board.
    Display format from https://github.com/geoffreyyip/numpy-tictactoe/blob/master/tictactoe.py
    """

    def display_board(self):
        board_list = []
        internal_arr = self.board_arr.flatten()
        for i in range(0, 9):
            if internal_arr[i] == 0:
                board_list.append('O')
            elif internal_arr[i] == 1:
                board_list.append('X')
            elif internal_arr[i] == 3:
                board_list.append(' ')
            else:
                raise Exception("Error: Unable to display board")

        # A board representation (ASCII)
        # From https://github.com/geoffreyyip/numpy-tictactoe/blob/master/tictactoe.py
        print("""
         {} | {} | {}
        ---+---+---
         {} | {} | {}
        ---+---+---
         {} | {} | {}
        """.format(*board_list))

        print("Numpy array representation: ")
        print(self.board_arr)

    """
    Checks for open slots using boolean arrays.
    """

    def return_open_slots(self):
        open_slots = []
        bool_arr = (self.board_arr == 3)
        bool_arr = bool_arr.flatten()

        for i in range(len(bool_arr)):
            if bool_arr[i]:
                open_slots.append(i + 1)

        return open_slots

    """
    Terminate the game and determine the winner or draw
    Then we will evaluate the board and train the agent
    """

    def terminate(self, last_played_num):
        self.display_board()
        if last_played_num == self.user_num:
            print("You win!")
        elif last_played_num == self.comp_num:
            print("You lost!")
        elif last_played_num == "Draw":
            print("Draw!")

        # Evaluate the game and train the model
        self.evaluate()

    """
    Evaluate the game and train the neural agent
    """

    def evaluate(self):
        if input("Train the neural agent? (y/n)") == 'y':
            board_input = np.array(self.record_board_arr)
            print("The length:", len(self.record_choice_arr))
            board_input = board_input.reshape((len(self.record_choice_arr), 3, 3))
            user_choice_output = np.array(self.record_choice_arr)
            user_choice_output = user_choice_output.reshape((len(self.record_choice_arr), 3, 3))
            self.agent.train(board_input, user_choice_output)

            if input("Save model? (y/n)") == 'y':
                self.agent.save_model()

        if input("Do you want to play again? (y/n)") == 'y':
            self.new_game()
        else:
            print("Thank you for playing!")
            sys.exit()

    """
    Checks rows, columns, and diagonals for winning
    """

    def check_for_winner(self, last_played_num):

        # Check if all slots are taken, then it is a draw
        if not self.return_open_slots():
            self.terminate("Draw")

        # Check rows and columns first. For loop 3 times for checking.
        for i in range(3):
            if (self.board_arr[i, :] == last_played_num).all() or \
                    (self.board_arr[:, i] == last_played_num).all():
                self.terminate(last_played_num)

        # Check diagonal then. Use diag() and fliplr() method from numpy array.
        if (np.diag(self.board_arr) == last_played_num).all() or \
                (np.diag(np.fliplr(self.board_arr)) == last_played_num).all():
            self.terminate(last_played_num)

        # If not terminated, then the game continues
        self.next_turn(last_played_num)

    """
    Determine who will play next
    """

    def next_turn(self, last_played_num):
        if last_played_num == self.user_num:
            self.comp_turn()
        elif last_played_num == self.comp_num:
            self.user_turn()

    """
    Place the "O" or "X" into the board
    """

    def place_letter(self, current_num, current_input):
        self.board_arr[np.where(self.tutorial_arr == current_input)] = current_num

    """
    Predict user turn and user actually makes the turn
    """

    def user_turn(self):
        self.display_board()
        self.predict()

        user_input = int(input("Pick an open slot: "))

        if user_input in self.return_open_slots():
            self.record_user_choice(user_input)
        else:
            print("That's not a open slots.")
            self.user_turn()
        self.check_for_winner(self.user_num)

    """
    Record user choice to the game board and save the record for future training
    """

    def record_user_choice(self, user_input):
        arr = [0 for _ in range(9)]
        arr[user_input - 1] = 1
        arr = np.reshape(arr, (-1, 3))
        self.record_choice_arr += [arr.tolist()]
        self.place_letter(self.user_num, user_input)
        self.record_board_arr += [self.board_arr.tolist()]

    """
    Neural network predict the next move of human player based on current state
    """

    def predict(self):
        floating_format = "{0:.9f}"
        prediction = self.board_arr.reshape((1, 3, 3))
        print("Prediction: ")
        result = self.agent.predict(prediction)
        result = result.flatten().tolist()
        for i in range(len(result)):
            if i + 1 not in self.return_open_slots():
                result[i] = 'Closed Slot'
            else:
                result[i] = floating_format.format(result[i])
        print(np.reshape(result, (-1, 3)))

    """
    Computer randomly picks an open slots
    """

    def comp_turn(self):
        open_slots = self.return_open_slots()
        comp_input = random.choice(open_slots)
        self.place_letter(self.comp_num, comp_input)
        self.check_for_winner(self.comp_num)
