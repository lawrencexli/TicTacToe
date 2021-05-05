import sys
import numpy as np
import random

from NeuralAgent import NeuralAgent


class GameBoard:

    def __init__(self):
        self.board_arr = None
        self.tutorial_arr = None
        self.record_arr = None
        self.user_num = None
        self.comp_num = None
        self.record_choice_arr = None
        self.record_board_arr = None
        print("Welcome to the game of tic tac toe!")

    # Creates blank 3x3 array. Randomizes who goes first.
    def new_game(self):
        # 0's act as O's
        # 1's act as X's
        # 3's act as placeholders for blank spots
        self.board_arr = np.array([[3, 3, 3],
                                   [3, 3, 3],
                                   [3, 3, 3]])
        self.tutorial_arr = np.array([[1, 2, 3],
                                      [4, 5, 6],
                                      [7, 8, 9]])
        self.record_board_arr = []
        self.record_choice_arr = []

        # Flip a coin to see who goes first with X's
        coin_flip = np.random.randint(0, 2)
        self.user_num = 0
        self.comp_num = 1
        if coin_flip == 0:
            print("Computer goes first. Your letter is O.")
            self.comp_turn()
        elif coin_flip == 1:
            print("You go first. Your letter is O.")
            self.user_turn()

    # Converts internal numpy array into a visual ASCII board.
    def display_board(self):
        board_list = []

        # loops through flattened board array to scan for 0's, 1's and 3's
        # converts them into O's, X's, and blank spots
        internal_arr = self.board_arr.flatten()
        for i in range(0, 9):
            if internal_arr[i] == 0:
                board_list.append('O')
            elif internal_arr[i] == 1:
                board_list.append('X')
            elif internal_arr[i] == 3:
                board_list.append(' ')
            else:
                raise Exception("display_board Error")

        # inputs O's, X's, and blank spots into an ASCII tic tac toe board
        print("""
         {} | {} | {}
        ---+---+---
         {} | {} | {}
        ---+---+---
         {} | {} | {}
        """.format(*board_list))

        print("Numpy array representation: ")
        print(self.board_arr)

    def return_open_slots(self):
        # Checks for open slots using Boolean arrays.
        # Important when checking for winner (if draw) and checking if user's input...
        # ...is valid
        open_slots = []

        bool_arr = (self.board_arr == 3)
        flat_bool_arr = bool_arr.flatten()

        # is spot taken by 3's? If so, then spot is open.
        # appends (i + 1) because inputs are indexed to 1
        for i in range(0, len(flat_bool_arr)):
            if flat_bool_arr[i]:
                open_slots.append(i + 1)

        return open_slots

    def terminate(self, last_played_num):
        # if last played number is user's number, declares user to be winner
        # if last played number is comp's number, declares comp to be winner
        # if return_open_slots() came up blank, declares draw
        if last_played_num == self.user_num:
            print("You win!")
        elif last_played_num == self.comp_num:
            print("You lost!")
        elif last_played_num == "Draw!":
            print("Draw!")

        self.evaluate()

    def evaluate(self):
        if input("Print the numpy array? (y/n)") == 'y':
            print("Board history: ")
            print(self.record_board_arr)

            print("User choice history: ")
            print(self.record_choice_arr)

        if input("Train the neural agent? (y/n)") == 'y':
            agent = NeuralAgent()
            agent.load_model()

            X = np.array(self.record_board_arr)
            y = np.array(self.record_choice_arr)
            agent.train(X, y)
            agent.plot(X, y)

            if input("Save model? (y/n)") == 'y':
                agent.save_model()

        if input("Do you want to play again? (y/n)") == 'y':
            self.new_game()
        else:
            print("Thank you for playing!")
            sys.exit()

    def check_for_winner(self, last_played_num):
        # Scans rows, columns, and diagonals for last-played number
        # Ex. if 1 was the last number played, this function would scan for 1's
        # Declares draw is open_slots is blank
        # Else proceeds to next turn

        if not self.return_open_slots:
            # Checks if no open slots
            self.terminate("Draw!")

        for i in range(0, 3):
            # Checks rows and columns for match
            rows_win = (self.board_arr[i, :] == last_played_num).all()
            cols_win = (self.board_arr[:, i] == last_played_num).all()

            if rows_win or cols_win:
                self.terminate(last_played_num)

        diag1_win = (np.diag(self.board_arr) == last_played_num).all()
        diag2_win = (np.diag(np.fliplr(self.board_arr)) == last_played_num).all()

        if diag1_win or diag2_win:
            # Checks both diagonals for match
            self.terminate(last_played_num)

        self.next_turn(last_played_num)

    def next_turn(self, last_played_num):
        if last_played_num == self.user_num:
            self.comp_turn()
        elif last_played_num == self.comp_num:
            self.user_turn()

    def place_letter(self, current_num, current_input):
        # Takes comp_num and comp_choice (or user_num and user_choice)...
        # ...and inputs that into the global board_arr
        # Current_input is either randomly chosen by computer or input by user
        # Current_num is either user_num or comp_num
        index = np.where(self.tutorial_arr == current_input)
        self.board_arr[index] = current_num

    def user_turn(self):
        self.display_board()

        user_input = input("Pick an open slot: ")
        user_input = int(user_input)

        if user_input in self.return_open_slots():
            self.record_choice_arr += [user_input]
            self.place_letter(self.user_num, user_input)
            self.record_board_arr += [self.board_arr.tolist()]
        else:
            print("That's not a open slots.")
            self.user_turn()
        self.check_for_winner(self.user_num)

    def comp_turn(self):
        # Randomly chooses from open_slots to place its letter
        open_slots = self.return_open_slots()
        comp_input = random.choice(open_slots)
        self.place_letter(self.comp_num, comp_input)
        self.check_for_winner(self.comp_num)
