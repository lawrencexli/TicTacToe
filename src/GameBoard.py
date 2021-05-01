import numpy as np
from random import randint


class GameBoard:
    valid_play = [11, 12, 13, 21, 22, 23, 31, 32, 33]

    """
    Initialize the game
    State of the game: 'X' if Player_X won, 'O' if Player_O won, '+' if draw, '.' if still ongoing
    """
    def __init__(self):
        self.play = [0] * 9
        self.winner = '.'
        self.state = ['.'] * 9
        self.verbose = False

    def play2state(self):
        # rewrite self.state after a play
        current = -1
        poss = ['.', 'O', 'X']
        for i in self.play:
            if i != 0:
                j = (i // 10 - 1) * 3 + (i % 10) - 1
                self.state[j] = poss[current]
            current = current * (-1)

    def newPlay(self, coord):
        # Add new play (if valid) and update other variables
        if self.winner != '.':
            print("Error: Game already finished")
        elif (not coord in self.valid_play):
            print("Error: shot off board")
        elif (coord in self.play):
            print("Error: move already played.")
        else:
            self.play[self.play.index(0)] = coord
            self.play2state()
            if self.verbose:
                self.display()
            self.status()

    def available(self):
        # Return list of available move to pick from
        t = self.valid_play.copy()
        for i in self.play:
            if i != 0:
                t.remove(i)
        return t

    def print(self):
        # Debugging function
        print("Play : {}.\nState : {}.\nWinner = {}.".format(self.play, self.state, self.winner))

    def display(self):
        # Print the board state
        print("")
        for i in range(9):
            print(self.state[i], end='')
            if (i + 1) % 3 == 0:
                print('\n', end='')
            else:
                print(' ', end='')
        print("")

    def status(self):
        # Check the board, if there is a winner
        if ((self.state[0] == self.state[4] == self.state[8]) or (
                self.state[2] == self.state[4] == self.state[6])) and (self.state[4] != '.'):
            self.winner = self.state[4]
        else:
            for i in range(3):
                if (self.state[i] == self.state[i + 3] == self.state[i + 6]) and (self.state[i] != '.'):
                    self.winner = self.state[i]
                elif (self.state[3 * i] == self.state[3 * i + 1] == self.state[3 * i + 2]) and (
                        self.state[3 * i] != '.'):
                    self.winner = self.state[3 * i]
        if self.winner != '.':
            if self.verbose:
                print("The winner is", self.winner, '!')
        elif (self.winner == '.') and (len(self.available()) == 0):
            self.winner = '+'
            if self.verbose:
                print("Match draw!")


def randomIA(B):
    # Make a random move on a Board B
    if B.winner != '.':
        return None
    poss = B.available()
    L = len(poss) - 1
    new = poss[randint(0, L)]
    B.newPlay(new)


def autoGame(B):
    # Playing random moves to get a valid board
    while B.winner == '.':
        randomIA(B)


def simulation(nsim=10):
    # Simulating games and exporting their results (plays, winner)
    # With outputs encoded for a LSTM neural network
    dataset = []
    win = []
    for i in range(nsim):
        B = GameBoard()
        autoGame(B)
        dataset.append([[x] for x in B.play])
        foo = [0] * 3
        if B.winner == 'X':
            foo[0] = 1
        elif B.winner == 'O':
            foo[2] = 1
        elif B.winner == '+':
            foo[1] = 1
        win.append(foo)
    return dataset, win

if __name__ == "__main__":
    a, b = simulation(5)  # Simulate 5 games
    print(np.squeeze(a))  # List of plays
    print(b)  # List of end-game status