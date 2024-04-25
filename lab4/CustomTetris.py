import random
from enum import Enum
import numpy as np


class Action(Enum):
    NONE = 0
    LEFT = 1
    RIGHT = 2


class CustomTetris:
    def __init__(self, grid_rows=3, grid_cols=3):
        self.board = np.zeros((grid_rows, grid_cols))
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.block_position = None
        self.win_moves_count = 500
        self.move_count = 0
        self.is_over = False
        self.reset()

    def get_amount_of_placed_elements(self):
        return [np.sum(self.board[:, column] == 1) for column in range(self.grid_cols)]

    def reset(self, seed=None):
        random.seed(seed)
        random_col = random.randint(0, self.grid_cols - 1)
        self.board = np.zeros((self.grid_rows, self.grid_cols))
        self.is_over = False
        self.move_count = 0
        self.block_position = [0, random_col]

    def create_new_block(self):
        random_col = random.randint(0, self.grid_cols - 1)
        self.block_position = [0, random_col]

    def perform_action(self, action: Action):
        if action == "1":
            action = Action.LEFT
        elif action == "2":
            action = Action.NONE
        else:
            action = Action.RIGHT

        if action == Action.LEFT:
            if self.block_position[1] > 0:
                if self.block_position[0] == self.grid_rows -1 or self.board[self.block_position[0] + 1, self.block_position[1] -1] != 1:
                    self.block_position[1] = max(self.block_position[1] - 1, 0)

        if action == Action.RIGHT:
            if self.block_position[1] < self.grid_cols - 1:
                if self.block_position[0] == self.grid_rows -1 or self.board[self.block_position[0] + 1, self.block_position[1] + 1] != 1:
                    self.block_position[1] = min(self.block_position[1] + 1, self.grid_cols - 1)

        if self.block_position[0] == 0 and self.board[1, self.block_position[1]] == 1:
            self.is_over = True

        self.block_position[0] += 1
        self.move_count += 1

        if self.move_count >= self.win_moves_count:
            self.is_over = True
            self.create_new_block()
            return True

        if self.block_position[0] == self.grid_rows-1 or self.board[self.block_position[0] + 1, self.block_position[1]] == 1:
            self.board[self.block_position[0], self.block_position[1]] = 1

            if all(self.get_amount_of_placed_elements()) >= 1:
                self.board = np.vstack((np.zeros(self.grid_cols, dtype=self.board.dtype), self.board[:-1]))

            self.create_new_block()

    def get_observations(self):
        position = self.block_position
        block_amounts = self.get_amount_of_placed_elements()
        return np.array(position + block_amounts)


    def render(self):
        print("")
        for i in range(self.grid_rows):
            for j in range(self.grid_cols):
                value_at_position = self.board[i, j]
                if self.block_position[0] == i and self.block_position[1] == j:
                    print("X", end="")
                elif value_at_position == 0:
                    print("-", end="")
                elif value_at_position == 1:
                    print("O", end="")

            print("")
        print("")



if __name__ == "__main__":
    x = CustomTetris(5,5)
#     # for i in range(100):
#     #     x.get_amount_of_placed_elements()
#     #     x.show_board()
#     #     rand_action = random.choice(list(Action))
#     #     print(rand_action)
#     #     x.perform_action(rand_action)
#
#
    while not x.is_over:
        x.render()
        print(x.get_observations())
        move = input("move?")
        x.perform_action(move)

    # x.show_board()

