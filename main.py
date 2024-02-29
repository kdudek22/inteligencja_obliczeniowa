from easyAI import TwoPlayerGame, AI_Player, Negamax
import random


class GameOfNim(TwoPlayerGame):
    """Game of nim, you take elements of the stack. the player to take the last one loses"""
    def __init__(self, players=None):
        self.players = players
        self.pile = 30
        self.current_player = 1
        self.player_moves = {}

    def possible_moves(self):
        return ["1", "2", "3"]

    def make_move(self, move):
        move = self.generate_move_amount(move)
        self.add_move_to_player_moves(move)
        self.pile -= int(move)

    def generate_move_amount(self, move):
        if random.randint(1, 10) <= 1:
            return str(int(move) + 1)
        return move

    def add_move_to_player_moves(self, move):
        if self.current_player in self.player_moves:
            self.player_moves[self.current_player].append(move)
        else:
            self.player_moves[self.current_player] = [move]

    def win(self):
        return self.pile <= 0

    def is_over(self):
        return self.win()

    def show(self):
        print(f"{self.pile} elements on the pile left")

    def scoring(self):
        return 100 if self.win() else 0


ai1 = Negamax(10)
ai2 = Negamax(10)

game = GameOfNim([AI_Player(ai1), AI_Player(ai2)])

game.play()

print(game.player_moves)
print("Winner: " + "1" if len(game.player_moves[1]) < len(game.player_moves[2]) else "2")
