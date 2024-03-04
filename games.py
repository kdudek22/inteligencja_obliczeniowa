from easyAI import TwoPlayerGame, AI_Player, Negamax
import matplotlib.pyplot as plt
import random


class GameOfNim(TwoPlayerGame):
    """Game of nim, you take elements of the stack. the player to take the last one loses"""

    def __init__(self, players=None, deterministic=True, random_chance=0.1):
        self.players = players
        self.pile = 20
        self.current_player = 1
        self.player_moves = {}
        self.deterministic = deterministic
        self.random_chance = random_chance
        self.player_move_times = {1: [], 2: []}

    def possible_moves(self):
        return ["1", "2", "3"]

    def make_move(self, move):
        move = self.generate_move_amount(move)
        self.add_move_to_player_moves(move)
        self.pile -= int(move)

    def generate_move_amount(self, move):
        if not self.deterministic and random.random() <= self.random_chance:
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

    def get_winner(self):
        return "1" if len(self.player_moves[1]) <= len(self.player_moves[2]) else "2"


class PlayALotOfGAmes:
    def __init__(self, player_1_parameters, player_2_parameters, num_of_games=50, deterministic=True, plot=False):
        self.parameters_1 = player_1_parameters
        self.parameters_2 = player_2_parameters
        self.player_2 = self.create_player_from_parameters(player_2_parameters)
        self.num_of_games = num_of_games
        self.deterministic = deterministic
        self.game_results = {"1": 0, "2": 0}
        self.player_moves = {}
        self.plot = plot
        self.player_move_times = {1: [], 2: []}
        self.title = (
            f"{self.num_of_games} {'deterministic' if self.deterministic else 'nondeterministic'} games depth "
            f"{self.parameters_1['algorithm'].depth} {self.parameters_2['algorithm'].depth}")

    def play_games(self):
        for i in range(self.num_of_games):
            player_1, player_2 = self.create_player_from_parameters(
                self.parameters_1), self.create_player_from_parameters(self.parameters_2)
            game = GameOfNim([player_1, player_2], deterministic=self.deterministic)
            game.play(verbose=False)
            self.update_game_results(game.get_winner())
            self.update_move_times(game.player_move_times)
            self.player_moves[i] = game.player_moves
            print(i, end=" ")

        if self.plot:
            self.plot_results()

    def update_game_results(self, winner):
        self.game_results[winner] += 1

    def update_move_times(self, player_move_times):
        for player in player_move_times:
            for time in player_move_times[player]:
                self.player_move_times[player].append(time)

    def plot_results(self):
        self.plot_wins()
        self.plot_moves()
        self.plot_move_times()
        self.plot_average_moves_amount()

    def plot_average_moves_amount(self):
        move_sum_p1 = 0
        move_sum_p2 = 0
        for game in self.player_moves:
            move_sum_p1 += len(self.player_moves[game][1])
            move_sum_p2 += len(self.player_moves[game][2])

        average_moves_p1 = move_sum_p1/len(self.player_moves.keys())
        average_moves_p2 = move_sum_p2/len(self.player_moves.keys())

        plt.title("Average moves count")
        plt.bar(["p1", "p2"], [average_moves_p1, average_moves_p2], color=["blue", "red"])
        plt.savefig(f"./plots/move_count/average_{self.title.replace(' ', '_')}.png")
        plt.show()

    def plot_move_times(self):
        plt.title("Average move time [s]")
        labels = ["p1", "p2"]
        avg_times = [sum(self.player_move_times[1])/len(self.player_move_times[1]),
                     sum(self.player_move_times[2])/len(self.player_move_times[2])]
        plt.bar(labels, avg_times, color=["blue", "red"])
        plt.savefig(f"./plots/times/times_{self.title.replace(' ', '_')}.png")
        plt.show()

    def plot_moves(self):
        move_counter = {1: {}, 2: {}}
        for game in self.player_moves:
            for player in self.player_moves[game]:
                for move in self.player_moves[game][player]:
                    if move in move_counter[player]:
                        move_counter[player][move] += 1
                    else:
                        move_counter[player][move] = 1

        all_moves = {}
        for player in move_counter:
            for move in move_counter[player]:
                if move in all_moves:
                    all_moves[move] += move_counter[player][move]
                else:
                    all_moves[move] = move_counter[player][move]

        move_counter[1] = dict(sorted(move_counter[1].items()))
        move_counter[2] = dict(sorted(move_counter[2].items()))

        plt.bar(move_counter[1].keys(), move_counter[1].values(), label=move_counter[1].values(),
                color=["green", "blue", "orange", "red"])
        plt.title("p1")
        plt.savefig(f"./plots/moves/p1_{self.title.replace(' ', '_')}.png")
        plt.show()
        plt.title("p2")
        plt.bar(move_counter[2].keys(), move_counter[2].values(), label=move_counter[2].values(),
                color=["green", "blue", "orange", "red"])
        plt.savefig(f"./plots/moves/p2_{self.title.replace(' ', '_')}.png")
        plt.show()

    def plot_wins(self):
        plt.title(self.title)
        plt.bar(self.get_label_for_plot(), self.get_scored_for_plot(), color=["blue", "red"])
        plt.savefig(f"./plots/wins/{self.title.replace(' ', '_')}.png")
        plt.show()

    def get_label_for_plot(self):
        return ["p1: " + self.parameters_1["alg_name"] + " " + str(self.parameters_1["algorithm"].depth) + " depth",
                "p2: " + self.parameters_2["alg_name"] + " " + str(self.parameters_2["algorithm"].depth) + " depth"]

    def get_scored_for_plot(self):
        return [self.game_results["1"], self.game_results["2"]]

    def create_player_from_parameters(self, parameters):
        return AI_Player(parameters["algorithm"])
