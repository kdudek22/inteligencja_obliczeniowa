from games import PlayALotOfGAmes
from easyAI import Negamax, SSS
import time


class CustomNegamax(Negamax):
    def __call__(self, game):
        start = time.time()
        move = super().__call__(game)
        move_time = time.time()-start
        game.player_move_times[game.current_player].append(move_time)
        return move


def main():
    params_1 = {"algorithm": SSS(2), "alg_name": "Negmax"}
    params_2 = {"algorithm": CustomNegamax(4), "alg_name": "Negmax"}

    lot_of_games = PlayALotOfGAmes(params_1, params_2, deterministic=False, plot=True, num_of_games=100)

    lot_of_games.play_games()


if __name__ == "__main__":
    main()
