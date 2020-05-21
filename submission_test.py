import traceback

import signal
import time

from isolation import Board
from test_players import RandomPlayer


class OpenMoveEvalFn:
    def score(self, game, my_player=None):
        """Score the current game state
        Evaluation function that outputs a score equal to how many
        moves are open for AI player on the board minus how many moves
        are open for Opponent's player on the board.

        Note:
            If you think of better evaluation function, do it in CustomEvalFn below.

            Args
                game (Board): The board and game state.
                my_player (Player object): This specifies which player you are.

            Returns:
                float: The current state's score. MyMoves-OppMoves.

            """
        active_player = game.get_active_player()
        active_player_moves = game.get_player_moves(active_player)
        inactive_player_moves = game.get_opponent_moves(active_player)

        active_player_moves_len = len(active_player_moves)
        inactive_player_moves_len = len(inactive_player_moves)

        if my_player != game.get_active_player():
            if active_player_moves_len == 0:
                final_score = float("-inf")
            else:
                final_score = inactive_player_moves_len - active_player_moves_len
        else:
            if inactive_player_moves_len == 0:
                final_score = float("+inf")
            else:
                final_score = active_player_moves_len - inactive_player_moves_len

        return final_score

    ######################################################################


class CustomEvalFn:
    def __init__(self):
        pass

    def score(self, game, my_player=None):
        """Score the current game state.

        Custom evaluation function that acts however you think it should. This
        is not required but highly encouraged if you want to build the best
        AI possible.

        Args:
            game (Board): The board and game state.
            my_player (Player object): This specifies which player you are.

        Returns:
            float: The current state's score, based on your own heuristic.
        """

        active_player = game.get_active_player()
        active_player_moves = game.get_player_moves(active_player)
        inactive_player_moves = game.get_opponent_moves(active_player)

        active_player_moves_len = len(active_player_moves)
        inactive_player_moves_len = len(inactive_player_moves)

        w, h = (game.width - 1) / 2., (game.height - 1) / 2.
        y, x = game.get_player_position(my_player)
        dist = abs(h - y) + abs(w - x)

        if my_player != game.get_active_player():
            final_score = inactive_player_moves_len - active_player_moves_len
        else:
            final_score = active_player_moves_len - inactive_player_moves_len

        return final_score - .6 * dist


class MiniMaxPlayer:
    """Player that chooses a move using your evaluation function
    and a minimax algorithm with alpha-beta pruning.
    You must finish and test this player to make sure it properly
    uses minimax and alpha-beta to return a good move."""

    def __init__(self, search_depth=2, eval_fn=OpenMoveEvalFn(), name="MiniMaxPlayer"):
        """Initializes your player.

        if you find yourself with a superior eval function, update the default
        value of `eval_fn` to `CustomEvalFn()`

        Args:
            search_depth (int): The depth to which your agent will search
            eval_fn (function): Evaluation function used by your agent
        """
        self.eval_fn = eval_fn
        self.search_depth = search_depth
        self.name = name

    def move(self, game, time_left):
        """Called to determine one move by your agent

        Note:
            1. Do NOT change the name of this 'move' function. We are going to call
            this function directly.
            2. Call alphabeta instead of minimax once implemented.
        Args:
            game (Board): The board and game state.
            time_left (function): Used to determine time left before timeout

        Returns:
            tuple: (int,int): Your best move
        """
        signal.signal(signal.SIGALRM, self.handler)
        signal.setitimer(0, ((time_left() / 1000) - 0.01), 0.01)
        if not game.get_player_moves(self):
            return None
        else:
            best_move = None
            try:
                start_time = int(round(time.time() * 1000))
                while time_left() - (int(round(time.time() * 1000)) - start_time):
                    for depth in range(2, 50):
                        best_move, score = minimax(self, game, time_left, depth=depth)
            except Exception:
                signal.alarm(0)
                # traceback.print_exc()
                return best_move

    def utility(self, game, my_turn=None):
        """You can handle special cases here (e.g. endgame)"""
        return self.eval_fn.score(game, self)

    def handler(self):
        """Signal Handler to pass the exception to the exception block and return the best_move"""
        raise Exception


def minimax(player, game, time_left, depth, my_turn=True, current_depth=0):
    # For BFS we want to cap the depth so that we can check the tree over a large breath
    # So once the current depth reaches the given depth - just evaluate the score
    if current_depth == depth:
        return player.utility(game)

    if my_turn:
        # In case if we cant set the best move, then atleast return something
        best_move_min = (-1, -1)
        max_score = float("-inf")
        # This will have the scores from the maximizer
        score_list = []
        # Get all the available moves, since we tackled the no available moves before, we dont have to worry for now
        active_moves = game.get_active_moves()

        # Means that the active player has lost so return least val so that minimizer can pick it up
        if not active_moves:
            return float("-inf")

        for available_move in active_moves:
            # Create a new state - now the active player for new board is opponent & will return the list of min values
            # So we need to chose the max of the list of min values
            new_board, is_over, winner = game.forecast_move(available_move)
            # Now ret_val will be either a tuple which is returned from the opposite player, or it will be a score
            ret_val = minimax(player, new_board, time_left, depth, False, current_depth + 1)
            if type(ret_val) == tuple:
                ret_val = ret_val[1]
            score_list.append((available_move, ret_val))
            # Every time ret_val is computed, we append it to the score_list and sort it again
            # We then return the max value of all the scores back to the user for a given depth
            best_move_min, max_score = sorted(score_list, key=lambda x: x[1], reverse=True)[0]

        return best_move_min, max_score

    else:
        best_move_max = (-1, -1)
        # Just in case we cant set the best moves, we have something to return
        min_score = float("+inf")
        # This will have the scores from the minimizer
        score_list = []

        active_moves = game.get_active_moves()

        # Means that the active player has won so return max val so that maximizer can pick it up
        if not active_moves:
            return float("+inf")

        for available_move in active_moves:
            new_board, is_over, winner = game.forecast_move(available_move)
            ret_val = minimax(player, new_board, time_left, depth, True, current_depth + 1)
            if type(ret_val) == tuple:
                ret_val = ret_val[1]
            score_list.append((available_move, ret_val))
            best_move_max, min_score = sorted(score_list, key=lambda x: x[1], reverse=False)[0]

        return best_move_max, min_score


class AlphaBetaPlayer:
    """Player that chooses a move using your evaluation function
    and a minimax algorithm with alpha-beta pruning.
    You must finish and test this player to make sure it properly
    uses minimax and alpha-beta to return a good move."""

    def __init__(self, search_depth=4, eval_fn=OpenMoveEvalFn(), name="AlphaBetaPlayer"):
        """Initializes your player.

        if you find yourself with a superior eval function, update the default
        value of `eval_fn` to `CustomEvalFn()`

        Args:
            search_depth (int): The depth to which your agent will search
            eval_fn (function): Evaluation function used by your agent
        """
        self.eval_fn = eval_fn
        self.search_depth = search_depth
        self.name = name

    def move(self, game, time_left):
        """Called to determine one move by your agent

        Note:
            1. Do NOT change the name of this 'move' function. We are going to call
            this function directly.
            2. Call alphabeta instead of minimax once implemented.
        Args:
            game (Board): The board and game state.
            time_left (function): Used to determine time left before timeout

        Returns:
            tuple: (int,int): Your best move
        """
        signal.signal(signal.SIGALRM, self.handler)
        signal.setitimer(0, ((time_left() / 1000) - 0.01), 0.01)
        if not game.get_player_moves(self):
            return None
        else:
            best_move = None
            try:
                start_time = int(round(time.time() * 1000))
                while time_left() - (int(round(time.time() * 1000)) - start_time):
                    for depth in range(2, 50):
                        best_move, score = alphabeta(self, game, time_left, depth=depth)
            except Exception:
                signal.alarm(0)
                # traceback.print_exc()
                return best_move

        # best_move, score = alphabeta(self, game, time_left, depth=self.search_depth)
        # return best_move

        # signal.signal(signal.SIGALRM, self.handler)
        # signal.setitimer(0, ((time_left() / 1000) - 0.01), 0.01)
        #
        # if not game.get_player_moves(self):
        #     print("No moves left for player {}".format(game.get_active_player()))
        #     return None
        # else:
        #     score_max = float("-inf")
        #     score_journal = {}
        #     best_move = None
        #     try:
        #         for depth in range(2, 10):
        #             best_move, score = alphabeta(self, game, time_left, depth=depth)
        #             # score_journal[depth] = {best_move: score}
        #             #
        #             # # For this move() method we need to set final_selected_value
        #             # # when the best_move is returned for depth 2
        #             # # If there is a better score in future, then replace the final_selected_move
        #             # # If not then just return the move at depth 2
        #             # if not final_selected_move:
        #             #     final_selected_move = best_move
        #             #     score_max = score
        #             # else:
        #             #     if score >= score_max:
        #             #         score_max = score
        #             #         final_selected_move = best_move
        #
        #     except TimeoutError:
        #         # print("Minimax depth: {}".format(depth))
        #         # print("Score Journal {}".format(score_journal))
        #         # final_move = best_move if not final_selected_move else final_selected_move
        #         signal.alarm(0)
        #         return best_move

    def utility(self, game, my_turn=None):
        """You can handle special cases here (e.g. endgame)"""
        return self.eval_fn.score(game, self)

    def handler(self):
        """
        Signal Handler to pass the exception to the exception block and return the best_move
        :return: Exception
        """
        raise Exception


def alphabeta(player, game, time_left, depth, alpha=float("-inf"), beta=float("inf"), my_turn=True, current_depth=0):
    # For BFS we want to cap the depth so that we can check the tree over a large breath
    # So once the current depth reaches the given depth - just evaluate the score
    if current_depth == depth:
        return player.utility(game)

    if my_turn:
        # In case if we cant set the best move, then atleast return something
        best_move_min = (-1, -1)
        max_score = float("-inf")
        # This will have the scores from the maximizer
        score_list = []
        # Get all the available moves, since we tackled the no available moves before, we dont have to worry for now
        active_moves = game.get_active_moves()

        if not active_moves:
            return float("-inf")

        for available_move in active_moves:
            # Create a new state - now the active player for new board is opponent & will return the list of min values
            # So we need to chose the max of the list of min values
            new_board, is_over, winner = game.forecast_move(available_move)
            # Now ret_val will be either a tuple which is returned from the opposite player, or it will be a score
            ret_val = alphabeta(player, new_board, time_left, depth, alpha, beta, False, current_depth + 1)
            if type(ret_val) == tuple:
                ret_val = ret_val[1]
            score_list.append((available_move, ret_val))
            # Every time ret_val is computed, we append it to the score_list and sort it again
            # We then return the max value of all the scores back to the user/player for a given depth
            best_move_min, max_score = sorted(score_list, key=lambda x: x[1], reverse=True)[0]

            # Main AB Pruning Logic
            if max_score >= beta:
                break
            alpha = max(alpha, max_score)

        return best_move_min, max_score

    else:
        best_move_min = (-1, -1)
        # Just in case we cant set the best moves, we have something to return
        min_score = float("+inf")
        score_list = []
        active_moves = game.get_active_moves()

        if not active_moves:
            return float("+inf")

        for available_move in active_moves:
            new_board, is_over, winner = game.forecast_move(available_move)
            ret_val = alphabeta(player, new_board, time_left, depth, alpha, beta, True, current_depth + 1)
            # Now ret_val will be either a tuple which is returned from the opposite player, or it will be a score
            if type(ret_val) == tuple:
                ret_val = ret_val[1]
            score_list.append((available_move, ret_val))
            # Every time ret_val is computed, we append it to the score_list and sort it again
            # We then return the min value of all the scores back to the player for a given depth
            best_move_min, min_score = sorted(score_list, key=lambda x: x[1], reverse=False)[0]

            # Main AB Pruning Logic
            if min_score <= alpha:
                break
            beta = min(beta, min_score)

        return best_move_min, min_score


if __name__ == '__main__':
    winnings = 0
    losses = 0
    errors_not_implemented = 0
    errors_others = 0
    others = 0
    games = 3
    for i in range(0, games):
        print("Playing the game: {} iteration".format(i))
        try:
            r = RandomPlayer()
            p = MiniMaxPlayer()
            q = AlphaBetaPlayer()
            game = Board(q, p, 7, 7)
            winner, move_history, termination = game.play_isolation(time_limit=1000, print_moves=False)
            print("\n", winner, " has won. Reason: ", termination)
            if "Q1" in winner:
                winnings += 1
            else:
                losses += 1
            # Uncomment to see game
            # print(game_as_text(winner, move_history, termination, output_b))
        except NotImplementedError:
            print('CustomPlayer Test: Not Implemented')
            errors_not_implemented += 1
        except:
            errors_others += 1
            print('CustomPlayer Test: ERROR OCCURRED')
            print(traceback.format_exc())

    print("\n\n\n\n")
    print("Total Games: {}".format(len(range(0, games))))
    print("Winnings Minimax: {}".format(winnings))
    print("Winnings AlphaBeta: {}".format(losses))
    print("Not determined: {}".format(others))
    print("Not Implemented Errors: {}".format(errors_not_implemented))
    print("Other Errors: {}".format(errors_others))

    # try:
    #     def time_left():  # For these testing purposes, let's ignore timeouts
    #         return 1000
    #
    #     player = CustomPlayer()  # using as a dummy player to create a board
    #     sample_board = Board(player, RandomPlayer())
    #     # setting up the board as though we've been playing
    #     board_state = [
    #         [" ", "X", "X", " ", "X", "X", " "],
    #         [" ", " ", "X", " ", " ", "X", " "],
    #         ["X", " ", " ", " ", " ", "Q1", " "],
    #         [" ", "X", "X", "Q2", "X", " ", " "],
    #         ["X", " ", "X", " ", " ", " ", " "],
    #         [" ", " ", "X", " ", "X", " ", " "],
    #         ["X", " ", "X", " ", " ", " ", " "]
    #     ]
    #     sample_board.set_state(board_state, True)
    #
    #     test_pass = True
    #
    #     expected_depth_scores = [(1, -2), (2, 1), (3, 4), (4, 3), (5, 5)]
    #     # expected_depth_scores = [(1, -2), (2, 1), (3, 4)]
    #
    #     for depth, exp_score in expected_depth_scores:
    #         move, score = minimax(player, sample_board, time_left, depth=depth, my_turn=True)
    #         if exp_score != score:
    #             print("Expected: Depth: {}, Score: {}".format(depth, exp_score))
    #             test_pass = False
    #         else:
    #             print("Minimax passed for depth: ", depth)
    #
    #     if test_pass:
    #         player = CustomPlayer()
    #         sample_board = Board(RandomPlayer(), player)
    #         # setting up the board as though we've been playing
    #         board_state = [
    #             [" ", " ", " ", " ", "X", " ", "X"],
    #             ["X", "X", "X", " ", "X", "Q2", " "],
    #             [" ", "X", "X", " ", "X", " ", " "],
    #             ["X", " ", "X", " ", "X", "X", " "],
    #             ["X", " ", "Q1", " ", "X", " ", "X"],
    #             [" ", " ", " ", " ", "X", "X", " "],
    #             ["X", " ", " ", " ", " ", " ", " "]
    #         ]
    #         sample_board.set_state(board_state, p1_turn=True)
    #
    #         test_pass = True
    #
    #         expected_depth_scores = [(1, -7), (2, -7), (3, -7), (4, -9), (5, -8)]
    #         # expected_depth_scores = [(1, -7), (2, -7)]
    #
    #         for depth, exp_score in expected_depth_scores:
    #             move, score = minimax(player, sample_board, time_left, depth=depth, my_turn=False)
    #             if exp_score != score:
    #                 print("Minimax failed for depth: ", depth)
    #                 test_pass = False
    #             else:
    #                 print("Minimax passed for depth: ", depth)
    #
    #     if test_pass:
    #         print("Minimax Test: Runs Successfully!")
    #
    #     else:
    #         print("Minimax Test: Failed")
    #
    # except NotImplementedError:
    #     print('Minimax Test: Not implemented')
    # except:
    #     print('Minimax Test: ERROR OCCURRED')
    #     print(traceback.format_exc())
    # import player_submission_tests as tests
    # tests.beatRandom(CustomPlayer)
    # tests.minimaxTest(CustomPlayer, minimax)
