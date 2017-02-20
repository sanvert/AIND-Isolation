"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random
import math

MAX_DEPTH = 11


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def score_calculation_1(game, player):
    """Heuristic value of the game state is calculated as number
    of legal moves of current player minus two times number of legal moves of
    opponent.

    Parameters
    ----------
    game : `isolation.Board`
    player : object

    Returns
    -------
    float : Number of current player's legal moves
    """
    return float(len(game.get_legal_moves(player)) - 2 * len(game.get_legal_moves(game.get_opponent(player))))


def score_calculation_2(game, player):
    """Heuristic value of the game state is calculated as number
    of legal moves of current player minus minimum number of opponent
    moves after each current player move is applied on board.
    Parameters
    ----------
    game : `isolation.Board`
    player : object

    Returns
    -------
    float : Number of current player's legal moves
    """
    player_moves = game.get_legal_moves(player)
    opponent_moves = float("inf")
    for move in player_moves:
        forecast_game = game.forecast_move(move)
        forecast_moves_opponent = forecast_game.get_legal_moves(game.get_opponent(player))
        if not opponent_moves == float("inf") or len(forecast_moves_opponent) < opponent_moves:
            opponent_moves = len(forecast_moves_opponent)
    return float(len(player_moves) - opponent_moves)


def ultimate_score_calculation(game, player):
    """Heuristic value of the game state is calculated as number
    of legal moves of current player minus two times minimum number of opponent
    moves after each current player move is applied on board.
    Parameters
    ----------
    game : `isolation.Board`
    player : object

    Returns
    -------
    float : Number of current player's legal moves
    """
    player_moves = game.get_legal_moves(player)
    opponent_moves = float("inf")
    for move in player_moves:
        forecast_game = game.forecast_move(move)
        forecast_moves_opponent = forecast_game.get_legal_moves(game.get_opponent(player))
        if not opponent_moves == float("inf") or len(forecast_moves_opponent) < opponent_moves:
            opponent_moves = len(forecast_moves_opponent)
    return float(len(player_moves) - 2 * opponent_moves)


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    return ultimate_score_calculation(game, player)


class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.maximizing_player=False
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left
        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves
        score, move = float("-inf"), (-1, -1)
        if not legal_moves:
            return move
        elif game.is_initial_move():
            self.maximizing_player = True
            return random.choice(game.get_open_move_book())
        else:
            # Return the best move from the last completed search iteration
            try:
                # The search method call (alpha beta or minimax) should happen in
                # here in order to avoid timeout. The try/except block will
                # automatically catch the exception raised by the search method
                # when the timer gets close to expiring
                selected_method = self.minimax if self.method == 'minimax' else self.alphabeta
                if self.iterative:
                    for iter_depth_idx in range(MAX_DEPTH):
                        iter_score, iter_move = selected_method(game, iter_depth_idx + 1, self.maximizing_player)
                        if iter_score >= score:
                            score = iter_score
                            move = iter_move
                else:
                    score, move = selected_method(game, self.search_depth, self.maximizing_player)
                return move
            except Timeout:
                # Handle any actions required at timeout, if necessary
                return move

    def min(self, game, depth):
        """ Minimizing function of minimax algorithm.
            According to the depth value given, it searches the min one
            among all legal moves.
            Parameters
            ----------
            game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

            depth : int (optional)
            A strictly positive integer (i.e., 1, 2, 3,...) for the number of
            layers in the game tree to explore for fixed-depth search.

            Returns
            -------
            float
            The score for the current search branch
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        utility = game.utility(self)
        if math.isclose(utility, 0):
            if depth == 0:
                return self.score(game, self)
            lowest_score = float("inf")
            player_valid_moves = game.get_legal_moves(game.active_player)
            for move in player_valid_moves:
                forecast_game = game.forecast_move(move)
                forecast_score = self.max(forecast_game, depth - 1)
                if forecast_score < lowest_score:
                    lowest_score = forecast_score
            return lowest_score
        else:
            return utility

    def max(self, game, depth):
        """ Maximizing function of minimax algorithm.
            According to the depth value given, it searches the max one
            among all legal moves.
            Parameters
            ----------
            game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

            depth : int (optional)
            A strictly positive integer (i.e., 1, 2, 3,...) for the number of
            layers in the game tree to explore for fixed-depth search.

            Returns
            -------
            float
            The score for the current search branch
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        utility = game.utility(self)
        if math.isclose(utility, 0):
            if depth == 0:
                return self.score(game, self)
            highest_score = float("-inf")
            player_valid_moves = game.get_legal_moves(game.active_player)
            for move in player_valid_moves:
                forecast_game = game.forecast_move(move)
                forecast_score = self.min(forecast_game, depth - 1)
                if forecast_score > highest_score:
                    highest_score = forecast_score
            return highest_score
        else:
            return utility

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        player_valid_moves = game.get_legal_moves(self)
        score = float("-inf") if maximizing_player else float("inf")
        selected_move = (-1, -1)
        if player_valid_moves and depth > 0:
            for move in player_valid_moves:
                forecast_game = game.forecast_move(move)
                if maximizing_player:
                    forecast_score = self.min(forecast_game, depth - 1)
                    if forecast_score > score:
                        score = forecast_score
                        selected_move = move
                else:
                    forecast_score = self.max(forecast_game, depth - 1)
                    if forecast_score < score:
                        score = forecast_score
                        selected_move = move
        return score, selected_move

    def min_alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """ Minimizing function of alphabeta algorithm.
            According to the depth value given, it searches the min one
            among all legal moves.
            Parameters
            ----------
            game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

            depth : int (optional)
            A strictly positive integer (i.e., 1, 2, 3,...) for the number of
            layers in the game tree to explore for fixed-depth search.

            alpha : float
            Alpha limits the lower bound of search on minimizing layers

            beta : float
            Beta limits the upper bound of search on maximizing layers

            Returns
            -------
            float
            The score for the current search branch
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        utility = game.utility(self)
        if math.isclose(utility, 0):
            if depth == 0:
                return self.score(game, self)
            lowest_score = float("inf")
            player_valid_moves = game.get_legal_moves(game.active_player)
            for move in player_valid_moves:
                forecast_game = game.forecast_move(move)
                forecast_score = self.max_alphabeta(forecast_game, depth - 1, alpha, beta)
                if forecast_score <= alpha:
                    return forecast_score
                elif forecast_score <= lowest_score:
                    beta = lowest_score = forecast_score
            return lowest_score
        else:
            return utility

    def max_alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"),):
        """ Maximizing function of alphabeta algorithm.
            According to the depth value given, it searches the max one
            among all legal moves.

            Parameters
            ----------
            game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

            depth : int (optional)
            A strictly positive integer (i.e., 1, 2, 3,...) for the number of
            layers in the game tree to explore for fixed-depth search.

            alpha : float
            Alpha limits the lower bound of search on minimizing layers

            beta : float
            Beta limits the upper bound of search on maximizing layers

            Returns
            -------
            float
            The score for the current search branch
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        utility = game.utility(self)
        if math.isclose(utility, 0):
            if depth == 0:
                return self.score(game, self)
            highest_score = float("-inf")
            player_valid_moves = game.get_legal_moves(game.active_player)
            for move in player_valid_moves:
                forecast_game = game.forecast_move(move)
                forecast_score = self.min_alphabeta(forecast_game, depth - 1, alpha, beta)
                if forecast_score >= beta:
                    return forecast_score
                elif forecast_score >= highest_score:
                    alpha = highest_score = forecast_score
            return highest_score
        else:
            return utility

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        player_valid_moves = game.get_legal_moves(self)
        score = float("-inf") if maximizing_player else float("inf")
        selected_move = (-1, -1)
        if player_valid_moves:
            for move in player_valid_moves:
                forecast_game = game.forecast_move(move)
                if maximizing_player:
                    forecast_score = self.min_alphabeta(forecast_game, depth - 1, alpha, beta)
                    if forecast_score > score:
                        alpha = score = forecast_score
                        selected_move = move
                else:
                    forecast_score = self.max_alphabeta(forecast_game, depth - 1, alpha, beta)
                    if forecast_score < score:
                        beta = score = forecast_score
                        selected_move = move

        return score, selected_move
