"""
Connect Four Gymnasium environment.
Wraps the exact board/minimax logic from connect_4.py (reproduced here
standalone, verbatim in behavior) so DQN can train against:
  - a random opponent
  - a Minimax opponent at a configurable depth

Observation: 42-length vector (flattened 6x7 board), values in {0,1,2}
Action: column index 0-6 (invalid/full columns are masked)
Reward: +1 win, -1 loss, 0 draw, small step penalty optional
"""

import math
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

ROW_COUNT = 6
COLUMN_COUNT = 7
EMPTY = 0
PLAYER_PIECE = 1   # the DQN agent
AI_PIECE = 2        # the opponent (random or minimax)
WINDOW_LENGTH = 4


# ---------------------------------------------------------------------------
# Board helpers (verbatim logic from connect_4.py)
# ---------------------------------------------------------------------------
def create_board():
    return np.zeros((ROW_COUNT, COLUMN_COUNT), dtype=np.int8)


def drop_piece(board, row, col, piece):
    board[row][col] = piece


def is_valid_location(board, col):
    return board[ROW_COUNT - 1][col] == 0


def get_next_open_row(board, col):
    for r in range(ROW_COUNT):
        if board[r][col] == 0:
            return r


def get_valid_locations(board):
    return [c for c in range(COLUMN_COUNT) if is_valid_location(board, c)]


def winning_move(board, piece):
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT):
            if all(board[r][c + i] == piece for i in range(4)):
                return True
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT - 3):
            if all(board[r + i][c] == piece for i in range(4)):
                return True
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT - 3):
            if all(board[r + i][c + i] == piece for i in range(4)):
                return True
    for c in range(COLUMN_COUNT - 3):
        for r in range(3, ROW_COUNT):
            if all(board[r - i][c + i] == piece for i in range(4)):
                return True
    return False


def is_terminal_node(board):
    return (winning_move(board, PLAYER_PIECE) or
            winning_move(board, AI_PIECE) or
            len(get_valid_locations(board)) == 0)


def evaluate_window(window, piece):
    score = 0
    opp_piece = PLAYER_PIECE if piece == AI_PIECE else AI_PIECE
    if window.count(piece) == 4:
        score += 100
    elif window.count(piece) == 3 and window.count(EMPTY) == 1:
        score += 5
    elif window.count(piece) == 2 and window.count(EMPTY) == 2:
        score += 2
    if window.count(opp_piece) == 3 and window.count(EMPTY) == 1:
        score -= 4
    return score


def score_position(board, piece):
    score = 0
    center_array = [int(i) for i in list(board[:, COLUMN_COUNT // 2])]
    score += center_array.count(piece) * 3

    for r in range(ROW_COUNT):
        row_array = [int(i) for i in list(board[r, :])]
        for c in range(COLUMN_COUNT - 3):
            score += evaluate_window(row_array[c:c + WINDOW_LENGTH], piece)

    for c in range(COLUMN_COUNT):
        col_array = [int(i) for i in list(board[:, c])]
        for r in range(ROW_COUNT - 3):
            score += evaluate_window(col_array[r:r + WINDOW_LENGTH], piece)

    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            window = [board[r + i][c + i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)

    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            window = [board[r + 3 - i][c + i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)

    return score


def minimax(board, depth, alpha, beta, maximizingPlayer):
    valid_locations = get_valid_locations(board)
    is_terminal = is_terminal_node(board)
    if depth == 0 or is_terminal:
        if is_terminal:
            if winning_move(board, AI_PIECE):
                return (None, 1e14)
            elif winning_move(board, PLAYER_PIECE):
                return (None, -1e13)
            else:
                return (None, 0)
        return (None, score_position(board, AI_PIECE))

    if maximizingPlayer:
        value = -math.inf
        column = random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, AI_PIECE)
            new_score = minimax(b_copy, depth - 1, alpha, beta, False)[1]
            if new_score > value:
                value, column = new_score, col
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return column, value
    else:
        value = math.inf
        column = random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, PLAYER_PIECE)
            new_score = minimax(b_copy, depth - 1, alpha, beta, True)[1]
            if new_score < value:
                value, column = new_score, col
            beta = min(beta, value)
            if alpha >= beta:
                break
        return column, value


# ---------------------------------------------------------------------------
# Gymnasium environment
# ---------------------------------------------------------------------------
class ConnectFourEnv(gym.Env):
    """
    The DQN agent always plays as PLAYER_PIECE (1) and moves first.
    The opponent (random or minimax) plays as AI_PIECE (2) and responds.
    """
    metadata = {"render_modes": []}

    def __init__(self, opponent="random", opponent_depth=3, invalid_move_penalty=-2.0):
        super().__init__()
        self.observation_space = spaces.Box(low=0, high=2, shape=(42,), dtype=np.int8)
        self.action_space = spaces.Discrete(COLUMN_COUNT)
        self.opponent = opponent  # "random" or "minimax"
        self.opponent_depth = opponent_depth
        self.invalid_move_penalty = invalid_move_penalty
        self.board = None

    def _obs(self):
        return self.board.flatten().astype(np.int8)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = create_board()
        return self._obs(), {}

    def _opponent_move(self):
        valid = get_valid_locations(self.board)
        if not valid:
            return None
        if self.opponent == "random":
            return random.choice(valid)
        else:
            col, _ = minimax(self.board, self.opponent_depth, -math.inf, math.inf, True)
            if col is None or col not in valid:
                col = random.choice(valid)
            return col

    def step(self, action):
        # Invalid move handling: penalize but don't crash
        if not is_valid_location(self.board, action):
            return self._obs(), self.invalid_move_penalty, False, False, {"invalid": True}

        row = get_next_open_row(self.board, action)
        drop_piece(self.board, row, action, PLAYER_PIECE)

        if winning_move(self.board, PLAYER_PIECE):
            return self._obs(), 1.0, True, False, {"result": "win"}
        if len(get_valid_locations(self.board)) == 0:
            return self._obs(), 0.0, True, False, {"result": "draw"}

        # Opponent responds
        opp_col = self._opponent_move()
        if opp_col is not None:
            opp_row = get_next_open_row(self.board, opp_col)
            drop_piece(self.board, opp_row, opp_col, AI_PIECE)

            if winning_move(self.board, AI_PIECE):
                return self._obs(), -1.0, True, False, {"result": "loss"}
            if len(get_valid_locations(self.board)) == 0:
                return self._obs(), 0.0, True, False, {"result": "draw"}

        return self._obs(), 0.0, False, False, {}

    def action_masks(self):
        """Optional helper for masking invalid actions (used by evaluation code)."""
        return np.array([is_valid_location(self.board, c) for c in range(COLUMN_COUNT)])
