"""
Headless self-play data generator for Connect Four (Phase 2, Step 1)
Reuses the exact minimax / score_position / evaluate_window / win-check
logic from connect_4.py, with no Pygame/display dependency.

Outputs: datasets/continuous_gameplay_log.csv (same schema as the
original game's CSV_HEADERS) and datasets/game_stats.csv (per-game results).
"""

import csv
import math
import os
import random
from datetime import datetime

import numpy as np

# ---------------------------------------------------------------------------
# Constants (copied verbatim from connect_4.py)
# ---------------------------------------------------------------------------
ROW_COUNT = 6
COLUMN_COUNT = 7

PLAYER = 0
AI = 1

EMPTY = 0
PLAYER_PIECE = 1
AI_PIECE = 2

WINDOW_LENGTH = 4

CSV_DIR = "datasets"
CONTINUOUS_CSV_FILE = os.path.join(CSV_DIR, "continuous_gameplay_log.csv")
GAME_STATS_CSV_FILE = os.path.join(CSV_DIR, "game_stats.csv")

CSV_HEADERS = [
    "session_id",
    "game_number",
    "move_number",
    "player",
    "match_type",
    "game_mode",
    "difficulty",
    "chosen_column",
    "minimax_score",
    "ai_explanation",
    "p1_towers",
    "p2_towers",
    "board_flat",
    "timestamp",
]

GAME_STATS_HEADERS = [
    "session_id",
    "game_number",
    "winner",
    "total_moves",
    "p1_depth",
    "p2_depth",
]

os.makedirs(CSV_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Board helper functions (verbatim logic from connect_4.py)
# ---------------------------------------------------------------------------

def create_board():
    return np.zeros((ROW_COUNT, COLUMN_COUNT))


def drop_piece(board, row, col, piece):
    board[row][col] = piece


def is_valid_location(board, col):
    return board[ROW_COUNT - 1][col] == 0


def get_next_open_row(board, col):
    for r in range(ROW_COUNT):
        if board[r][col] == 0:
            return r


def get_valid_locations(board):
    valid_locations = []
    for col in range(COLUMN_COUNT):
        if is_valid_location(board, col):
            valid_locations.append(col)
    return valid_locations


def winning_move(board, piece):
    # Horizontal
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT):
            if (board[r][c] == piece and board[r][c + 1] == piece and
                    board[r][c + 2] == piece and board[r][c + 3] == piece):
                return True
    # Vertical
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT - 3):
            if (board[r][c] == piece and board[r + 1][c] == piece and
                    board[r + 2][c] == piece and board[r + 3][c] == piece):
                return True
    # Positive diagonal
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT - 3):
            if (board[r][c] == piece and board[r + 1][c + 1] == piece and
                    board[r + 2][c + 2] == piece and board[r + 3][c + 3] == piece):
                return True
    # Negative diagonal
    for c in range(COLUMN_COUNT - 3):
        for r in range(3, ROW_COUNT):
            if (board[r][c] == piece and board[r - 1][c + 1] == piece and
                    board[r - 2][c + 2] == piece and board[r - 3][c + 3] == piece):
                return True
    return False


def is_terminal_node(board):
    return (winning_move(board, PLAYER_PIECE) or
            winning_move(board, AI_PIECE) or
            len(get_valid_locations(board)) == 0)


# ---------------------------------------------------------------------------
# Evaluation function (verbatim logic from connect_4.py)
# ---------------------------------------------------------------------------

def evaluate_window(window, piece):
    score = 0
    opp_piece = PLAYER_PIECE
    if piece == PLAYER_PIECE:
        opp_piece = AI_PIECE

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
    center_count = center_array.count(piece)
    score += center_count * 3

    for r in range(ROW_COUNT):
        row_array = [int(i) for i in list(board[r, :])]
        for c in range(COLUMN_COUNT - 3):
            window = row_array[c:c + WINDOW_LENGTH]
            score += evaluate_window(window, piece)

    for c in range(COLUMN_COUNT):
        col_array = [int(i) for i in list(board[:, c])]
        for r in range(ROW_COUNT - 3):
            window = col_array[r:r + WINDOW_LENGTH]
            score += evaluate_window(window, piece)

    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            window = [board[r + i][c + i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)

    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            window = [board[r + 3 - i][c + i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)

    return score


# ---------------------------------------------------------------------------
# Minimax (verbatim logic from connect_4.py)
# ---------------------------------------------------------------------------

def minimax(board, depth, alpha, beta, maximizingPlayer):
    valid_locations = get_valid_locations(board)
    is_terminal = is_terminal_node(board)
    if depth == 0 or is_terminal:
        if is_terminal:
            if winning_move(board, AI_PIECE):
                return (None, 100000000000000)
            elif winning_move(board, PLAYER_PIECE):
                return (None, -10000000000000)
            else:
                return (None, 0)
        else:
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
                value = new_score
                column = col
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
                value = new_score
                column = col
            beta = min(beta, value)
            if alpha >= beta:
                break
        return column, value


# ---------------------------------------------------------------------------
# AI explanation generator (simplified rule-based, mirrors connect_4.py XAI)
# ---------------------------------------------------------------------------

def generate_explanation(board_after_move, col, piece, minimax_score):
    """board_after_move: the board state AFTER the piece has been dropped."""
    if winning_move(board_after_move, piece):
        return "Winning move: completes four in a row."
    if minimax_score >= 100000000000000:
        return "Forced winning sequence detected."
    if minimax_score <= -10000000000000:
        return "Avoiding forced loss."
    if col == COLUMN_COUNT // 2:
        return "Strategic center column control."
    if minimax_score > 50:
        return "Strong offensive positioning."
    if minimax_score < -3:
        return "Defensive block of opponent threat."
    return "Positional development move."


# ---------------------------------------------------------------------------
# Self-play simulation
# ---------------------------------------------------------------------------

DIFFICULTY_DEPTH = {"EASY": 1, "MEDIUM": 3, "HARD": 5}


def play_game(session_id, game_number, p1_depth, p2_depth, csv_writer, move_counter_start):
    """
    Plays one Minimax-vs-Minimax game.
    'PLAYER_PIECE' side uses p1_depth, 'AI_PIECE' side uses p2_depth.
    Both sides use the same minimax()/score_position() (AI-perspective scoring),
    so for the PLAYER_PIECE side we negate the chosen column's perspective by
    calling minimax with maximizingPlayer=False (as in the original game loop).
    Returns (winner_label, total_moves, next_move_counter).
    """
    board = create_board()
    move_counter = move_counter_start
    turn = random.choice([PLAYER, AI])  # which side moves first
    game_over = False

    # CONQUER-mode tower counters (kept for schema compatibility; not used
    # for win/loss logic in this CLASSIC self-play simulation)
    p1_towers = 3
    p2_towers = 3
    difficulty_label = f"P1D{p1_depth}_P2D{p2_depth}"

    while not game_over:
        valid_locations = get_valid_locations(board)
        if not valid_locations:
            break

        if turn == PLAYER:
            depth = p1_depth
            col, minimax_score = minimax(board, depth, -math.inf, math.inf, False)
            player_label = "Player_Sim"
            piece = PLAYER_PIECE
        else:
            depth = p2_depth
            col, minimax_score = minimax(board, depth, -math.inf, math.inf, True)
            player_label = "AI_Warrior"
            piece = AI_PIECE

        if col is None or not is_valid_location(board, col):
            col = random.choice(valid_locations)

        row = get_next_open_row(board, col)
        drop_piece(board, row, col, piece)

        ai_explanation = generate_explanation(board, col, piece, minimax_score) \
            if player_label == "AI_Warrior" else None

        board_flat_str = ",".join(str(int(cell)) for r in board for cell in r)

        csv_writer.writerow({
            "session_id": session_id,
            "game_number": game_number,
            "move_number": move_counter,
            "player": player_label,
            "match_type": "PVE_SIM",
            "game_mode": "CLASSIC",
            "difficulty": difficulty_label,
            "chosen_column": int(col),
            "minimax_score": float(minimax_score)
            if minimax_score not in (math.inf, -math.inf) else "N/A",
            "ai_explanation": ai_explanation if ai_explanation else "N/A",
            "p1_towers": p1_towers,
            "p2_towers": p2_towers,
            "board_flat": board_flat_str,
            "timestamp": str(datetime.now()),
        })

        move_counter += 1

        if winning_move(board, piece):
            winner = "PLAYER_SIM" if piece == PLAYER_PIECE else "AI_WARRIOR"
            game_over = True
        elif len(get_valid_locations(board)) == 0:
            winner = "DRAW"
            game_over = True
        else:
            turn = AI if turn == PLAYER else PLAYER

    total_moves = move_counter - move_counter_start
    return winner, total_moves, move_counter


def main(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_SELFPLAY"

    # All depth matchups, mirroring EASY/MEDIUM/HARD = depth 1/3/5
    # Game count scaled down for deeper (slower) matchups
    depths = [1, 3, 5]
    games_for_depth_pair = {
        (1, 1): 30, (1, 3): 20, (1, 5): 10,
        (3, 1): 20, (3, 3): 15, (3, 5): 6,
        (5, 1): 10, (5, 3): 6,  (5, 5): 3,
    }
    matchups = [(d1, d2) for d1 in depths for d2 in depths]

    with open(CONTINUOUS_CSV_FILE, "w", newline="", encoding="utf-8") as f_moves, \
         open(GAME_STATS_CSV_FILE, "w", newline="", encoding="utf-8") as f_stats:

        move_writer = csv.DictWriter(f_moves, fieldnames=CSV_HEADERS)
        move_writer.writeheader()

        stats_writer = csv.DictWriter(f_stats, fieldnames=GAME_STATS_HEADERS)
        stats_writer.writeheader()

        game_number = 1
        move_counter = 1

        for (p1_depth, p2_depth) in matchups:
            num_games = games_for_depth_pair[(p1_depth, p2_depth)]
            for _ in range(num_games):
                winner, total_moves, move_counter = play_game(
                    session_id, game_number, p1_depth, p2_depth,
                    move_writer, move_counter
                )
                stats_writer.writerow({
                    "session_id": session_id,
                    "game_number": game_number,
                    "winner": winner,
                    "total_moves": total_moves,
                    "p1_depth": p1_depth,
                    "p2_depth": p2_depth,
                })
                game_number += 1

    print(f"Done. Generated {game_number - 1} games, {move_counter - 1} moves.")
    print(f"Moves CSV:  {os.path.abspath(CONTINUOUS_CSV_FILE)}")
    print(f"Stats CSV:  {os.path.abspath(GAME_STATS_CSV_FILE)}")


if __name__ == "__main__":
    main()
