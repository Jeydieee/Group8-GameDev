import numpy as np

EMPTY = 0
BLACK = 1
WHITE = 2

QUAD_ORIGINS = [(0, 0), (0, 3), (3, 0), (3, 3)]
_CW  = [6, 3, 0, 7, 4, 1, 8, 5, 2]
_CCW = [2, 5, 8, 1, 4, 7, 0, 3, 6]

_CENTERS = [(1, 1), (1, 4), (4, 1), (4, 4)]
_CORNERS = [(0, 0), (0, 2), (0, 3), (0, 5),
            (2, 0), (2, 2), (2, 3), (2, 5),
            (3, 0), (3, 2), (3, 3), (3, 5),
            (5, 0), (5, 2), (5, 3), (5, 5)]

def new_board() -> np.ndarray:
    return np.zeros((6, 6), dtype=np.int8)

def copy_board(board: np.ndarray) -> np.ndarray:
    return board.copy()

def place_marble(board: np.ndarray, row: int, col: int, player: int) -> np.ndarray:
    nb = copy_board(board)
    nb[row, col] = player
    return nb

def rotate_quadrant(board: np.ndarray, quad: int, direction: int) -> np.ndarray:
    nb = copy_board(board)
    qr, qc = QUAD_ORIGINS[quad]
    perm = _CW if direction == 1 else _CCW
    cells = [board[qr + r, qc + c] for r in range(3) for c in range(3)]
    rotated = [cells[perm[i]] for i in range(9)]
    for i, val in enumerate(rotated):
        r, c = divmod(i, 3)
        nb[qr + r, qc + c] = val
    return nb

def _get_all_5_windows(board: np.ndarray):
    for r in range(6):
        for s in range(2): yield board[r, s:s+5]
    for c in range(6):
        for s in range(2): yield board[s:s+5, c]
    for r in range(2):
        for c in range(2): yield np.array([board[r+i, c+i] for i in range(5)])
    for r in range(2):
        for c in range(4, 6): yield np.array([board[r+i, c-i] for i in range(5)])

def check_winner(board: np.ndarray) -> int:
    black_wins = white_wins = False
    for window in _get_all_5_windows(board):
        if window[0] != EMPTY and np.all(window == window[0]):
            if window[0] == BLACK: black_wins = True
            else: white_wins = True
    if black_wins and white_wins: return 0
    if black_wins: return BLACK
    if white_wins: return WHITE
    return -1

def is_board_full(board: np.ndarray) -> bool:
    return not np.any(board == EMPTY)

def legal_placements(board: np.ndarray) -> list:
    return list(zip(*np.where(board == EMPTY)))

def legal_moves(board: np.ndarray) -> list:
    moves = []
    for r, c in legal_placements(board):
        for q in range(4):
            for d in (1, -1): moves.append((r, c, q, d))
    return moves

def apply_move(board: np.ndarray, move: tuple, player: int) -> np.ndarray:
    r, c, q, d = move
    nb = place_marble(board, r, c, player)
    nb = rotate_quadrant(nb, q, d)
    return nb