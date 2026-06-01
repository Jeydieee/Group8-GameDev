import pygame
import numpy as np
from typing import Optional
import math
import sys
import os
import time as _time

# ==========================================
# 1. CORE ENGINE (Raw Math & Rules)
# ==========================================
EMPTY = 0
BLACK = 1
WHITE = 2

QUAD_ORIGINS = [(0, 0), (0, 3), (3, 0), (3, 3)]
_CW  = [6, 3, 0, 7, 4, 1, 8, 5, 2]
_CCW = [2, 5, 8, 1, 4, 7, 0, 3, 6]

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

def legal_placements(board: np.ndarray) -> list[tuple[int, int]]:
    return list(zip(*np.where(board == EMPTY)))

def legal_moves(board: np.ndarray) -> list[tuple[int, int, int, int]]:
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

class GameState:
    def __init__(self):
        self.board = new_board()
        self.turn = WHITE
        self.phase = "place"
        self.pending_place: Optional[tuple[int, int]] = None
        self.winner = -1
        self.move_count = 0

    def place(self, row: int, col: int) -> bool:
        if self.winner != -1 or self.phase != "place": return False
        if self.board[row, col] != EMPTY: return False
        self.board = place_marble(self.board, row, col, self.turn)
        w = check_winner(self.board)
        if w >= 0:
            self.winner = w
            return True
        self.phase = "rotate"
        self.pending_place = (row, col)
        return True

    def rotate(self, quad: int, direction: int, next_turn_player: int) -> bool:
        if self.winner != -1 or self.phase != "rotate": return False
        self.board = rotate_quadrant(self.board, quad, direction)
        w = check_winner(self.board)
        if w >= 0: self.winner = w
        elif is_board_full(self.board): self.winner = 0
        self.turn = next_turn_player
        self.phase = "place"
        self.pending_place = None
        self.move_count += 1
        return True

    def is_over(self) -> bool:
        return self.winner != -1


# ==========================================
# 2. AI MODULE  (FIXED)
# ==========================================
#
# WHAT WAS WRONG IN THE ORIGINAL:
#   a) evaluate_board only checked 4 center squares (max ±40 score).
#      The AI literally couldn't distinguish a near-win from a random board.
#   b) depth=2 means "I place → opponent places → static eval."
#      It cannot see any threat that resolves in 3 moves.
#   c) No window scoring — consecutive marbles were invisible to the heuristic.
#   d) No immediate-threat pre-scan before deep search.
#   e) No move ordering — alpha-beta pruned almost nothing.
#   f) No iterative deepening — UI blocked with no time-budget safety valve.
#
# FIXES:
#   1. Window-based heuristic (every 5-cell window, all directions)
#   2. Immediate win/block pre-scan  (O(36×8) before any tree search)
#   3. Move ordering before alpha-beta  (dramatically improves pruning)
#   4. Iterative Deepening (IDDFS) with wall-clock time limit
#   5. Default depth raised to 3 (viable because of improved pruning)
#
# NOTE ON GENETIC ALGORITHMS:
#   A GA is an *offline* tool for tuning the heuristic weight constants below
#   (run GA tournaments to evolve _FOUR_SCORE, _THREE_SCORE, etc.).
#   It does NOT replace alpha-beta during live play.  IDDFS + alpha-beta is
#   the correct real-time algorithm; use GA only to tune weights separately.

_WIN_SCORE    = 100_000
_FOUR_SCORE   = 22417
_THREE_SCORE  = 2530
_TWO_SCORE    = 106
_CENTER_BONUS = 66
_CORNER_BONUS = 6

_CENTERS = [(1, 1), (1, 4), (4, 1), (4, 4)]
_CORNERS = [(0, 0), (0, 2), (0, 3), (0, 5),
            (2, 0), (2, 2), (2, 3), (2, 5),
            (3, 0), (3, 2), (3, 3), (3, 5),
            (5, 0), (5, 2), (5, 3), (5, 5)]


def _score_window_for(window: np.ndarray, player: int) -> int:
    opp   = WHITE if player == BLACK else BLACK
    ai_n  = int(np.sum(window == player))
    opp_n = int(np.sum(window == opp))
    if ai_n > 0 and opp_n > 0: return 0   # contested — no value to either
    if ai_n == 5: return _WIN_SCORE
    if ai_n == 4: return _FOUR_SCORE
    if ai_n == 3: return _THREE_SCORE
    if ai_n == 2: return _TWO_SCORE
    return 0


class PentagoAI:
    def __init__(self, ai_player: int, human_player: int):
        self.ai_player    = ai_player
        self.human_player = human_player

    # ── Heuristic evaluation ─────────────────────────────────────────────────
    def evaluate_board(self, board: np.ndarray) -> int:
        winner = check_winner(board)
        if winner == self.ai_player:    return  _WIN_SCORE
        if winner == self.human_player: return -_WIN_SCORE
        if winner == 0:                 return 0

        score = 0
        for window in _get_all_5_windows(board):
            score += _score_window_for(window, self.ai_player)
            score -= _score_window_for(window, self.human_player) * 1.4  # fear opponent more

        for r, c in _CENTERS:
            if   board[r, c] == self.ai_player:    score += _CENTER_BONUS
            elif board[r, c] == self.human_player: score -= _CENTER_BONUS * 2
        for r, c in _CORNERS:
            if   board[r, c] == self.ai_player:    score += _CORNER_BONUS
            elif board[r, c] == self.human_player: score -= _CORNER_BONUS * 2
        return score

    # ── Immediate-threat scanner ──────────────────────────────────────────────
    def _immediate_win_cell(self, board: np.ndarray, player: int):
        """
        Returns (row, col) if `player` can place there and win (with or
        without any subsequent rotation).  O(36 × 8) — very fast.
        """
        empties = list(zip(*np.where(board == EMPTY)))
        for (r, c) in empties:
            nb = place_marble(board, r, c, player)
            if check_winner(nb) == player:
                return (r, c)
            for q in range(4):
                for d in (1, -1):
                    if check_winner(rotate_quadrant(nb, q, d)) == player:
                        return (r, c)
        return None

    def _best_block_move(self, board: np.ndarray, threat_rc: tuple) -> tuple:
        """Block the threat cell with the rotation that leaves AI best off."""
        r, c = threat_rc
        best_score, best_move = -math.inf, None
        for q in range(4):
            for d in (1, -1):
                nb = apply_move(board, (r, c, q, d), self.ai_player)
                s  = self.evaluate_board(nb)
                if s > best_score:
                    best_score, best_move = s, (r, c, q, d)
        return best_move

    # ── Move ordering ─────────────────────────────────────────────────────────
    def _order_moves(self, moves: list, board: np.ndarray, maximizing: bool) -> list:
        player = self.ai_player if maximizing else self.human_player
        scored = [(self.evaluate_board(apply_move(board, m, player)), m)
                  for m in moves]
        scored.sort(key=lambda x: x[0], reverse=maximizing)
        return [m for _, m in scored]

    # ── Alpha-beta with deadline ──────────────────────────────────────────────
    def alpha_beta(self, board: np.ndarray, depth: int,
                   alpha: float, beta: float,
                   maximizing: bool,
                   deadline: float = math.inf):
        winner = check_winner(board)
        if depth == 0 or winner != -1 or is_board_full(board):
            return self.evaluate_board(board), None

        if _time.monotonic() > deadline:
            return self.evaluate_board(board), None

        moves = self._order_moves(legal_moves(board), board, maximizing)
        best_move = moves[0] if moves else None

        if maximizing:
            max_eval = -math.inf
            for move in moves:
                if _time.monotonic() > deadline: break
                nb = apply_move(board, move, self.ai_player)
                score, _ = self.alpha_beta(nb, depth - 1, alpha, beta, False, deadline)
                if score > max_eval:
                    max_eval, best_move = score, move
                alpha = max(alpha, score)
                if beta <= alpha: break
            return max_eval, best_move
        else:
            min_eval = math.inf
            for move in moves:
                if _time.monotonic() > deadline: break
                nb = apply_move(board, move, self.human_player)
                score, _ = self.alpha_beta(nb, depth - 1, alpha, beta, True, deadline)
                if score < min_eval:
                    min_eval, best_move = score, move
                beta = min(beta, score)
                if beta <= alpha: break
            return min_eval, best_move

    # ── Public entry point: iterative deepening ───────────────────────────────
    def get_best_move(self, board: np.ndarray, depth: int = 4, time_limit: float = 3.5) -> tuple:
        """
        Iterative-deepening alpha-beta.  Searches d=1,2,...,depth within
        time_limit seconds and returns the best move from the deepest
        completed iteration.
        """
        # Fast path A: AI can win immediately
        win_cell = self._immediate_win_cell(board, self.ai_player)
        if win_cell is not None:
            r, c = win_cell
            for q in range(4):
                for d in (1, -1):
                    if check_winner(apply_move(board, (r, c, q, d), self.ai_player)) == self.ai_player:
                        return (r, c, q, d)
            return (r, c, 0, 1)

        # Fast path B: block opponent immediate win
        threat_cell = self._immediate_win_cell(board, self.human_player)
        if threat_cell is not None:
            return self._best_block_move(board, threat_cell)

        # Iterative deepening
        deadline  = _time.monotonic() + time_limit
        best_move = None
        for d in range(1, depth + 1):
            if _time.monotonic() >= deadline: break
            _, move = self.alpha_beta(board, d, -math.inf, math.inf, True, deadline)
            if move is not None:
                best_move = move

        if best_move is None:
            moves = legal_moves(board)
            best_move = moves[0] if moves else (0, 0, 0, 1)
        return best_move


# ==========================================
# 3. PYGAME INIT & THEME
# ==========================================
pygame.init()

WIDTH, HEIGHT = 1060, 820
CELL_SIZE = 82
BOARD_SIZE = CELL_SIZE * 6
MARGIN_X = (WIDTH - BOARD_SIZE) // 2
MARGIN_Y = 115

C_BG_TOP        = (18,  12,  8)
C_BG_BTM        = (36,  22, 12)
C_BOARD_DARK    = (58,  32, 14)
C_BOARD_MID     = (78,  44, 18)
C_QUAD_LIGHT    = (92,  54, 22)
C_QUAD_BORDER   = (120, 76, 32)
C_DIVIDER       = (160,100, 40)
C_CELL_LINE     = (70,  42, 18)
C_GOLD          = (210,170, 60)
C_GOLD_DIM      = (140,108, 38)
C_TEXT_BRIGHT   = (240,220,170)
C_TEXT_MID      = (190,165,110)
C_TEXT_DIM      = (130,108, 70)
C_BTN_FACE      = (52,  30, 12)
C_BTN_HOVER     = (78,  48, 18)
C_BTN_BORDER    = (130, 88, 36)
C_BTN_HOT_FACE  = (80,  46, 14)
C_BTN_HOT_BDR   = (200,148, 50)
C_RED_BTN       = (90,  22, 18)
C_RED_HOVER     = (120, 32, 26)
C_RED_BORDER    = (180, 60, 50)
C_SHADOW        = (0,   0,  0, 120)

BLACK_MARBLE_OUTER = (28, 28, 35)
BLACK_MARBLE_MID   = (50, 50, 62)
BLACK_MARBLE_SHINE = (110,110,135)
WHITE_MARBLE_OUTER = (195,185,168)
WHITE_MARBLE_MID   = (238,232,218)
WHITE_MARBLE_SHINE = (255,253,248)

FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(
    __file__ if "__file__" in dir() else __import__("sys").argv[0])), "fonts")

def load_font(name, size, bold=False):
    path = os.path.join(FONT_DIR, name)
    if os.path.exists(path):
        return pygame.font.Font(path, size)
    return pygame.font.SysFont("Georgia", size, bold=bold)

font_title   = load_font("Cinzel.ttf", 38)
font_heading = load_font("Cinzel.ttf", 22)
font_label   = load_font("Cinzel.ttf", 15)
font_ui      = load_font("Raleway.ttf", 18)
font_ui_sm   = load_font("Raleway.ttf", 14)
font_symbol  = pygame.font.SysFont("segoeuisymbol", 20)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pentago — Group 8")


# ==========================================
# DRAWING UTILITIES
# ==========================================
def draw_gradient_rect(surface, rect, top_color, bot_color, radius=0):
    x, y, w, h = rect
    for i in range(h):
        t = i / max(h - 1, 1)
        r = int(top_color[0] + (bot_color[0] - top_color[0]) * t)
        g = int(top_color[1] + (bot_color[1] - top_color[1]) * t)
        b = int(top_color[2] + (bot_color[2] - top_color[2]) * t)
        pygame.draw.line(surface, (r, g, b), (x, y + i), (x + w - 1, y + i))

def draw_background(surface):
    draw_gradient_rect(surface, (0, 0, WIDTH, HEIGHT), C_BG_TOP, C_BG_BTM)
    noise_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    rng = np.random.default_rng(42)
    for _ in range(500):
        rx = int(rng.integers(0, WIDTH))
        ry = int(rng.integers(0, HEIGHT))
        alpha = int(rng.integers(8, 22))
        pygame.draw.circle(noise_surf, (200, 160, 80, alpha), (rx, ry), 1)
    surface.blit(noise_surf, (0, 0))

def draw_marble(surface, cx, cy, radius, player):
    if player == BLACK:
        layers = [
            (BLACK_MARBLE_OUTER, 0),
            (BLACK_MARBLE_MID,   radius * 0.55),
            (BLACK_MARBLE_SHINE, radius * 0.72),
        ]
        shine_off = (-radius * 0.28, -radius * 0.28)
        shine_r   = radius * 0.22
        shine_col = (165, 165, 195)
    else:
        layers = [
            (WHITE_MARBLE_OUTER, 0),
            (WHITE_MARBLE_MID,   radius * 0.55),
            (WHITE_MARBLE_SHINE, radius * 0.72),
        ]
        shine_off = (-radius * 0.28, -radius * 0.28)
        shine_r   = radius * 0.22
        shine_col = (255, 255, 255)

    shadow_surf = pygame.Surface((int(radius*2+8), int(radius*2+8)), pygame.SRCALPHA)
    pygame.draw.circle(shadow_surf, (0, 0, 0, 80),
                       (int(radius+4), int(radius+6)), int(radius))
    surface.blit(shadow_surf, (int(cx - radius), int(cy - radius + 2)))

    for i in range(int(radius), 0, -1):
        t = 1.0 - (i / radius)
        if t < 0.55:
            t2 = t / 0.55
            col = tuple(int(layers[0][0][j] + (layers[1][0][j] - layers[0][0][j]) * t2) for j in range(3))
        elif t < 0.72:
            t2 = (t - 0.55) / 0.17
            col = tuple(int(layers[1][0][j] + (layers[2][0][j] - layers[1][0][j]) * t2) for j in range(3))
        else:
            col = layers[2][0]
        pygame.draw.circle(surface, col, (cx, cy), i)

    sx = cx + int(shine_off[0])
    sy = cy + int(shine_off[1])
    shine_surf = pygame.Surface((int(shine_r*4), int(shine_r*4)), pygame.SRCALPHA)
    for ri in range(int(shine_r), 0, -1):
        alpha = int(200 * (1 - ri / shine_r) ** 1.5)
        pygame.draw.circle(shine_surf, (*shine_col, alpha),
                           (int(shine_r*2), int(shine_r*2)), ri)
    surface.blit(shine_surf, (sx - int(shine_r*2), sy - int(shine_r*2)))

def draw_gold_text(surface, text, font, cx, cy, color=None, shadow=True):
    col = color or C_GOLD
    if shadow:
        sh = font.render(text, True, (0, 0, 0))
        surface.blit(sh, sh.get_rect(center=(cx+2, cy+2)))
    surf = font.render(text, True, col)
    surface.blit(surf, surf.get_rect(center=(cx, cy)))

def draw_decorative_line(surface, x1, y1, x2, y2):
    pygame.draw.line(surface, C_GOLD_DIM, (x1, y1), (x2, y2), 1)
    pygame.draw.circle(surface, C_GOLD_DIM, (x1, y1), 3)
    pygame.draw.circle(surface, C_GOLD_DIM, (x2, y2), 3)

def draw_panel(surface, rect, radius=10):
    x, y, w, h = rect
    draw_gradient_rect(surface, rect, C_BOARD_MID, C_BOARD_DARK, radius)
    pygame.draw.rect(surface, C_QUAD_BORDER, rect, 2, border_radius=radius)
    pygame.draw.line(surface, C_QUAD_LIGHT, (x+3, y+3), (x+w-4, y+3), 1)
    pygame.draw.line(surface, C_QUAD_LIGHT, (x+3, y+3), (x+3, y+h-4), 1)


# ==========================================
# BOARD DRAWING
# ==========================================
QUAD_NAMES = ["Q1", "Q2", "Q3", "Q4"]

def draw_board(surface, game: GameState, animating_quad=-1):
    shadow_surf = pygame.Surface((BOARD_SIZE + 24, BOARD_SIZE + 24), pygame.SRCALPHA)
    pygame.draw.rect(shadow_surf, (0, 0, 0, 100),
                     (0, 0, BOARD_SIZE + 24, BOARD_SIZE + 24), border_radius=18)
    surface.blit(shadow_surf, (MARGIN_X - 12, MARGIN_Y - 12))

    board_rect = (MARGIN_X, MARGIN_Y, BOARD_SIZE, BOARD_SIZE)
    draw_gradient_rect(surface, board_rect, C_BOARD_MID, C_BOARD_DARK)
    pygame.draw.rect(surface, C_QUAD_BORDER,
                     pygame.Rect(*board_rect), 3, border_radius=12)

    for q in range(4):
        if q == animating_quad:
            continue
        qr, qc = QUAD_ORIGINS[q]
        quad_x = MARGIN_X + qc * CELL_SIZE
        quad_y = MARGIN_Y + qr * CELL_SIZE
        quad_rect = (quad_x + 2, quad_y + 2, 3*CELL_SIZE - 4, 3*CELL_SIZE - 4)

        draw_gradient_rect(surface, quad_rect, C_QUAD_LIGHT, C_BOARD_MID, 8)
        pygame.draw.rect(surface, C_QUAD_BORDER,
                         pygame.Rect(*quad_rect), 1, border_radius=8)

        wm_surf = font_title.render(QUAD_NAMES[q], True, C_GOLD)
        wm_surf.set_alpha(22)
        wm_rect = wm_surf.get_rect(center=(quad_x + 1.5*CELL_SIZE, quad_y + 1.5*CELL_SIZE))
        surface.blit(wm_surf, wm_rect)

        for r in range(3):
            for c in range(3):
                cx = quad_x + c * CELL_SIZE
                cy = quad_y + r * CELL_SIZE
                cell_rect = pygame.Rect(cx, cy, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(surface, C_CELL_LINE, cell_rect, 1)

                slot_cx = cx + CELL_SIZE // 2
                slot_cy = cy + CELL_SIZE // 2
                pygame.draw.circle(surface, C_BOARD_DARK, (slot_cx, slot_cy), CELL_SIZE // 2 - 10)
                pygame.draw.circle(surface, C_CELL_LINE,  (slot_cx, slot_cy), CELL_SIZE // 2 - 10, 1)

                val = game.board[qr + r, qc + c]
                if val != EMPTY:
                    draw_marble(surface, slot_cx, slot_cy,
                                CELL_SIZE // 2 - 11, val)

    lx = MARGIN_X + 3 * CELL_SIZE
    ly = MARGIN_Y + 3 * CELL_SIZE
    pygame.draw.line(surface, C_DIVIDER,
                     (MARGIN_X + 4, ly), (MARGIN_X + BOARD_SIZE - 4, ly), 4)
    pygame.draw.line(surface, C_DIVIDER,
                     (lx, MARGIN_Y + 4), (lx, MARGIN_Y + BOARD_SIZE - 4), 4)
    pygame.draw.circle(surface, C_GOLD, (lx, ly), 7)
    pygame.draw.circle(surface, C_BOARD_DARK, (lx, ly), 4)

def draw_quadrant_anim(surface, board, q, offset_x, offset_y):
    qr, qc = QUAD_ORIGINS[q]
    quad_rect = (0, 0, 3*CELL_SIZE, 3*CELL_SIZE)
    draw_gradient_rect(surface, quad_rect, C_QUAD_LIGHT, C_BOARD_MID)
    pygame.draw.rect(surface, C_QUAD_BORDER,
                     pygame.Rect(*quad_rect), 1, border_radius=8)

    for r in range(3):
        for c in range(3):
            cx = c * CELL_SIZE
            cy = r * CELL_SIZE
            cell_rect = pygame.Rect(cx, cy, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(surface, C_CELL_LINE, cell_rect, 1)
            slot_cx = cx + CELL_SIZE // 2
            slot_cy = cy + CELL_SIZE // 2
            pygame.draw.circle(surface, C_BOARD_DARK, (slot_cx, slot_cy), CELL_SIZE // 2 - 10)
            pygame.draw.circle(surface, C_CELL_LINE,  (slot_cx, slot_cy), CELL_SIZE // 2 - 10, 1)
            val = board[qr + r, qc + c]
            if val != EMPTY:
                draw_marble(surface, slot_cx, slot_cy,
                            CELL_SIZE // 2 - 11, val)

def get_board_cell(mouse_pos: tuple) -> Optional[tuple[int, int]]:
    x, y = mouse_pos
    if not (MARGIN_X <= x <= MARGIN_X + BOARD_SIZE and MARGIN_Y <= y <= MARGIN_Y + BOARD_SIZE):
        return None
    return (y - MARGIN_Y) // CELL_SIZE, (x - MARGIN_X) // CELL_SIZE


# ==========================================
# BUTTON CLASS
# ==========================================
class Button:
    def __init__(self, x, y, w, h, text, action_val=None,
                 danger=False, symbol=False):
        self.rect       = pygame.Rect(x, y, w, h)
        self.text       = text
        self.action_val = action_val
        self.danger     = danger
        self.symbol     = symbol

    def draw(self, surface):
        hovered = self.rect.collidepoint(pygame.mouse.get_pos())
        if self.danger:
            face   = C_RED_HOVER   if hovered else C_RED_BTN
            border = C_RED_BORDER
        else:
            face   = C_BTN_HOVER   if hovered else C_BTN_FACE
            border = C_BTN_HOT_BDR if hovered else C_BTN_BORDER

        draw_gradient_rect(surface, self.rect, face,
                           tuple(max(0, c - 20) for c in face))
        pygame.draw.rect(surface, border, self.rect, 2, border_radius=8)
        inner = self.rect.inflate(-4, -4)
        pygame.draw.line(surface, tuple(min(255, c + 30) for c in face),
                         (inner.x, inner.y), (inner.right, inner.y), 1)

        f   = font_symbol if self.symbol else font_label
        col = C_TEXT_BRIGHT if hovered else C_TEXT_MID
        ts  = f.render(self.text, True, col)
        tr  = ts.get_rect(center=self.rect.center)
        surface.blit(ts, tr)


# ==========================================
# MAIN LOOP
# ==========================================
def main():
    clock     = pygame.time.Clock()
    app_state = "MENU"

    # ---- Menu widgets ----
    input_box        = pygame.Rect(WIDTH//2 - 160, HEIGHT//2 - 155, 320, 42)
    color_btn_white  = Button(WIDTH//2 - 170, HEIGHT//2 - 88,  160, 42, "White  (1st)", WHITE)
    color_btn_black  = Button(WIDTH//2 +  10, HEIGHT//2 - 88,  160, 42, "Black  (2nd)", BLACK)
    bo1_btn          = Button(WIDTH//2 - 165, HEIGHT//2 - 22,  100, 38, "Best of 1", 1)
    bo3_btn          = Button(WIDTH//2 -  50, HEIGHT//2 - 22,  100, 38, "Best of 3", 3)
    bo5_btn          = Button(WIDTH//2 +  65, HEIGHT//2 - 22,  100, 38, "Best of 5", 5)
    start_btn        = Button(WIDTH//2 - 110, HEIGHT//2 + 40,  220, 52, "START SERIES")

    # ---- Series / game vars ----
    player_name         = ""
    active_input        = False
    base_human_color    = WHITE
    best_of_series      = 1
    target_wins         = 1
    human_wins          = 0
    ai_wins             = 0
    current_human_color = WHITE
    current_ai_color    = BLACK
    game = None
    ai   = None

    # ---- Rotation buttons ----
    btn_w, btn_h, btn_gap = 106, 46, 6
    grid_total_w = 4 * btn_w + 3 * btn_gap
    grid_x0 = (WIDTH - grid_total_w) // 2
    grid_y0 = MARGIN_Y + BOARD_SIZE + 16

    rot_buttons = []
    for qi in range(4):
        row      = qi // 2
        col_base = (qi % 2) * 2
        bx_cw    = grid_x0 + col_base       * (btn_w + btn_gap)
        bx_ccw   = grid_x0 + (col_base + 1) * (btn_w + btn_gap)
        by       = grid_y0 + row * (btn_h + btn_gap)
        label    = f"Q{qi+1}"
        rot_buttons.append(Button(bx_cw,  by, btn_w, btn_h,
                                  f"{label} ↻", (qi,  1), symbol=True))
        rot_buttons.append(Button(bx_ccw, by, btn_w, btn_h,
                                  f"{label} ↺", (qi, -1), symbol=True))

    # ---- In-game overlay buttons ----
    overlay_y     = grid_y0 + 2 * (btn_h + btn_gap) + 10
    surrender_btn = Button(WIDTH//2 - 220, overlay_y, 200, 40, "Surrender", danger=True)
    main_menu_btn = Button(WIDTH//2 +  20, overlay_y, 200, 40, "Main Menu",  danger=False)

    # ---- Post-game buttons ----
    next_match_btn   = Button(WIDTH//2 - 215, overlay_y, 200, 46, "Next Match")
    restart_menu_btn = Button(WIDTH//2 +  15, overlay_y, 200, 46, "Main Menu")
    full_restart_btn = Button(WIDTH//2 - 110, overlay_y, 220, 46, "Main Menu")

    # ---- Animation state ----
    animating    = False
    anim_q       = 0
    anim_d       = 0
    anim_angle   = 0
    target_angle = 0
    anim_surf    = None
    quad_center  = (0, 0)

    status_msg = ""

    bg_surf = pygame.Surface((WIDTH, HEIGHT))
    draw_background(bg_surf)

    running = True
    while running:
        mouse_pos = pygame.mouse.get_pos()
        screen.blit(bg_surf, (0, 0))

        # ==========================================
        # STATE: MAIN MENU
        # ==========================================
        if app_state == "MENU":
            frame = pygame.Rect(WIDTH//2 - 240, HEIGHT//2 - 280, 480, 400)
            draw_panel(screen, frame, radius=14)
            draw_decorative_line(screen, frame.x + 20, frame.y + 8,
                                 frame.right - 20, frame.y + 8)
            draw_decorative_line(screen, frame.x + 20, frame.bottom - 8,
                                 frame.right - 20, frame.bottom - 8)

            draw_gold_text(screen, "PENTAGO", font_title,  WIDTH//2, HEIGHT//2 - 240)
            draw_gold_text(screen, "Game Setup", font_heading,
                           WIDTH//2, HEIGHT//2 - 205, color=C_TEXT_MID)

            name_lbl = font_ui_sm.render("Player Name", True, C_TEXT_DIM)
            screen.blit(name_lbl, (input_box.x, input_box.y - 18))
            ib_color = C_GOLD if active_input else C_BTN_BORDER
            draw_gradient_rect(screen, input_box, C_BTN_FACE,
                               tuple(max(0, c - 10) for c in C_BTN_FACE))
            pygame.draw.rect(screen, ib_color, input_box, 2, border_radius=6)
            name_surf = font_ui.render(player_name + ("|" if active_input else ""),
                                       True, C_TEXT_BRIGHT)
            screen.blit(name_surf, (input_box.x + 10, input_box.y + 10))

            col_lbl = font_ui_sm.render("Choose Color", True, C_TEXT_DIM)
            screen.blit(col_lbl, (color_btn_white.rect.x, color_btn_white.rect.y - 18))
            color_btn_white.draw(screen)
            color_btn_black.draw(screen)
            active_col_rect = (color_btn_white.rect if base_human_color == WHITE
                               else color_btn_black.rect)
            pygame.draw.rect(screen, C_GOLD, active_col_rect, 2, border_radius=8)

            ser_lbl = font_ui_sm.render("Series Format", True, C_TEXT_DIM)
            screen.blit(ser_lbl, (bo1_btn.rect.x, bo1_btn.rect.y - 18))
            bo1_btn.draw(screen); bo3_btn.draw(screen); bo5_btn.draw(screen)
            for btn, val in [(bo1_btn, 1), (bo3_btn, 3), (bo5_btn, 5)]:
                if best_of_series == val:
                    pygame.draw.rect(screen, C_GOLD, btn.rect, 2, border_radius=8)

            start_btn.draw(screen)

            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if input_box.collidepoint(event.pos): active_input = True
                    else: active_input = False
                    if color_btn_white.rect.collidepoint(event.pos): base_human_color = WHITE
                    if color_btn_black.rect.collidepoint(event.pos): base_human_color = BLACK
                    if bo1_btn.rect.collidepoint(event.pos): best_of_series = 1
                    if bo3_btn.rect.collidepoint(event.pos): best_of_series = 3
                    if bo5_btn.rect.collidepoint(event.pos): best_of_series = 5
                    if start_btn.rect.collidepoint(event.pos):
                        if not player_name: player_name = "Player"
                        human_wins = ai_wins = 0
                        target_wins         = (best_of_series // 2) + 1
                        current_human_color = base_human_color
                        current_ai_color    = BLACK if current_human_color == WHITE else WHITE
                        game = GameState()
                        ai   = PentagoAI(ai_player=current_ai_color,
                                         human_player=current_human_color)
                        app_state = "PLAYING"
                if event.type == pygame.KEYDOWN and active_input:
                    if event.key == pygame.K_BACKSPACE: player_name = player_name[:-1]
                    elif len(player_name) < 15: player_name += event.unicode

        # ==========================================
        # STATES: PLAYING / ROUND_OVER / SERIES_OVER
        # ==========================================
        elif app_state in ("PLAYING", "ROUND_OVER", "SERIES_OVER", "EVALUATE_WIN"):

            # ---- Header ----
            score_txt = f"{player_name}  {human_wins}  —  {ai_wins}  AI"
            draw_gold_text(screen, "PENTAGO", font_heading, WIDTH//2, 28)
            draw_gold_text(screen, score_txt, font_ui, WIDTH//2, 58, color=C_TEXT_MID)
            c_str = "White" if current_human_color == WHITE else "Black"
            draw_gold_text(screen, f"You are {c_str}  ·  Best of {best_of_series}",
                           font_ui_sm, WIDTH//2, 80, color=C_TEXT_DIM)
            draw_decorative_line(screen, MARGIN_X, 95, MARGIN_X + BOARD_SIZE, 95)

            # ---- Update status message ----
            if app_state == "PLAYING" and not animating:
                curr_name = player_name if game.turn == current_human_color else "AI"
                if game.phase == "place":
                    status_msg = f"{curr_name}  ·  Place a marble"
                else:
                    status_msg = f"{curr_name}  ·  Rotate a quadrant"

            # ---- AI Move ----
            if app_state == "PLAYING" and game.turn == current_ai_color and not animating:
                if game.phase == "place":
                    draw_board(screen, game)
                    think_surf = font_heading.render("AI is thinking…", True, C_GOLD)
                    think_rect = think_surf.get_rect(
                        centerx=WIDTH//2,
                        centery=grid_y0 + btn_h + btn_gap // 2
                    )
                    pad = 16
                    bg = pygame.Surface((think_surf.get_width() + pad*2,
                                         think_surf.get_height() + pad), pygame.SRCALPHA)
                    bg.fill((0, 0, 0, 160))
                    screen.blit(bg, (think_rect.x - pad, think_rect.y - pad // 2))
                    screen.blit(think_surf, think_rect)
                    pygame.display.flip()

                    # ---- AI uses improved get_best_move (IDDFS, depth=3) ----
                    r, c, q, d = ai.get_best_move(game.board, depth=3, time_limit=2.5)
                    game.place(r, c)

                    if not game.is_over():
                        animating    = True
                        anim_q, anim_d = q, d
                        anim_angle   = 0
                        target_angle = -90 if anim_d == 1 else 90
                        anim_surf    = pygame.Surface((3*CELL_SIZE, 3*CELL_SIZE),
                                                      pygame.SRCALPHA)
                        draw_quadrant_anim(anim_surf, game.board, anim_q, 0, 0)
                        qr, qc = QUAD_ORIGINS[anim_q]
                        quad_center = (MARGIN_X + qc*CELL_SIZE + 1.5*CELL_SIZE,
                                       MARGIN_Y + qr*CELL_SIZE + 1.5*CELL_SIZE)
                    else:
                        app_state = "EVALUATE_WIN"
                    continue

            # ---- Render board ----
            if animating:
                step = -10 if target_angle < 0 else 10
                anim_angle += step
                draw_board(screen, game, animating_quad=anim_q)
                rotated_surf = pygame.transform.rotate(anim_surf, anim_angle)
                rotated_rect = rotated_surf.get_rect(center=quad_center)
                screen.blit(rotated_surf, rotated_rect.topleft)

                if (step < 0 and anim_angle <= target_angle) or \
                   (step > 0 and anim_angle >= target_angle):
                    animating = False
                    next_turn = (current_ai_color
                                 if game.turn == current_human_color
                                 else current_human_color)
                    game.rotate(anim_q, anim_d, next_turn)
                    if game.is_over():
                        app_state = "EVALUATE_WIN"
            else:
                draw_board(screen, game)

            # ---- Evaluate win ----
            if app_state == "EVALUATE_WIN":
                if game.winner == current_human_color:
                    human_wins += 1
                    status_msg = f"🎉  {player_name} wins the round!"
                elif game.winner == current_ai_color:
                    ai_wins += 1
                    status_msg = "🤖  AI wins the round!"
                else:
                    status_msg = "🤝  Round drawn!"
                app_state = ("SERIES_OVER"
                             if human_wins == target_wins or ai_wins == target_wins
                             else "ROUND_OVER")

            # ---- Rotation buttons ----
            if app_state == "PLAYING" and game.phase == "rotate" \
                    and game.turn == current_human_color and not animating:
                for btn in rot_buttons:
                    btn.draw(screen)

            # ---- In-game surrender / menu ----
            if app_state == "PLAYING" and not animating:
                surrender_btn.draw(screen)
                main_menu_btn.draw(screen)

            # ---- Post-round / series buttons ----
            if app_state == "ROUND_OVER":
                next_match_btn.draw(screen)
                restart_menu_btn.draw(screen)
            elif app_state == "SERIES_OVER":
                if human_wins > ai_wins:
                    status_msg = f"🏆  {player_name} wins the series!"
                else:
                    status_msg = "💀  AI wins the series!"
                full_restart_btn.draw(screen)

            # ---- Status text ----
            draw_gold_text(screen, status_msg, font_heading,
                           WIDTH//2, HEIGHT - 30,
                           color=C_TEXT_BRIGHT if app_state == "PLAYING" else C_GOLD)

            # ---- Event handling ----
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False

                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:

                    if app_state == "PLAYING" and not animating:
                        if surrender_btn.rect.collidepoint(event.pos):
                            ai_wins += 1
                            status_msg = "You surrendered. AI wins the round."
                            app_state = ("SERIES_OVER"
                                         if ai_wins == target_wins
                                         else "ROUND_OVER")
                        elif main_menu_btn.rect.collidepoint(event.pos):
                            app_state = "MENU"

                    if app_state == "PLAYING" and not animating \
                            and game.turn == current_human_color:
                        if game.phase == "place":
                            cell = get_board_cell(mouse_pos)
                            if cell and game.place(cell[0], cell[1]):
                                if game.is_over():
                                    app_state = "EVALUATE_WIN"
                        elif game.phase == "rotate":
                            for btn in rot_buttons:
                                if btn.rect.collidepoint(event.pos):
                                    anim_q, anim_d = btn.action_val
                                    animating    = True
                                    anim_angle   = 0
                                    target_angle = -90 if anim_d == 1 else 90
                                    anim_surf = pygame.Surface(
                                        (3*CELL_SIZE, 3*CELL_SIZE), pygame.SRCALPHA)
                                    draw_quadrant_anim(anim_surf, game.board,
                                                       anim_q, 0, 0)
                                    qr, qc = QUAD_ORIGINS[anim_q]
                                    quad_center = (
                                        MARGIN_X + qc*CELL_SIZE + 1.5*CELL_SIZE,
                                        MARGIN_Y + qr*CELL_SIZE + 1.5*CELL_SIZE)
                                    break

                    elif app_state == "ROUND_OVER":
                        if next_match_btn.rect.collidepoint(event.pos):
                            current_human_color = (BLACK if current_human_color == WHITE
                                                   else WHITE)
                            current_ai_color    = (BLACK if current_human_color == WHITE
                                                   else WHITE)
                            game = GameState()
                            ai   = PentagoAI(ai_player=current_ai_color,
                                             human_player=current_human_color)
                            app_state = "PLAYING"
                        elif restart_menu_btn.rect.collidepoint(event.pos):
                            app_state = "MENU"

                    elif app_state == "SERIES_OVER":
                        if full_restart_btn.rect.collidepoint(event.pos):
                            app_state = "MENU"

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()