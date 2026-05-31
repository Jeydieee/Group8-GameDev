import pygame
import numpy as np
from typing import Optional
import math
import sys

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
        self.turn = BLACK
        self.phase = "place"
        self.pending_place: Optional[tuple[int, int]] = None
        self.winner = -1
        self.move_count = 0
        self.history: list[dict] = []

    def _snapshot(self) -> dict:
        return {
            "board": copy_board(self.board), "turn": self.turn,
            "phase": self.phase, "pending_place": self.pending_place,
            "winner": self.winner, "move_count": self.move_count,
        }

    def place(self, row: int, col: int) -> bool:
        if self.winner != -1 or self.phase != "place": return False
        if self.board[row, col] != EMPTY: return False
        self.history.append(self._snapshot())
        self.board = place_marble(self.board, row, col, self.turn)
        w = check_winner(self.board)
        if w >= 0:
            self.winner = w
            return True
        self.phase = "rotate"
        self.pending_place = (row, col)
        return True

    def rotate(self, quad: int, direction: int) -> bool:
        if self.winner != -1 or self.phase != "rotate": return False
        self.board = rotate_quadrant(self.board, quad, direction)
        w = check_winner(self.board)
        if w >= 0: self.winner = w
        elif is_board_full(self.board): self.winner = 0
        self.turn = WHITE if self.turn == BLACK else BLACK
        self.phase = "place"
        self.pending_place = None
        self.move_count += 1
        return True

    def apply_full_move(self, move: tuple) -> bool:
        r, c, q, d = move
        if not self.place(r, c): return False
        if self.winner != -1: return True
        return self.rotate(q, d)
        
    def is_over(self) -> bool:
        return self.winner != -1

# ==========================================
# 2. AI MODULE (Using Raw Engine)
# ==========================================
class PentagoAI:
    def __init__(self, ai_player=WHITE):
        self.ai_player = ai_player
        self.human_player = BLACK if ai_player == WHITE else WHITE

    def evaluate_board(self, board_state: np.ndarray):
        winner = check_winner(board_state)
        if winner == self.ai_player: return 100000
        if winner == self.human_player: return -100000
        if winner == 0: return 0
        
        score = 0
        centers = [(1,1), (1,4), (4,1), (4,4)]
        for r, c in centers:
            if board_state[r, c] == self.ai_player: score += 10
            elif board_state[r, c] == self.human_player: score -= 10
        return score

    def alpha_beta(self, board_state: np.ndarray, depth: int, alpha: float, beta: float, maximizing_player: bool):
        winner = check_winner(board_state)
        if depth == 0 or winner != -1:
            return self.evaluate_board(board_state), None

        moves = legal_moves(board_state)
        best_move = None

        if maximizing_player:
            max_eval = -math.inf
            for move in moves:
                new_board = apply_move(board_state, move, self.ai_player)
                eval_score, _ = self.alpha_beta(new_board, depth - 1, alpha, beta, False)
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                alpha = max(alpha, eval_score)
                if beta <= alpha: break
            return max_eval, best_move
        else:
            min_eval = math.inf
            for move in moves:
                new_board = apply_move(board_state, move, self.human_player)
                eval_score, _ = self.alpha_beta(new_board, depth - 1, alpha, beta, True)
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                beta = min(beta, eval_score)
                if beta <= alpha: break
            return min_eval, best_move

    def get_best_move(self, board_state: np.ndarray, depth=2):
        score, move = self.alpha_beta(board_state, depth, -math.inf, math.inf, True)
        return move

# ==========================================
# 3. PYGAME UI
# ==========================================
pygame.init()

WIDTH, HEIGHT = 1000, 750
CELL_SIZE = 80
BOARD_SIZE = CELL_SIZE * 6
MARGIN_X = (WIDTH - BOARD_SIZE) // 2
MARGIN_Y = 100

BG_COLOR = (245, 245, 240)
LINE_COLOR = (200, 200, 190)
QUAD_LINE_COLOR = (180, 180, 170)
BLACK_MARBLE = (40, 40, 40)
WHITE_MARBLE = (250, 250, 250)
TEXT_COLOR = (60, 60, 60)
BTN_COLOR = (255, 255, 255)
BTN_HOVER = (235, 235, 235)
BTN_OUTLINE = (200, 200, 200)

# Using Windows built-in symbol font for the curved arrows
font = pygame.font.SysFont("segoeuisymbol", 22)
large_font = pygame.font.SysFont("segoeui", 32, bold=True)
screen = pygame.display.set_mode((WIDTH, HEIGHT))
screen_rect = screen.get_rect()
pygame.display.set_caption("Pentago AI - Group 8")


class Button:
    def __init__(self, x, y, w, h, text, action_val):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.action_val = action_val

    def draw(self, surface):
        mouse_pos = pygame.mouse.get_pos()
        color = BTN_HOVER if self.rect.collidepoint(mouse_pos) else BTN_COLOR
        pygame.draw.rect(surface, color, self.rect, border_radius=8)
        pygame.draw.rect(surface, BTN_OUTLINE, self.rect, width=1, border_radius=8)
        text_surf = font.render(self.text, True, TEXT_COLOR)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)

def draw_board(surface, game: GameState):
    board_rect = pygame.Rect(MARGIN_X, MARGIN_Y, BOARD_SIZE, BOARD_SIZE)
    pygame.draw.rect(surface, (235, 230, 220), board_rect, border_radius=10)

    for r in range(6):
        for c in range(6):
            cell_rect = pygame.Rect(MARGIN_X + c * CELL_SIZE, MARGIN_Y + r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(surface, LINE_COLOR, cell_rect, 1)
            
            val = game.board[r, c]
            if val != EMPTY:
                center = cell_rect.center
                color = BLACK_MARBLE if val == BLACK else WHITE_MARBLE
                pygame.draw.circle(surface, color, center, CELL_SIZE // 2 - 10)
                shadow_color = (70, 70, 70) if val == BLACK else (220, 220, 220)
                pygame.draw.circle(surface, shadow_color, (center[0]-5, center[1]-5), CELL_SIZE // 2 - 25)

    pygame.draw.line(surface, QUAD_LINE_COLOR, (MARGIN_X, MARGIN_Y + 3*CELL_SIZE), (MARGIN_X + BOARD_SIZE, MARGIN_Y + 3*CELL_SIZE), 4)
    pygame.draw.line(surface, QUAD_LINE_COLOR, (MARGIN_X + 3*CELL_SIZE, MARGIN_Y), (MARGIN_X + 3*CELL_SIZE, MARGIN_Y + BOARD_SIZE), 4)

    q_labels = [("Q1", 0, 0), ("Q2", 3, 0), ("Q3", 0, 3), ("Q4", 3, 3)]
    for text, cx, cy in q_labels:
        # Revert to standard segoeui for the board quadrant labels so they match the title
        lbl = pygame.font.SysFont("segoeui", 20).render(text, True, (150, 150, 150))
        surface.blit(lbl, (MARGIN_X + cx*CELL_SIZE + 5, MARGIN_Y + cy*CELL_SIZE + 5))

def main():
    game = GameState()
    ai = PentagoAI(ai_player=WHITE)
    status_msg = "Game Started. You are Black. Place a marble."

    btn_y = MARGIN_Y + BOARD_SIZE + 30
    btn_w = 70
    
    buttons = [
        Button(MARGIN_X + 0,   btn_y, btn_w, 50, "Q1 ↺", (0, -1)),
        Button(MARGIN_X + 75,  btn_y, btn_w, 50, "Q1 ↻",  (0, 1)),
        Button(MARGIN_X + 160, btn_y, btn_w, 50, "Q2 ↺", (1, -1)),
        Button(MARGIN_X + 235, btn_y, btn_w, 50, "Q2 ↻",  (1, 1)),
        Button(MARGIN_X + 320, btn_y, btn_w, 50, "Q3 ↺", (2, -1)),
        Button(MARGIN_X + 395, btn_y, btn_w, 50, "Q3 ↻",  (2, 1)),
        Button(MARGIN_X + 480, btn_y, btn_w, 50, "Q4 ↺", (3, -1)),
        Button(MARGIN_X + 555, btn_y, btn_w, 50, "Q4 ↻",  (3, 1)),
    ]

    clock = pygame.time.Clock()
    running = True

    while running:
        screen.fill(BG_COLOR)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and not game.is_over():
                mouse_pos = event.pos
                
                if game.turn == BLACK and game.phase == "place":
                    if MARGIN_X <= mouse_pos[0] <= MARGIN_X + BOARD_SIZE and MARGIN_Y <= mouse_pos[1] <= MARGIN_Y + BOARD_SIZE:
                        c = (mouse_pos[0] - MARGIN_X) // CELL_SIZE
                        r = (mouse_pos[1] - MARGIN_Y) // CELL_SIZE
                        if game.place(r, c):
                            status_msg = "Marble placed. Now select a quadrant to rotate."

                elif game.turn == BLACK and game.phase == "rotate":
                    for btn in buttons:
                        if btn.rect.collidepoint(mouse_pos):
                            q, d = btn.action_val
                            if game.rotate(q, d):
                                status_msg = "AI is thinking..."
                            break

        # AI Turn
        if game.turn == WHITE and not game.is_over():
            draw_board(screen, game)
            status_surf = large_font.render(status_msg, True, TEXT_COLOR)
            status_rect = status_surf.get_rect(centerx=screen_rect.centerx, y=30)
            screen.blit(status_surf, status_rect)
            pygame.display.flip()
            
            # AI uses the raw numpy board from the state
            best_move = ai.get_best_move(game.board, depth=2)
            game.apply_full_move(best_move)
            
            if not game.is_over():
                status_msg = "AI played. Your turn."

        draw_board(screen, game)
        
        if game.phase == "rotate" and not game.is_over():
            for btn in buttons: btn.draw(screen)

        if game.is_over():
            if game.winner == BLACK: status_msg = "Game Over! You Win! 🎉"
            elif game.winner == WHITE: status_msg = "Game Over! AI Wins! 🤖"
            else: status_msg = "Game Over! It's a Draw! 🤝"
            
        status_surf = large_font.render(status_msg, True, TEXT_COLOR)
        status_rect = status_surf.get_rect(centerx=screen_rect.centerx, y=30)
        screen.blit(status_surf, status_rect)

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()