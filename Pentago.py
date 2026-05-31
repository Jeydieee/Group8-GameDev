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
        # White always goes first per the new rules
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
# 2. AI MODULE (Using Raw Engine)
# ==========================================
class PentagoAI:
    def __init__(self, ai_player, human_player):
        self.ai_player = ai_player
        self.human_player = human_player

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
# 3. PYGAME UI & STATE MACHINE
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

try:
    font = pygame.font.SysFont("segoeuisymbol", 22)
except:
    font = pygame.font.Font(None, 22)
large_font = pygame.font.SysFont("segoeui", 32, bold=True)
small_font = pygame.font.SysFont("segoeui", 20)
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pentago AI - Group 8")

class Button:
    def __init__(self, x, y, w, h, text, action_val=None):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.action_val = action_val

    def draw(self, surface):
        mouse_pos = pygame.mouse.get_pos()
        color = BTN_HOVER if self.rect.collidepoint(mouse_pos) else BTN_COLOR
        pygame.draw.rect(surface, color, self.rect, border_radius=8)
        pygame.draw.rect(surface, BTN_OUTLINE, self.rect, width=1, border_radius=8)
        
        # Use standard font if it's regular text, symbol font if it has arrows
        f = font if '↺' in self.text or '↻' in self.text else small_font
        text_surf = f.render(self.text, True, TEXT_COLOR)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)

def draw_quadrant(surface, board, q, offset_x, offset_y):
    """Draws a single 3x3 quadrant onto a given surface."""
    pygame.draw.rect(surface, (235, 230, 220), (0, 0, 3*CELL_SIZE, 3*CELL_SIZE), border_radius=10)
    qr, qc = QUAD_ORIGINS[q]
    
    for r in range(3):
        for c in range(3):
            cell_rect = pygame.Rect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(surface, LINE_COLOR, cell_rect, 1)
            
            val = board[qr + r, qc + c]
            if val != EMPTY:
                center = cell_rect.center
                color = BLACK_MARBLE if val == BLACK else WHITE_MARBLE
                pygame.draw.circle(surface, color, center, CELL_SIZE // 2 - 10)
                shadow_color = (70, 70, 70) if val == BLACK else (220, 220, 220)
                pygame.draw.circle(surface, shadow_color, (center[0]-5, center[1]-5), CELL_SIZE // 2 - 25)

def draw_board(surface, game: GameState, animating_quad=-1):
    board_rect = pygame.Rect(MARGIN_X, MARGIN_Y, BOARD_SIZE, BOARD_SIZE)
    pygame.draw.rect(surface, (235, 230, 220), board_rect, border_radius=10)

    for q in range(4):
        if q == animating_quad:
            continue
            
        qr, qc = QUAD_ORIGINS[q]
        quad_x = MARGIN_X + qc * CELL_SIZE
        quad_y = MARGIN_Y + qr * CELL_SIZE
        
        for r in range(3):
            for c in range(3):
                cell_rect = pygame.Rect(quad_x + c * CELL_SIZE, quad_y + r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(surface, LINE_COLOR, cell_rect, 1)
                
                val = game.board[qr + r, qc + c]
                if val != EMPTY:
                    center = cell_rect.center
                    color = BLACK_MARBLE if val == BLACK else WHITE_MARBLE
                    pygame.draw.circle(surface, color, center, CELL_SIZE // 2 - 10)
                    shadow_color = (70, 70, 70) if val == BLACK else (220, 220, 220)
                    pygame.draw.circle(surface, shadow_color, (center[0]-5, center[1]-5), CELL_SIZE // 2 - 25)

    pygame.draw.line(surface, QUAD_LINE_COLOR, (MARGIN_X, MARGIN_Y + 3*CELL_SIZE), (MARGIN_X + BOARD_SIZE, MARGIN_Y + 3*CELL_SIZE), 4)
    pygame.draw.line(surface, QUAD_LINE_COLOR, (MARGIN_X + 3*CELL_SIZE, MARGIN_Y), (MARGIN_X + 3*CELL_SIZE, MARGIN_Y + BOARD_SIZE), 4)

def get_board_cell(mouse_pos: tuple) -> Optional[tuple[int, int]]:
    x, y = mouse_pos
    if not (MARGIN_X <= x <= MARGIN_X + BOARD_SIZE and MARGIN_Y <= y <= MARGIN_Y + BOARD_SIZE):
        return None
    return (y - MARGIN_Y) // CELL_SIZE, (x - MARGIN_X) // CELL_SIZE

def main():
    clock = pygame.time.Clock()
    
    # State Control
    app_state = "MENU"
    
    # Menu UI Elements
    input_box = pygame.Rect(WIDTH//2 - 150, HEIGHT//2 - 150, 300, 40)
    color_btn_white = Button(WIDTH//2 - 160, HEIGHT//2 - 50, 150, 40, "White (Goes 1st)", WHITE)
    color_btn_black = Button(WIDTH//2 + 10, HEIGHT//2 - 50, 150, 40, "Black (Goes 2nd)", BLACK)
    
    bo1_btn = Button(WIDTH//2 - 160, HEIGHT//2 + 20, 90, 40, "Best of 1", 1)
    bo3_btn = Button(WIDTH//2 - 50, HEIGHT//2 + 20, 90, 40, "Best of 3", 3)
    bo5_btn = Button(WIDTH//2 + 60, HEIGHT//2 + 20, 90, 40, "Best of 5", 5)
    start_btn = Button(WIDTH//2 - 100, HEIGHT//2 + 100, 200, 50, "START SERIES")

    # In-Game Setup
    player_name = ""
    active_input = False
    base_human_color = WHITE
    best_of_series = 1
    target_wins = 1
    
    # Series tracking
    human_wins = 0
    ai_wins = 0
    current_human_color = WHITE
    current_ai_color = BLACK
    
    game = None
    ai = None
    
    # Rotation UI
    QUAD_LABELS = ["Q1", "Q2", "Q3", "Q4"]
    btn_y = MARGIN_Y + BOARD_SIZE + 30
    btn_w, btn_gap = 70, 5
    rot_buttons = []
    for i, label in enumerate(QUAD_LABELS):
        base_x = MARGIN_X + i * (btn_w * 2 + btn_gap * 3)
        rot_buttons.append(Button(base_x, btn_y, btn_w, 50, f"{label} ↺", (i, -1)))
        rot_buttons.append(Button(base_x + btn_w + btn_gap, btn_y, btn_w, 50, f"{label} ↻", (i,  1)))

    # Post-Game UI
    next_match_btn = Button(WIDTH//2 - 160, HEIGHT - 100, 150, 50, "Next Match")
    restart_menu_btn = Button(WIDTH//2 + 10, HEIGHT - 100, 150, 50, "Main Menu")
    full_restart_btn = Button(WIDTH//2 - 100, HEIGHT - 100, 200, 50, "Main Menu")

    # Animation Tracking
    animating = False
    anim_q = 0
    anim_d = 0
    anim_angle = 0
    target_angle = 0
    anim_surf = None
    quad_center = (0, 0)
    
    status_msg = ""

    running = True
    while running:
        screen.fill(BG_COLOR)
        mouse_pos = pygame.mouse.get_pos()
        
        # ==========================================
        # STATE: MAIN MENU
        # ==========================================
        if app_state == "MENU":
            title = large_font.render("Pentago Game Setup", True, TEXT_COLOR)
            screen.blit(title, (WIDTH//2 - title.get_width()//2, HEIGHT//2 - 230))
            
            txt_surface = small_font.render("Enter Name: " + player_name, True, TEXT_COLOR)
            screen.blit(txt_surface, (input_box.x + 5, input_box.y + 5))
            pygame.draw.rect(screen, (100,100,250) if active_input else BTN_OUTLINE, input_box, 2)
            
            color_btn_white.draw(screen)
            color_btn_black.draw(screen)
            bo1_btn.draw(screen)
            bo3_btn.draw(screen)
            bo5_btn.draw(screen)
            start_btn.draw(screen)
            
            # Highlights
            active_col = color_btn_white.rect if base_human_color == WHITE else color_btn_black.rect
            pygame.draw.rect(screen, (0,200,0), active_col, 3, border_radius=8)
            
            if best_of_series == 1: pygame.draw.rect(screen, (0,200,0), bo1_btn.rect, 3, border_radius=8)
            elif best_of_series == 3: pygame.draw.rect(screen, (0,200,0), bo3_btn.rect, 3, border_radius=8)
            elif best_of_series == 5: pygame.draw.rect(screen, (0,200,0), bo5_btn.rect, 3, border_radius=8)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    
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
                        # Initialize series data
                        human_wins = 0
                        ai_wins = 0
                        target_wins = (best_of_series // 2) + 1
                        current_human_color = base_human_color
                        current_ai_color = BLACK if current_human_color == WHITE else WHITE
                        
                        game = GameState()
                        ai = PentagoAI(ai_player=current_ai_color, human_player=current_human_color)
                        app_state = "PLAYING"
                        
                if event.type == pygame.KEYDOWN and active_input:
                    if event.key == pygame.K_BACKSPACE: player_name = player_name[:-1]
                    elif len(player_name) < 15: player_name += event.unicode

        # ==========================================
        # STATE: PLAYING OR ANIMATING
        # ==========================================
        elif app_state in ["PLAYING", "ROUND_OVER", "SERIES_OVER"]:
            # Header texts
            score_txt = large_font.render(f"Series: {player_name} {human_wins} - {ai_wins} AI (Best of {best_of_series})", True, TEXT_COLOR)
            screen.blit(score_txt, (MARGIN_X, 20))
            
            c_str = "White" if current_human_color == WHITE else "Black"
            info_txt = small_font.render(f"You are playing as {c_str}", True, (120, 120, 120))
            screen.blit(info_txt, (MARGIN_X, 60))

            if app_state == "PLAYING" and not animating:
                curr_name = player_name if game.turn == current_human_color else "AI"
                if game.phase == "place": status_msg = f"{curr_name}'s Turn: Place a marble."
                else: status_msg = f"{curr_name}'s Turn: Rotate a quadrant."

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if app_state == "PLAYING" and not animating:
                        if game.turn == current_human_color:
                            if game.phase == "place":
                                cell = get_board_cell(mouse_pos)
                                if cell:
                                    if game.place(cell[0], cell[1]):
                                        # Check if placing won the game
                                        if game.is_over(): app_state = "EVALUATE_WIN"
                            
                            elif game.phase == "rotate":
                                for btn in rot_buttons:
                                    if btn.rect.collidepoint(mouse_pos):
                                        anim_q, anim_d = btn.action_val
                                        animating = True
                                        anim_angle = 0
                                        target_angle = -90 if anim_d == 1 else 90
                                        
                                        anim_surf = pygame.Surface((3*CELL_SIZE, 3*CELL_SIZE), pygame.SRCALPHA)
                                        draw_quadrant(anim_surf, game.board, anim_q, 0, 0)
                                        
                                        qr, qc = QUAD_ORIGINS[anim_q]
                                        quad_center = (MARGIN_X + qc*CELL_SIZE + 1.5*CELL_SIZE, MARGIN_Y + qr*CELL_SIZE + 1.5*CELL_SIZE)
                                        break
                                        
                    elif app_state == "ROUND_OVER":
                        if next_match_btn.rect.collidepoint(mouse_pos):
                            # Alternate colors for rematch
                            current_human_color = BLACK if current_human_color == WHITE else WHITE
                            current_ai_color = BLACK if current_human_color == WHITE else WHITE
                            game = GameState()
                            ai = PentagoAI(ai_player=current_ai_color, human_player=current_human_color)
                            app_state = "PLAYING"
                        elif restart_menu_btn.rect.collidepoint(mouse_pos):
                            app_state = "MENU"
                            
                    elif app_state == "SERIES_OVER":
                        if full_restart_btn.rect.collidepoint(mouse_pos):
                            app_state = "MENU"

            # AI Move Calculation
            if app_state == "PLAYING" and game.turn == current_ai_color and not animating:
                if game.phase == "place":
                    status_surf = large_font.render("AI is thinking...", True, TEXT_COLOR)
                    screen.blit(status_surf, status_surf.get_rect(centerx=screen.get_rect().centerx, y=HEIGHT - 120))
                    pygame.display.flip()
                    
                    r, c, q, d = ai.get_best_move(game.board, depth=2)
                    game.place(r, c)
                    
                    if not game.is_over():
                        animating = True
                        anim_q, anim_d = q, d
                        anim_angle = 0
                        target_angle = -90 if anim_d == 1 else 90
                        
                        anim_surf = pygame.Surface((3*CELL_SIZE, 3*CELL_SIZE), pygame.SRCALPHA)
                        draw_quadrant(anim_surf, game.board, anim_q, 0, 0)
                        qr, qc = QUAD_ORIGINS[anim_q]
                        quad_center = (MARGIN_X + qc*CELL_SIZE + 1.5*CELL_SIZE, MARGIN_Y + qr*CELL_SIZE + 1.5*CELL_SIZE)
                    else:
                        app_state = "EVALUATE_WIN"

            # Render logic for PLAYING, ROUND_OVER, SERIES_OVER
            if animating:
                step = -10 if target_angle < 0 else 10
                anim_angle += step
                
                draw_board(screen, game, animating_quad=anim_q)
                
                rotated_surf = pygame.transform.rotate(anim_surf, anim_angle)
                rotated_rect = rotated_surf.get_rect(center=quad_center)
                screen.blit(rotated_surf, rotated_rect.topleft)
                
                if (step < 0 and anim_angle <= target_angle) or (step > 0 and anim_angle >= target_angle):
                    animating = False
                    next_turn = current_ai_color if game.turn == current_human_color else current_human_color
                    game.rotate(anim_q, anim_d, next_turn)
                    
                    if game.is_over():
                        app_state = "EVALUATE_WIN"
            else:
                draw_board(screen, game)
                if app_state == "PLAYING" and game.phase == "rotate":
                    for btn in rot_buttons:
                        btn.draw(screen)

            # Check logic for intermediate state evaluation
            if app_state == "EVALUATE_WIN":
                if game.winner == current_human_color:
                    human_wins += 1
                    status_msg = f"{player_name} Wins this round! 🎉"
                elif game.winner == current_ai_color:
                    ai_wins += 1
                    status_msg = "AI Wins this round! 🤖"
                else:
                    status_msg = "Round is a Draw! 🤝"
                
                if human_wins == target_wins or ai_wins == target_wins:
                    app_state = "SERIES_OVER"
                else:
                    app_state = "ROUND_OVER"

            # Draw buttons depending on end states
            if app_state == "ROUND_OVER":
                next_match_btn.draw(screen)
                restart_menu_btn.draw(screen)
            elif app_state == "SERIES_OVER":
                if human_wins > ai_wins: status_msg = f"🏆 {player_name} WINS THE SERIES! 🏆"
                else: status_msg = f"💀 AI WINS THE SERIES! 💀"
                full_restart_btn.draw(screen)

            # Draw Status Message
            status_surf = large_font.render(status_msg, True, TEXT_COLOR)
            status_rect = status_surf.get_rect(centerx=screen.get_rect().centerx, y=HEIGHT - 60 if app_state == "PLAYING" else HEIGHT - 160)
            screen.blit(status_surf, status_rect)

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()