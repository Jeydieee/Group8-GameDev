import numpy as np
import random
import math
import time as _time
import sys
from multiprocessing import Pool

from pentago_engine import (
    new_board, apply_move, check_winner,
    is_board_full, legal_moves, place_marble,
    rotate_quadrant, _get_all_5_windows,
    _CENTERS, _CORNERS,
    BLACK, WHITE, EMPTY
)

# ── GA Configuration ────────────────────────────────────────────
POP_SIZE       = 20     # number of agents per generation
GENERATIONS    = 15     # how many generations to evolve
GAMES_PER_PAIR = 2      # games per matchup (each side plays both colors)
MUTATION_RATE  = 0.3    # probability of mutating each weight
MUTATION_STD   = 0.15   # how much to mutate (as fraction of current value)
SEARCH_DEPTH   = 2      # depth for self-play
SEARCH_TIME    = 0.5    # time limit per move in self-play (seconds)

# Weight index mapping
I_FOUR   = 0
I_THREE  = 1
I_TWO    = 2
I_CENTER = 3
I_CORNER = 4

WIN_SCORE = 100_000     # fixed terminal value, never evolved


# ── Agent ────────────────────────────────────────────────────────
class Agent:
    def __init__(self, weights: np.ndarray):
        self.weights = weights.copy()
        self.wins    = 0
        self.losses  = 0
        self.draws   = 0

    @property
    def fitness(self):
        total = self.wins + self.losses + self.draws
        return self.wins / total if total > 0 else 0.0

    def evaluate(self, board: np.ndarray, ai_player: int) -> float:
        human_player = WHITE if ai_player == BLACK else BLACK
        winner = check_winner(board)
        if winner == ai_player:    return  WIN_SCORE
        if winner == human_player: return -WIN_SCORE
        if winner == 0:            return  0

        four, three, two, center_b, corner_b = self.weights
        score = 0.0

        for window in _get_all_5_windows(board):
            ai_n  = int(np.sum(window == ai_player))
            opp_n = int(np.sum(window == human_player))
            if ai_n > 0 and opp_n > 0:
                continue
            if ai_n == 4:    score += four
            elif ai_n == 3:  score += three
            elif ai_n == 2:  score += two
            if opp_n == 4:   score -= four   * 1.4
            elif opp_n == 3: score -= three  * 1.4
            elif opp_n == 2: score -= two    * 1.4

        for r, c in _CENTERS:
            if   board[r, c] == ai_player:    score += center_b
            elif board[r, c] == human_player: score -= center_b * 2
        for r, c in _CORNERS:
            if   board[r, c] == ai_player:    score += corner_b
            elif board[r, c] == human_player: score -= corner_b * 2

        return score

    def get_move(self, board: np.ndarray, ai_player: int) -> tuple:
        human_player = WHITE if ai_player == BLACK else BLACK
        deadline  = _time.monotonic() + SEARCH_TIME
        best_move = None

        for depth in range(1, SEARCH_DEPTH + 1):
            if _time.monotonic() >= deadline:
                break
            _, move = self._alpha_beta(
                board, depth, -math.inf, math.inf,
                True, ai_player, human_player, deadline
            )
            if move is not None:
                best_move = move

        if best_move is None:
            moves = legal_moves(board)
            best_move = moves[0] if moves else (0, 0, 0, 1)
        return best_move

    def _alpha_beta(self, board, depth, alpha, beta,
                    maximizing, ai_player, human_player, deadline):
        winner = check_winner(board)
        if depth == 0 or winner != -1 or is_board_full(board):
            return self.evaluate(board, ai_player), None
        if _time.monotonic() > deadline:
            return self.evaluate(board, ai_player), None

        moves = legal_moves(board)
        player = ai_player if maximizing else human_player
        moves.sort(
            key=lambda m: self.evaluate(apply_move(board, m, player), ai_player),
            reverse=maximizing
        )

        best_move = moves[0] if moves else None

        if maximizing:
            max_eval = -math.inf
            for move in moves:
                if _time.monotonic() > deadline: break
                nb = apply_move(board, move, ai_player)
                score, _ = self._alpha_beta(nb, depth - 1, alpha, beta,
                                            False, ai_player, human_player, deadline)
                if score > max_eval:
                    max_eval, best_move = score, move
                alpha = max(alpha, score)
                if beta <= alpha: break
            return max_eval, best_move
        else:
            min_eval = math.inf
            for move in moves:
                if _time.monotonic() > deadline: break
                nb = apply_move(board, move, human_player)
                score, _ = self._alpha_beta(nb, depth - 1, alpha, beta,
                                            True, ai_player, human_player, deadline)
                if score < min_eval:
                    min_eval, best_move = score, move
                beta = min(beta, score)
                if beta <= alpha: break
            return min_eval, best_move


# ── Self-play: one full game between two agents ──────────────────
def play_game(agent_black: Agent, agent_white: Agent) -> int:
    """Returns BLACK, WHITE, or 0 (draw)."""
    sys.stdout.flush()
    board      = new_board()
    turn       = BLACK
    phase      = "place"
    pending_rc = None

    for _ in range(36 * 2):
        winner = check_winner(board)
        if winner != -1:
            return winner
        if is_board_full(board):
            return 0

        agent = agent_black if turn == BLACK else agent_white
        opp   = WHITE if turn == BLACK else BLACK

        if phase == "place":
            r, c, q, d = agent.get_move(board, turn)
            board = place_marble(board, r, c, turn)
            w = check_winner(board)
            if w != -1:
                return w
            phase      = "rotate"
            pending_rc = (r, c, q, d)
        else:
            _, _, q, d = pending_rc
            board = rotate_quadrant(board, q, d)
            w = check_winner(board)
            if w != -1:
                return w
            if is_board_full(board):
                return 0
            turn  = opp
            phase = "place"

    return 0


# ── Multiprocessed pair runner ───────────────────────────────────
def _play_pair(args):
    """Plays two games between agent i and agent j. Standalone for multiprocessing."""
    i, j, w_i, w_j = args
    agent_i = Agent(w_i)
    agent_j = Agent(w_j)
    results = []
    results.append(play_game(agent_i, agent_j))   # i=BLACK, j=WHITE
    results.append(play_game(agent_j, agent_i))   # j=BLACK, i=WHITE
    return i, j, results


# ── Round-robin tournament ───────────────────────────────────────
def tournament(population: list):
    for agent in population:
        agent.wins = agent.losses = agent.draws = 0

    pairs = []
    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            pairs.append((i, j,
                          population[i].weights.copy(),
                          population[j].weights.copy()))

    # Kaggle gives 4 CPU cores — use all of them
    with Pool(processes=4) as pool:
        results = pool.map(_play_pair, pairs)

    for i, j, game_results in results:
        r0 = game_results[0]   # i=BLACK, j=WHITE
        if r0 == BLACK:
            population[i].wins   += 1; population[j].losses += 1
        elif r0 == WHITE:
            population[j].wins   += 1; population[i].losses += 1
        else:
            population[i].draws  += 1; population[j].draws  += 1

        r1 = game_results[1]   # j=BLACK, i=WHITE
        if r1 == BLACK:
            population[j].wins   += 1; population[i].losses += 1
        elif r1 == WHITE:
            population[i].wins   += 1; population[j].losses += 1
        else:
            population[i].draws  += 1; population[j].draws  += 1


# ── GA operators ─────────────────────────────────────────────────
def random_weights() -> np.ndarray:
    return np.array([
        random.uniform(20_000, 80_000),   # four
        random.uniform(500,     5_000),   # three
        random.uniform(20,        300),   # two
        random.uniform(10,        100),   # center
        random.uniform(5,          50),   # corner
    ])

def crossover(parent_a: Agent, parent_b: Agent) -> Agent:
    mask    = np.random.rand(5) > 0.5
    child_w = np.where(mask, parent_a.weights, parent_b.weights)
    return Agent(child_w)

def mutate(agent: Agent) -> Agent:
    new_w = agent.weights.copy()
    for i in range(len(new_w)):
        if random.random() < MUTATION_RATE:
            delta  = np.random.normal(0, abs(new_w[i]) * MUTATION_STD)
            new_w[i] = max(1.0, new_w[i] + delta)
    return Agent(new_w)


# ── Main GA loop ─────────────────────────────────────────────────
def run_ga():
    print("Initializing population...")
    sys.stdout.flush()

    population = [Agent(random_weights()) for _ in range(POP_SIZE)]

    # Seed agent 0 with current hand-tuned weights as baseline
    population[0] = Agent(np.array([50_000, 1_500, 80, 40, 15], dtype=float))

    for gen in range(1, GENERATIONS + 1):
        print(f"\n── Generation {gen}/{GENERATIONS} ──")
        sys.stdout.flush()

        tournament(population)
        population.sort(key=lambda a: a.fitness, reverse=True)

        best = population[0]
        print(f"  Best fitness : {best.fitness:.3f}  "
              f"(W{best.wins} L{best.losses} D{best.draws})")
        print(f"  Best weights : "
              f"four={best.weights[I_FOUR]:.0f}  "
              f"three={best.weights[I_THREE]:.0f}  "
              f"two={best.weights[I_TWO]:.0f}  "
              f"center={best.weights[I_CENTER]:.0f}  "
              f"corner={best.weights[I_CORNER]:.0f}")
        sys.stdout.flush()

        # Elitism: keep top 25%, fill rest with offspring
        cutoff    = max(1, POP_SIZE // 4)
        survivors = population[:cutoff]
        offspring = []
        while len(offspring) < POP_SIZE - cutoff:
            pa, pb = random.sample(survivors, 2)
            child  = mutate(crossover(pa, pb))
            offspring.append(child)
        population = survivors + offspring

    # ── Final result ─────────────────────────────────────────────
    best = population[0]
    with open("evolved_weights.txt", "w") as f:
        f.write(f"_FOUR_SCORE   = {best.weights[I_FOUR]:.0f}\n")
        f.write(f"_THREE_SCORE  = {best.weights[I_THREE]:.0f}\n")
        f.write(f"_TWO_SCORE    = {best.weights[I_TWO]:.0f}\n")
        f.write(f"_CENTER_BONUS = {best.weights[I_CENTER]:.0f}\n")
        f.write(f"_CORNER_BONUS = {best.weights[I_CORNER]:.0f}\n")
    print("Weights saved to evolved_weights.txt")
    sys.stdout.flush()


if __name__ == "__main__":
    run_ga()