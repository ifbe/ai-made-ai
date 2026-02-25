# game.py
import random
from functools import lru_cache

def board_to_str(board):
    s = ""
    for i in range(9):
        c = '.' if board[i]==0 else 'X' if board[i]==1 else 'O'
        s += c
        if (i+1)%3 == 0:
            s += "\n"
    return s.strip()

def check_win(board, player):
    win_combs = [
        [0,1,2],[3,4,5],[6,7,8],
        [0,3,6],[1,4,7],[2,5,8],
        [0,4,8],[2,4,6]
    ]
    for comb in win_combs:
        if all(board[i] == player for i in comb):
            return True
    return False

def is_full(board):
    return all(x != 0 for x in board)

def get_legal_moves(board):
    return [i for i in range(9) if board[i] == 0]

def is_winning_move(board, player, move):
    if move not in get_legal_moves(board):
        return False
    board[move] = player
    result = check_win(board, player)
    board[move] = 0
    return result

def is_threat_move(board, player, move):
    opponent = 3 - player
    if move not in get_legal_moves(board):
        return False
    board[move] = opponent
    result = check_win(board, opponent)
    board[move] = 0
    return result

@lru_cache(maxsize=200000)
def minimax_ab_cached(board_tuple, player, alpha=-float('inf'), beta=float('inf')):
    board = list(board_tuple)
    opponent = 3 - player

    if check_win(board, player):
        return 10 if player == 1 else -10, None
    if check_win(board, opponent):
        return -10 if player == 1 else 10, None
    if is_full(board):
        return 0, None

    best_move = None

    if player == 1:
        best_score = -float('inf')
        for pos in get_legal_moves(board):
            board[pos] = player
            score, _ = minimax_ab_cached(tuple(board), opponent, alpha, beta)
            board[pos] = 0
            if score > best_score:
                best_score = score
                best_move = pos
            alpha = max(alpha, score)
            if beta <= alpha:
                break
    else:
        best_score = float('inf')
        for pos in get_legal_moves(board):
            board[pos] = player
            score, _ = minimax_ab_cached(tuple(board), opponent, alpha, beta)
            board[pos] = 0
            if score < best_score:
                best_score = score
                best_move = pos
            beta = min(beta, score)
            if beta <= alpha:
                break

    return best_score, best_move

def minimax_move(board, player):
    board_tuple = tuple(board)
    _, pos = minimax_ab_cached(board_tuple, player)
    return pos

def generate_expert_game(start_player=1, opponent_is_random=False):
    board = [0] * 9
    player = start_player
    data = []

    while True:
        legals = get_legal_moves(board)
        if not legals:
            break

        if player == start_player:
            pos = minimax_move(board, player)
        else:
            if opponent_is_random:
                pos = random.choice(legals)
            else:
                pos = minimax_move(board, player)

        if pos is None:
            break

        data.append((board[:], pos, None))

        board[pos] = player

        if check_win(board, player):
            value = 1 if player == 1 else -1
            for i in range(len(data)):
                data[i] = (data[i][0], data[i][1], value)
            return data

        if is_full(board):
            for i in range(len(data)):
                data[i] = (data[i][0], data[i][1], 0)
            return data

        player = 3 - player

    for i in range(len(data)):
        data[i] = (data[i][0], data[i][1], 0)
    return data
