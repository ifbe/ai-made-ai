# game.py
import random
from functools import lru_cache
import re

BOARD_SIZE = 15
BOARD_POSITIONS = BOARD_SIZE * BOARD_SIZE
EMPTY = 0
BLACK = 1
WHITE = 2

def pos_to_str(pos):
    """将内部索引转换为显示坐标 (0-e 0-e)"""
    row = pos // BOARD_SIZE
    col = pos % BOARD_SIZE
    
    def num_to_char(n):
        if n < 10:
            return str(n)
        else:
            return chr(ord('a') + (n - 10))
    
    return f"{num_to_char(col)}{num_to_char(row)}"

def str_to_pos(s):
    """将显示坐标转换为内部索引"""
    s = s.lower().strip()
    if len(s) != 2:
        return None
    
    def char_to_num(c):
        if c.isdigit():
            return int(c)
        elif 'a' <= c <= 'e':
            return 10 + (ord(c) - ord('a'))
        else:
            return -1
    
    col = char_to_num(s[0])
    row = char_to_num(s[1])
    
    if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
        return row * BOARD_SIZE + col
    return None

def check_win(board, player):
    """检查五子连珠"""
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
    
    for i in range(BOARD_SIZE * BOARD_SIZE):
        if board[i] != player:
            continue
        row, col = i // BOARD_SIZE, i % BOARD_SIZE
        
        for dr, dc in directions:
            count = 1
            # 正方向
            for step in range(1, 5):
                nr, nc = row + dr * step, col + dc * step
                if nr < 0 or nr >= BOARD_SIZE or nc < 0 or nc >= BOARD_SIZE:
                    break
                if board[nr * BOARD_SIZE + nc] != player:
                    break
                count += 1
            # 反方向
            for step in range(1, 5):
                nr, nc = row - dr * step, col - dc * step
                if nr < 0 or nr >= BOARD_SIZE or nc < 0 or nc >= BOARD_SIZE:
                    break
                if board[nr * BOARD_SIZE + nc] != player:
                    break
                count += 1
            
            if count >= 5:
                return True
    return False

def is_full(board):
    return all(x != EMPTY for x in board)

def get_legal_moves(board):
    """获取合法位置"""
    return [i for i in range(BOARD_SIZE * BOARD_SIZE) if board[i] == EMPTY]

def get_nearby_moves(board, distance=2):
    """获取已有棋子附近的空位"""
    stones = [i for i in range(BOARD_SIZE * BOARD_SIZE) if board[i] != EMPTY]
    
    if not stones:
        center = BOARD_SIZE // 2
        return [center * BOARD_SIZE + center]
    
    nearby = set()
    for pos in stones:
        r, c = pos // BOARD_SIZE, pos % BOARD_SIZE
        for dr in range(-distance, distance + 1):
            for dc in range(-distance, distance + 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                    idx = nr * BOARD_SIZE + nc
                    if board[idx] == EMPTY:
                        nearby.add(idx)
    
    return list(nearby)

def get_nearby_positions(board, pos, distance=2):
    """获取某个位置附近的空位"""
    r, c = pos // BOARD_SIZE, pos % BOARD_SIZE
    nearby = []
    
    for dr in range(-distance, distance + 1):
        for dc in range(-distance, distance + 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                idx = nr * BOARD_SIZE + nc
                if board[idx] == EMPTY:
                    nearby.append(idx)
    
    return list(set(nearby))

def get_empty_near_stones(board, distance=2):
    """获取所有棋子附近的空位"""
    stones = [i for i in range(BOARD_SIZE * BOARD_SIZE) if board[i] != EMPTY]
    
    if not stones:
        center = BOARD_SIZE // 2
        return [center * BOARD_SIZE + center]
    
    nearby = set()
    for pos in stones:
        nearby.update(get_nearby_positions(board, pos, distance))
    
    return list(nearby)

def count_in_direction(board, pos, dr, dc, player):
    """计算某个方向上的连续棋子数"""
    r, c = pos // BOARD_SIZE, pos % BOARD_SIZE
    count = 0
    
    for step in range(1, 5):
        nr, nc = r + dr * step, c + dc * step
        if nr < 0 or nr >= BOARD_SIZE or nc < 0 or nc >= BOARD_SIZE:
            break
        if board[nr * BOARD_SIZE + nc] == player:
            count += 1
        else:
            break
    
    return count

def get_all_directions():
    return [(1, 0), (0, 1), (1, 1), (1, -1)]

def is_five_in_a_row(board, pos, player):
    """快速检查是否形成五连"""
    r, c = pos // BOARD_SIZE, pos % BOARD_SIZE
    
    for dr, dc in get_all_directions():
        count = 1
        
        for step in range(1, 5):
            nr, nc = r + dr * step, c + dc * step
            if nr < 0 or nr >= BOARD_SIZE or nc < 0 or nc >= BOARD_SIZE:
                break
            if board[nr * BOARD_SIZE + nc] == player:
                count += 1
            else:
                break
        
        for step in range(1, 5):
            nr, nc = r - dr * step, c - dc * step
            if nr < 0 or nr >= BOARD_SIZE or nc < 0 or nc >= BOARD_SIZE:
                break
            if board[nr * BOARD_SIZE + nc] == player:
                count += 1
            else:
                break
        
        if count >= 5:
            return True
    
    return False
