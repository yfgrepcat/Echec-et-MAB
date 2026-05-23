import chess
from random import choice
import os

BASE_DIR = os.path.dirname(__file__)


def load_openings(path=None):

    if path is None:
        path = os.path.join(BASE_DIR, "books", "openings.txt")

    openings = []

    with open(path, "r") as f:

        for line in f:

            line = line.strip()

            if line:
                openings.append(line)

    return openings


def apply_random_opening(board, openings):

    line = choice(openings)

    moves = [
        move.strip()
        for move in line.split()
        if move.strip()
    ]

    for move in moves:

        chess_move = chess.Move.from_uci(move)

        if chess_move in board.legal_moves:
            board.push(chess_move)

        else:
            break

    return board


def load_random_middlegame():
    path = os.path.join(BASE_DIR, "positions", "middlegame_fens.txt")

    with open(path, "r") as f:

        positions = [
            line.strip()
            for line in f
            if line.strip()
        ]

    fen = choice(positions)

    return chess.Board(fen)
