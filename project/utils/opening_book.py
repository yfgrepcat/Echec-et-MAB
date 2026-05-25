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


def apply_random_opening(board: chess.Board, openings: list[str]) -> chess.Board:
    """Performs a legal random opening.

    :param board: The chess board to apply the opening moves on.
    :type board: chess.Board
    :param openings: A list of openings, where each opening is a string of space-separated moves in UCI format.
    :type openings: list of str
    :return: The updated chess board after applying the opening moves.
    :rtype: chess.Board
    """

    line = choice(openings)

    moves = [move.strip() for move in line.split() if move.strip()]

    for move in moves:
        chess_move = chess.Move.from_uci(move)

        if chess_move in board.legal_moves:
            board.push(chess_move)

        else:
            break

    return board


def load_random_middlegame() -> chess.Board:
    """ Loads a chess game that is already on a middlegame position, randomly chosen from a list of FENs.
    This can be used to start training from more complex positions than the initial chess position, 
    which can help the model learn more quickly and effectively by exposing it to a wider 
    variety of positions and scenarios.

    :return: A chess board initialized to a random middlegame position.
    :rtype: chess.Board
    """
    path = os.path.join(BASE_DIR, "positions", "middlegame_fens.txt")

    with open(path, "r") as f:
        positions = [line.strip() for line in f if line.strip()]

    fen = choice(positions)

    return chess.Board(fen)