import chess

# Int values for each type, the higher the value the powerfull the piece is
PIECE_VALUES = {
    chess.PAWN: 1,  
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9
}

def material_balance(board: chess.Board) -> int:
    """ Util for computing the material balance of a chess position, which is the difference in total piece value between White and Black.
    This serves as a simple heuristic for evaluating the position and knowing the state of the current game.

    :param board: Chess board.
    :type board: chess.Board
    :return: Standard material balance score.
    :rtype: int
    """
    score = 0                                                           # Positive score favors White, negative favors Black
    for piece_type, value in PIECE_VALUES.items():                      
        score += len(board.pieces(piece_type, chess.WHITE)) * value     # Multiply the number of (piece_type) pieces by their value
        score -= len(board.pieces(piece_type, chess.BLACK)) * value     # Subtract the same for Black pieces to get the net material balance
    return score                               

def is_endgame(board: chess.Board) -> bool:
    """ Util for determining if a chess position is in the endgame phase, 
    which is typically characterized by a significant reduction in material on the board.
    This is important for the TimeManager to adjust time budgets based on the game phase, 
    as endgame positions often require more precise calculation and less exploration.

    :param board: Chess board to evaluate.
    :type board: chess.Board
    :return: True if the position is considered an endgame, False otherwise.
    :rtype: bool
    """
    queens = len(board.pieces(chess.QUEEN, chess.WHITE)) + \
             len(board.pieces(chess.QUEEN, chess.BLACK))                # Count total queens on the board
    minor_pieces = (
        len(board.pieces(chess.BISHOP, chess.WHITE)) +
        len(board.pieces(chess.BISHOP, chess.BLACK)) +                  # Count total bishops
        len(board.pieces(chess.KNIGHT, chess.WHITE)) +  
        len(board.pieces(chess.KNIGHT, chess.BLACK))                    # Count total knights
    )
    return queens == 0 or minor_pieces <= 2                             # Consider endgame if no queens or very few minor pieces remaining  
