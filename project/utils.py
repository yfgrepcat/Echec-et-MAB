import chess

# Int values for each type, the higher the value the powerfull the piece is
PIECE_VALUES = {
    chess.PAWN: 1,  
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9
}

# Material balance util for reward shaping
def material_balance(board):
    score = 0                                                           # Positive score favors White, negative favors Black
    for piece_type, value in PIECE_VALUES.items():                      
        score += len(board.pieces(piece_type, chess.WHITE)) * value     # Multiply the number of (piece_type) pieces by their value
        score -= len(board.pieces(piece_type, chess.BLACK)) * value     # Subtract the same for Black pieces to get the net material balance
    return score                                

# Util to detect endgame phase
def is_endgame(board):
    queens = len(board.pieces(chess.QUEEN, chess.WHITE)) + \
             len(board.pieces(chess.QUEEN, chess.BLACK))                # Count total queens on the board
    minor_pieces = (
        len(board.pieces(chess.BISHOP, chess.WHITE)) +
        len(board.pieces(chess.BISHOP, chess.BLACK)) +                  # Count total bishops
        len(board.pieces(chess.KNIGHT, chess.WHITE)) +  
        len(board.pieces(chess.KNIGHT, chess.BLACK))                    # Count total knights
    )
    return queens == 0 or minor_pieces <= 2                             # Consider endgame if no queens or very few minor pieces remaining  
    