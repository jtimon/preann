/*
 * chessBoard.h
 *
 * Chess board implementation for PREANN
 * Supports full chess rules including castling, en passant, promotion, check, and checkmate
 */

#ifndef CHESS_BOARD_H_
#define CHESS_BOARD_H_

#include "board.h"
#include "genetic/individual.h"
#include <vector>
#include <utility>

// Piece types for chess
enum PieceType {
    NO_PIECE = 0,
    PAWN = 1,
    KNIGHT = 2,
    BISHOP = 3,
    ROOK = 4,
    QUEEN = 5,
    KING = 6
};

// Combined piece representation (type + owner)
struct ChessPiece {
    PieceType type;
    SquareState owner;  // EMPTY, PLAYER_1, or PLAYER_2

    ChessPiece() : type(NO_PIECE), owner(EMPTY) {}
    ChessPiece(PieceType t, SquareState o) : type(t), owner(o) {}

    bool isEmpty() const { return type == NO_PIECE || owner == EMPTY; }
};

// Move representation for history tracking
struct ChessMove {
    unsigned fromX, fromY;
    unsigned toX, toY;
    ChessPiece movedPiece;
    ChessPiece capturedPiece;
    bool wasCastle;
    bool wasEnPassant;
    bool wasPromotion;

    ChessMove() : fromX(0), fromY(0), toX(0), toY(0),
                  wasCastle(false), wasEnPassant(false), wasPromotion(false) {}
};

class ChessBoard : public Board
{
private:
    // Chess-specific board representation
    ChessPiece** pieces;  // 8x8 array of pieces

    // Game state tracking
    bool whiteCanCastleKingside;
    bool whiteCanCastleQueenside;
    bool blackCanCastleKingside;
    bool blackCanCastleQueenside;

    // En passant state
    int enPassantColumn;  // -1 if no en passant available, otherwise column of pawn that just moved
    SquareState enPassantPlayer;  // Which player can capture en passant

    // Move history
    std::vector<ChessMove> moveHistory;

    // Helper methods for piece movement
    bool isValidPosition(int x, int y) const;
    bool isPieceAt(unsigned x, unsigned y, PieceType type, SquareState owner) const;
    void setPieceAt(unsigned x, unsigned y, const ChessPiece& piece);

    // Move validation helpers
    bool canPawnMove(unsigned fromX, unsigned fromY, unsigned toX, unsigned toY, SquareState player) const;
    bool canKnightMove(unsigned fromX, unsigned fromY, unsigned toX, unsigned toY) const;
    bool canBishopMove(unsigned fromX, unsigned fromY, unsigned toX, unsigned toY) const;
    bool canRookMove(unsigned fromX, unsigned fromY, unsigned toX, unsigned toY) const;
    bool canQueenMove(unsigned fromX, unsigned fromY, unsigned toX, unsigned toY) const;
    bool canKingMove(unsigned fromX, unsigned fromY, unsigned toX, unsigned toY) const;

    // Path checking
    bool isPathClear(unsigned fromX, unsigned fromY, unsigned toX, unsigned toY) const;

    // Check detection
    bool isSquareAttacked(unsigned x, unsigned y, SquareState byPlayer) const;
    bool isInCheck(SquareState player) const;
    bool wouldBeInCheck(unsigned fromX, unsigned fromY, unsigned toX, unsigned toY, SquareState player) const;

    // Castling helpers
    bool canCastleKingside(SquareState player) const;
    bool canCastleQueenside(SquareState player) const;

    // King position finding
    bool findKing(SquareState player, unsigned& kingX, unsigned& kingY) const;

    // Legal move generation
    void getAllLegalMovesForPiece(unsigned x, unsigned y, std::vector<std::pair<unsigned, unsigned>>& moves) const;
    bool hasAnyLegalMoves(SquareState player) const;

public:
    ChessBoard(unsigned size, BufferType bufferType);
    ChessBoard(ChessBoard* other);
    virtual ~ChessBoard();

    virtual void initBoard();
    virtual bool legalMove(unsigned xPos, unsigned yPos, SquareState player);
    virtual void makeMove(unsigned xPos, unsigned yPos, SquareState player);
    virtual float computerEstimation(unsigned xPos, unsigned yPos, SquareState player);
    virtual float individualEstimation(unsigned xPos, unsigned yPos, SquareState player, Individual* individual);

    // Override to handle chess-specific game ending
    virtual bool endGame();

    // Override to provide 768-input piece-aware encoding
    virtual Interface* updateInterface();

    // Note: countPoints() inherited from Board (just counts pieces, not used for chess scoring)

    // Chess-specific methods
    bool isCheckmate(SquareState player) const;
    bool isStalemate(SquareState player) const;
    float getMaterialValue(SquareState player) const;

    // Piece access for display purposes
    ChessPiece getPieceAt(unsigned x, unsigned y) const;

    // Board visualization
    void printBoard(std::ostream& out) const;
};

#endif /* CHESS_BOARD_H_ */
