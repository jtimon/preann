/*
 * chessBoard.cpp
 *
 * Chess board implementation with full rules support
 */

#include "chessBoard.h"
#include <cassert>
#include <cstring>
#include <cmath>
#include <algorithm>

using namespace std;

// ============================================================================
// Constructors and Destructor
// ============================================================================

ChessBoard::ChessBoard(unsigned size, BufferType bufferType) :
        Board(size, bufferType),
        whiteCanCastleKingside(true), whiteCanCastleQueenside(true),
        blackCanCastleKingside(true), blackCanCastleQueenside(true),
        enPassantColumn(-1), enPassantPlayer(EMPTY)
{
    // Chess board must be 8x8
    if (size != 8) {
        string error = "Chess board must be 8x8. Provided size: " + to_string(size);
        throw error;
    }

    // Recreate interface with correct size for chess (768 inputs = 8x8x12)
    // Base Board constructor created it with size * size * 2 (128), we need size * size * 12 (768)
    delete tInterface;
    tInterface = new Interface(size * size * 12, bufferType);

    // Allocate piece array
    pieces = new ChessPiece*[8];
    for (int i = 0; i < 8; i++) {
        pieces[i] = new ChessPiece[8];
    }
}

ChessBoard::ChessBoard(ChessBoard* other) :
        Board(other),
        whiteCanCastleKingside(other->whiteCanCastleKingside),
        whiteCanCastleQueenside(other->whiteCanCastleQueenside),
        blackCanCastleKingside(other->blackCanCastleKingside),
        blackCanCastleQueenside(other->blackCanCastleQueenside),
        enPassantColumn(other->enPassantColumn),
        enPassantPlayer(other->enPassantPlayer)
{
    // Recreate interface with correct size for chess (768 inputs = 8x8x12)
    // Base Board constructor created it with size * size * 2 (128), we need size * size * 12 (768)
    delete tInterface;
    tInterface = new Interface(8 * 8 * 12, other->getBufferType());

    // Allocate and copy piece array
    pieces = new ChessPiece*[8];
    for (int i = 0; i < 8; i++) {
        pieces[i] = new ChessPiece[8];
        for (int j = 0; j < 8; j++) {
            pieces[i][j] = other->pieces[i][j];
        }
    }

    // Copy move history
    moveHistory = other->moveHistory;
}

ChessBoard::~ChessBoard()
{
    for (int i = 0; i < 8; i++) {
        delete[] pieces[i];
    }
    delete[] pieces;
}

// ============================================================================
// Board Initialization
// ============================================================================

void ChessBoard::initBoard()
{
    // Clear the inherited board representation
    Board::initBoard();

    // Set up standard chess starting position
    // Black pieces (PLAYER_2) on rows 0-1
    // White pieces (PLAYER_1) on rows 6-7

    // Clear all squares first
    for (int x = 0; x < 8; x++) {
        for (int y = 0; y < 8; y++) {
            pieces[x][y] = ChessPiece(NO_PIECE, EMPTY);
            tBoard[x][y] = EMPTY;
        }
    }

    // Black back rank (row 0)
    pieces[0][0] = ChessPiece(ROOK, PLAYER_2);
    pieces[1][0] = ChessPiece(KNIGHT, PLAYER_2);
    pieces[2][0] = ChessPiece(BISHOP, PLAYER_2);
    pieces[3][0] = ChessPiece(QUEEN, PLAYER_2);
    pieces[4][0] = ChessPiece(KING, PLAYER_2);
    pieces[5][0] = ChessPiece(BISHOP, PLAYER_2);
    pieces[6][0] = ChessPiece(KNIGHT, PLAYER_2);
    pieces[7][0] = ChessPiece(ROOK, PLAYER_2);

    // Black pawns (row 1)
    for (int x = 0; x < 8; x++) {
        pieces[x][1] = ChessPiece(PAWN, PLAYER_2);
    }

    // White pawns (row 6)
    for (int x = 0; x < 8; x++) {
        pieces[x][6] = ChessPiece(PAWN, PLAYER_1);
    }

    // White back rank (row 7)
    pieces[0][7] = ChessPiece(ROOK, PLAYER_1);
    pieces[1][7] = ChessPiece(KNIGHT, PLAYER_1);
    pieces[2][7] = ChessPiece(BISHOP, PLAYER_1);
    pieces[3][7] = ChessPiece(QUEEN, PLAYER_1);
    pieces[4][7] = ChessPiece(KING, PLAYER_1);
    pieces[5][7] = ChessPiece(BISHOP, PLAYER_1);
    pieces[6][7] = ChessPiece(KNIGHT, PLAYER_1);
    pieces[7][7] = ChessPiece(ROOK, PLAYER_1);

    // Update the base board representation
    for (int x = 0; x < 8; x++) {
        for (int y = 0; y < 8; y++) {
            tBoard[x][y] = pieces[x][y].owner;
        }
    }

    // Reset game state
    whiteCanCastleKingside = true;
    whiteCanCastleQueenside = true;
    blackCanCastleKingside = true;
    blackCanCastleQueenside = true;
    enPassantColumn = -1;
    enPassantPlayer = EMPTY;
    moveHistory.clear();
}

// ============================================================================
// Helper Methods
// ============================================================================

bool ChessBoard::isValidPosition(int x, int y) const
{
    return x >= 0 && x < 8 && y >= 0 && y < 8;
}

bool ChessBoard::isPieceAt(unsigned x, unsigned y, PieceType type, SquareState owner) const
{
    if (!isValidPosition(x, y)) return false;
    return pieces[x][y].type == type && pieces[x][y].owner == owner;
}

ChessPiece ChessBoard::getPieceAt(unsigned x, unsigned y) const
{
    if (!isValidPosition(x, y)) return ChessPiece();
    return pieces[x][y];
}

void ChessBoard::setPieceAt(unsigned x, unsigned y, const ChessPiece& piece)
{
    if (!isValidPosition(x, y)) return;
    pieces[x][y] = piece;
    tBoard[x][y] = piece.owner;
}

// ============================================================================
// Piece Movement Rules
// ============================================================================

bool ChessBoard::canPawnMove(unsigned fromX, unsigned fromY, unsigned toX, unsigned toY, SquareState player) const
{
    int direction = (player == PLAYER_1) ? -1 : 1;  // White moves up (y decreases), black moves down
    int startRow = (player == PLAYER_1) ? 6 : 1;
    int dx = (int)toX - (int)fromX;
    int dy = (int)toY - (int)fromY;

    // Forward one square
    if (dx == 0 && dy == direction) {
        return getPieceAt(toX, toY).isEmpty();
    }

    // Forward two squares from starting position
    if (dx == 0 && dy == 2 * direction && fromY == startRow) {
        unsigned middleY = fromY + direction;
        return getPieceAt(toX, toY).isEmpty() && getPieceAt(toX, middleY).isEmpty();
    }

    // Diagonal capture
    if (abs(dx) == 1 && dy == direction) {
        ChessPiece target = getPieceAt(toX, toY);
        // Normal capture
        if (!target.isEmpty() && target.owner != player) {
            return true;
        }
        // En passant capture
        if ((int)toX == enPassantColumn && enPassantPlayer == Board::opponent(player)) {
            return true;
        }
    }

    return false;
}

bool ChessBoard::canKnightMove(unsigned fromX, unsigned fromY, unsigned toX, unsigned toY) const
{
    int dx = abs((int)toX - (int)fromX);
    int dy = abs((int)toY - (int)fromY);
    return (dx == 2 && dy == 1) || (dx == 1 && dy == 2);
}

bool ChessBoard::canBishopMove(unsigned fromX, unsigned fromY, unsigned toX, unsigned toY) const
{
    int dx = abs((int)toX - (int)fromX);
    int dy = abs((int)toY - (int)fromY);
    return dx == dy && dx > 0 && isPathClear(fromX, fromY, toX, toY);
}

bool ChessBoard::canRookMove(unsigned fromX, unsigned fromY, unsigned toX, unsigned toY) const
{
    bool sameRow = (fromX == toX);
    bool sameCol = (fromY == toY);
    return (sameRow || sameCol) && !(sameRow && sameCol) && isPathClear(fromX, fromY, toX, toY);
}

bool ChessBoard::canQueenMove(unsigned fromX, unsigned fromY, unsigned toX, unsigned toY) const
{
    return canRookMove(fromX, fromY, toX, toY) || canBishopMove(fromX, fromY, toX, toY);
}

bool ChessBoard::canKingMove(unsigned fromX, unsigned fromY, unsigned toX, unsigned toY) const
{
    int dx = abs((int)toX - (int)fromX);
    int dy = abs((int)toY - (int)fromY);
    return dx <= 1 && dy <= 1 && !(dx == 0 && dy == 0);
}

bool ChessBoard::isPathClear(unsigned fromX, unsigned fromY, unsigned toX, unsigned toY) const
{
    int dx = (toX > fromX) ? 1 : ((toX < fromX) ? -1 : 0);
    int dy = (toY > fromY) ? 1 : ((toY < fromY) ? -1 : 0);

    int x = fromX + dx;
    int y = fromY + dy;

    while (x != (int)toX || y != (int)toY) {
        if (!getPieceAt(x, y).isEmpty()) {
            return false;
        }
        x += dx;
        y += dy;
    }

    return true;
}

// ============================================================================
// Check Detection
// ============================================================================

bool ChessBoard::isSquareAttacked(unsigned x, unsigned y, SquareState byPlayer) const
{
    // Check if square (x, y) is attacked by any piece of byPlayer

    // Check for pawn attacks
    int pawnDir = (byPlayer == PLAYER_1) ? -1 : 1;
    if (isValidPosition(x - 1, y + pawnDir) && isPieceAt(x - 1, y + pawnDir, PAWN, byPlayer)) return true;
    if (isValidPosition(x + 1, y + pawnDir) && isPieceAt(x + 1, y + pawnDir, PAWN, byPlayer)) return true;

    // Check for knight attacks
    int knightMoves[8][2] = {{-2,-1},{-2,1},{-1,-2},{-1,2},{1,-2},{1,2},{2,-1},{2,1}};
    for (int i = 0; i < 8; i++) {
        int nx = x + knightMoves[i][0];
        int ny = y + knightMoves[i][1];
        if (isValidPosition(nx, ny) && isPieceAt(nx, ny, KNIGHT, byPlayer)) return true;
    }

    // Check for king attacks
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            if (dx == 0 && dy == 0) continue;
            int nx = x + dx;
            int ny = y + dy;
            if (isValidPosition(nx, ny) && isPieceAt(nx, ny, KING, byPlayer)) return true;
        }
    }

    // Check for bishop/queen attacks (diagonals)
    int diagDirs[4][2] = {{-1,-1},{-1,1},{1,-1},{1,1}};
    for (int i = 0; i < 4; i++) {
        int nx = x + diagDirs[i][0];
        int ny = y + diagDirs[i][1];
        while (isValidPosition(nx, ny)) {
            ChessPiece p = getPieceAt(nx, ny);
            if (!p.isEmpty()) {
                if (p.owner == byPlayer && (p.type == BISHOP || p.type == QUEEN)) return true;
                break;
            }
            nx += diagDirs[i][0];
            ny += diagDirs[i][1];
        }
    }

    // Check for rook/queen attacks (straight lines)
    int straightDirs[4][2] = {{-1,0},{1,0},{0,-1},{0,1}};
    for (int i = 0; i < 4; i++) {
        int nx = x + straightDirs[i][0];
        int ny = y + straightDirs[i][1];
        while (isValidPosition(nx, ny)) {
            ChessPiece p = getPieceAt(nx, ny);
            if (!p.isEmpty()) {
                if (p.owner == byPlayer && (p.type == ROOK || p.type == QUEEN)) return true;
                break;
            }
            nx += straightDirs[i][0];
            ny += straightDirs[i][1];
        }
    }

    return false;
}

bool ChessBoard::isInCheck(SquareState player) const
{
    unsigned kingX, kingY;
    if (!findKing(player, kingX, kingY)) {
        return false;  // No king found (shouldn't happen in valid game)
    }
    return isSquareAttacked(kingX, kingY, Board::opponent(player));
}

bool ChessBoard::wouldBeInCheck(unsigned fromX, unsigned fromY, unsigned toX, unsigned toY, SquareState player) const
{
    // Create a copy of the board to test the move
    ChessBoard* testBoard = new ChessBoard(const_cast<ChessBoard*>(this));

    // Make the move on the test board
    testBoard->pieces[toX][toY] = testBoard->pieces[fromX][fromY];
    testBoard->pieces[fromX][fromY] = ChessPiece(NO_PIECE, EMPTY);

    // Check if the move results in check
    bool inCheck = testBoard->isInCheck(player);

    // Clean up
    delete testBoard;

    return inCheck;
}

bool ChessBoard::findKing(SquareState player, unsigned& kingX, unsigned& kingY) const
{
    for (unsigned x = 0; x < 8; x++) {
        for (unsigned y = 0; y < 8; y++) {
            if (isPieceAt(x, y, KING, player)) {
                kingX = x;
                kingY = y;
                return true;
            }
        }
    }
    return false;
}

// ============================================================================
// Castling
// ============================================================================

bool ChessBoard::canCastleKingside(SquareState player) const
{
    unsigned kingRow = (player == PLAYER_1) ? 7 : 0;
    bool canCastle = (player == PLAYER_1) ? whiteCanCastleKingside : blackCanCastleKingside;

    if (!canCastle) return false;
    if (!isPieceAt(4, kingRow, KING, player)) return false;
    if (!isPieceAt(7, kingRow, ROOK, player)) return false;
    if (isInCheck(player)) return false;

    // Check squares between king and rook are empty
    if (!getPieceAt(5, kingRow).isEmpty() || !getPieceAt(6, kingRow).isEmpty()) return false;

    // Check king doesn't move through check
    if (isSquareAttacked(5, kingRow, Board::opponent(player))) return false;
    if (isSquareAttacked(6, kingRow, Board::opponent(player))) return false;

    return true;
}

bool ChessBoard::canCastleQueenside(SquareState player) const
{
    unsigned kingRow = (player == PLAYER_1) ? 7 : 0;
    bool canCastle = (player == PLAYER_1) ? whiteCanCastleQueenside : blackCanCastleQueenside;

    if (!canCastle) return false;
    if (!isPieceAt(4, kingRow, KING, player)) return false;
    if (!isPieceAt(0, kingRow, ROOK, player)) return false;
    if (isInCheck(player)) return false;

    // Check squares between king and rook are empty
    if (!getPieceAt(1, kingRow).isEmpty() || !getPieceAt(2, kingRow).isEmpty() || !getPieceAt(3, kingRow).isEmpty()) return false;

    // Check king doesn't move through check
    if (isSquareAttacked(2, kingRow, Board::opponent(player))) return false;
    if (isSquareAttacked(3, kingRow, Board::opponent(player))) return false;

    return true;
}

// ============================================================================
// Legal Move Generation
// ============================================================================

void ChessBoard::getAllLegalMovesForPiece(unsigned x, unsigned y, vector<pair<unsigned, unsigned>>& moves) const
{
    ChessPiece piece = getPieceAt(x, y);
    if (piece.isEmpty()) return;

    for (unsigned toX = 0; toX < 8; toX++) {
        for (unsigned toY = 0; toY < 8; toY++) {
            // Skip same square
            if (toX == x && toY == y) continue;

            // Skip if destination has our own piece
            ChessPiece dest = getPieceAt(toX, toY);
            if (!dest.isEmpty() && dest.owner == piece.owner) continue;

            // Check if this piece can make this move
            bool canMove = false;
            switch (piece.type) {
                case PAWN: canMove = canPawnMove(x, y, toX, toY, piece.owner); break;
                case KNIGHT: canMove = canKnightMove(x, y, toX, toY); break;
                case BISHOP: canMove = canBishopMove(x, y, toX, toY); break;
                case ROOK: canMove = canRookMove(x, y, toX, toY); break;
                case QUEEN: canMove = canQueenMove(x, y, toX, toY); break;
                case KING: canMove = canKingMove(x, y, toX, toY); break;
                default: break;
            }

            if (canMove && !wouldBeInCheck(x, y, toX, toY, piece.owner)) {
                moves.push_back(make_pair(toX, toY));
            }
        }
    }

    // Special case: castling for king
    if (piece.type == KING) {
        unsigned kingRow = (piece.owner == PLAYER_1) ? 7 : 0;
        if (canCastleKingside(piece.owner)) {
            moves.push_back(make_pair(6, kingRow));  // King moves to g-file
        }
        if (canCastleQueenside(piece.owner)) {
            moves.push_back(make_pair(2, kingRow));  // King moves to c-file
        }
    }
}

bool ChessBoard::hasAnyLegalMoves(SquareState player) const
{
    for (unsigned x = 0; x < 8; x++) {
        for (unsigned y = 0; y < 8; y++) {
            ChessPiece piece = getPieceAt(x, y);
            if (piece.isEmpty() || piece.owner != player) continue;

            vector<pair<unsigned, unsigned>> moves;
            getAllLegalMovesForPiece(x, y, moves);
            if (!moves.empty()) return true;
        }
    }
    return false;
}

// ============================================================================
// Core Board Interface Implementation
// ============================================================================

bool ChessBoard::legalMove(unsigned xPos, unsigned yPos, SquareState player)
{
    // NOTE: For chess, this method expects the DESTINATION square
    // The FROM square must be determined by finding which piece can legally move there
    // This is a limitation of the Board interface designed for Reversi

    // For simplicity, we'll return true if ANY piece of the player can legally move to (xPos, yPos)
    // The actual "from" square will be determined in the turn() method

    if (xPos >= 8 || yPos >= 8) {
        string error = "ChessBoard::legalMove : Position out of range";
        throw error;
    }

    if (player == EMPTY) {
        string error = "ChessBoard::legalMove : Empty is not a valid player";
        throw error;
    }

    // Check if any of player's pieces can move to this square
    for (unsigned fromX = 0; fromX < 8; fromX++) {
        for (unsigned fromY = 0; fromY < 8; fromY++) {
            ChessPiece piece = getPieceAt(fromX, fromY);
            if (piece.isEmpty() || piece.owner != player) continue;

            // Check if this piece can move to destination
            bool canMove = false;
            switch (piece.type) {
                case PAWN: canMove = canPawnMove(fromX, fromY, xPos, yPos, player); break;
                case KNIGHT: canMove = canKnightMove(fromX, fromY, xPos, yPos); break;
                case BISHOP: canMove = canBishopMove(fromX, fromY, xPos, yPos); break;
                case ROOK: canMove = canRookMove(fromX, fromY, xPos, yPos); break;
                case QUEEN: canMove = canQueenMove(fromX, fromY, xPos, yPos); break;
                case KING: canMove = canKingMove(fromX, fromY, xPos, yPos); break;
                default: break;
            }

            if (canMove) {
                // Make sure destination is empty or has opponent piece
                ChessPiece dest = getPieceAt(xPos, yPos);
                if (dest.isEmpty() || dest.owner != player) {
                    // Verify this move doesn't leave player in check
                    if (!wouldBeInCheck(fromX, fromY, xPos, yPos, player)) {
                        return true;
                    }
                }
            }
        }
    }

    // Check castling moves
    if (player == PLAYER_1 && yPos == 7) {
        if (xPos == 6 && canCastleKingside(PLAYER_1)) return true;
        if (xPos == 2 && canCastleQueenside(PLAYER_1)) return true;
    }
    if (player == PLAYER_2 && yPos == 0) {
        if (xPos == 6 && canCastleKingside(PLAYER_2)) return true;
        if (xPos == 2 && canCastleQueenside(PLAYER_2)) return true;
    }

    return false;
}

void ChessBoard::makeMove(unsigned xPos, unsigned yPos, SquareState player)
{
    // Find which piece should move to this destination
    // Priority: piece with lowest value (to avoid moving valuable pieces unnecessarily)

    int bestValue = 999;
    unsigned bestFromX = 0, bestFromY = 0;
    bool foundMove = false;

    for (unsigned fromX = 0; fromX < 8; fromX++) {
        for (unsigned fromY = 0; fromY < 8; fromY++) {
            ChessPiece piece = getPieceAt(fromX, fromY);
            if (piece.isEmpty() || piece.owner != player) continue;

            bool canMove = false;
            switch (piece.type) {
                case PAWN: canMove = canPawnMove(fromX, fromY, xPos, yPos, player); break;
                case KNIGHT: canMove = canKnightMove(fromX, fromY, xPos, yPos); break;
                case BISHOP: canMove = canBishopMove(fromX, fromY, xPos, yPos); break;
                case ROOK: canMove = canRookMove(fromX, fromY, xPos, yPos); break;
                case QUEEN: canMove = canQueenMove(fromX, fromY, xPos, yPos); break;
                case KING: canMove = canKingMove(fromX, fromY, xPos, yPos); break;
                default: break;
            }

            if (canMove && !wouldBeInCheck(fromX, fromY, xPos, yPos, player)) {
                int value = (int)piece.type;
                if (value < bestValue) {
                    bestValue = value;
                    bestFromX = fromX;
                    bestFromY = fromY;
                    foundMove = true;
                }
            }
        }
    }

    if (!foundMove) {
        // Check if it's a castling move
        unsigned kingRow = (player == PLAYER_1) ? 7 : 0;
        if (yPos == kingRow && isPieceAt(4, kingRow, KING, player)) {
            if (xPos == 6 && canCastleKingside(player)) {
                // Kingside castle
                setPieceAt(6, kingRow, ChessPiece(KING, player));
                setPieceAt(5, kingRow, ChessPiece(ROOK, player));
                setPieceAt(4, kingRow, ChessPiece(NO_PIECE, EMPTY));
                setPieceAt(7, kingRow, ChessPiece(NO_PIECE, EMPTY));

                if (player == PLAYER_1) {
                    whiteCanCastleKingside = false;
                    whiteCanCastleQueenside = false;
                } else {
                    blackCanCastleKingside = false;
                    blackCanCastleQueenside = false;
                }
                return;
            } else if (xPos == 2 && canCastleQueenside(player)) {
                // Queenside castle
                setPieceAt(2, kingRow, ChessPiece(KING, player));
                setPieceAt(3, kingRow, ChessPiece(ROOK, player));
                setPieceAt(4, kingRow, ChessPiece(NO_PIECE, EMPTY));
                setPieceAt(0, kingRow, ChessPiece(NO_PIECE, EMPTY));

                if (player == PLAYER_1) {
                    whiteCanCastleKingside = false;
                    whiteCanCastleQueenside = false;
                } else {
                    blackCanCastleKingside = false;
                    blackCanCastleQueenside = false;
                }
                return;
            }
        }

        string error = "ChessBoard::makeMove : No piece can legally move to destination";
        throw error;
    }

    // Execute the move
    ChessPiece movingPiece = getPieceAt(bestFromX, bestFromY);
    ChessPiece capturedPiece = getPieceAt(xPos, yPos);

    // Handle en passant capture
    bool isEnPassant = false;
    if (movingPiece.type == PAWN && (int)xPos == enPassantColumn && capturedPiece.isEmpty()) {
        unsigned capturedPawnY = (player == PLAYER_1) ? yPos + 1 : yPos - 1;
        setPieceAt(xPos, capturedPawnY, ChessPiece(NO_PIECE, EMPTY));
        isEnPassant = true;
    }

    // Move the piece
    setPieceAt(xPos, yPos, movingPiece);
    setPieceAt(bestFromX, bestFromY, ChessPiece(NO_PIECE, EMPTY));

    // Handle pawn promotion (auto-promote to Queen)
    if (movingPiece.type == PAWN) {
        if ((player == PLAYER_1 && yPos == 0) || (player == PLAYER_2 && yPos == 7)) {
            setPieceAt(xPos, yPos, ChessPiece(QUEEN, player));
        }
    }

    // Update en passant state
    enPassantColumn = -1;
    if (movingPiece.type == PAWN && abs((int)yPos - (int)bestFromY) == 2) {
        enPassantColumn = xPos;
        enPassantPlayer = player;
    }

    // Update castling rights
    if (movingPiece.type == KING) {
        if (player == PLAYER_1) {
            whiteCanCastleKingside = false;
            whiteCanCastleQueenside = false;
        } else {
            blackCanCastleKingside = false;
            blackCanCastleQueenside = false;
        }
    }
    if (movingPiece.type == ROOK) {
        if (player == PLAYER_1) {
            if (bestFromX == 0 && bestFromY == 7) whiteCanCastleQueenside = false;
            if (bestFromX == 7 && bestFromY == 7) whiteCanCastleKingside = false;
        } else {
            if (bestFromX == 0 && bestFromY == 0) blackCanCastleQueenside = false;
            if (bestFromX == 7 && bestFromY == 0) blackCanCastleKingside = false;
        }
    }
}

// ============================================================================
// Heuristic Evaluation
// ============================================================================

float ChessBoard::computerEstimation(unsigned xPos, unsigned yPos, SquareState player)
{
    // Random player - just return random value
    // Chess doesn't need a heuristic opponent like Reversi does
    return (float)(Random::positiveInteger(100));
}

float ChessBoard::individualEstimation(unsigned xPos, unsigned yPos, SquareState player, Individual* individual)
{
    // Neural network evaluation
    ChessBoard* futureBoard = new ChessBoard(this);
    futureBoard->makeMove(xPos, yPos, player);
    individual->updateInput(0, futureBoard->updateInterface());
    individual->calculateOutput();
    delete futureBoard;
    return individual->getOutput(individual->getNumLayers() - 1)->getElement(0);
}

// ============================================================================
// Game Ending and Scoring
// ============================================================================

bool ChessBoard::endGame()
{
    // Game ends if either player is checkmated or stalemated
    return isCheckmate(PLAYER_1) || isCheckmate(PLAYER_2) ||
           isStalemate(PLAYER_1) || isStalemate(PLAYER_2);
}

bool ChessBoard::isCheckmate(SquareState player) const
{
    return isInCheck(player) && !hasAnyLegalMoves(player);
}

bool ChessBoard::isStalemate(SquareState player) const
{
    return !isInCheck(player) && !hasAnyLegalMoves(player);
}

float ChessBoard::getMaterialValue(SquareState player) const
{
    float value = 0.0;
    for (unsigned x = 0; x < 8; x++) {
        for (unsigned y = 0; y < 8; y++) {
            ChessPiece piece = getPieceAt(x, y);
            if (piece.owner == player) {
                switch (piece.type) {
                    case PAWN: value += 1.0; break;
                    case KNIGHT: value += 3.0; break;
                    case BISHOP: value += 3.0; break;
                    case ROOK: value += 5.0; break;
                    case QUEEN: value += 9.0; break;
                    case KING: value += 0.0; break;  // King is priceless
                    default: break;
                }
            }
        }
    }
    return value;
}

// ============================================================================
// Neural Network Interface
// ============================================================================

Interface* ChessBoard::updateInterface()
{
    // 768-input piece-aware encoding: 12 inputs per square
    // For each square: 6 piece types × 2 players = 12 binary inputs

    unsigned index = 0;
    for (int x = 0; x < 8; ++x) {
        for (int y = 0; y < 8; ++y) {
            ChessPiece piece = getPieceAt(x, y);

            // PLAYER_1 pieces (white)
            tInterface->setElement(index++, (piece.owner == PLAYER_1 && piece.type == PAWN) ? 1.0 : 0.0);
            tInterface->setElement(index++, (piece.owner == PLAYER_1 && piece.type == KNIGHT) ? 1.0 : 0.0);
            tInterface->setElement(index++, (piece.owner == PLAYER_1 && piece.type == BISHOP) ? 1.0 : 0.0);
            tInterface->setElement(index++, (piece.owner == PLAYER_1 && piece.type == ROOK) ? 1.0 : 0.0);
            tInterface->setElement(index++, (piece.owner == PLAYER_1 && piece.type == QUEEN) ? 1.0 : 0.0);
            tInterface->setElement(index++, (piece.owner == PLAYER_1 && piece.type == KING) ? 1.0 : 0.0);

            // PLAYER_2 pieces (black)
            tInterface->setElement(index++, (piece.owner == PLAYER_2 && piece.type == PAWN) ? 1.0 : 0.0);
            tInterface->setElement(index++, (piece.owner == PLAYER_2 && piece.type == KNIGHT) ? 1.0 : 0.0);
            tInterface->setElement(index++, (piece.owner == PLAYER_2 && piece.type == BISHOP) ? 1.0 : 0.0);
            tInterface->setElement(index++, (piece.owner == PLAYER_2 && piece.type == ROOK) ? 1.0 : 0.0);
            tInterface->setElement(index++, (piece.owner == PLAYER_2 && piece.type == QUEEN) ? 1.0 : 0.0);
            tInterface->setElement(index++, (piece.owner == PLAYER_2 && piece.type == KING) ? 1.0 : 0.0);
        }
    }

    return tInterface;
}

// ============================================================================
// Board Visualization
// ============================================================================

void ChessBoard::printBoard(std::ostream& out) const
{
    // Chess board with piece symbols
    // Pieces: Uppercase for White (P R N B Q K), lowercase for Black (p r n b q k)

    out << "  +---+---+---+---+---+---+---+---+" << std::endl;
    for (int y = 0; y < 8; y++) {
        out << (8 - y) << " |";  // Rank numbers (8 down to 1)

        for (int x = 0; x < 8; x++) {
            ChessPiece piece = getPieceAt(x, y);

            if (piece.owner == EMPTY) {
                // Empty square - checkered pattern
                if ((x + y) % 2 == 0) {
                    out << "   ";
                } else {
                    out << " . ";
                }
            } else {
                // Determine piece character
                char pieceChar;
                switch (piece.type) {
                    case PAWN:   pieceChar = 'P'; break;
                    case ROOK:   pieceChar = 'R'; break;
                    case KNIGHT: pieceChar = 'N'; break;
                    case BISHOP: pieceChar = 'B'; break;
                    case QUEEN:  pieceChar = 'Q'; break;
                    case KING:   pieceChar = 'K'; break;
                    default:     pieceChar = '?'; break;
                }

                // White = uppercase, Black = lowercase
                if (piece.owner == PLAYER_2) {
                    pieceChar = tolower(pieceChar);
                }

                out << " " << pieceChar << " ";
            }
            out << "|";
        }
        out << " " << (8 - y) << std::endl;
        out << "  +---+---+---+---+---+---+---+---+" << std::endl;
    }
    out << "    a   b   c   d   e   f   g   h  " << std::endl;
}
