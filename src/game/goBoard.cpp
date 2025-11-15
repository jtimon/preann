#include "goBoard.h"
#include <cassert>
#include <cstring>

using namespace std;

GoBoard::GoBoard(unsigned size, BufferType bufferType) :
        Board(size, bufferType)
{
    // Standard Go board sizes are 9x9, 13x13, and 19x19
    // But we'll allow any size for experimentation
    if (size < 5) {
        string error = "The minimum size of a Go board is 5.";
        throw error;
    }
    if (size > 19) {
        string error = "The maximum size of a Go board is 19 (standard).";
        throw error;
    }
}

GoBoard::GoBoard(GoBoard* other) :
        Board(other)
{
}

GoBoard::~GoBoard()
{
}

void GoBoard::initBoard()
{
    // Go starts with an empty board
    Board::initBoard();
}

bool GoBoard::legalMove(unsigned xPos, unsigned yPos, SquareState player)
{
    // Low-hanging fruit: basic checks that apply to both Go and Reversi

    // Check 1: Position must be on the board
    if (xPos >= tSize || yPos >= tSize) {
        string error = "GoBoard::legalMove : The position (" + to_string(xPos) + ", "
                + to_string(yPos) + ") is out of range. The size of the board is " + to_string(tSize);
        throw error;
    }

    // Check 2: Must be a valid player (not EMPTY)
    if (player == EMPTY) {
        string error = "GoBoard::legalMove : Empty square is not a player.";
        throw error;
    }

    // Check 3: Square must be empty (Rule 1 of Go: cannot play where there's already a stone)
    if (getSquare(xPos, yPos) != EMPTY) {
        return false;
    }

    // TODO: Additional Go-specific checks needed:
    // - Suicide rule: Move must not create a group with zero liberties (unless it captures)
    // - Ko rule: Move must not recreate previous board position
    // For now, we allow all moves to empty squares (will capture some illegal moves but playable)

    return true;  // Temporarily allow all moves to empty squares
}

void GoBoard::makeMove(unsigned xPos, unsigned yPos, SquareState player)
{
    // Low-hanging fruit: basic validations and stone placement

    // Check 1: Position must be on the board
    if (xPos >= tSize || yPos >= tSize) {
        string error = "GoBoard::makeMove : The position (" + to_string(xPos) + ", "
                + to_string(yPos) + ") is out of range. The size of the board is " + to_string(tSize);
        throw error;
    }

    // Check 2: Must be a valid player (not EMPTY)
    if (player == EMPTY) {
        string error = "GoBoard::makeMove : Empty square is not a player.";
        throw error;
    }

    // Check 3: Square must be empty
    if (getSquare(xPos, yPos) != EMPTY) {
        string error = "GoBoard::makeMove : the square at position (" + to_string(xPos) + ", "
                + to_string(yPos) + ") is already occupied";
        throw error;
    }

    // Step 1: Place the stone (simple and universal for Go)
    setSquare(xPos, yPos, player);

    // TODO: Step 2: Remove captured opponent groups
    // Need to:
    // - Check all 4 adjacent positions
    // - For each opponent stone, check if its group has liberties
    // - Remove groups with zero liberties
    // This is the moderately complex part that could use Fuego or be reimplemented
}

float GoBoard::computerEstimation(unsigned xPos, unsigned yPos, SquareState player)
{
    // NOT IMPLEMENTED
    // The plan is to include Fuego library or reimplement here.
    //
    // For a simple heuristic, we could:
    // 1. Count liberties gained by this move
    // 2. Count opponent stones captured
    // 3. Consider edge/corner positions
    //
    // This could be reimplemented here as a simple heuristic without Fuego.

    assert(false && "GoBoard::computerEstimation() NOT IMPLEMENTED - plan is to include Fuego or reimplement here");
    return 0.0;
}

float GoBoard::individualEstimation(unsigned xPos, unsigned yPos, SquareState player, Individual* individual)
{
    // This method uses the neural network to evaluate a position
    // Following the same pattern as ReversiBoard
    //
    // NOTE: This will not work until makeMove() is implemented
    // because we need to create a future board state to evaluate

    GoBoard* futureBoard = new GoBoard(this);
    futureBoard->makeMove(xPos, yPos, player);  // Will assert - NOT IMPLEMENTED
    individual->updateInput(0, futureBoard->updateInterface());
    individual->calculateOutput();
    delete futureBoard;

    // The first element of the last layer
    return individual->getOutput(individual->getNumLayers() - 1)->getElement(0);
}
