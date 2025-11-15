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
    // NOT IMPLEMENTED
    // The plan is to include Fuego library or reimplement here.
    //
    // Go legal move checking requires:
    // 1. Square must be empty
    // 2. Move must not be suicide (unless it captures opponent stones)
    // 3. Move must not violate ko rule (cannot immediately recapture)
    //
    // This is complex enough that using Fuego is preferred, but could
    // be reimplemented here for simple cases.

    assert(false && "GoBoard::legalMove() NOT IMPLEMENTED - plan is to include Fuego or reimplement here");
    return false;
}

void GoBoard::makeMove(unsigned xPos, unsigned yPos, SquareState player)
{
    // NOT IMPLEMENTED
    // The plan is to include Fuego library or reimplement here.
    //
    // Go move execution requires:
    // 1. Place stone at position
    // 2. Remove captured opponent groups (groups with no liberties)
    // 3. Update board state
    //
    // The capture detection algorithm is moderately complex but could
    // be reimplemented here without Fuego.

    assert(false && "GoBoard::makeMove() NOT IMPLEMENTED - plan is to include Fuego or reimplement here");
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
