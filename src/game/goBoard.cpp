#include "goBoard.h"
#include <cassert>
#include <cstring>

using namespace std;

GoBoard::GoBoard(unsigned size, BufferType bufferType) :
        Board(size, bufferType), tPreviousBoardHash(0)
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

    // Check 4: Suicide rule - cannot play where resulting group has zero liberties
    // Exception: Legal if the move captures opponent stones
    if (!wouldCapture(xPos, yPos, player)) {
        // Won't capture, so check if we'd have liberties
        tBoard[xPos][yPos] = player;  // Temporarily place stone
        int liberties = countLiberties(xPos, yPos, player);
        tBoard[xPos][yPos] = EMPTY;  // Remove temporary stone

        if (liberties == 0) {
            return false;  // Suicide - illegal
        }
    }

    // Check 5: Ko rule - cannot immediately recapture to recreate previous position
    // Simulate the move and check board hash
    tBoard[xPos][yPos] = player;
    unsigned long long newHash = calculateBoardHash();
    tBoard[xPos][yPos] = EMPTY;

    if (newHash == tPreviousBoardHash) {
        return false;  // Ko violation - would recreate previous board
    }

    return true;  // Move is legal!
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

    // Save current board state for ko rule
    tPreviousBoardHash = calculateBoardHash();

    // Step 1: Place the stone
    setSquare(xPos, yPos, player);

    // Step 2: Remove captured opponent groups
    SquareState opponent = Board::opponent(player);
    int dx[] = {-1, 1, 0, 0};
    int dy[] = {0, 0, -1, 1};

    // Check all 4 adjacent positions
    for (int dir = 0; dir < 4; dir++) {
        int nx = xPos + dx[dir];
        int ny = yPos + dy[dir];

        if (insideBoard(nx, ny) && tBoard[nx][ny] == opponent) {
            // Check if this opponent group has liberties
            if (countLiberties(nx, ny, opponent) == 0) {
                // Capture! Remove the entire group
                bool visited[19][19] = {{false}};
                vector<pair<unsigned, unsigned>> capturedGroup;
                findGroup(nx, ny, opponent, visited, capturedGroup);
                removeGroup(capturedGroup);
            }
        }
    }
}

float GoBoard::computerEstimation(unsigned xPos, unsigned yPos, SquareState player)
{
    // TODO: Optional: competent challenger for faster bootstrap
    //
    // This method provides a heuristic for a computer opponent to play against
    // during neural network training. Having a competent opponent helps the
    // neural network learn faster (bootstrap).
    //
    // For now, neural networks can train by playing against each other
    // (individualEstimation is implemented). This is optional but helpful.
    //
    // NOT IMPLEMENTED
    // The plan is to include Fuego library or reimplement here.
    //
    // Simple heuristic ideas:
    // 1. Count liberties gained by this move
    // 2. Count opponent stones that would be captured
    // 3. Consider strategic positions (corners, edges, center)
    //
    // Can use Fuego library or reimplement here as simple heuristic.

    assert(false && "GoBoard::computerEstimation() NOT IMPLEMENTED - TODO: optional: competent challenger for faster bootstrap");
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

// ============================================================================
// Helper Functions for Go Rules Implementation
// ============================================================================

unsigned long long GoBoard::calculateBoardHash()
{
    // Simple hash function for ko rule detection
    // Uses Zobrist-like hashing: XOR positions with prime multipliers
    unsigned long long hash = 0;
    unsigned long long prime1 = 73856093;
    unsigned long long prime2 = 19349663;

    for (unsigned x = 0; x < tSize; x++) {
        for (unsigned y = 0; y < tSize; y++) {
            if (tBoard[x][y] != EMPTY) {
                unsigned long long posHash = (x * prime1) ^ (y * prime2);
                if (tBoard[x][y] == PLAYER_1) {
                    hash ^= posHash;
                } else {
                    hash ^= (posHash * 2);  // Different for PLAYER_2
                }
            }
        }
    }
    return hash;
}

void GoBoard::findGroup(unsigned xPos, unsigned yPos, SquareState player,
                        bool visited[][19], vector<pair<unsigned, unsigned>>& group)
{
    // Flood fill to find all connected stones of same color
    if (!insideBoard(xPos, yPos)) return;
    if (visited[xPos][yPos]) return;
    if (tBoard[xPos][yPos] != player) return;

    visited[xPos][yPos] = true;
    group.push_back(make_pair(xPos, yPos));

    // Recursively check 4 adjacent positions (not diagonals in Go)
    if (xPos > 0) findGroup(xPos - 1, yPos, player, visited, group);
    if (xPos < tSize - 1) findGroup(xPos + 1, yPos, player, visited, group);
    if (yPos > 0) findGroup(xPos, yPos - 1, player, visited, group);
    if (yPos < tSize - 1) findGroup(xPos, yPos + 1, player, visited, group);
}

int GoBoard::countLiberties(unsigned xPos, unsigned yPos, SquareState player)
{
    // Count liberties (empty adjacent squares) for the group containing this stone
    if (!insideBoard(xPos, yPos)) return 0;
    if (tBoard[xPos][yPos] != player) return 0;

    // Find the group
    bool visited[19][19] = {{false}};
    vector<pair<unsigned, unsigned>> group;
    findGroup(xPos, yPos, player, visited, group);

    // Count unique liberties
    bool libertyFound[19][19] = {{false}};
    int libertyCount = 0;

    for (size_t i = 0; i < group.size(); i++) {
        unsigned x = group[i].first;
        unsigned y = group[i].second;

        // Check 4 adjacent positions
        int dx[] = {-1, 1, 0, 0};
        int dy[] = {0, 0, -1, 1};

        for (int dir = 0; dir < 4; dir++) {
            int nx = x + dx[dir];
            int ny = y + dy[dir];

            if (insideBoard(nx, ny) && tBoard[nx][ny] == EMPTY) {
                if (!libertyFound[nx][ny]) {
                    libertyFound[nx][ny] = true;
                    libertyCount++;
                }
            }
        }
    }

    return libertyCount;
}

void GoBoard::removeGroup(const vector<pair<unsigned, unsigned>>& group)
{
    // Remove all stones in the group
    for (size_t i = 0; i < group.size(); i++) {
        tBoard[group[i].first][group[i].second] = EMPTY;
    }
}

bool GoBoard::wouldCapture(unsigned xPos, unsigned yPos, SquareState player)
{
    // Check if placing a stone here would capture any opponent groups
    SquareState opponent = Board::opponent(player);

    // Check all 4 adjacent positions
    int dx[] = {-1, 1, 0, 0};
    int dy[] = {0, 0, -1, 1};

    for (int dir = 0; dir < 4; dir++) {
        int nx = xPos + dx[dir];
        int ny = yPos + dy[dir];

        if (insideBoard(nx, ny) && tBoard[nx][ny] == opponent) {
            // Temporarily place our stone
            tBoard[xPos][yPos] = player;

            // Check if opponent group would have zero liberties
            int opponentLiberties = countLiberties(nx, ny, opponent);

            // Remove our temporary stone
            tBoard[xPos][yPos] = EMPTY;

            if (opponentLiberties == 0) {
                return true;  // Would capture this group
            }
        }
    }

    return false;
}
