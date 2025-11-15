#include <iostream>
#include <unistd.h>  // for usleep
#include "game/goBoard.h"

using namespace std;

void printBoard(GoBoard* board) {
    // Print column headers
    cout << "    ";
    for (unsigned x = 0; x < board->getSize(); x++) {
        cout << x << "   ";
    }
    cout << endl;

    for (unsigned y = 0; y < board->getSize(); y++) {
        // Print row with intersections and stones
        cout << y << "   ";
        for (unsigned x = 0; x < board->getSize(); x++) {
            SquareState state = board->getSquare(x, y);
            if (state == EMPTY) cout << ".";
            else if (state == PLAYER_1) cout << "@";
            else cout << "O";

            // Print horizontal line (except after last column)
            if (x < board->getSize() - 1) {
                cout << "---";
            }
        }
        cout << endl;

        // Print vertical lines (except after last row)
        if (y < board->getSize() - 1) {
            cout << "    ";
            for (unsigned x = 0; x < board->getSize(); x++) {
                cout << "|";
                if (x < board->getSize() - 1) {
                    cout << "   ";
                }
            }
            cout << endl;
        }
    }
    cout << endl;
}

// Custom board class to override estimation functions
class SimpleGoBoard : public GoBoard {
public:
    SimpleGoBoard(unsigned size, BufferType bufferType) : GoBoard(size, bufferType) {}

    virtual float computerEstimation(unsigned xPos, unsigned yPos, SquareState player) {
        // Player O (PLAYER_2) uses random 0-10 strategy
        // Player @ (PLAYER_1) uses constant 1.0 strategy
        // But this is called for heuristic, not individual
        return 1.0;
    }

    virtual float individualEstimation(unsigned xPos, unsigned yPos, SquareState player, Individual* individual) {
        // Player @ (PLAYER_1) always returns 1.0 (constant strategy)
        if (player == PLAYER_1) {
            return 1.0;
        }
        // Player O (PLAYER_2) returns random 0-10 (random strategy)
        else {
            return (float)(rand() % 11);  // 0 to 10
        }
    }
};

int main(int argc, char *argv[])
{
    cout << "=== Simple Go Game: Constant vs Random Strategy ===" << endl << endl;
    cout << "Player @ (Black/PLAYER_1): Always returns 1.0 (picks first legal move)" << endl;
    cout << "Player O (White/PLAYER_2): Returns random 0-10 (picks random best move)" << endl;
    cout << "Half second delay between moves for viewing..." << endl << endl;

    srand(time(NULL));  // Seed random number generator

    try {
        SimpleGoBoard board(9, BT_BIT);
        board.initBoard();

        SquareState turn = PLAYER_1;
        int moveCount = 0;
        int maxMoves = 50;  // Stop after 50 moves

        cout << "Initial board:" << endl;
        printBoard(&board);

        // Create dummy individuals for both players
        Individual player1(IT_C);
        Individual player2(IT_C);

        while (!board.endGame() && moveCount < maxMoves) {
            if (board.canMove(turn)) {
                cout << "Move " << (moveCount + 1) << ": Player " << (turn == PLAYER_1 ? "@" : "O") << endl;

                // Use the individual with the board's custom individualEstimation
                if (turn == PLAYER_1) {
                    board.turn(turn, &player1);
                } else {
                    board.turn(turn, &player2);
                }

                printBoard(&board);

                // Sleep for 0.5 seconds (500000 microseconds)
                usleep(500000);

                moveCount++;
            } else {
                cout << "Player " << (turn == PLAYER_1 ? "@" : "O") << " cannot move (passing)" << endl;
            }

            turn = Board::opponent(turn);
        }

        cout << "=== Game ended after " << moveCount << " moves ===" << endl;
        cout << "Player @ (PLAYER_1) points: " << board.countPoints(PLAYER_1) << endl;
        cout << "Player O (PLAYER_2) points: " << board.countPoints(PLAYER_2) << endl;

    } catch (string& error) {
        cerr << "Error: " << error << endl;
        return 1;
    }

    cout << endl << "Exit success." << endl;
    return 0;
}
