#include <iostream>
#include <unistd.h>  // for usleep
#include "game/goBoard.h"
#include "tasks/goTask.h"

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

    virtual float individualEstimation(unsigned xPos, unsigned yPos, SquareState player, Individual* individual) {
        // Player O (PLAYER_2) returns random 0-10 (random strategy)
        if (player == PLAYER_2) {
            return (float)(rand() % 11);  // 0 to 10
        }

        // Player @ (PLAYER_1) uses ACTUAL NEURAL NETWORK (re-enabled!)
        // This is the original implementation from goBoard.cpp
        GoBoard* futureBoard = new GoBoard(this);
        futureBoard->makeMove(xPos, yPos, player);
        individual->updateInput(0, futureBoard->updateInterface());
        individual->calculateOutput();
        delete futureBoard;

        return individual->getOutput(individual->getNumLayers() - 1)->getElement(0);
    }
};

int main(int argc, char *argv[])
{
    cout << "=== Go: Neural Network vs Random Player ===" << endl << endl;
    cout << "Player @ (Black/PLAYER_1): NEURAL NETWORK (re-enabled!)" << endl;
    cout << "Player O (White/PLAYER_2): Random 0-10 strategy" << endl;
    cout << "Half second delay between moves..." << endl << endl;
    cout << "Network architecture:" << endl;
    cout << "  Input: 81 (9x9 board)" << endl;
    cout << "  Hidden: 9 neurons (SMALL for speed)" << endl;
    cout << "  Output: 1 (position evaluation)" << endl;
    cout << "  Total weights: ~738" << endl << endl;

    srand(time(NULL));  // Seed random number generator

    try {
        SimpleGoBoard board(9, BT_BIT);
        board.initBoard();

        // Create SMALL neural network for speed with C++ implementation
        ParametersMap params;
        params.putNumber(Enumerations::enumTypeToString(ET_IMPLEMENTATION), IT_C);  // C++ not CUDA!
        params.putNumber(Enumerations::enumTypeToString(ET_BUFFER), BT_FLOAT);
        params.putNumber(Enumerations::enumTypeToString(ET_FUNCTION), FT_IDENTITY);

        // Create small network manually
        Individual player1(IT_C);
        player1.addInputLayer(board.getInterface());
        player1.addLayer(9, BT_FLOAT, FT_IDENTITY);      // Small hidden layer (9 neurons)
        player1.addLayer(1, BT_FLOAT, FT_IDENTITY);       // Output layer
        player1.addInputConnection(0, 0);                 // Input -> Hidden
        player1.addLayersConnection(0, 1);                // Hidden -> Output

        // Dummy individual for random player
        Individual player2(IT_C);

        SquareState turn = PLAYER_1;
        int moveCount = 0;
        int maxMoves = 50;

        cout << "Initial board:" << endl;
        printBoard(&board);

        while (!board.endGame() && moveCount < maxMoves) {
            if (board.canMove(turn)) {
                cout << "Move " << (moveCount + 1) << ": Player " << (turn == PLAYER_1 ? "@" : "O");
                if (turn == PLAYER_1) cout << " (NN)";
                else cout << " (Random)";
                cout << endl;

                if (turn == PLAYER_1) {
                    board.turn(turn, &player1);  // NN player
                } else {
                    board.turn(turn, &player2);  // Random player
                }

                printBoard(&board);
                usleep(500000);  // 0.5 second delay

                moveCount++;
            } else {
                cout << "Player " << (turn == PLAYER_1 ? "@" : "O") << " cannot move (passing)" << endl;
            }

            turn = Board::opponent(turn);
        }

        cout << "=== Game ended after " << moveCount << " moves ===" << endl;
        cout << "Player @ (NN) points: " << board.countPoints(PLAYER_1) << endl;
        cout << "Player O (Random) points: " << board.countPoints(PLAYER_2) << endl;

    } catch (string& error) {
        cerr << "Error: " << error << endl;
        return 1;
    }

    cout << endl << "Exit success." << endl;
    return 0;
}
