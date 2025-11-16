#include <iostream>
#include <fstream>
#include "tasks/goTask.h"
#include "genetic/individual.h"

using namespace std;

void printBoard(GoBoard& board, ostream& out) {
    // Print column headers
    out << "    ";
    for (unsigned x = 0; x < board.getSize(); x++) {
        out << x << "   ";
    }
    out << endl;

    for (unsigned y = 0; y < board.getSize(); y++) {
        // Print row with intersections and stones
        out << y << "   ";
        for (unsigned x = 0; x < board.getSize(); x++) {
            SquareState state = board.getSquare(x, y);
            if (state == EMPTY) out << ".";
            else if (state == PLAYER_1) out << "@";
            else out << "O";

            // Print horizontal line (except after last column)
            if (x < board.getSize() - 1) {
                out << "---";
            }
        }
        out << endl;

        // Print vertical lines (except after last row)
        if (y < board.getSize() - 1) {
            out << "    ";
            for (unsigned x = 0; x < board.getSize(); x++) {
                out << "|";
                if (x < board.getSize() - 1) {
                    out << "   ";
                }
            }
            out << endl;
        }
    }
}

int main(int argc, char *argv[])
{
    cout << "=== PREANN Go Demo ===" << endl << endl;

    try {
        // Parse board size argument
        int boardSize = 9;  // Default to 9x9
        if (argc > 1) {
            boardSize = atoi(argv[1]);
            if (boardSize < 5 || boardSize > 19) {
                cerr << "Board size must be between 5 and 19" << endl;
                return 1;
            }
        }

        cout << "Board size: " << boardSize << "x" << boardSize << endl << endl;

        // Create a GoTask
        cout << "Creating GoTask..." << endl;
        GoTask task(boardSize, BT_BIT, 1);
        cout << "Task created successfully!" << endl << endl;

        // Create two random untrained neural networks
        ParametersMap params;
        params.putNumber(Enumerations::enumTypeToString(ET_IMPLEMENTATION), IT_CUDA_REDUC);
        params.putNumber(Enumerations::enumTypeToString(ET_BUFFER), BT_BIT);
        params.putNumber(Enumerations::enumTypeToString(ET_FUNCTION), FT_BIPOLAR_SIGMOID);

        cout << "Creating two random neural networks..." << endl;
        Individual* player1 = task.getExample(&params);
        Individual* player2 = task.getExample(&params);
        cout << "Neural networks created" << endl << endl;

        // Play a game and save board states
        cout << "Playing game and saving boards to output/data/go_demo_game.txt..." << endl;
        ofstream gameFile("output/data/go_demo_game.txt");

        if (!gameFile.is_open()) {
            throw string("Could not open output/data/go_demo_game.txt for writing");
        }

        GoBoard board(boardSize, BT_BIT);
        board.initBoard();

        gameFile << "=== Go Game Demo ===" << endl;
        gameFile << "Board size: " << boardSize << "x" << boardSize << endl;
        gameFile << "Players: Two random untrained neural networks" << endl << endl;

        gameFile << "Initial board:" << endl;
        printBoard(board, gameFile);
        gameFile << endl;

        int moveNum = 0;
        SquareState turn = PLAYER_1;

        while (!board.endGame()) {
            if (!board.canMove(turn)) {
                gameFile << "Player " << (turn == PLAYER_1 ? "@" : "O") << " cannot move (passing)" << endl;
                turn = Board::opponent(turn);
                continue;
            }

            moveNum++;

            // Select which player's turn it is
            Individual* currentPlayer = (turn == PLAYER_1) ? player1 : player2;

            // Make a move
            board.turn(turn, currentPlayer);

            // Print board state
            gameFile << "After move " << moveNum << " (Player " << (turn == PLAYER_1 ? "@" : "O") << "):" << endl;
            printBoard(board, gameFile);
            gameFile << endl;

            turn = Board::opponent(turn);

            // Safety limit to prevent infinite games
            if (moveNum > boardSize * boardSize * 2) {
                gameFile << "Game ended due to move limit" << endl;
                break;
            }
        }

        int scoreP1 = board.countPoints(PLAYER_1);
        int scoreP2 = board.countPoints(PLAYER_2);
        gameFile << "Final score - Player @: " << scoreP1 << ", Player O: " << scoreP2 << endl;
        gameFile << "Game ended after " << moveNum << " moves" << endl;

        gameFile.close();
        cout << "Game saved to output/data/go_demo_game.txt" << endl;
        cout << "Moves played: " << moveNum << endl;
        cout << "Final score - Player @: " << scoreP1 << ", Player O: " << scoreP2 << endl;

        delete player1;
        delete player2;

    } catch (string& error) {
        cerr << "Error: " << error << endl;
        return 1;
    }

    cout << "Exit success." << endl;
    return 0;
}
