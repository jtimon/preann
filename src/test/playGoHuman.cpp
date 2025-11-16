#include <iostream>
#include <string>
#include <cctype>
#include "game/goBoard.h"

using namespace std;

// Convert column letter to x coordinate (A=0, B=1, ..., skipping I)
int letterToX(char letter) {
    letter = toupper(letter);
    if (letter < 'A' || letter > 'T') return -1;
    if (letter >= 'I') letter--;  // Skip I
    return letter - 'A';
}

// Convert x coordinate to column letter
char xToLetter(int x) {
    if (x >= 8) x++;  // Skip I
    return 'A' + x;
}

void printBoard(GoBoard& board) {
    int size = board.getSize();

    // Print column headers (A-T, skipping I)
    cout << "   ";
    for (int x = 0; x < size; x++) {
        cout << xToLetter(x) << "   ";
    }
    cout << endl;

    for (int y = 0; y < size; y++) {
        // Print row number (top to bottom: size...1)
        int rowNum = size - y;
        if (rowNum < 10) cout << " ";
        cout << rowNum << " ";

        for (int x = 0; x < size; x++) {
            SquareState state = board.getSquare(x, y);

            // Print the intersection with stone or empty
            if (state == EMPTY) cout << ".";
            else if (state == PLAYER_1) cout << "@";
            else cout << "O";

            // Print horizontal line connection (except after last column)
            if (x < size - 1) {
                cout << "---";
            }
        }

        cout << " " << rowNum;  // Print row number on right side too
        cout << endl;

        // Print vertical lines (except after last row)
        if (y < size - 1) {
            cout << "   ";
            for (int x = 0; x < size; x++) {
                cout << "|   ";
            }
            cout << endl;
        }
    }

    // Print column headers at bottom
    cout << "   ";
    for (int x = 0; x < size; x++) {
        cout << xToLetter(x) << "   ";
    }
    cout << endl << endl;
}

int main(int argc, char *argv[])
{
    cout << "=== Go: Human vs Random Computer ===" << endl << endl;

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

        cout << "Board size: " << boardSize << "x" << boardSize << endl;
        cout << "You are Player @ (Black/PLAYER_1)" << endl;
        cout << "Computer is Player O (White/PLAYER_2) - random moves" << endl;
        cout << endl;
        cout << "To move: enter column letter + row number (e.g., 'D4' or 'd4')" << endl;
        cout << "To pass: enter 'pass'" << endl;
        cout << "Columns: A-" << xToLetter(boardSize - 1) << " (skip I), Rows: " << boardSize << "-1" << endl;
        cout << endl;

        GoBoard board(boardSize, BT_BIT);
        board.initBoard();

        SquareState turn = PLAYER_1;
        int moveCount = 0;

        cout << "Initial board:" << endl;
        printBoard(board);

        while (!board.endGame()) {
            if (turn == PLAYER_1) {
                // Human player
                cout << "Your move (@ - Black): ";
                string input;
                getline(cin, input);

                if (input == "pass") {
                    cout << "You passed." << endl;
                    board.pass();
                } else {
                    // Parse Go notation: letter + number (e.g., D4, d16)
                    if (input.length() >= 2) {
                        char colLetter = input[0];
                        int rowNum;
                        if (sscanf(input.c_str() + 1, "%d", &rowNum) == 1) {
                            int x = letterToX(colLetter);
                            int y = boardSize - rowNum;  // Convert from Go notation to array index

                            if (x >= 0 && x < boardSize && y >= 0 && y < boardSize) {
                                if (board.legalMove(x, y, PLAYER_1)) {
                                    board.makeMove(x, y, PLAYER_1);
                                    moveCount++;
                                    cout << "You played at " << (char)toupper(colLetter) << rowNum << endl;
                                } else {
                                    cout << "Illegal move! Try again." << endl;
                                    continue;  // Don't switch turns
                                }
                            } else {
                                cout << "Position out of range! Try again." << endl;
                                continue;
                            }
                        } else {
                            cout << "Invalid input! Enter like 'D4' or 'pass'" << endl;
                            continue;
                        }
                    } else {
                        cout << "Invalid input! Enter like 'D4' or 'pass'" << endl;
                        continue;  // Don't switch turns
                    }
                }
            } else {
                // Computer player (random)
                cout << "Computer's move (O - White): ";
                board.turn(PLAYER_2, NULL);  // Uses computerEstimation
                moveCount++;
                cout << endl;
            }

            printBoard(board);

            turn = Board::opponent(turn);
        }

        cout << "=== Game ended after " << moveCount << " moves ===" << endl;
        cout << "Player @ (You) points: " << board.countPoints(PLAYER_1) << endl;
        cout << "Player O (Computer) points: " << board.countPoints(PLAYER_2) << endl;

    } catch (string& error) {
        cerr << "Error: " << error << endl;
        return 1;
    }

    cout << endl << "Exit success." << endl;
    return 0;
}
