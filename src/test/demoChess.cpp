/*
 * demoChess.cpp
 *
 * Plays a complete chess game and saves board states
 */

#include <iostream>
#include <fstream>
#include "tasks/chessTask.h"
#include "genetic/individual.h"

using namespace std;

void printBoard(ChessBoard& board, ostream& out) {
    // Chess board with piece symbols
    // Standard chess notation: files a-h, ranks 1-8
    // Pieces: Uppercase for White (P R N B Q K), lowercase for Black (p r n b q k)

    out << "  +---+---+---+---+---+---+---+---+" << endl;
    for (int y = 0; y < 8; y++) {
        out << (8 - y) << " |";  // Rank numbers (8 down to 1)

        for (int x = 0; x < 8; x++) {
            ChessPiece piece = board.getPieceAt(x, y);

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
        out << " " << (8 - y) << endl;
        out << "  +---+---+---+---+---+---+---+---+" << endl;
    }
    out << "    a   b   c   d   e   f   g   h  " << endl;
}

int main(int argc, char *argv[])
{
    cout << "=== PREANN Chess Demo ===" << endl << endl;

    try {
        cout << "Board size: 8x8 (standard chess)" << endl << endl;

        // Create a ChessTask
        cout << "Creating ChessTask..." << endl;
        ChessTask task(BT_BIT, 1);
        cout << "Task created successfully!" << endl << endl;

        // Create two random untrained neural networks
        ParametersMap params;
        params.putNumber(Enumerations::enumTypeToString(ET_IMPLEMENTATION), IT_C);
        params.putNumber(Enumerations::enumTypeToString(ET_BUFFER), BT_BIT);
        params.putNumber(Enumerations::enumTypeToString(ET_FUNCTION), FT_BIPOLAR_SIGMOID);

        cout << "Creating two random neural networks..." << endl;
        Individual* player1 = task.getExample(&params);
        Individual* player2 = task.getExample(&params);
        cout << "Neural networks created (768-16-16-1 architecture)" << endl << endl;

        // Play a game and save board states
        cout << "Playing game and saving boards to output/games/chess_demo.txt..." << endl;
        ofstream gameFile("output/games/chess_demo.txt");

        if (!gameFile.is_open()) {
            throw string("Could not open output/games/chess_demo.txt for writing");
        }

        ChessBoard board(8, BT_BIT);
        board.initBoard();

        gameFile << "=== Chess Game Demo ===" << endl;
        gameFile << "Board size: 8x8 (standard chess)" << endl;
        gameFile << "Players: Two random untrained neural networks" << endl;
        gameFile << "White pieces (PLAYER_1): P R N B Q K (uppercase)" << endl;
        gameFile << "Black pieces (PLAYER_2): p r n b q k (lowercase)" << endl << endl;

        gameFile << "Initial board:" << endl;
        printBoard(board, gameFile);
        gameFile << endl;

        int moveNum = 0;
        SquareState turn = PLAYER_1;

        while (!board.endGame()) {
            if (!board.canMove(turn)) {
                gameFile << (turn == PLAYER_1 ? "White" : "Black") << " has no legal moves!" << endl;
                break;
            }

            moveNum++;

            // Select which player's turn it is
            Individual* currentPlayer = (turn == PLAYER_1) ? player1 : player2;

            // Make a move
            board.turn(turn, currentPlayer);

            // Print board state
            gameFile << "After move " << moveNum << " (" << (turn == PLAYER_1 ? "White" : "Black") << "):" << endl;
            printBoard(board, gameFile);
            gameFile << endl;

            turn = Board::opponent(turn);

            // Safety limit to prevent infinite games
            if (moveNum > 200) {
                gameFile << "Game ended due to move limit (200 moves)" << endl;
                break;
            }
        }

        // Determine game result
        if (board.isCheckmate(PLAYER_1)) {
            gameFile << "CHECKMATE! Black wins!" << endl;
        } else if (board.isCheckmate(PLAYER_2)) {
            gameFile << "CHECKMATE! White wins!" << endl;
        } else if (board.isStalemate(PLAYER_1) || board.isStalemate(PLAYER_2)) {
            gameFile << "STALEMATE! Game is a draw." << endl;
        } else {
            gameFile << "Game ended (possibly by move limit)" << endl;
        }

        gameFile << "Total moves: " << moveNum << endl;

        gameFile.close();
        cout << "Game saved to output/games/chess_demo.txt" << endl;
        cout << "Moves played: " << moveNum << endl;

        delete player1;
        delete player2;

    } catch (string& error) {
        cerr << "Error: " << error << endl;
        return 1;
    }

    cout << "Exit success." << endl;
    return 0;
}
