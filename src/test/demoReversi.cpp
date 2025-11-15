#include <iostream>
#include "tasks/reversiTask.h"
#include "genetic/individual.h"

using namespace std;

void playAndShowGame(ReversiBoard* board, Individual* neuralNet, bool showBoard) {
    board->initBoard();

    SquareState aiPlayer = PLAYER_1;  // Neural network plays as PLAYER_1
    SquareState turn = PLAYER_1;

    if (showBoard) {
        cout << "Initial board:" << endl;
        board->print();
        cout << endl;
    }

    int moveCount = 0;
    while (!board->endGame()) {
        if (board->canMove(turn)) {
            if (turn == aiPlayer) {
                // AI (neural network) move
                board->turn(turn, neuralNet);
                if (showBoard) {
                    cout << "After AI (O) move " << ++moveCount << ":" << endl;
                    board->print();
                    cout << endl;
                }
            } else {
                // Computer opponent (simple heuristic) move
                board->turn(turn, NULL);
                if (showBoard) {
                    cout << "After opponent (X) move:" << endl;
                    board->print();
                    cout << endl;
                }
            }
        }
        turn = Board::opponent(turn);
    }

    int aiPoints = board->countPoints(aiPlayer);
    int oppPoints = board->countPoints(Board::opponent(aiPlayer));

    cout << "Game over! AI: " << aiPoints << " points, Opponent: " << oppPoints << " points";
    if (aiPoints > oppPoints) {
        cout << " - AI WINS!";
    } else if (oppPoints > aiPoints) {
        cout << " - AI LOSES";
    } else {
        cout << " - TIE";
    }
    cout << endl;
}

int main(int argc, char *argv[])
{
    cout << "=== PREANN Reversi Demo ===" << endl << endl;

    try {
        // Create a Reversi board
        ReversiBoard board(8, BT_BIT);

        // Create a random untrained neural network
        ParametersMap params;
        params.putNumber(Enumerations::enumTypeToString(ET_IMPLEMENTATION), IT_CUDA_REDUC);
        params.putNumber(Enumerations::enumTypeToString(ET_BUFFER), BT_BIT);
        params.putNumber(Enumerations::enumTypeToString(ET_FUNCTION), FT_BIPOLAR_SIGMOID);

        ReversiTask task(8, BT_BIT, 1);
        Individual* randomAI = task.getExample(&params);

        cout << "Created random untrained neural network" << endl;
        cout << "Neural network: " << randomAI->getNumLayers() << " layers" << endl << endl;

        cout << "========================================" << endl;
        cout << "Playing 3 demo games (showing boards):" << endl;
        cout << "O = AI (neural network)" << endl;
        cout << "X = Opponent (simple heuristic)" << endl;
        cout << "========================================" << endl << endl;

        // Play 3 games and show the boards
        for (int i = 0; i < 3; i++) {
            cout << "--- GAME " << (i+1) << " ---" << endl;
            playAndShowGame(&board, randomAI, true);
            cout << endl;
        }

        // Now play 10 more games without showing boards for statistics
        cout << "========================================" << endl;
        cout << "Playing 10 more games for statistics:" << endl;
        cout << "========================================" << endl;

        int wins = 0, losses = 0, ties = 0;
        int totalAIPoints = 0, totalOppPoints = 0;

        for (int i = 0; i < 10; i++) {
            board.initBoard();
            SquareState aiPlayer = PLAYER_1;
            SquareState turn = PLAYER_1;

            while (!board.endGame()) {
                if (board.canMove(turn)) {
                    if (turn == aiPlayer) {
                        board.turn(turn, randomAI);
                    } else {
                        board.turn(turn, NULL);
                    }
                }
                turn = Board::opponent(turn);
            }

            int aiPoints = board.countPoints(aiPlayer);
            int oppPoints = board.countPoints(Board::opponent(aiPlayer));

            totalAIPoints += aiPoints;
            totalOppPoints += oppPoints;

            if (aiPoints > oppPoints) wins++;
            else if (oppPoints > aiPoints) losses++;
            else ties++;

            cout << "Game " << (i+1) << ": AI " << aiPoints << " - " << oppPoints << " Opp";
            if (aiPoints > oppPoints) cout << " (WIN)";
            else if (aiPoints < oppPoints) cout << " (LOSS)";
            else cout << " (TIE)";
            cout << endl;
        }

        cout << endl << "========================================" << endl;
        cout << "STATISTICS (10 games):" << endl;
        cout << "========================================" << endl;
        cout << "Wins: " << wins << ", Losses: " << losses << ", Ties: " << ties << endl;
        cout << "Win rate: " << (wins * 100.0 / 10.0) << "%" << endl;
        cout << "Average AI points: " << (totalAIPoints / 10.0) << endl;
        cout << "Average Opponent points: " << (totalOppPoints / 10.0) << endl;
        cout << "Average point differential: " << ((totalAIPoints - totalOppPoints) / 10.0) << endl;

        delete randomAI;

    } catch (string& error) {
        cerr << "Error: " << error << endl;
        return 1;
    }

    cout << endl << "Exit success." << endl;
    return 0;
}
