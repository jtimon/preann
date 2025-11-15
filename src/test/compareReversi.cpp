#include <iostream>
#include "tasks/reversiTask.h"
#include "genetic/individual.h"
#include "genetic/population.h"

using namespace std;

void playGames(ReversiBoard* board, Individual* player, const string& playerName, int numGames) {
    int wins = 0, losses = 0, ties = 0;
    int totalP1Points = 0, totalP2Points = 0;

    cout << "Playing " << numGames << " games: " << playerName << " vs Heuristic Opponent" << endl;

    for (int i = 0; i < numGames; i++) {
        board->initBoard();
        SquareState p1 = PLAYER_1;
        SquareState turn = PLAYER_1;

        while (!board->endGame()) {
            if (board->canMove(turn)) {
                if (turn == p1) {
                    board->turn(turn, player);  // Our player
                } else {
                    board->turn(turn, NULL);    // Heuristic opponent
                }
            }
            turn = Board::opponent(turn);
        }

        int p1Points = board->countPoints(p1);
        int p2Points = board->countPoints(Board::opponent(p1));

        totalP1Points += p1Points;
        totalP2Points += p2Points;

        if (p1Points > p2Points) wins++;
        else if (p2Points > p1Points) losses++;
        else ties++;
    }

    cout << "Results: " << wins << " wins, " << losses << " losses, " << ties << " ties" << endl;
    cout << "Win rate: " << (wins * 100.0 / numGames) << "%" << endl;
    cout << "Avg points: " << (totalP1Points / (float)numGames) << " vs " << (totalP2Points / (float)numGames) << endl;
    cout << "Avg differential: " << ((totalP1Points - totalP2Points) / (float)numGames) << endl << endl;
}

int main(int argc, char *argv[])
{
    cout << "=== PREANN Reversi Comparison ===" << endl << endl;

    try {
        ReversiBoard board(8, BT_BIT);

        // Create parameters
        ParametersMap params;
        params.putNumber(Enumerations::enumTypeToString(ET_IMPLEMENTATION), IT_CUDA_REDUC);
        params.putNumber(Enumerations::enumTypeToString(ET_BUFFER), BT_BIT);
        params.putNumber(Enumerations::enumTypeToString(ET_FUNCTION), FT_BIPOLAR_SIGMOID);

        ReversiTask task(8, BT_BIT, 1);

        // Test 1: Heuristic vs itself (baseline)
        cout << "========================================" << endl;
        cout << "TEST 1: Static Heuristic vs Itself" << endl;
        cout << "========================================" << endl;
        playGames(&board, NULL, "Heuristic", 20);

        // Test 2: Random untrained neural network
        cout << "========================================" << endl;
        cout << "TEST 2: Untrained Random Neural Network" << endl;
        cout << "========================================" << endl;
        Individual* randomNN = task.getExample(&params);
        playGames(&board, randomNN, "Random NN", 20);

        // Test 3: Train a network and test again
        cout << "========================================" << endl;
        cout << "TEST 3: Training Neural Network..." << endl;
        cout << "========================================" << endl;

        params.putNumber(Population::MUTATION_RANGE, 0.5);
        params.putNumber(Population::SIZE, 50);
        params.putNumber(Population::NUM_SELECTION, 25);
        params.putNumber(Population::NUM_CROSSOVER, 20);
        params.putNumber(Population::NUM_PRESERVE, 5);
        params.putNumber(Population::RESET_NUM, 0);
        params.putNumber(Enumerations::enumTypeToString(ET_SELECTION_ALGORITHM), SA_TOURNAMENT);
        params.putNumber(Population::TOURNAMENT_SIZE, 5);
        params.putNumber(Enumerations::enumTypeToString(ET_CROSS_LEVEL), CL_WEIGH);
        params.putNumber(Enumerations::enumTypeToString(ET_CROSS_ALG), CA_UNIFORM);
        params.putNumber(Population::UNIFORM_CROSS_PROB, 0.5);
        params.putNumber(Enumerations::enumTypeToString(ET_MUTATION_ALG), MA_PER_INDIVIDUAL);
        params.putNumber(Population::MUTATION_NUM, 5);

        ReversiTask trainTask(8, BT_BIT, 2);
        Individual* example = trainTask.getExample(&params);
        Population population(&trainTask, example, 50, 100);
        population.setParams(&params);

        cout << "Training for 100 generations (population: 50)..." << endl;
        for (unsigned gen = 0; gen < 100; gen++) {
            population.nextGeneration();
            if ((gen + 1) % 20 == 0) {
                Individual* best = population.getBestIndividual();
                cout << "Generation " << (gen + 1) << ": Best = " << best->getFitness();
                cout << ", Avg = " << population.getAverageFitness() << endl;
            }
        }

        Individual* trainedNN = population.getBestIndividual();
        cout << "\nTraining complete!" << endl << endl;

        cout << "========================================" << endl;
        cout << "TEST 4: Trained Neural Network (100 gen)" << endl;
        cout << "========================================" << endl;
        playGames(&board, trainedNN, "Trained NN (100 gen)", 20);

        delete randomNN;

    } catch (string& error) {
        cerr << "Error: " << error << endl;
        return 1;
    }

    cout << "Exit success." << endl;
    return 0;
}
