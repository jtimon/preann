/*
 * playChess.cpp
 *
 * Train neural networks to play chess through genetic evolution
 * Supports persistent training with automatic save/load
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <sys/stat.h>
#include "tasks/chessTask.h"
#include "genetic/population.h"
#include "common/chronometer.h"
#include "common/dummy.h"

using namespace std;

bool fileExists(const char* filename)
{
    struct stat buffer;
    return (stat(filename, &buffer) == 0);
}

int main(int argc, char *argv[])
{
    cout << "=== PREANN Chess: Persistent Evolution Training ===" << endl << endl;

    try {
        // Parse command-line arguments
        if (argc < 3) {
            cout << "Usage: " << argv[0] << " <generations> <checkpoint_interval>" << endl;
            cout << "Example: " << argv[0] << " 100 25" << endl;
            cout << endl;
            cout << "Trains a population of neural networks to play chess." << endl;
            cout << "Automatically saves progress to output/data/chess_persist.pop" << endl;
            cout << "Resume training by running the same command again." << endl;
            return 1;
        }

        unsigned generations = atoi(argv[1]);
        unsigned checkpointInterval = atoi(argv[2]);
        const char* saveFile = "data/populations/chess_persist.pop";

        cout << "Configuration:" << endl;
        cout << "  Generations: " << generations << endl;
        cout << "  Checkpoint interval: " << checkpointInterval << endl;
        cout << "  Save file: " << saveFile << endl << endl;

        // Create Chess task
        cout << "Creating Chess task (8x8 board, 2 test games per individual)..." << endl;
        ChessTask chessTask(BT_BIT, 2);

        // Set up population parameters
        ParametersMap params;
        params.putNumber(Enumerations::enumTypeToString(ET_IMPLEMENTATION), IT_C);
        params.putNumber(Enumerations::enumTypeToString(ET_BUFFER), BT_BIT);
        params.putNumber(Enumerations::enumTypeToString(ET_FUNCTION), FT_BIPOLAR_SIGMOID);
        params.putNumber(Dummy::WEIGHS_RANGE, 5.0);
        params.putNumber(Population::MUTATION_RANGE, 0.5);

        // Population structure
        unsigned populationSize = 50;
        params.putNumber(Population::SIZE, populationSize);
        params.putNumber(Population::NUM_SELECTION, populationSize / 2);      // 25 selected
        params.putNumber(Population::NUM_CROSSOVER, populationSize / 2 - 5);  // 20 crossover
        params.putNumber(Population::NUM_PRESERVE, 5);                        // 5 elite

        // Genetic operators
        params.putNumber(Enumerations::enumTypeToString(ET_CROSS_LEVEL), CL_WEIGH);
        params.putNumber(Enumerations::enumTypeToString(ET_CROSS_ALG), CA_UNIFORM);
        params.putNumber(Population::UNIFORM_CROSS_PROB, 0.5);
        params.putNumber(Enumerations::enumTypeToString(ET_SELECTION_ALGORITHM), SA_TOURNAMENT);
        params.putNumber(Population::TOURNAMENT_SIZE, 3);
        params.putNumber(Enumerations::enumTypeToString(ET_MUTATION_ALG), MA_PER_INDIVIDUAL);
        params.putNumber(Population::MUTATION_NUM, 3);
        params.putNumber(Enumerations::enumTypeToString(ET_RESET_ALG), RA_DISABLED);

        // Load or create population
        Population* population = NULL;
        if (fileExists(saveFile)) {
            cout << "Loading existing population from " << saveFile << "..." << endl;
            FILE* loadStream = fopen(saveFile, "rb");
            if (!loadStream) {
                cerr << "Error: Could not open save file for reading" << endl;
                return 1;
            }

            population = new Population(&chessTask, populationSize);
            population->load(loadStream);
            population->setParams(&params);
            fclose(loadStream);

            // Enable competitive co-evolution for loaded populations
            cout << "Enabling competitive co-evolution (population vs best)..." << endl;
            chessTask.setBestOpponent(population->getBestIndividual());
            cout << endl;

            cout << "Loaded population:" << endl;
            cout << "  Generation: " << population->getGeneration() << endl;
            cout << "  Population size: " << populationSize << endl;
            cout << endl;

            cout << "Fitness from previous run (may use old fitness function): ";
            cout << "Best=" << population->getBestIndividual()->getFitness();
            cout << " | Avg=" << population->getAverageFitness() << endl;

            // Re-evaluate all individuals with current fitness function
            cout << "Re-evaluating all individuals with current fitness function..." << endl;
            population->reevaluateAndSort();

            cout << "Fitness after re-evaluation: ";
            cout << "Best=" << population->getBestIndividual()->getFitness();
            cout << " | Avg=" << population->getAverageFitness() << endl;
            cout << endl;

        } else {
            cout << "Creating new population (no save file found)..." << endl;
            Individual* example = chessTask.getExample(&params);
            population = new Population(&chessTask, example, populationSize, 5.0);
            population->setParams(&params);

            cout << "Neural network architecture:" << endl;
            cout << "  Input layer: 768 neurons (8x8x12 piece-aware encoding)" << endl;
            cout << "  Hidden layer 1: 16 neurons" << endl;
            cout << "  Hidden layer 2: 16 neurons" << endl;
            cout << "  Output layer: 1 neuron (position evaluation)" << endl;
            cout << "  Total connections: 768->16->16->1" << endl;
            cout << endl;

            // Keep random opponent for new populations (bootstrap)
            cout << "Using random opponent for bootstrapping..." << endl;
        }

        // Evolution loop
        cout << "=== EVOLVING FOR " << generations << " GENERATIONS ===" << endl;
        Chronometer chrono;
        chrono.start();

        for (unsigned gen = 0; gen < generations; gen++) {
            population->nextGeneration();

            // Print progress every generation
            cout << "Gen " << population->getGeneration() << ": ";
            cout << "Best=" << population->getBestIndividual()->getFitness();
            cout << " | Avg=" << population->getAverageFitness();
            cout << endl;

            // Save checkpoint at specified intervals
            if ((gen + 1) % checkpointInterval == 0) {
                FILE* saveStream = fopen(saveFile, "wb");
                if (saveStream) {
                    population->save(saveStream);
                    fclose(saveStream);
                    cout << "  --> Checkpoint saved at generation " << population->getGeneration() << endl;
                }
            }
        }

        chrono.stop();

        // Save final state
        FILE* finalSaveStream = fopen(saveFile, "wb");
        if (finalSaveStream) {
            population->save(finalSaveStream);
            fclose(finalSaveStream);
            cout << endl << "Final population saved to " << saveFile << endl;
        }

        // Print final statistics
        cout << endl << "=== FINAL RESULTS ===" << endl;
        cout << "Generation " << population->getGeneration() << ": ";
        cout << "Best=" << population->getBestIndividual()->getFitness();
        cout << " | Avg=" << population->getAverageFitness() << endl;
        cout << endl;

        // Convert time to hours/minutes/seconds
        float seconds = chrono.getSeconds();
        int hours = (int)(seconds / 3600);
        int minutes = (int)((seconds - hours * 3600) / 60);
        int secs = (int)(seconds - hours * 3600 - minutes * 60);

        cout << "Evolution completed in " << seconds << " seconds";
        if (hours > 0 || minutes > 0) {
            cout << " (" << hours << "h " << minutes << "m " << secs << "s)";
        }
        cout << endl;

        // Play a demonstration game with the best individual vs computer
        cout << endl << "Playing demonstration game (best individual vs computer)..." << endl;

        ostringstream gameFilename;
        gameFilename << "output/games/chess_generation" << population->getGeneration() << ".txt";

        ofstream gameFile(gameFilename.str().c_str());
        if (gameFile.is_open()) {
            ChessBoard demoBoard(8, BT_BIT);
            demoBoard.initBoard();

            gameFile << "=== Chess Training Game ===" << endl;
            gameFile << "Generation: " << population->getGeneration() << endl;
            gameFile << "Best individual fitness: " << population->getBestIndividual()->getFitness() << endl;
            gameFile << "White (trained NN) vs Black (random computer)" << endl;
            gameFile << "White pieces: P R N B Q K (uppercase)" << endl;
            gameFile << "Black pieces: p r n b q k (lowercase)" << endl << endl;

            gameFile << "Initial board:" << endl;
            demoBoard.printBoard(gameFile);
            gameFile << endl;

            Individual* bestIndividual = population->getBestIndividual();
            int moveNum = 0;
            SquareState turn = PLAYER_1;

            while (!demoBoard.endGame()) {
                if (!demoBoard.canMove(turn)) {
                    gameFile << (turn == PLAYER_1 ? "White" : "Black") << " has no legal moves!" << endl;
                    break;
                }

                moveNum++;

                // Best individual plays as White, computer as Black
                if (turn == PLAYER_1) {
                    demoBoard.turn(turn, bestIndividual);
                } else {
                    demoBoard.turn(turn, NULL);  // NULL triggers computer opponent
                }

                gameFile << "After move " << moveNum << " (" << (turn == PLAYER_1 ? "White" : "Black") << "):" << endl;
                demoBoard.printBoard(gameFile);
                gameFile << endl;

                turn = Board::opponent(turn);

                if (moveNum > 400) {
                    gameFile << "Game ended due to move limit (400 moves)" << endl;
                    break;
                }
            }

            // Determine game result
            if (demoBoard.isCheckmate(PLAYER_1)) {
                gameFile << "CHECKMATE! Black wins!" << endl;
            } else if (demoBoard.isCheckmate(PLAYER_2)) {
                gameFile << "CHECKMATE! White wins!" << endl;
            } else if (demoBoard.isStalemate(PLAYER_1) || demoBoard.isStalemate(PLAYER_2)) {
                gameFile << "STALEMATE! Game is a draw." << endl;
            } else {
                gameFile << "Game ended (possibly by move limit)" << endl;
            }

            gameFile << "Total moves: " << moveNum << endl;
            gameFile.close();

            cout << "Demonstration game saved to " << gameFilename.str() << endl;
            cout << "Moves played: " << moveNum << endl;
        } else {
            cout << "Warning: Could not save demonstration game" << endl;
        }

        delete population;

    } catch (string& error) {
        cerr << "Error: " << error << endl;
        return 1;
    }

    cout << endl << "Exit success." << endl;
    return 0;
}
