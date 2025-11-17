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

string getTopologyString(Individual* individual)
{
    // Generate topology string in format: b768_b128_b128_p32_f1
    // where b=BT_BIT, p=BT_SIGN (bipolar), f=BT_FLOAT
    ostringstream topology;

    // For chess, input is always 768 BIT neurons (8x8x12 piece encoding)
    // This is hardcoded since playChess.cpp is chess-specific
    topology << "b768";

    // Add all layers (hidden + output)
    for (unsigned i = 0; i < individual->getNumLayers(); i++) {
        Interface* layer = individual->getOutput(i);
        BufferType layerType = layer->getBufferType();
        unsigned layerSize = layer->getSize();

        char prefix = 'f';
        if (layerType == BT_BIT) prefix = 'b';
        else if (layerType == BT_SIGN) prefix = 'p';

        topology << "_" << prefix << layerSize;
    }

    return topology.str();
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
            cout << "Automatically saves progress to data/populations/chess_[topology]_P[size].pop" << endl;
            cout << "Resume training by running the same command again." << endl;
            return 1;
        }

        unsigned generations = atoi(argv[1]);
        unsigned checkpointInterval = atoi(argv[2]);

        // Create Chess task (game logging disabled for now)
        cout << "Creating Chess task (8x8 board, 2 test games per individual)..." << endl;
        ChessTask chessTask(BT_BIT, 2, false);

        // Set up population parameters
        ParametersMap params;

        // Select implementation based on build type
        #if defined(CUDA_IMPL)
            cout << "Using CUDA implementation (IT_CUDA_OUT)" << endl;
            params.putNumber(Enumerations::enumTypeToString(ET_IMPLEMENTATION), IT_CUDA_OUT);
        #elif defined(SSE2_IMPL)
            cout << "Using SSE2 implementation (IT_SSE2)" << endl;
            params.putNumber(Enumerations::enumTypeToString(ET_IMPLEMENTATION), IT_SSE2);
        #else
            cout << "Using C++ implementation (IT_C)" << endl;
            params.putNumber(Enumerations::enumTypeToString(ET_IMPLEMENTATION), IT_C);
        #endif

        params.putNumber(Enumerations::enumTypeToString(ET_BUFFER), BT_BIT);
        params.putNumber(Enumerations::enumTypeToString(ET_FUNCTION), FT_BIPOLAR_SIGMOID);
        params.putNumber(Dummy::WEIGHS_RANGE, 5.0);
        params.putNumber(Population::MUTATION_RANGE, 0.5);

        // Population structure
        unsigned populationSize = 50;
        params.putNumber(Population::SIZE, populationSize);
        params.putNumber(Population::NUM_SELECTION, populationSize / 2);      // 25 selected
        params.putNumber(Population::NUM_CROSSOVER, populationSize / 2 - 1);  // 24 crossover (REM: must be even)
        params.putNumber(Population::NUM_PRESERVE, 1);                        // 1 elite

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
        Individual* example = NULL;
        string topology;
        string saveFileString;
        const char* saveFile = NULL;

        // First, try to create example to determine topology and filename
        example = chessTask.getExample(&params);
        topology = getTopologyString(example);

        // Build save filename: chess_[topology]_P[populationSize].pop
        ostringstream saveFileStream;
        saveFileStream << "data/populations/chess_" << topology << "_P" << populationSize << ".pop";
        saveFileString = saveFileStream.str();
        saveFile = saveFileString.c_str();

        cout << "Configuration:" << endl;
        cout << "  Generations: " << generations << endl;
        cout << "  Checkpoint interval: " << checkpointInterval << endl;
        cout << "  Topology: " << topology << endl;
        cout << "  Population size: " << populationSize << endl;
        cout << "  Save file: " << saveFile << endl << endl;

        if (fileExists(saveFile)) {
            cout << "Loading existing population from " << saveFile << "..." << endl;

            // Don't need the example when loading
            delete example;
            example = NULL;

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
            cout << "Enabling competitive co-evolution (population vs adversary)..." << endl;
            chessTask.setAdversary(population->getBestIndividual());
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

            // Update adversary after re-evaluation (population may have re-sorted)
            chessTask.setAdversary(population->getBestIndividual());

            cout << "Fitness after re-evaluation: ";
            cout << "Best=" << population->getBestIndividual()->getFitness();
            cout << " | Avg=" << population->getAverageFitness() << endl;
            cout << endl;

        } else {
            cout << "Creating new population (no save file found)..." << endl;
            population = new Population(&chessTask, example, populationSize, 5.0);
            population->setParams(&params);

            cout << "Neural network architecture:" << endl;
            cout << "  Input layer: 768 neurons (8x8x12 piece-aware encoding)" << endl;
            cout << "  Hidden layer 1: 8 neurons (FLOAT buffer)" << endl;
            cout << "  Hidden layer 2: 4 neurons (FLOAT buffer)" << endl;
            cout << "  Hidden layer 3: 2 neurons (SIGN buffer)" << endl;
            cout << "  Output layer: 1 neuron (FLOAT buffer, IDENTITY function)" << endl;
            cout << "  Feedforward: 768->8->4->2->1" << endl;
            cout << "  Recurrent: NONE (feedforward only)" << endl;
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
        // Calculate bootstrap fitness (vs random opponent) as progress indicator
        cout << endl << "Testing best individual against random opponent..." << endl;
        Individual* bestIndividual = population->getBestIndividual();
        float adversaryFitness = bestIndividual->getFitness();  // Save original fitness
        Individual* currentAdversary = chessTask.getAdversary();
        chessTask.setAdversary(NULL);  // Test against random opponent (NULL = random moves)
        chessTask.test(bestIndividual);
        float bootstrapFitness = bestIndividual->getFitness();
        bestIndividual->setFitness(adversaryFitness);  // Restore original fitness
        chessTask.setAdversary(currentAdversary);  // Restore adversary

        cout << endl << "=== FINAL RESULTS ===" << endl;
        cout << "Generation: " << population->getGeneration() << endl;
        cout << "Best fitness (vs adversary): " << adversaryFitness << endl;
        cout << "Bootstrap fitness (vs random): " << bootstrapFitness << endl;
        cout << "Avg fitness:  " << population->getAverageFitness() << endl;
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

        // Game logging happens automatically during training
        // No need for separate demo game - checkmate games are logged as they occur

        delete population;

    } catch (string& error) {
        cerr << "Error: " << error << endl;
        return 1;
    }

    cout << endl << "Exit success." << endl;
    return 0;
}
