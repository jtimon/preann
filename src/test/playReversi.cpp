#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <sys/stat.h>
#include "tasks/reversiTask.h"
#include "genetic/population.h"
#include "common/chronometer.h"
#include "common/dummy.h"

using namespace std;

bool fileExists(const char* filename) {
    struct stat buffer;
    return (stat(filename, &buffer) == 0);
}

string getTopologyString(Individual* individual, unsigned boardSize)
{
    // Generate topology string in format: b64_b8_b8_f1
    // where b=BT_BIT, p=BT_SIGN (bipolar), f=BT_FLOAT
    ostringstream topology;

    // For Reversi, input is boardSize² BIT neurons (8x8=64 for standard board)
    // This is hardcoded since playReversi.cpp is reversi-specific
    topology << "b" << (boardSize * boardSize);

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
    cout << "=== PREANN Reversi: Persistent Evolution Training ===" << endl << endl;

    try {
        // Parse command-line arguments
        if (argc < 3) {
            cout << "Usage: " << argv[0] << " <generations> <checkpoint_interval>" << endl;
            cout << "Example: " << argv[0] << " 100 25" << endl;
            cout << endl;
            cout << "Trains a population of neural networks to play Reversi." << endl;
            cout << "Automatically saves progress to data/populations/reversi_[topology]_P[size].pop" << endl;
            cout << "Resume training by running the same command again." << endl;
            return 1;
        }

        unsigned generations = atoi(argv[1]);
        unsigned checkpointInterval = atoi(argv[2]);

        // Create Reversi task (8x8 board, 2 test games per individual)
        cout << "Creating Reversi task (8x8 board, 2 test games per individual)..." << endl;
        unsigned boardSize = 8;
        ReversiTask reversiTask(boardSize, BT_BIT, 2);

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
        // REM: crossover must be even
        unsigned populationSize = 100;
        params.putNumber(Population::SIZE, populationSize);
        params.putNumber(Population::NUM_SELECTION, populationSize / 2);      // 50 selected
        params.putNumber(Population::NUM_CROSSOVER, populationSize / 2 - 10); // 40 crossover
        params.putNumber(Population::NUM_PRESERVE, 10);                       // 10 elite

        // Genetic operators
        params.putNumber(Enumerations::enumTypeToString(ET_CROSS_LEVEL), CL_WEIGH);
        params.putNumber(Enumerations::enumTypeToString(ET_CROSS_ALG), CA_UNIFORM);
        params.putNumber(Population::UNIFORM_CROSS_PROB, 0.5);
        params.putNumber(Enumerations::enumTypeToString(ET_SELECTION_ALGORITHM), SA_TOURNAMENT);
        params.putNumber(Population::TOURNAMENT_SIZE, 5);
        params.putNumber(Enumerations::enumTypeToString(ET_MUTATION_ALG), MA_PER_INDIVIDUAL);
        params.putNumber(Population::MUTATION_NUM, 5);
        params.putNumber(Enumerations::enumTypeToString(ET_RESET_ALG), RA_DISABLED);

        // Load or create population
        Population* population = NULL;
        Individual* example = NULL;
        string topology;
        string saveFileString;
        const char* saveFile = NULL;

        // First, try to create example to determine topology and filename
        example = reversiTask.getExample(&params);
        topology = getTopologyString(example, boardSize);

        // Build save filename: reversi_[topology]_P[populationSize].pop
        ostringstream saveFileStream;
        saveFileStream << "data/populations/reversi_" << topology << "_P" << populationSize << ".pop";
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

            population = new Population(&reversiTask, populationSize);
            population->load(loadStream);
            population->setParams(&params);
            fclose(loadStream);

            cout << "Loaded population:" << endl;
            cout << "  Generation: " << population->getGeneration() << endl;
            cout << "  Population size: " << populationSize << endl;
            cout << endl;

            cout << "Fitness from previous run (may use old fitness function): ";
            cout << "Best=" << population->getBestIndividual()->getFitness();
            cout << " | Avg=" << population->getAverageFitness() << endl;

            // Enable competitive co-evolution BEFORE re-evaluation
            // This ensures re-evaluation uses the same fitness function as evolution
            cout << "Enabling competitive co-evolution (population vs adversary)..." << endl;
            reversiTask.setAdversary(population->getBestIndividual());
            cout << endl;

            // Re-evaluate all individuals with current fitness function
            cout << "Re-evaluating all individuals with current fitness function..." << endl;
            population->reevaluateAndSort();

            cout << "Fitness after re-evaluation: ";
            cout << "Best=" << population->getBestIndividual()->getFitness();
            cout << " | Avg=" << population->getAverageFitness() << endl;
            cout << endl;

        } else {
            cout << "Creating new population (no save file found)..." << endl;
            population = new Population(&reversiTask, example, populationSize, 5.0);
            population->setParams(&params);

            cout << "Neural network architecture:" << endl;
            cout << "  Input layer: " << (boardSize * boardSize) << " neurons (8x8 board encoding)" << endl;
            cout << "  Hidden layer 1: " << boardSize << " neurons (BIT buffer)" << endl;
            cout << "  Hidden layer 2: " << boardSize << " neurons (BIT buffer)" << endl;
            cout << "  Output layer: 1 neuron (FLOAT buffer)" << endl;
            cout << "  Feedforward with skip connections" << endl;
            cout << endl;

            // Keep greedy computer opponent for new populations (bootstrap)
            cout << "Using greedy computer for bootstrapping..." << endl;
        }

        // Evolution loop
        cout << "=== EVOLVING FOR " << generations << " GENERATIONS ===" << endl;
        cout << "Checkpoint interval: " << checkpointInterval << " generations" << endl << endl;

        Chronometer chrono;
        chrono.start();

        for (unsigned gen = 0; gen < generations; gen++) {
            population->nextGeneration();

            unsigned currentGen = population->getGeneration();

            // Print progress every generation
            cout << "Generation " << currentGen << ": ";
            cout << "Best = " << population->getBestIndividual()->getFitness();
            cout << " | Avg = " << population->getAverageFitness();
            cout << endl;

            // Save checkpoint
            if ((gen + 1) % checkpointInterval == 0) {
                FILE* saveStream = fopen(saveFile, "wb");
                if (saveStream) {
                    population->save(saveStream);
                    fclose(saveStream);
                    cout << "  --> Checkpoint saved at generation " << currentGen << endl;
                } else {
                    cout << "  WARNING: Could not save checkpoint!" << endl;
                }
            }
        }

        chrono.stop();

        // Save final state
        FILE* finalSaveStream = fopen(saveFile, "wb");
        if (finalSaveStream) {
            population->save(finalSaveStream);
            fclose(finalSaveStream);
            cout << endl << "Final state saved to " << saveFile << endl;
        }

        // Test best individual separately against each opponent
        cout << endl << "Testing best individual with fresh games..." << endl;
        Individual* bestIndividual = population->getBestIndividual();
        float storedFitness = bestIndividual->getFitness();  // Fitness from training
        bool hasAdversary = reversiTask.hasAdversary();

        // Test vs greedy only (2 new games)
        reversiTask.testBootstrap(bestIndividual);
        float greedyFitness = bestIndividual->getFitness();

        // Test vs neural adversary only (2 new games), if adversary exists
        float neuralFitness = 0;
        float combinedFitness = greedyFitness;
        if (hasAdversary) {
            reversiTask.testAdversary(bestIndividual);
            neuralFitness = bestIndividual->getFitness();
            // Calculate weighted average of 10 new games (2 neural + 8 greedy)
            // Weight by number of games: (2*neural + 8*greedy) / 10
            combinedFitness = (2.0 * neuralFitness + 8.0 * greedyFitness) / 10.0;
        }

        // Restore original fitness
        bestIndividual->setFitness(storedFitness);

        // Print final results
        cout << endl << "=== FINAL RESULTS ===" << endl;
        cout << "Generation: " << population->getGeneration() << endl;
        cout << "Stored fitness (from training): " << storedFitness << endl;
        if (hasAdversary) {
            cout << "Fresh test results (10 new games):" << endl;
            cout << "  Combined average: " << combinedFitness << endl;
            cout << "  vs neural adversary (2 games): " << neuralFitness << endl;
            cout << "  vs greedy computer (8 games):  " << greedyFitness << endl;
        } else {
            cout << "Fresh test results (8 new games):" << endl;
            cout << "  vs greedy computer: " << greedyFitness << endl;
        }
        cout << "Avg fitness (population):  " << population->getAverageFitness() << endl;
        cout << endl;

        float seconds = chrono.getSeconds();
        int hours = (int)(seconds / 3600);
        int minutes = (int)((seconds - hours * 3600) / 60);
        float remainingSeconds = seconds - hours * 3600 - minutes * 60;
        cout << "Evolution completed in " << seconds << " seconds";
        cout << " (" << hours << "h " << minutes << "m " << remainingSeconds << "s)" << endl;

    } catch (string& error) {
        cerr << "Error: " << error << endl;
        return 1;
    }

    cout << endl << "Exit success." << endl;
    return 0;
}
