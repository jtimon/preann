#include <iostream>
#include <cstdlib>
#include <sys/stat.h>
#include "tasks/reversiTask.h"
#include "genetic/population.h"
#include "common/chronometer.h"

using namespace std;

bool fileExists(const char* filename) {
    struct stat buffer;
    return (stat(filename, &buffer) == 0);
}

int main(int argc, char *argv[])
{
    cout << "=== PREANN Reversi Persistent Training ===" << endl << endl;

    try {
        // Parse positional arguments: ./bin/playReversi.exe <generations> <checkpoint>
        if (argc < 3) {
            cout << "Usage: " << argv[0] << " <generations> <checkpoint_interval>" << endl;
            cout << "  <generations>          Number of generations to evolve" << endl;
            cout << "  <checkpoint_interval>  Save checkpoint every N generations" << endl;
            cout << endl;
            cout << "Example: " << argv[0] << " 100 25" << endl;
            cout << "  First run: creates population, evolves 100 generations, saves every 25" << endl;
            cout << "  Second run: loads existing, evolves 100 more generations" << endl;
            return 1;
        }

        unsigned generations = atoi(argv[1]);
        unsigned checkpointInterval = atoi(argv[2]);

        if (generations == 0 || checkpointInterval == 0) {
            cout << "Error: generations and checkpoint_interval must be positive integers" << endl;
            return 1;
        }

        const char* saveFile = "output/data/populations/reversi_persist.pop";

        // Create Reversi task (8x8 board, 2 test games)
        cout << "Creating Reversi task (8x8 board, 2 test games)..." << endl;
        ReversiTask reversiTask(8, BT_BIT, 2);

        // Set up population parameters
        ParametersMap params;
        params.putNumber(Enumerations::enumTypeToString(ET_IMPLEMENTATION), IT_CUDA_REDUC);
        params.putNumber(Enumerations::enumTypeToString(ET_BUFFER), BT_BIT);
        params.putNumber(Enumerations::enumTypeToString(ET_FUNCTION), FT_BIPOLAR_SIGMOID);
        params.putNumber(Population::MUTATION_RANGE, 0.5);
        params.putNumber(Population::SIZE, 100);
        params.putNumber(Population::NUM_SELECTION, 50);
        params.putNumber(Population::NUM_CROSSOVER, 40);
        params.putNumber(Population::NUM_PRESERVE, 10);
        params.putNumber(Population::RESET_NUM, 0);
        params.putNumber(Enumerations::enumTypeToString(ET_SELECTION_ALGORITHM), SA_TOURNAMENT);
        params.putNumber(Population::TOURNAMENT_SIZE, 5);
        params.putNumber(Enumerations::enumTypeToString(ET_CROSS_LEVEL), CL_WEIGH);
        params.putNumber(Enumerations::enumTypeToString(ET_CROSS_ALG), CA_UNIFORM);
        params.putNumber(Population::UNIFORM_CROSS_PROB, 0.5);
        params.putNumber(Enumerations::enumTypeToString(ET_MUTATION_ALG), MA_PER_INDIVIDUAL);
        params.putNumber(Population::MUTATION_NUM, 5);

        Population* population = NULL;
        unsigned startGeneration = 0;

        // Check if save file exists
        if (fileExists(saveFile)) {
            cout << "Found existing save file: " << saveFile << endl;
            cout << "Loading population..." << endl;

            FILE* loadStream = fopen(saveFile, "rb");
            if (!loadStream) {
                string error = "Error: Could not open file for reading: " + string(saveFile);
                throw error;
            }

            population = new Population(&reversiTask, 100);
            population->load(loadStream);
            population->setParams(&params);
            fclose(loadStream);

            startGeneration = population->getGeneration();
            cout << "Loaded generation " << startGeneration << " with " << population->getSize() << " individuals" << endl;
        } else {
            cout << "No existing save file found. Creating new population..." << endl;

            Individual* example = reversiTask.getExample(&params);
            cout << "Created example neural network with " << example->getNumLayers() << " layers" << endl;

            population = new Population(&reversiTask, example, 100, 5.0);
            population->setParams(&params);

            startGeneration = 0;
            cout << "Created population of 100 individuals" << endl;
        }

        // Print fitness before first generation of this run
        cout << endl << "=== FITNESS BEFORE THIS RUN ===" << endl;
        cout << "Generation: " << population->getGeneration() << endl;
        cout << "Best fitness: " << population->getBestIndividual()->getFitness() << endl;
        cout << "Avg fitness:  " << population->getAverageFitness() << endl;
        cout << endl;

        // Evolution loop
        cout << "=== EVOLVING FOR " << generations << " GENERATIONS ===" << endl;
        cout << "Checkpoint interval: " << checkpointInterval << " generations" << endl;
        cout << "Goal fitness: " << reversiTask.getGoal() << endl << endl;

        Chronometer chrono;
        chrono.start();

        for (unsigned gen = 0; gen < generations; gen++) {
            population->nextGeneration();

            unsigned currentGen = population->getGeneration();

            // Print progress
            if ((gen + 1) % 10 == 0 || gen == 0) {
                cout << "Generation " << currentGen << ": ";
                cout << "Best = " << population->getBestIndividual()->getFitness();
                cout << " | Avg = " << population->getAverageFitness();
                cout << endl;
            }

            // Save checkpoint
            if ((gen + 1) % checkpointInterval == 0) {
                // Print fitness at checkpoint
                if ((gen + 1) % 10 != 0 && gen != 0) {  // Don't duplicate if already printed
                    cout << "Generation " << currentGen << ": ";
                    cout << "Best = " << population->getBestIndividual()->getFitness();
                    cout << " | Avg = " << population->getAverageFitness();
                    cout << endl;
                }
                FILE* saveStream = fopen(saveFile, "wb");
                if (saveStream) {
                    population->save(saveStream);
                    fclose(saveStream);
                    cout << "  --> Checkpoint saved at generation " << currentGen << endl;
                } else {
                    cout << "  WARNING: Could not save checkpoint!" << endl;
                }
            }

            // Check if goal reached
            if (population->getBestIndividual()->getFitness() >= reversiTask.getGoal()) {
                cout << endl << "GOAL REACHED at generation " << currentGen << "!" << endl;
                break;
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

        // Print fitness after last generation of this run
        cout << endl << "=== FITNESS AFTER THIS RUN ===" << endl;
        cout << "Generation: " << population->getGeneration() << endl;
        cout << "Best fitness: " << population->getBestIndividual()->getFitness() << endl;
        cout << "Avg fitness:  " << population->getAverageFitness() << endl;
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
