#include <iostream>
#include <cstring>
#include "tasks/goTask.h"
#include "genetic/population.h"
#include "common/chronometer.h"
#include "common/util.h"

using namespace std;

// Simple command-line argument parser
class Args {
public:
    string loadFile;
    string savePrefix;
    unsigned generations;
    unsigned checkpointInterval;

    Args() : loadFile(""), savePrefix("output/data/go_gen"), generations(300), checkpointInterval(50) {}

    void parse(int argc, char* argv[]) {
        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "--load") == 0 && i + 1 < argc) {
                loadFile = argv[++i];
            } else if (strcmp(argv[i], "--save") == 0 && i + 1 < argc) {
                savePrefix = argv[++i];
            } else if (strcmp(argv[i], "--generations") == 0 && i + 1 < argc) {
                generations = atoi(argv[++i]);
            } else if (strcmp(argv[i], "--checkpoint") == 0 && i + 1 < argc) {
                checkpointInterval = atoi(argv[++i]);
            } else if (strcmp(argv[i], "--help") == 0) {
                printHelp();
                exit(0);
            } else {
                cout << "Unknown argument: " << argv[i] << endl;
                printHelp();
                exit(1);
            }
        }
    }

    void printHelp() {
        cout << "Usage: playGo [options]" << endl;
        cout << "Options:" << endl;
        cout << "  --load <file>          Load population from file" << endl;
        cout << "  --save <prefix>        Save prefix (default: output/data/go_gen)" << endl;
        cout << "  --generations <num>    Number of generations (default: 300)" << endl;
        cout << "  --checkpoint <num>     Checkpoint interval (default: 50)" << endl;
        cout << "  --help                 Show this help" << endl;
        cout << endl;
        cout << "Examples:" << endl;
        cout << "  ./bin/playGo.exe" << endl;
        cout << "  ./bin/playGo.exe --generations 100" << endl;
        cout << "  ./bin/playGo.exe --load output/data/go_gen_050.pop --generations 200" << endl;
    }
};

int main(int argc, char *argv[])
{
    cout << "=== PREANN Go with CUDA Neural Networks ===" << endl << endl;

    try {
        // Parse command-line arguments
        Args args;
        args.parse(argc, argv);

        // Create a 9x9 Go board with bit-packed representation
        // and 2 test games per fitness evaluation
        cout << "Creating Go task (9x9 board, 2 test games)..." << endl;
        GoTask goTask(9, BT_BIT, 2);

        // Create example neural network individual with CUDA implementation
        ParametersMap params;
        params.putNumber(Enumerations::enumTypeToString(ET_IMPLEMENTATION), IT_CUDA_REDUC);
        params.putNumber(Enumerations::enumTypeToString(ET_BUFFER), BT_BIT);
        params.putNumber(Enumerations::enumTypeToString(ET_FUNCTION), FT_BIPOLAR_SIGMOID);

        // Set up population parameters
        params.putNumber(Population::MUTATION_RANGE, 0.5);
        params.putNumber(Population::SIZE, 100);
        params.putNumber(Population::NUM_SELECTION, 50);
        params.putNumber(Population::NUM_CROSSOVER, 40);  // Must be even
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

        // Load or create population
        if (!args.loadFile.empty()) {
            cout << "Loading population from " << args.loadFile << "..." << endl;
            FILE* loadStream = fopen(args.loadFile.c_str(), "rb");
            if (!loadStream) {
                string error = "Error: Could not open file for reading: " + args.loadFile;
                throw error;
            }

            population = new Population(&goTask, 100);  // Create with task reference
            population->load(loadStream);
            population->setParams(&params);  // Set parameters after loading
            fclose(loadStream);

            startGeneration = population->getGeneration();
            cout << "Loaded generation " << startGeneration << " with " << population->getSize() << " individuals" << endl;
            cout << "Best fitness: " << population->getBestIndividual()->getFitness() << endl;
            cout << "Continuing from generation " << (startGeneration + 1) << "..." << endl << endl;
        } else {
            cout << "Creating example neural network (CUDA implementation)..." << endl;
            Individual* example = goTask.getExample(&params);
            cout << "Neural network created with " << example->getNumLayers() << " layers" << endl;

            // Create population
            cout << "Creating population of 100 neural networks..." << endl;
            unsigned populationSize = 100;
            population = new Population(&goTask, example, populationSize, args.generations);
            population->setParams(&params);

            // Test generation 1 (random individuals)
            Individual* gen1Best = population->getBestIndividual();
            cout << "=== GENERATION 1 (Random/Untrained) ===" << endl;
            cout << "Testing best random individual with 10 games..." << endl;

            GoTask testTask(9, BT_BIT, 10);
            testTask.test(gen1Best);
            float gen1Fitness = gen1Best->getFitness();
            cout << "Generation 1 best fitness (10 games): " << gen1Fitness << endl;
            cout << "Average points per game: " << (gen1Fitness - 81*10) / 10.0 << " out of 81" << endl << endl;
        }

        cout << "Starting evolution (" << args.generations << " generations):" << endl;
        cout << "Goal fitness: " << goTask.getGoal() << endl;
        cout << "Checkpoint interval: every " << args.checkpointInterval << " generations" << endl << endl;

        Chronometer chrono;
        chrono.start();

        // Evolve the population
        cout << "=== EVOLVING ===" << endl;
        for (unsigned generation = startGeneration; generation < args.generations; generation++) {
            population->nextGeneration();

            Individual* best = population->getBestIndividual();
            unsigned currentGen = population->getGeneration();

            // Print progress every 50 generations
            if (currentGen % 50 == 0 || generation == startGeneration) {
                cout << "Generation " << currentGen << ": ";
                cout << "Best fitness = " << best->getFitness();
                cout << " (avg: " << population->getAverageFitness() << ")";
                cout << endl;
            }

            // Auto-checkpoint
            if (currentGen % args.checkpointInterval == 0) {
                char filename[256];
                snprintf(filename, sizeof(filename), "%s_%03d.pop", args.savePrefix.c_str(), currentGen);

                FILE* saveStream = fopen(filename, "wb");
                if (saveStream) {
                    population->save(saveStream);
                    fclose(saveStream);
                    cout << "Checkpoint saved: " << filename << endl;
                } else {
                    cerr << "Warning: Could not save checkpoint to " << filename << endl;
                }
            }

            if (best->getFitness() >= goTask.getGoal()) {
                cout << endl << "GOAL REACHED at generation " << currentGen << "!" << endl;
                break;
            }
        }

        chrono.stop();
        cout << endl << "Evolution completed in " << chrono.getSeconds() << " seconds";
        cout << " (" << (chrono.getSeconds()/60.0) << " minutes)" << endl;

        // Test final best individual
        Individual* finalBest = population->getBestIndividual();
        unsigned finalGen = population->getGeneration();
        cout << endl << "=== GENERATION " << finalGen << " (Evolved) ===" << endl;
        cout << "Testing best evolved individual with 10 games..." << endl;
        GoTask testTask(9, BT_BIT, 10);
        testTask.test(finalBest);
        float finalFitness = finalBest->getFitness();
        cout << "Generation " << finalGen << " best fitness (10 games): " << finalFitness << endl;
        cout << "Average points per game: " << (finalFitness - 81*10) / 10.0 << " out of 81" << endl << endl;

        // Save final population
        char finalFilename[256];
        snprintf(finalFilename, sizeof(finalFilename), "%s_%03d_final.pop", args.savePrefix.c_str(), finalGen);
        FILE* finalStream = fopen(finalFilename, "wb");
        if (finalStream) {
            population->save(finalStream);
            fclose(finalStream);
            cout << "Final population saved: " << finalFilename << endl;
        }

        delete population;

    } catch (string& error) {
        cerr << "Error: " << error << endl;
        return 1;
    }

    cout << endl << "Exit success." << endl;
    return 0;
}
