#include <iostream>
#include "tasks/goTask.h"
#include "genetic/population.h"
#include "common/chronometer.h"

using namespace std;

int main(int argc, char *argv[])
{
    cout << "=== PREANN Go with CUDA Neural Networks ===" << endl << endl;

    try {
        // Create a 9x9 Go board with bit-packed representation
        // and 2 test games per fitness evaluation
        cout << "Creating Go task (9x9 board, 2 test games)..." << endl;
        GoTask goTask(9, BT_BIT, 2);

        // Create example neural network individual with CUDA implementation
        ParametersMap params;
        params.putNumber(Enumerations::enumTypeToString(ET_IMPLEMENTATION), IT_CUDA_REDUC);
        params.putNumber(Enumerations::enumTypeToString(ET_BUFFER), BT_BIT);
        params.putNumber(Enumerations::enumTypeToString(ET_FUNCTION), FT_BIPOLAR_SIGMOID);

        cout << "Creating example neural network (CUDA implementation)..." << endl;
        Individual* example = goTask.getExample(&params);
        cout << "Neural network created with " << example->getNumLayers() << " layers" << endl;

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

        // Create population
        cout << "Creating population of 100 neural networks..." << endl;
        unsigned populationSize = 100;
        unsigned maxGenerations = 300;

        Population population(&goTask, example, populationSize, maxGenerations);
        population.setParams(&params);

        cout << endl << "Starting evolution (300 generations):" << endl;
        cout << "Goal fitness: " << goTask.getGoal() << endl << endl;

        Chronometer chrono;
        chrono.start();

        // Test generation 1 (random individuals)
        Individual* gen1Best = population.getBestIndividual();
        cout << "=== GENERATION 1 (Random/Untrained) ===" << endl;
        cout << "Testing best random individual with 10 games..." << endl;

        GoTask testTask(9, BT_BIT, 10);
        testTask.test(gen1Best);
        float gen1Fitness = gen1Best->getFitness();
        cout << "Generation 1 best fitness (10 games): " << gen1Fitness << endl;
        cout << "Average points per game: " << (gen1Fitness - 81*10) / 10.0 << " out of 81" << endl << endl;

        // Evolve the population
        cout << "=== EVOLVING ===" << endl;
        for (unsigned generation = 0; generation < maxGenerations; generation++) {
            population.nextGeneration();

            Individual* best = population.getBestIndividual();

            // Print progress every 50 generations
            if ((generation + 1) % 50 == 0 || generation == 0) {
                cout << "Generation " << (generation + 1) << ": ";
                cout << "Best fitness = " << best->getFitness();
                cout << " (avg: " << population.getAverageFitness() << ")";
                cout << endl;
            }

            if (best->getFitness() >= goTask.getGoal()) {
                cout << endl << "GOAL REACHED at generation " << (generation + 1) << "!" << endl;
                break;
            }
        }

        chrono.stop();
        cout << endl << "Evolution completed in " << chrono.getSeconds() << " seconds";
        cout << " (" << (chrono.getSeconds()/60.0) << " minutes)" << endl;

        // Test final best individual
        Individual* finalBest = population.getBestIndividual();
        cout << endl << "=== GENERATION 300 (Evolved) ===" << endl;
        cout << "Testing best evolved individual with 10 games..." << endl;
        testTask.test(finalBest);
        float finalFitness = finalBest->getFitness();
        cout << "Generation 300 best fitness (10 games): " << finalFitness << endl;
        cout << "Average points per game: " << (finalFitness - 81*10) / 10.0 << " out of 81" << endl << endl;

        // Show improvement
        cout << "=== RESULTS ===" << endl;
        cout << "Improvement: " << (finalFitness - gen1Fitness) << " fitness points" << endl;
        cout << "Points per game improvement: " << ((finalFitness - gen1Fitness) / 10.0) << endl;
        cout << "Percentage improvement: " << (((finalFitness - gen1Fitness) / gen1Fitness) * 100.0) << "%" << endl;

    } catch (string& error) {
        cerr << "Error: " << error << endl;
        return 1;
    }

    cout << endl << "Exit success." << endl;
    return 0;
}
