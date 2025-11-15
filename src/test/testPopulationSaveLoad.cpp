#include <iostream>
#include "tasks/binaryTask.h"
#include "genetic/population.h"
#include "common/dummy.h"

using namespace std;

int main(int argc, char *argv[])
{
    cout << "=== Test Population Save/Load with XOR ===" << endl << endl;

    try {
        // Create XOR task: 2 bit inputs (0/1), 2 bit outputs
        // This way we can test all 4 input combinations
        cout << "Creating XOR task (2 bits)..." << endl;
        BinaryTask xorTask(BO_XOR, BT_BIT, 2, 0);
        cout << "Goal fitness: " << xorTask.getGoal() << endl << endl;

        // Create parameters for neural network
        // Binary inputs (BT_BIT) use byte weights (BT_BYTE) for quantization
        ParametersMap params;
        params.putNumber(Enumerations::enumTypeToString(ET_IMPLEMENTATION), IT_C);
        params.putNumber(Enumerations::enumTypeToString(ET_BUFFER), BT_BYTE);  // byte weights for binary inputs
        params.putNumber(Enumerations::enumTypeToString(ET_FUNCTION), FT_IDENTITY);
        params.putNumber(Dummy::SIZE, 2);

        // Create example individual (2 layers: hidden + output)
        Individual* example = xorTask.getExample(&params);
        cout << "Created neural network with " << example->getNumLayers() << " layers" << endl;

        // Set up population parameters
        params.putNumber(Population::MUTATION_RANGE, 2.0);
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

        // Create and evolve population
        cout << "Creating population of 100 individuals..." << endl;
        Population population1(&xorTask, example, 100, 5.0);
        population1.setParams(&params);

        cout << "Initial best fitness: " << population1.getBestIndividual()->getFitness() << endl;
        cout << "Evolving for 50 generations..." << endl;

        for (unsigned gen = 0; gen < 50; gen++) {
            population1.nextGeneration();
            if (gen % 10 == 9) {
                cout << "Generation " << population1.getGeneration()
                     << ": best = " << population1.getBestIndividual()->getFitness() << endl;
            }
            if (population1.getBestIndividual()->getFitness() >= xorTask.getGoal()) {
                cout << "Goal reached at generation " << population1.getGeneration() << "!" << endl;
                break;
            }
        }

        float finalFitness = population1.getBestIndividual()->getFitness();
        unsigned finalGen = population1.getGeneration();
        cout << endl << "Final generation: " << finalGen << endl;
        cout << "Final best fitness: " << finalFitness << endl << endl;

        // Save population
        const char* filename = "output/data/test_xor_population.pop";
        cout << "Saving population to " << filename << "..." << endl;
        FILE* saveStream = fopen(filename, "wb");
        if (!saveStream) {
            string error = "Could not open file for writing: " + string(filename);
            throw error;
        }
        population1.save(saveStream);
        fclose(saveStream);
        cout << "Saved successfully!" << endl << endl;

        // Load population
        cout << "Loading population from " << filename << "..." << endl;
        FILE* loadStream = fopen(filename, "rb");
        if (!loadStream) {
            string error = "Could not open file for reading: " + string(filename);
            throw error;
        }

        Population population2(&xorTask, 100);
        population2.load(loadStream);
        fclose(loadStream);
        cout << "Loaded successfully!" << endl << endl;

        // Verify loaded data matches
        cout << "=== VERIFICATION ===" << endl;
        cout << "Original generation: " << population1.getGeneration()
             << " | Loaded generation: " << population2.getGeneration() << endl;
        cout << "Original best fitness: " << population1.getBestIndividual()->getFitness()
             << " | Loaded best fitness: " << population2.getBestIndividual()->getFitness() << endl;
        cout << "Original size: " << population1.getSize()
             << " | Loaded size: " << population2.getSize() << endl;

        // Check if they match
        bool success = true;
        if (population1.getGeneration() != population2.getGeneration()) {
            cout << "ERROR: Generation numbers don't match!" << endl;
            success = false;
        }
        if (population1.getSize() != population2.getSize()) {
            cout << "ERROR: Population sizes don't match!" << endl;
            success = false;
        }

        float fitness_diff = abs(population1.getBestIndividual()->getFitness() -
                                 population2.getBestIndividual()->getFitness());
        if (fitness_diff > 0.001) {
            cout << "ERROR: Best fitness values don't match! (diff: " << fitness_diff << ")" << endl;
            success = false;
        }

        // Test the loaded population by evolving it further
        if (success) {
            cout << endl << "=== TESTING LOADED POPULATION ===" << endl;
            cout << "Evolving loaded population for 10 more generations..." << endl;

            for (unsigned gen = 0; gen < 10; gen++) {
                population2.nextGeneration();
            }

            cout << "After 10 more generations: " << population2.getGeneration() << endl;
            cout << "Best fitness: " << population2.getBestIndividual()->getFitness() << endl;
        }

        if (success) {
            cout << endl << "SUCCESS: All values match! Population save/load working correctly." << endl;
        } else {
            cout << endl << "FAILURE: Some values don't match." << endl;
            return 1;
        }

    } catch (string& error) {
        cerr << "Error: " << error << endl;
        return 1;
    }

    cout << endl << "Exit success." << endl;
    return 0;
}
