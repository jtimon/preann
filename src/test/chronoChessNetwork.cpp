/*
 * chronoChessNetwork.cpp
 *
 * Benchmark C++ vs CUDA performance for chess network topology
 * Tests networks with structure: 768→N1→N2→N3→1 with recurrent connection
 * Finds the crossover point where CUDA becomes faster than C++
 */

#include <iostream>
#include <fstream>

#include "common/dummy.h"
#include "genetic/individual.h"
#include "loopTest/chronoPlotter.h"
#include "cuda/cuda.h"

#define INPUT_SIZE "INPUT_SIZE"
#define N1_SIZE "N1_SIZE"
#define N2_SIZE "N2_SIZE"
#define N3_SIZE "N3_SIZE"
#define OUTPUT_SIZE "OUTPUT_SIZE"

float chronoChessNetwork(ParametersMap* parametersMap, unsigned repetitions)
{
    // Get network sizes from parameters
    unsigned inputSize = parametersMap->getNumber(INPUT_SIZE);
    unsigned n1Size = parametersMap->getNumber(N1_SIZE);

    // Check if we're using proportional scaling mode
    unsigned n2Size, n3Size;
    if (parametersMap->getNumber(N2_SIZE) == 0) {
        // Proportional scaling: N2 = N1/2, N3 = N1/4
        n2Size = n1Size / 2;
        n3Size = n1Size / 4;
    } else {
        // Fixed sizes
        n2Size = parametersMap->getNumber(N2_SIZE);
        n3Size = parametersMap->getNumber(N3_SIZE);
    }

    unsigned outputSize = parametersMap->getNumber(OUTPUT_SIZE);

    ImplementationType implementationType = (ImplementationType)parametersMap->getNumber(
            Enumerations::enumTypeToString(ET_IMPLEMENTATION));
    FunctionType functionType = (FunctionType)parametersMap->getNumber(
            Enumerations::enumTypeToString(ET_FUNCTION));

    // Create input buffer (768 neurons for 8x8x12 chess board encoding)
    Interface* input = new Interface(inputSize, BT_FLOAT);
    input->random(5.0);

    float weighsRange = parametersMap->getNumber(Dummy::WEIGHS_RANGE);

    // Declare variables outside timing block
    Individual* individual = NULL;

    // Benchmark: Build chess network and initialize weights
    // This is the operation that scales with network size
    START_CHRONO
        individual = new Individual(implementationType);
        individual->addInputLayer(input);

        // Hidden layers - using BIT buffer for first two layers (byte weights)
        individual->addLayer(n1Size, BT_BIT, functionType);      // Layer 0
        individual->addLayer(n2Size, BT_BIT, functionType);      // Layer 1

        // Third hidden layer - SIGN buffer for bipolar values
        individual->addLayer(n3Size, BT_SIGN, functionType);     // Layer 2 (recurrent source)

        // Output layer - FLOAT with IDENTITY function
        individual->addLayer(outputSize, BT_FLOAT, FT_IDENTITY); // Layer 3

        // Feedforward connections
        individual->addInputConnection(0, 0);       // Input → Layer 0
        individual->addLayersConnection(0, 1);      // Layer 0 → Layer 1
        individual->addLayersConnection(1, 2);      // Layer 1 → Layer 2
        individual->addLayersConnection(2, 3);      // Layer 2 → Output

        // Recurrent connection (layer 2 feeds back to layer 0)
        individual->addLayersConnection(2, 0);      // Recurrent: Layer 2 → Layer 0

        // Initialize network with random weights
        individual->randomWeighs(weighsRange);

        delete individual;
    STOP_CHRONO

    delete input;

    return chrono.getSeconds();
}

int main(int argc, char *argv[])
{
    Chronometer total;
    total.start();
    try {
        Util::check(argv[1] == NULL, "You must specify an output directory.");

        // X-axis: N1 size (first hidden layer) from 128 to 1024
        ChronoPlotter plotter(argv[1], new RangeLoop(N1_SIZE, 128, 1025, 128), "Tiempo (ms)");
        unsigned repetitions = 1000;  // Fewer reps than activation (networks are slower)

        // Fixed parameters
        plotter.parameters.putNumber(INPUT_SIZE, 768);           // Chess board: 8x8x12
        plotter.parameters.putNumber(OUTPUT_SIZE, 1);            // Position evaluation
        plotter.parameters.putNumber(Dummy::WEIGHS_RANGE, 5.0);
        plotter.parameters.putNumber(Enumerations::enumTypeToString(ET_FUNCTION), FT_BIPOLAR_SIGMOID);

        // Benchmark 1: Proportional scaling (N2=N1/2, N3=N1/4)
        // Compare C++ vs CUDA implementations

        cout << "=== Benchmark 1: Proportional Network Scaling ===" << endl;
        cout << "Testing topology: 768→N1→(N1/2)→(N1/4)→1 with recurrent" << endl;
        cout << "N1 range: 128 to 1024 (step 128)" << endl;

        // Set N2_SIZE and N3_SIZE to 0 to enable proportional scaling mode
        // (N2 = N1/2, N3 = N1/4 computed inside benchmark function)
        plotter.parameters.putNumber(N2_SIZE, 0);
        plotter.parameters.putNumber(N3_SIZE, 0);

        // Lines: C++ vs CUDA
        EnumLoop linesLoop(ET_IMPLEMENTATION, 2, IT_C, IT_CUDA_OUT);

        linesLoop.print();
        plotter.plotChrono(chronoChessNetwork, "chess_proportional", &linesLoop, repetitions);

        // Benchmark 2: Fixed N2=128, N3=32, vary N1
        // Shows effect of scaling just first layer

        cout << "\n=== Benchmark 2: Fixed N2/N3, Variable N1 ===" << endl;
        cout << "Testing topology: 768→N1→128→32→1 with recurrent" << endl;

        plotter.parameters.putNumber(N2_SIZE, 128);
        plotter.parameters.putNumber(N3_SIZE, 32);

        plotter.plotChrono(chronoChessNetwork, "chess_fixed_n2n3", &linesLoop, repetitions);

        // Benchmark 3: Vary N2 with fixed N1=512, N3=32
        // Shows effect of scaling middle layer

        cout << "\n=== Benchmark 3: Fixed N1/N3, Variable N2 ===" << endl;
        cout << "Testing topology: 768→512→N2→32→1 with recurrent" << endl;

        plotter.parameters.putNumber(N1_SIZE, 512);
        plotter.parameters.putNumber(N3_SIZE, 32);

        ChronoPlotter plotter2(argv[1], new RangeLoop(N2_SIZE, 64, 513, 64), "Tiempo (ms)");
        plotter2.parameters.putNumber(INPUT_SIZE, 768);
        plotter2.parameters.putNumber(OUTPUT_SIZE, 1);
        plotter2.parameters.putNumber(N1_SIZE, 512);
        plotter2.parameters.putNumber(N3_SIZE, 32);
        plotter2.parameters.putNumber(Dummy::WEIGHS_RANGE, 5.0);
        plotter2.parameters.putNumber(Enumerations::enumTypeToString(ET_FUNCTION), FT_BIPOLAR_SIGMOID);

        plotter2.plotChrono(chronoChessNetwork, "chess_fixed_n1n3", &linesLoop, repetitions);

        printf("\nExit success.\n");
    } catch (std::string error) {
        cout << "Error: " << error << endl;
    }

    MemoryManagement::printTotalAllocated();
    MemoryManagement::printTotalPointers();
    total.stop();
    printf("Total time spent: %f seconds\n", total.getSeconds());
    return EXIT_SUCCESS;
}
