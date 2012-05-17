#include <iostream>
#include <fstream>

#include "loopTest/chronoPlotter.h"
#include "common/dummy.h"

#define START                                                                           \
    Interface* input = Dummy::interface(parametersMap);                                 \
    Individual* individual = Dummy::individual(parametersMap, input);

#define END                                                                             \
    delete (individual);                                                                \
    delete (input);                                                                     \
    return chrono.getSeconds();

float chronoCrossover(ParametersMap* parametersMap, unsigned repetitions)
{
    START

    Individual* other = individual->newCopy(false);
    other->setFitness(1.5);

    CrossoverAlgorithm crossoverAlgorithm = (CrossoverAlgorithm) parametersMap->getNumber(
            Enumerations::enumTypeToString(ET_CROSS_ALG));
    CrossoverLevel crossoverLevel = (CrossoverLevel) parametersMap->getNumber(
            Enumerations::enumTypeToString(ET_CROSS_LEVEL));

    float uniformCrossProb = 0;
    try {
        uniformCrossProb =parametersMap->getNumber(Population::UNIFORM_CROSS_PROB);
    } catch (string& e) {
        cout << "Warning : " + e <<endl;
    }
    unsigned numPoints = 0;
    try {
        numPoints =parametersMap->getNumber(Population::MULTIPOINT_NUM);
    } catch (string& e) {
        cout << "Warning : " + e <<endl;
    }

    START_CHRONO
    switch (crossoverAlgorithm) {
        case CA_UNIFORM:
            individual->uniformCrossover(crossoverLevel, other, uniformCrossProb);
            break;
        case CA_PROPORTIONAL:
            individual->proportionalCrossover(crossoverLevel, other);
            break;
        case CA_MULTIPOINT:
            individual->multipointCrossover(crossoverLevel, other, numPoints);
            break;
    }
    STOP_CHRONO

    delete (other);

    END
}

float colorPointTest(ParametersMap* parametersMap, unsigned repetitions)
{
    return parametersMap->getNumber(Dummy::SIZE) + parametersMap->getNumber("testValue");
}

int main(int argc, char *argv[])
{
    Chronometer total;
    total.start();
    try {
        ChronoPlotter plotter(PREANN_DIR + to_string("output/"));
        plotter.parameters.putNumber(Dummy::WEIGHS_RANGE, 5);
        plotter.parameters.putNumber(Enumerations::enumTypeToString(ET_IMPLEMENTATION), IT_C);
        plotter.parameters.putNumber(Enumerations::enumTypeToString(ET_FUNCTION), FT_IDENTITY);
        plotter.parameters.putNumber(Dummy::NUM_INPUTS, 2);

//        plotter.parameters.putNumber(Population::UNIFORM_CROSS_PROB, 0.1);

        JoinEnumLoop* crossAlgLoop = new JoinEnumLoop(ET_CROSS_ALG);
        plotter.addLoop(crossAlgLoop);

        RangeLoop* uniformCrossProbLoop = new RangeLoop(Population::UNIFORM_CROSS_PROB, 0.05, 0.20, 0.1);
        crossAlgLoop->addEnumLoop(CA_UNIFORM, uniformCrossProbLoop);
//
//        crossAlgLoop->addEnumLoop(CA_PROPORTIONAL, NULL);

        RangeLoop* numPointsLoop = new RangeLoop(Population::MULTIPOINT_NUM, 1, 3, 1);
        crossAlgLoop->addEnumLoop(CA_MULTIPOINT, numPointsLoop);


        plotter.addLoop(new EnumLoop(ET_CROSS_LEVEL));
//        plotter.parameters.putNumber(Enumerations::enumTypeToString(ET_CROSS_LEVEL), CL_LAYER);

        Loop* averageLoop = new EnumLoop(ET_BUFFER, 3, BT_FLOAT, BT_BIT, BT_SIGN);
        averageLoop->addInnerLoop(new RangeLoop(Dummy::NUM_LAYERS, 1, 3, 1));

        plotter.getLoop()->print();

        RangeLoop xToPlot(Dummy::SIZE, 50, 101, 50);
        string yLabel = "Time (seconds)";
        unsigned repetitions = 2;

        plotter.plotChronoAveraged(chronoCrossover, "Individual_crossover", &xToPlot, yLabel, averageLoop, repetitions);

        ChronoPlotter plotter2(PREANN_DIR + to_string("output/"));
        plotter2.addLoop(new RangeLoop("testValue", 0, 21, 1));
        plotter2.plotChrono(colorPointTest, "colorTest", &xToPlot, yLabel, repetitions);

        delete(averageLoop);

        printf("Exit success.\n");
    } catch (std::string& error) {
        cout << "Error: " << error << endl;
    }

    MemoryManagement::printTotalAllocated();
    MemoryManagement::printTotalPointers();
    MemoryManagement::printListOfPointers();

    total.stop();
    printf("Total time spent: %f \n", total.getSeconds());
    return EXIT_SUCCESS;
}
