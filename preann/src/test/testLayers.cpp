#include <iostream>
#include <fstream>

using namespace std;

#include "common/chronometer.h"
#include "loopTest/test.h"
#include "common/dummy.h"

const string LAYER_PATH = "layerPath";

unsigned testCalculateOutput(ParametersMap* parametersMap)
{
    unsigned differencesCounter = 0;
    string path = parametersMap->getString(LAYER_PATH);
    ImplementationType implementationType = (ImplementationType) (parametersMap->getNumber(
                Enumerations::enumTypeToString(ET_IMPLEMENTATION)));
    Interface* interfInput = Dummy::interface(parametersMap);

    Layer* input = new InputLayer(interfInput, implementationType);
    Layer* inputC = new InputLayer(interfInput, IT_C);
    Layer* layer = Dummy::layer(parametersMap, input);

    FILE* stream = fopen(path.data(), "w+b");
    layer->save(stream);
    layer->saveWeighs(stream);
    fclose(stream);

    stream = fopen(path.data(), "r+b");
    Layer* layerC = new Layer(stream, IT_C);

    unsigned numInputs = (unsigned) parametersMap->getNumber(Dummy::NUM_INPUTS);
    for (unsigned i = 0; i < numInputs; ++i) {
        layerC->addInput(inputC);
    }
    layerC->loadWeighs(stream);
    fclose(stream);

    //test calculation
    layer->calculateOutput();
    layerC->calculateOutput();

    differencesCounter += Test::assertEquals(layer->getOutput(), layerC->getOutput());

    delete (layerC);
    delete (layer);
    delete (inputC);
    delete (input);
    delete (interfInput);

    return differencesCounter;
}

int main(int argc, char *argv[])
{
    Chronometer total;
    total.start();
    try {
        Util::check(argv[1] == NULL, "You must specify a directory.");
        Test test;
        test.parameters.putNumber(Dummy::WEIGHS_RANGE, 20);
        test.parameters.putNumber(Dummy::NUM_INPUTS, 2);
        test.parameters.putString(LAYER_PATH, argv[1] + to_string("/data/layer.lay"));
        test.parameters.putNumber(Enumerations::enumTypeToString(ET_FUNCTION), FT_IDENTITY);

        RangeLoop loop(Dummy::SIZE, 1, 51, 49);
        loop.addInnerLoop(new EnumLoop(ET_BUFFER, 3, BT_BIT, BT_SIGN, BT_FLOAT));
//        loop.addInnerLoop(new EnumLoop(ET_IMPLEMENTATION, 2, IT_C, IT_SSE2));
        loop.addInnerLoop(new EnumLoop(ET_IMPLEMENTATION));

        loop.print();

        test.test(testCalculateOutput, "Layer::calculateOutput", &loop);

        printf("Exit success.\n");
    } catch (std::string error) {
        cout << "Error: " << error << endl;
        //	} catch (...) {
        //		printf("An error was thrown.\n", 1);
    }

    MemoryManagement::printTotalAllocated();
    MemoryManagement::printTotalPointers();
    //MemoryManagement::mem_printListOfPointers();
    total.stop();
    printf("Total time spent: %f \n", total.getSeconds());
    return EXIT_SUCCESS;
}
