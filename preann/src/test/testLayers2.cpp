#include <iostream>
#include <fstream>

using namespace std;

#include "chronometer.h"
#include "loop.h"
#include "dummy.h"
#include "test.h"

void testCalculateOutput(ParametersMap* parametersMap)
{
    unsigned differencesCounter = 0;
    string path = parametersMap->getString("layerPath");
    Buffer* buffer = Dummy::buffer(parametersMap);
    Buffer* bufferC = Factory::newBuffer(buffer, IT_C);
    Layer* layer = Dummy::layer(parametersMap, buffer);

    FILE* stream = fopen(path.data(), "w+b");
	layer->save(stream);
	layer->saveWeighs(stream);

    stream = fopen(path.data(), "r+b");
	Layer* layerC = new Layer(stream, IT_C);

	unsigned numInputs = (unsigned)parametersMap->getNumber("numInputs");
	for (unsigned i = 0; i < numInputs; ++i) {
		layerC->addInput(bufferC);
	}
	layerC->loadWeighs(stream);
	fclose(stream);

    //test calculation
    layer->calculateOutput();
    layerC->calculateOutput();

    differencesCounter += Test::assertEquals(layer->getOutput(),
            layerC->getOutput());

    delete (layerC);
    delete (bufferC);
    delete (layer);
    delete (buffer);

    parametersMap->putNumber("differencesCounter", differencesCounter);
}

int main(int argc, char *argv[])
{
    Chronometer total;
    total.start();
    try {
        Loop* loop;
        ParametersMap parametersMap;
        parametersMap.putNumber("initialWeighsRange", 20);
        parametersMap.putNumber("numInputs", 2);
        parametersMap.putString("layerPath", "/home/timon/layer.lay");
        parametersMap.putNumber(Enumerations::enumTypeToString(ET_FUNCTION),
                FT_IDENTITY);

        loop = new RangeLoop("size", 1, 51, 49, NULL);

        EnumLoop* bufferTypeLoop = new EnumLoop(Enumerations::enumTypeToString(
                ET_BUFFER), ET_BUFFER, loop, 3, BT_BIT, BT_SIGN, BT_FLOAT);
        loop = bufferTypeLoop;

        loop = new EnumLoop(Enumerations::enumTypeToString(ET_IMPLEMENTATION),
                ET_IMPLEMENTATION, loop);
        loop->print();

        //TODO arreglar
        loop->test(testCalculateOutput, &parametersMap, "Layer::calculateOutput");

        printf("Exit success.\n");
        MemoryManagement::printTotalAllocated();
        MemoryManagement::printTotalPointers();
    } catch (std::string error) {
        cout << "Error: " << error << endl;
        //	} catch (...) {
        //		printf("An error was thrown.\n", 1);
    }

    //MemoryManagement::mem_printListOfPointers();
    total.stop();
    printf("Total time spent: %f \n", total.getSeconds());
    return EXIT_SUCCESS;
}