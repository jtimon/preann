#include <iostream>
#include <fstream>

using namespace std;

#include "common/chronometer.h"
#include "loop/test.h"
#include "common/dummy.h"

#define START                                                                           \
    unsigned differencesCounter = 0;                                                    \
    Buffer* buffer = Dummy::buffer(parametersMap);

#define END                                                                             \
    delete (buffer);                                                                    \
    return differencesCounter;

unsigned testCopyFromInterface(ParametersMap* parametersMap)
{
    START

    Interface interface(buffer->getSize(), buffer->getBufferType());
    interface.random(parametersMap->getNumber(Dummy::WEIGHS_RANGE));

    Buffer* cBuffer = Factory::newBuffer(buffer, IT_C);

    buffer->copyFromInterface(&interface);
    cBuffer->copyFromInterface(&interface);

    differencesCounter += Test::assertEquals(cBuffer, buffer);

    delete (cBuffer);

    END
}

unsigned testCopyToInterface(ParametersMap* parametersMap)
{
    START

    Interface interface = Interface(buffer->getSize(), buffer->getBufferType());

    Buffer* cBuffer = Factory::newBuffer(buffer, IT_C);
    Interface cInterface = Interface(buffer->getSize(), buffer->getBufferType());

    buffer->copyToInterface(&interface);
    cBuffer->copyToInterface(&cInterface);

    differencesCounter = Test::assertEqualsInterfaces(&cInterface, &interface);

    delete (cBuffer);

    END
}

unsigned testSaveLoad(ParametersMap* parametersMap)
{
    START
    ImplementationType implementationType = (ImplementationType) (parametersMap->getNumber(
            Enumerations::enumTypeToString(ET_IMPLEMENTATION)));

    string path = parametersMap->getString("path");
    FILE* stream = fopen(path.data(), "w+b");
    buffer->save(stream);
    fclose(stream);

    stream = fopen(path.data(), "r+b");
    Buffer* loadedBuffer = Factory::newBuffer(stream, implementationType);

    differencesCounter += Test::assertEquals(buffer, loadedBuffer);

    delete (loadedBuffer);

    END
}

unsigned testClone(ParametersMap* parametersMap)
{
    START

    Buffer* copy = buffer->clone();
    differencesCounter += Test::assertEquals(buffer, copy);

    delete (copy);

    END
}

int main(int argc, char *argv[])
{
    Chronometer total;
    total.start();
    try {
        Util::check(argv[1] == NULL, "You must specify a directory.");
        Test test;
        test.parameters.putString("path", argv[1] + to_string("data/testBuffer.buf"));
        test.parameters.putNumber(Dummy::WEIGHS_RANGE, 20);
        test.parameters.putNumber(Enumerations::enumTypeToString(ET_FUNCTION), FT_IDENTITY);

        RangeLoop loop(Dummy::SIZE, 100, 101, 100);
//        loop.addInnerLoop(new EnumLoop(ET_IMPLEMENTATION, 2, IT_C, IT_SSE2));
        loop.addInnerLoop(new EnumLoop(ET_IMPLEMENTATION));

        EnumLoop* bufferTypeLoop = new EnumLoop(ET_BUFFER);
        loop.addInnerLoop(bufferTypeLoop);

        loop.print();

        test.test(testClone, "Buffer::clone", &loop);
        test.test(testCopyFromInterface, "Buffer::copyFromInterface", &loop);
        test.test(testCopyToInterface, "Buffer::copyToInterface", &loop);
        test.test(testSaveLoad, "Buffer::saveLoad", &loop);

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
