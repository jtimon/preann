#ifndef TEST_H_
#define TEST_H_

#include "common/parametersMap.h"

#include "common/loop/loop.h"
#include "common/loop/rangeLoop.h"
#include "common/loop/enumLoop.h"
#include "common/loop/joinLoop.h"
#include "common/loop/enumValueLoop.h"

#include "neural/buffer.h"

class Test
{
protected:
    static void createGnuPlotScript(Loop* loop, ParametersMap* parametersMap);
public:
    static unsigned char areEqual(float expected, float actual, BufferType bufferType);
    static unsigned assertEqualsInterfaces(Interface* expected, Interface* actual);
    static unsigned assertEquals(Buffer* expected, Buffer* actual);
    static void checkEmptyMemory(ParametersMap* parametersMap);

    static void test(Loop* loop, void(*func)(ParametersMap*), ParametersMap* parametersMap,
                     std::string functionLabel);
    static void plot(Loop* loop, void(*func)(ParametersMap*), ParametersMap* parametersMap,
                     std::string functionLabel, std::string plotVarKey, float min, float max, float inc);
    static void plotTask(Loop* loop, ParametersMap* parametersMap, unsigned maxGenerations);

};

#endif /* TEST_H_ */
