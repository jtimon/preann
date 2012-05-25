#ifndef TEST_H_
#define TEST_H_

#include "common/chronometer.h"
#include "neural/buffer.h"

#include "parametersMap.h"
#include "loop.h"
#include "rangeLoop.h"
#include "enumLoop.h"
#include "joinEnumLoop.h"


typedef unsigned (*TestFunctionPtr)(ParametersMap*);

#define START_CHRONO                                                                    \
    Chronometer chrono;                                                                 \
    chrono.start();                                                                     \
    for (unsigned i = 0; i < repetitions; ++i) {

#define STOP_CHRONO                                                                     \
    }                                                                                   \
    chrono.stop();


class Test
{
public:
    ParametersMap parameters;
    Test();
    ~Test();
    static void check(bool condition, string message);
    static unsigned char areEqual(float expected, float actual, BufferType bufferType);
    static unsigned assertEqualsInterfaces(Interface* expected, Interface* actual);
    static unsigned assertEquals(Buffer* expected, Buffer* actual);

    void testMemoryLosses(GenericLoopFuncPtr function, string label, Loop* loop);
    void test(TestFunctionPtr func, std::string label, Loop* loop);

};

#endif /* TEST_H_ */
