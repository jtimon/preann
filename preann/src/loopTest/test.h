#ifndef TEST_H_
#define TEST_H_

#include "common/chronometer.h"
#include "neural/buffer.h"

#include "loop/loop.h"
#include "loop/rangeLoop.h"
#include "loop/expLoop.h"
#include "loop/enumLoop.h"
#include "loop/joinEnumLoop.h"


typedef unsigned (*TestFunctionPtr)(ParametersMap*);

class Test
{
public:
    ParametersMap parameters;
    Test();
    ~Test();
    static unsigned char areEqual(float expected, float actual, BufferType bufferType);
    static unsigned assertEqualsInterfaces(Interface* expected, Interface* actual);
    static unsigned assertEquals(Buffer* expected, Buffer* actual);

    void testMemoryLosses(GenericLoopFuncPtr function, string label, Loop* loop);
    void test(TestFunctionPtr func, std::string label, Loop* loop);

};

#endif /* TEST_H_ */
