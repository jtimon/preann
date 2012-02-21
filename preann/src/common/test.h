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
public:
    static unsigned char areEqual(float expected, float actual,
            BufferType bufferType);
    static unsigned assertEqualsInterfaces(Interface* expected,
            Interface* actual);
    static unsigned assertEquals(Buffer* expected, Buffer* actual);
    static void checkEmptyMemory(ParametersMap* parametersMap);

};

#endif /* TEST_H_ */
