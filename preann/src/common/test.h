#ifndef TEST_H_
#define TEST_H_

#include "parametersMap.h"
#include "chronometer.h"

#include "loop/loop.h"
#include "loop/rangeLoop.h"
#include "loop/enumLoop.h"
#include "loop/joinLoop.h"
#include "loop/enumValueLoop.h"

#include "neural/buffer.h"

#define PLOT_LOOP "__LOOP__PLOT_LOOP"
#define PLOT_X_AXIS "__LOOP__PLOT_X_AXIS"
#define PLOT_Y_AXIS "__LOOP__PLOT_Y_AXIS"
#define PLOT_LINE_COLOR_LOOP "__LOOP__PLOT_LINE_COLOR_LOOP"
#define PLOT_POINT_TYPE_LOOP "__LOOP__PLOT_POINT_TYPE_LOOP"
#define PLOT_MIN "__LOOP__PLOT_MIN"
#define PLOT_MAX "__LOOP__PLOT_MAX"
#define PLOT_INC "__LOOP__PLOT_INC"

#define START_CHRONO                                                                    \
    Chronometer chrono;                                                                 \
    unsigned repetitions = parametersMap->getNumber(Test::REPETITIONS);                 \
    chrono.start();                                                                     \
    for (unsigned i = 0; i < repetitions; ++i) {

#define STOP_CHRONO                                                                     \
    }                                                                                   \
    chrono.stop();                                                                      \
    parametersMap->putNumber(Test::TIME_COUNT, chrono.getSeconds());

class Test
{
public:
    static const string DIFF_COUNT;
    static const string MEM_LOSSES;
    static const string REPETITIONS;
    static const string TIME_COUNT;
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
