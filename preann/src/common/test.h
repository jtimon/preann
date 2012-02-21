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
    static const string PLOT_LOOP;
    static const string PLOT_X_AXIS;
    static const string PLOT_Y_AXIS;
    static const string PLOT_MIN;
    static const string PLOT_MAX;
    static const string PLOT_INC;
    static const string LINE_COLOR;
    static const string POINT_TYPE;
    static const string PLOT_PATH;
    static const string PLOT_FILE;
    static const string FIRST_STATE;
    static const string SUB_PATH;
    static const string INITIAL_POPULATION;
    static const string EXAMPLE_INDIVIDUAL;
    static const string TASK;
    static const string MAX_GENERATIONS;
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
