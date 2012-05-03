#ifndef TEST_H_
#define TEST_H_

#include "parametersMap.h"
#include "chronometer.h"

#include "loop/loop.h"
#include "loop/rangeLoop.h"
#include "loop/enumLoop.h"
#include "loop/joinEnumLoop.h"

#include "neural/buffer.h"

typedef unsigned (*TestFunctionPtr)(ParametersMap*);
typedef float (*ChronoFunctionPtr)(ParametersMap*, unsigned);

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
    static const string LINE_COLOR_LEVEL;
    static const string POINT_TYPE_LEVEL;
    static const string PLOT_PATH;
protected:
    void createGnuPlotScript(string& path, string& title, string& xLabel, string& yLabel);
    Loop* tLoop;
public:
    ParametersMap parameters;
    Test();
    ~Test();
    static void check(bool condition, string message);
    static unsigned char areEqual(float expected, float actual, BufferType bufferType);
    static unsigned assertEqualsInterfaces(Interface* expected, Interface* actual);
    static unsigned assertEquals(Buffer* expected, Buffer* actual);

    void testMemoryLosses(ParamMapFuncPtr function, string label);
    void test(TestFunctionPtr func, std::string label);
    void plotChrono(ChronoFunctionPtr func, std::string label, RangeLoop* xToPlot, string yLabel, unsigned repetitions);
    void plotTask(Task* task, std::string label, RangeLoop* xToPlot);
    void plotTask(Task* task, std::string label, RangeLoop* xToPlot, Loop* toAverage);
    void addLoop(Loop* loop);
    Loop* getLoop();

};

#endif /* TEST_H_ */
