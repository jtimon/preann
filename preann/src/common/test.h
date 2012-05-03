#ifndef TEST_H_
#define TEST_H_

#include "parametersMap.h"
#include "chronometer.h"

#include "loop/loop.h"
#include "loop/rangeLoop.h"
#include "loop/enumLoop.h"
#include "loop/joinEnumLoop.h"

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
    static const string REPETITIONS;
    static const string TIME_COUNT;
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
    static void checkDifferences(ParametersMap* parametersMap);
    static void checkEmptyMemory(ParametersMap* parametersMap);

    void testMemoryLosses(ParamMapFuncPtr function, string label);
    void test(ParamMapFuncPtr func, std::string label);
    void plot(ParamMapFuncPtr func, std::string label, RangeLoop* xToPlot, string yLabel, unsigned repetitions);
    void plotTask(Task* task, std::string label, RangeLoop* xToPlot);
    void plotTask(Task* task, std::string label, RangeLoop* xToPlot, Loop* toAverage);
    void addLoop(Loop* loop);
    Loop* getLoop();

};

class TestAction : public LoopFunction
{
    ParametersMap* tParameters;
    LoopFunction* tInnerFunction;
    std::string tLabel;
public:
    TestAction(ParametersMap* parameters, LoopFunction* innerFunction, std::string label)
    {
        tParameters = parameters;
        tInnerFunction = innerFunction;
        tLabel = label;
    }
    ;
    virtual ~TestAction()
    {
    }
    ;
    virtual void __executeImpl()
    {
        try {
            tInnerFunction->execute(tCallerLoop);
            Test::checkEmptyMemory(tParameters);
        } catch (string e) {
            cout << " while testing " + tLabel + " at state " + tCallerLoop->getState(true) << " : " << endl;
            cout << e << endl;
        }
    };
};

#endif /* TEST_H_ */
