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
    static const string TEST_FUNCTION;
    static const string X_TO_PLOT;
    static const string TO_AVERAGE;
    static const string X_ARRAY;
    static const string Y_ARRAY;
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
    static const string LINE_COLOR_LEVEL;
    static const string POINT_TYPE_LEVEL;
    static const string PLOT_PATH;
    static const string PLOT_FILE;
    static const string FIRST_STATE;
    static const string SUB_PATH;
    static const string POPULATION;
    static const string EXAMPLE_INDIVIDUAL;
    static const string TASK;
    static const string MAX_GENERATIONS;
protected:
    void createGnuPlotScript(ParametersMap* parametersMap, string functionLabel);
    void createGnuPlotScriptTask(RangeLoop*& xToPlot, std::string& path, std::string& label);
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

    void test(ParamMapFuncPtr func, std::string functionLabel);
    void plot(ParamMapFuncPtr func, std::string functionLabel, std::string plotVarKey, float min, float max,
              float inc);
    void plotTask(unsigned maxGenerations);
    void plotTask2(std::string functionLabel, RangeLoop* xToPlot);
    void plotTask2(std::string functionLabel, RangeLoop* xToPlot, Loop* toAverage);
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
