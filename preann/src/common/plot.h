
#ifndef PLOT_H_
#define PLOT_H_

#include "chronometer.h"
#include "test.h"
#include "factory.h"
#include "population.h"
#include "task.h"

#define START_PLOT Chronometer chrono; unsigned repetitions = test->getValue("repetitions");
#define END_PLOT return chrono.getSeconds();

#define START_BUFFER_PLOT START_PLOT START_BUFFER
#define END_BUFFER_PLOT END_BUFFER END_PLOT

#define START_CONNECTION_PLOT START_PLOT START_CONNECTION
#define END_CONNECTION_PLOT END_CONNECTION END_PLOT

#define FOR_PLOT_ITERATOR for(*plotIterator.variable = plotIterator.min; *plotIterator.variable <= plotIterator.max; *plotIterator.variable += plotIterator.increment)

class Plot : public Test {
protected:
	std::string plotIteratorKey;
	IteratorConfig plotIterator;

	EnumType colorEnum;
	EnumType pointEnum;

//	FILE* preparePlotAndDataFile(string path, string testedMethod);
//	void plotDataFile(string path, string testedMethod);
	string getPlotPath(string path, string testedMethod);
	void createPlotScript(string path, string testedMethod);
	void plotFile(string path, string testedMethod);
public:
	Plot();
	virtual ~Plot();

	int getPointType();
	int getLineColor();

	virtual float getValue(std::string key);
	void setColorEnum(EnumType colorEnum);
	void setPointEnum(EnumType pointEnum);
	void putPlotIterator(std::string key, float min, float max, float increment);
	IteratorConfig getPlotIterator();

//	void plot(string path, float (*f)(Test*, unsigned), unsigned repetitions, string testedMethod);
//	void plot2(string path, float (*f)(Test*, unsigned), unsigned repetitions, string testedMethod);
	void plot(float (*f)(Test*), string path, string testedMethod);
	void plotTask(string path, Task* task, Individual* example, unsigned populationSize, float weighsRange);
};

#endif /* PLOT_H_ */
