
#ifndef PLOT_H_
#define PLOT_H_

#include "chronometer.h"
#include "test.h"
#include "factory.h"
#include "population.h"
#include "task.h"

class Plot : public Test {
public:
	Plot();
	virtual ~Plot();

	float plot(string path, float (*f)(Test*, unsigned), unsigned repetitions, string testedMethod);
	float plotTask(string path, Population* population);

};

#endif /* PLOT_H_ */
