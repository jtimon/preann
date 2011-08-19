
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

	float plot(string path, ClassID classID, Method method, unsigned repetitions);
	float doMethod(ClassID classID, Method method, unsigned repetitions);
	float doMethodBuffer(Buffer* buffer, Method method, unsigned repetitions);
	float doMethodConnection(Connection* connection, Method method, unsigned repetitions);
	float plotTask(string path, Population* population);

};

#endif /* PLOT_H_ */
