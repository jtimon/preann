
#ifndef PLOT_H_
#define PLOT_H_

#include "chronometer.h"
#include "test.h"
#include "factory.h"

class Plot {
public:
	Plot();
	virtual ~Plot();

	static void plot(string path, ClassID classID, Method method, Test test);
	static string methodToString(Method method);
	static string classToString(ClassID classID);
	static float doMethod(ClassID classID, Method method, Test test);
	static float doMethod(Vector* vector, Method method);
	static float doMethod(Connection* connection, Method method);

};

#endif /* PLOT_H_ */
