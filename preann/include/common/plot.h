
#ifndef PLOT_H_
#define PLOT_H_

#include "chronometer.h"
#include "test.h"
#include "factory.h"

class Plot : public Test {
public:
	Plot();
	virtual ~Plot();

	static string methodToString(Method method);
	static string classToString(ClassID classID);
	static string toString(ClassID classID, Method method);
	float plot(string path, ClassID classID, Method method, unsigned repetitions);
	static float doMethod(ClassID classID, Method method, unsigned repetitions);
	static float doMethod(Vector* vector, Method method, unsigned repetitions);
	static float doMethod(Connection* connection, Method method, unsigned repetitions);

};

#endif /* PLOT_H_ */
