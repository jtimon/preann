
#include <iostream>
#include <fstream>

using namespace std;

#include "chronometer.h"
#include "test.h"
#include "factory.h"


int main(int argc, char *argv[]) {

	Chronometer total;
	total.start();
	
	Test test;

	test.fromToBySize(2, 10, 10);
	test.fromToByOutputSize(1, 3, 2);
	test.setInitialWeighsRange(20);
	test.printParameters();

	try {
		test.test(BUFFER, CLONE);
		test.test(BUFFER, COPYFROMINTERFACE);
		test.test(BUFFER, COPYTOINTERFACE);
		test.disableBufferType(BYTE);
		test.test(BUFFER, ACTIVATION);
		
		test.test(CONNECTION, CALCULATEANDADDTO);
		test.test(CONNECTION, MUTATE);
		test.test(CONNECTION, CROSSOVER);
		
		printf("Exit success.\n");
		MemoryManagement::printTotalAllocated();
		MemoryManagement::printTotalPointers();
	} catch (std::string error) {
		cout << "Error: " << error << endl;
//	} catch (...) {
//		printf("An error was thrown.\n", 1);
	}

	//MemoryManagement::mem_printListOfPointers();
	total.stop();
	printf("Total time spent: %f \n", total.getSeconds());
	return EXIT_SUCCESS;
}
