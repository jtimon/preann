#include <iostream>
#include <fstream>

using namespace std;

#include "chronometer.h"
#include "plot.h"
#include "factory.h"

Test test;

int main(int argc, char *argv[])
{
	string path = "/home/timon/workspace/preann/output/plotData/";


	Chronometer total;
	total.start();
	unsigned errorCount = 0;
	test.fromToBySize(100, 1000, 100);
	test.fromToByOutputSize(100, 300, 100);
	test.printParameters();

	try {

		test.disableVectorType(BYTE);
		cout << Plot::toString(VECTOR, COPYFROMINTERFACE) << " total: "
			 << Plot::plot(path, VECTOR, COPYFROMINTERFACE, test, 1) << endl;

		cout << Plot::toString(VECTOR, COPYTOINTERFACE) << " total: "
		 	 << Plot::plot(path, VECTOR, COPYTOINTERFACE, test, 1) << endl;

		cout << Plot::toString(VECTOR, ACTIVATION) << " total: "
			 << Plot::plot(path, VECTOR, ACTIVATION, test, 10) << endl;

		cout << Plot::toString(CONNECTION, CALCULATEANDADDTO) << " total: "
			 << Plot::plot(path, CONNECTION, CALCULATEANDADDTO, test, 1) << endl;

		cout << Plot::toString(CONNECTION, MUTATE) << " total: "
			 << Plot::plot(path, CONNECTION, MUTATE, test, 100) << endl;

		cout << Plot::toString(CONNECTION, CROSSOVER) << " total: "
			 << Plot::plot(path, CONNECTION, CROSSOVER, test, 1) << endl;
		/*
		for (test.vectorTypeToMin(); test.vectorTypeIncrement(); ) {
			FunctionType functionType = (FunctionType)(test.getVectorType());
			for (test.implementationTypeToMin(); test.implementationTypeIncrement(); ) {

				test.printCurrentState();

				//				printf("\n CopyFrom: ");
				//				for (unsigned size = SIZE_MIN; size <= SIZE_MAX; size
				//						+= SIZE_INC) {
				//					Vector* vector = Factory::newVector(size,
				//							test.getVectorType(), implementationType);
				//					vector->random(INITIAL_WEIGHS_RANGE);
				//					printf(" %f ", chronoCopyFrom(vector));
				//					delete (vector);
				//				}
				//				printf("\n CopyTo: ");
				//				for (unsigned size = SIZE_MIN; size <= SIZE_MAX; size
				//						+= SIZE_INC) {
				//					Vector* vector = Factory::newVector(size,
				//							test.getVectorType(), test.getImplementationType());
				//					vector->random(INITIAL_WEIGHS_RANGE);
				//					printf(" %f ", chronoCopyTo(vector));
				//					delete (vector);
				//				}
				if (test.getVectorType() != BYTE) {
					test.openFile("activation");
					for (test.sizeToMin(); test.sizeIncrement(); ) {
						Vector* vector = Factory::newVector(test.getSize(),
								test.getVectorType(), test.getImplementationType());
						vector->random(INITIAL_WEIGHS_RANGE);

						test.plotToFile(chronoActivation(vector, functionType));

						delete (vector);
					}
					test.closeFile();

					for (int outputSize = OUTPUT_SIZE_MIN; outputSize
							<= OUTPUT_SIZE_MAX; outputSize += OUTPUT_SIZE_INC) {

						std::stringstream path;
						path << "addToResults_outSize_" << outputSize;
						test.openFile(path.str());
						for (test.sizeToMin(); test.sizeIncrement(); ) {
							Vector* vector = Factory::newVector(test.getSize(),
									test.getVectorType(), test.getImplementationType());
							vector->random(INITIAL_WEIGHS_RANGE);
							Connection* connection = Factory::newConnection(
									vector, outputSize, test.getImplementationType());
							connection->random(INITIAL_WEIGHS_RANGE);

							test.plotToFile(chronoAddToResults(connection));

							delete (connection);
							delete (vector);
						}
						test.closeFile();
						//						printf("\n mutate: ");
						//						for (unsigned size = SIZE_MIN; size <= SIZE_MAX; size
						//								+= SIZE_INC) {
						//							Vector* vector = Factory::newVector(size,
						//									test.getVectorType(), test.getImplementationType());
						//							vector->random(INITIAL_WEIGHS_RANGE);
						//							Connection* connection = Factory::newConnection(
						//									vector, outputSize, test.getImplementationType());
						//							connection->random(INITIAL_WEIGHS_RANGE);
						//							printf(" %f ", chronoMutate(connection, NUM_MUTATIONS));
						//							delete (connection);
						//							delete (vector);
						//						}
						std::stringstream path2;
						path << "crossover_outSize_" << outputSize;
						test.openFile(path2.str());
						for (test.sizeToMin(); test.sizeIncrement(); ) {
							Vector* vector = Factory::newVector(test.getSize(),
									test.getVectorType(), test.getImplementationType());
							vector->random(INITIAL_WEIGHS_RANGE);
							Connection* connection = Factory::newConnection(
									vector, outputSize, test.getImplementationType());
							connection->random(INITIAL_WEIGHS_RANGE);

							test.plotToFile(chronoCrossover(connection));

							delete (connection);
							delete (vector);
						}
						test.closeFile();
					}
				}
			}
		}*/
		printf("Exit success.\n");
		mem_printTotalAllocated();
		mem_printTotalPointers();
	} catch (std::string error) {
		cout << "Error: " << error << endl;
		//	} catch (...) {
		//		printf("An error was thrown.\n", 1);
	}

	//mem_printListOfPointers();
	total.stop();
	printf("Total time spent: %f \n", total.getSeconds());
	return EXIT_SUCCESS;
}
