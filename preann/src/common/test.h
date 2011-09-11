
#include "factory.h"

#ifndef TEST_H_
#define TEST_H_

#define START_TEST unsigned differencesCounter = 0;
#define END_TEST return differencesCounter;

#define START_BUFFER Buffer* buffer = Factory::newBuffer(size, test->getBufferType(), test->getImplementationType()); buffer->random(initialWeighsRange);
#define END_BUFFER delete(buffer);

#define START_BUFFER_TEST START_TEST START_BUFFER
#define END_BUFFER_TEST END_BUFFER END_TEST

#define START_CONNECTION START_BUFFER Connection* connection = Factory::newConnection(buffer, outputSize, buffer->getImplementationType()); connection->random(initialWeighsRange);
#define END_CONNECTION delete(connection); END_BUFFER

#define START_CONNECTION_TEST START_TEST START_CONNECTION
#define END_CONNECTION_TEST  END_CONNECTION END_TEST

#define FOR_ALL_ITERATORS for(int _ite=0; _ite < iterators.size(); ++_ite)             \
							  (*iterators[_ite].variable) = iterators[_ite].min;       \
						  for(int _ite=0; _ite < iterators.size(); ++_ite)             \
							  for ((*iterators[_ite].variable) = iterators[_ite].min; (*iterators[_ite].variable) <= iterators[_ite].max; (*iterators[_ite].variable) += iterators[_ite].increment)
#define FOR_ALL_ENUMERATIONS for(int _enu=0; _enu < ENUM_TYPE_DIM; ++_enu)             \
								 itEnumType[_enu] = enumTypes[_enu].begin();           \
							 for(int _enu=0; _enu < ENUM_TYPE_DIM; ++_enu)             \
								 FOR_EACH (itEnumType[_enu], enumTypes[_enu])

class Test {
protected:
	struct IteratorConfig{
		int* variable;
		int min;
		int max;
		int increment;
	};
	std::vector<IteratorConfig> iterators;

	std::vector<unsigned> enumTypes[ENUM_TYPE_DIM];
	std::vector<unsigned>::iterator itEnumType[ENUM_TYPE_DIM];

public:
	Test();
	virtual ~Test();

	void addIterator(int* variable, unsigned min, unsigned max, unsigned increment);
    void withAll(EnumType enumType);
    void with(EnumType enumType, unsigned count, ...);
    void exclude(EnumType enumType, unsigned count, ...);

	BufferType getBufferType();
	ImplementationType getImplementationType();
	FunctionType getFunctionType();

	std::string getCurrentState();
    void printParameters();
    void printCurrentState();
    static unsigned char areEqual(float expected, float actual, BufferType bufferType);
    static unsigned assertEqualsInterfaces(Interface* expected, Interface* actual);
    static unsigned assertEquals(Buffer* expected, Buffer* actual);

    FILE* openFile(string path);

    void testFunction( void (*f)(Test*), string testedMethod );
    void test ( unsigned (*f)(Test*), string testedMethod );

};

#endif /* TEST_H_ */
