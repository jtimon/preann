
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
								 for (itEnumType[_enu] = enumTypes[_enu].begin(); itEnumType[_enu] != enumTypes[_enu].end(); ++itEnumType[_enu])
//								 FOR_EACH (itEnumType[_enu], enumTypes[_enu])

struct IteratorConfig{
	int* variable;
	int min;
	int max;
	int increment;
};

#define FOR_ITER_CONF(_iter) for(*_iter.variable = _iter.min; *_iter.variable <= _iter.max; *_iter.variable += _iter.increment)

class Test {
protected:
	std::map<string, void*> variables;
	std::vector<IteratorConfig> iterators;

	std::vector<unsigned> enumTypes[ENUM_TYPE_DIM];
	std::vector<unsigned>::iterator itEnumType[ENUM_TYPE_DIM];

//	std::map<string, IteratorConfig> iterators2;
//	std::map<EnumType, vector<unsigned> > enumTypes2;
//	std::map<EnumType, vector<unsigned>::iterator> itEnumType2;

public:
	Test();
	virtual ~Test();

	void putVariable(std::string key, void* variable);
	void* getVariable(std::string key);
	void addIterator(int* variable, unsigned min, unsigned max, unsigned increment);
    void withAll(EnumType enumType);
    void with(EnumType enumType, unsigned count, ...);
    void exclude(EnumType enumType, unsigned count, ...);

    unsigned getEnum(EnumType enumType);
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

    void simpleTest( void (*f)(Test*), string testedMethod);
    void test( unsigned (*f)(Test*), string testedMethod);

    template <class classTempl>
    void loopFunction( void (*action)(classTempl (*)(Test*), Test*), classTempl (*f)(Test*), string testedMethod )
    {
    	try {
    		forEnumsIters(action, f, 0);
    	} catch (string error) {
    		cout << "Error: " << error << endl;
    		cout << " While looping "<< testedMethod << " State: " << getCurrentState() << endl;
    	}
    }

    template <class classTempl>
    void forEnumsIters (void (*action)(classTempl (*)(Test*), Test*), classTempl (*f)(Test*), unsigned enu)
    {
    	if (enu == ENUM_TYPE_DIM){
    		forIters(action, f, 0);
    	} else {
    		FOR_EACH(itEnumType[enu], enumTypes[enu]) {
    			forEnumsIters(action, f, enu + 1);
    		}
    	}
    }

    template <class classTempl>
    void forEnums (void (*action)(classTempl (*)(Test*), Test*), classTempl (*f)(Test*), unsigned enu)
    {
    	if (enu == ENUM_TYPE_DIM){
//    		printCurrentState();
    		(*action)(f, this);
    	} else {
    		FOR_EACH(itEnumType[enu], enumTypes[enu]) {
    			forEnums(action, f, enu + 1);
    		}
    	}
    }

    template <class classTempl>
    void forIters (void (*action)(classTempl (*)(Test*), Test*), classTempl (*f)(Test*), unsigned iter)
    {
    	if (iter == iterators.size()){
//    		printCurrentState();
    		(*action)(f, this);
    	} else {
    		FOR_ITER_CONF(iterators[iter]){
    			forIters(action, f, iter + 1);
    		}
    	}
    }
};

void simpleAction(void (*g)(Test*), Test* test);

#endif /* TEST_H_ */
