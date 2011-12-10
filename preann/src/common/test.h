
#include "factory.h"

#ifndef TEST_H_
#define TEST_H_

#define START_TEST unsigned differencesCounter = 0;
#define END_TEST return differencesCounter;

#define GET_SIZE unsigned size = test->getIterValue("size");
#define GET_INITIAL_WEIGHS_RANGE float initialWeighsRange = *((float*)test->getVariable("initialWeighsRange"));
#define GET_OUTPUT_SIZE unsigned outputSize = test->getIterValue("outputSize");

#define START_BUFFER GET_SIZE GET_INITIAL_WEIGHS_RANGE Buffer* buffer = Factory::newBuffer(size, (BufferType)test->getEnum(ET_BUFFER), (ImplementationType)test->getEnum(ET_IMPLEMENTATION)); buffer->random(initialWeighsRange);
#define END_BUFFER delete(buffer);

#define START_BUFFER_TEST START_TEST START_BUFFER
#define END_BUFFER_TEST END_BUFFER END_TEST

#define START_CONNECTION START_BUFFER GET_OUTPUT_SIZE Connection* connection = Factory::newConnection(buffer, outputSize, buffer->getImplementationType()); connection->random(initialWeighsRange);
#define END_CONNECTION delete(connection); END_BUFFER

#define START_CONNECTION_TEST START_TEST START_CONNECTION
#define END_CONNECTION_TEST  END_CONNECTION END_TEST

struct IteratorConfig{
	float value;
	float min;
	float max;
	float increment;
};

struct EnumIterConfig{
	vector<unsigned> valueVector;
	unsigned index;
};


#define FOR_ITER_CONF(_iter) for(_iter.value = _iter.min; _iter.value < _iter.max; _iter.value += _iter.increment)

class Test {
protected:
	std::map<string, void*> variables;
	std::vector<IteratorConfig> iterators;
	std::map<std::string, unsigned > iterMap;
	std::vector<EnumIterConfig*> enumerations;
	std::map<EnumType, unsigned > enumMap;

//	map<EnumType, unsigned> enumTypePos;
//	vector< vector<unsigned> > enumTypes;
//	vector< vector<unsigned>::iterator > enumTypeIters;

	void initEnumType(EnumType enumType);
	EnumType enumTypeAtPos(unsigned pos);
	EnumIterConfig* getEnumConfig(EnumType enumType);
	EnumIterConfig* getEnumConfigAtPos(unsigned pos);
	IteratorConfig getIterator(std::string key);
public:
	Test();
	virtual ~Test();

	void putVariable(std::string key, void* variable);
	void* getVariable(std::string key);

//	template <class T>
//	void putVariable(std::string key, T variable)
//	{
//		variables.erase(key);
//		variables.insert( pair<string, void*>(key, (void*)variable) );
//	};
//
////	template <class T>
//	void* getVariable(std::string key)
//	{
//		if(!variables.count(key)){
//			std::string error = " Test::getVariable : variable \"" + key + "\" not found.";
//			throw error;
//		}
//		return variables[key];
//	};

	void addIterator(std::string key, float min, float max, float increment);
	virtual float getIterValue(std::string key);
    void withAll(EnumType enumType);
    void with(EnumType enumType, unsigned count, ...);
    void exclude(EnumType enumType, unsigned count, ...);

    unsigned getEnum(EnumType enumType);

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
    void loopFunction( void (*action)(classTempl (*)(Test*), Test*), classTempl (*f)(Test*), string& testedMethod )
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
    	if (enu == enumerations.size()){
//    		printCurrentState();
    		forIters(action, f, 0);
    	} else {
    		EnumIterConfig* enumConfig = getEnumConfigAtPos(enu);
    		for (enumConfig->index = 0; enumConfig->index < enumConfig->valueVector.size(); ++enumConfig->index) {
    			forEnumsIters(action, f, enu + 1);
			}
    	}
    }

    template <class classTempl>
    void forEnums (void (*action)(classTempl (*)(Test*), Test*), classTempl (*f)(Test*), unsigned enu)
    {
    	if (enu == enumerations.size()){
//    		printCurrentState();
    		(*action)(f, this);
    	} else {
    		EnumIterConfig* enumConfig = getEnumConfigAtPos(enu);
    		for (enumConfig->index = 0; enumConfig->index < enumConfig->valueVector.size(); ++enumConfig->index) {
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
