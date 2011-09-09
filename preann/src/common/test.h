
#include "factory.h"

#ifndef TEST_H_
#define TEST_H_

#define NUM_MUTATIONS 10

class Test {
protected:
	unsigned size;
	unsigned minSize;
	unsigned maxSize;
	unsigned incSize;
	unsigned outputSize;
	unsigned minOutputSize;
	unsigned maxOutputSize;
	unsigned incOutputSize;

	vector<unsigned> bufferTypes;
	vector<unsigned>::iterator itBufferType;
	vector<ImplementationType> implementationTypes;
	vector<ImplementationType>::iterator itImplemType;

	float initialWeighsRange;

	FILE *file;

public:
	Test();
	virtual ~Test();

    unsigned getSize();
    void fromToBySize(unsigned minSize, unsigned maxSize, unsigned incSize);

    unsigned getOutputSize();
    void fromToByOutputSize(unsigned minOutputSize, unsigned maxOutputSize, unsigned incOutputSize);

    float getInitialWeighsRange();
    void setInitialWeighsRange(float initialWeighsRange);

	BufferType getBufferType();
    void withAllBufferTypes();
    void withBufferTypes(vector<unsigned> bufferTypes);
    void excludeBufferTypes(vector<unsigned> bufferTypes);

	ImplementationType getImplementationType();
    void withAllImplementationTypes();
    void withImplementationTypes(vector<ImplementationType> implementationTypes);
    void excludeImplementationTypes(vector<ImplementationType> implementationTypes);

	FunctionType getFunctionType();

    void printParameters();
    void printCurrentState();
    static unsigned char areEqual(float expected, float actual, BufferType bufferType);
    static unsigned assertEqualsInterfaces(Interface* expected, Interface* actual);
    static unsigned assertEquals(Buffer* expected, Buffer* actual);


    string bufferTypeToString();
    string implementationTypeToString();
    FILE* openFile(string path);
    void openFile(string path, ClassID classID, Method method);
    void closeFile();
    void plotToFile(float data);

    string getFileName(ClassID& classID, Method& method);

    void testFunction( void (*f)(Test*), string testedMethod );
    void test ( unsigned (*f)(Test*), string testedMethod );
    Buffer* buildBuffer();
    Connection* buildConnection(Buffer* buffer);

    static string toString(ClassID classID, Method method);
	static string methodToString(Method method);
	static string classToString(ClassID classID);

};

#endif /* TEST_H_ */
