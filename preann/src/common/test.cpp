/*
 * test.cpp
 *
 *  Created on: Apr 22, 2011
 *      Author: timon
 */

#include "test.h"

Test::Test()
{
	for(int i=0; i < ENUM_TYPE_DIM; ++i){
		enumTypes[i].push_back(0);
		itEnumType[i] = enumTypes[i].begin();
	}
	int baseIterator;
	addIterator(&baseIterator, 1, 1, 1);
}

Test::~Test()
{
	iterators.clear();
	variables.clear();

	for (unsigned i = 0; i < ENUM_TYPE_DIM; ++i) {
		enumTypes[i].clear();
	}
}

unsigned Test::getEnum(EnumType enumType)
{
	return *itEnumType[enumType];
}

BufferType Test::getBufferType()
{
	return (BufferType)*itEnumType[ET_BUFFER];
}
ImplementationType Test::getImplementationType()
{
	return (ImplementationType)*itEnumType[ET_IMPLEMENTATION];
}

FunctionType Test::getFunctionType()
{
	return (FunctionType)*itEnumType[ET_FUNCTION];
}


void testAction(unsigned (*g)(Test*), Test* test)
{
	unsigned differencesCounter = g(test);
	if (differencesCounter > 0){
		test->printCurrentState();
		cout << differencesCounter << " differences detected." << endl;
	}
}
void Test::test( unsigned (*f)(Test*), string testedMethod )
{
	loopFunction( testAction, f, testedMethod );
	cout << testedMethod << endl;
}

void simpleAction(void (*g)(Test*), Test* test)
{
	g(test);
}
void Test::simpleTest( void (*f)(Test*), string testedMethod )
{
	loopFunction( simpleAction, f, testedMethod );
	cout << testedMethod << endl;
}

void Test::addIterator(int* variable, unsigned min, unsigned max, unsigned increment)
{
	IteratorConfig iterator;
	iterator.variable = variable;
	*iterator.variable = min;
	iterator.min = min;
	iterator.max = max;
	iterator.increment = increment;
	iterators.push_back(iterator);
}

void Test::putVariable(std::string key, void* variable)
{
	variables.erase(key);
	variables.insert( pair<string, void*>(key, variable) );
}

void* Test::getVariable(std::string key)
{
	return variables[key];
}

void Test::withAll(EnumType enumType)
{
	enumTypes[enumType].clear();
	unsigned dim = Enumerations::enumTypeDim(enumType);
	for(unsigned i=0; i < dim; i++){
		enumTypes[enumType].push_back( i);
	}
	itEnumType[enumType] = enumTypes[enumType].begin();
}

void Test::with(EnumType enumType, unsigned count, ...)
{
	enumTypes[enumType].clear();
	va_list ap;
	va_start (ap, count);

	for (unsigned i = 0; i < count; i++){
		unsigned arg = va_arg (ap, unsigned);
		enumTypes[enumType].push_back(arg);
	}
	va_end (ap);
}

void Test::exclude(EnumType enumType, unsigned count, ...)
{
	withAll(enumType);
	va_list ap;
	va_start (ap, count);

	for (unsigned i = 0; i < count; i++){
		unsigned arg = va_arg (ap, unsigned);
		vector<unsigned>::iterator j;
		FOR_EACH(j, enumTypes[enumType]) {
			if (*j == arg){
				enumTypes[enumType].erase(j);
				break;
			}
		}
	}
	va_end (ap);
}

std::string Test::getCurrentState()
{
	string state;
	for(int i=0; i < ENUM_TYPE_DIM; ++i) {
		if (enumTypes[i].size() > 1)  {
			unsigned value = *(itEnumType[i]);
			state += "_" + Enumerations::toString((EnumType)i, value);
		}
	}
	for(int i=0; i < iterators.size(); ++i){
		if (iterators[i].min != iterators[i].max){
			state += "_" + to_string(*iterators[i].variable);
		}
	}
	if (state.length() > 1) {
		state.erase(0, 1);
	}
	return state;
}

void Test::printCurrentState()
{
	string state = getCurrentState();
	if (state.length() > 0) {
		printf("%s \n", state.data());
	} else {
		printf("There's no bucles defined for the test.\n", state.data());
	}
}

void Test::printParameters()
{
    std::vector<unsigned>::iterator it;
    //TODO imprimir iteradores
//    for (int i=0; i < PARAMS_DIM; ++i){
//		printf("-param %s: min = %d max = %d increment = %d \n", Enumerations::paramToString(params[i].param).data(), params[i].min, params[i].max, params[i].increment);
//    }

    printf("-Implementations: ");
    FOR_EACH(it, enumTypes[ET_IMPLEMENTATION]) {

		switch ((ImplementationType)*it){
			case C: 			printf(" C        "); 	break;
			case SSE2: 			printf(" SSE2     ");	break;
			case CUDA: 			printf(" CUDA     ");	break;
			case CUDA_REDUC:	printf(" CUDA_REDUC    ");	break;
			case CUDA_INV:		printf(" CUDA_INV ");	break;
		}
    }
    printf("\n");
    printf("-Buffer Types: ");
    std::vector<unsigned>::iterator it2;
    FOR_EACH(it, enumTypes[ET_BUFFER]) {

		switch ((BufferType)*it){
			case FLOAT: printf(" FLOAT "); 	break;
			case BIT: 	printf(" BIT   ");	break;
			case SIGN: 	printf(" SIGN  ");	break;
			case BYTE:	printf(" BYTE  ");	break;
		}
    }
    printf("\n");
}

unsigned char Test::areEqual(float expected, float actual, BufferType bufferType)
{
	if (bufferType == FLOAT){
		return (expected - 1 < actual
			 && expected + 1 > actual);
	} else {
		return expected == actual;
	}
}

unsigned Test::assertEqualsInterfaces(Interface* expected, Interface* actual)
{
    if(expected->getBufferType() != actual->getBufferType()){
        throw "The interfaces are not even of the same type!";
    }
    if(expected->getSize() != actual->getSize()){
        throw "The interfaces are not even of the same size!";
    }
	unsigned differencesCounter = 0;

    for(unsigned i = 0;i < expected->getSize();i++){
        if(!areEqual(expected->getElement(i), actual->getElement(i), expected->getBufferType())){
            printf("The interfaces are not equal at the position %d (expected = %f actual %f).\n", i, expected->getElement(i), actual->getElement(i));
            ++differencesCounter;
        }
    }
	return differencesCounter;
}

unsigned Test::assertEquals(Buffer* expected, Buffer* actual)
{
    if(expected->getBufferType() != actual->getBufferType()){
        throw "The buffers are not even of the same type!";
    }
    if(expected->getSize() != actual->getSize()){
        throw "The buffers are not even of the same size!";
    }

	unsigned differencesCounter = 0;
	Interface* expectedInt = expected->toInterface();
	Interface* actualInt = actual->toInterface();

    for(unsigned i = 0;i < expectedInt->getSize();i++){
        if(!areEqual(expectedInt->getElement(i), actualInt->getElement(i), expectedInt->getBufferType())){
            printf("The buffers are not equal at the position %d (expected = %f actual %f).\n", i, expectedInt->getElement(i), actualInt->getElement(i));
            ++differencesCounter;
        }
    }
    delete(expectedInt);
	delete(actualInt);
	return differencesCounter;
}

FILE* Test::openFile(string path)
{
	FILE* dataFile;
	if (!(dataFile = fopen(path.data(), "w")))
	{
		string error = "Error opening " + path;
		throw error;
	}
	return dataFile;
}
