/*
 * test.cpp
 *
 *  Created on: Apr 22, 2011
 *      Author: timon
 */

#include "test.h"

Test::Test()
{
}

Test::~Test()
{
	iterators.clear();
	variables.clear();

	for (unsigned i = 0; i < enumerations.size(); ++i) {
		enumerations[i]->valueVector.clear();
		delete(enumerations[i]);
	}
	enumerations.clear();
	enumMap.clear();
}

EnumType Test::enumTypeAtPos(unsigned pos)
{
	std::map<EnumType, unsigned >::iterator it;
	FOR_EACH(it, enumMap){
		if(it->second == pos){
			return it->first;
		}
	}
    if (pos >= enumerations.size()){
		string error = "Test::enumTypeAtPos : cannot access the pos " + to_string(pos) +
				" there's only " + to_string(enumerations.size()) + " enumerations.";
		throw error;
    }
	string error = "Test::enumTypeAtPos : No enum defined at pos " + pos;
	throw error;
}

EnumIterConfig* Test::getEnumConfig(EnumType enumType)
{
	if (!enumMap.count(enumType)){
		string error = "Test::getEnumConfig : The test has no defined vector for enum " + Enumerations::enumTypeToString(enumType);
		throw error;
	}
	return getEnumConfigAtPos( enumMap[enumType] );
}

EnumIterConfig* Test::getEnumConfigAtPos(unsigned pos)
{
    if (pos >= enumerations.size()){
		string error = "Test::getEnumConfigAtPos : cannot access the pos " + to_string(pos) +
				" there's only " + to_string(enumerations.size()) + " enumerations.";
		throw error;
    }
    return enumerations[pos];
}

unsigned Test::getEnum(EnumType enumType)
{
	EnumIterConfig* enumConfig = getEnumConfig(enumType);
	unsigned index = enumConfig->index;
    unsigned vectorSize = enumConfig->valueVector.size();
    if(index >= vectorSize) {
		string error = "Test::getEnum : The index points to " + to_string(index) +
				" but the vector has only " + to_string(vectorSize) + " values.";
		throw error;
	}
	return enumConfig->valueVector[ index ];
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

void setToIteratorConfig(IteratorConfig &iteratorConfig, float min, float max, float increment)
{
	iteratorConfig.value = min;
	iteratorConfig.min = min;
	iteratorConfig.max = max;
	iteratorConfig.increment = increment;
}

void Test::putConstant(std::string key, float constant)
{
	if (iterMap.count(key)){
		unsigned pos = iterMap[key];
		setToIteratorConfig(iterators[pos], constant, constant + 1, 2);
	} else {
		IteratorConfig iterator;
		setToIteratorConfig(iterator, constant, constant + 1, 2);
		iterMap[key] = iterators.size();
		iterators.push_back(iterator);
	}
}

void Test::putIterator(std::string key, float min, float max, float increment)
{
	if (iterMap.count(key)){
		unsigned pos = iterMap[key];
		setToIteratorConfig(iterators[pos], min, max, increment);
	} else {
		IteratorConfig iterator;
		setToIteratorConfig(iterator, min, max, increment);
		iterMap[key] = iterators.size();
		iterators.push_back(iterator);
	}
}

float Test::getValue(std::string key)
{
	return getIterator(key).value;
}

IteratorConfig Test::getIterator(std::string key)
{
	if (!iterMap.count(key)){
		string error = "Test::getIterValue : The test has no iterator " + key;
		throw error;
	}
	unsigned pos = iterMap[key];
	return iterators[pos];
}

void Test::putVariable(std::string key, void* variable)
{
	variables.erase(key);
	variables.insert( pair<string, void*>(key, variable) );
}

void* Test::getVariable(std::string key)
{
	if(!variables.count(key)){
		std::string error = " Test::getVariable : variable \"" + key + "\" not found.";
		throw error;
	}
	return variables[key];
}

void Test::initEnumType(EnumType enumType)
{
	if (!enumMap.count(enumType)){
		EnumIterConfig* config = new EnumIterConfig();
		config->valueVector.clear();
		config->index = 0;
		enumMap[enumType] = enumerations.size();
		enumerations.push_back(config);
	}
}

void Test::withAll(EnumType enumType)
{
	initEnumType(enumType);
	EnumIterConfig* enumConfig = getEnumConfig(enumType);
	enumConfig->valueVector.clear();

	unsigned dim = Enumerations::enumTypeDim(enumType);
	for(unsigned i=0; i < dim; i++){
		enumConfig->valueVector.push_back( i);
	}
}

void Test::with(EnumType enumType, unsigned count, ...)
{
	initEnumType(enumType);
	EnumIterConfig* enumConfig = getEnumConfig(enumType);
	enumConfig->valueVector.clear();

	va_list ap;
	va_start (ap, count);

	unsigned dim = Enumerations::enumTypeDim(enumType);
	for (unsigned i = 0; i < count; i++){
		unsigned arg = va_arg (ap, unsigned);
		if (arg < dim){
			enumConfig->valueVector.push_back(arg);
		}
	}
	va_end (ap);
}

void Test::exclude(EnumType enumType, unsigned count, ...)
{
	withAll(enumType);
	EnumIterConfig* enumConfig = getEnumConfig(enumType);

	va_list ap;
	va_start (ap, count);

	for (unsigned i = 0; i < count; i++){
		unsigned arg = va_arg (ap, unsigned);

		vector<unsigned>::iterator it;
		FOR_EACH(it, enumConfig->valueVector) {
			if (*it == arg){
				enumConfig->valueVector.erase(it);
				break;
			}
		}
	}
	va_end (ap);
}

std::string Test::getCurrentState()
{
	string state;

    map<EnumType, unsigned>::iterator it;
	FOR_EACH(it, enumMap){
		if(getEnumConfigAtPos(it->second)->valueVector.size() > 1) {
			state += "_" + Enumerations::toString(it->first, getEnum(it->first));
		}
	}

	for(unsigned i=0; i < iterators.size(); ++i){
		if (iterators[i].min != iterators[i].max){
			state += "_" + to_string(iterators[i].value);
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
    //TODO imprimir iteradores
//    for (int i=0; i < PARAMS_DIM; ++i){
//		printf("-param %s: min = %d max = %d increment = %d \n", Enumerations::paramToString(params[i].param).data(), params[i].min, params[i].max, params[i].increment);
//    }
	for (unsigned i = 0; i < enumerations.size(); ++i) {
		EnumType enumType = enumTypeAtPos(i);

		string str = Enumerations::enumTypeToString(enumType) + " : ";

		vector<unsigned> enumVect = enumerations[i]->valueVector;
		for (unsigned i = 0; i < enumVect.size(); ++i) {
			str += Enumerations::toString(enumType, enumVect[i]) + " ";
		}
		printf(" %s \n", str.data());
	}
}

unsigned char Test::areEqual(float expected, float actual, BufferType bufferType)
{
	if (bufferType == BT_FLOAT){
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
//	printf(" opening file \"%s\"\n", path.data());
	return dataFile;
}
