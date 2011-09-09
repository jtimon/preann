/*
 * test.cpp
 *
 *  Created on: Apr 22, 2011
 *      Author: timon
 */

#include "test.h"

Test::Test()
{
	size = minSize = maxSize = incSize = 1;
	withAllBufferTypes();
	withAllImplementationTypes();
	initialWeighsRange = 0;
	file = NULL;
}

Test::~Test()
{
	// TODO Auto-generated destructor stub
}

BufferType Test::getBufferType()
{
	return (BufferType)*itBufferType;
}
ImplementationType Test::getImplementationType()
{
	return *itImplemType;
}

Buffer* Test::buildBuffer()
{
	Buffer* buffer = Factory::newBuffer(getSize(), getBufferType(), getImplementationType());
	buffer->random(initialWeighsRange);
	return buffer;
}

Connection* Test::buildConnection(Buffer* buffer)
{
	Connection* connection = Factory::newConnection(buffer, outputSize, getImplementationType());
	connection->random(initialWeighsRange);
	return connection;
}

FunctionType Test::getFunctionType()
{
	//TODO hacer bucle para esto también
	return IDENTITY;
}

void Test::test ( unsigned (*f)(Test*), string testedMethod )
{
	for (size = minSize; size <= maxSize; size += incSize) {
		FOR_EACH(itBufferType, bufferTypes){
			FOR_EACH(itImplemType, implementationTypes){
				try {
					unsigned differencesCounter = f(this);
					if (differencesCounter > 0){
						printCurrentState();
						cout << differencesCounter << " differences detected." << endl;
					}
				} catch (string error) {
					cout << "Error: " << error << endl;
					cout << " While testing "<< testedMethod << endl;
					printCurrentState();
				}
			}
		}
	}
	cout << testedMethod << endl;
}

void Test::testFunction( void (*f)(Test*), string testedMethod )
{
	for (size = minSize; size <= maxSize; size += incSize) {
		FOR_EACH(itBufferType, bufferTypes){
			FOR_EACH(itImplemType, implementationTypes){
				try {
					(*f)(this);
				} catch (string error) {
					cout << "Error: " << error << endl;
					cout << " While testing "<< testedMethod << endl;
					printCurrentState();
				}
			}
		}
	}
}

unsigned Test::getSize()
{
    return size;
}

void Test::withAllImplementationTypes()
{
	implementationTypes.clear();
	for(unsigned i=0; i < IMPLEMENTATION_TYPE_DIM; i++){
		implementationTypes.push_back( (ImplementationType) i);
	}
}

void Test::withImplementationTypes(vector<ImplementationType> implTypes)
{
	implementationTypes.clear();
	implementationTypes.insert(implementationTypes.end(), implTypes.begin(), implTypes.end());
}

void Test::excludeImplementationTypes(vector<ImplementationType> implTypes)
{
	for (vector<ImplementationType>::iterator itExt = implTypes.begin(); itExt != implTypes.end(); ++itExt) {
	    ImplementationType implementationType = *itExt;
		for (vector<ImplementationType>::iterator it = implementationTypes.begin(); it != implementationTypes.end(); ++it) {
			if (*it == implementationType){
				implementationTypes.erase(it);
				break;
			}
		}
	}
}

void Test::withAllBufferTypes()
{
	bufferTypes.clear();
	for(unsigned i=0; i < BUFFER_TYPE_DIM; ++i){
		bufferTypes.push_back( (BufferType) i);
	}
}

void Test::withBufferTypes(vector<unsigned> buffTypes)
{
	bufferTypes.clear();
	std::vector<unsigned>::iterator it;
	FOR_EACH(it, buffTypes){
		bufferTypes.push_back((BufferType)(*it));
	}
}

void Test::excludeBufferTypes(vector<unsigned> buffTypes)
{
	vector<unsigned>::iterator i, j;
	FOR_EACH(i, buffTypes) {
		FOR_EACH(j, bufferTypes) {
			if (*i == *j){
				bufferTypes.erase(j);
				break;
			}
		}
	}
}

void Test::printCurrentState()
{
    printf("-Implementation Type = ");
    switch (getImplementationType()){
        case C: 		printf(" C        "); 	break;
        case SSE2: 		printf(" SSE2     ");	break;
        case CUDA: 		printf(" CUDA     ");	break;
        case CUDA_REDUC:		printf(" CUDA_REDUC    ");	break;
        case CUDA_INV:	printf(" CUDA_INV ");	break;
    }
    printf(" Buffer Type = ");
    switch (getBufferType()){
        case FLOAT: printf(" FLOAT "); 	break;
        case BIT: 	printf(" BIT   ");	break;
        case SIGN: 	printf(" SIGN  ");	break;
        case BYTE:	printf(" BYTE  ");	break;
    }
    printf("Size = %d ", size);
    if (initialWeighsRange != 0){
		printf("Weighs Range %f ", initialWeighsRange);
    }
    printf("\n");
}

void Test::printParameters()
{
    printf("-Size paramenters: min size = %d max size = %d increment = %d \n", minSize, maxSize, incSize);

    printf("-Implementations: ");
    std::vector<ImplementationType>::iterator it;
    FOR_EACH(it, implementationTypes) {

		switch (*it){
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
    FOR_EACH(it2, bufferTypes) {

		switch (*it2){
			case FLOAT: printf(" FLOAT "); 	break;
			case BIT: 	printf(" BIT   ");	break;
			case SIGN: 	printf(" SIGN  ");	break;
			case BYTE:	printf(" BYTE  ");	break;
		}
    }
    printf("\n");
    if (initialWeighsRange != 0){
		printf("-Weighs Range %f \n", initialWeighsRange);
		printf("\n");
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

float Test::getInitialWeighsRange()
{
    return initialWeighsRange;
}

void Test::setInitialWeighsRange(float initialWeighsRange)
{
    this->initialWeighsRange = initialWeighsRange;
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

void Test::openFile(string path, ClassID classID, Method method)
{
	if (file){
		fclose(file);
		file = NULL;
	}
	path += getFileName(classID, method);
	if (!(file = fopen(path.data(), "w")))
	{
		string error = "Error opening " + path;
		throw error;
	}
}

void Test::closeFile()
{
	if (file){
		fclose(file);
		file = NULL;
	}
}

void Test::plotToFile(float data)
{
	if (file){
		fprintf(file, "%d %f \n", size, data );
	} else {
		string error = "There is no opened file.";
		throw error;
	}
}

std::string Test::bufferTypeToString()
{
	std::string toReturn;
	switch (getBufferType()){
	case FLOAT:
		toReturn = "FLOAT";
		break;
	case BYTE:
		toReturn = "BYTE";
		break;
	case BIT:
		toReturn = "BIT";
		break;
	case SIGN:
		toReturn = "SIGN";
		break;
	}
	return toReturn;
}

std::string Test::implementationTypeToString()
{
	std::string toReturn;
	switch (getImplementationType()){
	case C:
		toReturn = "C";
		break;
	case SSE2:
		toReturn = "SSE2";
		break;
	case CUDA:
		toReturn = "CUDA";
		break;
	case CUDA_REDUC:
		toReturn = "CUDA_REDUC";
		break;
	case CUDA_INV:
		toReturn = "CUDA_INV";
		break;
	}
	return toReturn;
}

void Test::fromToBySize(unsigned minSize, unsigned maxSize, unsigned incSize)
{
	this->size = minSize;
	this->minSize = minSize;
	this->maxSize = maxSize;
	this->incSize = incSize;
}

void Test::fromToByOutputSize(unsigned minOutputSize, unsigned maxOutputSize, unsigned incOutputSize)
{
	this->outputSize = minOutputSize;
	this->minOutputSize = minOutputSize;
	this->maxOutputSize = maxOutputSize;
	this->incOutputSize = incOutputSize;
}

unsigned Test::getOutputSize()
{
	return outputSize;
}

string Test::getFileName(ClassID& classID, Method& method)
{
    return
//    classToString(classID) + "_" + methodToString(method) + "_" +
       bufferTypeToString() + "_" + implementationTypeToString() + ".DAT";
}

string Test::toString(ClassID classID, Method method)
{
	return classToString(classID) + methodToString(method);
}

string Test::classToString(ClassID classID)
{
	string toReturn;
	switch (classID){

		case BUFFER: toReturn = "BUFFER";			break;
		case CONNECTION: toReturn = "CONNECTION";	break;
		default:
			string error = "There's no such class to test.";
			throw error;

	}
	return toReturn;
}

string Test::methodToString(Method method)
{
	string toReturn;
	switch (method){
		case COPYFROM: 			toReturn = "COPYFROM";			break;
		case COPYTO: 			toReturn = "COPYTO";			break;
		case CLONE: 			toReturn = "CLONE";				break;
		case COPYFROMINTERFACE: toReturn = "COPYFROMINTERFACE";	break;
		case COPYTOINTERFACE: 	toReturn = "COPYTOINTERFACE";	break;
		case ACTIVATION: 		toReturn = "ACTIVATION";		break;
		case CALCULATEANDADDTO: toReturn = "CALCULATEANDADDTO";	break;
		case MUTATE: 			toReturn = "MUTATE";			break;
		case CROSSOVER: 		toReturn = "CROSSOVER";			break;
		default:
			string error = "There's no such method to test.";
			throw error;

	}
	return toReturn;
}
