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
	enableAllBufferTypes();
	enableAllImplementationTypes();
	initialWeighsRange = 0;
	file = NULL;
}

Test::~Test()
{
	// TODO Auto-generated destructor stub
}

void Test::test(ClassID classID, Method method)
{
	for (size = minSize; size <= maxSize; size += incSize) {
		for (bufferType = (BufferType) 0; bufferType < BUFFER_TYPE_DIM; bufferType = (BufferType) ((unsigned)bufferType + 1) ) if (bufferTypes[bufferType]){
			for (implementationType = (ImplementationType) 0; implementationType < IMPLEMENTATION_TYPE_DIM; implementationType = (ImplementationType) ((unsigned)implementationType + 1)) if (implementationTypes[implementationType]) {
				try {
					unsigned differencesCounter = doMethod(classID, method);
					if (differencesCounter > 0){
						printCurrentState();
						cout << differencesCounter << " differences detected." << endl;
					}
				} catch (string error) {
					cout << "Error: " << error << endl;
					cout << " While testing...";
					printCurrentState();
				}
			}
		}
	}
	cout << toString(classID, method) << endl;
}

unsigned Test::doMethod(ClassID classID, Method method)
{
	unsigned toReturn;

	Buffer* buffer = Factory::newBuffer(getSize(), getBufferType(), getImplementationType());
	buffer->random(initialWeighsRange);

	switch (classID){

		case BUFFER:
			toReturn = doMethodBuffer(buffer, method);
		break;
		case CONNECTION:
		{
			Connection* connection = Factory::newConnection(buffer, outputSize, getImplementationType());
			connection->random(initialWeighsRange);
			toReturn = doMethodConnection(connection, method);
			delete(connection);
		}

		break;
		default:
			string error = "there's no such class to test";
			throw error;

	}
	delete(buffer);
	return toReturn;
}

unsigned Test::doMethodConnection(Connection* connection, Method method)
{
	unsigned differencesCounter;

	switch (method){

	case CALCULATEANDADDTO:
	{
		Buffer* results = Factory::newBuffer(outputSize, FLOAT, connection->getImplementationType());

		Buffer* cInput = Factory::newBuffer(connection->getInput(), C);
		Connection* cConnection = Factory::newConnection(cInput, outputSize, C);
		cConnection->copyFrom(connection);

		Buffer* cResults = Factory::newBuffer(outputSize, FLOAT, C);

		connection->calculateAndAddTo(results);
		cConnection->calculateAndAddTo(cResults);

		differencesCounter = Test::assertEquals(cResults, results);

		delete(results);
		delete(cInput);
		delete(cConnection);
		delete(cResults);
	}
	break;
	case MUTATE:
	{
		unsigned pos = Random::positiveInteger(connection->getSize());
		float mutation = Random::floatNum(initialWeighsRange);
		connection->mutate(pos, mutation);

		Buffer* cInput = Factory::newBuffer(connection->getInput(), C);
		Connection* cConnection = Factory::newConnection(cInput, outputSize, C);
		cConnection->copyFrom(connection);

		for(unsigned i=0; i < NUM_MUTATIONS; i++) {
			float mutation = Random::floatNum(initialWeighsRange);
			unsigned pos = Random::positiveInteger(connection->getSize());
			connection->mutate(pos, mutation);
			cConnection->mutate(pos, mutation);
		}

		differencesCounter = Test::assertEquals(cConnection, connection);
		delete(cInput);
		delete(cConnection);
	}
	break;
	case CROSSOVER:
	{
		Connection* other = Factory::newConnection(connection->getInput(), outputSize, connection->getImplementationType());
		other->random(initialWeighsRange);

		Buffer* cInput = Factory::newBuffer(connection->getInput(), C);
		Connection* cConnection = Factory::newConnection(cInput, outputSize, C);
		cConnection->copyFrom(connection);
		Connection* cOther = Factory::newConnection(cInput, outputSize, C);
		cOther->copyFrom(other);

		Interface bitBuffer = Interface(connection->getSize(), BIT);
		//TODO bitBuffer.random(2); ??
		bitBuffer.random(1);

		connection->crossover(other, &bitBuffer);
		cConnection->crossover(cOther, &bitBuffer);

		differencesCounter = Test::assertEquals(cConnection, connection);
		differencesCounter += Test::assertEquals(cOther, other);

		delete(other);
		delete(cInput);
		delete(cConnection);
		delete(cOther);
	}
	break;
	default:
		string error = "The method " + methodToString(method) + " is not defined to test for Connection.";
		throw error;

	}

	return differencesCounter;
}

unsigned Test::doMethodBuffer(Buffer* buffer, Method method)
{
	unsigned differencesCounter;
	switch (method){

	case ACTIVATION:
	{
		FunctionType functionType = IDENTITY;
		Buffer* results = Factory::newBuffer(buffer->getSize(), FLOAT, buffer->getImplementationType());
		results->random(initialWeighsRange);

		Buffer* cResults = Factory::newBuffer(results, C);
		Buffer* cBuffer = Factory::newBuffer(buffer->getSize(), buffer->getBufferType(), C);

		buffer->activation(results, functionType);
		cBuffer->activation(cResults, functionType);
		differencesCounter = Test::assertEquals(cBuffer, buffer);

		delete(results);
		delete(cBuffer);
		delete(cResults);
	}
	break;
	case COPYFROMINTERFACE:
	{
		Interface interface = Interface(buffer->getSize(), buffer->getBufferType());
		interface.random(initialWeighsRange);

		Buffer* cBuffer = Factory::newBuffer(buffer, C);

		buffer->copyFromInterface(&interface);
		cBuffer->copyFromInterface(&interface);

		differencesCounter = Test::assertEquals(cBuffer, buffer);

		delete(cBuffer);
	}
	break;
	case COPYTOINTERFACE:
	{
		Interface interface = Interface(buffer->getSize(), buffer->getBufferType());

		Buffer* cBuffer = Factory::newBuffer(buffer, C);
		Interface cInterface = Interface(buffer->getSize(), buffer->getBufferType());

		buffer->copyToInterface(&interface);
		cBuffer->copyToInterface(&cInterface);

		differencesCounter = Test::assertEqualsInterfaces(&cInterface, &interface);

		delete(cBuffer);
	}
	break;
	case CLONE:
	{
		Buffer* copy = buffer->clone();
		differencesCounter = Test::assertEquals(buffer, copy);

		if (buffer->getImplementationType() != copy->getImplementationType()){
			printf("The buffers are not of the same implementation type.\n");
			++differencesCounter;
		}
		delete(copy);
	}
	break;

	default:
		string error = "The method " + methodToString(method) + " is not defined to test for Buffer.";
		throw error;
	}
	return differencesCounter;
}


unsigned Test::getIncSize()
{
    return incSize;
}

unsigned Test::getMaxSize()
{
    return maxSize;
}

unsigned Test::getMinSize()
{
    return minSize;
}

unsigned Test::getSize()
{
    return size;
}

void Test::setIncSize(unsigned  incSize)
{
    this->incSize = incSize;
}

void Test::setMaxSize(unsigned  maxSize)
{
    this->maxSize = maxSize;
}

void Test::setMinSize(unsigned  minSize)
{
    this->minSize = minSize;
}

void Test::sizeToMin()
{
    size = minSize;
}

int Test::hasNextSize()
{
    return size + incSize <= maxSize;
}

void Test::sizeIncrement()
{
    size += incSize;
}

ImplementationType Test::getImplementationType()
{
    return implementationType;
}

int Test::hasNextImplementationType()
{
	unsigned i = (unsigned)implementationType;
	do{
		if (++i >= IMPLEMENTATION_TYPE_DIM) {
			return 0;
		}
	} while (implementationTypes[i] == 0);

	return 1;
}

void Test::implementationTypeIncrement()
{
	unsigned i = (unsigned)implementationType;
	do{
		if (++i >= IMPLEMENTATION_TYPE_DIM) {
			return;
		}
	} while (implementationTypes[i] == 0);

	implementationType = (ImplementationType) i;
}

void Test::enableAllImplementationTypes()
{
	for(int i=0; i < IMPLEMENTATION_TYPE_DIM; i++){
		implementationTypes[i] = 1;
	}
}

void Test::implementationTypeToMin()
{
	implementationType = (ImplementationType) 0;
}



void Test::disableAllImplementationTypes()
{
	for(int i=0; i < IMPLEMENTATION_TYPE_DIM; i++){
		implementationTypes[i] = 0;
	}
}

void Test::disableImplementationType(ImplementationType implementationType)
{
	implementationTypes[ implementationType ] = 0;
}

void Test::enableImplementationType(ImplementationType implementationType)
{
	implementationTypes[ implementationType ] = 1;
}

BufferType Test::getBufferType()
{
	return bufferType;
}

void Test::bufferTypeToMin()
{
	bufferType = (BufferType) 0;
}

int Test::hasNextBufferType()
{
	unsigned i = (unsigned)bufferType;
	do{
		if (++i >= BUFFER_TYPE_DIM) {
			return 0;
		}
	} while (bufferTypes[i] == 0);

	return 1;
}

void Test::bufferTypeIncrement()
{
	unsigned i = (unsigned)bufferType;
	do{
		if (++i >= BUFFER_TYPE_DIM) {
			return;
		}
	} while (bufferTypes[i] == 0);

	bufferType = (BufferType) i;
}

void Test::enableBufferType(BufferType bufferType)
{
	bufferTypes[bufferType] = 1;
}

void Test::disableBufferType(BufferType bufferType)
{
	bufferTypes[bufferType] = 0;
}

void Test::enableAllBufferTypes()
{
	for(int i=0; i < BUFFER_TYPE_DIM; i++){
		bufferTypes[i] = 1;
	}
}

void Test::disableAllBufferTypes()
{
	for(int i=0; i < BUFFER_TYPE_DIM; i++){
		bufferTypes[i] = 0;
	}
}

void Test::printCurrentState()
{
    printf("-Implementation Type = ");
    switch (implementationType){
        case C: 		printf(" C        "); 	break;
        case SSE2: 		printf(" SSE2     ");	break;
        case CUDA: 		printf(" CUDA     ");	break;
        case CUDA2:		printf(" CUDA2    ");	break;
        case CUDA_INV:	printf(" CUDA_INV ");	break;
    }
    printf(" Buffer Type = ");
    switch (bufferType){
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
    for (int i=0; i < IMPLEMENTATION_TYPE_DIM; i++){
    	if (implementationTypes[i] == 1){

			switch ((ImplementationType) i){
				case C: 		printf(" C        "); 	break;
				case SSE2: 		printf(" SSE2     ");	break;
				case CUDA: 		printf(" CUDA     ");	break;
				case CUDA2:		printf(" CUDA2    ");	break;
				case CUDA_INV:	printf(" CUDA_INV ");	break;
			}
    	}
    }
    printf("\n");
    printf("-Buffer Types: ");
    for (int i=0; i < BUFFER_TYPE_DIM; i++){
    	if (bufferTypes[i] == 1){

			switch ((BufferType) i){
				case FLOAT: printf(" FLOAT "); 	break;
				case BIT: 	printf(" BIT   ");	break;
				case SIGN: 	printf(" SIGN  ");	break;
				case BYTE:	printf(" BYTE  ");	break;
			}
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
	switch (bufferType){
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
	switch (implementationType){
	case C:
		toReturn = "C";
		break;
	case SSE2:
		toReturn = "SSE2";
		break;
	case CUDA:
		toReturn = "CUDA";
		break;
	case CUDA2:
		toReturn = "CUDA2";
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

void Test::outputSizeToMin()
{
	outputSize = minOutputSize;
}

int Test::outputSizeIncrement()
{
    outputSize += incOutputSize;

    return outputSize <= maxOutputSize;
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
