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
	withAll(ET_BUFFER);
	withAll(ET_IMPLEMENTATION);
	with(ET_FUNCTION, 1, IDENTITY);
	initialWeighsRange = 10;
	file = NULL;
}

Test::~Test()
{
	// TODO Auto-generated destructor stub
}

BufferType Test::getBufferType()
{
	return (BufferType)*itEnumType[ET_BUFFER];
}
ImplementationType Test::getImplementationType()
{
	return (ImplementationType)*itEnumType[ET_IMPLEMENTATION];
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
		FOR_EACH(itEnumType[ET_BUFFER], enumTypes[ET_BUFFER]){
			FOR_EACH(itEnumType[ET_IMPLEMENTATION], enumTypes[ET_IMPLEMENTATION]){
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
		FOR_EACH(itEnumType[ET_BUFFER], enumTypes[ET_BUFFER]){
			FOR_EACH(itEnumType[ET_IMPLEMENTATION], enumTypes[ET_IMPLEMENTATION]){
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

void Test::withAll(EnumType enumType)
{
	enumTypes[enumType].clear();
	unsigned dim = enumTypeDim(enumType);
	for(unsigned i=0; i < dim; i++){
		enumTypes[enumType].push_back( i);
	}
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
    std::vector<unsigned>::iterator it;
    printf("-Size paramenters: min size = %d max size = %d increment = %d \n", minSize, maxSize, incSize);

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
