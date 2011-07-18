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
	enableAllVectorTypes();
	enableAllImplementationTypes();
	initialWeighsRange = 0;
	file = NULL;
}

Test::~Test()
{
	// TODO Auto-generated destructor stub
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

VectorType Test::getVectorType()
{
	return vectorType;
}

void Test::vectorTypeToMin()
{
	vectorType = (VectorType) 0;
}

int Test::hasNextVectorType()
{
	unsigned i = (unsigned)vectorType;
	do{
		if (++i >= VECTOR_TYPE_DIM) {
			return 0;
		}
	} while (vectorTypes[i] == 0);

	return 1;
}

void Test::vectorTypeIncrement()
{
	unsigned i = (unsigned)vectorType;
	do{
		if (++i >= VECTOR_TYPE_DIM) {
			return;
		}
	} while (vectorTypes[i] == 0);

	vectorType = (VectorType) i;
}

void Test::enableVectorType(VectorType vectorType)
{
	vectorTypes[vectorType] = 1;
}

void Test::disableVectorType(VectorType vectorType)
{
	vectorTypes[vectorType] = 0;
}

void Test::enableAllVectorTypes()
{
	for(int i=0; i < VECTOR_TYPE_DIM; i++){
		vectorTypes[i] = 1;
	}
}

void Test::disableAllVectorTypes()
{
	for(int i=0; i < VECTOR_TYPE_DIM; i++){
		vectorTypes[i] = 0;
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
    printf(" Vector Type = ");
    switch (vectorType){
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
    printf("-Vector Types: ");
    for (int i=0; i < VECTOR_TYPE_DIM; i++){
    	if (vectorTypes[i] == 1){

			switch ((VectorType) i){
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


unsigned char Test::areEqual(float expected, float actual, VectorType vectorType)
{
	if (vectorType == FLOAT){
		return (expected - 1 < actual
			 && expected + 1 > actual);
	} else {
		return expected == actual;
	}
}

unsigned Test::assertEqualsInterfaces(Interface* expected, Interface* actual)
{
    if(expected->getVectorType() != actual->getVectorType()){
        throw "The interfaces are not even of the same type!";
    }
    if(expected->getSize() != actual->getSize()){
        throw "The interfaces are not even of the same size!";
    }
	unsigned differencesCounter = 0;

    for(unsigned i = 0;i < expected->getSize();i++){
        if(!areEqual(expected->getElement(i), actual->getElement(i), expected->getVectorType())){
            printf("The interfaces are not equal at the position %d (expected = %f actual %f).\n", i, expected->getElement(i), actual->getElement(i));
            ++differencesCounter;
        }
    }
	return differencesCounter;
}

unsigned Test::assertEquals(Vector* expected, Vector* actual)
{
    if(expected->getVectorType() != actual->getVectorType()){
        throw "The vectors are not even of the same type!";
    }
    if(expected->getSize() != actual->getSize()){
        throw "The vectors are not even of the same size!";
    }

	unsigned differencesCounter = 0;
	Interface* expectedInt = expected->toInterface();
	Interface* actualInt = actual->toInterface();

    for(unsigned i = 0;i < expectedInt->getSize();i++){
        if(!areEqual(expectedInt->getElement(i), actualInt->getElement(i), expectedInt->getVectorType())){
            printf("The vectors are not equal at the position %d (expected = %f actual %f).\n", i, expectedInt->getElement(i), actualInt->getElement(i));
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

std::string Test::vectorTypeToString()
{
	std::string toReturn;
	switch (vectorType){
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
       vectorTypeToString() + "_" + implementationTypeToString() + ".DAT";
}

