/*
 * loop.cpp
 *
 *  Created on: Dec 12, 2011
 *      Author: timon
 */

#include "loop.h"

// class ParametersMap 
ParametersMap::ParametersMap()
{
}

ParametersMap::~ParametersMap()
{
	tNumbers.clear();
	tPtrs.clear();
}

void ParametersMap::putNumber(std::string key, float number)
{
	tNumbers[key] = number;
}

float ParametersMap::getNumber(std::string key)
{
	if (!tNumbers.count(key)){
		string error = "ParametersMap::getNumber : There's no number named " + key;
		throw error;
	}
	return tNumbers[key];
}

void ParametersMap::putPtr(std::string key, void* ptr)
{
	tPtrs[key] = ptr;
}

void* ParametersMap::getPtr(std::string key)
{
	if (!tPtrs.count(key)){
		string error = "ParametersMap::getPtr : There's no pointer named " + key;
		throw error;
	}
	return tPtrs[key];
}

void ParametersMap::putString(std::string key, std::string str)
{
	tStrings[key] = str;
}
std::string ParametersMap::getString(std::string key)
{
	if (!tStrings.count(key)){
		string error = "ParametersMap::getString : There's no string named " + key;
		throw error;
	}
	return tStrings[key];
}

// class Loop 
Loop::Loop()
{
	tKey = "Not Named Loop";
	tInnerLoop = NULL;
	tCallerLoop = NULL;
}

Loop::Loop(Loop* innerLoop, std::string key)
{
	tKey = key;
	tInnerLoop = innerLoop;
	tCallerLoop = NULL;
}

Loop::~Loop()
{
	if (tInnerLoop){
		delete(tInnerLoop);
	}
}

void Loop::repeatFunctionBase(void (*func)(ParametersMap*), ParametersMap* parametersMap)
{
	if (tInnerLoop){
		tInnerLoop->setCallerLoop(this);
		tInnerLoop->repeatFunction(func, parametersMap);
	} else {
		(*func)(parametersMap);
	}
}

void Loop::repeatActionBase(void (*action)(void (*)(ParametersMap*), ParametersMap* parametersMap, Loop* functionLoop),
		void (*func)(ParametersMap*), ParametersMap* parametersMap, Loop* functionLoop)
{
	if (tInnerLoop){
		tInnerLoop->setCallerLoop(this);
		tInnerLoop->repeatAction(action, func, parametersMap, functionLoop);
	} else {
		parametersMap->putPtr("actionLoop", this);
		(*action)(func, parametersMap, functionLoop);
	}
}

void Loop::setCallerLoop(Loop* callerLoop)
{
	tCallerLoop = callerLoop;
}

void testAction(unsigned (*f)(ParametersMap*), ParametersMap* parametersMap, Loop* functionLoop)
{
	try {
		f(parametersMap);
		unsigned differencesCounter = parametersMap->getNumber("differencesCounter");
		if (differencesCounter > 0){
			Loop* actionLoop = (Loop*)parametersMap->getPtr("actionLoop");
			cout << actionLoop->getState() << " : " << differencesCounter << " differences detected." << endl;
		}
	} catch (string e) {
		string functionLabel = parametersMap->getString("functionLabel");
		cout << " while testing " + functionLabel + " : " + e << endl;
	}
}
void Loop::test(void (*func)(ParametersMap*), ParametersMap* parametersMap, std::string functionLabel)
{
	parametersMap->putString("functionLabel", functionLabel);
	repeatAction(testAction, func, parametersMap, NULL);
}

unsigned Loop::valueToUnsigned()
{
	string error = "valueToUnsigned not implemented for this kind of Loop.";
	throw error;
}

int mapPointType(unsigned value)
{
// pt : 1=+, 2=X, 3=*, 4=square, 5=filled square, 6=circle,
//            7=filled circle, 8=triangle, 9=filled triangle, etc.
	switch (getEnum(pointEnum)){
		case 0:
			return 2;
		case 1:
			return 6;
		case 2:
			return 4;
		case 3:
			return 8;
		default:
		case 4:
			return 1;
		case 5:
			return 3;
	}
}
int mapLineColor(unsigned value)
{
// lt is for color of the points: -1=black 1=red 2=grn 3=blue 4=purple 5=aqua 6=brn 7=orange 8=light-brn
	switch (getEnum(colorEnum)){
		case 0:
			return 1;
		case 1:
			return 2;
		case 2:
			return 3;
		case 3:
			return 5;
		default:
		case 4:
			return -1;
		case 5:
			return 7;
		case 6:
			return 4;
	}
}
int Loop::getLineColor(ParametersMap* parametersMap)
{
	try {
		string lineColorParam = parametersMap->getString("lineColor");
		if(lineColorParam.compare(tKey) == 0){
			return mapLineColor(valueToUnsigned());
		} else {
			return this->tCallerLoop->getLineColor(parametersMap);
		}
	} catch (string e) {
		return mapLineColor(0);
	}
}

int Loop::getPointType(ParametersMap* parametersMap)
{
	try {
		string pointTypeParam = parametersMap->getString("pointType");
		if(pointTypeParam.compare(tKey) == 0){
			return mapLineColor(valueToUnsigned());
		} else {
			return this->tCallerLoop->getLineColor(parametersMap);
		}
	} catch (string e) {
		return mapLineColor(0);
	}
}

void preparePlotFunction(ParametersMap* parametersMap)
{
	string subPath = parametersMap->getString("subPath");
    FILE* plotFile = (FILE*)parametersMap->getPtr("plotFile");
    
    unsigned first = parametersMap->getNumber("first");
	Loop* actionLoop = (Loop*)parametersMap->getPtr("actionLoop");
    string state = actionLoop->getState();

    if (!first){
        fprintf(plotFile, " , ");
        parametersMap->putNumber("first", 0);
    }
    string dataPath = subPath + state + ".DAT";
    int lineColor = actionLoop->getLineColor(parametersMap);
    int pointType = actionLoop->getPointType(parametersMap);

    string line = " \"" + dataPath + "\" using 1:2 title \"" + state + "\" with linespoints lt " + to_string(lineColor) + " pt " + to_string(pointType);
    fprintf(plotFile, "%s", line.data());
}
void Loop::createGnuPlotScript(void (*func)(ParametersMap*), ParametersMap* parametersMap)
{
	string path = parametersMap->getString("path");
	string functionLabel = parametersMap->getString("functionLabel");
	
	string plotPath = path + "gnuplot/" + functionLabel + ".plt";
	string outputPath = path + "images/" + functionLabel + ".png";

	FILE* plotFile = openFile(plotPath);

	fprintf(plotFile, "set terminal png \n");
	fprintf(plotFile, "set output \"%s\" \n", outputPath.data());
	fprintf(plotFile, "plot ");

	unsigned count = 0;
	string subPath = path + "data/" + functionLabel + "_";

	parametersMap->putString("subPath", subPath);
	parametersMap->putPtr("plotFile", plotFile);
	parametersMap->putNumber("first", 1);

	try {
		repeatFunction(preparePlotFunction, parametersMap);
	} catch (string e) {
		string error = " while repeating preparePlotFunction : " + e;
		
	}
    string functionName = "preparePlotFunction";
    loopFunction(simpleAction, preparePlotFunction, functionName);

	fprintf(plotFile, "\n");
	fclose(plotFile);
}


void plotAction(unsigned (*f)(ParametersMap*), ParametersMap* parametersMap, Loop* functionLoop)
{
	string path = parametersMap->getSize()("path");
	string functionLabel = parametersMap->getVariable("functionLabel");
	Loop* actionLoop = (Loop*)parametersMap->getPtr("actionLoop");
	string state = actionLoop->getState();

	string dataPath = path + "data/" + functionLabel + "_" + state + ".DAT";
	FILE* dataFile = test->openFile(dataPath);
	fprintf(dataFile, "# Iterator %s \n", state.data());
	
	parametersMap->putNumber("totalTime", 0);
	parametersMap->putNumber("repetitions", 0);
	parametersMap->putNumber("repetitions", 0);
	
	//TODO AAAA pensar esto bien
//	functionLoop->repeatFunction(f, parametersMap);
//	fprintf(dataFile, " %f %f \n", plotIter.value, total/test->getValue("repetitions"));
//	
//	IteratorConfig plotIter = ((Plot*)test)->getPlotIterator();
//	FOR_ITER_CONF(plotIter){
//		float total = g(test);
//	}
	
	fclose(dataFile);
}

void plotFile(string path, string functionLabel)
{
	string plotPath = path + "gnuplot/" + functionLabel + ".plt";
	string syscommand = "gnuplot " + plotPath;
	system(syscommand.data());
}
void Loop::plot(void (*func)(ParametersMap*), ParametersMap* parametersMap, Loop* innerLoop, std::string functionLabel)
{
	parametersMap->putString("functionLabel", functionLabel);
	createGnuPlotScript(func, parametersMap);
	
	this->repeatAction(plotAction, func, parametersMap, innerLoop);

	string path = parametersMap->getString("path");
	plotFile(path, functionLabel);
	cout << functionLabel << endl;
}

// class RangeLoop
RangeLoop::RangeLoop(std::string key, float min, float max, float inc, Loop* innerLoop) : Loop(key, innerLoop)
{
	tMin = min;
	tMax = max;
	tInc = inc;
};

RangeLoop::~RangeLoop()
{
}

unsigned RangeLoop::valueToUnsigned()
{
	unsigned toReturn = 0;
	for(float auxValue = tMin; auxValue < tMax; auxValue += tInc){
		if (auxValue == tValue){
			return toReturn;
		}
		++toReturn;
	}
	return toReturn;
}

std::string RangeLoop::getState(){
	return tCallerLoop->getState() + "_" + tKey + "_" + to_string(tValue);
};

void RangeLoop::repeatFunction(void (*func)(ParametersMap*), ParametersMap* parametersMap)
{
	for(tValue = tMin; tValue < tMax; tValue += tInc){
		parametersMap->putNumber(tKey, tValue);
		repeatFunctionBase(func, parametersMap);
	}
}

void RangeLoop::repeatAction(void (*action)(void (*)(ParametersMap*), ParametersMap* parametersMap, Loop* functionLoop),
				void (*func)(ParametersMap*), ParametersMap* parametersMap, Loop* functionLoop)
{
	for(tValue = tMin; tValue < tMax; tValue += tInc){
		parametersMap->putNumber(tKey, tValue);
		repeatActionBase(action, func, parametersMap, functionLoop);
	}
}

// class EnumLoop 
EnumLoop::EnumLoop(std::string key, EnumType enumType, Loop* innerLoop) : Loop(key, innerLoop)
{
	unsigned dim = Enumerations::enumTypeDim(enumType);
	for(unsigned i=0; i < dim; i++){
		tValueVector.push_back( i);
	}
}

unsigned EnumLoop::valueToUnsigned()
{
	return tValueVector[tIndex];
}

EnumLoop::EnumLoop(Loop* innerLoop, std::string key, EnumType enumType, unsigned count, ...) : Loop(key, innerLoop)
{
	if (count == 0){
		string error = "EnumLoop : at least one enum value must be specified.";
		throw error;
	}
	va_list ap;
	va_start (ap, count);

	unsigned dim = Enumerations::enumTypeDim(enumType);
	for (unsigned i = 0; i < count; i++){
		unsigned arg = va_arg (ap, unsigned);
		if (arg > dim){
			string error = "EnumLoop : the enumType " + enumTypeToString(enumType) + " only has " + to_string(dim) + "possible values.";
			throw error;
		} else {
			tValueVector.push_back(arg);
		}
	}
	va_end (ap);
}

EnumLoop::~EnumLoop()
{
}

std::string EnumLoop::getState(){
	return tCallerLoop->getState() + "_" + tKey + "_" + Enumerations::toString(tEnumType, tValueVector[tIndex]);
};

void EnumLoop::repeatFunction(void (*func)(ParametersMap*), ParametersMap* parametersMap)
{
	for (int i = 0; i < tValueVector.size(); ++i) {
		parametersMap->putNumber(tKey, tValueVector[i]);
		repeatFunctionBase(func, parametersMap);
	}
}

void EnumLoop::repeatAction(void (*action)(void (*)(ParametersMap*), ParametersMap* parametersMap, Loop* functionLoop),
		void (*func)(ParametersMap*), ParametersMap* parametersMap, Loop* functionLoop)
{
	for (int i = 0; i < tValueVector.size(); ++i) {
		parametersMap->putNumber(tKey, tValueVector[i]);
		repeatActionBase(action, func, parametersMap, functionLoop);
	}
}

// class JoinLoop 
JoinLoop::JoinLoop(unsigned count, ...)
{
	if (count < 2){
		string error = "JoinLoop : at least 2 inner loops must be specified.";
		throw error;
	}
	va_list ap;
	va_start (ap, count);

	for (unsigned i = 0; i < count; i++){
		Loop* arg = va_arg (ap, Loop*);
		tInnerLoops.push_back(arg);
	}
	va_end (ap);
}

JoinLoop::~JoinLoop()
{
	for (int i = 0; i < tInnerLoops.size(); ++i) {
		delete(tInnerLoops[i]);
	}
	tInnerLoops.clear();
}

std::string JoinLoop::getState()
{
	return tCallerLoop->getState();
}

void JoinLoop::repeatFunction(void (*func)(ParametersMap*), ParametersMap* parametersMap)
{
	for (int i = 0; i < tInnerLoops.size(); ++i) {
		tInnerLoops[i]->setCallerLoop(this);
		tInnerLoops[i]->repeatFunction(func, parametersMap);
	}
}

void JoinLoop::repeatAction(void (*action)(void (*)(ParametersMap*), ParametersMap* parametersMap, Loop* functionLoop),
				void (*func)(ParametersMap*), ParametersMap* parametersMap, Loop* functionLoop)
{
	for (int i = 0; i < tInnerLoops.size(); ++i) {
		tInnerLoops[i]->setCallerLoop(this);
		tInnerLoops[i]->repeatAction(action, func, parametersMap, functionLoop);
	}
}

// class EnumValueLoop
EnumValueLoop::EnumValueLoop(std::string key, EnumType enumType, unsigned enumValue, Loop* innerLoop) : Loop(key, innerLoop)
{
	if (innerLoop == NULL){
		string error = "EnumValueLoop : EnumValueLoop makes no sense if it has no inner loop.";
		throw error;
	}
	tEnumType = enumType;
	tEnumValue = enumValue;
}

EnumValueLoop::~EnumValueLoop()
{
}

std::string EnumValueLoop::getState()
{
	return tCallerLoop->getState() + "_" + tKey + "_" + Enumerations::toString(tEnumType, tEnumValue);
}

void EnumValueLoop::repeatFunction(void (*func)(ParametersMap*), ParametersMap* parametersMap)
{
	parametersMap->putNumber(tKey, tEnumValue);
	tInnerLoop->setCallerLoop(this);
	tInnerLoop->repeatFunction(func, parametersMap);
}

void EnumValueLoop::repeatAction(void (*action)(void (*)(ParametersMap*), ParametersMap* parametersMap, Loop* functionLoop),
				void (*func)(ParametersMap*), ParametersMap* parametersMap, Loop* functionLoop)
{
	parametersMap->putNumber(tKey, tEnumValue);
	tInnerLoop->setCallerLoop(this);
	tInnerLoop->repeatAction(action, func, parametersMap, functionLoop);
}
