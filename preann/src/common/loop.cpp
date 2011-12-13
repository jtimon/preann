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
	f(parametersMap);
	unsigned differencesCounter = parametersMap->getNumber("differencesCounter");
	if (differencesCounter > 0){
		Loop* actionLoop = (Loop*)parametersMap->getPtr("actionLoop");
		cout << actionLoop->getState() << " : " << differencesCounter << " differences detected." << endl;
	}
}
void Loop::test(void (*func)(ParametersMap*), ParametersMap* parametersMap)
{
	repeatAction(testAction, func, parametersMap, NULL);
}

//void preparePlotFunction(Test* test)
//{
//	string* subPath = (string*)test->getVariable("subPath");
//    FILE* plotFile = (FILE*)test->getVariable("plotFile");
//    //TODO substitute count variable for an iterator
//    unsigned* count = (unsigned*)test->getVariable("count");
//    string functionName = test->getCurrentState();
//
//    if ((*count)++ > 0){
//        fprintf(plotFile, " , ");
//    }
//    string dataPath = (*subPath) + functionName + ".DAT";
//    int lineColor = ((Plot*)test)->getLineColor();
//    int pointType = ((Plot*)test)->getPointType();
//
//    string line = " \"" + dataPath + "\" using 1:2 title \"" + functionName + "\" with linespoints lt " + to_string(lineColor) + " pt " + to_string(pointType);
//    fprintf(plotFile, "%s", line.data());
//}
//
//void Plot::createPlotScript(string path, string testedMethod)
//{
//	string plotPath = getPlotPath(path, testedMethod);
//	string outputPath = path + "images/" + testedMethod + ".png";
//
//	FILE* plotFile = openFile(plotPath);
//
//	fprintf(plotFile, "set terminal png \n");
//	fprintf(plotFile, "set output \"%s\" \n", outputPath.data());
//	fprintf(plotFile, "plot ");
//
//	unsigned count = 0;
//	string subPath = path + "data/" + testedMethod + "_";
//
//	putVariable("subPath", &subPath);
//	putVariable("plotFile", plotFile);
//	putVariable("count", &count);
//
//    string functionName = "preparePlotFunction";
//    loopFunction(simpleAction, preparePlotFunction, functionName);
//
//	fprintf(plotFile, "\n");
//	fclose(plotFile);
//}

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
