/*
 * loop.h
 *
 *  Created on: Dec 12, 2011
 *      Author: timon
 */

#ifndef LOOP_H_
#define LOOP_H_

#include "enumerations.h"

class ParametersMap {
public:
	ParametersMap();
	virtual ~ParametersMap();
	void putNumber(std::string key, float number);
};

class Loop {
protected:
	std::string tKey;
	Loop* tInnerLoop;
public:
	Loop();
	virtual ~Loop();
	void repeatFunctionBase(void (*func)(ParametersMap*), ParametersMap* parametersMap){
			if (tInnerLoop){
				tInnerLoop->repeatFunction(func, parametersMap);
			} else {
				(*func)(parametersMap);
			}
	};
	void repeatActionBase(void (*action)(void (*)(ParametersMap*), ParametersMap* parametersMap, Loop* functionLoop),
			void (*func)(ParametersMap*), ParametersMap* parametersMap, Loop* functionLoop)
	{
			if (tInnerLoop){
				tInnerLoop->repeatAction(action, func, parametersMap, functionLoop);
			} else {
				(*action)(func, parametersMap, functionLoop);
//				functionLoop->repeatFunction(func, parametersMap);
			}
	};
	virtual void repeatFunction(void (*func)(ParametersMap*), ParametersMap* parametersMap) = 0;

	virtual void repeatAction(void (*action)(void (*)(ParametersMap*), ParametersMap* parametersMap, Loop* functionLoop),
			void (*func)(ParametersMap*), ParametersMap* parametersMap, Loop* functionLoop) = 0;
};

class RangeLoop : public Loop {
	float tMin, tMax, tInc;
public:
	RangeLoop(std::string key, float min, float max, float inc){
		tKey = key;
		tMin = min;
		tMax = max;
		tInc = inc;
	};
	virtual ~RangeLoop();
	virtual void repeatFunction(void (*func)(ParametersMap*), ParametersMap* parametersMap)
	{
		for(float value = tMin; value < tMax; value += tInc){
			parametersMap->putNumber(tKey, value);
			repeatFunctionBase(func, parametersMap);
		}
	};
	void repeatAction(void (*action)(void (*)(ParametersMap*), ParametersMap* parametersMap, Loop* functionLoop),
			void (*func)(ParametersMap*), ParametersMap* parametersMap, Loop* functionLoop)
	{
		for(float value = tMin; value < tMax; value += tInc){
			parametersMap->putNumber(tKey, value);
			repeatActionBase(action, func, parametersMap, functionLoop);
		}
	};
};

class JoinLoop : public Loop {
	vector<Loop*> innerLoops;
};

class EnumLoop : public Loop {
public:
	EnumLoop(std::string key, EnumType enumType){
		tKey = key;
	};
	EnumLoop(std::string key, EnumType enumType, unsigned count, ...){
		tKey = key;
	};
	virtual ~EnumLoop();
	virtual void repeatFunction(void (*func)(ParametersMap*), ParametersMap* parametersMap)
	{
//		for(float value = tMin; value < tMax; value += tInc){
//			parametersMap->putNumber(key, value);
//			repeatFunctionBase(func, parametersMap);
//		}
	};
	void repeatAction(void (*action)(void (*)(ParametersMap*), ParametersMap* parametersMap, Loop* functionLoop),
			void (*func)(ParametersMap*), ParametersMap* parametersMap, Loop* functionLoop)
	{
//		for(float value = tMin; value < tMax; value += tInc){
//			parametersMap->putNumber(key, value);
//			repeatActionBase(func, parametersMap, functionLoop);
//		}
//		(*action)(func, this);
	};
};

#endif /* LOOP_H_ */
