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
	std::map<string, float> tNumbers;
	std::map<string, void*> tPtrs;
	std::map<string, string> tStrings;
public:
	ParametersMap();
	virtual ~ParametersMap();
	void putNumber(std::string key, float number);
	float getNumber(std::string key);
	void putPtr(std::string key, void* ptr);
	void* getPtr(std::string key);
	void putString(std::string key, std::string str);
	std::string getString(std::string key);
};

class Loop {
protected:
	std::string tKey;
	Loop* tInnerLoop;
	Loop* tCallerLoop;
	
	void repeatFunctionBase(void (*func)(ParametersMap*), ParametersMap* parametersMap);
	void repeatActionBase(void (*action)(void (*)(ParametersMap*), ParametersMap* parametersMap, Loop* functionLoop),
			void (*func)(ParametersMap*), ParametersMap* parametersMap, Loop* functionLoop);
	void setCallerLoop(Loop* callerLoop);
public:
	Loop();
	Loop(std::string key, Loop* innerLoop);
	virtual ~Loop();
	
	void test(void (*func)(ParametersMap*), ParametersMap* parametersMap);
	
	virtual std::string getState() = 0;
	virtual void repeatFunction(void (*func)(ParametersMap*), ParametersMap* parametersMap) = 0;
	virtual void repeatAction(void (*action)(void (*)(ParametersMap*), ParametersMap* parametersMap, Loop* functionLoop),
					void (*func)(ParametersMap*), ParametersMap* parametersMap, Loop* functionLoop) = 0;
};

class RangeLoop : public Loop {
	float tValue, tMin, tMax, tInc;
public:
	RangeLoop(std::string key, float min, float max, float inc, Loop* innerLoop);
	virtual ~RangeLoop();

	virtual std::string getState();
	virtual void repeatFunction(void (*func)(ParametersMap*), ParametersMap* parametersMap);
	virtual void repeatAction(void (*action)(void (*)(ParametersMap*), ParametersMap* parametersMap, Loop* functionLoop),
					void (*func)(ParametersMap*), ParametersMap* parametersMap, Loop* functionLoop);
};

class EnumLoop : public Loop {
	EnumType tEnumType;
	vector<unsigned> tValueVector;
	unsigned tIndex;
public:
	EnumLoop(std::string key, EnumType enumType, Loop* innerLoop);
	EnumLoop(Loop* innerLoop, std::string key, EnumType enumType, unsigned count, ...);
	virtual ~EnumLoop();

	virtual std::string getState();
	virtual void repeatFunction(void (*func)(ParametersMap*), ParametersMap* parametersMap);
	virtual void repeatAction(void (*action)(void (*)(ParametersMap*), ParametersMap* parametersMap, Loop* functionLoop),
					void (*func)(ParametersMap*), ParametersMap* parametersMap, Loop* functionLoop);
};

class JoinLoop : public Loop {
	vector<Loop*> tInnerLoops;
public:
	JoinLoop(unsigned count, ...);
	virtual ~JoinLoop();
	
	virtual std::string getState();
	virtual void repeatFunction(void (*func)(ParametersMap*), ParametersMap* parametersMap);
	virtual void repeatAction(void (*action)(void (*)(ParametersMap*), ParametersMap* parametersMap, Loop* functionLoop),
					void (*func)(ParametersMap*), ParametersMap* parametersMap, Loop* functionLoop);
};

class EnumValueLoop : public Loop {
	EnumType tEnumType;
	unsigned tEnumValue;
public:
	EnumValueLoop(std::string key, EnumType enumType, unsigned enumValue, Loop* innerLoop);
	virtual ~EnumValueLoop();
	
	virtual std::string getState();
	virtual void repeatFunction(void (*func)(ParametersMap*), ParametersMap* parametersMap);
	virtual void repeatAction(void (*action)(void (*)(ParametersMap*), ParametersMap* parametersMap, Loop* functionLoop),
					void (*func)(ParametersMap*), ParametersMap* parametersMap, Loop* functionLoop);
};

#endif /* LOOP_H_ */
